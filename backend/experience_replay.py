"""
Experience Replay Buffer with Prioritization
Implements best practices for reinforcement learning:
- Experience replay (prevents forgetting)
- Prioritized replay (learn more from important experiences)
- Efficient sampling
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from collections import deque
from typing import List, Dict, Tuple
import random

# Use centralized P&L calculation to avoid the 100x contract multiplier bug
try:
    from utils.pnl import compute_pnl_pct
except ImportError:
    # Fallback if utils not available
    def compute_pnl_pct(trade: Dict) -> float:
        """Fallback P&L calculation with correct 100x multiplier"""
        quantity = trade.get("quantity", 1)
        cost_basis = trade.get("premium_paid", 1) * quantity * 100
        return trade.get("pnl", 0) / cost_basis if cost_basis > 0 else 0.0


class PrioritizedExperienceReplay:
    """
    Stores trading experiences with priority-based sampling.
    Important experiences (big wins/losses, surprises) are sampled more often.
    """
    
    def __init__(self, max_size: int = 10000, alpha: float = 0.6, beta: float = 0.4):
        """
        Args:
            max_size: Maximum buffer size
            alpha: How much prioritization to use (0 = uniform, 1 = full priority)
            beta: Importance sampling correction (starts low, anneals to 1)
        """
        self.max_size = max_size
        self.alpha = alpha  # Prioritization exponent
        self.beta = beta    # Importance sampling exponent
        self.beta_increment = 0.001  # Slowly increase beta to 1
        
        self.buffer = deque(maxlen=max_size)
        self.priorities = deque(maxlen=max_size)
        
    def add(self, experience: Dict, priority: float = None):
        """
        Add an experience to the buffer.
        
        Args:
            experience: Dict with trade information (signal, outcome, P&L, etc.)
            priority: Importance of this experience (defaults to max priority for new experiences)
        """
        # New experiences get max priority so they're replayed at least once
        if priority is None:
            priority = max(self.priorities) if self.priorities else 1.0
        
        self.buffer.append(experience)
        self.priorities.append(priority)
    
    def sample(self, batch_size: int) -> Tuple[List[Dict], np.ndarray, np.ndarray]:
        """
        Sample a batch of experiences based on priority.
        
        Returns:
            experiences: List of experience dicts
            indices: Indices of sampled experiences (for priority updates)
            weights: Importance sampling weights
        """
        if len(self.buffer) == 0:
            return [], np.array([]), np.array([])
        
        batch_size = min(batch_size, len(self.buffer))
        
        # Convert priorities to probabilities
        priorities = np.array(self.priorities)
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample indices based on probabilities
        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
        
        # Get experiences
        experiences = [self.buffer[idx] for idx in indices]
        
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize
        
        # Anneal beta towards 1
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return experiences, indices, weights
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities for sampled experiences based on TD error"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + 1e-6  # Small epsilon to ensure non-zero
    
    def __len__(self):
        return len(self.buffer)
    
    def get_stats(self) -> Dict:
        """Get buffer statistics"""
        if len(self.buffer) == 0:
            return {
                'size': 0,
                'avg_priority': 0,
                'max_priority': 0,
                'min_priority': 0
            }
        
        priorities = np.array(self.priorities)
        return {
            'size': len(self.buffer),
            'avg_priority': priorities.mean(),
            'max_priority': priorities.max(),
            'min_priority': priorities.min(),
            'beta': self.beta
        }


class TradeExperienceManager:
    """
    Manages trading experiences for learning.
    Converts closed trades into learning experiences.
    """
    
    def __init__(self, max_size: int = 10000):
        self.replay_buffer = PrioritizedExperienceReplay(max_size=max_size)
        self.processed_trade_ids = set()  # Track which trades we've already added
    
    def add_trade(self, trade: Dict, signal: Dict = None):
        """
        Convert a closed trade into a learning experience.
        
        Args:
            trade: Trade dict with entry/exit info and P&L
            signal: Original signal that led to this trade
        """
        # Don't add same trade twice
        if trade['id'] in self.processed_trade_ids:
            return
        
        # Use centralized P&L calculation (prevents 100x multiplier bug)
        pnl_pct = compute_pnl_pct(trade)
        
        # Priority factors:
        # 1. Large absolute P&L (big wins/losses are important)
        # 2. Unexpected outcomes (if we had low confidence but won, or high confidence but lost)
        abs_pnl = abs(pnl_pct)
        
        # Base priority on absolute P&L (learn more from extreme outcomes)
        priority = 1.0 + abs_pnl * 2.0
        
        # Bonus priority for unexpected outcomes
        if signal and 'confidence' in signal:
            confidence = signal['confidence']
            # If high confidence but lost, or low confidence but won = unexpected
            if (confidence > 0.7 and pnl_pct < 0) or (confidence < 0.4 and pnl_pct > 0):
                priority *= 1.5  # 50% more important
        
        # Create experience
        experience = {
            'trade_id': trade['id'],
            'timestamp': trade['timestamp'],
            'action': trade.get('option_type', 'UNKNOWN'),
            'entry_price': trade.get('entry_price', 0),
            'exit_price': trade.get('exit_price', 0),
            'premium_paid': trade['premium_paid'],
            'pnl': trade['pnl'],
            'pnl_pct': pnl_pct,
            'duration': trade.get('duration', 0),
            'signal': signal,
            'outcome': 'WIN' if pnl_pct > 0 else 'LOSS'
        }
        
        self.replay_buffer.add(experience, priority)
        self.processed_trade_ids.add(trade['id'])
    
    def sample_batch(self, batch_size: int = 32) -> Tuple[List[Dict], np.ndarray, np.ndarray]:
        """Sample a batch of experiences for learning"""
        return self.replay_buffer.sample(batch_size)
    
    def update_from_learning(self, indices: np.ndarray, td_errors: np.ndarray):
        """
        Update priorities based on learning outcomes.
        Higher TD error = more to learn from this experience.
        """
        # Convert TD errors to priorities
        priorities = np.abs(td_errors) + 1e-6
        self.replay_buffer.update_priorities(indices, priorities)
    
    def get_stats(self) -> Dict:
        """Get statistics about experiences"""
        stats = self.replay_buffer.get_stats()
        stats['processed_trades'] = len(self.processed_trade_ids)
        return stats
    
    def __len__(self):
        return len(self.replay_buffer)


def calculate_priority_from_trade(trade: Dict, predicted_pnl: float = None) -> float:
    """
    Calculate learning priority for a trade.
    
    Higher priority for:
    - Large wins/losses (extreme outcomes)
    - Unexpected outcomes (prediction error)
    - Rare events
    """
    # Use centralized P&L calculation (prevents 100x multiplier bug)
    actual_pnl = compute_pnl_pct(trade)
    
    # Base priority on absolute outcome
    priority = 1.0 + abs(actual_pnl) * 2.0
    
    # If we have a prediction, increase priority based on error
    if predicted_pnl is not None:
        prediction_error = abs(actual_pnl - predicted_pnl)
        priority *= (1.0 + prediction_error)
    
    return priority





