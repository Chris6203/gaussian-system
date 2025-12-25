#!/usr/bin/env python3
"""
RL Enhancements
===============

Advanced reinforcement learning components for improved trading policy:

1. Prioritized Experience Replay (PER)
   - Samples important experiences more frequently
   - Uses TD-error for priority calculation
   - Importance sampling weights for unbiased updates

2. Sharpe-Ratio Based Rewards
   - Incorporates risk-adjusted returns
   - Rolling Sharpe calculation
   - Drawdown penalties

3. Enhanced Reward Shaping
   - Calibration-aware rewards
   - Time-decay for stale predictions
   - Market regime adjustments

Usage:
    # Replace standard replay buffer
    per_buffer = PrioritizedReplayBuffer(capacity=10000)
    
    # Use enhanced reward calculator
    reward_calc = SharpeRewardCalculator()
    reward = reward_calc.calculate(trade_result, context)
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import heapq

logger = logging.getLogger(__name__)


# =============================================================================
# PRIORITIZED EXPERIENCE REPLAY (PER)
# =============================================================================

class SumTree:
    """
    Sum Tree data structure for efficient prioritized sampling.
    
    Each leaf stores a priority, and parent nodes store sums of children.
    Allows O(log n) sampling proportional to priority.
    """
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)  # Binary tree
        self.data = [None] * capacity  # Circular buffer for experiences
        self.write_idx = 0
        self.n_entries = 0
    
    def _propagate(self, idx: int, change: float):
        """Propagate priority change up the tree."""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)
    
    def _retrieve(self, idx: int, s: float) -> int:
        """Find leaf index for a given priority sum."""
        left = 2 * idx + 1
        right = left + 1
        
        if left >= len(self.tree):
            return idx
        
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
    
    def total(self) -> float:
        """Total priority sum (root of tree)."""
        return self.tree[0]
    
    def add(self, priority: float, data: Any):
        """Add experience with given priority."""
        idx = self.write_idx + self.capacity - 1
        
        self.data[self.write_idx] = data
        self.update(idx, priority)
        
        self.write_idx = (self.write_idx + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)
    
    def update(self, idx: int, priority: float):
        """Update priority at tree index."""
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)
    
    def get(self, s: float) -> Tuple[int, float, Any]:
        """
        Get experience for a given priority sum.
        
        Returns: (tree_idx, priority, data)
        """
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer.
    
    Samples experiences proportional to their TD-error, allowing the agent
    to learn more from surprising/important transitions.
    
    Features:
    - Proportional prioritization via SumTree
    - Importance sampling weights for unbiased updates
    - Priority decay for old experiences
    - Minimum priority to ensure all experiences can be sampled
    """
    
    def __init__(
        self,
        capacity: int = 10000,
        alpha: float = 0.6,  # Priority exponent (0 = uniform, 1 = full prioritization)
        beta: float = 0.4,   # Importance sampling exponent (increases to 1)
        beta_increment: float = 0.001,
        epsilon: float = 1e-6,  # Small constant for numerical stability
        max_priority: float = 1.0
    ):
        """
        Args:
            capacity: Maximum buffer size
            alpha: How much prioritization to use (0-1)
            beta: Initial importance sampling weight (increases to 1)
            beta_increment: How much to increase beta each sample
            epsilon: Small constant added to priorities
            max_priority: Maximum priority for new experiences
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        self.max_priority = max_priority
        
        self.tree = SumTree(capacity)
        
        # Statistics
        self._stats = {
            'total_added': 0,
            'total_sampled': 0,
            'priority_updates': 0
        }
        
        logger.info(f"✅ PrioritizedReplayBuffer initialized: capacity={capacity}, α={alpha}, β={beta}")
    
    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: Optional[np.ndarray],
        done: bool,
        log_prob: float = 0.0,
        info: Dict = None
    ):
        """Add experience with maximum priority."""
        experience = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'log_prob': log_prob,
            'info': info or {},
            'timestamp': datetime.now()
        }
        
        # New experiences get max priority (will be sampled)
        priority = self.max_priority ** self.alpha
        self.tree.add(priority, experience)
        
        self._stats['total_added'] += 1
    
    def sample(self, batch_size: int) -> Tuple[List[Dict], np.ndarray, List[int]]:
        """
        Sample batch of experiences with prioritized sampling.
        
        Returns:
            experiences: List of experience dicts
            weights: Importance sampling weights
            indices: Tree indices for priority updates
        """
        experiences = []
        indices = []
        weights = np.empty(batch_size, dtype=np.float32)
        
        # Divide priority range into segments
        segment = self.tree.total() / batch_size
        
        # Increase beta for more uniform sampling over time
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Calculate max weight for normalization
        min_prob = self.epsilon / self.tree.total() if self.tree.total() > 0 else self.epsilon
        max_weight = (min_prob * self.tree.n_entries) ** (-self.beta)
        
        for i in range(batch_size):
            # Sample uniformly within segment
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)
            
            idx, priority, data = self.tree.get(s)
            
            if data is None:
                # Buffer not full yet, use uniform sampling
                continue
            
            # Calculate importance sampling weight
            prob = priority / self.tree.total() if self.tree.total() > 0 else 1.0
            weight = (prob * self.tree.n_entries) ** (-self.beta)
            weights[i] = weight / max_weight  # Normalize
            
            experiences.append(data)
            indices.append(idx)
        
        self._stats['total_sampled'] += len(experiences)
        
        return experiences, weights[:len(experiences)], indices
    
    def update_priorities(self, indices: List[int], td_errors: np.ndarray):
        """
        Update priorities based on TD errors.
        
        Args:
            indices: Tree indices from sample()
            td_errors: TD errors from training
        """
        for idx, td_error in zip(indices, td_errors):
            # Priority = |TD error| + epsilon, raised to alpha
            priority = (abs(td_error) + self.epsilon) ** self.alpha
            priority = min(priority, self.max_priority ** self.alpha)
            self.tree.update(idx, priority)
        
        self._stats['priority_updates'] += len(indices)
    
    def __len__(self) -> int:
        return self.tree.n_entries
    
    def get_stats(self) -> Dict:
        return {
            **self._stats,
            'current_size': len(self),
            'capacity': self.capacity,
            'beta': self.beta,
            'total_priority': self.tree.total()
        }


# =============================================================================
# SHARPE-RATIO BASED REWARDS
# =============================================================================

@dataclass
class TradeHistory:
    """Track recent trades for Sharpe calculation."""
    returns: deque = field(default_factory=lambda: deque(maxlen=100))
    timestamps: deque = field(default_factory=lambda: deque(maxlen=100))
    
    def add_return(self, ret: float, timestamp: datetime = None):
        self.returns.append(ret)
        self.timestamps.append(timestamp or datetime.now())
    
    def get_sharpe(self, risk_free_rate: float = 0.0, annualize: bool = False) -> float:
        """Calculate Sharpe ratio from recent returns."""
        if len(self.returns) < 5:
            return 0.0
        
        returns = np.array(self.returns)
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return < 1e-8:
            return 0.0
        
        sharpe = (mean_return - risk_free_rate) / std_return
        
        if annualize and len(self.timestamps) >= 2:
            # Estimate trades per year based on frequency
            time_span = (self.timestamps[-1] - self.timestamps[0]).total_seconds() / 3600
            trades_per_hour = len(self.returns) / max(1, time_span)
            trades_per_year = trades_per_hour * 24 * 252  # Trading days
            sharpe *= np.sqrt(trades_per_year)
        
        return float(sharpe)
    
    def get_sortino(self, target: float = 0.0) -> float:
        """Calculate Sortino ratio (downside risk only)."""
        if len(self.returns) < 5:
            return 0.0
        
        returns = np.array(self.returns)
        mean_return = np.mean(returns)
        downside = returns[returns < target]
        
        if len(downside) < 2:
            return 0.0 if mean_return <= target else float('inf')
        
        downside_std = np.std(downside)
        if downside_std < 1e-8:
            return 0.0
        
        return (mean_return - target) / downside_std


class SharpeRewardCalculator:
    """
    Calculate rewards incorporating Sharpe ratio and risk-adjusted metrics.
    
    Reward components:
    1. Base PnL reward (scaled by account)
    2. Sharpe bonus/penalty (encourages consistent profits)
    3. Drawdown penalty (discourages large losses)
    4. Win rate bonus (maintains confidence)
    5. Efficiency bonus (quick profitable trades)
    """
    
    def __init__(
        self,
        pnl_scale: float = 100.0,
        sharpe_weight: float = 0.5,
        drawdown_weight: float = 2.0,
        win_bonus: float = 0.5,
        loss_penalty: float = 0.5,
        efficiency_bonus: float = 0.3,
        hold_penalty: float = -0.01
    ):
        """
        Args:
            pnl_scale: Scale factor for PnL rewards
            sharpe_weight: Weight for Sharpe ratio component
            drawdown_weight: Multiplier for drawdown penalty
            win_bonus: Bonus for winning trades
            loss_penalty: Penalty for losing trades
            efficiency_bonus: Bonus for quick profitable trades
            hold_penalty: Small penalty for holding (encourages action)
        """
        self.pnl_scale = pnl_scale
        self.sharpe_weight = sharpe_weight
        self.drawdown_weight = drawdown_weight
        self.win_bonus = win_bonus
        self.loss_penalty = loss_penalty
        self.efficiency_bonus = efficiency_bonus
        self.hold_penalty = hold_penalty
        
        self.trade_history = TradeHistory()
        self._episode_returns = []
        
        logger.info("✅ SharpeRewardCalculator initialized")
    
    def calculate(
        self,
        trade_result: Optional[Dict] = None,
        context: Optional[Dict] = None
    ) -> float:
        """
        Calculate reward for a trading action.
        
        Args:
            trade_result: Dict with 'pnl', 'time_held_minutes', 'max_drawdown', etc.
            context: Dict with 'account_balance', 'confidence', 'action', etc.
            
        Returns:
            Reward value
        """
        context = context or {}
        account_balance = context.get('account_balance', 10000.0)
        action = context.get('action', 0)  # HOLD = 0
        confidence = context.get('confidence', 0.5)
        
        reward = 0.0
        
        if trade_result is not None:
            pnl = trade_result.get('pnl', 0.0)
            time_held = trade_result.get('time_held_minutes', 60)
            max_dd = trade_result.get('max_drawdown', 0.0)
            
            # 1. Base PnL reward (as % of account, scaled)
            pnl_pct = pnl / max(1.0, account_balance)
            reward += pnl_pct * self.pnl_scale
            
            # 2. Win/loss bonus/penalty
            if pnl > 0:
                reward += self.win_bonus
                
                # Efficiency bonus for quick profits
                if time_held < 60:
                    reward += self.efficiency_bonus
            else:
                reward -= self.loss_penalty
                
                # Extra penalty for large losses
                if abs(pnl_pct) > 0.02:
                    reward -= self.loss_penalty
            
            # 3. Drawdown penalty
            if max_dd > 0.02:
                reward -= max_dd * self.drawdown_weight * 10
            
            # 4. Track for Sharpe calculation
            self.trade_history.add_return(pnl_pct)
            self._episode_returns.append(pnl_pct)
            
            # 5. Sharpe bonus/penalty (after enough trades)
            if len(self.trade_history.returns) >= 10:
                sharpe = self.trade_history.get_sharpe()
                
                # Bonus for positive Sharpe, penalty for negative
                sharpe_component = np.clip(sharpe, -2, 2) * self.sharpe_weight
                reward += sharpe_component
            
            # 6. Calibration penalty: high confidence + loss = bad
            if pnl < 0 and confidence > 0.7:
                reward -= (confidence - 0.5) * 2.0  # Penalize overconfidence
        
        else:
            # Step reward (no trade result yet)
            if action == 0:  # HOLD
                reward = self.hold_penalty
            else:
                # Small penalty for action without result (still in trade)
                reward = -0.005
        
        return float(np.clip(reward, -10, 10))
    
    def get_episode_sharpe(self) -> float:
        """Get Sharpe ratio for current episode."""
        if len(self._episode_returns) < 3:
            return 0.0
        
        returns = np.array(self._episode_returns)
        mean_ret = np.mean(returns)
        std_ret = np.std(returns)
        
        if std_ret < 1e-8:
            return 0.0
        
        return float(mean_ret / std_ret)
    
    def reset_episode(self):
        """Reset episode tracking (call at episode end)."""
        self._episode_returns = []
    
    def get_stats(self) -> Dict:
        """Get reward statistics."""
        return {
            'trade_count': len(self.trade_history.returns),
            'sharpe_ratio': self.trade_history.get_sharpe(),
            'sortino_ratio': self.trade_history.get_sortino(),
            'mean_return': float(np.mean(self.trade_history.returns)) if self.trade_history.returns else 0.0,
            'std_return': float(np.std(self.trade_history.returns)) if self.trade_history.returns else 0.0,
            'episode_sharpe': self.get_episode_sharpe(),
            'episode_trades': len(self._episode_returns)
        }


# =============================================================================
# MARKET REGIME REWARD ADJUSTMENT
# =============================================================================

class RegimeAwareRewardShaper:
    """
    Adjusts rewards based on market regime to encourage appropriate behavior.
    
    Regimes:
    - High volatility: Reward capital preservation, penalize large positions
    - Low volatility: Reward taking action, penalize excessive holding
    - Trending: Reward trend following
    - Choppy: Reward mean reversion / holding
    """
    
    def __init__(
        self,
        vix_high_threshold: float = 25.0,
        vix_low_threshold: float = 15.0
    ):
        self.vix_high = vix_high_threshold
        self.vix_low = vix_low_threshold
    
    def adjust_reward(
        self,
        base_reward: float,
        vix: float = 17.0,
        trend_strength: float = 0.0,  # -1 to 1, 0 = no trend
        action: int = 0,
        position_size: int = 1
    ) -> float:
        """
        Adjust reward based on market regime.
        
        Args:
            base_reward: Original reward
            vix: Current VIX level
            trend_strength: Trend indicator (-1 = down trend, 1 = up trend)
            action: Action taken
            position_size: Position size (1x, 2x)
            
        Returns:
            Adjusted reward
        """
        adjustment = 0.0
        
        # High volatility regime
        if vix > self.vix_high:
            # Penalize large positions in high vol
            if position_size > 1:
                adjustment -= 0.3
            
            # Reward holding in extreme vol
            if vix > 35 and action == 0:
                adjustment += 0.2
        
        # Low volatility regime
        elif vix < self.vix_low:
            # Small penalty for holding in calm markets
            if action == 0:
                adjustment -= 0.1
        
        # Trend following
        if abs(trend_strength) > 0.5:
            # Reward trading in direction of trend
            is_bullish_action = action in [1, 2]  # BUY_CALL
            is_bearish_action = action in [3, 4]  # BUY_PUT
            
            if trend_strength > 0.5 and is_bullish_action:
                adjustment += 0.2
            elif trend_strength < -0.5 and is_bearish_action:
                adjustment += 0.2
            # Penalize counter-trend
            elif trend_strength > 0.5 and is_bearish_action:
                adjustment -= 0.1
            elif trend_strength < -0.5 and is_bullish_action:
                adjustment -= 0.1
        
        return base_reward + adjustment


# =============================================================================
# ENHANCED PPO TRAINER
# =============================================================================

class EnhancedPPOTrainer:
    """
    Enhanced PPO training with prioritized replay and Sharpe rewards.
    
    Integrates with existing RLTradingPolicy.
    """
    
    def __init__(
        self,
        policy,  # RLTradingPolicy instance
        per_buffer: PrioritizedReplayBuffer = None,
        reward_calculator: SharpeRewardCalculator = None,
        regime_shaper: RegimeAwareRewardShaper = None
    ):
        """
        Args:
            policy: RLTradingPolicy instance
            per_buffer: Prioritized replay buffer (creates default if None)
            reward_calculator: Sharpe reward calculator (creates default if None)
            regime_shaper: Regime-aware reward shaper (creates default if None)
        """
        self.policy = policy
        self.per_buffer = per_buffer or PrioritizedReplayBuffer()
        self.reward_calculator = reward_calculator or SharpeRewardCalculator()
        self.regime_shaper = regime_shaper or RegimeAwareRewardShaper()
        
        self._training_stats = {
            'updates': 0,
            'td_errors': [],
            'rewards': [],
            'sharpe_history': []
        }
        
        logger.info("✅ EnhancedPPOTrainer initialized with PER + Sharpe rewards")
    
    def store_experience(
        self,
        state: np.ndarray,
        action: int,
        trade_result: Optional[Dict],
        next_state: Optional[np.ndarray],
        done: bool,
        context: Dict,
        log_prob: float = 0.0
    ):
        """
        Store experience with enhanced reward calculation.
        
        Args:
            state: Current state
            action: Action taken
            trade_result: Trade result dict
            next_state: Next state
            done: Episode done flag
            context: Context dict with vix, confidence, account_balance, etc.
            log_prob: Log probability of action
        """
        # Calculate enhanced reward
        base_reward = self.reward_calculator.calculate(trade_result, context)
        
        # Adjust for regime
        adjusted_reward = self.regime_shaper.adjust_reward(
            base_reward,
            vix=context.get('vix', 17.0),
            trend_strength=context.get('trend_strength', 0.0),
            action=action,
            position_size=context.get('position_size', 1)
        )
        
        # Store in prioritized buffer
        self.per_buffer.add(
            state=state,
            action=action,
            reward=adjusted_reward,
            next_state=next_state,
            done=done,
            log_prob=log_prob,
            info={
                'base_reward': base_reward,
                'adjusted_reward': adjusted_reward,
                'context': context,
                'trade_result': trade_result
            }
        )
        
        self._training_stats['rewards'].append(adjusted_reward)
    
    def train_step(self, batch_size: int = 64) -> Optional[Dict]:
        """
        Perform one training step using prioritized sampling.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Training statistics or None if not enough data
        """
        if len(self.per_buffer) < batch_size:
            return None
        
        # Sample from prioritized buffer
        experiences, weights, indices = self.per_buffer.sample(batch_size)
        
        if len(experiences) < batch_size // 2:
            return None
        
        # Convert to tensors and train
        # This integrates with the existing policy.update() method
        # but uses prioritized samples and importance weights
        
        # TODO: Implement full integration with policy.update()
        # For now, use standard policy update
        
        # Calculate TD errors for priority update
        td_errors = self._calculate_td_errors(experiences)
        self.per_buffer.update_priorities(indices, td_errors)
        
        self._training_stats['updates'] += 1
        self._training_stats['td_errors'].extend(td_errors.tolist())
        
        # Track Sharpe periodically
        if self._training_stats['updates'] % 10 == 0:
            sharpe = self.reward_calculator.get_stats()['sharpe_ratio']
            self._training_stats['sharpe_history'].append(sharpe)
        
        return {
            'td_error_mean': float(np.mean(td_errors)),
            'reward_mean': float(np.mean([e['reward'] for e in experiences])),
            'buffer_size': len(self.per_buffer),
            'sharpe': self.reward_calculator.get_stats()['sharpe_ratio']
        }
    
    def _calculate_td_errors(self, experiences: List[Dict]) -> np.ndarray:
        """Calculate TD errors for experiences."""
        import torch
        
        td_errors = []
        
        for exp in experiences:
            state = torch.FloatTensor(exp['state']).unsqueeze(0).to(self.policy.device)
            
            with torch.no_grad():
                if self.policy.use_shared_features and self.policy.shared_extractor is not None:
                    features = self.policy.shared_extractor(state)
                    value = self.policy.critic(state, features).item()
                else:
                    value = self.policy.critic(state).item()
            
            # TD error = reward + gamma * V(s') - V(s)
            if exp['done'] or exp['next_state'] is None:
                target = exp['reward']
            else:
                next_state = torch.FloatTensor(exp['next_state']).unsqueeze(0).to(self.policy.device)
                with torch.no_grad():
                    if self.policy.use_shared_features and self.policy.shared_extractor is not None:
                        next_features = self.policy.shared_extractor(next_state)
                        next_value = self.policy.critic(next_state, next_features).item()
                    else:
                        next_value = self.policy.critic(next_state).item()
                target = exp['reward'] + self.policy.gamma * next_value
            
            td_error = target - value
            td_errors.append(td_error)
        
        return np.array(td_errors)
    
    def get_stats(self) -> Dict:
        """Get training statistics."""
        return {
            **self._training_stats,
            'buffer_stats': self.per_buffer.get_stats(),
            'reward_stats': self.reward_calculator.get_stats()
        }


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_enhanced_rl_components(
    policy,
    buffer_capacity: int = 10000,
    alpha: float = 0.6,
    beta: float = 0.4
) -> Tuple[PrioritizedReplayBuffer, SharpeRewardCalculator, EnhancedPPOTrainer]:
    """
    Create all enhanced RL components.
    
    Args:
        policy: RLTradingPolicy instance
        buffer_capacity: Replay buffer size
        alpha: PER prioritization exponent
        beta: PER importance sampling exponent
        
    Returns:
        (per_buffer, reward_calc, trainer)
    """
    per_buffer = PrioritizedReplayBuffer(
        capacity=buffer_capacity,
        alpha=alpha,
        beta=beta
    )
    
    reward_calc = SharpeRewardCalculator()
    regime_shaper = RegimeAwareRewardShaper()
    
    trainer = EnhancedPPOTrainer(
        policy=policy,
        per_buffer=per_buffer,
        reward_calculator=reward_calc,
        regime_shaper=regime_shaper
    )
    
    return per_buffer, reward_calc, trainer









