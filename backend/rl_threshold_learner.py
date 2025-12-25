#!/usr/bin/env python3
"""
RL-Based Threshold Learning System

Instead of hard-coded thresholds (ALL must pass):
  confidence >= 0.40 AND
  return >= 0.002 AND
  momentum >= 0.001 AND
  volume >= 0.5

Use RL to learn optimal WEIGHTS:
  score = w1*confidence + w2*return + w3*momentum + w4*volume
  trade_if = score >= learned_threshold

The RL agent learns from outcomes which factors matter most!
"""

import os
# GPU acceleration enabled
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


# Feature names for attribution (matches the 16-feature input vector)
FEATURE_NAMES = [
    'confidence',        # 0: Signal confidence (0-1)
    'predicted_return',  # 1: Predicted return magnitude
    'momentum',          # 2: Price momentum
    'volume_spike',      # 3: Volume spike multiplier
    'vix_level',         # 4: VIX level (normalized)
    'vix_bb_pos',        # 5: VIX Bollinger Band position
    'vix_roc',           # 6: VIX rate of change
    'vix_percentile',    # 7: VIX historical percentile
    'hmm_trend',         # 8: HMM trend state (0-2 -> 0-1)
    'hmm_vol',           # 9: HMM volatility state
    'hmm_liq',           # 10: HMM liquidity state
    'time_of_day',       # 11: Normalized time (0=open, 1=close)
    'sector_strength',   # 12: Sector momentum
    'recent_win_rate',   # 13: Recent win rate
    'drawdown',          # 14: Current drawdown level
    'price_jerk',        # 15: Rate of change of acceleration
]


class ThresholdLearner(nn.Module):
    """
    ENHANCED Neural network that learns optimal weights for trading factors.

    Takes signal metrics as input, outputs a composite score AND position sizing.
    Learns from trade outcomes which factors matter most.

    IMPROVED:
    - Expanded inputs (12 features vs 4)
    - Deeper network with layer normalization
    - Dual output: trade score AND position size multiplier
    - Better gradient flow with residual connections
    - Feature attribution via gradient saliency
    """

    def __init__(self, input_dim=12, hidden_dim=64):  # Expanded from 4 to 12 inputs
        super().__init__()
        self.input_dim = input_dim
        
        # Input projection with layer norm
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        
        # Deeper hidden layers with residual connections
        self.hidden1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.15),  # Reduced dropout to retain more signal
        )
        
        self.hidden2 = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(0.1),  # Reduced dropout
        )
        
        # Trade quality score head (0-1)
        self.score_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        # Position sizing head (0.5x to 2x multiplier)
        self.size_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()  # Will scale to 0.5-2.0
        )
        
        # Initialize with reasonable weights
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x, return_size=False):
        """
        Calculate composite trading score from signal metrics
        
        Args:
            x: [confidence, return, momentum, volume, vix, hmm_trend, hmm_vol, hmm_liq,
                time_of_day, sector_strength, recent_win_rate, drawdown]
            return_size: If True, also return position size multiplier
        
        Returns:
            score: 0-1 score indicating trade quality
            size (optional): 0.5-2.0 position size multiplier
        """
        # Project input
        h = self.input_proj(x)
        
        # Hidden layers with residual
        h = h + self.hidden1(h) * 0.1  # Scaled residual
        h = self.hidden2(h)
        
        # Trade quality score
        score = self.score_head(h)
        
        if return_size:
            # Position size: scale sigmoid output to 0.5-2.0 range
            size_raw = self.size_head(h)
            size = 0.5 + size_raw * 1.5  # Maps [0,1] to [0.5, 2.0]
            return score, size

        return score

    def compute_feature_attribution(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute gradient-based feature attribution (saliency).

        Uses integrated gradients approximation: gradient of score w.r.t. input.
        Larger absolute gradient = more important feature for this prediction.

        Args:
            x: Input tensor [input_dim] or [batch, input_dim]

        Returns:
            Attribution tensor with same shape as input (higher = more important)
        """
        # Ensure input requires grad
        x = x.detach().clone()
        x.requires_grad_(True)

        # Forward pass
        score = self.forward(x)

        # Backward pass to get gradients
        if x.grad is not None:
            x.grad.zero_()
        score.sum().backward()

        # Gradient magnitude as attribution
        attribution = x.grad.abs() if x.grad is not None else torch.zeros_like(x)

        return attribution


class RLThresholdPolicy:
    """
    ENHANCED Reinforcement learning policy for threshold optimization.
    
    Issue 2 Fix: Threshold now derives from a canonical base, with RL only
    providing a SMALL delta (±0.03 max) on top of regime-adjusted threshold.
    
    Architecture:
    - base_threshold (from config) = 0.55
    - regime_multiplier (from regime_mapper) = 0.82 to 1.36
    - regime_adjusted = base_threshold * regime_multiplier
    - rl_delta (learned) = -0.03 to +0.03
    - effective_threshold = clamp(regime_adjusted + rl_delta, floor, ceiling)
    
    Learns from trade outcomes:
    - Which factors matter most (expanded from 4 to 12)
    - What composite score threshold to use
    - How to weight each factor
    - IMPROVED: Position sizing based on signal quality
    - IMPROVED: Context-aware decisions (VIX, regime, time)
    """
    
    def __init__(
        self,
        learning_rate=0.003,  # FIXED: Sensible learning rate (was 0.1 = way too fast!)
        device='cpu',
        use_expanded_inputs=True,
        # Issue 2 fix: Centralized threshold management
        base_threshold: float = 0.55,
        rl_delta_max: float = 0.05,  # FIXED: Allow slightly more adaptation range (was 0.03)
        threshold_floor: float = 0.45,  # FIXED: Lower floor to allow more trades (was 0.50)
        threshold_ceiling: float = 0.70
    ):
        self.device = device
        self.use_expanded_inputs = use_expanded_inputs
        
        # Issue 2 fix: Derive from canonical base, not arbitrary starting point
        self.base_threshold = base_threshold
        self.rl_delta_max = rl_delta_max  # RL can only adjust ±3%
        self.threshold_floor = threshold_floor
        self.threshold_ceiling = threshold_ceiling
        
        # Current RL delta (starts at 0, learns small adjustments)
        self.rl_delta = 0.0
        
        # Expanded inputs: 4 base + 12 context = 16 total
        # Added: vix_bb_pos, vix_roc, vix_percentile, price_jerk
        input_dim = 16 if use_expanded_inputs else 4
        self.model = ThresholdLearner(input_dim=input_dim).to(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        
        # Experience replay buffer
        self.experiences = []
        self.max_experiences = 10000
        
        # Track rejected signals to learn from missed opportunities
        self.rejected_signals = []  # Signals that were below threshold
        self.max_rejected = 1000
        
        # Configurable minimum move for counterfactual analysis
        # Need at least this % move in underlying to count as winner/loser
        self.missed_min_move = 0.005  # 0.5% minimum move - increased from 0.3% to focus on bigger plays
        
        # Issue 2 fix: Tighter bounds derived from config
        self.min_threshold = threshold_floor  # Hard floor from config
        self.max_threshold = threshold_ceiling  # Hard ceiling from config
        
        # FIXED: Much smaller learning rate for counterfactual adaptation
        # Previous 0.01 was still causing yo-yo effects
        self.counterfactual_alpha = 0.002  # Very conservative to prevent threshold whiplash
        
        # Cumulative tracking for counterfactual learning
        self.total_missed_winners = 0
        self.total_missed_losers = 0
        self.total_counterfactual_batches = 0
        
        # Store analyzed signals for dashboard visualization (keeps history)
        self.analyzed_signals_history = []  # Signals with their outcomes
        self.max_analyzed_history = 500  # Keep last 500 for plotting
        
        # Legacy: composite_threshold now computed from base + regime + delta
        # This is kept for backward compatibility but computed dynamically
        self.composite_threshold = base_threshold
        self.threshold_adaptation_rate = 0.001  # FIXED: Much slower adaptation (was 0.005)
        
        # Performance tracking
        self.training_stats = {
            'total_decisions': 0,
            'trades_executed': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_updates': 0,
            'avg_score_winners': [],
            'avg_score_losers': []
        }

        # Feature attribution tracking (Phase 14 improvement)
        self.feature_attribution_enabled = True
        self.attribution_history = []  # List of attribution vectors
        self.attribution_by_outcome = {
            'winners': [],  # Attribution for winning trades
            'losers': []    # Attribution for losing trades
        }
        self.max_attribution_history = 500
        self.feature_names = FEATURE_NAMES if use_expanded_inputs else FEATURE_NAMES[:4]

        logger.info("🎯 RL Threshold Learner initialized (FIXED - stable adaptation)")
        logger.info(f"   Device: {device}, LR: {learning_rate} (FIXED from 0.1)")
        logger.info(f"   Base threshold: {base_threshold:.2f}")
        logger.info(f"   RL delta max: ±{rl_delta_max:.2f}")
        logger.info(f"   Effective range: [{threshold_floor:.2f}, {threshold_ceiling:.2f}]")
        logger.info(f"   Counterfactual alpha: {self.counterfactual_alpha} (slow adaptation)")
        logger.info(f"   Threshold adaptation rate: {self.threshold_adaptation_rate}")
    
    def get_effective_threshold(self, regime_multiplier: float = 1.0) -> float:
        """
        Compute effective threshold from base + regime + RL delta.
        
        Issue 2 fix: Single canonical threshold derivation.
        
        Args:
            regime_multiplier: From regime_mapper (0.82 for ultra_low, 1.36 for extreme)
            
        Returns:
            Clamped effective threshold
        """
        regime_adjusted = self.base_threshold * regime_multiplier
        effective = regime_adjusted + self.rl_delta
        
        # Clamp to hard bounds
        clamped = max(self.threshold_floor, min(self.threshold_ceiling, effective))
        
        # Update legacy property for backward compat
        self.composite_threshold = clamped
        
        if clamped != effective:
            logger.debug(f"⚠️ Threshold clamped: {effective:.3f} -> {clamped:.3f}")
        
        return clamped
    
    def evaluate_signal(self, confidence: float, predicted_return: float,
                       momentum: float, volume_spike: float,
                       # Expanded context inputs (12 features)
                       vix_level: float = 18.0,
                       vix_bb_pos: float = 0.5,  # VIX Bollinger Band position (0-1)
                       vix_roc: float = 0.0,  # VIX rate of change
                       vix_percentile: float = 0.5,  # VIX historical percentile
                       hmm_trend: int = 1,  # 0=down, 1=neutral, 2=up
                       hmm_vol: int = 1,  # 0=low, 1=normal, 2=high
                       hmm_liq: int = 1,  # 0=low, 1=normal, 2=high
                       time_of_day: float = 0.5,  # 0=open, 1=close
                       sector_strength: float = 0.5,  # -1 to 1
                       recent_win_rate: float = 0.5,  # 0-1
                       drawdown: float = 0.0,  # 0-1
                       price_jerk: float = 0.0,  # Rate of change of acceleration
                       # For counterfactual learning
                       current_price: float = None,  # Current underlying price
                       timestamp = None,  # Current timestamp
                       proposed_action: str = None  # e.g. "BUY_CALLS", "BUY_PUTS"
                       ) -> Tuple[bool, float, Dict]:
        """
        Evaluate if signal should be traded using learned weights.
        
        Args:
            confidence: Signal confidence (0-1)
            predicted_return: Predicted return (normalized)
            momentum: Momentum value (normalized)
            volume_spike: Volume spike multiplier (normalized)
            vix_level: Current VIX level
            hmm_trend: HMM trend state (0-2)
            hmm_vol: HMM volatility state (0-2)
            hmm_liq: HMM liquidity state (0-2)
            time_of_day: Normalized time (0=open, 1=close)
            sector_strength: Sector momentum (-1 to 1)
            recent_win_rate: Recent trade win rate (0-1)
            drawdown: Current drawdown level (0-1)
        
        Returns:
            should_trade: Boolean decision
            composite_score: Calculated score (0-1)
            details: Dictionary with breakdown
        """
        self.training_stats['total_decisions'] += 1
        
        # Base 4 inputs (always used)
        base_inputs = [
            min(1.0, max(0.0, confidence)),  # Already 0-1
            min(1.0, max(0.0, abs(predicted_return) * 100)),  # Scale to 0-1
            min(1.0, max(0.0, abs(momentum) * 500)),  # Scale to 0-1
            min(1.0, max(0.0, volume_spike))  # Already ratio
        ]
        
        # Expanded context inputs (12 features for 16 total)
        if self.use_expanded_inputs:
            expanded_inputs = [
                min(1.0, vix_level / 50.0),  # VIX normalized (0-50 -> 0-1)
                min(1.0, max(0.0, vix_bb_pos)),  # VIX BB position (already 0-1)
                min(1.0, max(-1.0, vix_roc)) / 2.0 + 0.5,  # VIX ROC (-1,1 -> 0-1)
                min(1.0, max(0.0, vix_percentile)),  # VIX percentile (already 0-1)
                hmm_trend / 2.0,  # Trend state (0-2 -> 0-1)
                hmm_vol / 2.0,  # Volatility state (0-2 -> 0-1)
                hmm_liq / 2.0,  # Liquidity state (0-2 -> 0-1)
                min(1.0, max(0.0, time_of_day)),  # Time of day (0-1)
                (sector_strength + 1.0) / 2.0,  # Sector strength (-1,1 -> 0-1)
                min(1.0, max(0.0, recent_win_rate)),  # Win rate (0-1)
                min(1.0, max(0.0, drawdown)),  # Drawdown (0-1)
                min(1.0, max(-1.0, price_jerk * 1000)) / 2.0 + 0.5  # Jerk normalized (-1,1 -> 0-1)
            ]
            all_inputs = base_inputs + expanded_inputs
        else:
            all_inputs = base_inputs
        
        inputs = torch.tensor(all_inputs, dtype=torch.float32).to(self.device)
        
        # Get composite score and optional position size from neural network
        with torch.no_grad():
            if self.use_expanded_inputs:
                composite_score, position_size = self.model(inputs, return_size=True)
                composite_score = composite_score.item()
                position_size = position_size.item()
            else:
                composite_score = self.model(inputs).item()
                position_size = 1.0  # Default size

        # Compute feature attribution (Phase 14 improvement)
        attribution = None
        if self.feature_attribution_enabled:
            try:
                attribution = self.model.compute_feature_attribution(inputs).detach().cpu().numpy()
                self.attribution_history.append(attribution)
                # Keep history bounded
                if len(self.attribution_history) > self.max_attribution_history:
                    self.attribution_history.pop(0)
            except Exception as e:
                logger.debug(f"Attribution computation failed: {e}")
        
        # Get effective threshold (Issue 2 fix: derives from base + regime + RL delta)
        # Extract regime multiplier from context (default to 1.0 if not provided)
        regime_mult = 1.0
        if self.use_expanded_inputs and vix_level > 0:
            # Infer regime multiplier from VIX (simplified)
            if vix_level < 12:
                regime_mult = 0.82
            elif vix_level < 15:
                regime_mult = 0.91
            elif vix_level < 20:
                regime_mult = 1.0
            elif vix_level < 25:
                regime_mult = 1.09
            elif vix_level < 35:
                regime_mult = 1.18
            else:
                regime_mult = 1.36
        
        effective_threshold = self.get_effective_threshold(regime_mult)
        
        # FIXED: Exploration logic - don't force trades, just log the situation
        # Previous version lowered thresholds too aggressively after 50 checks
        # This caused taking bad trades just because we hadn't traded in a while
        decisions_without_trade = self.training_stats['total_decisions'] - self.training_stats.get('last_trade_decision', 0)
        
        if decisions_without_trade > 100:
            # Only log after 100 decisions without a trade (not 50)
            # Don't lower threshold - let the market bring good opportunities
            if decisions_without_trade % 50 == 0:  # Log every 50 decisions
                logger.info(f"ℹ️ {decisions_without_trade} signals evaluated without trading. Waiting for quality setup (threshold={effective_threshold:.3f})")
        
        # Simple threshold comparison - no forced exploration
        should_trade = composite_score >= effective_threshold
        
        if should_trade:
            self.training_stats['trades_executed'] += 1
            self.training_stats['last_trade_decision'] = self.training_stats['total_decisions']
        else:
            # Store rejected signal to learn from missed opportunities later
            # Include price info so we can check actual outcome
            self.rejected_signals.append({
                'timestamp': timestamp,  # When signal was rejected
                'price_at_signal': current_price,  # Market price at rejection
                'confidence': confidence,
                'predicted_return': predicted_return,
                'momentum': momentum,
                'volume': volume_spike,
                'composite_score': composite_score,
                'threshold': self.composite_threshold,
                'proposed_action': proposed_action,  # BUY_CALLS, BUY_PUTS, etc.
                'actual_return': None,  # Will be filled in later
                'was_checked': False
            })
            
            # Limit rejected signals buffer
            if len(self.rejected_signals) > self.max_rejected:
                self.rejected_signals.pop(0)
        
        details = {
            'composite_score': composite_score,
            'threshold': effective_threshold,  # Issue 2 fix: use effective, not raw
            'base_threshold': self.base_threshold,
            'regime_multiplier': regime_mult,
            'rl_delta': self.rl_delta,
            'threshold_floor': self.threshold_floor,
            'threshold_ceiling': self.threshold_ceiling,
            'position_size_multiplier': position_size,  # Recommended position sizing
            'confidence': confidence,
            'return': predicted_return,
            'momentum': momentum,
            'volume': volume_spike,
            'vix_level': vix_level,
            'hmm_regime': f"trend={hmm_trend}/vol={hmm_vol}/liq={hmm_liq}",
            'decision': 'TRADE' if should_trade else 'HOLD',
            'model_inputs': all_inputs,  # Store inputs for training
            'feature_attribution': attribution.tolist() if attribution is not None else None
        }

        return should_trade, composite_score, details
    
    def store_experience(self, confidence: float, predicted_return: float,
                        momentum: float, volume_spike: float,
                        composite_score: float, outcome_pnl: float,
                        model_inputs: List[float] = None,
                        feature_attribution: List[float] = None):
        """
        Store trade experience for learning.

        Args:
            confidence, predicted_return, momentum, volume_spike: Signal metrics
            composite_score: Score that was calculated
            outcome_pnl: Actual P&L from trade
            model_inputs: Full input vector used for inference (preferred)
            feature_attribution: Attribution scores for each feature (Phase 14)
        """
        # Use provided model inputs if available (supports expanded features)
        if model_inputs is not None:
            inputs = model_inputs
        else:
            # Fallback for legacy calls (only 4 features)
            inputs = [confidence, abs(predicted_return), abs(momentum), volume_spike]
            # Pad with zeros if model expects more
            if len(inputs) < self.model.input_dim:
                inputs += [0.0] * (self.model.input_dim - len(inputs))

        experience = {
            'inputs': inputs,
            'score': composite_score,
            'pnl': outcome_pnl,
            'won': outcome_pnl > 0,
            'attribution': feature_attribution
        }

        self.experiences.append(experience)

        # Track winning vs losing scores
        if outcome_pnl > 0:
            self.training_stats['winning_trades'] += 1
            self.training_stats['avg_score_winners'].append(composite_score)
            # Track attribution for winners (Phase 14)
            if feature_attribution is not None:
                self.attribution_by_outcome['winners'].append(feature_attribution)
                if len(self.attribution_by_outcome['winners']) > 100:
                    self.attribution_by_outcome['winners'].pop(0)
        else:
            self.training_stats['losing_trades'] += 1
            self.training_stats['avg_score_losers'].append(composite_score)
            # Track attribution for losers (Phase 14)
            if feature_attribution is not None:
                self.attribution_by_outcome['losers'].append(feature_attribution)
                if len(self.attribution_by_outcome['losers']) > 100:
                    self.attribution_by_outcome['losers'].pop(0)

        # Limit buffer size
        if len(self.experiences) > self.max_experiences:
            self.experiences.pop(0)

        # Keep only recent averages (last 100 trades)
        if len(self.training_stats['avg_score_winners']) > 100:
            self.training_stats['avg_score_winners'].pop(0)
        if len(self.training_stats['avg_score_losers']) > 100:
            self.training_stats['avg_score_losers'].pop(0)
    
    def train_from_experiences(self, batch_size=32):
        """
        Train the model from stored experiences.
        Uses supervised learning: learn to output higher scores for winning trades.
        """
        if len(self.experiences) < batch_size:
            return
        
        # Sample random batch
        indices = np.random.choice(len(self.experiences), batch_size, replace=False)
        batch = [self.experiences[i] for i in indices]
        
        # Prepare training data
        inputs = torch.tensor([exp['inputs'] for exp in batch], dtype=torch.float32).to(self.device)
        
        # Target: 1.0 for winners, 0.0 for losers (binary classification)
        targets = torch.tensor([1.0 if exp['won'] else 0.0 for exp in batch], 
                              dtype=torch.float32).unsqueeze(1).to(self.device)
        
        # Forward pass
        predictions = self.model(inputs)
        
        # Loss: Binary cross-entropy (learn to predict winners vs losers)
        loss = nn.BCELoss()(predictions, targets)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.training_stats['total_updates'] += 1
        
        # Adapt composite threshold based on win rate
        self._adapt_threshold()
        
        if self.training_stats['total_updates'] % 10 == 0:
            logger.info(f"🎓 RL Threshold trained: loss={loss.item():.4f}, threshold={self.composite_threshold:.3f}")
    
    def learn_from_missed_opportunities(self, missed_winners: int, missed_losers: int):
        """
        Learn from signals we rejected but should/shouldn't have traded.
        
        Issue 2 fix: Adjusts rl_delta, not composite_threshold directly.
        
        Uses a smooth, ratio-based approach with small learning rate to avoid
        threshold yo-yo effects from one weird batch.
        
        Args:
            missed_winners: Count of rejected signals that would have been profitable
            missed_losers: Count of rejected signals that would have lost money
        
        Adaptation logic:
        - Treat missed_winners vs missed_losers as a ratio
        - Target: 50% (equal filtering of winners/losers means threshold is calibrated)
        - Error > 0: missing too many winners → lower rl_delta
        - Error < 0: filtering correctly → raise rl_delta slightly
        """
        total_missed = missed_winners + missed_losers
        if total_missed == 0:
            return
        
        # Update cumulative tracking
        self.total_missed_winners += missed_winners
        self.total_missed_losers += missed_losers
        self.total_counterfactual_batches += 1
        
        # Calculate missed winner rate
        missed_winner_rate = missed_winners / total_missed
        
        # Target: 50% means threshold is well-calibrated
        # If we're missing mostly winners (>50%), we're too conservative
        # If we're avoiding mostly losers (<50%), we're filtering correctly
        target = 0.5
        
        # Signed error: positive = too conservative, negative = good filtering
        error = missed_winner_rate - target
        
        # Store old delta for logging
        old_delta = self.rl_delta
        
        # Issue 2 fix: Adjust rl_delta, not composite_threshold
        # Lower delta if missing winners, raise if filtering correctly
        self.rl_delta -= self.counterfactual_alpha * error
        
        # Clamp delta to bounds
        self.rl_delta = max(-self.rl_delta_max, min(self.rl_delta_max, self.rl_delta))
        
        # Log the outcome with context
        effective = self.get_effective_threshold()
        if missed_winner_rate > 0.6:
            logger.warning(f"⚠️  MISSED {missed_winners}/{total_missed} WINNERS ({missed_winner_rate:.0%})! "
                          f"RL delta: {old_delta:.3f} → {self.rl_delta:.3f} (effective: {effective:.3f})")
        elif missed_winner_rate < 0.4:
            logger.info(f"✅ Good filtering: {missed_losers}/{total_missed} avoided losses ({1-missed_winner_rate:.0%}). "
                       f"RL delta: {old_delta:.3f} → {self.rl_delta:.3f} (effective: {effective:.3f})")
        else:
            logger.info(f"⚖️  Balanced: {missed_winners}W/{missed_losers}L. "
                       f"RL delta: {old_delta:.3f} → {self.rl_delta:.3f} (effective: {effective:.3f})")
    
    def analyze_missed_opportunities(self, current_price: float, current_timestamp=None, 
                                       lookback_minutes: int = 30):
        """
        Analyze rejected signals to see what would have happened if we had traded.
        
        This is the COUNTERFACTUAL learning mechanism - it checks what the market
        actually did after we decided to HOLD, and learns from the outcome.
        
        Args:
            current_price: Current underlying price (e.g., SPY price)
            current_timestamp: Current timestamp (for time-based filtering)
            lookback_minutes: How many minutes back to look for matured signals
            
        Returns:
            dict: Statistics about missed opportunities
        """
        if not self.rejected_signals:
            return {'missed_winners': 0, 'missed_losers': 0, 'pending': 0, 'total_analyzed': 0}
        
        missed_winners = 0
        missed_losers = 0
        pending = 0
        signals_to_analyze = []
        
        for signal in self.rejected_signals:
            # Skip already-checked signals
            if signal.get('was_checked', False):
                continue
            
            # Skip if no price at signal time
            if signal.get('price_at_signal') is None:
                pending += 1
                continue
            
            # Skip if signal is too recent (need time for outcome to develop)
            if current_timestamp and signal.get('timestamp'):
                from datetime import datetime, timedelta
                if isinstance(signal['timestamp'], str):
                    signal_time = datetime.fromisoformat(signal['timestamp'])
                else:
                    signal_time = signal['timestamp']
                
                if current_timestamp - signal_time < timedelta(minutes=lookback_minutes):
                    pending += 1
                    continue
            
            signals_to_analyze.append(signal)
        
        # Analyze matured signals
        for signal in signals_to_analyze:
            entry_price = signal['price_at_signal']
            
            # Calculate underlying return (always positive = up, negative = down)
            underlying_return = (current_price - entry_price) / entry_price
            
            # Determine direction based on proposed_action
            # For CALLs: up move is good
            # For PUTs: down move is good
            proposed_action = signal.get('proposed_action', '')
            
            # Classify as CALL-type or PUT-type based on action string
            is_call_type = proposed_action in ('BUY_CALL', 'BUY_CALLS', 'CALL', 'CALLS')
            is_put_type = proposed_action in ('BUY_PUT', 'BUY_PUTS', 'PUT', 'PUTS')
            
            # If no action stored, fall back to predicted_return direction
            if not is_call_type and not is_put_type:
                predicted_dir = signal.get('predicted_return', 0)
                is_call_type = predicted_dir > 0
                is_put_type = predicted_dir < 0
            
            signal['was_checked'] = True
            
            # Use configurable minimum move threshold
            min_move = self.missed_min_move  # e.g., 0.003 = 0.3%
            
            # Determine if this was a missed winner or correctly avoided loser
            if is_call_type:
                # CALL: up move = winner, down move = loser
                if underlying_return >= min_move:
                    missed_winners += 1
                    signal['actual_return'] = underlying_return * 100
                    logger.debug(f"📈 MISSED CALL WINNER: Underlying UP {underlying_return*100:+.2f}% "
                               f"(score: {signal['composite_score']:.3f}, threshold: {signal['threshold']:.3f})")
                elif underlying_return <= -min_move:
                    missed_losers += 1
                    signal['actual_return'] = underlying_return * 100
                    logger.debug(f"✅ AVOIDED CALL LOSER: Underlying DOWN {underlying_return*100:+.2f}% "
                               f"(score: {signal['composite_score']:.3f})")
                # else: break-even, don't count
                    
            elif is_put_type:
                # PUT: down move = winner, up move = loser
                if underlying_return <= -min_move:
                    missed_winners += 1
                    signal['actual_return'] = -underlying_return * 100  # Invert for PUT P&L
                    logger.debug(f"📈 MISSED PUT WINNER: Underlying DOWN {underlying_return*100:+.2f}% "
                               f"(score: {signal['composite_score']:.3f}, threshold: {signal['threshold']:.3f})")
                elif underlying_return >= min_move:
                    missed_losers += 1
                    signal['actual_return'] = -underlying_return * 100  # Invert for PUT P&L
                    logger.debug(f"✅ AVOIDED PUT LOSER: Underlying UP {underlying_return*100:+.2f}% "
                               f"(score: {signal['composite_score']:.3f})")
                # else: break-even, don't count
        
        # If we have enough data, learn from it
        total_analyzed = missed_winners + missed_losers
        if total_analyzed >= 5:  # Need at least 5 signals to draw conclusions
            self.learn_from_missed_opportunities(missed_winners, missed_losers)
            
            # Also train the network on missed opportunities
            # Teach it: high-score signals that would have won should score even higher
            # Teach it: high-score signals that would have lost should score lower
            # NOTE: actual_return is stored as percentage (e.g., 0.5 = 0.5%)
            # Positive = winner, Negative = loser (already adjusted for CALL/PUT direction)
            min_move_pct = self.missed_min_move * 100  # Convert to percentage
            
            for signal in signals_to_analyze:
                if signal.get('actual_return') is None:
                    continue
                    
                # Create pseudo-experience for training
                if signal['actual_return'] > min_move_pct:
                    # This was a good signal we missed - train to increase score
                    self._train_on_missed_winner(signal)
                elif signal['actual_return'] < -min_move_pct:
                    # Good that we avoided this - reinforce the rejection
                    self._train_on_avoided_loser(signal)
        
        # Store analyzed signals in history for dashboard visualization BEFORE cleanup
        for signal in signals_to_analyze:
            if signal.get('actual_return') is not None:
                # Determine if it was a missed winner or avoided loser
                is_winner = signal['actual_return'] > min_move_pct
                is_loser = signal['actual_return'] < -min_move_pct
                
                self.analyzed_signals_history.append({
                    'timestamp': str(signal.get('timestamp', '')),
                    'price': signal.get('price_at_signal', 0),
                    'action': signal.get('proposed_action', ''),
                    'confidence': signal.get('confidence', 0),
                    'composite_score': signal.get('composite_score', 0),
                    'threshold': signal.get('threshold', 0),
                    'actual_return': signal['actual_return'],
                    'outcome': 'missed_winner' if is_winner else ('avoided_loser' if is_loser else 'breakeven')
                })
        
        # Trim history to max size
        if len(self.analyzed_signals_history) > self.max_analyzed_history:
            self.analyzed_signals_history = self.analyzed_signals_history[-self.max_analyzed_history:]
        
        # Clean up old analyzed signals
        self.rejected_signals = [s for s in self.rejected_signals 
                                 if not s.get('was_checked', False)]
        
        return {
            'missed_winners': missed_winners,
            'missed_losers': missed_losers,
            'pending': pending,
            'total_analyzed': total_analyzed
        }
    
    def _train_on_missed_winner(self, signal: Dict):
        """Train the network that this signal should have scored higher"""
        # Create a synthetic experience with positive reward
        self.store_experience(
            confidence=signal['confidence'],
            predicted_return=signal.get('predicted_return', 0),
            momentum=signal.get('momentum', 0),
            volume_spike=signal.get('volume', 1.0),
            composite_score=signal.get('composite_score', 0.5),
            outcome_pnl=abs(signal['actual_return']) * 10  # Scale for learning, positive
        )

    def _train_on_avoided_loser(self, signal: Dict):
        """Reinforce that rejecting this signal was correct"""
        # Record as if we had traded and lost (so network learns to avoid)
        self.store_experience(
            confidence=signal['confidence'],
            predicted_return=signal.get('predicted_return', 0),
            momentum=signal.get('momentum', 0),
            volume_spike=signal.get('volume', 1.0),
            composite_score=signal.get('composite_score', 0.5),
            outcome_pnl=-abs(signal['actual_return']) * 10  # Negative, so it learns to avoid
        )
    
    def _adapt_threshold(self):
        """
        Adapt the RL delta based on performance.
        
        Issue 2 fix: Only adjusts rl_delta within ±rl_delta_max bounds.
        Does NOT directly set composite_threshold.
        
        If winning trades have higher scores than losing trades,
        we can keep the threshold. Otherwise, adjust delta.
        """
        if (len(self.training_stats['avg_score_winners']) < 10 or 
            len(self.training_stats['avg_score_losers']) < 10):
            return
        
        avg_winner_score = np.mean(self.training_stats['avg_score_winners'][-50:])
        avg_loser_score = np.mean(self.training_stats['avg_score_losers'][-50:])
        
        old_delta = self.rl_delta
        
        # If winners and losers have similar scores, increase selectivity (raise delta)
        if abs(avg_winner_score - avg_loser_score) < 0.1:
            self.rl_delta = min(self.rl_delta_max, self.rl_delta + self.threshold_adaptation_rate)
            if self.rl_delta != old_delta:
                logger.info(f"📈 Raising RL delta: {old_delta:.3f} -> {self.rl_delta:.3f} (scores too similar)")
        
        # If winners have much higher scores, we can lower threshold to trade more
        elif avg_winner_score - avg_loser_score > 0.3:
            self.rl_delta = max(-self.rl_delta_max, self.rl_delta - self.threshold_adaptation_rate)
            if self.rl_delta != old_delta:
                logger.info(f"📉 Lowering RL delta: {old_delta:.3f} -> {self.rl_delta:.3f} (clear winner signal)")
        
        # Log warning if hitting bounds
        if abs(self.rl_delta) >= self.rl_delta_max:
            logger.warning(f"⚠️ RL delta at bound: {self.rl_delta:.3f} (max ±{self.rl_delta_max:.3f})")

    def get_feature_importance(self) -> Dict:
        """
        Get feature importance based on gradient attribution (Phase 14).

        Returns a dictionary with:
        - overall_importance: Average attribution across all evaluations
        - winner_importance: Average attribution for winning trades
        - loser_importance: Average attribution for losing trades
        - differential_importance: Winner - Loser (positive = predicts winners)
        - ranked_features: List of (feature_name, importance) sorted by differential

        Higher differential importance means the feature is MORE important
        for identifying winning trades vs losing trades.
        """
        result = {
            'overall_importance': {},
            'winner_importance': {},
            'loser_importance': {},
            'differential_importance': {},
            'ranked_features': [],
            'sample_counts': {
                'total': len(self.attribution_history),
                'winners': len(self.attribution_by_outcome['winners']),
                'losers': len(self.attribution_by_outcome['losers'])
            }
        }

        # Need at least some data
        if not self.attribution_history:
            return result

        # Compute overall importance (average attribution)
        overall_attr = np.array(self.attribution_history)
        overall_mean = overall_attr.mean(axis=0)
        for i, name in enumerate(self.feature_names):
            if i < len(overall_mean):
                result['overall_importance'][name] = float(overall_mean[i])

        # Compute importance for winners
        if self.attribution_by_outcome['winners']:
            winner_attr = np.array(self.attribution_by_outcome['winners'])
            winner_mean = winner_attr.mean(axis=0)
            for i, name in enumerate(self.feature_names):
                if i < len(winner_mean):
                    result['winner_importance'][name] = float(winner_mean[i])

        # Compute importance for losers
        if self.attribution_by_outcome['losers']:
            loser_attr = np.array(self.attribution_by_outcome['losers'])
            loser_mean = loser_attr.mean(axis=0)
            for i, name in enumerate(self.feature_names):
                if i < len(loser_mean):
                    result['loser_importance'][name] = float(loser_mean[i])

        # Compute differential importance (winner - loser)
        # Positive = feature is MORE important for winners
        if result['winner_importance'] and result['loser_importance']:
            for name in self.feature_names:
                w_imp = result['winner_importance'].get(name, 0)
                l_imp = result['loser_importance'].get(name, 0)
                result['differential_importance'][name] = w_imp - l_imp

            # Rank features by differential importance
            result['ranked_features'] = sorted(
                result['differential_importance'].items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )

        return result

    def get_stats(self) -> Dict:
        """Get training statistics including counterfactual learning metrics"""
        win_rate = (self.training_stats['winning_trades'] / 
                   max(1, self.training_stats['winning_trades'] + self.training_stats['losing_trades']))
        
        # Calculate cumulative counterfactual stats
        total_counterfactual = self.total_missed_winners + self.total_missed_losers
        counterfactual_miss_rate = (self.total_missed_winners / total_counterfactual 
                                    if total_counterfactual > 0 else 0)
        
        return {
            # Core trading stats
            'total_decisions': self.training_stats['total_decisions'],
            'trades_executed': self.training_stats['trades_executed'],
            'win_rate': win_rate,
            'total_updates': self.training_stats['total_updates'],
            'avg_winner_score': np.mean(self.training_stats['avg_score_winners']) if self.training_stats['avg_score_winners'] else 0,
            'avg_loser_score': np.mean(self.training_stats['avg_score_losers']) if self.training_stats['avg_score_losers'] else 0,
            
            # Issue 2 fix: Centralized threshold stats
            'base_threshold': self.base_threshold,
            'rl_delta': self.rl_delta,
            'rl_delta_max': self.rl_delta_max,
            'effective_threshold': self.get_effective_threshold(),
            'threshold_floor': self.threshold_floor,
            'threshold_ceiling': self.threshold_ceiling,
            'composite_threshold': self.composite_threshold,  # Backward compat (same as effective)
            
            # Counterfactual learning stats
            'total_missed_winners': self.total_missed_winners,
            'total_missed_losers': self.total_missed_losers,
            'counterfactual_batches': self.total_counterfactual_batches,
            'counterfactual_miss_rate': counterfactual_miss_rate,  # Higher = too conservative
            'pending_rejected_signals': len(self.rejected_signals),

            # Threshold bounds (for monitoring)
            'min_threshold': self.min_threshold,
            'max_threshold': self.max_threshold,

            # Feature attribution stats (Phase 14)
            'feature_attribution': self.get_feature_importance()
        }
    
    def get_analyzed_signals(self, limit: int = 200) -> List[Dict]:
        """
        Get analyzed signals history for dashboard visualization.
        
        Returns list of signals with their outcomes (missed_winner, avoided_loser, breakeven).
        """
        return self.analyzed_signals_history[-limit:]
    
    def save(self, path: str):
        """Save model and stats including counterfactual learning state"""
        torch.save({
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'threshold': self.composite_threshold,
            'stats': self.training_stats,
            'version': 'enhanced_v4',  # Bumped for Issue 2 threshold management
            'use_expanded_inputs': self.use_expanded_inputs,
            'input_dim': self.model.input_dim,
            # Issue 2 fix: Centralized threshold state
            'threshold_management': {
                'base_threshold': self.base_threshold,
                'rl_delta': self.rl_delta,
                'rl_delta_max': self.rl_delta_max,
                'threshold_floor': self.threshold_floor,
                'threshold_ceiling': self.threshold_ceiling,
            },
            # Counterfactual learning state
            'counterfactual': {
                'total_missed_winners': self.total_missed_winners,
                'total_missed_losers': self.total_missed_losers,
                'total_batches': self.total_counterfactual_batches,
                'min_threshold': self.min_threshold,
                'max_threshold': self.max_threshold,
                'alpha': self.counterfactual_alpha,
                'min_move': self.missed_min_move
            },
            # Feature attribution state (Phase 14)
            'feature_attribution': {
                'attribution_by_outcome': self.attribution_by_outcome,
                'feature_names': self.feature_names
            }
        }, path)
        logger.info(f"💾 RL Threshold Learner saved to {path}")
    
    def load(self, path: str):
        """Load model and stats with architecture migration support"""
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            
            # Check saved model dimensions
            saved_state = checkpoint.get('model_state', {})
            saved_input_dim = checkpoint.get('input_dim', 4)  # Default to old 4-input model
            current_input_dim = self.model.input_dim
            
            # Try to infer input dim from saved weights if not stored
            if 'input_proj.0.weight' in saved_state:
                saved_input_dim = saved_state['input_proj.0.weight'].shape[1]
            elif 'network.0.weight' in saved_state:
                saved_input_dim = saved_state['network.0.weight'].shape[1]
            
            # Load weights if dimensions match
            if saved_input_dim == current_input_dim:
                try:
                    self.model.load_state_dict(saved_state, strict=False)
                    logger.info(f"   ✅ Loaded model weights (input_dim={saved_input_dim})")
                except Exception as e:
                    logger.warning(f"   ⚠️ Partial weight load: {e}")
            else:
                logger.warning(f"⚠️ Model architecture changed (saved: {saved_input_dim}, current: {current_input_dim})")
                logger.warning(f"   Starting fresh model but keeping stats")
            
            # Load optimizer state
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            except Exception:
                logger.warning("   ⚠️ Could not load optimizer state, using fresh optimizer")
            
            self.composite_threshold = checkpoint.get('threshold', self.composite_threshold)
            self.training_stats = checkpoint.get('stats', self.training_stats)
            
            # Issue 2 fix: Load threshold management state if available
            tm_state = checkpoint.get('threshold_management', {})
            if tm_state:
                self.base_threshold = tm_state.get('base_threshold', self.base_threshold)
                self.rl_delta = tm_state.get('rl_delta', 0.0)
                self.rl_delta_max = tm_state.get('rl_delta_max', self.rl_delta_max)
                self.threshold_floor = tm_state.get('threshold_floor', self.threshold_floor)
                self.threshold_ceiling = tm_state.get('threshold_ceiling', self.threshold_ceiling)
            else:
                # Migrate from old format: infer rl_delta from composite_threshold
                old_threshold = checkpoint.get('threshold', 0.55)
                self.rl_delta = max(-self.rl_delta_max, min(self.rl_delta_max, 
                                    old_threshold - self.base_threshold))
                logger.info(f"   ℹ️ Migrated from v3: inferred rl_delta={self.rl_delta:.3f}")
            
            # Load counterfactual learning state if available
            cf_state = checkpoint.get('counterfactual', {})
            if cf_state:
                self.total_missed_winners = cf_state.get('total_missed_winners', 0)
                self.total_missed_losers = cf_state.get('total_missed_losers', 0)
                self.total_counterfactual_batches = cf_state.get('total_batches', 0)
                self.min_threshold = cf_state.get('min_threshold', self.min_threshold)
                self.max_threshold = cf_state.get('max_threshold', self.max_threshold)
                self.counterfactual_alpha = cf_state.get('alpha', self.counterfactual_alpha)
                self.missed_min_move = cf_state.get('min_move', self.missed_min_move)

            # Load feature attribution state if available (Phase 14)
            fa_state = checkpoint.get('feature_attribution', {})
            if fa_state:
                self.attribution_by_outcome = fa_state.get('attribution_by_outcome', self.attribution_by_outcome)
                # Feature names may have changed, so we keep current names

            logger.info(f"✅ RL Threshold Learner loaded from {path}")
            logger.info(f"   Base threshold: {self.base_threshold:.3f}, RL delta: {self.rl_delta:.3f}")
            logger.info(f"   Effective threshold: {self.get_effective_threshold():.3f}")
            logger.info(f"   Updates: {self.training_stats['total_updates']}")
            if cf_state:
                logger.info(f"   Counterfactual: {self.total_missed_winners}W / {self.total_missed_losers}L over {self.total_counterfactual_batches} batches")
            if fa_state:
                w_count = len(self.attribution_by_outcome.get('winners', []))
                l_count = len(self.attribution_by_outcome.get('losers', []))
                logger.info(f"   Feature Attribution: {w_count} winners, {l_count} losers tracked")
        except FileNotFoundError:
            logger.info(f"No saved model found at {path}, starting fresh")
        except Exception as e:
            logger.error(f"Error loading RL threshold learner: {e}")

