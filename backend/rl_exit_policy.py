#!/usr/bin/env python3
"""
RL-Based Exit Policy for Options Trading (PREDICTION-AWARE)

SIMPLIFIED VERSION - Based on output2's better-performing approach

Critical for options because:
- Time decay (theta) eats profits every minute
- Need to let predicted moves play out
- Should cut losses faster than with stocks
- Different timeframes need different holding periods

Strategy (3 Phases):
1. Before 25% of prediction: HOLD (too early)
2. Before 75% of prediction: HOLD (unless emergency stop/take profit)
3. After 75% of prediction: RL + Rules decide

Example: 30-minute prediction
- Minutes 0-7: Always hold
- Minutes 8-22: Hold unless P&L < -20% or > +50%
- Minutes 23+: RL evaluates whether to exit or hold longer

Learns:
- When to take profits after prediction window
- When to cut losses early vs ride it out
- When to hold past prediction for bigger moves
- How to balance time decay vs movement continuation
"""

import os
# GPU acceleration enabled
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ExitPolicyNetwork(nn.Module):
    """
    ENHANCED Neural network that learns optimal exit timing for options trades.
    
    Takes into account:
    - Predicted timeframe (15min, 30min, 1hr, 4hr)
    - Current P&L
    - Time held so far
    - Days to expiration (theta decay factor)
    - How much of predicted move has happened
    - IMPROVED: Market context (IV, momentum, regime)
    - IMPROVED: Greeks exposure
    - IMPROVED: Deeper network with residual connections
    - NEW: Position direction vs signal direction alignment
    """
    
    def __init__(self, input_dim=14):  # Expanded from 12 to 14 inputs (added position-signal alignment)
        super().__init__()
        
        # Deeper network with residual-style connections
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
        )
        
        # Hidden block with residual connection
        self.hidden1 = nn.Sequential(
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.25),
        )
        
        self.hidden2 = nn.Sequential(
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        
        # Output heads
        self.exit_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()  # 0-1 score: 0=hold, 1=exit
        )
        
        # NEW: Optimal hold time prediction (in minutes)
        self.hold_time_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Softplus()  # Non-negative output
        )
        
    def forward(self, x, return_hold_time=False):
        # Project input
        h = self.input_proj(x)
        
        # Hidden with residual
        h = h + self.hidden1(h) * 0.1  # Scaled residual
        h = self.hidden2(h)
        
        exit_score = self.exit_head(h)
        
        if return_hold_time:
            hold_time = self.hold_time_head(h)
            return exit_score, hold_time
        
        return exit_score


class RLExitPolicy:
    """
    Reinforcement learning policy for optimal exit timing on options trades.
    
    Key insight: Options have time decay, so holding too long = losing money
    even if direction is right!
    
    SIMPLIFIED: Removed Thompson Sampling, trailing stops, momentum tracking,
    and VIX-adaptive stops for better training convergence.
    """
    
    def __init__(self, learning_rate=0.001, device='cpu', use_online_learning=False, use_expanded_inputs=True):
        # FIXED: Sensible learning rate (was 0.2 = way too high!)
        self.device = device
        self.use_expanded_inputs = use_expanded_inputs
        self.learning_rate = learning_rate
        # Expanded inputs: 8 base + 6 market context + 3 HMM + 3 VIX extended = 20
        input_dim = 20 if use_expanded_inputs else 8
        self.model = ExitPolicyNetwork(input_dim=input_dim).to(device)
        # FIXED: Proper weight decay for regularization (was 0 = overfitting risk)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        
        # FIXED: Learning rate scheduler with sensible floor (was min_lr=0.01, now 0.0001)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=100, min_lr=0.0001
        )
        
        # Online learning disabled in simplified version
        self.use_online_learning = False
        self.online_learner = None
        
        # Experience buffer - larger for better learning
        self.experiences = []
        self.max_experiences = 10000  # Increased from 5000
        
        # Prioritized experience replay weights
        self.experience_priorities = []
        
        # Stats
        self.stats = {
            'total_exits': 0,
            'early_exits_won': 0,
            'early_exits_lost': 0,
            'full_hold_won': 0,
            'full_hold_lost': 0,
            'avg_hold_time_winners': [],
            'avg_hold_time_losers': [],
            'avg_pnl_winners': [],
            'avg_pnl_losers': [],
            'training_losses': [],
            'best_trade_pnl': 0.0,
            'worst_trade_pnl': 0.0,
        }
        
        # Update counter for compatibility
        self.update_count = 0
        self.train_count = 0
        
        logger.info(f"🚪 RL Exit Policy initialized (FIXED - stable learning)")
        logger.info(f"   LR: {learning_rate} (FIXED from 0.2)")
        logger.info(f"   Min hold: 10 minutes (anti-churn)")
        logger.info(f"   Profit target: 40%, Stop loss: -20%")
    
    def should_exit(self, 
                   prediction_timeframe_minutes: int,
                   time_held_minutes: int,
                   current_pnl_pct: float,
                   days_to_expiration: int,
                   predicted_move_pct: float,
                   actual_move_pct: float,
                   entry_confidence: float,
                   # LIVE PREDICTION PARAMETERS - NOW USED!
                   live_predicted_move_pct: Optional[float] = None,
                   live_confidence: Optional[float] = None,
                   live_hmm_state: Optional[str] = None,
                   entry_hmm_state: Optional[str] = None,
                   trade_id: Optional[str] = None,
                   vix_level: float = 18.0,
                   bid_ask_spread_pct: float = 0.0,
                   hmm_regime_info: Optional[Dict] = None,
                   # NEW: Position direction awareness
                   position_is_call: Optional[bool] = None,
                   bullish_timeframe_ratio: float = 0.5,  # Ratio of timeframes predicting UP
                   bearish_timeframe_ratio: float = 0.5,  # Ratio of timeframes predicting DOWN
                   # NEW: HMM regime features (trend/vol/liquidity as floats 0-1)
                   hmm_trend: float = 0.5,  # 0=bearish, 0.5=neutral, 1=bullish
                   hmm_vol: float = 0.5,    # 0=low, 0.5=normal, 1=high
                   hmm_liq: float = 0.5,    # 0=low, 0.5=normal, 1=high
                   # NEW: Extended VIX features
                   vix_bb_pos: float = 0.5,    # VIX Bollinger Band position (0=lower, 0.5=mid, 1=upper)
                   vix_roc: float = 0.0,       # VIX rate of change (-1 to 1)
                   vix_percentile: float = 0.5 # VIX percentile over history (0-1)
                   ) -> Tuple[bool, float, Dict]:
        """
        Decide if we should exit the options position now.
        
        Args:
            prediction_timeframe_minutes: What timeframe was the prediction for (15, 30, 60, 240)
            time_held_minutes: How long we've held this position
            current_pnl_pct: Current profit/loss percentage
            days_to_expiration: Days until option expires (theta decay factor)
            predicted_move_pct: What % move we predicted
            actual_move_pct: What % move has actually happened
            entry_confidence: Confidence when we entered (helps weight decision)
        
        Returns:
            should_exit: Boolean decision
            exit_score: 0-1 score (higher = more urgent to exit)
            details: Dictionary with reasoning
        """
        
        # Normalize inputs to 0-1 range (with division-by-zero safety)
        prediction_timeframe_safe = max(1.0, prediction_timeframe_minutes)  # Avoid division by zero
        predicted_move_safe = abs(predicted_move_pct) if abs(predicted_move_pct) > 0.001 else 1.0  # Avoid division by zero
        
        # Base 8 inputs (always used)
        base_inputs = [
            min(1.0, time_held_minutes / prediction_timeframe_safe),  # % of predicted time elapsed
            min(1.0, max(-1.0, current_pnl_pct / 50.0)),  # P&L normalized (-50% to +50%)
            min(1.0, max(0.0, days_to_expiration / 30.0)),  # Days to expiry (0-30 days)
            min(1.0, max(0.0, abs(actual_move_pct / predicted_move_safe))),  # How much of predicted move happened
            min(1.0, max(0.0, entry_confidence)),  # Entry confidence
            1.0 if current_pnl_pct > 0 else 0.0,  # Currently winning?
            1.0 if time_held_minutes >= prediction_timeframe_safe else 0.0,  # Past predicted time?
            min(1.0, max(0.0, (prediction_timeframe_safe - time_held_minutes) / prediction_timeframe_safe))  # Time remaining
        ]
        
        # EXPANDED inputs (4 context + 2 position-signal alignment = 6 additional features)
        if self.use_expanded_inputs:
            # Live prediction direction change (if available)
            live_pred_change = 0.0
            if live_predicted_move_pct is not None:
                live_pred_change = (live_predicted_move_pct - predicted_move_pct) / max(0.01, abs(predicted_move_pct))
                live_pred_change = min(1.0, max(-1.0, live_pred_change))
            
            # Live confidence ratio (if available)
            conf_ratio = live_confidence / max(0.01, entry_confidence) if live_confidence is not None else 1.0
            conf_ratio = min(2.0, max(0.0, conf_ratio)) / 2.0  # Normalize to 0-1
            
            # VIX level normalized (higher = more volatile market)
            vix_norm = min(1.0, max(0.0, vix_level / 50.0)) if vix_level else 0.36  # Default ~18 VIX
            
            # Bid-ask spread (liquidity indicator)
            spread_norm = min(1.0, max(0.0, bid_ask_spread_pct / 10.0))  # 10% = max
            
            # NEW: Position-Signal Alignment Features
            # Signal alignment: Does market direction support our position?
            # CALL benefits from bullish, PUT benefits from bearish
            signal_alignment = 0.5  # Neutral default
            if position_is_call is not None:
                if position_is_call:
                    # CALL position: bullish market = aligned, bearish = misaligned
                    signal_alignment = bullish_timeframe_ratio
                else:
                    # PUT position: bearish market = aligned, bullish = misaligned
                    signal_alignment = bearish_timeframe_ratio
            
            # Signal conflict: How strongly is market opposing our position?
            # 0 = no conflict (market supports us), 1 = strong conflict (market against us)
            signal_conflict = 0.0
            if position_is_call is not None:
                if position_is_call:
                    signal_conflict = bearish_timeframe_ratio  # Bearish signals conflict with CALL
                else:
                    signal_conflict = bullish_timeframe_ratio  # Bullish signals conflict with PUT
            
            expanded_inputs = [
                live_pred_change,  # How much prediction changed
                conf_ratio,  # Current vs entry confidence
                vix_norm,  # Market volatility regime
                spread_norm,  # Liquidity cost
                signal_alignment,  # Does market support our position? (0-1)
                signal_conflict,  # How much is market opposing us? (0-1)
                # HMM regime features (already 0-1 normalized)
                hmm_trend,  # 0=bearish, 0.5=neutral, 1=bullish
                hmm_vol,    # 0=low, 0.5=normal, 1=high
                hmm_liq,    # 0=low, 0.5=normal, 1=high
                # Extended VIX features
                vix_bb_pos,  # VIX Bollinger Band position (0-1)
                min(1.0, max(-1.0, vix_roc)),  # VIX rate of change (clamped)
                vix_percentile,  # VIX percentile (0-1)
            ]
            all_inputs = base_inputs + expanded_inputs
        else:
            all_inputs = base_inputs
            
        inputs = torch.tensor(all_inputs, dtype=torch.float32).to(self.device)
            
        # Get exit score from neural network
        with torch.no_grad():
            exit_score = self.model(inputs).item()
        
        # Decision logic with learned score
        should_exit = False
        reason = ""
        
        # ✅ RL-FIRST EXIT STRATEGY
        # The neural network sees all the data (P&L, time, VIX, HMM, etc.)
        # Let IT decide when to exit - don't override with hard-coded rules!
        # Only use safety backstops for extreme situations.
        
        # SAFETY BACKSTOPS ONLY (not timing rules):
        # 1. Big profit (>40%) - definitely take it (raised from 30%)
        # 2. Critical loss (<-20%) - cut it early (reduced from -25% to protect capital)
        # 3. About to expire (< 1 day) - exit to avoid pin risk
        
        # FIXED: Raised profit target to let winners run longer
        if current_pnl_pct >= 40.0:
            return True, 1.0, {
                'exit_score': 1.0,
                'should_exit': True,
                'reason': f'Taking big profit: {current_pnl_pct:+.1f}%',
                'time_held': time_held_minutes,
                'pnl_pct': current_pnl_pct,
                'days_to_expiry': days_to_expiration
            }
        
        # FIXED: Tighter stop loss to protect capital (-20% vs -25%)
        if current_pnl_pct <= -20.0:
            return True, 1.0, {
                'exit_score': 1.0,
                'should_exit': True,
                'reason': f'Cutting critical loss: {current_pnl_pct:+.1f}%',
                'time_held': time_held_minutes,
                'pnl_pct': current_pnl_pct,
                'days_to_expiry': days_to_expiration
            }
        
        if days_to_expiration < 1.0:
            return True, 1.0, {
                'exit_score': 1.0,
                'should_exit': True,
                'reason': f'Expiration approaching ({days_to_expiration:.1f} days)',
                'time_held': time_held_minutes,
                'pnl_pct': current_pnl_pct,
                'days_to_expiry': days_to_expiration
            }
        
        # Now let the RL network decide based on all the features it saw
        # Higher exit_score = more urgency to exit
        # The network has learned from experience when to exit
        
        # RL threshold: above 0.6 = exit, below = hold
        # This threshold can be tuned based on performance
        rl_exit_threshold = 0.6
        
        # ============= HMM-AWARE LIVE PREDICTION CHECK =============
        # Uses HMM regime to set smarter thresholds and detect regime changes
        prediction_reversed = False
        confidence_dropped = False
        regime_changed = False
        
        # ===== VOLATILITY-ADJUSTED THRESHOLDS =====
        # In high-vol regimes, predictions swing more - need higher thresholds
        # In low-vol regimes, smaller moves are more meaningful
        volatility_multiplier = 1.0
        current_vol_regime = "Normal"  # Default
        current_trend_regime = "Unknown"  # Default
        
        if hmm_regime_info:
            # Use structured volatility field if available, fall back to parsing
            current_vol_regime = hmm_regime_info.get('volatility', hmm_regime_info.get('volatility_name', 'Normal'))
            current_trend_regime = hmm_regime_info.get('trend', hmm_regime_info.get('trend_name', 'Unknown'))
            hmm_confidence = hmm_regime_info.get('confidence', hmm_regime_info.get('combined_confidence', 0.5))
            
            if 'High' in current_vol_regime:
                volatility_multiplier = 2.0  # Much higher thresholds in high vol
            elif 'Low' in current_vol_regime:
                volatility_multiplier = 0.6  # Lower thresholds in low vol (moves are meaningful)
        
        # Adjust thresholds based on volatility regime
        BASE_ENTRY_SIGNAL = 0.15   # Base: entry must be ±0.15% (was 0.05%)
        BASE_REVERSAL_SIGNAL = 0.10  # Base: reversal must be ±0.10% opposite (was 0.03%)
        
        MIN_ENTRY_SIGNAL = BASE_ENTRY_SIGNAL * volatility_multiplier
        MIN_REVERSAL_SIGNAL = BASE_REVERSAL_SIGNAL * volatility_multiplier
        
        # ===== REGIME CHANGE DETECTION =====
        # If HMM trend direction changed since entry, that's significant
        # Use direct trend comparison when available
        def is_bullish_trend(trend_str):
            if not trend_str:
                return False
            t = trend_str.lower()
            return 'uptrend' in t or 'up trend' in t or 'bullish' in t
        
        def is_bearish_trend(trend_str):
            if not trend_str:
                return False
            t = trend_str.lower()
            return 'downtrend' in t or 'down trend' in t or 'bearish' in t
        
        if entry_hmm_state and live_hmm_state:
            entry_bullish = is_bullish_trend(entry_hmm_state)
            entry_bearish = is_bearish_trend(entry_hmm_state)
            live_bullish = is_bullish_trend(live_hmm_state)
            live_bearish = is_bearish_trend(live_hmm_state)
            
            # True regime change: trend flipped from bullish to bearish or vice versa
            if (entry_bullish and live_bearish) or (entry_bearish and live_bullish):
                regime_changed = True
                logger.info(f"🔄 HMM REGIME CHANGED: {entry_hmm_state} → {live_hmm_state}")
        
        # ===== LSTM PREDICTION REVERSAL CHECK =====
        if live_predicted_move_pct is not None and predicted_move_pct != 0:
            # Only check for reversal if entry prediction was meaningful
            if abs(predicted_move_pct) >= MIN_ENTRY_SIGNAL:
                entry_direction = 1 if predicted_move_pct > 0 else -1
                
                # Live prediction must be meaningfully opposite
                if entry_direction > 0:  # Was bullish
                    if live_predicted_move_pct < -MIN_REVERSAL_SIGNAL:
                        prediction_reversed = True
                else:  # Was bearish
                    if live_predicted_move_pct > MIN_REVERSAL_SIGNAL:
                        prediction_reversed = True
                
                if prediction_reversed:
                    logger.info(f"🔄 Prediction REVERSED: entry={predicted_move_pct:+.2f}% → live={live_predicted_move_pct:+.2f}% (vol={current_vol_regime}, mult={volatility_multiplier:.1f}x)")
        
        # ===== CONFIDENCE DROP CHECK =====
        if live_confidence is not None and entry_confidence > 0:
            confidence_ratio = live_confidence / entry_confidence
            # In high vol, confidence naturally fluctuates more - less sensitive
            confidence_threshold = 0.6 if 'High' in current_vol_regime else 0.7
            if confidence_ratio < confidence_threshold:
                confidence_dropped = True
                logger.info(f"📉 Confidence DROPPED: entry={entry_confidence:.1%} → live={live_confidence:.1%}")
        
        # ===== EXIT DECISIONS =====
        # These are market condition triggers - NO min hold time gates!
        # If the market tells us to exit, we exit (regardless of time held)
        
        # Priority 1: Both LSTM reversed AND regime confirms (strongest signal)
        if prediction_reversed and regime_changed:
            return True, 0.95, {
                'exit_score': 0.95,
                'should_exit': True,
                'reason': f'LSTM + HMM both reversed! ({predicted_move_pct:+.2f}% → {live_predicted_move_pct:+.2f}%)',
                'prediction_reversed': True,
                'regime_changed': True,
                'time_held': time_held_minutes,
                'timeframe': prediction_timeframe_minutes,
                'pnl_pct': current_pnl_pct,
                'days_to_expiry': days_to_expiration,
                'move_completion': actual_move_pct / predicted_move_pct if predicted_move_pct != 0 else 0
            }
        
        # Priority 2: LSTM reversed alone (but respects vol-adjusted thresholds)
        # RELAXED: Only exit if RL agrees (score > 0.4) or PnL is bad
        if prediction_reversed and current_pnl_pct < 20.0:  # Exit unless winning big
            # Only force exit if RL is at least somewhat concerned
            if exit_score > 0.4:
                return True, max(exit_score, 0.9), {
                    'exit_score': max(exit_score, 0.9),
                    'should_exit': True,
                    'reason': f'Prediction REVERSED: {predicted_move_pct:+.2f}% → {live_predicted_move_pct:+.2f}%',
                    'prediction_reversed': True,
                    'vol_regime': current_vol_regime,
                    'vol_multiplier': volatility_multiplier,
                    'time_held': time_held_minutes,
                    'timeframe': prediction_timeframe_minutes,
                    'pnl_pct': current_pnl_pct,
                    'days_to_expiry': days_to_expiration,
                    'move_completion': actual_move_pct / predicted_move_pct if predicted_move_pct != 0 else 0
                }
        
        # Priority 3: HMM regime changed from trending to opposite trend
        if regime_changed and current_pnl_pct < 15.0:  # Exit unless winning big
             # Only force exit if RL is at least somewhat concerned
            if exit_score > 0.4:
                return True, max(exit_score, 0.85), {
                    'exit_score': max(exit_score, 0.85),
                    'should_exit': True,
                    'reason': f'HMM regime changed: {entry_hmm_state} → {live_hmm_state}',
                    'regime_changed': True,
                    'time_held': time_held_minutes,
                    'timeframe': prediction_timeframe_minutes,
                    'pnl_pct': current_pnl_pct,
                    'days_to_expiry': days_to_expiration,
                    'move_completion': actual_move_pct / predicted_move_pct if predicted_move_pct != 0 else 0
                }
        
        # Priority 4: Confidence dropped while losing
        if confidence_dropped and current_pnl_pct < 0:
            # RELAXED: Only exit if RL is at least somewhat concerned
            if exit_score > 0.4:
                return True, max(exit_score, 0.75), {
                    'exit_score': max(exit_score, 0.75),
                    'should_exit': True,
                    'reason': f'Confidence dropped ({entry_confidence:.1%}→{live_confidence:.1%}) while losing',
                    'confidence_dropped': True,
                    'time_held': time_held_minutes,
                    'timeframe': prediction_timeframe_minutes,
                    'pnl_pct': current_pnl_pct,
                    'days_to_expiry': days_to_expiration,
                    'move_completion': actual_move_pct / predicted_move_pct if predicted_move_pct != 0 else 0
                }
        
        # Priority 5: Strong signal conflict (market moving against position)
        if position_is_call is not None:
            # Calculate conflict score from the expanded inputs
            if position_is_call:
                signal_conflict = bearish_timeframe_ratio  # Bearish hurts CALLs
            else:
                signal_conflict = bullish_timeframe_ratio  # Bullish hurts PUTs
            
            # Strong conflict: 70%+ of timeframes predict against our position
            # AND we're losing money - exit regardless of time held
            # RELAXED: Only exit if RL is at least somewhat concerned
            if signal_conflict >= 0.70 and current_pnl_pct < 0 and exit_score > 0.4:
                conflict_direction = "bearish" if position_is_call else "bullish"
                position_type = "CALL" if position_is_call else "PUT"
                logger.warning(f"🔄 SIGNAL CONFLICT: {signal_conflict:.0%} of timeframes are {conflict_direction} (holding {position_type}, P&L: {current_pnl_pct:+.1f}%)")
                return True, max(exit_score, 0.80), {
                    'exit_score': max(exit_score, 0.80),
                    'should_exit': True,
                    'reason': f'Signal conflict: {signal_conflict:.0%} TFs against {position_type}, losing {current_pnl_pct:.1f}%',
                    'signal_conflict': True,
                    'conflict_ratio': signal_conflict,
                    'time_held': time_held_minutes,
                    'timeframe': prediction_timeframe_minutes,
                    'pnl_pct': current_pnl_pct,
                    'days_to_expiry': days_to_expiration,
                    'move_completion': actual_move_pct / predicted_move_pct if predicted_move_pct != 0 else 0
                }
            
            # Moderate conflict: 60%+ against us but we're slightly profitable
            # Let it run but log a warning
            elif signal_conflict >= 0.60 and current_pnl_pct < 5.0:
                position_type = "CALL" if position_is_call else "PUT"
                logger.info(f"⚠️ Signal leaning against {position_type}: {signal_conflict:.0%} opposition (P&L: {current_pnl_pct:+.1f}%)")
        
        # ============= END HMM-AWARE PREDICTION CHECK =============
        
        # ============= WEAK MOMENTUM EXIT (for flat markets) =============
        # In "No Trend" markets, predictions are weak (<0.1%) and noisy
        # Only exit if: held LONG time (45+ min) AND LOSING money AND weak predictions
        # This gives trades time to overcome the bid-ask spread
        weak_momentum = abs(live_predicted_move_pct or 0) < 0.10 if live_predicted_move_pct is not None else False
        
        # Check for No Trend using hmm_regime_info (properly populated)
        no_trend_regime = False
        if hmm_regime_info:
            trend = hmm_regime_info.get('trend', hmm_regime_info.get('trend_name', ''))
            no_trend_regime = trend in ['No Trend', 'Neutral', 'Sideways']
        
        # Only exit if LOSING money (< -3%) after 45+ minutes with no momentum
        # This prevents closing trades that are just slow to develop
        if weak_momentum and no_trend_regime and time_held_minutes > 45 and current_pnl_pct < -3.0:
            # Market is flat, predictions are weak, we're LOSING - exit to cut losses
            logger.info(f"🌊 Weak momentum + losing: {abs(live_predicted_move_pct or 0):.2f}% in No Trend, held {time_held_minutes}m, P&L: {current_pnl_pct:.1f}%")
            return True, 0.70, {
                'exit_score': 0.70,
                'should_exit': True,
                'reason': f'Weak momentum exit: {abs(live_predicted_move_pct or 0):.2f}% predicted, P&L {current_pnl_pct:.1f}%',
                'weak_momentum': True,
                'no_trend': True,
                'time_held': time_held_minutes,
                'pnl_pct': current_pnl_pct,
            }
        
        # ============= RL-FIRST EXIT DECISION =============
        # The neural network has seen all the features (P&L, time, VIX, HMM, etc.)
        # Let it decide - the exit_score represents its learned judgment
        
        # Dynamic threshold based on training progress
        # Starts conservative (0.7), becomes more confident as it learns (0.5)
        base_threshold = 0.70
        trained_threshold = 0.50
        training_progress = min(1.0, self.update_count / 500.0)  # Full training after 500 experiences
        dynamic_threshold = base_threshold - (base_threshold - trained_threshold) * training_progress
        
        # 🛡️ MINIMUM HOLDING PERIOD PROTECTION
        # Prevent "churning" where bot buys and immediately sells due to spread loss/fear.
        # FIXED: Extended minimum hold from 5 to 10 minutes, and tightened the P&L gate
        if time_held_minutes < 10 and current_pnl_pct > -10.0:
            # Boost the threshold significantly for young trades
            # This forces the bot to give the trade a chance to breathe
            dynamic_threshold = max(dynamic_threshold, 0.90)  # Raised from 0.85
            logger.debug(f"🛡️ Trade young ({time_held_minutes}m), raised exit threshold to {dynamic_threshold:.2f}")

        # RL says EXIT
        if exit_score >= dynamic_threshold:
            should_exit = True
            reason = f"RL decision: EXIT (score {exit_score:.3f} >= {dynamic_threshold:.2f}, P&L: {current_pnl_pct:+.1f}%)"
        else:
            # RL says HOLD - respect it
            should_exit = False
            reason = f"RL decision: HOLD (score {exit_score:.3f} < {dynamic_threshold:.2f}, P&L: {current_pnl_pct:+.1f}%)"
        
        details = {
            'exit_score': exit_score,
            'should_exit': should_exit,
            'reason': reason,
            'time_held': time_held_minutes,
            'timeframe': prediction_timeframe_minutes,
            'pnl_pct': current_pnl_pct,
            'days_to_expiry': days_to_expiration,
            'move_completion': actual_move_pct / predicted_move_pct if predicted_move_pct != 0 else 0
        }
        
        if should_exit:
            self.stats['total_exits'] += 1
        
        return should_exit, exit_score, details
    
    def store_exit_experience(self, 
                             prediction_timeframe_minutes: int,
                             time_held_minutes: int,
                             exit_pnl_pct: float,
                             days_to_expiration: int,
                             predicted_move_pct: float,
                             actual_move_pct: float,
                             entry_confidence: float,
                             exited_early: bool,
                             # Accept but ignore for compatibility
                             high_water_mark: float = None,
                             vix_level: float = 18.0,
                             momentum_at_exit: float = 0.0):
        """
        Store experience from a completed exit.
        
        Args:
            All same as should_exit, plus:
            exited_early: Did we exit before stop/target/expiry?
        """
        
        # Calculate what the exit score was (with division-by-zero safety)
        prediction_timeframe_safe = max(1.0, prediction_timeframe_minutes)  # Avoid division by zero
        predicted_move_safe = abs(predicted_move_pct) if abs(predicted_move_pct) > 0.001 else 1.0  # Avoid division by zero
        
        # Base inputs
        base_inputs = [
            min(1.0, time_held_minutes / prediction_timeframe_safe),
            min(1.0, max(-1.0, exit_pnl_pct / 50.0)),
            min(1.0, max(0.0, days_to_expiration / 30.0)),
            min(1.0, max(0.0, abs(actual_move_pct / predicted_move_safe))),
            min(1.0, max(0.0, entry_confidence)),
            1.0 if exit_pnl_pct > 0 else 0.0,
            1.0 if time_held_minutes >= prediction_timeframe_safe else 0.0,
            min(1.0, max(0.0, (prediction_timeframe_safe - time_held_minutes) / prediction_timeframe_safe))
        ]
        
        # Expanded inputs (fill with defaults when storing exit - we have limited info at exit time)
        if self.use_expanded_inputs:
            # We need to provide 12 inputs here to match the 20 total inputs expected (8 base + 12 context)
            # The previous implementation only provided 4 inputs, causing a shape mismatch
            expanded_inputs = [
                0.0,  # live_pred_change
                1.0 if exit_pnl_pct > 0 else 0.5,  # conf_ratio
                min(1.0, vix_level / 50.0) if vix_level else 0.36,  # vix_norm
                0.05,  # spread_norm
                0.5,  # signal_alignment
                0.0,  # signal_conflict
                0.5,  # hmm_trend
                0.5,  # hmm_vol
                0.5,  # hmm_liq
                0.5,  # vix_bb_pos
                0.0,  # vix_roc
                0.5,  # vix_percentile
            ]
            all_inputs = base_inputs + expanded_inputs
        else:
            all_inputs = base_inputs
            
        inputs = torch.tensor(all_inputs, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            exit_score = self.model(inputs).item()
        
        # Store experience
        experience = {
            'inputs': inputs.cpu().numpy(),
            'exit_score': exit_score,
            'pnl': exit_pnl_pct,
            'won': exit_pnl_pct > 0,
            'exited_early': exited_early,
            'time_held': time_held_minutes,
            'timeframe': prediction_timeframe_minutes
        }
        
        self.experiences.append(experience)
        self.update_count += 1
        
        # Initial priority based on PNL magnitude (important trades get higher priority)
        priority = abs(exit_pnl_pct) / 50.0 + 0.1  # Base priority 0.1, scales with PNL
        if not exit_pnl_pct > 0:  # Losses get extra priority (learn from mistakes)
            priority *= 1.5
        self.experience_priorities.append(priority)
        
        # Update stats
        if exit_pnl_pct > 0:
            self.stats['avg_hold_time_winners'].append(time_held_minutes)
            self.stats['avg_pnl_winners'].append(exit_pnl_pct)
            if exit_pnl_pct > self.stats['best_trade_pnl']:
                self.stats['best_trade_pnl'] = exit_pnl_pct
            if exited_early:
                self.stats['early_exits_won'] += 1
            else:
                self.stats['full_hold_won'] += 1
        else:
            self.stats['avg_hold_time_losers'].append(time_held_minutes)
            self.stats['avg_pnl_losers'].append(exit_pnl_pct)
            if exit_pnl_pct < self.stats['worst_trade_pnl']:
                self.stats['worst_trade_pnl'] = exit_pnl_pct
            if exited_early:
                self.stats['early_exits_lost'] += 1
            else:
                self.stats['full_hold_lost'] += 1
        
        # Limit buffers
        if len(self.experiences) > self.max_experiences:
            self.experiences.pop(0)
            if self.experience_priorities:
                self.experience_priorities.pop(0)
        
        # Keep only recent stats (increased to 200 for better averages)
        max_stat_history = 200
        if len(self.stats['avg_hold_time_winners']) > max_stat_history:
            self.stats['avg_hold_time_winners'] = self.stats['avg_hold_time_winners'][-max_stat_history:]
        if len(self.stats['avg_hold_time_losers']) > max_stat_history:
            self.stats['avg_hold_time_losers'] = self.stats['avg_hold_time_losers'][-max_stat_history:]
        if len(self.stats['avg_pnl_winners']) > max_stat_history:
            self.stats['avg_pnl_winners'] = self.stats['avg_pnl_winners'][-max_stat_history:]
        if len(self.stats['avg_pnl_losers']) > max_stat_history:
            self.stats['avg_pnl_losers'] = self.stats['avg_pnl_losers'][-max_stat_history:]
    
    def train_from_experiences(self, batch_size=64):
        """
        ENHANCED training with:
        1. PNL-weighted reward shaping (bigger wins/losses have more impact)
        2. Time-efficiency bonus (quick wins > slow wins for options)
        3. Prioritized experience replay (learn more from surprising outcomes)
        4. Gradient clipping for stability
        
        Goal: Learn to exit early when trades are going bad,
              hold longer when trades are going well.
        """
        if len(self.experiences) < batch_size:
            return None
        
        # Prioritized sampling: higher priority for surprising/important experiences
        if self.experience_priorities and len(self.experience_priorities) == len(self.experiences):
            # Sample with priority weights
            probs = np.array(self.experience_priorities)
            probs = probs / probs.sum()  # Normalize
            indices = np.random.choice(len(self.experiences), batch_size, replace=False, p=probs)
        else:
            # Fallback to uniform sampling
            indices = np.random.choice(len(self.experiences), batch_size, replace=False)
        
        batch = [self.experiences[i] for i in indices]
        
        # Prepare training data
        inputs = torch.tensor([exp['inputs'] for exp in batch], dtype=torch.float32).to(self.device)
        
        # ENHANCED REWARD SHAPING
        # Key insight: Exit score should reflect WHEN to exit optimally
        # - High score (close to 1.0) = EXIT NOW
        # - Low score (close to 0.0) = HOLD LONGER
        targets = []
        sample_weights = []
        
        for exp in batch:
            pnl = exp['pnl']
            won = exp['won']
            exited_early = exp['exited_early']
            time_held = exp.get('time_held', 0)
            timeframe = exp.get('timeframe', 30)
            
            # Time efficiency: what % of predicted timeframe was used
            time_efficiency = time_held / max(1, timeframe)
            
            # PNL magnitude factor (bigger trades matter more)
            pnl_magnitude = min(abs(pnl) / 50.0, 1.0)  # Cap at 50% moves
            
            if won:
                # WINNERS: We want to learn to HOLD winners longer
                if pnl >= 30.0:
                    # BIG WIN: Should have held - very low exit score
                    # Unless we exited very quickly (efficiency bonus)
                    if time_efficiency < 0.5:
                        target = 0.3  # Quick big win = good exit timing
                    else:
                        target = 0.2  # Slow big win = could hold longer
                elif pnl >= 15.0:
                    # MEDIUM WIN: Good timing
                    target = 0.4 if exited_early else 0.35
                else:
                    # SMALL WIN: OK but not great
                    target = 0.5 if exited_early else 0.45
                
                # Weight: Bigger wins matter more
                weight = 1.0 + pnl_magnitude
                
            else:
                # LOSERS: We want to learn to EXIT losers earlier
                if pnl <= -30.0:
                    # BIG LOSS: Should have exited much earlier - very high exit score
                    target = 0.95 if not exited_early else 0.85
                    weight = 2.0 + pnl_magnitude  # Extra weight on big losses
                elif pnl <= -15.0:
                    # MEDIUM LOSS: Should have exited earlier
                    target = 0.85 if not exited_early else 0.75
                    weight = 1.5 + pnl_magnitude
                else:
                    # SMALL LOSS: Close to break-even
                    # 🛡️ ANTI-CHURN PENALTY:
                    # If we exited very quickly (e.g. < 5 mins) with a small loss (spread),
                    # we should have HELD. So target should be LOW (Hold).
                    if time_held < 5.0 and exited_early:
                        target = 0.2  # Should have HELD! Don't panic sell spread.
                        weight = 2.0  # High importance to stop churning
                    else:
                        # Normal small loss, gradually lean towards exit
                        target = 0.7 if not exited_early else 0.6
                        weight = 1.0 + pnl_magnitude * 0.5
            
            targets.append(target)
            sample_weights.append(weight)
        
        targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(1).to(self.device)
        weights = torch.tensor(sample_weights, dtype=torch.float32).unsqueeze(1).to(self.device)
        
        # Forward pass
        self.model.train()
        predictions = self.model(inputs)
        
        # Weighted MSE Loss
        squared_errors = (predictions - targets) ** 2
        weighted_loss = (squared_errors * weights).mean()
        
        # Backward pass with gradient clipping
        self.optimizer.zero_grad()
        weighted_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Update learning rate based on loss
        self.scheduler.step(weighted_loss.item())
        
        # Track training stats
        self.train_count += 1
        self.stats['training_losses'].append(weighted_loss.item())
        if len(self.stats['training_losses']) > 1000:
            self.stats['training_losses'] = self.stats['training_losses'][-500:]
        
        # Update priorities for sampled experiences (TD-error style)
        with torch.no_grad():
            td_errors = torch.abs(predictions - targets).cpu().numpy().flatten()
            for i, idx in enumerate(indices):
                if idx < len(self.experience_priorities):
                    # Increase priority for high-error experiences
                    self.experience_priorities[idx] = float(td_errors[i]) + 0.01
        
        if self.train_count % 100 == 0:
            avg_loss = np.mean(self.stats['training_losses'][-100:])
            current_lr = self.optimizer.param_groups[0]['lr']
            logger.info(f"🚪 RL Exit training: loss={weighted_loss.item():.4f} (avg100={avg_loss:.4f}, lr={current_lr:.6f})")
        
        return {'loss': weighted_loss.item(), 'train_count': self.train_count}
    
    def get_optimal_hold_time(self, prediction_timeframe_minutes: int) -> int:
        """
        Get learned optimal hold time based on stats.
        
        Returns: Suggested hold time in minutes
        """
        if len(self.stats['avg_hold_time_winners']) > 10:
            avg_winner_time = np.mean(self.stats['avg_hold_time_winners'])
            # Winners tend to be held for X minutes on average
            return int(avg_winner_time)
        else:
            # Default: 80% of predicted timeframe
            return int(prediction_timeframe_minutes * 0.8)
    
    def get_stats(self) -> Dict:
        """Get comprehensive exit policy statistics"""
        total_early = self.stats['early_exits_won'] + self.stats['early_exits_lost']
        total_full = self.stats['full_hold_won'] + self.stats['full_hold_lost']
        total_trades = total_early + total_full
        
        # Calculate average PNLs
        avg_winner_pnl = np.mean(self.stats['avg_pnl_winners']) if self.stats.get('avg_pnl_winners') else 0
        avg_loser_pnl = np.mean(self.stats['avg_pnl_losers']) if self.stats.get('avg_pnl_losers') else 0
        
        # Calculate win/loss ratio (risk-reward)
        win_loss_ratio = abs(avg_winner_pnl / avg_loser_pnl) if avg_loser_pnl != 0 else 0
        
        # Overall win rate
        total_wins = self.stats['early_exits_won'] + self.stats['full_hold_won']
        overall_win_rate = total_wins / max(1, total_trades)
        
        # Training stats
        avg_training_loss = np.mean(self.stats.get('training_losses', [0])[-100:]) if self.stats.get('training_losses') else 0
        
        return {
            'total_exits': self.stats['total_exits'],
            'total_trades': total_trades,
            'overall_win_rate': overall_win_rate,
            'early_exit_win_rate': self.stats['early_exits_won'] / max(1, total_early),
            'full_hold_win_rate': self.stats['full_hold_won'] / max(1, total_full),
            'avg_winner_hold_time': np.mean(self.stats['avg_hold_time_winners']) if self.stats['avg_hold_time_winners'] else 0,
            'avg_loser_hold_time': np.mean(self.stats['avg_hold_time_losers']) if self.stats['avg_hold_time_losers'] else 0,
            'avg_winner_pnl': avg_winner_pnl,
            'avg_loser_pnl': avg_loser_pnl,
            'win_loss_ratio': win_loss_ratio,
            'best_trade': self.stats.get('best_trade_pnl', 0),
            'worst_trade': self.stats.get('worst_trade_pnl', 0),
            'update_count': self.update_count,
            'train_count': getattr(self, 'train_count', 0),
            'avg_training_loss': avg_training_loss,
            'experience_buffer_size': len(self.experiences),
            'current_lr': self.optimizer.param_groups[0]['lr'] if hasattr(self, 'optimizer') else 0
        }
    
    def save(self, path: str):
        """Save model and stats"""
        torch.save({
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict() if hasattr(self, 'scheduler') else None,
            'stats': self.stats,
            'update_count': self.update_count,
            'train_count': getattr(self, 'train_count', 0),
            'experiences': self.experiences[-2000:],  # Save more recent experiences
            'experience_priorities': self.experience_priorities[-2000:] if self.experience_priorities else [],
            'version': 'enhanced_v3',
            'use_expanded_inputs': self.use_expanded_inputs,
            'input_dim': 20 if self.use_expanded_inputs else 8
        }, path)
        logger.info(f"💾 RL Exit Policy saved to {path} ({len(self.experiences)} experiences)")
    
    def load(self, path: str):
        """Load model and stats with architecture migration support"""
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            
            # Handle different model versions
            version = checkpoint.get('version', 'unknown')
            saved_state = checkpoint.get('model_state', {})
            
            # Check input dimension of saved model
            input_proj_key = 'input_proj.0.weight'
            network_key = 'network.0.weight'
            
            saved_input_dim = None
            if input_proj_key in saved_state:
                saved_input_dim = saved_state[input_proj_key].shape[1]
            elif network_key in saved_state:
                saved_input_dim = saved_state[network_key].shape[1]
            
            current_input_dim = 20 if self.use_expanded_inputs else 8
            
            # Try to load weights, handling architecture changes
            if saved_input_dim == current_input_dim:
                # Exact match - load directly
                try:
                    self.model.load_state_dict(saved_state, strict=False)
                    logger.info(f"   ✅ Loaded model weights (input_dim={saved_input_dim})")
                except Exception as e:
                    logger.warning(f"   ⚠️ Partial weight load: {e}")
            elif saved_input_dim is not None:
                # Architecture mismatch - log and continue with fresh model
                logger.warning(f"⚠️ Model architecture changed (saved: {saved_input_dim}, current: {current_input_dim})")
                logger.warning(f"   Starting fresh model but loading stats")
            
            # Always try to load optimizer state
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            except Exception:
                logger.warning("   ⚠️ Could not load optimizer state, using fresh optimizer")
            
            self.stats = checkpoint.get('stats', self.stats)
            self.update_count = checkpoint.get('update_count', 0)
            self.train_count = checkpoint.get('train_count', 0)
            
            # Load experiences if available
            if 'experiences' in checkpoint:
                self.experiences = checkpoint['experiences']
            
            # Load experience priorities
            if 'experience_priorities' in checkpoint:
                self.experience_priorities = checkpoint['experience_priorities']
            else:
                # Initialize priorities for existing experiences
                self.experience_priorities = [0.5] * len(self.experiences)
            
            # Load scheduler state
            if 'scheduler_state' in checkpoint and checkpoint['scheduler_state'] and hasattr(self, 'scheduler'):
                try:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state'])
                except Exception:
                    pass  # Scheduler state may not be compatible
            
            version = checkpoint.get('version', 'unknown')
            logger.info(f"✅ RL Exit Policy loaded from {path} (v{version}, {self.update_count} updates, {self.train_count} trains)")
        except FileNotFoundError:
            logger.info(f"No saved exit policy found at {path}, starting fresh")
        except Exception as e:
            logger.warning(f"⚠️ Error loading exit policy: {e} - starting fresh")
