#!/usr/bin/env python3
"""
Unified RL Policy for Options Trading

REPLACES: rl_trading_policy.py, rl_threshold_learner.py, rl_exit_policy.py

Key Improvements:
1. Single policy handles ENTRY, HOLD, and EXIT (no fighting between systems)
2. Continuous rewards from mark-to-market (not just trade close)
3. Options-aware: explicitly models theta decay and gamma exposure
4. Simpler state space with proven predictive features
5. Contextual bandit mode for faster initial learning

Architecture:
- Simple MLP with proven features (not overengineered)
- Outputs: action probabilities + value estimate + exit urgency
- Learns from every price tick, not just trade closes
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
from dataclasses import dataclass
from datetime import datetime
import logging

# Regime filter for quality gating
from backend.regime_filter import get_regime_filter, RegimeDecision

# Event calendar for CPI/FOMC halts (Jerry's improvement)
from backend.event_calendar import should_halt_for_event

# Kill switches for consecutive losses, drawdowns (Jerry's improvement)
from backend.kill_switches import check_kill_switches

# Fuzzy logic scoring for trade quality (Jerry's improvement)
from backend.fuzzy_scoring import should_allow_trade as fuzzy_allow_trade

logger = logging.getLogger(__name__)


@dataclass
class TradeState:
    """Clean state representation for a single trade"""
    # Position info
    is_in_trade: bool = False
    is_call: bool = True  # True = CALL, False = PUT
    entry_price: float = 0.0
    current_price: float = 0.0
    position_size: int = 0
    
    # P&L
    unrealized_pnl_pct: float = 0.0
    max_pnl_seen: float = 0.0  # High water mark
    max_drawdown: float = 0.0
    
    # Time
    minutes_held: int = 0
    minutes_to_expiry: int = 1440  # 24 hours default
    
    # Market context
    predicted_direction: float = 0.0  # -1 to +1
    prediction_confidence: float = 0.0  # 0 to 1 (YOUR MODEL'S CONFIDENCE)
    vix_level: float = 18.0
    momentum_5m: float = 0.0
    volume_spike: float = 1.0
    
    # HMM Regime (YOUR HMM MODEL)
    hmm_trend: float = 0.5      # 0=bearish, 0.5=neutral, 1=bullish
    hmm_volatility: float = 0.5  # 0=low, 0.5=normal, 1=high
    hmm_liquidity: float = 0.5   # 0=low, 0.5=normal, 1=high
    hmm_confidence: float = 0.5  # HMM model's confidence in regime detection
    
    # Greeks exposure (critical for options!)
    estimated_theta_decay: float = 0.0  # Daily theta as %
    estimated_delta: float = 0.5


class UnifiedPolicyNetwork(nn.Module):
    """
    Simple but effective policy network.
    
    Key insight: Simpler networks often outperform complex ones
    when data is noisy (which trading data always is).
    
    State features (18 total):
    - Position state (4): in_trade, is_call, pnl%, drawdown
    - Time (2): minutes_held, minutes_to_expiry  
    - Prediction (3): direction, confidence, momentum
    - Market (2): vix, volume_spike
    - HMM Regime (4): trend, volatility, liquidity, hmm_confidence
    - Greeks (2): theta, delta
    """
    
    def __init__(self, state_dim: int = 18, hidden_dim: int = 64):  # 18 features now
        super().__init__()
        
        # Shared feature extractor (simple!)
        self.features = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        
        # Action head: 4 actions (HOLD, BUY_CALL, BUY_PUT, EXIT)
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 4)
        )
        
        # Value head (for advantage estimation)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # Exit urgency head (0-1, how urgently should we exit?)
        self.exit_urgency_head = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            action_logits: [batch, 4] logits for HOLD/BUY_CALL/BUY_PUT/EXIT
            value: [batch, 1] state value estimate
            exit_urgency: [batch, 1] how urgently to exit (0-1)
        """
        features = self.features(state)
        action_logits = self.action_head(features)
        value = self.value_head(features)
        exit_urgency = self.exit_urgency_head(features)
        return action_logits, value, exit_urgency


class UnifiedRLPolicy:
    """
    Single unified policy for all trading decisions.
    
    Modes:
    1. BANDIT mode (first 100 trades): Simple contextual bandit, fast learning
    2. FULL_RL mode (after 100 trades): Full PPO with temporal credit assignment
    
    This replaces:
    - RLTradingPolicy (entry decisions)
    - RLThresholdPolicy (threshold learning)  
    - RLExitPolicy (exit timing)
    """
    
    # Actions
    HOLD = 0
    BUY_CALL = 1
    BUY_PUT = 2
    EXIT = 3
    
    def __init__(
        self,
        learning_rate: float = 0.0003,
        gamma: float = 0.99,
        device: str = 'cpu',
        bandit_mode_trades: int = 100000,  # REDUCED: Fewer random trades before learning kicks in
        min_confidence_to_trade: float = 0.55,  # HIGH threshold - only trade high-quality signals
        max_position_loss_pct: float = None,  # Loaded from exit_config
        profit_target_pct: float = None,  # Loaded from exit_config
        trailing_stop_activation: float = None,  # Loaded from exit_config
        trailing_stop_distance: float = None,  # Loaded from exit_config
    ):
        # Load exit parameters from centralized config (SINGLE SOURCE OF TRUTH)
        try:
            from backend.exit_config import get_exit_config
            exit_cfg = get_exit_config()
            self.max_position_loss_pct = max_position_loss_pct if max_position_loss_pct is not None else exit_cfg.stop_loss_pct
            self.profit_target_pct = profit_target_pct if profit_target_pct is not None else exit_cfg.take_profit_pct
            self.trailing_stop_activation = trailing_stop_activation if trailing_stop_activation is not None else exit_cfg.trailing_activation_pct
            self.trailing_stop_distance = trailing_stop_distance if trailing_stop_distance is not None else exit_cfg.trailing_distance_pct
        except Exception as e:
            # Fallback to reasonable defaults if config fails
            import logging
            logging.getLogger(__name__).warning(f"Could not load exit_config: {e}, using defaults")
            self.max_position_loss_pct = max_position_loss_pct or 0.08
            self.profit_target_pct = profit_target_pct or 0.15
            self.trailing_stop_activation = trailing_stop_activation or 0.08
            self.trailing_stop_distance = trailing_stop_distance or 0.04

        self.device = device
        self.gamma = gamma
        self.bandit_mode_trades = bandit_mode_trades
        self.min_confidence_to_trade = min_confidence_to_trade
        
        # Track consecutive losses to reduce risk after losing streak
        self.consecutive_losses = 0
        self.max_consecutive_losses = 3  # After 3 losses, require higher confidence
        
        # Track win/loss history for adaptive position sizing
        self.recent_results = []  # Last 20 trades
        
        # Network (18 features: position + time + prediction + market + HMM + greeks)
        self.network = UnifiedPolicyNetwork(state_dim=18, hidden_dim=64).to(device)
        self.optimizer = torch.optim.AdamW(
            self.network.parameters(), 
            lr=learning_rate,
            weight_decay=1e-5
        )
        
        # Experience buffer
        self.experience_buffer: deque = deque(maxlen=10000)
        self.current_trade_experiences: List[Dict] = []
        
        # Stats
        self.total_trades = 0
        self.wins = 0
        self.losses = 0
        self.total_pnl = 0.0
        
        # Current trade tracking
        self.current_trade: Optional[TradeState] = None
        
        # Action history for analysis
        self.action_history: deque = deque(maxlen=1000)
        
        logger.info(f"🎯 Unified RL Policy initialized")
        logger.info(f"   Mode: BANDIT for first {bandit_mode_trades} trades, then FULL_RL")
        logger.info(f"   LR: {learning_rate}, Device: {device}")
        logger.info(f"   Profit target: {profit_target_pct:.0%}, Stop loss: {max_position_loss_pct:.0%}")
    
    @property
    def is_bandit_mode(self) -> bool:
        """Are we still in simple bandit learning mode?"""
        return self.total_trades < self.bandit_mode_trades
    
    @property
    def win_rate(self) -> float:
        total = self.wins + self.losses
        return self.wins / total if total > 0 else 0.5
    
    def state_to_tensor(self, state: TradeState) -> torch.Tensor:
        """Convert TradeState to tensor for network input (18 features)"""
        features = [
            # Position state (4 features)
            float(state.is_in_trade),
            float(state.is_call) if state.is_in_trade else 0.5,
            np.clip(state.unrealized_pnl_pct, -0.5, 0.5),  # Clip to [-50%, +50%]
            np.clip(state.max_drawdown, 0, 0.3),  # Max 30% drawdown
            
            # Time features (2 features)
            min(1.0, state.minutes_held / 120.0),  # Normalize to 2 hours
            min(1.0, state.minutes_to_expiry / 1440.0),  # Normalize to 24 hours
            
            # Prediction features (3 features) - YOUR MODEL'S OUTPUT
            np.clip(state.predicted_direction, -1, 1),
            np.clip(state.prediction_confidence, 0, 1),  # YOUR CONFIDENCE SCORE
            np.clip(state.momentum_5m * 100, -1, 1),  # Scale momentum
            
            # Market context (2 features)
            min(1.0, state.vix_level / 50.0),  # VIX normalized
            min(2.0, state.volume_spike) / 2.0,  # Volume spike normalized
            
            # HMM Regime (4 features) - YOUR HMM MODEL'S OUTPUT
            np.clip(state.hmm_trend, 0, 1),        # 0=bearish, 0.5=neutral, 1=bullish
            np.clip(state.hmm_volatility, 0, 1),   # 0=low, 0.5=normal, 1=high
            np.clip(state.hmm_liquidity, 0, 1),    # 0=low, 0.5=normal, 1=high
            np.clip(state.hmm_confidence, 0, 1),   # HMM confidence in regime
            
            # Greeks (2 features) - Critical for options!
            np.clip(state.estimated_theta_decay * 100, -0.1, 0),  # Theta is always negative
            np.clip(state.estimated_delta, 0, 1),

            # Padding / misc (1 feature)
            # NOTE: The network expects 18 inputs. We append a stable extra feature rather than
            # shifting the meaning/order of the existing 17 features.
            np.clip(float(state.position_size) / 10.0, 0.0, 1.0),
        ]
        return torch.tensor(features, dtype=torch.float32, device=self.device)
    
    def select_action(
        self, 
        state: TradeState,
        deterministic: bool = False
    ) -> Tuple[int, float, Dict]:
        """
        Select action given current state.

        Returns:
            action: 0=HOLD, 1=BUY_CALL, 2=BUY_PUT, 3=EXIT
            confidence: How confident the policy is
            details: Extra info for logging
        """
        # JERRY'S EVENT-DRIVEN HALTS: Block new entries during CPI/FOMC/NFP
        if not state.is_in_trade:
            halted, event_info = should_halt_for_event()
            if halted:
                event_type = event_info.get('type', 'unknown')
                mins_to = event_info.get('minutes_to_event', 0)
                mins_since = event_info.get('minutes_since_event', 0)
                if mins_to > 0:
                    logger.info(f"📅 EVENT HALT: {event_type} in {mins_to} min - blocking new entries")
                else:
                    logger.info(f"📅 EVENT HALT: {event_type} was {abs(mins_since)} min ago - blocking new entries")
                return self.HOLD, 0.0, {'reason': 'event_halt', 'event': event_info}

        # JERRY'S KILL SWITCHES: Non-overridable halt after consecutive losses/drawdown
        if not state.is_in_trade:
            kill_halted, kill_reason = check_kill_switches()
            if kill_halted:
                logger.warning(f"🛑 KILL SWITCH ACTIVE: {kill_reason} - blocking all entries")
                return self.HOLD, 0.0, {'reason': 'kill_switch', 'kill_reason': kill_reason}

        # RULE-BASED SAFETY CHECKS (always apply, even in RL mode)
        if state.is_in_trade:
            # Force exit on stop loss
            if state.unrealized_pnl_pct <= -self.max_position_loss_pct:
                logger.info(f"🛑 STOP LOSS: {state.unrealized_pnl_pct:.1%} <= -{self.max_position_loss_pct:.0%}")
                return self.EXIT, 1.0, {'reason': 'stop_loss'}
            
            # TRAILING STOP: Once we hit activation level, trail by distance
            if state.max_pnl_seen >= self.trailing_stop_activation:
                trailing_stop_level = state.max_pnl_seen - self.trailing_stop_distance
                if state.unrealized_pnl_pct <= trailing_stop_level:
                    logger.info(f"📈 TRAILING STOP: PnL {state.unrealized_pnl_pct:.1%} fell below trail at {trailing_stop_level:.1%} (max was {state.max_pnl_seen:.1%})")
                    return self.EXIT, 1.0, {'reason': 'trailing_stop'}
            
            # Force exit on profit target (only if not using trailing stop)
            if state.max_pnl_seen < self.trailing_stop_activation:
                if state.unrealized_pnl_pct >= self.profit_target_pct:
                    logger.info(f"🎯 PROFIT TARGET: {state.unrealized_pnl_pct:.1%} >= {self.profit_target_pct:.0%}")
                    return self.EXIT, 1.0, {'reason': 'profit_target'}
            
            # Force exit near expiry (< 30 mins)
            if state.minutes_to_expiry < 30:
                logger.info(f"⏰ EXPIRY EXIT: {state.minutes_to_expiry} minutes left")
                return self.EXIT, 1.0, {'reason': 'expiry'}
            
            # Force exit if position is old and not profitable (avoid theta decay eating small profits)
            if state.minutes_held > 90 and state.unrealized_pnl_pct < 0.03:
                logger.info(f"⏱️ TIME EXIT: {state.minutes_held}m held with only {state.unrealized_pnl_pct:.1%} profit")
                return self.EXIT, 0.9, {'reason': 'time_decay_risk'}
        
        # BANDIT MODE: Simpler decision making
        if self.is_bandit_mode:
            return self._bandit_decision(state, deterministic)
        
        # FULL RL MODE: Use neural network
        return self._rl_decision(state, deterministic)
    
    def _bandit_decision(
        self, 
        state: TradeState, 
        deterministic: bool
    ) -> Tuple[int, float, Dict]:
        """
        Smart contextual bandit decision with QUALITY-FIRST approach.
        
        Key principles for high win rate:
        1. Only trade high-quality setups (confluence of signals)
        2. Trade WITH the trend, not against it
        3. Tight stop losses, let winners run
        4. Reduce risk after losses
        """
        details = {'mode': 'bandit'}
        
        if state.is_in_trade:
            # ==================== EXIT LOGIC ====================
            # Exit logic is handled in select_action() safety checks
            # Here we only handle nuanced exit decisions
            
            # TREND REVERSAL: Exit if trend changed against our position
            if state.is_call and state.hmm_trend < 0.35 and state.hmm_confidence > 0.6:
                return self.EXIT, 0.85, {**details, 'reason': f'trend_bearish_reversal (trend={state.hmm_trend:.2f})'}
            elif not state.is_call and state.hmm_trend > 0.65 and state.hmm_confidence > 0.6:
                return self.EXIT, 0.85, {**details, 'reason': f'trend_bullish_reversal (trend={state.hmm_trend:.2f})'}
            
            # MOMENTUM REVERSAL: Quick exit on strong reversals
            momentum_threshold = 0.003  # 0.3% per 5 min is significant
            if state.is_call and state.momentum_5m < -momentum_threshold:
                if state.unrealized_pnl_pct < 0.05:  # Not winning much
                    return self.EXIT, 0.8, {**details, 'reason': 'momentum_reversal_bearish'}
            elif not state.is_call and state.momentum_5m > momentum_threshold:
                if state.unrealized_pnl_pct < 0.05:
                    return self.EXIT, 0.8, {**details, 'reason': 'momentum_reversal_bullish'}
            
            # THETA DECAY: Exit before theta eats our profits
            if state.estimated_theta_decay < -0.015 and state.unrealized_pnl_pct < 0.04:
                return self.EXIT, 0.7, {**details, 'reason': 'theta_decay_risk'}
            
            # Default: hold position
            return self.HOLD, 0.6, {**details, 'reason': 'hold_position'}
        
        else:
            # ==================== ENTRY LOGIC ====================
            # HMM + Neural Confirmation Strategy
            # Only trade when BOTH HMM and neural network agree on direction

            # === REGIME FILTER (Phase 12): Quality-based entry gating ===
            # Enable via env var: REGIME_FILTER_ENABLED=1 (disabled by default for testing)
            regime_enabled = os.environ.get('REGIME_FILTER_ENABLED', '0')
            if regime_enabled == '1':
                logger.info(f'[REGIME] Filter enabled, checking regime quality...')
                regime_filter = get_regime_filter()
                regime_decision = regime_filter.should_trade(
                    hmm_trend=state.hmm_trend,
                    hmm_volatility=state.hmm_volatility,
                    hmm_liquidity=getattr(state, 'hmm_liquidity', 0.5),
                    hmm_confidence=state.hmm_confidence,
                    vix_level=state.vix_level,
                )
                if not regime_decision.can_trade:
                    # Track regime vetoes
                    gate_tracker = getattr(self, '_gate_tracker', None)
                    if gate_tracker:
                        gate_tracker['regime_veto'] = gate_tracker.get('regime_veto', 0) + 1
                    logger.info(f"[REGIME VETO] {regime_decision.veto_reason} | quality={regime_decision.quality_score:.2f} state={regime_decision.regime_state}")
                    return self.HOLD, 0.95, {**details, 
                        'reason': f'regime_veto ({regime_decision.veto_reason})',
                        'regime_state': regime_decision.regime_state,
                        'quality_score': regime_decision.quality_score
                    }
                # Store position scale for later use (could scale position size)
                details['regime_quality'] = regime_decision.quality_score
                details['regime_state'] = regime_decision.regime_state
                details['position_scale'] = regime_decision.position_scale

            # === JERRY'S FUZZY LOGIC: Trade quality scoring (when enabled) ===
            # Check fuzzy score before proceeding with entry
            try:
                # Determine proposed action from signals
                if state.hmm_trend > 0.55 and state.predicted_direction > 0:
                    proposed_action = 'BUY_CALL'
                elif state.hmm_trend < 0.45 and state.predicted_direction < 0:
                    proposed_action = 'BUY_PUT'
                else:
                    proposed_action = 'HOLD'

                fuzzy_allowed, fuzzy_score, fuzzy_breakdown = fuzzy_allow_trade(
                    vix=state.vix_level,
                    hmm_confidence=state.hmm_confidence,
                    predicted_direction=state.predicted_direction,
                    hmm_trend=state.hmm_trend,
                    model_confidence=state.prediction_confidence,
                    proposed_action=proposed_action
                )
                details['fuzzy_score'] = fuzzy_score
                details['fuzzy_breakdown'] = fuzzy_breakdown

                if not fuzzy_allowed:
                    logger.info(f"FUZZY VETO: Score {fuzzy_score:.3f} below threshold")
                    return self.HOLD, fuzzy_score, {**details, 'reason': 'fuzzy_veto'}
            except Exception as e:
                logger.debug(f"Fuzzy scoring error: {e}")

            # === THRESHOLDS (configurable via env vars for testing) ===
            # Updated defaults based on Phase 11 analysis: relaxed thresholds = 3x better P&L
            HMM_STRONG_BULLISH = float(os.environ.get('HMM_STRONG_BULLISH', '0.70'))
            HMM_STRONG_BEARISH = float(os.environ.get('HMM_STRONG_BEARISH', '0.30'))
            HMM_MIN_CONFIDENCE = float(os.environ.get('HMM_MIN_CONFIDENCE', '0.70'))
            HMM_MAX_VOLATILITY = float(os.environ.get('HMM_MAX_VOLATILITY', '0.70'))

            # Neural confirmation settings (enable via env var)
            REQUIRE_NEURAL_CONFIRM = os.environ.get('REQUIRE_NEURAL_CONFIRM', '1') == '1'
            NEURAL_MIN_DIRECTION = float(os.environ.get('NEURAL_MIN_DIRECTION', '0.1'))

            # Track gate rejections for analysis
            gate_tracker = getattr(self, '_gate_tracker', None)
            if gate_tracker is None:
                self._gate_tracker = {'hmm_uncertain': 0, 'high_volatility': 0, 'neural_disagrees': 0, 'hmm_neutral': 0, 'regime_veto': 0, 'total': 0}
                gate_tracker = self._gate_tracker
            gate_tracker['total'] += 1

            # Check HMM confidence first
            if state.hmm_confidence < HMM_MIN_CONFIDENCE:
                gate_tracker['hmm_uncertain'] += 1
                return self.HOLD, 0.9, {**details, 'reason': f'hmm_uncertain ({state.hmm_confidence:.1%} < {HMM_MIN_CONFIDENCE:.0%})'}

            # Check volatility
            if state.hmm_volatility >= HMM_MAX_VOLATILITY:
                gate_tracker['high_volatility'] += 1
                return self.HOLD, 0.85, {**details, 'reason': f'high_volatility ({state.hmm_volatility:.2f} >= {HMM_MAX_VOLATILITY})'}

            # === VOLUME FILTER (Phase 45): Require active market ===
            # Analysis shows vol_spike > 1.2 gives +14.5% higher win rate
            MIN_VOLUME_SPIKE = float(os.environ.get('MIN_VOLUME_SPIKE', '0.5'))  # Default low, set to 1.2 for strict
            if state.volume_spike < MIN_VOLUME_SPIKE:
                gate_tracker['low_volume'] = gate_tracker.get('low_volume', 0) + 1
                return self.HOLD, 0.85, {**details, 'reason': f'low_volume ({state.volume_spike:.2f} < {MIN_VOLUME_SPIKE})'}

            # Trade with strong HMM trend + neural confirmation
            if state.hmm_trend > HMM_STRONG_BULLISH:
                # Strong bullish HMM - check neural agrees
                if REQUIRE_NEURAL_CONFIRM and state.predicted_direction < NEURAL_MIN_DIRECTION:
                    gate_tracker['neural_disagrees'] += 1
                    return self.HOLD, 0.8, {**details, 'reason': f'neural_disagrees_bullish (neural={state.predicted_direction:+.2f}, need>{NEURAL_MIN_DIRECTION})'}
                logger.info(f"📈 HMM+NEURAL ENTRY: CALL - hmm_trend={state.hmm_trend:.2f}, neural={state.predicted_direction:+.2f}")
                return self.BUY_CALL, state.hmm_confidence, {**details, 'reason': f'hmm_neural_bullish (hmm={state.hmm_trend:.2f}, neural={state.predicted_direction:+.2f})'}

            elif state.hmm_trend < HMM_STRONG_BEARISH:
                # Strong bearish HMM - check neural agrees
                if REQUIRE_NEURAL_CONFIRM and state.predicted_direction > -NEURAL_MIN_DIRECTION:
                    gate_tracker['neural_disagrees'] += 1
                    return self.HOLD, 0.8, {**details, 'reason': f'neural_disagrees_bearish (neural={state.predicted_direction:+.2f}, need<{-NEURAL_MIN_DIRECTION})'}
                logger.info(f"📉 HMM+NEURAL ENTRY: PUT - hmm_trend={state.hmm_trend:.2f}, neural={state.predicted_direction:+.2f}")
                return self.BUY_PUT, state.hmm_confidence, {**details, 'reason': f'hmm_neural_bearish (hmm={state.hmm_trend:.2f}, neural={state.predicted_direction:+.2f})'}

            # No strong trend - don't trade
            gate_tracker['hmm_neutral'] += 1
            return self.HOLD, 0.9, {**details, 'reason': f'hmm_neutral (trend={state.hmm_trend:.2f})'}

            # === LEGACY NEURAL-BASED GATES (disabled - neural predictions unreliable) ===
            # Uncomment below when neural network is trained
            """
            # === GATE 1: Confidence Check (STRICTER) ===
            # After losses, require even higher confidence
            confidence_required = self.min_confidence_to_trade
            if self.consecutive_losses >= 2:
                confidence_required = min(0.70, confidence_required + 0.10)
                details['loss_adjusted_threshold'] = True

            if state.prediction_confidence < confidence_required:
                return self.HOLD, 0.9, {**details, 'reason': f'low_confidence ({state.prediction_confidence:.1%} < {confidence_required:.1%})'}
            
            # === GATE 2: VIX Check (avoid extreme volatility) ===
            if state.vix_level > 30:
                return self.HOLD, 0.85, {**details, 'reason': f'high_vix ({state.vix_level:.1f} > 30)'}
            if state.vix_level < 12:  # Too quiet, hard to profit
                return self.HOLD, 0.8, {**details, 'reason': f'low_vix ({state.vix_level:.1f} < 12)'}
            
            # === GATE 3: Direction Strength Check ===
            # Require a meaningful predicted direction
            # predicted_direction is scaled: 1.0 = 1% predicted move
            # (predicted_return * 100, clipped to [-1, 1])
            # TIGHTENED: Require at least 0.5% predicted move (50 basis points)
            MIN_DIRECTION_THRESHOLD = 0.5  # 0.5% predicted move (was 0.2%)
            if abs(state.predicted_direction) < MIN_DIRECTION_THRESHOLD:
                return self.HOLD, 0.85, {**details, 'reason': f'weak_direction ({state.predicted_direction:.3f} < {MIN_DIRECTION_THRESHOLD})'}
            
            # === GATE 4: HMM Regime Alignment (CRITICAL) ===
            # ONLY trade when prediction aligns with HMM trend
            bullish_prediction = state.predicted_direction > MIN_DIRECTION_THRESHOLD
            bearish_prediction = state.predicted_direction < -MIN_DIRECTION_THRESHOLD

            # TIGHTENED: Require strong HMM signal (was 0.55/0.45)
            hmm_bullish = state.hmm_trend > 0.65  # Strong bullish required
            hmm_bearish = state.hmm_trend < 0.35  # Strong bearish required
            hmm_choppy = 0.35 <= state.hmm_trend <= 0.65 and state.hmm_volatility > 0.6

            # AVOID: Trading in choppy markets (low win rate)
            if hmm_choppy and state.hmm_confidence > 0.5:
                return self.HOLD, 0.8, {**details, 'reason': f'choppy_market (trend={state.hmm_trend:.2f}, vol={state.hmm_volatility:.2f})'}

            # CHECK ALIGNMENT
            effective_confidence = state.prediction_confidence

            if bullish_prediction:
                if hmm_bullish:
                    # STRONG SETUP: Bullish prediction + Bullish HMM
                    effective_confidence = min(1.0, state.prediction_confidence * 1.15)
                    details['setup_quality'] = 'aligned_bullish'
                elif hmm_bearish:
                    # CONFLICT: Don't fight the trend
                    return self.HOLD, 0.85, {**details, 'reason': 'bullish_vs_bearish_trend'}
                else:
                    # NEUTRAL trend - DON'T TRADE (predictions unreliable in neutral markets)
                    return self.HOLD, 0.8, {**details, 'reason': f'neutral_trend_bullish (trend={state.hmm_trend:.2f})'}

            elif bearish_prediction:
                if hmm_bearish:
                    # STRONG SETUP: Bearish prediction + Bearish HMM
                    effective_confidence = min(1.0, state.prediction_confidence * 1.15)
                    details['setup_quality'] = 'aligned_bearish'
                elif hmm_bullish:
                    # CONFLICT: Don't fight the trend
                    return self.HOLD, 0.85, {**details, 'reason': 'bearish_vs_bullish_trend'}
                else:
                    # NEUTRAL trend - DON'T TRADE (predictions unreliable in neutral markets)
                    return self.HOLD, 0.8, {**details, 'reason': f'neutral_trend_bearish (trend={state.hmm_trend:.2f})'}
            
            # === GATE 5: Volume/Liquidity Check ===
            if state.volume_spike < 0.5:  # Very low volume
                return self.HOLD, 0.75, {**details, 'reason': f'low_volume ({state.volume_spike:.2f})'}
            
            # === GATE 6: Exploration (REDUCED) ===
            # Only 5% exploration rate to avoid random losses
            explore_rate = 0.05 if self.is_bandit_mode else 0.02
            is_exploring = np.random.random() < explore_rate and not deterministic
            
            if is_exploring:
                # TIGHTENED: Even when exploring, require high confidence
                if state.prediction_confidence >= 0.55 and (hmm_bullish or hmm_bearish):
                    if bullish_prediction and hmm_bullish:
                        return self.BUY_CALL, 0.5, {**details, 'reason': 'exploration_bullish_aligned'}
                    elif bearish_prediction and hmm_bearish:
                        return self.BUY_PUT, 0.5, {**details, 'reason': 'exploration_bearish_aligned'}
            
            # === FINAL ENTRY DECISION ===
            if bullish_prediction:
                logger.info(f"📈 ENTRY: CALL - conf={effective_confidence:.1%}, direction={state.predicted_direction:.3f}, hmm={state.hmm_trend:.2f}")
                logger.info(f"   [DEBUG] Neural conf={state.prediction_confidence:.1%}, VIX={state.vix_level:.1f}, "
                           f"HMM vol={state.hmm_volatility:.2f}, volume={state.volume_spike:.2f}")
                return self.BUY_CALL, effective_confidence, {**details, 'reason': 'bullish_signal'}
            elif bearish_prediction:
                logger.info(f"📉 ENTRY: PUT - conf={effective_confidence:.1%}, direction={state.predicted_direction:.3f}, hmm={state.hmm_trend:.2f}")
                logger.info(f"   [DEBUG] Neural conf={state.prediction_confidence:.1%}, VIX={state.vix_level:.1f}, "
                           f"HMM vol={state.hmm_volatility:.2f}, volume={state.volume_spike:.2f}")
                return self.BUY_PUT, effective_confidence, {**details, 'reason': 'bearish_signal'}
            else:
                return self.HOLD, 0.7, {**details, 'reason': 'no_clear_signal'}
            """  # End of commented-out neural-based entry logic

    def _rl_decision(
        self, 
        state: TradeState, 
        deterministic: bool
    ) -> Tuple[int, float, Dict]:
        """Full RL decision using neural network"""
        details = {'mode': 'rl'}
        
        state_tensor = self.state_to_tensor(state).unsqueeze(0)
        
        with torch.no_grad():
            action_logits, value, exit_urgency = self.network(state_tensor)
            probs = F.softmax(action_logits, dim=-1).squeeze()
            
            # Mask invalid actions
            if state.is_in_trade:
                # Can only HOLD or EXIT when in a trade
                probs[self.BUY_CALL] = 0
                probs[self.BUY_PUT] = 0
            else:
                # Can't EXIT when not in a trade
                probs[self.EXIT] = 0
            
            # Renormalize
            probs = probs / (probs.sum() + 1e-8)
            
            if deterministic:
                action = probs.argmax().item()
            else:
                # Temperature-based sampling
                temp = 0.8 if self.total_trades < 200 else 1.0
                probs_tempered = (probs ** (1/temp))
                probs_tempered = probs_tempered / probs_tempered.sum()
                action = torch.multinomial(probs_tempered, 1).item()
            
            confidence = probs[action].item()
            
            details['probs'] = probs.cpu().numpy().tolist()
            details['value'] = value.item()
            details['exit_urgency'] = exit_urgency.item()
        
        # Quality gate for entries (even in RL mode)
        if action in [self.BUY_CALL, self.BUY_PUT]:
            if state.prediction_confidence < self.min_confidence_to_trade:
                return self.HOLD, 0.8, {**details, 'reason': 'quality_gate'}
        
        return action, confidence, details
    
    def record_step_reward(
        self, 
        state: TradeState, 
        action: int, 
        reward: float,
        next_state: TradeState,
        done: bool,
        trade_pnl: float = 0.0  # NEW: Actual P&L when trade closes
    ):
        """
        Record experience for learning.
        
        KEY IMPROVEMENT: We learn from every step, not just trade closes!
        Also tracks consecutive wins/losses for adaptive risk management.
        """
        experience = {
            'state': self.state_to_tensor(state).cpu().numpy(),
            'action': action,
            'reward': reward,
            'next_state': self.state_to_tensor(next_state).cpu().numpy(),
            'done': done,
            'timestamp': datetime.now()
        }
        
        self.current_trade_experiences.append(experience)
        
        if done:
            # Trade completed - add all experiences to buffer
            self.experience_buffer.extend(self.current_trade_experiences)
            
            # Calculate total reward from this trade
            total_reward = sum(exp['reward'] for exp in self.current_trade_experiences) if self.current_trade_experiences else reward
            
            # Use actual P&L if provided, otherwise use reward
            was_winner = trade_pnl > 0 if trade_pnl != 0 else total_reward > 0
            
            self.current_trade_experiences = []
            
            # Update stats
            self.total_trades += 1
            if was_winner:
                self.wins += 1
                self.consecutive_losses = 0  # Reset on win
            else:
                self.losses += 1
                self.consecutive_losses += 1
            self.total_pnl += total_reward
            
            # Track recent results for adaptive behavior
            self.recent_results.append(was_winner)
            if len(self.recent_results) > 20:
                self.recent_results.pop(0)
            
            # Calculate recent win rate
            recent_wins = sum(self.recent_results) if self.recent_results else 0
            recent_win_rate = recent_wins / len(self.recent_results) if self.recent_results else 0.5
            
            logger.info(f"📊 Trade #{self.total_trades}: {'✅ WIN' if was_winner else '❌ LOSS'} | "
                       f"Reward={total_reward:.3f} | Win rate={self.win_rate:.1%} | "
                       f"Recent 20={recent_win_rate:.1%} | Consec losses={self.consecutive_losses} | "
                       f"Mode={'BANDIT' if self.is_bandit_mode else 'RL'}")
    
    def calculate_step_reward(
        self,
        prev_state: TradeState,
        current_state: TradeState,
        action: int
    ) -> float:
        """
        Calculate immediate reward for a single step.
        
        IMPROVED REWARD SHAPING for higher win rates:
        1. Strong rewards for profitable exits
        2. Quick loss cutting is rewarded
        3. Patience for good setups is rewarded
        4. Penalize random trading
        """
        reward = 0.0
        
        # === NOT IN A TRADE ===
        if not prev_state.is_in_trade and not current_state.is_in_trade:
            # Reward patience when signals are weak
            if current_state.prediction_confidence < self.min_confidence_to_trade:
                reward = 0.02  # Good job waiting
            elif abs(current_state.predicted_direction) < 0.02:  # No clear direction
                reward = 0.01  # OK to wait
            else:
                # Strong signal but didn't trade - depends on context
                if current_state.hmm_volatility > 0.6 and 0.4 < current_state.hmm_trend < 0.6:
                    reward = 0.01  # Good to avoid choppy markets
                else:
                    reward = -0.01  # Slight penalty for missing clear signals
            return reward
        
        # === ENTERING A TRADE ===
        if not prev_state.is_in_trade and current_state.is_in_trade:
            # Just entered - reward based on setup quality
            setup_quality = current_state.prediction_confidence
            
            # Bonus for trend-aligned entries
            is_call = current_state.is_call
            if (is_call and current_state.hmm_trend > 0.6) or (not is_call and current_state.hmm_trend < 0.4):
                setup_quality *= 1.2
            
            reward = 0.05 * setup_quality  # Small positive for taking quality trades
            return float(np.clip(reward, -0.5, 0.5))
        
        # === IN A TRADE ===
        if current_state.is_in_trade:
            # Track P&L change (main learning signal)
            pnl_change = current_state.unrealized_pnl_pct - prev_state.unrealized_pnl_pct
            
            # Scale reward: 1% gain = +0.15, 1% loss = -0.15
            reward += pnl_change * 15.0
            
            # Theta decay cost (continuous time penalty)
            theta_per_minute = abs(current_state.estimated_theta_decay) / 390.0
            reward -= theta_per_minute * 3.0  # Gentle time pressure
            
            # Drawdown penalty (risk management signal)
            if current_state.max_drawdown > prev_state.max_drawdown:
                new_drawdown = current_state.max_drawdown - prev_state.max_drawdown
                reward -= new_drawdown * 5.0  # Strong drawdown penalty
            
            # Momentum alignment bonus (holding WITH momentum)
            if current_state.is_call and current_state.momentum_5m > 0.001:
                reward += 0.01  # Aligned with momentum
            elif not current_state.is_call and current_state.momentum_5m < -0.001:
                reward += 0.01  # Aligned with momentum
        
        # === EXITING A TRADE ===
        if action == self.EXIT:
            pnl = current_state.unrealized_pnl_pct
            
            # WINNING EXITS: Strong positive reward
            if pnl > 0.10:
                reward += 0.5  # Excellent win
            elif pnl > 0.05:
                reward += 0.3  # Good win
            elif pnl > 0:
                reward += 0.1  # Small win
            
            # LOSING EXITS: Reward quick loss cuts, penalize big losses
            elif pnl > -0.05:
                reward += 0.1  # Good loss management
            elif pnl > -0.08:
                reward += 0.0  # OK, near stop loss
            else:
                reward -= 0.2  # Let it run too far
            
            # Time-based efficiency bonus
            if pnl > 0 and current_state.minutes_held < 30:
                reward += 0.15  # Quick profitable trade
            elif pnl > 0 and current_state.minutes_held < 60:
                reward += 0.05  # Reasonably quick
            
            # Penalize churning (very quick exits without profit)
            if current_state.minutes_held < 5 and abs(pnl) < 0.02:
                reward -= 0.25  # Churning penalty
        
        return float(np.clip(reward, -2.0, 2.0))
    
    def train_step(self, batch_size: int = 32) -> Optional[Dict]:
        """Train the policy from experiences"""
        if len(self.experience_buffer) < batch_size:
            return None
        
        # Sample batch
        indices = np.random.choice(len(self.experience_buffer), batch_size, replace=False)
        batch = [self.experience_buffer[i] for i in indices]
        
        # Prepare tensors
        states = torch.tensor(
            np.array([exp['state'] for exp in batch]), 
            dtype=torch.float32, device=self.device
        )
        actions = torch.tensor(
            [exp['action'] for exp in batch], 
            dtype=torch.long, device=self.device
        )
        rewards = torch.tensor(
            [exp['reward'] for exp in batch], 
            dtype=torch.float32, device=self.device
        )
        next_states = torch.tensor(
            np.array([exp['next_state'] for exp in batch]), 
            dtype=torch.float32, device=self.device
        )
        dones = torch.tensor(
            [exp['done'] for exp in batch], 
            dtype=torch.float32, device=self.device
        )
        
        # Forward pass
        action_logits, values, exit_urgency = self.network(states)
        _, next_values, _ = self.network(next_states)
        
        # Compute targets (TD learning)
        with torch.no_grad():
            targets = rewards + self.gamma * next_values.squeeze() * (1 - dones)
        
        # Value loss
        value_loss = F.smooth_l1_loss(values.squeeze(), targets)
        
        # Policy loss (simple cross-entropy with advantage weighting)
        advantages = (targets - values.squeeze()).detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        log_probs = F.log_softmax(action_logits, dim=-1)
        action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze()
        policy_loss = -(action_log_probs * advantages).mean()
        
        # Entropy bonus (encourage exploration)
        probs = F.softmax(action_logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).mean()
        entropy_bonus = 0.01 * entropy
        
        # Total loss
        total_loss = policy_loss + 0.5 * value_loss - entropy_bonus
        
        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5)
        self.optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
            'total_loss': total_loss.item()
        }
    
    def get_stats(self) -> Dict:
        """Get policy statistics"""
        recent_win_rate = sum(self.recent_results) / len(self.recent_results) if self.recent_results else 0.5

        # Get gate rejection stats
        gate_tracker = getattr(self, '_gate_tracker', {})

        return {
            'total_trades': self.total_trades,
            'wins': self.wins,
            'losses': self.losses,
            'win_rate': self.win_rate,
            'recent_20_win_rate': recent_win_rate,
            'consecutive_losses': self.consecutive_losses,
            'total_pnl': self.total_pnl,
            'mode': 'BANDIT' if self.is_bandit_mode else 'RL',
            'experience_buffer_size': len(self.experience_buffer),
            'min_confidence': self.min_confidence_to_trade,
            'profit_target': self.profit_target_pct,
            'stop_loss': self.max_position_loss_pct,
            'trailing_stop_activation': self.trailing_stop_activation,
            'trailing_stop_distance': self.trailing_stop_distance,
            # Gate rejection stats
            'gate_total_signals': gate_tracker.get('total', 0),
            'gate_hmm_uncertain': gate_tracker.get('hmm_uncertain', 0),
            'gate_high_volatility': gate_tracker.get('high_volatility', 0),
            'gate_neural_disagrees': gate_tracker.get('neural_disagrees', 0),
            'gate_hmm_neutral': gate_tracker.get('hmm_neutral', 0),
        }
    
    def log_debug_state(self, state: TradeState, action: int, confidence: float, details: Dict):
        """
        Log comprehensive debug information for troubleshooting.
        Call this after select_action() to see what's happening.
        """
        action_names = {0: 'HOLD', 1: 'BUY_CALL', 2: 'BUY_PUT', 3: 'EXIT'}
        
        logger.info("=" * 60)
        logger.info(f"🔍 DEBUG: Decision Analysis")
        logger.info("=" * 60)
        
        # Current position status
        if state.is_in_trade:
            logger.info(f"📊 POSITION: {'CALL' if state.is_call else 'PUT'} | "
                       f"PnL: {state.unrealized_pnl_pct:+.1%} | Max: {state.max_pnl_seen:+.1%} | "
                       f"Held: {state.minutes_held}m")
        else:
            logger.info(f"📊 POSITION: None (looking for entry)")
        
        # Neural network signals
        logger.info(f"🧠 NEURAL: direction={state.predicted_direction:+.3f} | "
                   f"confidence={state.prediction_confidence:.1%} | "
                   f"momentum={state.momentum_5m:+.4f}")
        
        # HMM regime
        trend_label = "BULLISH" if state.hmm_trend > 0.6 else "BEARISH" if state.hmm_trend < 0.4 else "NEUTRAL"
        vol_label = "HIGH" if state.hmm_volatility > 0.6 else "LOW" if state.hmm_volatility < 0.4 else "NORMAL"
        logger.info(f"🎰 HMM: trend={state.hmm_trend:.2f} ({trend_label}) | "
                   f"vol={state.hmm_volatility:.2f} ({vol_label}) | "
                   f"conf={state.hmm_confidence:.1%}")
        
        # Market conditions
        logger.info(f"📈 MARKET: VIX={state.vix_level:.1f} | volume_spike={state.volume_spike:.2f} | "
                   f"theta={state.estimated_theta_decay:.2%}/day")
        
        # Quality gate analysis
        logger.info(f"🚪 GATES:")
        conf_req = self.min_confidence_to_trade
        if self.consecutive_losses >= 2:
            conf_req = min(0.70, conf_req + 0.10)
        logger.info(f"   • Confidence: {state.prediction_confidence:.1%} {'✅' if state.prediction_confidence >= conf_req else '❌'} (need {conf_req:.1%})")
        logger.info(f"   • VIX range: {state.vix_level:.1f} {'✅' if 13 <= state.vix_level <= 30 else '❌'} (need 13-30)")
        logger.info(f"   • Direction: {abs(state.predicted_direction):.3f} {'✅' if abs(state.predicted_direction) >= 0.02 else '❌'} (need ≥0.02)")
        
        # Alignment check
        is_bullish = state.predicted_direction > 0.02
        is_bearish = state.predicted_direction < -0.02
        hmm_bullish = state.hmm_trend > 0.55
        hmm_bearish = state.hmm_trend < 0.45
        aligned = (is_bullish and hmm_bullish) or (is_bearish and hmm_bearish)
        conflicting = (is_bullish and hmm_bearish) or (is_bearish and hmm_bullish)
        logger.info(f"   • HMM alignment: {'✅ ALIGNED' if aligned else '⚠️ NEUTRAL' if not conflicting else '❌ CONFLICTING'}")
        
        # Decision
        logger.info(f"➡️ ACTION: {action_names[action]} ({confidence:.1%}) - {details.get('reason', 'N/A')}")
        
        # Policy state
        logger.info(f"📉 POLICY: mode={'BANDIT' if self.is_bandit_mode else 'RL'} | "
                   f"trades={self.total_trades} | win_rate={self.win_rate:.1%} | "
                   f"consec_losses={self.consecutive_losses}")
        logger.info("=" * 60)
    
    def save(self, path: str):
        """Save policy state"""
        torch.save({
            'network_state': self.network.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'stats': {
                'total_trades': self.total_trades,
                'wins': self.wins,
                'losses': self.losses,
                'total_pnl': self.total_pnl,
                'consecutive_losses': self.consecutive_losses,
                'recent_results': self.recent_results,
            },
            'config': {
                'min_confidence': self.min_confidence_to_trade,
                'profit_target': self.profit_target_pct,
                'stop_loss': self.max_position_loss_pct,
                'trailing_stop_activation': self.trailing_stop_activation,
                'trailing_stop_distance': self.trailing_stop_distance,
            },
            'version': 'unified_v2'  # Updated version
        }, path)
        logger.info(f"💾 Unified RL Policy saved to {path} (v2, win_rate={self.win_rate:.1%})")
    
    def load(self, path: str) -> bool:
        """Load policy state"""
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            self.network.load_state_dict(checkpoint['network_state'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            
            stats = checkpoint.get('stats', {})
            self.total_trades = stats.get('total_trades', 0)
            self.wins = stats.get('wins', 0)
            self.losses = stats.get('losses', 0)
            self.total_pnl = stats.get('total_pnl', 0.0)
            self.consecutive_losses = stats.get('consecutive_losses', 0)
            self.recent_results = stats.get('recent_results', [])
            
            config = checkpoint.get('config', {})
            self.min_confidence_to_trade = config.get('min_confidence', self.min_confidence_to_trade)
            self.profit_target_pct = config.get('profit_target', self.profit_target_pct)
            self.max_position_loss_pct = config.get('stop_loss', self.max_position_loss_pct)
            self.trailing_stop_activation = config.get('trailing_stop_activation', self.trailing_stop_activation)
            self.trailing_stop_distance = config.get('trailing_stop_distance', self.trailing_stop_distance)
            
            version = checkpoint.get('version', 'v1')
            logger.info(f"✅ Unified RL Policy loaded from {path} ({version})")
            logger.info(f"   Trades: {self.total_trades}, Win rate: {self.win_rate:.1%}, Consec losses: {self.consecutive_losses}")
            return True
        except FileNotFoundError:
            logger.info(f"No saved policy at {path}, starting fresh")
            return False
        except Exception as e:
            logger.error(f"Error loading policy: {e}")
            return False


# ============================================================================
# ALTERNATIVE: Even simpler rule-based approach with learned parameters
# ============================================================================

class AdaptiveRulesPolicy:
    """
    Rule-based policy with learned parameters.
    
    Sometimes simpler is better! This uses explicit rules
    but learns optimal thresholds from experience.
    
    Advantages:
    - Fully interpretable
    - Faster to learn (fewer parameters)
    - More stable in production
    - Easier to debug
    
    OPTIMIZED FOR HIGH WIN RATE:
    - Conservative entries (fewer but higher quality trades)
    - Tight stop losses (cut losers fast)
    - Let winners run with trailing stops
    """
    
    def __init__(self):
        # Learned parameters (OPTIMIZED for high win rate)
        self.params = {
            # Entry (CONSERVATIVE)
            'min_confidence': 0.55,           # RAISED from 0.40 - require higher confidence
            'min_predicted_return': 0.02,     # RAISED from 0.003 - require clearer signals
            'max_vix_for_entry': 28.0,        # LOWERED from 30 - avoid high vol
            'min_vix_for_entry': 13.0,        # NEW - avoid low vol (hard to profit)
            'require_trend_alignment': True,  # NEW - only trade with trend
            
            # Exit - profit taking (LET WINNERS RUN)
            'profit_target_low_conf': 0.12,   # LOWERED from 0.15 - take profits faster
            'profit_target_high_conf': 0.20,  # LOWERED from 0.30 - more realistic
            'confidence_threshold': 0.60,     # What counts as "high confidence"
            'trailing_stop_activation': 0.10, # NEW - activate trailing stop at 10%
            'trailing_stop_distance': 0.05,   # NEW - trail by 5%
            
            # Exit - stop loss (TIGHT)
            'stop_loss_tight': 0.06,  # TIGHTENED from 0.10 - cut losses faster
            'stop_loss_loose': 0.10,  # TIGHTENED from 0.20 - still tight for high conf
            
            # Exit - time based
            'max_hold_minutes_no_profit': 60,  # EXTENDED from 45 - give trades time
            'min_hold_minutes': 5,  # Minimum hold to avoid churn
            
            # Exit - theta
            'theta_exit_threshold': 0.015,  # TIGHTENED from 0.02 - exit earlier on theta decay
        }
        
        # Learning: track which parameter values lead to wins/losses
        self.param_history: Dict[str, List[Tuple[float, bool]]] = {
            k: [] for k in self.params
        }
        
        self.total_trades = 0
        self.wins = 0
    
    def should_enter(
        self, 
        confidence: float, 
        predicted_return: float, 
        vix: float,
        direction: str,  # 'CALL' or 'PUT'
        hmm_trend: float = 0.5,  # NEW: HMM trend (0=bearish, 0.5=neutral, 1=bullish)
        hmm_volatility: float = 0.5  # NEW: HMM volatility
    ) -> Tuple[bool, str]:
        """Rule-based entry decision with trend alignment"""
        
        # Gate 1: Confidence check
        if confidence < self.params['min_confidence']:
            return False, f"Low confidence: {confidence:.1%} < {self.params['min_confidence']:.1%}"
        
        # Gate 2: Signal strength check
        if abs(predicted_return) < self.params['min_predicted_return']:
            return False, f"Weak signal: {predicted_return:.2%} < {self.params['min_predicted_return']:.2%}"
        
        # Gate 3: VIX range check
        if vix > self.params['max_vix_for_entry']:
            return False, f"High VIX: {vix:.1f} > {self.params['max_vix_for_entry']:.1f}"
        if vix < self.params['min_vix_for_entry']:
            return False, f"Low VIX: {vix:.1f} < {self.params['min_vix_for_entry']:.1f}"
        
        # Gate 4: Trend alignment check (CRITICAL for win rate)
        if self.params['require_trend_alignment']:
            # Avoid choppy markets
            if 0.4 <= hmm_trend <= 0.6 and hmm_volatility > 0.6:
                return False, f"Choppy market: trend={hmm_trend:.2f}, vol={hmm_volatility:.2f}"
            
            # CALL must align with bullish trend
            if direction == 'CALL':
                if hmm_trend < 0.45:  # Trend is bearish
                    return False, f"CALL vs bearish trend: {hmm_trend:.2f}"
            # PUT must align with bearish trend  
            elif direction == 'PUT':
                if hmm_trend > 0.55:  # Trend is bullish
                    return False, f"PUT vs bullish trend: {hmm_trend:.2f}"
        
        return True, "Signal meets all criteria"
    
    def should_exit(
        self,
        pnl_pct: float,
        minutes_held: int,
        entry_confidence: float,
        theta_decay: float,
        momentum_reversed: bool,
        max_pnl_seen: float = 0.0  # NEW: High water mark for trailing stop
    ) -> Tuple[bool, str]:
        """Rule-based exit decision with trailing stop"""
        
        # Determine thresholds based on entry confidence
        high_conf = entry_confidence >= self.params['confidence_threshold']
        profit_target = self.params['profit_target_high_conf'] if high_conf else self.params['profit_target_low_conf']
        stop_loss = self.params['stop_loss_loose'] if high_conf else self.params['stop_loss_tight']
        
        # Rule 0: Minimum hold time check (unless losing badly)
        if minutes_held < self.params['min_hold_minutes'] and pnl_pct > -stop_loss * 0.5:
            return False, f"Minimum hold: {minutes_held}m < {self.params['min_hold_minutes']}m"
        
        # Rule 1: Stop loss (FIRST - protect capital)
        if pnl_pct <= -stop_loss:
            return True, f"Stop loss hit: {pnl_pct:.1%} <= -{stop_loss:.1%}"
        
        # Rule 2: Trailing stop (protect profits)
        if max_pnl_seen >= self.params['trailing_stop_activation']:
            trailing_stop_level = max_pnl_seen - self.params['trailing_stop_distance']
            if pnl_pct <= trailing_stop_level:
                return True, f"Trailing stop: {pnl_pct:.1%} <= {trailing_stop_level:.1%} (max was {max_pnl_seen:.1%})"
        
        # Rule 3: Fixed profit target (only if not using trailing stop)
        if max_pnl_seen < self.params['trailing_stop_activation']:
            if pnl_pct >= profit_target:
                return True, f"Profit target hit: {pnl_pct:.1%} >= {profit_target:.1%}"
        
        # Rule 4: Momentum reversal (quick exit to protect gains)
        if momentum_reversed:
            if pnl_pct > 0.05:
                return True, f"Momentum reversed - protecting {pnl_pct:.1%} profit"
            elif pnl_pct < 0:
                return True, f"Momentum reversed while losing {pnl_pct:.1%}"
        
        # Rule 5: Theta eating profits (options-specific)
        if abs(theta_decay) > self.params['theta_exit_threshold'] and pnl_pct < 0.04:
            return True, f"Theta exit: {abs(theta_decay):.1%} daily decay, only {pnl_pct:.1%} profit"
        
        # Rule 6: Time-based exit for flat trades
        if minutes_held > self.params['max_hold_minutes_no_profit'] and pnl_pct < 0.03:
            return True, f"Time exit: {minutes_held}m with only {pnl_pct:.1%} profit"
        
        return False, "Hold position"
    
    def record_trade_outcome(self, was_win: bool, entry_confidence: float):
        """Learn from trade outcome by tracking which parameters worked"""
        self.total_trades += 1
        if was_win:
            self.wins += 1
        
        # Simple parameter adaptation: if winning, slightly widen targets
        # If losing, slightly tighten risk controls
        adaptation_rate = 0.01
        
        if was_win:
            # Winning - we can be slightly more aggressive
            self.params['stop_loss_tight'] = min(0.15, self.params['stop_loss_tight'] + adaptation_rate)
            self.params['profit_target_low_conf'] = max(0.10, self.params['profit_target_low_conf'] - adaptation_rate)
        else:
            # Losing - be more conservative
            self.params['stop_loss_tight'] = max(0.05, self.params['stop_loss_tight'] - adaptation_rate)
            self.params['min_confidence'] = min(0.50, self.params['min_confidence'] + adaptation_rate * 0.5)
    
    @property
    def win_rate(self) -> float:
        return self.wins / self.total_trades if self.total_trades > 0 else 0.5
    
    def get_params(self) -> Dict:
        """Get current learned parameters"""
        return {
            **self.params,
            'total_trades': self.total_trades,
            'win_rate': self.win_rate,
        }

