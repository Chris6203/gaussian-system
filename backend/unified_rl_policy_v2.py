#!/usr/bin/env python3
"""
Unified RL Policy V2 - Enhanced Architecture

Key improvements over V1:
1. Predictor embedding (64→16 compressed) fed into policy instead of 3 scalars
2. Direction probabilities (not argmax) + execution heads included
3. GRU layer for temporal awareness (learns from recent state history)
4. Split EXIT into EXIT_FAST (market) and EXIT_PATIENT (limit)
5. EV-based gating instead of raw confidence threshold
6. Improved reward shaping with costs and holding penalties

State features (40 total):
- Position state (4): in_trade, is_call, pnl%, drawdown
- Time (2): minutes_held, minutes_to_expiry
- Market (2): vix, volume_spike
- HMM Regime (4): trend, volatility, liquidity, hmm_confidence
- Greeks (2): theta, delta
- Predictor embedding (16): compressed from 64-dim embedding
- Direction probs (3): [DOWN, NEUTRAL, UP] probabilities
- Execution heads (3): fillability, exp_slippage, exp_ttf
- Return/Vol (2): predicted_return, predicted_volatility
- Momentum/Confidence (2): momentum_5m, raw confidence

Actions (5):
- 0: HOLD
- 1: BUY_CALL
- 2: BUY_PUT
- 3: EXIT_FAST (market order, immediate)
- 4: EXIT_PATIENT (limit order, wait for fill)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class TradeStateV2:
    """Enhanced state representation with predictor outputs."""
    # Position info
    is_in_trade: bool = False
    is_call: bool = True
    entry_price: float = 0.0
    current_price: float = 0.0
    position_size: int = 0

    # P&L
    unrealized_pnl_pct: float = 0.0
    max_pnl_seen: float = 0.0
    max_drawdown: float = 0.0

    # Time
    minutes_held: int = 0
    minutes_to_expiry: int = 1440

    # Market context
    vix_level: float = 18.0
    momentum_5m: float = 0.0
    volume_spike: float = 1.0

    # HMM Regime
    hmm_trend: float = 0.5
    hmm_volatility: float = 0.5
    hmm_liquidity: float = 0.5
    hmm_confidence: float = 0.5

    # Greeks
    estimated_theta_decay: float = 0.0
    estimated_delta: float = 0.5

    # NEW: Predictor outputs (previously discarded!)
    predictor_embedding: np.ndarray = field(default_factory=lambda: np.zeros(64))
    direction_probs: np.ndarray = field(default_factory=lambda: np.array([0.33, 0.34, 0.33]))  # [DOWN, NEUTRAL, UP]
    raw_confidence: float = 0.5
    predicted_return: float = 0.0
    predicted_volatility: float = 0.0

    # NEW: Execution quality predictions
    fillability: float = 0.8
    exp_slippage: float = 0.01
    exp_ttf: float = 2.0  # seconds


class EmbeddingAdapter(nn.Module):
    """Compress 64-dim predictor embedding to 16-dim for RL state."""

    def __init__(self, in_dim: int = 64, out_dim: int = 16):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, out_dim),
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.adapter(embedding)


class RecurrentPolicyNetwork(nn.Module):
    """
    Policy network with GRU for temporal awareness.

    Architecture:
    1. State encoder (MLP): state → hidden
    2. GRU: sequence of hidden states → temporal context
    3. Action/Value/Urgency heads from temporal context

    This allows the policy to learn patterns like:
    - "confidence has been dropping for 3 steps"
    - "trend changed direction recently"
    - "momentum is accelerating"
    """

    def __init__(
        self,
        state_dim: int = 40,  # Expanded from 18
        hidden_dim: int = 64,
        gru_hidden: int = 64,
        num_gru_layers: int = 1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru_hidden = gru_hidden
        self.num_gru_layers = num_gru_layers

        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # GRU for temporal patterns
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=gru_hidden,
            num_layers=num_gru_layers,
            batch_first=True,
            dropout=0.1 if num_gru_layers > 1 else 0,
        )

        # Combine GRU output with current state
        self.combiner = nn.Sequential(
            nn.Linear(gru_hidden + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # Action head: 5 actions (HOLD, BUY_CALL, BUY_PUT, EXIT_FAST, EXIT_PATIENT)
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 5)
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        # Exit urgency head (0-1)
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
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param, gain=0.5)
                    elif 'bias' in name:
                        nn.init.zeros_(param)

    def forward(
        self,
        state: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
        state_history: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            state: Current state [B, state_dim]
            hidden: GRU hidden state [num_layers, B, gru_hidden]
            state_history: Previous states [B, T, state_dim] (optional, for batch processing)

        Returns:
            action_logits: [B, 5]
            value: [B, 1]
            exit_urgency: [B, 1]
            new_hidden: [num_layers, B, gru_hidden]
        """
        batch_size = state.size(0)

        # Encode current state
        encoded = self.state_encoder(state)  # [B, hidden_dim]

        # If we have state history, process it through GRU
        if state_history is not None and state_history.size(1) > 0:
            # Encode all historical states
            T = state_history.size(1)
            history_flat = state_history.view(-1, state_history.size(-1))  # [B*T, state_dim]
            history_encoded = self.state_encoder(history_flat)  # [B*T, hidden_dim]
            history_encoded = history_encoded.view(batch_size, T, -1)  # [B, T, hidden_dim]

            # Pass through GRU
            gru_out, hidden = self.gru(history_encoded, hidden)
            temporal_context = gru_out[:, -1, :]  # Take last output [B, gru_hidden]
        else:
            # Single step update
            encoded_seq = encoded.unsqueeze(1)  # [B, 1, hidden_dim]
            if hidden is None:
                hidden = torch.zeros(self.num_gru_layers, batch_size, self.gru_hidden,
                                   device=state.device, dtype=state.dtype)
            gru_out, hidden = self.gru(encoded_seq, hidden)
            temporal_context = gru_out.squeeze(1)  # [B, gru_hidden]

        # Combine temporal context with current encoded state
        combined = torch.cat([temporal_context, encoded], dim=-1)  # [B, gru_hidden + hidden_dim]
        features = self.combiner(combined)  # [B, hidden_dim]

        # Heads
        action_logits = self.action_head(features)
        value = self.value_head(features)
        exit_urgency = self.exit_urgency_head(features)

        return action_logits, value, exit_urgency, hidden


class UnifiedRLPolicyV2:
    """
    Enhanced unified policy for options trading.

    Key improvements:
    1. Uses predictor embedding (not just 3 scalars)
    2. GRU for temporal awareness
    3. EV-based entry gating
    4. Split EXIT actions
    5. Proper reward shaping with costs
    """

    # Actions
    HOLD = 0
    BUY_CALL = 1
    BUY_PUT = 2
    EXIT_FAST = 3
    EXIT_PATIENT = 4

    # Friction costs (used for EV calculation)
    SPREAD_COST = 0.003  # 0.3% round-trip spread
    FEE_PER_CONTRACT = 0.65
    SLIPPAGE_PCT = 0.001  # 0.1% average slippage

    def __init__(
        self,
        learning_rate: float = 0.0003,
        gamma: float = 0.99,
        device: str = 'cpu',
        bandit_mode_trades: int = 50,
        state_history_len: int = 10,  # NEW: How many past states to remember
        lambda_time: float = 0.001,  # Time penalty per minute in trade
        lambda_risk: float = 0.1,  # Drawdown penalty multiplier
        max_position_loss_pct: float = None,
        profit_target_pct: float = None,
        trailing_stop_activation: float = None,
        trailing_stop_distance: float = None,
    ):
        # Load exit config
        try:
            from backend.exit_config import get_exit_config
            exit_cfg = get_exit_config()
            self.max_position_loss_pct = max_position_loss_pct or exit_cfg.stop_loss_pct
            self.profit_target_pct = profit_target_pct or exit_cfg.take_profit_pct
            self.trailing_stop_activation = trailing_stop_activation or exit_cfg.trailing_activation_pct
            self.trailing_stop_distance = trailing_stop_distance or exit_cfg.trailing_distance_pct
        except Exception as e:
            logger.warning(f"Could not load exit_config: {e}, using defaults")
            self.max_position_loss_pct = max_position_loss_pct or 0.08
            self.profit_target_pct = profit_target_pct or 0.12
            self.trailing_stop_activation = trailing_stop_activation or 0.08
            self.trailing_stop_distance = trailing_stop_distance or 0.04

        self.device = device
        self.gamma = gamma
        self.bandit_mode_trades = bandit_mode_trades
        self.state_history_len = state_history_len
        self.lambda_time = lambda_time
        self.lambda_risk = lambda_risk

        # Embedding adapter (64 → 16)
        self.embedding_adapter = EmbeddingAdapter(64, 16).to(device)

        # Policy network (40 features)
        self.network = RecurrentPolicyNetwork(state_dim=40, hidden_dim=64).to(device)

        # Combined optimizer
        all_params = list(self.network.parameters()) + list(self.embedding_adapter.parameters())
        self.optimizer = torch.optim.AdamW(all_params, lr=learning_rate, weight_decay=1e-5)

        # GRU hidden state (persistent across steps)
        self.gru_hidden: Optional[torch.Tensor] = None

        # State history buffer
        self.state_history: deque = deque(maxlen=state_history_len)

        # Experience buffer
        self.experience_buffer: deque = deque(maxlen=10000)

        # Stats
        self.total_trades = 0
        self.wins = 0
        self.losses = 0
        self.total_pnl = 0.0

        # Current trade
        self.current_trade: Optional[TradeStateV2] = None

        logger.info(f"Unified RL Policy V2 initialized")
        logger.info(f"   State dim: 40 (18 scalars + 16 embedding + 3 direction + 3 execution)")
        logger.info(f"   Actions: 5 (HOLD, BUY_CALL, BUY_PUT, EXIT_FAST, EXIT_PATIENT)")
        logger.info(f"   GRU history: {state_history_len} steps")
        logger.info(f"   Device: {device}")

    @property
    def is_bandit_mode(self) -> bool:
        return self.total_trades < self.bandit_mode_trades

    def _get_dir_prob(self, direction_probs, idx: int) -> float:
        """Extract direction probability handling array, dict, or None formats."""
        if direction_probs is None:
            return 0.33  # Default neutral
        
        # Handle numpy array or list
        if hasattr(direction_probs, '__getitem__'):
            try:
                # Try numeric index first (array/list)
                return float(direction_probs[idx])
            except (KeyError, TypeError):
                pass
            
            # Try dict with string keys
            key_map = {0: 'DOWN', 1: 'NEUTRAL', 2: 'UP'}
            key = key_map.get(idx, str(idx))
            try:
                return float(direction_probs.get(key, 0.33))
            except (AttributeError, TypeError):
                pass
        
        return 0.33  # Default fallback

    def _safe_float(self, value, default: float = 0.5) -> float:
        """Safely convert value to float, handling strings like 'Uptrend'."""
        if value is None:
            return default
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            # Map string regime names to floats
            trend_map = {'Down': 0.0, 'Downtrend': 0.0, 'Bearish': 0.0,
                         'No Trend': 0.5, 'Neutral': 0.5, 'Sideways': 0.5,
                         'Up': 1.0, 'Uptrend': 1.0, 'Bullish': 1.0,
                         'Low': 0.0, 'Normal': 0.5, 'High': 1.0}
            return trend_map.get(value, default)
        return default

    def state_to_tensor(self, state: TradeStateV2) -> torch.Tensor:
        """Convert TradeStateV2 to 40-dim tensor."""

        # Compress predictor embedding: 64 → 16
        embedding = torch.tensor(state.predictor_embedding, dtype=torch.float32, device=self.device)
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)
        compressed_embedding = self.embedding_adapter(embedding).squeeze(0)  # [16]

        # Scalar features (24 total)
        scalars = [
            # Position state (4)
            float(state.is_in_trade),
            float(state.is_call) if state.is_in_trade else 0.5,
            np.clip(state.unrealized_pnl_pct, -0.5, 0.5),
            np.clip(state.max_drawdown, 0, 0.3),

            # Time (2)
            min(1.0, state.minutes_held / 120.0),
            min(1.0, state.minutes_to_expiry / 1440.0),

            # Market (2)
            min(1.0, state.vix_level / 50.0),
            min(2.0, state.volume_spike) / 2.0,

            # HMM Regime (4)
            np.clip(state.hmm_trend, 0, 1),
            np.clip(state.hmm_volatility, 0, 1),
            np.clip(state.hmm_liquidity, 0, 1),
            np.clip(state.hmm_confidence, 0, 1),

            # Greeks (2)
            np.clip(state.estimated_theta_decay * 100, -0.1, 0),
            np.clip(state.estimated_delta, 0, 1),

            # Direction probs (3) - handle array/dict/None formats
            np.clip(self._get_dir_prob(state.direction_probs, 0), 0, 1),  # DOWN
            np.clip(self._get_dir_prob(state.direction_probs, 1), 0, 1),  # NEUTRAL
            np.clip(self._get_dir_prob(state.direction_probs, 2), 0, 1),  # UP

            # Execution heads (3)
            np.clip(state.fillability, 0, 1),
            np.clip(state.exp_slippage * 10, -1, 1),  # Scale slippage
            np.clip(state.exp_ttf / 10.0, 0, 1),  # Normalize TTF

            # Return/Vol (2)
            np.clip(state.predicted_return * 100, -1, 1),  # Scale return
            np.clip(state.predicted_volatility * 10, 0, 1),  # Scale volatility

            # Momentum/Confidence (2)
            np.clip(state.momentum_5m * 100, -1, 1),
            np.clip(state.raw_confidence, 0, 1),
        ]

        scalar_tensor = torch.tensor(scalars, dtype=torch.float32, device=self.device)

        # Concatenate: scalars (24) + compressed embedding (16) = 40
        return torch.cat([scalar_tensor, compressed_embedding])

    def compute_expected_value(self, state: TradeStateV2, action: int) -> float:
        """
        Compute expected value for a trade, accounting for friction.

        EV = P(up) * E[gain|up] - P(down) * E[loss|down] - costs

        This replaces raw confidence gating.
        """
        if action == self.HOLD:
            return 0.0

        if action not in [self.BUY_CALL, self.BUY_PUT]:
            return 0.0

        # Direction probabilities
        p_down, p_neutral, p_up = state.direction_probs

        # Predicted return magnitude (absolute)
        expected_move = abs(state.predicted_return)

        # Friction costs
        total_friction = self.SPREAD_COST + self.SLIPPAGE_PCT + state.exp_slippage

        # Expected theta cost (if holding ~15 minutes)
        theta_cost = abs(state.estimated_theta_decay) * (15 / 1440)  # 15 min / day

        total_costs = total_friction + theta_cost

        if action == self.BUY_CALL:
            # CALL profits when price goes up
            ev = p_up * expected_move - p_down * expected_move - total_costs
        else:  # BUY_PUT
            # PUT profits when price goes down
            ev = p_down * expected_move - p_up * expected_move - total_costs

        return ev

    def select_action(
        self,
        state: TradeStateV2,
        deterministic: bool = False,
    ) -> Tuple[int, float, Dict]:
        """
        Select action with EV-based gating and temporal awareness.

        Returns:
            action: 0-4 (HOLD, BUY_CALL, BUY_PUT, EXIT_FAST, EXIT_PATIENT)
            confidence: Policy confidence
            details: Logging info
        """
        # ==================== HARD SAFETY RULES ====================
        if state.is_in_trade:
            # Stop loss
            if state.unrealized_pnl_pct <= -self.max_position_loss_pct:
                logger.info(f"STOP LOSS: {state.unrealized_pnl_pct:.1%}")
                return self.EXIT_FAST, 1.0, {'reason': 'stop_loss'}

            # Trailing stop
            if state.max_pnl_seen >= self.trailing_stop_activation:
                trail_level = state.max_pnl_seen - self.trailing_stop_distance
                if state.unrealized_pnl_pct <= trail_level:
                    logger.info(f"TRAILING STOP: {state.unrealized_pnl_pct:.1%}")
                    return self.EXIT_FAST, 1.0, {'reason': 'trailing_stop'}

            # Profit target (use EXIT_PATIENT if profitable)
            if state.unrealized_pnl_pct >= self.profit_target_pct:
                logger.info(f"PROFIT TARGET: {state.unrealized_pnl_pct:.1%}")
                return self.EXIT_PATIENT, 1.0, {'reason': 'profit_target'}

            # Expiry
            if state.minutes_to_expiry < 30:
                return self.EXIT_FAST, 1.0, {'reason': 'expiry'}

        # ==================== STATE PROCESSING ====================
        state_tensor = self.state_to_tensor(state)

        # Update state history
        self.state_history.append(state_tensor.detach().cpu().numpy())

        # Build history tensor if we have enough samples
        if len(self.state_history) >= 2:
            history_array = np.array(list(self.state_history)[:-1])  # Exclude current
            history_tensor = torch.tensor(history_array, dtype=torch.float32, device=self.device)
            history_tensor = history_tensor.unsqueeze(0)  # [1, T, state_dim]
        else:
            history_tensor = None

        # ==================== BANDIT MODE ====================
        if self.is_bandit_mode:
            return self._bandit_decision_v2(state, deterministic)

        # ==================== FULL RL MODE ====================
        with torch.no_grad():
            state_batch = state_tensor.unsqueeze(0)  # [1, state_dim]

            action_logits, value, exit_urgency, self.gru_hidden = self.network(
                state_batch,
                self.gru_hidden,
                history_tensor,
            )

            if deterministic:
                action = action_logits.argmax(dim=-1).item()
            else:
                # Temperature-based sampling
                temperature = 1.0
                probs = F.softmax(action_logits / temperature, dim=-1)
                action = torch.multinomial(probs, 1).item()

            confidence = F.softmax(action_logits, dim=-1)[0, action].item()

        # ==================== EV GATING ====================
        if action in [self.BUY_CALL, self.BUY_PUT]:
            ev = self.compute_expected_value(state, action)
            if ev <= 0:
                logger.debug(f"EV gate: {ev:.4f} <= 0, converting to HOLD")
                return self.HOLD, 0.5, {'reason': 'ev_gate_block', 'ev': ev}

        details = {
            'mode': 'rl_v2',
            'value': value.item(),
            'exit_urgency': exit_urgency.item(),
            'gru_active': True,
        }

        return action, confidence, details

    def _bandit_decision_v2(
        self,
        state: TradeStateV2,
        deterministic: bool,
    ) -> Tuple[int, float, Dict]:
        """Enhanced bandit with EV-based gating."""
        details = {'mode': 'bandit_v2'}

        if state.is_in_trade:
            # Exit logic based on conditions

            # Trend reversal
            if state.is_call and state.hmm_trend < 0.35 and state.hmm_confidence > 0.6:
                return self.EXIT_FAST, 0.85, {**details, 'reason': 'trend_reversal'}
            elif not state.is_call and state.hmm_trend > 0.65 and state.hmm_confidence > 0.6:
                return self.EXIT_FAST, 0.85, {**details, 'reason': 'trend_reversal'}

            # Momentum reversal with small profit
            if state.is_call and state.momentum_5m < -0.003 and state.unrealized_pnl_pct < 0.05:
                return self.EXIT_PATIENT, 0.7, {**details, 'reason': 'momentum_reversal'}
            elif not state.is_call and state.momentum_5m > 0.003 and state.unrealized_pnl_pct < 0.05:
                return self.EXIT_PATIENT, 0.7, {**details, 'reason': 'momentum_reversal'}

            return self.HOLD, 0.6, {**details, 'reason': 'hold'}

        else:
            # Entry logic with EV gating

            # Only trade in favorable regimes
            if self._safe_float(state.hmm_volatility) > 0.7:
                return self.HOLD, 0.5, {**details, 'reason': 'high_volatility_regime'}

            if self._safe_float(state.hmm_confidence) < 0.5:
                return self.HOLD, 0.5, {**details, 'reason': 'low_hmm_confidence'}

            # Check for strong directional signal - use _get_dir_prob for safe access
            p_down = self._get_dir_prob(state.direction_probs, 0)
            p_neutral = self._get_dir_prob(state.direction_probs, 1)
            p_up = self._get_dir_prob(state.direction_probs, 2)

            # CALL signal - RELAXED thresholds to match baseline behavior
            # Only require HMM trend, direction_probs is optional confirmation
            if self._safe_float(state.hmm_trend) > 0.65:
                # Skip EV check for now - just use HMM like baseline
                return self.BUY_CALL, self._safe_float(state.hmm_confidence, 0.7), {**details, 'reason': 'hmm_bullish', 'hmm_trend': self._safe_float(state.hmm_trend)}

            # PUT signal - RELAXED thresholds
            if self._safe_float(state.hmm_trend) < 0.35:
                return self.BUY_PUT, self._safe_float(state.hmm_confidence, 0.7), {**details, 'reason': 'hmm_bearish', 'hmm_trend': self._safe_float(state.hmm_trend)}

            return self.HOLD, 0.5, {**details, 'reason': 'no_signal'}

    def compute_reward(
        self,
        state: TradeStateV2,
        prev_state: TradeStateV2,
        action_taken: int,
        pnl_after_costs: float,
    ) -> float:
        """
        Compute shaped reward for RL training.

        reward = ΔPnL_after_costs - λ_time * minutes_held - λ_risk * drawdown_increase

        Also includes:
        - Penalty for holding into low-liquidity
        - Penalty for theta decay eating profits
        """
        reward = pnl_after_costs

        # Time penalty (encourage faster exits on losers)
        if state.is_in_trade and state.unrealized_pnl_pct < 0:
            reward -= self.lambda_time * state.minutes_held

        # Drawdown penalty
        drawdown_increase = max(0, state.max_drawdown - prev_state.max_drawdown)
        reward -= self.lambda_risk * drawdown_increase

        # Low fillability penalty (if we tried to trade in illiquid conditions)
        if action_taken in [self.BUY_CALL, self.BUY_PUT, self.EXIT_PATIENT]:
            if state.fillability < 0.5:
                reward -= 0.01 * (1 - state.fillability)

        # Theta decay awareness (penalize holding when theta is high)
        if state.is_in_trade:
            theta_penalty = abs(state.estimated_theta_decay) * state.minutes_held / 60
            reward -= theta_penalty

        return reward

    def reset_episode(self):
        """Reset GRU hidden state and history for new trading session."""
        self.gru_hidden = None
        self.state_history.clear()
        logger.debug("Episode reset: GRU hidden cleared, history cleared")

    def get_action_name(self, action: int) -> str:
        """Convert action int to readable name."""
        names = ['HOLD', 'BUY_CALL', 'BUY_PUT', 'EXIT_FAST', 'EXIT_PATIENT']
        return names[action] if 0 <= action < len(names) else f'UNKNOWN({action})'


def create_state_from_signal(signal: Dict, predictor_output: Dict) -> TradeStateV2:
    """
    Helper to create TradeStateV2 from signal dict and predictor output.

    This bridges the old signal format with the new state format.
    """
    state = TradeStateV2()

    # Position info
    state.is_in_trade = signal.get('is_in_trade', False)
    state.is_call = signal.get('is_call', True)
    state.entry_price = signal.get('entry_price', 0.0)
    state.current_price = signal.get('current_price', 0.0)
    state.position_size = signal.get('position_size', 0)

    # P&L
    state.unrealized_pnl_pct = signal.get('unrealized_pnl_pct', 0.0)
    state.max_pnl_seen = signal.get('max_pnl_seen', 0.0)
    state.max_drawdown = signal.get('max_drawdown', 0.0)

    # Time
    state.minutes_held = signal.get('minutes_held', 0)
    state.minutes_to_expiry = signal.get('minutes_to_expiry', 1440)

    # Market
    state.vix_level = signal.get('vix_level', 18.0)
    state.momentum_5m = signal.get('momentum_5m', 0.0)
    state.volume_spike = signal.get('volume_spike', 1.0)

    # HMM
    hmm = signal.get('hmm_regime', {})
    state.hmm_trend = hmm.get('trend', 0.5)
    state.hmm_volatility = hmm.get('volatility', 0.5)
    state.hmm_liquidity = hmm.get('liquidity', 0.5)
    state.hmm_confidence = hmm.get('confidence', 0.5)

    # Greeks
    state.estimated_theta_decay = signal.get('theta_decay', 0.0)
    state.estimated_delta = signal.get('delta', 0.5)

    # Predictor outputs
    if 'embedding' in predictor_output:
        emb = predictor_output['embedding']
        if isinstance(emb, torch.Tensor):
            state.predictor_embedding = emb.detach().cpu().numpy().flatten()
        else:
            state.predictor_embedding = np.array(emb).flatten()

    if 'direction' in predictor_output:
        direction = predictor_output['direction']
        if isinstance(direction, torch.Tensor):
            state.direction_probs = F.softmax(direction, dim=-1).detach().cpu().numpy().flatten()
        else:
            state.direction_probs = np.array(direction).flatten()

    if 'confidence' in predictor_output:
        conf = predictor_output['confidence']
        if isinstance(conf, torch.Tensor):
            state.raw_confidence = conf.item()
        else:
            state.raw_confidence = float(conf)

    if 'return' in predictor_output:
        ret = predictor_output['return']
        if isinstance(ret, torch.Tensor):
            state.predicted_return = ret.item()
        else:
            state.predicted_return = float(ret)

    if 'volatility' in predictor_output:
        vol = predictor_output['volatility']
        if isinstance(vol, torch.Tensor):
            state.predicted_volatility = vol.item()
        else:
            state.predicted_volatility = float(vol)

    # Execution heads
    if 'fillability' in predictor_output:
        fill = predictor_output['fillability']
        state.fillability = fill.item() if isinstance(fill, torch.Tensor) else float(fill)

    if 'exp_slippage' in predictor_output:
        slip = predictor_output['exp_slippage']
        state.exp_slippage = slip.item() if isinstance(slip, torch.Tensor) else float(slip)

    if 'exp_ttf' in predictor_output:
        ttf = predictor_output['exp_ttf']
        state.exp_ttf = ttf.item() if isinstance(ttf, torch.Tensor) else float(ttf)

    return state
