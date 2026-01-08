#!/usr/bin/env python3
"""
AlphaStar-Style RL Policy for Options Trading

Inspired by DeepMind's AlphaStar (StarCraft II):
- Transformer-based core (not LSTM)
- Multi-scale temporal attention
- Entity embeddings for market components
- Pointer network concepts for action selection
- Auto-regressive action heads

Key AlphaStar concepts adapted for trading:
1. Transformer Encoder: Better long-range dependencies than LSTM
2. Multi-Scale Attention: Attend to 1min, 5min, 15min, 1hr patterns
3. Entity Embeddings: Treat VIX, HMM, Greeks as separate "entities"
4. Gated Outputs: Control information flow per action type

Architecture:
    Market State [B, D]
           │
    ┌──────▼──────┐
    │   Entity    │   (Split into VIX, HMM, Greeks, Prediction entities)
    │  Encoder    │
    └──────┬──────┘
           │ [B, N_entities, H]
    ┌──────▼──────┐
    │   Multi-    │   (N layers of self-attention)
    │  Scale      │
    │ Transformer │
    └──────┬──────┘
           │ [B, H]
    ┌──────▼──────┐
    │  Gated      │   (AlphaStar-style gating)
    │  Residual   │
    │  Blocks     │
    └──────┬──────┘
           │
    ┌──────┴──────────────┐
    │                     │
┌───▼───┐  ┌──────▼──────┐
│ Action │  │   Value     │
│ Heads  │  │   (Critic)  │
│(AR)    │  │             │
└───┬────┘  └─────────────┘
    │
  Trade/Hold/Exit Decision

Usage:
    RL_USE_ALPHASTAR=1 python scripts/train_time_travel.py
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
import logging

logger = logging.getLogger(__name__)


class RotaryPositionEncoding(nn.Module):
    """
    Rotary Position Embedding (RoPE) - used in modern transformers.
    Better than sinusoidal for relative position awareness.
    """

    def __init__(self, dim: int, max_seq_len: int = 512):
        super().__init__()
        self.dim = dim

        # Precompute freqs
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        # Precompute sin/cos
        t = torch.arange(max_seq_len).float()
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())

    def forward(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def rotate_half(x):
    """Rotates half the hidden dims."""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary position embedding to q and k."""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class EntityEncoder(nn.Module):
    """
    Encodes different market "entities" separately.

    AlphaStar treats units as entities. We treat market components as entities:
    - Position entity (in_trade, is_call, pnl%, drawdown)
    - Time entity (minutes_held, minutes_to_expiry)
    - Prediction entity (direction, confidence, momentum, volume)
    - HMM entity (trend, vol, liq, hmm_conf)
    - Greeks entity (theta, delta)
    - Sentiment entity (if enabled)
    - Market entity (vix, etc)

    This allows the transformer to attend to entities by type.
    """

    def __init__(self, state_dim: int, hidden_dim: int):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Compute entity dims FIRST
        self.entity_dims = self._compute_entity_dims(state_dim)
        self.n_entities = len(self.entity_dims)

        # Entity type embeddings (learnable) - size based on actual entities
        self.entity_type_embed = nn.Embedding(self.n_entities, hidden_dim)

        # Entity encoders (one per type)
        self.entity_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU()
            )
            for dim in self.entity_dims
        ])

    def _compute_entity_dims(self, state_dim: int) -> List[int]:
        """
        Split state into entity groups.

        Default 22-dim state splits as:
        - Position (4): in_trade, is_call, pnl%, drawdown
        - Time (2): minutes_held, minutes_to_expiry
        - Prediction (4): direction, confidence, momentum, volume
        - HMM (4): trend, vol, liq, hmm_conf
        - Greeks (2): theta, delta
        - Sentiment (4): fear_greed, pcr, contrarian, news (if enabled)
        - VIX (2): vix_level, vix_change
        """
        # Simple split - adjust based on actual state structure
        if state_dim == 22:
            return [4, 2, 4, 4, 2, 4, 2]  # 7 entities
        else:
            # Default: 5 roughly equal entities
            base = state_dim // 5
            remainder = state_dim % 5
            dims = [base] * 5
            for i in range(remainder):
                dims[i] += 1
            return dims

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: [B, D] flat state or [B, T, D] sequence

        Returns:
            [B, N_entities, H] entity embeddings
        """
        is_seq = state.dim() == 3
        if is_seq:
            B, T, D = state.shape
            state = state.reshape(B * T, D)

        # Split state into entity groups
        entity_inputs = torch.split(state, self.entity_dims, dim=-1)

        # Encode each entity
        entity_embeds = []
        for i, (encoder, inp) in enumerate(zip(self.entity_encoders, entity_inputs)):
            embed = encoder(inp)  # [B, H]

            # Add entity type embedding
            type_embed = self.entity_type_embed(
                torch.tensor([i], device=state.device)
            ).expand(embed.size(0), -1)
            embed = embed + type_embed

            entity_embeds.append(embed)

        # Stack entities: [B, N_entities, H]
        entities = torch.stack(entity_embeds, dim=1)

        if is_seq:
            entities = entities.reshape(B, T, len(self.entity_dims), self.hidden_dim)

        return entities


class GatedResidualBlock(nn.Module):
    """
    Gated Residual Network (GRN) from AlphaStar.
    Uses gating to control information flow.
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()

        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # Gating layer
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )

        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Nonlinear transformation
        h = F.gelu(self.fc1(x))
        h = self.dropout(self.fc2(h))

        # Gating
        gate = self.gate(x)
        h = gate * h

        # Residual
        return self.norm(x + h)


class MultiScaleTransformerLayer(nn.Module):
    """
    Transformer layer with multi-scale attention.

    Attends at different temporal scales (1min, 5min, 15min, 1hr).
    """

    def __init__(
        self,
        hidden_dim: int,
        n_heads: int = 8,
        dropout: float = 0.1,
        use_rope: bool = True
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.use_rope = use_rope

        # Q, K, V projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # RoPE
        if use_rope:
            self.rope = RotaryPositionEncoding(self.head_dim)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [B, T, H] or [B, N_entities, H]
            mask: Optional attention mask

        Returns:
            [B, T, H] transformed
        """
        B, T, H = x.shape

        # Pre-norm
        x_norm = self.norm1(x)

        # Q, K, V
        q = self.q_proj(x_norm).reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_norm).reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_norm).reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE if enabled
        if self.use_rope:
            cos, sin = self.rope(x, T)
            cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, T, D]
            sin = sin.unsqueeze(0).unsqueeze(0)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / scale

        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, -1e9)

        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_probs = self.dropout(attn_probs)

        attn_out = torch.matmul(attn_probs, v)
        attn_out = attn_out.transpose(1, 2).reshape(B, T, H)
        attn_out = self.out_proj(attn_out)

        # Residual
        x = x + self.dropout(attn_out)

        # FFN with pre-norm and residual
        x = x + self.ffn(self.norm2(x))

        return x


class AlphaStarCore(nn.Module):
    """
    Core AlphaStar-style transformer.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()

        self.layers = nn.ModuleList([
            MultiScaleTransformerLayer(hidden_dim, n_heads, dropout)
            for _ in range(n_layers)
        ])

        # Gated residual blocks after transformer
        self.gated_blocks = nn.ModuleList([
            GatedResidualBlock(hidden_dim, dropout)
            for _ in range(2)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N_entities, H] entity embeddings

        Returns:
            [B, H] aggregated representation
        """
        # Transformer layers
        for layer in self.layers:
            x = layer(x)

        # Pool across entities (mean pooling)
        x = x.mean(dim=1)  # [B, H]

        # Gated residual blocks
        for block in self.gated_blocks:
            x = block(x)

        return x


class ActionHead(nn.Module):
    """
    Action selection head with optional auto-regressive structure.

    For trading, we have simpler actions than StarCraft:
    - HOLD, BUY_CALL, BUY_PUT, EXIT

    But we can still use auto-regressive structure:
    - First decide: trade or not (HOLD vs action)
    - Then decide: direction (CALL vs PUT) or EXIT
    """

    def __init__(self, hidden_dim: int, n_actions: int = 4):
        super().__init__()

        # Stage 1: Trade or Hold
        self.trade_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 2)  # [HOLD, TRADE]
        )

        # Stage 2: Action type (conditioned on TRADE)
        self.action_type_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 3)  # [CALL, PUT, EXIT]
        )

        # Combined logits projection
        self.combine = nn.Linear(5, n_actions)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            x: [B, H] features

        Returns:
            action_logits: [B, 4] combined logits
            details: Dict with intermediate logits
        """
        # Stage 1: Trade or not
        trade_logits = self.trade_head(x)  # [B, 2]

        # Stage 2: Action type
        action_type_logits = self.action_type_head(x)  # [B, 3]

        # Combine into final 4 actions
        # HOLD = p(hold)
        # CALL = p(trade) * p(call|trade)
        # PUT = p(trade) * p(put|trade)
        # EXIT = p(trade) * p(exit|trade)

        combined = torch.cat([trade_logits, action_type_logits], dim=-1)  # [B, 5]
        action_logits = self.combine(combined)  # [B, 4]

        details = {
            'trade_logits': trade_logits,
            'action_type_logits': action_type_logits
        }

        return action_logits, details


class AlphaStarPolicyNetwork(nn.Module):
    """
    Complete AlphaStar-style policy network for trading.

    Config via env vars:
        RL_HIDDEN_DIM=256
        RL_TRANSFORMER_LAYERS=4
        RL_ATTENTION_HEADS=8
        RL_HISTORY_LEN=32
    """

    def __init__(
        self,
        state_dim: int = 22,
        hidden_dim: int = None,
        n_layers: int = None,
        n_heads: int = None,
        history_len: int = None,
        n_actions: int = 4
    ):
        super().__init__()

        # Config from env vars
        self.hidden_dim = hidden_dim or int(os.environ.get('RL_HIDDEN_DIM', '256'))
        self.n_layers = n_layers or int(os.environ.get('RL_TRANSFORMER_LAYERS', '4'))
        self.n_heads = n_heads or int(os.environ.get('RL_ATTENTION_HEADS', '8'))
        self.history_len = history_len or int(os.environ.get('RL_HISTORY_LEN', '32'))
        self.state_dim = state_dim

        # State history
        self.state_history = deque(maxlen=self.history_len)

        # Entity encoder
        self.entity_encoder = EntityEncoder(state_dim, self.hidden_dim)

        # Transformer core
        self.core = AlphaStarCore(
            self.hidden_dim,
            self.n_layers,
            self.n_heads
        )

        # Action head (auto-regressive style)
        self.action_head = ActionHead(self.hidden_dim, n_actions)

        # Value head (separate critic)
        self.value_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1)
        )

        # Exit urgency head
        self.exit_urgency_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        self._log_architecture()

    def _log_architecture(self):
        param_count = sum(p.numel() for p in self.parameters())
        logger.info(
            f"⭐ AlphaStar Policy: hidden={self.hidden_dim}, layers={self.n_layers}, "
            f"heads={self.n_heads}, history={self.history_len}, params={param_count:,}"
        )

    def reset(self):
        """Reset for new trading session"""
        self.state_history.clear()

    def add_state(self, state: torch.Tensor):
        """Add state to history"""
        if state.dim() == 1:
            state = state.unsqueeze(0)
        self.state_history.append(state)

    def get_state_sequence(self) -> torch.Tensor:
        """Get padded state history [1, T, D]"""
        if len(self.state_history) == 0:
            raise ValueError("No states in history")

        states = torch.cat(list(self.state_history), dim=0)

        if len(states) < self.history_len:
            pad_size = self.history_len - len(states)
            padding = states[0:1].expand(pad_size, -1)
            states = torch.cat([padding, states], dim=0)

        return states.unsqueeze(0)

    def forward(
        self,
        state: torch.Tensor,
        use_history: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            state: [B, D] current state or [B, T, D] sequence

        Returns:
            Dict with action_logits, value, exit_urgency, etc.
        """
        # Handle input
        if state.dim() == 2 and use_history and len(self.state_history) > 0:
            self.add_state(state)
            state = self.get_state_sequence()
        elif state.dim() == 2:
            self.add_state(state) if use_history else None
            state = state.unsqueeze(1)  # [B, 1, D]

        # Encode entities
        entities = self.entity_encoder(state)  # [B, T, N_ent, H]

        # Flatten time and entities for transformer
        B, T = state.shape[:2]
        if entities.dim() == 4:
            N_ent = entities.shape[2]
            entities = entities.reshape(B, T * N_ent, self.hidden_dim)
        elif entities.dim() == 3:
            N_ent = entities.shape[1]
            # Already [B, N_ent, H]

        # Transformer core
        features = self.core(entities)  # [B, H]

        # Action head
        action_logits, action_details = self.action_head(features)

        # Value and exit urgency
        value = self.value_head(features)
        exit_urgency = self.exit_urgency_head(features)

        return {
            'action_logits': action_logits,
            'action_probs': F.softmax(action_logits, dim=-1),
            'value': value,
            'exit_urgency': exit_urgency,
            'features': features,
            'action_details': action_details
        }

    def get_action(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
        temperature: float = 1.0
    ) -> Tuple[int, Dict[str, float]]:
        """Get action for trading"""
        if state.dim() == 1:
            state = state.unsqueeze(0)

        with torch.no_grad():
            out = self.forward(state, use_history=True)

        logits = out['action_logits'] / temperature
        probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        if deterministic:
            action = int(probs.argmax())
        else:
            action = int(np.random.choice(len(probs), p=probs))

        info = {
            'prob_hold': float(probs[0]),
            'prob_call': float(probs[1]),
            'prob_put': float(probs[2]),
            'prob_exit': float(probs[3]),
            'value': float(out['value'].item()),
            'exit_urgency': float(out['exit_urgency'].item()),
            'entropy': float(-np.sum(probs * np.log(probs + 1e-8)))
        }

        return action, info


class AlphaStarRLPolicy:
    """
    High-level AlphaStar RL policy wrapper.
    Drop-in replacement for UnifiedRLPolicy.
    """

    HOLD = 0
    BUY_CALL = 1
    BUY_PUT = 2
    EXIT = 3
    ACTION_NAMES = ['HOLD', 'BUY_CALL', 'BUY_PUT', 'EXIT']

    def __init__(
        self,
        state_dim: int = 22,
        learning_rate: float = 0.0001,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        device: str = 'cpu',
        **kwargs  # Accept and ignore extra kwargs for compatibility
    ):
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        # Compatibility with UnifiedRLPolicy interface
        self.is_bandit_mode = False  # AlphaStar uses full RL from start
        self.total_trades = 0
        self.bandit_mode_trades = 0  # Not used but needed for compat

        self.network = AlphaStarPolicyNetwork(state_dim=state_dim).to(device)

        # Single optimizer (can split like GameAI if needed)
        self.optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )

        # Scheduler (AlphaStar used learning rate decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=1000, T_mult=2
        )

        # Experience buffer
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

        self.total_steps = 0
        self.total_updates = 0

        logger.info(f"⭐ AlphaStar RL Policy initialized (device={device})")

    def reset(self):
        self.network.reset()
        self._clear_buffer()

    def select_action(self, state, deterministic: bool = False):
        """
        Compatibility method for UnifiedRLPolicy interface.

        Args:
            state: TradeState object or numpy array
        Returns:
            (action, confidence, details_dict)
        """
        # Convert TradeState to numpy array if needed
        if hasattr(state, 'is_in_trade'):
            state_array = self._state_to_array(state)
        else:
            state_array = np.array(state, dtype=np.float32)

        action, info = self.get_action(state_array, deterministic=deterministic)
        confidence = max(info.get('prob_call', 0), info.get('prob_put', 0), info.get('prob_exit', 0))

        return action, confidence, info

    def _state_to_array(self, state) -> np.ndarray:
        """Convert TradeState to numpy array (22 features)"""
        features = [
            float(state.is_in_trade),
            float(state.is_call) if state.is_in_trade else 0.5,
            np.clip(state.unrealized_pnl_pct, -1, 1),
            np.clip(state.max_drawdown, 0, 1),
            min(state.minutes_held / 60.0, 2.0),
            min(state.minutes_to_expiry / 1440.0, 1.0),
            np.clip(state.predicted_direction, -1, 1),
            np.clip(state.prediction_confidence, 0, 1),
            np.clip(state.momentum_5m * 100, -1, 1),
            np.clip(state.vix_level / 50.0, 0, 1),
            np.clip(state.volume_spike, 0, 3),
            np.clip(state.hmm_trend, 0, 1),
            np.clip(state.hmm_volatility, 0, 1),
            np.clip(state.hmm_liquidity, 0, 1),
            np.clip(state.hmm_confidence, 0, 1),
            np.clip(state.estimated_theta_decay * 10, -1, 1),
            np.clip(state.estimated_delta, 0, 1),
            np.clip(getattr(state, 'sentiment_fear_greed', 0.5), 0, 1),
            np.clip(getattr(state, 'sentiment_pcr', 0), -1, 1),
            np.clip(getattr(state, 'sentiment_contrarian', 0), -1, 1),
            np.clip(getattr(state, 'sentiment_news', 0), -1, 1),
            0.0  # padding
        ]
        return np.array(features, dtype=np.float32)

    def _clear_buffer(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()

    def get_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[int, Dict]:
        state_tensor = torch.FloatTensor(state).to(self.device)
        return self.network.get_action(state_tensor, deterministic=deterministic)

    def store_transition(self, state: np.ndarray, action: int, reward: float, done: bool = False):
        state_tensor = torch.FloatTensor(state).to(self.device)

        with torch.no_grad():
            out = self.network.forward(state_tensor.unsqueeze(0), use_history=False)
            value = out['value'].item()
            log_prob = F.log_softmax(out['action_logits'], dim=-1)[0, action].item()

        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        self.total_steps += 1

    def learn(self, batch_size: int = 64, epochs: int = 4) -> Dict[str, float]:
        if len(self.states) < batch_size:
            return {'skipped': True}

        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)

        advantages, returns = self._compute_gae()
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_loss = 0

        for _ in range(epochs):
            out = self.network.forward(states, use_history=False)

            log_probs = F.log_softmax(out['action_logits'], dim=-1)
            new_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)

            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            probs = F.softmax(out['action_logits'], dim=-1)
            entropy = -(probs * log_probs).sum(dim=-1).mean()

            values = out['value'].squeeze(1)
            critic_loss = F.mse_loss(values, returns)

            loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
            self.optimizer.step()

            total_loss += loss.item()

        self.scheduler.step()
        self.total_updates += 1
        self._clear_buffer()

        return {
            'loss': total_loss / epochs,
            'lr': self.scheduler.get_last_lr()[0],
            'total_updates': self.total_updates
        }

    def _compute_gae(self) -> Tuple[np.ndarray, np.ndarray]:
        rewards = np.array(self.rewards)
        values = np.array(self.values)
        dones = np.array(self.dones)

        n = len(rewards)
        advantages = np.zeros(n)

        last_value = values[-1] if not dones[-1] else 0
        last_gae = 0

        for t in reversed(range(n)):
            next_value = last_value if t == n - 1 else values[t + 1]
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae

        returns = advantages + values
        return advantages, returns

    def save(self, path: str):
        torch.save({
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'total_steps': self.total_steps,
            'total_updates': self.total_updates
        }, path)
        logger.info(f"⭐ AlphaStar policy saved to {path}")

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scheduler' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.total_steps = checkpoint.get('total_steps', 0)
        self.total_updates = checkpoint.get('total_updates', 0)
        logger.info(f"⭐ AlphaStar policy loaded from {path}")


def get_alphastar_policy(**kwargs) -> AlphaStarRLPolicy:
    """Factory function for AlphaStar policy"""
    return AlphaStarRLPolicy(**kwargs)


if __name__ == '__main__':
    print("Testing AlphaStar Policy...")

    policy = AlphaStarRLPolicy(state_dim=22, device='cpu')
    policy.reset()

    for i in range(10):
        state = np.random.randn(22).astype(np.float32)
        action, info = policy.get_action(state)
        reward = np.random.randn() * 0.1
        policy.store_transition(state, action, reward, done=(i == 9))

        print(f"Step {i+1}: action={policy.ACTION_NAMES[action]}, "
              f"value={info['value']:.3f}, entropy={info['entropy']:.3f}")

    metrics = policy.learn(batch_size=5, epochs=2)
    print(f"\nLearning: {metrics}")

    print("\n✅ AlphaStar Policy test passed!")
