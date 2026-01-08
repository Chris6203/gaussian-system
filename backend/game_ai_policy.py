#!/usr/bin/env python3
"""
Game-AI Style RL Policy for Options Trading

Inspired by:
- OpenAI Five (Dota 2): LSTM + large hidden dims + PPO
- AlphaStar (StarCraft II): Transformer + pointer networks + multi-agent
- MuZero: Learned world model + planning

Key differences from UnifiedPolicyNetwork:
1. LSTM memory (remembers market patterns across time)
2. Much larger hidden dims (256-512 vs 64)
3. Separate Actor and Critic networks (better value estimation)
4. Self-attention over state history
5. Residual connections throughout

Architecture:
    State History [B, T, D]
           │
    ┌──────▼──────┐
    │ Observation │   (Linear projection + LayerNorm)
    │   Encoder   │
    └──────┬──────┘
           │ [B, T, H]
    ┌──────▼──────┐
    │    LSTM     │   (2-layer, bidirectional)
    │   Memory    │
    └──────┬──────┘
           │ [B, H*2]
    ┌──────▼──────┐
    │   Self-     │   (Multi-head attention over time)
    │  Attention  │
    └──────┬──────┘
           │
    ┌──────┴──────┐
    │             │
┌───▼───┐   ┌─────▼─────┐
│ Actor │   │  Critic   │   (Separate networks!)
│  Net  │   │   Net     │
└───┬───┘   └─────┬─────┘
    │             │
 Actions       Value

Usage:
    # Enable via env var
    RL_USE_GAME_AI=1 python scripts/train_time_travel.py

    # Configure
    RL_HIDDEN_DIM=256
    RL_LSTM_LAYERS=2
    RL_HISTORY_LEN=32  # Number of past states to remember
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
import logging

logger = logging.getLogger(__name__)


class ObservationEncoder(nn.Module):
    """
    Encodes raw state observations into a rich representation.
    Similar to OpenAI Five's observation processing.
    """

    def __init__(self, state_dim: int, hidden_dim: int):
        super().__init__()

        # Multi-layer encoding with residual connections
        self.input_proj = nn.Linear(state_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)

        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, D] or [B, D] state observations
        Returns:
            [B, T, H] or [B, H] encoded observations
        """
        # Handle both batched and sequence inputs
        squeeze_out = False
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B, D] -> [B, 1, D]
            squeeze_out = True

        # Project to hidden dim
        h = F.gelu(self.norm1(self.input_proj(x)))

        # Residual block 1
        h = h + F.gelu(self.norm2(self.fc1(h)))

        # Residual block 2
        h = h + F.gelu(self.norm3(self.fc2(h)))

        if squeeze_out:
            h = h.squeeze(1)

        return h


class LSTMMemory(nn.Module):
    """
    LSTM-based memory module like OpenAI Five.
    Remembers patterns across market states.
    """

    def __init__(self, hidden_dim: int, num_layers: int = 2, bidirectional: bool = True):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=0.1 if num_layers > 1 else 0.0
        )

        # Project bidirectional output back to hidden_dim
        if bidirectional:
            self.output_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        else:
            self.output_proj = nn.Identity()

        self.norm = nn.LayerNorm(hidden_dim)

        # Persistent hidden state (for online trading)
        self._hidden = None
        self._cell = None

    def reset_hidden(self, batch_size: int = 1, device: str = 'cpu'):
        """Reset LSTM hidden state (call at start of trading session)"""
        self._hidden = torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_dim,
            device=device
        )
        self._cell = torch.zeros_like(self._hidden)

    def forward(self, x: torch.Tensor, use_persistent: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, T, H] encoded observations
            use_persistent: Use persistent hidden state for online trading

        Returns:
            output: [B, T, H] LSTM output (all timesteps)
            final: [B, H] final hidden state (for action selection)
        """
        if use_persistent and self._hidden is not None:
            lstm_out, (self._hidden, self._cell) = self.lstm(x, (self._hidden, self._cell))
        else:
            lstm_out, _ = self.lstm(x)

        # Project bidirectional back to hidden_dim
        lstm_out = self.output_proj(lstm_out)
        lstm_out = self.norm(lstm_out)

        # Get final timestep
        final = lstm_out[:, -1, :]  # [B, H]

        return lstm_out, final


class TemporalAttention(nn.Module):
    """
    Self-attention over temporal history.
    Like AlphaStar's transformer layer - attends to important past states.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_dim)

        # Feedforward
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [B, T, H] LSTM outputs
            mask: Optional attention mask

        Returns:
            [B, H] attended representation (uses last position as query)
        """
        # Self-attention with residual
        attn_out, _ = self.attention(x, x, x, key_padding_mask=mask)
        x = self.norm1(x + attn_out)

        # Feedforward with residual
        x = self.norm2(x + self.ff(x))

        # Return last position (aggregated context)
        return x[:, -1, :]


class ActorNetwork(nn.Module):
    """
    Separate Actor network (policy).
    Outputs action probabilities.
    """

    def __init__(self, hidden_dim: int, n_actions: int = 4):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, n_actions)
        )

        # Initialize final layer with small weights (for stable initial policy)
        nn.init.orthogonal_(self.net[-1].weight, gain=0.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns action logits [B, n_actions]"""
        return self.net(x)


class CriticNetwork(nn.Module):
    """
    Separate Critic network (value function).
    Outputs state value estimate.

    Key insight from game AI: Separate network prevents actor updates
    from destabilizing value estimates.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()

        # Larger critic for better value estimation
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns value estimate [B, 1]"""
        return self.net(x)


class GameAIPolicyNetwork(nn.Module):
    """
    Complete Game-AI style policy network.

    Combines:
    - Observation encoder (ResNet-style)
    - LSTM memory (OpenAI Five)
    - Temporal attention (AlphaStar)
    - Separate Actor/Critic (standard PPO best practice)

    Config via env vars:
        RL_HIDDEN_DIM=256
        RL_LSTM_LAYERS=2
        RL_HISTORY_LEN=32
        RL_ATTENTION_HEADS=4
    """

    def __init__(
        self,
        state_dim: int = 22,
        hidden_dim: int = None,
        lstm_layers: int = None,
        history_len: int = None,
        n_actions: int = 4
    ):
        super().__init__()

        # Config from env vars with defaults
        self.hidden_dim = hidden_dim or int(os.environ.get('RL_HIDDEN_DIM', '256'))
        self.lstm_layers = lstm_layers or int(os.environ.get('RL_LSTM_LAYERS', '2'))
        self.history_len = history_len or int(os.environ.get('RL_HISTORY_LEN', '32'))
        self.n_actions = n_actions

        # State history buffer
        self.state_history = deque(maxlen=self.history_len)

        # Core modules
        self.obs_encoder = ObservationEncoder(state_dim, self.hidden_dim)
        self.lstm_memory = LSTMMemory(self.hidden_dim, self.lstm_layers, bidirectional=True)
        self.temporal_attn = TemporalAttention(
            self.hidden_dim,
            num_heads=int(os.environ.get('RL_ATTENTION_HEADS', '4'))
        )

        # Separate actor and critic
        self.actor = ActorNetwork(self.hidden_dim, n_actions)
        self.critic = CriticNetwork(self.hidden_dim)

        # Exit urgency head (specialized for trading)
        self.exit_urgency = nn.Sequential(
            nn.Linear(self.hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        self._log_architecture()

    def _log_architecture(self):
        """Log architecture summary"""
        param_count = sum(p.numel() for p in self.parameters())
        logger.info(
            f"🎮 GameAI Policy: hidden={self.hidden_dim}, lstm_layers={self.lstm_layers}, "
            f"history={self.history_len}, params={param_count:,}"
        )

    def reset(self, batch_size: int = 1, device: str = 'cpu'):
        """Reset for new trading session"""
        self.state_history.clear()
        self.lstm_memory.reset_hidden(batch_size, device)

    def add_state(self, state: torch.Tensor):
        """Add state to history buffer"""
        if state.dim() == 1:
            state = state.unsqueeze(0)  # [D] -> [1, D]
        self.state_history.append(state)

    def get_state_sequence(self) -> torch.Tensor:
        """Get padded state history as tensor [1, T, D]"""
        if len(self.state_history) == 0:
            raise ValueError("No states in history. Call add_state first.")

        # Stack history
        states = torch.cat(list(self.state_history), dim=0)  # [T, D]

        # Pad if needed
        if len(states) < self.history_len:
            pad_size = self.history_len - len(states)
            padding = states[0:1].expand(pad_size, -1)  # Repeat first state
            states = torch.cat([padding, states], dim=0)

        return states.unsqueeze(0)  # [1, T, D]

    def forward(
        self,
        state: torch.Tensor,
        return_value: bool = True,
        use_history: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            state: Current state [B, D] or full sequence [B, T, D]
            return_value: Also compute critic value
            use_history: Use accumulated state history (for online trading)

        Returns:
            Dict with:
                action_logits: [B, 4]
                action_probs: [B, 4]
                value: [B, 1] (if return_value)
                exit_urgency: [B, 1]
        """
        # Handle input format
        if state.dim() == 2 and use_history:
            # Single state - add to history and use full sequence
            self.add_state(state)
            state_seq = self.get_state_sequence()  # [1, T, D]
        elif state.dim() == 2:
            # Single state, no history - expand to sequence
            state_seq = state.unsqueeze(1)  # [B, 1, D]
        else:
            # Already a sequence
            state_seq = state  # [B, T, D]

        # Encode observations
        encoded = self.obs_encoder(state_seq)  # [B, T, H]

        # LSTM memory processing
        lstm_out, lstm_final = self.lstm_memory(encoded)  # [B, T, H], [B, H]

        # Temporal attention (aggregate across time)
        attended = self.temporal_attn(lstm_out)  # [B, H]

        # Actor output
        action_logits = self.actor(attended)  # [B, 4]
        action_probs = F.softmax(action_logits, dim=-1)

        # Exit urgency (trading-specific)
        exit_urg = self.exit_urgency(attended)  # [B, 1]

        result = {
            'action_logits': action_logits,
            'action_probs': action_probs,
            'exit_urgency': exit_urg,
            'features': attended  # For downstream use
        }

        # Critic output (separate computation)
        if return_value:
            result['value'] = self.critic(attended)  # [B, 1]

        return result

    def get_action(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
        temperature: float = 1.0
    ) -> Tuple[int, Dict[str, float]]:
        """
        Get action for trading.

        Args:
            state: Current state [D] or [1, D]
            deterministic: If True, take argmax action
            temperature: Softmax temperature (lower = more deterministic)

        Returns:
            action: int (0=HOLD, 1=CALL, 2=PUT, 3=EXIT)
            info: Dict with probabilities and value
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)

        with torch.no_grad():
            out = self.forward(state, return_value=True, use_history=True)

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


class GameAIRLPolicy:
    """
    High-level RL policy wrapper using GameAI architecture.

    Drop-in replacement for UnifiedRLPolicy with game-AI architecture.

    Usage:
        policy = GameAIRLPolicy(device='cuda')
        policy.reset()  # Start of trading session

        while trading:
            state = get_market_state()
            action, info = policy.get_action(state)
            execute(action)
            policy.learn(reward)
    """

    # Actions (same as UnifiedRLPolicy)
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
        self.is_bandit_mode = False  # Game-AI uses full RL from start
        self.total_trades = 0
        self.bandit_mode_trades = 0  # Not used but needed for compat

        # Create network
        self.network = GameAIPolicyNetwork(state_dim=state_dim).to(device)

        # Separate optimizers for actor and critic (game AI best practice)
        self.actor_optimizer = torch.optim.Adam(
            list(self.network.obs_encoder.parameters()) +
            list(self.network.lstm_memory.parameters()) +
            list(self.network.temporal_attn.parameters()) +
            list(self.network.actor.parameters()) +
            list(self.network.exit_urgency.parameters()),
            lr=learning_rate
        )
        self.critic_optimizer = torch.optim.Adam(
            self.network.critic.parameters(),
            lr=learning_rate * 2  # Critic learns faster
        )

        # Experience buffer for PPO updates
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

        # Stats
        self.total_steps = 0
        self.total_updates = 0

        logger.info(f"🎮 GameAI RL Policy initialized (device={device})")

    def reset(self):
        """Reset for new trading session"""
        self.network.reset(batch_size=1, device=self.device)
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
        """Clear experience buffer"""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()

    def get_action(
        self,
        state: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[int, Dict[str, float]]:
        """Get action for current state"""
        state_tensor = torch.FloatTensor(state).to(self.device)
        action, info = self.network.get_action(state_tensor, deterministic=deterministic)
        return action, info

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        done: bool = False
    ):
        """Store transition for learning"""
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
        """
        PPO update step.

        Returns:
            Dict with loss metrics
        """
        if len(self.states) < batch_size:
            return {'skipped': True, 'buffer_size': len(self.states)}

        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)

        # Compute advantages (GAE)
        advantages, returns = self._compute_gae()
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO epochs
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0

        for _ in range(epochs):
            # Forward pass (no history for batch training)
            out = self.network.forward(states, return_value=True, use_history=False)

            # Actor loss (PPO clipped objective)
            log_probs = F.log_softmax(out['action_logits'], dim=-1)
            new_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)

            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # Entropy bonus
            probs = F.softmax(out['action_logits'], dim=-1)
            entropy = -(probs * log_probs).sum(dim=-1).mean()

            # Critic loss
            values = out['value'].squeeze(1)
            critic_loss = F.mse_loss(values, returns)

            # Update actor
            self.actor_optimizer.zero_grad()
            actor_total_loss = actor_loss - self.entropy_coef * entropy
            actor_total_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()

            # Update critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()

            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            total_entropy += entropy.item()

        self.total_updates += 1
        self._clear_buffer()

        return {
            'actor_loss': total_actor_loss / epochs,
            'critic_loss': total_critic_loss / epochs,
            'entropy': total_entropy / epochs,
            'total_updates': self.total_updates
        }

    def _compute_gae(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Generalized Advantage Estimation"""
        rewards = np.array(self.rewards)
        values = np.array(self.values)
        dones = np.array(self.dones)

        n = len(rewards)
        advantages = np.zeros(n)
        returns = np.zeros(n)

        # Bootstrap from last value
        last_value = values[-1] if not dones[-1] else 0
        last_gae = 0

        for t in reversed(range(n)):
            if t == n - 1:
                next_value = last_value
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae

        returns = advantages + values
        return advantages, returns

    def save(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'network_state': self.network.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'total_steps': self.total_steps,
            'total_updates': self.total_updates
        }, path)
        logger.info(f"🎮 GameAI policy saved to {path}")

    def load(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.total_steps = checkpoint.get('total_steps', 0)
        self.total_updates = checkpoint.get('total_updates', 0)
        logger.info(f"🎮 GameAI policy loaded from {path}")


# Factory function for easy switching
def get_rl_policy(use_game_ai: bool = None, **kwargs):
    """
    Get RL policy based on config.

    Args:
        use_game_ai: Use Game-AI architecture. If None, reads from RL_USE_GAME_AI env var.
        **kwargs: Passed to policy constructor

    Returns:
        Either GameAIRLPolicy or standard UnifiedRLPolicy
    """
    if use_game_ai is None:
        use_game_ai = os.environ.get('RL_USE_GAME_AI', '0') == '1'

    if use_game_ai:
        logger.info("🎮 Using Game-AI style RL policy")
        return GameAIRLPolicy(**kwargs)
    else:
        # Import here to avoid circular imports
        from backend.unified_rl_policy import UnifiedRLPolicy
        return UnifiedRLPolicy(**kwargs)


if __name__ == '__main__':
    # Quick test
    print("Testing GameAI Policy Network...")

    policy = GameAIRLPolicy(state_dim=22, device='cpu')
    policy.reset()

    # Simulate a few steps
    for i in range(10):
        state = np.random.randn(22).astype(np.float32)
        action, info = policy.get_action(state)
        reward = np.random.randn() * 0.1
        policy.store_transition(state, action, reward, done=(i == 9))

        print(f"Step {i+1}: action={policy.ACTION_NAMES[action]}, "
              f"value={info['value']:.3f}, entropy={info['entropy']:.3f}")

    # Learn
    metrics = policy.learn(batch_size=5, epochs=2)
    print(f"\nLearning metrics: {metrics}")

    print("\n✅ GameAI Policy test passed!")
