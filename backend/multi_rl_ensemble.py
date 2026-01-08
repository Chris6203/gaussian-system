#!/usr/bin/env python3
"""
Multi-RL Ensemble Architecture

Multiple RL agents with different characteristics, combined via meta-selector.

Architecture:
    [Encoder Features]
           │
    ┌──────┼──────┬──────┬──────┐
    │      │      │      │      │
   RL1    RL2    RL3    RL4
 (Aggro) (Cons) (Trend) (Mean-Rev)
    │      │      │      │      │
 Action  Action  Action  Action
 Conf    Conf    Conf    Conf
    │      │      │      │      │
    └──────┴──────┴──────┴──────┘
           │
    [Meta-Selector]
    Learns which RL to trust
           │
    Final Action + Combined Confidence

Each RL agent has:
- Different learning rate (fast vs slow adapters)
- Different confidence thresholds
- Different action biases (aggressive vs conservative)

Meta-selector tracks:
- Recent win rate per agent
- Confidence calibration per agent
- Market regime suitability
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class RLAgentConfig:
    """Configuration for a single RL agent in the ensemble"""
    name: str
    learning_rate: float = 0.0003
    min_confidence: float = 0.55
    action_temperature: float = 1.0  # Lower = more deterministic
    bias_calls: float = 0.0  # Positive = more bullish
    bias_puts: float = 0.0  # Positive = more bearish
    description: str = ""


class SubPolicyNetwork(nn.Module):
    """
    Small policy network for a single RL agent.
    Each agent in the ensemble has one of these.
    """

    def __init__(self, state_dim: int, hidden_dim: int = 48, n_actions: int = 4):
        super().__init__()
        self.state_dim = state_dim

        # Smaller network than main policy (efficiency)
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Action head
        self.action_head = nn.Linear(hidden_dim, n_actions)

        # Confidence head (how confident is this agent in its action)
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.GELU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

        # Value head for PPO
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, temperature: float = 1.0) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Returns:
            action_probs: [B, 4] action probabilities
            confidence: [B, 1] agent confidence in its recommendation
            value: [B, 1] state value estimate
        """
        h = self.shared(x)

        # Action logits with temperature
        action_logits = self.action_head(h) / temperature
        action_probs = F.softmax(action_logits, dim=-1)

        # Agent's confidence in its recommendation
        confidence = self.confidence_head(h)

        # Value estimate
        value = self.value_head(h)

        return {
            'action_probs': action_probs,
            'action_logits': action_logits,
            'confidence': confidence,
            'value': value,
            'features': h  # For meta-selector
        }


class MetaSelector(nn.Module):
    """
    Meta-selector that learns which RL agent to trust.

    Inputs:
    - Agent features (hidden states from each agent)
    - Agent confidences
    - Recent performance metrics
    - Market state features

    Outputs:
    - Agent weights (which agent to trust)
    """

    def __init__(self, n_agents: int, agent_hidden_dim: int, state_dim: int):
        super().__init__()
        self.n_agents = n_agents

        # Input: concatenated agent features + confidences + market state
        input_dim = n_agents * agent_hidden_dim + n_agents + state_dim

        self.selector = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, n_agents),
        )

    def forward(
        self,
        agent_features: List[torch.Tensor],
        agent_confidences: List[torch.Tensor],
        market_state: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute agent weights.

        Returns:
            weights: [B, n_agents] weights for each agent (softmax normalized)
        """
        # Concatenate all inputs
        features_cat = torch.cat(agent_features, dim=-1)  # [B, n_agents * hidden]
        confs_cat = torch.cat(agent_confidences, dim=-1)  # [B, n_agents]

        x = torch.cat([features_cat, confs_cat, market_state], dim=-1)

        # Get raw weights
        weights = self.selector(x)

        # Softmax to normalize
        weights = F.softmax(weights, dim=-1)

        return weights


class MultiRLEnsemble:
    """
    Ensemble of multiple RL agents with meta-selection.

    Each agent has different characteristics:
    - Aggressive: Low confidence threshold, higher learning rate
    - Conservative: High confidence threshold, slower learning
    - Trend-follower: Biased toward following HMM trend
    - Mean-reversion: Biased toward fading extremes
    """

    # Actions (same as UnifiedRLPolicy)
    HOLD = 0
    BUY_CALL = 1
    BUY_PUT = 2
    EXIT = 3

    def __init__(
        self,
        state_dim: int = 18,
        n_agents: int = 4,
        device: str = 'cpu',
        agent_configs: List[RLAgentConfig] = None,
    ):
        self.device = device
        self.n_agents = n_agents
        self.state_dim = state_dim

        # Default agent configurations
        if agent_configs is None:
            agent_configs = [
                RLAgentConfig(
                    name="aggressive",
                    learning_rate=0.001,
                    min_confidence=0.40,
                    action_temperature=0.8,
                    description="Fast learner, takes more trades"
                ),
                RLAgentConfig(
                    name="conservative",
                    learning_rate=0.0001,
                    min_confidence=0.70,
                    action_temperature=1.2,
                    description="Slow learner, only high-confidence trades"
                ),
                RLAgentConfig(
                    name="trend_follower",
                    learning_rate=0.0003,
                    min_confidence=0.55,
                    action_temperature=1.0,
                    bias_calls=0.1,  # Slight bullish bias when HMM is bullish
                    description="Follows HMM trend direction"
                ),
                RLAgentConfig(
                    name="mean_reverter",
                    learning_rate=0.0003,
                    min_confidence=0.55,
                    action_temperature=1.0,
                    bias_calls=-0.1,  # Contrarian when HMM is extreme
                    description="Fades extreme moves"
                ),
            ]

        self.agent_configs = agent_configs[:n_agents]

        # Create agent networks
        agent_hidden_dim = 48
        self.agents = nn.ModuleList([
            SubPolicyNetwork(state_dim, hidden_dim=agent_hidden_dim, n_actions=4).to(device)
            for _ in range(n_agents)
        ])

        # Create optimizers (one per agent with its own LR)
        self.optimizers = [
            torch.optim.AdamW(
                agent.parameters(),
                lr=cfg.learning_rate,
                weight_decay=1e-5
            )
            for agent, cfg in zip(self.agents, self.agent_configs)
        ]

        # Meta-selector
        self.meta_selector = MetaSelector(
            n_agents=n_agents,
            agent_hidden_dim=agent_hidden_dim,
            state_dim=state_dim
        ).to(device)

        self.meta_optimizer = torch.optim.AdamW(
            self.meta_selector.parameters(),
            lr=0.0003,
            weight_decay=1e-5
        )

        # Performance tracking per agent
        self.agent_stats = {
            cfg.name: {
                'wins': 0,
                'losses': 0,
                'total_pnl': 0.0,
                'recent_results': deque(maxlen=50),  # Last 50 decisions
                'selection_count': 0,
            }
            for cfg in self.agent_configs
        }

        # Track which agent was selected for each trade
        self.current_agent_idx: Optional[int] = None
        self.current_agent_name: Optional[str] = None

        # Overall stats
        self.total_trades = 0

        # Experience buffer (shared across agents for meta-learning)
        self.experience_buffer: deque = deque(maxlen=10000)

        logger.info(f"🎭 Multi-RL Ensemble initialized with {n_agents} agents:")
        for i, cfg in enumerate(self.agent_configs):
            logger.info(f"   Agent {i}: {cfg.name} - {cfg.description}")
            logger.info(f"      LR={cfg.learning_rate}, min_conf={cfg.min_confidence}, temp={cfg.action_temperature}")

    def get_agent_win_rates(self) -> Dict[str, float]:
        """Get recent win rate for each agent"""
        rates = {}
        for name, stats in self.agent_stats.items():
            recent = list(stats['recent_results'])
            if len(recent) >= 5:
                rates[name] = sum(1 for r in recent if r > 0) / len(recent)
            else:
                rates[name] = 0.5  # Default when not enough data
        return rates

    def state_to_tensor(self, state_dict: Dict) -> torch.Tensor:
        """Convert state dict to tensor"""
        # Extract features from state dict
        features = [
            float(state_dict.get('is_in_trade', False)),
            float(state_dict.get('is_call', True)) if state_dict.get('is_in_trade', False) else 0.5,
            np.clip(state_dict.get('unrealized_pnl_pct', 0.0), -0.5, 0.5),
            np.clip(state_dict.get('max_drawdown', 0.0), 0, 0.3),
            min(1.0, state_dict.get('minutes_held', 0) / 120.0),
            min(1.0, state_dict.get('minutes_to_expiry', 1440) / 1440.0),
            np.clip(state_dict.get('predicted_direction', 0.0), -1, 1),
            np.clip(state_dict.get('prediction_confidence', 0.5), 0, 1),
            np.clip(state_dict.get('momentum_5m', 0.0) * 100, -1, 1),
            min(1.0, state_dict.get('vix_level', 18.0) / 50.0),
            min(2.0, state_dict.get('volume_spike', 1.0)) / 2.0,
            np.clip(state_dict.get('hmm_trend', 0.5), 0, 1),
            np.clip(state_dict.get('hmm_volatility', 0.5), 0, 1),
            np.clip(state_dict.get('hmm_liquidity', 0.5), 0, 1),
            np.clip(state_dict.get('hmm_confidence', 0.5), 0, 1),
            np.clip(state_dict.get('estimated_theta_decay', 0.0) * 100, -0.1, 0),
            np.clip(state_dict.get('estimated_delta', 0.5), 0, 1),
            np.clip(float(state_dict.get('position_size', 0)) / 10.0, 0.0, 1.0),
        ]

        # Pad or truncate to state_dim
        while len(features) < self.state_dim:
            features.append(0.0)
        features = features[:self.state_dim]

        return torch.tensor([features], dtype=torch.float32, device=self.device)

    def select_action(
        self,
        state_dict: Dict,
        explore: bool = True,
    ) -> Tuple[int, float, Dict]:
        """
        Select action using ensemble of agents.

        Returns:
            action: Selected action (HOLD, BUY_CALL, BUY_PUT, EXIT)
            confidence: Combined confidence in the action
            info: Debug info including which agent was selected
        """
        state_tensor = self.state_to_tensor(state_dict)

        # Get outputs from all agents
        agent_outputs = []
        with torch.no_grad():
            for i, (agent, cfg) in enumerate(zip(self.agents, self.agent_configs)):
                out = agent(state_tensor, temperature=cfg.action_temperature)

                # Apply biases based on HMM trend
                hmm_trend = state_dict.get('hmm_trend', 0.5)
                if cfg.bias_calls != 0:
                    # Bias toward calls when HMM is bullish
                    trend_factor = (hmm_trend - 0.5) * 2  # -1 to 1
                    out['action_probs'][:, self.BUY_CALL] += cfg.bias_calls * trend_factor
                    out['action_probs'][:, self.BUY_PUT] -= cfg.bias_calls * trend_factor
                    out['action_probs'] = F.softmax(out['action_probs'], dim=-1)

                agent_outputs.append(out)

        # Get agent features and confidences for meta-selector
        agent_features = [out['features'] for out in agent_outputs]
        agent_confidences = [out['confidence'] for out in agent_outputs]

        # Meta-selector chooses agent weights
        with torch.no_grad():
            agent_weights = self.meta_selector(agent_features, agent_confidences, state_tensor)

        # Incorporate recent performance into weights
        win_rates = self.get_agent_win_rates()
        performance_bonus = torch.tensor([
            [win_rates.get(cfg.name, 0.5) for cfg in self.agent_configs]
        ], device=self.device)

        # Combine learned weights with performance
        combined_weights = agent_weights * 0.7 + performance_bonus * 0.3
        combined_weights = combined_weights / combined_weights.sum(dim=-1, keepdim=True)

        # Select best agent (or weighted combination)
        best_agent_idx = combined_weights.argmax(dim=-1).item()
        best_agent_cfg = self.agent_configs[best_agent_idx]
        best_agent_output = agent_outputs[best_agent_idx]

        # Check confidence threshold for this agent
        agent_confidence = best_agent_output['confidence'].item()
        prediction_confidence = state_dict.get('prediction_confidence', 0.5)

        # Get action from best agent
        action_probs = best_agent_output['action_probs'].squeeze(0)

        if explore and np.random.random() < 0.1:  # 10% exploration
            action = np.random.choice(4, p=action_probs.cpu().numpy())
        else:
            action = action_probs.argmax().item()

        # Apply confidence gating
        if action in [self.BUY_CALL, self.BUY_PUT]:
            if prediction_confidence < best_agent_cfg.min_confidence:
                action = self.HOLD  # Override to HOLD if confidence too low

        # Combined confidence from agent and meta-selector
        combined_confidence = agent_confidence * combined_weights[0, best_agent_idx].item()

        # Store which agent was selected
        self.current_agent_idx = best_agent_idx
        self.current_agent_name = best_agent_cfg.name

        # Debug info
        info = {
            'selected_agent': best_agent_cfg.name,
            'selected_agent_idx': best_agent_idx,
            'agent_confidence': agent_confidence,
            'agent_weights': combined_weights.squeeze(0).cpu().numpy().tolist(),
            'action_probs': action_probs.cpu().numpy().tolist(),
            'agent_win_rates': win_rates,
        }

        return action, combined_confidence, info

    def record_outcome(self, pnl: float, info: Dict = None):
        """
        Record trade outcome for the selected agent.

        Called when a trade closes to update agent performance tracking.
        """
        if self.current_agent_name is None:
            return

        agent_name = self.current_agent_name
        stats = self.agent_stats[agent_name]

        # Update stats
        if pnl > 0:
            stats['wins'] += 1
        else:
            stats['losses'] += 1

        stats['total_pnl'] += pnl
        stats['recent_results'].append(pnl)
        stats['selection_count'] += 1

        self.total_trades += 1

        # Store experience for meta-learning
        if info is not None:
            self.experience_buffer.append({
                'agent_idx': self.current_agent_idx,
                'agent_name': agent_name,
                'pnl': pnl,
                'info': info,
            })

        # Reset current agent tracking
        self.current_agent_idx = None
        self.current_agent_name = None

    def update(self, batch_size: int = 32):
        """
        Update all agents and meta-selector from experience buffer.
        """
        if len(self.experience_buffer) < batch_size:
            return

        # Sample batch
        indices = np.random.choice(len(self.experience_buffer), batch_size, replace=False)
        batch = [self.experience_buffer[i] for i in indices]

        # Update each agent based on its outcomes
        agent_losses = {cfg.name: [] for cfg in self.agent_configs}

        for exp in batch:
            agent_idx = exp['agent_idx']
            pnl = exp['pnl']

            # Simple reward: positive for profit, negative for loss
            reward = 1.0 if pnl > 0 else -1.0

            # This is a simplified update - full implementation would use PPO
            # For now, just track that updates would happen
            agent_losses[self.agent_configs[agent_idx].name].append(reward)

        logger.debug(f"[MULTI-RL] Updated from {batch_size} experiences")

    def get_stats(self) -> Dict:
        """Get ensemble statistics"""
        stats = {
            'total_trades': self.total_trades,
            'agents': {}
        }

        for cfg in self.agent_configs:
            name = cfg.name
            agent_stats = self.agent_stats[name]
            total = agent_stats['wins'] + agent_stats['losses']
            stats['agents'][name] = {
                'wins': agent_stats['wins'],
                'losses': agent_stats['losses'],
                'win_rate': agent_stats['wins'] / total if total > 0 else 0.0,
                'total_pnl': agent_stats['total_pnl'],
                'selection_count': agent_stats['selection_count'],
            }

        return stats

    def save(self, path: str):
        """Save ensemble state"""
        state = {
            'agents': [agent.state_dict() for agent in self.agents],
            'meta_selector': self.meta_selector.state_dict(),
            'agent_stats': {k: dict(v) for k, v in self.agent_stats.items()},
            'total_trades': self.total_trades,
        }
        torch.save(state, path)
        logger.info(f"[MULTI-RL] Saved ensemble to {path}")

    def load(self, path: str):
        """Load ensemble state"""
        state = torch.load(path, map_location=self.device)

        for agent, agent_state in zip(self.agents, state['agents']):
            agent.load_state_dict(agent_state)

        self.meta_selector.load_state_dict(state['meta_selector'])

        # Restore stats
        for name, stats in state['agent_stats'].items():
            if name in self.agent_stats:
                self.agent_stats[name].update(stats)

        self.total_trades = state.get('total_trades', 0)
        logger.info(f"[MULTI-RL] Loaded ensemble from {path}")


def get_multi_rl_ensemble(state_dim: int = 18, device: str = 'cpu') -> MultiRLEnsemble:
    """Factory function to create Multi-RL ensemble"""
    n_agents = int(os.environ.get('MULTI_RL_N_AGENTS', '4'))
    return MultiRLEnsemble(state_dim=state_dim, n_agents=n_agents, device=device)


# Test
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # Create ensemble
    ensemble = get_multi_rl_ensemble(state_dim=18)

    # Test state
    test_state = {
        'is_in_trade': False,
        'predicted_direction': 0.3,
        'prediction_confidence': 0.65,
        'hmm_trend': 0.7,
        'hmm_confidence': 0.8,
        'vix_level': 18.0,
    }

    # Select action
    action, confidence, info = ensemble.select_action(test_state)
    print(f"Action: {action}, Confidence: {confidence:.2f}")
    print(f"Selected agent: {info['selected_agent']}")
    print(f"Agent weights: {info['agent_weights']}")

    # Record outcome
    ensemble.record_outcome(pnl=50.0, info=info)

    # Get stats
    stats = ensemble.get_stats()
    print(f"Stats: {stats}")
