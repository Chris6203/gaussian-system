"""
Reinforcement Learning Trading Policy
Implements Proximal Policy Optimization (PPO) for optimal trading decisions.

This upgraded policy replaces the previous Actor-Critic (A2C) implementation with PPO,
offering significantly better stability and sample efficiency through clipped updates.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
import logging
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class SharedFeatureExtractor(nn.Module):
    """
    Shared feature extractor for Actor-Critic architecture.
    Both actor and critic share this backbone for efficient feature learning.
    """
    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.output_dim = 128
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class ActorNetwork(nn.Module):
    """
    Policy network (Actor) - outputs action probabilities
    Maps state -> action distribution
    Uses shared feature extractor for efficient learning
    """
    def __init__(self, state_dim: int, action_dim: int, shared_extractor: SharedFeatureExtractor = None):
        super().__init__()
        self.shared = shared_extractor
        
        # If no shared extractor, create standalone network
        if self.shared is None:
            self.standalone = nn.Sequential(
                nn.Linear(state_dim, 128),
                nn.LayerNorm(128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.LayerNorm(64),
                nn.ReLU(),
            )
            input_dim = 64
        else:
            self.standalone = None
            input_dim = self.shared.output_dim
        
        # Actor-specific head with larger capacity
        self.head = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim)
        )
    
    def forward(self, state: torch.Tensor, shared_features: torch.Tensor = None) -> torch.Tensor:
        """Return action logits (use softmax for probabilities)"""
        if shared_features is not None:
            features = shared_features
        elif self.shared is not None:
            features = self.shared(state)
        else:
            features = self.standalone(state)
        return self.head(features)


class CriticNetwork(nn.Module):
    """
    Value network (Critic) - estimates state value
    Maps state -> expected return
    Uses shared feature extractor for efficient learning
    """
    def __init__(self, state_dim: int, shared_extractor: SharedFeatureExtractor = None):
        super().__init__()
        self.shared = shared_extractor
        
        # If no shared extractor, create standalone network
        if self.shared is None:
            self.standalone = nn.Sequential(
                nn.Linear(state_dim, 128),
                nn.LayerNorm(128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.LayerNorm(64),
                nn.ReLU(),
            )
            input_dim = 64
        else:
            self.standalone = None
            input_dim = self.shared.output_dim
        
        # Critic-specific head
        self.head = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # Single value output
        )
    
    def forward(self, state: torch.Tensor, shared_features: torch.Tensor = None) -> torch.Tensor:
        """Return state value estimate"""
        if shared_features is not None:
            features = shared_features
        elif self.shared is not None:
            features = self.shared(state)
        else:
            features = self.standalone(state)
        return self.head(features)


class TradingAction:
    """Enumeration of possible trading actions"""
    HOLD = 0
    BUY_CALL_1X = 1
    BUY_CALL_2X = 2
    BUY_PUT_1X = 3
    BUY_PUT_2X = 4
    
    @staticmethod
    def to_string(action_idx: int) -> str:
        actions = ['HOLD', 'BUY_CALL', 'BUY_CALL', 'BUY_PUT', 'BUY_PUT']
        return actions[action_idx]
    
    @staticmethod
    def get_position_size(action_idx: int) -> int:
        sizes = [0, 1, 2, 1, 2]
        return sizes[action_idx]
    
    @staticmethod
    def get_direction(action_idx: int) -> str:
        directions = ['HOLD', 'CALL', 'CALL', 'PUT', 'PUT']
        return directions[action_idx]


class RLTradingPolicy:
    """
    Reinforcement Learning Trading Policy using PPO (Proximal Policy Optimization)
    
    PPO Improvements over A2C:
    - Clipped objective function prevents destructive policy updates
    - Multiple epochs of training on the same batch (sample efficiency)
    - More stable hyperparameter tuning
    """
    
    def __init__(
        self,
        state_dim: int = 32,  # Updated: Increased to 32 for context features (was 28)
        action_dim: int = 5,
        learning_rate: float = 0.0003,  # FIXED: Standard PPO learning rate (was 0.05 = way too high!)
        gamma: float = 0.99,
        entropy_coef: float = 0.05,  # FIXED: Higher entropy for more exploration (was 0.01)
        entropy_min: float = 0.01,   # FIXED: Higher floor to prevent collapse (was 0.001)
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,  # Tighter clipping for PPO
        ppo_clip_range: float = 0.2,
        ppo_epochs: int = 4,
        device: str = 'cpu',
        use_shared_features: bool = True,
        # NEW: Warmup period to gather data before making real decisions
        warmup_episodes: int = 50,    # Gather 50 episodes before RL takes over
        exploration_temp: float = 2.0  # Higher temp = more random during warmup
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.entropy_min = entropy_min
        self.entropy_decay = 0.9995
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_clip_range = ppo_clip_range
        self.ppo_epochs = ppo_epochs
        self.device = device
        self.use_shared_features = use_shared_features
        
        # NEW: Warmup and exploration parameters
        self.warmup_episodes = warmup_episodes
        self.exploration_temp = exploration_temp
        self.episodes_completed = 0
        self.is_warmed_up = False
        
        # Track signal quality to avoid trading garbage
        self.signal_quality_threshold = 0.4  # Minimum signal quality to consider
        self.consecutive_losses = 0
        self.max_consecutive_losses = 3  # Stop trading after 3 losses in a row
        
        # Initialize networks
        if use_shared_features:
            self.shared_extractor = SharedFeatureExtractor(state_dim, hidden_dim=256).to(device)
            self.actor = ActorNetwork(state_dim, action_dim, self.shared_extractor).to(device)
            self.critic = CriticNetwork(state_dim, self.shared_extractor).to(device)
            
            all_params = list(self.shared_extractor.parameters()) + \
                        list(self.actor.head.parameters()) + \
                        list(self.critic.head.parameters())
            self.optimizer = torch.optim.AdamW(all_params, lr=learning_rate, weight_decay=1e-5)
            
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, T_0=1000, T_mult=2, eta_min=1e-6
            )
            self.actor_optimizer = self.optimizer
            self.critic_optimizer = self.optimizer
        else:
            self.shared_extractor = None
            self.actor = ActorNetwork(state_dim, action_dim).to(device)
            self.critic = CriticNetwork(state_dim).to(device)
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=learning_rate)
            self.scheduler = None
            self.optimizer = None
        
        self.experience_buffer = deque(maxlen=10000)
        self.current_episode = []
        
        self.training_stats = {
            'total_updates': 0,
            'actor_losses': [],
            'critic_losses': [],
            'rewards': [],
            'entropy': [],
            'policy_accuracy': []
        }
        
        logger.info(f"🎯 RL Trading Policy initialized (PPO - FIXED)")
        logger.info(f"   State dim: {state_dim}, Action dim: {action_dim}")
        logger.info(f"   Device: {device}, LR: {learning_rate} (FIXED from 0.05)")
        logger.info(f"   Entropy: {entropy_coef} (encourages exploration)")
        logger.info(f"   Warmup: {warmup_episodes} episodes before real trading")
        logger.info(f"   Signal quality threshold: {self.signal_quality_threshold}")
    
    def build_state_vector(self, predicted_return, predicted_volatility, confidence, direction_probs,
                          current_balance, num_open_positions, max_positions, total_pnl,
                          vix_level, momentum_5min, momentum_15min, volume_spike, rsi,
                          win_rate_last_10, avg_return_last_10, recent_drawdown, trades_today,
                          minutes_since_market_open, days_until_expiry, is_last_hour,
                          hmm_trend=0.5, hmm_vol=0.5, hmm_liq=0.5,
                          vix_bb_pos=0.5, vix_roc=0.0, vix_percentile=0.5,
                          price_jerk=0.0,
                          # NEW: Context features
                          context_trend_alignment=0.5, tech_momentum=0.0, crypto_momentum=0.0, sector_rotation=0.0):
        """Build comprehensive state vector for policy network"""
        balance_log = np.log1p(current_balance) / 10.0
        pnl_ratio = total_pnl / max(1.0, current_balance) * 10.0
        
        state = np.array([
            predicted_return * 100,
            predicted_volatility * 100,
            confidence,
            direction_probs[0],
            direction_probs[1],
            direction_probs[2],
            balance_log,
            num_open_positions / max(max_positions, 1),
            pnl_ratio,
            min(1.0, vix_level / 80.0),
            momentum_5min * 100,
            momentum_15min * 100,
            volume_spike / 2.0,
            (rsi - 50) / 50.0,
            win_rate_last_10,
            avg_return_last_10 * 10,
            recent_drawdown * 10,
            trades_today / 10.0,
            minutes_since_market_open / 390.0,
            days_until_expiry / 7.0,
            float(is_last_hour),
            hmm_trend,
            hmm_vol,
            hmm_liq,
            vix_bb_pos,
            min(1.0, max(-1.0, vix_roc)),
            vix_percentile,
            min(1.0, max(-1.0, price_jerk / 10.0)),
            # NEW: Context features
            context_trend_alignment,
            min(1.0, max(-1.0, tech_momentum * 100)),
            min(1.0, max(-1.0, crypto_momentum * 100)),
            min(1.0, max(-1.0, sector_rotation * 100))
        ], dtype=np.float32)
        
        return np.clip(state, -10, 10)
    
    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False,
        temperature: float = 1.0,
        force_explore: bool = False  # NEW: Force exploration when stuck
    ) -> Tuple[int, float, float]:
        """
        Select action using PPO policy with improved exploration
        
        FIXED: Added warmup period, signal quality check, and consecutive loss protection
        
        Returns: action_idx, action_prob, log_prob
        """
        # SAFETY: After consecutive losses, prefer HOLD to stop bleeding
        if self.consecutive_losses >= self.max_consecutive_losses:
            logger.warning(f"⚠️ {self.consecutive_losses} consecutive losses - forcing HOLD to stop bleeding")
            return TradingAction.HOLD, 1.0, 0.0
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Check signal quality from state vector
        # state[2] = confidence, state[0] = predicted_return
        confidence = state[2] if len(state) > 2 else 0.5
        predicted_return = state[0] / 100.0 if len(state) > 0 else 0.0  # Undo the *100 scaling
        
        # QUALITY GATE: Don't trade weak signals regardless of what RL says
        if abs(predicted_return) < 0.001 or confidence < self.signal_quality_threshold:
            logger.debug(f"🚫 Signal too weak (conf={confidence:.1%}, return={predicted_return:.2%}) - HOLD")
            return TradingAction.HOLD, 1.0, 0.0
        
        with torch.no_grad():
            if self.use_shared_features and self.shared_extractor is not None:
                shared_features = self.shared_extractor(state_tensor)
                logits = self.actor(state_tensor, shared_features)
            else:
                logits = self.actor(state_tensor)
            
            # WARMUP: Use higher temperature during warmup for more exploration
            effective_temp = temperature
            if not self.is_warmed_up:
                effective_temp = self.exploration_temp  # Higher temp = more random
                if self.episodes_completed % 10 == 0:
                    logger.info(f"🔥 WARMUP: {self.episodes_completed}/{self.warmup_episodes} episodes (temp={effective_temp:.1f})")
            
            # Apply temperature
            logits = logits / effective_temp
            probs = F.softmax(logits, dim=-1)
            
            # ANTI-COLLAPSE: Ensure minimum probability for non-HOLD actions
            # This prevents the policy from collapsing to always HOLD
            if not deterministic and not self.is_warmed_up:
                min_action_prob = 0.05  # At least 5% chance for each action during warmup
                probs = probs.clamp(min=min_action_prob)
                probs = probs / probs.sum()  # Re-normalize
            
            if deterministic:
                action_idx = probs.argmax().item()
                log_prob = 0.0
            else:
                dist = torch.distributions.Categorical(probs)
                action_idx = dist.sample().item()
                log_prob = dist.log_prob(torch.tensor(action_idx).to(self.device)).item()
            
            action_prob = probs[0, action_idx].item()
        
        return action_idx, action_prob, log_prob

    def calculate_reward(self, trade_result: Optional[Dict] = None, predicted_return: float = 0.0,
                        confidence: float = 0.0, action_taken: int = TradingAction.HOLD,
                        account_balance: float = 1000.0, time_held_minutes: int = 0) -> float:
        """
        FIXED: Stable reward function with bounded range [-5, +5]
        
        Previous version scaled rewards by 200x which caused:
        - Gradient explosions
        - Policy collapse
        - Inability to learn stable patterns
        
        New approach: Risk-adjusted returns with bounded rewards
        """
        if trade_result is not None:
            pnl = trade_result.get('pnl', 0.0)
            entry_price = trade_result.get('entry_price', 1.0)
            
            # FIXED: Use percentage return, not dollar amount scaled by 200
            # This keeps rewards in a stable range
            pnl_pct = pnl / max(1.0, account_balance)
            
            # Clip to reasonable range: -10% to +10% maps to reward
            pnl_pct = np.clip(pnl_pct, -0.10, 0.10)
            
            # Base reward: Map -10% to +10% PnL to -3 to +3 reward
            reward = pnl_pct * 30.0  # 1% gain = +0.3 reward, 1% loss = -0.3 reward
            
            # WIN/LOSS bonuses (small but meaningful)
            if pnl > 0:
                reward += 0.5  # Small win bonus
                # Efficiency bonus for quick wins (options benefit from quick exits)
                if time_held_minutes < 30:
                    reward += 0.3
                elif time_held_minutes < 60:
                    reward += 0.1
                    
                # Track consecutive wins/losses
                self.consecutive_losses = 0
            else:
                reward -= 0.3  # Small loss penalty
                self.consecutive_losses += 1
                
                # Extra penalty for large losses (risk management)
                if pnl_pct < -0.03:  # Lost more than 3%
                    reward -= 0.5
            
            # Drawdown penalty (gentle, not catastrophic)
            max_dd = trade_result.get('max_drawdown', 0.0)
            if max_dd > 0.05:
                reward -= min(1.0, max_dd * 5.0)  # Cap penalty at -1.0
            
            # CLIP FINAL REWARD to stable range
            reward = np.clip(reward, -5.0, 5.0)
                
        else:
            # Step reward for holding/not trading
            reward = 0.0
            
            if action_taken == TradingAction.HOLD:
                # Neutral reward for holding - don't punish waiting for good setups
                reward = 0.0
            elif action_taken in [TradingAction.BUY_CALL_2X, TradingAction.BUY_PUT_2X]:
                # Penalize high leverage on low confidence
                if confidence < 0.5:
                    reward = -0.3
                    
        return float(reward)

    def store_experience(self, state: np.ndarray, action: int, reward: float,
                        next_state: Optional[np.ndarray], done: bool, info: Dict = None,
                        log_prob: float = 0.0):
        """Store experience including log_prob for PPO"""
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
        self.current_episode.append(experience)
        if done:
            self.experience_buffer.append(self.current_episode.copy())
            self.current_episode = []
            episode_reward = sum(exp['reward'] for exp in self.experience_buffer[-1])
            self.training_stats['rewards'].append(episode_reward)
            
            # FIXED: Track episode count for warmup
            self.episodes_completed += 1
            if not self.is_warmed_up and self.episodes_completed >= self.warmup_episodes:
                self.is_warmed_up = True
                logger.info(f"🎓 WARMUP COMPLETE! {self.episodes_completed} episodes gathered. RL policy now active.")
            
            # Track win/loss for consecutive loss tracking
            was_win = episode_reward > 0 or (info and info.get('pnl', 0) > 0)
            if was_win:
                self.consecutive_losses = 0
            
            logger.info(f"📊 Episode {self.episodes_completed}: Reward={episode_reward:.3f}, Steps={len(self.experience_buffer[-1])}, Warmed={'✅' if self.is_warmed_up else '🔥'}")

    def train_step(self, batch_size: int = 32) -> Dict[str, float]:
        """Train using PPO algorithm"""
        if len(self.experience_buffer) < batch_size:
            return {}
            
        episodes = np.random.choice(len(self.experience_buffer), size=min(batch_size, len(self.experience_buffer)), replace=False)
        
        # Flatten episodes into a single batch of steps
        batch_states = []
        batch_actions = []
        batch_log_probs = []
        batch_returns = []
        batch_advantages = []
        
        for ep_idx in episodes:
            episode = self.experience_buffer[ep_idx]
            
            states = torch.FloatTensor(np.array([exp['state'] for exp in episode])).to(self.device)
            rewards = torch.FloatTensor([exp['reward'] for exp in episode]).to(self.device)
            
            # Compute GAE (Generalized Advantage Estimation) or simple returns
            # Using simple discounted returns for now
            returns = []
            R = 0
            for r in reversed(rewards.tolist()):
                R = r + self.gamma * R
                returns.insert(0, R)
            returns = torch.FloatTensor(returns).to(self.device)
            
            # Get values for advantage
            with torch.no_grad():
                if self.use_shared_features:
                    features = self.shared_extractor(states)
                    values = self.critic(states, features).squeeze()
                else:
                    values = self.critic(states).squeeze()
            
            advantages = returns - values
            
            batch_states.extend([exp['state'] for exp in episode])
            batch_actions.extend([exp['action'] for exp in episode])
            batch_log_probs.extend([exp.get('log_prob', 0.0) for exp in episode]) # Handle legacy data
            batch_returns.extend(returns.cpu().numpy())
            batch_advantages.extend(advantages.cpu().numpy())
            
        # Convert to tensors
        states = torch.FloatTensor(np.array(batch_states)).to(self.device)
        actions = torch.LongTensor(np.array(batch_actions)).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(batch_log_probs)).to(self.device)
        returns = torch.FloatTensor(np.array(batch_returns)).to(self.device)
        advantages = torch.FloatTensor(np.array(batch_advantages)).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO Epochs
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        
        for _ in range(self.ppo_epochs):
            if self.use_shared_features:
                features = self.shared_extractor(states)
                logits = self.actor(states, features)
                values = self.critic(states, features).squeeze()
            else:
                logits = self.actor(states)
                values = self.critic(states).squeeze()
            
            dist = torch.distributions.Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            
            # Ratio
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # Surrogate Loss
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.ppo_clip_range, 1.0 + self.ppo_clip_range) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Critic Loss
            critic_loss = F.smooth_l1_loss(values, returns)
            
            # Total Loss
            loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy
            
            # Optimize
            if self.use_shared_features:
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.optimizer.param_groups[0]['params'], self.max_grad_norm)
                self.optimizer.step()
            else:
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()
                self.actor_optimizer.step()
                self.critic_optimizer.step()
                
            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            total_entropy += entropy.item()
            
        # Decay entropy
        self.entropy_coef = max(self.entropy_min, self.entropy_coef * self.entropy_decay)
        
        metrics = {
            'actor_loss': total_actor_loss / self.ppo_epochs,
            'critic_loss': total_critic_loss / self.ppo_epochs,
            'entropy': total_entropy / self.ppo_epochs,
            'learning_rate': self.optimizer.param_groups[0]['lr'] if self.optimizer else 0.0
        }
        
        self.training_stats['total_updates'] += 1
        self.training_stats['actor_losses'].append(metrics['actor_loss'])
        self.training_stats['critic_losses'].append(metrics['critic_loss'])
        self.training_stats['entropy'].append(metrics['entropy'])
        
        return metrics

    def save(self, path: str):
        state = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'training_stats': self.training_stats,
            'config': {
                'state_dim': self.state_dim,
                'action_dim': self.action_dim
            }
        }
        if self.use_shared_features and self.shared_extractor:
            state['shared_extractor_state_dict'] = self.shared_extractor.state_dict()
            state['optimizer_state_dict'] = self.optimizer.state_dict()
            
        torch.save(state, path)
        logger.info(f"💾 PPO Policy saved to {path}")

    def load(self, path: str):
        try:
            state = torch.load(path, map_location=self.device)
            # Load logic similar to before, handling shared/separate
            if self.use_shared_features and 'shared_extractor_state_dict' in state:
                self.shared_extractor.load_state_dict(state['shared_extractor_state_dict'])
                self.optimizer.load_state_dict(state['optimizer_state_dict'])
            
            self.actor.load_state_dict(state['actor_state_dict'])
            self.critic.load_state_dict(state['critic_state_dict'])
            self.training_stats = state.get('training_stats', self.training_stats)
            return True
        except Exception as e:
            logger.error(f"❌ Failed to load PPO policy: {e}")
            return False

    def get_action_distribution(self, state: np.ndarray) -> Dict[str, float]:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if self.use_shared_features:
                features = self.shared_extractor(state_tensor)
                logits = self.actor(state_tensor, features)
            else:
                logits = self.actor(state_tensor)
            probs = F.softmax(logits, dim=-1).squeeze().cpu().numpy()
        return {
            'HOLD': float(probs[0]),
            'BUY_CALL_1X': float(probs[1]),
            'BUY_CALL_2X': float(probs[2]),
            'BUY_PUT_1X': float(probs[3]),
            'BUY_PUT_2X': float(probs[4])
        }

    def reset_consecutive_losses(self):
        """Reset consecutive loss counter (call after a manual intervention or new session)"""
        self.consecutive_losses = 0
        logger.info("🔄 Consecutive loss counter reset")
    
    def get_training_summary(self) -> Dict:
        if not self.training_stats['rewards']:
            return {'status': 'No training data'}
        return {
            'total_updates': self.training_stats['total_updates'],
            'avg_reward': np.mean(self.training_stats['rewards'][-100:]),
            'avg_entropy': np.mean(self.training_stats['entropy'][-100:]),
            'episodes_completed': self.episodes_completed,
            'is_warmed_up': self.is_warmed_up,
            'consecutive_losses': self.consecutive_losses
        }
