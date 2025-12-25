"""
Direction-Based Entry Controller
================================

Uses the dedicated DirectionPredictor for entry decisions.
Optimized for higher win rate by focusing 100% on direction prediction.

Usage:
    ENTRY_CONTROLLER=direction python scripts/train_time_travel.py
"""

import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DirectionDecision:
    """Result of direction-based entry decision."""
    action: str  # 'HOLD', 'BUY_CALLS', 'BUY_PUTS'
    confidence: float
    direction: int  # 1=UP, -1=DOWN, 0=NEUTRAL
    magnitude: float
    details: Dict[str, Any]


class DirectionEntryController:
    """
    Entry controller using dedicated DirectionPredictor.

    Key features:
    - Dedicated direction model (not shared with other outputs)
    - Online learning from trade outcomes
    - Confidence-based position sizing
    - HMM alignment checking
    """

    def __init__(
        self,
        feature_dim: int = 50,
        sequence_length: int = 60,
        min_confidence: float = 0.55,
        min_magnitude: float = 0.002,  # 0.2% minimum expected move
        device: str = 'cpu',
        training_mode: bool = True,  # Start in training mode
        exploration_trades: int = 100,  # Explore for first N trades
    ):
        self.feature_dim = feature_dim
        self.sequence_length = sequence_length
        self.min_confidence = min_confidence
        self.min_magnitude = min_magnitude
        self.device = device
        self.training_mode = training_mode
        self.exploration_trades = exploration_trades

        # Import here to avoid circular imports
        from bot_modules.neural_networks import DirectionPredictor

        # Initialize the direction predictor
        self.model = DirectionPredictor(feature_dim, sequence_length).to(device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.0003, weight_decay=0.01)
        self.criterion = nn.CrossEntropyLoss()

        # Training stats
        self.predictions = []  # Store for online learning
        self.pending_outcomes = {}  # Map trade_id -> prediction info
        self.trade_count = 0
        self.correct_count = 0
        self.total_trained = 0
        self.training_buffer = []  # (features, seq, label) tuples

        logger.info("🎯 DirectionEntryController initialized")
        logger.info(f"   Training mode: {training_mode}")
        logger.info(f"   Exploration trades: {exploration_trades}")
        logger.info(f"   Min confidence: {min_confidence:.0%}")
        logger.info(f"   Min magnitude: {min_magnitude:.2%}")
        logger.info(f"   Device: {device}")

    def decide(
        self,
        current_features: torch.Tensor,
        sequence_features: torch.Tensor,
        hmm_trend: float = 0.5,
        hmm_confidence: float = 0.5,
        vix_level: float = 18.0,
    ) -> DirectionDecision:
        """
        Make entry decision based on direction prediction.

        Args:
            current_features: [D] current feature vector
            sequence_features: [T, D] sequence of features
            hmm_trend: HMM trend value (0=bearish, 0.5=neutral, 1=bullish)
            hmm_confidence: HMM confidence
            vix_level: Current VIX

        Returns:
            DirectionDecision with action and details
        """
        import random

        self.model.eval()

        # Prepare inputs
        if current_features.dim() == 1:
            current_features = current_features.unsqueeze(0)
        if sequence_features.dim() == 2:
            sequence_features = sequence_features.unsqueeze(0)

        current_features = current_features.to(self.device)
        sequence_features = sequence_features.to(self.device)

        with torch.no_grad():
            output = self.model(current_features, sequence_features)

        direction_probs = output['direction_probs'][0].cpu().numpy()
        direction = output['direction'][0].item()  # 0=DOWN, 1=UP
        confidence = output['confidence'][0].item()
        magnitude = output['magnitude'][0].item()

        # Determine if we're in exploration phase
        in_exploration = self.training_mode and self.trade_count < self.exploration_trades

        details = {
            'direction_probs': direction_probs.tolist(),
            'raw_direction': direction,
            'raw_confidence': confidence,
            'magnitude': magnitude,
            'hmm_trend': hmm_trend,
            'vix': vix_level,
            'in_exploration': in_exploration,
            'trade_count': self.trade_count,
        }

        # Convert direction to +1/-1
        pred_direction = 1 if direction == 1 else -1

        # === EXPLORATION MODE ===
        if in_exploration:
            # During exploration, use MODEL direction but require HMM alignment
            # This gathers quality labeled data for learning

            # Rate limit: only trade every 20 cycles
            current_cycle = getattr(self, '_cycle_counter', 0)
            last_trade_cycle = getattr(self, '_last_trade_cycle', -100)
            cycle_gap = current_cycle - last_trade_cycle
            if cycle_gap < 20:
                return DirectionDecision(
                    action='HOLD',
                    confidence=confidence,
                    direction=0,
                    magnitude=magnitude,
                    details={**details, 'reason': f'exploration_rate_limit (gap={cycle_gap}<20)'}
                )

            # Block extreme VIX
            if vix_level > 30:
                return DirectionDecision(
                    action='HOLD',
                    confidence=confidence,
                    direction=0,
                    magnitude=magnitude,
                    details={**details, 'reason': f'high_vix ({vix_level:.1f} > 30)'}
                )

            # === ALIGNMENT CHECK: Model direction must match HMM ===
            hmm_bullish = hmm_trend > 0.55
            hmm_bearish = hmm_trend < 0.45
            hmm_neutral = not hmm_bullish and not hmm_bearish

            # Only trade when model and HMM agree
            model_bullish = pred_direction == 1
            model_bearish = pred_direction == -1

            aligned = False
            if model_bullish and hmm_bullish:
                action = 'BUY_CALLS'
                details['reason'] = 'exploration_aligned_bullish'
                aligned = True
            elif model_bearish and hmm_bearish:
                action = 'BUY_PUTS'
                details['reason'] = 'exploration_aligned_bearish'
                aligned = True
            elif hmm_neutral:
                # In neutral HMM, follow model with lower probability (30%)
                if random.random() < 0.30:
                    if model_bullish:
                        action = 'BUY_CALLS'
                        details['reason'] = 'exploration_neutral_call'
                    else:
                        action = 'BUY_PUTS'
                        details['reason'] = 'exploration_neutral_put'
                    aligned = True

            if not aligned:
                return DirectionDecision(
                    action='HOLD',
                    confidence=confidence,
                    direction=0,
                    magnitude=magnitude,
                    details={**details, 'reason': 'exploration_misaligned'}
                )

            # Require minimum magnitude for exploration trades
            if magnitude < 0.001:  # 0.1% minimum expected move
                return DirectionDecision(
                    action='HOLD',
                    confidence=confidence,
                    direction=0,
                    magnitude=magnitude,
                    details={**details, 'reason': f'exploration_low_magnitude ({magnitude:.2%})'}
                )

            # Store prediction for learning
            self.predictions.append({
                'current_features': current_features.cpu(),
                'sequence_features': sequence_features.cpu(),
                'predicted_direction': pred_direction,
                'confidence': confidence,
            })

            details['effective_confidence'] = confidence

            return DirectionDecision(
                action=action,
                confidence=confidence,
                direction=pred_direction,
                magnitude=magnitude,
                details=details
            )

        # === NORMAL MODE (post-exploration) ===

        # === GATE 1: Confidence check ===
        if confidence < self.min_confidence:
            return DirectionDecision(
                action='HOLD',
                confidence=confidence,
                direction=0,
                magnitude=magnitude,
                details={**details, 'reason': f'low_confidence ({confidence:.1%} < {self.min_confidence:.0%})'}
            )

        # === GATE 2: Magnitude check ===
        if magnitude < self.min_magnitude:
            return DirectionDecision(
                action='HOLD',
                confidence=confidence,
                direction=0,
                magnitude=magnitude,
                details={**details, 'reason': f'low_magnitude ({magnitude:.2%} < {self.min_magnitude:.2%})'}
            )

        # === GATE 3: VIX extremes ===
        if vix_level > 35:
            return DirectionDecision(
                action='HOLD',
                confidence=confidence,
                direction=0,
                magnitude=magnitude,
                details={**details, 'reason': f'extreme_vix ({vix_level:.1f} > 35)'}
            )

        # === GATE 4: HMM alignment (soft) ===
        effective_confidence = confidence

        hmm_bullish = hmm_trend > 0.55
        hmm_bearish = hmm_trend < 0.45

        if pred_direction == 1:  # Predicting UP
            if hmm_bearish and hmm_confidence > 0.6:
                # Counter-trend: apply penalty
                effective_confidence *= 0.75
                details['hmm_penalty'] = 'counter_trend_bullish'
            elif hmm_bullish:
                # Aligned: small boost
                effective_confidence = min(1.0, effective_confidence * 1.10)
                details['hmm_boost'] = 'aligned_bullish'
        else:  # Predicting DOWN
            if hmm_bullish and hmm_confidence > 0.6:
                # Counter-trend: apply penalty
                effective_confidence *= 0.75
                details['hmm_penalty'] = 'counter_trend_bearish'
            elif hmm_bearish:
                # Aligned: small boost
                effective_confidence = min(1.0, effective_confidence * 1.10)
                details['hmm_boost'] = 'aligned_bearish'

        details['effective_confidence'] = effective_confidence

        # === GATE 5: Final threshold after penalties ===
        if effective_confidence < 0.50:
            return DirectionDecision(
                action='HOLD',
                confidence=effective_confidence,
                direction=0,
                magnitude=magnitude,
                details={**details, 'reason': f'post_penalty_low_conf ({effective_confidence:.1%} < 50%)'}
            )

        # === ENTRY DECISION ===
        if pred_direction == 1:
            action = 'BUY_CALLS'
            details['reason'] = 'bullish_direction'
        else:
            action = 'BUY_PUTS'
            details['reason'] = 'bearish_direction'

        # Store prediction for learning
        self.predictions.append({
            'current_features': current_features.cpu(),
            'sequence_features': sequence_features.cpu(),
            'predicted_direction': pred_direction,
            'confidence': confidence,
        })

        return DirectionDecision(
            action=action,
            confidence=effective_confidence,
            direction=pred_direction,
            magnitude=magnitude,
            details=details
        )

    def set_cycle(self, cycle: int):
        """Set current cycle number for rate limiting."""
        self._cycle_counter = cycle

    def register_trade(self, trade_id: str, features: torch.Tensor, sequences: torch.Tensor, direction: int):
        """
        Register a trade for later outcome learning.

        Args:
            trade_id: Unique trade identifier
            features: Current features at entry
            sequences: Sequence features at entry
            direction: Predicted direction (1=UP, -1=DOWN)
        """
        self.pending_outcomes[trade_id] = {
            'features': features.cpu() if isinstance(features, torch.Tensor) else features,
            'sequences': sequences.cpu() if isinstance(sequences, torch.Tensor) else sequences,
            'direction': direction,
        }
        # Update last trade cycle for rate limiting
        self._last_trade_cycle = getattr(self, '_cycle_counter', 0)

    def record_outcome(self, predicted_direction: int, actual_return: float, trade_id: Optional[str] = None):
        """
        Record trade outcome for online learning.

        Args:
            predicted_direction: 1 for UP, -1 for DOWN
            actual_return: Actual price return (positive = up)
            trade_id: Optional trade ID to retrieve stored features
        """
        self.trade_count += 1

        # Determine if prediction was correct
        actual_direction = 1 if actual_return > 0 else -1
        correct = (predicted_direction == actual_direction)

        if correct:
            self.correct_count += 1

        # If we have stored features, add to training buffer
        if trade_id and trade_id in self.pending_outcomes:
            trade_info = self.pending_outcomes.pop(trade_id)
            # Label: 0=DOWN, 1=UP based on actual outcome
            label = 1 if actual_direction == 1 else 0
            self.training_buffer.append({
                'features': trade_info['features'],
                'sequences': trade_info['sequences'],
                'label': label,
            })

            # Train if buffer is large enough
            if len(self.training_buffer) >= 16:
                self._train_from_outcomes()

        # Log periodically
        if self.trade_count % 25 == 0:
            accuracy = self.correct_count / self.trade_count if self.trade_count > 0 else 0
            mode = "EXPLORE" if self.trade_count < self.exploration_trades else "LEARNED"
            logger.info(f"📊 Direction [{mode}] accuracy: {accuracy:.1%} ({self.correct_count}/{self.trade_count})")

    def _train_from_outcomes(self):
        """Train from accumulated outcome data."""
        if len(self.training_buffer) < 8:
            return

        self.model.train()

        # Stack all samples
        features_list = []
        sequences_list = []
        labels_list = []

        for item in self.training_buffer:
            features_list.append(item['features'])
            sequences_list.append(item['sequences'])
            labels_list.append(item['label'])

        features = torch.cat(features_list, dim=0).to(self.device)
        sequences = torch.cat(sequences_list, dim=0).to(self.device)
        labels = torch.tensor(labels_list, dtype=torch.long).to(self.device)

        # Train for a few steps
        for _ in range(3):
            self.optimizer.zero_grad()

            output = self.model(features, sequences)
            direction_logits = self.model.direction_logits(output['embedding'])

            loss = self.criterion(direction_logits, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

        self.total_trained += len(self.training_buffer)
        logger.info(f"🧠 Trained on {len(self.training_buffer)} outcomes (loss: {loss.item():.4f})")

        # Clear buffer
        self.training_buffer = []
        self.model.eval()

    def train_step(self, features: torch.Tensor, sequences: torch.Tensor, labels: torch.Tensor):
        """
        Single training step for the direction predictor.

        Args:
            features: [B, D] current features
            sequences: [B, T, D] sequence features
            labels: [B] direction labels (0=DOWN, 1=UP)
        """
        self.model.train()

        features = features.to(self.device)
        sequences = sequences.to(self.device)
        labels = labels.to(self.device).long()

        self.optimizer.zero_grad()

        output = self.model(features, sequences)
        direction_logits = self.model.direction_logits(output['embedding'])

        loss = self.criterion(direction_logits, labels)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()
        self.total_trained += 1

        return loss.item()

    def train_from_buffer(self, batch_size: int = 32):
        """Train from accumulated predictions if we have enough."""
        if len(self.predictions) < batch_size:
            return None

        # This would need actual outcome labels - for now just return
        # In practice, this is called after trades complete with actual outcomes
        return None

    def save(self, path: str):
        """Save model and optimizer state."""
        torch.save({
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'trade_count': self.trade_count,
            'correct_count': self.correct_count,
            'total_trained': self.total_trained,
        }, path)
        logger.info(f"💾 DirectionEntryController saved to {path}")

    def load(self, path: str):
        """Load model and optimizer state."""
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            self.trade_count = checkpoint.get('trade_count', 0)
            self.correct_count = checkpoint.get('correct_count', 0)
            self.total_trained = checkpoint.get('total_trained', 0)
            logger.info(f"📂 DirectionEntryController loaded from {path}")
            return True
        return False

    @property
    def accuracy(self) -> float:
        """Current direction prediction accuracy."""
        if self.trade_count == 0:
            return 0.0
        return self.correct_count / self.trade_count


# Global instance for easy access
_controller_instance: Optional[DirectionEntryController] = None


def get_direction_controller(
    feature_dim: int = 50,
    model_path: Optional[str] = None,
) -> DirectionEntryController:
    """Get or create the direction controller instance."""
    global _controller_instance

    if _controller_instance is None:
        _controller_instance = DirectionEntryController(feature_dim=feature_dim)

        if model_path and os.path.exists(model_path):
            _controller_instance.load(model_path)

    return _controller_instance


def decide_from_signal(
    signal: Dict[str, Any],
    current_features: Optional[torch.Tensor] = None,
    sequence_features: Optional[torch.Tensor] = None,
    **kwargs
) -> Tuple[str, float, Dict]:
    """
    Entry point for direction-based decisions.

    Compatible with existing entry controller interface.
    """
    controller = get_direction_controller()

    # Extract features from signal if not provided
    if current_features is None:
        # Try to get from signal
        if 'features' in signal:
            current_features = torch.tensor(signal['features'], dtype=torch.float32)
        else:
            # Fallback: create dummy features
            current_features = torch.zeros(50)

    if sequence_features is None:
        if 'sequence_features' in signal:
            sequence_features = torch.tensor(signal['sequence_features'], dtype=torch.float32)
        else:
            # Fallback: create dummy sequence
            sequence_features = torch.zeros(60, 50)

    # Get HMM values
    hmm_trend = kwargs.get('hmm_trend', signal.get('hmm_trend', 0.5))
    hmm_confidence = kwargs.get('hmm_confidence', signal.get('hmm_confidence', 0.5))
    vix_level = kwargs.get('vix_level', signal.get('vix', 18.0))

    decision = controller.decide(
        current_features,
        sequence_features,
        hmm_trend=hmm_trend,
        hmm_confidence=hmm_confidence,
        vix_level=vix_level,
    )

    return decision.action, decision.confidence, decision.details
