"""
V3 Entry Controller - Uses DirectionPredictorV3 for entry decisions.

This controller uses the standalone direction predictor (V3) which achieved
56% validation accuracy for UP/DOWN prediction.

Usage:
    Set ENTRY_CONTROLLER=v3_direction environment variable
    Ensure models/direction_v3.pt exists

Features expected by V3 (computed internally):
    - return_1m, return_5m, return_15m
    - volatility (20-bar rolling std)
    - momentum (price position in range)
    - volume_spike

The controller extracts these from the existing feature set or computes them
from recent price history.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class V3Decision:
    """Entry decision from V3 predictor."""
    action: str  # "HOLD", "BUY_CALLS", "BUY_PUTS"
    confidence: float
    direction_prob: float  # P(UP) if action is BUY_CALLS, P(DOWN) if BUY_PUTS
    raw_direction: int  # 0=DOWN, 1=UP
    details: Dict[str, Any]


class V3EntryController:
    """
    Entry controller using DirectionPredictorV3.

    The V3 predictor is a simplified direction-only model that achieved
    56% validation accuracy on UP/DOWN prediction.
    """

    def __init__(self, model_path: str = None, min_confidence: float = 0.55):
        """
        Initialize V3 entry controller.

        Args:
            model_path: Path to trained V3 model (.pt file)
            min_confidence: Minimum confidence to trade (default 0.55)
        """
        self.min_confidence = min_confidence
        self.model = None
        self.config = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Price history for feature computation
        self.price_history = []
        self.volume_history = []
        self.max_history = 100

        # Load model
        if model_path is None:
            model_path = os.environ.get('V3_MODEL_PATH', 'models/direction_v3.pt')

        self._load_model(model_path)

    def _load_model(self, model_path: str):
        """Load the V3 predictor model."""
        path = Path(model_path)
        if not path.is_absolute():
            # Try relative to current directory and script directory
            for base in [Path.cwd(), Path(__file__).parent.parent]:
                candidate = base / model_path
                if candidate.exists():
                    path = candidate
                    break

        if not path.exists():
            logger.warning(f"V3 model not found at {path}, will use random baseline")
            return

        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.config = checkpoint.get('config', {})

            # Import and create model
            from bot_modules.neural_networks import DirectionPredictorV3

            feature_dim = self.config.get('feature_dim', 6)
            sequence_length = self.config.get('sequence_length', 30)

            self.model = DirectionPredictorV3(
                feature_dim=feature_dim,
                sequence_length=sequence_length
            ).to(self.device)

            self.model.load_state_dict(checkpoint['state_dict'])
            self.model.eval()

            training_info = checkpoint.get('training', {})
            logger.info(f"V3 model loaded from {path}")
            logger.info(f"  - Feature dim: {feature_dim}")
            logger.info(f"  - Sequence length: {sequence_length}")
            logger.info(f"  - Val accuracy: {training_info.get('best_val_acc', 'unknown'):.2%}")

        except Exception as e:
            logger.error(f"Failed to load V3 model: {e}")
            self.model = None

    def update_price(self, price: float, volume: float = 1.0):
        """Update price history for feature computation."""
        self.price_history.append(price)
        self.volume_history.append(volume)

        if len(self.price_history) > self.max_history:
            self.price_history = self.price_history[-self.max_history:]
            self.volume_history = self.volume_history[-self.max_history:]

    def _compute_features(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute V3 features from price history.

        Returns:
            current_features: [D] current feature vector
            sequence: [T, D] historical feature sequence
        """
        prices = np.array(self.price_history, dtype=np.float32)
        volumes = np.array(self.volume_history, dtype=np.float32)

        n = len(prices)
        sequence_length = self.config.get('sequence_length', 30) if self.config else 30

        if n < sequence_length + 1:
            # Pad with zeros if not enough history
            pad_len = sequence_length + 1 - n
            prices = np.concatenate([np.full(pad_len, prices[0]), prices])
            volumes = np.concatenate([np.ones(pad_len), volumes])
            n = len(prices)

        # Compute features for each timestep
        feature_dim = 6
        all_features = np.zeros((n, feature_dim), dtype=np.float32)

        # Returns
        ret_1m = np.zeros(n)
        ret_1m[1:] = (prices[1:] - prices[:-1]) / (prices[:-1] + 1e-8)

        ret_5m = np.zeros(n)
        if n > 5:
            ret_5m[5:] = (prices[5:] - prices[:-5]) / (prices[:-5] + 1e-8)

        ret_15m = np.zeros(n)
        if n > 15:
            ret_15m[15:] = (prices[15:] - prices[:-15]) / (prices[:-15] + 1e-8)

        # Volatility (rolling std of returns)
        volatility = np.zeros(n)
        for i in range(20, n):
            volatility[i] = np.std(ret_1m[i-20:i])

        # Momentum (price position in 20-bar range)
        momentum = np.zeros(n)
        for i in range(20, n):
            high_20 = np.max(prices[i-20:i])
            low_20 = np.min(prices[i-20:i])
            if high_20 > low_20:
                momentum[i] = (prices[i] - low_20) / (high_20 - low_20)
            else:
                momentum[i] = 0.5

        # Volume spike
        vol_spike = np.zeros(n)
        vol_mean = np.mean(volumes) + 1e-8
        for i in range(20, n):
            vol_mean_20 = np.mean(volumes[i-20:i]) + 1e-8
            vol_spike[i] = volumes[i] / vol_mean_20

        # Stack features
        all_features[:, 0] = ret_1m
        all_features[:, 1] = ret_5m
        all_features[:, 2] = ret_15m
        all_features[:, 3] = volatility
        all_features[:, 4] = momentum
        all_features[:, 5] = vol_spike

        # Current features and sequence
        current_features = all_features[-1]
        sequence = all_features[-sequence_length:]

        return current_features, sequence

    def decide(self, signal: Dict = None, vix_level: float = None,
               market_open: bool = True, hmm_trend: float = None) -> V3Decision:
        """
        Make entry decision using V3 predictor with HMM filter.

        Args:
            signal: Optional signal dict with additional info
            vix_level: Current VIX level
            market_open: Whether market is open
            hmm_trend: HMM trend value (0=bearish, 0.5=neutral, 1=bullish)

        Returns:
            V3Decision with action, confidence, and details
        """
        # Default to HOLD
        default_decision = V3Decision(
            action="HOLD",
            confidence=0.0,
            direction_prob=0.5,
            raw_direction=1,
            details={"reason": "default"}
        )

        if not market_open:
            default_decision.details["reason"] = "market_closed"
            return default_decision

        if self.model is None:
            # Fallback to random if no model
            direction = np.random.choice([0, 1])
            return V3Decision(
                action="BUY_PUTS" if direction == 0 else "BUY_CALLS",
                confidence=0.5,
                direction_prob=0.5,
                raw_direction=direction,
                details={"reason": "no_model_random_fallback"}
            )

        # Extract HMM trend from signal if not provided
        if hmm_trend is None and signal:
            hmm_trend = signal.get('hmm_trend', 0.5)

        if len(self.price_history) < 20:
            default_decision.details["reason"] = "insufficient_history"
            return default_decision

        try:
            # Compute features
            current_features, sequence = self._compute_features()

            # Convert to tensors
            cur_tensor = torch.FloatTensor(current_features).unsqueeze(0).to(self.device)
            seq_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)

            # Forward pass
            with torch.no_grad():
                out = self.model(cur_tensor, seq_tensor)

            direction = out['direction'].item()  # 0=DOWN, 1=UP
            direction_probs = out['direction_probs'][0].cpu().numpy()  # [P(DOWN), P(UP)]
            confidence = out['confidence'].item()

            # Determine action based on direction
            if direction == 1:  # UP
                action = "BUY_CALLS"
                direction_prob = direction_probs[1]  # P(UP)
            else:  # DOWN
                action = "BUY_PUTS"
                direction_prob = direction_probs[0]  # P(DOWN)

            # Apply confidence threshold
            if direction_prob < self.min_confidence:
                action = "HOLD"

            # HMM ALIGNMENT FILTER: Only trade when V3 agrees with HMM trend
            hmm_aligned = True
            hmm_veto_reason = None
            if hmm_trend is not None and action != "HOLD":
                # Ensure hmm_trend is float
                try:
                    hmm_val = float(hmm_trend)
                except (TypeError, ValueError):
                    hmm_val = 0.5

                # Bullish HMM (>0.6) should only allow CALLS
                # Bearish HMM (<0.4) should only allow PUTS
                # Neutral HMM (0.4-0.6) allows either but with caution
                if hmm_val > 0.6 and action == "BUY_PUTS":
                    hmm_aligned = False
                    hmm_veto_reason = f"HMM bullish ({hmm_val:.2f}) conflicts with PUT"
                elif hmm_val < 0.4 and action == "BUY_CALLS":
                    hmm_aligned = False
                    hmm_veto_reason = f"HMM bearish ({hmm_val:.2f}) conflicts with CALL"

                if not hmm_aligned:
                    action = "HOLD"

            # Get hmm_val for details (already computed above if filter ran)
            try:
                hmm_val_for_details = float(hmm_trend) if hmm_trend is not None else 0.5
            except (TypeError, ValueError):
                hmm_val_for_details = 0.5

            return V3Decision(
                action=action,
                confidence=confidence,
                direction_prob=float(direction_prob),
                raw_direction=direction,
                details={
                    "p_up": float(direction_probs[1]),
                    "p_down": float(direction_probs[0]),
                    "model_confidence": confidence,
                    "threshold": self.min_confidence,
                    "hmm_trend": hmm_val_for_details,
                    "hmm_aligned": hmm_aligned,
                    "hmm_veto_reason": hmm_veto_reason,
                }
            )

        except Exception as e:
            logger.error(f"V3 prediction error: {e}")
            default_decision.details["reason"] = f"prediction_error: {e}"
            return default_decision


# Global controller instance
_controller = None


def get_v3_controller() -> V3EntryController:
    """Get or create global V3 controller instance."""
    global _controller
    if _controller is None:
        model_path = os.environ.get('V3_MODEL_PATH', 'models/direction_v3.pt')
        min_conf = float(os.environ.get('V3_MIN_CONFIDENCE', '0.55'))
        _controller = V3EntryController(model_path=model_path, min_confidence=min_conf)
    return _controller


def decide_from_signal(signal: Dict, account_state: Dict = None,
                       vix_level: float = None, market_open: bool = True,
                       verbose: bool = False) -> Tuple[int, float, Dict]:
    """
    Entry point for V3 entry decisions.

    Args:
        signal: Signal dictionary with price, prediction info
        account_state: Current account state
        vix_level: VIX level
        market_open: Whether market is open
        verbose: Print debug info

    Returns:
        (action_int, confidence, details)
        action_int: 0=HOLD, 1=BUY_CALLS, 2=BUY_PUTS
    """
    controller = get_v3_controller()

    # Update price history from signal
    price = signal.get('current_price', signal.get('spot_price', 0))
    volume = signal.get('volume', 1.0)
    if price > 0:
        controller.update_price(price, volume)

    # Get decision
    decision = controller.decide(signal=signal, vix_level=vix_level, market_open=market_open)

    # Map action to int
    action_map = {"HOLD": 0, "BUY_CALLS": 1, "BUY_PUTS": 2}
    action_int = action_map.get(decision.action, 0)

    if verbose:
        logger.info(f"V3 Decision: {decision.action} (conf={decision.confidence:.3f}, "
                   f"p_dir={decision.direction_prob:.3f})")

    return action_int, decision.confidence, decision.details
