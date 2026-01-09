"""
Best Plays Gate

Uses pretrained model to filter entries down to ~3-4 high-quality plays per day.

Trading day = 6.5 hours = 390 minutes
3-4 trades/day = 1 trade every ~100 minutes

Usage:
    BEST_PLAYS_MODEL=models/pretrained/best_plays.pt python scripts/train_time_travel.py
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple

import torch
import numpy as np

logger = logging.getLogger(__name__)


class BestPlaysGate:
    """
    Gate that uses pretrained "best plays" model to filter entries.

    Only allows trades that:
    1. Model predicts is a "best play" (high profit potential)
    2. We haven't exceeded daily trade limit
    3. Minimum time since last trade (prevents clustering)
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        max_daily_trades: int = 4,
        min_minutes_between_trades: int = 60,
        min_best_play_prob: float = 0.6,
        device: str = 'cpu'
    ):
        self.max_daily_trades = max_daily_trades
        self.min_minutes_between = min_minutes_between_trades
        self.min_prob = min_best_play_prob
        self.device = device

        self.model = None
        self.model_loaded = False

        # Tracking
        self.trades_today = 0
        self.last_trade_time: Optional[datetime] = None
        self.current_date: Optional[datetime] = None

        # Stats
        self.total_signals = 0
        self.passed_signals = 0
        self.blocked_by_model = 0
        self.blocked_by_limit = 0
        self.blocked_by_cooldown = 0

        # Load model if path provided
        if model_path:
            self._load_model(model_path)
        elif os.environ.get('BEST_PLAYS_MODEL'):
            self._load_model(os.environ['BEST_PLAYS_MODEL'])

    def _load_model(self, path: str):
        """Load pretrained best plays model"""
        if not os.path.exists(path):
            logger.warning(f"Best plays model not found: {path}")
            return

        try:
            # Import here to avoid circular imports
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).parent.parent))

            checkpoint = torch.load(path, map_location=self.device)

            # Create model
            from scripts.pretrain_best_plays import BestPlaysModel
            self.model = BestPlaysModel().to(self.device)
            self.model.load_state_dict(checkpoint['model_state'])
            self.model.eval()

            self.model_loaded = True
            logger.info(f"✅ Best plays model loaded from {path}")
            logger.info(f"   Trained up to: {checkpoint.get('cutoff_date', 'unknown')}")
            logger.info(f"   Min profit threshold: {checkpoint.get('min_profit_pct', 'unknown')}%")

        except Exception as e:
            logger.warning(f"Error loading best plays model: {e}")
            self.model = None

    def _build_features(self, features_dict: Dict) -> torch.Tensor:
        """Build feature tensor from features dict"""
        feature_dim = 50
        state = np.zeros(feature_dim, dtype=np.float32)

        feature_names = [
            'vix', 'vix_percentile', 'vix_term_structure',
            'spy_return_1m', 'spy_return_5m', 'spy_return_15m', 'spy_return_30m',
            'spy_volatility_5m', 'spy_volatility_15m',
            'rsi_14', 'macd', 'macd_signal', 'bb_position',
            'volume_ratio', 'price_momentum',
            'hmm_trend', 'hmm_volatility', 'hmm_liquidity', 'hmm_confidence',
            'predicted_return', 'confidence', 'direction_up', 'direction_down',
            'fear_greed', 'pcr', 'news_sentiment',
            'theta', 'delta', 'gamma', 'vega', 'iv',
            'time_to_expiry', 'moneyness',
            'sector_spy_corr', 'sector_strength',
            'tda_entropy_h0', 'tda_entropy_h1', 'tda_complexity',
            'spy_price', 'momentum_5m', 'momentum_15m'
        ]

        for i, name in enumerate(feature_names):
            if i >= feature_dim:
                break
            val = features_dict.get(name, 0.0)
            if val is None:
                val = 0.0
            # Normalize
            if name == 'vix':
                val = float(val) / 50.0
            elif name == 'spy_price':
                val = (float(val) - 500) / 100
            state[i] = float(val)

        return torch.FloatTensor(state).unsqueeze(0).to(self.device)

    def should_trade(
        self,
        features: Dict,
        current_time: Optional[datetime] = None,
        signal_direction: str = 'CALL'
    ) -> Tuple[bool, str, float]:
        """
        Check if this is a best play worth trading.

        Returns:
            (should_trade, reason, best_play_probability)
        """
        self.total_signals += 1

        if current_time is None:
            current_time = datetime.now()

        # Reset daily counter if new day
        if self.current_date is None or current_time.date() != self.current_date.date():
            self.trades_today = 0
            self.current_date = current_time

        # Check 1: Daily trade limit
        if self.trades_today >= self.max_daily_trades:
            self.blocked_by_limit += 1
            return False, f"Daily limit reached ({self.trades_today}/{self.max_daily_trades})", 0.0

        # Check 2: Cooldown between trades
        if self.last_trade_time is not None:
            minutes_since = (current_time - self.last_trade_time).total_seconds() / 60
            if minutes_since < self.min_minutes_between:
                self.blocked_by_cooldown += 1
                return False, f"Cooldown ({minutes_since:.0f}/{self.min_minutes_between} min)", 0.0

        # Check 3: Model prediction (if available)
        best_prob = 0.5  # Default if no model

        if self.model_loaded and self.model is not None:
            try:
                with torch.no_grad():
                    feat_tensor = self._build_features(features)
                    out = self.model(feat_tensor)

                    best_prob = torch.sigmoid(out['best_play_logit']).item()
                    dir_probs = torch.softmax(out['direction_logits'], dim=1)

                    # Check if model agrees with signal direction
                    pred_dir = out['direction_logits'].argmax(dim=1).item()
                    signal_dir = 1 if signal_direction.upper() == 'CALL' else 0

                    # Must be best play AND direction must match
                    if best_prob < self.min_prob:
                        self.blocked_by_model += 1
                        return False, f"Not best play (prob={best_prob:.1%} < {self.min_prob:.0%})", best_prob

                    if pred_dir != signal_dir:
                        self.blocked_by_model += 1
                        return False, f"Direction mismatch (model={['PUT','CALL'][pred_dir]}, signal={signal_direction})", best_prob

            except Exception as e:
                logger.warning(f"Best plays model error: {e}")
                best_prob = 0.5

        # All checks passed!
        self.passed_signals += 1
        self.trades_today += 1
        self.last_trade_time = current_time

        return True, f"Best play! (prob={best_prob:.1%})", best_prob

    def get_stats(self) -> Dict:
        """Get gate statistics"""
        return {
            'total_signals': self.total_signals,
            'passed': self.passed_signals,
            'pass_rate': self.passed_signals / max(1, self.total_signals) * 100,
            'blocked_by_model': self.blocked_by_model,
            'blocked_by_limit': self.blocked_by_limit,
            'blocked_by_cooldown': self.blocked_by_cooldown,
            'trades_today': self.trades_today,
            'model_loaded': self.model_loaded
        }

    def log_stats(self):
        """Log gate statistics"""
        stats = self.get_stats()
        logger.info(f"[BEST_PLAYS_GATE] Stats:")
        logger.info(f"   Total signals: {stats['total_signals']}")
        logger.info(f"   Passed: {stats['passed']} ({stats['pass_rate']:.1f}%)")
        logger.info(f"   Blocked by model: {stats['blocked_by_model']}")
        logger.info(f"   Blocked by daily limit: {stats['blocked_by_limit']}")
        logger.info(f"   Blocked by cooldown: {stats['blocked_by_cooldown']}")


# Global instance
_best_plays_gate: Optional[BestPlaysGate] = None


def get_best_plays_gate() -> BestPlaysGate:
    """Get or create the global best plays gate"""
    global _best_plays_gate

    if _best_plays_gate is None:
        model_path = os.environ.get('BEST_PLAYS_MODEL')
        max_trades = int(os.environ.get('BEST_PLAYS_MAX_DAILY', '4'))
        min_between = int(os.environ.get('BEST_PLAYS_MIN_BETWEEN', '60'))
        min_prob = float(os.environ.get('BEST_PLAYS_MIN_PROB', '0.6'))

        _best_plays_gate = BestPlaysGate(
            model_path=model_path,
            max_daily_trades=max_trades,
            min_minutes_between_trades=min_between,
            min_best_play_prob=min_prob
        )

    return _best_plays_gate
