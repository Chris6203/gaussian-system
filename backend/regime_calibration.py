"""
Per-Regime Calibration - Separate calibrators per HMM regime and time-of-day.

Key insight: Calibration behavior differs massively by regime and market session.
A 70% confidence in low-vol open is very different from 70% in high-vol close.

Maintains separate calibration buffers and fits per:
- HMM regime bucket (low/normal/high volatility)
- Time-of-day bucket (open/midday/close)
"""

import os
import numpy as np
import logging
from collections import defaultdict
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field
from datetime import datetime
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


@dataclass
class CalibrationBuffer:
    """Buffer for a single regime's calibration data."""
    confidences: List[float] = field(default_factory=list)
    outcomes: List[int] = field(default_factory=list)  # 1=win, 0=loss
    model: Optional[IsotonicRegression] = None
    is_fitted: bool = False
    last_fit_size: int = 0
    win_rate: float = 0.5


class RegimeCalibrator:
    """
    Per-regime calibration system.

    Maintains separate calibrators for each combination of:
    - Volatility regime: low (<0.4), normal (0.4-0.6), high (>0.6)
    - Time bucket: open (9:30-10:30), midday (10:30-14:30), close (14:30-16:00)
    """

    def __init__(
        self,
        min_samples: int = 30,
        refit_interval: int = 20,
        enabled: bool = True,
    ):
        self.min_samples = min_samples
        self.refit_interval = refit_interval
        self.enabled = os.environ.get('REGIME_CALIBRATION', '1') == '1' if enabled else False

        # Separate buffers per regime
        self.buffers: Dict[str, CalibrationBuffer] = defaultdict(CalibrationBuffer)

        # Global fallback calibrator
        self.global_buffer = CalibrationBuffer()

        logger.info(f"[REGIME_CAL] Initialized: enabled={self.enabled}, min_samples={min_samples}")

    def get_regime_key(
        self,
        hmm_volatility: float,
        timestamp: Optional[datetime] = None,
    ) -> str:
        """
        Get the regime key for calibration lookup.

        Args:
            hmm_volatility: HMM volatility state (0-1)
            timestamp: Current time for time-of-day bucket

        Returns:
            Regime key string
        """
        # Volatility bucket
        if hmm_volatility < 0.4:
            vol_bucket = "low_vol"
        elif hmm_volatility > 0.6:
            vol_bucket = "high_vol"
        else:
            vol_bucket = "normal_vol"

        # Time bucket
        if timestamp:
            hour = timestamp.hour
            minute = timestamp.minute
            time_minutes = hour * 60 + minute

            if time_minutes < 630:  # Before 10:30
                time_bucket = "open"
            elif time_minutes > 870:  # After 14:30
                time_bucket = "close"
            else:
                time_bucket = "midday"
        else:
            time_bucket = "unknown"

        return f"{vol_bucket}_{time_bucket}"

    def record_outcome(
        self,
        confidence: float,
        outcome: int,  # 1=win, 0=loss
        hmm_volatility: float,
        timestamp: Optional[datetime] = None,
    ):
        """
        Record a trade outcome for calibration.

        Args:
            confidence: Raw model confidence (0-1)
            outcome: 1 if profitable, 0 if loss
            hmm_volatility: HMM volatility state
            timestamp: Trade entry time
        """
        if not self.enabled:
            return

        # Get regime key
        regime_key = self.get_regime_key(hmm_volatility, timestamp)

        # Add to regime buffer
        buffer = self.buffers[regime_key]
        buffer.confidences.append(confidence)
        buffer.outcomes.append(outcome)

        # Also add to global buffer
        self.global_buffer.confidences.append(confidence)
        self.global_buffer.outcomes.append(outcome)

        # Check if refit needed
        if len(buffer.confidences) >= self.min_samples:
            if len(buffer.confidences) - buffer.last_fit_size >= self.refit_interval:
                self._fit_buffer(buffer, regime_key)

        if len(self.global_buffer.confidences) >= self.min_samples:
            if len(self.global_buffer.confidences) - self.global_buffer.last_fit_size >= self.refit_interval:
                self._fit_buffer(self.global_buffer, "global")

    def _fit_buffer(self, buffer: CalibrationBuffer, name: str):
        """Fit isotonic regression on a buffer."""
        try:
            X = np.array(buffer.confidences).reshape(-1, 1)
            y = np.array(buffer.outcomes)

            # Isotonic regression for calibration
            model = IsotonicRegression(out_of_bounds='clip')
            model.fit(X.ravel(), y)

            buffer.model = model
            buffer.is_fitted = True
            buffer.last_fit_size = len(buffer.confidences)
            buffer.win_rate = np.mean(y)

            logger.debug(f"[REGIME_CAL] Fitted {name}: {len(y)} samples, WR={buffer.win_rate:.1%}")

        except Exception as e:
            logger.debug(f"[REGIME_CAL] Fit failed for {name}: {e}")

    def calibrate(
        self,
        confidence: float,
        hmm_volatility: float,
        timestamp: Optional[datetime] = None,
    ) -> Tuple[float, str]:
        """
        Get calibrated confidence for a regime.

        Args:
            confidence: Raw model confidence
            hmm_volatility: HMM volatility state
            timestamp: Current time

        Returns:
            Tuple of (calibrated_confidence, regime_used)
        """
        if not self.enabled:
            return confidence, "disabled"

        # Get regime key
        regime_key = self.get_regime_key(hmm_volatility, timestamp)
        buffer = self.buffers.get(regime_key)

        # Use regime-specific calibrator if fitted
        if buffer and buffer.is_fitted and buffer.model is not None:
            try:
                calibrated = buffer.model.predict([confidence])[0]
                return float(calibrated), regime_key
            except Exception:
                pass

        # Fall back to global calibrator
        if self.global_buffer.is_fitted and self.global_buffer.model is not None:
            try:
                calibrated = self.global_buffer.model.predict([confidence])[0]
                return float(calibrated), "global"
            except Exception:
                pass

        # No calibration available
        return confidence, "uncalibrated"

    def get_regime_stats(self) -> Dict[str, Dict]:
        """Get statistics for each regime."""
        stats = {}

        for key, buffer in self.buffers.items():
            if len(buffer.outcomes) > 0:
                stats[key] = {
                    'samples': len(buffer.outcomes),
                    'win_rate': np.mean(buffer.outcomes),
                    'is_fitted': buffer.is_fitted,
                    'avg_confidence': np.mean(buffer.confidences),
                }

        # Add global
        if len(self.global_buffer.outcomes) > 0:
            stats['global'] = {
                'samples': len(self.global_buffer.outcomes),
                'win_rate': np.mean(self.global_buffer.outcomes),
                'is_fitted': self.global_buffer.is_fitted,
                'avg_confidence': np.mean(self.global_buffer.confidences),
            }

        return stats


# Singleton
_regime_calibrator = None

def get_regime_calibrator() -> RegimeCalibrator:
    """Get or create the regime calibrator singleton."""
    global _regime_calibrator
    if _regime_calibrator is None:
        _regime_calibrator = RegimeCalibrator()
    return _regime_calibrator
