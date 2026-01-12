#!/usr/bin/env python3
"""
Simons-Style Enhanced Predictor

Based on correlation analysis findings:
1. spy_vix_corr: -0.34 (STRONGEST signal - when SPY/VIX correlated, expect DOWN)
2. bb_position: -0.058 (mean reversion)
3. rsi_14: -0.054 (mean reversion)
4. momentum features: all negative (fade momentum)
5. vix_sma_ratio: +0.035 (when VIX elevated, expect UP)
6. is_lunch: -0.028 (lunch hours bearish)

Environment Variables:
- SIMONS_PREDICTOR=1: Enable Simons predictor
- SIMONS_MIN_SIGNAL=0.3: Minimum composite signal to trade
- SIMONS_USE_VIX_CORR=1: Use SPY-VIX correlation signal
"""

import os
import logging
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple
from datetime import datetime
from collections import deque

logger = logging.getLogger(__name__)

# Configuration
SIMONS_PREDICTOR = os.environ.get('SIMONS_PREDICTOR', '0') == '1'
SIMONS_MIN_SIGNAL = float(os.environ.get('SIMONS_MIN_SIGNAL', '0.3'))
SIMONS_USE_VIX_CORR = os.environ.get('SIMONS_USE_VIX_CORR', '1') == '1'


@dataclass
class SimonsPrediction:
    """Result from Simons predictor."""
    action: str  # 'BUY_CALLS', 'BUY_PUTS', 'HOLD'
    confidence: float  # 0.0 to 1.0
    signal_strength: float  # -1 to +1 (negative = bearish, positive = bullish)
    components: dict  # Individual signal contributions
    reason: str


class SimonsPredictor:
    """
    Simons-style predictor combining multiple weak signals.

    Philosophy: Many weak signals (r=0.03-0.05) combined = strong edge
    Key insight: spy_vix_corr has r=-0.34, use it as primary signal
    """

    def __init__(self):
        self.enabled = SIMONS_PREDICTOR
        self.min_signal = SIMONS_MIN_SIGNAL
        self.use_vix_corr = SIMONS_USE_VIX_CORR

        # Price histories
        self.spy_prices: deque = deque(maxlen=200)
        self.vix_prices: deque = deque(maxlen=200)
        self.qqq_prices: deque = deque(maxlen=200)
        self.volumes: deque = deque(maxlen=200)

        # Signal weights from correlation analysis
        self.weights = {
            'spy_vix_corr': 0.40,     # r=-0.34, strongest signal
            'bb_position': 0.15,      # r=-0.058
            'rsi': 0.15,              # r=-0.054
            'momentum': 0.10,         # r=-0.05
            'vix_regime': 0.10,       # r=+0.035
            'time_of_day': 0.05,      # r=-0.028
            'volume': 0.05,           # r=+0.035
        }

        if self.enabled:
            logger.info(f"📊 Simons Predictor ENABLED")
            logger.info(f"   Min signal: {self.min_signal}")
            logger.info(f"   Use VIX corr: {self.use_vix_corr}")

    def is_enabled(self) -> bool:
        return self.enabled

    def update(self, spy_price: float, vix_price: Optional[float] = None,
               qqq_price: Optional[float] = None, volume: Optional[float] = None):
        """Update price histories."""
        self.spy_prices.append(spy_price)
        if vix_price:
            self.vix_prices.append(vix_price)
        if qqq_price:
            self.qqq_prices.append(qqq_price)
        if volume:
            self.volumes.append(volume)

    def _calculate_rsi(self, period: int = 14) -> float:
        """Calculate RSI."""
        if len(self.spy_prices) < period + 1:
            return 50.0

        prices = np.array(list(self.spy_prices))[-period-1:]
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _calculate_bb_position(self, period: int = 20) -> float:
        """Calculate Bollinger Band position (-1 to +1)."""
        if len(self.spy_prices) < period:
            return 0.0

        prices = np.array(list(self.spy_prices))[-period:]
        middle = np.mean(prices)
        std = np.std(prices)

        if std == 0:
            return 0.0

        current = self.spy_prices[-1]
        return np.clip((current - middle) / (2 * std), -2, 2)

    def _calculate_momentum(self, period: int = 15) -> float:
        """Calculate price momentum."""
        if len(self.spy_prices) < period + 1:
            return 0.0

        prices = list(self.spy_prices)
        return (prices[-1] - prices[-period-1]) / prices[-period-1]

    def _calculate_spy_vix_correlation(self, period: int = 20) -> float:
        """
        Calculate rolling SPY-VIX correlation.
        This is the STRONGEST signal (r=-0.34).

        When SPY and VIX are highly correlated (unusual), expect DOWN.
        Normal: SPY up = VIX down (negative correlation)
        Abnormal: SPY and VIX moving together = stress/dislocation
        """
        if len(self.spy_prices) < period or len(self.vix_prices) < period:
            return 0.0

        spy = np.array(list(self.spy_prices)[-period:])
        vix = np.array(list(self.vix_prices)[-period:])

        if len(spy) != len(vix):
            return 0.0

        # Calculate returns
        spy_ret = np.diff(spy) / spy[:-1]
        vix_ret = np.diff(vix) / vix[:-1]

        if len(spy_ret) < 5:
            return 0.0

        # Correlation
        corr = np.corrcoef(spy_ret, vix_ret)[0, 1]
        if np.isnan(corr):
            return 0.0

        return corr

    def _calculate_vix_regime(self) -> float:
        """
        Calculate VIX regime signal.
        vix_sma_ratio has r=+0.035 (when VIX elevated, expect UP - contrarian)
        """
        if len(self.vix_prices) < 20:
            return 0.0

        vix = np.array(list(self.vix_prices))
        current = vix[-1]
        sma = np.mean(vix[-20:])

        # When VIX > SMA, market is fearful → expect bounce (positive signal)
        return (current / sma - 1) * 2  # Scale to roughly -1 to +1

    def _calculate_volume_signal(self) -> float:
        """
        Volume signal - higher volume = market UP (r=+0.035)
        """
        if len(self.volumes) < 20:
            return 0.0

        volumes = np.array(list(self.volumes))
        current = volumes[-1]
        sma = np.mean(volumes[-20:])

        if sma == 0:
            return 0.0

        # High volume ratio = bullish
        return np.clip((current / sma - 1) * 2, -1, 1)

    def predict(self, timestamp: Optional[datetime] = None,
                features: Optional[dict] = None) -> SimonsPrediction:
        """
        Generate Simons-style prediction combining multiple signals.
        """
        if not self.enabled:
            return SimonsPrediction(
                action='HOLD',
                confidence=0.0,
                signal_strength=0.0,
                components={},
                reason='disabled'
            )

        # Need warmup
        if len(self.spy_prices) < 60:
            return SimonsPrediction(
                action='HOLD',
                confidence=0.0,
                signal_strength=0.0,
                components={},
                reason='warming_up'
            )

        components = {}

        # 1. SPY-VIX Correlation (STRONGEST - r=-0.34)
        # High correlation = bearish (expect DOWN)
        if self.use_vix_corr and len(self.vix_prices) >= 20:
            spy_vix_corr = self._calculate_spy_vix_correlation()
            # Correlation of +0.5 (unusual) → strongly bearish
            # Correlation of -0.5 (normal) → neutral to bullish
            components['spy_vix_corr'] = -spy_vix_corr  # Invert: high corr = bearish
        else:
            components['spy_vix_corr'] = 0.0

        # 2. BB Position (r=-0.058)
        # High BB = overbought = bearish
        bb_pos = self._calculate_bb_position()
        components['bb_position'] = -bb_pos / 2  # Normalize to -1/+1 range

        # 3. RSI (r=-0.054)
        # High RSI = overbought = bearish
        rsi = self._calculate_rsi()
        components['rsi'] = -(rsi - 50) / 50  # Normalize: 70 RSI = -0.4 signal

        # 4. Momentum (r=-0.05)
        # Positive momentum = bearish (fade it)
        momentum = self._calculate_momentum()
        components['momentum'] = -np.clip(momentum * 20, -1, 1)  # Scale

        # 5. VIX Regime (r=+0.035)
        # High VIX = bullish (contrarian)
        if len(self.vix_prices) >= 20:
            components['vix_regime'] = self._calculate_vix_regime()
        else:
            components['vix_regime'] = 0.0

        # 6. Time of Day (r=-0.028)
        if timestamp:
            hour = timestamp.hour if hasattr(timestamp, 'hour') else 12
            # Lunch hours (11-14) are bearish
            if 11 <= hour < 14:
                components['time_of_day'] = -0.3  # Slightly bearish
            # End of day (15:30+) bearish
            elif hour >= 15:
                components['time_of_day'] = -0.2
            # Morning (9:30-11) neutral to bullish
            elif hour < 11:
                components['time_of_day'] = 0.1
            else:
                components['time_of_day'] = 0.0
        else:
            components['time_of_day'] = 0.0

        # 7. Volume (r=+0.035)
        components['volume'] = self._calculate_volume_signal()

        # Calculate weighted composite signal
        # Positive = bullish (buy calls), Negative = bearish (buy puts)
        composite = sum(
            self.weights.get(k, 0) * v
            for k, v in components.items()
        )

        # Determine action and confidence
        abs_signal = abs(composite)

        if abs_signal < self.min_signal:
            return SimonsPrediction(
                action='HOLD',
                confidence=abs_signal,
                signal_strength=composite,
                components=components,
                reason=f'signal_weak ({composite:.2f} < {self.min_signal})'
            )

        if composite > 0:
            action = 'BUY_CALLS'
            reason = f'bullish (signal={composite:.2f})'
        else:
            action = 'BUY_PUTS'
            reason = f'bearish (signal={composite:.2f})'

        # Confidence scales with signal strength
        confidence = min(0.8, 0.4 + abs_signal * 0.4)

        return SimonsPrediction(
            action=action,
            confidence=confidence,
            signal_strength=composite,
            components=components,
            reason=reason
        )


# Global instance
_simons_predictor = None


def get_simons_predictor() -> SimonsPredictor:
    """Get or create the global Simons predictor."""
    global _simons_predictor
    if _simons_predictor is None:
        _simons_predictor = SimonsPredictor()
    return _simons_predictor


def simons_predict(spy_price: float,
                   vix_price: Optional[float] = None,
                   qqq_price: Optional[float] = None,
                   volume: Optional[float] = None,
                   timestamp: Optional[datetime] = None,
                   features: Optional[dict] = None) -> SimonsPrediction:
    """
    Convenience function for Simons prediction.

    Usage:
        pred = simons_predict(spy_price=610.50, vix_price=16.5, timestamp=now)
        if pred.action != 'HOLD':
            print(f"Signal: {pred.action} (conf={pred.confidence:.1%})")
            print(f"Components: {pred.components}")
    """
    predictor = get_simons_predictor()
    predictor.update(spy_price, vix_price, qqq_price, volume)
    return predictor.predict(timestamp, features)
