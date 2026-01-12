#!/usr/bin/env python3
"""
Momentum Exhaustion Detector

Carmack Philosophy: Don't catch falling knives - wait for confirmation.
Simons Philosophy: Multiple weak signals of exhaustion = strong reversal signal.

Key Insight: Mean reversion works AFTER momentum exhausts, not during.

Exhaustion Signals:
1. RSI Divergence - Price new high/low, RSI doesn't confirm
2. Volume Exhaustion - Move on declining volume
3. Momentum Slowdown - Rate of change decreasing
4. Volatility Contraction - Squeeze after expansion
5. Candle Exhaustion - Small bodies after big moves

Environment Variables:
- MOMENTUM_EXHAUSTION=1: Enable momentum exhaustion detection
- MOM_EX_MIN_SIGNALS=2: Minimum exhaustion signals to confirm
- MOM_EX_RSI_DIVERGENCE=1: Check RSI divergence
- MOM_EX_VOLUME_DECLINE=1: Check volume decline
- MOM_EX_MOMENTUM_SLOW=1: Check momentum slowdown
"""

import os
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from collections import deque

logger = logging.getLogger(__name__)

# Configuration
MOMENTUM_EXHAUSTION = os.environ.get('MOMENTUM_EXHAUSTION', '0') == '1'
MOM_EX_MIN_SIGNALS = int(os.environ.get('MOM_EX_MIN_SIGNALS', '2'))
MOM_EX_RSI_DIVERGENCE = os.environ.get('MOM_EX_RSI_DIVERGENCE', '1') == '1'
MOM_EX_VOLUME_DECLINE = os.environ.get('MOM_EX_VOLUME_DECLINE', '1') == '1'
MOM_EX_MOMENTUM_SLOW = os.environ.get('MOM_EX_MOMENTUM_SLOW', '1') == '1'


@dataclass
class ExhaustionSignal:
    """Individual exhaustion signal."""
    name: str
    detected: bool
    strength: float  # 0.0 to 1.0
    details: str


@dataclass
class ExhaustionResult:
    """Result of momentum exhaustion analysis."""
    is_exhausted: bool  # True if momentum is exhausted (safe to enter)
    direction: str  # 'BULLISH_EXHAUSTION' (buy puts) or 'BEARISH_EXHAUSTION' (buy calls)
    exhaustion_score: float  # 0.0 to 1.0 (higher = more exhausted)
    signals: List[ExhaustionSignal] = field(default_factory=list)
    should_wait: bool = False  # True if momentum still strong - wait
    reason: str = ""


class MomentumExhaustionDetector:
    """
    Detects when price momentum is exhausted and reversal is likely.

    Don't catch falling knives - wait for exhaustion signals.
    """

    def __init__(self):
        self.enabled = MOMENTUM_EXHAUSTION
        self.min_signals = MOM_EX_MIN_SIGNALS
        self.check_rsi_divergence = MOM_EX_RSI_DIVERGENCE
        self.check_volume_decline = MOM_EX_VOLUME_DECLINE
        self.check_momentum_slow = MOM_EX_MOMENTUM_SLOW

        # Price and indicator history
        self.prices: deque = deque(maxlen=100)
        self.highs: deque = deque(maxlen=100)
        self.lows: deque = deque(maxlen=100)
        self.volumes: deque = deque(maxlen=100)
        self.rsi_history: deque = deque(maxlen=50)

        if self.enabled:
            logger.info(f"🔍 Momentum Exhaustion Detector ENABLED")
            logger.info(f"   Min signals: {self.min_signals}")
            logger.info(f"   RSI divergence: {self.check_rsi_divergence}")
            logger.info(f"   Volume decline: {self.check_volume_decline}")
            logger.info(f"   Momentum slow: {self.check_momentum_slow}")

    def update(self, price: float, high: Optional[float] = None,
               low: Optional[float] = None, volume: Optional[float] = None):
        """Update price history."""
        self.prices.append(price)
        self.highs.append(high if high else price)
        self.lows.append(low if low else price)
        if volume:
            self.volumes.append(volume)

        # Calculate and store RSI
        if len(self.prices) >= 15:
            rsi = self._calculate_rsi(14)
            self.rsi_history.append(rsi)

    def _calculate_rsi(self, period: int = 14) -> float:
        """Calculate RSI."""
        if len(self.prices) < period + 1:
            return 50.0

        prices = np.array(list(self.prices)[-period-1:])
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _detect_rsi_divergence(self) -> ExhaustionSignal:
        """
        Detect RSI divergence - price makes new extreme but RSI doesn't.

        Bullish divergence: Price lower low, RSI higher low → buy signal
        Bearish divergence: Price higher high, RSI lower high → sell signal
        """
        if len(self.prices) < 20 or len(self.rsi_history) < 10:
            return ExhaustionSignal("rsi_divergence", False, 0.0, "insufficient_data")

        prices = np.array(list(self.prices)[-20:])
        rsis = np.array(list(self.rsi_history)[-10:])

        # Find recent price extremes
        price_max_idx = np.argmax(prices[-10:])
        price_min_idx = np.argmin(prices[-10:])

        # Check for bearish divergence (price higher high, RSI lower high)
        if price_max_idx >= 7:  # Recent high
            recent_price_high = prices[-10:].max()
            older_price_high = prices[-20:-10].max()

            if recent_price_high > older_price_high:
                # Price made higher high
                recent_rsi_max = rsis[-5:].max() if len(rsis) >= 5 else rsis.max()
                older_rsi_max = rsis[:-5].max() if len(rsis) >= 5 else rsis.max()

                if recent_rsi_max < older_rsi_max - 3:  # RSI didn't confirm
                    strength = min(1.0, (older_rsi_max - recent_rsi_max) / 10)
                    return ExhaustionSignal(
                        "rsi_divergence", True, strength,
                        f"BEARISH: Price HH but RSI LH ({recent_rsi_max:.0f} < {older_rsi_max:.0f})"
                    )

        # Check for bullish divergence (price lower low, RSI higher low)
        if price_min_idx >= 7:  # Recent low
            recent_price_low = prices[-10:].min()
            older_price_low = prices[-20:-10].min()

            if recent_price_low < older_price_low:
                # Price made lower low
                recent_rsi_min = rsis[-5:].min() if len(rsis) >= 5 else rsis.min()
                older_rsi_min = rsis[:-5].min() if len(rsis) >= 5 else rsis.min()

                if recent_rsi_min > older_rsi_min + 3:  # RSI didn't confirm
                    strength = min(1.0, (recent_rsi_min - older_rsi_min) / 10)
                    return ExhaustionSignal(
                        "rsi_divergence", True, strength,
                        f"BULLISH: Price LL but RSI HL ({recent_rsi_min:.0f} > {older_rsi_min:.0f})"
                    )

        return ExhaustionSignal("rsi_divergence", False, 0.0, "no_divergence")

    def _detect_volume_exhaustion(self) -> ExhaustionSignal:
        """
        Detect volume exhaustion - price moving on declining volume.

        Strong moves should have volume confirmation.
        Moves on declining volume = exhaustion.
        """
        if len(self.volumes) < 20:
            return ExhaustionSignal("volume_exhaustion", False, 0.0, "insufficient_data")

        volumes = np.array(list(self.volumes)[-20:])
        prices = np.array(list(self.prices)[-20:])

        # Calculate price change over period
        price_change = (prices[-1] - prices[0]) / prices[0]

        # Calculate volume trend
        recent_vol = np.mean(volumes[-5:])
        older_vol = np.mean(volumes[-15:-5])

        if older_vol == 0:
            return ExhaustionSignal("volume_exhaustion", False, 0.0, "no_volume_data")

        vol_change = (recent_vol - older_vol) / older_vol

        # Exhaustion: Price moving but volume declining
        if abs(price_change) > 0.002:  # Significant price move
            if vol_change < -0.15:  # Volume declining 15%+
                strength = min(1.0, abs(vol_change) * 2)
                direction = "UP" if price_change > 0 else "DOWN"
                return ExhaustionSignal(
                    "volume_exhaustion", True, strength,
                    f"Price {direction} {abs(price_change)*100:.2f}% on declining volume ({vol_change*100:.0f}%)"
                )

        return ExhaustionSignal("volume_exhaustion", False, 0.0, "volume_confirmed")

    def _detect_momentum_slowdown(self) -> ExhaustionSignal:
        """
        Detect momentum slowdown - rate of change decreasing.

        When momentum slows, reversal is near.
        """
        if len(self.prices) < 30:
            return ExhaustionSignal("momentum_slowdown", False, 0.0, "insufficient_data")

        prices = np.array(list(self.prices)[-30:])

        # Calculate momentum at different periods
        mom_5 = (prices[-1] - prices[-6]) / prices[-6]  # Recent 5-bar momentum
        mom_10 = (prices[-6] - prices[-16]) / prices[-16]  # Previous 10-bar momentum
        mom_15 = (prices[-16] - prices[-26]) / prices[-26] if len(prices) >= 26 else 0

        # Detect momentum slowdown
        # Uptrend exhaustion: momentum was positive but slowing
        if mom_15 > 0.002 and mom_10 > 0.001:  # Was trending up
            if mom_5 < mom_10 * 0.5:  # Recent momentum < 50% of previous
                strength = min(1.0, (mom_10 - mom_5) / (abs(mom_10) + 0.001) * 2)
                return ExhaustionSignal(
                    "momentum_slowdown", True, strength,
                    f"UPTREND exhausting: mom slowed from {mom_10*100:.2f}% to {mom_5*100:.2f}%"
                )

        # Downtrend exhaustion: momentum was negative but slowing
        if mom_15 < -0.002 and mom_10 < -0.001:  # Was trending down
            if mom_5 > mom_10 * 0.5:  # Recent momentum > 50% of previous (less negative)
                strength = min(1.0, (mom_5 - mom_10) / (abs(mom_10) + 0.001) * 2)
                return ExhaustionSignal(
                    "momentum_slowdown", True, strength,
                    f"DOWNTREND exhausting: mom slowed from {mom_10*100:.2f}% to {mom_5*100:.2f}%"
                )

        return ExhaustionSignal("momentum_slowdown", False, 0.0, "momentum_strong")

    def _detect_volatility_contraction(self) -> ExhaustionSignal:
        """
        Detect volatility contraction after expansion.

        After big moves, volatility contracts before reversal.
        """
        if len(self.prices) < 30:
            return ExhaustionSignal("volatility_contraction", False, 0.0, "insufficient_data")

        prices = np.array(list(self.prices)[-30:])

        # Calculate rolling volatility
        recent_vol = np.std(prices[-10:]) / np.mean(prices[-10:])
        older_vol = np.std(prices[-20:-10]) / np.mean(prices[-20:-10])

        if older_vol == 0:
            return ExhaustionSignal("volatility_contraction", False, 0.0, "no_volatility")

        vol_change = (recent_vol - older_vol) / older_vol

        # Contraction after expansion suggests exhaustion
        if vol_change < -0.20:  # Volatility contracted 20%+
            strength = min(1.0, abs(vol_change))
            return ExhaustionSignal(
                "volatility_contraction", True, strength,
                f"Volatility contracted {abs(vol_change)*100:.0f}% (squeeze forming)"
            )

        return ExhaustionSignal("volatility_contraction", False, 0.0, "volatility_normal")

    def _detect_price_rejection(self) -> ExhaustionSignal:
        """
        Detect price rejection - long wicks showing rejection of extremes.
        """
        if len(self.highs) < 10 or len(self.lows) < 10:
            return ExhaustionSignal("price_rejection", False, 0.0, "insufficient_data")

        prices = np.array(list(self.prices)[-10:])
        highs = np.array(list(self.highs)[-10:])
        lows = np.array(list(self.lows)[-10:])

        # Calculate candle bodies and wicks
        recent_price = prices[-1]
        recent_high = highs[-1]
        recent_low = lows[-1]
        prev_price = prices[-2]

        body = abs(recent_price - prev_price)
        upper_wick = recent_high - max(recent_price, prev_price)
        lower_wick = min(recent_price, prev_price) - recent_low
        total_range = recent_high - recent_low

        if total_range == 0:
            return ExhaustionSignal("price_rejection", False, 0.0, "no_range")

        # Upper rejection (bearish)
        if upper_wick > body * 2 and upper_wick > total_range * 0.6:
            strength = min(1.0, upper_wick / total_range)
            return ExhaustionSignal(
                "price_rejection", True, strength,
                f"Upper wick rejection (bearish exhaustion)"
            )

        # Lower rejection (bullish)
        if lower_wick > body * 2 and lower_wick > total_range * 0.6:
            strength = min(1.0, lower_wick / total_range)
            return ExhaustionSignal(
                "price_rejection", True, strength,
                f"Lower wick rejection (bullish exhaustion)"
            )

        return ExhaustionSignal("price_rejection", False, 0.0, "no_rejection")

    def analyze(self, proposed_action: str = 'UNKNOWN') -> ExhaustionResult:
        """
        Analyze momentum exhaustion for trade entry.

        Args:
            proposed_action: 'BUY_CALLS' or 'BUY_PUTS'

        Returns:
            ExhaustionResult with exhaustion signals and recommendation
        """
        if not self.enabled:
            return ExhaustionResult(
                is_exhausted=True,  # Allow trades when disabled
                direction='UNKNOWN',
                exhaustion_score=0.5,
                should_wait=False,
                reason='detector_disabled'
            )

        if len(self.prices) < 30:
            return ExhaustionResult(
                is_exhausted=False,
                direction='UNKNOWN',
                exhaustion_score=0.0,
                should_wait=True,
                reason='warming_up'
            )

        signals = []

        # Collect exhaustion signals
        if self.check_rsi_divergence:
            signals.append(self._detect_rsi_divergence())

        if self.check_volume_decline:
            signals.append(self._detect_volume_exhaustion())

        if self.check_momentum_slow:
            signals.append(self._detect_momentum_slowdown())

        signals.append(self._detect_volatility_contraction())
        signals.append(self._detect_price_rejection())

        # Count positive signals
        positive_signals = [s for s in signals if s.detected]
        exhaustion_score = sum(s.strength for s in positive_signals) / max(len(signals), 1)

        # Determine direction
        direction = 'UNKNOWN'
        for sig in positive_signals:
            if 'BEARISH' in sig.details or 'UPTREND exhausting' in sig.details or 'Upper' in sig.details:
                direction = 'BULLISH_EXHAUSTION'  # Uptrend exhausted → buy puts
                break
            elif 'BULLISH' in sig.details or 'DOWNTREND exhausting' in sig.details or 'Lower' in sig.details:
                direction = 'BEARISH_EXHAUSTION'  # Downtrend exhausted → buy calls
                break

        # Decision logic
        is_exhausted = len(positive_signals) >= self.min_signals

        # Check if action aligns with exhaustion
        should_wait = False
        if proposed_action == 'BUY_CALLS':
            # For calls, we want downtrend exhaustion (BEARISH_EXHAUSTION)
            if direction == 'BULLISH_EXHAUSTION':
                should_wait = True  # Uptrend exhausting - don't buy calls!
            elif not is_exhausted and len(self.prices) > 0:
                # Check if we're in a downtrend that hasn't exhausted
                mom = (self.prices[-1] - self.prices[-10]) / self.prices[-10] if len(self.prices) >= 10 else 0
                if mom < -0.003:  # In downtrend
                    should_wait = True  # Wait for exhaustion

        elif proposed_action == 'BUY_PUTS':
            # For puts, we want uptrend exhaustion (BULLISH_EXHAUSTION)
            if direction == 'BEARISH_EXHAUSTION':
                should_wait = True  # Downtrend exhausting - don't buy puts!
            elif not is_exhausted and len(self.prices) > 0:
                # Check if we're in an uptrend that hasn't exhausted
                mom = (self.prices[-1] - self.prices[-10]) / self.prices[-10] if len(self.prices) >= 10 else 0
                if mom > 0.003:  # In uptrend
                    should_wait = True  # Wait for exhaustion

        reason_parts = [s.details for s in positive_signals]
        reason = f"{len(positive_signals)}/{len(signals)} signals: " + "; ".join(reason_parts) if reason_parts else "no exhaustion signals"

        return ExhaustionResult(
            is_exhausted=is_exhausted,
            direction=direction,
            exhaustion_score=exhaustion_score,
            signals=signals,
            should_wait=should_wait,
            reason=reason
        )


# Global instance
_exhaustion_detector = None


def get_exhaustion_detector() -> MomentumExhaustionDetector:
    """Get or create the global exhaustion detector."""
    global _exhaustion_detector
    if _exhaustion_detector is None:
        _exhaustion_detector = MomentumExhaustionDetector()
    return _exhaustion_detector


def check_momentum_exhaustion(
    proposed_action: str,
    price: float,
    high: Optional[float] = None,
    low: Optional[float] = None,
    volume: Optional[float] = None
) -> ExhaustionResult:
    """
    Convenience function to check momentum exhaustion.

    Usage:
        result = check_momentum_exhaustion('BUY_CALLS', 610.50, volume=1000000)

        if result.should_wait:
            print(f"Wait - momentum still strong: {result.reason}")
        elif result.is_exhausted:
            print(f"Good entry - exhaustion detected: {result.reason}")
    """
    detector = get_exhaustion_detector()
    detector.update(price, high, low, volume)
    return detector.analyze(proposed_action)
