#!/usr/bin/env python3
"""
Mean Reversion Entry Gate

Based on Carmack-Simons Analysis (Phase 58):
- All features show NEGATIVE correlation with forward returns
- Market MEAN REVERTS at 15-minute timeframe
- Strategy: FADE overbought/oversold conditions

Key Findings:
- bb_position: -0.058 correlation (strongest signal)
- rsi_14: -0.054 correlation
- momentum_15m: -0.050 correlation
- CALLS outperform PUTS (67-80% WR vs 51-62% WR)

Environment Variables:
- MEAN_REVERSION_GATE=1: Enable mean reversion filtering
- MR_ENTRY_THRESHOLD=0.5: Min signal strength to trade (0.0-1.0)
- MR_MIN_RSI_OVERSOLD=30: RSI threshold for oversold
- MR_MAX_RSI_OVERBOUGHT=70: RSI threshold for overbought
- MR_AVOID_LUNCH=1: Skip trades 11:00-14:00
- MR_CALLS_ONLY=0: Only allow CALL trades (best performers)
"""

import os
import logging
import numpy as np
from dataclasses import dataclass
from typing import Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)

# Configuration from environment
MEAN_REVERSION_GATE = os.environ.get('MEAN_REVERSION_GATE', '0') == '1'
MR_ENTRY_THRESHOLD = float(os.environ.get('MR_ENTRY_THRESHOLD', '0.5'))
MR_MIN_RSI_OVERSOLD = float(os.environ.get('MR_MIN_RSI_OVERSOLD', '30'))
MR_MAX_RSI_OVERBOUGHT = float(os.environ.get('MR_MAX_RSI_OVERBOUGHT', '70'))
MR_AVOID_LUNCH = os.environ.get('MR_AVOID_LUNCH', '1') == '1'
MR_CALLS_ONLY = os.environ.get('MR_CALLS_ONLY', '0') == '1'


@dataclass
class MeanReversionResult:
    """Result from mean reversion analysis."""
    should_trade: bool
    recommended_action: str  # 'BUY_CALLS', 'BUY_PUTS', 'HOLD'
    signal_strength: float  # -1 to +1 (negative = oversold, positive = overbought)
    rsi: float
    bb_position: float
    momentum: float
    reason: str
    confidence_boost: float  # Additional confidence if signal is strong


class MeanReversionGate:
    """
    Gate that filters trades based on mean reversion signals.

    Philosophy: Only trade when market is at extremes and likely to revert.
    """

    def __init__(self):
        self.price_history: List[float] = []
        self.enabled = MEAN_REVERSION_GATE
        self.entry_threshold = MR_ENTRY_THRESHOLD
        self.rsi_oversold = MR_MIN_RSI_OVERSOLD
        self.rsi_overbought = MR_MAX_RSI_OVERBOUGHT
        self.avoid_lunch = MR_AVOID_LUNCH
        self.calls_only = MR_CALLS_ONLY

        if self.enabled:
            logger.info(f"📊 Mean Reversion Gate ENABLED")
            logger.info(f"   Entry threshold: {self.entry_threshold}")
            logger.info(f"   RSI oversold: <{self.rsi_oversold}")
            logger.info(f"   RSI overbought: >{self.rsi_overbought}")
            logger.info(f"   Avoid lunch: {self.avoid_lunch}")
            logger.info(f"   Calls only: {self.calls_only}")

    def is_enabled(self) -> bool:
        return self.enabled

    def update_price(self, price: float):
        """Add new price to history."""
        self.price_history.append(price)
        # Keep last 200 prices
        if len(self.price_history) > 200:
            self.price_history = self.price_history[-200:]

    def calculate_rsi(self, period: int = 14) -> float:
        """Calculate RSI indicator."""
        if len(self.price_history) < period + 1:
            return 50.0

        prices = np.array(self.price_history[-period-1:])
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def calculate_bb_position(self, period: int = 20) -> float:
        """Calculate Bollinger Band position (-1 to +1 scale)."""
        if len(self.price_history) < period:
            return 0.0

        recent = np.array(self.price_history[-period:])
        middle = np.mean(recent)
        std = np.std(recent)

        if std == 0:
            return 0.0

        current = self.price_history[-1]
        bb_pos = (current - middle) / (2 * std)
        return np.clip(bb_pos, -2, 2)

    def calculate_momentum(self, period: int = 15) -> float:
        """Calculate price momentum."""
        if len(self.price_history) < period + 1:
            return 0.0

        return (self.price_history[-1] - self.price_history[-period-1]) / self.price_history[-period-1]

    def evaluate(
        self,
        proposed_action: str,
        timestamp: Optional[datetime] = None,
        features: Optional[dict] = None
    ) -> MeanReversionResult:
        """
        Evaluate whether a trade should proceed based on mean reversion signals.

        Args:
            proposed_action: 'BUY_CALLS' or 'BUY_PUTS'
            timestamp: Current timestamp for time-based filtering
            features: Optional feature dict (can contain pre-calculated RSI, etc.)

        Returns:
            MeanReversionResult with recommendation
        """
        if not self.enabled:
            return MeanReversionResult(
                should_trade=True,
                recommended_action=proposed_action,
                signal_strength=0.0,
                rsi=50.0,
                bb_position=0.0,
                momentum=0.0,
                reason="gate_disabled",
                confidence_boost=0.0
            )

        # Check if we have enough data
        if len(self.price_history) < 60:
            return MeanReversionResult(
                should_trade=False,
                recommended_action='HOLD',
                signal_strength=0.0,
                rsi=50.0,
                bb_position=0.0,
                momentum=0.0,
                reason="warming_up",
                confidence_boost=0.0
            )

        # Time-based filter
        if self.avoid_lunch and timestamp:
            hour = timestamp.hour if hasattr(timestamp, 'hour') else 12
            minute = timestamp.minute if hasattr(timestamp, 'minute') else 0

            # Avoid lunch (11:00-14:00)
            if 11 <= hour < 14:
                return MeanReversionResult(
                    should_trade=False,
                    recommended_action='HOLD',
                    signal_strength=0.0,
                    rsi=50.0,
                    bb_position=0.0,
                    momentum=0.0,
                    reason="lunch_hours",
                    confidence_boost=0.0
                )

            # Avoid last 30 minutes
            if hour >= 15 and minute >= 30:
                return MeanReversionResult(
                    should_trade=False,
                    recommended_action='HOLD',
                    signal_strength=0.0,
                    rsi=50.0,
                    bb_position=0.0,
                    momentum=0.0,
                    reason="end_of_day",
                    confidence_boost=0.0
                )

        # Calculate indicators
        rsi = features.get('rsi_14', self.calculate_rsi(14)) if features else self.calculate_rsi(14)
        bb_pos = features.get('bb_position', self.calculate_bb_position(20)) if features else self.calculate_bb_position(20)
        momentum = features.get('momentum_15m', self.calculate_momentum(15)) if features else self.calculate_momentum(15)

        # Composite overbought score
        # Higher = more overbought = expect DOWN = should buy PUTS
        # Lower = more oversold = expect UP = should buy CALLS
        overbought_score = (
            0.40 * (bb_pos / 1.5) +           # BB position (normalized)
            0.35 * ((rsi - 50) / 50) +        # RSI (normalized to -1/+1)
            0.25 * np.clip(momentum * 20, -1, 1)  # Momentum (scaled)
        )

        # Determine recommended action based on mean reversion
        if overbought_score > self.entry_threshold and rsi > self.rsi_overbought:
            # Overbought → expect DOWN → BUY PUTS
            recommended = 'BUY_PUTS'
            signal_reason = f"overbought (RSI={rsi:.0f}, BB={bb_pos:.2f})"
            confidence_boost = min(0.15, abs(overbought_score) * 0.1)
        elif overbought_score < -self.entry_threshold and rsi < self.rsi_oversold:
            # Oversold → expect UP → BUY CALLS
            recommended = 'BUY_CALLS'
            signal_reason = f"oversold (RSI={rsi:.0f}, BB={bb_pos:.2f})"
            confidence_boost = min(0.15, abs(overbought_score) * 0.1)
        else:
            # Not at extreme → don't trade
            return MeanReversionResult(
                should_trade=False,
                recommended_action='HOLD',
                signal_strength=overbought_score,
                rsi=rsi,
                bb_position=bb_pos,
                momentum=momentum,
                reason=f"not_extreme (score={overbought_score:.2f}, RSI={rsi:.0f})",
                confidence_boost=0.0
            )

        # Check if proposed action aligns with mean reversion signal
        if proposed_action != recommended:
            # Proposed action conflicts with mean reversion
            return MeanReversionResult(
                should_trade=False,
                recommended_action=recommended,
                signal_strength=overbought_score,
                rsi=rsi,
                bb_position=bb_pos,
                momentum=momentum,
                reason=f"action_mismatch: proposed {proposed_action}, MR suggests {recommended}",
                confidence_boost=0.0
            )

        # Check calls-only filter
        if self.calls_only and proposed_action == 'BUY_PUTS':
            return MeanReversionResult(
                should_trade=False,
                recommended_action='HOLD',
                signal_strength=overbought_score,
                rsi=rsi,
                bb_position=bb_pos,
                momentum=momentum,
                reason="calls_only_mode",
                confidence_boost=0.0
            )

        # All checks passed - allow trade with confidence boost
        return MeanReversionResult(
            should_trade=True,
            recommended_action=proposed_action,
            signal_strength=overbought_score,
            rsi=rsi,
            bb_position=bb_pos,
            momentum=momentum,
            reason=signal_reason,
            confidence_boost=confidence_boost
        )


# Global instance
_mean_reversion_gate = None


def get_mean_reversion_gate() -> MeanReversionGate:
    """Get or create the global mean reversion gate instance."""
    global _mean_reversion_gate
    if _mean_reversion_gate is None:
        _mean_reversion_gate = MeanReversionGate()
    return _mean_reversion_gate


def check_mean_reversion(
    proposed_action: str,
    price: float,
    timestamp: Optional[datetime] = None,
    features: Optional[dict] = None
) -> MeanReversionResult:
    """
    Convenience function to check mean reversion gate.

    Usage:
        result = check_mean_reversion('BUY_CALLS', current_price, timestamp)
        if result.should_trade:
            # Proceed with trade
            confidence += result.confidence_boost
        else:
            print(f"Trade blocked: {result.reason}")
    """
    gate = get_mean_reversion_gate()
    gate.update_price(price)
    return gate.evaluate(proposed_action, timestamp, features)
