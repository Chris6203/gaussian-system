"""
Market Regime Filter for Gaussian System
=========================================

Adapted from Quantor-MTFuzz regime_filter.py
Original: 5-regime classification for Iron Condor suitability
Modified: Extended regime detection for directional options trading

Key Features:
- VIX-based crash detection (blocks all trading)
- ADX trend strength classification
- Price vs SMA trend direction
- Liquidity gate filtering
"""

from enum import Enum
from typing import Dict, Optional, Tuple
import numpy as np


class MarketRegime(Enum):
    """
    Market regime classification.

    For directional trading (gaussian-system):
    - BULL_TREND: Favor CALL entries
    - BEAR_TREND: Favor PUT entries
    - LOW_VOL_RANGE: Cautious, smaller positions
    - HIGH_VOL_RANGE: Cautious, tighter stops
    - CRASH_MODE: NO TRADING
    """
    LOW_VOL_RANGE = "Low Volatility Range"
    HIGH_VOL_RANGE = "High Volatility Range"
    BULL_TREND = "Bull Trend"
    BEAR_TREND = "Bear Trend"
    CRASH_MODE = "Crash Mode"


class RegimeFilter:
    """
    Market regime classification and filtering for gaussian-system.

    Combines VIX levels, ADX trend strength, and price vs SMA
    to determine the current market regime.
    """

    def __init__(
        self,
        vix_crash_threshold: float = 35.0,
        vix_high_vol_threshold: float = 20.0,
        adx_trend_threshold: float = 25.0,
        min_volume_threshold: int = 100,
        max_range_pct: float = 0.02
    ):
        """
        Initialize regime filter.

        Args:
            vix_crash_threshold: VIX level above which trading is blocked
            vix_high_vol_threshold: VIX level above which regime is HIGH_VOL
            adx_trend_threshold: ADX above which market is considered trending
            min_volume_threshold: Minimum volume for liquidity check
            max_range_pct: Maximum price range % for liquidity check
        """
        self.vix_crash_threshold = vix_crash_threshold
        self.vix_high_vol_threshold = vix_high_vol_threshold
        self.adx_trend_threshold = adx_trend_threshold
        self.min_volume_threshold = min_volume_threshold
        self.max_range_pct = max_range_pct

    def classify_regime(
        self,
        vix: float,
        adx: float = None,
        close: float = None,
        sma_200: float = None
    ) -> MarketRegime:
        """
        Classify current market regime.

        Args:
            vix: Current VIX level
            adx: Average Directional Index (trend strength)
            close: Current price
            sma_200: 200-period SMA for trend direction

        Returns:
            MarketRegime enum value
        """
        # 1. Crash Mode - VIX explosion
        if vix is None:
            return MarketRegime.LOW_VOL_RANGE  # Default if no VIX data

        if vix > self.vix_crash_threshold:
            return MarketRegime.CRASH_MODE

        # 2. Check for trending vs ranging
        is_trending = (adx is not None and adx > self.adx_trend_threshold)

        # 3. Determine trend direction (if trending)
        if is_trending:
            if close is not None and sma_200 is not None:
                is_bullish = close > sma_200
            else:
                # If no SMA data, assume neutral trending
                is_bullish = None

            if is_bullish is True:
                return MarketRegime.BULL_TREND
            elif is_bullish is False:
                return MarketRegime.BEAR_TREND
            else:
                # Trending but direction unclear - treat as high vol range
                return MarketRegime.HIGH_VOL_RANGE

        # 4. Ranging market - check volatility level
        if vix > self.vix_high_vol_threshold:
            return MarketRegime.HIGH_VOL_RANGE
        else:
            return MarketRegime.LOW_VOL_RANGE

    def check_liquidity(
        self,
        volumes: list,
        highs: list = None,
        lows: list = None,
        current_close: float = None
    ) -> Tuple[bool, str]:
        """
        Check if market has sufficient liquidity for trading.

        Args:
            volumes: List of recent volume values (e.g., last 5 bars)
            highs: List of recent high prices
            lows: List of recent low prices
            current_close: Current close price

        Returns:
            Tuple of (passes_liquidity_check, reason)
        """
        if not volumes or len(volumes) < 3:
            return False, "Insufficient volume data"

        # Check minimum volume
        avg_volume = sum(volumes) / len(volumes)
        if avg_volume < self.min_volume_threshold:
            return False, f"Low volume: {avg_volume:.0f} < {self.min_volume_threshold}"

        # Check price range stability (if we have high/low data)
        if highs and lows and current_close and len(highs) >= 3:
            avg_range = sum(h - l for h, l in zip(highs, lows)) / len(highs)
            range_pct = avg_range / current_close if current_close > 0 else 0

            if range_pct > self.max_range_pct:
                return False, f"High volatility: {range_pct:.2%} > {self.max_range_pct:.2%}"

        return True, "OK"

    def is_trading_allowed(
        self,
        regime: MarketRegime,
        direction: str = None
    ) -> Tuple[bool, str]:
        """
        Check if trading is allowed for the given regime and direction.

        Args:
            regime: Current market regime
            direction: 'CALL' or 'PUT' (optional)

        Returns:
            Tuple of (trading_allowed, reason)
        """
        # Crash mode - no trading
        if regime == MarketRegime.CRASH_MODE:
            return False, "CRASH_MODE - All trading blocked"

        # Direction alignment checks (optional enhancement)
        if direction:
            if direction == 'CALL' and regime == MarketRegime.BEAR_TREND:
                return False, "CALL blocked in BEAR_TREND regime"
            if direction == 'PUT' and regime == MarketRegime.BULL_TREND:
                return False, "PUT blocked in BULL_TREND regime"

        return True, f"Trading allowed in {regime.value}"

    def get_regime_adjustments(
        self,
        regime: MarketRegime
    ) -> Dict[str, float]:
        """
        Get recommended parameter adjustments for current regime.

        Returns dict with adjustment factors for:
        - position_size: Multiplier for position sizing
        - stop_loss: Multiplier for stop loss width
        - take_profit: Multiplier for take profit target
        - confidence_threshold: Minimum confidence required
        """
        adjustments = {
            'position_size': 1.0,
            'stop_loss': 1.0,
            'take_profit': 1.0,
            'confidence_threshold': 0.5
        }

        if regime == MarketRegime.CRASH_MODE:
            # Should not trade, but if checked:
            adjustments['position_size'] = 0.0
            adjustments['confidence_threshold'] = 1.0  # Impossible

        elif regime == MarketRegime.HIGH_VOL_RANGE:
            # Reduce size, widen stops, require higher confidence
            adjustments['position_size'] = 0.5
            adjustments['stop_loss'] = 1.5
            adjustments['take_profit'] = 1.3
            adjustments['confidence_threshold'] = 0.65

        elif regime == MarketRegime.LOW_VOL_RANGE:
            # Cautious - tighter stops, modest targets
            adjustments['position_size'] = 0.75
            adjustments['stop_loss'] = 0.8
            adjustments['take_profit'] = 0.9
            adjustments['confidence_threshold'] = 0.55

        elif regime == MarketRegime.BULL_TREND:
            # Full size for calls, wider targets
            adjustments['position_size'] = 1.0
            adjustments['stop_loss'] = 1.0
            adjustments['take_profit'] = 1.2
            adjustments['confidence_threshold'] = 0.5

        elif regime == MarketRegime.BEAR_TREND:
            # Full size for puts, wider targets
            adjustments['position_size'] = 1.0
            adjustments['stop_loss'] = 1.0
            adjustments['take_profit'] = 1.2
            adjustments['confidence_threshold'] = 0.5

        return adjustments

    def get_direction_bias(
        self,
        regime: MarketRegime
    ) -> float:
        """
        Get directional bias for the regime.

        Returns:
            Float from -1 (bearish) to +1 (bullish), 0 = neutral
        """
        if regime == MarketRegime.BULL_TREND:
            return 0.7
        elif regime == MarketRegime.BEAR_TREND:
            return -0.7
        elif regime == MarketRegime.HIGH_VOL_RANGE:
            return 0.0  # Neutral, could go either way
        elif regime == MarketRegime.LOW_VOL_RANGE:
            return 0.0  # Neutral, range-bound
        else:  # CRASH_MODE
            return -1.0  # Extremely bearish bias
