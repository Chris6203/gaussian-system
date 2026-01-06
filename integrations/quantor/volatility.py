"""
Volatility Analytics for Gaussian System
=========================================

Adapted from Quantor-MTFuzz analytics:
- Realized volatility calculator (log returns method)
- IV skew estimator for crash-risk detection
- Volatility Risk Premium (VRP) calculation
- ATR-based dynamic stop/target adjustment

Key Concepts:
- RV = sqrt((252/N) * sum(r_t^2)) where r_t = ln(P_t/P_{t-1})
- VRP = IV - RV (positive = premium for selling options)
- Skew = (IV_put - IV_call) / IV_atm (high = crash risk)
"""

import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class VolatilityMetrics:
    """Container for all volatility-related metrics."""
    realized_vol: float = 0.0       # Annualized realized volatility
    implied_vol: float = 0.0        # Current ATM implied volatility
    vrp: float = 0.0                # Volatility risk premium (IV - RV)
    skew: float = 0.0               # Put-call skew
    atr_pct: float = 0.0            # ATR as % of price
    vol_regime: str = "normal"      # "low", "normal", "high", "extreme"


class VolatilityAnalyzer:
    """
    Comprehensive volatility analysis for gaussian-system.

    Combines realized vol, implied vol, skew, and VRP metrics
    to inform position sizing and risk management.
    """

    def __init__(
        self,
        annualization_factor: float = 252.0,
        low_vol_threshold: float = 0.10,
        high_vol_threshold: float = 0.25,
        extreme_vol_threshold: float = 0.40,
        steep_skew_threshold: float = 0.15
    ):
        """
        Initialize volatility analyzer.

        Args:
            annualization_factor: Trading days per year (252 standard)
            low_vol_threshold: RV below this is "low" regime
            high_vol_threshold: RV above this is "high" regime
            extreme_vol_threshold: RV above this is "extreme" regime
            steep_skew_threshold: Skew above this indicates crash risk
        """
        self.annualization_factor = annualization_factor
        self.low_vol_threshold = low_vol_threshold
        self.high_vol_threshold = high_vol_threshold
        self.extreme_vol_threshold = extreme_vol_threshold
        self.steep_skew_threshold = steep_skew_threshold

    # =========================================================================
    # REALIZED VOLATILITY
    # =========================================================================

    def compute_realized_variance(
        self,
        close_prices: np.ndarray,
        window: int = 20
    ) -> float:
        """
        Compute annualized realized variance over `window` observations.

        Uses sum of squared log returns method:
            RV² = (252/N) * Σ(r_t²)

        Args:
            close_prices: Array of close prices (time-ordered)
            window: Rolling window length

        Returns:
            Annualized realized variance
        """
        if len(close_prices) < window + 1:
            return 0.0

        px = np.asarray(close_prices, dtype=float)
        r = np.log(px[1:] / px[:-1])
        r2 = r[-window:] ** 2

        return (self.annualization_factor / window) * float(np.sum(r2))

    def compute_realized_vol(
        self,
        close_prices: np.ndarray,
        window: int = 20
    ) -> float:
        """
        Compute annualized realized volatility (sqrt of variance).

        Args:
            close_prices: Array of close prices
            window: Rolling window length

        Returns:
            Annualized realized volatility (decimal, e.g., 0.15 = 15%)
        """
        rv2 = self.compute_realized_variance(close_prices, window)
        return float(np.sqrt(max(0, rv2)))

    def compute_realized_vol_intraday(
        self,
        prices: np.ndarray,
        bars_per_day: int = 78  # 5-min bars in trading day
    ) -> float:
        """
        Compute realized vol from intraday bars with proper annualization.

        Args:
            prices: Array of intraday close prices
            bars_per_day: Number of bars in a trading day

        Returns:
            Annualized realized volatility
        """
        if len(prices) < 2:
            return 0.0

        px = np.asarray(prices, dtype=float)
        r = np.log(px[1:] / px[:-1])
        r2 = r ** 2

        # Scale up to daily, then annualize
        daily_var = np.sum(r2) * bars_per_day / len(r2)
        annual_var = daily_var * self.annualization_factor

        return float(np.sqrt(max(0, annual_var)))

    # =========================================================================
    # IMPLIED VOLATILITY SKEW
    # =========================================================================

    def compute_skew(
        self,
        iv_put: float,
        iv_call: float,
        iv_atm: float
    ) -> float:
        """
        Compute IV skew metric.

        Skew = (IV_put - IV_call) / IV_atm

        Positive skew = puts more expensive (crash protection)
        Negative skew = calls more expensive (rare)

        Args:
            iv_put: IV of OTM put
            iv_call: IV of OTM call
            iv_atm: IV of ATM option

        Returns:
            Skew metric (typically 0.05 to 0.20 for SPY)
        """
        if iv_atm <= 0:
            return 0.0
        return (iv_put - iv_call) / iv_atm

    def is_steep_skew(self, skew: float) -> bool:
        """Check if skew indicates elevated crash risk."""
        return skew > self.steep_skew_threshold

    def compute_term_structure_slope(
        self,
        iv_near: float,
        iv_far: float
    ) -> float:
        """
        Compute IV term structure slope.

        Positive = contango (normal)
        Negative = backwardation (fear/hedging demand)

        Args:
            iv_near: Near-term IV
            iv_far: Far-term IV

        Returns:
            Term structure slope
        """
        if iv_near <= 0:
            return 0.0
        return (iv_far - iv_near) / iv_near

    # =========================================================================
    # VOLATILITY RISK PREMIUM
    # =========================================================================

    def compute_vrp(
        self,
        implied_vol: float,
        realized_vol: float
    ) -> float:
        """
        Compute Volatility Risk Premium.

        VRP = IV - RV

        Positive VRP = options overpriced (good for sellers)
        Negative VRP = options underpriced (good for buyers)

        Args:
            implied_vol: Current ATM implied volatility
            realized_vol: Recent realized volatility

        Returns:
            VRP in decimal form
        """
        return implied_vol - realized_vol

    def vrp_percentile(
        self,
        current_vrp: float,
        historical_vrps: np.ndarray
    ) -> float:
        """
        Compute VRP percentile vs historical distribution.

        Args:
            current_vrp: Current VRP value
            historical_vrps: Array of historical VRP values

        Returns:
            Percentile rank (0-100)
        """
        if len(historical_vrps) == 0:
            return 50.0
        return float(np.mean(historical_vrps <= current_vrp) * 100)

    # =========================================================================
    # ATR-BASED ANALYTICS
    # =========================================================================

    def compute_atr(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        period: int = 14
    ) -> float:
        """
        Compute Average True Range.

        Args:
            highs: Array of high prices
            lows: Array of low prices
            closes: Array of close prices
            period: ATR period

        Returns:
            ATR value
        """
        if len(highs) < period + 1:
            return 0.0

        # True Range = max(H-L, |H-Pc|, |L-Pc|)
        h = np.asarray(highs, dtype=float)
        l = np.asarray(lows, dtype=float)
        c = np.asarray(closes, dtype=float)

        tr1 = h[1:] - l[1:]
        tr2 = np.abs(h[1:] - c[:-1])
        tr3 = np.abs(l[1:] - c[:-1])

        tr = np.maximum(tr1, np.maximum(tr2, tr3))

        # Simple average (could use EMA)
        atr = np.mean(tr[-period:])
        return float(atr)

    def compute_atr_pct(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        period: int = 14
    ) -> float:
        """
        Compute ATR as percentage of current price.

        Args:
            highs: Array of high prices
            lows: Array of low prices
            closes: Array of close prices
            period: ATR period

        Returns:
            ATR as percentage (e.g., 0.01 = 1%)
        """
        atr = self.compute_atr(highs, lows, closes, period)
        current_price = closes[-1] if len(closes) > 0 else 0

        if current_price <= 0:
            return 0.0
        return atr / current_price

    def atr_stop_multiplier(
        self,
        atr_pct: float,
        base_multiplier: float = 1.5
    ) -> float:
        """
        Calculate dynamic stop-loss multiplier based on ATR.

        Low volatility → tighter stop (1.0x)
        Medium volatility → standard stop (1.5x)
        High volatility → wider stop (2.0x)

        Args:
            atr_pct: ATR as percentage of price
            base_multiplier: Base stop multiplier

        Returns:
            Adjusted stop multiplier
        """
        if atr_pct is None or np.isnan(atr_pct):
            return base_multiplier

        # Low volatility: tighter stop
        if atr_pct <= 0.005:  # 0.5%
            return max(1.0, base_multiplier - 0.5)

        # High volatility: wider stop
        elif atr_pct >= 0.02:  # 2.0%
            return min(2.5, base_multiplier + 0.5)

        # Medium volatility: interpolate
        else:
            adjustment = ((atr_pct - 0.005) / 0.015) * 0.5
            return base_multiplier + adjustment

    # =========================================================================
    # REGIME CLASSIFICATION
    # =========================================================================

    def classify_vol_regime(self, realized_vol: float) -> str:
        """
        Classify volatility regime.

        Args:
            realized_vol: Current realized volatility

        Returns:
            Regime string: "low", "normal", "high", "extreme"
        """
        if realized_vol >= self.extreme_vol_threshold:
            return "extreme"
        elif realized_vol >= self.high_vol_threshold:
            return "high"
        elif realized_vol <= self.low_vol_threshold:
            return "low"
        else:
            return "normal"

    # =========================================================================
    # COMPREHENSIVE ANALYSIS
    # =========================================================================

    def analyze(
        self,
        close_prices: np.ndarray,
        highs: np.ndarray = None,
        lows: np.ndarray = None,
        implied_vol: float = None,
        iv_put: float = None,
        iv_call: float = None,
        iv_atm: float = None,
        rv_window: int = 20,
        atr_period: int = 14
    ) -> VolatilityMetrics:
        """
        Compute comprehensive volatility metrics.

        Args:
            close_prices: Array of close prices
            highs: Array of high prices (optional)
            lows: Array of low prices (optional)
            implied_vol: Current ATM IV (optional)
            iv_put: OTM put IV for skew (optional)
            iv_call: OTM call IV for skew (optional)
            iv_atm: ATM IV for skew (optional)
            rv_window: Window for realized vol
            atr_period: Period for ATR

        Returns:
            VolatilityMetrics dataclass with all computed values
        """
        metrics = VolatilityMetrics()

        # Realized volatility
        metrics.realized_vol = self.compute_realized_vol(close_prices, rv_window)

        # Vol regime
        metrics.vol_regime = self.classify_vol_regime(metrics.realized_vol)

        # Implied volatility
        if implied_vol is not None:
            metrics.implied_vol = implied_vol
            metrics.vrp = self.compute_vrp(implied_vol, metrics.realized_vol)

        # Skew
        if iv_put is not None and iv_call is not None and iv_atm is not None:
            metrics.skew = self.compute_skew(iv_put, iv_call, iv_atm)

        # ATR
        if highs is not None and lows is not None:
            metrics.atr_pct = self.compute_atr_pct(highs, lows, close_prices, atr_period)

        return metrics

    def get_position_adjustments(
        self,
        metrics: VolatilityMetrics
    ) -> Dict[str, float]:
        """
        Get recommended position adjustments based on volatility metrics.

        Returns dict with:
        - size_multiplier: Scale position size
        - stop_multiplier: Scale stop loss distance
        - profit_multiplier: Scale take profit distance
        """
        adjustments = {
            'size_multiplier': 1.0,
            'stop_multiplier': 1.0,
            'profit_multiplier': 1.0
        }

        # Adjust for vol regime
        if metrics.vol_regime == "extreme":
            adjustments['size_multiplier'] = 0.25
            adjustments['stop_multiplier'] = 2.0
            adjustments['profit_multiplier'] = 1.5
        elif metrics.vol_regime == "high":
            adjustments['size_multiplier'] = 0.5
            adjustments['stop_multiplier'] = 1.5
            adjustments['profit_multiplier'] = 1.3
        elif metrics.vol_regime == "low":
            adjustments['size_multiplier'] = 1.2  # Slightly larger in calm markets
            adjustments['stop_multiplier'] = 0.8
            adjustments['profit_multiplier'] = 0.9

        # Penalize steep skew (crash risk)
        if self.is_steep_skew(metrics.skew):
            adjustments['size_multiplier'] *= 0.7

        # Adjust for negative VRP (options expensive)
        if metrics.vrp < -0.05:
            adjustments['size_multiplier'] *= 0.8

        return adjustments
