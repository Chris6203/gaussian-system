"""
Greeks-Aware Exit System - Dynamic stop/TP based on option Greeks and volatility.

Key insight: Static -8%/+12% stops don't account for option characteristics.
A high-IV near-expiry option moves differently than a low-IV far-dated one.

Adjusts exits based on:
- Delta: Higher delta = tighter stops (moves more with underlying)
- Theta: Higher theta decay rate = tighter time limit
- IV percentile: High IV = wider stops (more volatile price swings)
- Time to expiry: Shorter = tighter stops (gamma increases)
"""

import os
import logging
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class DynamicExitLevels:
    """Calculated exit levels for a position."""
    stop_loss_pct: float
    take_profit_pct: float
    max_hold_minutes: int
    trailing_activation_pct: float
    trailing_distance_pct: float
    reason: str


class GreeksAwareExitManager:
    """
    Calculates dynamic exit levels based on option Greeks.

    Philosophy:
    - High delta options move faster -> tighter stops to lock gains/cut losses
    - High IV options swing more -> wider stops to avoid noise exits
    - Near expiry -> tighter max hold (gamma risk, theta decay)
    - Use uncertainty for volatility-adjusted stops, not entry selection
    """

    def __init__(
        self,
        base_stop_loss: float = 8.0,
        base_take_profit: float = 12.0,
        base_max_hold: int = 45,
        base_trailing_activation: float = 4.0,
        base_trailing_distance: float = 2.0,
        iv_adjustment_factor: float = 0.5,  # How much IV affects stops
        delta_adjustment_factor: float = 0.3,  # How much delta affects stops
        enabled: bool = True,
    ):
        self.base_stop_loss = float(os.environ.get('BASE_STOP_LOSS', str(base_stop_loss)))
        self.base_take_profit = float(os.environ.get('BASE_TAKE_PROFIT', str(base_take_profit)))
        self.base_max_hold = int(os.environ.get('BASE_MAX_HOLD', str(base_max_hold)))
        self.base_trailing_activation = float(os.environ.get('BASE_TRAILING_ACTIVATION', str(base_trailing_activation)))
        self.base_trailing_distance = float(os.environ.get('BASE_TRAILING_DISTANCE', str(base_trailing_distance)))

        self.iv_factor = float(os.environ.get('GREEKS_IV_FACTOR', str(iv_adjustment_factor)))
        self.delta_factor = float(os.environ.get('GREEKS_DELTA_FACTOR', str(delta_adjustment_factor)))
        self.enabled = os.environ.get('GREEKS_AWARE_EXITS', '1') == '1' if enabled else False

        logger.info(f"[GREEKS_EXIT] Initialized: enabled={self.enabled}, "
                   f"base_sl={self.base_stop_loss}%, base_tp={self.base_take_profit}%")

    def calculate_exit_levels(
        self,
        delta: Optional[float] = None,
        gamma: Optional[float] = None,
        theta: Optional[float] = None,
        iv: Optional[float] = None,
        iv_percentile: Optional[float] = None,
        minutes_to_expiry: Optional[float] = None,
        mc_uncertainty: Optional[float] = None,
        vix_level: Optional[float] = None,
    ) -> DynamicExitLevels:
        """
        Calculate dynamic exit levels based on Greeks and market conditions.

        Args:
            delta: Option delta (0-1 for calls, 0 to -1 for puts)
            gamma: Option gamma
            theta: Option theta (daily decay)
            iv: Implied volatility (decimal, e.g., 0.25 for 25%)
            iv_percentile: IV rank/percentile (0-100)
            minutes_to_expiry: Minutes until expiration
            mc_uncertainty: Monte Carlo uncertainty from predictor
            vix_level: Current VIX level

        Returns:
            DynamicExitLevels with adjusted stop/TP levels
        """
        if not self.enabled:
            return DynamicExitLevels(
                stop_loss_pct=self.base_stop_loss,
                take_profit_pct=self.base_take_profit,
                max_hold_minutes=self.base_max_hold,
                trailing_activation_pct=self.base_trailing_activation,
                trailing_distance_pct=self.base_trailing_distance,
                reason="disabled"
            )

        adjustments = []

        # Start with base levels
        stop_loss = self.base_stop_loss
        take_profit = self.base_take_profit
        max_hold = self.base_max_hold
        trailing_act = self.base_trailing_activation
        trailing_dist = self.base_trailing_distance

        # 1. Delta adjustment
        # Higher delta = tighter stops (option moves more with underlying)
        if delta is not None:
            abs_delta = abs(delta)
            if abs_delta > 0.7:
                # Deep ITM - tighter stops
                delta_mult = 0.8
                adjustments.append("high_delta")
            elif abs_delta > 0.5:
                # ATM - normal
                delta_mult = 1.0
            else:
                # OTM - wider stops (options move less)
                delta_mult = 1.2
                adjustments.append("low_delta")

            stop_loss *= delta_mult
            take_profit *= (2 - delta_mult)  # Inverse for TP

        # 2. IV/VIX adjustment
        # Higher IV = wider stops (more price swings)
        if iv_percentile is not None or vix_level is not None:
            vol_score = 0.5  # Default neutral

            if iv_percentile is not None:
                vol_score = iv_percentile / 100.0
            elif vix_level is not None:
                # Normalize VIX (12 low, 30 high)
                vol_score = min(1.0, max(0.0, (vix_level - 12) / 18))

            # High vol = wider stops
            vol_mult = 1.0 + (vol_score - 0.5) * self.iv_factor
            stop_loss *= vol_mult
            take_profit *= vol_mult
            trailing_dist *= vol_mult

            if vol_score > 0.7:
                adjustments.append("high_vol")
            elif vol_score < 0.3:
                adjustments.append("low_vol")

        # 3. Time to expiry adjustment
        # Near expiry = tighter holds (gamma risk)
        if minutes_to_expiry is not None:
            if minutes_to_expiry < 60:  # Less than 1 hour
                max_hold = min(max_hold, 15)
                stop_loss *= 0.8  # Tighter stops near expiry
                adjustments.append("near_expiry")
            elif minutes_to_expiry < 240:  # Less than 4 hours
                max_hold = min(max_hold, 30)
                stop_loss *= 0.9
                adjustments.append("same_day")

        # 4. Theta adjustment
        # High theta decay = tighter time limits
        if theta is not None:
            theta_pct = abs(theta) * 100  # Convert to percentage of premium
            if theta_pct > 2.0:  # Losing >2% per day to theta
                max_hold = min(max_hold, 20)
                adjustments.append("high_theta")
            elif theta_pct > 1.0:
                max_hold = min(max_hold, 30)

        # 5. MC uncertainty adjustment (use for risk, not entry)
        # High uncertainty = wider stops, smaller positions (handled elsewhere)
        if mc_uncertainty is not None:
            unc_mult = 1.0 + min(mc_uncertainty * 2, 0.3)  # Max 30% wider
            stop_loss *= unc_mult
            if mc_uncertainty > 0.15:
                adjustments.append("high_uncertainty")

        # Clamp to reasonable ranges
        stop_loss = max(3.0, min(20.0, stop_loss))  # 3% to 20%
        take_profit = max(5.0, min(50.0, take_profit))  # 5% to 50%
        max_hold = max(5, min(120, max_hold))  # 5 to 120 minutes
        trailing_act = max(2.0, min(15.0, trailing_act))
        trailing_dist = max(1.0, min(8.0, trailing_dist))

        reason = ",".join(adjustments) if adjustments else "standard"

        logger.debug(f"[GREEKS_EXIT] Levels: SL={stop_loss:.1f}%, TP={take_profit:.1f}%, "
                    f"hold={max_hold}min ({reason})")

        return DynamicExitLevels(
            stop_loss_pct=round(stop_loss, 1),
            take_profit_pct=round(take_profit, 1),
            max_hold_minutes=max_hold,
            trailing_activation_pct=round(trailing_act, 1),
            trailing_distance_pct=round(trailing_dist, 1),
            reason=reason,
        )

    def get_position_size_multiplier(
        self,
        mc_uncertainty: Optional[float] = None,
        iv_percentile: Optional[float] = None,
        vix_level: Optional[float] = None,
    ) -> float:
        """
        Get position size multiplier based on uncertainty.

        Use MC uncertainty for position sizing (risk management),
        NOT for entry selection.

        Returns:
            Multiplier 0.5-1.0 for position sizing
        """
        if not self.enabled:
            return 1.0

        multiplier = 1.0

        # MC uncertainty reduces position size
        if mc_uncertainty is not None:
            # High uncertainty = smaller positions
            # uncertainty 0.0 = full size, 0.25 = half size
            multiplier *= max(0.5, 1.0 - mc_uncertainty * 2)

        # High VIX reduces position size
        if vix_level is not None and vix_level > 25:
            vix_factor = min(0.3, (vix_level - 25) / 50)  # Max 30% reduction
            multiplier *= (1.0 - vix_factor)

        return round(max(0.5, min(1.0, multiplier)), 2)


# Singleton
_greeks_exit_manager = None

def get_greeks_exit_manager() -> GreeksAwareExitManager:
    """Get or create the Greeks-aware exit manager singleton."""
    global _greeks_exit_manager
    if _greeks_exit_manager is None:
        _greeks_exit_manager = GreeksAwareExitManager()
    return _greeks_exit_manager
