"""
Expected Value (EV) Gate - Gate trades on positive expectancy after costs.

Key insight: High confidence doesn't mean positive EV when theta/spreads dominate.
This is especially important for 0-1 DTE options where costs can exceed edge.

EV = P(win) * E[win] - P(loss) * E[loss] - costs
where costs = spread + slippage + theta_decay

Gate requires:
1. EV > 0 (positive expectancy)
2. EV / max_loss > min_ratio (risk-adjusted return)
"""

import os
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


@dataclass
class EVGateResult:
    """Result of EV gate evaluation."""
    passed: bool
    ev: float
    ev_ratio: float  # EV / risk
    breakdown: Dict[str, float]
    rejection_reason: Optional[str] = None


class EVGate:
    """
    Expected Value gate for trade filtering.

    Calculates EV considering:
    - Calibrated win probability
    - Expected win/loss amounts from stops
    - Spread costs (bid-ask)
    - Slippage estimates
    - Theta decay over expected hold time
    """

    def __init__(
        self,
        min_ev: float = 0.0,  # Minimum EV to trade (0 = just be positive)
        min_ev_ratio: float = 0.0,  # Minimum EV/risk ratio (0 = just require positive EV)
        spread_cost_pct: float = 0.002,  # Default 0.2% spread cost
        slippage_pct: float = 0.001,  # Default 0.1% slippage
        theta_decay_per_hour: float = 0.003,  # 0.3% theta decay per hour (conservative)
        # Bayesian prior blending (replaces hard floor)
        win_prob_prior: float = 0.40,  # Prior P(win) based on historical base rate
        prior_weight: float = 0.3,  # Weight of prior (0=use model only, 1=use prior only)
    ):
        self.min_ev = min_ev
        self.min_ev_ratio = min_ev_ratio
        self.spread_cost_pct = spread_cost_pct
        self.slippage_pct = slippage_pct
        self.theta_decay_per_hour = theta_decay_per_hour
        self.win_prob_prior = win_prob_prior
        self.prior_weight = prior_weight

        # Load from environment
        self.enabled = os.environ.get('EV_GATE_ENABLED', '0') == '1'
        if os.environ.get('EV_MIN_RATIO'):
            self.min_ev_ratio = float(os.environ.get('EV_MIN_RATIO'))
        if os.environ.get('EV_SPREAD_COST'):
            self.spread_cost_pct = float(os.environ.get('EV_SPREAD_COST'))
        if os.environ.get('EV_THETA_DECAY'):
            self.theta_decay_per_hour = float(os.environ.get('EV_THETA_DECAY'))
        if os.environ.get('EV_WIN_PROB_PRIOR'):
            self.win_prob_prior = float(os.environ.get('EV_WIN_PROB_PRIOR'))
        if os.environ.get('EV_PRIOR_WEIGHT'):
            self.prior_weight = float(os.environ.get('EV_PRIOR_WEIGHT'))

        logger.info(f"[EV_GATE] Initialized: enabled={self.enabled}, min_ratio={self.min_ev_ratio}, "
                   f"prior={self.win_prob_prior:.0%} (w={self.prior_weight:.1f})")

    def calculate_ev(
        self,
        win_prob: float,  # Calibrated P(profit)
        take_profit_pct: float,  # Take profit threshold (e.g., 0.12 = 12%)
        stop_loss_pct: float,  # Stop loss threshold (e.g., 0.08 = 8%)
        expected_hold_minutes: float,  # Expected holding time
        spread_pct: Optional[float] = None,  # Actual spread if known
        delta: Optional[float] = None,  # Option delta for Greeks-aware calc
        theta: Optional[float] = None,  # Option theta (daily decay $)
        premium: Optional[float] = None,  # Option premium paid
    ) -> EVGateResult:
        """
        Calculate expected value of a trade.

        Args:
            win_prob: Calibrated probability of profit (0-1)
            take_profit_pct: Take profit as decimal (0.12 = 12%)
            stop_loss_pct: Stop loss as decimal (0.08 = 8%)
            expected_hold_minutes: Expected time in trade
            spread_pct: Bid-ask spread as % of premium
            delta: Option delta (0-1)
            theta: Daily theta decay in $
            premium: Option premium paid

        Returns:
            EVGateResult with pass/fail and breakdown
        """
        # Apply Bayesian prior blending (not a hard floor)
        # P(win)_posterior = w * prior + (1-w) * model
        # This keeps the gate honest while preventing "no-trade" when model is underconfident
        win_prob_posterior = self.prior_weight * self.win_prob_prior + (1 - self.prior_weight) * win_prob
        win_prob = win_prob_posterior

        # Expected win/loss amounts
        expected_win = take_profit_pct
        expected_loss = stop_loss_pct

        # Cost components
        spread_cost = spread_pct if spread_pct is not None else self.spread_cost_pct
        slippage_cost = self.slippage_pct

        # Theta cost (convert to % of premium for hold time)
        if theta is not None and premium is not None and premium > 0:
            # Theta is daily decay, convert to hold period
            theta_cost_pct = abs(theta) * (expected_hold_minutes / (60 * 24)) / premium
        else:
            # Use default decay estimate
            theta_cost_pct = self.theta_decay_per_hour * (expected_hold_minutes / 60)

        # Total round-trip costs (entry + exit)
        total_costs = (spread_cost * 2) + (slippage_cost * 2) + theta_cost_pct

        # Calculate EV
        # EV = P(win) * (win - costs) - P(loss) * (loss + costs)
        # Simplified: EV = P(win)*win - P(loss)*loss - costs
        ev_gross = win_prob * expected_win - (1 - win_prob) * expected_loss
        ev_net = ev_gross - total_costs

        # Risk is the max expected loss
        max_loss = expected_loss + total_costs

        # EV ratio (risk-adjusted)
        ev_ratio = ev_net / max_loss if max_loss > 0 else 0

        breakdown = {
            'win_prob': win_prob,
            'expected_win': expected_win,
            'expected_loss': expected_loss,
            'spread_cost': spread_cost * 2,
            'slippage_cost': slippage_cost * 2,
            'theta_cost': theta_cost_pct,
            'total_costs': total_costs,
            'ev_gross': ev_gross,
            'ev_net': ev_net,
            'max_loss': max_loss,
            'ev_ratio': ev_ratio,
        }

        # Determine pass/fail
        passed = True
        rejection_reason = None

        if ev_net < self.min_ev:
            passed = False
            rejection_reason = f"EV={ev_net:.2%} < min={self.min_ev:.2%}"
        elif ev_ratio < self.min_ev_ratio:
            passed = False
            rejection_reason = f"EV/risk={ev_ratio:.2f} < min={self.min_ev_ratio:.2f}"

        return EVGateResult(
            passed=passed,
            ev=ev_net,
            ev_ratio=ev_ratio,
            breakdown=breakdown,
            rejection_reason=rejection_reason,
        )

    def evaluate(
        self,
        signal: Dict[str, Any],
        config: Dict[str, Any],
        option_greeks: Optional[Dict[str, float]] = None,
    ) -> EVGateResult:
        """
        Evaluate a trade signal against the EV gate.

        Args:
            signal: Trading signal with confidence, etc.
            config: Trading config with stops, etc.
            option_greeks: Optional Greeks dict with delta, theta, etc.

        Returns:
            EVGateResult
        """
        if not self.enabled:
            return EVGateResult(passed=True, ev=0, ev_ratio=0, breakdown={})

        # Extract parameters
        win_prob = signal.get('calibrated_confidence', signal.get('confidence', 0.5))
        take_profit = config.get('take_profit_pct', 0.12)
        stop_loss = config.get('stop_loss_pct', 0.08)
        max_hold = config.get('max_hold_minutes', 30)

        # Greeks if available
        delta = None
        theta = None
        premium = signal.get('premium', None)

        if option_greeks:
            delta = option_greeks.get('delta')
            theta = option_greeks.get('theta')

        # Calculate spread from signal if available
        spread_pct = None
        if signal.get('bid') and signal.get('ask') and signal.get('mid'):
            mid = signal['mid']
            if mid > 0:
                spread_pct = (signal['ask'] - signal['bid']) / mid

        return self.calculate_ev(
            win_prob=win_prob,
            take_profit_pct=take_profit,
            stop_loss_pct=stop_loss,
            expected_hold_minutes=max_hold * 0.7,  # Expect to exit before max
            spread_pct=spread_pct,
            delta=delta,
            theta=theta,
            premium=premium,
        )


# Singleton instance
_ev_gate = None

def get_ev_gate() -> EVGate:
    """Get or create the EV gate singleton."""
    global _ev_gate
    if _ev_gate is None:
        _ev_gate = EVGate()
    return _ev_gate
