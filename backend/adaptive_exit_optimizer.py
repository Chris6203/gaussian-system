#!/usr/bin/env python3
"""
Adaptive Exit Optimizer

Carmack Philosophy: Simple, measurable, debuggable
Simons Philosophy: Data-driven, multiple signals, adaptive

This module:
1. Predicts optimal hold time using multi-horizon analysis
2. Maintains crash protection via VIX monitoring
3. Learns from actual trade outcomes
4. No hard switches - all probabilities and confidence scores

Environment Variables:
- ADAPTIVE_EXIT=1: Enable adaptive exit optimizer
- ADAPTIVE_MIN_HOLD=5: Minimum hold time (crash protection)
- ADAPTIVE_MAX_HOLD=60: Maximum hold time
- ADAPTIVE_VIX_CRASH=25: VIX level for crash mode (fast exit)
"""

import os
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from datetime import datetime, timedelta
from collections import deque

logger = logging.getLogger(__name__)

# Configuration
ADAPTIVE_EXIT = os.environ.get('ADAPTIVE_EXIT', '0') == '1'
ADAPTIVE_MIN_HOLD = int(os.environ.get('ADAPTIVE_MIN_HOLD', '5'))
ADAPTIVE_MAX_HOLD = int(os.environ.get('ADAPTIVE_MAX_HOLD', '60'))
ADAPTIVE_VIX_CRASH = float(os.environ.get('ADAPTIVE_VIX_CRASH', '25'))


@dataclass
class HorizonPrediction:
    """Prediction for a specific time horizon."""
    horizon_mins: int
    predicted_return: float  # Expected return at this horizon
    confidence: float  # 0-1
    win_probability: float  # P(profitable) at this horizon


@dataclass
class ExitRecommendation:
    """Recommendation from the adaptive exit optimizer."""
    action: str  # 'HOLD', 'EXIT_NOW', 'EXIT_AT_TARGET'
    optimal_hold_mins: int
    expected_return: float
    crash_mode: bool  # True if VIX spike detected
    reason: str
    horizon_analysis: List[HorizonPrediction] = field(default_factory=list)


@dataclass
class TradeOutcome:
    """Record of a completed trade for learning."""
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    direction: str  # 'CALL' or 'PUT'
    entry_rsi: float
    entry_vix: float
    actual_hold_mins: int
    pnl_pct: float
    optimal_exit_mins: Optional[int] = None  # When was the best exit?


class AdaptiveExitOptimizer:
    """
    ML-driven exit optimizer that learns optimal hold times.

    Carmack: Keep it simple, measure everything
    Simons: Combine weak signals, adapt to data
    """

    def __init__(self):
        self.enabled = ADAPTIVE_EXIT
        self.min_hold = ADAPTIVE_MIN_HOLD
        self.max_hold = ADAPTIVE_MAX_HOLD
        self.vix_crash_threshold = ADAPTIVE_VIX_CRASH

        # Price/feature history for multi-horizon analysis
        self.price_history: deque = deque(maxlen=200)
        self.vix_history: deque = deque(maxlen=200)
        self.rsi_history: deque = deque(maxlen=200)

        # Learning from outcomes (Simons approach)
        self.trade_outcomes: List[TradeOutcome] = []
        self.horizon_win_rates: Dict[int, List[bool]] = {
            h: [] for h in [5, 10, 15, 20, 30, 45, 60]
        }

        # Learned parameters (start with priors, update with data)
        self.call_optimal_hold = 45  # Prior: calls need 45 min
        self.put_optimal_hold = 15   # Prior: puts need less time
        self.oversold_hold_bonus = 15  # Extra time for oversold

        if self.enabled:
            logger.info(f"🎯 Adaptive Exit Optimizer ENABLED")
            logger.info(f"   Min hold: {self.min_hold} min")
            logger.info(f"   Max hold: {self.max_hold} min")
            logger.info(f"   VIX crash threshold: {self.vix_crash_threshold}")

    def update(self, price: float, vix: Optional[float] = None,
               rsi: Optional[float] = None):
        """Update price history for analysis."""
        self.price_history.append(price)
        if vix:
            self.vix_history.append(vix)
        if rsi:
            self.rsi_history.append(rsi)

    def _calculate_horizon_prediction(self, direction: str,
                                       horizon_mins: int) -> HorizonPrediction:
        """
        Predict return at a specific horizon.
        Uses historical patterns + learned win rates.
        """
        if len(self.price_history) < 60:
            return HorizonPrediction(
                horizon_mins=horizon_mins,
                predicted_return=0.0,
                confidence=0.0,
                win_probability=0.5
            )

        prices = np.array(list(self.price_history))

        # Calculate momentum at different scales
        momentum_5 = (prices[-1] - prices[-6]) / prices[-6] if len(prices) > 6 else 0
        momentum_15 = (prices[-1] - prices[-16]) / prices[-16] if len(prices) > 16 else 0
        momentum_30 = (prices[-1] - prices[-31]) / prices[-31] if len(prices) > 31 else 0

        # Mean reversion expectation (Simons: fade momentum)
        # Stronger recent momentum = stronger reversion expected
        reversion_strength = -np.sign(momentum_5) * abs(momentum_5) * 0.5

        # Scale by horizon (longer horizon = more reversion time)
        horizon_factor = np.sqrt(horizon_mins / 15)  # Sqrt for diminishing returns

        # Base prediction
        if direction == 'CALL':
            # For calls, we want price to go UP
            # Oversold conditions (negative momentum) should revert UP
            predicted_return = reversion_strength * horizon_factor
        else:
            # For puts, we want price to go DOWN
            # Overbought conditions (positive momentum) should revert DOWN
            predicted_return = -reversion_strength * horizon_factor

        # Confidence based on data quality and pattern strength
        confidence = min(0.8, 0.3 + abs(momentum_5) * 10)

        # Win probability from learned data (if available)
        if horizon_mins in self.horizon_win_rates:
            outcomes = self.horizon_win_rates[horizon_mins]
            if len(outcomes) >= 10:
                win_probability = sum(outcomes[-50:]) / len(outcomes[-50:])
            else:
                # Prior: slight edge based on correlation analysis
                win_probability = 0.55 if direction == 'CALL' else 0.48
        else:
            win_probability = 0.5

        return HorizonPrediction(
            horizon_mins=horizon_mins,
            predicted_return=predicted_return * 100,  # As percentage
            confidence=confidence,
            win_probability=win_probability
        )

    def _detect_crash_mode(self) -> Tuple[bool, str]:
        """
        Detect if we should be in crash protection mode.
        Carmack: Simple, clear conditions
        """
        if len(self.vix_history) < 2:
            return False, ""

        current_vix = self.vix_history[-1]

        # Hard VIX threshold
        if current_vix >= self.vix_crash_threshold:
            return True, f"VIX={current_vix:.1f} >= {self.vix_crash_threshold}"

        # VIX spike detection (sudden increase)
        if len(self.vix_history) >= 5:
            vix_5_ago = self.vix_history[-5]
            vix_change = (current_vix - vix_5_ago) / vix_5_ago
            if vix_change > 0.10:  # 10% VIX spike in 5 minutes
                return True, f"VIX spike: +{vix_change:.1%} in 5min"

        return False, ""

    def get_exit_recommendation(
        self,
        direction: str,  # 'CALL' or 'PUT'
        entry_price: float,
        current_price: float,
        mins_held: int,
        entry_rsi: Optional[float] = None
    ) -> ExitRecommendation:
        """
        Get adaptive exit recommendation.

        Returns probability-based recommendation, not hard rules.
        """
        if not self.enabled:
            return ExitRecommendation(
                action='HOLD',
                optimal_hold_mins=30,
                expected_return=0.0,
                crash_mode=False,
                reason='adaptive_exit_disabled'
            )

        # Check crash protection first (Carmack: safety first)
        crash_mode, crash_reason = self._detect_crash_mode()
        if crash_mode and mins_held >= self.min_hold:
            return ExitRecommendation(
                action='EXIT_NOW',
                optimal_hold_mins=mins_held,
                expected_return=0.0,
                crash_mode=True,
                reason=f"CRASH_PROTECTION: {crash_reason}"
            )

        # Calculate current P&L
        if direction == 'CALL':
            pnl_pct = (current_price - entry_price) / entry_price * 5  # 5x leverage
        else:
            pnl_pct = (entry_price - current_price) / entry_price * 5

        # Multi-horizon analysis (Simons: look at multiple timeframes)
        horizons = [5, 10, 15, 20, 30, 45, 60]
        predictions = []

        for h in horizons:
            remaining = h - mins_held
            if remaining > 0:  # Only future horizons
                pred = self._calculate_horizon_prediction(direction, h)
                predictions.append(pred)

        if not predictions:
            # Already past all horizons - exit
            return ExitRecommendation(
                action='EXIT_NOW',
                optimal_hold_mins=mins_held,
                expected_return=pnl_pct,
                crash_mode=False,
                reason=f"max_horizon_reached (held {mins_held}min)",
                horizon_analysis=[]
            )

        # Find optimal horizon (highest expected return * win probability)
        best_horizon = None
        best_score = float('-inf')

        for pred in predictions:
            # Risk-adjusted score: expected return * confidence * win_prob
            score = pred.predicted_return * pred.confidence * pred.win_probability

            # Penalize longer holds (theta decay)
            remaining = pred.horizon_mins - mins_held
            theta_penalty = remaining * 0.01  # 1% penalty per minute
            score -= theta_penalty

            if score > best_score:
                best_score = score
                best_horizon = pred

        # Decision logic
        if best_horizon is None:
            action = 'EXIT_NOW'
            reason = 'no_positive_horizon'
        elif mins_held < self.min_hold:
            action = 'HOLD'
            reason = f'min_hold ({mins_held}/{self.min_hold}min)'
        elif best_score < 0 and pnl_pct > 0:
            # Current profit and no better future - take profit
            action = 'EXIT_NOW'
            reason = f'take_profit (P&L={pnl_pct:.1%}, no better horizon)'
        elif best_score < -0.5 and mins_held >= 15:
            # Negative outlook - exit
            action = 'EXIT_NOW'
            reason = f'negative_outlook (score={best_score:.2f})'
        elif best_horizon.horizon_mins - mins_held <= 5:
            # Close to optimal - prepare to exit
            action = 'EXIT_AT_TARGET'
            reason = f'approaching_optimal ({best_horizon.horizon_mins - mins_held}min to optimal)'
        else:
            action = 'HOLD'
            reason = f'wait_for_optimal ({best_horizon.horizon_mins}min, score={best_score:.2f})'

        return ExitRecommendation(
            action=action,
            optimal_hold_mins=best_horizon.horizon_mins if best_horizon else mins_held,
            expected_return=best_horizon.predicted_return if best_horizon else 0.0,
            crash_mode=crash_mode,
            reason=reason,
            horizon_analysis=predictions
        )

    def record_outcome(self, outcome: TradeOutcome):
        """
        Learn from trade outcomes (Simons: adapt to data).
        """
        self.trade_outcomes.append(outcome)

        # Update horizon win rates
        actual_hold = outcome.actual_hold_mins
        was_winner = outcome.pnl_pct > 0

        # Record for nearest horizon
        for h in [5, 10, 15, 20, 30, 45, 60]:
            if abs(actual_hold - h) <= 5:
                self.horizon_win_rates[h].append(was_winner)

        # Update learned parameters
        if len(self.trade_outcomes) >= 20:
            self._update_learned_parameters()

    def _update_learned_parameters(self):
        """
        Update optimal hold times based on outcomes.
        """
        call_outcomes = [o for o in self.trade_outcomes if o.direction == 'CALL']
        put_outcomes = [o for o in self.trade_outcomes if o.direction == 'PUT']

        if len(call_outcomes) >= 10:
            # Find best hold time for calls
            winners = [o for o in call_outcomes if o.pnl_pct > 0]
            if winners:
                self.call_optimal_hold = int(np.median([o.actual_hold_mins for o in winners]))
                logger.info(f"[ADAPTIVE] Updated call optimal hold: {self.call_optimal_hold}min")

        if len(put_outcomes) >= 10:
            winners = [o for o in put_outcomes if o.pnl_pct > 0]
            if winners:
                self.put_optimal_hold = int(np.median([o.actual_hold_mins for o in winners]))
                logger.info(f"[ADAPTIVE] Updated put optimal hold: {self.put_optimal_hold}min")

    def get_stats(self) -> Dict:
        """Get optimizer statistics for debugging (Carmack: measure everything)."""
        stats = {
            'total_outcomes': len(self.trade_outcomes),
            'call_optimal_hold': self.call_optimal_hold,
            'put_optimal_hold': self.put_optimal_hold,
            'horizon_win_rates': {}
        }

        for h, outcomes in self.horizon_win_rates.items():
            if outcomes:
                stats['horizon_win_rates'][h] = sum(outcomes) / len(outcomes)

        return stats


# Global instance
_adaptive_optimizer = None


def get_adaptive_optimizer() -> AdaptiveExitOptimizer:
    """Get or create the global optimizer."""
    global _adaptive_optimizer
    if _adaptive_optimizer is None:
        _adaptive_optimizer = AdaptiveExitOptimizer()
    return _adaptive_optimizer


def should_exit_adaptive(
    direction: str,
    entry_price: float,
    current_price: float,
    mins_held: int,
    vix: Optional[float] = None,
    rsi: Optional[float] = None
) -> ExitRecommendation:
    """
    Convenience function to check if we should exit.

    Usage:
        rec = should_exit_adaptive('CALL', 610.0, 611.0, 15, vix=16.5)
        if rec.action == 'EXIT_NOW':
            exit_trade(reason=rec.reason)
        elif rec.action == 'HOLD':
            print(f"Wait for optimal: {rec.optimal_hold_mins}min")
    """
    optimizer = get_adaptive_optimizer()
    optimizer.update(current_price, vix, rsi)
    return optimizer.get_exit_recommendation(
        direction, entry_price, current_price, mins_held
    )
