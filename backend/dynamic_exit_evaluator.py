"""
Dynamic Exit Evaluator - Continuously re-evaluates open positions

Instead of passively waiting for stop/TP, this evaluator asks every cycle:
1. Is my entry thesis still valid?
2. Has the market regime changed?
3. Is theta decay exceeding my expected profit?
4. Should I exit early based on time/momentum?

This addresses the core issue: 97% of trades exit via time limit because
we're not actively managing positions.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any, Tuple

logger = logging.getLogger(__name__)


class ExitReason(Enum):
    HOLD = "hold"
    THESIS_INVALIDATED = "thesis_invalidated"
    CONFIDENCE_COLLAPSED = "confidence_collapsed"
    REGIME_CHANGED = "regime_changed"
    THETA_EXCEEDS_EXPECTED = "theta_exceeds_expected"
    MOMENTUM_EXHAUSTED = "momentum_exhausted"
    TIME_DECAY_EXIT = "time_decay_exit"
    PROFIT_PROTECTION = "profit_protection"


@dataclass
class ExitDecision:
    should_exit: bool
    reason: ExitReason
    confidence: float  # 0-1, how confident we are in the exit decision
    details: str


class DynamicExitEvaluator:
    """
    Evaluates whether to exit a position based on current market conditions.

    Key principles:
    - Cut losers early when thesis is invalidated
    - Let winners run but protect profits
    - Don't wait for time limit if trade is clearly not working
    """

    def __init__(self, config: Optional[Dict] = None):
        config = config or {}

        # Thesis invalidation thresholds
        self.prediction_flip_exit = config.get('prediction_flip_exit', True)
        self.confidence_collapse_threshold = config.get('confidence_collapse_threshold', 0.5)  # Exit if conf drops by 50%

        # Regime change settings
        self.regime_change_exit = config.get('regime_change_exit', True)

        # Theta/time decay settings
        self.theta_check_enabled = config.get('theta_check_enabled', True)
        self.theta_daily_pct = config.get('theta_daily_pct', 0.05)  # Assume 5% daily theta decay
        self.min_expected_move_ratio = config.get('min_expected_move_ratio', 1.5)  # Expected move must be 1.5x theta

        # Momentum exhaustion
        self.momentum_check_enabled = config.get('momentum_check_enabled', True)
        self.momentum_reversal_threshold = config.get('momentum_reversal_threshold', 0.3)  # 30% momentum reversal

        # Time-based probability
        self.time_decay_exit_enabled = config.get('time_decay_exit_enabled', True)
        self.flat_trade_exit_minutes = config.get('flat_trade_exit_minutes', 10)  # Exit flat trades after 10 min
        self.flat_threshold_pct = config.get('flat_threshold_pct', 2.0)  # "Flat" = within +/- 2%

        # Profit protection
        self.profit_protection_enabled = config.get('profit_protection_enabled', True)
        self.profit_lock_threshold = config.get('profit_lock_threshold', 5.0)  # Lock profits above 5%
        self.profit_pullback_exit = config.get('profit_pullback_exit', 3.0)  # Exit if pulls back 3% from peak

        # Track peak profit for each trade
        self.peak_profits: Dict[str, float] = {}

        logger.info("DynamicExitEvaluator initialized:")
        logger.info(f"  Prediction flip exit: {self.prediction_flip_exit}")
        logger.info(f"  Confidence collapse threshold: {self.confidence_collapse_threshold:.0%}")
        logger.info(f"  Regime change exit: {self.regime_change_exit}")
        logger.info(f"  Flat trade exit: {self.flat_trade_exit_minutes} min")
        logger.info(f"  Profit protection: {self.profit_lock_threshold}% lock, {self.profit_pullback_exit}% pullback")

    def evaluate(
        self,
        trade_id: str,
        trade_type: str,  # 'CALL' or 'PUT'
        entry_prediction_direction: str,  # 'UP' or 'DOWN'
        entry_confidence: float,
        entry_hmm_regime: Dict[str, Any],
        current_prediction_direction: str,
        current_confidence: float,
        current_hmm_regime: Dict[str, Any],
        current_pnl_pct: float,
        minutes_held: float,
        current_momentum: float,  # -1 to 1, current price momentum
        entry_momentum: float,  # momentum at entry
    ) -> ExitDecision:
        """
        Evaluate whether to exit a position.

        Returns ExitDecision with should_exit, reason, and confidence.
        """

        # Track peak profit for profit protection
        if trade_id not in self.peak_profits:
            self.peak_profits[trade_id] = current_pnl_pct
        else:
            self.peak_profits[trade_id] = max(self.peak_profits[trade_id], current_pnl_pct)

        peak_profit = self.peak_profits[trade_id]

        # === CHECK 1: Thesis Invalidation (Prediction Flipped) ===
        if self.prediction_flip_exit:
            expected_direction = 'UP' if trade_type == 'CALL' else 'DOWN'
            if current_prediction_direction != expected_direction and current_prediction_direction != 'NEUTRAL':
                # Prediction flipped against our position
                # Only exit if we're not already profitable
                if current_pnl_pct < 3.0:  # Not profitable enough to ignore the flip
                    return ExitDecision(
                        should_exit=True,
                        reason=ExitReason.THESIS_INVALIDATED,
                        confidence=0.8,
                        details=f"Prediction flipped from {expected_direction} to {current_prediction_direction}, P&L: {current_pnl_pct:+.1f}%"
                    )

        # === CHECK 2: Confidence Collapsed ===
        if entry_confidence > 0:
            confidence_ratio = current_confidence / entry_confidence
            if confidence_ratio < self.confidence_collapse_threshold:
                # Confidence dropped significantly
                if current_pnl_pct < 0:  # Only exit if losing
                    return ExitDecision(
                        should_exit=True,
                        reason=ExitReason.CONFIDENCE_COLLAPSED,
                        confidence=0.7,
                        details=f"Confidence collapsed: {entry_confidence:.1%} -> {current_confidence:.1%} ({confidence_ratio:.0%})"
                    )

        # === CHECK 3: Regime Changed ===
        if self.regime_change_exit:
            entry_trend = entry_hmm_regime.get('trend_state', 1)
            current_trend = current_hmm_regime.get('trend_state', 1)

            # Check if regime flipped against our position
            if trade_type == 'CALL' and entry_trend >= 1.5 and current_trend <= 0.5:
                # Was bullish, now bearish - exit calls
                if current_pnl_pct < 5.0:  # Not super profitable
                    return ExitDecision(
                        should_exit=True,
                        reason=ExitReason.REGIME_CHANGED,
                        confidence=0.75,
                        details=f"HMM regime flipped bearish (trend: {entry_trend:.1f} -> {current_trend:.1f})"
                    )
            elif trade_type == 'PUT' and entry_trend <= 0.5 and current_trend >= 1.5:
                # Was bearish, now bullish - exit puts
                if current_pnl_pct < 5.0:
                    return ExitDecision(
                        should_exit=True,
                        reason=ExitReason.REGIME_CHANGED,
                        confidence=0.75,
                        details=f"HMM regime flipped bullish (trend: {entry_trend:.1f} -> {current_trend:.1f})"
                    )

        # === CHECK 4: Theta Decay vs Expected Move ===
        if self.theta_check_enabled and minutes_held > 5:
            # Calculate theta decay so far
            minutes_per_day = 6.5 * 60  # ~390 trading minutes
            theta_decay_pct = self.theta_daily_pct * (minutes_held / minutes_per_day)

            # If we're flat/losing and theta is eating us
            if current_pnl_pct < theta_decay_pct and current_pnl_pct < 2.0:
                # Our P&L isn't beating theta decay
                if minutes_held > 15:  # Give it some time first
                    return ExitDecision(
                        should_exit=True,
                        reason=ExitReason.THETA_EXCEEDS_EXPECTED,
                        confidence=0.6,
                        details=f"Theta decay ({theta_decay_pct:.1f}%) exceeding gains ({current_pnl_pct:+.1f}%)"
                    )

        # === CHECK 5: Momentum Exhausted ===
        if self.momentum_check_enabled:
            # Check if momentum reversed significantly
            if trade_type == 'CALL' and entry_momentum > 0:
                # We entered on positive momentum for a call
                if current_momentum < -self.momentum_reversal_threshold:
                    # Momentum flipped negative
                    if current_pnl_pct < 3.0:
                        return ExitDecision(
                            should_exit=True,
                            reason=ExitReason.MOMENTUM_EXHAUSTED,
                            confidence=0.65,
                            details=f"Momentum reversed: {entry_momentum:+.2f} -> {current_momentum:+.2f}"
                        )
            elif trade_type == 'PUT' and entry_momentum < 0:
                # We entered on negative momentum for a put
                if current_momentum > self.momentum_reversal_threshold:
                    if current_pnl_pct < 3.0:
                        return ExitDecision(
                            should_exit=True,
                            reason=ExitReason.MOMENTUM_EXHAUSTED,
                            confidence=0.65,
                            details=f"Momentum reversed: {entry_momentum:+.2f} -> {current_momentum:+.2f}"
                        )

        # === CHECK 6: Time-Based Flat Trade Exit ===
        if self.time_decay_exit_enabled:
            if minutes_held >= self.flat_trade_exit_minutes:
                if abs(current_pnl_pct) < self.flat_threshold_pct:
                    # Trade is flat after X minutes - it's not working
                    return ExitDecision(
                        should_exit=True,
                        reason=ExitReason.TIME_DECAY_EXIT,
                        confidence=0.6,
                        details=f"Trade flat ({current_pnl_pct:+.1f}%) after {minutes_held:.0f} min"
                    )

        # === CHECK 7: Profit Protection ===
        if self.profit_protection_enabled:
            if peak_profit >= self.profit_lock_threshold:
                # We've been profitable - protect it
                pullback = peak_profit - current_pnl_pct
                if pullback >= self.profit_pullback_exit:
                    return ExitDecision(
                        should_exit=True,
                        reason=ExitReason.PROFIT_PROTECTION,
                        confidence=0.85,
                        details=f"Protecting profit: peak {peak_profit:+.1f}% -> current {current_pnl_pct:+.1f}% (pullback: {pullback:.1f}%)"
                    )

        # === NO EXIT - HOLD ===
        return ExitDecision(
            should_exit=False,
            reason=ExitReason.HOLD,
            confidence=0.5,
            details=f"Holding: P&L {current_pnl_pct:+.1f}%, {minutes_held:.0f}min, conf {current_confidence:.1%}"
        )

    def cleanup_trade(self, trade_id: str):
        """Remove tracking data for closed trade."""
        if trade_id in self.peak_profits:
            del self.peak_profits[trade_id]

    def get_stats(self) -> Dict[str, Any]:
        """Get evaluator statistics."""
        return {
            'active_trades_tracked': len(self.peak_profits),
            'peak_profits': dict(self.peak_profits)
        }
