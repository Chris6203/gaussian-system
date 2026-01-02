"""
Skew-Optimized Exit Manager - Capture fat-tail winners while protecting profits.

Key insight: With ~38% WR, the edge comes from occasional 100%+ winners.
A hard 12% TP chops the best outcomes. Instead:
1. Partial take-profit: Take 50% at target, let runner continue
2. Trailing stop on runner: Lock in gains while letting it run
3. Trend-conditioned TP: Wider or no TP in trending regimes

Based on analysis: One +584% trade drove 1293% of P&L in v3_calibrated.
"""

import os
import logging
from dataclasses import dataclass
from typing import Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class ExitMode(Enum):
    """Exit mode determines TP behavior."""
    FIXED = "fixed"  # Standard fixed TP
    PARTIAL = "partial"  # Take partial, leave runner
    TRAILING = "trailing"  # Trailing stop only
    TREND_ADAPTIVE = "trend_adaptive"  # Widen/remove TP in trends


@dataclass
class SkewExitConfig:
    """Configuration for skew-optimized exits."""
    # Partial take-profit settings
    partial_tp_pct: float = 0.10  # Take profit at 10%
    partial_take_fraction: float = 0.50  # Take 50% of position

    # Runner settings (after partial TP)
    runner_trailing_activation: float = 0.15  # Activate trail at 15%
    runner_trailing_distance: float = 0.05  # Trail 5% behind peak

    # Trend-conditioned settings
    trend_tp_multiplier: float = 2.0  # 2x TP in strong trends
    trend_threshold: float = 0.70  # HMM trend > 0.70 = strong trend

    # Maximum runner potential
    max_runner_pct: float = 3.0  # Cap runner at 300% (emergency exit)


@dataclass
class ExitDecision:
    """Result of exit evaluation."""
    should_exit: bool
    exit_fraction: float  # 0.0 to 1.0 (partial or full)
    exit_reason: str
    new_stop_pct: Optional[float] = None  # Updated trailing stop
    runner_active: bool = False


class SkewExitManager:
    """
    Manages exits to capture positive skew (fat-tail winners).

    Replaces fixed TP with:
    1. Partial TP at target (take 50%, leave runner)
    2. Trailing stop on runner
    3. Trend-adaptive TP widening
    """

    def __init__(
        self,
        base_tp_pct: float = 0.12,  # 12% base take profit
        base_sl_pct: float = 0.08,  # 8% stop loss
        mode: ExitMode = ExitMode.PARTIAL,
        config: Optional[SkewExitConfig] = None,
    ):
        self.base_tp_pct = float(os.environ.get('BASE_TP_PCT', str(base_tp_pct)))
        self.base_sl_pct = float(os.environ.get('BASE_SL_PCT', str(base_sl_pct)))
        self.mode = ExitMode(os.environ.get('SKEW_EXIT_MODE', mode.value))
        self.config = config or SkewExitConfig()

        # Override config from env
        if os.environ.get('PARTIAL_TP_PCT'):
            self.config.partial_tp_pct = float(os.environ.get('PARTIAL_TP_PCT'))
        if os.environ.get('PARTIAL_TAKE_FRACTION'):
            self.config.partial_take_fraction = float(os.environ.get('PARTIAL_TAKE_FRACTION'))
        if os.environ.get('RUNNER_TRAIL_ACTIVATION'):
            self.config.runner_trailing_activation = float(os.environ.get('RUNNER_TRAIL_ACTIVATION'))
        if os.environ.get('RUNNER_TRAIL_DISTANCE'):
            self.config.runner_trailing_distance = float(os.environ.get('RUNNER_TRAIL_DISTANCE'))

        self.enabled = os.environ.get('SKEW_EXIT_ENABLED', '0') == '1'

        # Track runner positions
        self._runners = {}  # trade_id -> {'peak_pnl': float, 'partial_taken': bool}

        logger.info(f"[SKEW_EXIT] Initialized: mode={self.mode.value}, enabled={self.enabled}")

    def evaluate_exit(
        self,
        trade_id: str,
        current_pnl_pct: float,
        peak_pnl_pct: float,
        hmm_trend: Optional[float] = None,
        minutes_held: float = 0,
        max_hold_minutes: float = 45,
    ) -> ExitDecision:
        """
        Evaluate whether to exit and how much.

        Args:
            trade_id: Unique trade identifier
            current_pnl_pct: Current P&L as decimal (0.12 = 12%)
            peak_pnl_pct: Peak P&L reached (for trailing)
            hmm_trend: HMM trend state (0-1)
            minutes_held: Minutes in trade
            max_hold_minutes: Maximum hold time

        Returns:
            ExitDecision with action to take
        """
        if not self.enabled:
            return self._fixed_exit(current_pnl_pct, minutes_held, max_hold_minutes)

        if self.mode == ExitMode.FIXED:
            return self._fixed_exit(current_pnl_pct, minutes_held, max_hold_minutes)
        elif self.mode == ExitMode.PARTIAL:
            return self._partial_exit(trade_id, current_pnl_pct, peak_pnl_pct,
                                      minutes_held, max_hold_minutes)
        elif self.mode == ExitMode.TRAILING:
            return self._trailing_exit(trade_id, current_pnl_pct, peak_pnl_pct,
                                       minutes_held, max_hold_minutes)
        elif self.mode == ExitMode.TREND_ADAPTIVE:
            return self._trend_adaptive_exit(trade_id, current_pnl_pct, peak_pnl_pct,
                                             hmm_trend, minutes_held, max_hold_minutes)
        else:
            return self._fixed_exit(current_pnl_pct, minutes_held, max_hold_minutes)

    def _fixed_exit(
        self,
        current_pnl_pct: float,
        minutes_held: float,
        max_hold_minutes: float,
    ) -> ExitDecision:
        """Standard fixed stop/TP exit."""
        # Stop loss
        if current_pnl_pct <= -self.base_sl_pct:
            return ExitDecision(
                should_exit=True,
                exit_fraction=1.0,
                exit_reason=f"stop_loss:{current_pnl_pct:.1%}",
            )

        # Take profit
        if current_pnl_pct >= self.base_tp_pct:
            return ExitDecision(
                should_exit=True,
                exit_fraction=1.0,
                exit_reason=f"take_profit:{current_pnl_pct:.1%}",
            )

        # Max hold
        if minutes_held >= max_hold_minutes:
            return ExitDecision(
                should_exit=True,
                exit_fraction=1.0,
                exit_reason=f"max_hold:{minutes_held:.0f}min",
            )

        return ExitDecision(should_exit=False, exit_fraction=0.0, exit_reason="hold")

    def _partial_exit(
        self,
        trade_id: str,
        current_pnl_pct: float,
        peak_pnl_pct: float,
        minutes_held: float,
        max_hold_minutes: float,
    ) -> ExitDecision:
        """
        Partial take-profit with runner.

        1. At first TP target, take 50% of position
        2. Remaining 50% becomes "runner" with trailing stop
        3. Runner exits on trail or emergency
        """
        # Initialize runner tracking
        if trade_id not in self._runners:
            self._runners[trade_id] = {
                'peak_pnl': current_pnl_pct,
                'partial_taken': False,
            }

        runner = self._runners[trade_id]
        runner['peak_pnl'] = max(runner['peak_pnl'], current_pnl_pct)

        # Stop loss (full exit)
        if current_pnl_pct <= -self.base_sl_pct:
            self._cleanup_runner(trade_id)
            return ExitDecision(
                should_exit=True,
                exit_fraction=1.0,
                exit_reason=f"stop_loss:{current_pnl_pct:.1%}",
            )

        # Partial TP (first target reached, take 50%)
        if not runner['partial_taken'] and current_pnl_pct >= self.config.partial_tp_pct:
            runner['partial_taken'] = True
            logger.info(f"[SKEW_EXIT] Partial TP triggered at {current_pnl_pct:.1%}, "
                       f"taking {self.config.partial_take_fraction:.0%}")
            return ExitDecision(
                should_exit=True,
                exit_fraction=self.config.partial_take_fraction,
                exit_reason=f"partial_tp:{current_pnl_pct:.1%}",
                runner_active=True,
            )

        # Runner trailing stop (after partial taken)
        if runner['partial_taken']:
            # Emergency cap
            if current_pnl_pct >= self.config.max_runner_pct:
                self._cleanup_runner(trade_id)
                return ExitDecision(
                    should_exit=True,
                    exit_fraction=1.0,
                    exit_reason=f"runner_max:{current_pnl_pct:.1%}",
                )

            # Trailing activation
            if runner['peak_pnl'] >= self.config.runner_trailing_activation:
                trail_stop = runner['peak_pnl'] - self.config.runner_trailing_distance
                if current_pnl_pct <= trail_stop:
                    self._cleanup_runner(trade_id)
                    return ExitDecision(
                        should_exit=True,
                        exit_fraction=1.0,
                        exit_reason=f"runner_trail:{current_pnl_pct:.1%}(peak:{runner['peak_pnl']:.1%})",
                    )

                # Update trailing stop
                return ExitDecision(
                    should_exit=False,
                    exit_fraction=0.0,
                    exit_reason="runner_hold",
                    new_stop_pct=trail_stop,
                    runner_active=True,
                )

        # Max hold (with slight extension for runners)
        hold_limit = max_hold_minutes * (1.5 if runner['partial_taken'] else 1.0)
        if minutes_held >= hold_limit:
            self._cleanup_runner(trade_id)
            return ExitDecision(
                should_exit=True,
                exit_fraction=1.0,
                exit_reason=f"max_hold:{minutes_held:.0f}min",
            )

        return ExitDecision(should_exit=False, exit_fraction=0.0, exit_reason="hold")

    def _trailing_exit(
        self,
        trade_id: str,
        current_pnl_pct: float,
        peak_pnl_pct: float,
        minutes_held: float,
        max_hold_minutes: float,
    ) -> ExitDecision:
        """Pure trailing stop (no fixed TP)."""
        # Stop loss
        if current_pnl_pct <= -self.base_sl_pct:
            return ExitDecision(
                should_exit=True,
                exit_fraction=1.0,
                exit_reason=f"stop_loss:{current_pnl_pct:.1%}",
            )

        # Emergency cap
        if current_pnl_pct >= self.config.max_runner_pct:
            return ExitDecision(
                should_exit=True,
                exit_fraction=1.0,
                exit_reason=f"max_gain:{current_pnl_pct:.1%}",
            )

        # Trailing stop
        if peak_pnl_pct >= self.config.runner_trailing_activation:
            trail_stop = peak_pnl_pct - self.config.runner_trailing_distance
            if current_pnl_pct <= trail_stop:
                return ExitDecision(
                    should_exit=True,
                    exit_fraction=1.0,
                    exit_reason=f"trailing:{current_pnl_pct:.1%}(peak:{peak_pnl_pct:.1%})",
                )

        # Max hold
        if minutes_held >= max_hold_minutes:
            return ExitDecision(
                should_exit=True,
                exit_fraction=1.0,
                exit_reason=f"max_hold:{minutes_held:.0f}min",
            )

        return ExitDecision(should_exit=False, exit_fraction=0.0, exit_reason="hold")

    def _trend_adaptive_exit(
        self,
        trade_id: str,
        current_pnl_pct: float,
        peak_pnl_pct: float,
        hmm_trend: Optional[float],
        minutes_held: float,
        max_hold_minutes: float,
    ) -> ExitDecision:
        """
        Trend-conditioned TP:
        - Strong trend (HMM > 0.70): Use trailing stop, no fixed TP
        - Weak/no trend: Use fixed TP
        """
        # Determine if in strong trend
        in_strong_trend = hmm_trend is not None and (
            hmm_trend > self.config.trend_threshold or
            hmm_trend < (1 - self.config.trend_threshold)
        )

        if in_strong_trend:
            # Use trailing exit in trends (let winners run)
            return self._trailing_exit(trade_id, current_pnl_pct, peak_pnl_pct,
                                       minutes_held, max_hold_minutes * 1.5)
        else:
            # Use partial exit in choppy markets
            return self._partial_exit(trade_id, current_pnl_pct, peak_pnl_pct,
                                      minutes_held, max_hold_minutes)

    def _cleanup_runner(self, trade_id: str):
        """Remove runner tracking for closed trade."""
        if trade_id in self._runners:
            del self._runners[trade_id]

    def get_runner_stats(self) -> dict:
        """Get current runner positions stats."""
        return {
            'active_runners': len([r for r in self._runners.values() if r['partial_taken']]),
            'total_tracked': len(self._runners),
        }


# Singleton
_skew_exit_manager = None

def get_skew_exit_manager() -> SkewExitManager:
    """Get or create the skew exit manager singleton."""
    global _skew_exit_manager
    if _skew_exit_manager is None:
        _skew_exit_manager = SkewExitManager()
    return _skew_exit_manager
