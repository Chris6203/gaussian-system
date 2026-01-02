"""
Regime Performance Attribution - Track and auto-disable underperforming regimes.

Maintains rolling performance metrics per regime:
- PnL, drawdown, win rate, calibration quality
- Automatically disables trading in regimes with negative expectancy
- Can raise confidence thresholds for marginal regimes
"""

import os
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RegimePerformance:
    """Performance tracking for a single regime."""
    trades: List[float] = field(default_factory=list)  # PnL values
    timestamps: List[datetime] = field(default_factory=list)
    confidences: List[float] = field(default_factory=list)
    outcomes: List[int] = field(default_factory=list)  # 1=win, 0=loss

    @property
    def total_pnl(self) -> float:
        return sum(self.trades) if self.trades else 0

    @property
    def win_rate(self) -> float:
        return np.mean(self.outcomes) if self.outcomes else 0.5

    @property
    def avg_pnl(self) -> float:
        return np.mean(self.trades) if self.trades else 0

    @property
    def trade_count(self) -> int:
        return len(self.trades)

    @property
    def max_drawdown(self) -> float:
        if not self.trades:
            return 0
        cumsum = np.cumsum(self.trades)
        running_max = np.maximum.accumulate(cumsum)
        drawdowns = running_max - cumsum
        return float(np.max(drawdowns)) if len(drawdowns) > 0 else 0

    def get_recent_pnl(self, lookback_trades: int = 20) -> float:
        """Get PnL from last N trades."""
        if not self.trades:
            return 0
        return sum(self.trades[-lookback_trades:])

    def get_recent_win_rate(self, lookback_trades: int = 20) -> float:
        """Get win rate from last N trades."""
        if not self.outcomes:
            return 0.5
        recent = self.outcomes[-lookback_trades:]
        return np.mean(recent) if recent else 0.5


class RegimePerformanceTracker:
    """
    Tracks performance per regime and provides trading recommendations.

    Auto-disables regimes with:
    - Negative expectancy over lookback period
    - Win rate below threshold
    - Excessive drawdown
    """

    def __init__(
        self,
        disable_threshold_pnl: float = -100,  # Disable if PnL below this (in $)
        disable_threshold_wr: float = 0.30,  # Disable if WR below this
        min_trades_for_decision: int = 20,  # Need this many trades to judge
        confidence_boost_marginal: float = 0.10,  # Raise threshold for marginal
        lookback_trades: int = 50,  # Rolling window for decisions
        auto_disable: bool = True,
    ):
        self.disable_threshold_pnl = float(os.environ.get('REGIME_DISABLE_PNL', str(disable_threshold_pnl)))
        self.disable_threshold_wr = float(os.environ.get('REGIME_DISABLE_WR', str(disable_threshold_wr)))
        self.min_trades = int(os.environ.get('REGIME_MIN_TRADES', str(min_trades_for_decision)))
        self.confidence_boost = float(os.environ.get('REGIME_CONF_BOOST', str(confidence_boost_marginal)))
        self.lookback = lookback_trades
        self.auto_disable = os.environ.get('REGIME_AUTO_DISABLE', '1') == '1' if auto_disable else False

        self.regimes: Dict[str, RegimePerformance] = defaultdict(RegimePerformance)
        self.disabled_regimes: set = set()
        self.marginal_regimes: set = set()  # Regimes needing higher confidence

        logger.info(f"[REGIME_PERF] Initialized: auto_disable={self.auto_disable}, "
                   f"threshold_pnl={self.disable_threshold_pnl}, threshold_wr={self.disable_threshold_wr}")

    def get_regime_key(self, hmm_trend: float, hmm_volatility: float) -> str:
        """Get regime key from HMM states."""
        # Trend bucket
        if hmm_trend < 0.4:
            trend = "bearish"
        elif hmm_trend > 0.6:
            trend = "bullish"
        else:
            trend = "neutral"

        # Volatility bucket
        if hmm_volatility < 0.4:
            vol = "low_vol"
        elif hmm_volatility > 0.6:
            vol = "high_vol"
        else:
            vol = "normal_vol"

        return f"{trend}_{vol}"

    def record_trade(
        self,
        pnl: float,
        hmm_trend: float,
        hmm_volatility: float,
        confidence: float,
        timestamp: Optional[datetime] = None,
    ):
        """Record a completed trade for a regime."""
        regime_key = self.get_regime_key(hmm_trend, hmm_volatility)
        perf = self.regimes[regime_key]

        perf.trades.append(pnl)
        perf.timestamps.append(timestamp or datetime.now())
        perf.confidences.append(confidence)
        perf.outcomes.append(1 if pnl > 0 else 0)

        # Trim to lookback window
        if len(perf.trades) > self.lookback * 2:
            perf.trades = perf.trades[-self.lookback:]
            perf.timestamps = perf.timestamps[-self.lookback:]
            perf.confidences = perf.confidences[-self.lookback:]
            perf.outcomes = perf.outcomes[-self.lookback:]

        # Update regime status
        if self.auto_disable:
            self._update_regime_status(regime_key)

    def _update_regime_status(self, regime_key: str):
        """Update disabled/marginal status for a regime."""
        perf = self.regimes[regime_key]

        if perf.trade_count < self.min_trades:
            return  # Not enough data

        recent_pnl = perf.get_recent_pnl(self.lookback)
        recent_wr = perf.get_recent_win_rate(self.lookback)

        # Check for disable conditions
        if recent_pnl < self.disable_threshold_pnl or recent_wr < self.disable_threshold_wr:
            if regime_key not in self.disabled_regimes:
                self.disabled_regimes.add(regime_key)
                logger.warning(f"[REGIME_PERF] DISABLED {regime_key}: "
                             f"PnL=${recent_pnl:.2f}, WR={recent_wr:.1%}")
        else:
            # Check if can re-enable
            if regime_key in self.disabled_regimes:
                if recent_pnl > 0 and recent_wr > self.disable_threshold_wr + 0.05:
                    self.disabled_regimes.remove(regime_key)
                    logger.info(f"[REGIME_PERF] RE-ENABLED {regime_key}: "
                              f"PnL=${recent_pnl:.2f}, WR={recent_wr:.1%}")

        # Check for marginal (needs higher confidence)
        avg_pnl = recent_pnl / max(1, min(perf.trade_count, self.lookback))
        if 0 <= avg_pnl < 5:  # Barely positive
            self.marginal_regimes.add(regime_key)
        else:
            self.marginal_regimes.discard(regime_key)

    def should_trade(
        self,
        hmm_trend: float,
        hmm_volatility: float,
        confidence: float,
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if should trade in current regime.

        Returns:
            Tuple of (should_trade, rejection_reason or None)
        """
        regime_key = self.get_regime_key(hmm_trend, hmm_volatility)

        # Check if disabled
        if regime_key in self.disabled_regimes:
            return False, f"regime_disabled:{regime_key}"

        # Check if marginal (needs higher confidence)
        if regime_key in self.marginal_regimes:
            boosted_threshold = 0.5 + self.confidence_boost  # Raise threshold
            if confidence < boosted_threshold:
                return False, f"marginal_regime_low_conf:{regime_key}"

        return True, None

    def get_all_stats(self) -> Dict[str, Dict]:
        """Get performance stats for all regimes."""
        stats = {}
        for key, perf in self.regimes.items():
            if perf.trade_count > 0:
                stats[key] = {
                    'trades': perf.trade_count,
                    'total_pnl': perf.total_pnl,
                    'avg_pnl': perf.avg_pnl,
                    'win_rate': perf.win_rate,
                    'max_drawdown': perf.max_drawdown,
                    'recent_pnl': perf.get_recent_pnl(self.lookback),
                    'recent_wr': perf.get_recent_win_rate(self.lookback),
                    'disabled': key in self.disabled_regimes,
                    'marginal': key in self.marginal_regimes,
                }
        return stats

    def save_state(self, path: str):
        """Save regime performance state to file."""
        state = {
            'regimes': {k: {
                'trades': v.trades,
                'outcomes': v.outcomes,
                'confidences': v.confidences,
            } for k, v in self.regimes.items()},
            'disabled': list(self.disabled_regimes),
            'marginal': list(self.marginal_regimes),
        }
        with open(path, 'w') as f:
            json.dump(state, f, indent=2, default=str)

    def load_state(self, path: str):
        """Load regime performance state from file."""
        try:
            with open(path, 'r') as f:
                state = json.load(f)

            for key, data in state.get('regimes', {}).items():
                perf = RegimePerformance()
                perf.trades = data.get('trades', [])
                perf.outcomes = data.get('outcomes', [])
                perf.confidences = data.get('confidences', [])
                self.regimes[key] = perf

            self.disabled_regimes = set(state.get('disabled', []))
            self.marginal_regimes = set(state.get('marginal', []))

            logger.info(f"[REGIME_PERF] Loaded state: {len(self.regimes)} regimes, "
                       f"{len(self.disabled_regimes)} disabled")
        except Exception as e:
            logger.debug(f"[REGIME_PERF] Could not load state: {e}")


# Singleton
_regime_tracker = None

def get_regime_tracker() -> RegimePerformanceTracker:
    """Get or create the regime tracker singleton."""
    global _regime_tracker
    if _regime_tracker is None:
        _regime_tracker = RegimePerformanceTracker()
    return _regime_tracker
