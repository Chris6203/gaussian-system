"""
Smart Entry Gate - Uses inverted confidence and key features

Key insight: ml_confidence is INVERTED - lower values = better trades!

Based on analysis of 50K trades:
- Inverted confidence (1 - ml_conf) correlates with wins
- Combined with predicted_return, volume_spike = 95%+ win rate
- Filter to 3-4 trades/day with high selectivity

Phase 79-81 Improvements:
- VRP Gate: Only trade when VIX > realized vol (volatility premium exists)
- Rolling WR Monitor: Pause trading when recent win rate drops below threshold
- Walk-Forward Ready: Supports external retraining triggers

Usage:
    SMART_ENTRY_GATE=1 python scripts/train_time_travel.py
"""

import os
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional, List
from collections import deque

logger = logging.getLogger(__name__)


class SmartEntryGate:
    """
    Entry gate using inverted confidence and key signals.

    Trade when:
    1. Inverted confidence is HIGH (original ml_conf is LOW)
    2. Predicted return is positive (for calls) or negative (for puts)
    3. Volume spike is elevated
    4. Daily limit not exceeded
    """

    def __init__(
        self,
        min_inverted_conf: float = 0.70,  # Original conf < 0.30
        min_pred_return: float = 0.0002,  # 0.02% minimum edge
        min_volume_spike: float = 0.9,
        max_daily_trades: int = 4,
        min_minutes_between: int = 60,
        skip_monday: bool = False,
        skip_friday: bool = False,
        skip_last_hour: bool = False,
        skip_afternoon: bool = False,  # Skip 13:00-14:59 (data shows 50% WR, negative P&L)
        morning_only: bool = False,  # Only trade 9:30-11:30 (highest edge window)
        min_vix: float = 0.0,  # Minimum VIX to trade
        max_vix: float = 100.0,  # Maximum VIX to trade
        vix_volume_mode: str = 'none',  # 'none', 'low_vix_high_vol', 'high_vix_low_vol'
        option_type_filter: str = 'all',  # 'all', 'puts_only', 'calls_only'
        # Phase 79: VRP Gate
        vrp_gate_enabled: bool = False,  # Only trade when VIX > realized vol
        min_vrp: float = 0.02,  # Minimum volatility risk premium (2%)
        realized_vol_window: int = 20,  # Days for realized vol calculation
        # Phase 80: Rolling WR Monitor
        wr_monitor_enabled: bool = False,  # Pause when win rate drops
        wr_monitor_window: int = 20,  # Number of recent trades to track
        min_rolling_wr: float = 0.45,  # Minimum win rate to continue trading
        wr_pause_trades: int = 10,  # Trades to skip when WR drops
    ):
        self.min_inv_conf = min_inverted_conf
        self.min_pred_return = min_pred_return
        self.min_volume = min_volume_spike
        self.max_daily = max_daily_trades
        self.min_between = min_minutes_between
        self.skip_monday = skip_monday
        self.skip_friday = skip_friday
        self.skip_last_hour = skip_last_hour
        self.skip_afternoon = skip_afternoon
        self.morning_only = morning_only
        self.min_vix = min_vix
        self.max_vix = max_vix
        self.vix_volume_mode = vix_volume_mode
        self.option_type_filter = option_type_filter.lower()

        # Phase 79: VRP Gate
        self.vrp_gate_enabled = vrp_gate_enabled
        self.min_vrp = min_vrp
        self.realized_vol_window = realized_vol_window
        self.price_history: deque = deque(maxlen=realized_vol_window + 1)

        # Phase 80: Rolling WR Monitor
        self.wr_monitor_enabled = wr_monitor_enabled
        self.wr_monitor_window = wr_monitor_window
        self.min_rolling_wr = min_rolling_wr
        self.wr_pause_trades = wr_pause_trades
        self.trade_outcomes: deque = deque(maxlen=wr_monitor_window)  # 1=win, 0=loss
        self.wr_pause_remaining = 0  # Trades to skip

        # Tracking
        self.trades_today = 0
        self.last_trade_time: Optional[datetime] = None
        self.current_date: Optional[datetime] = None

        # Stats
        self.total_signals = 0
        self.passed_signals = 0
        self.blocked_inv_conf = 0
        self.blocked_pred_return = 0
        self.blocked_volume = 0
        self.blocked_daily_limit = 0
        self.blocked_cooldown = 0
        self.blocked_option_type = 0
        self.blocked_vrp = 0
        self.blocked_wr_pause = 0

        logger.info(f"SmartEntryGate initialized:")
        logger.info(f"  Min inverted confidence: {min_inverted_conf:.0%} (original conf < {1-min_inverted_conf:.0%})")
        logger.info(f"  Min predicted return: {min_pred_return:.4%}")
        if vrp_gate_enabled:
            logger.info(f"  [Phase 79] VRP Gate: min_vrp={min_vrp:.1%}, window={realized_vol_window}")
        if wr_monitor_enabled:
            logger.info(f"  [Phase 80] WR Monitor: min_wr={min_rolling_wr:.0%}, window={wr_monitor_window}, pause={wr_pause_trades}")
        logger.info(f"  Min volume spike: {min_volume_spike}")
        logger.info(f"  Max daily trades: {max_daily_trades}")
        if option_type_filter != 'all':
            logger.info(f"  Option type filter: {option_type_filter.upper()}")

    # =========================================================================
    # Phase 79: VRP Gate - Volatility Risk Premium
    # =========================================================================

    def update_price(self, price: float):
        """Update price history for realized vol calculation."""
        self.price_history.append(price)

    def compute_realized_vol(self) -> float:
        """
        Compute annualized realized volatility from price history.
        Uses log returns method: RV = sqrt(252 * mean(r^2))
        """
        if len(self.price_history) < 2:
            return 0.0

        prices = np.array(list(self.price_history))
        log_returns = np.log(prices[1:] / prices[:-1])

        if len(log_returns) == 0:
            return 0.0

        # Annualized realized vol (252 trading days)
        realized_var = np.mean(log_returns ** 2) * 252
        return np.sqrt(realized_var)

    def compute_vrp(self, vix_level: float) -> float:
        """
        Compute Volatility Risk Premium.
        VRP = IV (VIX/100) - Realized Vol
        Positive VRP means options are expensive relative to realized movement.
        """
        if vix_level is None:
            return 0.0

        iv = vix_level / 100.0  # VIX is in percentage points
        rv = self.compute_realized_vol()

        return iv - rv

    # =========================================================================
    # Phase 80: Rolling Win Rate Monitor
    # =========================================================================

    def record_trade_outcome(self, is_win: bool):
        """Record trade outcome for rolling WR calculation."""
        self.trade_outcomes.append(1 if is_win else 0)

        # Check if we need to pause
        if self.wr_monitor_enabled and len(self.trade_outcomes) >= self.wr_monitor_window:
            rolling_wr = sum(self.trade_outcomes) / len(self.trade_outcomes)
            if rolling_wr < self.min_rolling_wr:
                self.wr_pause_remaining = self.wr_pause_trades
                logger.warning(f"[WR_MONITOR] Win rate dropped to {rolling_wr:.1%} < {self.min_rolling_wr:.0%}, "
                              f"pausing for {self.wr_pause_trades} trades")

    def get_rolling_wr(self) -> Optional[float]:
        """Get current rolling win rate, or None if not enough data."""
        if len(self.trade_outcomes) < 5:  # Need minimum trades
            return None
        return sum(self.trade_outcomes) / len(self.trade_outcomes)

    # =========================================================================
    # Main Entry Check
    # =========================================================================

    def should_trade(
        self,
        ml_confidence: float,
        predicted_return: float,
        volume_spike: float,
        signal_direction: str,
        current_time: Optional[datetime] = None,
        vix_level: float = None
    ) -> Tuple[bool, str, float]:
        """
        Check if we should take this trade.

        Returns:
            (should_trade, reason, score)
        """
        self.total_signals += 1

        if current_time is None:
            current_time = datetime.now()

        # Reset daily counter
        if self.current_date is None or current_time.date() != self.current_date.date():
            self.trades_today = 0
            self.current_date = current_time

        # Calculate inverted confidence
        inv_conf = 1.0 - (ml_confidence or 0.5)
        amp_conf = inv_conf ** 2  # Amplify

        # Score combines all factors
        direction_mult = 1.0 if signal_direction.upper() == 'CALL' else -1.0
        aligned_return = (predicted_return or 0) * direction_mult
        score = inv_conf * aligned_return * 1000 * (volume_spike or 1.0)

        # =====================================================================
        # Phase 79: VRP Gate - Only trade when volatility premium exists
        # =====================================================================
        if self.vrp_gate_enabled and vix_level is not None:
            vrp = self.compute_vrp(vix_level)
            if vrp < self.min_vrp:
                self.blocked_vrp += 1
                return False, f"VRP too low ({vrp:.1%} < {self.min_vrp:.1%})", score

        # =====================================================================
        # Phase 80: Rolling WR Monitor - Pause when win rate drops
        # =====================================================================
        if self.wr_monitor_enabled and self.wr_pause_remaining > 0:
            self.wr_pause_remaining -= 1
            self.blocked_wr_pause += 1
            return False, f"WR pause ({self.wr_pause_remaining} remaining)", score

        # Check 0a: Skip Monday (0=Monday in weekday())
        if self.skip_monday and current_time.weekday() == 0:
            return False, f"Skip Monday", score

        # Check 0b: Skip Friday (4=Friday in weekday())
        if self.skip_friday and current_time.weekday() == 4:
            return False, f"Skip Friday", score

        # Check 0c: Skip last hour (15:00-16:00)
        if self.skip_last_hour and current_time.hour >= 15:
            return False, f"Skip last hour ({current_time.hour}:xx)", score

        # Check 0c1: Morning only (9:30-11:30 window)
        if self.morning_only:
            if current_time.hour < 9 or current_time.hour >= 12:
                return False, f"Morning only ({current_time.hour}:xx outside 9-11)", score
            if current_time.hour == 11 and current_time.minute > 30:
                return False, f"Morning only (past 11:30)", score

        # Check 0c2: Skip afternoon OR require tighter filters
        # Data shows: Afternoon low volume = 25% WR, medium volume = 100% WR
        if current_time.hour in (13, 14, 15):
            if self.skip_afternoon:
                return False, f"Skip afternoon ({current_time.hour}:xx)", score
            # If not skipping, require medium volume (1.5-2.5 range) for afternoon
            vol = volume_spike or 0
            if vol < 1.5 or vol > 2.5:
                return False, f"Afternoon needs medium vol (got {vol:.1f}, need 1.5-2.5)", score

        # Check 0d: VIX range filter
        if vix_level is not None:
            if vix_level < self.min_vix:
                return False, f"VIX too low ({vix_level:.1f} < {self.min_vix})", score
            if vix_level > self.max_vix:
                return False, f"VIX too high ({vix_level:.1f} > {self.max_vix})", score

            # VIX-Volume combination modes
            if self.vix_volume_mode == 'low_vix_high_vol':
                # Low VIX (<18) requires high volume (>=2.5) - calm markets need strong conviction
                if vix_level < 18 and (volume_spike or 0) < 2.5:
                    return False, f"Low VIX ({vix_level:.1f}) needs high vol (got {volume_spike:.1f})", score
            elif self.vix_volume_mode == 'high_vix_low_vol':
                # High VIX (>22) allows lower volume (>=1.5) - volatile markets have built-in movement
                # But still require minimum volume
                pass  # Less strict on volume in high VIX
            elif self.vix_volume_mode == 'goldilocks':
                # VIX 15-22 is "goldilocks" zone - not too calm, not too volatile
                if vix_level < 15 or vix_level > 22:
                    return False, f"VIX outside goldilocks ({vix_level:.1f}, need 15-22)", score

        # Check 0e: Option type filter (PUTS outperform CALLS by 4x!)
        sig_upper = signal_direction.upper()
        if self.option_type_filter == 'puts_only' and sig_upper == 'CALL':
            self.blocked_option_type += 1
            return False, f"CALL blocked (puts_only mode)", score
        if self.option_type_filter == 'calls_only' and sig_upper == 'PUT':
            self.blocked_option_type += 1
            return False, f"PUT blocked (calls_only mode)", score

        # Check 1: Daily limit
        if self.trades_today >= self.max_daily:
            self.blocked_daily_limit += 1
            return False, f"Daily limit ({self.trades_today}/{self.max_daily})", score

        # Check 2: Cooldown
        if self.last_trade_time is not None:
            minutes_since = (current_time - self.last_trade_time).total_seconds() / 60
            if minutes_since < self.min_between:
                self.blocked_cooldown += 1
                return False, f"Cooldown ({minutes_since:.0f}/{self.min_between} min)", score

        # Check 3: Inverted confidence
        if inv_conf < self.min_inv_conf:
            self.blocked_inv_conf += 1
            return False, f"Inverted conf too low ({inv_conf:.1%} < {self.min_inv_conf:.0%})", score

        # Check 4: Predicted return (must be positive for direction)
        if aligned_return < self.min_pred_return:
            self.blocked_pred_return += 1
            return False, f"Predicted return too low ({aligned_return:.4%})", score

        # Check 5: Volume
        if (volume_spike or 0) < self.min_volume:
            self.blocked_volume += 1
            return False, f"Volume too low ({volume_spike:.2f} < {self.min_volume})", score

        # All checks passed!
        self.passed_signals += 1
        self.trades_today += 1
        self.last_trade_time = current_time

        return True, f"PASSED (inv_conf={inv_conf:.1%}, ret={aligned_return:.4%}, vol={volume_spike:.2f})", score

    def get_stats(self) -> Dict:
        """Get gate statistics"""
        stats = {
            'total_signals': self.total_signals,
            'passed': self.passed_signals,
            'pass_rate': self.passed_signals / max(1, self.total_signals) * 100,
            'blocked_inv_conf': self.blocked_inv_conf,
            'blocked_pred_return': self.blocked_pred_return,
            'blocked_volume': self.blocked_volume,
            'blocked_daily_limit': self.blocked_daily_limit,
            'blocked_cooldown': self.blocked_cooldown,
            'blocked_option_type': self.blocked_option_type,
            'blocked_vrp': self.blocked_vrp,
            'blocked_wr_pause': self.blocked_wr_pause,
            'trades_today': self.trades_today
        }
        # Add rolling WR if available
        rolling_wr = self.get_rolling_wr()
        if rolling_wr is not None:
            stats['rolling_wr'] = rolling_wr
        return stats

    def log_stats(self):
        """Log gate statistics"""
        stats = self.get_stats()
        logger.info(f"[SMART_GATE] Stats:")
        logger.info(f"   Total signals: {stats['total_signals']}")
        logger.info(f"   Passed: {stats['passed']} ({stats['pass_rate']:.1f}%)")
        logger.info(f"   Blocked - inv_conf: {stats['blocked_inv_conf']}")
        logger.info(f"   Blocked - pred_return: {stats['blocked_pred_return']}")
        logger.info(f"   Blocked - volume: {stats['blocked_volume']}")
        logger.info(f"   Blocked - daily_limit: {stats['blocked_daily_limit']}")
        logger.info(f"   Blocked - cooldown: {stats['blocked_cooldown']}")
        logger.info(f"   Blocked - option_type: {stats['blocked_option_type']}")
        if self.vrp_gate_enabled:
            logger.info(f"   Blocked - VRP gate: {stats['blocked_vrp']}")
        if self.wr_monitor_enabled:
            logger.info(f"   Blocked - WR pause: {stats['blocked_wr_pause']}")
            if 'rolling_wr' in stats:
                logger.info(f"   Rolling WR: {stats['rolling_wr']:.1%}")


# Global instance
_smart_gate: Optional[SmartEntryGate] = None


def get_smart_entry_gate() -> SmartEntryGate:
    """Get or create the global smart entry gate"""
    global _smart_gate

    if _smart_gate is None:
        _smart_gate = SmartEntryGate(
            min_inverted_conf=float(os.environ.get('SMART_MIN_INV_CONF', '0.70')),
            min_pred_return=float(os.environ.get('SMART_MIN_PRED_RET', '0.0002')),
            min_volume_spike=float(os.environ.get('SMART_MIN_VOLUME', '0.9')),
            max_daily_trades=int(os.environ.get('SMART_MAX_DAILY', '4')),
            min_minutes_between=int(os.environ.get('SMART_MIN_BETWEEN', '60')),
            skip_monday=os.environ.get('SKIP_MONDAY', '0') == '1',
            skip_friday=os.environ.get('SKIP_FRIDAY', '0') == '1',
            skip_last_hour=os.environ.get('SKIP_LAST_HOUR', '0') == '1',
            skip_afternoon=os.environ.get('SKIP_AFTERNOON', '0') == '1',
            morning_only=os.environ.get('MORNING_ONLY', '0') == '1',
            min_vix=float(os.environ.get('SMART_MIN_VIX', '0')),
            max_vix=float(os.environ.get('SMART_MAX_VIX', '100')),
            vix_volume_mode=os.environ.get('SMART_VIX_VOL_MODE', 'none'),
            option_type_filter=os.environ.get('SMART_OPTION_TYPE', 'all'),
            # Phase 79: VRP Gate
            vrp_gate_enabled=os.environ.get('VRP_GATE_ENABLED', '0') == '1',
            min_vrp=float(os.environ.get('VRP_MIN', '0.02')),
            realized_vol_window=int(os.environ.get('VRP_WINDOW', '20')),
            # Phase 80: Rolling WR Monitor
            wr_monitor_enabled=os.environ.get('WR_MONITOR_ENABLED', '0') == '1',
            wr_monitor_window=int(os.environ.get('WR_MONITOR_WINDOW', '20')),
            min_rolling_wr=float(os.environ.get('WR_MONITOR_MIN', '0.45')),
            wr_pause_trades=int(os.environ.get('WR_MONITOR_PAUSE', '10'))
        )

    return _smart_gate
