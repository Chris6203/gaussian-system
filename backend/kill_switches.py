#!/usr/bin/env python3
"""
Kill Switches - Deterministic, Non-Overridable Halt Mechanisms

Jerry's "Kill Switches" feature:
- HARD halts that CANNOT be overridden by any other system
- Triggered by consecutive losses, daily drawdown, etc.
- Once triggered, trading stops until manually reset or new day

From Jerry's PDF:
    "Modus Ponens (MP) Decision Framework:
     Trade if and only if ALL constraints satisfied.
     Any single veto → no trade."

Usage:
    from backend.kill_switches import check_kill_switches, KillSwitchManager

    blocked, reason = check_kill_switches()
    if blocked:
        logger.warning(f"KILL SWITCH ACTIVE: {reason}")
        return HOLD
"""

import os
import json
import logging
from datetime import datetime, date, timedelta
from typing import Dict, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class KillSwitchManager:
    """
    Manages deterministic kill switches that halt trading.

    Kill switches are NON-OVERRIDABLE - once triggered, they stay
    active until reset conditions are met.

    Switches:
    1. Consecutive Losses: N losses in a row → halt
    2. Daily Drawdown: P&L < -X% for the day → halt until tomorrow
    3. Weekly Drawdown: P&L < -Y% for the week → halt until next week
    4. Max Daily Trades: Too many trades in one day → halt
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize kill switch manager.

        Config options (jerry.kill_switches section):
            enabled: bool - Enable/disable kill switches
            max_consecutive_losses: int - Halt after N consecutive losses
            daily_drawdown_limit_pct: float - Halt if daily P&L < this %
            weekly_drawdown_limit_pct: float - Halt if weekly P&L < this %
            max_daily_trades: int - Max trades per day (0 = unlimited)
        """
        self.config = config or {}

        # Load from jerry config section
        jerry_cfg = self.config.get('jerry', {}).get('kill_switches', {})

        self.enabled = jerry_cfg.get('enabled', False)
        self.max_consecutive_losses = jerry_cfg.get('max_consecutive_losses', 5)
        self.daily_drawdown_limit = jerry_cfg.get('daily_drawdown_limit_pct', -15.0)
        self.weekly_drawdown_limit = jerry_cfg.get('weekly_drawdown_limit_pct', -20.0)
        self.max_daily_trades = jerry_cfg.get('max_daily_trades', 0)  # 0 = unlimited

        # State tracking
        self._consecutive_losses = 0
        self._daily_pnl = 0.0
        self._weekly_pnl = 0.0
        self._daily_trades = 0
        self._current_day = None
        self._current_week = None

        # Kill switch status
        self._consecutive_loss_halt = False
        self._daily_drawdown_halt = False
        self._weekly_drawdown_halt = False
        self._daily_trades_halt = False

        logger.info(f"Kill Switches initialized (enabled={self.enabled})")
        if self.enabled:
            logger.info(f"   Max consecutive losses: {self.max_consecutive_losses}")
            logger.info(f"   Daily drawdown limit: {self.daily_drawdown_limit}%")
            logger.info(f"   Weekly drawdown limit: {self.weekly_drawdown_limit}%")
            if self.max_daily_trades > 0:
                logger.info(f"   Max daily trades: {self.max_daily_trades}")

    def update_date(self, current_date: datetime):
        """Update date tracking - resets daily/weekly counters if needed."""
        today = current_date.date() if isinstance(current_date, datetime) else current_date
        week_num = today.isocalendar()[1]

        # Reset daily counters on new day
        if self._current_day != today:
            self._current_day = today
            self._daily_pnl = 0.0
            self._daily_trades = 0
            self._daily_drawdown_halt = False
            self._daily_trades_halt = False
            logger.debug(f"Kill switches: new day {today}, daily counters reset")

        # Reset weekly counters on new week
        if self._current_week != week_num:
            self._current_week = week_num
            self._weekly_pnl = 0.0
            self._weekly_drawdown_halt = False
            logger.debug(f"Kill switches: new week {week_num}, weekly counters reset")

    def record_trade_result(self, pnl_pct: float, is_win: bool):
        """Record a trade result for kill switch tracking."""
        if not self.enabled:
            return

        # Update P&L tracking
        self._daily_pnl += pnl_pct
        self._weekly_pnl += pnl_pct
        self._daily_trades += 1

        # Update consecutive losses
        if is_win:
            self._consecutive_losses = 0
            self._consecutive_loss_halt = False
        else:
            self._consecutive_losses += 1
            if self._consecutive_losses >= self.max_consecutive_losses:
                self._consecutive_loss_halt = True
                logger.warning(
                    f"KILL SWITCH: {self._consecutive_losses} consecutive losses"
                )

        # Check daily drawdown
        if self._daily_pnl <= self.daily_drawdown_limit:
            self._daily_drawdown_halt = True
            logger.warning(
                f"KILL SWITCH: Daily drawdown {self._daily_pnl:.1f}% <= {self.daily_drawdown_limit}%"
            )

        # Check weekly drawdown
        if self._weekly_pnl <= self.weekly_drawdown_limit:
            self._weekly_drawdown_halt = True
            logger.warning(
                f"KILL SWITCH: Weekly drawdown {self._weekly_pnl:.1f}% <= {self.weekly_drawdown_limit}%"
            )

        # Check daily trade limit
        if self.max_daily_trades > 0 and self._daily_trades >= self.max_daily_trades:
            self._daily_trades_halt = True
            logger.warning(
                f"KILL SWITCH: Daily trade limit {self._daily_trades} >= {self.max_daily_trades}"
            )

    def is_halted(self) -> Tuple[bool, Optional[str]]:
        """
        Check if any kill switch is active.

        Returns:
            (is_halted, reason) - True if trading should be halted
        """
        if not self.enabled:
            return False, None

        if self._consecutive_loss_halt:
            return True, f"consecutive_losses ({self._consecutive_losses})"

        if self._daily_drawdown_halt:
            return True, f"daily_drawdown ({self._daily_pnl:.1f}%)"

        if self._weekly_drawdown_halt:
            return True, f"weekly_drawdown ({self._weekly_pnl:.1f}%)"

        if self._daily_trades_halt:
            return True, f"daily_trade_limit ({self._daily_trades})"

        return False, None

    def reset_consecutive_losses(self):
        """Manually reset consecutive loss counter (e.g., after a pause)."""
        self._consecutive_losses = 0
        self._consecutive_loss_halt = False
        logger.info("Kill switch: consecutive loss counter reset")

    def get_status(self) -> Dict:
        """Get current kill switch status."""
        return {
            'enabled': self.enabled,
            'consecutive_losses': self._consecutive_losses,
            'daily_pnl': self._daily_pnl,
            'weekly_pnl': self._weekly_pnl,
            'daily_trades': self._daily_trades,
            'halts': {
                'consecutive_loss': self._consecutive_loss_halt,
                'daily_drawdown': self._daily_drawdown_halt,
                'weekly_drawdown': self._weekly_drawdown_halt,
                'daily_trades': self._daily_trades_halt
            }
        }


# =============================================================================
# Global Manager Instance
# =============================================================================

_manager_instance: Optional[KillSwitchManager] = None


def get_kill_switch_manager(config: Optional[Dict] = None) -> KillSwitchManager:
    """Get or create the global kill switch manager."""
    global _manager_instance

    if _manager_instance is None:
        # Try to load config from file
        if config is None:
            config_path = Path('config.json')
            if config_path.exists():
                try:
                    with open(config_path) as f:
                        config = json.load(f)
                except Exception as e:
                    logger.warning(f"Could not load config.json: {e}")
                    config = {}

        _manager_instance = KillSwitchManager(config)

    return _manager_instance


def check_kill_switches(current_time: Optional[datetime] = None) -> Tuple[bool, Optional[str]]:
    """
    Quick check if any kill switch is active.

    Usage:
        halted, reason = check_kill_switches()
        if halted:
            logger.warning(f"KILL SWITCH: {reason}")
            return HOLD

    Returns:
        (is_halted, reason) - True if trading should be halted
    """
    manager = get_kill_switch_manager()

    # Update date tracking
    if current_time:
        manager.update_date(current_time)

    return manager.is_halted()


def record_trade(pnl_pct: float, is_win: bool):
    """Record a trade result for kill switch tracking."""
    manager = get_kill_switch_manager()
    manager.record_trade_result(pnl_pct, is_win)


def reset_manager():
    """Reset the global manager instance (for testing)."""
    global _manager_instance
    _manager_instance = None
