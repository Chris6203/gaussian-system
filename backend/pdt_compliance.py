"""
PDT Compliance Module
=====================

Pattern Day Trader (PDT) rule compliance for trading accounts.

PDT Rules:
- If account < $25,000: Max 3 day trades per rolling 5 business days
- Day trade = Open and close same position in same day
- Violation can result in account restrictions

Usage:
    from backend.pdt_compliance import PDTTracker, check_pdt_compliance
"""

import logging
import sqlite3
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# Data Classes
# ------------------------------------------------------------------------------
@dataclass
class DayTrade:
    """Record of a day trade."""
    trade_id: str
    symbol: str
    trade_date: datetime
    entry_time: datetime
    exit_time: datetime
    pnl: float = 0.0


@dataclass
class PDTStatus:
    """Current PDT compliance status."""
    account_balance: float
    day_trades_5_days: int
    max_day_trades: int
    is_restricted: bool
    can_day_trade: bool
    trades_remaining: int
    is_pdt_account: bool  # True if account >= $25k
    warning_message: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            'account_balance': self.account_balance,
            'day_trades_5_days': self.day_trades_5_days,
            'max_day_trades': self.max_day_trades,
            'is_restricted': self.is_restricted,
            'can_day_trade': self.can_day_trade,
            'trades_remaining': self.trades_remaining,
            'is_pdt_account': self.is_pdt_account,
            'warning_message': self.warning_message
        }


# ------------------------------------------------------------------------------
# PDT Tracker
# ------------------------------------------------------------------------------
class PDTTracker:
    """
    Tracks Pattern Day Trader compliance.
    
    Monitors day trades and enforces PDT rules for accounts under $25,000.
    """
    
    PDT_THRESHOLD = 25000.0  # Account balance threshold
    MAX_DAY_TRADES = 3       # Max day trades in 5 days for non-PDT accounts
    ROLLING_DAYS = 5         # Rolling window for day trade count
    
    def __init__(self, db_path: str = "data/paper_trading.db", disabled: bool = False):
        """
        Initialize PDT tracker.
        
        Args:
            db_path: Path to trading database
            disabled: If True, PDT rules are not enforced
        """
        self.db_path = db_path
        self.disabled = disabled
        self.day_trades: List[DayTrade] = []
        self._ensure_table()
        self._load_recent_day_trades()
    
    def _ensure_table(self):
        """Ensure day_trades table exists."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS day_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_id TEXT UNIQUE,
                    symbol TEXT,
                    trade_date TEXT,
                    entry_time TEXT,
                    exit_time TEXT,
                    pnl REAL DEFAULT 0
                )
            """)
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"Could not create day_trades table: {e}")
    
    def _load_recent_day_trades(self):
        """Load recent day trades from database."""
        try:
            cutoff = datetime.now() - timedelta(days=self.ROLLING_DAYS)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT trade_id, symbol, trade_date, entry_time, exit_time, pnl
                FROM day_trades
                WHERE date(trade_date) >= date(?)
                ORDER BY trade_date DESC
            """, (cutoff.strftime('%Y-%m-%d'),))
            
            rows = cursor.fetchall()
            conn.close()
            
            self.day_trades = []
            for row in rows:
                self.day_trades.append(DayTrade(
                    trade_id=row[0],
                    symbol=row[1],
                    trade_date=datetime.strptime(row[2], '%Y-%m-%d') if row[2] else datetime.now(),
                    entry_time=datetime.strptime(row[3], '%Y-%m-%d %H:%M:%S') if row[3] else datetime.now(),
                    exit_time=datetime.strptime(row[4], '%Y-%m-%d %H:%M:%S') if row[4] else datetime.now(),
                    pnl=row[5] or 0.0
                ))
            
            logger.debug(f"Loaded {len(self.day_trades)} recent day trades")
            
        except Exception as e:
            logger.warning(f"Could not load day trades: {e}")
            self.day_trades = []
    
    def record_day_trade(
        self,
        trade_id: str,
        symbol: str,
        entry_time: datetime,
        exit_time: datetime,
        pnl: float = 0.0
    ):
        """
        Record a day trade.
        
        Args:
            trade_id: Unique trade identifier
            symbol: Trading symbol
            entry_time: Trade entry time
            exit_time: Trade exit time
            pnl: Profit/loss from trade
        """
        trade = DayTrade(
            trade_id=trade_id,
            symbol=symbol,
            trade_date=entry_time.date() if hasattr(entry_time, 'date') else entry_time,
            entry_time=entry_time,
            exit_time=exit_time,
            pnl=pnl
        )
        
        self.day_trades.append(trade)
        
        # Save to database
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO day_trades 
                (trade_id, symbol, trade_date, entry_time, exit_time, pnl)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                trade_id,
                symbol,
                trade.trade_date.strftime('%Y-%m-%d') if hasattr(trade.trade_date, 'strftime') else str(trade.trade_date),
                entry_time.strftime('%Y-%m-%d %H:%M:%S'),
                exit_time.strftime('%Y-%m-%d %H:%M:%S'),
                pnl
            ))
            conn.commit()
            conn.close()
            
            logger.info(f"📝 Day trade recorded: {symbol} ({trade_id})")
            
        except Exception as e:
            logger.error(f"Could not save day trade: {e}")
    
    def get_day_trade_count(self, days: int = None) -> int:
        """
        Get count of day trades in rolling window.
        
        Args:
            days: Number of days to look back (default: ROLLING_DAYS)
            
        Returns:
            Number of day trades
        """
        if days is None:
            days = self.ROLLING_DAYS
        
        cutoff = datetime.now() - timedelta(days=days)
        
        count = 0
        for trade in self.day_trades:
            trade_dt = trade.trade_date
            if hasattr(trade_dt, 'date'):
                trade_dt = datetime.combine(trade_dt.date() if hasattr(trade_dt, 'date') else trade_dt, datetime.min.time())
            elif not isinstance(trade_dt, datetime):
                continue
            
            if trade_dt >= cutoff:
                count += 1
        
        return count
    
    def get_status(self, account_balance: float) -> PDTStatus:
        """
        Get current PDT compliance status.
        
        Args:
            account_balance: Current account balance
            
        Returns:
            PDTStatus with compliance information
        """
        is_pdt_account = account_balance >= self.PDT_THRESHOLD
        day_trade_count = self.get_day_trade_count()
        
        if self.disabled:
            return PDTStatus(
                account_balance=account_balance,
                day_trades_5_days=day_trade_count,
                max_day_trades=999,
                is_restricted=False,
                can_day_trade=True,
                trades_remaining=999,
                is_pdt_account=is_pdt_account,
                warning_message="PDT tracking disabled"
            )
        
        if is_pdt_account:
            return PDTStatus(
                account_balance=account_balance,
                day_trades_5_days=day_trade_count,
                max_day_trades=999,
                is_restricted=False,
                can_day_trade=True,
                trades_remaining=999,
                is_pdt_account=True,
                warning_message=None
            )
        
        # Non-PDT account: enforce 3 day trade limit
        trades_remaining = max(0, self.MAX_DAY_TRADES - day_trade_count)
        is_restricted = day_trade_count >= self.MAX_DAY_TRADES
        can_day_trade = trades_remaining > 0
        
        warning = None
        if is_restricted:
            warning = f"PDT RESTRICTED: {day_trade_count} day trades in 5 days (max {self.MAX_DAY_TRADES})"
        elif trades_remaining == 1:
            warning = f"PDT WARNING: Only 1 day trade remaining in 5-day window"
        
        return PDTStatus(
            account_balance=account_balance,
            day_trades_5_days=day_trade_count,
            max_day_trades=self.MAX_DAY_TRADES,
            is_restricted=is_restricted,
            can_day_trade=can_day_trade,
            trades_remaining=trades_remaining,
            is_pdt_account=False,
            warning_message=warning
        )
    
    def check_compliance(self, account_balance: float) -> Tuple[bool, str]:
        """
        Check if a day trade is allowed.
        
        Args:
            account_balance: Current account balance
            
        Returns:
            Tuple of (is_allowed, reason)
        """
        status = self.get_status(account_balance)
        
        if status.can_day_trade:
            return True, "OK"
        else:
            return False, status.warning_message or "PDT limit reached"
    
    def is_day_trade(self, entry_time: datetime, exit_time: datetime) -> bool:
        """
        Check if a round trip qualifies as a day trade.
        
        A day trade is opening and closing the same position in the same day.
        
        Args:
            entry_time: Position entry time
            exit_time: Position exit time
            
        Returns:
            True if this is a day trade
        """
        if entry_time is None or exit_time is None:
            return False
        
        entry_date = entry_time.date() if hasattr(entry_time, 'date') else entry_time
        exit_date = exit_time.date() if hasattr(exit_time, 'date') else exit_time
        
        return entry_date == exit_date
    
    def clean_old_trades(self, days: int = 10):
        """Remove day trades older than specified days."""
        cutoff = datetime.now() - timedelta(days=days)
        
        self.day_trades = [
            t for t in self.day_trades 
            if hasattr(t.trade_date, 'date') and t.trade_date >= cutoff
        ]
        
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("""
                DELETE FROM day_trades 
                WHERE date(trade_date) < date(?)
            """, (cutoff.strftime('%Y-%m-%d'),))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"Could not clean old day trades: {e}")


# ------------------------------------------------------------------------------
# Convenience Functions
# ------------------------------------------------------------------------------
def check_pdt_compliance(
    account_balance: float,
    db_path: str = "data/paper_trading.db"
) -> Tuple[bool, str]:
    """
    Quick check for PDT compliance.
    
    Args:
        account_balance: Current account balance
        db_path: Path to trading database
        
    Returns:
        Tuple of (can_day_trade, message)
    """
    tracker = PDTTracker(db_path)
    return tracker.check_compliance(account_balance)


def get_pdt_status(
    account_balance: float,
    db_path: str = "data/paper_trading.db"
) -> dict:
    """
    Get PDT status as dictionary.
    
    Args:
        account_balance: Current account balance
        db_path: Path to trading database
        
    Returns:
        Dictionary with PDT status
    """
    tracker = PDTTracker(db_path)
    return tracker.get_status(account_balance).to_dict()


__all__ = [
    'DayTrade',
    'PDTStatus',
    'PDTTracker',
    'check_pdt_compliance',
    'get_pdt_status',
]








