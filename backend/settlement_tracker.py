"""
Settlement Tracking Module
==========================

Handles T+1 settlement for options trading.

Options settle T+1 (next business day), meaning:
- Funds from closed trades aren't immediately available
- Buying power calculation must account for pending settlements

Usage:
    from backend.settlement_tracker import SettlementTracker, Settlement
"""

import logging
import sqlite3
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# Data Classes
# ------------------------------------------------------------------------------
@dataclass
class Settlement:
    """Pending settlement record."""
    trade_id: str
    amount: float
    settlement_date: datetime
    created_at: datetime = None
    is_complete: bool = False
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
    
    @property
    def is_due(self) -> bool:
        """Check if settlement is due (settlement date has passed)."""
        return datetime.now() >= self.settlement_date


# ------------------------------------------------------------------------------
# Settlement Tracker
# ------------------------------------------------------------------------------
class SettlementTracker:
    """
    Tracks T+1 settlement for options trades.
    
    When a position is closed, the proceeds aren't immediately available.
    This tracker manages pending settlements and calculates available funds.
    """
    
    def __init__(self, db_path: str = "data/paper_trading.db", simulate_settlement: bool = True):
        """
        Initialize settlement tracker.
        
        Args:
            db_path: Path to database
            simulate_settlement: If True, enforce T+1 settlement
        """
        self.db_path = db_path
        self.simulate_settlement = simulate_settlement
        self.pending_settlements: List[Settlement] = []
        self._ensure_table()
        self._load_pending()
    
    def _ensure_table(self):
        """Ensure settlements table exists."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS settlements (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_id TEXT UNIQUE,
                    amount REAL,
                    settlement_date TEXT,
                    created_at TEXT,
                    is_complete INTEGER DEFAULT 0
                )
            """)
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"Could not create settlements table: {e}")
    
    def _load_pending(self):
        """Load pending settlements from database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT trade_id, amount, settlement_date, created_at
                FROM settlements
                WHERE is_complete = 0
            """)
            
            rows = cursor.fetchall()
            conn.close()
            
            self.pending_settlements = []
            for row in rows:
                self.pending_settlements.append(Settlement(
                    trade_id=row[0],
                    amount=row[1],
                    settlement_date=datetime.strptime(row[2], '%Y-%m-%d %H:%M:%S') if row[2] else datetime.now(),
                    created_at=datetime.strptime(row[3], '%Y-%m-%d %H:%M:%S') if row[3] else datetime.now(),
                    is_complete=False
                ))
            
            logger.debug(f"Loaded {len(self.pending_settlements)} pending settlements")
            
        except Exception as e:
            logger.warning(f"Could not load settlements: {e}")
            self.pending_settlements = []
    
    def get_next_business_day(self, from_date: datetime = None) -> datetime:
        """
        Get the next business day (T+1).
        
        Skips weekends. Does not account for holidays.
        
        Args:
            from_date: Starting date (default: now)
            
        Returns:
            Next business day datetime
        """
        if from_date is None:
            from_date = datetime.now()
        
        next_day = from_date + timedelta(days=1)
        
        # Skip weekends
        while next_day.weekday() >= 5:  # Saturday = 5, Sunday = 6
            next_day += timedelta(days=1)
        
        # Set to market close time (4:00 PM)
        return next_day.replace(hour=16, minute=0, second=0, microsecond=0)
    
    def add_settlement(self, trade_id: str, amount: float, trade_close_time: datetime = None) -> Settlement:
        """
        Add a pending settlement.
        
        Args:
            trade_id: Trade identifier
            amount: Settlement amount (proceeds from closing)
            trade_close_time: Time trade was closed (default: now)
            
        Returns:
            Created Settlement object
        """
        if trade_close_time is None:
            trade_close_time = datetime.now()
        
        settlement_date = self.get_next_business_day(trade_close_time)
        
        settlement = Settlement(
            trade_id=trade_id,
            amount=amount,
            settlement_date=settlement_date,
            created_at=trade_close_time
        )
        
        self.pending_settlements.append(settlement)
        
        # Save to database
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO settlements 
                (trade_id, amount, settlement_date, created_at, is_complete)
                VALUES (?, ?, ?, ?, 0)
            """, (
                trade_id,
                amount,
                settlement_date.strftime('%Y-%m-%d %H:%M:%S'),
                trade_close_time.strftime('%Y-%m-%d %H:%M:%S')
            ))
            conn.commit()
            conn.close()
            
            logger.info(f"📅 Settlement pending: ${amount:,.2f} for {trade_id}, settles {settlement_date.strftime('%Y-%m-%d')}")
            
        except Exception as e:
            logger.error(f"Could not save settlement: {e}")
        
        return settlement
    
    def process_settlements(self, current_time: datetime = None) -> float:
        """
        Process due settlements and return total settled amount.
        
        Args:
            current_time: Current time for settlement checking
            
        Returns:
            Total amount that settled
        """
        if current_time is None:
            current_time = datetime.now()
        
        total_settled = 0.0
        settled_ids = []
        
        for settlement in self.pending_settlements:
            if current_time >= settlement.settlement_date:
                total_settled += settlement.amount
                settled_ids.append(settlement.trade_id)
                settlement.is_complete = True
                
                logger.info(f"✅ Settlement complete: ${settlement.amount:,.2f} from {settlement.trade_id}")
        
        # Mark complete in database
        if settled_ids:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                for trade_id in settled_ids:
                    cursor.execute("""
                        UPDATE settlements SET is_complete = 1 WHERE trade_id = ?
                    """, (trade_id,))
                conn.commit()
                conn.close()
            except Exception as e:
                logger.error(f"Could not update settlement status: {e}")
        
        # Remove completed from pending list
        self.pending_settlements = [s for s in self.pending_settlements if not s.is_complete]
        
        return total_settled
    
    @property
    def pending_amount(self) -> float:
        """Total amount pending settlement."""
        return sum(s.amount for s in self.pending_settlements)
    
    @property
    def pending_count(self) -> int:
        """Number of pending settlements."""
        return len(self.pending_settlements)
    
    def get_settled_balance(self, total_balance: float) -> float:
        """
        Get balance that's actually settled (available for trading).
        
        Args:
            total_balance: Total account balance including unsettled
            
        Returns:
            Settled (available) balance
        """
        if not self.simulate_settlement:
            return total_balance
        
        return total_balance - self.pending_amount
    
    def get_status(self) -> dict:
        """Get settlement status summary."""
        return {
            'pending_count': self.pending_count,
            'pending_amount': self.pending_amount,
            'settlements': [
                {
                    'trade_id': s.trade_id,
                    'amount': s.amount,
                    'settlement_date': s.settlement_date.isoformat(),
                    'is_due': s.is_due
                }
                for s in self.pending_settlements
            ]
        }
    
    def clear_completed(self, days_old: int = 7):
        """Clear completed settlements older than specified days."""
        try:
            cutoff = datetime.now() - timedelta(days=days_old)
            conn = sqlite3.connect(self.db_path)
            conn.execute("""
                DELETE FROM settlements 
                WHERE is_complete = 1 AND date(settlement_date) < date(?)
            """, (cutoff.strftime('%Y-%m-%d'),))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"Could not clean old settlements: {e}")


__all__ = [
    'Settlement',
    'SettlementTracker',
]








