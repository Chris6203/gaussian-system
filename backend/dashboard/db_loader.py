#!/usr/bin/env python3
"""
Database Loader
===============

Loads data from SQLite databases for dashboard display.
Handles paper trading DB and historical data DB.
"""

import sqlite3
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

from backend.dashboard.state import TrainingState, PositionDetail, TradeRecord
from backend.dashboard.pricing import OptionPricing

# Import centralized time utilities
try:
    from backend.time_utils import get_market_time, format_timestamp, parse_timestamp
except ImportError:
    # Fallback if time_utils not available
    import pytz
    def get_market_time():
        market_tz = pytz.timezone('US/Eastern')
        return pytz.UTC.localize(datetime.utcnow()).astimezone(market_tz).replace(tzinfo=None)
    def format_timestamp(dt):
        return dt.strftime('%Y-%m-%dT%H:%M:%S') if dt else ""
    def parse_timestamp(s):
        return datetime.fromisoformat(s.replace(' ', 'T')) if s else None

logger = logging.getLogger(__name__)


class DatabaseLoader:
    """
    Loads data from databases for dashboard display.
    
    Handles:
    - Paper trading database (trades, positions, account state)
    - Historical database (price data for charts)
    """
    
    def __init__(
        self,
        paper_db_path: str = 'data/paper_trading.db',
        historical_db_path: str = 'data/db/historical.db',
        trade_filter: str = 'all'
    ):
        """
        Initialize database loader.

        Args:
            paper_db_path: Path to paper trading database
            historical_db_path: Path to historical data database
            trade_filter: Filter trades by type - 'all', 'paper', or 'live'
        """
        self.paper_db_path = paper_db_path
        self.historical_db_path = historical_db_path
        self.trade_filter = trade_filter  # 'all', 'paper', or 'live'

    def _get_trade_filter_sql(self) -> str:
        """Get SQL WHERE clause for trade filtering."""
        if self.trade_filter == 'paper':
            return "AND (is_real_trade = 0 OR is_real_trade IS NULL)"
        elif self.trade_filter == 'live':
            return "AND is_real_trade = 1"
        return ""  # 'all' - no filter

    def load_account_state(self, state: TrainingState) -> None:
        """Load account state from paper trading database."""
        if not Path(self.paper_db_path).exists():
            return
        
        try:
            conn = sqlite3.connect(self.paper_db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT balance, winning_trades, losing_trades, total_profit_loss 
                FROM account_state ORDER BY updated_at DESC LIMIT 1
            ''')
            row = cursor.fetchone()
            
            if row:
                # Only use DB win/loss if log parser hasn't already set them
                if state.account.winning_trades == 0 and state.account.losing_trades == 0:
                    state.account.winning_trades = row[1] or 0
                    state.account.losing_trades = row[2] or 0
                db_balance = row[0] or 0
                
                # Only use DB balance if state hasn't been updated from logs
                if db_balance > 0 and abs(state.account.current_balance - state.account.initial_balance) < 0.01:
                    state.account.current_balance = db_balance
                    state.account.pnl = row[3] or 0
                    state.account.update_pnl()
            
            conn.close()
            
        except Exception as e:
            logger.error(f"Error loading account state: {e}")
    
    def load_open_positions(self, state: TrainingState) -> None:
        """Load open positions with current values."""
        if not Path(self.paper_db_path).exists():
            return
        
        try:
            conn = sqlite3.connect(self.paper_db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, timestamp, option_type, strike_price, premium_paid, quantity,
                       entry_price, expiration_date, stop_loss, take_profit
                FROM trades 
                WHERE status IN ('FILLED', 'OPEN')
            ''')
            open_trades = cursor.fetchall()
            
            total_invested = 0.0
            total_current_value = 0.0
            state.open_positions_details = []
            
            for trade in open_trades:
                position = self._process_open_trade(trade, state)
                if position:
                    state.open_positions_details.append(position)
                    total_invested += position.invested
                    total_current_value += position.current_value
            
            state.account.open_positions_invested = total_invested
            state.account.open_positions_value = total_current_value
            state.current_positions = len(open_trades)
            state.account.unrealized_pnl = total_current_value - total_invested if total_invested > 0 else 0
            
            conn.close()
            
        except Exception as e:
            logger.error(f"Error loading open positions: {e}")
    
    def _process_open_trade(self, trade_row: tuple, state: TrainingState) -> Optional[PositionDetail]:
        """Process a single open trade row into a PositionDetail."""
        (trade_id, timestamp, option_type, strike_price, premium_paid, quantity,
         entry_price, expiration_date, stop_loss, take_profit) = trade_row
        
        entry_premium = premium_paid or entry_price or 0
        qty = quantity or 1
        invested = entry_premium * qty * 100
        
        # Get current time
        current_datetime = self._get_current_datetime(state)
        
        # Calculate minutes held
        minutes_held = 0.0
        if timestamp:
            try:
                entry_dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                if entry_dt.tzinfo is not None:
                    entry_dt = entry_dt.replace(tzinfo=None)
                minutes_held = max(0, (current_datetime - entry_dt).total_seconds() / 60)
            except Exception:
                pass
        
        # Calculate days to expiry
        days_to_expiry = 1.0
        if expiration_date:
            try:
                if isinstance(expiration_date, str):
                    exp_date = datetime.fromisoformat(expiration_date.replace('Z', '+00:00'))
                    if exp_date.tzinfo is not None:
                        exp_date = exp_date.replace(tzinfo=None)
                else:
                    exp_date = expiration_date
                days_to_expiry = max(0.1, (exp_date - current_datetime).days)
            except Exception:
                pass
        
        # Calculate current value
        current_value = 0.0
        unrealized_pnl = 0.0
        unrealized_pnl_pct = 0.0
        
        underlying_entry_price = entry_price or state.market.spy_price
        
        if state.market.spy_price > 0 and entry_premium > 0 and underlying_entry_price > 0:
            current_value, unrealized_pnl, unrealized_pnl_pct = OptionPricing.calculate_unrealized_pnl(
                option_type or 'CALL',
                underlying_entry_price,
                entry_premium,
                state.market.spy_price,
                qty,
                minutes_held,
                days_to_expiry
            )
        
        # Parse entry time
        entry_time_str = '--:--'
        entry_date_str = '--'
        if timestamp:
            try:
                entry_dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                entry_time_str = entry_dt.strftime('%H:%M:%S')
                entry_date_str = entry_dt.strftime('%m/%d')
            except Exception:
                pass
        
        return PositionDetail(
            id=str(trade_id),
            entry_date=entry_date_str,
            entry_time=entry_time_str,
            entry_timestamp=timestamp,
            option_type=option_type or 'UNKNOWN',
            strike_price=strike_price or 0,
            entry_price=entry_premium,
            quantity=qty,
            invested=invested,
            current_value=current_value,
            current_premium=current_value / (qty * 100) if qty > 0 else 0,
            unrealized_pnl=unrealized_pnl,
            unrealized_pnl_pct=unrealized_pnl_pct,
            stop_loss=stop_loss,
            take_profit=take_profit,
            days_to_expiry=days_to_expiry
        )
    
    def _get_current_datetime(self, state: TrainingState) -> datetime:
        """Get current datetime, using simulated time if in training."""
        if state.session.simulated_date and state.session.simulated_date != "Not started":
            try:
                if state.session.simulated_time:
                    return datetime.strptime(
                        f"{state.session.simulated_date} {state.session.simulated_time}",
                        '%Y-%m-%d %H:%M:%S'
                    )
                else:
                    return datetime.strptime(state.session.simulated_date, '%Y-%m-%d')
            except Exception:
                pass
        # Use market time (US/Eastern) instead of local time
        return get_market_time()
    
    def load_recent_trades(self, state: TrainingState, limit: int = 10) -> None:
        """Load recent trades from database."""
        if not Path(self.paper_db_path).exists():
            return
        
        try:
            conn = sqlite3.connect(self.paper_db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT timestamp, option_type, entry_price, profit_loss, status,
                       ml_confidence, exit_price, exit_timestamp, strike_price, premium_paid
                FROM trades 
                ORDER BY CASE WHEN status IN ('FILLED', 'OPEN') THEN 0 ELSE 1 END,
                         timestamp DESC 
                LIMIT ?
            ''', (limit,))
            
            state.recent_trades = []
            
            for row in cursor.fetchall():
                trade = self._process_trade_row(row)
                if trade:
                    state.recent_trades.append(trade)
            
            conn.close()
            
        except Exception as e:
            logger.error(f"Error loading recent trades: {e}")
    
    def _process_trade_row(self, row: tuple) -> Optional[TradeRecord]:
        """Process a trade row into a TradeRecord."""
        (timestamp, option_type, entry_price, profit_loss, status,
         confidence, exit_price, exit_timestamp, strike_price, premium_paid) = row
        
        try:
            trade_dt = datetime.fromisoformat(timestamp)
            trade_date = trade_dt.strftime('%m/%d')
            trade_time = trade_dt.strftime('%H:%M')
        except Exception:
            trade_date = '--'
            trade_time = '00:00'
        
        exit_time_str = '--'
        if exit_timestamp:
            try:
                exit_dt = datetime.fromisoformat(exit_timestamp)
                exit_time_str = exit_dt.strftime('%H:%M')
            except Exception:
                pass
        
        actual_return_pct = 0.0
        if profit_loss and premium_paid and premium_paid > 0:
            actual_return_pct = (profit_loss / premium_paid) * 100
        
        return TradeRecord(
            date=trade_date,
            time=trade_time,
            action=option_type or 'UNKNOWN',
            strike=strike_price or 0,
            entry_price=premium_paid or entry_price or 0,
            exit_price=exit_price or 0,
            exit_time=exit_time_str,
            pnl=profit_loss or 0,
            status=status or 'UNKNOWN',
            actual_return_pct=actual_return_pct,
            confidence=confidence or 0
        )
    
    def load_all(self, state: TrainingState) -> None:
        """Load all data from database."""
        self.load_account_state(state)
        self.load_open_positions(state)
        self.load_recent_trades(state)
        self.load_latest_simulated_time(state)
    
    def load_latest_simulated_time(self, state: TrainingState) -> None:
        """Load the latest simulated time from database if not already set."""
        # Only set if not already set from log file
        if state.session.simulated_date != "Not started" and state.session.simulated_time:
            return
        
        if not Path(self.paper_db_path).exists():
            return
        
        try:
            conn = sqlite3.connect(self.paper_db_path)
            cursor = conn.cursor()
            
            # Get the most recent timestamp from trades or account_state
            cursor.execute('''
                SELECT MAX(timestamp) FROM trades
            ''')
            row = cursor.fetchone()
            
            if row and row[0]:
                from datetime import datetime
                timestamp_str = row[0]
                try:
                    # Parse the timestamp
                    dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    state.session.simulated_date = dt.strftime('%Y-%m-%d')
                    state.session.simulated_time = dt.strftime('%H:%M:%S')
                except Exception as e:
                    logger.debug(f"Could not parse timestamp {timestamp_str}: {e}")
            
            conn.close()
            
        except Exception as e:
            logger.error(f"Error loading latest simulated time: {e}")
    
    # =========================================================================
    # Chart Data Loading
    # =========================================================================
    
    def load_spy_prices(
        self,
        reference_time: Optional[str] = None,
        lookback_days: int = 3,
        max_points: int = 10000
    ) -> List[Dict[str, Any]]:
        """
        Load SPY price history for chart.
        
        Args:
            reference_time: Reference datetime (ISO format) to center on
            lookback_days: Days before/after reference to include
            max_points: Maximum data points to return
            
        Returns:
            List of price point dicts with timestamp, close, high, low, open
        """
        prices = []
        
        if not Path(self.historical_db_path).exists():
            return prices
        
        try:
            conn = sqlite3.connect(self.historical_db_path)
            cursor = conn.cursor()
            
            if reference_time:
                # Calculate date range in Python
                from datetime import datetime as dt, timedelta
                try:
                    ref_dt = dt.fromisoformat(reference_time.replace('Z', '').replace(' ', 'T'))
                    start_dt = ref_dt - timedelta(days=lookback_days)
                    # Load up to current simulated time + buffer for predictions
                    end_dt = ref_dt + timedelta(hours=3)
                    # Use date-only for start (catches all timestamps on that day)
                    start_str = start_dt.strftime('%Y-%m-%d')
                    # Use end of day + buffer for end
                    end_str = end_dt.strftime('%Y-%m-%d %H:%M:%S')
                except:
                    # Fallback: just get recent data
                    start_str = '2020-01-01'
                    end_str = '2099-12-31'
                
                # Use REPLACE to normalize timestamps for comparison
                # This handles both "2025-06-09 18:34:00" and "2025-06-09T18:34:00" formats
                cursor.execute('''
                    SELECT SUBSTR(REPLACE(timestamp, 'T', ' '), 1, 19) as ts_norm, 
                           close_price, high_price, low_price, open_price
                    FROM historical_data 
                    WHERE symbol = 'SPY'
                    AND REPLACE(timestamp, 'T', ' ') >= ?
                    AND REPLACE(timestamp, 'T', ' ') <= ?
                    GROUP BY ts_norm
                    ORDER BY ts_norm ASC
                    LIMIT ?
                ''', (start_str, end_str, max_points))
            else:
                # Use REPLACE+SUBSTR to normalize timestamps
                cursor.execute('''
                    SELECT ts_norm, close_price, high_price, low_price, open_price
                    FROM (
                        SELECT SUBSTR(REPLACE(timestamp, 'T', ' '), 1, 19) as ts_norm, 
                               close_price, high_price, low_price, open_price
                        FROM historical_data 
                        WHERE symbol = 'SPY'
                        GROUP BY ts_norm
                        ORDER BY ts_norm DESC
                        LIMIT ?
                    ) 
                    ORDER BY ts_norm ASC
                ''', (max_points,))
            
            for row in cursor.fetchall():
                try:
                    close_price = row[1]
                    # Ensure timestamp uses 'T' separator for consistent JS Date parsing
                    # DB may store "2025-06-11 17:22:00" but JS needs "2025-06-11T17:22:00"
                    timestamp = row[0].replace(' ', 'T') if row[0] else row[0]
                    prices.append({
                        'timestamp': timestamp,
                        'close': close_price,
                        'high': row[2] or close_price,
                        'low': row[3] or close_price,
                        'open': row[4] or close_price
                    })
                except Exception:
                    continue
            
            conn.close()
            
        except Exception as e:
            logger.error(f"Error loading SPY prices: {e}")
        
        return prices
    
    def load_vix_prices(
        self,
        reference_time: Optional[str] = None,
        lookback_days: int = 3,
        max_points: int = 10000,
        bb_period: int = 20
    ) -> Dict[str, Any]:
        """
        Load VIX price history with Bollinger Bands for chart.
        
        Args:
            reference_time: Reference datetime (ISO format) to center on
            lookback_days: Days before/after reference to include
            max_points: Maximum data points to return
            bb_period: Period for Bollinger Band calculation (default 20)
            
        Returns:
            Dict with vix_prices, bb_upper, bb_lower, bb_middle lists
        """
        result = {
            'vix_prices': [],
            'bb_upper': [],
            'bb_lower': [],
            'bb_middle': []
        }
        
        if not Path(self.historical_db_path).exists():
            return result
        
        try:
            conn = sqlite3.connect(self.historical_db_path)
            cursor = conn.cursor()
            
            if reference_time:
                from datetime import datetime as dt, timedelta
                try:
                    ref_dt = dt.fromisoformat(reference_time.replace('Z', '').replace(' ', 'T'))
                    start_dt = ref_dt - timedelta(days=lookback_days)
                    end_dt = ref_dt + timedelta(hours=3)
                    start_str = start_dt.strftime('%Y-%m-%d')
                    end_str = end_dt.strftime('%Y-%m-%d %H:%M:%S')
                except:
                    start_str = '2020-01-01'
                    end_str = '2099-12-31'
                
                cursor.execute('''
                    SELECT SUBSTR(REPLACE(timestamp, 'T', ' '), 1, 19) as ts_norm, 
                           close_price, high_price, low_price, open_price
                    FROM historical_data 
                    WHERE symbol IN ('VIX', '^VIX')
                    AND REPLACE(timestamp, 'T', ' ') >= ?
                    AND REPLACE(timestamp, 'T', ' ') <= ?
                    GROUP BY ts_norm
                    ORDER BY ts_norm ASC
                    LIMIT ?
                ''', (start_str, end_str, max_points))
            else:
                cursor.execute('''
                    SELECT ts_norm, close_price, high_price, low_price, open_price
                    FROM (
                        SELECT SUBSTR(REPLACE(timestamp, 'T', ' '), 1, 19) as ts_norm, 
                               close_price, high_price, low_price, open_price
                        FROM historical_data 
                        WHERE symbol IN ('VIX', '^VIX')
                        GROUP BY ts_norm
                        ORDER BY ts_norm DESC
                        LIMIT ?
                    ) 
                    ORDER BY ts_norm ASC
                ''', (max_points,))
            
            raw_prices = []
            for row in cursor.fetchall():
                try:
                    close_price = row[1]
                    timestamp = row[0].replace(' ', 'T') if row[0] else row[0]
                    raw_prices.append({
                        'timestamp': timestamp,
                        'close': close_price,
                        'high': row[2] or close_price,
                        'low': row[3] or close_price,
                        'open': row[4] or close_price
                    })
                except Exception:
                    continue
            
            conn.close()
            
            # Calculate Bollinger Bands
            # Only include BB data where there's actual variance (std > min_std)
            # This prevents flat lines when VIX data is stale/unchanged
            if len(raw_prices) >= bb_period:
                import numpy as np
                closes = [p['close'] for p in raw_prices]
                min_std = 0.01  # Minimum std deviation to consider meaningful
                
                for i, price in enumerate(raw_prices):
                    result['vix_prices'].append(price)
                    
                    if i >= bb_period - 1:
                        # Calculate BB for this point using lookback window
                        window = closes[i - bb_period + 1:i + 1]
                        sma = np.mean(window)
                        std = np.std(window)
                        
                        # Only include BB data if there's actual variance
                        if std > min_std:
                            result['bb_middle'].append({
                                'timestamp': price['timestamp'],
                                'value': float(sma)
                            })
                            result['bb_upper'].append({
                                'timestamp': price['timestamp'],
                                'value': float(sma + 2 * std)
                            })
                            result['bb_lower'].append({
                                'timestamp': price['timestamp'],
                                'value': float(sma - 2 * std)
                            })
                        # else: skip this point - no meaningful BB data
                    # Skip early points - don't add placeholders
            else:
                result['vix_prices'] = raw_prices
            
        except Exception as e:
            logger.error(f"Error loading VIX prices: {e}")
        
        return result
    
    def load_trades_for_chart(
        self,
        reference_time: Optional[str] = None,
        lookback_days: int = 3,
        max_trades: int = 200
    ) -> tuple:
        """
        Load trades and annotations for chart.
        
        Args:
            reference_time: Reference datetime to center on
            lookback_days: Days before reference to include
            max_trades: Maximum trades to return
            
        Returns:
            Tuple of (trades list, annotations list)
        """
        trades = []
        annotations = []
        
        if not Path(self.paper_db_path).exists():
            return trades, annotations
        
        try:
            conn = sqlite3.connect(self.paper_db_path)
            cursor = conn.cursor()
            
            trade_filter_sql = self._get_trade_filter_sql()

            if reference_time:
                cursor.execute(f'''
                    SELECT timestamp, option_type, strike_price, premium_paid, entry_price,
                           exit_price, exit_timestamp, profit_loss, status
                    FROM trades
                    WHERE datetime(timestamp) >= datetime(?, '-{lookback_days} days')
                    {trade_filter_sql}
                    ORDER BY timestamp ASC
                    LIMIT ?
                ''', (reference_time, max_trades))
            else:
                cursor.execute(f'''
                    SELECT timestamp, option_type, strike_price, premium_paid, entry_price,
                           exit_price, exit_timestamp, profit_loss, status
                    FROM trades
                    WHERE 1=1 {trade_filter_sql}
                    ORDER BY timestamp DESC
                    LIMIT ?
                ''', (max_trades,))
            
            for row in cursor.fetchall():
                try:
                    trade, entry_annotation, exit_annotation = self._process_chart_trade(row)
                    if trade:
                        trades.append(trade)
                    if entry_annotation:
                        annotations.append(entry_annotation)
                    if exit_annotation:
                        annotations.append(exit_annotation)
                except Exception as e:
                    logger.debug(f"Error processing chart trade: {e}")
            
            conn.close()
            
        except Exception as e:
            logger.error(f"Error loading trades for chart: {e}")
        
        return trades, annotations
    
    def _process_chart_trade(self, row: tuple) -> tuple:
        """Process a trade row for chart display."""
        (timestamp, option_type, strike, premium_paid, entry_price,
         exit_price, exit_time, pnl, status) = row
        
        option_type = option_type or 'UNKNOWN'
        
        # Determine color
        if status in ('PROFIT_TAKEN', 'STOPPED_OUT', 'CLOSED', 'EXPIRED'):
            color = '#22c55e' if (pnl and pnl > 0) else '#ef4444'
        else:
            color = '#a855f7'  # Purple for open
        
        # Ensure timestamps use 'T' separator for consistent JS Date parsing
        entry_ts = timestamp.replace(' ', 'T') if timestamp else timestamp
        exit_ts = exit_time.replace(' ', 'T') if exit_time else exit_time
        
        trade = {
            'entry_time': entry_ts,
            'exit_time': exit_ts,
            'type': option_type,
            'entry_price': premium_paid or entry_price,
            'exit_price': exit_price,
            'strike': strike,
            'pnl': pnl,
            'status': status,
            'color': color
        }
        
        # Entry annotation
        entry_annotation = None
        if entry_ts:
            strike_str = f" ${strike:.0f}" if strike else ""
            emoji = "ðŸŸ¢" if 'CALL' in option_type.upper() else "ðŸ”´"
            entry_annotation = {
                'timestamp': entry_ts,
                'type': 'entry',
                'option_type': option_type,
                'strike': strike,
                'label': f"{emoji} {option_type}{strike_str}"
            }
        
        # Exit annotation
        exit_annotation = None
        if exit_ts and status in ('PROFIT_TAKEN', 'STOPPED_OUT', 'CLOSED', 'EXPIRED'):
            pnl_label = f"${pnl:+.2f}" if pnl else ""
            exit_emoji = "âœ…" if (pnl and pnl > 0) else "âŒ"
            exit_annotation = {
                'timestamp': exit_ts,
                'type': 'exit',
                'pnl': pnl,
                'label': f"{exit_emoji} {pnl_label}"
            }
        
        return trade, entry_annotation, exit_annotation



