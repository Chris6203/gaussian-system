#!/usr/bin/env python3
"""
Standalone Data Collector for Training Data

This script runs independently in the background to:
1. Collect minute-by-minute price data for all configured symbols
2. Fill gaps in historical data when the bot wasn't running
3. Collect real liquidity data (bid/ask spread, volume, open interest) for options every minute
4. Store everything to the historical database for realistic training data

Usage:
    python data_collector.py                    # Run continuously
    python data_collector.py --backfill 7      # Backfill last 7 days first, then run continuously
    python data_collector.py --backfill-only 30 # Only backfill last 30 days, then exit
    python data_collector.py --status          # Show data collection status

Run in background:
    Windows: start /B python data_collector.py
    Linux/Mac: nohup python data_collector.py &
"""

import sys
import os
import time
import json
import logging
import argparse
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from logging.handlers import TimedRotatingFileHandler
import threading

# Fix encoding on Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# =============================================================================
# CONFIGURATION
# =============================================================================

class CollectorConfig:
    """Configuration loader for data collector"""
    
    # Default symbols if config.json is missing or incomplete
    try:
        from config.symbols import ALL_DATA_COLLECTION_SYMBOLS
        DEFAULT_SYMBOLS = ALL_DATA_COLLECTION_SYMBOLS
    except ImportError:
        # Fallback if config module not found
        DEFAULT_SYMBOLS = ['SPY', '^VIX', 'UUP']
    
    def __init__(self, config_path: str = "config.json"):
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> dict:
        """Load configuration from file"""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"[WARN] Could not load config.json: {e}, using defaults")
            return {}
    
    def get_symbols(self) -> List[str]:
        """
        Get all symbols to fetch.
        
        Priority:
        1. config.json data_fetching.symbols
        2. DEFAULT_SYMBOLS constant (includes MEGA_CAP_TECH_SYMBOLS)
        
        Always ensures primary trading symbol is included.
        """
        symbols = self.config.get('data_fetching', {}).get('symbols', self.DEFAULT_SYMBOLS.copy())
        primary = self.config.get('trading', {}).get('symbol', 'SPY')
        if primary not in symbols:
            symbols.insert(0, primary)
        return symbols
    
    def get_trading_symbol(self) -> str:
        """Get primary trading symbol for options liquidity"""
        return self.config.get('trading', {}).get('symbol', 'SPY')
    
    def get_tradier_credentials(self) -> Optional[Dict]:
        """Get Tradier API credentials"""
        creds = self.config.get('credentials', {}).get('tradier', {})
        data_token = creds.get('data_api_token')
        
        # Try sandbox first for data, fall back to live
        sandbox = creds.get('sandbox', {})
        live = creds.get('live', {})
        
        if sandbox.get('access_token'):
            return {
                'access_token': sandbox['access_token'],
                'account_number': sandbox.get('account_number'),
                'is_sandbox': True,
                'data_token': data_token
            }
        elif live.get('access_token'):
            return {
                'access_token': live['access_token'],
                'account_number': live.get('account_number'),
                'is_sandbox': False,
                'data_token': data_token
            }
        return None
    
    def get_liquidity_config(self) -> Dict:
        """Get liquidity settings"""
        return self.config.get('liquidity', {
            'max_bid_ask_spread_pct': 5.0,
            'min_volume': 20,
            'min_open_interest': 10
        })


# =============================================================================
# MARKET HOURS
# =============================================================================

def is_market_hours() -> bool:
    """Check if US stock market is currently open (9:30 AM - 4:00 PM ET)"""
    try:
        import pytz
        et = pytz.timezone('US/Eastern')
        now = datetime.now(et)
        
        # Weekend check
        if now.weekday() >= 5:
            return False
        
        # Time check (9:30 AM - 4:00 PM ET)
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        
        return market_open <= now <= market_close
        
    except ImportError:
        # Fallback without pytz - assume UTC-5
        now = datetime.utcnow() - timedelta(hours=5)
        
        if now.weekday() >= 5:
            return False
        
        hour = now.hour
        minute = now.minute
        
        # Market hours: 9:30 - 16:00 ET
        if hour < 9 or hour > 16:
            return False
        if hour == 9 and minute < 30:
            return False
        if hour == 16 and minute > 0:
            return False
        
        return True


def get_market_status() -> Tuple[bool, str]:
    """Get market status and reason"""
    try:
        import pytz
        et = pytz.timezone('US/Eastern')
        now = datetime.now(et)
        
        if now.weekday() >= 5:
            return False, f"Weekend ({now.strftime('%A')})"
        
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        
        if now < market_open:
            minutes_until = (market_open - now).seconds // 60
            return False, f"Pre-market ({minutes_until} min until open)"
        elif now > market_close:
            return False, "After-hours"
        else:
            minutes_left = (market_close - now).seconds // 60
            return True, f"Open ({minutes_left} min remaining)"
            
    except ImportError:
        is_open = is_market_hours()
        return is_open, "Open" if is_open else "Closed"


# =============================================================================
# DATA SOURCES
# =============================================================================

class DataFetcher:
    """Fetches price data from multiple sources with fallback"""
    
    def __init__(self, config: CollectorConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self._yf = None
        self._tradier_token = None
        
        # Initialize Tradier credentials
        creds = config.get_tradier_credentials()
        if creds:
            self._tradier_token = creds.get('data_token') or creds.get('access_token')
    
    def get_price_data(self, symbol: str, period: str = '1d', interval: str = '1m') -> Optional[Dict]:
        """
        Fetch price data for a symbol
        
        Returns dict with: timestamp, open, high, low, close, volume
        """
        # Try yfinance first (most reliable for price data)
        data = self._fetch_yfinance(symbol, period, interval)
        if data is not None:
            return data
        
        # Try Tradier as fallback
        data = self._fetch_tradier(symbol)
        if data is not None:
            return data
        
        self.logger.warning(f"[DATA] Could not fetch data for {symbol}")
        return None
    
    def _fetch_yfinance(self, symbol: str, period: str, interval: str) -> Optional[Dict]:
        """Fetch from yfinance"""
        try:
            import yfinance as yf
            
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            
            if df.empty:
                return None
            
            # Get the most recent data point
            latest = df.iloc[-1]
            timestamp = df.index[-1]
            
            return {
                'timestamp': timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp),
                'open': float(latest.get('Open', 0)),
                'high': float(latest.get('High', 0)),
                'low': float(latest.get('Low', 0)),
                'close': float(latest.get('Close', 0)),
                'volume': int(latest.get('Volume', 0)),
                'source': 'yfinance',
                'df': df  # Include full dataframe for backfill
            }
            
        except Exception as e:
            self.logger.debug(f"[YFINANCE] Error fetching {symbol}: {e}")
            return None
    
    def _fetch_tradier(self, symbol: str) -> Optional[Dict]:
        """Fetch from Tradier API"""
        if not self._tradier_token:
            return None
        
        try:
            import requests
            
            # Use production API for real market data
            url = "https://api.tradier.com/v1/markets/quotes"
            
            response = requests.get(
                url,
                params={'symbols': symbol},
                headers={
                    'Authorization': f'Bearer {self._tradier_token}',
                    'Accept': 'application/json'
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                quotes = data.get('quotes', {})
                quote = quotes.get('quote', {})
                
                if isinstance(quote, list):
                    quote = quote[0] if quote else {}
                
                if quote and quote.get('last'):
                    return {
                        'timestamp': datetime.now().isoformat(),
                        'open': float(quote.get('open', 0) or 0),
                        'high': float(quote.get('high', 0) or 0),
                        'low': float(quote.get('low', 0) or 0),
                        'close': float(quote.get('last', 0) or 0),
                        'volume': int(quote.get('volume', 0) or 0),
                        'source': 'tradier'
                    }
            
        except Exception as e:
            self.logger.debug(f"[TRADIER] Error fetching {symbol}: {e}")
        
        return None


class LiquidityFetcher:
    """Fetches options liquidity data from Tradier"""
    
    def __init__(self, config: CollectorConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.creds = config.get_tradier_credentials()
        
        if self.creds:
            self.api_token = self.creds.get('data_token') or self.creds.get('access_token')
            self.base_url = "https://api.tradier.com/v1"
        else:
            self.api_token = None
            self.base_url = None
    
    def _make_request(self, endpoint: str, params: dict = None) -> Optional[Dict]:
        """Make API request to Tradier"""
        if not self.api_token:
            return None
        
        try:
            import requests
            
            url = f"{self.base_url}{endpoint}"
            response = requests.get(
                url,
                params=params,
                headers={
                    'Authorization': f'Bearer {self.api_token}',
                    'Accept': 'application/json'
                }
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                self.logger.debug(f"[TRADIER] Request failed: {response.status_code}")
                
        except Exception as e:
            self.logger.debug(f"[TRADIER] Request error: {e}")
        
        return None
    
    def get_expirations(self, symbol: str) -> List[str]:
        """Get available expiration dates"""
        response = self._make_request(
            "/markets/options/expirations",
            params={'symbol': symbol}
        )
        
        if response and 'expirations' in response:
            return response['expirations'].get('date', [])
        return []
    
    def get_options_chain(self, symbol: str, expiration: str) -> List[Dict]:
        """Get full options chain"""
        response = self._make_request(
            "/markets/options/chains",
            params={'symbol': symbol, 'expiration': expiration, 'greeks': 'true'}
        )
        
        if response and 'options' in response and response['options']:
            options = response['options'].get('option', [])
            return options if isinstance(options, list) else [options]
        return []
    
    def get_atm_liquidity(self, symbol: str, current_price: float) -> List[Dict]:
        """
        Get liquidity data for ATM options (calls and puts)
        
        Returns list of liquidity snapshots
        """
        if not self.api_token:
            return []
        
        snapshots = []
        
        try:
            # Get nearest expiration (~30-45 days out for best liquidity)
            expirations = self.get_expirations(symbol)
            if not expirations:
                return []
            
            # Find expiration ~30-45 days out
            target_date = datetime.now() + timedelta(days=35)
            best_exp = min(
                expirations,
                key=lambda x: abs((datetime.strptime(x, '%Y-%m-%d') - target_date).days)
            )
            
            # Get options chain
            chain = self.get_options_chain(symbol, best_exp)
            if not chain:
                return []
            
            # Find ATM strikes for calls and puts
            for option_type in ['call', 'put']:
                # Filter by type
                filtered = [opt for opt in chain if opt.get('option_type') == option_type]
                
                if not filtered:
                    continue
                
                # Find closest strike to current price
                atm_option = min(
                    filtered,
                    key=lambda x: abs(float(x.get('strike', 0)) - current_price)
                )
                
                # Extract liquidity data
                bid = float(atm_option.get('bid') or 0)
                ask = float(atm_option.get('ask') or 0)
                volume = int(atm_option.get('volume') or 0)
                oi = int(atm_option.get('open_interest') or 0)
                
                if bid > 0 and ask > 0:
                    mid = (bid + ask) / 2
                    spread_pct = ((ask - bid) / mid) * 100 if mid > 0 else 0
                    
                    # Calculate quality score (0-100)
                    quality = 0
                    quality += max(0, 40 - spread_pct * 8)  # Spread score
                    quality += min(30, volume / 10)  # Volume score
                    quality += min(30, oi / 50)  # OI score
                    
                    # Get Greeks if available
                    greeks = atm_option.get('greeks', {})
                    
                    snapshot = {
                        'timestamp': datetime.now().isoformat(),
                        'symbol': symbol,
                        'option_symbol': atm_option.get('symbol'),
                        'underlying_price': current_price,
                        'strike_price': float(atm_option.get('strike', 0)),
                        'option_type': option_type.upper(),
                        'expiration_date': best_exp,
                        'bid': bid,
                        'ask': ask,
                        'spread_pct': spread_pct,
                        'mid_price': mid,
                        'volume': volume,
                        'open_interest': oi,
                        'quality_score': quality,
                        'implied_volatility': float(greeks.get('mid_iv') or 0),
                        'delta': float(greeks.get('delta') or 0),
                        'gamma': float(greeks.get('gamma') or 0),
                        'theta': float(greeks.get('theta') or 0),
                        'vega': float(greeks.get('vega') or 0),
                        'signal_action': None,  # Not from trading signals
                        'signal_confidence': None,
                        'trade_executed': 0,
                        'trade_blocked_reason': 'Data collection only',
                        'hmm_regime': None,
                        'vix_value': None
                    }
                    snapshots.append(snapshot)
            
        except Exception as e:
            self.logger.error(f"[LIQUIDITY] Error fetching ATM liquidity: {e}")
        
        return snapshots
    
    def get_strike_range_liquidity(self, symbol: str, current_price: float, 
                                   range_pct: float = 0.05) -> List[Dict]:
        """
        Get liquidity for a range of strikes around ATM
        
        Args:
            symbol: Underlying symbol
            current_price: Current price of underlying
            range_pct: Range around ATM (e.g., 0.05 = ±5%)
        
        Returns list of liquidity snapshots
        """
        if not self.api_token:
            return []
        
        snapshots = []
        min_strike = current_price * (1 - range_pct)
        max_strike = current_price * (1 + range_pct)
        
        try:
            expirations = self.get_expirations(symbol)
            if not expirations:
                return []
            
            # Get nearest weekly/monthly expiration
            target_date = datetime.now() + timedelta(days=35)
            best_exp = min(
                expirations,
                key=lambda x: abs((datetime.strptime(x, '%Y-%m-%d') - target_date).days)
            )
            
            chain = self.get_options_chain(symbol, best_exp)
            if not chain:
                return []
            
            for option in chain:
                strike = float(option.get('strike', 0))
                
                # Only include strikes in range
                if strike < min_strike or strike > max_strike:
                    continue
                
                bid = float(option.get('bid') or 0)
                ask = float(option.get('ask') or 0)
                volume = int(option.get('volume') or 0)
                oi = int(option.get('open_interest') or 0)
                
                if bid > 0 and ask > 0:
                    mid = (bid + ask) / 2
                    spread_pct = ((ask - bid) / mid) * 100 if mid > 0 else 0
                    
                    quality = 0
                    quality += max(0, 40 - spread_pct * 8)
                    quality += min(30, volume / 10)
                    quality += min(30, oi / 50)
                    
                    greeks = option.get('greeks', {})
                    
                    snapshot = {
                        'timestamp': datetime.now().isoformat(),
                        'symbol': symbol,
                        'option_symbol': option.get('symbol'),
                        'underlying_price': current_price,
                        'strike_price': strike,
                        'option_type': option.get('option_type', '').upper(),
                        'expiration_date': best_exp,
                        'bid': bid,
                        'ask': ask,
                        'spread_pct': spread_pct,
                        'mid_price': mid,
                        'volume': volume,
                        'open_interest': oi,
                        'quality_score': quality,
                        'implied_volatility': float(greeks.get('mid_iv') or 0),
                        'delta': float(greeks.get('delta') or 0),
                        'gamma': float(greeks.get('gamma') or 0),
                        'theta': float(greeks.get('theta') or 0),
                        'vega': float(greeks.get('vega') or 0),
                        'signal_action': None,
                        'signal_confidence': None,
                        'trade_executed': 0,
                        'trade_blocked_reason': 'Data collection only',
                        'hmm_regime': None,
                        'vix_value': None
                    }
                    snapshots.append(snapshot)
            
        except Exception as e:
            self.logger.error(f"[LIQUIDITY] Error fetching strike range: {e}")
        
        return snapshots


# =============================================================================
# DATA STORAGE
# =============================================================================

class DataStorage:
    """Stores collected data to historical database"""
    
    def __init__(self, db_path: str = "data/db/historical.db", logger: logging.Logger = None):
        self.db_path = db_path
        self.logger = logger or logging.getLogger(__name__)
        self._lock = threading.Lock()
        self._ensure_database()
    
    def _ensure_database(self):
        """Ensure database and tables exist"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Historical price data
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS historical_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                open_price REAL,
                high_price REAL,
                low_price REAL,
                close_price REAL,
                volume INTEGER,
                UNIQUE(symbol, timestamp)
            )
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_symbol_timestamp 
            ON historical_data(symbol, timestamp)
        ''')
        
        # Liquidity snapshots
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS liquidity_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                symbol TEXT NOT NULL,
                option_symbol TEXT,
                underlying_price REAL,
                strike_price REAL,
                option_type TEXT,
                expiration_date TEXT,
                bid REAL,
                ask REAL,
                spread_pct REAL,
                mid_price REAL,
                volume INTEGER,
                open_interest INTEGER,
                quality_score REAL,
                implied_volatility REAL,
                delta REAL,
                gamma REAL,
                theta REAL,
                vega REAL,
                signal_action TEXT,
                signal_confidence REAL,
                trade_executed INTEGER DEFAULT 0,
                trade_blocked_reason TEXT,
                hmm_regime TEXT,
                vix_value REAL,
                UNIQUE(symbol, option_symbol, timestamp)
            )
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_liquidity_symbol_timestamp 
            ON liquidity_snapshots(symbol, timestamp)
        ''')
        
        # Data collection log (for tracking gaps)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS collection_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                symbols_collected TEXT,
                price_records INTEGER,
                liquidity_records INTEGER,
                market_open INTEGER,
                duration_ms INTEGER,
                notes TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _normalize_timestamp(self, timestamp: str) -> str:
        """
        Normalize timestamp to consistent format without timezone suffix.
        
        Handles:
        - '2025-11-12T09:30:00-05:00' -> '2025-11-12T09:30:00'
        - '2025-11-12T09:30:00+00:00' -> '2025-11-12T09:30:00'
        - '2025-11-12T09:30:00Z' -> '2025-11-12T09:30:00'
        - '2025-11-12T09:30:00' -> '2025-11-12T09:30:00' (no change)
        """
        if not timestamp:
            return timestamp
        
        # Strip timezone info - keep only first 19 chars (YYYY-MM-DDTHH:MM:SS)
        ts = str(timestamp)
        if len(ts) > 19:
            ts = ts[:19]
        return ts
    
    def save_price_data(self, symbol: str, data: Dict) -> bool:
        """Save a single price data point"""
        try:
            with self._lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Normalize timestamp to prevent duplicates from different timezone formats
                timestamp = self._normalize_timestamp(data.get('timestamp'))
                
                cursor.execute('''
                    INSERT OR REPLACE INTO historical_data 
                    (symbol, timestamp, open_price, high_price, low_price, close_price, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    symbol,
                    timestamp,
                    data.get('open', 0),
                    data.get('high', 0),
                    data.get('low', 0),
                    data.get('close', 0),
                    data.get('volume', 0)
                ))
                
                conn.commit()
                conn.close()
            return True
            
        except Exception as e:
            self.logger.error(f"[DB] Error saving price data: {e}")
            return False
    
    def save_price_dataframe(self, symbol: str, df) -> int:
        """Save a DataFrame of price data (for backfill)"""
        if df is None or df.empty:
            return 0
        
        saved = 0
        try:
            with self._lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                for idx, row in df.iterrows():
                    try:
                        timestamp_raw = idx.isoformat() if hasattr(idx, 'isoformat') else str(idx)
                        # Normalize timestamp to prevent timezone-format duplicates
                        timestamp = self._normalize_timestamp(timestamp_raw)
                        
                        cursor.execute('''
                            INSERT OR REPLACE INTO historical_data 
                            (symbol, timestamp, open_price, high_price, low_price, close_price, volume)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            symbol,
                            timestamp,
                            float(row.get('Open', row.get('open', 0)) or 0),
                            float(row.get('High', row.get('high', 0)) or 0),
                            float(row.get('Low', row.get('low', 0)) or 0),
                            float(row.get('Close', row.get('close', 0)) or 0),
                            int(row.get('Volume', row.get('volume', 0)) or 0)
                        ))
                        saved += 1
                    except Exception:
                        pass
                
                conn.commit()
                conn.close()
            
        except Exception as e:
            self.logger.error(f"[DB] Error saving dataframe: {e}")
        
        return saved
    
    def save_liquidity_snapshot(self, snapshot: Dict) -> bool:
        """Save a liquidity snapshot"""
        try:
            with self._lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT OR REPLACE INTO liquidity_snapshots 
                    (timestamp, symbol, option_symbol, underlying_price, strike_price,
                     option_type, expiration_date, bid, ask, spread_pct, mid_price,
                     volume, open_interest, quality_score, implied_volatility,
                     delta, gamma, theta, vega, signal_action, signal_confidence,
                     trade_executed, trade_blocked_reason, hmm_regime, vix_value)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    snapshot.get('timestamp'),
                    snapshot.get('symbol'),
                    snapshot.get('option_symbol'),
                    snapshot.get('underlying_price'),
                    snapshot.get('strike_price'),
                    snapshot.get('option_type'),
                    snapshot.get('expiration_date'),
                    snapshot.get('bid'),
                    snapshot.get('ask'),
                    snapshot.get('spread_pct'),
                    snapshot.get('mid_price'),
                    snapshot.get('volume'),
                    snapshot.get('open_interest'),
                    snapshot.get('quality_score'),
                    snapshot.get('implied_volatility'),
                    snapshot.get('delta'),
                    snapshot.get('gamma'),
                    snapshot.get('theta'),
                    snapshot.get('vega'),
                    snapshot.get('signal_action'),
                    snapshot.get('signal_confidence'),
                    snapshot.get('trade_executed', 0),
                    snapshot.get('trade_blocked_reason'),
                    snapshot.get('hmm_regime'),
                    snapshot.get('vix_value')
                ))
                
                conn.commit()
                conn.close()
            return True
            
        except Exception as e:
            self.logger.debug(f"[DB] Error saving liquidity snapshot: {e}")
            return False
    
    def log_collection(self, symbols: List[str], price_records: int, 
                       liquidity_records: int, market_open: bool, 
                       duration_ms: int, notes: str = None):
        """Log a collection cycle"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO collection_log 
                (timestamp, symbols_collected, price_records, liquidity_records, 
                 market_open, duration_ms, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                ','.join(symbols),
                price_records,
                liquidity_records,
                1 if market_open else 0,
                duration_ms,
                notes
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"[DB] Error logging collection: {e}")
    
    def get_data_stats(self) -> Dict:
        """Get statistics about collected data"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            stats = {'symbols': {}, 'liquidity': {}, 'collection': {}}
            
            # Price data stats per symbol
            cursor.execute('''
                SELECT symbol, COUNT(*), MIN(timestamp), MAX(timestamp)
                FROM historical_data
                GROUP BY symbol
            ''')
            
            for row in cursor.fetchall():
                stats['symbols'][row[0]] = {
                    'records': row[1],
                    'first': row[2],
                    'last': row[3]
                }
            
            # Liquidity stats
            cursor.execute('''
                SELECT COUNT(*), MIN(timestamp), MAX(timestamp), AVG(quality_score)
                FROM liquidity_snapshots
            ''')
            row = cursor.fetchone()
            stats['liquidity'] = {
                'total_snapshots': row[0],
                'first': row[1],
                'last': row[2],
                'avg_quality': round(row[3], 1) if row[3] else 0
            }
            
            # Recent collection stats
            cursor.execute('''
                SELECT COUNT(*), SUM(price_records), SUM(liquidity_records)
                FROM collection_log
                WHERE timestamp > datetime('now', '-24 hours')
            ''')
            row = cursor.fetchone()
            stats['collection'] = {
                'cycles_24h': row[0],
                'price_records_24h': row[1] or 0,
                'liquidity_records_24h': row[2] or 0
            }
            
            conn.close()
            return stats
            
        except Exception as e:
            self.logger.error(f"[DB] Error getting stats: {e}")
            return {}
    
    def get_data_gaps(self, symbol: str, days: int = 7) -> List[Tuple[str, str]]:
        """Find gaps in data (market hours with no data)"""
        gaps = []
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get timestamps for the last N days
            cursor.execute('''
                SELECT timestamp FROM historical_data
                WHERE symbol = ? 
                AND timestamp > datetime('now', ?)
                ORDER BY timestamp
            ''', (symbol, f'-{days} days'))
            
            rows = cursor.fetchall()
            conn.close()
            
            if len(rows) < 2:
                return gaps
            
            # Find gaps > 5 minutes during market hours
            import pandas as pd
            timestamps = [pd.Timestamp(row[0]) for row in rows]
            
            for i in range(1, len(timestamps)):
                diff = timestamps[i] - timestamps[i-1]
                if diff > timedelta(minutes=5):
                    gaps.append((str(timestamps[i-1]), str(timestamps[i])))
            
        except Exception as e:
            self.logger.error(f"[DB] Error finding gaps: {e}")
        
        return gaps
    
    def check_integrity(self) -> Dict:
        """
        Run comprehensive data integrity checks
        
        Returns dict with:
            - duplicates: count of duplicate records found
            - null_values: count of records with NULL critical fields
            - invalid_prices: count of records with invalid prices
            - issues: list of specific issues found
            - fixed: count of issues auto-fixed
        """
        results = {
            'duplicates': 0,
            'null_values': 0,
            'invalid_prices': 0,
            'liquidity_duplicates': 0,
            'issues': [],
            'fixed': 0
        }
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # ===== CHECK PRICE DATA =====
            
            # 1. Find duplicate timestamps per symbol
            cursor.execute('''
                SELECT symbol, timestamp, COUNT(*) as cnt 
                FROM historical_data 
                GROUP BY symbol, timestamp 
                HAVING cnt > 1
            ''')
            duplicates = cursor.fetchall()
            results['duplicates'] = len(duplicates)
            
            if duplicates:
                results['issues'].append(f"Found {len(duplicates)} duplicate price records")
                for dup in duplicates[:5]:  # Show first 5
                    results['issues'].append(f"  - {dup[0]} @ {dup[1]}: {dup[2]} copies")
            
            # 2. Check for NULL critical fields
            cursor.execute('''
                SELECT COUNT(*) FROM historical_data 
                WHERE close_price IS NULL OR timestamp IS NULL
            ''')
            null_count = cursor.fetchone()[0]
            results['null_values'] = null_count
            
            if null_count > 0:
                results['issues'].append(f"Found {null_count} records with NULL close_price or timestamp")
            
            # 3. Check for invalid prices (negative, zero, or unreasonable)
            cursor.execute('''
                SELECT COUNT(*) FROM historical_data 
                WHERE close_price <= 0 OR close_price > 100000
            ''')
            invalid_count = cursor.fetchone()[0]
            results['invalid_prices'] = invalid_count
            
            if invalid_count > 0:
                results['issues'].append(f"Found {invalid_count} records with invalid prices")
            
            # ===== CHECK LIQUIDITY DATA =====
            
            # 4. Find duplicate liquidity snapshots
            cursor.execute('''
                SELECT symbol, option_symbol, timestamp, COUNT(*) as cnt 
                FROM liquidity_snapshots 
                GROUP BY symbol, option_symbol, timestamp 
                HAVING cnt > 1
            ''')
            liq_duplicates = cursor.fetchall()
            results['liquidity_duplicates'] = len(liq_duplicates)
            
            if liq_duplicates:
                results['issues'].append(f"Found {len(liq_duplicates)} duplicate liquidity records")
            
            # 5. Check for invalid liquidity data
            cursor.execute('''
                SELECT COUNT(*) FROM liquidity_snapshots 
                WHERE bid < 0 OR ask < 0 OR bid > ask
            ''')
            invalid_liq = cursor.fetchone()[0]
            if invalid_liq > 0:
                results['issues'].append(f"Found {invalid_liq} liquidity records with invalid bid/ask")
            
            conn.close()
            
        except Exception as e:
            results['issues'].append(f"Error during integrity check: {e}")
            self.logger.error(f"[INTEGRITY] Error: {e}")
        
        return results
    
    def remove_duplicates(self) -> Dict:
        """
        Remove duplicate records from database
        
        Returns dict with counts of removed records
        """
        results = {
            'price_duplicates_removed': 0,
            'liquidity_duplicates_removed': 0,
            'errors': []
        }
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # ===== REMOVE PRICE DATA DUPLICATES =====
            # Keep the record with the highest ID (most recent insert)
            
            # First, find duplicates
            cursor.execute('''
                SELECT symbol, timestamp, COUNT(*) as cnt, MIN(id) as min_id, MAX(id) as max_id
                FROM historical_data 
                GROUP BY symbol, timestamp 
                HAVING cnt > 1
            ''')
            duplicates = cursor.fetchall()
            
            if duplicates:
                self.logger.info(f"[DEDUP] Found {len(duplicates)} duplicate price records to clean")
                
                # Delete all but the latest (max id) for each duplicate
                for symbol, timestamp, cnt, min_id, max_id in duplicates:
                    cursor.execute('''
                        DELETE FROM historical_data 
                        WHERE symbol = ? AND timestamp = ? AND id != ?
                    ''', (symbol, timestamp, max_id))
                    results['price_duplicates_removed'] += cnt - 1
                
                conn.commit()
                self.logger.info(f"[DEDUP] Removed {results['price_duplicates_removed']} duplicate price records")
            
            # ===== REMOVE LIQUIDITY DATA DUPLICATES =====
            
            cursor.execute('''
                SELECT symbol, option_symbol, timestamp, COUNT(*) as cnt, MIN(id) as min_id, MAX(id) as max_id
                FROM liquidity_snapshots 
                GROUP BY symbol, option_symbol, timestamp 
                HAVING cnt > 1
            ''')
            liq_duplicates = cursor.fetchall()
            
            if liq_duplicates:
                self.logger.info(f"[DEDUP] Found {len(liq_duplicates)} duplicate liquidity records to clean")
                
                for symbol, option_symbol, timestamp, cnt, min_id, max_id in liq_duplicates:
                    cursor.execute('''
                        DELETE FROM liquidity_snapshots 
                        WHERE symbol = ? AND option_symbol = ? AND timestamp = ? AND id != ?
                    ''', (symbol, option_symbol, timestamp, max_id))
                    results['liquidity_duplicates_removed'] += cnt - 1
                
                conn.commit()
                self.logger.info(f"[DEDUP] Removed {results['liquidity_duplicates_removed']} duplicate liquidity records")
            
            conn.close()
            
        except Exception as e:
            results['errors'].append(str(e))
            self.logger.error(f"[DEDUP] Error removing duplicates: {e}")
        
        return results
    
    def remove_invalid_records(self) -> Dict:
        """
        Remove records with invalid data
        
        Returns dict with counts of removed records
        """
        results = {
            'null_records_removed': 0,
            'invalid_price_removed': 0,
            'invalid_liquidity_removed': 0,
            'errors': []
        }
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Remove records with NULL critical fields
            cursor.execute('''
                DELETE FROM historical_data 
                WHERE close_price IS NULL OR timestamp IS NULL OR symbol IS NULL
            ''')
            results['null_records_removed'] = cursor.rowcount
            
            # Remove records with invalid prices
            cursor.execute('''
                DELETE FROM historical_data 
                WHERE close_price <= 0 OR close_price > 100000
            ''')
            results['invalid_price_removed'] = cursor.rowcount
            
            # Remove liquidity records with invalid bid/ask
            cursor.execute('''
                DELETE FROM liquidity_snapshots 
                WHERE bid < 0 OR ask < 0 OR (bid > 0 AND ask > 0 AND bid > ask)
            ''')
            results['invalid_liquidity_removed'] = cursor.rowcount
            
            conn.commit()
            conn.close()
            
            total_removed = sum([
                results['null_records_removed'],
                results['invalid_price_removed'],
                results['invalid_liquidity_removed']
            ])
            
            if total_removed > 0:
                self.logger.info(f"[CLEANUP] Removed {total_removed} invalid records")
            
        except Exception as e:
            results['errors'].append(str(e))
            self.logger.error(f"[CLEANUP] Error removing invalid records: {e}")
        
        return results
    
    def optimize_database(self) -> bool:
        """
        Optimize database (vacuum and reindex)
        
        Returns True if successful
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            self.logger.info("[OPTIMIZE] Running VACUUM...")
            cursor.execute("VACUUM")
            
            self.logger.info("[OPTIMIZE] Rebuilding indexes...")
            cursor.execute("REINDEX")
            
            conn.commit()
            conn.close()
            
            self.logger.info("[OPTIMIZE] Database optimization complete")
            return True
            
        except Exception as e:
            self.logger.error(f"[OPTIMIZE] Error: {e}")
            return False
    
    def full_integrity_check_and_fix(self) -> Dict:
        """
        Run full integrity check and fix all issues
        
        Returns comprehensive results
        """
        self.logger.info("[INTEGRITY] Starting full integrity check...")
        
        results = {
            'check': self.check_integrity(),
            'duplicates_removed': {},
            'invalid_removed': {},
            'optimized': False
        }
        
        # If issues found, fix them
        if results['check']['duplicates'] > 0 or results['check']['liquidity_duplicates'] > 0:
            self.logger.info("[INTEGRITY] Removing duplicates...")
            results['duplicates_removed'] = self.remove_duplicates()
        
        if results['check']['null_values'] > 0 or results['check']['invalid_prices'] > 0:
            self.logger.info("[INTEGRITY] Removing invalid records...")
            results['invalid_removed'] = self.remove_invalid_records()
        
        # Optimize if we made changes
        total_removed = (
            results['duplicates_removed'].get('price_duplicates_removed', 0) +
            results['duplicates_removed'].get('liquidity_duplicates_removed', 0) +
            results['invalid_removed'].get('null_records_removed', 0) +
            results['invalid_removed'].get('invalid_price_removed', 0) +
            results['invalid_removed'].get('invalid_liquidity_removed', 0)
        )
        
        if total_removed > 0:
            self.logger.info("[INTEGRITY] Optimizing database...")
            results['optimized'] = self.optimize_database()
        
        # Re-check after fixes
        results['final_check'] = self.check_integrity()
        
        self.logger.info("[INTEGRITY] Integrity check complete")
        return results


# =============================================================================
# MAIN COLLECTOR
# =============================================================================

class DataCollector:
    """Main data collection orchestrator"""
    
    def __init__(self, config_path: str = "config.json"):
        self.config = CollectorConfig(config_path)
        
        # Set up logging
        self._setup_logging()
        
        # Initialize components
        self.data_fetcher = DataFetcher(self.config, self.logger)
        self.liquidity_fetcher = LiquidityFetcher(self.config, self.logger)
        self.storage = DataStorage(logger=self.logger)
        
        # State
        self.running = False
        self.cycle_count = 0
        self.last_vix = None
    
    def _setup_logging(self):
        """Set up logging with daily rotation"""
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        
        # Create rotating file handler
        file_handler = TimedRotatingFileHandler(
            filename='logs/data_collector.log',
            when='midnight',
            interval=1,
            backupCount=30,
            encoding='utf-8'
        )
        file_handler.suffix = '%Y%m%d.log'
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        
        # Configure logger
        self.logger = logging.getLogger('DataCollector')
        self.logger.setLevel(logging.INFO)
        self.logger.handlers = []
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def backfill(self, days: int = 7, skip_integrity_check: bool = False):
        """
        Backfill historical data for the specified number of days
        
        Args:
            days: Number of days to backfill
            skip_integrity_check: Skip the pre-backfill integrity check
        """
        # Run integrity check before backfill (unless skipped)
        if not skip_integrity_check:
            print("[*] Running pre-backfill integrity check...")
            self.logger.info("[BACKFILL] Running integrity check before backfill...")
            
            results = self.storage.check_integrity()
            total_issues = (
                results['duplicates'] + 
                results['null_values'] + 
                results['invalid_prices'] + 
                results['liquidity_duplicates']
            )
            
            if total_issues > 0:
                print(f"[!] Found {total_issues} issues - cleaning up first...")
                self.storage.full_integrity_check_and_fix()
                print("[OK] Cleanup complete")
            else:
                print("[OK] Data integrity verified")
        
        self.logger.info(f"[BACKFILL] Starting backfill for last {days} days...")
        print(f"[*] Backfilling {days} days of data...")
        
        symbols = self.config.get_symbols()
        total_saved = 0
        
        for symbol in symbols:
            self.logger.info(f"[BACKFILL] Fetching {symbol}...")
            
            try:
                import yfinance as yf
                
                ticker = yf.Ticker(symbol)
                df = ticker.history(period=f'{days}d', interval='1m')
                
                if not df.empty:
                    saved = self.storage.save_price_dataframe(symbol, df)
                    total_saved += saved
                    self.logger.info(f"[BACKFILL] Saved {saved} records for {symbol}")
                else:
                    self.logger.warning(f"[BACKFILL] No data for {symbol}")
                    
            except Exception as e:
                self.logger.error(f"[BACKFILL] Error fetching {symbol}: {e}")
            
            time.sleep(1)  # Rate limiting
        
        self.logger.info(f"[BACKFILL] Complete! Saved {total_saved} total records")
        print(f"[OK] Backfill complete - saved {total_saved:,} records")
        
        # Post-backfill integrity check and dedup
        print("[*] Running post-backfill integrity check...")
        self.logger.info("[BACKFILL] Running post-backfill integrity check...")
        
        results = self.storage.check_integrity()
        if results['duplicates'] > 0 or results['liquidity_duplicates'] > 0:
            print(f"[!] Found {results['duplicates']} duplicates - removing...")
            self.storage.remove_duplicates()
            print("[OK] Duplicates removed")
        
        # Optimize after large insert
        if total_saved > 1000:
            print("[*] Optimizing database...")
            self.storage.optimize_database()
            print("[OK] Database optimized")
        
        return total_saved
    
    def collect_cycle(self) -> Tuple[int, int]:
        """
        Run a single collection cycle
        
        Returns:
            Tuple of (price_records_saved, liquidity_records_saved)
        """
        self.cycle_count += 1
        cycle_start = time.time()
        
        is_open, status = get_market_status()
        symbols = self.config.get_symbols()
        trading_symbol = self.config.get_trading_symbol()
        
        self.logger.info(f"[CYCLE {self.cycle_count}] Market: {status}")
        
        price_saved = 0
        liquidity_saved = 0
        
        # ===== PRICE DATA COLLECTION =====
        # Collect even when market is closed (for after-hours context)
        for symbol in symbols:
            try:
                data = self.data_fetcher.get_price_data(symbol, period='1d', interval='1m')
                
                if data:
                    # Save latest data point
                    if self.storage.save_price_data(symbol, data):
                        price_saved += 1
                        
                        # Store VIX for liquidity context
                        if 'VIX' in symbol.upper():
                            self.last_vix = data.get('close')
                        
                        self.logger.debug(f"[PRICE] {symbol}: ${data.get('close', 0):.2f}")
                        
            except Exception as e:
                self.logger.error(f"[PRICE] Error collecting {symbol}: {e}")
        
        # ===== LIQUIDITY DATA COLLECTION (Market Hours Only) =====
        if is_open:
            try:
                # Get current price of trading symbol
                price_data = self.data_fetcher.get_price_data(trading_symbol)
                
                if price_data:
                    current_price = price_data.get('close', 0)
                    
                    # Collect ATM liquidity
                    snapshots = self.liquidity_fetcher.get_atm_liquidity(
                        trading_symbol, current_price
                    )
                    
                    # Also collect wider range every 5 minutes
                    if self.cycle_count % 5 == 0:
                        range_snapshots = self.liquidity_fetcher.get_strike_range_liquidity(
                            trading_symbol, current_price, range_pct=0.05
                        )
                        snapshots.extend(range_snapshots)
                    
                    # Add VIX context
                    for snapshot in snapshots:
                        snapshot['vix_value'] = self.last_vix
                    
                    # Save liquidity snapshots
                    for snapshot in snapshots:
                        if self.storage.save_liquidity_snapshot(snapshot):
                            liquidity_saved += 1
                    
                    if snapshots:
                        atm_call = next((s for s in snapshots if s['option_type'] == 'CALL'), None)
                        atm_put = next((s for s in snapshots if s['option_type'] == 'PUT'), None)
                        
                        if atm_call:
                            self.logger.info(
                                f"[LIQUIDITY] {trading_symbol} ATM CALL: "
                                f"${atm_call['strike_price']:.0f} | "
                                f"Spread: {atm_call['spread_pct']:.1f}% | "
                                f"Vol: {atm_call['volume']} | "
                                f"OI: {atm_call['open_interest']} | "
                                f"Quality: {atm_call['quality_score']:.0f}/100"
                            )
                        if atm_put:
                            self.logger.info(
                                f"[LIQUIDITY] {trading_symbol} ATM PUT: "
                                f"${atm_put['strike_price']:.0f} | "
                                f"Spread: {atm_put['spread_pct']:.1f}% | "
                                f"Vol: {atm_put['volume']} | "
                                f"OI: {atm_put['open_interest']} | "
                                f"Quality: {atm_put['quality_score']:.0f}/100"
                            )
                    
            except Exception as e:
                self.logger.error(f"[LIQUIDITY] Error collecting: {e}")
        
        # Log cycle
        duration_ms = int((time.time() - cycle_start) * 1000)
        self.storage.log_collection(
            symbols=symbols,
            price_records=price_saved,
            liquidity_records=liquidity_saved,
            market_open=is_open,
            duration_ms=duration_ms
        )
        
        self.logger.info(
            f"[CYCLE {self.cycle_count}] Saved: {price_saved} price, "
            f"{liquidity_saved} liquidity ({duration_ms}ms)"
        )
        
        return price_saved, liquidity_saved
    
    def run(self, interval_seconds: int = 60, integrity_check_interval: int = 60):
        """
        Run the collector continuously
        
        Args:
            interval_seconds: Time between collection cycles
            integrity_check_interval: Run integrity check every N cycles (default: 60 = ~1 hour)
        """
        self.running = True
        self.logger.info("="*70)
        self.logger.info("DATA COLLECTOR STARTED")
        self.logger.info(f"Symbols: {', '.join(self.config.get_symbols())}")
        self.logger.info(f"Trading Symbol (Options): {self.config.get_trading_symbol()}")
        self.logger.info(f"Interval: {interval_seconds}s")
        self.logger.info(f"Integrity check every: {integrity_check_interval} cycles")
        self.logger.info("="*70)
        
        print(f"\n[*] Data Collector starting...")
        print(f"[*] Symbols: {', '.join(self.config.get_symbols())}")
        print(f"[*] Interval: {interval_seconds}s")
        print(f"[*] Log file: logs/data_collector.log")
        print()
        
        # ===== STARTUP INTEGRITY CHECK =====
        print("[*] Running startup integrity check...")
        self.logger.info("[STARTUP] Running integrity check...")
        
        results = self.storage.check_integrity()
        total_issues = (
            results['duplicates'] + 
            results['null_values'] + 
            results['invalid_prices'] + 
            results['liquidity_duplicates']
        )
        
        if total_issues > 0:
            print(f"[!] Found {total_issues} issues - auto-fixing...")
            self.logger.warning(f"[STARTUP] Found {total_issues} issues:")
            for issue in results['issues']:
                self.logger.warning(f"  - {issue}")
            
            # Auto-fix
            fix_results = self.storage.full_integrity_check_and_fix()
            
            fixed = (
                fix_results.get('duplicates_removed', {}).get('price_duplicates_removed', 0) +
                fix_results.get('duplicates_removed', {}).get('liquidity_duplicates_removed', 0) +
                fix_results.get('invalid_removed', {}).get('null_records_removed', 0) +
                fix_results.get('invalid_removed', {}).get('invalid_price_removed', 0) +
                fix_results.get('invalid_removed', {}).get('invalid_liquidity_removed', 0)
            )
            
            print(f"[OK] Fixed {fixed} issues")
            self.logger.info(f"[STARTUP] Fixed {fixed} issues")
            
            # Verify fix
            final_check = fix_results.get('final_check', {})
            remaining = (
                final_check.get('duplicates', 0) + 
                final_check.get('null_values', 0) + 
                final_check.get('invalid_prices', 0)
            )
            
            if remaining > 0:
                print(f"[WARN] {remaining} issues remain - check logs")
                self.logger.warning(f"[STARTUP] {remaining} issues remain after fix")
            else:
                print("[OK] All issues resolved")
        else:
            print("[OK] Data integrity verified - no issues found")
            self.logger.info("[STARTUP] Data integrity check passed ✓")
        
        # Show quick stats
        stats = self.storage.get_data_stats()
        total_price_records = sum(s.get('records', 0) for s in stats.get('symbols', {}).values())
        total_liquidity = stats.get('liquidity', {}).get('total_snapshots', 0)
        print(f"[*] Database: {total_price_records:,} price records, {total_liquidity:,} liquidity snapshots")
        
        is_open, status = get_market_status()
        print(f"[*] Market: {status}")
        print()
        print(f"[*] Data Collector running (Ctrl+C to stop)")
        print(f"[*] Integrity check: every {integrity_check_interval} cycles (~{integrity_check_interval} min)")
        print()
        
        try:
            while self.running:
                cycle_start = time.time()
                
                self.collect_cycle()
                
                # Periodic integrity check and cleanup
                if self.cycle_count % integrity_check_interval == 0 and self.cycle_count > 0:
                    self.logger.info("[INTEGRITY] Running periodic integrity check...")
                    results = self.storage.check_integrity()
                    
                    total_issues = (
                        results['duplicates'] + 
                        results['null_values'] + 
                        results['invalid_prices'] + 
                        results['liquidity_duplicates']
                    )
                    
                    if total_issues > 0:
                        self.logger.warning(f"[INTEGRITY] Found {total_issues} issues - auto-fixing...")
                        fix_results = self.storage.full_integrity_check_and_fix()
                        
                        fixed = (
                            fix_results.get('duplicates_removed', {}).get('price_duplicates_removed', 0) +
                            fix_results.get('duplicates_removed', {}).get('liquidity_duplicates_removed', 0) +
                            fix_results.get('invalid_removed', {}).get('null_records_removed', 0) +
                            fix_results.get('invalid_removed', {}).get('invalid_price_removed', 0)
                        )
                        self.logger.info(f"[INTEGRITY] Fixed {fixed} issues")
                    else:
                        self.logger.info("[INTEGRITY] No issues found ✓")
                
                # Wait for next cycle
                elapsed = time.time() - cycle_start
                wait_time = max(0, interval_seconds - elapsed)
                
                if wait_time > 0:
                    time.sleep(wait_time)
                    
        except KeyboardInterrupt:
            self.logger.info("[STOP] Collector stopped by user")
            print("\n[*] Stopped")
            
            # Final integrity check on shutdown
            self.logger.info("[SHUTDOWN] Running final integrity check...")
            results = self.storage.check_integrity()
            if results['duplicates'] > 0 or results['liquidity_duplicates'] > 0:
                self.logger.info("[SHUTDOWN] Cleaning up duplicates before exit...")
                self.storage.remove_duplicates()
            
        except Exception as e:
            self.logger.error(f"[ERROR] Collector crashed: {e}")
            import traceback
            traceback.print_exc()
    
    def print_status(self):
        """Print collection status"""
        stats = self.storage.get_data_stats()
        
        print("\n" + "="*70)
        print("DATA COLLECTOR STATUS")
        print("="*70)
        
        print("\n📊 PRICE DATA:")
        for symbol, info in stats.get('symbols', {}).items():
            print(f"  {symbol}:")
            print(f"    Records: {info['records']:,}")
            print(f"    Range: {info['first']} → {info['last']}")
        
        print("\n💧 LIQUIDITY DATA:")
        liq = stats.get('liquidity', {})
        print(f"  Total Snapshots: {liq.get('total_snapshots', 0):,}")
        print(f"  Average Quality: {liq.get('avg_quality', 0)}/100")
        print(f"  Range: {liq.get('first')} → {liq.get('last')}")
        
        print("\n📈 LAST 24 HOURS:")
        col = stats.get('collection', {})
        print(f"  Collection Cycles: {col.get('cycles_24h', 0)}")
        print(f"  Price Records: {col.get('price_records_24h', 0):,}")
        print(f"  Liquidity Records: {col.get('liquidity_records_24h', 0):,}")
        
        is_open, status = get_market_status()
        print(f"\n🕐 MARKET STATUS: {status}")
        
        print("="*70 + "\n")
    
    def print_integrity_check(self):
        """Print data integrity check results"""
        print("\n" + "="*70)
        print("DATA INTEGRITY CHECK")
        print("="*70)
        
        results = self.storage.check_integrity()
        
        print("\n🔍 CHECKING DATA INTEGRITY...")
        
        # Price data issues
        print("\n📊 PRICE DATA:")
        print(f"  Duplicate Records: {results['duplicates']}", end="")
        print(" ✓" if results['duplicates'] == 0 else " ⚠️")
        
        print(f"  NULL Values: {results['null_values']}", end="")
        print(" ✓" if results['null_values'] == 0 else " ⚠️")
        
        print(f"  Invalid Prices: {results['invalid_prices']}", end="")
        print(" ✓" if results['invalid_prices'] == 0 else " ⚠️")
        
        # Liquidity data issues
        print("\n💧 LIQUIDITY DATA:")
        print(f"  Duplicate Records: {results['liquidity_duplicates']}", end="")
        print(" ✓" if results['liquidity_duplicates'] == 0 else " ⚠️")
        
        # Issues detail
        if results['issues']:
            print("\n⚠️  ISSUES FOUND:")
            for issue in results['issues']:
                print(f"  - {issue}")
        else:
            print("\n✅ NO ISSUES FOUND - Data integrity is good!")
        
        # Recommendations
        total_issues = (
            results['duplicates'] + 
            results['null_values'] + 
            results['invalid_prices'] + 
            results['liquidity_duplicates']
        )
        
        if total_issues > 0:
            print("\n💡 RECOMMENDATION:")
            print("  Run: python data_collector.py --fix")
            print("  This will remove duplicates and invalid records")
        
        print("\n" + "="*70 + "\n")
    
    def fix_integrity_issues(self):
        """Fix all data integrity issues"""
        print("\n" + "="*70)
        print("FIXING DATA INTEGRITY ISSUES")
        print("="*70)
        
        # Run full check and fix
        results = self.storage.full_integrity_check_and_fix()
        
        print("\n🔧 INITIAL CHECK:")
        initial = results['check']
        print(f"  Duplicates found: {initial['duplicates']}")
        print(f"  Liquidity duplicates: {initial['liquidity_duplicates']}")
        print(f"  NULL values: {initial['null_values']}")
        print(f"  Invalid prices: {initial['invalid_prices']}")
        
        # Report what was fixed
        dup_removed = results.get('duplicates_removed', {})
        inv_removed = results.get('invalid_removed', {})
        
        print("\n🗑️  REMOVED:")
        print(f"  Price duplicates: {dup_removed.get('price_duplicates_removed', 0)}")
        print(f"  Liquidity duplicates: {dup_removed.get('liquidity_duplicates_removed', 0)}")
        print(f"  NULL records: {inv_removed.get('null_records_removed', 0)}")
        print(f"  Invalid prices: {inv_removed.get('invalid_price_removed', 0)}")
        print(f"  Invalid liquidity: {inv_removed.get('invalid_liquidity_removed', 0)}")
        
        if results.get('optimized'):
            print("\n📦 DATABASE OPTIMIZED")
        
        # Final check
        final = results.get('final_check', {})
        print("\n✅ FINAL CHECK:")
        print(f"  Duplicates: {final.get('duplicates', 0)}")
        print(f"  NULL values: {final.get('null_values', 0)}")
        print(f"  Invalid prices: {final.get('invalid_prices', 0)}")
        
        if not final.get('issues'):
            print("\n✅ ALL ISSUES FIXED - Data integrity restored!")
        else:
            print("\n⚠️  Some issues may remain - check logs")
        
        print("\n" + "="*70 + "\n")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Standalone Data Collector for Training Data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python data_collector.py                    # Run continuously
  python data_collector.py --backfill 7      # Backfill 7 days, then run
  python data_collector.py --backfill-only 30 # Only backfill 30 days
  python data_collector.py --status          # Show data status
  python data_collector.py --check           # Check data integrity
  python data_collector.py --fix             # Fix integrity issues
  python data_collector.py --optimize        # Optimize database
        '''
    )
    
    parser.add_argument('--backfill', type=int, metavar='DAYS',
                       help='Backfill N days of data before running')
    parser.add_argument('--backfill-only', type=int, metavar='DAYS',
                       help='Only backfill N days of data, then exit')
    parser.add_argument('--status', action='store_true',
                       help='Show data collection status and exit')
    parser.add_argument('--check', action='store_true',
                       help='Run data integrity check and exit')
    parser.add_argument('--fix', action='store_true',
                       help='Fix data integrity issues (remove duplicates, invalid records)')
    parser.add_argument('--optimize', action='store_true',
                       help='Optimize database (vacuum and reindex)')
    parser.add_argument('--interval', type=int, default=60,
                       help='Collection interval in seconds (default: 60)')
    parser.add_argument('--config', type=str, default='config.json',
                       help='Path to config file (default: config.json)')
    
    args = parser.parse_args()
    
    # Initialize collector
    collector = DataCollector(config_path=args.config)
    
    # Status only
    if args.status:
        collector.print_status()
        return
    
    # Integrity check only
    if args.check:
        collector.print_integrity_check()
        return
    
    # Fix integrity issues
    if args.fix:
        collector.fix_integrity_issues()
        return
    
    # Optimize database
    if args.optimize:
        print("[*] Optimizing database...")
        if collector.storage.optimize_database():
            print("[OK] Database optimized successfully")
        else:
            print("[ERROR] Database optimization failed")
        return
    
    # Backfill only
    if args.backfill_only:
        collector.backfill(days=args.backfill_only)
        collector.print_status()
        return
    
    # Backfill then run
    if args.backfill:
        collector.backfill(days=args.backfill)
    
    # Run continuously
    collector.run(interval_seconds=args.interval)


if __name__ == "__main__":
    main()

