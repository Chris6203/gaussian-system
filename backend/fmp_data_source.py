#!/usr/bin/env python3
"""
Financial Modeling Prep (FMP) Data Source
Fetches real-time and historical 1-minute data for stocks and ETFs

API Documentation: https://financialmodelingprep.com/developer/docs
"""
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List
import time
import logging
import sqlite3
from pathlib import Path

class FMPDataSource:
    """Data source for Financial Modeling Prep API"""
    
    def __init__(self, api_key: str, db_path: str = 'data/db/historical.db', 
                 requests_per_minute: int = 300, config: dict = None):
        self.api_key = api_key
        self.base_url = "https://financialmodelingprep.com"
        self.stable_url = "https://financialmodelingprep.com/stable"
        self.logger = logging.getLogger(__name__)
        self.db_path = db_path
        
        # Rate limiting - FMP Starter plan: 300 requests/minute
        # Can be configured via config dict or constructor parameter
        self.request_times = []
        self.rate_limit_window = 60  # seconds
        
        # Get rate limit from config if provided
        if config:
            fmp_config = config.get('credentials', {}).get('fmp', {})
            self.max_requests_per_minute = fmp_config.get('requests_per_minute', requests_per_minute)
        else:
            self.max_requests_per_minute = requests_per_minute
        
        # Track daily usage (for logging purposes)
        self.daily_requests = 0
        self.last_reset_date = datetime.now().date()
        
        # Initialize database for storing historical data
        self._init_database()
        
        self.logger.info(f"[FMP] Data source initialized (rate limit: {self.max_requests_per_minute} req/min)")
    
    def _init_database(self):
        """Initialize database tables for FMP historical data"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create table for 1-minute historical data
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS fmp_intraday_1m (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        open_price REAL NOT NULL,
                        high_price REAL NOT NULL,
                        low_price REAL NOT NULL,
                        close_price REAL NOT NULL,
                        volume INTEGER NOT NULL,
                        source TEXT DEFAULT 'fmp',
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(symbol, timestamp)
                    )
                ''')
                
                # Create index for faster queries
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_fmp_intraday_symbol_time 
                    ON fmp_intraday_1m(symbol, timestamp)
                ''')
                
                conn.commit()
                self.logger.debug("[FMP] Database initialized")
        except Exception as e:
            self.logger.error(f"[FMP] Database init error: {e}")
    
    def _check_daily_limit(self) -> bool:
        """Check daily request count (for logging, Starter plan has no daily limit)"""
        today = datetime.now().date()
        if today != self.last_reset_date:
            self.logger.info(f"[FMP] Daily stats reset - yesterday used {self.daily_requests} requests")
            self.daily_requests = 0
            self.last_reset_date = today
        
        return True  # Starter plan has no daily limit
    
    def _rate_limit(self):
        """Enforce rate limiting"""
        current_time = time.time()
        
        # Remove requests older than window
        self.request_times = [t for t in self.request_times 
                            if current_time - t < self.rate_limit_window]
        
        # Wait if at limit
        if len(self.request_times) >= self.max_requests_per_minute:
            oldest = self.request_times[0]
            wait_time = self.rate_limit_window - (current_time - oldest)
            
            if wait_time > 0:
                self.logger.info(f"[FMP] Rate limit: waiting {wait_time:.1f}s...")
                time.sleep(wait_time + 0.1)
                current_time = time.time()
                self.request_times = [t for t in self.request_times 
                                     if current_time - t < self.rate_limit_window]
        
        self.request_times.append(current_time)
        self.daily_requests += 1
    
    def _make_request(self, endpoint: str, params: dict = None, use_stable: bool = False) -> dict:
        """Make API request with rate limiting (300 req/min for Starter plan)"""
        self._check_daily_limit()  # Just for logging
        self._rate_limit()
        
        if params is None:
            params = {}
        params['apikey'] = self.api_key
        
        base = self.stable_url if use_stable else self.base_url
        url = f"{base}{endpoint}"
        
        try:
            response = requests.get(url, params=params, timeout=15)
            
            if response.status_code == 403:
                # Log more details for debugging
                self.logger.error(f"[FMP] 403 Forbidden - URL: {url[:100]}...")
                self.logger.error(f"[FMP] Response: {response.text[:200]}")
                return {}
            elif response.status_code == 429:
                self.logger.warning("[FMP] Rate limit exceeded")
                time.sleep(60)  # Wait a minute
                return {}
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"[FMP] API request failed: {e}")
            return {}
    
    def get_quote(self, symbol: str) -> dict:
        """Get current quote for a symbol"""
        endpoint = f"/quote?symbol={symbol}"
        data = self._make_request(endpoint, use_stable=True)
        
        if not data or not isinstance(data, list) or len(data) == 0:
            return {}
        
        quote = data[0]
        return {
            'symbol': quote.get('symbol', symbol),
            'price': quote.get('price', 0),
            'change': quote.get('change', 0),
            'change_percent': quote.get('changesPercentage', 0),
            'volume': quote.get('volume', 0),
            'avg_volume': quote.get('avgVolume', 0),
            'market_cap': quote.get('marketCap', 0),
            'pe': quote.get('pe', 0),
            'open': quote.get('open', 0),
            'high': quote.get('dayHigh', 0),
            'low': quote.get('dayLow', 0),
            'previous_close': quote.get('previousClose', 0),
            'timestamp': quote.get('timestamp', 0),
        }
    
    def get_historical_intraday(
        self,
        symbol: str,
        interval: str = '1min',
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Get historical intraday data from FMP.
        
        Args:
            symbol: Stock symbol (e.g., 'SPY', 'AAPL')
            interval: Time interval ('1min', '5min', '15min', '30min', '1hour', '4hour')
            from_date: Start date
            to_date: End date
            
        Returns:
            DataFrame with OHLCV data
        """
        # FMP Starter plan uses the NEW stable API (as of Aug 2025)
        # Correct format: /historical-chart/{interval}?symbol=XXX
        # NOT the old v3 format: /api/v3/historical-chart/{interval}/XXX
        
        # Build params with symbol as query parameter
        params = {'symbol': symbol}
        if from_date:
            params['from'] = from_date.strftime('%Y-%m-%d')
        if to_date:
            params['to'] = to_date.strftime('%Y-%m-%d')
        
        # Use stable API with symbol as query param (correct new format)
        endpoint = f"/historical-chart/{interval}"
        data = self._make_request(endpoint, params=params, use_stable=True)
        
        if not data or not isinstance(data, list):
            self.logger.warning(f"[FMP] No intraday data for {symbol}")
            return pd.DataFrame()
        
        try:
            df = pd.DataFrame(data)
            
            if df.empty:
                return pd.DataFrame()
            
            # Rename columns to standard format
            column_map = {
                'date': 'timestamp',
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume'
            }
            
            df = df.rename(columns=column_map)
            
            # Parse timestamp
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
            
            # Ensure proper column order
            df = df[['open', 'high', 'low', 'close', 'volume']]
            
            # Sort by timestamp (FMP returns newest first)
            df = df.sort_index()
            
            self.logger.info(f"[FMP] Got {len(df)} intraday records for {symbol}")
            
            # Save to database
            self._save_to_database(symbol, df)
            
            return df
            
        except Exception as e:
            self.logger.error(f"[FMP] Error parsing intraday data: {e}")
            return pd.DataFrame()
    
    def get_historical_daily(
        self,
        symbol: str,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Get historical daily data from FMP.
        
        Args:
            symbol: Stock symbol
            from_date: Start date
            to_date: End date
            
        Returns:
            DataFrame with OHLCV data
        """
        endpoint = f"/historical-price-eod/full?symbol={symbol}"
        
        params = {}
        if from_date:
            params['from'] = from_date.strftime('%Y-%m-%d')
        if to_date:
            params['to'] = to_date.strftime('%Y-%m-%d')
        
        data = self._make_request(endpoint, params=params, use_stable=True)
        
        if not data or 'historical' not in data:
            self.logger.warning(f"[FMP] No daily data for {symbol}")
            return pd.DataFrame()
        
        try:
            df = pd.DataFrame(data['historical'])
            
            if df.empty:
                return pd.DataFrame()
            
            # Parse date
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            
            # Select and rename columns
            df = df[['open', 'high', 'low', 'close', 'volume']]
            
            # Sort by date
            df = df.sort_index()
            
            self.logger.info(f"[FMP] Got {len(df)} daily records for {symbol}")
            return df
            
        except Exception as e:
            self.logger.error(f"[FMP] Error parsing daily data: {e}")
            return pd.DataFrame()
    
    def get_historical_data(
        self,
        symbol: str,
        period: str = '1d',
        interval: str = '1m'
    ) -> pd.DataFrame:
        """
        Unified interface matching other data sources.
        
        Args:
            symbol: Stock symbol
            period: Data period ('1d', '5d', '7d', '30d', '90d', 'max')
            interval: Data interval ('1m', '5m', '15m', '1h', '1d')
            
        Returns:
            DataFrame with OHLCV data
        """
        # Map period to date range
        end_date = datetime.now()
        
        period_map = {
            '1d': 1,
            '2d': 2,
            '5d': 5,
            '7d': 7,
            '1mo': 30,
            '30d': 30,
            '3mo': 90,
            '90d': 90,
            '6mo': 180,
            '1y': 365,
            'max': 365 * 5
        }
        
        days = period_map.get(period, 1)
        start_date = end_date - timedelta(days=days)
        
        # Map interval to FMP format
        interval_map = {
            '1m': '1min',
            '1min': '1min',
            '5m': '5min',
            '5min': '5min',
            '15m': '15min',
            '15min': '15min',
            '30m': '30min',
            '30min': '30min',
            '1h': '1hour',
            '1hour': '1hour',
            '4h': '4hour',
            '4hour': '4hour',
            '1d': 'daily',
            '1day': 'daily',
            'daily': 'daily'
        }
        
        fmp_interval = interval_map.get(interval, '1min')
        
        if fmp_interval == 'daily':
            return self.get_historical_daily(symbol, start_date, end_date)
        else:
            return self.get_historical_intraday(symbol, fmp_interval, start_date, end_date)
    
    def get_data(self, symbol: str, period: str = '1d', interval: str = '1m') -> pd.DataFrame:
        """Alias for get_historical_data for compatibility"""
        return self.get_historical_data(symbol, period, interval)
    
    def _save_to_database(self, symbol: str, df: pd.DataFrame):
        """Save intraday data to database for persistence"""
        if df.empty:
            return
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                saved_count = 0
                for timestamp, row in df.iterrows():
                    try:
                        cursor.execute('''
                            INSERT OR REPLACE INTO fmp_intraday_1m 
                            (symbol, timestamp, open_price, high_price, low_price, close_price, volume, source)
                            VALUES (?, ?, ?, ?, ?, ?, ?, 'fmp')
                        ''', (
                            symbol,
                            timestamp.isoformat(),
                            float(row['open']),
                            float(row['high']),
                            float(row['low']),
                            float(row['close']),
                            int(row['volume'])
                        ))
                        saved_count += 1
                    except Exception as e:
                        self.logger.debug(f"[FMP] Could not save record: {e}")
                
                conn.commit()
                self.logger.debug(f"[FMP] Saved {saved_count} records to database for {symbol}")
                
        except Exception as e:
            self.logger.error(f"[FMP] Database save error: {e}")
    
    def get_from_database(
        self,
        symbol: str,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Get historical data from local database.
        
        Args:
            symbol: Stock symbol
            from_date: Start date
            to_date: End date
            limit: Maximum records to return
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = '''
                    SELECT timestamp, open_price, high_price, low_price, close_price, volume
                    FROM fmp_intraday_1m
                    WHERE symbol = ?
                '''
                params = [symbol]
                
                if from_date:
                    query += ' AND timestamp >= ?'
                    params.append(from_date.isoformat())
                
                if to_date:
                    query += ' AND timestamp <= ?'
                    params.append(to_date.isoformat())
                
                query += ' ORDER BY timestamp DESC LIMIT ?'
                params.append(limit)
                
                df = pd.read_sql_query(query, conn, params=params)
                
                if df.empty:
                    return pd.DataFrame()
                
                # Rename columns
                df = df.rename(columns={
                    'open_price': 'open',
                    'high_price': 'high',
                    'low_price': 'low',
                    'close_price': 'close'
                })
                
                # Parse timestamp and set as index
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.set_index('timestamp')
                df = df.sort_index()
                
                return df
                
        except Exception as e:
            self.logger.error(f"[FMP] Database read error: {e}")
            return pd.DataFrame()
    
    def fetch_and_store_historical(
        self,
        symbols: List[str],
        days_back: int = 30,
        interval: str = '1min'
    ) -> dict:
        """
        Fetch historical data for multiple symbols and store in database.
        
        This is useful for backfilling historical 1-minute data.
        
        Args:
            symbols: List of stock symbols
            days_back: Number of days of history to fetch
            interval: Data interval
            
        Returns:
            Dict with results per symbol
        """
        results = {}
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        self.logger.info(f"[FMP] Fetching {days_back} days of {interval} data for {len(symbols)} symbols")
        
        for symbol in symbols:
            try:
                self.logger.info(f"[FMP] Fetching {symbol}...")
                df = self.get_historical_intraday(symbol, interval, start_date, end_date)
                
                results[symbol] = {
                    'success': not df.empty,
                    'records': len(df),
                    'start': df.index.min() if not df.empty else None,
                    'end': df.index.max() if not df.empty else None
                }
                
                # Small delay between symbols to be nice to the API
                time.sleep(0.5)
                
            except Exception as e:
                self.logger.error(f"[FMP] Error fetching {symbol}: {e}")
                results[symbol] = {'success': False, 'error': str(e)}
        
        return results
    
    def get_available_symbols(self) -> List[str]:
        """Get list of all available stock symbols from FMP"""
        endpoint = "/stock-list"
        data = self._make_request(endpoint, use_stable=True)
        
        if not data or not isinstance(data, list):
            return []
        
        return [item.get('symbol') for item in data if item.get('symbol')]
    
    def search_symbol(self, query: str) -> List[dict]:
        """Search for symbols by name or ticker"""
        endpoint = f"/search-symbol?query={query}"
        data = self._make_request(endpoint, use_stable=True)
        
        if not data or not isinstance(data, list):
            return []
        
        return data


# Standalone test
if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    # Test with API key from command line or environment
    api_key = sys.argv[1] if len(sys.argv) > 1 else "demo"
    
    fmp = FMPDataSource(api_key)
    
    # Test quote
    print("\n=== Testing Quote ===")
    quote = fmp.get_quote("AAPL")
    print(f"AAPL Quote: {quote}")
    
    # Test intraday data
    print("\n=== Testing Intraday Data ===")
    df = fmp.get_historical_data("SPY", period="1d", interval="1m")
    print(f"SPY 1-min data: {len(df)} records")
    if not df.empty:
        print(df.tail())
    
    # Test daily data
    print("\n=== Testing Daily Data ===")
    df_daily = fmp.get_historical_data("SPY", period="30d", interval="1d")
    print(f"SPY daily data: {len(df_daily)} records")
    if not df_daily.empty:
        print(df_daily.tail())



