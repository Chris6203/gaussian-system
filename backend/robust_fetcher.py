#!/usr/bin/env python3
"""
Robust Data Fetcher
===================

Wraps data fetching with:
1. Exponential backoff retries
2. Fallback to local cache
3. Request rate limiting
4. Health monitoring
5. Incremental updates (append new bars only)

Usage:
    fetcher = RobustFetcher(data_source)
    data = fetcher.get_data('SPY', period='7d', interval='1m')
"""

import logging
import time
import sqlite3
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Callable
from functools import wraps
from threading import Lock
import pandas as pd

logger = logging.getLogger(__name__)


class RateLimiter:
    """Simple rate limiter with sliding window."""
    
    def __init__(self, max_calls: int = 60, window_seconds: int = 60):
        self.max_calls = max_calls
        self.window_seconds = window_seconds
        self._calls: list = []
        self._lock = Lock()
    
    def acquire(self) -> bool:
        """Try to acquire a rate limit slot. Returns True if allowed."""
        with self._lock:
            now = time.time()
            # Remove old calls outside window
            self._calls = [t for t in self._calls if now - t < self.window_seconds]
            
            if len(self._calls) < self.max_calls:
                self._calls.append(now)
                return True
            return False
    
    def wait_and_acquire(self, timeout: float = 30.0) -> bool:
        """Wait until rate limit allows, then acquire."""
        start = time.time()
        while time.time() - start < timeout:
            if self.acquire():
                return True
            time.sleep(0.5)
        return False


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exceptions: tuple = (Exception,)
):
    """
    Decorator for retrying with exponential backoff.
    
    Args:
        max_retries: Maximum retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        exceptions: Tuple of exceptions to catch
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt < max_retries:
                        delay = min(base_delay * (2 ** attempt), max_delay)
                        logger.warning(
                            f"Retry {attempt + 1}/{max_retries} for {func.__name__}: {e}. "
                            f"Waiting {delay:.1f}s..."
                        )
                        time.sleep(delay)
                    else:
                        logger.error(f"All {max_retries} retries failed for {func.__name__}: {e}")
            
            raise last_exception
        return wrapper
    return decorator


class LocalCache:
    """
    Local SQLite cache for market data.
    
    Stores data by (symbol, interval) with timestamps for incremental updates.
    """
    
    def __init__(self, db_path: str = "data/cache/market_cache.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self._lock = Lock()
    
    def _init_db(self):
        """Initialize cache database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS market_data (
                symbol TEXT,
                interval TEXT,
                timestamp TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                PRIMARY KEY (symbol, interval, timestamp)
            )
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_market_symbol_interval 
            ON market_data(symbol, interval)
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cache_meta (
                symbol TEXT,
                interval TEXT,
                last_update TEXT,
                last_timestamp TEXT,
                PRIMARY KEY (symbol, interval)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def get(self, symbol: str, interval: str, lookback_days: int = 7) -> Optional[pd.DataFrame]:
        """
        Get cached data for a symbol.
        
        Args:
            symbol: Symbol to fetch
            interval: Data interval (e.g., '1m', '15m')
            lookback_days: How many days to look back
            
        Returns:
            DataFrame or None if not cached
        """
        with self._lock:
            try:
                conn = sqlite3.connect(self.db_path)
                cutoff = (datetime.now() - timedelta(days=lookback_days)).isoformat()
                
                df = pd.read_sql_query(
                    """
                    SELECT timestamp, open, high, low, close, volume
                    FROM market_data
                    WHERE symbol = ? AND interval = ? AND timestamp >= ?
                    ORDER BY timestamp
                    """,
                    conn,
                    params=(symbol, interval, cutoff)
                )
                conn.close()
                
                if df.empty:
                    return None
                
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                
                logger.debug(f"Cache hit: {symbol}/{interval} - {len(df)} rows")
                return df
                
            except Exception as e:
                logger.warning(f"Cache read error: {e}")
                return None
    
    def put(self, symbol: str, interval: str, data: pd.DataFrame) -> int:
        """
        Store data in cache.
        
        Args:
            symbol: Symbol
            interval: Data interval
            data: DataFrame with OHLCV data
            
        Returns:
            Number of rows inserted/updated
        """
        if data.empty:
            return 0
        
        with self._lock:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                rows_inserted = 0
                for idx, row in data.iterrows():
                    timestamp = idx.isoformat() if hasattr(idx, 'isoformat') else str(idx)
                    
                    cursor.execute('''
                        INSERT OR REPLACE INTO market_data 
                        (symbol, interval, timestamp, open, high, low, close, volume)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        symbol, interval, timestamp,
                        float(row.get('open', row.get('Open', 0))),
                        float(row.get('high', row.get('High', 0))),
                        float(row.get('low', row.get('Low', 0))),
                        float(row.get('close', row.get('Close', 0))),
                        int(row.get('volume', row.get('Volume', 0)))
                    ))
                    rows_inserted += 1
                
                # Update metadata
                last_ts = data.index[-1].isoformat() if hasattr(data.index[-1], 'isoformat') else str(data.index[-1])
                cursor.execute('''
                    INSERT OR REPLACE INTO cache_meta (symbol, interval, last_update, last_timestamp)
                    VALUES (?, ?, ?, ?)
                ''', (symbol, interval, datetime.now().isoformat(), last_ts))
                
                conn.commit()
                conn.close()
                
                logger.debug(f"Cache write: {symbol}/{interval} - {rows_inserted} rows")
                return rows_inserted
                
            except Exception as e:
                logger.warning(f"Cache write error: {e}")
                return 0
    
    def get_last_timestamp(self, symbol: str, interval: str) -> Optional[datetime]:
        """Get the last timestamp for incremental updates."""
        with self._lock:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute(
                    "SELECT last_timestamp FROM cache_meta WHERE symbol = ? AND interval = ?",
                    (symbol, interval)
                )
                row = cursor.fetchone()
                conn.close()
                
                if row and row[0]:
                    return datetime.fromisoformat(row[0])
                return None
                
            except Exception as e:
                logger.warning(f"Error getting last timestamp: {e}")
                return None
    
    def cleanup(self, max_age_days: int = 30):
        """Remove old cached data."""
        with self._lock:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cutoff = (datetime.now() - timedelta(days=max_age_days)).isoformat()
                cursor.execute("DELETE FROM market_data WHERE timestamp < ?", (cutoff,))
                deleted = cursor.rowcount
                
                conn.commit()
                conn.close()
                
                if deleted > 0:
                    logger.info(f"Cache cleanup: removed {deleted} rows older than {max_age_days} days")
                    
            except Exception as e:
                logger.warning(f"Cache cleanup error: {e}")


class RobustFetcher:
    """
    Robust data fetcher with retries, caching, and rate limiting.
    """
    
    def __init__(
        self,
        data_source: Any,
        cache_enabled: bool = True,
        cache_db_path: str = "data/cache/market_cache.db",
        max_retries: int = 3,
        rate_limit_calls: int = 60,
        rate_limit_window: int = 60
    ):
        """
        Args:
            data_source: Primary data source (e.g., DataSourceRouter)
            cache_enabled: Whether to use local cache
            cache_db_path: Path to cache database
            max_retries: Max retry attempts
            rate_limit_calls: Max calls per rate limit window
            rate_limit_window: Rate limit window in seconds
        """
        self.data_source = data_source
        self.cache_enabled = cache_enabled
        self.max_retries = max_retries
        
        # Initialize cache
        self.cache = LocalCache(cache_db_path) if cache_enabled else None
        
        # Initialize rate limiter
        self.rate_limiter = RateLimiter(rate_limit_calls, rate_limit_window)
        
        # Track fetch statistics
        self._stats = {
            'total_fetches': 0,
            'cache_hits': 0,
            'api_calls': 0,
            'retries': 0,
            'failures': 0
        }
        
        # Track last fetch times for incremental updates
        self._last_fetch: Dict[str, datetime] = {}
        
        logger.info(f"✅ RobustFetcher initialized (cache={'enabled' if cache_enabled else 'disabled'})")
    
    def get_data(
        self,
        symbol: str,
        period: str = "7d",
        interval: str = "1m",
        incremental: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Fetch data with retries, caching, and optional incremental updates.
        
        Args:
            symbol: Symbol to fetch
            period: Data period (e.g., '7d')
            interval: Data interval (e.g., '1m')
            incremental: If True, only fetch new data since last fetch
            
        Returns:
            DataFrame with OHLCV data, or None on failure
        """
        self._stats['total_fetches'] += 1
        cache_key = f"{symbol}_{interval}"
        
        # Try incremental fetch first
        if incremental and self.cache and cache_key in self._last_fetch:
            result = self._fetch_incremental(symbol, period, interval)
            if result is not None:
                return result
        
        # Full fetch with retries
        result = self._fetch_with_retries(symbol, period, interval)
        
        if result is not None:
            self._last_fetch[cache_key] = datetime.now()
            
            # Update cache
            if self.cache:
                self.cache.put(symbol, interval, result)
        
        elif self.cache:
            # Fallback to cache on failure
            cached = self.cache.get(symbol, interval, lookback_days=self._period_to_days(period))
            if cached is not None:
                self._stats['cache_hits'] += 1
                logger.warning(f"Using cached data for {symbol} (API fetch failed)")
                return cached
        
        return result
    
    def _fetch_incremental(
        self,
        symbol: str,
        period: str,
        interval: str
    ) -> Optional[pd.DataFrame]:
        """
        Fetch only new data since last fetch.
        
        Returns combined cached + new data, or None to trigger full fetch.
        """
        if not self.cache:
            return None
        
        last_ts = self.cache.get_last_timestamp(symbol, interval)
        if not last_ts:
            return None
        
        # Only do incremental if last fetch was recent
        age = (datetime.now() - last_ts).total_seconds() / 60
        if age > 60:  # More than 60 minutes old, do full fetch
            return None
        
        try:
            # Fetch new data since last timestamp
            # Most APIs support start/end parameters
            new_data = self._fetch_with_retries(
                symbol, 
                period="1d",  # Fetch last day for incremental
                interval=interval
            )
            
            if new_data is None or new_data.empty:
                # Return cached data if API returns nothing
                return self.cache.get(symbol, interval, lookback_days=self._period_to_days(period))
            
            # Filter to only truly new data
            new_data = new_data[new_data.index > last_ts]
            
            if not new_data.empty:
                # Update cache with new data
                self.cache.put(symbol, interval, new_data)
                logger.debug(f"Incremental update: {symbol}/{interval} - {len(new_data)} new rows")
            
            # Return combined data
            return self.cache.get(symbol, interval, lookback_days=self._period_to_days(period))
            
        except Exception as e:
            logger.debug(f"Incremental fetch failed: {e}")
            return None
    
    def _fetch_with_retries(
        self,
        symbol: str,
        period: str,
        interval: str
    ) -> Optional[pd.DataFrame]:
        """
        Fetch data with exponential backoff retries.
        """
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                # Rate limiting
                if not self.rate_limiter.wait_and_acquire(timeout=30):
                    logger.warning("Rate limit timeout")
                    continue
                
                self._stats['api_calls'] += 1
                
                # Call underlying data source
                result = self.data_source.get_data(symbol, period=period, interval=interval)
                
                if result is not None and not result.empty:
                    return result
                
                # Empty result, might retry
                logger.warning(f"Empty result for {symbol}")
                
            except Exception as e:
                last_exception = e
                self._stats['retries'] += 1
                
                if attempt < self.max_retries:
                    delay = min(1.0 * (2 ** attempt), 30.0)
                    logger.warning(
                        f"Fetch retry {attempt + 1}/{self.max_retries} for {symbol}: {e}. "
                        f"Waiting {delay:.1f}s..."
                    )
                    time.sleep(delay)
        
        self._stats['failures'] += 1
        if last_exception:
            logger.error(f"All retries failed for {symbol}: {last_exception}")
        
        return None
    
    def _period_to_days(self, period: str) -> int:
        """Convert period string to days."""
        period = period.lower()
        if 'd' in period:
            return int(period.replace('d', ''))
        elif 'w' in period:
            return int(period.replace('w', '')) * 7
        elif 'm' in period:
            return int(period.replace('m', '')) * 30
        return 7
    
    def get_stats(self) -> Dict:
        """Get fetch statistics."""
        stats = self._stats.copy()
        if stats['total_fetches'] > 0:
            stats['cache_hit_rate'] = stats['cache_hits'] / stats['total_fetches']
            stats['failure_rate'] = stats['failures'] / stats['total_fetches']
        return stats
    
    def clear_cache(self, symbol: Optional[str] = None):
        """Clear cache for a symbol or all symbols."""
        if self.cache:
            if symbol:
                # Clear specific symbol (would need to implement in LocalCache)
                pass
            else:
                self.cache.cleanup(max_age_days=0)


def create_robust_fetcher(data_source: Any, config: Optional[Dict] = None) -> RobustFetcher:
    """Factory function to create RobustFetcher."""
    defaults = {
        'cache_enabled': True,
        'cache_db_path': 'data/cache/market_cache.db',
        'max_retries': 3,
        'rate_limit_calls': 60,
        'rate_limit_window': 60
    }
    
    if config:
        defaults.update(config)
    
    return RobustFetcher(data_source, **defaults)









