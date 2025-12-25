#!/usr/bin/env python3
"""
Unified Data Provider Clients

Clean abstractions for Tradier and Polygon.io APIs with:
- Standardized interfaces
- Built-in caching
- Rate limiting
- Error handling
"""

import os
import json
import time
import logging
import hashlib
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any, Union
from functools import wraps

logger = logging.getLogger(__name__)

# =============================================================================
# DISK CACHE
# =============================================================================

class DiskCache:
    """
    Simple disk-based cache for API responses using parquet/JSON.
    
    Cache key format: {provider}_{symbol}_{timeframe}_{start}_{end}.parquet
    """
    
    def __init__(self, cache_dir: str = "data_cache"):
        """
        Initialize disk cache.
        
        Args:
            cache_dir: Directory for cached files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict:
        """Load cache metadata from disk."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return {}
    
    def _save_metadata(self):
        """Save cache metadata to disk."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Failed to save cache metadata: {e}")
    
    def _get_cache_key(self, provider: str, symbol: str, timeframe: str, 
                       start: Union[date, datetime], end: Union[date, datetime]) -> str:
        """Generate cache key from parameters."""
        start_str = start.strftime('%Y%m%d') if hasattr(start, 'strftime') else str(start)
        end_str = end.strftime('%Y%m%d') if hasattr(end, 'strftime') else str(end)
        key = f"{provider}_{symbol}_{timeframe}_{start_str}_{end_str}"
        return hashlib.md5(key.encode()).hexdigest()[:16]
    
    def get(self, provider: str, symbol: str, timeframe: str,
            start: Union[date, datetime], end: Union[date, datetime],
            max_age_hours: float = 24.0) -> Optional[pd.DataFrame]:
        """
        Retrieve cached data if available and fresh.
        
        Args:
            provider: Data provider name
            symbol: Symbol name
            timeframe: Data timeframe (e.g., '1m', '1d')
            start: Start date/datetime
            end: End date/datetime
            max_age_hours: Maximum age of cache in hours
            
        Returns:
            Cached DataFrame or None if not available/stale
        """
        cache_key = self._get_cache_key(provider, symbol, timeframe, start, end)
        cache_file = self.cache_dir / f"{cache_key}.parquet"
        
        if not cache_file.exists():
            return None
        
        # Check age
        meta = self.metadata.get(cache_key, {})
        cached_at = meta.get('cached_at')
        if cached_at:
            cached_time = datetime.fromisoformat(cached_at)
            age_hours = (datetime.now() - cached_time).total_seconds() / 3600
            if age_hours > max_age_hours:
                logger.debug(f"Cache stale for {symbol} ({age_hours:.1f}h old)")
                return None
        
        try:
            df = pd.read_parquet(cache_file)
            logger.debug(f"Cache hit: {symbol} ({len(df)} rows)")
            return df
        except Exception as e:
            logger.warning(f"Failed to read cache for {symbol}: {e}")
            return None
    
    def put(self, provider: str, symbol: str, timeframe: str,
            start: Union[date, datetime], end: Union[date, datetime],
            data: pd.DataFrame):
        """
        Store data in cache.
        
        Args:
            provider: Data provider name
            symbol: Symbol name
            timeframe: Data timeframe
            start: Start date/datetime
            end: End date/datetime
            data: DataFrame to cache
        """
        if data is None or data.empty:
            return
        
        cache_key = self._get_cache_key(provider, symbol, timeframe, start, end)
        cache_file = self.cache_dir / f"{cache_key}.parquet"
        
        try:
            data.to_parquet(cache_file)
            self.metadata[cache_key] = {
                'provider': provider,
                'symbol': symbol,
                'timeframe': timeframe,
                'start': str(start),
                'end': str(end),
                'rows': len(data),
                'cached_at': datetime.now().isoformat()
            }
            self._save_metadata()
            logger.debug(f"Cached {symbol}: {len(data)} rows")
        except Exception as e:
            logger.warning(f"Failed to cache {symbol}: {e}")
    
    def clear(self, older_than_days: int = 7):
        """Clear cache entries older than specified days."""
        cutoff = datetime.now() - timedelta(days=older_than_days)
        cleared = 0
        
        for cache_key, meta in list(self.metadata.items()):
            cached_at = meta.get('cached_at')
            if cached_at:
                cached_time = datetime.fromisoformat(cached_at)
                if cached_time < cutoff:
                    cache_file = self.cache_dir / f"{cache_key}.parquet"
                    try:
                        cache_file.unlink(missing_ok=True)
                        del self.metadata[cache_key]
                        cleared += 1
                    except Exception:
                        pass
        
        if cleared > 0:
            self._save_metadata()
            logger.info(f"Cleared {cleared} stale cache entries")


# Global cache instance
_cache = DiskCache()


# =============================================================================
# TRADIER DATA CLIENT
# =============================================================================

class TradierDataClient:
    """
    Clean Tradier API client for market data.
    
    Features:
    - Equity quotes and historical data
    - Options chains with greeks
    - Time series at multiple intervals
    - Built-in rate limiting and caching
    """
    
    BASE_URL = "https://api.tradier.com"
    SANDBOX_URL = "https://sandbox.tradier.com"
    
    # Symbols not supported by Tradier (crypto, forex)
    UNSUPPORTED_PATTERNS = ['-USD', '-EUR', '-GBP', '-JPY']
    UNSUPPORTED_SYMBOLS = {'BTC', 'ETH', 'DOGE', 'SOL', 'XRP'}
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        account_id: Optional[str] = None,
        sandbox: bool = False,
        cache: Optional[DiskCache] = None
    ):
        """
        Initialize Tradier client.
        
        Args:
            api_key: Tradier API key (or set TRADIER_API_KEY env var)
            account_id: Tradier account ID (or set TRADIER_ACCOUNT_ID env var)
            sandbox: Use sandbox environment
            cache: DiskCache instance for caching
        """
        self.api_key = api_key or os.getenv('TRADIER_API_KEY')
        self.account_id = account_id or os.getenv('TRADIER_ACCOUNT_ID')
        self.sandbox = sandbox
        self.base_url = self.SANDBOX_URL if sandbox else self.BASE_URL
        self.cache = cache or _cache
        
        # Rate limiting
        self._last_request_time = 0.0
        self._min_request_interval = 0.5  # 500ms between requests
        
        # Request headers
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Accept': 'application/json'
        }
        
        if not self.api_key:
            logger.warning("TradierDataClient: No API key provided")
    
    def _is_supported(self, symbol: str) -> bool:
        """Check if symbol is supported by Tradier."""
        symbol_upper = symbol.upper()
        if symbol_upper in self.UNSUPPORTED_SYMBOLS:
            return False
        for pattern in self.UNSUPPORTED_PATTERNS:
            if pattern in symbol_upper:
                return False
        return True
    
    def _normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol for Tradier API."""
        # Remove ^ prefix for indices
        if symbol.startswith('^'):
            return symbol[1:]
        return symbol
    
    def _rate_limit(self):
        """Enforce rate limiting."""
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()
    
    def _request(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """Make API request with rate limiting and error handling."""
        if not self.api_key:
            logger.error("Tradier: No API key configured")
            return {}
        
        self._rate_limit()
        url = f"{self.base_url}{endpoint}"
        
        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            logger.warning(f"Tradier HTTP error: {e}")
            return {}
        except requests.exceptions.RequestException as e:
            logger.warning(f"Tradier request failed: {e}")
            return {}
        except json.JSONDecodeError as e:
            logger.warning(f"Tradier JSON decode error: {e}")
            return {}
    
    def get_equity_quote(self, symbol: str) -> Dict[str, Any]:
        """
        Get current quote for an equity/ETF.
        
        Args:
            symbol: Stock symbol (e.g., 'SPY', 'QQQ', 'VIX')
            
        Returns:
            Dict with quote data:
            - symbol, price, bid, ask, volume, change, change_pct
            - high, low, open, previous_close
        """
        if not self._is_supported(symbol):
            logger.debug(f"Tradier: {symbol} not supported")
            return {}
        
        api_symbol = self._normalize_symbol(symbol)
        data = self._request("/v1/markets/quotes", {'symbols': api_symbol})
        
        if not data or 'quotes' not in data:
            return {}
        
        quote = data['quotes'].get('quote', {})
        if not quote:
            return {}
        
        # Handle single quote vs list
        if isinstance(quote, list):
            quote = quote[0] if quote else {}
        
        return {
            'symbol': symbol,
            'price': float(quote.get('last', 0) or 0),
            'bid': float(quote.get('bid', 0) or 0),
            'ask': float(quote.get('ask', 0) or 0),
            'volume': int(quote.get('volume', 0) or 0),
            'change': float(quote.get('change', 0) or 0),
            'change_pct': float(quote.get('change_percentage', 0) or 0),
            'high': float(quote.get('high', 0) or 0),
            'low': float(quote.get('low', 0) or 0),
            'open': float(quote.get('open', 0) or 0),
            'previous_close': float(quote.get('prevclose', 0) or 0),
            'timestamp': datetime.now().isoformat()
        }
    
    def get_equity_quotes(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Get quotes for multiple symbols in one request.
        
        Args:
            symbols: List of stock symbols
            
        Returns:
            Dict mapping symbol to quote data
        """
        # Filter to supported symbols
        supported = [s for s in symbols if self._is_supported(s)]
        if not supported:
            return {}
        
        api_symbols = [self._normalize_symbol(s) for s in supported]
        data = self._request("/v1/markets/quotes", {'symbols': ','.join(api_symbols)})
        
        if not data or 'quotes' not in data:
            return {}
        
        quotes = data['quotes'].get('quote', [])
        if not isinstance(quotes, list):
            quotes = [quotes] if quotes else []
        
        result = {}
        for quote in quotes:
            sym = quote.get('symbol', '')
            result[sym] = {
                'symbol': sym,
                'price': float(quote.get('last', 0) or 0),
                'bid': float(quote.get('bid', 0) or 0),
                'ask': float(quote.get('ask', 0) or 0),
                'volume': int(quote.get('volume', 0) or 0),
                'change': float(quote.get('change', 0) or 0),
                'change_pct': float(quote.get('change_percentage', 0) or 0),
                'high': float(quote.get('high', 0) or 0),
                'low': float(quote.get('low', 0) or 0),
                'open': float(quote.get('open', 0) or 0),
                'previous_close': float(quote.get('prevclose', 0) or 0),
            }
        
        return result
    
    def get_time_series(
        self,
        symbol: str,
        interval: str = '1m',
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Get historical time series data.
        
        Args:
            symbol: Stock symbol
            interval: Bar interval ('1m', '5m', '15m', '1h', '1d')
            start: Start datetime (default: 7 days ago)
            end: End datetime (default: now)
            use_cache: Whether to use disk cache
            
        Returns:
            DataFrame with columns: open, high, low, close, volume
            Index: datetime
        """
        if not self._is_supported(symbol):
            logger.debug(f"Tradier: {symbol} not supported")
            return pd.DataFrame()
        
        api_symbol = self._normalize_symbol(symbol)
        end = end or datetime.now()
        start = start or (end - timedelta(days=7))
        
        # Check cache
        if use_cache:
            cached = self.cache.get('tradier', symbol, interval, start, end)
            if cached is not None:
                return cached
        
        # Map interval to Tradier format
        interval_map = {
            '1m': '1min', '5m': '5min', '15m': '15min',
            '30m': '30min', '1h': '1hour', '1d': 'daily'
        }
        tradier_interval = interval_map.get(interval, '1min')
        
        if interval in ['1m', '5m', '15m', '30m', '1h']:
            # Intraday data via timesales
            df = self._get_timesales(api_symbol, tradier_interval, start, end)
        else:
            # Daily data via history
            df = self._get_history(api_symbol, start, end)
        
        # Cache result
        if use_cache and not df.empty:
            self.cache.put('tradier', symbol, interval, start, end, df)
        
        return df
    
    def _get_timesales(self, symbol: str, interval: str, 
                       start: datetime, end: datetime) -> pd.DataFrame:
        """Get intraday time series."""
        params = {
            'symbol': symbol,
            'interval': interval,
            'start': start.strftime('%Y-%m-%d %H:%M'),
            'end': end.strftime('%Y-%m-%d %H:%M'),
            'session_filter': 'all'
        }
        
        data = self._request("/v1/markets/timesales", params)
        
        if not data or 'series' not in data:
            return pd.DataFrame()
        
        series = data['series']
        if series is None or 'data' not in series:
            return pd.DataFrame()
        
        records = series['data']
        if not records:
            return pd.DataFrame()
        
        df = pd.DataFrame(records)
        df['timestamp'] = pd.to_datetime(df['time'])
        df = df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 
                                'c': 'close', 'v': 'volume'})
        
        # Use standard column names if available
        for old, new in [('open', 'open'), ('high', 'high'), ('low', 'low'), 
                         ('close', 'close'), ('volume', 'volume')]:
            if old in df.columns:
                df[new] = pd.to_numeric(df[old], errors='coerce')
        
        df = df.set_index('timestamp')
        return df[['open', 'high', 'low', 'close', 'volume']].dropna()
    
    def _get_history(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        """Get daily historical data."""
        params = {
            'symbol': symbol,
            'interval': 'daily',
            'start': start.strftime('%Y-%m-%d'),
            'end': end.strftime('%Y-%m-%d')
        }
        
        data = self._request("/v1/markets/history", params)
        
        if not data or 'history' not in data:
            return pd.DataFrame()
        
        history = data['history']
        if history is None or 'day' not in history:
            return pd.DataFrame()
        
        days = history['day']
        if not days:
            return pd.DataFrame()
        
        if not isinstance(days, list):
            days = [days]
        
        df = pd.DataFrame(days)
        df['timestamp'] = pd.to_datetime(df['date'])
        df = df.set_index('timestamp')
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df[['open', 'high', 'low', 'close', 'volume']].dropna()
    
    def get_option_chain(
        self,
        symbol: str,
        expiration: Optional[str] = None,
        include_greeks: bool = True
    ) -> pd.DataFrame:
        """
        Get options chain for a symbol.
        
        Args:
            symbol: Underlying symbol (e.g., 'SPY')
            expiration: Expiration date 'YYYY-MM-DD' (default: nearest)
            include_greeks: Include greeks in response
            
        Returns:
            DataFrame with columns:
            - symbol, type (call/put), strike, expiration
            - bid, ask, last, volume, open_interest
            - delta, gamma, theta, vega, rho (if include_greeks)
            - bid_iv, mid_iv, ask_iv
        """
        api_symbol = self._normalize_symbol(symbol)
        
        # Get expiration if not specified
        if not expiration:
            expirations = self.get_option_expirations(symbol)
            if not expirations:
                return pd.DataFrame()
            expiration = expirations[0]  # Nearest expiration
        
        params = {
            'symbol': api_symbol,
            'expiration': expiration,
            'greeks': str(include_greeks).lower()
        }
        
        data = self._request("/v1/markets/options/chains", params)
        
        if not data or 'options' not in data:
            return pd.DataFrame()
        
        options = data['options']
        if options is None or 'option' not in options:
            return pd.DataFrame()
        
        option_list = options['option']
        if not option_list:
            return pd.DataFrame()
        
        if not isinstance(option_list, list):
            option_list = [option_list]
        
        records = []
        for opt in option_list:
            record = {
                'symbol': opt.get('symbol'),
                'underlying': api_symbol,
                'type': opt.get('option_type'),
                'strike': float(opt.get('strike', 0)),
                'expiration': opt.get('expiration_date'),
                'bid': float(opt.get('bid', 0) or 0),
                'ask': float(opt.get('ask', 0) or 0),
                'last': float(opt.get('last', 0) or 0),
                'volume': int(opt.get('volume', 0) or 0),
                'open_interest': int(opt.get('open_interest', 0) or 0),
            }
            
            # Greeks
            greeks = opt.get('greeks', {}) or {}
            record.update({
                'delta': float(greeks.get('delta', 0) or 0),
                'gamma': float(greeks.get('gamma', 0) or 0),
                'theta': float(greeks.get('theta', 0) or 0),
                'vega': float(greeks.get('vega', 0) or 0),
                'rho': float(greeks.get('rho', 0) or 0),
                'bid_iv': float(greeks.get('bid_iv', 0) or 0),
                'mid_iv': float(greeks.get('mid_iv', 0) or 0),
                'ask_iv': float(greeks.get('ask_iv', 0) or 0),
            })
            
            records.append(record)
        
        return pd.DataFrame(records)
    
    def get_option_expirations(self, symbol: str) -> List[str]:
        """
        Get available option expiration dates.
        
        Args:
            symbol: Underlying symbol
            
        Returns:
            List of expiration dates in 'YYYY-MM-DD' format
        """
        api_symbol = self._normalize_symbol(symbol)
        data = self._request("/v1/markets/options/expirations", {'symbol': api_symbol})
        
        if not data or 'expirations' not in data:
            return []
        
        expirations = data['expirations']
        if expirations is None or 'date' not in expirations:
            return []
        
        dates = expirations['date']
        if not isinstance(dates, list):
            dates = [dates] if dates else []
        
        return sorted(dates)


# =============================================================================
# POLYGON DATA CLIENT
# =============================================================================

class PolygonDataClient:
    """
    Clean Polygon.io API client for market data.
    
    Features:
    - End-of-day bars for equities and crypto
    - Intraday bars (limited on free tier)
    - Built-in rate limiting (5 req/min free tier)
    - Disk caching to minimize API calls
    """
    
    BASE_URL = "https://api.polygon.io"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        cache: Optional[DiskCache] = None,
        requests_per_minute: int = 5
    ):
        """
        Initialize Polygon client.
        
        Args:
            api_key: Polygon API key (or set POLYGON_API_KEY env var)
            cache: DiskCache instance for caching
            requests_per_minute: Rate limit (5 for free tier)
        """
        self.api_key = api_key or os.getenv('POLYGON_API_KEY')
        self.cache = cache or _cache
        self.requests_per_minute = requests_per_minute
        
        # Rate limiting with sliding window
        self._request_times: List[float] = []
        self._rate_limit_window = 60.0  # seconds
        
        if not self.api_key:
            logger.warning("PolygonDataClient: No API key provided")
    
    def _rate_limit(self):
        """Enforce rate limiting with sliding window."""
        now = time.time()
        
        # Remove old timestamps
        self._request_times = [
            t for t in self._request_times 
            if now - t < self._rate_limit_window
        ]
        
        # Wait if at limit
        if len(self._request_times) >= self.requests_per_minute:
            oldest = self._request_times[0]
            wait_time = self._rate_limit_window - (now - oldest) + 0.1
            if wait_time > 0:
                logger.info(f"Polygon rate limit: waiting {wait_time:.1f}s")
                time.sleep(wait_time)
                now = time.time()
                self._request_times = [
                    t for t in self._request_times 
                    if now - t < self._rate_limit_window
                ]
        
        self._request_times.append(time.time())
    
    def _request(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """Make API request with rate limiting and error handling."""
        if not self.api_key:
            logger.error("Polygon: No API key configured")
            return {}
        
        self._rate_limit()
        
        params = params or {}
        params['apiKey'] = self.api_key
        url = f"{self.BASE_URL}{endpoint}"
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            logger.warning(f"Polygon HTTP error: {e}")
            return {}
        except requests.exceptions.RequestException as e:
            logger.warning(f"Polygon request failed: {e}")
            return {}
        except json.JSONDecodeError as e:
            logger.warning(f"Polygon JSON decode error: {e}")
            return {}
    
    def _normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol for Polygon API."""
        symbol = symbol.upper()
        # VIX -> I:VIX
        if symbol in ['VIX', '^VIX']:
            return 'I:VIX'
        # Crypto: BTC-USD -> X:BTCUSD
        if '-USD' in symbol:
            base = symbol.replace('-USD', '')
            return f'X:{base}USD'
        return symbol
    
    def get_eod_bars(
        self,
        symbol: str,
        start: Union[date, datetime],
        end: Union[date, datetime],
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Get end-of-day bars for a symbol.
        
        Args:
            symbol: Stock/ETF/crypto symbol
            start: Start date
            end: End date
            use_cache: Whether to use disk cache
            
        Returns:
            DataFrame with columns: open, high, low, close, volume
            Index: date
        """
        # Check cache
        if use_cache:
            cached = self.cache.get('polygon', symbol, '1d', start, end, max_age_hours=12)
            if cached is not None:
                return cached
        
        api_symbol = self._normalize_symbol(symbol)
        start_str = start.strftime('%Y-%m-%d') if hasattr(start, 'strftime') else str(start)
        end_str = end.strftime('%Y-%m-%d') if hasattr(end, 'strftime') else str(end)
        
        endpoint = f"/v2/aggs/ticker/{api_symbol}/range/1/day/{start_str}/{end_str}"
        params = {'adjusted': 'true', 'sort': 'asc', 'limit': 50000}
        
        data = self._request(endpoint, params)
        
        if not data or 'results' not in data:
            return pd.DataFrame()
        
        results = data['results']
        if not results:
            return pd.DataFrame()
        
        df = pd.DataFrame(results)
        df = df.rename(columns={
            't': 'timestamp', 'o': 'open', 'h': 'high',
            'l': 'low', 'c': 'close', 'v': 'volume'
        })
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('timestamp')
        
        result = df[['open', 'high', 'low', 'close', 'volume']].dropna()
        
        # Cache result
        if use_cache and not result.empty:
            self.cache.put('polygon', symbol, '1d', start, end, result)
        
        return result
    
    def get_intraday_bars(
        self,
        symbol: str,
        timespan: str,
        start: datetime,
        end: datetime,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Get intraday bars (limited availability on free tier).
        
        Note: Free tier has delayed/limited intraday data.
        
        Args:
            symbol: Stock/ETF/crypto symbol
            timespan: Bar size ('minute', 'hour')
            start: Start datetime
            end: End datetime
            use_cache: Whether to use disk cache
            
        Returns:
            DataFrame with columns: open, high, low, close, volume
            Index: datetime
        """
        cache_key = f"{timespan}"
        
        # Check cache
        if use_cache:
            cached = self.cache.get('polygon', symbol, cache_key, start, end, max_age_hours=1)
            if cached is not None:
                return cached
        
        api_symbol = self._normalize_symbol(symbol)
        start_str = start.strftime('%Y-%m-%d')
        end_str = end.strftime('%Y-%m-%d')
        
        # Map timespan
        multiplier = 1
        if timespan in ['5m', '5min']:
            multiplier = 5
            timespan = 'minute'
        elif timespan in ['15m', '15min']:
            multiplier = 15
            timespan = 'minute'
        elif timespan in ['1h', '1hour', 'hour']:
            multiplier = 1
            timespan = 'hour'
        elif timespan in ['1m', '1min', 'minute']:
            multiplier = 1
            timespan = 'minute'
        
        endpoint = f"/v2/aggs/ticker/{api_symbol}/range/{multiplier}/{timespan}/{start_str}/{end_str}"
        params = {'adjusted': 'true', 'sort': 'asc', 'limit': 50000}
        
        data = self._request(endpoint, params)
        
        if not data or 'results' not in data:
            return pd.DataFrame()
        
        results = data['results']
        if not results:
            return pd.DataFrame()
        
        df = pd.DataFrame(results)
        df = df.rename(columns={
            't': 'timestamp', 'o': 'open', 'h': 'high',
            'l': 'low', 'c': 'close', 'v': 'volume'
        })
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('timestamp')
        
        result = df[['open', 'high', 'low', 'close', 'volume']].dropna()
        
        # Cache result
        if use_cache and not result.empty:
            self.cache.put('polygon', symbol, cache_key, start, end, result)
        
        return result
    
    def get_crypto_bars(
        self,
        symbol: str,
        start: Union[date, datetime],
        end: Union[date, datetime],
        interval: str = '1d',
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Get crypto candles.
        
        Args:
            symbol: Crypto symbol (e.g., 'BTC-USD', 'BTCUSD', 'ETH-USD')
            start: Start date/datetime
            end: End date/datetime
            interval: Candle interval ('1d', '1h')
            use_cache: Whether to use disk cache
            
        Returns:
            DataFrame with OHLCV data
        """
        # Normalize to Polygon crypto format
        symbol_clean = symbol.upper().replace('-', '')
        if not symbol_clean.startswith('X:'):
            if 'USD' in symbol_clean:
                api_symbol = f"X:{symbol_clean}"
            else:
                api_symbol = f"X:{symbol_clean}USD"
        else:
            api_symbol = symbol_clean
        
        # Check cache
        if use_cache:
            cached = self.cache.get('polygon', symbol, interval, start, end, max_age_hours=4)
            if cached is not None:
                return cached
        
        start_str = start.strftime('%Y-%m-%d') if hasattr(start, 'strftime') else str(start)
        end_str = end.strftime('%Y-%m-%d') if hasattr(end, 'strftime') else str(end)
        
        # Map interval
        if interval in ['1h', '1hour', 'hour']:
            endpoint = f"/v2/aggs/ticker/{api_symbol}/range/1/hour/{start_str}/{end_str}"
        else:
            endpoint = f"/v2/aggs/ticker/{api_symbol}/range/1/day/{start_str}/{end_str}"
        
        params = {'adjusted': 'true', 'sort': 'asc', 'limit': 50000}
        data = self._request(endpoint, params)
        
        if not data or 'results' not in data:
            return pd.DataFrame()
        
        results = data['results']
        if not results:
            return pd.DataFrame()
        
        df = pd.DataFrame(results)
        df = df.rename(columns={
            't': 'timestamp', 'o': 'open', 'h': 'high',
            'l': 'low', 'c': 'close', 'v': 'volume'
        })
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('timestamp')
        
        result = df[['open', 'high', 'low', 'close', 'volume']].dropna()
        
        # Cache result
        if use_cache and not result.empty:
            self.cache.put('polygon', symbol, interval, start, end, result)
        
        return result


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_data_clients(config: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Create data clients from configuration.
    
    Args:
        config: Configuration dict with credentials
        
    Returns:
        Dict with 'tradier' and 'polygon' client instances
    """
    config = config or {}
    creds = config.get('credentials', {})
    
    # Get API keys
    tradier_key = (
        creds.get('tradier', {}).get('data_api_token') or
        creds.get('tradier', {}).get('live', {}).get('access_token') or
        os.getenv('TRADIER_API_KEY')
    )
    
    polygon_key = (
        creds.get('polygon', {}).get('api_key') or
        os.getenv('POLYGON_API_KEY')
    )
    
    # Create clients
    cache = DiskCache()
    
    return {
        'tradier': TradierDataClient(api_key=tradier_key, cache=cache),
        'polygon': PolygonDataClient(api_key=polygon_key, cache=cache),
        'cache': cache
    }


if __name__ == "__main__":
    """Test the data clients."""
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    # Load config
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        config = {}
    
    clients = create_data_clients(config)
    tradier = clients['tradier']
    polygon = clients['polygon']
    
    print("=" * 70)
    print("DATA PROVIDER CLIENT TEST")
    print("=" * 70)
    
    # Test Tradier
    print("\n[TRADIER]")
    if tradier.api_key:
        quote = tradier.get_equity_quote('SPY')
        print(f"  SPY Quote: ${quote.get('price', 'N/A')}")
        
        ts = tradier.get_time_series('SPY', interval='1d', 
                                      start=datetime.now() - timedelta(days=5))
        print(f"  SPY History: {len(ts)} bars")
        
        chain = tradier.get_option_chain('SPY')
        print(f"  SPY Options: {len(chain)} contracts")
    else:
        print("  No API key configured")
    
    # Test Polygon
    print("\n[POLYGON]")
    if polygon.api_key:
        bars = polygon.get_eod_bars('SPY', 
                                    start=datetime.now() - timedelta(days=30),
                                    end=datetime.now())
        print(f"  SPY EOD: {len(bars)} bars")
        
        crypto = polygon.get_crypto_bars('BTC-USD',
                                         start=datetime.now() - timedelta(days=7),
                                         end=datetime.now())
        print(f"  BTC-USD: {len(crypto)} bars")
    else:
        print("  No API key configured")
    
    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)




