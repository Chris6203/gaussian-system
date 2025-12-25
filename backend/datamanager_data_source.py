#!/usr/bin/env python3
"""
Data Manager Data Source

Fetches market data from a remote Data Manager server.
This is the primary data source when connected to a centralized data collection service.
"""

import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

# Import the client from data-manager
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'data-manager'))

try:
    from client import DataManagerClient, APIError
except ImportError:
    # Inline minimal client if data-manager not available
    DataManagerClient = None
    APIError = Exception


class DataManagerDataSource:
    """
    Data source that fetches from a remote Data Manager server.

    Provides the same interface as other data sources (Tradier, FMP, Yahoo)
    but fetches from a centralized data collection service.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout: int = 30,
        cache_ttl_seconds: int = 60
    ):
        """
        Initialize Data Manager data source.

        Args:
            base_url: Data Manager server URL (e.g., "http://31.97.215.206:5050")
            api_key: API key for authentication (dm_xxxxx format)
            timeout: Request timeout in seconds
            cache_ttl_seconds: Cache TTL for real-time data
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self.cache_ttl = cache_ttl_seconds

        # Simple in-memory cache
        self._cache: Dict[str, tuple] = {}  # key -> (data, timestamp)

        # Initialize client
        if DataManagerClient is None:
            self._init_minimal_client()
        else:
            self.client = DataManagerClient(base_url, api_key)

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"[DATA] DataManager source initialized: {base_url}")

    def _init_minimal_client(self):
        """Initialize minimal HTTP client if SDK not available."""
        import requests
        self._session = requests.Session()
        self._session.headers['X-API-Key'] = self.api_key
        self._session.headers['Content-Type'] = 'application/json'
        self.client = None

    def _make_request(self, method: str, endpoint: str, params: dict = None) -> dict:
        """Make HTTP request to Data Manager."""
        import requests

        url = f"{self.base_url}{endpoint}"
        try:
            if self.client:
                return self.client._request(method, endpoint, params=params)
            else:
                response = self._session.request(
                    method, url, params=params, timeout=self.timeout
                )
                response.raise_for_status()
                return response.json()
        except requests.exceptions.Timeout:
            self.logger.warning(f"[DATA] DataManager request timeout: {endpoint}")
            raise
        except requests.exceptions.RequestException as e:
            self.logger.warning(f"[DATA] DataManager request failed: {e}")
            raise

    def _get_cache_key(self, symbol: str, period: str, interval: str) -> str:
        """Generate cache key."""
        return f"{symbol}:{period}:{interval}"

    def _is_cache_valid(self, key: str) -> bool:
        """Check if cached data is still valid."""
        if key not in self._cache:
            return False
        _, timestamp = self._cache[key]
        return (datetime.now() - timestamp).total_seconds() < self.cache_ttl

    def _period_to_start_date(self, period: str) -> str:
        """Convert period string to ISO date."""
        now = datetime.now()

        period_map = {
            '1d': timedelta(days=1),
            '5d': timedelta(days=5),
            '7d': timedelta(days=7),
            '1mo': timedelta(days=30),
            '30d': timedelta(days=30),
            '3mo': timedelta(days=90),
            '90d': timedelta(days=90),
            '6mo': timedelta(days=180),
            '180d': timedelta(days=180),
            '1y': timedelta(days=365),
            '365d': timedelta(days=365),
        }

        delta = period_map.get(period, timedelta(days=7))
        start = now - delta
        return start.strftime('%Y-%m-%dT%H:%M:%S')

    def get_data(
        self,
        symbol: str,
        period: str = "1d",
        interval: str = "1m"
    ) -> pd.DataFrame:
        """
        Get historical data for a symbol.

        Args:
            symbol: Stock symbol (e.g., 'SPY', '^VIX', 'VIX')
            period: Data period ('1d', '5d', '7d', '30d', '90d', '180d')
            interval: Data interval (currently only '1m' is stored)

        Returns:
            DataFrame with OHLCV columns: Open, High, Low, Close, Volume
        """
        # Normalize symbol (remove ^ prefix, handle VIX variants)
        api_symbol = symbol.lstrip('^').upper()
        if api_symbol in ('VIX', 'VIXY'):
            api_symbol = 'VIX'

        # Check cache for real-time requests
        cache_key = self._get_cache_key(api_symbol, period, interval)
        if period == '1d' and self._is_cache_valid(cache_key):
            data, _ = self._cache[cache_key]
            return data.copy()

        try:
            start_date = self._period_to_start_date(period)

            # Fetch from Data Manager
            if self.client:
                records = self.client.get_prices(api_symbol, start=start_date, limit=50000)
            else:
                result = self._make_request(
                    'GET',
                    f'/api/v1/prices/{api_symbol}',
                    params={'start': start_date, 'limit': 50000}
                )
                records = result.get('data', [])

            if not records:
                self.logger.debug(f"[DATA] No data from DataManager for {symbol}")
                return pd.DataFrame()

            # Convert to DataFrame
            df = pd.DataFrame(records)

            # Ensure proper column names (capitalize for compatibility)
            column_map = {
                'timestamp': 'Datetime',
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            }
            df = df.rename(columns=column_map)

            # Parse datetime and set as index
            if 'Datetime' in df.columns:
                df['Datetime'] = pd.to_datetime(df['Datetime'])
                df = df.set_index('Datetime')

            # Sort by time
            df = df.sort_index()

            # Handle interval resampling if needed
            if interval != '1m' and interval in ('5m', '15m', '1h'):
                df = self._resample(df, interval)

            # Cache the result
            self._cache[cache_key] = (df.copy(), datetime.now())

            self.logger.debug(f"[DATA] Got {len(df)} records from DataManager for {symbol}")
            return df

        except Exception as e:
            self.logger.warning(f"[DATA] DataManager fetch failed for {symbol}: {e}")
            return pd.DataFrame()

    def _resample(self, df: pd.DataFrame, interval: str) -> pd.DataFrame:
        """Resample 1-minute data to larger intervals."""
        interval_map = {
            '5m': '5T',
            '15m': '15T',
            '1h': '1H',
            '1d': '1D'
        }

        rule = interval_map.get(interval, '5T')

        resampled = df.resample(rule).agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()

        return resampled

    def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Get current price for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Current price or None if unavailable
        """
        api_symbol = symbol.lstrip('^').upper()

        try:
            if self.client:
                prices = self.client.get_latest_prices([api_symbol])
            else:
                result = self._make_request(
                    'GET',
                    '/api/v1/prices',
                    params={'symbols': api_symbol}
                )
                prices = result.get('data', {})

            if api_symbol in prices:
                return float(prices[api_symbol].get('close', 0))

            # Fallback to getting last record from historical
            data = self.get_data(symbol, period='1d', interval='1m')
            if not data.empty:
                return float(data['Close'].iloc[-1])

        except Exception as e:
            self.logger.debug(f"[DATA] Could not get price for {symbol}: {e}")

        return None

    def get_options_chain(self, symbol: str) -> Dict[str, Any]:
        """
        Get options chain data for a symbol.

        Args:
            symbol: Underlying symbol (e.g., 'SPY')

        Returns:
            Dict with options chain data
        """
        api_symbol = symbol.lstrip('^').upper()

        try:
            if self.client:
                options = self.client.get_options(api_symbol, limit=5000)
            else:
                result = self._make_request(
                    'GET',
                    f'/api/v1/options/{api_symbol}',
                    params={'limit': 5000}
                )
                options = result.get('data', [])

            if not options:
                return {}

            # Group by expiration and type
            chain = {'calls': [], 'puts': [], 'expirations': set()}

            for opt in options:
                opt_type = opt.get('option_type', '').upper()
                exp = opt.get('expiration_date', '')
                chain['expirations'].add(exp)

                entry = {
                    'symbol': opt.get('option_symbol', ''),
                    'strike': opt.get('strike_price', 0),
                    'expiration': exp,
                    'bid': opt.get('bid', 0),
                    'ask': opt.get('ask', 0),
                    'mid': opt.get('mid_price', 0),
                    'volume': opt.get('volume', 0),
                    'open_interest': opt.get('open_interest', 0),
                    'iv': opt.get('implied_volatility', 0),
                    'delta': opt.get('delta', 0),
                    'gamma': opt.get('gamma', 0),
                    'theta': opt.get('theta', 0),
                    'vega': opt.get('vega', 0),
                }

                if opt_type == 'CALL':
                    chain['calls'].append(entry)
                elif opt_type == 'PUT':
                    chain['puts'].append(entry)

            chain['expirations'] = sorted(list(chain['expirations']))
            return chain

        except Exception as e:
            self.logger.warning(f"[DATA] Could not get options chain for {symbol}: {e}")
            return {}

    def get_historical_data(
        self,
        symbol: str,
        period: str = "7d",
        interval: str = "1m"
    ) -> pd.DataFrame:
        """Alias for get_data for compatibility."""
        return self.get_data(symbol, period, interval)

    def health_check(self) -> bool:
        """Check if the Data Manager server is healthy."""
        try:
            if self.client:
                return self.client.health_check()
            else:
                result = self._make_request('GET', '/api/health')
                return result.get('ok', False)
        except Exception:
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics from Data Manager."""
        try:
            if self.client:
                return self.client.get_stats()
            else:
                return self._make_request('GET', '/api/v1/stats')
        except Exception as e:
            self.logger.warning(f"[DATA] Could not get stats: {e}")
            return {}


__all__ = ['DataManagerDataSource']
