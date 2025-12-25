#!/usr/bin/env python3
"""
Polygon.io Data Source
Fetches real-time and historical data for BITX and VIX
"""
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
import time
import logging

class PolygonDataSource:
    """Data source for Polygon.io API"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.polygon.io"
        self.logger = logging.getLogger(__name__)
        
        # Rate limiting (Free tier: 5 requests/minute)
        self.request_times = []  # Track request timestamps
        self.max_requests_per_minute = 5
        self.rate_limit_window = 60  # seconds
        
    def _rate_limit(self):
        """
        Enforce rate limiting: maximum 5 requests per 60 seconds
        Uses sliding window approach to track requests
        """
        current_time = time.time()
        
        # Remove requests older than 60 seconds
        self.request_times = [t for t in self.request_times if current_time - t < self.rate_limit_window]
        
        # If we've made 5 requests in the last 60 seconds, wait
        if len(self.request_times) >= self.max_requests_per_minute:
            # Calculate how long to wait until the oldest request is 60 seconds old
            oldest_request = self.request_times[0]
            wait_time = self.rate_limit_window - (current_time - oldest_request)
            
            if wait_time > 0:
                self.logger.info(f"⏳ Rate limit: {len(self.request_times)}/5 requests used. Waiting {wait_time:.1f}s...")
                time.sleep(wait_time + 0.1)  # Add 0.1s buffer
                current_time = time.time()
                # Clean up again after waiting
                self.request_times = [t for t in self.request_times if current_time - t < self.rate_limit_window]
        
        # Record this request
        self.request_times.append(current_time)
    
    def _make_request(self, endpoint: str, params: dict = None) -> dict:
        """Make API request with rate limiting"""
        self._rate_limit()
        
        if params is None:
            params = {}
        params['apiKey'] = self.api_key
        
        url = f"{self.base_url}{endpoint}"
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Polygon API request failed: {e}")
            return {}
    
    def get_quote(self, symbol: str) -> dict:
        """Get current quote for a symbol"""
        # Map VIX to correct symbol
        if symbol.upper() in ['VIX', '^VIX']:
            symbol = 'I:VIX'
        
        endpoint = f"/v2/last/trade/{symbol}"
        data = self._make_request(endpoint)
        
        if not data or 'results' not in data:
            return {}
        
        result = data['results']
        return {
            'symbol': symbol,
            'price': result.get('p', 0),
            'size': result.get('s', 0),
            'timestamp': result.get('t', 0),
            'exchange': result.get('x', ''),
        }
    
    def get_historical_data(
        self,
        symbol: str,
        period: str = '1d',
        interval: str = '1m',
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Get historical aggregate bars for a symbol
        
        Args:
            symbol: Stock/index symbol
            period: Time period ('1d', '7d', '30d', etc.) - used if start_date not provided
            interval: Bar interval ('1m', '5m', '15m', '1h', '1d')
            start_date: Start date (optional, overrides period)
            end_date: End date (optional, defaults to now)
        
        Returns:
            DataFrame with OHLCV data
        """
        # Map VIX to correct symbol
        if symbol.upper() in ['VIX', '^VIX']:
            symbol = 'I:VIX'
        
        # Parse interval
        interval_map = {
            '1m': ('minute', 1),
            '5m': ('minute', 5),
            '15m': ('minute', 15),
            '1h': ('hour', 1),
            '1d': ('day', 1)
        }
        
        if interval not in interval_map:
            self.logger.error(f"Invalid interval: {interval}")
            return pd.DataFrame()
        
        timespan, multiplier = interval_map[interval]
        
        # Calculate date range
        if end_date is None:
            end_date = datetime.now()
        
        if start_date is None:
            # Parse period
            period_map = {
                '1d': 1,
                '7d': 7,
                '30d': 30,
                '60d': 60,
                '90d': 90
            }
            days = period_map.get(period, 1)
            start_date = end_date - timedelta(days=days)
        
        # Format dates for API
        from_date = start_date.strftime('%Y-%m-%d')
        to_date = end_date.strftime('%Y-%m-%d')
        
        # Build endpoint
        endpoint = f"/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
        
        params = {
            'adjusted': 'true',
            'sort': 'asc',
            'limit': 50000  # Max results
        }
        
        self.logger.info(f"Fetching Polygon data: {symbol} from {from_date} to {to_date} ({interval} bars)")
        
        data = self._make_request(endpoint, params)
        
        if not data or 'results' not in data or not data['results']:
            self.logger.warning(f"No data returned from Polygon for {symbol}")
            return pd.DataFrame()
        
        # Parse results
        results = data['results']
        
        df = pd.DataFrame(results)
        
        # Rename columns to match expected format
        df = df.rename(columns={
            't': 'timestamp',
            'o': 'open',
            'h': 'high',
            'l': 'low',
            'c': 'close',
            'v': 'volume',
            'vw': 'vwap',
            'n': 'transactions'
        })
        
        # Convert timestamp from milliseconds to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('timestamp')
        
        # Ensure we have the required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                df[col] = 0
        
        self.logger.info(f"✅ Got {len(df)} {interval} bars from Polygon for {symbol}")
        
        return df[['open', 'high', 'low', 'close', 'volume']]
    
    def test_connection(self) -> bool:
        """Test API connection and key validity"""
        try:
            # Test with a simple request
            endpoint = "/v2/last/trade/AAPL"
            data = self._make_request(endpoint)
            
            if 'status' in data and data['status'] == 'ERROR':
                self.logger.error(f"Polygon API error: {data.get('error', 'Unknown error')}")
                return False
            
            if 'results' in data:
                self.logger.info("✅ Polygon.io connection successful!")
                return True
            
            return False
        except Exception as e:
            self.logger.error(f"Polygon connection test failed: {e}")
            return False


if __name__ == "__main__":
    """Test the Polygon data source"""
    import sys
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Get API key from user
    if len(sys.argv) > 1:
        api_key = sys.argv[1]
    else:
        api_key = input("Enter your Polygon.io API key: ").strip()
    
    if not api_key:
        print("❌ No API key provided!")
        sys.exit(1)
    
    print("\n" + "=" * 70)
    print("🧪 TESTING POLYGON.IO DATA SOURCE")
    print("=" * 70)
    
    # Create data source
    polygon = PolygonDataSource(api_key)
    
    # Test connection
    print("\n1. Testing API connection...")
    if not polygon.test_connection():
        print("❌ Connection failed!")
        sys.exit(1)
    
    # Test BITX data
    print("\n2. Testing BITX historical data (1 day, 1min bars)...")
    bitx_data = polygon.get_historical_data('BITX', period='1d', interval='1m')
    
    if not bitx_data.empty:
        print(f"✅ Got {len(bitx_data)} BITX bars")
        print(f"   Date range: {bitx_data.index[0]} to {bitx_data.index[-1]}")
        print(f"   Latest close: ${bitx_data['close'].iloc[-1]:.2f}")
    else:
        print("❌ No BITX data received")
    
    # Test VIX data
    print("\n3. Testing VIX historical data (1 day, 1min bars)...")
    vix_data = polygon.get_historical_data('VIX', period='1d', interval='1m')
    
    if not vix_data.empty:
        print(f"✅ Got {len(vix_data)} VIX bars")
        print(f"   Date range: {vix_data.index[0]} to {vix_data.index[-1]}")
        print(f"   Latest value: {vix_data['close'].iloc[-1]:.2f}")
    else:
        print("❌ No VIX data received")
    
    # Test quote
    print("\n4. Testing real-time quote...")
    quote = polygon.get_quote('BITX')
    
    if quote:
        print(f"✅ Current BITX price: ${quote.get('price', 0):.2f}")
    else:
        print("❌ No quote data received")
    
    print("\n" + "=" * 70)
    print("✅ Testing complete!")
    print("=" * 70)
    
    # Save API key reminder
    print("\n💡 To use Polygon in your bot:")
    print("   1. Add to tradier_config.json:")
    print('      "polygon_api_key": "YOUR_API_KEY"')
    print("   2. Update enhanced_data_sources.py to use Polygon")
    print("   3. Restart your bot")

