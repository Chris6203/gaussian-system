#!/usr/bin/env python3
"""
Tradier Data Source - Real Broker Integration for Options Trading
Ready to activate when you have your Tradier account credentials
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Optional, Dict, List
import json
import time


class TradierDataSource:
    """
    Tradier broker integration for real options trading
    Supports both sandbox and live environments
    """

    def __init__(self, api_key: str = "", account_id: str = "", sandbox: bool = True):
        self.logger = logging.getLogger(__name__)
        self.api_key = api_key
        self.account_id = account_id
        self.sandbox = sandbox

        # API endpoints
        if sandbox:
            self.base_url = "https://sandbox.tradier.com"
        else:
            self.base_url = "https://api.tradier.com"

        self.headers = {
            'Authorization': f'Bearer {api_key}',
            'Accept': 'application/json'
        }

        # Rate limiting
        self.rate_limit_delay = 1  # 1 second between requests
        self.last_request_time = 0

        # Cache
        self.cache = {}
        self.cache_timeout = 30  # 30 seconds for real-time data

    def _rate_limit(self):
        """Implement rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last
            time.sleep(sleep_time)

        self.last_request_time = time.time()

    def _make_request(self, endpoint: str, params: Dict = None) -> Dict:
        """Make API request with error handling"""
        if not self.api_key:
            self.logger.error("No Tradier API key provided")
            return {}

        self._rate_limit()

        url = f"{self.base_url}{endpoint}"

        try:
            response = requests.get(
                url, headers=self.headers, params=params, timeout=30)
            response.raise_for_status()

            return response.json()

        except Exception as e:
            self.logger.error(f"Tradier API request failed: {e}")
            return {}

    def test_connection(self) -> bool:
        """Test if Tradier API is working"""
        if not self.api_key:
            self.logger.warning("No Tradier API key - connection test skipped")
            return False

        self.logger.info("Testing Tradier connection...")

        # Test with user profile endpoint
        data = self._make_request("/v1/user/profile")

        if data and 'profile' in data:
            self.logger.info("Tradier connection successful")
            return True
        else:
            self.logger.error("❌ Tradier connection failed")
            return False

    def get_quote(self, symbol: str) -> Dict:
        """Get current quote for a symbol"""
        # Map ^VIX to VIX for Tradier (VIX works, $VIX doesn't for historical)
        if symbol.upper() == '^VIX':
            symbol = 'VIX'
        
        data = self._make_request(
            "/v1/markets/quotes", params={'symbols': symbol})

        if data and 'quotes' in data and 'quote' in data['quotes']:
            quote = data['quotes']['quote']

            return {
                'symbol': quote.get('symbol', symbol),
                'price': float(quote.get('last', 0)),
                'bid': float(quote.get('bid', 0)),
                'ask': float(quote.get('ask', 0)),
                'volume': int(quote.get('volume', 0)),
                'change': float(quote.get('change', 0)),
                'change_percent': float(quote.get('change_percentage', 0)),
                'high': float(quote.get('high', 0)),
                'low': float(quote.get('low', 0)),
                'open': float(quote.get('open', 0)),
                'previous_close': float(quote.get('prevclose', 0))
            }

        return {}

    # Symbols that Tradier doesn't support (crypto, forex pairs, etc.)
    UNSUPPORTED_SYMBOLS = ['BTC-USD', 'ETH-USD', 'BTC', 'ETH', 'DOGE-USD']
    
    def _is_unsupported_symbol(self, symbol: str) -> bool:
        """Check if symbol is unsupported by Tradier (crypto, forex, etc.)"""
        symbol_upper = symbol.upper()
        # Check exact matches
        if symbol_upper in self.UNSUPPORTED_SYMBOLS:
            return True
        # Check patterns like XXX-USD (crypto pairs)
        if '-USD' in symbol_upper or '-EUR' in symbol_upper:
            return True
        return False
    
    def get_historical_data(self, symbol: str, period: str = "1month", interval: str = "daily") -> pd.DataFrame:
        """Get historical price data with configurable period and interval"""
        
        # Skip unsupported symbols (crypto, forex) - let fallback handle them
        if self._is_unsupported_symbol(symbol):
            self.logger.info(f"Tradier: Skipping unsupported symbol {symbol} (crypto/forex) → use Yahoo fallback")
            return pd.DataFrame()
        
        # Map ^VIX to VIX for Tradier (VIX works, $VIX doesn't)
        if symbol.upper() == '^VIX':
            symbol = 'VIX'
            self.logger.info(f"Mapped ^VIX to VIX for Tradier")

        # Map period strings to days
        period_days = {
            '1d': 1,
            '2d': 2,
            '7d': 7,
            '1week': 7,
            '1month': 30,
            '30d': 30,
            '3month': 90,
            '6month': 180,
            '1year': 365,
            '1y': 365
        }

        # Get number of days to fetch
        days = period_days.get(period, 30)

        # Check if we need intraday data
        if interval in ['1m', '5m', '15m', '30m', '1h']:
            return self._get_intraday_data(symbol, period, interval)

        # For daily data, use the historical endpoint
        tradier_interval = 'daily'

        params = {
            'symbol': symbol,
            'interval': tradier_interval,
            'start': (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d'),
            'end': datetime.now().strftime('%Y-%m-%d')
        }

        data = self._make_request("/v1/markets/history", params=params)

        if data and 'history' in data and 'day' in data['history']:
            days = data['history']['day']

            # Convert to DataFrame
            df_data = []
            for day in days:
                df_data.append({
                    'Date': pd.to_datetime(day['date']),
                    'Open': float(day['open']),
                    'High': float(day['high']),
                    'Low': float(day['low']),
                    'Close': float(day['close']),
                    'Volume': int(day['volume'])
                })

            df = pd.DataFrame(df_data)
            df.set_index('Date', inplace=True)
            df.sort_index(inplace=True)

            # Normalize column names to title case for consistency with yfinance/pandas
            df.columns = df.columns.str.title()

            # Tradier only provides daily data - no synthetic intraday generation
            # Limit the data to the requested period
            if len(df) > 0:
                df = self._limit_data_to_period(df, period)

            return df

        return pd.DataFrame()

    def _get_intraday_data(self, symbol: str, period: str, interval: str) -> pd.DataFrame:
        """Get intraday data using Tradier's timesales endpoint"""
        
        # VIX symbol is already mapped in get_historical_data, so it should be $VIX here
        # Just log it for debugging
        if symbol.startswith('$'):
            self.logger.debug(f"Fetching intraday data for {symbol} (Tradier format)")

        # Map period to days
        period_days = {
            '1d': 1,
            '2d': 2,
            '7d': 7,
            '1week': 7,
            '1month': 30,
            '30d': 30,
            '3month': 90,
            '6month': 180,
            '1year': 365,
            '1y': 365
        }

        days = period_days.get(period, 1)

        # Map interval to Tradier format
        interval_map = {
            '1m': '1min',
            '5m': '5min',
            '15m': '15min',
            '30m': '30min',
            '1h': '1hour'
        }

        tradier_interval = interval_map.get(interval, '1min')

        # For delisted symbols like BITX, try to get historical data
        # Calculate start and end times
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)

        # Format times for Tradier API
        start_str = start_time.strftime('%Y-%m-%d %H:%M')
        end_str = end_time.strftime('%Y-%m-%d %H:%M')

        params = {
            'symbol': symbol,
            'interval': tradier_interval,
            'start': start_str,
            'end': end_str,
            'session_filter': 'all'  # Include all trading sessions
        }

        self.logger.info(
            f"Getting {symbol} intraday data from Tradier ({tradier_interval})...")

        data = self._make_request("/v1/markets/timesales", params=params)

        # Debug: Log the actual API response for BITX (disabled for cleaner logs)
        # if symbol == "BITX":
        #     self.logger.debug(f"BITX API Response: {data}")

        if data and 'series' in data and 'data' in data['series']:
            timesales = data['series']['data']

            # Convert to DataFrame
            df_data = []
            for point in timesales:
                # Tradier returns times in PST, convert to EST by adding 3 hours
                raw_time = pd.to_datetime(point['time'])
                est_time = raw_time + pd.Timedelta(hours=3)

                df_data.append({
                    'Date': est_time,
                    'Open': float(point['open']),
                    'High': float(point['high']),
                    'Low': float(point['low']),
                    'Close': float(point['close']),
                    'Volume': int(point['volume'])
                })

            df = pd.DataFrame(df_data)
            df.set_index('Date', inplace=True)
            df.sort_index(inplace=True)

            # Normalize column names to title case for consistency with yfinance/pandas
            df.columns = df.columns.str.title()

            self.logger.debug(
                f"Got {len(df)} intraday data points from Tradier")
            return df

        if data and 'series' in data and data['series'] is None:
            self.logger.warning(
                f"⚠️ Tradier returned null series for {symbol} - this is normal for sandbox environment")
        else:
            self.logger.warning(
                f"⚠️ No intraday data returned from Tradier for {symbol}")
            if data:
                self.logger.warning(f"API Response: {data}")
        return pd.DataFrame()

    def _limit_data_to_period(self, df: pd.DataFrame, period: str) -> pd.DataFrame:
        """Limit the data to the requested period"""
        if df.empty:
            return df

        # Map period to number of days
        period_days = {
            '1d': 1,
            '2d': 2,
            '7d': 7,
            '1week': 7,
            '1month': 30,
            '30d': 30,
            '3month': 90,
            '6month': 180,
            '1year': 365,
            '1y': 365
        }

        days = period_days.get(period, 30)

        # Get the most recent data points
        cutoff_date = datetime.now() - timedelta(days=days)
        limited_df = df[df.index >= cutoff_date]

        # If we don't have enough data, return what we have
        if len(limited_df) == 0:
            limited_df = df.tail(days)  # Return last N days

        self.logger.info(
            f"Limited data from {len(df)} to {len(limited_df)} days for period '{period}'")
        return limited_df

    def _generate_intraday_data(self, daily_df: pd.DataFrame, interval: str) -> pd.DataFrame:
        """Generate intraday data from daily OHLC data with continuous 24-hour coverage"""
        import numpy as np

        # Map interval to minutes
        interval_minutes = {
            '1m': 1,
            '5m': 5,
            '15m': 15,
            '1h': 60
        }

        minutes = interval_minutes.get(interval, 60)

        # Generate intraday data for each day
        intraday_data = []

        for date, row in daily_df.iterrows():
            # Create time range for 24 hours (00:00 to 23:59)
            start_time = date.replace(
                hour=0, minute=0, second=0, microsecond=0)
            end_time = date.replace(
                hour=23, minute=59, second=59, microsecond=0)

            # Generate time series for full 24 hours
            time_range = pd.date_range(
                start=start_time, end=end_time, freq=f'{minutes}min')

            # Generate realistic intraday price movements
            open_price = row['Open']
            close_price = row['Close']
            high_price = row['High']
            low_price = row['Low']
            volume = row['Volume']

            # Create price path using random walk with trend
            n_periods = len(time_range)
            if n_periods > 1:
                # Generate price movements
                price_change = (close_price - open_price) / (n_periods - 1)
                volatility = (high_price - low_price) / (n_periods * 2)

                prices = [open_price]
                for i in range(1, n_periods):
                    # Add trend and random noise
                    trend = price_change
                    noise = np.random.normal(0, volatility)
                    new_price = prices[-1] + trend + noise

                    # Ensure price stays within daily range
                    new_price = max(low_price, min(high_price, new_price))
                    prices.append(new_price)

                # Ensure last price is close to actual close
                prices[-1] = close_price

                # Generate volume distribution with higher volume during market hours
                daily_volume = volume
                volumes = []

                for i, timestamp in enumerate(time_range):
                    # Higher volume during market hours (9:30 AM - 4:00 PM)
                    if 9.5 <= timestamp.hour + timestamp.minute/60 <= 16:
                        # Market hours: higher volume
                        # 40% of periods are market hours
                        base_volume = daily_volume / (n_periods * 0.4)
                        volume_multiplier = max(
                            0.1, 1 + np.random.normal(0, 0.3))  # Ensure positive
                    else:
                        # After hours: lower volume
                        base_volume = daily_volume / \
                            (n_periods * 0.6) * 0.1  # 10% of market volume
                        volume_multiplier = max(
                            0.1, 1 + np.random.normal(0, 0.1))  # Ensure positive

                    # Ensure minimum volume of 1
                    volumes.append(
                        max(1, int(base_volume * volume_multiplier)))

                # Create OHLC for each period
                for i, (timestamp, price) in enumerate(zip(time_range, prices)):
                    if i == 0:
                        ohlc = [price, price, price, price]
                    else:
                        prev_price = prices[i-1]
                        ohlc = [prev_price, max(prev_price, price), min(
                            prev_price, price), price]

                    intraday_data.append({
                        'Date': timestamp,
                        'Open': ohlc[0],
                        'High': ohlc[1],
                        'Low': ohlc[2],
                        'Close': ohlc[3],
                        'Volume': volumes[i]
                    })

        if intraday_data:
            intraday_df = pd.DataFrame(intraday_data)
            intraday_df.set_index('Date', inplace=True)
            intraday_df.sort_index(inplace=True)
            self.logger.info(
                f"Generated {len(intraday_df)} intraday data points from {len(daily_df)} daily points")
            return intraday_df

        self.logger.warning(
            "⚠️ No intraday data generated, returning daily data")
        return daily_df

    def get_options_chain(self, symbol: str, expiration: str = None) -> Dict:
        """
        Get options chain - this is where Tradier really shines!
        """
        params = {'symbol': symbol}
        if expiration:
            params['expiration'] = expiration

        data = self._make_request("/v1/markets/options/chains", params=params)

        if data and 'options' in data:
            return data['options']

        return {}

    def get_options_expirations(self, symbol: str) -> List[str]:
        """Get available options expiration dates"""
        data = self._make_request("/v1/markets/options/expirations",
                                  params={'symbol': symbol})

        if data and 'expirations' in data and 'date' in data['expirations']:
            return data['expirations']['date']

        return []

    def get_account_info(self) -> Dict:
        """Get account information"""
        if not self.account_id:
            return {}

        data = self._make_request(f"/v1/accounts/{self.account_id}")

        if data and 'account' in data:
            account = data['account']
            return {
                'account_number': account.get('account_number'),
                'buying_power': float(account.get('account_equity', 0)),
                'cash': float(account.get('cash', {}).get('cash_available', 0)),
                'market_value': float(account.get('market_value', 0)),
                'option_level': account.get('option_level', 0)
            }

        return {}

    def place_order(self, symbol: str, quantity: int, side: str, order_type: str = "market") -> Dict:
        """
        Place an options order (when you're ready for live trading)
        This is a placeholder - implement when ready for real trading
        """
        self.logger.warning(
            "Order placement not implemented - paper trading only")

        # Placeholder for order placement
        return {
            'status': 'simulated',
            'message': 'Paper trading mode - no real order placed'
        }


def setup_tradier_instructions():
    """Provide setup instructions for Tradier"""
    print("🏦 Tradier Setup Instructions")
    print("=" * 40)

    print("\n📋 Steps to activate Tradier:")
    print("1. Complete your Tradier account setup")
    print("2. Get API access:")
    print("   • Log into Tradier")
    print("   • Go to Settings > API Access")
    print("   • Create new application")
    print("   • Get your API key")

    print("\n3. Enable options trading:")
    print("   • Request Level 3 options approval")
    print("   • Consider margin account for spreads")
    print("   • Enable Tradier Pro for streaming data")

    print("\n4. Update your bot:")
    print("   • Add API key to multi_data_source.py")
    print("   • Test in sandbox first")
    print("   • Switch to live when ready")

    print("\nBenefits of Tradier:")
    print("   • Real options chains and Greeks")
    print("   • Live trade execution")
    print("   • OPRA data feed")
    print("   • No synthetic data")
    print("   • Professional options tools")


def main():
    """Test Tradier integration (placeholder)"""
    print("🏦 Tradier Data Source")
    print("=" * 30)

    # Create placeholder instance
    tradier = TradierDataSource()

    print("⚠️ Tradier credentials needed")
    print("This data source is ready but requires:")
    print("• API key from your Tradier account")
    print("• Account ID")
    print("• Options trading approval")

    setup_tradier_instructions()

    print(f"\n🔄 Status: Ready for activation")
    print(f"📋 When ready, update multi_data_source.py with:")
    print(f"   multi_source.add_tradier_credentials(api_key, account_id)")


if __name__ == "__main__":
    main()






