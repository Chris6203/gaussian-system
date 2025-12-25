#!/usr/bin/env python3
"""
Data Sources for BITX Options Trading Bot
Auto-uploaded to Tesla P40 server
"""
import yfinance as yf
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple


class DataSource:
    """Base class for data sources"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def get_data(self, symbol: str, period: str = "1d") -> pd.DataFrame:
        """Get historical data for a symbol"""
        raise NotImplementedError


class YahooFinanceDataSource(DataSource):
    """Yahoo Finance data source for market data with improved reliability"""

    def __init__(self):
        super().__init__()
        self.cache = {}
        self.cache_timeout = 300  # 5 minutes
        # 10 seconds between requests (increased for rate limiting)
        self.rate_limit_delay = 10
        self.last_request_time = 0
        self.request_count = 0
        self.daily_request_limit = 100  # Conservative daily limit

        # No fallback symbols - use only the requested symbol
        self.symbol_alternatives = {}

    def _rate_limit(self):
        """Implement enhanced rate limiting"""
        import time
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        # Check daily request limit
        if self.request_count >= self.daily_request_limit:
            self.logger.warning(
                f"Daily request limit ({self.daily_request_limit}) reached, using cached/synthetic data only")
            return False

        # Exponential backoff for rate limiting
        base_delay = self.rate_limit_delay
        if self.request_count > 50:  # If we've made many requests
            base_delay *= 2  # Double the delay

        if time_since_last < base_delay:
            sleep_time = base_delay - time_since_last
            self.logger.info(
                f"Rate limiting: sleeping {sleep_time:.1f}s (request #{self.request_count})")
            time.sleep(sleep_time)

        self.last_request_time = time.time()
        self.request_count += 1
        return True

    def get_data(self, symbol: str, period: str = "1d", interval: str = "1m") -> pd.DataFrame:
        """
        Get historical data from Yahoo Finance with fallbacks and synthetic data

        Args:
            symbol: Stock symbol (e.g., 'BITX', '^VIX')
            period: Data period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval: Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')

        Returns:
            DataFrame with OHLCV data
        """
        # Try primary symbol first
        data = self._get_single_symbol_data(symbol, period, interval)

        if not data.empty:
            return data

        # Try alternatives
        alternatives = self.symbol_alternatives.get(symbol, [])

        for alt_symbol in alternatives:
            self.logger.info(f"Trying alternative symbol: {alt_symbol}")
            data = self._get_single_symbol_data(alt_symbol, period, interval)

            if not data.empty:
                self.logger.info(
                    f"Successfully using {alt_symbol} as alternative to {symbol}")
                return data

        # If all fail, return empty DataFrame
        self.logger.warning(f"All data sources failed for {symbol}")
        return pd.DataFrame()

    def _get_single_symbol_data(self, symbol: str, period: str, interval: str) -> pd.DataFrame:
        """
        Get data for a single symbol with caching and rate limiting
        """
        cache_key = f"{symbol}_{period}_{interval}"
        current_time = datetime.now()

        # Check cache first
        if cache_key in self.cache:
            cached_data, cached_time = self.cache[cache_key]
            if (current_time - cached_time).seconds < self.cache_timeout:
                self.logger.debug(f"Returning cached data for {symbol}")
                return cached_data

        # Rate limiting - skip API call if limit reached
        if not self._rate_limit():
            self.logger.info(
                f"Skipping API call for {symbol} due to rate limiting")
            return pd.DataFrame()

        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)

            if not data.empty:
                # Cache successful data
                self.cache[cache_key] = (data, current_time)
                self.logger.info(
                    f"Retrieved {len(data)} data points for {symbol}")
                return data
            else:
                self.logger.warning(f"No data returned for {symbol}")
                return pd.DataFrame()

        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return pd.DataFrame()

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return info.get('regularMarketPrice') or info.get('previousClose')
        except Exception as e:
            self.logger.error(
                f"Error fetching current price for {symbol}: {str(e)}")
            return None

    def get_options_chain(self, symbol: str) -> Dict:
        """Get options chain for a symbol"""
        try:
            ticker = yf.Ticker(symbol)
            options_dates = ticker.options

            if not options_dates:
                self.logger.warning(f"No options available for {symbol}")
                return {}

            options_data = {}
            for date in options_dates[:5]:  # Get first 5 expiration dates
                try:
                    option_chain = ticker.option_chain(date)
                    options_data[date] = {
                        'calls': option_chain.calls,
                        'puts': option_chain.puts
                    }
                except Exception as e:
                    self.logger.warning(
                        f"Error fetching options for {date}: {str(e)}")
                    continue

            return options_data

        except Exception as e:
            self.logger.error(
                f"Error fetching options chain for {symbol}: {str(e)}")
            return {}


class MarketDataManager:
    """Manages all market data sources and provides unified interface"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.data_source = YahooFinanceDataSource()
        self.bitx_data = pd.DataFrame()
        self.vix_data = pd.DataFrame()
        self.last_update = None

    def update_data(self, bitx_symbol: str = 'BITX', vix_symbol: str = '^VIX'):
        """Update BITX and VIX data"""
        try:
            # Get BITX data
            self.bitx_data = self.data_source.get_data(
                bitx_symbol, period="5d", interval="1m")

            # Get VIX data
            self.vix_data = self.data_source.get_data(
                vix_symbol, period="5d", interval="1m")

            self.last_update = datetime.now()
            self.logger.info(f"Data updated at {self.last_update}")

        except Exception as e:
            self.logger.error(f"Error updating market data: {str(e)}")

    def get_bitx_data(self) -> pd.DataFrame:
        """Get BITX data"""
        return self.bitx_data.copy() if not self.bitx_data.empty else pd.DataFrame()

    def get_vix_data(self) -> pd.DataFrame:
        """Get VIX data"""
        return self.vix_data.copy() if not self.vix_data.empty else pd.DataFrame()

    def get_current_bitx_price(self) -> Optional[float]:
        """Get current BITX price"""
        return self.data_source.get_current_price('BITX')

    def get_current_vix_value(self) -> Optional[float]:
        """Get current VIX value"""
        return self.data_source.get_current_price('^VIX')

    def get_bitx_options(self) -> Dict:
        """Get BITX options chain"""
        return self.data_source.get_options_chain('BITX')

    def is_data_fresh(self, max_age_minutes: int = 5) -> bool:
        """Check if data is fresh enough"""
        if self.last_update is None:
            return False

        age = (datetime.now() - self.last_update).total_seconds() / 60
        return age <= max_age_minutes
