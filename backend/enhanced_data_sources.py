#!/usr/bin/env python3
"""
Enhanced Data Sources - Unified Data Manager

Provides a unified interface for fetching market data from multiple sources
with automatic fallback and configuration support.
"""

import logging
import pandas as pd
from typing import Dict, Optional, Any

logger = logging.getLogger(__name__)


class EnhancedDataSource:
    """
    Unified data source that combines multiple data providers.
    
    Provides automatic fallback between:
    1. Tradier (primary, requires API key) - Best for real-time intraday
    2. FMP (secondary, requires API key) - Best for historical 1-min data
    3. Polygon (tertiary, requires API key) - Good for crypto and extended hours
    4. Yahoo Finance (fallback, free) - Good for daily data
    
    Accepts config dict to configure data sources and API keys.
    """
    
    def __init__(self, config: Dict = None, db_path: str = 'data/db/historical.db'):
        """
        Initialize enhanced data source.

        Args:
            config: Configuration dict with credentials and settings
            db_path: Path to historical database for caching
        """
        self.config = config or {}
        self.db_path = db_path
        self.cache_db_path = db_path  # Alias for compatibility
        self.logger = logging.getLogger(__name__)

        # Initialize data sources based on config
        self._datamanager_source = None  # Remote Data Manager (highest priority)
        self._primary_source = None      # Tradier
        self._secondary_source = None    # FMP
        self._tertiary_source = None     # Polygon
        self._fallback_source = None     # Yahoo
        self._fmp_source = None          # Direct reference to FMP for historical backfill

        self._initialize_sources()
    
    def _initialize_sources(self):
        """Initialize data sources based on configuration."""
        # Try to initialize Data Manager as highest priority source (centralized data)
        dm_config = self.config.get('data_manager', {})
        if dm_config.get('enabled', False):
            try:
                from backend.datamanager_data_source import DataManagerDataSource

                base_url = dm_config.get('base_url', '')
                api_key = dm_config.get('api_key', '')

                if base_url and api_key:
                    self._datamanager_source = DataManagerDataSource(
                        base_url=base_url,
                        api_key=api_key,
                        timeout=dm_config.get('timeout_seconds', 30),
                        cache_ttl_seconds=dm_config.get('cache_ttl_seconds', 60)
                    )
                    # Health check
                    if self._datamanager_source.health_check():
                        self.logger.info(f"[DATA] Data Manager source initialized: {base_url}")
                    else:
                        self.logger.warning(f"[DATA] Data Manager at {base_url} not responding, will use fallback")
                        self._datamanager_source = None
                elif base_url and not api_key:
                    self.logger.warning("[DATA] Data Manager enabled but no API key configured")
            except ImportError:
                self.logger.debug("[DATA] Data Manager data source not available")
            except Exception as e:
                self.logger.warning(f"[DATA] Could not initialize Data Manager: {e}")

        # Try to initialize Tradier as primary source (best for real-time)
        try:
            from backend.tradier_data_source import TradierDataSource

            # Get Tradier API key from config
            tradier_config = self.config.get('credentials', {}).get('tradier', {})
            api_key = tradier_config.get('data_api_token')

            if api_key:
                self._primary_source = TradierDataSource(api_key=api_key, sandbox=False)
                self.logger.info("[DATA] Tradier data source initialized (primary)")
        except ImportError:
            self.logger.debug("[DATA] Tradier data source not available")
        except Exception as e:
            self.logger.warning(f"[DATA] Could not initialize Tradier: {e}")
        
        # Try to initialize FMP as secondary source (best for historical 1-min data)
        try:
            from backend.fmp_data_source import FMPDataSource
            
            fmp_config = self.config.get('credentials', {}).get('fmp', {})
            api_key = fmp_config.get('api_key')
            
            if api_key:
                # Pass config so FMP can read rate limits
                self._fmp_source = FMPDataSource(
                    api_key=api_key, 
                    db_path=self.db_path,
                    config=self.config
                )
                self._secondary_source = self._fmp_source
                rate_limit = fmp_config.get('requests_per_minute', 300)
                self.logger.info(f"[DATA] FMP data source initialized (secondary - {rate_limit} req/min)")
        except ImportError:
            self.logger.debug("[DATA] FMP data source not available")
        except Exception as e:
            self.logger.warning(f"[DATA] Could not initialize FMP: {e}")
        
        # Try to initialize Polygon as tertiary source (good for crypto/extended hours)
        try:
            from backend.polygon_data_source import PolygonDataSource
            
            polygon_config = self.config.get('credentials', {}).get('polygon', {})
            api_key = polygon_config.get('api_key')
            
            if api_key:
                self._tertiary_source = PolygonDataSource(api_key=api_key)
                self.logger.info("[DATA] Polygon data source initialized (tertiary)")
        except ImportError:
            self.logger.debug("[DATA] Polygon data source not available")
        except Exception as e:
            self.logger.warning(f"[DATA] Could not initialize Polygon: {e}")
        
        # Always initialize Yahoo Finance as fallback
        try:
            from backend.data_sources import YahooFinanceDataSource
            self._fallback_source = YahooFinanceDataSource()
            self.logger.info("[DATA] Yahoo Finance data source initialized (fallback)")
        except ImportError:
            self.logger.warning("[DATA] Yahoo Finance data source not available")
        except Exception as e:
            self.logger.warning(f"[DATA] Could not initialize Yahoo: {e}")
    
    def get_data(
        self,
        symbol: str,
        period: str = "1d",
        interval: str = "1m"
    ) -> pd.DataFrame:
        """
        Get historical data for a symbol with automatic fallback.

        Args:
            symbol: Stock symbol (e.g., 'SPY', '^VIX')
            period: Data period ('1d', '5d', '7d', '30d', '90d', etc.)
            interval: Data interval ('1m', '5m', '15m', '1h', '1d')

        Returns:
            DataFrame with OHLCV data
        """
        # Normalize symbol for API calls (some APIs don't like ^)
        api_symbol = symbol.lstrip('^') if symbol.startswith('^') else symbol

        # Try Data Manager first (centralized data - highest priority)
        if self._datamanager_source:
            try:
                data = self._datamanager_source.get_data(api_symbol, period=period, interval=interval)

                if data is not None and not data.empty:
                    self.logger.debug(f"[DATA] Got {len(data)} records from DataManager for {symbol}")
                    return data
            except Exception as e:
                self.logger.debug(f"[DATA] DataManager failed for {symbol}: {e}")

        # Try primary source (Tradier - best for real-time intraday)
        if self._primary_source:
            try:
                # Check if Tradier supports this symbol
                if hasattr(self._primary_source, 'get_historical_data'):
                    data = self._primary_source.get_historical_data(
                        api_symbol, period=period, interval=interval
                    )
                else:
                    data = self._primary_source.get_data(
                        api_symbol, period=period, interval=interval
                    )
                
                if data is not None and not data.empty:
                    self.logger.debug(f"[DATA] Got {len(data)} records from Tradier for {symbol}")
                    return data
            except Exception as e:
                self.logger.debug(f"[DATA] Tradier failed for {symbol}: {e}")
        
        # Try secondary source (FMP - best for historical 1-min data)
        if self._secondary_source:
            try:
                if hasattr(self._secondary_source, 'get_historical_data'):
                    data = self._secondary_source.get_historical_data(
                        api_symbol, period=period, interval=interval
                    )
                else:
                    data = self._secondary_source.get_data(
                        api_symbol, period=period, interval=interval
                    )
                
                if data is not None and not data.empty:
                    self.logger.debug(f"[DATA] Got {len(data)} records from FMP for {symbol}")
                    return data
            except Exception as e:
                self.logger.debug(f"[DATA] FMP failed for {symbol}: {e}")
        
        # Try tertiary source (Polygon - good for crypto/extended hours)
        if self._tertiary_source:
            try:
                if hasattr(self._tertiary_source, 'get_historical_data'):
                    data = self._tertiary_source.get_historical_data(
                        api_symbol, period=period, interval=interval
                    )
                else:
                    data = self._tertiary_source.get_data(
                        api_symbol, period=period, interval=interval
                    )
                
                if data is not None and not data.empty:
                    self.logger.debug(f"[DATA] Got {len(data)} records from Polygon for {symbol}")
                    return data
            except Exception as e:
                self.logger.debug(f"[DATA] Polygon failed for {symbol}: {e}")
        
        # Try fallback source (Yahoo Finance)
        if self._fallback_source:
            try:
                data = self._fallback_source.get_data(symbol, period=period, interval=interval)
                
                if data is not None and not data.empty:
                    self.logger.debug(f"[DATA] Got {len(data)} records from Yahoo for {symbol}")
                    return data
            except Exception as e:
                self.logger.debug(f"[DATA] Yahoo failed for {symbol}: {e}")
        
        self.logger.warning(f"[DATA] All data sources failed for {symbol}")
        return pd.DataFrame()
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Get current price for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Current price or None if unavailable
        """
        api_symbol = symbol.lstrip('^') if symbol.startswith('^') else symbol

        # Try Data Manager first
        if self._datamanager_source:
            try:
                price = self._datamanager_source.get_current_price(api_symbol)
                if price is not None:
                    return price
            except Exception as e:
                self.logger.debug(f"[DATA] DataManager price failed for {symbol}: {e}")

        # Try to get from recent data
        try:
            data = self.get_data(symbol, period="1d", interval="1m")
            if data is not None and not data.empty:
                # Normalize column names
                cols = [c.lower() for c in data.columns]
                if 'close' in cols:
                    return float(data['close'].iloc[-1])
                elif 'Close' in data.columns:
                    return float(data['Close'].iloc[-1])
        except Exception as e:
            self.logger.debug(f"[DATA] Could not get current price for {symbol}: {e}")

        # Try fallback method
        if self._fallback_source and hasattr(self._fallback_source, 'get_current_price'):
            try:
                return self._fallback_source.get_current_price(symbol)
            except Exception:
                pass

        return None
    
    def get_options_chain(self, symbol: str) -> Dict:
        """
        Get options chain for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Options chain data dict
        """
        api_symbol = symbol.lstrip('^') if symbol.startswith('^') else symbol

        # Try Data Manager first (has options with greeks)
        if self._datamanager_source:
            try:
                chain = self._datamanager_source.get_options_chain(api_symbol)
                if chain:
                    return chain
            except Exception as e:
                self.logger.debug(f"[DATA] DataManager options chain failed: {e}")

        # Try Tradier (best for live options)
        if self._primary_source and hasattr(self._primary_source, 'get_options_chain'):
            try:
                return self._primary_source.get_options_chain(symbol)
            except Exception as e:
                self.logger.debug(f"[DATA] Tradier options chain failed: {e}")

        # Fallback to Yahoo
        if self._fallback_source and hasattr(self._fallback_source, 'get_options_chain'):
            try:
                return self._fallback_source.get_options_chain(symbol)
            except Exception as e:
                self.logger.debug(f"[DATA] Yahoo options chain failed: {e}")

        return {}
    
    def fetch_historical_1min(
        self,
        symbols: list,
        days_back: int = 30
    ) -> dict:
        """
        Fetch and store historical 1-minute data using FMP.
        
        This method is specifically for backfilling historical data
        that can be used for training and backtesting.
        
        Args:
            symbols: List of stock symbols
            days_back: Number of days of history to fetch (FMP free tier may limit this)
            
        Returns:
            Dict with results per symbol
        """
        if not self._fmp_source:
            self.logger.error("[DATA] FMP data source not available for historical backfill")
            return {'error': 'FMP not configured'}
        
        self.logger.info(f"[DATA] Fetching {days_back} days of 1-min historical data for {len(symbols)} symbols via FMP")
        return self._fmp_source.fetch_and_store_historical(symbols, days_back, '1min')
    
    def get_fmp_stored_data(
        self,
        symbol: str,
        from_date: Optional[Any] = None,
        to_date: Optional[Any] = None,
        limit: int = 10000
    ) -> pd.DataFrame:
        """
        Get historical data from local FMP database.
        
        Args:
            symbol: Stock symbol
            from_date: Start date
            to_date: End date
            limit: Maximum records
            
        Returns:
            DataFrame with OHLCV data
        """
        if not self._fmp_source:
            self.logger.warning("[DATA] FMP data source not available")
            return pd.DataFrame()
        
        return self._fmp_source.get_from_database(symbol, from_date, to_date, limit)


# Backward compatibility aliases
TradierMarketDataManager = EnhancedDataSource
DataManager = EnhancedDataSource

__all__ = ['EnhancedDataSource', 'TradierMarketDataManager', 'DataManager']
