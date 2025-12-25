#!/usr/bin/env python3
"""
Smoke Tests for Feature Pipeline

Tests that:
- All feature families produce expected columns
- No exceptions on missing data
- Caching works correctly
- Feature consistency is maintained
"""

import sys
import os
import unittest
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestEquityETFFeatures(unittest.TestCase):
    """Test equity/ETF feature computation."""
    
    def setUp(self):
        """Create mock data."""
        np.random.seed(42)
        self.n_bars = 200
        
        self.data = {}
        for symbol in ['SPY', 'QQQ', 'IWM']:
            prices = 100 * np.exp(np.cumsum(np.random.randn(self.n_bars) * 0.01))
            df = pd.DataFrame({
                'open': prices * 0.99,
                'high': prices * 1.01,
                'low': prices * 0.98,
                'close': prices,
                'volume': np.random.randint(1000000, 10000000, self.n_bars),
            }, index=pd.date_range('2024-01-01', periods=self.n_bars, freq='1h'))
            self.data[symbol] = df
    
    def test_compute_equity_features(self):
        """Test that equity features are computed."""
        from features.equity_etf import compute_equity_etf_features
        
        features = compute_equity_etf_features(self.data, primary_symbol='SPY')
        
        self.assertIsInstance(features, pd.DataFrame)
        self.assertGreater(len(features.columns), 0)
        self.assertEqual(len(features), self.n_bars)
    
    def test_get_latest_features(self):
        """Test getting latest features as dict."""
        from features.equity_etf import get_latest_equity_etf_features
        
        features = get_latest_equity_etf_features(self.data, primary_symbol='SPY')
        
        self.assertIsInstance(features, dict)
        self.assertGreater(len(features), 0)
        
        # Check for expected feature patterns
        has_returns = any('ret_' in k for k in features.keys())
        self.assertTrue(has_returns, "Should have return features")
    
    def test_empty_data_handling(self):
        """Test graceful handling of empty data."""
        from features.equity_etf import compute_equity_etf_features
        
        # Empty dict
        features = compute_equity_etf_features({}, primary_symbol='SPY')
        self.assertTrue(features.empty)
        
        # Missing primary symbol
        features = compute_equity_etf_features({'QQQ': self.data['QQQ']}, primary_symbol='SPY')
        self.assertTrue(features.empty)


class TestOptionsSurfaceFeatures(unittest.TestCase):
    """Test options surface feature computation."""
    
    def setUp(self):
        """Create mock options chain."""
        self.spot = 450.0
        strikes = np.arange(430, 470, 2.5)
        
        records = []
        for strike in strikes:
            for opt_type in ['call', 'put']:
                records.append({
                    'type': opt_type,
                    'strike': strike,
                    'expiration': '2024-01-19',
                    'bid': 2.0,
                    'ask': 2.5,
                    'volume': np.random.randint(100, 5000),
                    'open_interest': np.random.randint(1000, 50000),
                    'delta': 0.5 if opt_type == 'call' else -0.5,
                    'gamma': 0.02,
                    'mid_iv': 0.15,
                })
        
        self.chain = pd.DataFrame(records)
    
    def test_compute_surface_features(self):
        """Test that surface features are computed."""
        from features.options_surface import compute_options_surface_features
        
        features = compute_options_surface_features(self.chain, self.spot)
        
        self.assertIsInstance(features, dict)
        self.assertIn('atm_iv_avg', features)
        self.assertIn('oi_put_call_ratio', features)
        self.assertIn('gex_total', features)
    
    def test_empty_chain_handling(self):
        """Test graceful handling of empty chain."""
        from features.options_surface import compute_options_surface_features
        
        features = compute_options_surface_features(pd.DataFrame(), self.spot)
        
        self.assertIsInstance(features, dict)
        self.assertEqual(features['atm_iv_avg'], 0.0)
    
    def test_zero_spot_handling(self):
        """Test graceful handling of zero spot price."""
        from features.options_surface import compute_options_surface_features
        
        features = compute_options_surface_features(self.chain, 0.0)
        
        self.assertIsInstance(features, dict)


class TestBreadthFeatures(unittest.TestCase):
    """Test breadth feature computation."""
    
    def setUp(self):
        """Create mock data."""
        np.random.seed(42)
        n_bars = 100
        
        self.data = {}
        for symbol in ['SPY', 'QQQ', 'IWM', 'XLF', 'XLK']:
            prices = 100 * np.exp(np.cumsum(np.random.randn(n_bars) * 0.01))
            df = pd.DataFrame({
                'close': prices,
                'volume': np.random.randint(1000000, 10000000, n_bars),
            })
            self.data[symbol] = df
    
    def test_compute_breadth_features(self):
        """Test that breadth features are computed."""
        from features.breadth import compute_breadth_features
        
        features = compute_breadth_features(self.data)
        
        self.assertIsInstance(features, dict)
        self.assertIn('breadth_up_frac_short', features)
        self.assertIn('xsec_ret_mean', features)
        
        # Values should be reasonable
        self.assertGreaterEqual(features['breadth_up_frac_short'], 0)
        self.assertLessEqual(features['breadth_up_frac_short'], 1)
    
    def test_single_symbol(self):
        """Test with single symbol."""
        from features.breadth import compute_breadth_features
        
        features = compute_breadth_features({'SPY': self.data['SPY']})
        
        self.assertIsInstance(features, dict)


class TestMacroFeatures(unittest.TestCase):
    """Test macro feature computation."""
    
    def setUp(self):
        """Create mock data."""
        np.random.seed(42)
        n_bars = 100
        
        self.data = {}
        for symbol in ['SPY', 'TLT', 'IEF', 'SHY', 'UUP']:
            prices = 100 * np.exp(np.cumsum(np.random.randn(n_bars) * 0.01))
            df = pd.DataFrame({
                'close': prices,
                'volume': np.random.randint(1000000, 10000000, n_bars),
            })
            self.data[symbol] = df
    
    def test_compute_macro_features(self):
        """Test that macro features are computed."""
        from features.macro import compute_macro_features
        
        features = compute_macro_features(self.data)
        
        self.assertIsInstance(features, dict)
        self.assertIn('tlt_ret_5', features)
        self.assertIn('dollar_strength', features)
    
    def test_missing_symbols(self):
        """Test with missing symbols."""
        from features.macro import compute_macro_features
        
        # Only SPY
        features = compute_macro_features({'SPY': self.data['SPY']})
        
        self.assertIsInstance(features, dict)
        self.assertEqual(features['tlt_ret_5'], 0.0)


class TestCryptoFeatures(unittest.TestCase):
    """Test crypto feature computation."""
    
    def setUp(self):
        """Create mock data."""
        np.random.seed(42)
        n_bars = 200
        
        self.data = {}
        for symbol in ['BTC-USD', 'ETH-USD', 'SPY']:
            base_price = 50000 if 'BTC' in symbol else (3000 if 'ETH' in symbol else 450)
            vol = 0.02 if 'USD' in symbol else 0.005
            prices = base_price * np.exp(np.cumsum(np.random.randn(n_bars) * vol))
            df = pd.DataFrame({
                'close': prices,
                'volume': np.random.randint(10000, 100000, n_bars),
            })
            self.data[symbol] = df
    
    def test_compute_crypto_features(self):
        """Test that crypto features are computed."""
        from features.crypto import compute_crypto_features
        
        features = compute_crypto_features(self.data)
        
        self.assertIsInstance(features, dict)
        self.assertIn('btc_ret_5', features)
        self.assertIn('crypto_risk_on', features)
    
    def test_no_crypto_data(self):
        """Test with no crypto data."""
        from features.crypto import compute_crypto_features
        
        features = compute_crypto_features({'SPY': self.data['SPY']})
        
        self.assertIsInstance(features, dict)
        self.assertEqual(features['btc_ret_5'], 0.0)


class TestMetaFeatures(unittest.TestCase):
    """Test meta feature computation."""
    
    def test_time_features(self):
        """Test time-of-day features."""
        from features.meta import compute_time_features
        
        # Market hours
        features = compute_time_features(datetime(2024, 1, 15, 10, 30))
        
        self.assertIsInstance(features, dict)
        self.assertIn('session_progress', features)
        self.assertGreater(features['session_progress'], 0)
        
        # First 30 min
        features = compute_time_features(datetime(2024, 1, 15, 9, 45))
        self.assertEqual(features['is_first_30min'], 1.0)
    
    def test_expiry_features(self):
        """Test expiry features."""
        from features.meta import compute_expiry_features
        
        features = compute_expiry_features(
            current_date=date(2024, 1, 15),
            expiry_date=date(2024, 1, 17)
        )
        
        self.assertIsInstance(features, dict)
        self.assertEqual(features['days_to_expiry'], 2.0)
        self.assertEqual(features['is_0dte'], 0.0)
    
    def test_vol_regime_features(self):
        """Test volatility regime features."""
        from features.meta import compute_vol_regime_features
        
        vol_history = np.abs(np.random.randn(100) * 0.02)
        
        # High vol
        features = compute_vol_regime_features(0.04, vol_history, vix_level=30)
        self.assertIsInstance(features, dict)
        self.assertEqual(features['vix_regime_high'], 1.0)


class TestFeaturePipeline(unittest.TestCase):
    """Test unified feature pipeline."""
    
    def setUp(self):
        """Create mock data."""
        np.random.seed(42)
        n_bars = 200
        
        self.equity_data = {}
        symbols = ['SPY', 'QQQ', 'IWM', 'TLT', 'UUP', 'BTC-USD']
        
        for symbol in symbols:
            prices = 100 * np.exp(np.cumsum(np.random.randn(n_bars) * 0.01))
            df = pd.DataFrame({
                'open': prices * 0.99,
                'high': prices * 1.01,
                'low': prices * 0.98,
                'close': prices,
                'volume': np.random.randint(1000000, 10000000, n_bars),
            }, index=pd.date_range('2024-01-01', periods=n_bars, freq='1h'))
            self.equity_data[symbol] = df
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        from features.pipeline import FeaturePipeline, FeatureConfig
        
        config = FeatureConfig()
        pipeline = FeaturePipeline(config)
        
        self.assertIsNotNone(pipeline)
        self.assertIsNotNone(pipeline.config)
    
    def test_compute_features(self):
        """Test feature computation."""
        from features.pipeline import FeaturePipeline, FeatureConfig
        
        config = FeatureConfig(
            equity_symbols=['SPY', 'QQQ'],
            breadth_symbols=['SPY', 'QQQ', 'IWM'],
        )
        pipeline = FeaturePipeline(config)
        
        features = pipeline.compute_features(
            equity_data=self.equity_data,
            spot_price=450.0
        )
        
        self.assertIsInstance(features, dict)
        self.assertGreater(len(features), 0)
    
    def test_feature_array(self):
        """Test getting features as array."""
        from features.pipeline import FeaturePipeline
        
        pipeline = FeaturePipeline()
        
        arr = pipeline.get_feature_array(
            equity_data=self.equity_data,
            spot_price=450.0
        )
        
        self.assertIsInstance(arr, np.ndarray)
        self.assertEqual(arr.dtype, np.float32)
    
    def test_feature_consistency(self):
        """Test that feature columns are consistent across calls."""
        from features.pipeline import FeaturePipeline
        
        pipeline = FeaturePipeline()
        
        # First call
        features1 = pipeline.compute_features(
            equity_data=self.equity_data,
            spot_price=450.0
        )
        
        # Second call
        features2 = pipeline.compute_features(
            equity_data=self.equity_data,
            spot_price=455.0
        )
        
        # Same keys
        self.assertEqual(set(features1.keys()), set(features2.keys()))


class TestDataProviders(unittest.TestCase):
    """Test data provider clients."""
    
    def test_cache_initialization(self):
        """Test cache initialization."""
        from data.providers import DiskCache
        
        cache = DiskCache(cache_dir='data_cache_test')
        self.assertIsNotNone(cache)
    
    def test_tradier_client_init(self):
        """Test Tradier client initialization."""
        from data.providers import TradierDataClient
        
        client = TradierDataClient(api_key='test_key')
        self.assertIsNotNone(client)
        self.assertEqual(client.api_key, 'test_key')
    
    def test_polygon_client_init(self):
        """Test Polygon client initialization."""
        from data.providers import PolygonDataClient
        
        client = PolygonDataClient(api_key='test_key')
        self.assertIsNotNone(client)
        self.assertEqual(client.api_key, 'test_key')
    
    def test_unsupported_symbols(self):
        """Test Tradier unsupported symbol detection."""
        from data.providers import TradierDataClient
        
        client = TradierDataClient()
        
        self.assertFalse(client._is_supported('BTC-USD'))
        self.assertFalse(client._is_supported('ETH-USD'))
        self.assertTrue(client._is_supported('SPY'))
        self.assertTrue(client._is_supported('QQQ'))


class TestIntegration(unittest.TestCase):
    """Test integration module."""
    
    def test_enhanced_feature_manager(self):
        """Test EnhancedFeatureManager initialization."""
        from features.integration import EnhancedFeatureManager
        
        manager = EnhancedFeatureManager(enable_enhanced_features=True)
        
        self.assertIsNotNone(manager)
        self.assertTrue(manager.is_enhanced_enabled)
    
    def test_data_update(self):
        """Test data update in manager."""
        from features.integration import EnhancedFeatureManager
        
        manager = EnhancedFeatureManager(enable_enhanced_features=True)
        
        # Create mock data
        df = pd.DataFrame({
            'close': [100, 101, 102],
            'volume': [1000, 1100, 1200],
        })
        
        manager.update_data('SPY', df)
        
        self.assertIn('SPY', manager._data_cache)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)




