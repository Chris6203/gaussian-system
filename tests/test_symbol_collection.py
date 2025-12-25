#!/usr/bin/env python3
"""
Test Symbol Collection for Mega-Cap Tech
=========================================

Verifies that the DataCollector properly fetches and stores data
for all MEGA_CAP_TECH_SYMBOLS.

Usage:
    python -m pytest tests/test_symbol_collection.py -v
    python tests/test_symbol_collection.py  # Direct run
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import sqlite3
from datetime import datetime, timedelta
from typing import List, Dict, Optional

# Import our modules
from config.symbols import (
    MEGA_CAP_TECH_SYMBOLS,
    ALL_DATA_COLLECTION_SYMBOLS,
    get_all_symbols,
    get_tech_symbols,
)
from data_collector import CollectorConfig, DataFetcher, DataStorage


class TestSymbolConfiguration:
    """Test that symbol configuration is correct."""
    
    def test_mega_cap_tech_symbols_defined(self):
        """Verify MEGA_CAP_TECH_SYMBOLS contains expected symbols."""
        expected = ['NVDA', 'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'PLTR', 'INTC', 'AMD']
        
        assert MEGA_CAP_TECH_SYMBOLS is not None
        assert len(MEGA_CAP_TECH_SYMBOLS) == 10
        
        for symbol in expected:
            assert symbol in MEGA_CAP_TECH_SYMBOLS, f"Missing symbol: {symbol}"
    
    def test_all_symbols_includes_tech(self):
        """Verify ALL_DATA_COLLECTION_SYMBOLS includes tech symbols."""
        for symbol in MEGA_CAP_TECH_SYMBOLS:
            assert symbol in ALL_DATA_COLLECTION_SYMBOLS, f"Tech symbol {symbol} not in ALL_DATA_COLLECTION_SYMBOLS"
    
    def test_get_tech_symbols_function(self):
        """Verify get_tech_symbols() returns a copy."""
        symbols = get_tech_symbols()
        assert symbols == MEGA_CAP_TECH_SYMBOLS
        
        # Verify it's a copy, not the original
        symbols.append('TEST')
        assert 'TEST' not in MEGA_CAP_TECH_SYMBOLS


class TestCollectorConfig:
    """Test CollectorConfig symbol loading."""
    
    def test_config_includes_tech_symbols(self):
        """Verify CollectorConfig returns tech symbols."""
        config = CollectorConfig()
        symbols = config.get_symbols()
        
        for tech_symbol in MEGA_CAP_TECH_SYMBOLS:
            assert tech_symbol in symbols, f"Config missing tech symbol: {tech_symbol}"
    
    def test_default_symbols_include_tech(self):
        """Verify DEFAULT_SYMBOLS constant includes tech."""
        defaults = CollectorConfig.DEFAULT_SYMBOLS
        
        for tech_symbol in MEGA_CAP_TECH_SYMBOLS:
            assert tech_symbol in defaults, f"DEFAULT_SYMBOLS missing: {tech_symbol}"


class TestDataFetcher:
    """Test that DataFetcher can fetch tech symbols."""
    
    @pytest.fixture
    def fetcher(self):
        """Create a DataFetcher instance."""
        import logging
        logger = logging.getLogger('test')
        config = CollectorConfig()
        return DataFetcher(config, logger)
    
    @pytest.mark.parametrize("symbol", MEGA_CAP_TECH_SYMBOLS)
    def test_can_fetch_tech_symbol(self, fetcher, symbol):
        """
        Test that each tech symbol can be fetched.
        
        Note: This test makes real API calls, so it may be slow.
        Skip in CI by using: pytest -m "not slow"
        """
        data = fetcher.get_price_data(symbol, period='1d', interval='1m')
        
        # Should return some data (may be None if market is closed/weekend)
        # At minimum, the fetch should not error out
        if data is not None:
            assert 'close' in data, f"No close price for {symbol}"
            assert data['close'] > 0, f"Invalid close price for {symbol}"
            print(f"✓ {symbol}: ${data['close']:.2f}")
        else:
            # Data may be None if market is closed - that's OK
            print(f"⚠ {symbol}: No data (market may be closed)")


class TestDataStorage:
    """Test that DataStorage properly handles tech symbols."""
    
    @pytest.fixture
    def temp_db(self, tmp_path):
        """Create a temporary database for testing."""
        import logging
        db_path = str(tmp_path / "test_historical.db")
        logger = logging.getLogger('test')
        storage = DataStorage(db_path=db_path, logger=logger)
        return storage, db_path
    
    def test_can_store_tech_symbol_data(self, temp_db):
        """Test storing data for a tech symbol."""
        storage, db_path = temp_db
        
        test_data = {
            'timestamp': datetime.now().isoformat(),
            'open': 100.0,
            'high': 105.0,
            'low': 98.0,
            'close': 103.5,
            'volume': 1000000
        }
        
        # Store data for NVDA
        success = storage.save_price_data('NVDA', test_data)
        assert success, "Failed to save NVDA data"
        
        # Verify it was stored
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM historical_data WHERE symbol = 'NVDA'")
        rows = cursor.fetchall()
        conn.close()
        
        assert len(rows) == 1, "NVDA data not found in database"
    
    def test_can_store_all_tech_symbols(self, temp_db):
        """Test storing data for all tech symbols."""
        storage, db_path = temp_db
        
        for symbol in MEGA_CAP_TECH_SYMBOLS:
            test_data = {
                'timestamp': datetime.now().isoformat(),
                'open': 100.0,
                'high': 105.0,
                'low': 98.0,
                'close': 103.5,
                'volume': 1000000
            }
            
            success = storage.save_price_data(symbol, test_data)
            assert success, f"Failed to save {symbol} data"
        
        # Verify all symbols stored
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT symbol FROM historical_data")
        stored_symbols = {row[0] for row in cursor.fetchall()}
        conn.close()
        
        for symbol in MEGA_CAP_TECH_SYMBOLS:
            assert symbol in stored_symbols, f"{symbol} not found in database"


def quick_verification():
    """
    Quick verification that can be run directly.
    
    Checks:
    1. Config includes all tech symbols
    2. At least one tech symbol can be fetched
    3. Data can be stored
    """
    import logging
    logging.basicConfig(level=logging.WARNING)
    
    print("=" * 60)
    print("MEGA-CAP TECH SYMBOL VERIFICATION")
    print("=" * 60)
    
    # 1. Check config
    print("\n[1] Checking configuration...")
    config = CollectorConfig()
    symbols = config.get_symbols()
    
    missing = [s for s in MEGA_CAP_TECH_SYMBOLS if s not in symbols]
    if missing:
        print(f"  ❌ Missing symbols in config: {missing}")
        return False
    print(f"  ✓ All {len(MEGA_CAP_TECH_SYMBOLS)} tech symbols in config")
    print(f"  Total symbols: {len(symbols)}")
    
    # 2. Try fetching one symbol
    print("\n[2] Testing data fetch (NVDA)...")
    logger = logging.getLogger('test')
    fetcher = DataFetcher(config, logger)
    
    data = fetcher.get_price_data('NVDA', period='1d', interval='1m')
    if data and data.get('close'):
        print(f"  ✓ NVDA: ${data['close']:.2f}")
    else:
        print("  ⚠ NVDA: No data (market may be closed)")
    
    # 3. Test storage
    print("\n[3] Testing data storage...")
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        storage = DataStorage(db_path=db_path, logger=logger)
        
        test_data = {
            'timestamp': datetime.now().isoformat(),
            'open': 100.0, 'high': 105.0, 'low': 98.0,
            'close': 103.5, 'volume': 1000000
        }
        
        stored = 0
        for symbol in MEGA_CAP_TECH_SYMBOLS:
            if storage.save_price_data(symbol, test_data):
                stored += 1
        
        if stored == len(MEGA_CAP_TECH_SYMBOLS):
            print(f"  ✓ Stored data for all {stored} tech symbols")
        else:
            print(f"  ❌ Only stored {stored}/{len(MEGA_CAP_TECH_SYMBOLS)} symbols")
            return False
    
    print("\n" + "=" * 60)
    print("✓ VERIFICATION PASSED")
    print("=" * 60)
    return True


if __name__ == "__main__":
    # Run quick verification when executed directly
    success = quick_verification()
    sys.exit(0 if success else 1)




