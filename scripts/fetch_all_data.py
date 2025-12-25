#!/usr/bin/env python3
"""
Fetch All Configured Data - Test Script

Fetches all symbols defined in config.json and verifies data sources work correctly.
"""

import sys
import os
from datetime import datetime
from config_loader import load_config
from src.data.manager import DataManager

print("="*80)
print("FETCH ALL DATA - Test Script")
print("="*80)
print()

# Load configuration
config = load_config("config.json")

# Get symbols to fetch
try:
    from config.symbols import ALL_DATA_COLLECTION_SYMBOLS
    default_symbols = ALL_DATA_COLLECTION_SYMBOLS
except ImportError:
    default_symbols = ['SPY', '^VIX', 'UUP']

symbols = config.get('data_fetching', 'symbols', default=default_symbols)

print(f"[CONFIG] Trading Symbol: {config.get_symbol()}")
print(f"[CONFIG] Data Symbols: {', '.join(symbols)}")
fallback_order = config.get('data_sources', 'fallback_order', default=['tradier', 'yahoo'])
print(f"[CONFIG] Data Sources: {' → '.join(fallback_order)}")
print()

# Initialize DataManager
print("[*] Initializing DataManager...")
data_manager = DataManager(config=config.config)
print()

# Fetch data for each symbol
print("="*80)
print("FETCHING DATA")
print("="*80)
print()

results = {}
for symbol in symbols:
    print(f"[{symbol}] Fetching data...")
    try:
        data = data_manager.get_data(
            symbol=symbol,
            period=config.get('data_sources', 'historical_period', default='1d'),
            interval=config.get('data_sources', 'interval', default='1m')
        )
        
        if not data.empty:
            print(f"  ✅ Got {len(data)} rows")
            print(f"     Date range: {data.index[0]} to {data.index[-1]}")
            if 'close' in data.columns:
                latest_price = data['close'].iloc[-1]
                print(f"     Latest price: ${latest_price:.2f}")
            results[symbol] = {
                'success': True,
                'rows': len(data),
                'data': data
            }
        else:
            print(f"  ❌ No data received")
            results[symbol] = {
                'success': False,
                'rows': 0,
                'data': None
            }
    except Exception as e:
        print(f"  ❌ Error: {e}")
        results[symbol] = {
            'success': False,
            'error': str(e),
            'data': None
        }
    
    print()

# Summary
print("="*80)
print("SUMMARY")
print("="*80)
print()

success_count = sum(1 for r in results.values() if r.get('success'))
total_count = len(results)

print(f"Successfully fetched: {success_count}/{total_count} symbols")
print()

if success_count == total_count:
    print("✅ ALL SYMBOLS FETCHED SUCCESSFULLY!")
    print()
    print("Data is ready for training and live trading.")
else:
    print("⚠️  SOME SYMBOLS FAILED")
    print()
    print("Failed symbols:")
    for symbol, result in results.items():
        if not result.get('success'):
            error = result.get('error', 'No data')
            print(f"  - {symbol}: {error}")
    print()
    print("Note: The bot can still work with available data,")
    print("but some features may be limited.")

print()
print("="*80)
print("DATA FETCH TEST COMPLETE")
print("="*80)

