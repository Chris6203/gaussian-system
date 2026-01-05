#!/usr/bin/env python3
"""
Test Data Manager Integration

Quick test to verify the Data Manager connection and data fetching works.

Usage:
    python scripts/test_datamanager.py
"""

import os
import sys
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    print("=" * 60)
    print("DATA MANAGER INTEGRATION TEST")
    print("=" * 60)

    # Load config
    config = {}
    for path in ['config.json', '../config.json']:
        if os.path.exists(path):
            with open(path, 'r') as f:
                config = json.load(f)
            break

    dm_config = config.get('data_manager', {})
    base_url = dm_config.get('base_url', '')
    api_key = dm_config.get('api_key', '')

    # Use server_config.json fallback if base_url not in config.json
    if not base_url:
        try:
            from config_loader import get_data_manager_url
            base_url = get_data_manager_url()
            print(f"[INFO] Using server_config.json fallback: {base_url}")
        except ImportError:
            pass

    print(f"\n[1] Configuration Check")
    print(f"    URL: {base_url or '(not set)'}")
    print(f"    API Key: {'*' * 8 + api_key[-4:] if api_key else '(not set)'}")
    print(f"    Enabled: {dm_config.get('enabled', False)}")

    if not base_url or not api_key:
        print("\n[ERROR] Data Manager not configured!")
        print("[HINT] Add to config.json:")
        print('    "data_manager": {')
        print('        "enabled": true,')
        print('        "base_url": "http://192.168.20.235:5050",')
        print('        "api_key": "dm_your_api_key_here"')
        print('    }')
        print("\nOr set server_config.json for automatic fallback to localhost.")
        return 1

    # Test DataManagerDataSource directly
    print(f"\n[2] Testing DataManagerDataSource...")
    try:
        from backend.datamanager_data_source import DataManagerDataSource

        dm = DataManagerDataSource(base_url, api_key)
        print("    [OK] DataManagerDataSource created")

        # Health check
        if dm.health_check():
            print("    [OK] Server is healthy")
        else:
            print("    [ERROR] Server not responding")
            return 1

        # Get stats
        stats = dm.get_stats()
        if stats:
            print(f"    [OK] Got stats: {len(stats)} keys")

    except ImportError as e:
        print(f"    [ERROR] Import failed: {e}")
        return 1
    except Exception as e:
        print(f"    [ERROR] Connection failed: {e}")
        return 1

    # Test data fetching
    print(f"\n[3] Testing Data Fetch...")
    try:
        # Test SPY
        df = dm.get_data('SPY', period='1d', interval='1m')
        if df is not None and not df.empty:
            print(f"    [OK] SPY: {len(df)} records")
            print(f"        Latest: {df.index[-1]} - ${df['Close'].iloc[-1]:.2f}")
        else:
            print("    [WARN] No SPY data returned")

        # Test VIX
        df = dm.get_data('VIX', period='1d', interval='1m')
        if df is not None and not df.empty:
            print(f"    [OK] VIX: {len(df)} records")
        else:
            print("    [WARN] No VIX data returned")

        # Test current price
        price = dm.get_current_price('SPY')
        if price:
            print(f"    [OK] SPY current price: ${price:.2f}")
        else:
            print("    [WARN] No current price available")

    except Exception as e:
        print(f"    [ERROR] Data fetch failed: {e}")
        return 1

    # Test EnhancedDataSource integration
    print(f"\n[4] Testing EnhancedDataSource Integration...")
    try:
        from backend.enhanced_data_sources import EnhancedDataSource

        eds = EnhancedDataSource(config=config)

        if eds._datamanager_source:
            print("    [OK] Data Manager is primary source")
        else:
            print("    [WARN] Data Manager not initialized (check config)")

        # Test via EnhancedDataSource
        df = eds.get_data('SPY', period='1d', interval='1m')
        if df is not None and not df.empty:
            print(f"    [OK] EnhancedDataSource.get_data: {len(df)} records")
        else:
            print("    [WARN] No data via EnhancedDataSource")

    except Exception as e:
        print(f"    [ERROR] EnhancedDataSource test failed: {e}")
        return 1

    print(f"\n" + "=" * 60)
    print("[SUCCESS] Data Manager integration is working!")
    print("=" * 60)

    print("""
Next steps:
1. Sync historical data for training:
   python scripts/sync_from_datamanager.py --days 30

2. Run time-travel training:
   python scripts/train_time_travel.py

3. Live/paper trading will automatically use Data Manager
""")

    return 0


if __name__ == '__main__':
    sys.exit(main())
