#!/usr/bin/env python3
"""
Sync Data from Remote Data Manager

Fetches historical data from the remote Data Manager server and stores it
in the local SQLite database for time-travel training.

Usage:
    python scripts/sync_from_datamanager.py                    # Sync last 7 days
    python scripts/sync_from_datamanager.py --days 30          # Sync last 30 days
    python scripts/sync_from_datamanager.py --symbols SPY QQQ  # Sync specific symbols
"""

import os
import sys
import json
import sqlite3
import argparse
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from backend.datamanager_data_source import DataManagerDataSource
except ImportError:
    print("[ERROR] Could not import DataManagerDataSource")
    print("[HINT] Make sure backend/datamanager_data_source.py exists")
    sys.exit(1)


def load_config():
    """Load configuration from config.json."""
    config_paths = ['config.json', '../config.json']
    for path in config_paths:
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
    return {}


def sync_symbol(dm_source, symbol, days, db_path):
    """Sync a single symbol from Data Manager to local database."""
    print(f"  [*] Syncing {symbol}...")

    try:
        # Fetch data from Data Manager
        period = f"{days}d"
        df = dm_source.get_data(symbol, period=period, interval='1m')

        if df is None or df.empty:
            print(f"      [WARN] No data returned for {symbol}")
            return 0

        # Normalize column names
        df = df.reset_index()
        if 'Datetime' in df.columns:
            df = df.rename(columns={'Datetime': 'timestamp'})
        elif 'index' in df.columns:
            df = df.rename(columns={'index': 'timestamp'})

        # Ensure required columns
        required_cols = {'timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'}
        df_cols = set(df.columns)

        # Handle lowercase columns
        col_map = {}
        for col in df.columns:
            if col.lower() in ['open', 'high', 'low', 'close', 'volume']:
                col_map[col] = col.capitalize()
        if col_map:
            df = df.rename(columns=col_map)

        # Connect to database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Ensure table exists
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS historical_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                open_price REAL,
                high_price REAL,
                low_price REAL,
                close_price REAL,
                volume INTEGER,
                source TEXT DEFAULT 'datamanager',
                UNIQUE(symbol, timestamp)
            )
        """)

        # Create index if not exists
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_historical_symbol_ts
            ON historical_data(symbol, timestamp)
        """)

        # Insert data
        inserted = 0
        skipped = 0

        for _, row in df.iterrows():
            try:
                ts = row.get('timestamp', row.name if hasattr(row, 'name') else None)
                if ts is None:
                    continue

                # Convert timestamp to string
                if hasattr(ts, 'isoformat'):
                    ts_str = ts.isoformat()
                else:
                    ts_str = str(ts)

                cursor.execute("""
                    INSERT OR REPLACE INTO historical_data
                    (symbol, timestamp, open_price, high_price, low_price, close_price, volume, source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, 'datamanager')
                """, (
                    symbol.upper(),
                    ts_str,
                    float(row.get('Open', 0)),
                    float(row.get('High', 0)),
                    float(row.get('Low', 0)),
                    float(row.get('Close', 0)),
                    int(row.get('Volume', 0))
                ))
                inserted += 1
            except sqlite3.IntegrityError:
                skipped += 1
            except Exception as e:
                print(f"      [WARN] Error inserting row: {e}")

        conn.commit()
        conn.close()

        print(f"      [OK] {symbol}: {inserted} records synced, {skipped} duplicates skipped")
        return inserted

    except Exception as e:
        print(f"      [ERROR] Failed to sync {symbol}: {e}")
        return 0


def main():
    parser = argparse.ArgumentParser(description='Sync data from remote Data Manager')
    parser.add_argument('--days', type=int, default=7, help='Number of days to sync (default: 7)')
    parser.add_argument('--symbols', nargs='+', help='Symbols to sync (default: from config)')
    parser.add_argument('--db', type=str, default='data/db/historical.db', help='Database path')
    parser.add_argument('--url', type=str, help='Override Data Manager URL')
    parser.add_argument('--key', type=str, help='Override API key')
    args = parser.parse_args()

    print("=" * 60)
    print("DATA MANAGER SYNC")
    print("=" * 60)

    # Load config
    config = load_config()
    dm_config = config.get('data_manager', {})

    # Get connection params (with server_config.json fallback)
    base_url = args.url or dm_config.get('base_url', '')
    if not base_url:
        try:
            from config_loader import get_data_manager_url
            base_url = get_data_manager_url()
            print(f"[INFO] Using server_config.json fallback: {base_url}")
        except ImportError:
            pass
    api_key = args.key or dm_config.get('api_key', '')

    if not base_url:
        print("[ERROR] No Data Manager URL configured")
        print("[HINT] Set data_manager.base_url in config.json, use --url, or configure server_config.json")
        sys.exit(1)

    if not api_key:
        print("[ERROR] No API key configured")
        print("[HINT] Set data_manager.api_key in config.json or use --key")
        sys.exit(1)

    # Get symbols
    if args.symbols:
        symbols = args.symbols
    else:
        # Get from config
        trading_symbol = config.get('trading', {}).get('symbol', 'SPY')
        fetch_symbols = config.get('data_fetching', {}).get('symbols', ['SPY', 'QQQ', 'VIX'])
        symbols = list(set([trading_symbol] + fetch_symbols))

    # Normalize symbols
    symbols = [s.replace('^', '') for s in symbols]

    print(f"\n[*] Configuration:")
    print(f"    URL: {base_url}")
    print(f"    Symbols: {', '.join(symbols)}")
    print(f"    Days: {args.days}")
    print(f"    Database: {args.db}")

    # Create data source
    try:
        dm_source = DataManagerDataSource(
            base_url=base_url,
            api_key=api_key,
            timeout=60
        )

        if not dm_source.health_check():
            print(f"\n[ERROR] Cannot connect to Data Manager at {base_url}")
            sys.exit(1)

        print(f"\n[OK] Connected to Data Manager")

    except Exception as e:
        print(f"\n[ERROR] Failed to connect: {e}")
        sys.exit(1)

    # Ensure database directory exists
    os.makedirs(os.path.dirname(args.db), exist_ok=True)

    # Sync each symbol
    print(f"\n[*] Syncing {len(symbols)} symbols...")
    total_records = 0

    for symbol in symbols:
        records = sync_symbol(dm_source, symbol, args.days, args.db)
        total_records += records

    print(f"\n[DONE] Synced {total_records} total records to {args.db}")

    # Show stats
    try:
        conn = sqlite3.connect(args.db)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT symbol, COUNT(*), MIN(timestamp), MAX(timestamp)
            FROM historical_data
            GROUP BY symbol
            ORDER BY symbol
        """)

        print("\n[*] Database Statistics:")
        for row in cursor.fetchall():
            symbol, count, min_ts, max_ts = row
            print(f"    {symbol}: {count:,} records ({min_ts[:10]} to {max_ts[:10]})")

        conn.close()
    except Exception as e:
        print(f"    [WARN] Could not get stats: {e}")


if __name__ == '__main__':
    main()
