#!/usr/bin/env python3
"""
Data Verification Script
========================
Checks all databases and data sources in the output3 folder.
Run this to diagnose data issues.
"""

import sqlite3
import os
from pathlib import Path
from datetime import datetime

def main():
    print("=" * 70)
    print("DATA VERIFICATION REPORT")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Working Directory: {os.getcwd()}")
    print("=" * 70)
    
    # Database paths to check
    db_paths = [
        'data/db/historical.db',
        'data/paper_trading.db',
        'data/unified_options_bot.db',
        'data/learning_trades.db',
        'data/market_data_cache.db',
        'data/shadow_trading.db',
    ]
    
    print("\n📁 DATABASE FILES:")
    print("-" * 70)
    
    for db_path in db_paths:
        path = Path(db_path)
        if path.exists():
            size = path.stat().st_size
            print(f"  ✅ {db_path}")
            print(f"     Size: {size:,} bytes ({size/1024/1024:.2f} MB)")
            
            # Try to get table info
            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall()]
                print(f"     Tables: {', '.join(tables) if tables else 'None'}")
                
                # For historical.db, check data range
                if 'historical_data' in tables:
                    cursor.execute("""
                        SELECT symbol, MIN(timestamp), MAX(timestamp), COUNT(*) 
                        FROM historical_data 
                        WHERE symbol IN ('SPY', 'VIX', 'QQQ')
                        GROUP BY symbol
                    """)
                    print(f"     Data ranges:")
                    for row in cursor.fetchall():
                        print(f"       {row[0]}: {row[1][:19] if row[1] else 'N/A'} to {row[2][:19] if row[2] else 'N/A'} ({row[3]:,} records)")
                
                conn.close()
            except Exception as e:
                print(f"     Error reading: {e}")
        else:
            print(f"  ❌ {db_path} - NOT FOUND")
    
    # Check for any .db files we might have missed
    print("\n🔍 SEARCHING FOR ALL .db FILES:")
    print("-" * 70)
    
    found_dbs = list(Path('.').rglob('*.db'))
    if found_dbs:
        for db in found_dbs:
            print(f"  📄 {db}")
    else:
        print("  No .db files found in directory tree")
    
    # Check data directory structure
    print("\n📂 DATA DIRECTORY STRUCTURE:")
    print("-" * 70)
    
    data_dir = Path('data')
    if data_dir.exists():
        for item in sorted(data_dir.rglob('*')):
            if item.is_file():
                rel_path = item.relative_to(data_dir)
                size = item.stat().st_size
                print(f"  {rel_path} ({size:,} bytes)")
    else:
        print("  data/ directory does not exist!")
    
    # Summary
    print("\n📊 SUMMARY:")
    print("-" * 70)
    
    historical_db = Path('data/db/historical.db')
    if historical_db.exists():
        print("  ✅ Historical database exists - dashboard should work")
    else:
        print("  ⚠️  Historical database MISSING!")
        print("     The dashboard chart needs this file.")
        print("     Run: python run_simulation.py  (to fetch and populate data)")
        print("     Or:  python fetch_polygon_historical.py  (for 6 months of data)")
    
    print("\n" + "=" * 70)

if __name__ == '__main__':
    main()


