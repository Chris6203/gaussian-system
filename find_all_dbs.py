#!/usr/bin/env python3
"""Find all databases in e:\gaussian and check VIX data in each."""
import sqlite3
from pathlib import Path
from datetime import datetime

# Places to check for databases
root = Path('E:/gaussian')
db_paths_to_check = [
    root / 'data' / 'db' / 'historical.db',
    root / 'data' / 'historical.db',
    root / 'output' / 'data' / 'db' / 'historical.db',
    root / 'output2' / 'data' / 'db' / 'historical.db',
    root / 'output3' / 'data' / 'db' / 'historical.db',
    root / 'output4' / 'data' / 'db' / 'historical.db',
    root / 'historical.db',
    root / 'market_data.db',
    root / 'data' / 'market_data.db',
]

# Also scan for any .db files
print("=" * 70)
print("SCANNING FOR ALL .DB FILES IN E:\\gaussian")
print("=" * 70)

found_dbs = []

# Scan entire directory tree
for db_file in root.rglob('*.db'):
    found_dbs.append(db_file)
    print(f"Found: {db_file} ({db_file.stat().st_size:,} bytes)")

# Also check specific paths even if not found by rglob
for db_path in db_paths_to_check:
    if db_path.exists() and db_path not in found_dbs:
        found_dbs.append(db_path)
        print(f"Found (specific): {db_path} ({db_path.stat().st_size:,} bytes)")

print(f"\nTotal databases found: {len(found_dbs)}")

# Check VIX data in each database
print("\n" + "=" * 70)
print("VIX DATA IN EACH DATABASE")
print("=" * 70)

for db_path in found_dbs:
    print(f"\n--- {db_path} ---")
    try:
        conn = sqlite3.connect(str(db_path))
        c = conn.cursor()
        
        # Check for historical_data table
        c.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in c.fetchall()]
        print(f"Tables: {tables}")
        
        if 'historical_data' in tables:
            # Check VIX data
            c.execute("""
                SELECT symbol, MIN(timestamp), MAX(timestamp), COUNT(*) 
                FROM historical_data 
                WHERE symbol LIKE '%VIX%'
                GROUP BY symbol
            """)
            for row in c.fetchall():
                print(f"  {row[0]}: {row[1]} to {row[2]} ({row[3]:,} records)")
        
        # Also check polygon_intraday_1m table if exists
        if 'polygon_intraday_1m' in tables:
            c.execute("""
                SELECT symbol, MIN(timestamp), MAX(timestamp), COUNT(*) 
                FROM polygon_intraday_1m 
                WHERE symbol LIKE '%VIX%'
                GROUP BY symbol
            """)
            results = c.fetchall()
            if results:
                print(f"  (polygon_intraday_1m table):")
                for row in results:
                    print(f"    {row[0]}: {row[1]} to {row[2]} ({row[3]:,} records)")
        
        conn.close()
    except Exception as e:
        print(f"  Error: {e}")

# Save results
with open('db_scan_results.txt', 'w') as f:
    f.write(f"Scan completed at {datetime.now()}\n")
    f.write(f"Found {len(found_dbs)} databases\n")
    for db in found_dbs:
        f.write(f"  {db}\n")

print("\n" + "=" * 70)
print("Results saved to db_scan_results.txt")


