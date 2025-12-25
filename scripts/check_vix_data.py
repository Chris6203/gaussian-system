#!/usr/bin/env python3
"""Check VIX data specifically."""
import sqlite3
from pathlib import Path
import sys

db_path = Path('data/db/historical.db')
output_file = Path('vix_data_report.txt')

with open(output_file, 'w') as f:
    if not db_path.exists():
        f.write(f"Database not found: {db_path}\n")
        print(f"Database not found: {db_path}")
        sys.exit(1)
    
    conn = sqlite3.connect(str(db_path))
    c = conn.cursor()
    
    # Check for VIX data with various symbol names
    vix_variants = ['VIX', '^VIX', 'VIXY', '$VIX']
    
    f.write("=== VIX DATA CHECK ===\n\n")
    
    for symbol in vix_variants:
        c.execute('SELECT MIN(timestamp), MAX(timestamp), COUNT(*) FROM historical_data WHERE symbol=?', (symbol,))
        r = c.fetchone()
        if r[2] > 0:
            line = f"{symbol}: {r[0]} to {r[1]} ({r[2]:,} records)\n"
        else:
            line = f"{symbol}: No data\n"
        f.write(line)
        print(line.strip())
    
    f.write("\n=== ALL SYMBOLS DATE RANGES ===\n\n")
    
    # Get all symbols and their date ranges
    c.execute('''
        SELECT symbol, MIN(timestamp), MAX(timestamp), COUNT(*) 
        FROM historical_data 
        GROUP BY symbol 
        ORDER BY symbol
    ''')
    
    for row in c.fetchall():
        start = row[1][:10] if row[1] else 'N/A'
        end = row[2][:10] if row[2] else 'N/A'
        line = f"  {row[0]:<12}: {start} to {end} ({row[3]:>8,} records)\n"
        f.write(line)
        print(line.strip())
    
    conn.close()
    
    f.write(f"\nReport saved to: {output_file}\n")
    print(f"\nReport saved to: {output_file}")



