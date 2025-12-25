#!/usr/bin/env python3
"""Check historical data date range - writes to data_report.txt"""
import sqlite3
from pathlib import Path

output_lines = []

def out(msg):
    print(msg)
    output_lines.append(msg)

db_path = Path('data/db/historical.db')
if not db_path.exists():
    out(f"Database not found: {db_path}")
else:
    conn = sqlite3.connect(str(db_path))
    c = conn.cursor()
    
    out("=" * 60)
    out("HISTORICAL DATA REPORT")
    out("=" * 60)
    
    # Get all symbols and their date ranges
    out("\nDATA BY SYMBOL:")
    out(f"{'Symbol':<15} {'Start Date':<12} {'End Date':<12} {'Records':>12}")
    out("-" * 55)
    
    c.execute('''
        SELECT symbol, MIN(timestamp), MAX(timestamp), COUNT(*) 
        FROM historical_data 
        GROUP BY symbol 
        ORDER BY symbol
    ''')
    
    for row in c.fetchall():
        start = row[1][:10] if row[1] else 'N/A'
        end = row[2][:10] if row[2] else 'N/A'
        out(f"{row[0]:<15} {start:<12} {end:<12} {row[3]:>12,}")
    
    # Specific VIX check
    out("\n" + "=" * 60)
    out("VIX DATA CHECK:")
    c.execute("SELECT MIN(timestamp), MAX(timestamp), COUNT(*) FROM historical_data WHERE symbol IN ('VIX', '^VIX')")
    r = c.fetchone()
    out(f"VIX range: {r[0]} to {r[1]}")
    out(f"VIX records: {r[2]:,}")
    
    # SPY check
    out("\nSPY DATA CHECK:")
    c.execute("SELECT MIN(timestamp), MAX(timestamp), COUNT(*) FROM historical_data WHERE symbol='SPY'")
    r = c.fetchone()
    out(f"SPY range: {r[0]} to {r[1]}")
    out(f"SPY records: {r[2]:,}")
    
    # QQQ check
    out("\nQQQ DATA CHECK:")
    c.execute("SELECT MIN(timestamp), MAX(timestamp), COUNT(*) FROM historical_data WHERE symbol='QQQ'")
    r = c.fetchone()
    out(f"QQQ range: {r[0]} to {r[1]}")
    out(f"QQQ records: {r[2]:,}")
    
    conn.close()
    
    out("\n" + "=" * 60)

# Save to file
with open('data_report.txt', 'w') as f:
    f.write('\n'.join(output_lines))
print(f"\nReport saved to data_report.txt")



