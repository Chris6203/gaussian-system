#!/usr/bin/env python3
"""
Copy VIX data from other database locations into the main historical.db
"""
import sqlite3
from pathlib import Path
from datetime import datetime

TARGET_DB = Path('E:/gaussian/output3/data/db/historical.db')
ROOT = Path('E:/gaussian')

# All potential database locations to check
DB_LOCATIONS = [
    ROOT / 'data' / 'db' / 'historical.db',
    ROOT / 'data' / 'historical.db',
    ROOT / 'output' / 'data' / 'db' / 'historical.db',
    ROOT / 'output2' / 'data' / 'db' / 'historical.db',
    ROOT / 'output4' / 'data' / 'db' / 'historical.db',
    ROOT / 'historical.db',
    ROOT / 'market_data.db',
]

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def get_vix_data_from_db(db_path):
    """Extract VIX data from a database."""
    if not db_path.exists():
        return []
    
    try:
        conn = sqlite3.connect(str(db_path))
        c = conn.cursor()
        
        # Check if table exists
        c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='historical_data'")
        if not c.fetchone():
            conn.close()
            return []
        
        # Get VIX data
        c.execute("""
            SELECT symbol, timestamp, open_price, high_price, low_price, close_price, volume
            FROM historical_data
            WHERE symbol LIKE '%VIX%'
        """)
        data = c.fetchall()
        conn.close()
        return data
    except Exception as e:
        log(f"  Error reading {db_path}: {e}")
        return []

def main():
    print("=" * 70)
    print("COPY VIX DATA FROM OTHER DATABASES")
    print("=" * 70)
    
    # Check current state of target DB
    if TARGET_DB.exists():
        conn = sqlite3.connect(str(TARGET_DB))
        c = conn.cursor()
        c.execute("SELECT MIN(timestamp), MAX(timestamp), COUNT(*) FROM historical_data WHERE symbol LIKE '%VIX%'")
        result = c.fetchone()
        log(f"Current VIX in target: {result[0]} to {result[1]} ({result[2]:,} records)")
        conn.close()
    else:
        log(f"Target DB not found: {TARGET_DB}")
        return
    
    total_copied = 0
    
    # Check each potential source database
    for db_path in DB_LOCATIONS:
        if db_path == TARGET_DB:
            continue
        
        log(f"\nChecking: {db_path}")
        
        if not db_path.exists():
            log(f"  Not found")
            continue
        
        # Get VIX data
        vix_data = get_vix_data_from_db(db_path)
        if not vix_data:
            log(f"  No VIX data found")
            continue
        
        log(f"  Found {len(vix_data):,} VIX records")
        
        # Copy to target
        conn = sqlite3.connect(str(TARGET_DB))
        c = conn.cursor()
        
        copied = 0
        for row in vix_data:
            try:
                c.execute("""
                    INSERT OR IGNORE INTO historical_data 
                    (symbol, timestamp, open_price, high_price, low_price, close_price, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, row)
                if c.rowcount > 0:
                    copied += 1
            except Exception:
                pass
        
        conn.commit()
        conn.close()
        
        log(f"  Copied {copied:,} new records")
        total_copied += copied
    
    # Also scan for any .db files in the entire tree
    log("\n" + "-" * 50)
    log("Scanning for additional .db files...")
    
    for db_file in ROOT.rglob('*.db'):
        if db_file == TARGET_DB:
            continue
        if db_file in DB_LOCATIONS:
            continue
        
        log(f"\nFound: {db_file}")
        vix_data = get_vix_data_from_db(db_file)
        
        if vix_data:
            log(f"  Found {len(vix_data):,} VIX records")
            
            conn = sqlite3.connect(str(TARGET_DB))
            c = conn.cursor()
            
            copied = 0
            for row in vix_data:
                try:
                    c.execute("""
                        INSERT OR IGNORE INTO historical_data 
                        (symbol, timestamp, open_price, high_price, low_price, close_price, volume)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, row)
                    if c.rowcount > 0:
                        copied += 1
                except Exception:
                    pass
            
            conn.commit()
            conn.close()
            
            log(f"  Copied {copied:,} new records")
            total_copied += copied
    
    # Final report
    print("\n" + "=" * 70)
    log(f"TOTAL: Copied {total_copied:,} new VIX records")
    
    # Check final state
    conn = sqlite3.connect(str(TARGET_DB))
    c = conn.cursor()
    c.execute("SELECT MIN(timestamp), MAX(timestamp), COUNT(*) FROM historical_data WHERE symbol LIKE '%VIX%'")
    result = c.fetchone()
    log(f"Final VIX data: {result[0]} to {result[1]} ({result[2]:,} records)")
    conn.close()
    
    print("=" * 70)

if __name__ == "__main__":
    main()


