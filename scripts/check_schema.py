#!/usr/bin/env python3
"""Check database schema."""
import sqlite3

for db_path in ['data/paper_trading.db', 'models/paper_trading.db', 'data/db/trading.db']:
    try:
        print(f"\n{'='*60}")
        print(f"Database: {db_path}")
        print('='*60)
        
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        
        # Get tables
        c.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [r[0] for r in c.fetchall()]
        print(f"Tables: {tables}")
        
        # Show schema for each table
        for table in tables:
            c.execute(f"PRAGMA table_info({table})")
            columns = [(r[1], r[2]) for r in c.fetchall()]
            print(f"\n  {table}: {columns}")
            
            # Show sample data
            c.execute(f"SELECT * FROM {table} LIMIT 3")
            rows = c.fetchall()
            if rows:
                print(f"  Sample rows: {len(rows)}")
                for row in rows:
                    print(f"    {row}")
            
            # Count
            c.execute(f"SELECT COUNT(*) FROM {table}")
            count = c.fetchone()[0]
            print(f"  Total rows: {count}")
        
        conn.close()
    except Exception as e:
        print(f"{db_path}: {e}")

