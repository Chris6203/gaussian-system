#!/usr/bin/env python3
"""
Fetch and save training data for the bot
"""
import warnings
warnings.filterwarnings('ignore')

import json
import sqlite3
import pandas as pd
from datetime import datetime

print("="*80)
print("FETCHING TRAINING DATA")
print("="*80)
print()

# Load config to get symbol
try:
    with open('config.json', 'r') as f:
        config = json.load(f)
    symbol = config.get('trading', {}).get('symbol', 'SPY')
    print(f"[CONFIG] Trading symbol: {symbol}")
except Exception as e:
    print(f"[WARN] Could not load config: {e}")
    symbol = 'SPY'

# Fetch data
print(f"\n[1/3] Fetching {symbol} data from Yahoo Finance...")
from backend.enhanced_data_sources import EnhancedDataSource

ds = EnhancedDataSource()
data = ds.get_data(symbol, period='1d', interval='1m')

if data.empty:
    print(f"[ERROR] No data received for {symbol}")
    print(f"[HINT] Check internet connection or try a different symbol")
    exit(1)

print(f"[OK] Got {len(data)} rows of {symbol} data")
print(f"[OK] Date range: {data.index[0]} to {data.index[-1]}")

# Prepare data for database
print(f"\n[2/3] Preparing data for database...")
data_to_insert = []
for timestamp, row in data.iterrows():
    data_to_insert.append((
        symbol,
        timestamp.strftime('%Y-%m-%d %H:%M:%S'),
        float(row.get('Open', row.get('open', 0))),
        float(row.get('High', row.get('high', 0))),
        float(row.get('Low', row.get('low', 0))),
        float(row.get('Close', row.get('close', 0))),
        int(row.get('Volume', row.get('volume', 0)))
    ))

print(f"[OK] Prepared {len(data_to_insert)} records")

# Save to database
print(f"\n[3/3] Saving to database...")
import os
os.makedirs('data/db', exist_ok=True)

conn = sqlite3.connect('data/db/historical.db')
cursor = conn.cursor()

# Create table if not exists
cursor.execute('''
    CREATE TABLE IF NOT EXISTS historical_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL,
        timestamp DATETIME NOT NULL,
        open_price REAL,
        high_price REAL,
        low_price REAL,
        close_price REAL,
        volume INTEGER,
        UNIQUE(symbol, timestamp)
    )
''')

# Clear existing data for this symbol
cursor.execute('DELETE FROM historical_data WHERE symbol = ?', (symbol,))
print(f"[OK] Cleared old {symbol} data")

# Insert new data
cursor.executemany('''
    INSERT OR REPLACE INTO historical_data 
    (symbol, timestamp, open_price, high_price, low_price, close_price, volume)
    VALUES (?, ?, ?, ?, ?, ?, ?)
''', data_to_insert)

conn.commit()
conn.close()

print(f"[OK] Saved {len(data_to_insert)} records to database")

# Verify
conn = sqlite3.connect('data/db/historical.db')
count = conn.execute('SELECT COUNT(*) FROM historical_data WHERE symbol = ?', (symbol,)).fetchone()[0]
conn.close()

print(f"\n[✓] Verification: {count} records in database for {symbol}")
print()
print("="*80)
print("DATA READY FOR TRAINING!")
print("="*80)
print()
print("Next step:")
print("  python train_then_go_live.py")
print()









