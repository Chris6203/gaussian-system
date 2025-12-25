#!/usr/bin/env python3
"""
Fetch longer VIX history from multiple sources.

Sources tried (in order):
1. FMP (Financial Modeling Prep) - Up to 6 months of 1-min data
2. Yahoo Finance - Up to 60 days of 1-min data
3. Daily data expanded to 1-minute for gap filling

This script fetches as much 1-min VIX data as possible.
"""

import os
import sys
import json
import sqlite3
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

DB_PATH = "data/db/historical.db"
CONFIG_PATH = "config.json"

def log(msg):
    """Print with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}")

def init_database():
    """Ensure database and table exist"""
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
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
    
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_historical_symbol_time 
        ON historical_data(symbol, timestamp)
    ''')
    
    conn.commit()
    conn.close()
    log("Database initialized")

def get_existing_vix_range():
    """Check what VIX data we already have"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT MIN(timestamp), MAX(timestamp), COUNT(*)
        FROM historical_data
        WHERE symbol IN ('VIX', '^VIX')
    ''')
    
    result = cursor.fetchone()
    conn.close()
    
    return result

def fetch_vix_from_yahoo(period='max', interval='1m'):
    """Fetch VIX data from Yahoo Finance"""
    try:
        import yfinance as yf
    except ImportError:
        log("Installing yfinance...")
        os.system(f"{sys.executable} -m pip install yfinance")
        import yfinance as yf
    
    log(f"Fetching VIX data from Yahoo Finance (period={period}, interval={interval})...")
    
    ticker = yf.Ticker("^VIX")
    
    try:
        data = ticker.history(period=period, interval=interval)
        if data is not None and not data.empty:
            log(f"  Got {len(data)} records from {data.index.min()} to {data.index.max()}")
            return data
        else:
            log(f"  No data returned for period={period}, interval={interval}")
            return None
    except Exception as e:
        log(f"  Error fetching: {e}")
        return None

def save_vix_data(data, symbol='VIX'):
    """Save VIX data to database"""
    if data is None or data.empty:
        return 0
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    inserted = 0
    for idx, row in data.iterrows():
        # Normalize timestamp (remove timezone info)
        ts_str = str(idx)[:19]
        
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO historical_data 
                (symbol, timestamp, open_price, high_price, low_price, close_price, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                symbol,
                ts_str,
                row.get('Open'),
                row.get('High'),
                row.get('Low'),
                row.get('Close'),
                row.get('Volume', 0)
            ))
            inserted += 1
        except Exception as e:
            pass  # Skip duplicates silently
    
    conn.commit()
    conn.close()
    
    return inserted

def load_fmp_api_key():
    """Load FMP API key from config"""
    try:
        with open(CONFIG_PATH, 'r') as f:
            config = json.load(f)
        return config.get('credentials', {}).get('fmp', {}).get('api_key')
    except Exception:
        return None

def fetch_vix_from_fmp(api_key, days_back=180):
    """Fetch VIX data from Financial Modeling Prep API"""
    if not api_key:
        log("  FMP API key not found")
        return None
    
    log(f"Fetching VIX data from FMP (last {days_back} days)...")
    
    # FMP uses ^VIX or VIX for volatility index
    # Try their intraday historical endpoint
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    from_str = start_date.strftime('%Y-%m-%d')
    to_str = end_date.strftime('%Y-%m-%d')
    
    # FMP's 1-minute historical endpoint
    url = f"https://financialmodelingprep.com/api/v3/historical-chart/1min/%5EVIX?from={from_str}&to={to_str}&apikey={api_key}"
    
    try:
        response = requests.get(url, timeout=60)
        if response.status_code == 200:
            data = response.json()
            if data and isinstance(data, list) and len(data) > 0:
                df = pd.DataFrame(data)
                log(f"  Got {len(df)} records from FMP")
                return df
            else:
                log(f"  FMP returned empty data for VIX")
        else:
            log(f"  FMP request failed: {response.status_code}")
    except Exception as e:
        log(f"  FMP error: {e}")
    
    return None

def save_fmp_vix_data(data):
    """Save FMP VIX data to database"""
    if data is None or data.empty:
        return 0
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    inserted = 0
    for _, row in data.iterrows():
        # FMP returns data with 'date', 'open', 'high', 'low', 'close', 'volume'
        ts_str = str(row.get('date', ''))[:19]
        
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO historical_data 
                (symbol, timestamp, open_price, high_price, low_price, close_price, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                'VIX',
                ts_str,
                row.get('open'),
                row.get('high'),
                row.get('low'),
                row.get('close'),
                row.get('volume', 0)
            ))
            inserted += 1
        except Exception:
            pass
    
    conn.commit()
    conn.close()
    return inserted

def main():
    print("=" * 60)
    print("VIX HISTORY FETCHER")
    print("=" * 60)
    print()
    
    init_database()
    
    # Check existing data
    existing = get_existing_vix_range()
    log(f"Existing VIX data: {existing[0]} to {existing[1]} ({existing[2]:,} records)")
    
    total_inserted = 0
    
    # Try FMP first (best source for long 1-min history)
    log("\n--- Trying FMP for VIX data ---")
    fmp_key = load_fmp_api_key()
    if fmp_key:
        fmp_data = fetch_vix_from_fmp(fmp_key, days_back=180)
        if fmp_data is not None and not fmp_data.empty:
            inserted = save_fmp_vix_data(fmp_data)
            total_inserted += inserted
            log(f"  Saved {inserted} records from FMP")
    else:
        log("  FMP API key not configured")
    
    # Try Yahoo Finance as backup
    log("\n--- Fetching 1-minute VIX data from Yahoo ---")
    
    for period in ['60d', '30d', '7d', '5d']:
        data = fetch_vix_from_yahoo(period=period, interval='1m')
        if data is not None and not data.empty:
            inserted = save_vix_data(data, 'VIX')
            total_inserted += inserted
            log(f"  Saved {inserted} records from period={period}")
            break  # Got data, no need to try shorter periods
    
    # Also try 5-minute data for potentially longer history
    log("\n--- Fetching 5-minute VIX data ---")
    for period in ['60d', '30d']:
        data = fetch_vix_from_yahoo(period=period, interval='5m')
        if data is not None and not data.empty:
            # Resample 5m to 1m by forward-filling (rough approximation)
            # Actually just save the 5m data points as is
            inserted = save_vix_data(data, 'VIX')
            total_inserted += inserted
            log(f"  Saved {inserted} 5-min records from period={period}")
            break
    
    # Get longer daily data and create 1-minute entries for market hours
    log("\n--- Fetching daily VIX data for gap filling ---")
    daily_data = fetch_vix_from_yahoo(period='6mo', interval='1d')
    if daily_data is not None and not daily_data.empty:
        log(f"  Got daily data: {daily_data.index.min()} to {daily_data.index.max()}")
        
        # Convert daily data to 1-minute data by replicating the close price
        # for each minute of market hours (9:30 AM - 4:00 PM ET)
        log("  Creating 1-minute records from daily data...")
        
        expanded_records = []
        for date_idx, row in daily_data.iterrows():
            date = date_idx.date()
            close_price = row['Close']
            
            # Skip weekends
            if date_idx.weekday() >= 5:
                continue
            
            # Create entries for each minute from 9:30 to 16:00
            for hour in range(9, 16):
                start_min = 30 if hour == 9 else 0
                end_min = 60 if hour < 15 else 1  # End at 16:00
                for minute in range(start_min, end_min):
                    ts = f"{date} {hour:02d}:{minute:02d}:00"
                    expanded_records.append({
                        'timestamp': ts,
                        'open': close_price,
                        'high': close_price,
                        'low': close_price,
                        'close': close_price,
                        'volume': 0
                    })
        
        # Save expanded records
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        daily_inserted = 0
        for rec in expanded_records:
            try:
                cursor.execute('''
                    INSERT OR IGNORE INTO historical_data 
                    (symbol, timestamp, open_price, high_price, low_price, close_price, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', ('VIX', rec['timestamp'], rec['open'], rec['high'], rec['low'], rec['close'], rec['volume']))
                if cursor.rowcount > 0:
                    daily_inserted += 1
            except Exception:
                pass
        
        conn.commit()
        conn.close()
        
        log(f"  Saved {daily_inserted} records from daily data expansion")
        total_inserted += daily_inserted
    
    # Final report
    print()
    print("=" * 60)
    log(f"TOTAL: Inserted/updated {total_inserted:,} VIX records")
    
    # Check final state
    final = get_existing_vix_range()
    log(f"Final VIX data: {final[0]} to {final[1]} ({final[2]:,} records)")
    print("=" * 60)

if __name__ == "__main__":
    main()



