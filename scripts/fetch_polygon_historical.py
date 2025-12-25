#!/usr/bin/env python3
"""
Fetch 6 months of 1-minute historical data from Polygon.io
Requires Polygon Stocks Starter plan ($29/mo)

This script will:
1. Fetch 1-minute data for all symbols in config.json
2. Go back 6 months (or as far as available)
3. Save to historical.db database
4. Handle rate limiting and pagination
"""

import os
import sys
import json
import time
import sqlite3
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# Configuration
MONTHS_BACK = 6
DB_PATH = "data/db/historical.db"
CONFIG_PATH = "config.json"

# Polygon Starter plan: Much higher rate limits than free tier
# Starter allows ~unlimited calls (reasonable use)
REQUESTS_PER_MINUTE = 100  # Conservative limit for Starter
BATCH_DAYS = 7  # Fetch 7 days at a time to avoid hitting result limits

def log(msg):
    """Print with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}")

def load_config():
    """Load API key and symbols from config"""
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
    
    api_key = config.get('credentials', {}).get('polygon', {}).get('api_key')
    symbols = config.get('data_fetching', {}).get('symbols', [])
    
    # Filter out VIX and crypto (VIX needs Indices plan, crypto separate)
    stock_symbols = []
    skipped = []
    for s in symbols:
        if s.startswith('^') or 'VIX' in s.upper():
            skipped.append(f"{s} (index - need Indices plan)")
        elif '-USD' in s.upper():
            skipped.append(f"{s} (crypto - need Crypto plan)")
        else:
            stock_symbols.append(s)
    
    return api_key, stock_symbols, skipped

def init_database():
    """Initialize the database tables for historical data"""
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create table for 1-minute historical data (Polygon-specific with extra fields)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS polygon_intraday_1m (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            open_price REAL NOT NULL,
            high_price REAL NOT NULL,
            low_price REAL NOT NULL,
            close_price REAL NOT NULL,
            volume INTEGER NOT NULL,
            vwap REAL,
            transactions INTEGER,
            source TEXT DEFAULT 'polygon',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(symbol, timestamp)
        )
    ''')
    
    # Create the historical_data table that dashboard and training expect
    # This is the STANDARD table format used throughout the system
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
    
    # Create indexes for faster queries
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_polygon_intraday_symbol_time 
        ON polygon_intraday_1m(symbol, timestamp)
    ''')
    
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_historical_data_symbol_time 
        ON historical_data(symbol, timestamp)
    ''')
    
    conn.commit()
    conn.close()
    log("Database initialized (both polygon_intraday_1m and historical_data tables)")

def fetch_polygon_data(api_key, symbol, start_date, end_date):
    """Fetch 1-minute data from Polygon for a date range"""
    base_url = "https://api.polygon.io"
    
    from_str = start_date.strftime('%Y-%m-%d')
    to_str = end_date.strftime('%Y-%m-%d')
    
    endpoint = f"/v2/aggs/ticker/{symbol}/range/1/minute/{from_str}/{to_str}"
    
    params = {
        'apiKey': api_key,
        'adjusted': 'true',
        'sort': 'asc',
        'limit': 50000  # Max per request
    }
    
    try:
        response = requests.get(f"{base_url}{endpoint}", params=params, timeout=30)
        
        if response.status_code == 429:
            log(f"  Rate limited - waiting 60s...")
            time.sleep(60)
            return fetch_polygon_data(api_key, symbol, start_date, end_date)
        
        if response.status_code == 403:
            log(f"  [ERROR] Access denied for {symbol} - check your plan")
            return []
        
        response.raise_for_status()
        data = response.json()
        
        if 'results' in data and data['results']:
            return data['results']
        else:
            return []
            
    except requests.exceptions.RequestException as e:
        log(f"  [ERROR] Request failed: {e}")
        return []

def save_to_database(symbol, results):
    """Save results to database - BOTH polygon table AND historical_data table for training"""
    if not results:
        return 0
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Also create the historical_data table (used by training)
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
    
    saved = 0
    for r in results:
        try:
            # Convert timestamp from milliseconds
            ts_full = datetime.fromtimestamp(r['t'] / 1000).isoformat()
            # Normalize to 19 chars (no timezone) for consistency
            ts = ts_full[:19] if len(ts_full) > 19 else ts_full
            
            # Save to polygon_intraday_1m table
            cursor.execute('''
                INSERT OR REPLACE INTO polygon_intraday_1m 
                (symbol, timestamp, open_price, high_price, low_price, close_price, 
                 volume, vwap, transactions, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'polygon')
            ''', (
                symbol,
                ts,
                r.get('o', 0),
                r.get('h', 0),
                r.get('l', 0),
                r.get('c', 0),
                r.get('v', 0),
                r.get('vw'),
                r.get('n')
            ))
            
            # ALSO save to historical_data table (used by training!)
            cursor.execute('''
                INSERT OR REPLACE INTO historical_data 
                (symbol, timestamp, open_price, high_price, low_price, close_price, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                symbol,
                ts,
                r.get('o', 0),
                r.get('h', 0),
                r.get('l', 0),
                r.get('c', 0),
                r.get('v', 0)
            ))
            
            saved += 1
        except Exception as e:
            pass  # Skip duplicates
    
    conn.commit()
    conn.close()
    return saved

def get_existing_data_range(symbol):
    """Check what data we already have for a symbol"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT MIN(timestamp), MAX(timestamp), COUNT(*)
        FROM polygon_intraday_1m
        WHERE symbol = ?
    ''', (symbol,))
    
    result = cursor.fetchone()
    conn.close()
    
    if result and result[0]:
        return result[0], result[1], result[2]
    return None, None, 0

def fetch_symbol_history(api_key, symbol, months_back):
    """Fetch full history for a symbol"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=months_back * 30)
    
    # Check existing data
    existing_start, existing_end, existing_count = get_existing_data_range(symbol)
    if existing_count > 0:
        log(f"  Existing data: {existing_count:,} records ({existing_start[:10]} to {existing_end[:10]})")
    
    total_saved = 0
    current_start = start_date
    
    while current_start < end_date:
        current_end = min(current_start + timedelta(days=BATCH_DAYS), end_date)
        
        results = fetch_polygon_data(api_key, symbol, current_start, current_end)
        
        if results:
            saved = save_to_database(symbol, results)
            total_saved += saved
            log(f"  {current_start.strftime('%Y-%m-%d')} to {current_end.strftime('%Y-%m-%d')}: {len(results):,} bars fetched, {saved:,} saved")
        else:
            log(f"  {current_start.strftime('%Y-%m-%d')} to {current_end.strftime('%Y-%m-%d')}: No data")
        
        current_start = current_end
        
        # Small delay between requests
        time.sleep(0.5)
    
    return total_saved

def main():
    print("=" * 70)
    print("POLYGON HISTORICAL DATA FETCHER")
    print("Fetching 6 months of 1-minute data for all symbols")
    print("=" * 70)
    print()
    
    # Load config
    api_key, symbols, skipped = load_config()
    
    if not api_key:
        log("[ERROR] No Polygon API key found in config.json!")
        log("Add your API key to: credentials.polygon.api_key")
        return
    
    log(f"API Key: {api_key[:10]}...{api_key[-4:]}")
    log(f"Symbols to fetch: {len(symbols)}")
    log(f"Months of history: {MONTHS_BACK}")
    log(f"Database: {DB_PATH}")
    
    if skipped:
        print()
        log("Skipped symbols (need different plans):")
        for s in skipped:
            log(f"  - {s}")
    
    print()
    
    # Initialize database
    init_database()
    
    # Test API connection first (use historical endpoint, not quote)
    log("Testing Polygon API connection...")
    test_url = f"https://api.polygon.io/v2/aggs/ticker/AAPL/range/1/minute/2024-12-01/2024-12-02?apiKey={api_key}&limit=5"
    try:
        resp = requests.get(test_url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if data.get('resultsCount', 0) > 0:
                log(f"[OK] API connection successful! Got {data['resultsCount']} test records")
            else:
                log("[OK] API connected (no data in test range)")
        elif resp.status_code == 403:
            log("[ERROR] API key invalid or plan doesn't include historical data")
            log(f"Response: {resp.text[:200]}")
            return
        else:
            log(f"[WARNING] Unexpected response: {resp.status_code}")
            log(f"Response: {resp.text[:200]}")
    except Exception as e:
        log(f"[ERROR] Connection failed: {e}")
        return
    
    print()
    print("=" * 70)
    log("Starting historical data fetch...")
    print("=" * 70)
    print()
    
    # Fetch data for each symbol
    results = {}
    start_time = time.time()
    
    for i, symbol in enumerate(symbols, 1):
        log(f"[{i}/{len(symbols)}] Fetching {symbol}...")
        
        try:
            saved = fetch_symbol_history(api_key, symbol, MONTHS_BACK)
            results[symbol] = {'success': True, 'records': saved}
            log(f"  [DONE] {symbol}: {saved:,} total records saved")
        except Exception as e:
            results[symbol] = {'success': False, 'error': str(e)}
            log(f"  [ERROR] {symbol}: {e}")
        
        print()
    
    # Summary
    elapsed = time.time() - start_time
    
    print("=" * 70)
    log("FETCH COMPLETE!")
    print("=" * 70)
    print()
    
    successful = sum(1 for r in results.values() if r.get('success'))
    total_records = sum(r.get('records', 0) for r in results.values() if r.get('success'))
    
    log(f"Symbols processed: {successful}/{len(symbols)}")
    log(f"Total records saved: {total_records:,}")
    log(f"Time elapsed: {elapsed/60:.1f} minutes")
    log(f"Database: {DB_PATH}")
    
    print()
    log("Results by symbol:")
    for symbol, result in results.items():
        if result.get('success'):
            log(f"  [OK] {symbol}: {result.get('records', 0):,} records")
        else:
            log(f"  [FAIL] {symbol}: {result.get('error', 'Unknown error')}")
    
    print()
    
    # Check database size
    if os.path.exists(DB_PATH):
        size_mb = os.path.getsize(DB_PATH) / (1024 * 1024)
        log(f"Database size: {size_mb:.1f} MB")

if __name__ == "__main__":
    main()

