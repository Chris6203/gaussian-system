#!/usr/bin/env python3
"""
Fetch 6 months of historical 1-minute data for all bot symbols.
Run this script to backfill historical data for training.
"""

import sys
import os
from datetime import datetime
import time

# Change to script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Simple logging (ASCII-safe for Windows)
log_file = open('fetch_6months.log', 'w', encoding='utf-8')

def log(msg):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{timestamp}] {msg}"
    print(line)
    log_file.write(line + '\n')
    log_file.flush()

log("=" * 60)
log("Starting 6-month historical data fetch")
log("=" * 60)
log(f"Working directory: {os.getcwd()}")

# Load config
try:
    from config_loader import load_config
    config = load_config("config.json")
    config_dict = config.config
    log("[OK] Config loaded")
except Exception as e:
    log(f"[ERROR] Could not load config: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Get FMP credentials
fmp_key = config_dict.get('credentials', {}).get('fmp', {}).get('api_key')
if not fmp_key:
    log("[ERROR] FMP API key not found in config.json")
    sys.exit(1)

log(f"[OK] FMP API key: {fmp_key[:10]}...")

# Initialize FMP
try:
    from backend.fmp_data_source import FMPDataSource
    fmp = FMPDataSource(api_key=fmp_key, config=config_dict)
    log("[OK] FMP data source initialized")
except Exception as e:
    log(f"[ERROR] Could not initialize FMP: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Symbols to fetch (all symbols the bot uses)
symbols = [
    # Main ETFs
    'SPY', 'QQQ', 'IWM', 'DIA',
    # Tech stocks
    'NVDA', 'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'PLTR', 'INTC', 'AMD',
    # Sector ETFs
    'XLF', 'XLK', 'XLE', 'XLU', 'XLY', 'XLP',
    # Bond ETFs
    'HYG', 'LQD', 'TLT', 'IEF', 'SHY',
    # Dollar
    'UUP'
]

days_back = 180  # 6 months

log("")
log(f"Fetching {days_back} days of 1-min data for {len(symbols)} symbols")
log(f"Symbols: {', '.join(symbols)}")
log("")

# Fetch data for each symbol
results = {}
total_records = 0
success_count = 0

for i, symbol in enumerate(symbols, 1):
    log(f"[{i}/{len(symbols)}] Fetching {symbol}...")
    
    try:
        from datetime import timedelta
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        df = fmp.get_historical_intraday(symbol, '1min', start_date, end_date)
        
        if df is not None and not df.empty:
            records = len(df)
            total_records += records
            success_count += 1
            results[symbol] = {
                'success': True,
                'records': records,
                'start': df.index.min(),
                'end': df.index.max()
            }
            log(f"  [OK] {symbol}: {records:,} records")
        else:
            results[symbol] = {'success': False, 'error': 'No data returned'}
            log(f"  [FAIL] {symbol}: No data returned")
            
    except Exception as e:
        results[symbol] = {'success': False, 'error': str(e)}
        log(f"  [FAIL] {symbol}: Error - {e}")
    
    # Small delay between requests
    time.sleep(0.2)

# Print summary
log("")
log("=" * 60)
log("Results Summary")
log("=" * 60)

for symbol, result in results.items():
    if result.get('success'):
        records = result.get('records', 0)
        start = result.get('start')
        end = result.get('end')
        log(f"[OK] {symbol}: {records:,} records ({start} to {end})")
    else:
        error = result.get('error', 'Unknown error')
        log(f"[FAIL] {symbol}: Failed - {error}")

log("")
log(f"Successfully fetched: {success_count}/{len(symbols)} symbols")
log(f"Total records saved: {total_records:,}")
log(f"Database: data/db/historical.db")
log("")
log("Done!")

log_file.close()



