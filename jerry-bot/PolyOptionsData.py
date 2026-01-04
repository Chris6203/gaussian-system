import requests
import pandas as pd
import datetime as dt
import time
import os
import sys
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Optional
import json
from dateutil.relativedelta import relativedelta

# Import config
sys.path.insert(0, str(Path(__file__).parent))
from core.config import RunConfig, StrategyConfig

# ============================================================================
# CONFIGURATION
# ============================================================================

POLYGON_BASE_URL = "https://api.polygon.io"
DATA_DIR = Path("data/polygon_options")
DATA_DIR.mkdir(parents=True, exist_ok=True)

TIMEFRAME_MAP = {
    '1': 1,
    '5': 5,
    '15': 15,
    '30': 30,
    '1H': 60,
    '2H': 120,
    '4H': 240,
    'D': 1440,
    'W': 10080,
    'M': 43200,
    'Y': 525600
}

TIMEFRAME_NAMES = {
    '1': 'M1',
    '5': 'M5',
    '15': 'M15',
    '30': 'M30',
    '1H': 'H1',
    '2H': 'H2',
    '4H': 'H4',
    'D': 'D',
    'W': 'W',
    'M': 'M',
    'Y': 'Y'
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_config() -> tuple:
    """Load API keys from config.py"""
    config = RunConfig()
    if config.polygon_key == "YOUR_POLYGON_KEY_HERE":
        print("\n⚠️  ERROR: Polygon API key not configured in config.py")
        print("Please update RunConfig.polygon_key with your actual key.\n")
        sys.exit(1)
    return config, StrategyConfig()

def get_available_instruments() -> List[str]:
    """Return list of commonly traded options instruments"""
    instruments = [
        "SPY", "QQQ", "IWM", "DIA",  # Major ETFs
        "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",  # Tech
        "GE", "XOM", "JPM", "BAC",  # Industrials/Finance
        "NFLX", "META", "NVDA", "AMD"  # Growth
    ]
    return sorted(instruments)

def prompt_instrument() -> str:
    """Interactive instrument selection"""
    while True:
        user_input = input("\nWhat Instrument [type ? to list them all]: ").upper().strip()
        
        if user_input == "?":
            instruments = get_available_instruments()
            print("\nAvailable Instruments:")
            for i, inst in enumerate(instruments, 1):
                print(f"  {i:2d}. {inst}")
            continue
        
        if user_input in get_available_instruments():
            return user_input
        
        print(f"❌ '{user_input}' not recognized. Type ? to see available instruments.")

def prompt_dataset_mode() -> str:
    """Prompt for dataset creation mode"""
    while True:
        mode = input("\nCreate new dataset or overwrite [new / overwrite - leave blank, Append - A]: ").lower().strip()
        if mode in ['new', 'overwrite', 'a', 'append', '']:
            if mode == '':
                return 'new'
            if mode in ['overwrite']:
                return 'overwrite'
            if mode in ['a', 'append']:
                return 'append'
            return mode
        print("❌ Invalid input. Enter 'new', 'overwrite', or 'A' for append.")

def prompt_timeframes() -> List[str]:
    """Prompt for timeframe selection"""
    valid_tf = list(TIMEFRAME_MAP.keys())
    print(f"\nAvailable timeframes: {', '.join(valid_tf)}, or ALL")
    
    while True:
        user_input = input("What timeframes [1,5,15,30,1H,2H,4H,D,W,Y, or ALL]: ").upper().strip()
        
        if user_input == "ALL":
            return valid_tf
        
        selected = [tf.strip() for tf in user_input.split(',')]
        
        if all(tf in valid_tf for tf in selected):
            return selected
        
        invalid = [tf for tf in selected if tf not in valid_tf]
        print(f"❌ Invalid timeframes: {', '.join(invalid)}")

def prompt_date_range() -> tuple:
    """Prompt for date range"""
    while True:
        try:
            start_str = input("Enter start date [year-month-day]: ").strip()
            start_date = dt.datetime.strptime(start_str, "%Y-%m-%d").date()
            
            end_str = input("Enter end date [year-month-day]: ").strip()
            end_date = dt.datetime.strptime(end_str, "%Y-%m-%d").date()
            
            if start_date >= end_date:
                print("❌ Start date must be before end date.")
                continue
            
            return start_date, end_date
        except ValueError:
            print("❌ Invalid date format. Use YYYY-MM-DD.")

# ============================================================================
# POLYGON API FUNCTIONS
# ============================================================================

def get_option_chain_snapshot(api_key: str, underlying: str, date: dt.date) -> Optional[pd.DataFrame]:
    """
    Fetch options chain snapshot for a specific date from Polygon.io
    
    Returns DataFrame with columns: strike, expiration, option_symbol, bid, ask, 
                                    last_price, volume, open_interest, delta, gamma, theta, vega, iv
    """
    
    url = f"{POLYGON_BASE_URL}/v3/snapshot/options/chains/{underlying}"
    params = {
        'apiKey': api_key,
        'order': 'asc',
        'limit': 1000
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if 'results' not in data or not data['results']:
            return None
        
        records = []
        for option in data['results']:
            try:
                details = option.get('details', {})
                last_quote = option.get('last_quote', {})
                last_trade = option.get('last_trade', {})
                
                record = {
                    'timestamp': dt.datetime.utcnow(),
                    'date': date,
                    'symbol': underlying,
                    'option_symbol': option.get('option_symbol', ''),
                    'strike': details.get('strike_price'),
                    'expiration': details.get('expiration_date'),
                    'contract_type': details.get('contract_type'),  # 'call' or 'put'
                    'bid': last_quote.get('bid'),
                    'ask': last_quote.get('ask'),
                    'last_price': last_trade.get('price'),
                    'bid_size': last_quote.get('bid_size'),
                    'ask_size': last_quote.get('ask_size'),
                    'volume': option.get('day', {}).get('volume'),
                    'open_interest': option.get('open_interest'),
                    'vega': details.get('vega'),
                    'gamma': details.get('gamma'),
                    'delta': details.get('delta'),
                    'theta': details.get('theta'),
                    'implied_volatility': details.get('implied_volatility'),
                }
                records.append(record)
            except Exception as e:
                continue
        
        if not records:
            return None
        
        df = pd.DataFrame(records)
        return df
    
    except requests.exceptions.RequestException as e:
        print(f"⚠️  API request failed for {date}: {e}")
        return None

def fetch_options_data(api_key: str, underlying: str, start_date: dt.date, 
                      end_date: dt.date) -> pd.DataFrame:
    """
    Fetch options data for date range
    """
    all_data = []
    current_date = start_date
    total_days = (end_date - start_date).days
    
    print(f"\nConnecting to Polygon.io...")
    print(f"Connected ✓\n")
    
    with tqdm(total=total_days, desc="Fetching options data", unit="day") as pbar:
        while current_date <= end_date:
            # Skip weekends
            if current_date.weekday() < 5:  # Monday-Friday
                df = get_option_chain_snapshot(api_key, underlying, current_date)
                if df is not None:
                    all_data.append(df)
            
            current_date += dt.timedelta(days=1)
            pbar.update(1)
            
            # Rate limiting for free tier
            time.sleep(0.2)
    
    if not all_data:
        print("❌ No options data retrieved.")
        return pd.DataFrame()
    
    combined = pd.concat(all_data, ignore_index=True)
    return combined

# ============================================================================
# DATA PROCESSING
# ============================================================================

def clean_options_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and normalize options data"""
    
    print("\nCleaning data...")
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['symbol', 'option_symbol', 'strike', 'expiration', 'date'], 
                            keep='first')
    
    # Fill missing bid-ask with last_price
    df['bid'] = df['bid'].fillna(df['last_price'])
    df['ask'] = df['ask'].fillna(df['last_price'])
    
    # Forward fill Greeks if missing
    for col in ['delta', 'gamma', 'theta', 'vega', 'implied_volatility']:
        df[col] = df.groupby('option_symbol')[col].fillna(method='ffill').fillna(method='bfill')
    
    # Remove rows with NaN strikes or expirations
    df = df.dropna(subset=['strike', 'expiration'])
    
    # Ensure numeric columns
    numeric_cols = ['bid', 'ask', 'last_price', 'strike', 'delta', 'gamma', 'theta', 'vega', 'implied_volatility']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def round_decimals(df: pd.DataFrame, decimals: int = 2) -> pd.DataFrame:
    """Round price columns to specified decimals"""
    price_cols = ['bid', 'ask', 'last_price', 'strike']
    for col in price_cols:
        if col in df.columns:
            df[col] = df[col].round(decimals)
    return df

def aggregate_to_timeframe(df: pd.DataFrame, timeframe_minutes: int) -> pd.DataFrame:
    """
    Aggregate options data to specified timeframe
    For options, we take the latest quote in each timeframe bucket
    """
    
    if timeframe_minutes >= 1440:  # Daily or larger
        df['bucket'] = df['date']
    else:
        df['bucket'] = df['timestamp'].dt.floor(f'{timeframe_minutes}T')
    
    # Group by strike, expiration, contract_type, and time bucket
    # Take the last (most recent) quote in each bucket
    agg_df = df.groupby(['symbol', 'strike', 'expiration', 'contract_type', 'bucket']).apply(
        lambda x: x.iloc[-1]
    ).reset_index(drop=True)
    
    return agg_df.sort_values('timestamp')

def validate_data_integrity(df: pd.DataFrame, timeframe_name: str) -> bool:
    """Validate data integrity and report issues"""
    
    print(f"\n{timeframe_name} Data Integrity Check...", end='')
    
    issues = []
    
    # Check for missing critical columns
    required_cols = ['strike', 'expiration', 'bid', 'ask', 'delta']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        issues.append(f"Missing columns: {missing}")
    
    # Check for missing values in critical columns
    null_counts = df[required_cols].isnull().sum()
    if (null_counts > 0).any():
        issues.append(f"Null values: {null_counts[null_counts > 0].to_dict()}")
    
    # Check for invalid prices (bid > ask)
    if (df['bid'] > df['ask']).any():
        issues.append(f"Invalid spreads (bid > ask): {(df['bid'] > df['ask']).sum()} rows")
    
    if issues:
        print("⚠️  ISSUES FOUND")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("DONE ✓")
        return True

def save_timeframe_data(df: pd.DataFrame, underlying: str, timeframe_name: str, 
                       mode: str = 'new') -> str:
    """Save data to CSV file"""
    
    filename = DATA_DIR / f"{underlying}_{timeframe_name}.csv"
    
    if mode == 'overwrite' or not filename.exists():
        df.to_csv(filename, index=False)
    elif mode == 'append':
        existing = pd.read_csv(filename)
        combined = pd.concat([existing, df], ignore_index=True)
        combined = combined.drop_duplicates(subset=['symbol', 'option_symbol', 'date'], keep='last')
        combined.to_csv(filename, index=False)
    else:  # new - create if doesn't exist
        if not filename.exists():
            df.to_csv(filename, index=False)
        else:
            existing = pd.read_csv(filename)
            combined = pd.concat([existing, df], ignore_index=True)
            combined = combined.drop_duplicates(subset=['symbol', 'option_symbol', 'date'], keep='last')
            combined.to_csv(filename, index=False)
    
    return str(filename)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution flow"""
    
    print("\n" + "="*70)
    print("Polygon.io Historical Options Data Scraper")
    print("="*70)
    
    # Load config
    run_config, strat_config = load_config()
    
    # Interactive prompts
    underlying = prompt_instrument()
    dataset_mode = prompt_dataset_mode()
    timeframes = prompt_timeframes()
    start_date, end_date = prompt_date_range()
    
    print(f"\nRunning...")
    print(f"Connecting...")
    
    # Fetch raw data once
    raw_df = fetch_options_data(run_config.polygon_key, underlying, start_date, end_date)
    
    if raw_df.empty:
        print("\n❌ Failed to retrieve any data. Check your API key and date range.")
        return
    
    # Clean data
    raw_df = clean_options_data(raw_df)
    raw_df = round_decimals(raw_df)
    
    print(f"\nRaw data retrieved: {len(raw_df)} records")
    
    # Process each timeframe
    processed_timeframes = []
    
    for tf_code in timeframes:
        tf_minutes = TIMEFRAME_MAP[tf_code]
        tf_name = TIMEFRAME_NAMES[tf_code]
        
        print(f"\nPulling data for {tf_name} timeframe...", end='', flush=True)
        
        # Aggregate to timeframe
        tf_df = aggregate_to_timeframe(raw_df, tf_minutes)
        
        if tf_df.empty:
            print(f"  ⚠️  No data for {tf_name}")
            continue
        
        print(f"  {len(tf_df)} records")
        
        # Validate integrity
        is_valid = validate_data_integrity(tf_df, tf_name)
        
        # Save to CSV
        filepath = save_timeframe_data(tf_df, underlying, tf_name, dataset_mode)
        print(f"  Saved: {filepath}")
        
        processed_timeframes.append(tf_name)
    
    # Summary
    print("\n" + "="*70)
    print("COMPLETION SUMMARY")
    print("="*70)
    print(f"Instrument: {underlying}")
    print(f"Date Range: {start_date} to {end_date}")
    print(f"Timeframes Processed: {', '.join(processed_timeframes)}")
    print(f"Data Directory: {DATA_DIR}")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()