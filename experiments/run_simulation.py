#!/usr/bin/env python3
"""
🎯 GAUSSIAN STANDALONE SIMULATION LAUNCHER

This is the main entry point for simulation training in the Gaussian system.
It runs the time-travel training on historical data - the exact same workflow
as the original run_simulation.py but from within the gaussian folder.

Usage: 
  python run_simulation.py (from gaussian/ folder)
  cd gaussian && python run_simulation.py (from parent directory)
"""

import sys
import os
import subprocess
from pathlib import Path

# Add current directory to Python path so we can import local modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import json

# Import credentials manager
try:
    from tradier_credentials import TradierCredentials
    HAS_CREDENTIALS = True
except ImportError:
    HAS_CREDENTIALS = False

# Fix encoding issues on Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def find_training_script():
    """Find the training script (time-travel training) - ISOLATED VERSION"""
    current_dir = Path(__file__).parent
    
    # 🚨 GAUSSIAN STANDALONE: Use isolated script, NOT parent's old one
    # First check in gaussian/scripts (preferred - isolated mode)
    if (current_dir / 'scripts' / 'train_time_travel.py').exists():
        return current_dir / 'scripts' / 'train_time_travel.py'
    
    # Fallback to gaussian/deprecated (legacy)
    if (current_dir / 'deprecated' / 'train_with_time_travel_v2.py').exists():
        return current_dir / 'deprecated' / 'train_with_time_travel_v2.py'
    
    # DO NOT use parent's old script (would cause overfitting)
    parent_dir = current_dir.parent
    if (parent_dir / 'train_with_time_travel_v2.py').exists():
        print("[!] WARNING: Found parent's old script but using gaussian/scripts/ instead (standalone mode)")
    
    return None

def find_data_directory():
    """Find the data directory - ISOLATED VERSION"""
    current_dir = Path(__file__).parent
    
    # 🚨 GAUSSIAN STANDALONE: Use only gaussian/data
    # Check current/data (gaussian's own data)
    if (current_dir / 'data').exists():
        return current_dir / 'data'
    
    # Data must exist within gaussian - no fallback to parent
    return None

def update_historical_data():
    """Update historical database with fresh market data
    
    Uses parallel fetching with ThreadPoolExecutor for ~3-5x faster data loading.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading
    
    print("\n" + "=" * 80)
    print("  FETCHING FRESH MARKET DATA (PARALLEL)")
    print("=" * 80)
    
    # Initialize database
    os.makedirs('data/db', exist_ok=True)
    db_path = 'data/db/historical.db'
    conn = sqlite3.connect(db_path, check_same_thread=False)
    cursor = conn.cursor()
    db_lock = threading.Lock()  # Thread-safe database access
    
    # Create table if it doesn't exist
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
    conn.commit()
    
    # Read symbols from config (with fallback including MEGA_CAP_TECH)
    DEFAULT_SYMBOLS = [
        # Core market indices
        'SPY', 'QQQ', 'IWM', 'DIA', '^VIX',
        # Mega-cap tech (SPY influencers)
        'NVDA', 'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'PLTR', 'INTC', 'AMD',
        # Sectors
        'XLF', 'XLK', 'XLE', 'XLU', 'XLY', 'XLP', 'HYG', 'LQD',
        # Macro
        'TLT', 'IEF', 'SHY', 'UUP',
        # Crypto
        'BTC-USD', 'ETH-USD'
    ]
    
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
        symbols = config.get('data_fetching', {}).get('symbols', DEFAULT_SYMBOLS)
        training_period = config.get('data_sources', {}).get('training_period', '90d')
    except:
        symbols = DEFAULT_SYMBOLS
        training_period = '90d'
    
    # IMPORTANT: Tradier only provides limited 1-minute historical data (~5-20 days)
    tradier_intraday_period = '15d'  # Realistic for Tradier 1-min data
    
    print(f"[*] Fetching {len(symbols)} symbols IN PARALLEL: {', '.join(symbols)}")
    print(f"[*] Configured training period: {training_period}")
    print(f"[*] Tradier intraday limit: ~{tradier_intraday_period} (API limitation)")
    print(f"[*] Training will use intersection of available data for all symbols")
    
    primary_source = None
    backup_source = None
    
    # Initialize Tradier with API key from credentials
    try:
        from backend.tradier_data_source import TradierDataSource as TradierSource
        
        if HAS_CREDENTIALS:
            creds = TradierCredentials()
            api_key = creds.get_data_api_token()
            if api_key:
                try:
                    tradier_source = TradierSource(api_key=api_key, sandbox=False)
                    primary_source = tradier_source
                    print("[✓] Tradier data source initialized (PRIMARY)")
                except Exception as init_error:
                    print(f"[WARN] Tradier source creation failed: {init_error}")
            else:
                print("[*] Tradier Data API key not configured - skipping")
        else:
            print("[*] TradierCredentials not available - skipping Tradier")
    except Exception as e:
        print(f"[WARN] Tradier initialization failed: {e}")
    
    try:
        from backend.polygon_data_source import PolygonDataSource
        polygon_api_key = None
        try:
            with open('config.json', 'r') as f:
                config = json.load(f)
            polygon_api_key = config.get('credentials', {}).get('polygon', {}).get('api_key')
        except:
            try:
                with open('tradier_config.json', 'r') as f:
                    config = json.load(f)
                polygon_api_key = config.get('polygon', {}).get('api_key')
            except:
                pass
        
        if polygon_api_key:
            polygon = PolygonDataSource(api_key=polygon_api_key)
            print("[*] Using Polygon as backup data source")
            backup_source = polygon
        else:
            print("[*] Polygon API key not configured - skipping")
    except Exception as e:
        print(f"[WARN] Polygon not available: {e}")
    
    def fetch_symbol_data(symbol):
        """Fetch data for a single symbol (runs in thread)"""
        api_symbol = symbol.lstrip('^')  # ^VIX -> VIX
        data = None
        source_used = None
        
        # Crypto symbols need Yahoo Finance (Tradier/Polygon don't support them)
        is_crypto = '-USD' in symbol or symbol in ['BTC', 'ETH', 'BTCUSD', 'ETHUSD']
        
        # For crypto, skip Tradier/Polygon and go straight to Yahoo Finance
        if not is_crypto:
            # Try primary source first (Tradier)
            if primary_source:
                try:
                    data = primary_source.get_historical_data(api_symbol, period=tradier_intraday_period, interval='1m')
                    if data is not None and not data.empty:
                        source_used = 'Tradier'
                except Exception:
                    pass
            
            # Try backup source (Polygon)
            if (data is None or data.empty) and backup_source:
                try:
                    data = backup_source.get_historical_data(api_symbol, period=training_period, interval='1m')
                    if data is not None and not data.empty:
                        source_used = 'Polygon'
                except Exception:
                    pass
        
        # Yahoo Finance fallback (especially for crypto and when others fail)
        if data is None or data.empty:
            try:
                import yfinance as yf
                ticker = yf.Ticker(symbol)
                # Use 5d for crypto (yfinance 1m data limited to ~7 days)
                yf_period = '5d' if is_crypto else tradier_intraday_period
                data = ticker.history(period=yf_period, interval='1m')
                if data is not None and not data.empty:
                    source_used = 'Yahoo'
            except Exception:
                pass
        
        return symbol, data, source_used
    
    def insert_symbol_data(symbol, data):
        """Insert data for a symbol into database (thread-safe)"""
        if data is None or data.empty:
            return 0
        
        insert_count = 0
        rows_to_insert = []
        
        def normalize_timestamp(ts_str):
            """Strip timezone suffix to prevent duplicates"""
            ts = str(ts_str)
            return ts[:19] if len(ts) > 19 else ts
        
        for idx, row in data.iterrows():
            # Normalize timestamp to prevent duplicates with different timezone formats
            ts_raw = pd.Timestamp(idx).isoformat()
            ts_normalized = normalize_timestamp(ts_raw)
            
            rows_to_insert.append((
                symbol,
                ts_normalized,
                row.get('Open', row.get('open', None)),
                row.get('High', row.get('high', None)),
                row.get('Low', row.get('low', None)),
                row.get('Close', row.get('close', None)),
                row.get('Volume', row.get('volume', 0))
            ))
        
        # Batch insert with lock
        with db_lock:
            try:
                cursor.executemany('''
                    INSERT OR REPLACE INTO historical_data 
                    (symbol, timestamp, open_price, high_price, low_price, close_price, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', rows_to_insert)
                conn.commit()
                insert_count = len(rows_to_insert)
            except Exception:
                pass
        
        return insert_count
    
    # Parallel fetch all symbols
    results = {}
    max_workers = min(len(symbols), 5)  # Limit concurrent API calls
    
    print(f"\n[*] Starting parallel fetch with {max_workers} workers...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all fetch tasks
        future_to_symbol = {executor.submit(fetch_symbol_data, sym): sym for sym in symbols}
        
        # Collect results as they complete
        for future in as_completed(future_to_symbol):
            symbol, data, source_used = future.result()
            results[symbol] = (data, source_used)
            
            if data is not None and not data.empty:
                print(f"    [OK] {symbol}: {len(data)} records from {source_used}")
            else:
                print(f"    [WARN] {symbol}: No data available")
    
    # Insert data (can also be parallelized but SQLite benefits less)
    print("\n[*] Inserting data into database...")
    total_inserted = 0
    
    for symbol, (data, source_used) in results.items():
        count = insert_symbol_data(symbol, data)
        if count > 0:
            total_inserted += count
    
    conn.close()
    print(f"\n[✓] Data update complete! ({total_inserted:,} total records)")

def main():
    """Main entry point - run training simulation"""
    
    print("\n" + "=" * 80)
    print("  GAUSSIAN SIMULATION TRAINING LAUNCHER")
    print("  Time-Travel Training: Fast-Forward Through Historical Data")
    print("=" * 80)
    print()
    
    # Find training script
    training_script = find_training_script()
    if not training_script:
        print("[ERROR] Could not find training script!")
        print("  Looking for: train_with_time_travel_v2.py")
        print("  Checked: parent directory and gaussian/deprecated/")
        sys.exit(1)
    
    print(f"[*] Training script found: {training_script}")
    
    # Find data directory
    data_dir = find_data_directory()
    if not data_dir:
        print("[ERROR] Could not find data directory!")
        print("  Looking for: data/")
        sys.exit(1)
    
    print(f"[*] Data directory found: {data_dir}")
    print()
    
    # Set working directory to GAUSSIAN folder (standalone mode)
    current_dir = Path(__file__).parent
    work_dir = current_dir
    
    # Update historical data with fresh market data
    if os.environ.get('SKIP_DATA_UPDATE') == '1':
        print("\n" + "=" * 80)
        print("  SKIPPING DATA UPDATE (Weekend/Configured)")
        print("=" * 80)
    else:
        try:
            update_historical_data()
        except Exception as e:
            print(f"[WARN] Could not fetch fresh data: {e}")
            print("[*] Continuing with existing data...")
    
    # Validate data integrity before starting training
    print("\n" + "=" * 80)
    print("  VALIDATING DATA INTEGRITY")
    print("=" * 80)
    
    try:
        from data_manager import DataManager
        dm = DataManager('data/db/historical.db')
        
        # Check if simulation is ready
        is_ready, issues = dm.validate_simulation_ready()
        
        if issues:
            print("\n[!] Data validation issues found:")
            for issue in issues:
                print(f"  - {issue}")
        
        # Print data summary
        print("\nData Summary:")
        summary = dm.get_data_summary()
        for symbol, info in summary['symbols'].items():
            if info['records'] > 0:
                print(f"  {symbol:6s}: {info['records']:,} records | {info['first_date']} to {info['last_date']}")
        
        if not is_ready:
            print("\n[ERROR] Simulation data is not ready!")
            print("[*] Please ensure:")
            print("  - Tradier/Polygon API is configured")
            print("  - API keys are valid")
            print("  - You have internet connectivity")
            sys.exit(1)
        
        print("\n[✓] Data validation passed - proceeding with training!")
        
    except Exception as e:
        print(f"[WARN] Could not validate data: {e}")
        print("[*] Continuing anyway...")
    
    # Create timestamped models directory for this run
    from datetime import datetime
    timestamp = os.environ.get('MODEL_RUN_TIMESTAMP')
    if not timestamp:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    env_run_dir = os.environ.get('MODEL_RUN_DIR')
    if env_run_dir:
        run_models_dir = Path(env_run_dir)
    else:
        run_models_dir = current_dir / "models" / f"run_{timestamp}"
    run_models_dir.mkdir(parents=True, exist_ok=True)
    
    # Create symlink to current run (so bot knows where to save)
    current_link = current_dir / "models" / "current"
    if current_link.exists() or current_link.is_symlink():
        current_link.unlink()
    
    # On Windows, use a marker file instead of symlink
    if sys.platform == 'win32':
        marker_file = current_dir / "models" / "current_run.txt"
        with open(marker_file, 'w') as f:
            f.write(str(run_models_dir))
        print(f"\n[*] Created new run directory: {run_models_dir.name}")
        print(f"[*] Current run marker: models/current_run.txt")
    else:
        current_link.symlink_to(run_models_dir, target_is_directory=True)
        print(f"\n[*] Created new run directory: {run_models_dir.name}")
        print(f"[*] Symlinked as: models/current")
    
    print("[*] ✅ Fresh model directory for this training run")
    print("[*] ✅ Previous runs preserved in models/ for comparison")
    print()
    
    print(f"[*] Working directory: {work_dir} (GAUSSIAN STANDALONE MODE)")
    
    log_hint = os.environ.get('TRAINING_LOG_FILE', 'logs/real_bot_simulation_*.log')
    print("=" * 80)
    print("[*] Starting simulation training (FRESH START - no old models)")
    print(f"[*] Logs will be saved to: {log_hint}")
    print("[*] Run 'python gaussian/watch_training_dashboard.py' to watch!")
    print("=" * 80)
    print()
    
    try:
        # Pass the run directory to training script via environment variable
        env = os.environ.copy()
        env['MODEL_RUN_DIR'] = str(run_models_dir)
        env['MODEL_RUN_TIMESTAMP'] = timestamp
        if 'TRAINING_LOG_FILE' in os.environ:
            env['TRAINING_LOG_FILE'] = os.environ['TRAINING_LOG_FILE']
        # Pass initial balance from parent process or use config
        if 'TRAINING_INITIAL_BALANCE' not in env:
            try:
                import json
                with open('config.json', 'r') as f:
                    cfg = json.load(f)
                env['TRAINING_INITIAL_BALANCE'] = str(cfg.get('trading', {}).get('initial_balance', 1000.0))
            except:
                env['TRAINING_INITIAL_BALANCE'] = '1000.0'
        
        # Run the training script with proper working directory
        # Let subprocess output stream directly (don't capture)
        result = subprocess.run(
            [sys.executable, str(training_script)],
            cwd=str(work_dir),
            env=env
        )
        
        sys.exit(result.returncode)
        
    except KeyboardInterrupt:
        print("\n\n[*] Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] Failed to start training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()



