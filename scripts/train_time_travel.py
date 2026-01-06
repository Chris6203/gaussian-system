#!/usr/bin/env python3
"""
Time-Travel Training V2: Feed the bot historical data as fast as possible
Works EXACTLY like live bot, just with historical data instead of waiting for live data
"""

import sys
import os

# GAUSSIAN STANDALONE: Add parent directory to path so we can import local modules
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)
sys.path.insert(0, os.path.join(_project_root, "core"))  # unified_options_trading_bot is in core/

# Dashboard heartbeat for tracking running experiments
try:
    from backend.dashboard.heartbeat import ExperimentHeartbeat
    HEARTBEAT_AVAILABLE = True
except ImportError:
    HEARTBEAT_AVAILABLE = False
    ExperimentHeartbeat = None

# Import centralized P&L calculation (prevents 100x multiplier bug)
try:
    from utils.pnl import compute_pnl_pct
except ImportError:
    # Fallback if utils not available
    def compute_pnl_pct(trade):
        quantity = trade.get("quantity", 1)
        cost_basis = trade.get("premium_paid", 1) * quantity * 100
        return trade.get("pnl", 0) / cost_basis if cost_basis > 0 else 0.0

import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import logging
from pathlib import Path
import time as real_time
import builtins
import random
import numpy as np
import torch

# Set random seeds for reproducibility (override with RANDOM_SEED env var)
RANDOM_SEED = int(os.environ.get('RANDOM_SEED', '42'))
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# Technical indicators for consensus entry controller (Signal 5)
try:
    from bot_modules.technical_indicators import add_indicators_to_signal
except ImportError:
    add_indicators_to_signal = None

# Phase 44: Ensemble predictor and advanced signal generators
try:
    from bot_modules.ensemble_predictor import (
        EnsemblePredictor, GammaExposureSignals, OrderFlowSignals,
        MultiIndicatorStack, create_ensemble_predictor, create_signal_generators
    )
    ENSEMBLE_AVAILABLE = True
except ImportError:
    ENSEMBLE_AVAILABLE = False
    EnsemblePredictor = None
    GammaExposureSignals = None
    OrderFlowSignals = None
    MultiIndicatorStack = None

# Set up comprehensive logging for both console and file (for dashboard)
log_file = os.environ.get('TRAINING_LOG_FILE')
if log_file:
    log_path = Path(log_file)
else:
    log_path = Path("logs") / f"real_bot_simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
log_path.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(str(log_path), encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION LOADING - All settings from config.json
# =============================================================================
def _load_training_config():
    """Load time-travel training config from config.json."""
    try:
        import json
        config_paths = ['config.json', '../config.json', 'E:/gaussian/output3/config.json']
        for path in config_paths:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    cfg = json.load(f)
                return cfg.get('time_travel_training', {}), cfg.get('entry_controller', {}), cfg
        return {}, {}, {}
    except Exception as e:
        logger.warning(f"Could not load config.json for training: {e}")
        return {}, {}, {}

_tt_config, _entry_config, _full_config = _load_training_config()

# Training settings from config.json
TT_VERBOSE = _tt_config.get('verbose', False)
TT_QUIET = _tt_config.get('quiet', False)
TT_PRINT_EVERY = int(os.environ.get('TT_PRINT_EVERY', _tt_config.get('print_every', 20)))
TT_DASH_EVERY = 5
TT_MAX_CYCLES = int(os.environ.get('TT_MAX_CYCLES', _tt_config.get('max_cycles', 0)))
TT_MAX_HOLD_MINUTES = int(os.environ.get('TT_MAX_HOLD_MINUTES', _tt_config.get('max_hold_minutes', 45)))
TT_MAX_POSITIONS = int(_tt_config.get('max_positions', 3))

# Data interval from config.json (default: 1m, can be 5m for less noise)
_data_sources_config = _full_config.get('data_sources', {})
TT_DATA_INTERVAL = _data_sources_config.get('interval', '1m')  # '1m' or '5m'
TT_INTERVAL_MINUTES = 5 if TT_DATA_INTERVAL == '5m' else 1

# Entry controller type from config.json
ENTRY_CONTROLLER_TYPE = _entry_config.get('type', 'bandit').lower()

# RSI+MACD confirmation filter for higher win rate
# When enabled, trades must have RSI and MACD confirmation
RSI_MACD_FILTER_ENABLED = os.environ.get('RSI_MACD_FILTER', '0') == '1'
RSI_OVERSOLD_THRESHOLD = float(os.environ.get('RSI_OVERSOLD', '40'))  # For calls: RSI < this
RSI_OVERBOUGHT_THRESHOLD = float(os.environ.get('RSI_OVERBOUGHT', '60'))  # For puts: RSI > this
MACD_CONFIRM_ENABLED = os.environ.get('MACD_CONFIRM', '1') == '1'  # Require MACD alignment
# Momentum mode: trade WITH trend (RSI>50 for calls, RSI<50 for puts) instead of mean reversion
RSI_MOMENTUM_MODE = os.environ.get('RSI_MOMENTUM_MODE', '0') == '1'
# Volume confirmation: require above-average volume for entry
VOLUME_CONFIRM_ENABLED = os.environ.get('VOLUME_CONFIRM', '0') == '1'
VOLUME_CONFIRM_THRESHOLD = float(os.environ.get('VOLUME_THRESHOLD', '1.2'))  # 1.2 = 20% above average

# Jerry's Quantor-MTFuzz integration
# JERRY_FEATURES=1: Add Jerry's fuzzy scores as additional NN input features
# JERRY_FILTER=1: Use Jerry's composite F_t score as trade confirmation filter
JERRY_FEATURES_ENABLED = os.environ.get('JERRY_FEATURES', '0') == '1'
JERRY_FILTER_ENABLED = os.environ.get('JERRY_FILTER', '0') == '1'
JERRY_FILTER_THRESHOLD = float(os.environ.get('JERRY_FILTER_THRESHOLD', '0.5'))
JERRY_MTF_MIN = float(os.environ.get('JERRY_MTF_MIN', '0.3'))

# Log loaded configuration
logger.info(f"[CONFIG] Entry controller: {ENTRY_CONTROLLER_TYPE}")
logger.info(f"[CONFIG] TT_MAX_CYCLES: {TT_MAX_CYCLES}, TT_PRINT_EVERY: {TT_PRINT_EVERY}")
logger.info(f"[CONFIG] TT_MAX_HOLD_MINUTES: {TT_MAX_HOLD_MINUTES}, TT_MAX_POSITIONS: {TT_MAX_POSITIONS}")
logger.info(f"[CONFIG] TT_DATA_INTERVAL: {TT_DATA_INTERVAL} ({TT_INTERVAL_MINUTES}-minute bars)")
if RSI_MACD_FILTER_ENABLED:
    logger.info(f"[CONFIG] RSI+MACD filter ENABLED: RSI oversold<{RSI_OVERSOLD_THRESHOLD}, overbought>{RSI_OVERBOUGHT_THRESHOLD}, MACD={MACD_CONFIRM_ENABLED}")
if JERRY_FEATURES_ENABLED or JERRY_FILTER_ENABLED:
    logger.info(f"[CONFIG] Jerry integration: features={JERRY_FEATURES_ENABLED}, filter={JERRY_FILTER_ENABLED} (threshold={JERRY_FILTER_THRESHOLD})")

if TT_QUIET:
    _orig_print = builtins.print

    def _filtered_print(*args, **kwargs):
        try:
            s = " ".join(str(a) for a in args)
        except Exception:
            s = ""

        if TT_VERBOSE:
            return _orig_print(*args, **kwargs)

        # Always allow key lifecycle + diagnostics lines.
        allow = (
            "[OK]", "[WARN]", "[ERROR]", "[TRAIN]", "[DIAG]", "[SUMMARY]",
            "TIME-TRAVEL", "TIME-TRAVELING", "HISTORICAL DATA COMPLETE", "SAVING MODELS",
            "Progress:", "[*] ",
        )
        # Special-case: avoid "empty" summary headers spamming the console.
        # Only show the full summary block periodically.
        if s.startswith("[SUMMARY] CYCLE SUMMARY"):
            cyc = globals().get("cycles", 0)
            if (cyc % TT_PRINT_EVERY) == 0:
                return _orig_print(*args, **kwargs)
            return None

        if s.startswith(allow) or ("HISTORICAL DATA COMPLETE" in s):
            return _orig_print(*args, **kwargs)

        # Allow cycle banners occasionally.
        if s.startswith("[CYCLE"):
            cyc = globals().get("cycles", 0)
            if (cyc % TT_PRINT_EVERY) == 0:
                return _orig_print(*args, **kwargs)
            return None

        # When we do print a cycle summary (every TT_PRINT_EVERY cycles), also allow its indented lines.
        if s.startswith("   - "):
            cyc = globals().get("cycles", 0)
            if (cyc % TT_PRINT_EVERY) == 0:
                return _orig_print(*args, **kwargs)
            return None

        return None

    builtins.print = _filtered_print

print(f"[*] Training log: {log_path}")

def _emit_dash(
    cycle: int,
    signals: int,
    trades: int,
    rl_updates: int,
    balance: float,
    spy: float,
    vix: float,
    qqq: float,
    btc: float,
) -> None:
    """
    Emit a compact dashboard line that is parsed by backend.dashboard.log_parser.LogParser.
    This keeps the dashboard updating even when stdout is quiet / not redirected.
    """
    try:
        if TT_DASH_EVERY <= 0:
            return
        if (cycle % TT_DASH_EVERY) != 0:
            return
        logger.info(
            "[DASH] cycle=%d signals=%d trades=%d rl_updates=%d balance=%.2f spy=%.2f vix=%.2f qqq=%.2f btc=%.2f",
            int(cycle),
            int(signals),
            int(trades),
            int(rl_updates),
            float(balance),
            float(spy),
            float(vix),
            float(qqq),
            float(btc),
        )
    except Exception:
        pass

print("="*70)
print("TIME-TRAVEL TRAINING V2: Fast-Forward Through History")
print("="*70)
print("""
This runs the bot EXACTLY as it does live:
  [OK] Same cycle logic
  [OK] Same signal generation
  [OK] Same trade execution
  [OK] Same learning
  
The ONLY difference:
  [>>] We feed it historical data as fast as it can process
  [>>] No waiting for new data (time-travel!)
  
This guarantees training = production behavior!
""")

# Load historical data
print("\n[*] Loading historical data from database...")
conn = sqlite3.connect('data/db/historical.db')

# Load symbol from config.json
import os
print(f"    [DEBUG] Working directory: {os.getcwd()}")
try:
    import json
    # Try current directory first, then parent directory
    config_paths = ['config.json', '../config.json', 'output/config.json']
    config = None
    for config_path in config_paths:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            print(f"    [CONFIG] Loaded from: {config_path}")
            break
    
    if config:
        trading_symbol = config.get('trading', {}).get('symbol', 'SPY')
        initial_balance_from_config = config.get('trading', {}).get('initial_balance', 5000.0)
        print(f"    [CONFIG] Trading symbol: {trading_symbol}")
        print(f"    [CONFIG] Initial balance from config: ${initial_balance_from_config:,.2f}")
    else:
        print(f"    [WARN] config.json not found in any expected location")
        trading_symbol = 'SPY'  # Default to SPY instead of BITX
except Exception as e:
    print(f"    [WARN] Could not load config.json: {e}")
    trading_symbol = 'SPY'  # Default to SPY instead of BITX

# Load primary symbol and VIX from database
symbols_data = {}

# Load trading symbol
df = pd.read_sql_query("""
    SELECT timestamp, open_price, high_price, low_price, close_price, volume
    FROM historical_data WHERE symbol = ? ORDER BY timestamp
""", conn, params=(trading_symbol,))

if not df.empty:
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601', utc=True)
    df = df.set_index('timestamp')
    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    # Convert to timezone-naive (remove timezone info for compatibility)
    df.index = df.index.tz_localize(None)
    
    # Remove duplicate timestamps (keep last occurrence)
    if df.index.duplicated().any():
        duplicates_count = df.index.duplicated().sum()
        df = df[~df.index.duplicated(keep='last')]
        print(f"    [WARN] Removed {duplicates_count} duplicate timestamps for {trading_symbol}")
    
    symbols_data[trading_symbol] = df
    print(f"    [OK] Loaded {len(df)} records for {trading_symbol}")
else:
    print(f"    [ERROR] No data found for {trading_symbol}")
    print(f"    [HINT] Run data fetching first or ensure data is in database")

# ============================================================================
# LOAD REQUIRED SYMBOLS: VIX, SPY, QQQ, BTC (for realistic training)
# Training only happens when ALL these symbols have data
# ============================================================================

# Load VIX data (required by bot for predictions)
vix_df = pd.read_sql_query("""
    SELECT timestamp, open_price, high_price, low_price, close_price, volume
    FROM historical_data WHERE symbol IN ('VIX', '^VIX') ORDER BY timestamp
""", conn, params=())

if not vix_df.empty:
    vix_df['timestamp'] = pd.to_datetime(vix_df['timestamp'], format='ISO8601', utc=True)
    vix_df = vix_df.set_index('timestamp')
    vix_df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    # Convert to timezone-naive (remove timezone info for compatibility)
    vix_df.index = vix_df.index.tz_localize(None)
    
    # Remove duplicate timestamps (keep last occurrence)
    if vix_df.index.duplicated().any():
        duplicates_count = vix_df.index.duplicated().sum()
        vix_df = vix_df[~vix_df.index.duplicated(keep='last')]
        print(f"    [WARN] Removed {duplicates_count} duplicate timestamps for VIX")
    
    symbols_data['VIX'] = vix_df
    symbols_data['^VIX'] = vix_df  # Alias for compatibility
    print(f"    [OK] Loaded {len(vix_df)} records for VIX")
else:
    print(f"    [WARN] No VIX data found - bot predictions may be limited")
    vix_df = pd.DataFrame()

# Load SPY data (required for market context)
spy_df = pd.read_sql_query("""
    SELECT timestamp, open_price, high_price, low_price, close_price, volume
    FROM historical_data WHERE symbol = 'SPY' ORDER BY timestamp
""", conn, params=())

if not spy_df.empty:
    spy_df['timestamp'] = pd.to_datetime(spy_df['timestamp'], format='ISO8601', utc=True)
    spy_df = spy_df.set_index('timestamp')
    spy_df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    spy_df.index = spy_df.index.tz_localize(None)
    if spy_df.index.duplicated().any():
        spy_df = spy_df[~spy_df.index.duplicated(keep='last')]
    symbols_data['SPY'] = spy_df
    print(f"    [OK] Loaded {len(spy_df)} records for SPY")
else:
    print(f"    [WARN] No SPY data found")
    spy_df = pd.DataFrame()

# Load QQQ data (required for market context)
qqq_df = pd.read_sql_query("""
    SELECT timestamp, open_price, high_price, low_price, close_price, volume
    FROM historical_data WHERE symbol = 'QQQ' ORDER BY timestamp
""", conn, params=())

if not qqq_df.empty:
    qqq_df['timestamp'] = pd.to_datetime(qqq_df['timestamp'], format='ISO8601', utc=True)
    qqq_df = qqq_df.set_index('timestamp')
    qqq_df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    qqq_df.index = qqq_df.index.tz_localize(None)
    if qqq_df.index.duplicated().any():
        qqq_df = qqq_df[~qqq_df.index.duplicated(keep='last')]
    symbols_data['QQQ'] = qqq_df
    print(f"    [OK] Loaded {len(qqq_df)} records for QQQ")
else:
    print(f"    [WARN] No QQQ data found")
    qqq_df = pd.DataFrame()

# Load BTC data (check multiple symbols: BTC-USD, BTCUSD, BTC)
btc_df = pd.read_sql_query("""
    SELECT timestamp, open_price, high_price, low_price, close_price, volume
    FROM historical_data WHERE symbol IN ('BTC-USD', 'BTCUSD', 'BTC') ORDER BY timestamp
""", conn, params=())

if not btc_df.empty:
    btc_df['timestamp'] = pd.to_datetime(btc_df['timestamp'], format='ISO8601', utc=True)
    btc_df = btc_df.set_index('timestamp')
    btc_df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    btc_df.index = btc_df.index.tz_localize(None)
    if btc_df.index.duplicated().any():
        btc_df = btc_df[~btc_df.index.duplicated(keep='last')]
    symbols_data['BTC'] = btc_df
    symbols_data['BTC-USD'] = btc_df  # Alias
    print(f"    [OK] Loaded {len(btc_df)} records for BTC")
else:
    print(f"    [WARN] No BTC data found")
    btc_df = pd.DataFrame()

# ============================================================================
# LOAD ADDITIONAL CONTEXT SYMBOLS (for enhanced features)
# These are optional - used if available for cross-asset analysis
# ============================================================================

ADDITIONAL_SYMBOLS = ['IWM', 'DIA', 'XLF', 'XLK', 'XLE', 'XLU', 'XLY', 'XLP', 
                      'HYG', 'LQD', 'TLT', 'IEF', 'SHY', 'UUP', 'ETH-USD']

print(f"\n[*] Loading additional context symbols...")

for symbol in ADDITIONAL_SYMBOLS:
    if symbol in symbols_data:
        continue  # Already loaded
    
    # Handle different symbol formats in database
    db_symbols = [symbol]
    if symbol == 'ETH-USD':
        db_symbols = ['ETH-USD', 'ETHUSD', 'ETH']
    
    placeholders = ','.join(['?' for _ in db_symbols])
    query = f"""
        SELECT timestamp, open_price, high_price, low_price, close_price, volume
        FROM historical_data WHERE symbol IN ({placeholders}) ORDER BY timestamp
    """
    
    try:
        symbol_df = pd.read_sql_query(query, conn, params=db_symbols)
        
        if not symbol_df.empty:
            symbol_df['timestamp'] = pd.to_datetime(symbol_df['timestamp'], format='ISO8601', utc=True)
            symbol_df = symbol_df.set_index('timestamp')
            symbol_df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            symbol_df.index = symbol_df.index.tz_localize(None)
            if symbol_df.index.duplicated().any():
                symbol_df = symbol_df[~symbol_df.index.duplicated(keep='last')]
            symbols_data[symbol] = symbol_df
            print(f"    [OK] Loaded {len(symbol_df)} records for {symbol}")
        else:
            print(f"    [--] No data for {symbol}")
    except Exception as e:
        print(f"    [--] Could not load {symbol}: {e}")

conn.close()

# ============================================================================
# MARKET HOURS HELPER: Only train during regular market hours (9:30 AM - 4:00 PM ET)
# ============================================================================

def is_market_hours(timestamp) -> bool:
    """
    Check if timestamp is during regular US market hours (9:30 AM - 4:00 PM ET)
    This ensures training uses realistic market conditions only.
    """
    # Handle timezone-naive timestamps (assume ET)
    hour = timestamp.hour
    minute = timestamp.minute
    weekday = timestamp.weekday()
    
    # Skip weekends
    if weekday >= 5:  # Saturday=5, Sunday=6
        return False
    
    # Market hours: 9:30 AM - 4:00 PM ET
    if hour < 9 or hour >= 16:
        return False
    if hour == 9 and minute < 30:
        return False
    
    return True

def has_required_data(tt_source, timestamp, required_symbols=['SPY', 'QQQ', 'VIX']) -> tuple:
    """
    Check if all required symbols have data at this timestamp.
    Returns (has_all_data, missing_symbols)
    """
    missing = []
    for symbol in required_symbols:
        price = tt_source.get_current_price(symbol)
        if not price or price <= 0:
            missing.append(symbol)
    
    return len(missing) == 0, missing

# ============================================================================
# COMPUTE INTERSECTION OF AVAILABLE DATA RANGES
# Tradier only provides limited intraday history (~5-20 days for 1-min data)
# Training should only use timestamps where ALL required symbols have data
# ============================================================================

def compute_data_intersection(symbols_data: dict, required_symbols: list) -> tuple:
    """
    Find the intersection of timestamps where all required symbols have data.
    
    Args:
        symbols_data: Dict mapping symbol -> DataFrame with DatetimeIndex
        required_symbols: List of symbols that must have data
        
    Returns:
        (common_timestamps, data_stats) where data_stats is a dict with per-symbol info
    """
    data_stats = {}
    
    # Get date ranges for each symbol
    for symbol in required_symbols:
        df = symbols_data.get(symbol)
        if df is not None and not df.empty:
            data_stats[symbol] = {
                'count': len(df),
                'start': df.index.min(),
                'end': df.index.max(),
                'available': True
            }
        else:
            data_stats[symbol] = {
                'count': 0,
                'start': None,
                'end': None,
                'available': False
            }
    
    # Find the intersection (latest start, earliest end)
    available_symbols = [s for s in required_symbols if data_stats[s]['available']]
    
    if not available_symbols:
        return [], data_stats
    
    # Compute the common date range
    latest_start = max(data_stats[s]['start'] for s in available_symbols)
    earliest_end = min(data_stats[s]['end'] for s in available_symbols)
    
    data_stats['_intersection'] = {
        'start': latest_start,
        'end': earliest_end,
        'available_symbols': available_symbols
    }
    
    if latest_start > earliest_end:
        print(f"    [WARN] No overlapping data range found!")
        return [], data_stats
    
    # Get timestamps from primary symbol that fall within intersection
    primary_df = symbols_data.get(required_symbols[0])
    if primary_df is None or primary_df.empty:
        return [], data_stats
    
    # Filter to intersection range
    mask = (primary_df.index >= latest_start) & (primary_df.index <= earliest_end)
    common_times = sorted(primary_df.index[mask])
    
    return common_times, data_stats

# Define required symbols for training
# Core symbols that MUST have data - training only proceeds where these exist
REQUIRED_SYMBOLS = ['SPY', 'QQQ', 'VIX']

# Optional context symbols - used if available, but not required
OPTIONAL_SYMBOLS = ['IWM', 'DIA', 'XLF', 'XLK', 'XLE', 'TLT', 'UUP', 'BTC-USD']

print(f"\n[*] Computing available data intersection...")
print(f"[*] REQUIRED symbols (training only where all exist): {', '.join(REQUIRED_SYMBOLS)}")
print(f"[*] OPTIONAL symbols (used if available): {', '.join(OPTIONAL_SYMBOLS)}")

# Compute intersection of required symbols
common_times, data_stats = compute_data_intersection(symbols_data, REQUIRED_SYMBOLS)

# Print data availability report
print(f"\n[*] Data availability report:")
print(f"    {'Symbol':<12} {'Records':<10} {'Start':<22} {'End':<22}")
print(f"    {'-'*12} {'-'*10} {'-'*22} {'-'*22}")

for symbol in REQUIRED_SYMBOLS + OPTIONAL_SYMBOLS:
    if symbol in data_stats:
        stats = data_stats[symbol]
        if stats['available']:
            print(f"    {symbol:<12} {stats['count']:<10} {str(stats['start']):<22} {str(stats['end']):<22}")
        else:
            print(f"    {symbol:<12} {'N/A':<10} {'No data':<22} {'':<22}")
    elif symbol in symbols_data:
        df = symbols_data[symbol]
        if not df.empty:
            print(f"    {symbol:<12} {len(df):<10} {str(df.index.min()):<22} {str(df.index.max()):<22}")
        else:
            print(f"    {symbol:<12} {'N/A':<10} {'No data':<22} {'':<22}")
    else:
        print(f"    {symbol:<12} {'N/A':<10} {'Not loaded':<22} {'':<22}")

# Print intersection info
if '_intersection' in data_stats:
    inter = data_stats['_intersection']
    print(f"\n[*] Data intersection (where all REQUIRED symbols have data):")
    print(f"    Start: {inter['start']}")
    print(f"    End:   {inter['end']}")
    if inter['start'] and inter['end']:
        duration = inter['end'] - inter['start']
        print(f"    Duration: {duration}")

# Filter by date range if specified via environment variables
import os
start_date_str = os.environ.get('TRAINING_START_DATE')
end_date_str = os.environ.get('TRAINING_END_DATE')

if start_date_str and end_date_str:
    try:
        start_date = pd.to_datetime(start_date_str)
        end_date = pd.to_datetime(end_date_str)
        
        # Filter timestamps to user-specified range
        common_times = [t for t in common_times if start_date <= t <= end_date]
        print(f"\n    [OK] Further filtered to user date range: {start_date_str} to {end_date_str}")
    except Exception as e:
        print(f"    [!] Could not parse date range: {e}")
        print(f"    [*] Using computed intersection range")

print(f"\n[*] Training will ONLY run during market hours (9:30 AM - 4:00 PM ET)")

# Keep backward compatibility
bitx_df = symbols_data.get(trading_symbol, pd.DataFrame())

if len(common_times) == 0:
    print(f"\n    [!] No overlapping data found for required symbols!")
    print(f"    [!] This usually means Tradier's limited 1-min history doesn't overlap")
    print(f"    [*] Try fetching fresh data or using daily data instead")
    
    # Fallback: use primary symbol data only with warning
    if not bitx_df.empty:
        print(f"\n    [FALLBACK] Using {trading_symbol} data only (reduced accuracy)")
        common_times = sorted(bitx_df.index)
    else:
        print(f"    [ERROR] No data available at all!")
        sys.exit(1)

print(f"\n    [OK] {len(common_times)} aligned timestamps for training")
print(f"    Range: {common_times[0]} to {common_times[-1]}")

# Filter to 5-minute steps if using 5m interval (reduces noise, faster training)
if TT_INTERVAL_MINUTES > 1:
    original_count = len(common_times)
    # Keep only timestamps where minute is divisible by interval
    common_times = [t for t in common_times if t.minute % TT_INTERVAL_MINUTES == 0]
    print(f"    [5M] Filtered from {original_count:,} to {len(common_times):,} timestamps ({TT_INTERVAL_MINUTES}-min steps)")
    if len(common_times) > 0:
        print(f"    [5M] New range: {common_times[0]} to {common_times[-1]}")

# Get initial balance from environment, config, or default
env_balance = os.environ.get('TRAINING_INITIAL_BALANCE')
if env_balance:
    initial_balance = float(env_balance)
    print(f"    [CONFIG] Initial balance from env: ${initial_balance:,.2f}")
elif 'initial_balance_from_config' in dir():
    initial_balance = initial_balance_from_config
    print(f"    [CONFIG] Initial balance from config: ${initial_balance:,.2f}")
else:
    initial_balance = 5000.0  # Default to $5000 instead of $1000
    print(f"    [CONFIG] Initial balance (default): ${initial_balance:,.2f}")

# Initialize bot with comprehensive setup
print(f"\n[*] Initializing bot with ${initial_balance:,.2f} training balance...")
logger.info(f"[INIT] Initializing UnifiedOptionsBot with ${initial_balance:,.2f}...")

from unified_options_trading_bot import UnifiedOptionsBot
from backend.experience_replay import TradeExperienceManager

# Try to import trading mode config (optional)
try:
    from trading_mode_config import TradingModeConfig
    HAS_TRADING_MODE_CONFIG = True
except ImportError:
    HAS_TRADING_MODE_CONFIG = False
    logger.warning("âš ï¸ Could not initialize config system: trading_mode_config not available")

# Check if running in shadow trading mode (places trades in Tradier Sandbox too)
use_shadow_trading = os.environ.get('USE_SHADOW_TRADING', '0') == '1'
tradier_shadow = None

if use_shadow_trading:
    print("[*] Shadow trading enabled - will mirror trades to Tradier Sandbox")
    try:
        from tradier_credentials import TradierCredentials
        from backend.tradier_trading_system import TradierTradingSystem
        
        creds = TradierCredentials()
        sandbox_creds = creds.get_sandbox_credentials()
        
        if sandbox_creds:
            # Create a Tradier shadow trading system (try multiple signatures for version compatibility)
            shadow_initialized = False
            
            # Method 1: Try simple sandbox-only signature (SSH server version)
            try:
                tradier_shadow = TradierTradingSystem(sandbox=True)
                print(f"[âœ“] Shadow trading initialized with auto-credentials")
                shadow_initialized = True
            except (TypeError, AttributeError) as e:
                logger.debug(f"Auto-credentials signature failed: {e}")
            
            # Method 2: Try with explicit credentials (local version)
            if not shadow_initialized:
                try:
                    tradier_shadow = TradierTradingSystem(
                        account_id=sandbox_creds['account_number'],
                        api_token=sandbox_creds['access_token'],
                        sandbox=True,
                        initial_balance=100000.0,
                        db_path='data/shadow_trading.db'
                    )
                    print(f"[âœ“] Shadow trading initialized with explicit credentials")
                    shadow_initialized = True
                except (TypeError, AttributeError) as e:
                    logger.debug(f"Explicit credentials signature failed: {e}")
            
            if shadow_initialized:
                print(f"[âœ“] Account: {sandbox_creds['account_number']}")
                print(f"[âœ“] Simulation trades will ALSO be placed in Tradier Sandbox")
                print(f"[âœ“] Check https://tradier.com/sandbox to see live trades!")
                logger.info(f"[SHADOW] Shadow trading enabled: {sandbox_creds['account_number']}")
            else:
                print("[!] Could not initialize shadow trading with any known signature")
                print("[*] Continuing with paper trading only")
                tradier_shadow = None
        else:
            print("[!] Shadow trading requested but no sandbox credentials found")
            print("[*] Continuing with paper trading only")
    except Exception as e:
        print(f"[!] Could not initialize shadow trading: {e}")
        print("[*] Continuing with paper trading only")
        tradier_shadow = None
else:
    print("[*] Paper trading mode - using in-memory simulation only")

# Initialize experience replay buffer for advanced learning (fresh each run)
experience_manager = TradeExperienceManager(max_size=10000)
logger.info("[OK] Experience Replay Buffer initialized (max 10k experiences, fresh buffer)")

# Map trade_id -> signal dict for better learning attribution (helps prioritized replay + RL updates)
signal_by_trade_id = {}

# ---------------------------------------------------------------------------
# MISSED-PLAY TRACKING (report-only; no learning)
# ---------------------------------------------------------------------------
# We track skipped opportunities (HOLD or gated-out signals) and score them after
# a fixed horizon (e.g., 15 minutes) using the realized underlying move.
missed_tracking = {
    "enabled": True,
    "horizon_minutes": 15,
    "min_confidence": 0.25,
    "min_abs_predicted_return": 0.0015,  # 0.15% underlying move
    # Minimum realized move to call something a "win" after costs/slippage.
    # This is on underlying, not option premium; used only for missed-play scoring.
    "min_realized_move_to_count": 0.0010,  # 0.10%
}

# ----------------------------
# Decision diagnostics (why trades are skipped)
# ----------------------------
decision_stats = {
    "signals_total": 0,
    "signals_non_hold": 0,
    "bandit_follow": 0,
    "bandit_skip_conf": 0,
    "bandit_skip_edge": 0,
    "bandit_skip_both": 0,
    "bandit_skip_other": 0,
    "fullrl_follow": 0,
    "fullrl_skip": 0,
    "blocked_max_positions": 0,
    "trade_failed": 0,
    "trade_placed": 0,
}

def _log_decision_stats(prefix: str = "[DIAG]") -> None:
    try:
        non_hold = int(decision_stats.get("signals_non_hold", 0))
        bandit_total = int(decision_stats.get("bandit_follow", 0)) + int(decision_stats.get("bandit_skip_conf", 0)) + int(decision_stats.get("bandit_skip_edge", 0)) + int(decision_stats.get("bandit_skip_both", 0)) + int(decision_stats.get("bandit_skip_other", 0))
        msg = (
            f"{prefix} decisions: non_hold={non_hold} | "
            f"bandit: follow={decision_stats.get('bandit_follow', 0)}, "
            f"skip_conf={decision_stats.get('bandit_skip_conf', 0)}, "
            f"skip_edge={decision_stats.get('bandit_skip_edge', 0)}, "
            f"skip_both={decision_stats.get('bandit_skip_both', 0)}, "
            f"skip_other={decision_stats.get('bandit_skip_other', 0)} (total={bandit_total}) | "
            f"fullrl: follow={decision_stats.get('fullrl_follow', 0)}, skip={decision_stats.get('fullrl_skip', 0)} | "
            f"exec: placed={decision_stats.get('trade_placed', 0)}, failed={decision_stats.get('trade_failed', 0)}, "
            f"blocked_maxpos={decision_stats.get('blocked_max_positions', 0)}"
        )
        print(msg)
        logger.info(msg)
    except Exception:
        pass

pending_missed = []  # list of dicts: {t, price, conf, pred_ret, action, reason}
missed_stats = {
    "tracked": 0,
    "evaluated": 0,
    "missed_winners": 0,
    "avoided_losers": 0,
    "neutral": 0,
    "by_reason": {},  # reason -> counts
    "by_action": {},  # BUY_CALLS/BUY_PUTS -> counts
}

# ---------------------------------------------------------------------------
# TRADABILITY GATE DATASET (supervised dataset generation)
# ---------------------------------------------------------------------------
# We generate a labeled dataset for a "tradability gate" model:
# - Inputs: predictor return/confidence + market context + regime
# - Label: 1 if realized move in the proposed direction exceeds a threshold over horizon, else 0
gate_dataset = {
    "enabled": True,
    "path": None,  # set after run_dir exists
    "horizon_minutes": 15,
    "min_realized_move_to_count": 0.0010,  # 0.10% underlying move
    "pending": [],  # list of dicts with features at time t
    "stats": {
        "collected": 0,
        "evaluated": 0,
        "positive": 0,
        "negative": 0,
        "dropped_no_future_price": 0,
    },
}

# Allow disabling tradability gate via environment variable (for testing)
if os.environ.get("TT_DISABLE_TRADABILITY_GATE"):
    gate_dataset["enabled"] = False
    logger.info("[GATE] Tradability gate DISABLED via TT_DISABLE_TRADABILITY_GATE env var")

# ============================================================================
# UNIFIED RL POLICY (replaces 3 separate systems)
# ============================================================================
unified_rl = None
rules_policy = None
q_entry_controller = None

try:
    from backend.unified_rl_policy import UnifiedRLPolicy, TradeState, AdaptiveRulesPolicy, compute_condor_regime_features, CONDOR_FEATURES_ENABLED
    
    # Use unified RL policy (single system for entry + exit)
    # TRAINING MODE: Lower thresholds to generate more trades for learning
    unified_rl = UnifiedRLPolicy(
        learning_rate=0.0003,
        gamma=0.99,
        device='cpu',
        bandit_mode_trades=100000,  # Use bandit mode for first 100 trades
        min_confidence_to_trade=0.15,  # VERY LOW for training (need trades to learn!)
        max_position_loss_pct=0.15,   # -15% stop loss
        profit_target_pct=0.20,       # +20% take profit (lower to lock in gains faster)
    )
    
    # Also create rule-based policy as comparison
    rules_policy = AdaptiveRulesPolicy()
    
    logger.info("[NEW] UNIFIED RL Policy initialized (single system for entry+exit)")
    logger.info(f"   Bandit mode: First 50 trades")
    logger.info(f"   Min confidence: 25%, Stop: -15%, Target: +25%")
    
    USE_UNIFIED_RL = True
except ImportError as e:
    USE_UNIFIED_RL = False
    logger.warning(f"[WARN] Unified RL not available: {e}")
    logger.warning("[WARN] Falling back to legacy RL systems")

# ============================================================================
# V2 ARCHITECTURE (Enhanced RL Policy with embedding + GRU + EV gating)
# ============================================================================
USE_RL_POLICY_V2 = os.environ.get('USE_RL_POLICY_V2', '0') == '1'
unified_rl_v2 = None
TradeStateV2 = None

if USE_RL_POLICY_V2:
    try:
        from backend.unified_rl_policy_v2 import UnifiedRLPolicyV2, TradeStateV2 as _TradeStateV2
        TradeStateV2 = _TradeStateV2

        unified_rl_v2 = UnifiedRLPolicyV2(
            learning_rate=0.0003,
            gamma=0.99,
            device='cpu',
            bandit_mode_trades=100000,  # Use bandit mode for first 100 trades
        )

        logger.info("[V2] Enhanced RL Policy V2 initialized!")
        logger.info("   State: 40 features (includes 16-dim embedding)")
        logger.info("   Actions: 5 (HOLD, BUY_CALL, BUY_PUT, EXIT_FAST, EXIT_PATIENT)")
        logger.info("   Entry gating: EV-based (friction-aware)")
        logger.info("   Temporal: GRU over last 10 states")
    except ImportError as e:
        USE_RL_POLICY_V2 = False
        logger.warning(f"[V2] V2 policy not available: {e}")
        logger.warning("[V2] Falling back to V1 policy")

# ============================================================================
# OPTIONAL: Q-SCORER ENTRY CONTROLLER (offline-supervised)
# ============================================================================
HAS_Q_ENTRY_CONTROLLER = False
try:
    from backend.q_entry_controller import QEntryController

    q_entry_controller = QEntryController()
    HAS_Q_ENTRY_CONTROLLER = bool(q_entry_controller.load())
    if HAS_Q_ENTRY_CONTROLLER:
        logger.info("[Q] QEntryController loaded (will be used when ENTRY_CONTROLLER=q_scorer)")
except Exception:
    HAS_Q_ENTRY_CONTROLLER = False
    q_entry_controller = None

# Direction-based entry controller (for improved win rate)
direction_controller = None
HAS_DIRECTION_CONTROLLER = False
try:
    from backend.direction_entry_controller import DirectionEntryController
    direction_controller = DirectionEntryController(feature_dim=50)
    HAS_DIRECTION_CONTROLLER = True
    logger.info("[DIR] DirectionEntryController loaded (use ENTRY_CONTROLLER=direction)")
except Exception as e:
    HAS_DIRECTION_CONTROLLER = False
    direction_controller = None
    logger.warning(f"[DIR] DirectionEntryController not available: {e}")

# ============================================================================
# V3 DIRECTION PREDICTOR (simplified architecture, 56% val accuracy)
# ============================================================================
v3_controller = None
HAS_V3_CONTROLLER = False
try:
    from backend.v3_entry_controller import get_v3_controller, decide_from_signal as v3_decide
    v3_controller = get_v3_controller()
    if v3_controller.model is not None:
        HAS_V3_CONTROLLER = True
        logger.info("[V3] V3EntryController loaded (use ENTRY_CONTROLLER=v3_direction)")
    else:
        logger.warning("[V3] V3EntryController model not loaded")
except Exception as e:
    HAS_V3_CONTROLLER = False
    v3_controller = None
    logger.warning(f"[V3] V3EntryController not available: {e}")

# ============================================================================
# CONSENSUS ENTRY CONTROLLER (multi-signal agreement)
# ============================================================================
consensus_controller = None
HAS_CONSENSUS_CONTROLLER = False
try:
    from backend.consensus_entry_controller import ConsensusEntryController, decide_from_signal as consensus_decide
    consensus_controller = ConsensusEntryController()
    HAS_CONSENSUS_CONTROLLER = True
    logger.info("[CONSENSUS] ConsensusEntryController loaded (use ENTRY_CONTROLLER=consensus)")
except Exception as e:
    HAS_CONSENSUS_CONTROLLER = False
    consensus_controller = None
    logger.warning(f"[CONSENSUS] ConsensusEntryController not available: {e}")

# ============================================================================
# ADAPTIVE LEARNING COMPONENTS (from output architecture)
# ============================================================================
# These run alongside the main entry controller as additional quality filters

rl_threshold_learner = None
rl_exit_policy = None

# Load adaptive learning config (from _full_config loaded at top of file)
adaptive_learning_cfg = _full_config.get("adaptive_learning", {})
rl_threshold_cfg = adaptive_learning_cfg.get("rl_threshold_learner", {})
adaptive_weights_cfg = adaptive_learning_cfg.get("adaptive_timeframe_weights", {})

# Initialize RLThresholdPolicy if enabled (works alongside main entry controller)
if rl_threshold_cfg.get("enabled", False):
    try:
        from backend.rl_threshold_learner import RLThresholdPolicy
        rl_threshold_learner = RLThresholdPolicy(
            learning_rate=rl_threshold_cfg.get("learning_rate", 0.003),
            base_threshold=rl_threshold_cfg.get("base_threshold", 0.25),
            threshold_floor=rl_threshold_cfg.get("threshold_floor", 0.15),
            threshold_ceiling=rl_threshold_cfg.get("threshold_ceiling", 0.50),
            rl_delta_max=rl_threshold_cfg.get("rl_delta_max", 0.05)
        )
        USE_RL_THRESHOLD_FILTER = rl_threshold_cfg.get("use_as_filter", True)
        logger.info("[ADAPTIVE] RL Threshold Learner initialized")
        logger.info(f"   Base threshold: {rl_threshold_cfg.get('base_threshold', 0.25)}")
        logger.info(f"   Use as filter: {USE_RL_THRESHOLD_FILTER}")

        # Load pretrained weights if available (from protected directory)
        if os.environ.get('LOAD_PRETRAINED', '0') == '1':
            pretrained_path = "models/pretrained/rl_threshold.pth"  # Protected directory
            if os.path.exists(pretrained_path):
                try:
                    rl_threshold_learner.load(pretrained_path)
                    logger.info(f"[PRETRAINED] Loaded RL threshold learner from {pretrained_path}")
                    logger.info(f"   Composite threshold: {rl_threshold_learner.composite_threshold:.4f}")
                except Exception as e_load:
                    logger.warning(f"[WARN] Failed to load pretrained RL threshold: {e_load}")
            else:
                logger.info(f"[PRETRAINED] No RL threshold file at {pretrained_path} (OK - not required)")
    except ImportError as e:
        rl_threshold_learner = None
        USE_RL_THRESHOLD_FILTER = False
        logger.warning(f"[WARN] RL threshold learner not available: {e}")
else:
    USE_RL_THRESHOLD_FILTER = False
    logger.info("[ADAPTIVE] RL Threshold Learner disabled in config")

# Legacy RL exit policy (only if unified RL not available)
if not USE_UNIFIED_RL:
    try:
        from backend.rl_exit_policy import RLExitPolicy
        rl_exit_policy = RLExitPolicy(learning_rate=0.001)
        logger.info("[LEGACY] RL exit policy initialized")
    except ImportError:
        rl_exit_policy = None
        logger.info("[WARN] RL exit policy not available")

# CRITICAL: Delete old models/state directory to ensure fresh start
# The bot's __init__ tries to load from hardcoded "models/state/" paths
# We must delete this BEFORE creating the bot to prevent loading old models
# Skip if LOAD_PRETRAINED=1 to allow loading pre-trained state
import shutil
from pathlib import Path
old_state_dir = Path("models/state")
if os.environ.get('LOAD_PRETRAINED', '0') == '1':
    logger.info(f"[PRETRAINED] Keeping models/state for pre-trained state loading")
elif old_state_dir.exists():
    try:
        shutil.rmtree(old_state_dir)
        logger.info(f"[CLEANUP] Deleted old models/state directory for fresh start")
    except Exception as e:
        logger.warning(f"[WARN] Could not delete models/state: {e}")
else:
    logger.info(f"[OK] No old models/state directory found")

# START FRESH EACH RUN - Models saved to timestamped directory
run_dir = os.environ.get('MODEL_RUN_DIR')
run_timestamp = os.environ.get('MODEL_RUN_TIMESTAMP', datetime.now().strftime('%Y%m%d_%H%M%S'))

if run_dir:
    logger.info(f"[NN] Using timestamped model directory: {run_dir}")
    logger.info(f"[NN] Starting with fresh neural network for this run")
else:
    run_dir = f"models/run_{run_timestamp}"
    logger.warning(f"[WARN] MODEL_RUN_DIR not set, using default: {run_dir}")

# Create state directory inside run directory
os.makedirs(os.path.join(run_dir, 'state'), exist_ok=True)

# Initialize heartbeat for dashboard tracking
_heartbeat = None
if HEARTBEAT_AVAILABLE:
    try:
        # Collect env vars that affect the experiment
        _env_vars_for_heartbeat = {
            k: v for k, v in os.environ.items()
            if k.startswith(('TT_', 'HARD_', 'HMM_', 'PREDICTOR_', 'TEMPORAL_', 'LOAD_', 'ENTRY_'))
        }
        _run_name = os.path.basename(run_dir)
        _heartbeat = ExperimentHeartbeat(
            run_name=_run_name,
            target_cycles=TT_MAX_CYCLES,
            env_vars=_env_vars_for_heartbeat
        )
        _heartbeat.start()
        logger.info(f"[HEARTBEAT] Registered experiment: {_run_name}")
    except Exception as e:
        logger.warning(f"[HEARTBEAT] Failed to initialize: {e}")
        _heartbeat = None

# Load bot config and override predictor checkpoint path for this run.
# We want BOTH:
# - per-run checkpoint: models/run_YYYYMMDD_HHMMSS/state/predictor_v2.pt (unique each run)
# - stable latest:      models/predictor_v2.pt (updated by bot.save_model export)
try:
    from config_loader import load_config
    # Load output3/config.json regardless of current working directory.
    _cfg_path = Path(__file__).resolve().parents[1] / "config.json"
    bot_config = load_config(str(_cfg_path))
    bot_config.config.setdefault("architecture", {}).setdefault("predictor", {})
    bot_config.config["architecture"]["predictor"]["checkpoint_path"] = os.path.join(
        run_dir, "state", "predictor_v2.pt"
    )
    logger.info(f"[NN] Predictor checkpoint (this run): {bot_config.config['architecture']['predictor']['checkpoint_path']}")

    # Training quality gates (prevent trading noise that can't beat spread/fees)
    bot_config.config.setdefault("architecture", {}).setdefault("entry_policy", {})
    entry_cfg = bot_config.config["architecture"]["entry_policy"]
    # Defaults tuned for optimal selectivity (Phase 26 optimization: -1.8% vs -62% baseline)
    # Higher thresholds = fewer trades = less losses. Sweet spot: 30% conf, 0.13% edge
    entry_cfg.setdefault("training_min_confidence", 0.30)              # 30% (was 15%)
    entry_cfg.setdefault("training_min_abs_predicted_return", 0.0013)  # 0.13% (was 0.05%)

    # Allow env overrides (fast iteration without editing config.json)
    try:
        env_min_conf = os.environ.get("TT_TRAIN_MIN_CONF", "").strip()
        env_min_abs = os.environ.get("TT_TRAIN_MIN_ABS_RET", "").strip()
        if env_min_conf:
            entry_cfg["training_min_confidence"] = float(env_min_conf)
        if env_min_abs:
            entry_cfg["training_min_abs_predicted_return"] = float(env_min_abs)
    except Exception:
        pass
    logger.info(f"[TRAIN] Entry gates: min_conf={entry_cfg['training_min_confidence']:.0%}, "
                f"min_abs_return={entry_cfg['training_min_abs_predicted_return']*100:.2f}%")

    # Missed-play tracking settings (report-only)
    try:
        missed_tracking["min_confidence"] = float(entry_cfg.get("training_min_confidence", missed_tracking["min_confidence"]))
        missed_tracking["min_abs_predicted_return"] = float(entry_cfg.get("training_min_abs_predicted_return", missed_tracking["min_abs_predicted_return"]))
        hz = bot_config.config.get("architecture", {}).get("horizons", {}).get("prediction_minutes", missed_tracking["horizon_minutes"])
        missed_tracking["horizon_minutes"] = int(hz)
        logger.info(f"[TRAIN] Missed-play tracking: horizon={missed_tracking['horizon_minutes']}m, "
                    f"min_conf={missed_tracking['min_confidence']:.0%}, "
                    f"min_abs_return={missed_tracking['min_abs_predicted_return']*100:.2f}%")
    except Exception as e:
        logger.warning(f"[WARN] Could not apply missed-play tracking config: {e}")

    # Gate dataset config (labeling horizon + realized-move threshold)
    try:
        gate_dataset["horizon_minutes"] = int(bot_config.config.get("architecture", {}).get("horizons", {}).get("prediction_minutes", gate_dataset["horizon_minutes"]))
        gate_cfg = bot_config.config.get("architecture", {}).get("tradability_gate", {})
        if "min_realized_move_to_count" in gate_cfg:
            gate_dataset["min_realized_move_to_count"] = float(gate_cfg.get("min_realized_move_to_count", gate_dataset["min_realized_move_to_count"]))
        logger.info(f"[TRAIN] Tradability-gate dataset: horizon={gate_dataset['horizon_minutes']}m, "
                    f"label_move_thresh={gate_dataset['min_realized_move_to_count']*100:.2f}%")
    except Exception as e:
        logger.warning(f"[WARN] Could not apply tradability gate dataset config: {e}")
except Exception as e:
    bot_config = None
    logger.warning(f"[WARN] Could not load BotConfig for predictor arch/checkpoint: {e}")

# Create bot (pass config so predictor arch + checkpoint_path are honored)
bot = UnifiedOptionsBot(initial_balance=initial_balance, config=bot_config)

# Set run_id on paper trader for trade tracking in dashboard
run_name = os.path.basename(run_dir)
if hasattr(bot, "paper_trader") and bot.paper_trader is not None:
    bot.paper_trader.set_run_id(run_name)
    logger.info(f"[TRAIN] Paper trader run_id set to: {run_name}")

    # Override stop_loss and take_profit from env vars (takes precedence over config.json)
    if os.environ.get('TT_STOP_LOSS_PCT'):
        sl_pct = float(os.environ['TT_STOP_LOSS_PCT']) / 100.0
        bot.paper_trader.stop_loss_pct = sl_pct
        logger.info(f"[TRAIN] Forced stop_loss to {sl_pct:.1%} from TT_STOP_LOSS_PCT")
    if os.environ.get('TT_TAKE_PROFIT_PCT'):
        tp_pct = float(os.environ['TT_TAKE_PROFIT_PCT']) / 100.0
        bot.paper_trader.take_profit_pct = tp_pct
        logger.info(f"[TRAIN] Forced take_profit to {tp_pct:.1%} from TT_TAKE_PROFIT_PCT")

# Phase 44: Initialize ensemble predictor and signal generators
phase44_ensemble = None
phase44_gex_signals = None
phase44_order_flow = None
phase44_multi_indicator = None

if ENSEMBLE_AVAILABLE and os.environ.get('ENSEMBLE_ENABLED', '0') == '1':
    try:
        phase44_ensemble = create_ensemble_predictor(feature_dim=50)
        # Load pre-trained weights if available
        if os.environ.get('ENSEMBLE_PRETRAINED', '0') == '1':
            pretrained_path = os.environ.get('ENSEMBLE_MODEL_PATH', 'models/pretrained_ensemble.pkl')
            if os.path.exists(pretrained_path):
                import pickle
                with open(pretrained_path, 'rb') as f:
                    saved = pickle.load(f)
                phase44_ensemble.tcn.load_state_dict(saved['tcn_state'])
                phase44_ensemble.lstm.load_state_dict(saved['lstm_state'])
                phase44_ensemble.attention.load_state_dict(saved['attention_state'])
                phase44_ensemble.xgb_model = saved['xgb_model']
                phase44_ensemble.xgb_scaler = saved['xgb_scaler']
                phase44_ensemble.meta_learner = saved['meta_learner']
                phase44_ensemble.meta_scaler = saved['meta_scaler']
                phase44_ensemble.is_fitted = saved['is_fitted']
                logger.info(f"[PHASE44] Loaded pre-trained ensemble from {pretrained_path}")
            else:
                logger.warning(f"[PHASE44] Pre-trained ensemble not found at {pretrained_path}")
        else:
            logger.info("[PHASE44] Ensemble predictor initialized (untrained)")
    except Exception as e:
        logger.warning(f"[PHASE44] Failed to create ensemble predictor: {e}")

if ENSEMBLE_AVAILABLE and os.environ.get('GEX_SIGNALS_ENABLED', '0') == '1':
    try:
        phase44_gex_signals = GammaExposureSignals()
        logger.info("[PHASE44] GEX signal generator initialized")
    except Exception as e:
        logger.warning(f"[PHASE44] Failed to create GEX signals: {e}")

if ENSEMBLE_AVAILABLE and os.environ.get('ORDER_FLOW_ENABLED', '0') == '1':
    try:
        phase44_order_flow = OrderFlowSignals()
        logger.info("[PHASE44] Order flow signal generator initialized")
    except Exception as e:
        logger.warning(f"[PHASE44] Failed to create order flow signals: {e}")

if ENSEMBLE_AVAILABLE and os.environ.get('MULTI_INDICATOR_ENABLED', '0') == '1':
    try:
        phase44_multi_indicator = MultiIndicatorStack()
        logger.info("[PHASE44] Multi-indicator stack initialized")
    except Exception as e:
        logger.warning(f"[PHASE44] Failed to create multi-indicator stack: {e}")

# Training capacity: allow more concurrent paper positions so learning isn't blocked by max_positions.
try:
    train_max_positions = int(os.environ.get("TT_TRAIN_MAX_POSITIONS", "10"))
    if train_max_positions > 0:
        bot.max_positions = train_max_positions
        if hasattr(bot, "paper_trader") and hasattr(bot.paper_trader, "max_positions"):
            bot.paper_trader.max_positions = train_max_positions
        # IMPORTANT: RiskManager has its own max concurrent position cap (default=5).
        # If we don't lift it in training, it will hard-block new trades even when
        # paper_trader.max_positions is higher.
        if hasattr(bot, "paper_trader") and getattr(bot.paper_trader, "risk_manager", None) is not None:
            try:
                rm = bot.paper_trader.risk_manager
                rm.limits.max_concurrent_positions = max(int(getattr(rm.limits, "max_concurrent_positions", 0) or 0), int(train_max_positions))
                logger.info(f"[TRAIN] RiskManager max_concurrent_positions set to {rm.limits.max_concurrent_positions} (training override)")
            except Exception:
                pass
        logger.info(f"[TRAIN] Max positions set to {train_max_positions} for time-travel training")
except Exception:
    pass

# TRAINING OVERRIDES (time-travel only)
# ------------------------------------
# Default risk limits (2% max loss per trade) are too small for 1-contract SPY options on a ~$5k account,
# so RiskManager frequently blocks with "Position too small", starving RL of experiences.
# In time-travel training we relax these limits to ensure we can actually take trades and learn.
try:
    relax = os.environ.get("TT_RELAX_RISK", "1").strip() == "1"
    train_max_risk_pct = float(os.environ.get("TT_TRAIN_MAX_RISK_PCT", "0.10"))   # 10% per-trade risk cap (training only)
    train_max_pos_pct = float(os.environ.get("TT_TRAIN_MAX_POS_PCT", "0.20"))     # 20% position size cap (training only)
    if relax and hasattr(bot, "paper_trader") and getattr(bot.paper_trader, "risk_manager", None) is not None:
        rm = bot.paper_trader.risk_manager
        rm.limits.max_risk_per_trade_pct = max(rm.limits.max_risk_per_trade_pct, train_max_risk_pct)
        rm.limits.max_position_size_pct = max(rm.limits.max_position_size_pct, train_max_pos_pct)
        # Also reduce cold-start suppression in training so sizing doesn't get stuck at tiny values.
        rm.limits.cold_start_position_scale = max(rm.limits.cold_start_position_scale, 0.75)
        logger.info(
            f"[TRAIN] Relaxed RiskManager limits for time-travel: "
            f"max_risk_per_trade_pct={rm.limits.max_risk_per_trade_pct:.0%}, "
            f"max_position_size_pct={rm.limits.max_position_size_pct:.0%}, "
            f"cold_start_position_scale={rm.limits.cold_start_position_scale:.0%}"
        )
except Exception as e:
    logger.warning(f"[WARN] Could not apply training risk overrides: {e}")

# SHORT TIMEFRAMES: Focus on intraday momentum (no multi-hour noise)
# Weight short-term higher (more reliable), long-term for trend confirmation
bot.prediction_timeframes = {
    "15min": {"minutes": 15, "weight": 0.35},    # Primary signal (highest weight)
    "30min": {"minutes": 30, "weight": 0.30},    # Confirmation
    "1hour": {"minutes": 60, "weight": 0.20},    # Trend direction
    "2hour": {"minutes": 120, "weight": 0.15},   # Longer confirm (lowest weight)
}
logger.info("[TIMEFRAMES] Short intraday: 15min(35%)â†’30min(30%)â†’1h(20%)â†’2h(15%)")

import torch
# GPU acceleration enabled - use all available resources
# torch.set_num_threads(4)  # Removed CPU-only limitation

# Verify fresh start (Path already imported above)
models_path = Path(run_dir) / "state" / "trained_model.pth"
if models_path.exists():
    logger.warning(f"   [WARN] Model file already exists at {models_path}")
    logger.warning(f"   [WARN] This shouldn't happen for a new run!")
else:
    logger.info(f"   [OK] No existing model file - starting fresh run")

if hasattr(bot, 'model') and bot.model is not None:
    # If model was somehow loaded, this is a problem
    logger.warning("   [WARN] Model was loaded - checking if it's from old files...")
    num_params = sum(p.numel() for p in bot.model.parameters())
    logger.info(f"   [INFO] Model has {num_params:,} parameters")
    
    if hasattr(bot, 'optimizer') and bot.optimizer is not None:
        logger.info("   [INFO] Optimizer exists")
    else:
        # Create optimizer if missing
        bot.optimizer = torch.optim.Adam(bot.model.parameters(), lr=0.001)
        logger.info("   [INFO] Optimizer created")
else:
    logger.info("   [OK] Starting with brand new neural network")

# Feature buffer status
if hasattr(bot, 'feature_buffer'):
    buffer_size = len(bot.feature_buffer) if bot.feature_buffer else 0
    if buffer_size > 0:
        logger.warning(f"   [WARN] Feature buffer has {buffer_size} features - should be empty for fresh start")
    else:
        logger.info(f"   [OK] Feature buffer is empty (fresh start)")

logger.info("[TRAIN] Each run is isolated in its own timestamped directory")

# Load pretrained model if available (BEFORE overwriting save paths!)
# CRITICAL: Use models/pretrained/ directory which is never overwritten by training runs
if os.environ.get('LOAD_PRETRAINED', '0') == '1':
    # Allow override via PRETRAINED_MODEL_PATH env var
    pretrained_model_path = os.environ.get('PRETRAINED_MODEL_PATH', "models/pretrained/trained_model.pth")
    if os.path.exists(pretrained_model_path):
        # Force model initialization if not already done
        # Auto-detect feature_dim from the checkpoint
        if not hasattr(bot, 'model') or bot.model is None:
            from bot_modules.neural_networks import create_predictor
            logger.info(f"[PRETRAINED] Forcing model initialization for pretrained loading...")
            predictor_arch = "v2_slim_bayesian"  # Default architecture

            # Auto-detect feature_dim from checkpoint
            pretrained_feature_dim = 50  # Default fallback
            try:
                checkpoint = torch.load(pretrained_model_path, map_location='cpu')
                # Check if it's a state_dict directly or wrapped
                state_dict = checkpoint if isinstance(checkpoint, dict) and 'temporal_encoder.input_proj.weight' in checkpoint else checkpoint.get('model_state_dict', checkpoint)
                if 'temporal_encoder.input_proj.weight' in state_dict:
                    pretrained_feature_dim = state_dict['temporal_encoder.input_proj.weight'].shape[1]
                    logger.info(f"[PRETRAINED] Auto-detected feature_dim={pretrained_feature_dim} from checkpoint")
            except Exception as e_detect:
                logger.warning(f"[PRETRAINED] Could not auto-detect feature_dim: {e_detect}, using default={pretrained_feature_dim}")

            try:
                cfg = bot.config.config if bot.config else {}
                predictor_arch = cfg.get("architecture", {}).get("predictor", {}).get("arch", predictor_arch)
            except Exception:
                pass
            bot.model = create_predictor(
                arch=predictor_arch,
                feature_dim=pretrained_feature_dim,  # Use pretrained dimensions
                sequence_length=bot.sequence_length,
                use_gaussian_kernels=True,
            )
            bot.model.to(bot.device)
            bot.model_initialized = True
            bot.feature_dim = pretrained_feature_dim  # Update bot's feature_dim to match
            logger.info(f"[PRETRAINED] Model created with arch={predictor_arch}, feature_dim={pretrained_feature_dim}")

        try:
            bot.model.load_state_dict(torch.load(pretrained_model_path, map_location=bot.device))
            logger.info(f"[PRETRAINED] Loaded neural network from {pretrained_model_path}")
            logger.info(f"   This model was trained during profitable long_run_20k")
            logger.info(f"   Expected: lower confidences -> ~1-2% trade rate")
        except Exception as e_load:
            logger.warning(f"[WARN] Failed to load pretrained model: {e_load}")
    else:
        logger.info(f"[PRETRAINED] No pretrained model found at {pretrained_model_path}")
        logger.info(f"   Copy from models/long_run_20k/state/trained_model.pth to use pretrained weights")

# Override bot's model save paths to use our timestamped directory
bot.model_save_path = os.path.join(run_dir, "state", "trained_model.pth")
bot.optimizer_save_path = os.path.join(run_dir, "state", "optimizer_state.pth")
bot.feature_buffer_save_path = os.path.join(run_dir, "state", "feature_buffer.pkl")
bot.gaussian_processor_save_path = os.path.join(run_dir, "state", "gaussian_processor.pkl")
bot.learning_state_save_path = os.path.join(run_dir, "state", "learning_state.pkl")
bot.prediction_history_save_path = os.path.join(run_dir, "state", "prediction_history.pkl")
bot.rl_policy_path = os.path.join(run_dir, "rl_trading_policy.pth")

logger.info(f"[SAVE] Models will be saved to: {run_dir}")

# Gate dataset output path (per-run)
try:
    gate_dataset["path"] = os.path.join(run_dir, "state", "tradability_gate_dataset.jsonl")
    # Start fresh each run
    with open(gate_dataset["path"], "w", encoding="utf-8") as f:
        f.write("")  # truncate
    logger.info(f"[GATE] Writing tradability dataset to: {gate_dataset['path']}")
except Exception as e:
    gate_dataset["enabled"] = False
    logger.warning(f"[WARN] Could not initialize tradability dataset file: {e}")

# ---------------------------------------------------------------------------
# DecisionRecord / MissedOpportunityRecord JSONL (offline Q-scorer dataset)
# ---------------------------------------------------------------------------
try:
    from backend.decision_pipeline import DecisionPipeline  # noqa: E402

    decision_records_path = os.path.join(run_dir, "state", "decision_records.jsonl")
    missed_opps_path = os.path.join(run_dir, "state", "missed_opportunities_full.jsonl")

    # Start fresh each run
    with open(decision_records_path, "w", encoding="utf-8") as f:
        f.write("")
    with open(missed_opps_path, "w", encoding="utf-8") as f:
        f.write("")

    decision_pipeline = DecisionPipeline(decisions_path=decision_records_path, missed_path=missed_opps_path)
    logger.info(f"[DATA] Writing DecisionRecords to: {decision_records_path}")
    logger.info(f"[DATA] Writing MissedOpps (realized) to: {missed_opps_path}")
except Exception as e:
    decision_pipeline = None
    decision_records_path = None
    missed_opps_path = None
    logger.warning(f"[WARN] Could not initialize DecisionPipeline JSONL writers: {e}")

# Q-label logging mode for time-travel datasets:
# - "hold_only": write labels only when executed_action==HOLD (smaller file, but often mostly in-trade)
# - "all": write labels for every cycle (best for entry Q training)
# - "flat_only": write labels only when bot is flat (is_in_trade==False)
TT_Q_LABELS = os.environ.get("TT_Q_LABELS", "hold_only").strip().lower()

# Enable simulation mode
bot.simulation_mode = True
bot.disable_live_data_fetch = True
bot.paper_trader.simulation_mode = True
logger.info("[OK] Simulation mode enabled")

# Disable data freshness check
if hasattr(bot, 'max_data_age_minutes'):
    bot.max_data_age_minutes = 999
    logger.info("[OK] Data freshness check disabled")

# Enable time-travel mode (Black-Scholes pricing)
bot.paper_trader.time_travel_mode = True
logger.info("[OK] Time-Travel Mode: ENABLED (Black-Scholes pricing)")

# Disable PDT
bot.paper_trader.pdt_protected = False
bot.paper_trader.day_trade_history = []
bot.paper_trader.day_trades_count = 0
logger.info("[OK] PDT FULLY DISABLED")

# Connect RL systems
if rl_exit_policy:
    bot.paper_trader.rl_exit_policy = rl_exit_policy
    logger.info("[OK] RL Exit Policy connected")

# =============================================================================
# PNL CALIBRATION GATE (Phase 13 Improvement)
# =============================================================================
# CONFIDENCE CALIBRATION SYSTEM (Phase 14 - ENABLED BY DEFAULT)
# =============================================================================
# Problem discovered: Raw confidence doesn't predict outcomes (0.3 conf = 37.5% WR)
# Solution: Online Platt/isotonic calibration to map confidence → P(profit)
#
# The CalibrationTracker learns the true mapping between model confidence and
# actual trade outcomes. Once it has enough samples, it blocks trades where
# the calibrated P(profit) is too low.
#
# ALWAYS learn (even if gate disabled) so we can analyze calibration quality.
PNL_CAL_GATE_ENABLED = os.environ.get('PNL_CAL_GATE', '1') == '1'  # ENABLED by default
PNL_CAL_MIN_PROB = float(os.environ.get('PNL_CAL_MIN_PROB', '0.42'))  # Min calibrated P(profit)
PNL_CAL_MIN_SAMPLES = int(os.environ.get('PNL_CAL_MIN_SAMPLES', '30'))  # Min trades before gating

# Always initialize calibration tracker for learning (even if gate disabled)
pnl_calibration_tracker = None
try:
    from backend.calibration_tracker import CalibrationTracker
    pnl_calibration_tracker = CalibrationTracker(
        direction_horizon=15,
        pnl_horizon=45,  # Match max_hold_minutes
        buffer_size=500,
        min_samples_for_calibration=PNL_CAL_MIN_SAMPLES,
        refit_interval=20,
        use_hybrid=True
    )
    if PNL_CAL_GATE_ENABLED:
        logger.info(f"[OK] PnL Calibration Gate ENABLED (min_prob={PNL_CAL_MIN_PROB:.0%}, min_samples={PNL_CAL_MIN_SAMPLES})")
    else:
        logger.info(f"[OK] PnL Calibration LEARNING (gate disabled, set PNL_CAL_GATE=1 to enable)")
except Exception as e:
    logger.warning(f"[WARN] Could not initialize PnL CalibrationTracker: {e}")
    PNL_CAL_GATE_ENABLED = False

# PARTIAL RESET - Only delete trades from THIS run, preserve history from other runs
logger.info(f"[RESET] Clearing old data for run: {run_name}...")
conn = sqlite3.connect(bot.paper_trader.db_path)
cursor = conn.cursor()

# Delete only trades from THIS run (preserve other runs for dashboard history)
# First check if run_id column exists (backwards compatibility)
cursor.execute("PRAGMA table_info(trades)")
columns = [col[1] for col in cursor.fetchall()]
if 'run_id' in columns:
    cursor.execute("DELETE FROM trades WHERE run_id = ?", (run_name,))
    deleted_trades = cursor.rowcount
    logger.info(f"   Deleted {deleted_trades} old trades from run {run_name}")
else:
    logger.info(f"   [SKIP] run_id column not in trades table - no per-run cleanup")

# Reset account state
cursor.execute("""
    UPDATE account_state 
    SET balance = ?, 
        total_profit_loss = 0,
        total_trades = 0,
        winning_trades = 0,
        losing_trades = 0
""", (initial_balance,))

conn.commit()
conn.close()

# Clear in-memory state
bot.paper_trader.active_trades = []
bot.paper_trader.total_trades = 0
bot.paper_trader.winning_trades = 0
bot.paper_trader.losing_trades = 0
bot.paper_trader.total_profit_loss = 0
bot.paper_trader.current_balance = initial_balance
bot.paper_trader.initial_balance = initial_balance

# CRITICAL: Disable stale position recovery during time-travel
# This prevents reload of old trades from database during simulation
bot.paper_trader._update_cycle_count = 999999  # Skip recovery check
logger.info("[OK] Stale position recovery DISABLED for time-travel")

logger.info(f"[OK] Full reset complete - Starting fresh with ${initial_balance:,.2f}")

# Force close any remaining active trades (shouldn't be any, but just in case)
if len(bot.paper_trader.active_trades) > 0:
    logger.warning(f"[WARN] Found {len(bot.paper_trader.active_trades)} stuck trades - force closing...")
    bot.paper_trader.active_trades = []
    logger.info(f"[OK] All stuck trades cleared")

bot.training_enabled = True
bot.learning_enabled = True

# TRAINING MODE: Use very low confidence threshold to allow more trades/learning
# TCN typically outputs 25-35% confidence, so we need threshold below that
trading_config = config.get('trading', {})
config_min_confidence = trading_config.get('min_confidence_threshold', 0.20)
# Force lower threshold during training to ensure trades happen
TRAINING_CONFIDENCE_THRESHOLD = 0.20  # 20% - allows most signals through
bot.min_confidence_threshold = min(config_min_confidence, TRAINING_CONFIDENCE_THRESHOLD)
bot.base_confidence_threshold = bot.min_confidence_threshold

logger.info("[CONFIG] Bot Configuration:")
logger.info(f"   Balance: ${bot.paper_trader.current_balance:,.2f}")
logger.info(f"   Max positions: {bot.paper_trader.max_positions}")
logger.info(f"   Confidence threshold: {bot.min_confidence_threshold:.0%}")
logger.info(f"   Learning enabled: {bot.learning_enabled}")
logger.info(f"   RL Policy: {'ENABLED' if bot.use_rl_policy else 'DISABLED'}")

print(f"\n    [OK] Bot initialized")
print(f"    [OK] Log file: {log_file}")
print(f"    [OK] Dashboard: watch_training_dashboard.py")


# Create a custom data source that serves historical data
class TimeTravelDataSource:
    """Serves historical data at specific timestamps"""
    
    def __init__(self, symbols_data, primary_symbol='BITX'):
        self.symbols_data = symbols_data
        self.primary_symbol = primary_symbol
        self.bitx_df = symbols_data.get(primary_symbol, pd.DataFrame())  # Backward compat
        self.vix_df = symbols_data.get('VIX', pd.DataFrame())    # Backward compat
        self.current_time = None
        self.cache_db_path = 'data/market_data_cache.db'
    
    def set_time(self, timestamp):
        """Set the 'current' time for data retrieval"""
        self.current_time = timestamp
    
    def get_data(self, symbol, period='1d', interval='1m'):
        """Return historical data up to current_time

        Supports intervals: '1m' (default), '5m' (resampled from 1m data)
        """
        # Normalize symbol (handle ^VIX -> VIX)
        if symbol.startswith('^'):
            symbol = symbol[1:]

        df = self.symbols_data.get(symbol, pd.DataFrame())

        if df.empty or self.current_time is None:
            return pd.DataFrame()

        # Convert current_time to pandas Timestamp if needed for proper comparison
        current_ts = pd.Timestamp(self.current_time)
        try:
            mask = df.index <= current_ts
            # For 5m interval, get more raw data to resample (3000 1m bars = 600 5m bars)
            tail_count = 3000 if interval == '5m' else 600
            recent = df[mask].tail(tail_count).copy()
        except Exception as e:
            # If comparison fails (e.g., timezone mismatch), return empty
            logger.debug(f"Error filtering data by time: {e}")
            return pd.DataFrame()

        # Normalize column names to lowercase (bot expects lowercase)
        recent.columns = [col.lower() for col in recent.columns]

        # Resample to 5-minute bars if requested
        if interval == '5m' and not recent.empty:
            try:
                # Resample OHLCV data properly
                resampled = recent.resample('5min').agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).dropna()
                recent = resampled.tail(600)  # Return up to 600 5-min bars
            except Exception as e:
                logger.debug(f"Error resampling to 5m: {e}")
                # Fall back to 1m data if resampling fails

        return recent
    
    def get_current_price(self, symbol):
        """Get price at current_time"""
        # Normalize symbol (handle ^VIX -> VIX)
        if symbol.startswith('^'):
            symbol = symbol[1:]
        
        df = self.symbols_data.get(symbol, pd.DataFrame())
        
        if not df.empty and self.current_time is not None:
            # Try exact match first
            if self.current_time in df.index:
                price = df.loc[self.current_time, 'Close']
                
                # Handle duplicate timestamps
                if isinstance(price, pd.Series):
                    price = price.iloc[-1]
                
                return float(price)
            
            # If no exact match, use nearest timestamp (forward fill)
            # Get all data up to current_time
            mask = df.index <= pd.Timestamp(self.current_time)
            if mask.any():
                recent = df[mask]
                if not recent.empty:
                    price = recent['Close'].iloc[-1]
                    return float(price)
            
            # If no data before current_time (e.g., VIX starts later than SPY)
            # Use first available price (backfill for data gaps)
            if not df.empty:
                price = df['Close'].iloc[0]
                return float(price)
        
        return None
    
    def clear_cache(self):
        """Mock method - does nothing in time-travel mode"""
        pass
    
    def get_options_chain(self, symbol, expiration_date):
        """Mock method - not used in time-travel (uses Black-Scholes)"""
        return None
    
    def get_latest_price(self, symbol):
        """Get latest price (same as get_current_price)"""
        return self.get_current_price(symbol)
    
    def get_quote(self, symbol):
        """Mock quote method"""
        price = self.get_current_price(symbol)
        if price:
            return {'last': price, 'symbol': symbol}
        return None
    
    def test_connection(self):
        """Mock connection test - always returns True"""
        return True


# Create time-travel data source with all symbols
tt_source = TimeTravelDataSource(symbols_data, primary_symbol=trading_symbol)

# CRITICAL: Replace bot's data source AND disable ALL fallbacks
old_data_source = bot.data_source
bot.data_source = tt_source

# Completely disable the old data source to prevent any background calls
if hasattr(old_data_source, 'has_tradier'):
    old_data_source.has_tradier = False
if hasattr(old_data_source, 'has_alpha_vantage'):
    old_data_source.has_alpha_vantage = False  
if hasattr(old_data_source, 'yahoo_source'):
    old_data_source.yahoo_source = None

# Prevent bot from using ANY fallback data sources
if hasattr(bot.data_source, 'sources'):
    bot.data_source.sources = []
if hasattr(bot.data_source, 'fallback_source'):
    bot.data_source.fallback_source = None

# Override safe_get_data to always use TimeTravelDataSource
original_safe_get_data = bot.safe_get_data
def time_travel_safe_get_data(symbol, period='1d', interval='1m'):
    return tt_source.get_data(symbol, period, interval)
bot.safe_get_data = time_travel_safe_get_data

logger.info("[OK] TimeTravelDataSource installed (ALL fallbacks disabled)")
logger.info("[OK] Bot will ONLY use historical database data")

# Also check if paper trader has a separate data source reference
if hasattr(bot, 'paper_trader') and hasattr(bot.paper_trader, 'data_source'):
    bot.paper_trader.data_source = tt_source
    logger.info("[OK] Paper trader data source also replaced")

print("\n" + "="*70)
print("TIME-TRAVELING THROUGH HISTORY")
print("="*70)
print(f"\n[*] Processing {len(common_times)} minutes...")
print("[*] Running bot's ACTUAL cycle for each minute")
print("[*] This is EXACTLY how the bot runs live!\n")
print("-"*70)

# Keep file handler at INFO, but reduce console spam
for handler in logging.getLogger().handlers:
    if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
        handler.setLevel(logging.WARNING)  # Console: warnings/errors only

start_time = real_time.time()
cycles = 0
signals_generated = 0
trades_made = 0
rl_updates_count = 0  # Track RL learning updates

# Phase 52: Track max drawdown for risk-adjusted metrics
max_drawdown_pct = 0.0
peak_balance_seen = initial_balance
balance_history = []  # Track balance over time for equity curve analysis

# SIMPLIFIED: No losing streak protection (let the bot learn naturally)
# REALISTIC: Only train during market hours with complete data

# Track skipped timestamps for reporting
skipped_off_hours = 0
skipped_missing_data = 0

# Helper: robustly lookup future close for realized-move scoring
def _lookup_close(df, ts, tolerance_minutes: int = 5):
    try:
        if df is None or df.empty:
            return None
        # exact match fast path
        if ts in df.index:
            return float(df.loc[ts, "Close"])
        # nearest match with tolerance
        idx = df.index.get_indexer([ts], method="nearest")[0]
        if idx < 0:
            return None
        nearest_ts = df.index[idx]
        delta_min = abs((nearest_ts - ts).total_seconds()) / 60.0
        if delta_min > tolerance_minutes:
            return None
        return float(df.iloc[idx]["Close"])
    except Exception:
        return None


def _record_missed(sim_time, current_price, predicted_return, confidence, proposed_action, reason: str):
    if not missed_tracking.get("enabled", True):
        return
    if proposed_action not in ("BUY_CALLS", "BUY_PUTS"):
        return
    pending_missed.append({
        "t": sim_time,
        "price": float(current_price),
        "pred_ret": float(predicted_return or 0.0),
        "conf": float(confidence or 0.0),
        "action": proposed_action,
        "reason": reason,
    })
    missed_stats["tracked"] += 1


def _eval_missed(sim_time, df_primary):
    """Evaluate any missed opportunities whose horizon has elapsed."""
    if not missed_tracking.get("enabled", True):
        return
    horizon = int(missed_tracking.get("horizon_minutes", 15))
    if horizon <= 0:
        return
    move_thresh = float(missed_tracking.get("min_realized_move_to_count", 0.001))

    remaining = []
    for item in pending_missed:
        target_time = item["t"] + timedelta(minutes=horizon)
        if sim_time < target_time:
            remaining.append(item)
            continue

        future_close = _lookup_close(df_primary, target_time, tolerance_minutes=5)
        if future_close is None:
            remaining.append(item)
            continue

        entry = item["price"]
        realized = (future_close - entry) / entry if entry > 0 else 0.0
        action = item["action"]
        reason = item["reason"]

        # Classify after-cost "would win" vs "avoided loss"
        would_win = (realized >= move_thresh) if action == "BUY_CALLS" else (realized <= -move_thresh)
        would_lose = (realized <= -move_thresh) if action == "BUY_CALLS" else (realized >= move_thresh)

        missed_stats["evaluated"] += 1
        missed_stats["by_reason"].setdefault(reason, {"evaluated": 0, "missed_winners": 0, "avoided_losers": 0, "neutral": 0})
        missed_stats["by_action"].setdefault(action, {"evaluated": 0, "missed_winners": 0, "avoided_losers": 0, "neutral": 0})
        missed_stats["by_reason"][reason]["evaluated"] += 1
        missed_stats["by_action"][action]["evaluated"] += 1

        if would_win:
            missed_stats["missed_winners"] += 1
            missed_stats["by_reason"][reason]["missed_winners"] += 1
            missed_stats["by_action"][action]["missed_winners"] += 1
        elif would_lose:
            missed_stats["avoided_losers"] += 1
            missed_stats["by_reason"][reason]["avoided_losers"] += 1
            missed_stats["by_action"][action]["avoided_losers"] += 1
        else:
            missed_stats["neutral"] += 1
            missed_stats["by_reason"][reason]["neutral"] += 1
            missed_stats["by_action"][action]["neutral"] += 1

    pending_missed[:] = remaining


df_primary_for_missed = symbols_data.get(trading_symbol, pd.DataFrame())


# ----------------------------
# Tradability-gate dataset I/O
# ----------------------------
def _gate_features_from_signal(signal: dict, hmm_trend: float, hmm_vol: float, hmm_liq: float, hmm_conf: float) -> dict:
    return {
        "predicted_return": float(signal.get("predicted_return", 0.0) or 0.0),
        "prediction_confidence": float(signal.get("confidence", 0.0) or 0.0),
        "vix_level": float(signal.get("vix_level", 0.0) or 0.0),
        "momentum_5m": float(signal.get("momentum_5min", 0.0) or 0.0),
        "volume_spike": float(signal.get("volume_spike", 1.0) or 1.0),
        "hmm_trend": float(hmm_trend),
        "hmm_volatility": float(hmm_vol),
        "hmm_liquidity": float(hmm_liq),
        "hmm_confidence": float(hmm_conf),
    }


def _record_gate_candidate(sim_time, entry_price, proposed_action: str, features: dict, reason: str):
    if not gate_dataset.get("enabled", True):
        return
    if proposed_action not in ("BUY_CALLS", "BUY_PUTS"):
        return
    gate_dataset["pending"].append({
        "t": sim_time,
        "price": float(entry_price),
        "action": proposed_action,
        "reason": reason,
        "features": dict(features),
    })
    gate_dataset["stats"]["collected"] += 1


def _write_gate_row(row: dict):
    path = gate_dataset.get("path")
    if not path:
        return
    import json
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def _eval_gate(sim_time, df_primary, force_all: bool = False):
    if not gate_dataset.get("enabled", True):
        return
    horizon = int(gate_dataset.get("horizon_minutes", 15))
    if horizon <= 0:
        return
    thresh = float(gate_dataset.get("min_realized_move_to_count", 0.001))

    remaining = []
    for item in gate_dataset["pending"]:
        target_time = item["t"] + timedelta(minutes=horizon)
        # If forcing at the end of run, evaluate using the best available future point (sim_time).
        # This ensures we still get labeled data even if the run ends before horizon elapses.
        if (not force_all) and sim_time < target_time:
            remaining.append(item)
            continue

        eval_time = sim_time if force_all and sim_time < target_time else target_time
        future_close = _lookup_close(df_primary, eval_time, tolerance_minutes=5)
        if future_close is None:
            gate_dataset["stats"]["dropped_no_future_price"] += 1
            continue

        entry = item["price"]
        realized = (future_close - entry) / entry if entry > 0 else 0.0
        action = item["action"]

        # Label: tradable if realized move is large enough in action direction
        y = 1 if ((realized >= thresh) if action == "BUY_CALLS" else (realized <= -thresh)) else 0
        gate_dataset["stats"]["evaluated"] += 1
        if y == 1:
            gate_dataset["stats"]["positive"] += 1
        else:
            gate_dataset["stats"]["negative"] += 1

        _write_gate_row({
            "timestamp": str(item["t"]),
            "horizon_minutes": horizon,
            "effective_horizon_minutes": float((eval_time - item["t"]).total_seconds() / 60.0),
            "proposed_action": action,
            "reason": item.get("reason", "unknown"),
            "entry_price": entry,
            "future_price": float(future_close),
            "future_time": str(eval_time),
            "realized_move": float(realized),
            "label": int(y),
            "features": item["features"],
        })

    gate_dataset["pending"][:] = remaining

for idx, sim_time in enumerate(common_times):
    try:
        
        # ============================================================================
        # MARKET HOURS CHECK: Skip off-market timestamps
        # ============================================================================
        if not is_market_hours(sim_time):
            skipped_off_hours += 1
            if skipped_off_hours % 1000 == 0:  # Log every 1000 skips
                logger.debug(f"[SKIP] Skipped {skipped_off_hours} off-market timestamps so far")
            continue
        
        # Track cycle timing
        import time as real_time_module
        cycle_start_time = real_time_module.time()
        
        # Set the current time in our time-travel data source
        tt_source.set_time(sim_time)

        # Set simulation time for event calendar (Jerry's event halts feature)
        try:
            from backend.event_calendar import set_simulation_time
            set_simulation_time(sim_time)
        except ImportError:
            pass  # Event calendar not available

        # ============================================================================
        # DATA AVAILABILITY CHECK: Skip if required symbols don't have data
        # ============================================================================
        has_data, missing = has_required_data(tt_source, sim_time, REQUIRED_SYMBOLS)
        if not has_data:
            skipped_missing_data += 1
            if skipped_missing_data % 100 == 0:  # Log every 100 skips
                logger.debug(f"[SKIP] Skipped {skipped_missing_data} timestamps with missing data (missing: {missing})")
            continue
        
        # Set simulated time on bot (CRITICAL: set simulation_mode=True for get_market_time() to work!)
        bot.simulated_time = sim_time
        bot.paper_trader.simulated_time = sim_time
        bot.paper_trader.simulation_mode = True  # Required for get_market_time() to return simulated_time
        bot.current_time = sim_time
        
        # Log timestamp for dashboard
        logger.info(f"[TIME] Processing timestamp: {sim_time}")
        
        # Get current price for the configured trading symbol
        current_price = tt_source.get_current_price(trading_symbol)
        if not current_price:
            continue
        
        # === VERBOSE OUTPUT FOR EACH CYCLE ===
        print(f"\n{'='*70}")
        print(f"[CYCLE {cycles}] | {sim_time}")
        print(f"{'='*70}")
        # Log cycle info for dashboard (CRITICAL for dashboard updates!)
        logger.info(f"[CYCLE {cycles}] | {sim_time}")
        logger.info(f"[TIME] Processing timestamp: {sim_time}")
        
        # Get VIX value
        vix_price = tt_source.get_current_price('VIX')
        vix_str = f"${vix_price:.2f}" if vix_price else "N/A"
        
        # Get other symbol prices for dashboard display
        spy_price = tt_source.get_current_price('SPY')
        qqq_price = tt_source.get_current_price('QQQ')
        uup_price = tt_source.get_current_price('UUP')
        btc_price = tt_source.get_current_price('BTC-USD') or tt_source.get_current_price('BTC')
        
        print(f"[PRICE] {trading_symbol}: ${current_price:.2f}")
        print(f"[VIX] {vix_str}")
        if spy_price:
            print(f"[SPY] ${spy_price:.2f}")
        if qqq_price:
            print(f"[QQQ] ${qqq_price:.2f}")
        if btc_price:
            print(f"[BTC-USD] ${btc_price:,.2f}")
        if uup_price:
            print(f"[UUP/Dollar] ${uup_price:.2f}")
        print(f"[BALANCE] ${bot.paper_trader.current_balance:,.2f}")
        print(f"[BUFFER] Feature Buffer: {len(bot.feature_buffer)}/{bot.sequence_length}")
        print(f"[POSITIONS] Active: {len(bot.paper_trader.active_trades)}/{bot.paper_trader.max_positions}")
        print()
        
        # Log for dashboard
        logger.info(f"[PRICE] {trading_symbol} Price: ${current_price:.2f}")
        if vix_price:
            logger.info(f"[VIX] VIX: ${vix_price:.2f}")
        if spy_price:
            logger.info(f"[SPY] SPY: ${spy_price:.2f}")
        if qqq_price:
            logger.info(f"[QQQ] QQQ: ${qqq_price:.2f}")
        if btc_price:
            logger.info(f"[BTC-USD] BTC-USD: ${btc_price:,.2f}")
        if uup_price:
            logger.info(f"[DOLLAR] Dollar: ${uup_price:.2f}")
        logger.info(f"[BALANCE] Balance: ${bot.paper_trader.current_balance:,.2f}")
        logger.info(f"[POSITIONS] Active Positions: {len(bot.paper_trader.active_trades)}/{bot.paper_trader.max_positions}")
        
        # === COMPLETE TRADING CYCLE (AS BOT SHOULD RUN) ===
        
        # STEP 1: Update open positions (check exits FIRST)
        # This updates existing trades with current prices and may close positions
        print("[STEP1] STEP 1: Checking active positions...")
        old_trade_count = len(bot.paper_trader.active_trades)
        
        # Show details of active trades before update
        if len(bot.paper_trader.active_trades) > 0:
            print(f"   [ACTIVE] Active trades ({len(bot.paper_trader.active_trades)} open)")
        
        bot.paper_trader.update_positions(trading_symbol, current_price=current_price)

        # FORCE-CLOSE MECHANISM: Close positions held too long
        # Use config value (TT_MAX_HOLD_MINUTES) which is set at module load from config.json
        max_hold_minutes = float(os.environ.get("TT_MAX_HOLD_MINUTES", str(TT_MAX_HOLD_MINUTES)))
        trades_to_close = []
        for trade in bot.paper_trader.active_trades:
            try:
                trade_time = trade.timestamp if isinstance(trade.timestamp, datetime) else datetime.fromisoformat(str(trade.timestamp).replace('Z', '+00:00').replace('+00:00', ''))
                if sim_time.tzinfo is None and trade_time.tzinfo is not None:
                    trade_time = trade_time.replace(tzinfo=None)
                minutes_held = (sim_time - trade_time).total_seconds() / 60.0
                if minutes_held >= max_hold_minutes:
                    trades_to_close.append((trade, minutes_held))
            except Exception:
                pass

        for trade, minutes_held in trades_to_close:
            try:
                pnl_pct = ((current_price / trade.entry_price) - 1.0) * 100.0 if trade.entry_price else 0
                is_call = str(trade.option_type) in ['CALL', 'BUY_CALL', 'OrderType.CALL', 'OrderType.BUY_CALL']
                if is_call:
                    exit_premium = trade.premium_paid * (1 + pnl_pct / 10)
                else:
                    exit_premium = trade.premium_paid * (1 - pnl_pct / 10)
                exit_premium = max(0.01, exit_premium)
                if hasattr(bot.paper_trader, '_close_trade'):
                    bot.paper_trader._close_trade(trade, exit_premium, f"FORCE_CLOSE: Held {minutes_held:.0f}min")
                    print(f"   [FORCE_CLOSE] {trade.option_type} held {minutes_held:.0f}min - closed")
            except Exception as e:
                import traceback
                print(f"[FORCE_CLOSE_ERROR] Exception closing trade: {e}")
                traceback.print_exc()

        new_trade_count = len(bot.paper_trader.active_trades)

        if new_trade_count < old_trade_count:
            closed_count = old_trade_count - new_trade_count
            print(f"   [CLOSE] Closed {closed_count} trade(s)")
            
            # Log closed trades with P&L for dashboard
            recent_closed = bot.paper_trader.get_recent_closed_trades(limit=closed_count)
            for trade in recent_closed:
                print(f"   [TRADE] Closed {trade['option_type']} trade")
                print(f"   â€¢ Net P&L: ${trade['pnl']:+.2f}")
                logger.info(f"[TRADE] Closed {trade['option_type']} trade")
                logger.info(f"  â€¢ Net P&L: ${trade['pnl']:+.2f}")
        elif old_trade_count > 0:
            print(f"   [WAIT] No trades closed (all {old_trade_count} positions still open)")
        
        # EVENT-DRIVEN LEARNING: If trades closed, add to experience replay
        if new_trade_count < old_trade_count:
            # Trades closed! Get recent closed trades and add to experience buffer
            recent_closed = bot.paper_trader.get_recent_closed_trades(limit=5)
            for trade in recent_closed:
                # PNL CALIBRATION: Record outcome for P(profit) learning
                if pnl_calibration_tracker:
                    try:
                        pnl = trade.get('pnl', 0)
                        hold_mins = trade.get('hold_time_minutes', 45)
                        recorded = pnl_calibration_tracker.record_pnl_outcome(
                            trade_id=str(trade['id']),
                            option_pnl=pnl,
                            hold_minutes=hold_mins
                        )
                        if recorded:
                            pnl_metrics = pnl_calibration_tracker.get_metrics()
                            if pnl_metrics.get('pnl_sample_count', 0) % 10 == 0:
                                print(f"   [PNL_CAL] {pnl_metrics['pnl_sample_count']} samples | "
                                      f"Win rate: {pnl_metrics.get('pnl_win_rate', 0):.1%}")
                    except Exception as e:
                        logger.debug(f"[PNL_CAL] Could not record PnL outcome: {e}")

                # Only add if we haven't processed it yet
                if trade['id'] not in experience_manager.processed_trade_ids:
                    # Find the original signal that led to this trade (if available)
                    signal_for_trade = signal_by_trade_id.get(trade['id'], {})  # best-effort attribution
                    experience_manager.add_trade(trade, signal_for_trade)

                    # Use centralized P&L calculation (prevents 100x multiplier bug)
                    pnl_pct = compute_pnl_pct(trade)

                    # JERRY'S KILL SWITCHES: Record trade result for consecutive loss tracking
                    try:
                        from backend.kill_switches import record_trade
                        is_win = trade.get('pnl', 0) > 0
                        record_trade(pnl_pct, is_win)
                    except ImportError:
                        pass  # Kill switches not available

                    # Direction controller learning - record outcome
                    if HAS_DIRECTION_CONTROLLER and direction_controller is not None:
                        try:
                            # Get direction from trade type (CALL=UP, PUT=DOWN)
                            is_call = 'CALL' in str(trade.get('order_type', '')).upper()
                            trade_direction = 1 if is_call else -1
                            # Actual return: positive if price moved in predicted direction
                            actual_return = pnl_pct if is_call else -pnl_pct
                            direction_controller.record_outcome(
                                predicted_direction=trade_direction,
                                actual_return=actual_return,
                                trade_id=str(trade['id']),
                            )
                        except Exception as dir_err:
                            logger.debug(f"Direction outcome recording error: {dir_err}")

                    if cycles < 50:
                        logger.info(f"[EXP] Added experience: Trade #{trade['id']} | P&L: {pnl_pct:+.1%}")
        
        # STEP 2: Mini-batch learning from experiences (FAST, every 3 cycles)
        # Sample prioritized experiences and learn from them
        print("[STEP2] STEP 2: Mini-batch learning...")
        if cycles > 0 and cycles % 3 == 0 and len(experience_manager) >= 16:
            print(f"   [LEARN] Learning from {len(experience_manager)} experiences")
            try:
                # Sample a mini-batch of experiences (prioritized)
                experiences, indices, weights = experience_manager.sample_batch(batch_size=16)
                
                if len(experiences) > 0:
                    # Quick confidence adjustments (always)
                    if hasattr(bot, 'update_confidence_from_closed_trades'):
                        bot.update_confidence_from_closed_trades()
                    
                    # Learn from sampled experiences
                    # Calculate TD errors for priority updates (how surprising each outcome was)
                    td_errors = []
                    for exp in experiences:
                        # Simple TD error: actual outcome vs expected
                        actual_pnl = exp['pnl_pct']
                        
                        # If we had a signal, compare to prediction
                        if exp['signal'] and 'predicted_return' in exp['signal']:
                            predicted_pnl = exp['signal']['predicted_return']
                            td_error = abs(actual_pnl - predicted_pnl)
                        else:
                            # No prediction = high error (we didn't see this coming)
                            td_error = abs(actual_pnl)
                        
                        td_errors.append(td_error)
                    
                    # Update experience priorities based on TD errors
                    import numpy as np
                    experience_manager.update_from_learning(indices, np.array(td_errors))
                    
                    # ================================================================
                    # UPDATE RL SYSTEMS WITH EXPERIENCE
                    # ================================================================
                    
                    # UNIFIED RL: Record trade outcomes and train
                    if USE_UNIFIED_RL and unified_rl is not None:
                        for exp in experiences:
                            # Calculate step reward from outcome
                            pnl_pct = exp.get('pnl_pct', 0.0)
                            
                            # Create a done=True experience for the trade outcome
                            if exp.get('signal'):
                                # Build approximate states from signal data
                                prev_state = TradeState(
                                    is_in_trade=True,
                                    is_call='CALL' in str(exp.get('order_type', '')).upper(),
                                    unrealized_pnl_pct=0.0,
                                    prediction_confidence=exp['signal'].get('confidence', 0.5),
                                )
                                
                                curr_state = TradeState(
                                    is_in_trade=False,  # Trade closed
                                    unrealized_pnl_pct=pnl_pct,
                                    prediction_confidence=exp['signal'].get('confidence', 0.5),
                                )
                                
                                # Calculate reward
                                reward = unified_rl.calculate_step_reward(prev_state, curr_state, unified_rl.EXIT)
                                
                                # Record the experience
                                unified_rl.record_step_reward(
                                    state=prev_state,
                                    action=unified_rl.EXIT,
                                    reward=reward,
                                    next_state=curr_state,
                                    done=True
                                )
                        
                        # Train unified policy
                        train_result = unified_rl.train_step(batch_size=32)
                        if train_result and cycles % 15 == 0:
                            print(f"   [UNIFIED] Training loss: {train_result.get('total_loss', 0):.4f}")
                    
                    # ADAPTIVE: Update RL threshold learner with prioritized experiences
                    # Runs alongside unified RL when USE_RL_THRESHOLD_FILTER is enabled
                    if USE_RL_THRESHOLD_FILTER and rl_threshold_learner:
                        for exp in experiences:
                            if exp['signal']:
                                # Get full model inputs if available
                                rl_details = exp['signal'].get('rl_details', {})
                                model_inputs = rl_details.get('model_inputs', None)
                                # Get feature attribution if available (Phase 14)
                                feature_attribution = rl_details.get('feature_attribution', None)

                                # Call store_experience to add to buffer
                                rl_threshold_learner.store_experience(
                                    confidence=exp['signal'].get('confidence', 0.5),
                                    predicted_return=abs(exp['signal'].get('predicted_return', 0)),
                                    momentum=abs(exp['signal'].get('momentum_5min', 0)),
                                    volume_spike=exp['signal'].get('volume_spike', 1.0),
                                    composite_score=rl_details.get('composite_score', 0.5),
                                    outcome_pnl=exp['pnl_pct'],
                                    model_inputs=model_inputs,
                                    feature_attribution=feature_attribution
                                )
                        
                        # Train the model
                        rl_threshold_learner.train_from_experiences(batch_size=32)
                    
                    # Increment RL updates counter
                    rl_updates_count += 1
                    
                    if cycles % 15 == 0:  # Log every 5 mini-batch updates
                        stats = experience_manager.get_stats()
                        logger.info(f"[LEARN] Mini-batch learning: {len(experiences)} experiences | Buffer: {stats['size']} | Avg priority: {stats['avg_priority']:.2f}")
            
            except Exception as e:
                logger.warning(f"Error in mini-batch learning: {e}")
        else:
            if cycles > 0 and cycles % 3 == 0:
                print(f"   [SKIP] Skipped (need 16+ experiences, have {len(experience_manager)})")
            else:
                print(f"   [SKIP] Skipped (runs every 3 cycles, next at cycle {(cycles//3 + 1) * 3})")
        
        # STEP 3: Full neural network retraining - LESS FREQUENT (every 20 cycles)
        # This is computationally expensive, so we do it less often now
        # Mini-batch updates (every 3 cycles) handle most of the learning
        # Full retraining more frequently (every 10 cycles) for better prediction accuracy
        print("[NN] STEP 3: Full neural network retraining...")
        if cycles % 10 == 0 and cycles > 0:
            print(f"   [NN] RETRAINING NEURAL NETWORK (cycle {cycles})")
            logger.info("[NN] Running full neural network retraining cycle...")
            bot.continuous_learning_cycle()
            rl_updates_count += 1  # Count full retraining as an RL update
            print(f"   [OK] Retraining complete")
        else:
            next_retrain = ((cycles // 10) + 1) * 10
            print(f"   [SKIP] Skipped (runs every 10 cycles, next at cycle {next_retrain})")
        
        # STEP 4: Make NEW predictions (generate signal)
        # Bot analyzes current data and makes predictions
        print("[PREDICT] STEP 4: Generating predictions...")
        
        data_for_signal = tt_source.get_data(trading_symbol)
        signal = None
        
        if data_for_signal is None or data_for_signal.empty:
            print("   [WARN] No historical window available for signal generation")
        else:
            signal = bot.generate_time_travel_signal(
                trading_symbol,
                data_for_signal,
                simulation_time=sim_time,
                interval='1m'
            )
        
        # Check if NN predictions are active
        buffer_len = len(bot.feature_buffer)
        if buffer_len >= bot.sequence_length:
            print(f"   [NN] Neural network active ({buffer_len} features)")
        else:
            print(f"   [WARMUP] Warmup mode ({buffer_len}/{bot.sequence_length} features) - using fallback")
        
        # Log HMM regime info if available (reuse fetched data)
        if bot.hmm_regime_detector and bot.hmm_regime_detector.is_fitted and data_for_signal is not None and not data_for_signal.empty:
            try:
                regime_info = bot.hmm_regime_detector.predict_current_regime(data_for_signal)
                print(f"   [HMM] Market Regime: {regime_info['combined_name']} (confidence: {regime_info['combined_confidence']:.1%})")
                logger.info(f"[HMM] Regime: {regime_info['combined_name']}, Confidence: {regime_info['combined_confidence']:.1%}, Trend: {regime_info['trend_name']}, Vol: {regime_info['volatility_name']}")
            except Exception as e:
                logger.debug(f"HMM regime logging skipped: {e}")
        
        # Debug: Check what we got
        print(f"   [DEBUG] Signal generated: {signal is not None}")
        print(f"   [DEBUG] Model exists: {bot.model is not None}")
        print(f"   [DEBUG] Feature buffer size: {buffer_len}/{bot.sequence_length}")
        
        # Show multi-timeframe predictions when available
        if signal and signal.get("multi_timeframe_predictions"):
            mtf_preds = signal["multi_timeframe_predictions"]
            print(f"   [LSTM] Multi-timeframe TCN predictions:")
            logger.info(f"[LSTM] Multi-timeframe TCN predictions:")
            for timeframe, pred in mtf_preds.items():
                predicted_return = pred.get('predicted_return', 0)
                neural_confidence = pred.get('neural_confidence', 0)
                direction = "UP" if predicted_return > 0 else "DOWN"
                print(f"      - {timeframe:6s}: {direction} | Return: {predicted_return:+.2%} | Confidence: {neural_confidence:.1%}")
                logger.info(f"[LSTM] {timeframe}: direction={direction}, return={predicted_return:+.2%}, confidence={neural_confidence:.1%}")
        elif buffer_len >= bot.sequence_length:
            print(f"   [WARN] No TCN predictions available (check for errors)")
            logger.warning(f"[LSTM] No TCN predictions available despite full buffer")
        
        if signal:
            action = signal.get('action', 'HOLD')
            confidence = signal.get('confidence', 0)

            # ============= INVERSE SIGNAL MODE =============
            # If predictor is consistently wrong, try inverting: CALLS->PUTS, PUTS->CALLS
            invert_signals = os.environ.get('TT_INVERT_SIGNALS', '0') == '1'
            if invert_signals and action in ['BUY_CALLS', 'BUY_PUTS']:
                original_action = action
                action = 'BUY_PUTS' if action == 'BUY_CALLS' else 'BUY_CALLS'
                signal['action'] = action  # Update the signal dict too
                # Also invert predicted_return so edge check passes
                if 'predicted_return' in signal:
                    signal['predicted_return'] = -signal['predicted_return']
                print(f"   [INVERT] Signal inverted: {original_action} -> {action}")

            has_nn_prediction = signal.get('neural_prediction') is not None
            prediction_source = "NN+Options" if has_nn_prediction else "Options Only"
            print(f"   [SIGNAL] FINAL Signal: {action} (confidence: {confidence:.1%}) [{prediction_source}]")
            # Log for dashboard
            logger.info(f"[SIGNAL] FINAL Signal: {action} (confidence: {confidence:.1%}) [{prediction_source}]")
            try:
                decision_stats["signals_total"] += 1
                if action != "HOLD":
                    decision_stats["signals_non_hold"] += 1
            except Exception:
                pass
        
        # STEP 5: Execute strategy (act on predictions)
        # Evaluate the signal and decide whether to trade
        print("[TRADE] STEP 5: Evaluating trade execution...")

        # -------------------------------------------------------------------
        # Offline dataset logging (DecisionRecord + realized MissedOpp)
        # -------------------------------------------------------------------
        proposed_action_for_log = "HOLD"
        executed_action_for_log = "HOLD"
        trade_placed_for_log = False
        rejection_reasons_for_log = []
        paper_rejection_for_log = None
        trade = None  # ensure defined even if we never place a trade
        try:
            if signal and isinstance(signal, dict):
                proposed_action_for_log = str(signal.get("action", "HOLD") or "HOLD")
        except Exception:
            pass
        if signal is None:
            rejection_reasons_for_log.append("no_signal")
        
        # COUNTERFACTUAL LEARNING: Track HOLD signals too so we can learn from missed opportunities
        # Even if the bot says HOLD, record what direction the prediction was leaning
        if signal and signal.get('action') == 'HOLD' and rl_threshold_learner:
            # Infer what the signal WOULD have been from predicted_return
            predicted_return = signal.get('predicted_return', 0)
            confidence = signal.get('confidence', 0)
            
            # Determine what the proposed action would have been
            if predicted_return > 0.001:  # Would have been CALLS
                proposed_action = 'BUY_CALLS'
            elif predicted_return < -0.001:  # Would have been PUTS  
                proposed_action = 'BUY_PUTS'
            else:
                proposed_action = None  # Truly neutral, no counterfactual needed
            
            # Track this HOLD as a potential missed opportunity
            if proposed_action and confidence > 0.10:  # Only track if there was some conviction
                rl_threshold_learner.rejected_signals.append({
                    'timestamp': sim_time,
                    'price_at_signal': current_price,
                    'confidence': confidence,
                    'predicted_return': predicted_return,
                    'momentum': signal.get('momentum_5min', 0),
                    'volume': signal.get('volume_spike', 1.0),
                    'composite_score': confidence,  # Use confidence as proxy
                    'threshold': 0.20,  # HOLD threshold
                    'proposed_action': proposed_action,
                    'actual_return': None,
                    'was_checked': False,
                    'signal_type': 'HOLD_OVERRIDE'  # Mark as forced HOLD
                })
                print(f"   [TRACK] Recording HOLD for counterfactual: {proposed_action} ({confidence:.1%} conf, {predicted_return*100:.2f}% return)")

        # MISSED-PLAY TRACKING (report-only; works for unified RL too)
        if signal and signal.get('action') == 'HOLD':
            try:
                try:
                    rejection_reasons_for_log.append("hold")
                except Exception:
                    pass
                pr = float(signal.get('predicted_return', 0.0) or 0.0)
                conf = float(signal.get('confidence', 0.0) or 0.0)
                min_conf = float(missed_tracking.get("min_confidence", 0.25))
                min_abs = float(missed_tracking.get("min_abs_predicted_return", 0.0015))
                if abs(pr) >= min_abs and conf >= min_conf:
                    proposed = "BUY_CALLS" if pr > 0 else "BUY_PUTS"
                    _record_missed(sim_time, current_price, pr, conf, proposed, reason="hold")
                    # Also record as gate dataset candidate (supervised)
                    # Use proposed_action inferred from sign of predicted_return
                    try:
                        # HMM floats are computed later in the unified RL block; for HOLD we may not have them.
                        feats = {
                            "predicted_return": float(pr),
                            "prediction_confidence": float(conf),
                            "vix_level": float(signal.get("vix_level", 0.0) or 0.0),
                            "momentum_5m": float(signal.get("momentum_5min", 0.0) or 0.0),
                            "volume_spike": float(signal.get("volume_spike", 1.0) or 1.0),
                            "hmm_trend": 0.5,
                            "hmm_volatility": 0.5,
                            "hmm_liquidity": 0.5,
                            "hmm_confidence": 0.5,
                        }
                        _record_gate_candidate(sim_time, current_price, proposed, feats, reason="hold")
                    except Exception:
                        pass
            except Exception:
                pass

        # ================================================================
        # ENTRY CONTROLLER OVERRIDE (Q-scorer) - BYPASSES WARMUP & HOLD
        # ================================================================
        # If Q-scorer mode enabled, use it instead of all other entry logic.
        # Q-scorer is a fully-trained model that doesn't need warmup.
        # Entry mode is loaded from config.json (or env var override) at startup
        entry_mode = ENTRY_CONTROLLER_TYPE  # Use global from config.json

        # DEBUG: Log entry mode (only on first cycle)
        if cycles == 0:
            logger.info(f"[ENTRY_MODE] ENTRY_CONTROLLER={repr(entry_mode)} (from config.json)")
            logger.info(f"[ENTRY_MODE] HAS_CONSENSUS_CONTROLLER={HAS_CONSENSUS_CONTROLLER}")

        # DEBUG: Log Q-scorer condition check
        q_mode_match = entry_mode in ("q_scorer", "q", "qscorer")
        q_scorer_active = q_mode_match and HAS_Q_ENTRY_CONTROLLER and q_entry_controller is not None

        if q_scorer_active:
            # Q-scorer takes over COMPLETELY - bypasses warmup and all other entry logic
            logger.info(f"   [Q_DEBUG] Q-scorer ACTIVE: entry_mode={repr(entry_mode)}")
            try:
                # Build q trade_state from the signal (best-effort, same idea as live bot).
                q_state = dict(signal or {})
                # Ensure embedding present if available.
                npred = q_state.get("neural_prediction") or q_state.get("neural_pred") or {}
                if isinstance(npred, dict) and q_state.get("predictor_embedding") is None:
                    q_state["predictor_embedding"] = npred.get("predictor_embedding", npred.get("embedding"))
                q_state.setdefault("is_in_trade", False)
                q_dec = q_entry_controller.decide(q_state)
                if q_dec is None:
                    logger.warning(f"   [Q_SCORER] decide() returned None - likely missing predictor_embedding (cycle {cycle_count})")
            except Exception as e:
                logger.error(f"   [Q_SCORER] Exception in decide(): {e}", exc_info=True)
                q_dec = None

            if q_dec is not None and q_dec.action in ("BUY_CALLS", "BUY_PUTS"):
                should_trade = True
                signals_generated += 1
                composite_score = float(max(q_dec.q_values.get("BUY_CALLS", 0.0), q_dec.q_values.get("BUY_PUTS", 0.0)))
                action = q_dec.action
                confidence = 0.5  # placeholder
                predicted_return = 0.0  # placeholder
                momentum_5min = signal.get('momentum_5min', 0) if signal else 0
                volume_spike = signal.get('volume_spike', 1.0) if signal else 1.0
                signal = signal or {}
                signal["action"] = action
                signal["q_values"] = dict(q_dec.q_values)
                signal["q_threshold"] = float(q_dec.threshold)
                try:
                    signal.setdefault("reasoning", [])
                    signal["reasoning"] = list(signal.get("reasoning", [])) + [q_dec.reason]
                except Exception:
                    pass
                rl_details = {"reason": f"q_scorer ({q_dec.reason})", "mode": "q_scorer"}
                print(f"   [Q_SCORER] TRADE: {action} (q={composite_score:+.2f}, threshold={q_dec.threshold:+.2f})")
            else:
                should_trade = False
                composite_score = 0.0
                rl_details = {"reason": "q_scorer_hold", "mode": "q_scorer"}
                try:
                    rejection_reasons_for_log.append("q_scorer_hold")
                except Exception:
                    pass
                print(f"   [Q_SCORER] HOLD (q_dec={'None' if q_dec is None else 'action='+q_dec.action})")

            # Q-SCORER TRADE EXECUTION
            if should_trade:
                # Check position limits
                current_positions = len(bot.paper_trader.active_trades)
                if current_positions >= bot.max_positions:
                    print(f"   [WARN] Max positions reached ({current_positions}/{bot.max_positions}) - cannot trade")
                    try:
                        rejection_reasons_for_log.append("max_positions")
                    except Exception:
                        pass
                else:
                    # Execute trade in simulation
                    print(f"   [Q_TRADE] Executing Q-scorer trade...")
                    trade = bot.paper_trader.place_trade(trading_symbol, signal, current_price)
                    if trade:
                        trades_made += 1
                        try:
                            trade_placed_for_log = True
                        except Exception:
                            pass
                        print(f"\n   [SIMON] SimonSays: {action}\n")

                        # SHADOW TRADING: Also place in Tradier Sandbox if enabled
                        if tradier_shadow:
                            try:
                                print(f"   [SHADOW] Mirroring trade to Tradier Sandbox...")
                                shadow_signal = signal.copy()
                                shadow_signal['quantity'] = trade.quantity
                                shadow_signal['strike_price'] = trade.strike_price
                                logger.info(f"[SHADOW] Using paper trade params: {trade.quantity} contracts @ ${trade.strike_price:.2f} strike")
                                shadow_trade = tradier_shadow.place_trade(trading_symbol, shadow_signal, current_price)
                                if shadow_trade:
                                    print(f"   [SHADOW] âœ“ Trade placed in Tradier Sandbox!")
                                    logger.info(f"[SHADOW] Trade mirrored to Tradier: {action}")
                                else:
                                    print(f"   [SHADOW] âœ— Failed to place in Tradier Sandbox")
                                    logger.warning(f"[SHADOW] Failed to mirror trade")
                            except Exception as e:
                                print(f"   [SHADOW] âœ— Error: {e}")
                                logger.error(f"[SHADOW] Error mirroring trade: {e}")
                        print(f"   [OK] TRADE PLACED: {action} @ ${current_price:.2f} (Strike: ${trade.strike_price:.2f})")
                        logger.info(f"[SIMON] SimonSays: {action}")
                        logger.info(f"[TRADE] TRADE PLACED: {action} at ${current_price:.2f} | Strike: ${trade.strike_price:.2f} | Qty: {trade.quantity}")
                    else:
                        print(f"   [ERROR] Trade failed to execute")
                        try:
                            rejection_reasons_for_log.append("trade_failed")
                        except Exception:
                            pass

        # ================================================================
        # V3 DIRECTION PREDICTOR (simplified, 56% val accuracy)
        # ================================================================
        v3_mode_match = entry_mode in ("v3_direction", "v3", "v3dir")
        v3_active = v3_mode_match and HAS_V3_CONTROLLER and v3_controller is not None

        if v3_active and signal and not q_scorer_active:
            # V3 direction predictor mode
            print(f"   [V3] Using DirectionPredictorV3 for entry decision")

            try:
                # Update price history in controller
                v3_controller.update_price(current_price, signal.get('volume', 1.0))

                # Get HMM trend for alignment filter from bot's current regime
                hmm_trend_val = 0.5  # Default neutral
                if hasattr(bot, 'current_hmm_regime') and bot.current_hmm_regime:
                    regime = bot.current_hmm_regime
                    trend_str = regime.get('trend', regime.get('trend_state', ''))
                    # Map trend string to value
                    trend_map = {
                        'Down': 0.0, 'Downtrend': 0.0, 'Bearish': 0.0,
                        'Sideways': 0.5, 'Neutral': 0.5, 'No Trend': 0.5,
                        'Up': 1.0, 'Uptrend': 1.0, 'Bullish': 1.0
                    }
                    hmm_trend_val = trend_map.get(str(trend_str), 0.5)

                # Get decision with HMM alignment filter
                v3_decision = v3_controller.decide(
                    signal=signal,
                    vix_level=signal.get('vix_level', 15.0),
                    market_open=True,
                    hmm_trend=hmm_trend_val
                )

                if v3_decision.action in ("BUY_CALLS", "BUY_PUTS"):
                    should_trade = True
                    signals_generated += 1
                    action = v3_decision.action
                    composite_score = v3_decision.direction_prob
                    confidence = v3_decision.confidence
                    predicted_return = 0.0
                    momentum_5min = signal.get('momentum_5min', 0) if signal else 0
                    volume_spike = signal.get('volume_spike', 1.0) if signal else 1.0
                    signal["action"] = action
                    rl_details = {"reason": f"v3_direction (p={v3_decision.direction_prob:.2f}, hmm={hmm_trend_val:.2f})", "mode": "v3"}
                    print(f"   [V3] TRADE: {action} (p_dir={v3_decision.direction_prob:.3f}, hmm={hmm_trend_val:.2f})")

                    # RSI+MACD confirmation filter for higher win rate
                    if RSI_MACD_FILTER_ENABLED and should_trade:
                        rsi_val = signal.get('rsi', signal.get('rsi_14', 50))
                        macd_val = signal.get('macd', 0)

                        rsi_confirms = False
                        macd_confirms = True  # Default to True if MACD check disabled

                        if RSI_MOMENTUM_MODE:
                            # Momentum mode: trade WITH trend
                            if action == "BUY_CALLS":
                                rsi_confirms = rsi_val > 50  # Bullish momentum
                                if MACD_CONFIRM_ENABLED:
                                    macd_confirms = macd_val > 0  # MACD bullish
                            else:  # BUY_PUTS
                                rsi_confirms = rsi_val < 50  # Bearish momentum
                                if MACD_CONFIRM_ENABLED:
                                    macd_confirms = macd_val < 0  # MACD bearish
                        else:
                            # Mean reversion mode (default): trade against extremes
                            if action == "BUY_CALLS":
                                rsi_confirms = rsi_val < RSI_OVERSOLD_THRESHOLD  # Oversold = bounce
                                if MACD_CONFIRM_ENABLED:
                                    macd_confirms = macd_val > 0  # MACD bullish
                            else:  # BUY_PUTS
                                rsi_confirms = rsi_val > RSI_OVERBOUGHT_THRESHOLD  # Overbought = drop
                                if MACD_CONFIRM_ENABLED:
                                    macd_confirms = macd_val < 0  # MACD bearish

                        if not rsi_confirms or not macd_confirms:
                            should_trade = False
                            rejection_reason = f"rsi_macd_filter (RSI={rsi_val:.1f}, MACD={macd_val:.4f})"
                            rl_details = {"reason": rejection_reason, "mode": "v3"}
                            try:
                                rejection_reasons_for_log.append("rsi_macd_filter")
                            except Exception:
                                pass
                            print(f"   [V3] RSI/MACD VETO: {action} blocked (RSI={rsi_val:.1f}, MACD={macd_val:.4f})")
                        else:
                            print(f"   [V3] RSI/MACD OK: RSI={rsi_val:.1f}, MACD={macd_val:.4f}")

                    # Volume confirmation filter
                    if VOLUME_CONFIRM_ENABLED and should_trade:
                        volume_spike = signal.get('volume_spike', 1.0)
                        if volume_spike < VOLUME_CONFIRM_THRESHOLD:
                            should_trade = False
                            rejection_reason = f"volume_filter (spike={volume_spike:.2f} < {VOLUME_CONFIRM_THRESHOLD})"
                            rl_details = {"reason": rejection_reason, "mode": "v3"}
                            try:
                                rejection_reasons_for_log.append("volume_filter")
                            except Exception:
                                pass
                            print(f"   [V3] VOLUME VETO: volume_spike={volume_spike:.2f} < {VOLUME_CONFIRM_THRESHOLD}")
                        else:
                            print(f"   [V3] VOLUME OK: spike={volume_spike:.2f}")

                    # Jerry's Quantor-MTFuzz confirmation filter
                    if JERRY_FILTER_ENABLED and should_trade:
                        try:
                            from features.jerry_features import compute_jerry_features, jerry_filter_check
                            # Compute Jerry's fuzzy scores
                            jerry_feats = compute_jerry_features(
                                vix_level=signal.get('vix_level', 18.0),
                                bars_in_regime=getattr(bot, '_bars_in_regime', 10),
                                volume_ratio=signal.get('volume_spike', 1.0),
                                hmm_confidence=signal.get('hmm_confidence', 0.5),
                            )
                            passed, reason = jerry_filter_check(
                                f_t=jerry_feats.f_t,
                                threshold=JERRY_FILTER_THRESHOLD,
                                min_mtf=JERRY_MTF_MIN,
                                mu_mtf=jerry_feats.mu_mtf
                            )
                            if not passed:
                                should_trade = False
                                rejection_reason = f"jerry_filter ({reason})"
                                rl_details = {"reason": rejection_reason, "mode": "v3"}
                                try:
                                    rejection_reasons_for_log.append("jerry_filter")
                                except Exception:
                                    pass
                                print(f"   [V3] JERRY VETO: {reason} (F_t={jerry_feats.f_t:.2f})")
                            else:
                                print(f"   [V3] JERRY OK: F_t={jerry_feats.f_t:.2f}")
                        except Exception as e:
                            logger.warning(f"Jerry filter error: {e}")
                else:
                    should_trade = False
                    composite_score = 0.0
                    hmm_veto = v3_decision.details.get('hmm_veto_reason')
                    if hmm_veto:
                        rl_details = {"reason": f"v3_hmm_veto: {hmm_veto}", "mode": "v3"}
                        try:
                            rejection_reasons_for_log.append("v3_hmm_veto")
                        except Exception:
                            pass
                        print(f"   [V3] HMM VETO: {hmm_veto}")
                    else:
                        rl_details = {"reason": "v3_hold", "mode": "v3"}
                        try:
                            rejection_reasons_for_log.append("v3_hold")
                        except Exception:
                            pass
                        print(f"   [V3] HOLD (p_up={v3_decision.details.get('p_up', 0):.3f}, hmm={hmm_trend_val:.2f})")

            except Exception as e:
                logger.error(f"   [V3] Exception: {e}", exc_info=True)
                should_trade = False
                rl_details = {"reason": f"v3_error: {e}", "mode": "v3"}

            # V3 TRADE EXECUTION
            if should_trade:
                current_positions = len(bot.paper_trader.active_trades)
                if current_positions >= bot.max_positions:
                    print(f"   [WARN] Max positions reached ({current_positions}/{bot.max_positions})")
                    try:
                        rejection_reasons_for_log.append("max_positions")
                    except Exception:
                        pass
                else:
                    print(f"   [V3_TRADE] Executing V3 trade...")
                    trade = bot.paper_trader.place_trade(trading_symbol, signal, current_price)
                    if trade:
                        trades_made += 1
                        try:
                            trade_placed_for_log = True
                        except Exception:
                            pass
                        print(f"   [OK] V3 TRADE PLACED: {action} @ ${current_price:.2f} (Strike: ${trade.strike_price:.2f})")
                        logger.info(f"[V3_TRADE] {action} at ${current_price:.2f} | Strike: ${trade.strike_price:.2f}")
                    else:
                        print(f"   [ERROR] V3 trade failed to execute")
                        try:
                            rejection_reasons_for_log.append("trade_failed")
                        except Exception:
                            pass

        # ================================================================
        # CONSENSUS ENTRY CONTROLLER (multi-signal agreement)
        # ================================================================
        consensus_mode_match = entry_mode in ("consensus", "multi", "multisignal")
        consensus_active = consensus_mode_match and HAS_CONSENSUS_CONTROLLER and consensus_controller is not None

        if consensus_active and signal and not q_scorer_active and not v3_active:
            # Consensus controller mode - requires ALL signals to agree
            print(f"   [CONSENSUS] Using multi-signal consensus for entry decision")

            try:
                # Build HMM regime dict from bot's current regime
                hmm_regime = {'trend': 0.5, 'volatility': 0.5, 'liquidity': 0.5, 'confidence': 0.5}
                if hasattr(bot, 'current_hmm_regime') and bot.current_hmm_regime:
                    regime = bot.current_hmm_regime
                    trend_str = regime.get('trend', regime.get('trend_state', ''))
                    # Map trend string to value
                    trend_map = {
                        'Down': 0.0, 'Downtrend': 0.0, 'Bearish': 0.0,
                        'Sideways': 0.5, 'Neutral': 0.5, 'No Trend': 0.5,
                        'Up': 1.0, 'Uptrend': 1.0, 'Bullish': 1.0
                    }
                    hmm_regime['trend'] = trend_map.get(str(trend_str), 0.5)
                    hmm_regime['confidence'] = regime.get('confidence', 0.5)
                    # Get volatility state
                    vol_str = regime.get('volatility', regime.get('volatility_state', ''))
                    vol_map = {'Low': 0.2, 'Normal': 0.5, 'High': 0.8, 'Panic': 1.0}
                    hmm_regime['volatility'] = vol_map.get(str(vol_str), 0.5)

                # Add multi-timeframe predictions to signal from bot
                if hasattr(bot, 'multi_timeframe_predictions') and bot.multi_timeframe_predictions:
                    signal['multi_timeframe_predictions'] = bot.multi_timeframe_predictions

                # Build features dict
                features = {
                    'momentum_5m': signal.get('momentum_5m', signal.get('momentum_5min', 0)),
                    'momentum_15m': signal.get('momentum_15m', signal.get('momentum_15min', 0)),
                    'price_jerk': signal.get('price_jerk', signal.get('jerk', 0)),
                    'rsi': signal.get('rsi', signal.get('rsi_14', 50)),
                    'vix_level': vix_price if vix_price else 18.0,
                    'hmm_volatility': hmm_regime.get('volatility', 0.5),
                    'volume_spike': signal.get('volume_spike', signal.get('volume_ratio', 1.0)),
                }

                # Get decision from consensus controller
                cons_action, cons_conf, cons_details = consensus_controller.decide(signal, hmm_regime, features)

                if cons_action in ("BUY_CALLS", "BUY_PUTS"):
                    should_trade = True
                    signals_generated += 1
                    action = cons_action
                    composite_score = cons_conf
                    confidence = cons_conf
                    predicted_return = 0.0
                    momentum_5min = signal.get('momentum_5min', 0) if signal else 0
                    volume_spike = signal.get('volume_spike', 1.0) if signal else 1.0
                    signal["action"] = action
                    rl_details = {"reason": f"consensus_all_agree", "mode": "consensus", "checks": cons_details.get('checks', {})}
                    print(f"   [CONSENSUS] TRADE: {action} (conf={cons_conf:.2f}) - All 4 signals agree!")
                else:
                    should_trade = False
                    composite_score = 0.0
                    failed = cons_details.get('failed', [])
                    rl_details = {"reason": f"consensus_veto: {failed}", "mode": "consensus"}
                    try:
                        rejection_reasons_for_log.append(f"consensus_veto_{','.join(failed)}")
                    except Exception:
                        pass
                    try:
                        print(f"   [CONSENSUS] HOLD - Failed checks: {failed}")
                    except OSError:
                        # Windows console encoding issue
                        logger.info(f"[CONSENSUS] HOLD - Failed checks: {failed}")

            except Exception as e:
                logger.error(f"   [CONSENSUS] Exception: {e}", exc_info=True)
                should_trade = False
                rl_details = {"reason": f"consensus_error: {e}", "mode": "consensus"}

            # CONSENSUS TRADE EXECUTION
            if should_trade:
                current_positions = len(bot.paper_trader.active_trades)
                if current_positions >= bot.max_positions:
                    print(f"   [WARN] Max positions reached ({current_positions}/{bot.max_positions})")
                    try:
                        rejection_reasons_for_log.append("max_positions")
                    except Exception:
                        pass
                else:
                    print(f"   [CONSENSUS_TRADE] Executing consensus trade...")
                    trade = bot.paper_trader.place_trade(trading_symbol, signal, current_price)
                    if trade:
                        trades_made += 1
                        try:
                            trade_placed_for_log = True
                        except Exception:
                            pass
                        print(f"   [OK] CONSENSUS TRADE PLACED: {action} @ ${current_price:.2f} (Strike: ${trade.strike_price:.2f})")
                        logger.info(f"[CONSENSUS_TRADE] {action} at ${current_price:.2f} | Strike: ${trade.strike_price:.2f}")
                    else:
                        print(f"   [ERROR] Consensus trade failed to execute")
                        try:
                            rejection_reasons_for_log.append("trade_failed")
                        except Exception:
                            pass

        # ================================================================
        # DIRECTION-BASED ENTRY CONTROLLER (for improved win rate)
        # ================================================================
        direction_mode_match = entry_mode in ("direction", "dir", "direction_predictor")
        direction_active = direction_mode_match and HAS_DIRECTION_CONTROLLER and direction_controller is not None

        if direction_active and signal:
            # Direction controller mode - uses dedicated DirectionPredictor
            print(f"   [DIRECTION] Using DirectionPredictor for entry decision")

            # Set cycle counter for rate limiting
            direction_controller.set_cycle(cycles)

            try:
                # Get features from signal
                current_features = signal.get('features')
                sequence_features = signal.get('sequence_features')

                # Get HMM data directly from bot
                hmm_trend_val = 0.5  # Default neutral
                hmm_conf_val = 0.5
                if bot.current_hmm_regime:
                    trend_map = {
                        'Down': 0.0, 'Downtrend': 0.0, 'Bearish': 0.0,
                        'No Trend': 0.5, 'Neutral': 0.5, 'Sideways': 0.5,
                        'Up': 1.0, 'Uptrend': 1.0, 'Bullish': 1.0
                    }
                    trend_name = bot.current_hmm_regime.get('trend', 'Neutral')
                    hmm_trend_val = trend_map.get(trend_name, 0.5)
                    hmm_conf_val = bot.current_hmm_regime.get('confidence', 0.5)
                vix_val = vix_price if vix_price else 18.0

                # Convert to tensors if needed
                import torch
                if current_features is not None:
                    if not isinstance(current_features, torch.Tensor):
                        current_features = torch.tensor(current_features, dtype=torch.float32)
                else:
                    current_features = torch.zeros(50)

                if sequence_features is not None:
                    if not isinstance(sequence_features, torch.Tensor):
                        sequence_features = torch.tensor(sequence_features, dtype=torch.float32)
                else:
                    sequence_features = torch.zeros(60, 50)

                print(f"   [DIRECTION] HMM trend={hmm_trend_val:.2f}, VIX={vix_val:.1f}, features_shape={current_features.shape if hasattr(current_features, 'shape') else 'N/A'}")
                decision = direction_controller.decide(
                    current_features,
                    sequence_features,
                    hmm_trend=hmm_trend_val,
                    hmm_confidence=hmm_conf_val,
                    vix_level=vix_val,
                )
                print(f"   [DIRECTION] Decision: {decision.action} (reason={decision.details.get('reason', 'N/A')})")

                if decision.action in ("BUY_CALLS", "BUY_PUTS"):
                    should_trade = True
                    signals_generated += 1
                    composite_score = decision.confidence
                    action = decision.action
                    confidence = decision.confidence
                    predicted_return = decision.magnitude * decision.direction
                    momentum_5min = signal.get('momentum_5min', 0)
                    volume_spike = signal.get('volume_spike', 1.0)
                    signal["action"] = action
                    rl_details = {"reason": f"direction ({decision.details.get('reason', 'N/A')})", "mode": "direction"}
                    print(f"   [DIRECTION] TRADE: {action} (conf={decision.confidence:.1%}, dir={decision.direction}, mag={decision.magnitude:.2%})")
                else:
                    should_trade = False
                    composite_score = decision.confidence
                    rl_details = {"reason": f"direction_hold ({decision.details.get('reason', 'N/A')})", "mode": "direction"}
                    print(f"   [DIRECTION] HOLD: {decision.details.get('reason', 'N/A')}")
                    try:
                        rejection_reasons_for_log.append("direction_hold")
                    except Exception:
                        pass

            except Exception as e:
                print(f"   [DIRECTION] ERROR: {e}")
                import traceback
                traceback.print_exc()
                logger.error(f"   [DIRECTION] Exception in decide(): {e}", exc_info=True)
                should_trade = False
                composite_score = 0.0
                rl_details = {"reason": "direction_error", "mode": "direction"}

            # DIRECTION CONTROLLER TRADE EXECUTION
            if should_trade:
                current_positions = len(bot.paper_trader.active_trades)
                if current_positions >= bot.max_positions:
                    print(f"   [WARN] Max positions reached ({current_positions}/{bot.max_positions}) - cannot trade")
                    try:
                        rejection_reasons_for_log.append("max_positions")
                    except Exception:
                        pass
                else:
                    print(f"   [DIRECTION] Executing trade...")
                    trade = bot.paper_trader.place_trade(trading_symbol, signal, current_price)
                    if trade:
                        trades_made += 1
                        try:
                            trade_placed_for_log = True
                        except Exception:
                            pass
                        # Register trade for online learning
                        try:
                            direction_controller.register_trade(
                                trade_id=str(trade.id) if hasattr(trade, 'id') else str(id(trade)),
                                features=current_features,
                                sequences=sequence_features,
                                direction=decision.direction,
                            )
                        except Exception as reg_err:
                            logger.debug(f"Trade registration error: {reg_err}")
                        print(f"\n   [SIMON] SimonSays: {action}\n")
                        print(f"   [OK] TRADE PLACED: {action} @ ${current_price:.2f} (Strike: ${trade.strike_price:.2f})")
                    else:
                        print(f"   [ERROR] Trade failed to execute")

        # ================================================================
        # NORMAL SIGNAL PROCESSING (non-Q-scorer, non-direction mode)
        # ================================================================
        elif signal and signal.get('action') != 'HOLD' and not direction_active and not consensus_active:
            signals_generated += 1

            action = signal.get('action')
            confidence = signal.get('confidence', 0)
            predicted_return = signal.get('predicted_return', 0)
            momentum_5min = signal.get('momentum_5min', 0)
            volume_spike = signal.get('volume_spike', 1.0)

            print(f"   [ACTION] Non-HOLD signal: {action}")

            # SIMPLIFIED: No cooldown - let RL learn from all market conditions

            # ================================================================
            # UNIFIED RL POLICY (preferred) vs LEGACY THRESHOLD LEARNER
            # ================================================================
            if USE_UNIFIED_RL and unified_rl is not None:
                # Use unified RL policy for entry decision
                should_trade = False
                composite_score = 0.0
                
                # Build trade state for unified policy
                from backend.unified_rl_policy import TradeState
                
                # Get VIX
                vix_for_rl = vix_price if vix_price else 18.0
                
                # Get HMM regime data directly from bot
                hmm_trend_float = 0.5  # Default: neutral
                hmm_vol_float = 0.5    # Default: normal
                hmm_liq_float = 0.5    # Default: normal
                hmm_conf_float = 0.5   # Default: medium confidence
                
                if bot.current_hmm_regime:
                    # Map trend names to 0-1 scale (0=bearish, 0.5=neutral, 1=bullish)
                    trend_map = {
                        'Down': 0.0, 'Downtrend': 0.0, 'Bearish': 0.0,
                        'No Trend': 0.5, 'Neutral': 0.5, 'Sideways': 0.5,
                        'Up': 1.0, 'Uptrend': 1.0, 'Bullish': 1.0
                    }
                    trend_name = bot.current_hmm_regime.get('trend', 'Neutral')
                    hmm_trend_float = trend_map.get(trend_name, 0.5)
                    
                    # Map volatility names to 0-1 scale
                    vol_map = {'Low': 0.0, 'Normal': 0.5, 'High': 1.0}
                    vol_name = bot.current_hmm_regime.get('volatility', 'Normal')
                    hmm_vol_float = vol_map.get(vol_name, 0.5)
                    
                    # Map liquidity names to 0-1 scale
                    liq_map = {'Low': 0.0, 'Normal': 0.5, 'High': 1.0}
                    liq_name = bot.current_hmm_regime.get('liquidity', 'Normal')
                    hmm_liq_float = liq_map.get(liq_name, 0.5)
                    
                    # HMM confidence
                    hmm_conf_float = bot.current_hmm_regime.get('confidence', 0.5)

                # ============================================================
                # CONDOR REGIME FEATURES: Use Iron Condor logic as NN weights
                # Only computed when CONDOR_FEATURES_ENABLED=1
                # ============================================================
                condor_suitability = 0.5
                mtf_consensus = 0.5
                trending_signal_count = 0.0

                if CONDOR_FEATURES_ENABLED:
                    condor_features_dict = {
                        'vix_level': vix_for_rl,
                        'hmm_trend': hmm_trend_float,
                        'hmm_confidence': hmm_conf_float,
                        'momentum_5m': momentum_5min,
                        'confidence': confidence,
                    }
                    condor_regime = compute_condor_regime_features(condor_features_dict)
                    condor_suitability = condor_regime['condor_suitability']
                    mtf_consensus = condor_regime['mtf_consensus']
                    trending_signal_count = condor_regime['trending_signal_count']

                trade_state = TradeState(
                    is_in_trade=False,  # Entry decision
                    is_call=action == 'BUY_CALLS',
                    predicted_direction=predicted_return * 100 if predicted_return else 0.0,
                    prediction_confidence=confidence,  # YOUR MODEL'S CONFIDENCE
                    vix_level=vix_for_rl,
                    momentum_5m=momentum_5min,
                    volume_spike=volume_spike,
                    # HMM REGIME DATA (YOUR HMM MODEL)
                    hmm_trend=hmm_trend_float,
                    hmm_volatility=hmm_vol_float,
                    hmm_liquidity=hmm_liq_float,
                    hmm_confidence=hmm_conf_float,
                    # Greeks
                    estimated_theta_decay=-0.03,  # Assume 0DTE
                    estimated_delta=0.5,
                    # CONDOR REGIME FEATURES (Iron Condor logic as NN weights)
                    # Only used when CONDOR_FEATURES_ENABLED=1
                    condor_suitability=condor_suitability,
                    mtf_consensus=mtf_consensus,
                    trending_signal_count=trending_signal_count,
                )

                # ============================================================
                # V2 STATE: Build enhanced state with predictor embedding
                # ============================================================
                trade_state_v2 = None
                if USE_RL_POLICY_V2 and unified_rl_v2 is not None and TradeStateV2 is not None:
                    import numpy as np  # Local import for V2 state building
                    
                    # Extract predictor outputs from signal
                    neural_pred = (signal or {}).get('neural_prediction', {}) or {}
                    
                    # Get predictor embedding (64-dim)
                    pred_embedding = neural_pred.get('predictor_embedding')
                    if pred_embedding is None:
                        pred_embedding = neural_pred.get('embedding')
                    if pred_embedding is None:
                        pred_embedding = np.zeros(64)
                    elif isinstance(pred_embedding, list):
                        pred_embedding = np.array(pred_embedding)
                    
                    # Get direction probabilities [DOWN, NEUTRAL, UP]
                    dir_probs = neural_pred.get('direction_probs')
                    if dir_probs is None:
                        # Derive from predicted return
                        if predicted_return and predicted_return > 0.01:
                            dir_probs = np.array([0.1, 0.2, 0.7])  # Bullish
                        elif predicted_return and predicted_return < -0.01:
                            dir_probs = np.array([0.7, 0.2, 0.1])  # Bearish
                        else:
                            dir_probs = np.array([0.33, 0.34, 0.33])  # Neutral
                    elif isinstance(dir_probs, list):
                        dir_probs = np.array(dir_probs)
                    
                    # Get execution quality predictions
                    fillability = neural_pred.get('fillability', neural_pred.get('fillability_mean', 0.8))
                    exp_slippage = neural_pred.get('exp_slippage', neural_pred.get('exp_slippage_mean', 0.01))
                    exp_ttf = neural_pred.get('exp_ttf', neural_pred.get('exp_ttf_mean', 2.0))
                    pred_vol = neural_pred.get('volatility', neural_pred.get('volatility_mean', 0.02))
                    
                    trade_state_v2 = TradeStateV2(
                        is_in_trade=False,
                        is_call=action == 'BUY_CALLS',
                        vix_level=vix_for_rl,
                        momentum_5m=momentum_5min,
                        volume_spike=volume_spike,
                        # HMM Regime
                        hmm_trend=hmm_trend_float,
                        hmm_volatility=hmm_vol_float,
                        hmm_liquidity=hmm_liq_float,
                        hmm_confidence=hmm_conf_float,
                        # Greeks
                        estimated_theta_decay=-0.03,
                        estimated_delta=0.5,
                        # NEW V2: Predictor outputs
                        predictor_embedding=pred_embedding,
                        direction_probs=dir_probs,
                        raw_confidence=confidence,
                        predicted_return=predicted_return or 0.0,
                        predicted_volatility=pred_vol,
                        # Execution quality
                        fillability=fillability,
                        exp_slippage=exp_slippage,
                        exp_ttf=exp_ttf,
                    )

                # ============================================================
                # REGIME FILTER: Filter trades by market regime
                # TT_REGIME_FILTER=1: Skip sideways (trade only trends)
                # TT_REGIME_FILTER=2: Skip trends (trade only sideways - mean reversion)
                # TT_REGIME_FILTER=3: Use full RegimeFilter (Phase 12) with quality scoring
                # ============================================================
                regime_filter_mode = os.environ.get('TT_REGIME_FILTER', '0')
                regime_blocked = False
                if regime_filter_mode == '3' and action in ['BUY_CALLS', 'BUY_PUTS']:
                    print(f"   [DEBUG] Regime filter mode 3 detected, action={action}")

                if regime_filter_mode == '1':
                    # Mode 1: Skip sideways (only trade trends)
                    is_sideways = 0.4 <= hmm_trend_float <= 0.6
                    if is_sideways and action in ['BUY_CALLS', 'BUY_PUTS']:
                        should_trade = False
                        composite_score = 0.0
                        rl_details = {'reason': 'regime_filter_sideways', 'mode': 'regime_gate'}
                        print(f"   [REGIME FILTER] Skipping: Sideways market (hmm_trend={hmm_trend_float:.2f})")
                        try:
                            decision_stats["regime_filter_skip"] = decision_stats.get("regime_filter_skip", 0) + 1
                        except Exception:
                            pass
                        try:
                            _record_missed(sim_time, current_price, predicted_return, confidence, action, reason="regime_sideways")
                        except Exception:
                            pass
                        regime_blocked = True

                elif regime_filter_mode == '2':
                    # Mode 2: Skip trends (only trade sideways - mean reversion)
                    is_trending = hmm_trend_float < 0.4 or hmm_trend_float > 0.6
                    if is_trending and action in ['BUY_CALLS', 'BUY_PUTS']:
                        should_trade = False
                        composite_score = 0.0
                        rl_details = {'reason': 'regime_filter_trending', 'mode': 'regime_gate'}
                        print(f"   [REGIME FILTER] Skipping: Trending market (hmm_trend={hmm_trend_float:.2f})")
                        try:
                            decision_stats["regime_filter_skip"] = decision_stats.get("regime_filter_skip", 0) + 1
                        except Exception:
                            pass
                        try:
                            _record_missed(sim_time, current_price, predicted_return, confidence, action, reason="regime_trending")
                        except Exception:
                            pass
                        regime_blocked = True

                elif regime_filter_mode == '3':
                    # Mode 3: Full RegimeFilter with quality scoring (Phase 12)
                    # Import the regime filter
                    try:
                        from backend.regime_filter import get_regime_filter
                        rf = get_regime_filter()
                        rf_decision = rf.should_trade(
                            hmm_trend=hmm_trend_float,
                            hmm_volatility=hmm_vol_float,
                            hmm_liquidity=hmm_liq_float,
                            hmm_confidence=hmm_conf_float,
                            vix_level=vix_price if vix_price else 18.0,
                        )
                        if not rf_decision.can_trade:
                            should_trade = False
                            composite_score = rf_decision.quality_score
                            rl_details = {'reason': f'regime_filter_v3 ({rf_decision.veto_reason})', 'mode': 'regime_gate_v3', 
                                         'quality_score': rf_decision.quality_score, 'regime_state': rf_decision.regime_state}
                            print(f"   [REGIME FILTER V3] Blocked: {rf_decision.veto_reason} | quality={rf_decision.quality_score:.2f} state={rf_decision.regime_state}")
                            try:
                                decision_stats["regime_filter_v3_veto"] = decision_stats.get("regime_filter_v3_veto", 0) + 1
                            except Exception:
                                pass
                            regime_blocked = True
                        else:
                            # Trade approved - log quality info
                            if rf_decision.quality_score < 0.5:
                                print(f"   [REGIME FILTER V3] Marginal: quality={rf_decision.quality_score:.2f} state={rf_decision.regime_state}")
                            else:
                                print(f"   [REGIME FILTER V3] Approved: quality={rf_decision.quality_score:.2f} state={rf_decision.regime_state}")
                    except Exception as e:
                        print(f"   [REGIME FILTER V3] Error: {e}")

                # TRAINING MODE: In bandit mode, just follow the signal to gather data!
                # The learning happens from outcomes, not from being picky about entries
                if regime_blocked:
                    # Regime filter already blocked this trade - skip to exit logic
                    pass
                elif unified_rl.is_bandit_mode:
                    # Just follow the signal's action in training mode
                    if action in ['BUY_CALLS', 'BUY_PUTS']:
                        # === SIGNAL INVERSION TEST (Phase 26) ===
                        # If win rate is ~37%, inverting should give ~63%
                        INVERT_SIGNALS = os.environ.get('INVERT_SIGNALS', '0') == '1'
                        if INVERT_SIGNALS:
                            original_action = action
                            action = 'BUY_PUTS' if action == 'BUY_CALLS' else 'BUY_CALLS'
                            print(f"   [INVERTED] {original_action} → {action}")
                        # === END SIGNAL INVERSION ===

                        # HMM-PURE MODE: Ignore neural predictions, use only HMM regime
                        hmm_pure = os.environ.get('HMM_PURE_MODE', '0') == '1'
                        hmm_decision_made = False
                        if hmm_pure:
                            # Only trade when HMM is VERY confident in a clear trend
                            HMM_STRONG_BULLISH = 0.75
                            HMM_STRONG_BEARISH = 0.25
                            HMM_MIN_CONF = 0.80
                            HMM_MAX_VOL = 0.60  # Low volatility only

                            if hmm_conf_float >= HMM_MIN_CONF and hmm_vol_float < HMM_MAX_VOL:
                                if hmm_trend_float > HMM_STRONG_BULLISH:
                                    action = 'BUY_CALLS'
                                    should_trade = True
                                    composite_score = hmm_conf_float
                                    rl_details = {'reason': f'hmm_pure_bullish ({hmm_trend_float:.0%})', 'mode': 'hmm_pure'}
                                    hmm_decision_made = True
                                    print(f"   [HMM-PURE] TRADE: Strong bullish trend ({hmm_trend_float:.0%}, conf={hmm_conf_float:.0%})")
                                elif hmm_trend_float < HMM_STRONG_BEARISH:
                                    action = 'BUY_PUTS'
                                    should_trade = True
                                    composite_score = hmm_conf_float
                                    rl_details = {'reason': f'hmm_pure_bearish ({hmm_trend_float:.0%})', 'mode': 'hmm_pure'}
                                    hmm_decision_made = True
                                    print(f"   [HMM-PURE] TRADE: Strong bearish trend ({hmm_trend_float:.0%}, conf={hmm_conf_float:.0%})")
                                else:
                                    should_trade = False
                                    hmm_decision_made = True
                            else:
                                should_trade = False
                                hmm_decision_made = True

                        # QUALITY GATES - load from config (skipped in HMM-pure mode or SKIP_QUALITY_GATES=1)
                        # Phase 26 optimization: Higher thresholds dramatically reduce losses
                        skip_gates = os.environ.get('SKIP_QUALITY_GATES', '0') == '1'
                        train_min_conf = 0.0 if skip_gates else 0.30  # 30% conf (Phase 26: -1.8% vs -62%)
                        train_max_conf = 1.0  # Default: no max (set TRAIN_MAX_CONF to enable inverted filter)
                        train_min_abs_ret = 0.0 if skip_gates else 0.0013  # 0.13% edge (Phase 26 optimal)
                        # Inverted confidence filter: Data shows LOW conf = better WR for intraday
                        max_conf_env = os.environ.get('TRAIN_MAX_CONF', '')
                        if max_conf_env:
                            train_max_conf = float(max_conf_env)
                        if not hmm_decision_made:
                            try:
                                if bot_config and hasattr(bot_config, "config") and not skip_gates:
                                    ep = bot_config.config.get("architecture", {}).get("entry_policy", {})
                                    train_min_conf = float(ep.get("training_min_confidence", train_min_conf))
                                    train_min_abs_ret = float(ep.get("training_min_abs_predicted_return", train_min_abs_ret))
                            except Exception:
                                pass

                            # Require predicted edge consistent with direction
                            is_call = action == 'BUY_CALLS'
                            edge_ok = (predicted_return >= train_min_abs_ret) if is_call else (predicted_return <= -train_min_abs_ret)
                            conf_ok = confidence >= train_min_conf and confidence <= train_max_conf

                            # MARKET OPEN FILTER: Big wins happen at market open (09:31)
                            # Set MARKET_OPEN_MINUTES=30 to only trade in first 30 min after open
                            market_open_minutes = int(os.environ.get('MARKET_OPEN_MINUTES', '0'))
                            time_ok = True
                            if market_open_minutes > 0:
                                try:
                                    market_open_time = sim_time.replace(hour=9, minute=30, second=0, microsecond=0)
                                    mins_since_open = (sim_time - market_open_time).total_seconds() / 60
                                    time_ok = 0 <= mins_since_open <= market_open_minutes
                                    if not time_ok:
                                        pass  # Will be logged below
                                except Exception:
                                    time_ok = True  # Default to allowing if time calc fails

                            # ================================================================
                            # PHASE 41: NEW QUALITY FILTERS FOR 60%+ WIN RATE
                            # ================================================================
                            filter_ok = True
                            filter_reason = None

                            # TIME ZONE FILTER: Avoid first hour and last 30 min
                            if os.environ.get('TIME_ZONE_FILTER', '0') == '1':
                                try:
                                    market_open_time = sim_time.replace(hour=9, minute=30, second=0, microsecond=0)
                                    market_close_time = sim_time.replace(hour=16, minute=0, second=0, microsecond=0)
                                    mins_since_open = (sim_time - market_open_time).total_seconds() / 60
                                    mins_to_close = (market_close_time - sim_time).total_seconds() / 60
                                    start_threshold = int(os.environ.get('TIME_ZONE_START_MINUTES', '60'))
                                    end_threshold = int(os.environ.get('TIME_ZONE_END_MINUTES', '30'))
                                    if mins_since_open < start_threshold:
                                        filter_ok = False
                                        filter_reason = f"time_zone_start ({mins_since_open:.0f}m < {start_threshold}m)"
                                    elif mins_to_close < end_threshold:
                                        filter_ok = False
                                        filter_reason = f"time_zone_end ({mins_to_close:.0f}m to close)"
                                except Exception:
                                    pass

                            # VIX GOLDILOCKS: Only trade VIX 15-22
                            if filter_ok and os.environ.get('VIX_GOLDILOCKS_FILTER', '0') == '1':
                                try:
                                    vix_min = float(os.environ.get('VIX_GOLDILOCKS_MIN', '15.0'))
                                    vix_max = float(os.environ.get('VIX_GOLDILOCKS_MAX', '22.0'))
                                    # Use vix_for_rl which is already calculated earlier in this scope
                                    current_vix = vix_for_rl if 'vix_for_rl' in dir() else (vix_price if vix_price else 18.0)
                                    if current_vix < vix_min:
                                        filter_ok = False
                                        filter_reason = f"vix_low ({current_vix:.1f} < {vix_min})"
                                    elif current_vix > vix_max:
                                        filter_ok = False
                                        filter_reason = f"vix_high ({current_vix:.1f} > {vix_max})"
                                except Exception:
                                    pass

                            # MOMENTUM CONFIRMATION: Price must be moving in trade direction
                            if filter_ok and os.environ.get('MOMENTUM_CONFIRM_FILTER', '0') == '1':
                                try:
                                    mom_threshold = float(os.environ.get('MOMENTUM_MIN_STRENGTH', '0.001'))
                                    # Use momentum_5min which is already calculated in this scope
                                    mom_5m = momentum_5min if 'momentum_5min' in dir() else 0.0
                                    is_call = action == 'BUY_CALLS'
                                    if is_call and mom_5m < mom_threshold:
                                        filter_ok = False
                                        filter_reason = f"momentum_against_call ({mom_5m:.4f})"
                                    elif not is_call and mom_5m > -mom_threshold:
                                        filter_ok = False
                                        filter_reason = f"momentum_against_put ({mom_5m:.4f})"
                                except Exception:
                                    pass

                            # DAY OF WEEK FILTER: Skip Monday and Friday
                            if filter_ok and os.environ.get('DAY_OF_WEEK_FILTER', '0') == '1':
                                try:
                                    weekday = sim_time.weekday()  # 0=Mon, 4=Fri
                                    skip_monday = os.environ.get('SKIP_MONDAY', '1') == '1'
                                    skip_friday = os.environ.get('SKIP_FRIDAY', '1') == '1'
                                    skip_thursday = os.environ.get('SKIP_THURSDAY', '0') == '1'
                                    if skip_monday and weekday == 0:
                                        filter_ok = False
                                        filter_reason = "skip_monday"
                                    elif skip_friday and weekday == 4:
                                        filter_ok = False
                                        filter_reason = "skip_friday"
                                    elif skip_thursday and weekday == 3:
                                        filter_ok = False
                                        filter_reason = "skip_thursday"
                                except Exception:
                                    pass

                            # PHASE 44: Advanced signal integration
                            phase44_boost = 0.0
                            phase44_veto = False

                            # Multi-Indicator Stack: Boost confidence if composite agrees
                            if filter_ok and phase44_multi_indicator is not None:
                                try:
                                    # Get recent prices for indicators
                                    price_history = getattr(bot, '_price_history', [])
                                    if len(price_history) < 30:
                                        price_history = [spy_price] * 30  # Fallback
                                    indicators = phase44_multi_indicator.calculate_indicators(
                                        np.array(price_history[-30:])
                                    )
                                    composite = indicators.get('composite', 0)
                                    is_call = action == 'BUY_CALLS'

                                    # Boost if indicators agree with direction
                                    if (is_call and composite > 0.3) or (not is_call and composite < -0.3):
                                        phase44_boost += 0.05  # 5% boost
                                    # Veto if strong disagreement
                                    elif (is_call and composite < -0.5) or (not is_call and composite > 0.5):
                                        phase44_veto = True
                                        filter_ok = False
                                        filter_reason = f"multi_ind_veto (composite={composite:.2f})"
                                except Exception:
                                    pass

                            # GEX Signals: Adjust based on dealer positioning
                            if filter_ok and phase44_gex_signals is not None:
                                try:
                                    # Use available options data (simplified)
                                    gex_data = phase44_gex_signals.calculate_gex_proxy(
                                        call_volume=1000, put_volume=1000,  # Placeholders
                                        call_oi=5000, put_oi=5000,
                                        spot_price=spy_price,
                                        vix=vix_for_rl if 'vix_for_rl' in dir() else 18.0
                                    )
                                    dealer_gamma = gex_data.get('dealer_gamma', 0.5)
                                    is_call = action == 'BUY_CALLS'

                                    # Long gamma = expect mean reversion (fade extremes)
                                    # Short gamma = expect trend (follow direction)
                                    if dealer_gamma > 0.6:  # Dealers long gamma
                                        # Mean reversion likely - boost contrarian trades
                                        if (is_call and momentum_5min < 0) or (not is_call and momentum_5min > 0):
                                            phase44_boost += 0.03
                                    elif dealer_gamma < 0.4:  # Dealers short gamma
                                        # Trend likely - boost momentum trades
                                        if (is_call and momentum_5min > 0) or (not is_call and momentum_5min < 0):
                                            phase44_boost += 0.03
                                except Exception:
                                    pass

                            # Order Flow: Boost if flow agrees with direction
                            if filter_ok and phase44_order_flow is not None:
                                try:
                                    flow_data = phase44_order_flow.calculate_imbalance(
                                        price=spy_price,
                                        volume=1000000  # Placeholder
                                    )
                                    flow_imb = flow_data.get('flow_imbalance', 0)
                                    is_call = action == 'BUY_CALLS'

                                    # Boost if flow agrees
                                    if (is_call and flow_imb > 0.3) or (not is_call and flow_imb < -0.3):
                                        phase44_boost += 0.04
                                except Exception:
                                    pass

                            # Ensemble Predictor: Use combined model predictions
                            if filter_ok and phase44_ensemble is not None:
                                try:
                                    # Get feature buffer from bot
                                    if hasattr(bot, 'feature_buffer') and len(bot.feature_buffer) >= 30:
                                        # Get last 30 timesteps of features
                                        feature_seq = np.array(list(bot.feature_buffer)[-30:])
                                        feature_tensor = torch.tensor(feature_seq, dtype=torch.float32)

                                        # Get ensemble prediction
                                        ensemble_result = phase44_ensemble.predict_ensemble(feature_tensor)
                                        ensemble_pred = ensemble_result.get('ensemble_avg', 0)

                                        is_call = action == 'BUY_CALLS'
                                        # Ensemble predicts direction: positive = bullish, negative = bearish
                                        if (is_call and ensemble_pred > 0.1) or (not is_call and ensemble_pred < -0.1):
                                            phase44_boost += 0.06  # Strong agreement boost
                                        elif (is_call and ensemble_pred < -0.2) or (not is_call and ensemble_pred > 0.2):
                                            phase44_veto = True
                                            filter_ok = False
                                            filter_reason = f"ensemble_veto (pred={ensemble_pred:.3f})"
                                except Exception as e:
                                    pass  # Silently continue if ensemble fails

                            # ============= PHASE 52: SELECTIVE FILTERS =============
                            # These filters were previously NOT implemented (bug discovered 2026-01-05)

                            # MONDAY_ONLY: Only trade on Mondays
                            if filter_ok and os.environ.get('MONDAY_ONLY', '0') == '1':
                                weekday = sim_time.weekday()  # 0=Mon, 6=Sun
                                if weekday != 0:
                                    filter_ok = False
                                    filter_reason = f"monday_only (today={['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][weekday]})"

                            # FIRST_90_MIN_ONLY: Only trade first 90 minutes (9:30-11:00)
                            if filter_ok and os.environ.get('FIRST_90_MIN_ONLY', '0') == '1':
                                market_open_time = sim_time.replace(hour=9, minute=30, second=0, microsecond=0)
                                mins_since_open = (sim_time - market_open_time).total_seconds() / 60
                                if mins_since_open < 0 or mins_since_open > 90:
                                    filter_ok = False
                                    filter_reason = f"first_90_min_only ({mins_since_open:.0f}m since open)"

                            # MIN_ABS_MOMENTUM: Require minimum momentum magnitude
                            if filter_ok and os.environ.get('MIN_ABS_MOMENTUM', ''):
                                min_mom = float(os.environ.get('MIN_ABS_MOMENTUM', '0'))
                                mom_5m = momentum_5min if 'momentum_5min' in dir() else 0.0
                                if abs(mom_5m) < min_mom:
                                    filter_ok = False
                                    filter_reason = f"min_abs_momentum ({abs(mom_5m):.5f} < {min_mom})"

                            # SENTIMENT_FEAR_GREED_MAX: Only trade when Fear & Greed below threshold (fear = opportunity)
                            if filter_ok and os.environ.get('SENTIMENT_FEAR_GREED_MAX', ''):
                                max_fg = float(os.environ.get('SENTIMENT_FEAR_GREED_MAX', '100'))
                                fg_value = 50  # Default neutral
                                if hasattr(bot, 'last_features') and bot.last_features:
                                    fg_value = bot.last_features.get('fear_greed_value', 50)
                                if fg_value > max_fg:
                                    filter_ok = False
                                    filter_reason = f"fear_greed_high ({fg_value:.0f} > {max_fg})"

                            # TDA_REGIME_FILTER: Only trade when TDA complexity is favorable
                            if filter_ok and os.environ.get('TDA_REGIME_FILTER', '0') == '1':
                                max_complexity = float(os.environ.get('TDA_MAX_COMPLEXITY', '0.5'))
                                tda_complexity = 0.3  # Default
                                if hasattr(bot, 'last_features') and bot.last_features:
                                    tda_complexity = bot.last_features.get('tda_complexity', 0.3)
                                if tda_complexity > max_complexity:
                                    filter_ok = False
                                    filter_reason = f"tda_complexity_high ({tda_complexity:.3f} > {max_complexity})"

                            # ============= END PHASE 52 FILTERS =============

                            # Apply Phase 44 boost to confidence (for logging)
                            if phase44_boost > 0:
                                print(f"   [PHASE44] Signal boost: +{phase44_boost:.1%}")

                            if conf_ok and edge_ok and time_ok and filter_ok:
                                should_trade = True
                                composite_score = confidence
                                rl_details = {'reason': f'bandit_follow_signal ({action})', 'mode': 'bandit'}
                                print(f"   [UNIFIED BANDIT] Following signal: {action} (conf={confidence:.1%}, edge={predicted_return:+.2%})")
                                try:
                                    decision_stats["bandit_follow"] += 1
                                except Exception:
                                    pass
                            else:
                                should_trade = False
                                composite_score = confidence
                                why = []
                                if not conf_ok:
                                    if confidence < train_min_conf:
                                        why.append(f"conf<{train_min_conf:.0%}")
                                    if confidence > train_max_conf:
                                        why.append(f"conf>{train_max_conf:.0%} (inverted)")
                                if not edge_ok:
                                    why.append(f"|ret|<{train_min_abs_ret*100:.2f}%")
                                if not time_ok:
                                    why.append(f"outside market open ({market_open_minutes}m)")
                                if not filter_ok and filter_reason:
                                    why.append(filter_reason)
                                rl_details = {'reason': f"bandit_gate ({', '.join(why)})", 'mode': 'bandit'}
                                print(f"   [UNIFIED BANDIT] Skipping: {', '.join(why)} (conf={confidence:.1%}, edge={predicted_return:+.2%})")
                                try:
                                    if (not conf_ok) and (not edge_ok):
                                        decision_stats["bandit_skip_both"] += 1
                                    elif not conf_ok:
                                        decision_stats["bandit_skip_conf"] += 1
                                    elif not edge_ok:
                                        decision_stats["bandit_skip_edge"] += 1
                                    else:
                                        decision_stats["bandit_skip_other"] += 1
                                except Exception:
                                    pass
                                # Track as missed opportunity (report-only)
                                try:
                                    try:
                                        rejection_reasons_for_log.append("bandit_gate")
                                    except Exception:
                                        pass
                                    _record_missed(sim_time, current_price, predicted_return, confidence, action, reason="bandit_gate")
                                except Exception:
                                    pass
                                # Record gate dataset candidate as well (supervised)
                                try:
                                    feats = _gate_features_from_signal(signal, hmm_trend_float, hmm_vol_float, hmm_liq_float, hmm_conf_float)
                                    _record_gate_candidate(sim_time, current_price, action, feats, reason="bandit_gate")
                                except Exception:
                                    pass
                    else:
                        should_trade = False
                        composite_score = 0.5
                        rl_details = {'reason': 'not_call_or_put', 'mode': 'bandit'}
                        print(f"   [UNIFIED BANDIT] Skipping: action is {action}")
                        try:
                            decision_stats["bandit_skip_other"] += 1
                        except Exception:
                            pass
                else:
                    # After bandit mode, use full RL decision
                    # ============================================================
                    # V2 POLICY: Use enhanced RL with embedding + EV gating
                    # ============================================================
                    if USE_RL_POLICY_V2 and unified_rl_v2 is not None and trade_state_v2 is not None:
                        rl_action, rl_confidence, rl_details = unified_rl_v2.select_action(trade_state_v2, deterministic=False)
                        
                        # Map V2 actions (5 actions: HOLD, BUY_CALL, BUY_PUT, EXIT_FAST, EXIT_PATIENT)
                        if rl_action == unified_rl_v2.BUY_CALL:
                            should_trade = True
                            composite_score = rl_confidence
                            action = 'BUY_CALLS'
                            signal['action'] = 'BUY_CALLS'
                        elif rl_action == unified_rl_v2.BUY_PUT:
                            should_trade = True
                            composite_score = rl_confidence
                            action = 'BUY_PUTS'
                            signal['action'] = 'BUY_PUTS'
                        else:
                            # HOLD, EXIT_FAST, EXIT_PATIENT all result in no entry
                            should_trade = False
                            composite_score = rl_confidence
                        
                        try:
                            if should_trade:
                                decision_stats['v2_fullrl_follow'] = decision_stats.get('v2_fullrl_follow', 0) + 1
                            else:
                                decision_stats['v2_fullrl_skip'] = decision_stats.get('v2_fullrl_skip', 0) + 1
                        except Exception:
                            pass
                        
                        action_names = ['HOLD', 'BUY_CALL', 'BUY_PUT', 'EXIT_FAST', 'EXIT_PATIENT']
                        print(f"   [V2 RL] Action: {action_names[rl_action]}, Score: {composite_score:.2f}, EV: {rl_details.get('ev', 0):.4f}")
                        print(f"   [V2] Reason: {rl_details.get('reason', 'N/A')}")
                        
                        signal['rl_details'] = rl_details
                        signal['unified_rl_action'] = rl_action
                        signal['rl_version'] = 'v2'
                    else:
                        # Fallback to V1 policy
                        rl_action, rl_confidence, rl_details = unified_rl.select_action(trade_state, deterministic=False)
                    
                        # Map unified action to trade decision
                    # IMPORTANT: Let RL choose direction (CALL vs PUT).
                    # If we require RL to match the signal's suggested direction, RL never
                    # learns the "other lever" when the signal stream is biased.
                    if rl_action == unified_rl.BUY_CALL:
                        should_trade = True
                        composite_score = rl_confidence
                        action = 'BUY_CALLS'
                        signal['action'] = 'BUY_CALLS'
                    elif rl_action == unified_rl.BUY_PUT:
                        should_trade = True
                        composite_score = rl_confidence
                        action = 'BUY_PUTS'
                        signal['action'] = 'BUY_PUTS'
                    else:
                        should_trade = False
                        composite_score = rl_confidence
                    try:
                        if should_trade:
                            decision_stats["fullrl_follow"] += 1
                        else:
                            decision_stats["fullrl_skip"] += 1
                    except Exception:
                        pass
                    
                    print(f"   [UNIFIED RL] Action: {['HOLD', 'BUY_CALL', 'BUY_PUT', 'EXIT'][rl_action]}, Score: {composite_score:.2f}")
                    print(f"   [UNIFIED] Reason: {rl_details.get('reason', 'N/A')}")
                    
                    # Store details for later (only in full RL mode where rl_action is defined)
                    signal['rl_details'] = rl_details
                    signal['unified_rl_action'] = rl_action

                    # ------------------------------------------------------------
                    # TRAINING EXPLORATION BOOST: "epsilon trade" when RL holds
                    # ------------------------------------------------------------
                    # In long runs, RL can get conservative and HOLD too often,
                    # which starves the system of fresh experiences.
                    # In time-travel training only, if the upstream signal is non-HOLD,
                    # we occasionally force an entry to keep exploration alive.
                    try:
                        epsilon_trade = float(os.environ.get("TT_EPSILON_TRADE", "0.05"))
                        eps_min_conf = float(os.environ.get("TT_EPSILON_MIN_CONF", "0.15"))
                        if (not should_trade) and (epsilon_trade > 0) and (confidence >= eps_min_conf):
                            import random
                            if random.random() < epsilon_trade:
                                should_trade = True
                                composite_score = float(confidence)
                                # Follow the signal action (or keep what RL already overwrote).
                                if action in ("BUY_CALLS", "BUY_PUTS"):
                                    signal["action"] = action
                                rl_details = dict(rl_details or {})
                                rl_details["epsilon_forced_trade"] = True
                                rl_details["epsilon_trade"] = epsilon_trade
                                signal["rl_details"] = rl_details
                                print(f"   [UNIFIED] Îµ-trade forced entry (eps={epsilon_trade:.2%}, conf={confidence:.1%})")
                                try:
                                    decision_stats["fullrl_epsilon_trade"] += 1
                                except Exception:
                                    pass
                    except Exception:
                        pass

                # Record gate dataset candidate for any non-HOLD proposal (supervised)
                try:
                    feats = _gate_features_from_signal(signal, hmm_trend_float, hmm_vol_float, hmm_liq_float, hmm_conf_float)
                    _record_gate_candidate(sim_time, current_price, action, feats, reason="signal")
                except Exception:
                    pass

                # ============================================================
                # RL THRESHOLD LEARNER FILTER (Adaptive Quality Gate)
                # ============================================================
                # If enabled, use RLThresholdPolicy as an additional quality filter
                # It learns optimal weights for trading factors from outcomes
                rl_filter_passed = True
                rl_filter_score = composite_score

                if should_trade and USE_RL_THRESHOLD_FILTER and rl_threshold_learner is not None:
                    try:
                        # Get volume spike and momentum from signal
                        filter_volume_spike = float(signal.get('volume_spike', 1.0)) if isinstance(signal, dict) else 1.0
                        filter_momentum = float(signal.get('momentum', momentum_5min)) if isinstance(signal, dict) else momentum_5min

                        # Evaluate signal using learned weights
                        rl_filter_passed, rl_filter_score, rl_filter_details = rl_threshold_learner.evaluate_signal(
                            confidence=confidence,
                            predicted_return=predicted_return,
                            momentum=filter_momentum,
                            volume_spike=filter_volume_spike,
                            vix_level=vix_for_rl,  # Use vix_for_rl from earlier
                            hmm_trend=int(hmm_trend_float * 2),  # Convert 0-1 to 0-2
                            hmm_vol=int(hmm_vol_float * 2),
                            hmm_liq=int(hmm_liq_float * 2),
                            time_of_day=0.5,  # Default
                            sector_strength=0.0,
                            recent_win_rate=0.5,
                            drawdown=0.0,
                            current_price=current_price,
                            timestamp=sim_time,
                            proposed_action=action
                        )

                        if not rl_filter_passed:
                            should_trade = False
                            print(f"   [RL-FILTER] Blocked: score {rl_filter_score:.3f} < threshold {rl_filter_details.get('threshold', 0):.3f}")
                            try:
                                decision_stats["rl_filter_blocked"] = decision_stats.get("rl_filter_blocked", 0) + 1
                            except Exception:
                                pass
                        else:
                            print(f"   [RL-FILTER] Passed: score {rl_filter_score:.3f} >= threshold {rl_filter_details.get('threshold', 0):.3f}")
                            # Store RL filter details in signal for feature attribution (Phase 14)
                            if isinstance(signal, dict):
                                signal['rl_details'] = rl_filter_details
                    except Exception as e:
                        logger.warning(f"[RL-FILTER] Error evaluating signal: {e}")
                        rl_filter_passed = True  # Don't block on error

                # ============================================================
                # PNL CALIBRATION GATE (Phase 13/14)
                # ============================================================
                pnl_cal_blocked = False
                calibrated_pnl_prob = None  # Store for trade record
                if pnl_calibration_tracker:
                    try:
                        pnl_metrics = pnl_calibration_tracker.get_metrics()
                        pnl_samples = pnl_metrics.get('pnl_sample_count', 0)

                        # Always calculate calibrated confidence for tracking
                        raw_conf = signal.get('confidence', 0.5)
                        if pnl_samples >= PNL_CAL_MIN_SAMPLES:
                            calibrated_pnl_prob = pnl_calibration_tracker.calibrate_pnl(raw_conf)
                            # Store in signal for trade record (Phase 14)
                            signal['calibrated_confidence'] = calibrated_pnl_prob

                            # Gate if enabled
                            if should_trade and PNL_CAL_GATE_ENABLED:
                                if calibrated_pnl_prob < PNL_CAL_MIN_PROB:
                                    pnl_cal_blocked = True
                                    should_trade = False
                                    print(f"   [PNL_CAL] BLOCKED: P(profit)={calibrated_pnl_prob:.1%} < {PNL_CAL_MIN_PROB:.0%} threshold")
                                    try:
                                        decision_stats["pnl_cal_blocked"] = decision_stats.get("pnl_cal_blocked", 0) + 1
                                    except Exception:
                                        pass
                                else:
                                    print(f"   [PNL_CAL] PASSED: P(profit)={calibrated_pnl_prob:.1%} >= {PNL_CAL_MIN_PROB:.0%}")
                        else:
                            print(f"   [PNL_CAL] Learning phase ({pnl_samples}/{PNL_CAL_MIN_SAMPLES} samples)")
                    except Exception as e:
                        logger.debug(f"[PNL_CAL] Error: {e}")

                # ============================================================
                # EXECUTE TRADE FOR UNIFIED RL
                # ============================================================
                if should_trade:
                    current_positions = len(bot.paper_trader.active_trades)
                    if current_positions >= bot.max_positions:
                        print(f"   [WARN] Max positions reached ({current_positions}/{bot.max_positions}) - cannot trade")
                        try:
                            rejection_reasons_for_log.append("max_positions")
                        except Exception:
                            pass
                        try:
                            decision_stats["blocked_max_positions"] += 1
                        except Exception:
                            pass
                    else:
                        print(f"   [UNIFIED] Placing trade...")
                        trade = bot.paper_trader.place_trade(trading_symbol, signal, current_price)
                        if trade:
                            trades_made += 1
                            try:
                                trade_placed_for_log = True
                            except Exception:
                                pass
                            try:
                                decision_stats["trade_placed"] += 1
                            except Exception:
                                pass
                            # Store signal for later attribution when the trade closes
                            try:
                                signal_by_trade_id[trade.id] = dict(signal) if isinstance(signal, dict) else {}
                            except Exception:
                                signal_by_trade_id[trade.id] = {}

                            # PNL CALIBRATION: Record trade entry for P(profit) learning
                            if pnl_calibration_tracker:
                                try:
                                    direction = 'UP' if action in ['BUY_CALLS', 'BUY_CALL'] else 'DOWN'
                                    pnl_calibration_tracker.record_trade_entry(
                                        trade_id=str(trade.id),
                                        confidence=signal.get('confidence', 0.5),
                                        predicted_direction=direction,
                                        entry_price=current_price,
                                        entry_option_price=trade.premium_paid,
                                        timestamp=sim_time
                                    )
                                except Exception as e:
                                    logger.debug(f"[PNL_CAL] Could not record trade entry: {e}")
                            print(f"\n   [SIMON] SimonSays: {action}\n")
                            
                            # SHADOW TRADING: Also place in Tradier Sandbox if enabled
                            if tradier_shadow:
                                try:
                                    print(f"   [SHADOW] Mirroring trade to Tradier Sandbox...")
                                    shadow_signal = signal.copy()
                                    shadow_signal['quantity'] = trade.quantity
                                    shadow_signal['strike_price'] = trade.strike_price
                                    shadow_trade = tradier_shadow.place_trade(trading_symbol, shadow_signal, current_price)
                                    if shadow_trade:
                                        print(f"   [SHADOW] âœ“ Trade placed in Tradier Sandbox!")
                                    else:
                                        print(f"   [SHADOW] âœ— Failed to place in Tradier Sandbox")
                                except Exception as e:
                                    print(f"   [SHADOW] âœ— Error: {e}")
                            
                            print(f"   [OK] TRADE PLACED: {action} @ ${current_price:.2f} (Strike: ${trade.strike_price:.2f})")
                            logger.info(f"[UNIFIED] Trade placed: {action} at ${current_price:.2f} | Strike: ${trade.strike_price:.2f}")
                        else:
                            print(f"   [ERROR] Trade failed to execute")
                            try:
                                rejection_reasons_for_log.append("trade_failed")
                            except Exception:
                                pass
                            try:
                                decision_stats["trade_failed"] += 1
                            except Exception:
                                pass
                else:
                    print(f"   [UNIFIED] RL holding - no trade (score: {composite_score:.3f})")

                # Periodic diagnostics (keeps logs readable)
                try:
                    if (cycles % 250) == 0 and cycles > 0:
                        _log_decision_stats(prefix=f"[DIAG] cycle={cycles}")
                except Exception:
                    pass
                
            elif rl_threshold_learner:
                # QUALITY OVER QUANTITY: Only trade strong signals
                MIN_CONFIDENCE_FLOOR = 0.30  # 30% - wait for strong signals
                
                should_trade = False
                composite_score = 0.0
                
                if confidence < MIN_CONFIDENCE_FLOOR:
                    print(f"   [REJECT] Confidence too low ({confidence:.1%} < {MIN_CONFIDENCE_FLOOR:.0%})")
                else:
                    # Extract HMM state from signal or bot for RL context
                    hmm_trend = 1  # Default: neutral
                    hmm_vol = 1   # Default: normal
                    hmm_liq = 1   # Default: normal
                    
                    if bot.current_hmm_regime:
                        # Map trend names to integers (0=down, 1=neutral, 2=up)
                        trend_map = {'Down': 0, 'Downtrend': 0, 'No Trend': 1, 'Neutral': 1, 'Up': 2, 'Uptrend': 2}
                        vol_map = {'Low': 0, 'Normal': 1, 'High': 2}
                        liq_map = {'Low': 0, 'Normal': 1, 'High': 2}
                        
                        hmm_trend = trend_map.get(bot.current_hmm_regime.get('trend', 'Neutral'), 1)
                        hmm_vol = vol_map.get(bot.current_hmm_regime.get('volatility', 'Normal'), 1)
                        hmm_liq = liq_map.get(bot.current_hmm_regime.get('liquidity', 'Normal'), 1)
                    
                    # Get VIX level and advanced VIX features
                    vix_for_rl = vix_price if vix_price else 18.0
                    vix_bb_pos = 0.5  # Default: middle of BB
                    vix_roc = 0.0     # Default: no change
                    vix_percentile = 0.5  # Default: median
                    
                    # Calculate VIX BB, ROC, and percentile from VIX data
                    try:
                        vix_data = tt_source.get_data('VIX', period='1d', interval=TT_DATA_INTERVAL)
                        if vix_data is not None and len(vix_data) >= 20:
                            vix_close = vix_data['close'].values
                            # VIX Bollinger Band position
                            vix_sma = float(np.mean(vix_close[-20:]))
                            vix_std = float(np.std(vix_close[-20:]))
                            if vix_std > 0:
                                vix_bb_upper = vix_sma + 2.0 * vix_std
                                vix_bb_lower = vix_sma - 2.0 * vix_std
                                vix_bb_pos = (vix_close[-1] - vix_bb_lower) / (vix_bb_upper - vix_bb_lower)
                                vix_bb_pos = float(min(max(vix_bb_pos, 0.0), 1.0))
                            # VIX rate of change (5-period)
                            if len(vix_close) >= 6:
                                vix_roc = float((vix_close[-1] - vix_close[-6]) / vix_close[-6]) if vix_close[-6] != 0 else 0.0
                            # VIX percentile
                            vix_percentile = float(np.sum(vix_close[-100:] < vix_close[-1]) / min(len(vix_close), 100))
                    except:
                        pass
                    
                    # Calculate price jerk (rate of change of acceleration)
                    price_jerk = 0.0
                    try:
                        price_data = tt_source.get_data(trading_symbol, period='1d', interval=TT_DATA_INTERVAL)
                        if price_data is not None and len(price_data) >= 10:
                            close_prices = price_data['close'].values
                            velocity = np.diff(close_prices)  # 1st derivative
                            acceleration = np.diff(velocity)  # 2nd derivative
                            jerk = np.diff(acceleration)      # 3rd derivative
                            if len(jerk) > 0 and close_prices[-1] != 0:
                                price_jerk = float(jerk[-1] / close_prices[-1])
                    except:
                        pass

                    # NEW: Add technical indicators for consensus Signal 5
                    if add_indicators_to_signal is not None and signal:
                        try:
                            price_data_ti = tt_source.get_data(trading_symbol, period='1d', interval=TT_DATA_INTERVAL)
                            spy_data_ti = tt_source.get_data('SPY', period='1d', interval=TT_DATA_INTERVAL)
                            qqq_data_ti = tt_source.get_data('QQQ', period='1d', interval=TT_DATA_INTERVAL)
                            add_indicators_to_signal(signal, price_data_ti, spy_data_ti, qqq_data_ti)
                        except:
                            pass

                    # Calculate time of day (0=market open, 1=market close)
                    # Market hours: 9:30 AM - 4:00 PM ET = 390 minutes
                    market_hour = sim_time.hour + sim_time.minute / 60.0
                    time_of_day = max(0.0, min(1.0, (market_hour - 9.5) / 6.5))  # 9:30-16:00
                    
                    # Calculate sector strength from QQQ momentum (tech sector proxy)
                    sector_strength = 0.0  # Default: neutral
                    qqq_price = tt_source.get_current_price('QQQ')
                    if qqq_price and hasattr(tt_source, 'get_data'):
                        try:
                            qqq_data = tt_source.get_data('QQQ', period='1d', interval=TT_DATA_INTERVAL)
                            if qqq_data is not None and len(qqq_data) >= 10:
                                qqq_close = qqq_data['close'].values
                                qqq_momentum = (qqq_close[-1] / qqq_close[-10] - 1) * 100  # 10-bar momentum
                                sector_strength = max(-1.0, min(1.0, qqq_momentum))  # Clamp to [-1, 1]
                        except:
                            pass
                    
                    # Get recent win rate and drawdown from paper trader
                    recent_win_rate = 0.5  # Default: 50%
                    current_drawdown = 0.0  # Default: no drawdown
                    
                    if bot.paper_trader.total_trades > 0:
                        recent_win_rate = bot.paper_trader.winning_trades / max(1, bot.paper_trader.total_trades)
                    
                    # Calculate drawdown
                    peak_balance = max(bot.paper_trader.initial_balance, 
                                      bot.paper_trader.initial_balance + bot.paper_trader.total_profit_loss)
                    current_drawdown = (peak_balance - bot.paper_trader.current_balance) / peak_balance if peak_balance > 0 else 0
                    current_drawdown = max(0.0, min(1.0, current_drawdown))  # Clamp to [0, 1]
                    
                    # REGIME AWARENESS: Log regime but DON'T penalize heavily
                    # Output6 doesn't penalize - let RL learn what regimes to trade
                    regime_penalty = 1.0  # No penalty - let RL learn
                    
                    if bot.current_hmm_regime:
                        hmm_regime_confidence = bot.current_hmm_regime.get('confidence', 0.0)
                        trend_name = bot.current_hmm_regime.get('trend', '')
                        vol_name = bot.current_hmm_regime.get('volatility', '')
                        
                        # Just log the regime for awareness (RL will learn from outcomes)
                        if trend_name in ['No Trend', 'Neutral'] and hmm_regime_confidence > 0.70:
                            regime_penalty = 0.85  # MILD penalty: 15% reduction (was 40-60%!)
                            print(f"   [REGIME] {trend_name} + {vol_name} Vol ({hmm_regime_confidence:.0%}) - mild 15% reduction")
                    
                    # Apply mild regime adjustment
                    adjusted_confidence = confidence * regime_penalty
                    
                    should_trade, composite_score, rl_details = rl_threshold_learner.evaluate_signal(
                        confidence=adjusted_confidence,  # USE ADJUSTED CONFIDENCE
                        predicted_return=abs(predicted_return),
                        momentum=abs(momentum_5min),
                        volume_spike=volume_spike,
                        vix_level=vix_for_rl,
                        vix_bb_pos=vix_bb_pos,
                        vix_roc=vix_roc,
                        vix_percentile=vix_percentile,
                        hmm_trend=hmm_trend,
                        hmm_vol=hmm_vol,
                        hmm_liq=hmm_liq,
                        time_of_day=time_of_day,
                        sector_strength=sector_strength,
                        recent_win_rate=recent_win_rate,
                        drawdown=current_drawdown,
                        price_jerk=price_jerk,
                        current_price=current_price,  # For counterfactual learning
                        timestamp=sim_time,  # For counterfactual learning
                        proposed_action=action  # For direction-aware counterfactual
                    )
                    
                    # Store RL details in signal for later learning
                    signal['rl_details'] = rl_details
                
                # Periodically analyze missed opportunities (every 10 cycles)
                if cycles % 10 == 0:
                    missed_stats = rl_threshold_learner.analyze_missed_opportunities(
                        current_price=current_price,
                        current_timestamp=sim_time,
                        lookback_minutes=30
                    )
                    if missed_stats['total_analyzed'] > 0:
                        print(f"   [COUNTERFACTUAL] Analyzed {missed_stats['total_analyzed']} missed signals: "
                              f"{missed_stats['missed_winners']} would have won, {missed_stats['missed_losers']} avoided losses")
                    
                    # Save analyzed signals for dashboard visualization
                    try:
                        import json
                        analyzed_signals = rl_threshold_learner.get_analyzed_signals(limit=200)
                        rl_stats = rl_threshold_learner.get_stats()
                        missed_data = {
                            'signals': analyzed_signals,
                            'stats': {
                                'total_missed_winners': rl_stats.get('total_missed_winners', 0),
                                'total_avoided_losers': rl_stats.get('total_missed_losers', 0),
                                'miss_rate': rl_stats.get('counterfactual_miss_rate', 0),
                                'pending_signals': rl_stats.get('pending_rejected_signals', 0)
                            },
                            'updated_at': str(sim_time)
                        }
                        os.makedirs('data', exist_ok=True)
                        with open('data/missed_opportunities.json', 'w') as f:
                            json.dump(missed_data, f)
                    except Exception as e:
                        logger.warning(f"Could not save missed opportunities: {e}")
                
                if should_trade:
                    # Check position limits
                    current_positions = len(bot.paper_trader.active_trades)
                    if current_positions >= bot.max_positions:
                        print(f"   [WARN] Max positions reached ({current_positions}/{bot.max_positions}) - cannot trade")
                        try:
                            rejection_reasons_for_log.append("max_positions")
                        except Exception:
                            pass
                    else:
                        # Execute trade in simulation
                        print(f"   [RL] RL approved! Placing trade...")
                        trade = bot.paper_trader.place_trade(trading_symbol, signal, current_price)
                        if trade:
                            trades_made += 1
                            try:
                                trade_placed_for_log = True
                            except Exception:
                                pass
                            print(f"\n   [SIMON] SimonSays: {action}\n")
                            
                            # SHADOW TRADING: Also place in Tradier Sandbox if enabled
                            if tradier_shadow:
                                try:
                                    print(f"   [SHADOW] Mirroring trade to Tradier Sandbox...")
                                    # Pass quantity and strike from paper trade to avoid recalculating
                                    shadow_signal = signal.copy()
                                    shadow_signal['quantity'] = trade.quantity
                                    shadow_signal['strike_price'] = trade.strike_price
                                    logger.info(f"[SHADOW] Using paper trade params: {trade.quantity} contracts @ ${trade.strike_price:.2f} strike")
                                    shadow_trade = tradier_shadow.place_trade(trading_symbol, shadow_signal, current_price)
                                    if shadow_trade:
                                        print(f"   [SHADOW] âœ“ Trade placed in Tradier Sandbox!")
                                        logger.info(f"[SHADOW] Trade mirrored to Tradier: {action}")
                                    else:
                                        print(f"   [SHADOW] âœ— Failed to place in Tradier Sandbox")
                                        logger.warning(f"[SHADOW] Failed to mirror trade")
                                except Exception as e:
                                    print(f"   [SHADOW] âœ— Error: {e}")
                                    logger.error(f"[SHADOW] Error mirroring trade: {e}")
                            print(f"   [OK] TRADE PLACED: {action} @ ${current_price:.2f} (Strike: ${trade.strike_price:.2f})")
                            logger.info(f"[SIMON] SimonSays: {action}")
                            logger.info(f"[TRADE] TRADE PLACED: {action} at ${current_price:.2f} | Strike: ${trade.strike_price:.2f} | Qty: {trade.quantity}")
                        else:
                            print(f"   [ERROR] Trade failed to execute")
                            try:
                                rejection_reasons_for_log.append("trade_failed")
                            except Exception:
                                pass
                else:
                    print(f"   [REJECT] RL rejected (composite score: {composite_score:.3f}, threshold: {rl_threshold_learner.composite_threshold:.3f})")
                    try:
                        rejection_reasons_for_log.append("rl_rejected")
                    except Exception:
                        pass
            else:
                # No RL - use basic threshold
                print(f"   [CHECK] Checking confidence threshold...")
                
                # QUALITY OVER QUANTITY: Only trade strong signals
                MIN_CONFIDENCE_FLOOR = 0.30  # 30% - wait for strong signals
                
                if confidence < MIN_CONFIDENCE_FLOOR:
                    print(f"   [REJECT] Confidence too low ({confidence:.1%} < {MIN_CONFIDENCE_FLOOR:.0%})")
                    try:
                        rejection_reasons_for_log.append("confidence_floor")
                    except Exception:
                        pass
                elif confidence >= bot.min_confidence_threshold:
                    current_positions = len(bot.paper_trader.active_trades)
                    if current_positions < bot.max_positions:
                        print(f"   [OK] Confidence met! Placing trade...")
                        trade = bot.paper_trader.place_trade(trading_symbol, signal, current_price)
                        if trade:
                            trades_made += 1
                            try:
                                trade_placed_for_log = True
                            except Exception:
                                pass
                            print(f"\n   [SIMON] SimonSays: {action}\n")
                            
                            # SHADOW TRADING: Also place in Tradier Sandbox if enabled
                            if tradier_shadow:
                                try:
                                    print(f"   [SHADOW] Mirroring trade to Tradier Sandbox...")
                                    # Pass quantity and strike from paper trade to avoid recalculating
                                    shadow_signal = signal.copy()
                                    shadow_signal['quantity'] = trade.quantity
                                    shadow_signal['strike_price'] = trade.strike_price
                                    logger.info(f"[SHADOW] Using paper trade params: {trade.quantity} contracts @ ${trade.strike_price:.2f} strike")
                                    shadow_trade = tradier_shadow.place_trade(trading_symbol, shadow_signal, current_price)
                                    if shadow_trade:
                                        print(f"   [SHADOW] âœ“ Trade placed in Tradier Sandbox!")
                                        logger.info(f"[SHADOW] Trade mirrored to Tradier: {action}")
                                    else:
                                        print(f"   [SHADOW] âœ— Failed to place in Tradier Sandbox")
                                        logger.warning(f"[SHADOW] Failed to mirror trade")
                                except Exception as e:
                                    print(f"   [SHADOW] âœ— Error: {e}")
                                    logger.error(f"[SHADOW] Error mirroring trade: {e}")
                            
                            print(f"   [OK] TRADE PLACED: {action} @ ${current_price:.2f} (Strike: ${trade.strike_price:.2f})")
                            logger.info(f"[SIMON] SimonSays: {action}")
                            logger.info(f"[TRADE] TRADE PLACED: {action} at ${current_price:.2f} | Strike: ${trade.strike_price:.2f} | Qty: {trade.quantity}")
                        else:
                            print(f"   [ERROR] Trade failed to execute")
                            try:
                                rejection_reasons_for_log.append("trade_failed")
                            except Exception:
                                pass
                    else:
                        print(f"   [WARN] Max positions reached ({current_positions}/{bot.max_positions})")
                        try:
                            rejection_reasons_for_log.append("max_positions")
                        except Exception:
                            pass
                else:
                    print(f"   [REJECT] Confidence too low ({confidence:.1%} < {bot.min_confidence_threshold:.1%})")
                    try:
                        rejection_reasons_for_log.append("confidence_threshold")
                    except Exception:
                        pass
        else:
            print(f"   [HOLD] HOLD signal - no trade action")

        # -------------------------------------------------------------------
        # Write DecisionRecord + realized MissedOpportunityRecord (once per cycle)
        # -------------------------------------------------------------------
        try:
            if decision_pipeline is not None:
                executed_action_for_log = proposed_action_for_log if trade_placed_for_log else "HOLD"
                st, vec, names, schema = decision_pipeline.record_decision(
                    timestamp=sim_time,
                    symbol=trading_symbol,
                    current_price=float(current_price or 0.0),
                    signal=signal or {},
                    proposed_action=str(proposed_action_for_log),
                    executed_action=str(executed_action_for_log),
                    trade_placed=bool(trade_placed_for_log),
                    rejection_reasons=list(rejection_reasons_for_log),
                    paper_rejection=paper_rejection_for_log,
                    bot=bot,
                    write_missed=False,  # we log realized future-based labels below
                )

                # Q-label logging mode (see TT_Q_LABELS at top of file)
                do_label = False
                try:
                    if TT_Q_LABELS == "all":
                        do_label = True
                    elif TT_Q_LABELS == "flat_only":
                        do_label = not bool(getattr(st, "is_in_trade", False))
                    else:  # hold_only
                        do_label = str(executed_action_for_log).upper() == "HOLD"
                except Exception:
                    do_label = str(executed_action_for_log).upper() == "HOLD"

                if do_label:
                    try:
                        q_h = int(os.environ.get("Q_HORIZON_MINUTES", "30"))
                    except Exception:
                        q_h = 30
                    # Compute realized returns for common horizons + the configured Q horizon.
                    horizons = sorted({10, 15, 20, 30, 60, int(q_h)})
                    horizon_returns = {}
                    for h in horizons:
                        fut = _lookup_close(df_primary_for_missed, sim_time + timedelta(minutes=int(h)), tolerance_minutes=5)
                        if fut is None:
                            continue
                        entry = float(current_price or 0.0)
                        if entry > 0:
                            horizon_returns[int(h)] = (float(fut) - entry) / entry
                    decision_pipeline.record_missed_opportunity(
                        state=st,
                        vec=vec,
                        names=names,
                        schema=schema,
                        veto_reasons=list(rejection_reasons_for_log),
                        horizon_returns=horizon_returns,
                        ghost_mode="realized",
                        executed_action=str(executed_action_for_log),
                    )
        except Exception:
            pass
        
        # Increment cycle counter first
        cycles += 1

        # Optional: stop after N cycles (useful for quick dataset smoke tests)
        if TT_MAX_CYCLES > 0 and cycles >= TT_MAX_CYCLES:
            logger.info(f"[STOP] TT_MAX_CYCLES reached ({cycles}); stopping time-travel run.")
            break
        
        # End of cycle summary
        print()
        print(f"[SUMMARY] CYCLE SUMMARY:")
        print(f"   - Completed cycles: {cycles}")
        print(f"   - Signals generated: {signals_generated}")
        print(f"   - Trades made: {trades_made}")
        print(f"   - RL updates: {rl_updates_count}")
        print(f"   - Current balance: ${bot.paper_trader.current_balance:,.2f}")
        print(f"   - Feature buffer: {len(bot.feature_buffer)}/{bot.sequence_length}")

        # Phase 52: Update max drawdown tracking
        current_bal = bot.paper_trader.current_balance
        balance_history.append(current_bal)
        if current_bal > peak_balance_seen:
            peak_balance_seen = current_bal
        if peak_balance_seen > 0:
            current_dd = (peak_balance_seen - current_bal) / peak_balance_seen * 100
            if current_dd > max_drawdown_pct:
                max_drawdown_pct = current_dd

        # Calculate time per cycle
        cycle_end_time = real_time_module.time()
        cycle_duration = cycle_end_time - cycle_start_time
        print(f"   - Cycle duration: {cycle_duration:.2f}s")
        
        # Log cycle summary to file for dashboard (every cycle)
        logger.info(f"   - Completed cycles: {cycles}")
        logger.info(f"   - Signals generated: {signals_generated}")
        logger.info(f"   - Trades made: {trades_made}")
        logger.info(f"   - RL updates: {rl_updates_count}")
        logger.info(f"   - Current balance: ${bot.paper_trader.current_balance:,.2f}")
        
        # Log cycle count every 50 cycles for dashboard updates
        if cycles % 50 == 0:
            current_positions = len(bot.paper_trader.active_trades)
            logger.info(f"[PROGRESS] Progress: Cycles: {cycles} | Signals: {signals_generated} | Trades: {trades_made} | Balance: ${bot.paper_trader.current_balance:,.2f} | RL Updates: {rl_updates_count} | Positions: {current_positions}/2")

        # Compact dashboard update (low-overhead, parseable)
        try:
            _emit_dash(
                cycle=cycles,
                signals=signals_generated,
                trades=trades_made,
                rl_updates=rl_updates_count,
                balance=float(getattr(bot.paper_trader, "current_balance", 0.0) or 0.0),
                spy=float(current_price or 0.0),
                vix=float(vix_price or 0.0),
                qqq=float(qqq_price or 0.0),
                btc=float(btc_price or 0.0),
            )
        except Exception:
            pass
        
        # STABILITY: Small delay to prevent memory crashes on Windows (like output6)
        import time as stability_time
        stability_time.sleep(0.05)  # 50ms pause per cycle
        
        # STABILITY: Periodic garbage collection (every 10 cycles, not every cycle)
        if cycles % 10 == 0:
            import gc
            gc.collect()

        # Update heartbeat for dashboard tracking (every 100 cycles)
        if cycles % 100 == 0 and _heartbeat:
            try:
                current_pnl = bot.paper_trader.current_balance - bot.paper_trader.initial_balance
                _heartbeat.update(cycles, current_pnl)
            except Exception:
                pass

        # Checkpoint logging (models NOT saved to prevent overfitting)
        if cycles % 1000 == 0 and cycles > 0:
            # Note: We DON'T save models during training to prevent overfitting
            # Each run starts fresh to ensure clean evaluation
            logger.info(f"[CHECKPOINT] Checkpoint {cycles}: Balance ${bot.paper_trader.current_balance:,.2f}")

        # Evaluate any matured missed opportunities (report-only)
        _eval_missed(sim_time, df_primary_for_missed)
        _eval_gate(sim_time, df_primary_for_missed)
        
        # Progress updates (log to both console and file for dashboard)
        if (idx + 1) % (len(common_times) // 10) == 0:
            current_balance = bot.paper_trader.current_balance
            elapsed = real_time.time() - start_time
            rate = cycles / elapsed if elapsed > 0 else 0
            eta = (len(common_times) - cycles) / rate if rate > 0 else 0
            
            progress = ((idx + 1) / len(common_times)) * 100
            progress_msg = (f"Progress: {progress:3.0f}% | {sim_time.strftime('%m/%d %H:%M')} | "
                           f"Cycles: {cycles} | Signals: {signals_generated} | Trades: {trades_made} | "
                           f"Balance: ${current_balance:,.0f} | ETA: {eta/60:.1f}m")
            
            # Log to file for dashboard
            logger.info(progress_msg)
            # Also print to console
            print(f"\n  {progress_msg}")
        
    except KeyboardInterrupt:
        print(f"\n\n[!] Training interrupted by user at cycle {cycles}/{len(common_times)}")
        print("[*] Finalizing run (datasets stay on disk).")
        # Break out of the loop cleanly so we still finalize / save summary.
        break
    except Exception as e:
        logger.error(f"Error in cycle {cycles}: {e}")
        import traceback
        traceback.print_exc()
        continue

# Historical data complete - check if we should continue live
elapsed = real_time.time() - start_time
final_balance = bot.paper_trader.current_balance

print(f"\n{'='*70}")
print(f"[OK] HISTORICAL DATA COMPLETE!")
print(f"{'='*70}")
print(f"\n  Processed: {cycles} cycles (market hours only)")
print(f"  Skipped off-market: {skipped_off_hours:,}")
print(f"  Skipped missing data: {skipped_missing_data:,}")
print(f"  Real time: {elapsed/60:.1f} minutes")
print(f"  Speed: {cycles/elapsed:.1f} cycles/second")
print(f"  Signals: {signals_generated}")
print(f"  Trades: {trades_made}")
print(f"  Start balance: ${initial_balance:,.2f}")
print(f"  Final balance: ${final_balance:,.2f}")
print(f"  P&L: ${final_balance - initial_balance:,.2f}")

# Finalize missed-play tracking (evaluate whatever can be evaluated at end time)
try:
    _eval_missed(common_times[-1], df_primary_for_missed)
except Exception:
    pass

if missed_tracking.get("enabled", True):
    evaluated = missed_stats.get("evaluated", 0)
    mw = missed_stats.get("missed_winners", 0)
    al = missed_stats.get("avoided_losers", 0)
    neu = missed_stats.get("neutral", 0)
    print(f"\n  Missed-Play Tracking (report-only, horizon={missed_tracking.get('horizon_minutes', 15)}m):")
    print(f"    Tracked:   {missed_stats.get('tracked', 0)}")
    print(f"    Evaluated: {evaluated}")
    if evaluated > 0:
        print(f"    Missed winners:  {mw} ({mw/max(1,evaluated)*100:.1f}%)")
        print(f"    Avoided losers:  {al} ({al/max(1,evaluated)*100:.1f}%)")
        print(f"    Neutral:         {neu} ({neu/max(1,evaluated)*100:.1f}%)")

# Finalize gate dataset evaluation
try:
    _eval_gate(common_times[-1], df_primary_for_missed, force_all=True)
except Exception:
    pass

# Final diagnostics summary
_log_decision_stats(prefix="[DIAG] FINAL")

if gate_dataset.get("enabled", True):
    gs = gate_dataset.get("stats", {})
    ev = int(gs.get("evaluated", 0))
    pos = int(gs.get("positive", 0))
    neg = int(gs.get("negative", 0))
    print(f"\n  Tradability-Gate Dataset:")
    print(f"    File:      {gate_dataset.get('path')}")
    print(f"    Collected: {int(gs.get('collected', 0))}")
    print(f"    Evaluated: {ev}")
    print(f"    Dropped (no future price): {int(gs.get('dropped_no_future_price', 0))}")
    if ev > 0:
        print(f"    Label positive rate: {pos/ev*100:.1f}% ({pos}/{ev})")

# Experience Replay Statistics
exp_stats = experience_manager.get_stats()
print(f"\n  Experience Replay Buffer:")
print(f"    Total experiences: {exp_stats['size']}")
print(f"    Processed trades: {exp_stats['processed_trades']}")
print(f"    Avg priority: {exp_stats['avg_priority']:.2f}")
print(f"    Max priority: {exp_stats['max_priority']:.2f}")
if 'beta' in exp_stats:
    print(f"    Beta (importance): {exp_stats['beta']:.3f}")

if bot.use_rl_policy:
    print(f"\n  RL Policy:")
    # Unified RL path (preferred)
    try:
        if 'USE_UNIFIED_RL' in globals() and USE_UNIFIED_RL and 'unified_rl' in globals() and unified_rl is not None:
            stats = unified_rl.get_stats()
            print(f"    Type:        UnifiedRLPolicy")
            print(f"    Trades:      {stats.get('total_trades', 0)}")
            print(f"    Win rate:    {stats.get('win_rate', 0) * 100:.1f}%")
            print(f"    RL updates:  {stats.get('total_updates', 0)}")
            print(f"    Buffer size: {stats.get('experience_buffer_size', 0)}")
        elif hasattr(bot, 'rl_policy') and bot.rl_policy is not None:
            print(f"    Type:        RLTradingPolicy")
            print(f"    Experiences: {len(bot.rl_policy.experience_buffer)}")
            print(f"    Updates:     {bot.rl_policy.training_stats.get('total_updates', 0)}")
        else:
            print("    [INFO] RL policy object not attached to bot (OK for unified RL runs)")
    except Exception as e:
        print(f"    [WARN] Could not summarize RL policy: {e}")

# Save models
print(f"\n{'='*70}")
print("SAVING MODELS (TIMESTAMPED + BEST)")
print(f"{'='*70}")

# Save this run with timestamp
import shutil
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
final_pnl = final_balance - initial_balance
pnl_pct = (final_pnl / initial_balance) * 100

# Save final model to the run directory (already created at startup)
logger.info(f"[SAVE] Saving final models to: {run_dir}")
print(f"\n[*] Saving models to: {run_dir}")

# Save models to timestamped directory
try:
    bot.save_model()
    print(f"  [OK] Neural network saved")
    logger.info(f"[SAVE] Neural network saved")
except Exception as e:
    print(f"  [WARN] Could not save neural network: {e}")
    logger.error(f"[ERROR] Failed to save model: {e}")

# Save performance summary for easy comparison
summary_file = os.path.join(run_dir, "SUMMARY.txt")
try:
    with open(summary_file, 'w') as f:
        f.write(f"Training Run Summary\n")
        f.write(f"=" * 60 + "\n\n")
        f.write(f"Timestamp:       {timestamp}\n")
        f.write(f"Run Directory:   {run_dir}\n\n")
        f.write(f"Performance:\n")
        f.write(f"  Initial Balance: ${initial_balance:,.2f}\n")
        f.write(f"  Final Balance:   ${final_balance:,.2f}\n")
        f.write(f"  P&L:             ${final_pnl:+,.2f} ({pnl_pct:+.2f}%)\n")
        f.write(f"  Max Drawdown:    {max_drawdown_pct:.2f}%\n")
        pnl_dd_ratio = pnl_pct / max_drawdown_pct if max_drawdown_pct > 0 else 0
        f.write(f"  P&L/DD Ratio:    {pnl_dd_ratio:.2f}\n\n")
        f.write(f"Trading Stats:\n")
        f.write(f"  Total Trades:    {trades_made}\n")
        f.write(f"  Total Signals:   {signals_generated}\n")
        f.write(f"  Total Cycles:    {cycles}\n")
        f.write(f"  Win Rate:        {(bot.paper_trader.winning_trades / bot.paper_trader.total_trades * 100) if bot.paper_trader.total_trades > 0 else 0:.1f}%\n")
        f.write(f"  Wins:            {bot.paper_trader.winning_trades}\n")
        f.write(f"  Losses:          {bot.paper_trader.losing_trades}\n\n")
        # Missed-play tracking (report-only)
        f.write(f"Missed-Play Tracking (report-only):\n")
        f.write(f"  Horizon:         {missed_tracking.get('horizon_minutes', 15)} minutes\n")
        f.write(f"  Tracked:         {missed_stats.get('tracked', 0)}\n")
        f.write(f"  Evaluated:       {missed_stats.get('evaluated', 0)}\n")
        if missed_stats.get('evaluated', 0) > 0:
            ev = missed_stats.get('evaluated', 0)
            mw = missed_stats.get('missed_winners', 0)
            al = missed_stats.get('avoided_losers', 0)
            neu = missed_stats.get('neutral', 0)
            f.write(f"  Missed winners:  {mw} ({mw/max(1,ev)*100:.1f}%)\n")
            f.write(f"  Avoided losers:  {al} ({al/max(1,ev)*100:.1f}%)\n")
            f.write(f"  Neutral:         {neu} ({neu/max(1,ev)*100:.1f}%)\n")
        f.write("\n")
        # Tradability-gate dataset
        f.write("Tradability-Gate Dataset:\n")
        f.write(f"  File:            {gate_dataset.get('path')}\n")
        f.write(f"  Collected:       {gate_dataset.get('stats', {}).get('collected', 0)}\n")
        f.write(f"  Evaluated:       {gate_dataset.get('stats', {}).get('evaluated', 0)}\n")
        f.write(f"  Positives:       {gate_dataset.get('stats', {}).get('positive', 0)}\n")
        f.write(f"  Negatives:       {gate_dataset.get('stats', {}).get('negative', 0)}\n")
        f.write(f"  Dropped(no fut): {gate_dataset.get('stats', {}).get('dropped_no_future_price', 0)}\n")
        f.write("\n")
        f.write(f"Models Saved:\n")
        f.write(f"  Neural Network:  state/trained_model.pth\n")
        f.write(f"  Optimizer:       state/optimizer_state.pth\n")
        f.write(f"  Feature Buffer:  state/feature_buffer.pkl\n")
    print(f"  [OK] Performance summary saved: SUMMARY.txt")
    logger.info(f"[SAVE] Summary saved to {summary_file}")
except Exception as e:
    logger.error(f"[ERROR] Failed to save summary: {e}")

if rl_threshold_learner:
    try:
        # Get RL stats including counterfactual learning
        rl_stats = rl_threshold_learner.get_stats()
        print(f"\nðŸ“Š RL THRESHOLD LEARNER STATS:")
        print(f"   Total decisions: {rl_stats.get('total_decisions', 0)}")
        print(f"   Trades executed: {rl_stats.get('trades_executed', 0)}")
        print(f"   Win rate: {rl_stats.get('win_rate', 0):.1%}")
        print(f"   Final threshold: {rl_stats.get('composite_threshold', 0):.3f}")
        print(f"   Threshold bounds: [{rl_stats.get('min_threshold', 0.2):.2f}, {rl_stats.get('max_threshold', 0.8):.2f}]")
        print(f"   Avg winner score: {rl_stats.get('avg_winner_score', 0):.3f}")
        print(f"   Avg loser score: {rl_stats.get('avg_loser_score', 0):.3f}")
        print(f"\nðŸ”„ COUNTERFACTUAL LEARNING:")
        print(f"   Missed winners: {rl_stats.get('total_missed_winners', 0)}")
        print(f"   Avoided losers: {rl_stats.get('total_missed_losers', 0)}")
        print(f"   Batches analyzed: {rl_stats.get('counterfactual_batches', 0)}")
        print(f"   Miss rate: {rl_stats.get('counterfactual_miss_rate', 0):.1%} (>50% = too conservative)")
        print(f"   Pending signals: {rl_stats.get('pending_rejected_signals', 0)}")

        # Feature Attribution Stats (Phase 14)
        fa_stats = rl_stats.get('feature_attribution', {})
        if fa_stats.get('ranked_features'):
            print(f"\n[ATTRIBUTION] FEATURE IMPORTANCE (Phase 14):")
            print(f"   Samples: {fa_stats.get('sample_counts', {}).get('total', 0)} total, "
                  f"{fa_stats.get('sample_counts', {}).get('winners', 0)} winners, "
                  f"{fa_stats.get('sample_counts', {}).get('losers', 0)} losers")
            print(f"   Top features (most predictive of winners):")
            for i, (name, importance) in enumerate(fa_stats.get('ranked_features', [])[:5]):
                direction = "+" if importance > 0 else "-"
                print(f"      {i+1}. {name}: {direction}{abs(importance):.4f}")

        rl_threshold_learner.save(f'{run_dir}/rl_threshold.pth')
        print(f"  [OK] RL threshold learner saved")
    except Exception as e_rl:
        print(f"  [WARN] Could not save RL threshold: {e_rl}")

# PNL CALIBRATION STATS (Phase 13/14 - Confidence Calibration)
if pnl_calibration_tracker:
    try:
        pnl_metrics = pnl_calibration_tracker.get_metrics()
        print(f"\n📊 CONFIDENCE CALIBRATION STATS (Phase 14):")
        print(f"   Samples collected: {pnl_metrics.get('pnl_sample_count', 0)}")
        print(f"   Overall win rate: {pnl_metrics.get('pnl_win_rate', 0):.1%}")
        print(f"   Brier score: {pnl_metrics.get('pnl_brier_score', 1.0):.4f} (lower=better, <0.25=good)")
        print(f"   ECE: {pnl_metrics.get('pnl_ece', 1.0):.4f} (lower=better, <0.10=well-calibrated)")
        print(f"   Calibration method: {pnl_metrics.get('pnl_calibration_method', 'none')}")
        print(f"   Is calibrated: {pnl_metrics.get('pnl_is_calibrated', False)}")

        if PNL_CAL_GATE_ENABLED:
            blocked = decision_stats.get('pnl_cal_blocked', 0)
            print(f"   PnL gate blocked: {blocked} trades")

        # Get bucket stats - THE KEY DIAGNOSTIC
        bucket_stats = pnl_calibration_tracker.get_pnl_bucket_stats()
        if bucket_stats:
            print(f"\n   📈 CALIBRATION CURVE (Raw Confidence → Actual Win Rate):")
            print(f"   {'Raw Conf':>10} │ {'Actual WR':>10} │ {'Samples':>8} │ {'Status':>12}")
            print(f"   {'─'*10}─┼─{'─'*10}─┼─{'─'*8}─┼─{'─'*12}")
            for bucket, stats in sorted(bucket_stats.items()):
                raw_conf = float(bucket.replace('_', '.'))
                actual_wr = stats['actual_win_rate']
                samples = stats['samples']
                # Check if calibrated (actual should be close to raw if well-calibrated)
                if samples >= 10:
                    diff = abs(actual_wr - raw_conf)
                    status = "✓ Good" if diff < 0.15 else "⚠ Needs fix" if diff < 0.25 else "✗ Broken"
                else:
                    status = "⏳ Learning"
                print(f"   {raw_conf:>10.0%} │ {actual_wr:>10.0%} │ {samples:>8} │ {status:>12}")

        # Save calibration state for persistence
        try:
            import pickle
            cal_save_path = f'{run_dir}/pnl_calibration.pkl'
            with open(cal_save_path, 'wb') as f:
                pickle.dump({
                    'metrics': pnl_metrics,
                    'bucket_stats': bucket_stats,
                }, f)
            print(f"   [OK] Calibration state saved to {cal_save_path}")
        except Exception as e_save:
            print(f"   [WARN] Could not save calibration: {e_save}")

    except Exception as e_pnl:
        print(f"  [WARN] Could not get PnL calibration stats: {e_pnl}")

# Save UNIFIED RL Policy (preferred system)
if USE_UNIFIED_RL and unified_rl is not None:
    try:
        unified_stats = unified_rl.get_stats()
        print(f"\nðŸŽ¯ UNIFIED RL POLICY:")
        print(f"   Total trades: {unified_stats.get('total_trades', 0)}")
        print(f"   Win rate: {unified_stats.get('win_rate', 0):.1%}")
        print(f"   Mode: {unified_stats.get('mode', 'unknown')}")
        print(f"   Total P&L: ${unified_stats.get('total_pnl', 0):+.2f}")
        print(f"   Experience buffer: {unified_stats.get('experience_buffer_size', 0)}")
        
        unified_rl.save(f'{run_dir}/unified_rl.pth')
        print(f"  [OK] Unified RL policy saved")
        
        # Also save adaptive rules policy stats
        if rules_policy:
            rules_stats = rules_policy.get_params()
            print(f"\nðŸ“‹ ADAPTIVE RULES POLICY:")
            print(f"   Win rate: {rules_stats.get('win_rate', 0):.1%}")
            print(f"   Min confidence: {rules_stats.get('min_confidence', 0):.1%}")
            print(f"   Profit target: {rules_stats.get('profit_target_high_conf', 0):.1%}")
            print(f"   Stop loss: {rules_stats.get('stop_loss_tight', 0):.1%}")
    except Exception as e:
        print(f"  [WARN] Could not save unified RL: {e}")

if rl_exit_policy:
    try:
        rl_exit_policy.save(f'{run_dir}/rl_exit.pth')
        print(f"  [OK] RL exit policy saved")
    except Exception as e:
        print(f"  [WARN] Could not save RL exit: {e}")

# Save run metadata
metadata = {
    'timestamp': timestamp,
    'start_date': str(common_times[0]),
    'end_date': str(common_times[-1]),
    'initial_balance': float(initial_balance),
    'final_balance': float(final_balance),
    'pnl': float(final_pnl),
    'pnl_pct': float(pnl_pct),
    'cycles': cycles,
    'signals': signals_generated,
    'trades': trades_made,
    'experience_buffer_size': len(experience_manager),
    'training_time_seconds': elapsed
}

import json
with open(f'{run_dir}/run_info.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"  [OK] Run metadata saved")
print(f"  [DIR] Location: {run_dir}")
logger.info(f"[SAVE] Run saved: {run_dir}")

# 2. Update best model if this run beat previous best
try:
    best_file = 'models/best_run.json'
    if os.path.exists(best_file):
        with open(best_file, 'r') as f:
            best = json.load(f)
            best_pnl = best.get('pnl', -999999)
    else:
        best_pnl = -999999  # First run
    
    if final_pnl > best_pnl:
        print(f"\n[BEST] NEW BEST MODEL!")
        print(f"   Previous best: ${best_pnl:+.2f}")
        print(f"   This run: ${final_pnl:+.2f}")
        print(f"   Improvement: ${final_pnl - best_pnl:+.2f}")
        logger.info(f"[BEST] NEW BEST! {final_pnl:+.2f} > {best_pnl:+.2f}")
        
        # Copy to best directory
        best_dir = 'models/best'
        os.makedirs(best_dir, exist_ok=True)
        
        if os.path.exists(f'{run_dir}/neural_network_full.pth'):
            shutil.copy(f'{run_dir}/neural_network_full.pth', f'{best_dir}/neural_network_full.pth')
            print(f"   [OK] Best neural network updated")
        
        if rl_threshold_learner and os.path.exists(f'{run_dir}/rl_threshold.pth'):
            shutil.copy(f'{run_dir}/rl_threshold.pth', f'{best_dir}/rl_threshold.pth')
            print(f"   [OK] Best RL threshold updated")
        
        if rl_exit_policy and os.path.exists(f'{run_dir}/rl_exit.pth'):
            shutil.copy(f'{run_dir}/rl_exit.pth', f'{best_dir}/rl_exit.pth')
            print(f"   [OK] Best RL exit policy updated")
        
        # Update best metadata
        with open(best_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"   [DIR] Best model location: {best_dir}")
        logger.info(f"[BEST] Best model updated: {best_dir}")
    else:
        print(f"\n[STATS] Not best model")
        print(f"   This run: ${final_pnl:+.2f}")
        print(f"   Best: ${best_pnl:+.2f}")
        print(f"   Difference: ${final_pnl - best_pnl:+.2f}")
        logger.info(f"Not best. Current: {final_pnl:+.2f}, Best: {best_pnl:+.2f}")

except Exception as e:
    print(f"  [WARN] Could not update best model: {e}")
    logger.warning(f"Could not update best model: {e}")

print(f"""
[OK] Historical training complete!
[OK] Bot processed {cycles} cycles (market hours only, with required data)

Data Quality:
  Total timestamps available: {len(common_times):,}
  Skipped (off-market hours): {skipped_off_hours:,}
  Skipped (missing required data): {skipped_missing_data:,}
  Actually trained on: {cycles:,} cycles ({cycles/max(len(common_times),1)*100:.1f}%)

Performance on historical data:
  Trades: {trades_made}
  Win rate: {(bot.paper_trader.winning_trades / max(bot.paper_trader.total_trades, 1)) * 100:.1f}%
  Final balance: ${final_balance:,.2f}
  P&L: ${final_balance - initial_balance:,.2f}

""")

# === CONTINUOUS LIVE MODE ===
# Check if we should continue with live data
if os.environ.get('ENABLE_LIVE_MODE', 'false').lower() == 'true':
    print(f"\n{'='*70}")
    print("[LIVE] SWITCHING TO CONTINUOUS LIVE MODE")
    print(f"{'='*70}")
    print("""
[LIVE] Historical data exhausted - continuing with LIVE data!
[LIVE] Will fetch new 1-minute bars as they complete
[LIVE] Training continues indefinitely (Ctrl+C to stop)
""")
    
    try:
        from continuous_data_fetcher import ContinuousDataFetcher  # type: ignore[reportMissingImports]
        
        fetcher = ContinuousDataFetcher(db_path='data/db/historical.db')
        
        def process_new_bar(new_bars):
            """Process a newly fetched bar through the trading cycle"""
            global cycles, signals_generated, trades_made, rl_updates_count
            
            # Get primary symbol bar
            if trading_symbol not in new_bars:
                logger.warning(f"[LIVE] No {trading_symbol} data in new bars")
                return True  # Continue anyway
            
            bitx_bar = new_bars[trading_symbol]
            sim_time = bitx_bar['timestamp']
            current_price = bitx_bar['close']
            
            # Set time on bot (CRITICAL: set simulation_mode=True for get_market_time() to work!)
            tt_source.set_time(sim_time)
            bot.simulated_time = sim_time
            bot.paper_trader.simulated_time = sim_time
            bot.paper_trader.simulation_mode = True  # Required for get_market_time() to return simulated_time
            bot.current_time = sim_time

            # Set simulation time for event calendar (Jerry's event halts feature)
            try:
                from backend.event_calendar import set_simulation_time
                set_simulation_time(sim_time)
            except ImportError:
                pass

            # Log for tracking
            logger.info(f"[LIVE] Processing new bar: {sim_time} | Price: ${current_price:.2f}")
            
            # Run complete trading cycle (same as historical)
            try:
                # STEP 1: Update positions
                bot.paper_trader.update_positions(trading_symbol, current_price=current_price)
                
                # STEP 2: Mini-batch learning (every 3 cycles)
                if cycles % 3 == 0 and len(experience_manager) >= 16:
                    experiences, indices, weights = experience_manager.sample_batch(batch_size=16)
                    if len(experiences) > 0:
                        # Learning logic here (simplified for live)
                        rl_updates_count += 1
                
                # STEP 3: Full retraining (every 10 cycles for better accuracy)
                if cycles % 10 == 0 and cycles > 0:
                    bot.continuous_learning_cycle()
                    rl_updates_count += 1
                
                # STEP 4: Generate signal
                signal = bot.generate_unified_signal(trading_symbol)
                
                # STEP 5: Execute trades
                if signal and signal.get('action') != 'HOLD':
                    signals_generated += 1
                    
                    if rl_threshold_learner:
                        from datetime import datetime
                        
                        # Extract HMM state for RL context
                        hmm_trend = 1  # Default: neutral
                        hmm_vol = 1   # Default: normal  
                        hmm_liq = 1   # Default: normal
                        
                        if bot.current_hmm_regime:
                            trend_map = {'Down': 0, 'Downtrend': 0, 'No Trend': 1, 'Neutral': 1, 'Up': 2, 'Uptrend': 2}
                            vol_map = {'Low': 0, 'Normal': 1, 'High': 2}
                            liq_map = {'Low': 0, 'Normal': 1, 'High': 2}
                            
                            hmm_trend = trend_map.get(bot.current_hmm_regime.get('trend', 'Neutral'), 1)
                            hmm_vol = vol_map.get(bot.current_hmm_regime.get('volatility', 'Normal'), 1)
                            hmm_liq = liq_map.get(bot.current_hmm_regime.get('liquidity', 'Normal'), 1)
                        
                        # Calculate time of day
                        now = datetime.now()
                        market_hour = now.hour + now.minute / 60.0
                        time_of_day = max(0.0, min(1.0, (market_hour - 9.5) / 6.5))
                        
                        # Get VIX features (use defaults for live mode if not available)
                        vix_bb_pos = 0.5
                        vix_roc = 0.0
                        vix_percentile = 0.5
                        price_jerk = 0.0
                        
                        # Get win rate and drawdown
                        recent_win_rate = 0.5
                        current_drawdown = 0.0
                        if bot.paper_trader.total_trades > 0:
                            recent_win_rate = bot.paper_trader.winning_trades / max(1, bot.paper_trader.total_trades)
                        peak_balance = max(bot.paper_trader.initial_balance,
                                          bot.paper_trader.initial_balance + bot.paper_trader.total_profit_loss)
                        current_drawdown = max(0, (peak_balance - bot.paper_trader.current_balance) / peak_balance) if peak_balance > 0 else 0
                        
                        should_trade, composite_score, rl_details = rl_threshold_learner.evaluate_signal(
                            confidence=signal.get('confidence', 0),
                            predicted_return=abs(signal.get('predicted_return', 0)),
                            momentum=abs(signal.get('momentum_5min', 0)),
                            volume_spike=signal.get('volume_spike', 1.0),
                            vix_bb_pos=vix_bb_pos,
                            vix_roc=vix_roc,
                            vix_percentile=vix_percentile,
                            hmm_trend=hmm_trend,
                            hmm_vol=hmm_vol,
                            hmm_liq=hmm_liq,
                            time_of_day=time_of_day,
                            recent_win_rate=recent_win_rate,
                            drawdown=current_drawdown,
                            price_jerk=price_jerk,
                            current_price=current_price,
                            timestamp=datetime.now(),
                            proposed_action=signal.get('action')  # For direction-aware counterfactual
                        )
                        
                        # Analyze missed opportunities every 10 cycles
                        if cycles % 10 == 0:
                            missed_stats = rl_threshold_learner.analyze_missed_opportunities(
                                current_price=current_price,
                                current_timestamp=datetime.now(),
                                lookback_minutes=30
                            )
                            if missed_stats.get('total_analyzed', 0) > 0:
                                logger.info(f"[COUNTERFACTUAL] {missed_stats['missed_winners']} missed winners, "
                                          f"{missed_stats['missed_losers']} avoided losers")
                        
                        if should_trade:
                            trade = bot.paper_trader.place_trade(trading_symbol, signal, current_price)
                            if trade:
                                trades_made += 1
                                logger.info(f"[LIVE] Trade placed: {signal.get('action')} @ ${current_price:.2f}")
                
                cycles += 1
                
                # Progress update every 10 cycles
                if cycles % 10 == 0:
                    current_balance = bot.paper_trader.current_balance
                    logger.info(f"[LIVE] Cycle {cycles} | Balance: ${current_balance:,.2f} | Trades: {trades_made}")
                
            except Exception as e:
                logger.error(f"[LIVE] Error in trading cycle: {e}")
                import traceback
                traceback.print_exc()
            
            return True  # Continue fetching
        
        # Start continuous fetching with callback
        logger.info("[LIVE] Starting continuous data fetcher...")
        # Get symbols from config or use defaults
        symbols_to_fetch = [trading_symbol]
        try:
            import json
            with open('config.json', 'r') as f:
                config = json.load(f)
            symbols_to_fetch = config.get('data_fetching', {}).get('symbols', [trading_symbol, 'VIX', 'SPY', 'QQQ', 'UUP'])
        except:
            symbols_to_fetch = [trading_symbol, 'VIX', 'SPY', 'QQQ', 'UUP']
        fetcher.run_continuous(symbols=symbols_to_fetch, callback=process_new_bar)
        
    except KeyboardInterrupt:
        print(f"\n\n[LIVE] Live mode interrupted by user")
    except Exception as e:
        logger.error(f"[LIVE] Error in live mode: {e}")
        import traceback
        traceback.print_exc()
else:
    # Stop heartbeat tracking
    if _heartbeat:
        try:
            _heartbeat.stop()
            logger.info("[HEARTBEAT] Experiment unregistered (complete)")
        except Exception:
            pass

    print(f"\n{'='*70}")
    print("[DONE] SIMULATION COMPLETE")
    print(f"{'='*70}")
    print(f"""
Simulation finished! Results saved to: {run_dir}

Next steps:
  - Review results: View logs in logs/ directory
  - Run again: python run_simulation.py
  - Enable live mode: ENABLE_LIVE_MODE=true python run_simulation.py
  - Go live: python start_everything.py (when ready for live trading)
""")


