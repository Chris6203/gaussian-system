#!/usr/bin/env python3
"""
Go Live Only - Standalone Configurable Bot

Usage:
    python go_live_only.py <model_directory>
    python go_live_only.py models/run_20251104_225752

Configuration:
    Edit config.json to change trading symbol and parameters
"""

import sys
import os
import time
import logging
from logging.handlers import TimedRotatingFileHandler
from datetime import datetime
from pathlib import Path

# Import market time utility for consistent Eastern time
from backend.time_utils import get_market_time

# Load configuration
from config_loader import load_config
config = load_config("config.json")

# Get symbol from config
SYMBOL = config.get_symbol()
print(f"[CONFIG] Trading Symbol: {SYMBOL}")
print(f"[CONFIG] Initial Balance (live): ${config.get_initial_balance():,.2f}")
print(f"[CONFIG] Paper Balance (simulated): ${config.get_paper_initial_balance():,.2f}")
print(f"[CONFIG] Data Source: {config.get_data_source()}")

# ============================================================================
# PAPER ONLY MODE - Dynamic switching via flag file
# Create 'go_live.flag' file to switch to live trading without restart
# Delete 'go_live.flag' to switch back to paper mode
# ============================================================================
PAPER_ONLY_MODE = True  # Start in paper mode by default
GO_LIVE_FLAG_FILE = Path("go_live.flag")

def check_live_mode():
    """Check if we should be in live mode (flag file exists)."""
    return GO_LIVE_FLAG_FILE.exists()

def get_current_mode():
    """Get current trading mode based on flag file."""
    if check_live_mode():
        return False  # Not paper only = live mode
    return True  # Paper only mode

print(f"[CONFIG] Initial Mode: {'PAPER' if PAPER_ONLY_MODE else 'LIVE'}")
print(f"[CONFIG] Dynamic switching enabled:")
print(f"[CONFIG]   - Create 'go_live.flag' to switch to LIVE trading")
print(f"[CONFIG]   - Delete 'go_live.flag' to switch back to PAPER mode")
if GO_LIVE_FLAG_FILE.exists():
    print(f"[CONFIG] ** go_live.flag EXISTS - Will trade LIVE **")
    PAPER_ONLY_MODE = False
else:
    print(f"[CONFIG] ** No go_live.flag - Paper trading only **")
print()

if len(sys.argv) < 2:
    print("Usage: python go_live_only.py <model_directory>")
    print(f"Example: python go_live_only.py models/run_20251104_225752")
    sys.exit(1)

model_dir = Path(sys.argv[1])
if not model_dir.exists():
    print(f"Error: Model directory not found: {model_dir}")
    sys.exit(1)

print("="*80)
print("LIVE SHADOW TRADING ONLY")
print("="*80)
print(f"Using models from: {model_dir}")
print(f"Trading: {SYMBOL}")
print()

# Set up daily rotating log files
os.makedirs('logs', exist_ok=True)
log_file = f"logs/trading_bot_{datetime.now().strftime('%Y%m%d')}.log"
print(f"[*] Log file: {log_file} (rotates daily, keeps 30 days)")

# Create rotating file handler - new file each day, keep 30 days
file_handler = TimedRotatingFileHandler(
    filename='logs/trading_bot.log',
    when='midnight',           # Rotate at midnight
    interval=1,                # Every 1 day
    backupCount=30,            # Keep 30 days of logs
    encoding='utf-8'
)
file_handler.suffix = '%Y%m%d.log'  # trading_bot.log.20251125.log
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

# Also log to console
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

# Configure root logger
logging.basicConfig(
    level=getattr(logging, config.get('logging', 'level', default='INFO')),
    handlers=[file_handler, console_handler],
    force=True
)
logger = logging.getLogger(__name__)

logger.info("="*70)
logger.info(f"LIVE SHADOW TRADING - Symbol: {SYMBOL}")
logger.info("="*70)

# Initialize bot
from unified_options_trading_bot import UnifiedOptionsBot
import shutil

# Copy models to models/state/ so bot can load them
trained_state_dir = model_dir / "state"
target_state_dir = Path("models/state")

if trained_state_dir.exists():
    logger.info(f"[LIVE] Loading models from {trained_state_dir}")
    target_state_dir.mkdir(parents=True, exist_ok=True)
    
    for file_path in trained_state_dir.glob("*"):
        shutil.copy2(file_path, target_state_dir / file_path.name)
        logger.info(f"   [OK] Loaded {file_path.name}")
else:
    logger.error(f"[ERROR] No models found at {trained_state_dir}")
    sys.exit(1)

logger.info("[LIVE] Syncing balance from Tradier...")
# Get actual Tradier balance BEFORE initializing bot
try:
    from tradier_credentials import TradierCredentials
    from backend.tradier_trading_system import TradierTradingSystem
    from trading_mode import get_trading_mode
    
    mode = get_trading_mode()
    creds = TradierCredentials()
    
    if mode['is_sandbox']:
        tradier_creds = creds.get_sandbox_credentials()
    else:
        tradier_creds = creds.get_live_credentials()
    
    # Create temporary Tradier system just to get balance
    temp_tradier = TradierTradingSystem(
        account_id=tradier_creds['account_number'],
        api_token=tradier_creds['access_token'],
        sandbox=mode['is_sandbox'],
        initial_balance=0  # Don't care, just need to get balance
    )
    
    # Get actual account balance from Tradier
    import requests
    base_url = f"https://{'sandbox' if mode['is_sandbox'] else 'api'}.tradier.com/v1"
    url = f"{base_url}/accounts/{tradier_creds['account_number']}/balances"
    logger.info(f"[SYNC] Mode: {mode['mode']} | Sandbox: {mode['is_sandbox']}")
    logger.info(f"[SYNC] Fetching balance from: {url}")
    response = requests.get(
        url,
        headers={
            'Authorization': f'Bearer {tradier_creds["access_token"]}',
            'Accept': 'application/json'
        }
    )
    
    if response.status_code == 200:
        data = response.json()
        balances = data.get('balances', {})
        actual_balance = float(balances.get('total_equity', config.get_initial_balance()))
        logger.info(f"✅ [SYNC] Actual Tradier balance: ${actual_balance:,.2f}")
    else:
        actual_balance = config.get_initial_balance()
        logger.warning(f"⚠️  [SYNC] Could not fetch Tradier balance, using config: ${actual_balance:,.2f}")
        
except Exception as e:
    actual_balance = config.get_initial_balance()
    logger.warning(f"⚠️  [SYNC] Error syncing balance: {e}, using config: ${actual_balance:,.2f}")

# Use PAPER balance for bot's paper trader (tracks hypothetical performance of ALL signals)
# The actual Tradier balance is used only for the shadow_trader (actual live trades)
paper_balance = config.get_paper_initial_balance()
logger.info(f"[LIVE] Initializing bot with PAPER balance: ${paper_balance:,.2f}")
logger.info(f"[LIVE] Tradier actual balance: ${actual_balance:,.2f} (used for shadow trader)")
bot = UnifiedOptionsBot(
    initial_balance=paper_balance,  # Paper trader uses separate simulated balance
    config=config  # Pass config to bot
)

# Disable simulation mode for live trading
bot.simulation_mode = False
if hasattr(bot, 'disable_live_data_fetch'):
    bot.disable_live_data_fetch = False
bot.paper_trader.simulation_mode = False
bot.paper_trader.time_travel_mode = False
logger.info("[LIVE] Live mode enabled")

# Verify paper trading balance (separate from live Tradier balance)
paper_balance_actual = bot.paper_trader.current_balance
logger.info(f"[LIVE] Paper trading balance: ${paper_balance_actual:,.2f}")
logger.info(f"[LIVE] This is INDEPENDENT from Tradier balance (${actual_balance:,.2f})")

# Configure settlement simulation from config
simulate_settlement = config.get('trading', 'simulate_settlement', default=True)
bot.paper_trader.simulate_settlement = simulate_settlement
if simulate_settlement:
    logger.info(f"[LIVE] Settlement simulation ENABLED (T+1 for options - funds settle next business day)")
else:
    logger.info(f"[LIVE] Settlement simulation DISABLED (instant settlement)")

# Enable liquidity checks if configured
if config.get('liquidity', 'enable_liquidity_checks', default=True):
    try:
        from trading_mode import get_trading_mode
        mode = get_trading_mode()
        
        if not mode['is_sandbox']:
            try:
                from backend.liquidity_validator import LiquidityValidator
                bot.paper_trader.liquidity_validator = LiquidityValidator(
                    max_bid_ask_spread_pct=config.get('liquidity', 'max_bid_ask_spread_pct', default=5.0),
                    min_volume=config.get('liquidity', 'min_volume', default=100),
                    min_open_interest=config.get('liquidity', 'min_open_interest', default=75),
                    max_strike_deviation_pct=config.get('liquidity', 'max_strike_deviation_pct', default=15.0)
                )
                bot.paper_trader.enable_liquidity_checks = True
                logger.info("[LIVE] Paper trader liquidity validation ENABLED")
            except ImportError:
                logger.warning("[LIVE] WARNING: Liquidity validator not available")
    except Exception as e:
        logger.warning(f"[LIVE] Could not configure liquidity checks: {e}")

# Initialize shadow trader if enabled
shadow_trader = None
if config.is_shadow_trading_enabled():
    try:
        from tradier_credentials import TradierCredentials
        from backend.tradier_trading_system import TradierTradingSystem
        from trading_mode import get_trading_mode, print_trading_mode_banner
        
        print_trading_mode_banner()
        
        mode_name = config.get_shadow_mode()
        creds = TradierCredentials()
        
        if mode_name == 'sandbox':
            if creds.has_sandbox_credentials():
                sandbox_creds = creds.get_sandbox_credentials()
                shadow_trader = TradierTradingSystem(
                    account_id=sandbox_creds['account_number'],
                    api_token=sandbox_creds['access_token'],
                    sandbox=True,
                    initial_balance=actual_balance
                )
                bot.shadow_trader = shadow_trader
                logger.info(f"[SANDBOX] Shadow trading enabled: {sandbox_creds['account_number']}")
                
                # Enable liquidity learning from paper trades
                if hasattr(bot, 'paper_trader') and bot.paper_trader:
                    # Connect shadow trader to paper trader for live position management
                    bot.paper_trader.tradier_trading_system = shadow_trader
                    logger.info("🔗 Paper trader connected to shadow trader for live position management")
                    
                    bot.paper_trader.set_bot_reference(bot)
                    logger.info("[LEARN] Liquidity learning enabled for paper trades")
                    
                    # Recover stale positions on startup
                    logger.info("[RECOVERY] Checking for stale positions...")
                    bot.paper_trader.recover_stale_positions(max_age_hours=48)
            else:
                logger.warning("[SANDBOX] No credentials")
        else:  # live mode
            if creds.has_live_credentials():
                live_creds = creds.get_live_credentials()
                shadow_trader = TradierTradingSystem(
                    account_id=live_creds['account_number'],
                    api_token=live_creds['access_token'],
                    sandbox=False,
                    initial_balance=actual_balance
                )
                bot.shadow_trader = shadow_trader
                logger.warning(f"[LIVE] REAL MONEY - LIVE trading enabled: {live_creds['account_number']}")
                
                # Enable liquidity learning from paper trades
                if hasattr(bot, 'paper_trader') and bot.paper_trader:
                    # Connect shadow trader to paper trader for live position management
                    bot.paper_trader.tradier_trading_system = shadow_trader
                    logger.info("🔗 Paper trader connected to shadow trader for live position management")
                    
                    bot.paper_trader.set_bot_reference(bot)
                    logger.info("[LEARN] Liquidity learning enabled for paper trades")
                    
                    # Recover stale positions on startup
                    logger.info("[RECOVERY] Checking for stale positions...")
                    bot.paper_trader.recover_stale_positions(max_age_hours=48)
            else:
                logger.error("[LIVE] No credentials")
                sys.exit(1)
    except Exception as e:
        logger.warning(f"[SHADOW] Could not initialize: {e}")

# Rest of go_live_only.py logic...
# (Position recovery, live trading loop, etc.)

print()
print("[*] Starting live trading loop...")
print(f"[*] Trading: {SYMBOL}")
print(f"[*] Cycle interval: {config.get('system', 'cycle_interval_seconds', default=60)}s")
print("[*] Press Ctrl+C to stop")
print()

cycle = 0
signals_generated = 0  # Track non-HOLD signals
cycle_interval = config.get('system', 'cycle_interval_seconds', default=60)
last_sync_time = 0  # Track last Tradier position sync
SYNC_INTERVAL = 300  # Sync every 5 minutes (300 seconds)

try:
    while True:
        cycle += 1
        cycle_start = time.time()
        
        logger.info("="*70)
        logger.info(f"[CYCLE {cycle}] | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("="*70)
        print(f"\n[CYCLE {cycle}] {datetime.now().strftime('%H:%M:%S')}")
        
        # Periodic Tradier position sync (prevent desync)
        if time.time() - last_sync_time > SYNC_INTERVAL:
            if hasattr(bot, 'paper_trader') and bot.paper_trader:
                try:
                    logger.info("[SYNC] Running periodic Tradier position sync...")
                    bot.paper_trader.sync_tradier_positions()
                    last_sync_time = time.time()
                except Exception as sync_err:
                    logger.error(f"[SYNC] Error syncing positions: {sync_err}")
        
        # Periodic model saving (every 10 cycles) - DON'T LOSE LEARNING!
        if cycle % 10 == 0 and cycle > 0:
            try:
                import os
                os.makedirs('models', exist_ok=True)
                os.makedirs('models/state', exist_ok=True)
                
                saved_models = []
                
                # Save Exit Policy (Thompson Sampling + Neural)
                if hasattr(bot, 'paper_trader') and bot.paper_trader:
                    if hasattr(bot.paper_trader, 'rl_exit_policy') and bot.paper_trader.rl_exit_policy:
                        bot.paper_trader.rl_exit_policy.save('models/rl_exit_policy_live.pth')
                        stats = bot.paper_trader.rl_exit_policy.get_stats()
                        saved_models.append(f"ExitPolicy({stats.get('total_exits', 0)} exits)")
                
                # Save Threshold Policy
                if hasattr(bot, 'rl_threshold_learner') and bot.rl_threshold_learner:
                    bot.rl_threshold_learner.save('models/rl_threshold_live.pth')
                    saved_models.append("ThresholdPolicy")
                
                # Save HMM Regime Detector
                if hasattr(bot, 'hmm_regime_detector') and bot.hmm_regime_detector:
                    if hasattr(bot.hmm_regime_detector, 'save'):
                        bot.hmm_regime_detector.save('models/multi_dimensional_hmm.pkl')
                        saved_models.append("HMM")
                
                # Save Adaptive Timeframe Weights
                if hasattr(bot, 'adaptive_weights') and bot.adaptive_weights:
                    if hasattr(bot.adaptive_weights, 'save'):
                        bot.adaptive_weights.save('models/adaptive_timeframe_weights.json')
                        saved_models.append("AdaptiveWeights")
                
                # Save Main Neural Network (Bayesian + LSTM)
                if hasattr(bot, 'save_model') and cycle % 50 == 0:  # Full save every 50 cycles
                    bot.save_model()
                    saved_models.append("NeuralNetwork")
                
                if saved_models:
                    logger.info(f"[SAVE] Periodic save: {', '.join(saved_models)}")
                    
            except Exception as save_err:
                logger.warning(f"[SAVE] Could not save models: {save_err}")
        
        try:
            # Fetch data for configured symbol
            logger.info(f"[LIVE] Fetching data for {SYMBOL}...")
            data = bot.data_source.get_data(
                SYMBOL,
                period=config.get('data_sources', 'historical_period', default='1d'),
                interval=config.get('data_sources', 'interval', default='1m')
            )
            
            if data.empty:
                logger.warning("[LIVE] No data received")
            else:
                data.columns = [col.lower() for col in data.columns]
                current_price = float(data['close'].iloc[-1])
                
                # Log in format dashboard expects: [PRICE] SPY: $X
                logger.info(f"[PRICE] {SYMBOL}: ${current_price:.2f}")
                print(f"  [DATA] {SYMBOL}: ${current_price:.2f}")
                
                # Save latest prices to historical database for dashboard chart
                try:
                    import sqlite3
                    historical_db = Path('data/db/historical.db')
                    if historical_db.exists():
                        conn = sqlite3.connect(historical_db)
                        cursor = conn.cursor()
                        # Save last 5 rows for dashboard
                        for idx, row in data.tail(5).iterrows():
                            timestamp = idx.isoformat() if hasattr(idx, 'isoformat') else str(idx).replace(' ', 'T')
                            cursor.execute('''
                                INSERT OR REPLACE INTO historical_data 
                                (symbol, timestamp, open_price, high_price, low_price, close_price, volume)
                                VALUES (?, ?, ?, ?, ?, ?, ?)
                            ''', (SYMBOL, timestamp,
                                  float(row.get('open', 0)), float(row.get('high', 0)),
                                  float(row.get('low', 0)), float(row.get('close', 0)),
                                  int(row.get('volume', 0))))
                        conn.commit()
                        conn.close()
                except Exception as db_err:
                    logger.debug(f"[DB] Could not save prices: {db_err}")
                
                # ===== LOG VIX VALUE FOR DASHBOARD =====
                try:
                    vix_data = bot.data_source.get_data('^VIX', period='1d', interval='1m')
                    if not vix_data.empty:
                        vix_data.columns = [col.lower() for col in vix_data.columns]
                        vix_value = float(vix_data['close'].iloc[-1])
                        logger.info(f"[MARKET] VIX: ${vix_value:.2f}")
                        print(f"  [DATA] VIX: ${vix_value:.2f}")
                except Exception as vix_err:
                    logger.debug(f"[VIX] Could not fetch VIX: {vix_err}")
                
                # ===== FETCH ALL CONFIGURED SYMBOLS =====
                # Fetch every cycle - data auto-persists to historical DB via DataManager
                # AND updates bot's market_context for cross-asset feature calculation
                fetch_symbols = config.get('data_fetching', 'symbols', default=['BTC-USD', '^VIX', 'SPY', 'QQQ', 'UUP'])
                
                for fetch_symbol in fetch_symbols:
                    # Skip primary trading symbol and VIX (already fetched above)
                    if fetch_symbol == SYMBOL or fetch_symbol in ['^VIX', 'VIX']:
                        continue
                    
                    try:
                        sym_data = bot.data_source.get_data(fetch_symbol, period='1d', interval='1m')
                        if sym_data is not None and not sym_data.empty:
                            # Store in bot's market context for cross-asset features
                            if hasattr(bot, 'update_market_context'):
                                bot.update_market_context(fetch_symbol, sym_data)
                            
                            sym_data.columns = [col.lower() for col in sym_data.columns]
                            sym_price = float(sym_data['close'].iloc[-1])
                            # Clean symbol for display (remove ^ prefix)
                            display_sym = fetch_symbol.replace('^', '')
                            logger.info(f"[MARKET] {display_sym} Price: ${sym_price:.2f}")
                    except Exception as sym_err:
                        logger.debug(f"[MARKET] Could not fetch {fetch_symbol}: {sym_err}")
                
                # ===== LOG CROSS-ASSET FEATURES =====
                # These features show market correlations and are used for signal validation
                try:
                    if hasattr(bot, 'get_cross_asset_features') and data is not None:
                        cross_features = bot.get_cross_asset_features(data)
                        if cross_features.get('context_confidence', 0) > 0:
                            logger.info(f"[CROSS-ASSET] SPY/QQQ corr: {cross_features['spy_qqq_corr']:.2f}, "
                                       f"QQQ rel: {cross_features['spy_qqq_rel_strength']:+.2f}, "
                                       f"USD: {cross_features['dollar_strength']:+.2f}, "
                                       f"Risk: {'ON' if cross_features['risk_on_off'] > 0 else 'OFF'}")
                except Exception as cross_err:
                    logger.debug(f"[CROSS-ASSET] Could not calculate: {cross_err}")
                
                # ===== LOG PROGRESS & TRAINING METRICS FOR DASHBOARD =====
                try:
                    if hasattr(bot, 'paper_trader') and bot.paper_trader:
                        pt = bot.paper_trader
                        total_trades = getattr(pt, 'total_trades', 0)
                        active_positions = len(getattr(pt, 'active_trades', []))
                        max_positions = getattr(pt, 'max_positions', 2)
                        wins = getattr(pt, 'winning_trades', 0)
                        losses = getattr(pt, 'losing_trades', 0)
                        balance = getattr(pt, 'current_balance', 0)
                        
                        # Log progress line for dashboard (Balance = Paper trading, tracks ALL signals)
                        logger.info(f"Progress: Cycles: {cycle}, Signals: {signals_generated}, Trades: {total_trades}, Balance: ${balance:.2f}")
                        logger.info(f"Active Positions: {active_positions}/{max_positions}")
                        
                        # Show pending settlements if any
                        pending_settlement = getattr(pt, 'pending_settlement_amount', 0)
                        if pending_settlement > 0:
                            logger.info(f"[SETTLEMENT] Pending: ${pending_settlement:.2f} | Available: ${balance:.2f}")
                        
                        logger.debug(f"[INFO] Paper balance (all signals): ${balance:.2f} | Tradier (live only): ${actual_balance:.2f}")
                        
                        # RL updates
                        rl_updates = 0
                        if hasattr(bot, 'rl_policy') and bot.rl_policy:
                            rl_updates = getattr(bot.rl_policy, 'update_count', 0)
                        if hasattr(bot, 'rl_exit_policy') and bot.rl_exit_policy:
                            rl_updates += getattr(bot.rl_exit_policy, 'update_count', 0)
                        logger.info(f"Learning: {rl_updates}")
                        
                        # RL Exit Policy Stats (for dashboard)
                        exit_exits = 0
                        exit_online = 0
                        if hasattr(bot, 'paper_trader') and bot.paper_trader:
                            pt = bot.paper_trader
                            if hasattr(pt, 'rl_exit_policy') and pt.rl_exit_policy:
                                ep = pt.rl_exit_policy
                                exit_exits = getattr(ep, 'total_exits', 0)
                                if hasattr(ep, 'online_learner') and ep.online_learner:
                                    exit_online = getattr(ep.online_learner, 'n_samples', 0)
                        logger.info(f"[RL] Exit Policy: {exit_exits} exits, {exit_online} online updates")
                except Exception as prog_err:
                    logger.debug(f"[PROGRESS] Could not log: {prog_err}")
                
                # UPDATE EXISTING POSITIONS FIRST (check stop loss / take profit)
                if hasattr(bot, 'paper_trader') and bot.paper_trader:
                    try:
                        bot.paper_trader.update_positions(SYMBOL, current_price)
                        logger.info(f"[POSITIONS] Updated positions for {SYMBOL}")
                    except Exception as pos_err:
                        logger.error(f"[POSITIONS] Error updating positions: {pos_err}")
                
                # Generate signal for configured symbol
                signal = bot.generate_unified_signal(SYMBOL)
                
                # ===== LOG HMM REGIME FOR DASHBOARD =====
                try:
                    if hasattr(bot, 'current_hmm_regime') and bot.current_hmm_regime:
                        regime = bot.current_hmm_regime
                        regime_name = regime.get('state', 'Unknown')
                        regime_conf = regime.get('confidence', 0) * 100
                        trend_name = regime.get('trend', 'N/A')
                        vol_name = regime.get('volatility', 'N/A')
                        logger.info(f"[HMM] Regime: {regime_name}, Confidence: {regime_conf:.1f}%, Trend: {trend_name}, Vol: {vol_name}")
                        print(f"  [HMM] Regime: {regime_name} ({regime_conf:.0f}%)")
                except Exception as hmm_err:
                    logger.debug(f"[HMM] Could not log regime: {hmm_err}")
                
                # ===== LOG TCN PREDICTIONS FOR DASHBOARD =====
                try:
                    predictions_logged = False
                    
                    # Try bot's stored multi_timeframe_predictions (most reliable source)
                    if hasattr(bot, 'multi_timeframe_predictions') and bot.multi_timeframe_predictions:
                        for tf, pred in bot.multi_timeframe_predictions.items():
                            # Get direction - either from 'direction' key or direction_probs
                            direction = pred.get('direction', 'N/A')
                            if direction == 'N/A' and 'direction_probs' in pred:
                                probs = pred['direction_probs']
                                if probs.get('UP', 0) > probs.get('DOWN', 0) and probs.get('UP', 0) > probs.get('NEUTRAL', 0):
                                    direction = 'UP'
                                elif probs.get('DOWN', 0) > probs.get('UP', 0) and probs.get('DOWN', 0) > probs.get('NEUTRAL', 0):
                                    direction = 'DOWN'
                                else:
                                    direction = 'NEUTRAL'
                            
                            # predicted_return is already in decimal (0.01 = 1%)
                            pred_return = pred.get('predicted_return', 0) * 100
                            # Use neural_confidence (not confidence)
                            pred_conf = pred.get('neural_confidence', pred.get('confidence', 0)) * 100
                            logger.info(f"[LSTM] {tf}: direction={direction}, return={pred_return:+.2f}%, confidence={pred_conf:.1f}%")
                            predictions_logged = True
                    
                    # Fallback to signal's multi_timeframe_predictions
                    elif signal and signal.get('multi_timeframe_predictions'):
                        for tf, pred in signal.get('multi_timeframe_predictions', {}).items():
                            direction = pred.get('direction', 'N/A')
                            pred_return = pred.get('predicted_return', 0) * 100
                            pred_conf = pred.get('neural_confidence', pred.get('confidence', 0)) * 100
                            logger.info(f"[LSTM] {tf}: direction={direction}, return={pred_return:+.2f}%, confidence={pred_conf:.1f}%")
                            predictions_logged = True
                    
                    # Log warmup status if no predictions yet
                    if not predictions_logged and hasattr(bot, 'feature_buffer'):
                        buffer_size = len(bot.feature_buffer)
                        buffer_needed = getattr(bot, 'sequence_length', 30)
                        if buffer_size < buffer_needed:
                            logger.info(f"[LSTM] WARMUP: {buffer_size}/{buffer_needed} cycles until predictions active")
                        else:
                            logger.info(f"[LSTM] Buffer ready ({buffer_size}/{buffer_needed}) but no predictions available")
                        
                except Exception as lstm_err:
                    logger.warning(f"[LSTM] Could not log predictions: {lstm_err}")
                
                # ===== LOG AND PERSIST LIQUIDITY DATA =====
                # Save liquidity snapshots for model retraining
                try:
                    if signal and signal.get('liquidity'):
                        liq = signal.get('liquidity', {})
                        spread = liq.get('spread_pct', 0)
                        volume = liq.get('volume', 0)
                        oi = liq.get('open_interest', 0)
                        quality = liq.get('quality_score', 0)
                        logger.info(f"[LIQUIDITY] Spread: {spread:.2f}%, Volume: {volume}, OI: {oi}, Quality: {quality}/100")
                        
                        # Persist liquidity snapshot for model retraining
                        try:
                            from src.data.persistence import get_persistence
                            persistence = get_persistence()
                            
                            # Get HMM regime info
                            hmm_regime = None
                            if hasattr(bot, 'current_hmm_regime') and bot.current_hmm_regime:
                                hmm_regime = bot.current_hmm_regime.get('state', 'Unknown')
                            
                            # Get VIX value
                            vix_value = None
                            try:
                                vix_data = bot.data_source.get_data('^VIX', period='1d', interval='1m')
                                if not vix_data.empty:
                                    vix_data.columns = [col.lower() for col in vix_data.columns]
                                    vix_value = float(vix_data['close'].iloc[-1])
                            except:
                                pass
                            
                            liquidity_snapshot = {
                                'timestamp': datetime.now().isoformat(),
                                'symbol': SYMBOL,
                                'option_symbol': liq.get('option_symbol'),
                                'underlying_price': current_price,
                                'strike_price': liq.get('strike'),
                                'option_type': signal.get('action', '').replace('BUY_', '').replace('S', ''),
                                'expiration_date': liq.get('expiration'),
                                'bid': liq.get('bid'),
                                'ask': liq.get('ask'),
                                'spread_pct': spread,
                                'mid_price': liq.get('mid_price'),
                                'volume': volume,
                                'open_interest': oi,
                                'quality_score': quality,
                                'implied_volatility': liq.get('implied_volatility'),
                                'delta': liq.get('delta'),
                                'gamma': liq.get('gamma'),
                                'theta': liq.get('theta'),
                                'vega': liq.get('vega'),
                                'signal_action': signal.get('action'),
                                'signal_confidence': signal.get('confidence', 0),
                                'trade_executed': 0,  # Will update after trade attempt
                                'trade_blocked_reason': None,
                                'hmm_regime': hmm_regime,
                                'vix_value': vix_value
                            }
                            
                            persistence.save_liquidity_snapshot(liquidity_snapshot)
                            logger.debug(f"[PERSIST] Saved liquidity snapshot for {SYMBOL}")
                        except Exception as persist_err:
                            logger.debug(f"[PERSIST] Could not save liquidity: {persist_err}")
                            
                except Exception as liq_err:
                    logger.debug(f"[LIQUIDITY] Could not log: {liq_err}")
                
                # ===== PERSIST PREDICTIONS, HMM, AND SIGNALS FOR TRAINING =====
                try:
                    from src.data.persistence import get_persistence
                    persistence = get_persistence()
                    
                    # Get VIX value for context
                    vix_value = None
                    try:
                        if hasattr(bot, 'data_source'):
                            vix_data = bot.data_source.get_data('^VIX', period='1d', interval='1m')
                            if not vix_data.empty:
                                vix_data.columns = [col.lower() for col in vix_data.columns]
                                vix_value = float(vix_data['close'].iloc[-1])
                    except:
                        pass
                    
                    # Get HMM regime info
                    hmm_regime_str = None
                    if hasattr(bot, 'current_hmm_regime') and bot.current_hmm_regime:
                        hmm_regime_str = bot.current_hmm_regime.get('state', 'Unknown')
                    
                    # 1. Save HMM regime state
                    if hasattr(bot, 'current_hmm_regime') and bot.current_hmm_regime:
                        regime = bot.current_hmm_regime
                        hmm_data = {
                            'timestamp': datetime.now().isoformat(),
                            'symbol': SYMBOL,
                            'trend_state': regime.get('trend_name', regime.get('trend', 'Unknown')),
                            'trend_confidence': regime.get('trend_confidence', 0),
                            'volatility_state': regime.get('volatility_name', regime.get('volatility', 'Unknown')),
                            'volatility_confidence': regime.get('volatility_confidence', 0),
                            'liquidity_state': regime.get('liquidity_name', 'Normal'),
                            'liquidity_confidence': regime.get('liquidity_confidence', 0),
                            'combined_regime': regime.get('state', 'Unknown'),
                            'combined_confidence': regime.get('confidence', regime.get('combined_confidence', 0)),
                            'underlying_price': current_price,
                            'vix_value': vix_value
                        }
                        persistence.save_hmm_regime(hmm_data)
                    
                    # 2. Save multi-timeframe predictions
                    multi_tf = None
                    if signal and signal.get('multi_timeframe_predictions'):
                        multi_tf = signal.get('multi_timeframe_predictions')
                    elif hasattr(bot, 'latest_predictions') and bot.latest_predictions:
                        multi_tf = bot.latest_predictions
                    
                    if multi_tf:
                        for tf, pred in multi_tf.items():
                            pred_data = {
                                'timestamp': datetime.now().isoformat(),
                                'symbol': SYMBOL,
                                'timeframe': tf,
                                'predicted_return': pred.get('predicted_return', pred.get('return', 0)),
                                'predicted_direction': pred.get('direction', 'UNKNOWN'),
                                'confidence': pred.get('confidence', pred.get('neural_confidence', 0)),
                                'underlying_price': current_price,
                                'vix_value': vix_value,
                                'hmm_regime': hmm_regime_str
                            }
                            persistence.save_prediction(pred_data)
                    
                    # 3. Save signal
                    if signal:
                        signal_data = {
                            'timestamp': datetime.now().isoformat(),
                            'symbol': SYMBOL,
                            'action': signal.get('action', 'HOLD'),
                            'confidence': signal.get('confidence', 0),
                            'predicted_return': signal.get('predicted_return', 0),
                            'underlying_price': current_price,
                            'vix_value': vix_value,
                            'hmm_regime': hmm_regime_str,
                            'trade_executed': 0,  # Will update later
                            'multi_timeframe_agreement': signal.get('agreement', 0),
                            'neural_confidence': signal.get('neural_confidence', 0)
                        }
                        persistence.save_signal(signal_data)
                    
                    logger.debug(f"[PERSIST] Saved predictions/HMM/signal for training")
                except Exception as persist_err:
                    logger.debug(f"[PERSIST] Could not save training data: {persist_err}")
                
                # ===== ALWAYS FETCH AND SAVE LIQUIDITY DATA (even for HOLD) =====
                # This provides continuous liquidity data for training
                try:
                    from backend.tradier_trading_system import TradierTradingSystem
                    from tradier_credentials import get_tradier_credentials
                    
                    # Get context variables for liquidity snapshot
                    liq_hmm_regime = None
                    liq_vix_value = None
                    if hasattr(bot, 'current_hmm_regime') and bot.current_hmm_regime:
                        liq_hmm_regime = bot.current_hmm_regime.get('state', 'Unknown')
                    try:
                        if hasattr(bot, 'data_source'):
                            vix_df = bot.data_source.get_data('^VIX', period='1d', interval='1m')
                            if not vix_df.empty:
                                vix_df.columns = [c.lower() for c in vix_df.columns]
                                liq_vix_value = float(vix_df['close'].iloc[-1])
                    except:
                        pass
                    
                    creds = get_tradier_credentials()
                    if creds:
                        tradier_liq = TradierTradingSystem(
                            account_id=creds['account_number'],
                            api_token=creds['access_token'],
                            sandbox=creds.get('is_sandbox', False)
                        )
                        
                        # Get ATM options for both calls and puts
                        for option_type in ['call', 'put']:
                            try:
                                # Select ATM strike
                                option_symbol, strike, expiration = tradier_liq.select_strike_and_expiration(
                                    SYMBOL, option_type, current_price, target_strike=current_price
                                )
                                
                                if option_symbol:
                                    # Get quote for this option
                                    quote = tradier_liq.get_option_quote(option_symbol)
                                    if quote:
                                        bid = float(quote.get('bid') or 0)
                                        ask = float(quote.get('ask') or 0)
                                        volume = int(quote.get('volume') or 0)
                                        oi = int(quote.get('open_interest') or 0)
                                        
                                        if bid > 0 and ask > 0:
                                            mid = (bid + ask) / 2
                                            spread_pct = ((ask - bid) / mid) * 100
                                            
                                            # Calculate quality score
                                            quality = 0
                                            quality += max(0, 40 - spread_pct * 8)  # Spread score
                                            quality += min(30, volume / 10)  # Volume score
                                            quality += min(30, oi / 50)  # OI score
                                            
                                            logger.info(f"[LIQUIDITY] {option_type.upper()} ATM ${strike}: "
                                                       f"Bid=${bid:.2f}, Ask=${ask:.2f}, Spread={spread_pct:.1f}%, "
                                                       f"Vol={volume}, OI={oi}, Quality={quality:.0f}/100")
                                            
                                            # Save to database
                                            from src.data.persistence import get_persistence
                                            persistence = get_persistence()
                                            
                                            liq_snapshot = {
                                                'timestamp': datetime.now().isoformat(),
                                                'symbol': SYMBOL,
                                                'option_symbol': option_symbol,
                                                'underlying_price': current_price,
                                                'strike_price': strike,
                                                'option_type': option_type.upper(),
                                                'expiration_date': expiration,
                                                'bid': bid,
                                                'ask': ask,
                                                'spread_pct': spread_pct,
                                                'mid_price': mid,
                                                'volume': volume,
                                                'open_interest': oi,
                                                'quality_score': quality,
                                                'implied_volatility': float(quote.get('greeks', {}).get('mid_iv') or 0),
                                                'delta': float(quote.get('greeks', {}).get('delta') or 0),
                                                'gamma': float(quote.get('greeks', {}).get('gamma') or 0),
                                                'theta': float(quote.get('greeks', {}).get('theta') or 0),
                                                'vega': float(quote.get('greeks', {}).get('vega') or 0),
                                                'signal_action': signal.get('action', 'HOLD') if signal else 'HOLD',
                                                'signal_confidence': signal.get('confidence', 0) if signal else 0,
                                                'trade_executed': 0,
                                                'trade_blocked_reason': 'HOLD signal' if (not signal or signal.get('action') == 'HOLD') else None,
                                                'hmm_regime': liq_hmm_regime,
                                                'vix_value': liq_vix_value
                                            }
                                            persistence.save_liquidity_snapshot(liq_snapshot)
                            except Exception as opt_err:
                                logger.debug(f"[LIQUIDITY] Could not fetch {option_type} liquidity: {opt_err}")
                        
                        logger.debug(f"[PERSIST] Saved ATM liquidity snapshots for training")
                except Exception as liq_fetch_err:
                    logger.debug(f"[LIQUIDITY] Could not fetch liquidity data: {liq_fetch_err}")
                
                if signal and signal.get('action') != 'HOLD':
                    signals_generated += 1  # Track signal count
                    action = signal.get('action', 'UNKNOWN')
                    confidence = signal.get('confidence', 0) * 100
                    
                    # Log with confidence for dashboard parsing
                    logger.info(f"FINAL Signal: {action} (confidence: {confidence:.1f}%)")
                    logger.info(f"[SIGNAL] {action} @ ${current_price:.2f}")
                    print(f"  [SIGNAL] {action} @ ${current_price:.2f} ({confidence:.1f}%)")
                    
                    # ===== CHECK TCN WARMUP STATUS =====
                    # Block live trades until TCN is fully warmed up
                    lstm_ready = True
                    if hasattr(bot, 'feature_buffer'):
                        buffer_size = len(bot.feature_buffer)
                        buffer_needed = getattr(bot, 'sequence_length', 30)
                        lstm_ready = buffer_size >= buffer_needed
                        
                        if not lstm_ready:
                            logger.warning(f"[SAFETY] 🛡️ TCN not ready ({buffer_size}/{buffer_needed}) - blocking LIVE trades")
                            logger.info(f"[SAFETY] Paper trades will continue for learning")
                            print(f"  [!] TCN WARMUP: {buffer_size}/{buffer_needed} - live trades blocked")
                    
                    # DUAL ACCOUNT SYSTEM: Try Tradier (live), conditionally do paper
                    tradier_success = False
                    bot_safety_block = False  # Track if bot's safety checks blocked it
                    shadow_result = None
                    block_reason = None  # Track why live trade was blocked
                    
                    # Only attempt live trades if LSTM is ready AND paper-only mode is off
                    # Check flag file each cycle for dynamic mode switching
                    current_paper_mode = get_current_mode()

                    # Track mode changes and notify user
                    if not hasattr(check_live_mode, '_last_mode'):
                        check_live_mode._last_mode = current_paper_mode
                    if check_live_mode._last_mode != current_paper_mode:
                        if current_paper_mode:
                            print("\n" + "="*60)
                            print("MODE CHANGED: Switched to PAPER trading")
                            print("(go_live.flag was deleted)")
                            print("="*60 + "\n")
                        else:
                            print("\n" + "="*60)
                            print("MODE CHANGED: Switched to LIVE trading!")
                            print("(go_live.flag detected)")
                            print("="*60 + "\n")
                        check_live_mode._last_mode = current_paper_mode

                    if current_paper_mode:
                        block_reason = "Paper Only Mode enabled - learning with paper trades"
                        logger.info(f"[PAPER-ONLY] Live trading disabled - {block_reason}")
                        print(f"  [📚] LEARNING MODE: Paper trades only")
                        tradier_success = False
                    elif lstm_ready and hasattr(bot, 'shadow_trader') and bot.shadow_trader:
                        logger.info(f"[SHADOW] Attempting Tradier trade FIRST...")
                        try:
                            shadow_result = bot.shadow_trader.place_trade(
                                symbol=SYMBOL,
                                prediction_data=signal,
                                current_price=current_price
                            )
                            
                            if shadow_result:
                                # Check if trade was blocked (insufficient funds, market closed, etc.)
                                if shadow_result.get('blocked'):
                                    block_reason = shadow_result.get('reason', 'Trade blocked')
                                    logger.warning(f"[SHADOW] Live trade BLOCKED: {block_reason}")
                                    print(f"  [!] LIVE BLOCKED: {block_reason}")
                                    print(f"  [✓] Paper trade will still execute for learning")
                                    tradier_success = False
                                    bot_safety_block = False  # Not bot safety, just insufficient funds
                                # Handle straddle orders (have call_order and put_order instead of single order_id)
                                elif shadow_result.get('strategy') == 'STRADDLE':
                                    call_order = shadow_result.get('call_order', {})
                                    put_order = shadow_result.get('put_order', {})
                                    order_id = f"{call_order.get('order_id')},{put_order.get('order_id')}"
                                    order_status = f"call:{call_order.get('status')}, put:{put_order.get('status')}"
                                    logger.info(f"[SHADOW] Straddle orders placed!")
                                    logger.info(f"[SHADOW] Call Order: {call_order.get('order_id')}, Put Order: {put_order.get('order_id')}")
                                    print(f"  [+] TRADIER: Straddle orders placed")
                                    # For straddles, we'll skip verification for now (complex)
                                    tradier_success = True
                                elif shadow_result.get('order_id'):
                                    # Regular single-leg order
                                    order_id = shadow_result.get('order_id')
                                    order_status = shadow_result.get('status')
                                    
                                    logger.info(f"[SHADOW] Tradier order placed!")
                                    logger.info(f"[SHADOW] Order ID: {order_id}, Status: {order_status}")
                                    print(f"  [+] TRADIER: Order placed (ID: {order_id})")
                                    
                                    # Log to audit trail for safety
                                    try:
                                        import json
                                        audit_file = Path('data/tradier_orders_audit.jsonl')
                                        audit_file.parent.mkdir(exist_ok=True)
                                        with open(audit_file, 'a') as f:
                                            audit_entry = {
                                                'timestamp': datetime.now().isoformat(),
                                                'order_id': order_id,
                                                'option_symbol': shadow_result.get('option_symbol'),
                                                'status': order_status,
                                                'action': action,
                                                'symbol': SYMBOL,
                                                'price': current_price
                                            }
                                            f.write(json.dumps(audit_entry) + '\n')
                                        logger.info(f"[AUDIT] Order logged to audit trail")
                                    except Exception as audit_err:
                                        logger.warning(f"[AUDIT] Failed to log to audit trail: {audit_err}")
                                    
                                    # Verify order fill (wait up to 5 seconds)
                                    logger.info(f"[SHADOW] Verifying order fill...")
                                    verification = bot.shadow_trader.verify_order_filled(order_id, max_wait_seconds=5)
                                    
                                    if verification['success']:
                                        logger.info(f"[SHADOW] Order {order_id} CONFIRMED FILLED")
                                        print(f"  [OK] TRADIER: Order filled successfully!")
                                        tradier_success = True
                                    elif verification['success'] is False:
                                        block_reason = verification.get('reason', 'Order rejected')
                                        logger.error(f"[SHADOW] Order {order_id} REJECTED: {block_reason}")
                                        print(f"  [!] TRADIER: Order REJECTED - {block_reason}")
                                        tradier_success = False
                                        bot_safety_block = False  # Tradier rejected, not bot's decision
                                    else:
                                        logger.warning(f"[SHADOW] Order {order_id} still pending after 5s")
                                        print(f"  [!] TRADIER: Order still processing - assuming success")
                                        tradier_success = True  # Assume success if pending
                                else:
                                    # Unexpected result format - treat as blocked
                                    block_reason = f"Unexpected Tradier response: {shadow_result}"
                                    logger.warning(f"[SHADOW] {block_reason}")
                                    print(f"  [!] TRADIER: Unexpected response")
                                    tradier_success = False
                                    bot_safety_block = False
                            else:
                                # shadow_result is None - could be HOLD signal or blocked
                                if signal.get('action') == 'HOLD':
                                    # HOLD signal - no trade attempted
                                    block_reason = 'Signal is HOLD'
                                    tradier_success = False
                                    bot_safety_block = True
                                else:
                                    block_reason = 'Liquidity/safety checks failed'
                                    logger.warning(f"[SHADOW] Bot blocked trade: {block_reason}")
                                    print(f"  [!] BOT BLOCKED: {block_reason}")
                                    print(f"  [✓] Correct decision - no paper trade needed")
                                    tradier_success = False
                                    bot_safety_block = True  # Bot correctly blocked this trade
                        except Exception as se:
                            block_reason = f"Error: {str(se)}"
                            logger.error(f"[SHADOW] ERROR placing Tradier trade: {se}")
                            import traceback
                            logger.error(traceback.format_exc())
                            print(f"  [!] TRADIER: Error - {se}")
                            tradier_success = False
                            bot_safety_block = False  # External error, not bot's decision
                    elif not lstm_ready:
                        # TCN warmup - block live trades but continue paper trading
                        block_reason = f"TCN warmup ({len(bot.feature_buffer)}/{getattr(bot, 'sequence_length', 30)} cycles)"
                        logger.info(f"[SHADOW] Live trading BLOCKED: {block_reason}")
                        print(f"  [!] LIVE BLOCKED: TCN still warming up")
                        tradier_success = False
                        bot_safety_block = False  # Not a bot safety block, just warmup
                    else:
                        block_reason = "Shadow trader not initialized"
                        logger.error(f"[SHADOW] Shadow trader NOT AVAILABLE!")
                        print(f"  [!] NO SHADOW TRADER - skipping trade")
                        tradier_success = False
                        bot_safety_block = False  # External error, not bot's decision
                    
                    # Only place paper trade if:
                    # 1. Tradier succeeded (will be marked as real trade)
                    # 2. External error (NOT bot's safety checks)
                    should_paper_trade = tradier_success or not bot_safety_block
                    
                    if should_paper_trade:
                        logger.info(f"[PAPER] Placing trade (Tradier: {'SUCCESS' if tradier_success else 'REJECTED'})")
                        
                        # Pass flag to indicate this might be a live trade (bypass paper limits)
                        prediction_data_with_flag = signal.copy()
                        prediction_data_with_flag['_is_live_trade'] = tradier_success
                        
                        trade_result = bot.paper_trader.place_trade(
                            symbol=SYMBOL,
                            prediction_data=prediction_data_with_flag,
                            current_price=current_price
                        )
                        
                        if trade_result:
                            logger.info(f"[PAPER] Paper trade placed: {action}")
                            print(f"  [OK] PAPER: Trade recorded (Strike: ${trade_result.strike_price:.2f})")
                            
                            # Link Tradier order to paper trade (if Tradier succeeded)
                            if tradier_success and shadow_result:
                                tradier_option_symbol = shadow_result.get('option_symbol')
                                order_id = shadow_result.get('order_id')
                                if tradier_option_symbol:
                                    trade_result.tradier_option_symbol = tradier_option_symbol
                                    trade_result.tradier_order_id = order_id
                                    trade_result.is_real_trade = True  # MARK AS REAL TRADE - TOP PRIORITY!
                                    logger.warning(f"[REAL TRADE] This is a REAL Tradier trade - TOP PRIORITY!")
                                    logger.info(f"[LINK] Paper trade {trade_result.id} <-> Tradier {tradier_option_symbol}")
                                    print(f"  [REAL] REAL MONEY TRADE - Priority management enabled")
                                    print(f"  [LINK] Paper trade linked to Tradier order")
                                    
                                    # Update database to mark as real trade (with retry and verification)
                                    db_update_success = False
                                    for attempt in range(3):  # Try up to 3 times
                                        try:
                                            import sqlite3
                                            conn = sqlite3.connect('data/paper_trading.db', timeout=10)
                                            cursor = conn.cursor()
                                            cursor.execute('''
                                                UPDATE trades 
                                                SET tradier_option_symbol = ?, 
                                                    tradier_order_id = ?,
                                                    is_real_trade = 1,
                                                    live_blocked_reason = NULL
                                                WHERE id = ?
                                            ''', (tradier_option_symbol, order_id, trade_result.id))
                                            
                                            # Verify the update worked
                                            if cursor.rowcount == 0:
                                                logger.error(f"[DB] UPDATE affected 0 rows! Trade ID might not exist: {trade_result.id}")
                                                conn.close()
                                                continue
                                            
                                            conn.commit()
                                            
                                            # Double-check the update persisted
                                            cursor.execute('''
                                                SELECT is_real_trade, tradier_option_symbol 
                                                FROM trades 
                                                WHERE id = ?
                                            ''', (trade_result.id,))
                                            row = cursor.fetchone()
                                            
                                            if row and row[0] == 1 and row[1] == tradier_option_symbol:
                                                logger.info(f"[DB] ✓ Trade {trade_result.id} verified as REAL in database")
                                                db_update_success = True
                                                conn.close()
                                                break
                                            else:
                                                logger.error(f"[DB] Verification failed: {row}")
                                                conn.close()
                                                continue
                                                
                                        except Exception as db_err:
                                            logger.error(f"[DB] Attempt {attempt + 1}/3 failed: {db_err}")
                                            if attempt < 2:
                                                time.sleep(0.5)  # Wait before retry
                                
                                if not db_update_success:
                                    logger.error(f"[DB] ⚠️ CRITICAL: Failed to mark trade as REAL after 3 attempts!")
                                    logger.error(f"[DB] Trade {trade_result.id} may be orphaned - sync will catch it")
                                    logger.error(f"[DB] Tradier: {tradier_option_symbol}, Order: {order_id}")
                            else:
                                # Mark as paper-only with block reason
                                trade_result.live_blocked_reason = block_reason
                                logger.info(f"[PAPER] Paper-only trade (no Tradier link) - bot continues learning")
                                logger.info(f"[PAPER] Live blocked reason: {block_reason}")
                                print(f"  [LEARN] Paper trade for learning only")
                                print(f"  [REASON] Live blocked: {block_reason}")
                                
                                # Update database with block reason
                                try:
                                    import sqlite3
                                    conn = sqlite3.connect('data/paper_trading.db')
                                    cursor = conn.cursor()
                                    cursor.execute('''
                                        UPDATE trades 
                                        SET live_blocked_reason = ?
                                        WHERE id = ?
                                    ''', (block_reason, trade_result.id))
                                    conn.commit()
                                    conn.close()
                                    logger.info(f"[DB] Saved block reason for trade {trade_result.id}")
                                except Exception as db_err:
                                    logger.error(f"[DB] Failed to save block reason: {db_err}")
                        else:
                            logger.error(f"[PAPER] Failed to place paper trade!")
                            print(f"  [!] PAPER: Failed to record trade")
                    else:
                        # Bot's safety checks blocked - no paper trade needed
                        logger.info(f"[SKIP] Not placing paper trade - bot correctly blocked for safety")
                        logger.info(f"[SKIP] Reason: {block_reason}")
                    
                    # Update liquidity snapshot with trade outcome
                    try:
                        from src.data.persistence import get_persistence
                        persistence = get_persistence()
                        
                        # Update the most recent liquidity snapshot for this symbol
                        import sqlite3
                        conn = sqlite3.connect('data/db/historical.db')
                        cursor = conn.cursor()
                        cursor.execute('''
                            UPDATE liquidity_snapshots 
                            SET trade_executed = ?, trade_blocked_reason = ?
                            WHERE symbol = ? 
                            AND id = (SELECT MAX(id) FROM liquidity_snapshots WHERE symbol = ?)
                        ''', (1 if tradier_success else 0, block_reason, SYMBOL, SYMBOL))
                        conn.commit()
                        conn.close()
                        logger.debug(f"[PERSIST] Updated liquidity snapshot: executed={tradier_success}")
                    except Exception as update_err:
                        logger.debug(f"[PERSIST] Could not update liquidity outcome: {update_err}")
                        
                else:
                    # Log HOLD signal with confidence for dashboard
                    # Get the calculated confidence from the signal, even if action is HOLD
                    hold_confidence = 0
                    if signal:
                        hold_confidence = signal.get('confidence', 0) * 100
                        # If signal has reasoning, it had a valid calculation
                        if signal.get('reasoning') and hold_confidence == 0:
                            # Check for confidence in reasoning text
                            for reason in signal.get('reasoning', []):
                                if 'confidence' in reason.lower():
                                    hold_confidence = 35.0  # Default moderate confidence
                                    break
                    
                    logger.info(f"FINAL Signal: HOLD (confidence: {hold_confidence:.1f}%)")
                    print(f"  [SIGNAL] HOLD ({hold_confidence:.1f}%)")
        
        except Exception as e:
            logger.error(f"[LIVE] Error in cycle: {e}")
            print(f"  [ERROR] {e}")
        
        # Wait for next cycle
        cycle_duration = time.time() - cycle_start
        wait_time = max(0, cycle_interval - cycle_duration)
        if wait_time > 0:
            time.sleep(wait_time)

except KeyboardInterrupt:
    print("\n[*] Stopping...")
    logger.info("[STOP] Stopped by user")
    print(f"[*] Final balance: ${bot.paper_trader.current_balance:,.2f}")
    
    # SAVE ALL LEARNING BEFORE EXIT - Don't lose progress!
    print("[*] Saving ALL learned models...")
    try:
        import os
        os.makedirs('models', exist_ok=True)
        os.makedirs('models/state', exist_ok=True)
        
        saved_count = 0
        
        # 1. Save Exit Policy (Thompson Sampling + Neural)
        if hasattr(bot, 'paper_trader') and bot.paper_trader:
            if hasattr(bot.paper_trader, 'rl_exit_policy') and bot.paper_trader.rl_exit_policy:
                bot.paper_trader.rl_exit_policy.save('models/rl_exit_policy_live.pth')
                stats = bot.paper_trader.rl_exit_policy.get_stats()
                print(f"  [OK] Exit Policy: {stats.get('total_exits', 0)} exits, {stats.get('online_updates', 0)} online updates")
                logger.info(f"[SAVE] Exit Policy saved: {stats}")
                saved_count += 1
        
        # 2. Save Threshold Policy
        if hasattr(bot, 'rl_threshold_learner') and bot.rl_threshold_learner:
            bot.rl_threshold_learner.save('models/rl_threshold_live.pth')
            print("  [OK] Threshold Policy saved")
            logger.info("[SAVE] Threshold Policy saved")
            saved_count += 1
        
        # 3. Save HMM Regime Detector
        if hasattr(bot, 'hmm_regime_detector') and bot.hmm_regime_detector:
            if hasattr(bot.hmm_regime_detector, 'save'):
                bot.hmm_regime_detector.save('models/multi_dimensional_hmm.pkl')
                print("  [OK] HMM Regime Detector saved")
                logger.info("[SAVE] HMM Regime Detector saved")
                saved_count += 1
        
        # 4. Save Adaptive Timeframe Weights
        if hasattr(bot, 'adaptive_weights') and bot.adaptive_weights:
            if hasattr(bot.adaptive_weights, 'save'):
                bot.adaptive_weights.save('models/adaptive_timeframe_weights.json')
                print("  [OK] Adaptive Timeframe Weights saved")
                logger.info("[SAVE] Adaptive Timeframe Weights saved")
                saved_count += 1
        
        # 5. Save Main Neural Network (Bayesian TCN + Gaussian Processor)
        if hasattr(bot, 'save_model'):
            if bot.save_model():
                print("  [OK] Neural Network (TCN + Gaussian) saved")
                logger.info("[SAVE] Neural Network state saved")
                saved_count += 1
            else:
                print("  [WARN] Neural Network had nothing to save")
        
        # 6. Save Gaussian Processor separately (backup)
        if hasattr(bot, 'gaussian_processor') and bot.gaussian_processor:
            if hasattr(bot.gaussian_processor, 'is_fitted') and bot.gaussian_processor.is_fitted:
                import pickle
                with open('models/state/gaussian_processor_backup.pkl', 'wb') as f:
                    pickle.dump(bot.gaussian_processor, f)
                print("  [OK] Gaussian Processor backup saved")
                saved_count += 1
            
        print(f"[OK] All {saved_count} models saved successfully!")
        logger.info(f"[SAVE] All {saved_count} models saved on shutdown")
        
    except Exception as save_err:
        print(f"  [WARN] Error saving models: {save_err}")
        logger.error(f"[SAVE] Error on shutdown: {save_err}")
