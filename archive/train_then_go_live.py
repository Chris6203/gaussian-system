#!/usr/bin/env python3
"""
Train Then Go Live
==================

Two-phase trading system:
1. Phase 1: Train on historical data using run_simulation.py
2. Phase 2: Live trading with real-time data (paper + optional shadow trading)

Configuration is loaded from config.json. Key settings:
- trading.symbol: Symbol to trade (e.g., "SPY")
- trading.initial_balance: Starting balance
- live_trading.paper_only_mode: If true, no real Tradier trades
- live_trading.position_recovery: Settings for handling stale positions

Usage:
    python train_then_go_live.py
"""

import sys
import os
import subprocess
import time
import logging
import json
import shutil
import glob
import importlib
import inspect
from datetime import datetime
from pathlib import Path
from typing import Optional, Any

# Fix encoding on Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


# =============================================================================
# PHASE 1: TRAINING
# =============================================================================

def run_training_phase(script_dir: Path, run_timestamp: str) -> Path:
    """
    Run Phase 1: Historical training via run_simulation.py.
    
    Args:
        script_dir: Directory containing run_simulation.py
        run_timestamp: Timestamp for this training run
        
    Returns:
        Path to the models directory for this run
    """
    print("=" * 80)
    print("TRAIN THEN GO LIVE")
    print("=" * 80)
    print()
    print("Phase 1: Training on historical data")
    print("Phase 2: Live shadow trading (1-minute cycles)")
    print()
    
    run_simulation_path = script_dir / 'run_simulation.py'
    run_models_dir = script_dir / "models" / f"run_{run_timestamp}"
    training_log_file = script_dir / "logs" / f"real_bot_simulation_{run_timestamp}.log"
    training_log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Load training balance from config
    training_balance = _load_training_balance(script_dir)
    
    print("-" * 80)
    print("PHASE 1: TRAINING")
    print("-" * 80)
    print()
    print(f"[*] Starting historical training...")
    print(f"[*] Using: {run_simulation_path}")
    print(f"[*] Working directory: {script_dir}")
    print(f"[*] Run directory: {run_models_dir.name}")
    print(f"[*] Training log: {training_log_file}")
    print(f"[*] Initial balance: ${training_balance:,.2f}")
    print()
    
    # Prepare environment for Phase 1
    phase1_env = os.environ.copy()
    phase1_env['MODEL_RUN_TIMESTAMP'] = run_timestamp
    phase1_env['MODEL_RUN_DIR'] = str(run_models_dir)
    phase1_env['TRAINING_LOG_FILE'] = str(training_log_file)
    phase1_env['TRAINING_INITIAL_BALANCE'] = str(training_balance)
    phase1_env['USE_SHADOW_TRADING'] = '0'  # Disable shadow trading during training
    
    # Check if weekend (Saturday=5, Sunday=6) to skip data collection
    if datetime.now().weekday() >= 5:
        print("[*] Weekend detected - disabling data collection for training")
        phase1_env['SKIP_DATA_UPDATE'] = '1'
    
    # Run training
    result = subprocess.run(
        [sys.executable, str(run_simulation_path)],
        cwd=str(script_dir),
        env=phase1_env
    )
    
    if result.returncode != 0:
        print()
        print("[ERROR] Training failed!")
        sys.exit(1)
    
    print()
    print("[OK] Training complete!")
    print()
    
    return run_models_dir


def _load_training_balance(script_dir: Path) -> float:
    """Load initial balance from config.json."""
    try:
        with open(script_dir / 'config.json', 'r') as f:
            cfg = json.load(f)
        return float(cfg.get('trading', {}).get('initial_balance', 1000.0))
    except (FileNotFoundError, json.JSONDecodeError, KeyError, ValueError):
        return 1000.0


# =============================================================================
# PHASE 2: LIVE TRADING
# =============================================================================

def setup_logging(script_dir: Path, training_log_file: Path) -> logging.Logger:
    """Configure logging for Phase 2."""
    os.makedirs('logs', exist_ok=True)
    
    # Use training log file if it exists, otherwise find most recent
    if training_log_file.exists():
        log_file = str(training_log_file)
        print(f"[*] Continuing to log file from Phase 1: {log_file}")
    else:
        log_files = glob.glob('logs/real_bot_simulation_*.log')
        if log_files:
            log_file = max(log_files, key=lambda x: os.path.getmtime(x))
            print(f"[*] Continuing to log file: {log_file}")
        else:
            log_file = f"logs/real_bot_simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            print(f"[*] Creating new log file: {log_file}")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='a', encoding='utf-8'),
            logging.StreamHandler()
        ],
        force=True
    )
    
    logger = logging.getLogger(__name__)
    logger.info("=" * 70)
    logger.info("PHASE 2: LIVE SHADOW TRADING")
    logger.info("=" * 70)
    
    return logger


def initialize_bot(config: Any, logger: logging.Logger) -> Any:
    """Initialize the trading bot with trained models."""
    from config_loader import get_config
    
    # Import bot module
    unified_bot_module = importlib.import_module('unified_options_trading_bot')
    
    # Bridge TradingModeConfig if available
    try:
        from trading_mode_config import TradingModeConfig
        unified_bot_module.TradingModeConfig = TradingModeConfig
        logger.info("[CONFIG] TradingModeConfig bridge ready for UnifiedOptionsBot")
    except ImportError as exc:
        logger.warning(f"[CONFIG] Could not import trading_mode_config: {exc}")
    
    UnifiedOptionsBot = unified_bot_module.UnifiedOptionsBot
    
    # Verify bot supports config parameter
    bot_init_sig = inspect.signature(UnifiedOptionsBot.__init__)
    if 'config' not in bot_init_sig.parameters:
        raise ImportError("ERROR: Wrong UnifiedOptionsBot version - expected 'config' parameter")
    
    initial_balance = config.get_initial_balance()
    logger.info(f"[CONFIG] Trading symbol: {config.get_symbol()}, initial_balance: ${initial_balance:,.2f}")
    
    # Initialize bot
    logger.info("[LIVE] Initializing bot for live trading...")
    bot = UnifiedOptionsBot(initial_balance=initial_balance, config=config)
    
    # Configure for live mode
    bot.simulation_mode = False
    if hasattr(bot, 'disable_live_data_fetch'):
        bot.disable_live_data_fetch = False
    bot.paper_trader.simulation_mode = False
    bot.paper_trader.time_travel_mode = False
    logger.info("[LIVE] Live mode enabled")
    
    return bot


def load_trained_models(logger: logging.Logger) -> Path:
    """Find and load trained models from most recent run."""
    models_dir = Path("models")
    logger.info(f"[LIVE] Looking for trained models in: {models_dir.resolve()}")
    
    run_dirs = sorted(models_dir.glob("run_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    
    if not run_dirs:
        logger.error("[ERROR] No training run directories found!")
        print("[ERROR] No trained models found! Did training complete successfully?")
        sys.exit(1)
    
    run_dir = run_dirs[0]
    logger.info(f"[LIVE] Using models from most recent training run: {run_dir.name}")
    print(f"[*] Loading models from: {run_dir.name}")
    
    # Copy trained state to models/state/
    trained_state_dir = run_dir / "state"
    target_state_dir = Path("models/state")
    
    if trained_state_dir.exists():
        logger.info(f"[LIVE] Copying trained models from {trained_state_dir} to {target_state_dir}")
        target_state_dir.mkdir(parents=True, exist_ok=True)
        
        for file_path in trained_state_dir.glob("*"):
            shutil.copy2(file_path, target_state_dir / file_path.name)
            logger.info(f"   [OK] Copied {file_path.name}")
    else:
        logger.warning(f"[WARN] No trained models found at {trained_state_dir}")
    
    return run_dir


def configure_liquidity_validation(bot: Any, config: Any, logger: logging.Logger) -> None:
    """Configure liquidity validation for paper trading."""
    from trading_mode import get_trading_mode
    
    liquidity_settings = config.get('liquidity') or {}
    liquidity_checks_enabled = liquidity_settings.get('enable_liquidity_checks', True)
    
    try:
        mode = get_trading_mode()
        
        if mode['is_sandbox']:
            logger.info("[SANDBOX] Paper trader liquidity validation DISABLED (simulation mode)")
            return
        
        if not liquidity_checks_enabled:
            bot.paper_trader.enable_liquidity_checks = False
            logger.info("[LIVE] Liquidity validation disabled via config.json")
            return
        
        # Enable liquidity validation
        from backend.liquidity_validator import LiquidityValidator
        
        bot.paper_trader.liquidity_validator = LiquidityValidator(
            max_bid_ask_spread_pct=liquidity_settings.get('max_bid_ask_spread_pct', 5.0),
            min_volume=liquidity_settings.get('min_volume', 100),
            min_open_interest=liquidity_settings.get('min_open_interest', 75),
            max_strike_deviation_pct=liquidity_settings.get('max_strike_deviation_pct', 15.0)
        )
        bot.paper_trader.enable_liquidity_checks = True
        
        logger.info("[LIVE] Paper trader liquidity validation ENABLED")
        logger.info(
            f"[LIVE] Liquidity thresholds -> spread: {liquidity_settings.get('max_bid_ask_spread_pct', 5.0):.1f}% | "
            f"volume: {liquidity_settings.get('min_volume', 100)} | "
            f"OI: {liquidity_settings.get('min_open_interest', 75)}"
        )
        
    except ImportError:
        logger.warning("[LIVE] Liquidity validator not available for paper trader")
    except Exception as e:
        logger.warning(f"[LIVE] Could not configure paper trader liquidity checks: {e}")


def initialize_shadow_trading(config: Any, logger: logging.Logger) -> Optional[Any]:
    """Initialize shadow trading system based on configuration."""
    from backend.tradier_trading_system import TradierTradingSystem
    from tradier_credentials import TradierCredentials
    from trading_mode import get_trading_mode, print_trading_mode_banner
    
    print_trading_mode_banner()
    
    try:
        mode = get_trading_mode()
        creds = TradierCredentials()
        
        if mode['is_sandbox']:
            # Sandbox mode - fake money
            sandbox_creds = creds.get_sandbox_credentials()
            
            if sandbox_creds:
                shadow_trader = TradierTradingSystem(
                    account_id=sandbox_creds['account_number'],
                    api_token=sandbox_creds['access_token'],
                    sandbox=True
                )
                logger.info(f"[SANDBOX] Connected to Tradier Sandbox: {sandbox_creds['account_number']}")
                print(f"[OK] Shadow trading to SANDBOX: {sandbox_creds['account_number']}")
                return shadow_trader
            else:
                logger.warning("[SANDBOX] No sandbox credentials found")
                print("[WARN] Shadow trading disabled - no credentials")
                return None
        else:
            # Live mode - real money!
            live_creds = creds.get_live_credentials()
            
            if live_creds:
                shadow_trader = TradierTradingSystem(
                    account_id=live_creds['account_number'],
                    api_token=live_creds['access_token'],
                    sandbox=False
                )
                logger.warning(f"[LIVE] 🔴 Connected to Tradier LIVE: {live_creds['account_number']}")
                logger.warning("[LIVE] 🔴 REAL MONEY MODE - Losses are permanent!")
                print(f"[🔴] LIVE trading to PRODUCTION: {live_creds['account_number']}")
                print("[🔴] REAL MONEY AT RISK!")
                return shadow_trader
            else:
                logger.error("[LIVE] No live credentials found")
                print("[ERROR] Live trading disabled - no credentials")
                print("        Run: python setup_credentials.py")
                sys.exit(1)
                
    except Exception as e:
        logger.error(f"[SHADOW] Could not initialize: {e}")
        print(f"[WARN] Shadow trading disabled - {e}")
        return None


def create_shadow_bridge(
    shadow_trader: Optional[Any], 
    config: Any,
    logger: logging.Logger
) -> Optional[Any]:
    """Create shadow trading bridge with configuration."""
    from backend.shadow_trading_bridge import ShadowTradingBridge
    
    # Get paper_only_mode from config
    paper_only_mode = config.get('live_trading', 'paper_only_mode', default=True)
    
    if paper_only_mode:
        logger.info("[PAPER-ONLY] Paper-only mode enabled - no live trades will be placed")
        print(">>> PAPER ONLY MODE - No live trades will be placed <<<")
    
    # Get shadow sync settings
    shadow_sync = config.get('live_trading', 'shadow_sync', default={})
    
    return ShadowTradingBridge(
        tradier_system=shadow_trader,
        paper_only_mode=paper_only_mode,
        verify_entry_fill=shadow_sync.get('verify_entry_fill', True),
        verify_exit_fill=shadow_sync.get('verify_exit_fill', True),
        entry_fill_timeout=shadow_sync.get('entry_fill_timeout_seconds', 5),
        exit_fill_timeout=shadow_sync.get('exit_fill_timeout_seconds', 10)
    )


def run_live_phase(
    script_dir: Path,
    run_models_dir: Path,
    training_log_file: Path
) -> None:
    """
    Run Phase 2: Live trading with real-time data.
    
    Args:
        script_dir: Script directory
        run_models_dir: Path to trained models directory
        training_log_file: Path to training log file
    """
    print("-" * 80)
    print("PHASE 2: LIVE SHADOW TRADING")
    print("-" * 80)
    print()
    
    # Ensure we're in the correct directory
    output_dir = Path(__file__).parent.resolve()
    if Path.cwd().resolve() != output_dir:
        os.chdir(output_dir)
        print(f"[*] Changed working directory to: {output_dir}")
    
    print("[*] Switching to live mode...")
    print("[*] Will fetch new data every 60 seconds")
    print("[*] Press Ctrl+C to stop")
    print()
    
    # Setup logging
    logger = setup_logging(script_dir, training_log_file)
    
    # Load configuration
    from config_loader import get_config
    config = get_config()
    
    # Load trained models
    run_dir = load_trained_models(logger)
    
    # Initialize bot
    bot = initialize_bot(config, logger)
    
    # ==========================================================================
    # INTEGRATE ENHANCED COMPONENTS
    # ==========================================================================
    try:
        from backend.integration import integrate_all_components, patch_bot_with_components
        
        logger.info("[INTEGRATION] Initializing enhanced components...")
        
        # Get symbols for parallel fetching
        symbols = config.get('data_fetching', 'symbols', default=['SPY'])
        if isinstance(symbols, list) and len(symbols) > 0:
            primary_symbol = config.get_symbol()
            if primary_symbol not in symbols:
                symbols = [primary_symbol] + symbols[:11]  # Primary + up to 11 context
        
        # Integrate all components
        components = integrate_all_components(
            bot=bot,
            config=config._data if hasattr(config, '_data') else {},
            symbols=symbols
        )
        
        # Patch bot to use enhanced components
        patch_bot_with_components(bot, components)
        
        # Store components for access
        bot._enhanced_components = components
        
        logger.info("[INTEGRATION] ✅ Enhanced components integrated:")
        logger.info(f"   - RobustFetcher: {'✓' if components.get('fetcher') else '✗'}")
        logger.info(f"   - FeatureCache: {'✓' if components.get('feature_cache') else '✗'}")
        logger.info(f"   - CalibrationTracker: {'✓' if components.get('calibration') else '✗'}")
        logger.info(f"   - HealthCheck: {'✓' if components.get('health_check') else '✗'}")
        logger.info(f"   - ModelHealth: {'✓' if components.get('model_health') else '✗'}")
        logger.info(f"   - RL Enhancements: {'✓' if components.get('trainer') else '✗'}")
        
    except ImportError as e:
        logger.warning(f"[INTEGRATION] Enhanced components not available: {e}")
    except Exception as e:
        logger.warning(f"[INTEGRATION] Could not integrate components: {e}")
        import traceback
        traceback.print_exc()
    
    # Log subsystem health
    from backend.live_trading_engine import log_subsystem_health
    log_subsystem_health(bot, config, logger)
    
    # Configure liquidity validation
    configure_liquidity_validation(bot, config, logger)
    
    # Initialize shadow trading
    shadow_trader = initialize_shadow_trading(config, logger)
    shadow_bridge = create_shadow_bridge(shadow_trader, config, logger)
    
    # Position recovery on restart
    from backend.position_recovery import evaluate_existing_positions
    
    historical_period = config.get('data_sources', 'historical_period', default='7d')
    print()
    evaluate_existing_positions(
        bot=bot,
        symbol=config.get_symbol(),
        config=config,
        historical_period=historical_period
    )
    
    # Create and run live trading engine
    from backend.live_trading_engine import LiveTradingEngine
    
    print()
    logger.info("[LIVE] Starting live trading loop...")
    
    engine = LiveTradingEngine(
        bot=bot,
        config=config,
        shadow_bridge=shadow_bridge,
        symbol=config.get_symbol()
    )
    
    try:
        engine.run_forever()
    except Exception as e:
        logger.error(f"[LIVE] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Print final info
    print()
    print(f"Models saved to: {run_dir}")
    print(f"Summary: {run_dir}/SUMMARY.txt")
    print()
    
    logger.info("[LIVE] Done")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main entry point for train-then-go-live workflow."""
    script_dir = Path(__file__).parent.resolve()
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    training_log_file = script_dir / "logs" / f"real_bot_simulation_{run_timestamp}.log"
    
    # Phase 1: Training
    run_models_dir = run_training_phase(script_dir, run_timestamp)
    
    # Phase 2: Live trading
    run_live_phase(script_dir, run_models_dir, training_log_file)


if __name__ == "__main__":
    main()
