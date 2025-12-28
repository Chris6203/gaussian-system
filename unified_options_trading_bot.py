#!/usr/bin/env python3
"""
UNIFIED OPTIONS TRADING BOT (Refactored)
=========================================

Combines:
- Options-specific volatility signals
- Bayesian neural network predictions with Gaussian kernels
- TCN/LSTM temporal patterns
- Paper trading execution

Architecture:
- Neural networks extracted to bot_modules/neural_networks.py
- Gaussian processor extracted to bot_modules/gaussian_processor.py
- Market utilities extracted to bot_modules/market_utils.py
- Calibration utilities extracted to bot_modules/calibration_utils.py
- Features extracted to bot_modules/features.py
- Signals extracted to bot_modules/signals.py

Key patches:
- Lazy neural model initialization once feature buffer reaches sequence_length
- Interval-aware volatility scaling (1m/5m/15m)
- 15-min prediction uses the correct symbol
- Safer RSI edge case
- Strategy→order mapping improved
- DB: update record after execution
- Market-closed backoff
"""

import os
import json
import pickle
import sqlite3
import logging
import time
import warnings
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd
import pytz
import torch
import torch.nn as nn

# ------------------------------------------------------------------------------
# Logging (set up early)
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("unified_trading_bot.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------------------
# Import Modular Components
# ------------------------------------------------------------------------------
# Neural network components (extracted to bot_modules)
from bot_modules.neural_networks import (
    BayesianLinear,
    RBFKernelLayer,
    TCNBlock,
    OptionsTCN,
    OptionsLSTM,
    ResidualBlock,
    UnifiedOptionsPredictor,
    create_predictor,
)

# Gaussian kernel processor
from bot_modules.gaussian_processor import GaussianKernelProcessor

# Market utilities (market hours, device management)
from bot_modules.market_utils import (
    MarketHours,
    DeviceManager,
    annualization_scale,
)

# Backend components
from backend.paper_trading_system import PaperTradingSystem, OrderType
from backend.enhanced_data_sources import EnhancedDataSource
from backend.adaptive_timeframe_weights import AdaptiveTimeframeWeights

# Edge-based entry controller (world-model scoring)
try:
    from backend.edge_entry_controller import decide_from_signal as edge_decide_from_signal
    HAS_EDGE_ENTRY_CONTROLLER = True
except Exception:
    edge_decide_from_signal = None  # type: ignore
    HAS_EDGE_ENTRY_CONTROLLER = False

# Full-state decision/missed logging sidecar (optional; used for Q-scorer dataset)
try:
    from backend.decision_pipeline import DecisionPipeline
    HAS_DECISION_PIPELINE = True
except Exception:
    DecisionPipeline = None  # type: ignore
    HAS_DECISION_PIPELINE = False

# Q-style entry controller (offline Q regression scorer)
try:
    from backend.q_entry_controller import QEntryController
    HAS_Q_ENTRY_CONTROLLER = True
except Exception:
    QEntryController = None  # type: ignore
    HAS_Q_ENTRY_CONTROLLER = False

# UNIFIED RL Policy (replaces separate entry/threshold/exit policies)
try:
    from backend.unified_rl_policy import UnifiedRLPolicy, TradeState, AdaptiveRulesPolicy
    HAS_UNIFIED_RL = True
    logger.info("✅ Using Unified RL Policy (single system for entry/exit)")
except ImportError:
    HAS_UNIFIED_RL = False
    logger.warning("⚠️ Unified RL Policy not available, falling back to legacy")
    
# Legacy RL (fallback if unified not available)
from backend.rl_trading_policy import RLTradingPolicy, TradingAction

# Liquidity-aware execution layer
try:
    from execution.liquidity_exec import LiquidityExecutor, LiquidityRules, OrderIntent
    from execution.tradier_adapter import TradierAdapter
    from backend.tradier_trading_system import TradierTradingSystem
    HAS_LIQUIDITY_EXEC = True
except ImportError:
    HAS_LIQUIDITY_EXEC = False
    logger.warning("⚠️ Liquidity execution layer not available")

# Bot Reporter - Reports trades and metrics to Data Manager
try:
    from backend.bot_reporter import BotReporter, init_reporter, get_reporter
    HAS_BOT_REPORTER = True
except ImportError:
    HAS_BOT_REPORTER = False
    BotReporter = None
    init_reporter = None
    get_reporter = None
    logger.info("📊 Bot Reporter not available (optional)")

# Multi-Dimensional HMM Market Regime Detection
try:
    from backend.multi_dimensional_hmm import MultiDimensionalHMM, add_multid_regime_to_features
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    logger.warning("⚠️ Multi-Dimensional HMM not available - install: pip install hmmlearn")

# Trading configuration and credentials
try:
    from tradier_credentials import TradierCredentials
    from trading_mode_config import TradingModeConfig
    HAS_CONFIG_SYSTEM = True
except ImportError:
    HAS_CONFIG_SYSTEM = False
    # Create a dummy class if not available
    class TradingModeConfig:
        MODE_SIMULATION = 'SIMULATION'
        MODE_PAPER_TRADING = 'PAPER_TRADING'
        MODE_LIVE_TRADIER = 'LIVE_TRADIER'

# Optional system monitoring imports
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    
try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False


# ------------------------------------------------------------------------------
# Neural Network Components (extracted to bot_modules/neural_networks.py)
# ------------------------------------------------------------------------------
# The following classes have been extracted to modular files:
#
# - BayesianLinear: bot_modules/neural_networks.py
# - RBFKernelLayer: bot_modules/neural_networks.py
# - TCNBlock: bot_modules/neural_networks.py
# - OptionsTCN: bot_modules/neural_networks.py
# - OptionsLSTM: bot_modules/neural_networks.py
# - ResidualBlock: bot_modules/neural_networks.py
# - UnifiedOptionsPredictor: bot_modules/neural_networks.py
# - GaussianKernelProcessor: bot_modules/gaussian_processor.py
#
# Import them from bot_modules:
#   from bot_modules.neural_networks import UnifiedOptionsPredictor, BayesianLinear
#   from bot_modules.gaussian_processor import GaussianKernelProcessor
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# Bot
# ------------------------------------------------------------------------------
class UnifiedOptionsBot:
    def _initialize_trading_system(self, initial_balance: float):
        """
        Initialize trading system based on config:
        - SIMULATION: In-memory + Tradier sandbox if available
        - LIVE_TRADIER: Real Tradier API with live account
        
        Returns:
            (paper_trading_system, tradier_system) tuple
        """
        # Disable PDT protection - let Tradier enforce it, not our paper trading
        trading_system = PaperTradingSystem(initial_balance, disable_pdt=True)
        tradier_system = None
        
        if not HAS_CONFIG_SYSTEM:
            return trading_system, tradier_system
        
        try:
            config = TradingModeConfig()
            creds = TradierCredentials()
            mode = config.get_mode()
            
            logging.info(f"🎯 Trading Mode: {mode}")
            
            # SIMULATION: Use in-memory with sandbox credentials if available
            if mode == TradingModeConfig.MODE_SIMULATION:
                if creds.has_sandbox_credentials():
                    logging.info("📊 Using Tradier SANDBOX for simulated trades (real data)")
                    account_id, api_token = creds.get_sandbox_credentials()
                    if account_id and api_token and HAS_LIQUIDITY_EXEC:
                        tradier_system = TradierTradingSystem(
                            account_id=account_id,
                            api_token=api_token,
                            sandbox=True,
                            initial_balance=initial_balance
                        )
                else:
                    logging.info("📊 Using IN-MEMORY simulation (no Tradier credentials)")
            
            # PAPER_TRADING: Same as simulation for now
            elif mode == TradingModeConfig.MODE_PAPER_TRADING:
                logging.info("📋 Paper trading mode (in-memory simulation)")
            
            # LIVE_TRADIER: Check for live credentials
            elif mode == TradingModeConfig.MODE_LIVE_TRADIER:
                if creds.has_live_credentials():
                    logging.warning("🔴 LIVE TRADIER MODE - REAL MONEY!")
                    logging.warning("🔴 Using real Tradier account for trades!")
                    account_id, api_token = creds.get_live_credentials()
                    if account_id and api_token and HAS_LIQUIDITY_EXEC:
                        tradier_system = TradierTradingSystem(
                            account_id=account_id,
                            api_token=api_token,
                            sandbox=False,
                            initial_balance=initial_balance
                        )
                else:
                    logging.warning("⚠️ Live mode selected but no credentials - falling back to simulation")
        
        except Exception as e:
            logging.warning(f"⚠️ Could not initialize config system: {e}")
        
        # Connect paper trader to Tradier system for live trade execution
        if tradier_system:
            trading_system.tradier_trading_system = tradier_system
            logging.info("🔗 Paper trader connected to Tradier system for live position management")
        
        return trading_system, tradier_system
    
    def __init__(self, initial_balance: float = 1000.0, config=None):
        """
        Initialize UnifiedOptionsBot.
        
        Args:
            initial_balance: Starting account balance
            config: Optional BotConfig object for configuration
        """
        # Store config and convert to dict for DataManager
        self.config = config
        config_dict = config.config if config else {}
        
        # Read historical period from config (default to 7d for better predictions)
        self.historical_period = config_dict.get('data_sources', {}).get('historical_period', '7d')
        
        # Initialize data source with config
        self.data_source = EnhancedDataSource(config=config_dict)
        self.paper_trader, self.tradier_system = self._initialize_trading_system(initial_balance)
        
        # --- APPLY CONFIG TO PAPER TRADER ---
        if config_dict:
            risk_config = config_dict.get('risk_management', {})
            if 'stop_loss_pct' in risk_config:
                self.paper_trader.stop_loss_pct = float(risk_config['stop_loss_pct'])
                logger.info(f"✅ Set stop loss to {self.paper_trader.stop_loss_pct:.1%}")
            
            if 'take_profit_pct' in risk_config:
                self.paper_trader.take_profit_pct = float(risk_config['take_profit_pct'])
                logger.info(f"✅ Set take profit to {self.paper_trader.take_profit_pct:.1%}")
                
            if 'max_positions' in config_dict.get('trading', {}):
                self.paper_trader.max_positions = int(config_dict['trading']['max_positions'])
                logger.info(f"✅ Set max positions to {self.paper_trader.max_positions}")
        
        # Strategy feature toggles (default OFF for safety)
        trading_cfg = config_dict.get('trading', {}) if config_dict else {}
        self.enable_straddles = bool(trading_cfg.get('enable_straddles', False))
        self.enable_premium_selling = bool(trading_cfg.get('enable_premium_selling', False))
        logger.info(f"🧩 Strategy toggles: straddles={'ON' if self.enable_straddles else 'OFF'}, "
                    f"premium_selling={'ON' if self.enable_premium_selling else 'OFF'}")
        
        # Connect paper trader to bot for live prediction access (dynamic hold time)
        self.paper_trader.set_bot_reference(self)
        
        # Initialize liquidity-aware execution layer
        self.liquidity_exec = None
        self.tradier_adapter = None
        if HAS_LIQUIDITY_EXEC and self.tradier_system:
            self.tradier_adapter = TradierAdapter(self.tradier_system)
            self.liquidity_exec = LiquidityExecutor(
                self.tradier_adapter,
                LiquidityRules(
                    min_oi=100,              # Minimum open interest
                    min_day_volume=150,      # Minimum daily volume
                    max_spread_pct=4.0,      # Max 4% bid-ask spread
                    target_delta=0.30,       # Target 0.30 delta
                    delta_band=0.08,         # ±0.08 delta band
                    min_quote_size=10,       # Minimum 10 contracts at bid/ask
                    max_work_time_sec=20,    # Work order for max 20 seconds
                    price_tick=0.01,         # $0.01 price ladder
                    max_price_lifts=4,       # Max 4 price improvements
                    ioc_probe_size=1,        # 1-contract IOC probe
                    cancel_replace_delay=2.0 # 2 second delay between price lifts
                ),
            )
            logger.info("🎯 Liquidity-aware execution layer initialized")
            logger.info("   - Contract screening by OI, volume, spread, delta")
            logger.info("   - Midpoint-peg orders with price ladder")
            logger.info("   - Automatic fallback to vertical spreads")
        else:
            if not HAS_LIQUIDITY_EXEC:
                logger.info("📊 Liquidity execution layer not available (module not imported)")
            else:
                logger.info("📊 No Tradier connection - liquidity execution disabled")

        # Initialize Bot Reporter for Data Manager integration
        self.bot_reporter = None
        if HAS_BOT_REPORTER and config_dict.get('bot_reporter', {}).get('enabled', False):
            try:
                self.bot_reporter = init_reporter(config_dict)
                if self.bot_reporter and self.bot_reporter.enabled:
                    # Register bot with Data Manager
                    if self.bot_reporter.register():
                        logger.info("📡 Bot Reporter initialized and registered with Data Manager")
                    else:
                        logger.warning("📡 Bot Reporter initialized but registration failed")
                else:
                    logger.info("📡 Bot Reporter not enabled (check data_manager config)")
            except Exception as e:
                logger.warning(f"📡 Bot Reporter initialization failed: {e}")
                self.bot_reporter = None
        else:
            logger.info("📡 Bot Reporter not configured (optional)")

        self.market_tz = pytz.timezone("US/Eastern")
        
        # Symbol-specific market hours (all times in Eastern)
        self.market_hours = {
            "BITX": {
                "open": datetime.strptime("09:30", "%H:%M").time(),
                "close": datetime.strptime("16:00", "%H:%M").time()
            },
            "VIX": {
                "open": datetime.strptime("08:30", "%H:%M").time(),  # VIX futures/options start earlier
                "close": datetime.strptime("16:15", "%H:%M").time()   # VIX closes slightly later
            }
        }
        
        # Default market hours for general market open check
        self.market_open = datetime.strptime("09:30", "%H:%M").time()
        self.market_close = datetime.strptime("16:00", "%H:%M").time()

        self.feature_dim = 50  # RESTORED: Match profitable run config (no time/gaussian features)
        self.sequence_length = 60  # Increased from 30 to 60 (1 hour lookback for better patterns)
        self.model: Optional[UnifiedOptionsPredictor] = None
        self.model_initialized = False                  # OK: lazy init flag
        self.optimizer = None
        self.loss_fn = nn.MSELoss()
        self.learning_rate = 0.0003  # INCREASED: Faster adaptation during training
        self.training_enabled = True
        
        # Mixed Precision Training (AMP) for GPU acceleration
        self.use_amp = torch.cuda.is_available()  # Only use AMP when GPU is available
        self.grad_scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

        # Model persistence (now organized in models/state/)
        self.model_save_path = "models/state/trained_model.pth"
        self.optimizer_save_path = "models/state/optimizer_state.pth"
        self.feature_buffer_save_path = "models/state/feature_buffer.pkl"
        self.gaussian_processor_save_path = "models/state/gaussian_processor.pkl"
        self.learning_state_save_path = "models/state/learning_state.pkl"
        self.prediction_history_save_path = "models/state/prediction_history.pkl"

        self.gaussian_processor = GaussianKernelProcessor()
        self.feature_buffer: List[np.ndarray] = []
        self.max_buffer_size = 500
        self.prediction_history: List[Dict] = []
        
        # Market context: store related symbols for cross-asset features
        # These are used for correlation, relative strength, and market breadth features
        self.market_context: Dict[str, pd.DataFrame] = {}  # symbol -> DataFrame
        self.market_context_prices: Dict[str, float] = {}  # symbol -> latest price
        
        # Read context symbols from config.data_fetching.symbols
        # Excludes primary trading symbol (SPY) and VIX which are handled separately
        all_symbols = config_dict.get('data_fetching', {}).get('symbols', ['SPY', '^VIX', 'UUP'])
        trading_symbol = config_dict.get('trading', {}).get('symbol', 'SPY')
        self.context_symbols = [s for s in all_symbols if s not in [trading_symbol, '^VIX', 'VIX']]
        logger.info(f"[CONTEXT] Tracking symbols from config: {self.context_symbols}")
        
        # Multi-Dimensional Hidden Markov Model for market regime detection
        # Tracks 3 dimensions: Trend (3 states) × Volatility (3 states) × Liquidity (3 states) = 27 regimes
        self.hmm_regime_detector = None
        self.last_hmm_train_time = None
        self.hmm_train_interval_hours = 24  # Retrain daily
        self.hmm_min_samples = 500  # Minimum samples needed to train
        self.current_hmm_regime = None  # Track current HMM state for exit decisions
        self.last_multi_timeframe_predictions = {}  # Store for paper trader's dynamic hold time
        if HMM_AVAILABLE:
            self.hmm_regime_detector = MultiDimensionalHMM(lookback_window=50)
            # Try to load pre-trained model
            if self.hmm_regime_detector.load():
                logger.info("🎰 Multi-Dimensional HMM model loaded from disk (3×3×3 = 27 regimes)")
                self.last_hmm_train_time = datetime.now()
            else:
                logger.info("🎰 Multi-Dimensional HMM initialized - will auto-train when enough data available")
        
        # Experience replay buffer for better learning
        self.experience_buffer: List[Dict] = []
        self.max_experience_size = 500  # Store more experiences for replay
        
        # =============================================================================
        # CALIBRATION: Clean Platt scaling via CalibrationTracker
        # =============================================================================
        # PRIMARY_HORIZON: The single horizon used for calibration and gating
        # This ensures consistent calibration - don't mix different horizons!
        self.PRIMARY_HORIZON_MINUTES = 15  # 15-minute predictions are our primary target
        
        # Initialize dedicated CalibrationTracker for Platt scaling
        try:
            from backend.calibration_tracker import CalibrationTracker
            self.calibration_tracker = CalibrationTracker(
                horizon_minutes=self.PRIMARY_HORIZON_MINUTES,
                buffer_size=1000,
                min_samples_for_calibration=50,
                refit_interval=50
            )
            logger.info(f"✅ CalibrationTracker initialized ({self.PRIMARY_HORIZON_MINUTES}m horizon)")
        except ImportError as e:
            logger.warning(f"CalibrationTracker not available: {e}")
            self.calibration_tracker = None
        
        # Legacy accuracy_tracker kept for monitoring/logging only
        self.accuracy_tracker = {
            'direction_correct': 0,
            'direction_total': 0,
            'confidence_calibration': [],  # Backup buffer (legacy)
            'recent_errors': [],  # Return errors for monitoring
            'horizon_stats': {
                5: {'correct': 0, 'total': 0},
                15: {'correct': 0, 'total': 0},
                30: {'correct': 0, 'total': 0},
                60: {'correct': 0, 'total': 0},
            }
        }
        self.CALIBRATION_BUFFER_SIZE = 1000  # For legacy buffer
        
        # System monitoring
        self.system_monitor = {
            'cpu_percent': 0.0,
            'memory_percent': 0.0,
            'memory_used_gb': 0.0,
            'memory_total_gb': 0.0,
            'gpu_percent': 0.0,
            'gpu_memory_percent': 0.0,
            'gpu_memory_used_gb': 0.0,
            'gpu_memory_total_gb': 0.0,
            'gpu_available': False,
            'process_cpu_percent': 0.0,
            'process_memory_mb': 0.0,
            'last_update': self.get_market_time()
        }
        
        # Start system monitoring thread (if packages available)
        if PSUTIL_AVAILABLE:
            self.monitoring_thread = threading.Thread(target=self._monitor_system_resources, daemon=True)
            self.monitoring_thread.start()
            logger.info("🖥️ System monitoring started")
        else:
            logger.warning("⚠️ System monitoring disabled (psutil not available)")

        # Try to load feature buffer on startup
        self.load_feature_buffer_only()

        # Learning system
        self.learning_enabled = True
        self.learning_batch_size = 5
        self.last_learning_time = self.get_market_time()
        self.learning_interval = 30  # seconds between learning cycles
        
        # Data change tracking - only learn when NEW data arrives
        self.last_bitx_timestamp = None
        self.last_vix_timestamp = None
        
        # Adaptive confidence system (reset for $1000 account)
        self.base_confidence_threshold = 0.45  # Higher starting threshold for smaller account
        self.target_confidence_threshold = 0.70  # Higher target for risk management
        self.confidence_adaptation_rate = 0.002  # Faster adaptation for small account
        self.min_confidence_threshold = self.base_confidence_threshold
        self.confidence_history = []  # Track confidence over time
        self.trades_since_last_adjustment = 0
        self.adjustment_interval = 5  # Adjust more frequently with smaller account
        
        # No-trade activity detector (NEW!)
        self.signals_since_last_trade = 0
        self.no_trade_threshold = 50  # After 50 signals with no trades, force threshold down
        self.last_trade_count = 0
        
        # Losing streak detector - pause trading after consecutive losses
        self.consecutive_losses = 0
        self.max_consecutive_losses = 5  # Pause after 5 consecutive losses
        self.loss_cooldown_cycles = 0  # Cycles to skip before resuming
        self.loss_cooldown_duration = 20  # Skip 20 cycles after losing streak
        
        # Performance-based confidence tracking (NEW!)
        self.confidence_brackets = {
            '20-30': {'trades': [], 'pnl': []},
            '30-40': {'trades': [], 'pnl': []},
            '40-50': {'trades': [], 'pnl': []},
            '50-60': {'trades': [], 'pnl': []},
            '60-70': {'trades': [], 'pnl': []},
            '70-80': {'trades': [], 'pnl': []},
            '80+': {'trades': [], 'pnl': []}
        }
        self.confidence_bracket_window = 30  # Keep last 30 trades per bracket
        
        # Adaptive signal generation thresholds (AUTO-TUNE from data every 100 signals)
        # LOWERED for training: Let RL decide what to trade, not hard thresholds
        self.neural_confidence_threshold = 0.10  # Very low - let RL learn from more signals
        self.neural_return_threshold = 0.0001  # 0.01% - capture smaller moves (predictions are often 0.0002-0.0005)
        self.momentum_threshold = 0.0005  # 0.05% - very sensitive to momentum  
        self.volume_threshold = 0.5  # Volume spike threshold (will auto-tune from data)
        self.signal_performance_history = []  # Track signal accuracy for adaptation
        self.signals_since_last_tune = 0  # Counter for auto-tuning
        self.auto_tune_interval = 100  # Auto-tune thresholds every N signals
        
        # Multi-timeframe prediction system with ADAPTIVE WEIGHTS
        # Weights will be learned based on which timeframes are most accurate!
        self.adaptive_weights = AdaptiveTimeframeWeights()
        
        # Try to load saved weights
        try:
            self.adaptive_weights.load('models/adaptive_timeframe_weights.json')
        except FileNotFoundError:
            logger.info("Starting with default timeframe weights (will learn from data)")
        except Exception as e:
            logger.warning(f"Could not load adaptive weights: {e}. Using defaults.")
        
        # Get current weights (learned or default)
        learned_weights = self.adaptive_weights.get_weights()
        
        # Multi-timeframe predictions: GRANULAR for first 2 hours (more accurate)
        # Short-term predictions (5-30min) are most reliable
        # 10-minute intervals for first 2 hours as requested
        self.prediction_timeframes = {
            "5min": {"minutes": 5, "weight": learned_weights.get('5min', 0.20)},      # Most reliable
            "10min": {"minutes": 10, "weight": learned_weights.get('10min', 0.18)},
            "15min": {"minutes": 15, "weight": learned_weights.get('15min', 0.15)},
            "20min": {"minutes": 20, "weight": learned_weights.get('20min', 0.12)},
            "30min": {"minutes": 30, "weight": learned_weights.get('30min', 0.10)},
            "40min": {"minutes": 40, "weight": learned_weights.get('40min', 0.07)},
            "50min": {"minutes": 50, "weight": learned_weights.get('50min', 0.05)},
            "1h": {"minutes": 60, "weight": learned_weights.get('1h', 0.05)},
            "1h20m": {"minutes": 80, "weight": learned_weights.get('1h20m', 0.03)},
            "1h40m": {"minutes": 100, "weight": learned_weights.get('1h40m', 0.03)},
            "2h": {"minutes": 120, "weight": learned_weights.get('2h', 0.02)}
        }
        # NOTE: Removed 4hr, 12hr, 24hr, 48hr - too far out to be accurate!
        self.multi_timeframe_predictions = {}  # Store predictions for each timeframe
        
        # Prediction trend tracking buffer (NEW!)
        # Stores recent consensus predictions to calculate stability and momentum
        from collections import deque
        self.consensus_history = deque(maxlen=20)  # Keep last 20 consensus predictions (approx 20 mins)
        
        # Aggregated predictions indexed by TARGET time (for overlapping prediction consensus)
        # Key: target timestamp (rounded to minute), Value: list of predictions targeting that time
        self.predictions_by_target_time: Dict[str, List[Dict]] = {}
        self.max_target_time_predictions = 500  # Max predictions to keep in memory
        
        logger.info(f"📊 Timeframe weights (learned from accuracy):")
        for tf, config in self.prediction_timeframes.items():
            logger.info(f"   {tf}: {config['weight']:.1%} ({config['minutes']} minutes)")
        
        self.predictions_since_weight_update = 0
        self.weight_update_interval = 100  # Update weights every 100 prediction cycles
        
        self.max_positions = 5  # Maximum concurrent positions (increased from 2)
        self.risk_per_trade = 0.05  # 5% risk per trade (higher % but lower $ amount)
        self.last_signal_time: Optional[datetime] = None
        self.signal_interval = 30  # 30 seconds to check for new 1-minute bars (data-driven)
        self.min_time_between_trades = 300  # 5 minutes minimum between trades (fast but not crazy)
        
        # Data freshness settings for learning (realistic for market conditions)
        self.max_data_age_minutes = 5  # Maximum age for data to be considered fresh (realistic for live trading)
        self.learning_requires_fresh_data = True  # CRITICAL: NEVER train on stale data - only fresh data!

        # Time machine / simulation mode support
        self.simulation_mode = False  # Set to True when running simulations/time machine
        self.simulated_time = None  # MUST be set when simulation_mode=True
        self.disable_live_data_fetch = False  # Set to True in simulation to prevent API calls

        # Optimization: GPU support with robust fallback
        self.device = self._initialize_device()
        logger.info(f"🖥️  Using device: {self.device}")
        
        # RL Trading Policy - UNIFIED (single policy for entry + exit)
        self.use_rl_policy = True  # ENABLED: Using unified policy
        self.use_unified_rl = HAS_UNIFIED_RL  # Use new unified system if available
        
        if self.use_unified_rl:
            # NEW: Unified RL Policy (replaces 3 separate systems)
            # OPTIMIZED PARAMETERS for higher win rate:
            self.unified_rl = UnifiedRLPolicy(
                learning_rate=0.0003,
                gamma=0.99,
                device=str(self.device),
                bandit_mode_trades=50,  # REDUCED: Fewer random trades before learning kicks in
                min_confidence_to_trade=0.55,  # RAISED from 0.35: Only trade high-quality signals
                max_position_loss_pct=0.08,  # TIGHTER: 8% stop loss (was 15%)
                profit_target_pct=0.15,  # FASTER exits: 15% profit target (was 25%)
                trailing_stop_activation=0.08,  # NEW: Activate trailing stop at 8% profit
                trailing_stop_distance=0.04,  # NEW: Trail by 4%
            )
            self.unified_rl_path = "models/unified_rl_policy.pth"
            
            # Also create adaptive rules policy as backup
            self.rules_policy = AdaptiveRulesPolicy()
            
            # Load unified policy if exists
            if os.path.exists(self.unified_rl_path):
                self.unified_rl.load(self.unified_rl_path)
            else:
                logger.info("🎯 Unified RL Policy: Starting fresh (no saved policy found)")
            
            # Legacy compatibility (keep old policy as backup)
            self.rl_policy = None  # Not used when unified is active
            logger.info("🚀 Using UNIFIED RL Policy (single system for entry + exit)")
        else:
            # Fallback to legacy separate policies
            self.unified_rl = None
            self.rules_policy = None
            self.rl_policy = RLTradingPolicy(
                state_dim=32,
                action_dim=5,
                learning_rate=3e-4,
                device=str(self.device)
            )
            self.rl_policy_path = "models/rl_trading_policy.pth"
            if os.path.exists(self.rl_policy_path):
                self.rl_policy.load(self.rl_policy_path)
            else:
                logger.info("🎯 Legacy RL Policy: Starting fresh (no saved policy found)")
        
        self.rl_temperature = 1.0  # Exploration temperature
        self.rl_last_state = None
        self.rl_last_action = None
        self.rl_episode_start_balance = initial_balance
        
        # Track current trade state for unified RL
        self.current_trade_state: Optional[TradeState] = None if not self.use_unified_rl else TradeState()
        
        # Force model initialization if we have enough saved features (after device is set)
        if len(self.feature_buffer) >= self.sequence_length and not self.model_initialized:
            logger.info(f"🧠 Force-initializing model with {len(self.feature_buffer)} saved features...")
            self._maybe_init_model()
        
        # State machine for market open/close
        self.market_state = 'CLOSED' # Start in a known state
        self.end_of_day_processed = False # Flag to run cleanup tasks only once
        self.last_close_price = None # Initialize the closing price attribute

        self.init_database()
        self.init_learning_database()
        
        # Check cache health on startup
        if not self.check_cache_health():
            logger.info("🔄 Cache will be rebuilt with fresh data")

    # ------------------------ Utilities ------------------------
    def safe_get_data(self, symbol: str, period: str = "1d", interval: str = "1m"):
        """
        Safely get data - skips API calls if in simulation with disable_live_data_fetch=True
        
        Returns empty DataFrame if data fetch is disabled to prevent API rate limiting.
        """
        if hasattr(self, 'disable_live_data_fetch') and self.disable_live_data_fetch:
            logger.debug(f"⏭️ Skipping live data fetch for {symbol} (simulation mode)")
            return pd.DataFrame()
        
        return self.data_source.get_data(symbol, period=period, interval=interval)
    
    def health_check(self) -> Dict[str, Any]:
        """
        Comprehensive health check for the bot
        Returns dict with status of all major components
        """
        health = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'checks': {}
        }
        
        # Model checks
        health['checks']['model_initialized'] = getattr(self, 'model_initialized', False)
        health['checks']['model_exists'] = self.model is not None if hasattr(self, 'model') else False
        health['checks']['optimizer_exists'] = self.optimizer is not None if hasattr(self, 'optimizer') else False
        
        # Feature buffer
        if hasattr(self, 'feature_buffer'):
            buffer_ready = len(self.feature_buffer) >= self.sequence_length
            health['checks']['feature_buffer_ready'] = buffer_ready
            health['checks']['feature_buffer_size'] = len(self.feature_buffer)
        else:
            health['checks']['feature_buffer_ready'] = False
            health['checks']['feature_buffer_size'] = 0
        
        # RL Policy
        if getattr(self, 'use_rl_policy', False):
            health['checks']['rl_enabled'] = True
            health['checks']['rl_policy_exists'] = self.rl_policy is not None
            if self.rl_policy:
                health['checks']['rl_episodes_trained'] = len(self.rl_policy.experience_buffer)
                health['checks']['rl_total_updates'] = self.rl_policy.training_stats.get('total_updates', 0)
        else:
            health['checks']['rl_enabled'] = False
        
        # Simulation mode
        if hasattr(self, 'simulation_mode'):
            health['checks']['simulation_mode'] = self.simulation_mode
            if self.simulation_mode:
                health['checks']['simulated_time_set'] = self.simulated_time is not None
        
        # Data source
        health['checks']['data_source_available'] = self.data_source is not None
        
        # Paper trading
        health['checks']['paper_trader_available'] = self.paper_trader is not None
        if self.paper_trader:
            try:
                account = self.paper_trader.get_account_summary()
                health['checks']['paper_trader_balance'] = account.get('current_balance', 0)
                health['checks']['paper_trader_positions'] = len(self.paper_trader.active_trades)
            except Exception as e:
                logger.debug(f"Paper trader health check failed: {e}")
                health['checks']['paper_trader_balance'] = 'error'
                health['checks']['paper_trader_positions'] = 'error'
        
        # Overall status
        critical_checks = [
            health['checks'].get('model_initialized', False),
            health['checks'].get('data_source_available', False),
            health['checks'].get('paper_trader_available', False)
        ]
        health['overall_healthy'] = all(critical_checks)
        
        return health
    
    def print_health_status(self):
        """Print formatted health check"""
        health = self.health_check()
        
        logger.info("="*60)
        logger.info("  🏥 BOT HEALTH CHECK")
        logger.info("="*60)
        
        for check_name, status in health['checks'].items():
            # Format check name
            name = check_name.replace('_', ' ').title()
            
            # Determine icon
            if isinstance(status, bool):
                icon = "✅" if status else "❌"
                status_text = "OK" if status else "FAIL"
            elif isinstance(status, (int, float)):
                icon = "📊"
                status_text = str(status)
            else:
                icon = "ℹ️"
                status_text = str(status)
            
            logger.info(f"  {icon} {name}: {status_text}")
        
        logger.info("="*60)
        overall = "✅ HEALTHY" if health['overall_healthy'] else "❌ ISSUES DETECTED"
        logger.info(f"  Overall Status: {overall}")
        logger.info("="*60)
        
        return health['overall_healthy']
    
    def _initialize_device(self):
        """
        Initialize PyTorch device with robust fallback to CPU
        
        Tries to use CUDA GPU if available, but gracefully falls back to CPU if:
        - CUDA is not available
        - CUDA initialization fails
        - GPU runs out of memory
        - Any other GPU-related error
        
        Returns:
            torch.device: Either 'cuda' or 'cpu'
        """
        try:
            # Try to use CUDA if available
            if torch.cuda.is_available():
                # Verify we can actually use the GPU by creating a small tensor
                try:
                    test_tensor = torch.zeros(1, device='cuda')
                    device_name = torch.cuda.get_device_name(0)
                    logger.info(f"✅ GPU detected: {device_name}")
                    logger.info(f"✅ CUDA version: {torch.version.cuda}")
                    logger.info(f"✅ PyTorch will use GPU acceleration")
                    del test_tensor  # Clean up
                    torch.cuda.empty_cache()
                    return torch.device("cuda")
                except RuntimeError as e:
                    logger.warning(f"⚠️  GPU detected but initialization failed: {e}")
                    logger.warning(f"⚠️  Falling back to CPU")
                    return torch.device("cpu")
            else:
                logger.info(f"ℹ️  No GPU detected, using CPU")
                logger.info(f"ℹ️  To enable GPU: Install CUDA-enabled PyTorch")
                return torch.device("cpu")
        except Exception as e:
            # Catch any other unexpected errors
            logger.warning(f"⚠️  Error during device initialization: {e}")
            logger.warning(f"⚠️  Falling back to CPU")
            return torch.device("cpu")
    
    def _safe_to_device(self, tensor_or_model, fallback_to_cpu=True):
        """
        Safely move tensor or model to device with automatic fallback
        
        Args:
            tensor_or_model: PyTorch tensor or model to move
            fallback_to_cpu: If True, falls back to CPU on error
            
        Returns:
            Tensor or model on the appropriate device
        """
        try:
            return tensor_or_model.to(self.device)
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and fallback_to_cpu:
                logger.warning(f"⚠️  GPU out of memory, falling back to CPU for this operation")
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                # Switch to CPU permanently if OOM happens
                self.device = torch.device("cpu")
                logger.warning(f"⚠️  Permanently switched to CPU mode")
                return tensor_or_model.to(self.device)
            else:
                raise
    
    def get_market_time(self) -> datetime:
        """
        Get current time in market timezone as naive datetime
        
        SIMULATION MODE: Must use simulated_time (never real time)
        LIVE MODE: Uses actual current time
        """
        # Check if we're in simulation mode
        if hasattr(self, 'simulation_mode') and self.simulation_mode:
            # SIMULATION MODE: simulated_time MUST be set
            if not hasattr(self, 'simulated_time') or self.simulated_time is None:
                raise RuntimeError(
                    "❌ SIMULATION MODE ERROR: simulated_time not set!\n"
                    "   You must set bot.simulated_time = timestamp before each cycle.\n"
                    "   Example: bot.simulated_time = historical_timestamp"
                )
            
            # Return simulated time
            if self.simulated_time.tzinfo is None:
                # Assume it's already in market timezone
                return self.simulated_time
            else:
                # Convert to market timezone
                return self.simulated_time.astimezone(self.market_tz).replace(tzinfo=None)
        
        # LIVE MODE: use actual current time
        market_dt = pytz.UTC.localize(datetime.utcnow()).astimezone(self.market_tz)
        return market_dt.replace(tzinfo=None)

    def is_market_open(self, symbol: str = "BITX") -> bool:
        """Check if the market is open for a specific symbol"""
        t = self.get_market_time()
        
        # Weekend check first
        if t.weekday() >= 5:
            return False
            
        # Get symbol-specific hours, fallback to default
        if symbol in self.market_hours:
            open_time = self.market_hours[symbol]["open"]
            close_time = self.market_hours[symbol]["close"]
        else:
            open_time = self.market_open
            close_time = self.market_close
            
        return open_time <= t.time() <= close_time
    
    def is_symbol_trading(self, symbol: str) -> bool:
        """Check if a specific symbol is currently trading"""
        return self.is_market_open(symbol)
    
    def check_cache_health(self) -> bool:
        """Check if the cache database is healthy and accessible"""
        try:
            import sqlite3
            conn = sqlite3.connect('data/market_data_cache.db')
            cursor = conn.cursor()
            
            # Check if tables exist
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            
            if not tables:
                logger.warning("Cache database is empty - will rebuild automatically")
                conn.close()
                return False
                
            # Check if we can read recent data
            cursor.execute("SELECT COUNT(*) FROM market_data_cache WHERE created_at > datetime('now', '-1 day')")
            recent_count = cursor.fetchone()[0]
            
            conn.close()
            logger.info(f"✅ Cache health check passed - {recent_count} recent entries")
            return True
            
        except Exception as e:
            logger.warning(f"⚠️ Cache health check failed: {e}")
            return False

    def _annualization_scale(self, interval: str) -> float:
        # 252 trading days * 390 minutes/day = steps per year (in minutes)
        minutes_per_year = 252 * 390
        step = {"1m": 1, "5m": 5, "15m": 15}.get(interval, 1)
        return np.sqrt(minutes_per_year / step)

    def init_database(self):
        conn = sqlite3.connect("data/unified_options_bot.db")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS unified_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                symbol TEXT,
                current_price REAL,

                -- Snapshot of the model input features at decision time.
                -- This is CRITICAL for any online/continuous learning to avoid
                -- training on "current" (future) features when backfilling labels.
                features_blob BLOB,
                -- Snapshot of the full temporal sequence [T, D] used for the model at decision time.
                -- Without this, online training can accidentally pair a past label with a *current*
                -- sequence window (a serious training corruption for TCN/LSTM encoders).
                sequence_blob BLOB,

                volatility_percentile REAL,
                momentum_5min REAL,
                momentum_15min REAL,
                volume_spike REAL,
                price_position REAL,

                predicted_return REAL,
                predicted_volatility REAL,
                direction_prob TEXT,
                temporal_confidence REAL,
                lstm_context TEXT,

                action TEXT,
                strategy TEXT,
                confidence REAL,
                reasoning TEXT,

                executed BOOLEAN,
                position_size REAL,
                entry_price REAL,
                exit_price REAL,
                pnl REAL,

                predicted_price_15min REAL,
                prediction_time TEXT,
                
                pred_15min_return REAL,
                pred_15min_confidence REAL,
                pred_30min_return REAL,
                pred_30min_confidence REAL,
                pred_1hour_return REAL,
                pred_1hour_confidence REAL,
                pred_4hour_return REAL,
                pred_4hour_confidence REAL,
                pred_12hour_return REAL,
                pred_12hour_confidence REAL,
                pred_24hour_return REAL,
                pred_24hour_confidence REAL,
                pred_48hour_return REAL,
                pred_48hour_confidence REAL,
                multi_timeframe_direction TEXT,
                multi_timeframe_agreement REAL,
                multi_timeframe_strength REAL
            )
            """
        )
        
        # Add new columns to existing table if they don't exist
        cursor = conn.cursor()
        new_columns = [
            ("features_blob", "BLOB"),
            ("sequence_blob", "BLOB"),
            ("pred_15min_return", "REAL"),
            ("pred_15min_confidence", "REAL"),
            ("pred_30min_return", "REAL"),
            ("pred_30min_confidence", "REAL"),
            ("pred_1hour_return", "REAL"),
            ("pred_1hour_confidence", "REAL"),
            ("pred_4hour_return", "REAL"),
            ("pred_4hour_confidence", "REAL"),
            ("pred_12hour_return", "REAL"),
            ("pred_12hour_confidence", "REAL"),
            ("pred_24hour_return", "REAL"),
            ("pred_24hour_confidence", "REAL"),
            ("pred_48hour_return", "REAL"),
            ("pred_48hour_confidence", "REAL"),
            ("multi_timeframe_direction", "TEXT"),
            ("multi_timeframe_agreement", "REAL"),
            ("multi_timeframe_strength", "REAL")
        ]
        
        for column_name, column_type in new_columns:
            try:
                cursor.execute(f"ALTER TABLE unified_signals ADD COLUMN {column_name} {column_type}")
            except sqlite3.OperationalError:
                pass  # Column already exists
                
        conn.commit()
        conn.close()
        logger.info("📊 Unified signals database initialized with multi-timeframe support")
        
        # Initialize execution outcomes table for fillability learning
        self._init_execution_outcomes_table()
    
    def _init_execution_outcomes_table(self):
        """Initialize table for logging execution outcomes to train fillability predictor"""
        conn = sqlite3.connect("data/unified_options_bot.db")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS exec_outcomes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT,
                occ_symbol TEXT,
                symbol TEXT,
                side TEXT,
                limit_price REAL,
                worked_secs INTEGER,
                filled_qty INTEGER,
                desired_qty INTEGER,
                avg_fill_price REAL,
                best_bid REAL,
                best_ask REAL,
                features BLOB,        -- pickled np.ndarray from build_liquidity_features
                label_fill INTEGER,   -- 1 if filled_qty==desired_qty within T, else 0
                label_slip REAL,      -- (avg_fill_price - limit_price) * sign
                label_ttf REAL        -- seconds to full fill (cap if not filled)
            )
        """)
        conn.commit()
        conn.close()
        logger.info("🎯 Execution outcomes table initialized for fillability learning")

    def init_learning_database(self):
        """Initialize separate database for learning trades (simulated trades during off-hours)"""
        conn = sqlite3.connect("data/learning_trades.db")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS learning_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                symbol TEXT,
                current_price REAL,
                
                signal_action TEXT,
                signal_confidence REAL,
                signal_strategy TEXT,
                signal_reasoning TEXT,
                
                simulated_entry_price REAL,
                simulated_exit_price REAL,
                simulated_pnl REAL,
                simulated_duration_minutes INTEGER,
                
                market_hours BOOLEAN,
                learning_mode TEXT
            )
            """
        )
        conn.commit()
        conn.close()
        logger.info("📚 Learning trades database initialized")

    # ------------------------ Feature engineering ------------------------
    def calculate_options_metrics(self, data: pd.DataFrame, interval_hint: str) -> Optional[Dict]:
        if len(data) < 60:
            return None

        close = data["close"].values
        high = data["high"].values
        low = data["low"].values
        volume = data["volume"].values
        returns = np.diff(close) / close[:-1]

        ann = self._annualization_scale(interval_hint)

        # Vol measures
        hist_vol_30 = (np.std(returns[-30:])
                       if len(returns) >= 30 else 0.0) * ann

        vol_percentile = 50.0
        if len(returns) > 30:
            rolling = [np.std(returns[i - 30: i]) *
                       ann for i in range(30, len(returns))]
            if len(rolling) > 0:
                vol_percentile = (
                    np.sum(np.array(rolling) < hist_vol_30) / len(rolling)) * 100.0

        # Momentum
        momentum_5min = (close[-1] - close[-6]) / \
            close[-6] if len(close) > 5 else 0.0
        momentum_15min = (close[-1] - close[-16]) / \
            close[-16] if len(close) > 15 else 0.0

        # Position in recent range
        recent_high = np.max(high[-30:])
        recent_low = np.min(low[-30:])
        price_position = (
            (close[-1] - recent_low) / (recent_high - recent_low)
            if recent_high != recent_low
            else 0.5
        )

        # Volume spike
        avg_volume = np.mean(volume[-20:])
        volume_spike = volume[-1] / avg_volume if avg_volume > 0 else 1.0

        return {
            "volatility_percentile": float(vol_percentile),
            "momentum_5min": float(momentum_5min),
            "momentum_15min": float(momentum_15min),
            "volume_spike": float(volume_spike),
            "price_position": float(price_position),
            "near_resistance": bool(price_position > 0.8),
            "near_support": bool(price_position < 0.2),
            "hist_vol_30min": float(hist_vol_30),
        }

    def update_market_context(self, symbol: str, data: pd.DataFrame) -> None:
        """
        Update market context with data from related symbols (QQQ, UUP, BITX).
        This data is used for cross-asset correlation features.
        
        Args:
            symbol: The symbol being updated (e.g., 'QQQ', 'UUP')
            data: OHLCV DataFrame for the symbol
        """
        if data is None or data.empty:
            return
        
        # Store the full data for correlation calculations
        self.market_context[symbol] = data.copy()
        
        # Store the latest price for quick access
        close_col = 'Close' if 'Close' in data.columns else 'close'
        if close_col in data.columns:
            self.market_context_prices[symbol] = float(data[close_col].iloc[-1])
            logger.debug(f"[CONTEXT] Updated {symbol}: ${self.market_context_prices[symbol]:.2f}")
    
    def get_cross_asset_features(self, primary_data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate cross-asset features for enhanced predictions.
        
        Features include:
        - SPY/QQQ correlation (tech vs broad market)
        - SPY/QQQ relative strength (tech leading/lagging)
        - Dollar strength (UUP) impact on equities
        - Crypto correlation (BITX if available)
        
        Returns:
            Dict of cross-asset features, all normalized to [-1, 1] or [0, 1] range
        """
        features = {
            'spy_qqq_corr': 0.0,        # Correlation between SPY and QQQ
            'spy_qqq_rel_strength': 0.0, # QQQ relative strength vs SPY
            'dollar_strength': 0.0,      # UUP momentum (dollar strength)
            'dollar_equity_corr': 0.0,   # Correlation: stronger dollar = weaker equities?
            'crypto_corr': 0.0,          # BITX correlation with SPY
            'market_breadth': 0.0,       # Are QQQ and SPY moving together?
            'risk_on_off': 0.0,          # Risk-on (crypto+tech up) vs risk-off
            'context_confidence': 0.0,   # How much context data we have (0-1)
        }
        
        try:
            # Get primary (SPY) close prices
            close_col = 'Close' if 'Close' in primary_data.columns else 'close'
            spy_close = primary_data[close_col].values
            
            if len(spy_close) < 20:
                return features
            
            # Calculate SPY returns
            spy_returns = np.diff(spy_close) / spy_close[:-1]
            
            context_count = 0
            
            # QQQ features (tech sector)
            if 'QQQ' in self.market_context and not self.market_context['QQQ'].empty:
                qqq_data = self.market_context['QQQ']
                qqq_close_col = 'Close' if 'Close' in qqq_data.columns else 'close'
                qqq_close = qqq_data[qqq_close_col].values[-len(spy_close):]
                
                if len(qqq_close) >= 20:
                    qqq_returns = np.diff(qqq_close) / qqq_close[:-1]
                    
                    # Align lengths
                    min_len = min(len(spy_returns), len(qqq_returns))
                    if min_len >= 10:
                        spy_ret = spy_returns[-min_len:]
                        qqq_ret = qqq_returns[-min_len:]
                        
                        # Correlation
                        corr = np.corrcoef(spy_ret, qqq_ret)[0, 1]
                        features['spy_qqq_corr'] = float(corr) if not np.isnan(corr) else 0.0
                        
                        # Relative strength: QQQ 5-period return vs SPY 5-period return
                        spy_5d = (spy_close[-1] / spy_close[-6] - 1) if len(spy_close) > 5 else 0
                        qqq_5d = (qqq_close[-1] / qqq_close[-6] - 1) if len(qqq_close) > 5 else 0
                        features['spy_qqq_rel_strength'] = float(np.clip(qqq_5d - spy_5d, -0.1, 0.1) * 10)  # Normalized
                        
                        # Market breadth: are they moving together?
                        same_direction = np.sign(spy_ret[-1]) == np.sign(qqq_ret[-1]) if len(spy_ret) > 0 and len(qqq_ret) > 0 else False
                        features['market_breadth'] = 1.0 if same_direction else -1.0
                        
                        context_count += 1
            
            # UUP features (dollar strength)
            if 'UUP' in self.market_context and not self.market_context['UUP'].empty:
                uup_data = self.market_context['UUP']
                uup_close_col = 'Close' if 'Close' in uup_data.columns else 'close'
                uup_close = uup_data[uup_close_col].values[-len(spy_close):]
                
                if len(uup_close) >= 10:
                    # Dollar momentum (5-period)
                    uup_5d = (uup_close[-1] / uup_close[-6] - 1) if len(uup_close) > 5 else 0
                    features['dollar_strength'] = float(np.clip(uup_5d * 100, -1, 1))  # Normalized
                    
                    # Dollar-equity correlation (typically negative)
                    uup_returns = np.diff(uup_close) / uup_close[:-1]
                    min_len = min(len(spy_returns), len(uup_returns))
                    if min_len >= 10:
                        corr = np.corrcoef(spy_returns[-min_len:], uup_returns[-min_len:])[0, 1]
                        features['dollar_equity_corr'] = float(corr) if not np.isnan(corr) else 0.0
                    
                    context_count += 1
            
            # Bitcoin/Crypto features (BTC-USD or legacy BITX)
            crypto_symbol = 'BTC-USD' if 'BTC-USD' in self.market_context else 'BITX'
            if crypto_symbol in self.market_context and not self.market_context[crypto_symbol].empty:
                bitx_data = self.market_context[crypto_symbol]
                bitx_close_col = 'Close' if 'Close' in bitx_data.columns else 'close'
                bitx_close = bitx_data[bitx_close_col].values[-len(spy_close):]
                
                if len(bitx_close) >= 10:
                    bitx_returns = np.diff(bitx_close) / bitx_close[:-1]
                    min_len = min(len(spy_returns), len(bitx_returns))
                    if min_len >= 10:
                        corr = np.corrcoef(spy_returns[-min_len:], bitx_returns[-min_len:])[0, 1]
                        features['crypto_corr'] = float(corr) if not np.isnan(corr) else 0.0
                    
                    context_count += 1
            
            # Risk-on/off indicator: combine QQQ relative strength + crypto momentum
            qqq_rel = features['spy_qqq_rel_strength']
            crypto_mom = features['crypto_corr']
            features['risk_on_off'] = float(np.clip((qqq_rel + crypto_mom) / 2, -1, 1))
            
            # Context confidence: how much data do we have?
            features['context_confidence'] = min(context_count / 3.0, 1.0)
            
        except Exception as e:
            logger.debug(f"[CONTEXT] Cross-asset feature calculation error: {e}")
        
        return features

    def create_features(self, data: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Create comprehensive feature set for neural network model.

        Feature breakdown (59 total):
        - Features 0-3: Price & returns (4)
        - Features 4-6: Volatility (3)
        - Features 7-12: Volume features + derivatives (6)
        - Feature 13: RSI (1)
        - Features 14-15: Bollinger Bands (2)
        - Feature 16: MACD (1)
        - Features 17-18: Range features (2)
        - Features 19-24: Jerk analysis (6)
        - Features 25-28: VIX features (4)
        - Features 29-38: HMM regime features (10)
        - Features 39-44: Momentum & trend indicators (6)
        - Features 45-49: Short-term momentum (5)
        - Features 50-53: Time-of-day (hour_sin, hour_cos, dow_sin, dow_cos) (4)
        - Features 54-58: Gaussian pattern similarity (5 key metrics)
        """
        if len(data) < 20:
            return None

        # Handle both capitalized and lowercase column names
        close = data["Close"].values if "Close" in data.columns else data["close"].values
        high = data["High"].values if "High" in data.columns else data["high"].values
        low = data["Low"].values if "Low" in data.columns else data["low"].values
        volume = data["Volume"].values if "Volume" in data.columns else data["volume"].values

        features: List[float] = []

        # ===== FEATURES 0-3: Price & returns (4 features) =====
        features.append(float(close[-1]))
        features.append(
            float((close[-1] - close[-2]) / close[-2]) if len(close) > 1 else 0.0)
        features.append(
            float((close[-1] - close[-5]) / close[-5]) if len(close) > 4 else 0.0)
        features.append(
            float((close[-1] - close[-10]) / close[-10]) if len(close) > 9 else 0.0)

        # ===== FEATURES 4-6: Volatility (3 features) =====
        returns = np.diff(close) / close[:-1]
        features.extend(
            [
                float(np.std(returns[-5:]) if len(returns) >= 5 else 0.0),
                float(np.std(returns[-10:]) if len(returns) >= 10 else 0.0),
                float(np.std(returns[-20:]) if len(returns) >= 20 else 0.0),
            ]
        )

        # ===== FEATURES 7-12: Volume features + derivatives (6 features) =====
        # Basic volume features
        vol_current = float(volume[-1])
        vol_avg_5 = float(np.mean(volume[-5:]) if len(volume) >= 5 else volume[-1])
        vol_ratio = float(volume[-1] / np.mean(volume[-10:]) if len(volume) >= 10 and np.mean(volume[-10:]) > 0 else 1.0)
        
        # Volume derivatives (velocity, acceleration, jerk)
        vol_velocity = float(volume[-1] - volume[-2]) / vol_avg_5 if len(volume) > 1 and vol_avg_5 > 0 else 0.0
        vol_accel = 0.0
        vol_jerk = 0.0
        if len(volume) >= 4:
            vol_diff1 = np.diff(volume[-4:])  # Velocity
            vol_diff2 = np.diff(vol_diff1)    # Acceleration
            vol_diff3 = np.diff(vol_diff2)    # Jerk
            vol_accel = float(vol_diff2[-1]) / vol_avg_5 if vol_avg_5 > 0 else 0.0
            vol_jerk = float(vol_diff3[-1]) / vol_avg_5 if vol_avg_5 > 0 else 0.0
        
        features.extend([vol_current, vol_avg_5, vol_ratio, vol_velocity, vol_accel, vol_jerk])

        # ===== FEATURE 13: RSI (1 feature) =====
        if len(close) >= 14:
            delta = np.diff(close)
            gain = np.clip(delta, 0, None)
            loss = np.clip(-delta, 0, None)
            avg_gain = np.mean(gain[-14:])
            avg_loss = np.mean(loss[-14:])
            rs = avg_gain / avg_loss if avg_loss > 0 else np.inf
            rsi = 100.0 if np.isinf(rs) else 100.0 - (100.0 / (1.0 + rs))
            features.append(float(rsi))
        else:
            features.append(50.0)

        # ===== FEATURES 14-15: Bollinger position + std (2 features) =====
        if len(close) >= 20:
            sma = float(np.mean(close[-20:]))
            std = float(np.std(close[-20:]))
            bb_upper = sma + 2.0 * std
            bb_lower = sma - 2.0 * std
            bb_pos = (close[-1] - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5
            bb_pos = float(min(max(bb_pos, 0.0), 1.0))
            features.extend([bb_pos, float(std)])
        else:
            features.extend([0.5, 0.0])

        # ===== FEATURE 16: MACD normalized (1 feature) =====
        macd_norm = 0.0
        if len(close) >= 26:
            ema12 = pd.Series(close).ewm(span=12).mean().iloc[-1]
            ema26 = pd.Series(close).ewm(span=26).mean().iloc[-1]
            macd = float(ema12 - ema26)
            macd_norm = macd / float(close[-1]) if close[-1] != 0 else 0.0
        features.append(float(macd_norm))

        # ===== FEATURES 17-18: Range features (2 features) =====
        features.extend(
            [
                float((high[-1] - low[-1]) / close[-1] if close[-1] != 0 else 0.0),
                float(np.mean((high[-5:] - low[-5:]) / close[-5:]) if len(close) >= 5 else 0.0),
            ]
        )

        # ===== FEATURES 19-24: Jerk Analysis (6 features) =====
        # Price derivatives: velocity, acceleration, jerk
        price_velocity = 0.0
        price_accel = 0.0
        price_jerk = 0.0
        jerk_smooth = 0.0
        jerk_momentum = 0.0
        jerk_volatility = 0.0
        
        if len(close) >= 10:
            # Calculate derivatives
            velocity = np.diff(close)  # 1st derivative
            acceleration = np.diff(velocity)  # 2nd derivative
            jerk = np.diff(acceleration)  # 3rd derivative
            
            # Normalize by price to make comparable across assets
            price_velocity = float(velocity[-1] / close[-1]) if close[-1] != 0 else 0.0
            price_accel = float(acceleration[-1] / close[-1]) if len(acceleration) > 0 and close[-1] != 0 else 0.0
            price_jerk = float(jerk[-1] / close[-1]) if len(jerk) > 0 and close[-1] != 0 else 0.0
            
            # Smoothed jerk (3-period rolling mean)
            if len(jerk) >= 3:
                jerk_smooth = float(np.mean(jerk[-3:])) / close[-1] if close[-1] != 0 else 0.0
            
            # Jerk momentum (positive vs negative count over 5 periods)
            if len(jerk) >= 5:
                jerk_momentum = float(np.sum(jerk[-5:] > 0) - np.sum(jerk[-5:] < 0)) / 5.0
            
            # Jerk volatility (std of jerk over 5 periods)
            if len(jerk) >= 5:
                jerk_volatility = float(np.std(jerk[-5:])) / close[-1] if close[-1] != 0 else 0.0
        
        # Store price jerk for RL policies
        self.current_price_jerk = price_jerk
        
        features.extend([price_velocity, price_accel, price_jerk, jerk_smooth, jerk_momentum, jerk_volatility])

        # ===== FEATURES 25-28: VIX Features (4 features) =====
        vix_level = 0.0
        vix_bb_pos = 0.5
        vix_roc = 0.0
        vix_percentile = 0.5
        
        try:
            # Get VIX data from data source
            vix_data = self.data_source.get_data("^VIX", period=self.historical_period, interval="1m")
            if vix_data is not None and not vix_data.empty:
                vix_close_col = 'Close' if 'Close' in vix_data.columns else 'close'
                vix_close = vix_data[vix_close_col].values
                
                if len(vix_close) >= 20:
                    # VIX level (normalized, typical range 10-40)
                    vix_level = float(vix_close[-1]) / 30.0  # Normalize around 30
                    
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
                    
                    # VIX percentile (where current VIX sits in recent history)
                    vix_percentile = float(np.sum(vix_close[-100:] < vix_close[-1]) / min(len(vix_close), 100))
        except Exception as vix_err:
            logger.debug(f"VIX features unavailable: {vix_err}")
        
        # Store VIX extended features for RL policies
        self.current_vix_bb_pos = vix_bb_pos
        self.current_vix_roc = vix_roc
        self.current_vix_percentile = vix_percentile
        
        features.extend([vix_level, vix_bb_pos, vix_roc, vix_percentile])

        # ===== FEATURES 29-38: Multi-Dimensional HMM Market Regime (10 features) =====
        if self.hmm_regime_detector and self.hmm_regime_detector.is_fitted:
            try:
                regime_info = self.hmm_regime_detector.predict_current_regime(data)
                hmm_features = [
                    regime_info['combined_id'] / 26.0,  # Normalized combined regime (0-26 → 0-1)
                    regime_info['combined_confidence'],  # Overall confidence
                    regime_info['trend_state'] / 2.0,  # Trend state (0-2 → 0-1)
                    regime_info['trend_confidence'],  # Trend confidence
                    regime_info['volatility_state'] / 2.0,  # Volatility state (0-2 → 0-1)
                    regime_info['volatility_confidence'],  # Volatility confidence
                    regime_info['liquidity_state'] / 2.0,  # Liquidity state (0-2 → 0-1)
                    regime_info['liquidity_confidence'],  # Liquidity confidence
                    1.0 if regime_info['trend_state'] == 2 else 0.0,  # Is strong uptrend?
                    1.0 if regime_info['volatility_state'] == 0 else 0.0  # Is low volatility?
                ]
                features.extend(hmm_features)
                # Store current HMM regime for exit decisions
                self.current_hmm_regime = {
                    'state': regime_info['combined_name'],
                    'confidence': regime_info['combined_confidence'],
                    'trend': regime_info.get('trend_name', 'UNKNOWN'),
                    'volatility': regime_info.get('volatility_name', 'UNKNOWN'),
                    'liquidity': regime_info.get('liquidity_name', 'UNKNOWN')
                }
                # Log in format that dashboard can parse
                logger.info(f"[HMM] Regime: {regime_info['combined_name']}, Confidence: {regime_info['combined_confidence']*100:.1f}%, Trend: {regime_info.get('trend_name', 'Unknown')}, Vol: {regime_info.get('volatility_name', 'Unknown')}")
            except Exception as e:
                logger.debug(f"Multi-D HMM regime detection skipped: {e}")
                features.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # Pad with 10 zeros if HMM fails
        else:
            features.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # Pad with 10 zeros if HMM not available

        # ===== FEATURES 39-44: Momentum & Trend Indicators (6 features) =====
        # These are critical for prediction accuracy!
        
        # Feature 39: EMA crossover signal (9/21 EMA)
        ema_crossover = 0.0
        if len(close) >= 21:
            ema9 = pd.Series(close).ewm(span=9, adjust=False).mean().iloc[-1]
            ema21 = pd.Series(close).ewm(span=21, adjust=False).mean().iloc[-1]
            # Normalized crossover: positive = bullish, negative = bearish
            ema_crossover = float((ema9 - ema21) / close[-1]) if close[-1] != 0 else 0.0
        features.append(ema_crossover)
        
        # Feature 40: Price position relative to 20 SMA (trend following)
        price_vs_sma20 = 0.0
        if len(close) >= 20:
            sma20 = float(np.mean(close[-20:]))
            price_vs_sma20 = float((close[-1] - sma20) / sma20) if sma20 != 0 else 0.0
        features.append(price_vs_sma20)
        
        # Feature 41: Momentum (ROC - Rate of Change) 10 periods
        momentum_10 = 0.0
        if len(close) >= 11:
            momentum_10 = float((close[-1] - close[-11]) / close[-11]) if close[-11] != 0 else 0.0
        features.append(momentum_10)
        
        # Feature 42: ADX-like trend strength (simplified)
        trend_strength = 0.0
        if len(close) >= 14 and len(high) >= 14 and len(low) >= 14:
            # True Range components
            tr_components = []
            for i in range(-14, 0):
                if i == -14:
                    continue
                tr = max(
                    high[i] - low[i],
                    abs(high[i] - close[i-1]),
                    abs(low[i] - close[i-1])
                )
                tr_components.append(tr)
            atr = np.mean(tr_components) if tr_components else 1.0
            # Directional movement
            price_change = abs(close[-1] - close[-14])
            trend_strength = float(price_change / (atr * 14)) if atr > 0 else 0.0
            trend_strength = min(trend_strength, 2.0)  # Cap at 2
        features.append(trend_strength)
        
        # Feature 43: Recent highs/lows momentum
        hl_momentum = 0.0
        if len(high) >= 10 and len(low) >= 10:
            recent_high = float(np.max(high[-10:]))
            recent_low = float(np.min(low[-10:]))
            range_size = recent_high - recent_low
            if range_size > 0:
                # Where is current price in recent range? (0 = at low, 1 = at high)
                hl_momentum = float((close[-1] - recent_low) / range_size)
        features.append(hl_momentum)
        
        # Feature 44: Short-term vs long-term volatility ratio
        vol_ratio = 1.0
        if len(close) >= 30:
            returns = np.diff(close) / close[:-1]
            short_vol = float(np.std(returns[-5:])) if len(returns) >= 5 else 0.001
            long_vol = float(np.std(returns[-30:])) if len(returns) >= 30 else 0.001
            vol_ratio = float(short_vol / long_vol) if long_vol > 0.0001 else 1.0
            vol_ratio = min(vol_ratio, 3.0)  # Cap at 3x
        features.append(vol_ratio)
        
        # ===== FEATURES 45-49: SHORT-TERM MOMENTUM (5 features) =====
        # These are CRITICAL for accurate 5-10 minute predictions!
        
        # Feature 45: 3-period momentum (most recent direction)
        momentum_3 = 0.0
        if len(close) >= 4:
            momentum_3 = float((close[-1] - close[-4]) / close[-4]) if close[-4] != 0 else 0.0
        features.append(momentum_3)
        
        # Feature 46: 5-period momentum (slightly longer)
        momentum_5 = 0.0
        if len(close) >= 6:
            momentum_5 = float((close[-1] - close[-6]) / close[-6]) if close[-6] != 0 else 0.0
        features.append(momentum_5)
        
        # Feature 47: Recent price acceleration (2nd derivative of price)
        # Positive = accelerating upward or decelerating downward
        price_acceleration = 0.0
        if len(close) >= 5:
            velocity_recent = close[-1] - close[-2]  # Last period change
            velocity_prior = close[-3] - close[-4]   # Prior period change
            price_acceleration = float((velocity_recent - velocity_prior) / close[-1]) if close[-1] != 0 else 0.0
        features.append(price_acceleration)
        
        # Feature 48: Volume-weighted momentum (if volume available)
        # Higher volume = more conviction in move
        vwm = 0.0
        if len(close) >= 5 and len(volume) >= 5:
            price_changes = np.diff(close[-5:])
            volumes = volume[-4:]  # 4 volumes for 4 price changes
            if len(volumes) == len(price_changes):
                weighted_sum = float(np.sum(price_changes * volumes))
                vol_sum = float(np.sum(volumes))
                vwm = float(weighted_sum / vol_sum / close[-1]) if vol_sum > 0 and close[-1] != 0 else 0.0
        features.append(vwm)
        
        # Feature 49: Consecutive direction count (momentum persistence)
        # Counts how many consecutive bars in same direction
        consec_direction = 0.0
        if len(close) >= 10:
            consecutive = 0
            last_direction = None
            for i in range(-1, -10, -1):
                current_direction = 1 if close[i] > close[i-1] else -1 if close[i] < close[i-1] else 0
                if last_direction is None:
                    last_direction = current_direction
                    consecutive = 1 if current_direction != 0 else 0
                elif current_direction == last_direction and current_direction != 0:
                    consecutive += 1
                else:
                    break
            # Normalize: positive = up streak, negative = down streak
            consec_direction = float(consecutive * last_direction / 10.0) if last_direction else 0.0
        features.append(consec_direction)

        # ===== FEATURES 50-53: TIME-OF-DAY FEATURES (4 features) =====
        # Cyclical encoding of time - critical for intraday patterns!
        # Markets behave differently at open, midday, and close
        try:
            # Get timestamp from data
            if hasattr(data, 'index') and len(data.index) > 0:
                last_ts = data.index[-1]
                if hasattr(last_ts, 'hour'):
                    hour = last_ts.hour + last_ts.minute / 60.0
                    dow = last_ts.dayofweek  # 0=Monday, 4=Friday
                else:
                    hour = 12.0  # Default to noon
                    dow = 2  # Default to Wednesday
            else:
                hour = 12.0
                dow = 2
        except:
            hour = 12.0
            dow = 2

        # Cyclical encoding (sin/cos) preserves continuity (23:59 close to 00:00)
        hour_sin = float(np.sin(2 * np.pi * hour / 24.0))
        hour_cos = float(np.cos(2 * np.pi * hour / 24.0))
        dow_sin = float(np.sin(2 * np.pi * dow / 5.0))  # 5 trading days
        dow_cos = float(np.cos(2 * np.pi * dow / 5.0))

        features.extend([hour_sin, hour_cos, dow_sin, dow_cos])

        # ===== FEATURES 54-58: GAUSSIAN PATTERN SIMILARITY (5 features) =====
        # How similar is current market to recent patterns?
        # Uses GaussianKernelProcessor to measure similarity to reference window
        if self.gaussian_processor.is_fitted:
            try:
                gfeat = self.gaussian_processor.transform(np.array(features[:50]))  # Use base features
                if len(gfeat) >= 25:
                    # Extract key statistics across gamma values:
                    # gfeat layout: [mean, std, max, p75, p25] for each of 5 gammas
                    # Use gamma=1.0 (index 2) as primary - balanced sensitivity
                    gamma_1_idx = 10  # gamma[2] * 5 stats = indices 10-14
                    similarity_mean = float(gfeat[gamma_1_idx])      # Mean similarity to recent
                    similarity_std = float(gfeat[gamma_1_idx + 1])   # Variability in similarity
                    similarity_max = float(gfeat[gamma_1_idx + 2])   # Best match to recent
                    # Also use tight gamma (0.5) for local patterns
                    gamma_05_idx = 5  # gamma[1] * 5 stats = indices 5-9
                    local_similarity = float(gfeat[gamma_05_idx])    # Local pattern match
                    # And wide gamma (2.0) for broad regime
                    gamma_2_idx = 15  # gamma[3] * 5 stats = indices 15-19
                    broad_similarity = float(gfeat[gamma_2_idx])     # Broad regime match

                    features.extend([similarity_mean, similarity_std, similarity_max,
                                   local_similarity, broad_similarity])
                else:
                    features.extend([0.5, 0.1, 0.8, 0.5, 0.5])  # Neutral defaults
            except Exception as e:
                logger.debug(f"Gaussian features failed: {e}")
                features.extend([0.5, 0.1, 0.8, 0.5, 0.5])
        else:
            features.extend([0.5, 0.1, 0.8, 0.5, 0.5])  # Neutral defaults

        # Return all features (must match self.feature_dim).
        # Current design: 59 features (0-58).
        # Features 0-49: Base features (price, vol, HMM, momentum, etc.)
        # Features 50-53: Time-of-day (hour_sin, hour_cos, dow_sin, dow_cos)
        # Features 54-58: Gaussian pattern similarity (5 key metrics)
        return np.array(features[:self.feature_dim], dtype=np.float32)

    def _maybe_train_hmm(self, data: pd.DataFrame):
        """
        Auto-train or update Multi-Dimensional HMM when conditions are met:
        - First time: train when we have enough data
        - Periodic: retrain every 24 hours with latest data
        """
        if not self.hmm_regime_detector:
            logger.debug("🎰 HMM: No regime detector available")
            return
        
        now = datetime.now()
        
        # Check if we should train/retrain
        should_train = False
        
        if not self.hmm_regime_detector.is_fitted:
            # First time training - need minimum samples
            logger.debug(f"🎰 HMM: Not fitted yet. Data length: {len(data)}, need: {self.hmm_min_samples}")
            if len(data) >= self.hmm_min_samples:
                should_train = True
                logger.info(f"🎰 First-time Multi-D HMM training with {len(data)} samples...")
        elif self.last_hmm_train_time:
            # Periodic retraining
            hours_since_train = (now - self.last_hmm_train_time).total_seconds() / 3600
            if hours_since_train >= self.hmm_train_interval_hours:
                should_train = True
                logger.info(f"🎰 Periodic Multi-D HMM retraining ({hours_since_train:.1f} hours since last train)...")
        
        if should_train:
            try:
                # Train on recent data (3 separate HMMs: Trend, Volatility, Liquidity)
                success = self.hmm_regime_detector.fit(data.tail(self.hmm_min_samples))
                if success:
                    self.last_hmm_train_time = now
                    self.hmm_regime_detector.save()
                    logger.info(f"✅ Multi-D HMM training complete - model saved (27 regimes: 3×3×3)")
                else:
                    logger.warning(f"⚠️ Multi-D HMM training failed")
            except Exception as e:
                logger.error(f"❌ Multi-D HMM training error: {e}")

    # ------------------------ Predictions ------------------------
    def _maybe_init_model(self):
        """OK: Lazy model init once the buffer is long enough."""
        if (not self.model_initialized) and (len(self.feature_buffer) >= self.sequence_length):
            # IMPORTANT (V2): instantiate the predictor architecture from config.json
            # This ensures "Option A" (train predictor) produces a checkpoint compatible
            # with ArchV2 / PredictorManager.
            predictor_arch = "v1_original"
            try:
                cfg = self.config.config if self.config else {}
                predictor_arch = (
                    cfg.get("architecture", {})
                       .get("predictor", {})
                       .get("arch", predictor_arch)
                )
            except Exception:
                predictor_arch = "v1_original"

            self.model = create_predictor(
                arch=predictor_arch,
                feature_dim=self.feature_dim,
                sequence_length=self.sequence_length,
                use_gaussian_kernels=True,
            )
            self._safe_to_device(self.model)  # Optimization: Move to GPU if available (with CPU fallback)
            
            # Initialize optimizer with weight decay for regularization
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), 
                lr=self.learning_rate,
                weight_decay=1e-5  # L2 regularization
            )
            
            # Learning rate scheduler: Cosine annealing with warm restarts
            # Restarts every 1000 steps with increasing period
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, 
                T_0=1000,  # Initial restart period
                T_mult=2,  # Double period after each restart
                eta_min=1e-6  # Minimum learning rate
            )
            
            self.model.train()  # Set to training mode for continuous learning
            self.model_initialized = True
            logger.info(
                "Neural network initialized with training capabilities.")
            logger.info(f"   Predictor arch: {predictor_arch}")
            logger.info(f"   Optimizer: AdamW (lr={self.learning_rate}, weight_decay=1e-5)")
            logger.info(f"   Scheduler: CosineAnnealingWarmRestarts (T_0=1000, T_mult=2)")

            # Try to load existing model if available
            self.load_model()

    def get_neural_prediction(self, current_features: np.ndarray) -> Optional[Dict]:
        if self.model is None or len(self.feature_buffer) < self.sequence_length:
            return None

        # Simple rolling standardization over the last 'sequence_length' frames
        buf = np.vstack(self.feature_buffer[-self.sequence_length:])
        mu, sd = buf.mean(axis=0), buf.std(axis=0) + 1e-6
        cf = (current_features - mu) / sd
        seq = (buf - mu) / sd

        x = self._safe_to_device(torch.from_numpy(cf).float().unsqueeze(0))  # [1, D]
        s = self._safe_to_device(torch.from_numpy(seq).float().unsqueeze(0))  # [1, T, D]

        # SIMPLIFIED: Fewer MC samples for stability (like output6)
        # More samples = more memory pressure = crashes on Windows
        mc_samples = 5
        
        # Sequential MC sampling (more stable than batched on Windows)
        with torch.no_grad():
            outs = []
            for i in range(mc_samples):
                out = self.model(x, s)
                outs.append({
                    'return': out["return"].item(),
                    'volatility': out["volatility"].item(),
                    'confidence': out["confidence"].item(),
                    'direction': torch.softmax(out["direction"], dim=-1).squeeze().cpu().numpy(),
                    # Execution heads (predicted by the model, previously unused by the policy)
                    'fillability': out.get("fillability").item() if isinstance(out.get("fillability", None), torch.Tensor) else 0.5,
                    'exp_slippage': out.get("exp_slippage").item() if isinstance(out.get("exp_slippage", None), torch.Tensor) else 0.0,
                    'exp_ttf': out.get("exp_ttf").item() if isinstance(out.get("exp_ttf", None), torch.Tensor) else 0.0,
                    # Predictor embedding (latent representation) for future downstream consumers
                    'embedding': out.get("embedding").squeeze().detach().cpu().numpy().astype(np.float32).tolist()
                    if isinstance(out.get("embedding", None), torch.Tensor) else None,
                })
                del out  # Immediate cleanup
        
        # Extract MC statistics from sequential outputs
        returns = np.array([o["return"] for o in outs])
        volatilities = np.array([o["volatility"] for o in outs])
        confidences = np.array([o["confidence"] for o in outs])
        directions = np.array([o["direction"] for o in outs])
        fillabilities = np.array([o.get("fillability", 0.5) for o in outs], dtype=np.float32)
        exp_slippages = np.array([o.get("exp_slippage", 0.0) for o in outs], dtype=np.float32)
        exp_ttfs = np.array([o.get("exp_ttf", 0.0) for o in outs], dtype=np.float32)
        
        # Calculate Monte Carlo statistics
        mc_return_mean = float(np.mean(returns))
        mc_return_std = float(np.std(returns))
        mc_return_median = float(np.median(returns))
        mc_return_percentiles = np.percentile(returns, [5, 25, 75, 95])
        
        # Apply bounds to predictions - realistic 15-minute returns should be ±2% max
        pred_ret = np.tanh(mc_return_mean) * 0.02  # Bound to ±2% using tanh
        
        # Enhanced volatility and confidence with Monte Carlo uncertainty
        pred_vol = float(np.mean(volatilities))
        mc_vol_std = float(np.std(volatilities))
        
        # Average direction probabilities across MC samples
        dirs = np.mean(directions, axis=0)  # [3]
        
        # Confidence with Monte Carlo uncertainty consideration
        base_conf = float(np.mean(confidences))
        mc_uncertainty = mc_return_std  # Higher std = lower confidence
        adjusted_conf = base_conf * (1.0 - min(mc_uncertainty * 10, 0.5))  # Reduce confidence if high uncertainty

        gfeat = self.gaussian_processor.transform(current_features)
        gconf = float(min(gfeat[0], 1.0)) if len(gfeat) > 0 else 0.5

        return {
            "predicted_return": pred_ret,
            "predicted_volatility": pred_vol,
            "direction_probs": {"DOWN": float(dirs[0]), "NEUTRAL": float(dirs[1]), "UP": float(dirs[2])},
            "neural_confidence": adjusted_conf,
            "gaussian_confidence": gconf,
            "temporal_confidence": (adjusted_conf + gconf) / 2.0,
            # Execution heads (MC-aggregated)
            "fillability_mean": float(np.mean(fillabilities)) if len(fillabilities) else 0.5,
            "fillability_std": float(np.std(fillabilities)) if len(fillabilities) else 0.0,
            "exp_slippage_mean": float(np.mean(exp_slippages)) if len(exp_slippages) else 0.0,
            "exp_slippage_std": float(np.std(exp_slippages)) if len(exp_slippages) else 0.0,
            "exp_ttf_mean": float(np.mean(exp_ttfs)) if len(exp_ttfs) else 0.0,
            "exp_ttf_std": float(np.std(exp_ttfs)) if len(exp_ttfs) else 0.0,
            # Embedding (use last sample; kept for logging/analysis, not used for trading yet)
            "predictor_embedding": outs[-1].get("embedding") if outs else None,
            "monte_carlo_stats": {
                "samples": mc_samples,
                "return_mean": mc_return_mean,
                "return_std": mc_return_std,
                "return_median": mc_return_median,
                "return_percentiles": {
                    "p5": float(mc_return_percentiles[0]),
                    "p25": float(mc_return_percentiles[1]), 
                    "p75": float(mc_return_percentiles[2]),
                    "p95": float(mc_return_percentiles[3])
                },
                "volatility_uncertainty": mc_vol_std,
                "prediction_uncertainty": mc_uncertainty
            },
            "lstm_context": {"mean": mc_return_mean, 
                           "std": mc_return_std,
                           "confidence": adjusted_conf}
        }

    def _apply_edge_entry_controller(
        self,
        signal: Dict,
        neural_pred: Optional[Dict],
        options_metrics: Dict,
        symbol: str,
    ) -> None:
        """
        Edge-based entry selection using predictor outputs + execution heads.

        This is intentionally ENTRY-only. Exits remain controlled by the existing exit stack.
        """
        if not HAS_EDGE_ENTRY_CONTROLLER or edge_decide_from_signal is None:
            return
        if not neural_pred:
            return
        try:
            # Controller toggle: default ON (better than RL-on-18-scalars for entries).
            import os
            mode = os.environ.get("ENTRY_CONTROLLER", "edge").strip().lower()
            if mode not in ("edge", "edge_scorer", "world_model"):
                return

            decision, updates = edge_decide_from_signal(
                signal=signal,
                neural_pred=neural_pred,
                options_metrics=options_metrics,
                bot=self,
            )
            # Apply updates (do not overwrite confidence; engine gating still uses calibrated confidence)
            for k, v in updates.items():
                if k == "confidence":
                    continue
                signal[k] = v
            if decision.action == "HOLD":
                # Keep existing action if already HOLD; otherwise convert to HOLD.
                signal["action"] = "HOLD"
                signal["strategy"] = signal.get("strategy", "EDGE_SCORER") or "EDGE_SCORER"
                try:
                    signal.setdefault("reasoning", [])
                    signal["reasoning"] = list(signal.get("reasoning", [])) + [decision.reason]
                except Exception:
                    signal["reasoning"] = [decision.reason]
        except Exception as e:
            logger.debug(f"Edge entry controller error: {e}")

    def _apply_q_entry_controller(
        self,
        signal: Dict,
        neural_pred: Optional[Dict],
        options_metrics: Dict,
        symbol: str,
    ) -> None:
        """
        Apply the offline Q-scorer controller for ENTRY decisions.

        Safety rules:
        - Only active when ENTRY_CONTROLLER=q_scorer
        - If model missing/unloadable, fall back to edge controller (no behavior change beyond env choice)
        """
        import os
        mode = os.environ.get("ENTRY_CONTROLLER", "edge").strip().lower()
        if mode not in ("q_scorer", "q", "qscorer"):
            return
        if not HAS_Q_ENTRY_CONTROLLER or QEntryController is None:
            return
        # Build trade_state dict from current signal (already contains scalars) and embedding
        trade_state = dict(signal or {})
        # Ensure predictor embedding present
        if neural_pred and neural_pred.get("predictor_embedding") is not None:
            trade_state["predictor_embedding"] = neural_pred.get("predictor_embedding")
        if trade_state.get("predictor_embedding") is None:
            # No embedding => cannot use q_scorer
            return
        # Ensure core scalar fields exist (best-effort)
        trade_state.setdefault("hmm_trend", 0.5)
        trade_state.setdefault("hmm_volatility", 0.5)
        trade_state.setdefault("hmm_liquidity", 0.5)
        trade_state.setdefault("hmm_confidence", 0.5)
        trade_state.setdefault("vix_bb_pos", getattr(self, "current_vix_bb_pos", 0.5))
        trade_state.setdefault("vix_level", getattr(self, "current_vix", 18.0))

        # Lazy singleton controller instance on the bot
        if not hasattr(self, "_q_entry_controller"):
            self._q_entry_controller = QEntryController()
            try:
                self._q_entry_controller.load()
            except Exception:
                pass

        qc = getattr(self, "_q_entry_controller", None)
        if qc is None or not getattr(qc, "model", None):
            # fallback to edge controller
            self._apply_edge_entry_controller(signal, neural_pred, options_metrics, symbol)
            return

        # Hot-reload model/metadata if a new artifact is deployed while the bot is running.
        try:
            if hasattr(qc, "maybe_reload"):
                qc.maybe_reload()
        except Exception:
            pass

        decision = qc.decide(trade_state)
        if decision is None:
            self._apply_edge_entry_controller(signal, neural_pred, options_metrics, symbol)
            return

        signal["q_values"] = decision.q_values
        signal["q_threshold"] = decision.threshold
        signal.setdefault("reasoning", [])
        try:
            signal["reasoning"] = list(signal.get("reasoning", [])) + [decision.reason]
        except Exception:
            signal["reasoning"] = [decision.reason]

        if decision.action == "HOLD":
            signal["action"] = "HOLD"
            signal["strategy"] = "Q_SCORER_HOLD"
            signal["rl_position_size"] = 0
        else:
            signal["action"] = decision.action
            signal["strategy"] = "Q_SCORER"
            signal["rl_position_size"] = int(decision.position_size or 1)

    def get_multi_timeframe_predictions(self, current_features: np.ndarray) -> Dict[str, Dict]:
        """Get neural network predictions for multiple timeframes (15min, 30min, 1hour, 4hour)"""
        predictions = {}
        
        # Show warmup message once when starting
        if len(self.feature_buffer) < self.sequence_length and not hasattr(self, '_warmup_message_shown'):
            logger.info(f"🔄 WARMUP: Building feature buffer ({len(self.feature_buffer)}/{self.sequence_length}) - NN predictions will start after {self.sequence_length - len(self.feature_buffer)} more cycles")
            self._warmup_message_shown = True
        
        # Show ready message once when warmup completes
        elif len(self.feature_buffer) >= self.sequence_length and not hasattr(self, '_warmup_complete_message_shown'):
            logger.info(f"🎉 WARMUP COMPLETE! Feature buffer full ({len(self.feature_buffer)}/{self.sequence_length}) - Neural network predictions now active!")
            self._warmup_complete_message_shown = True
        
        logger.info(f"🎯 Generating multi-timeframe predictions for {len(self.prediction_timeframes)} timeframes")
        
        for timeframe_name, config in self.prediction_timeframes.items():
            prediction = self._get_neural_prediction_for_timeframe(current_features, config["minutes"])
            if prediction:
                predictions[timeframe_name] = {
                    **prediction,
                    "weight": config["weight"],
                    "timeframe": timeframe_name
                }
                # Log in format dashboard expects: [LSTM] {timeframe}: direction={dir}, return={ret}%, confidence={conf}%
                pred_return = prediction.get('predicted_return', 0)
                pred_conf = prediction.get('neural_confidence', 0)
                direction = "UP" if pred_return > 0 else "DOWN" if pred_return < 0 else "NEUTRAL"
                logger.info(f"[LSTM] {timeframe_name}: direction={direction}, return={pred_return*100:+.2f}%, confidence={pred_conf*100:.1f}%")
            else:
                # Only show warning if we're past warmup period (feature buffer should be full by now)
                if len(self.feature_buffer) >= self.sequence_length:
                    logger.warning(f"[WARN] Failed to generate {timeframe_name} prediction")
                # During warmup (first 30 cycles), silently skip - this is expected
        
        logger.info(f"[PREDICT] Total multi-timeframe predictions generated: {len(predictions)}")
        
        # Store for later use in signal generation
        self.multi_timeframe_predictions = predictions
        
        # NOTE: We store the last known SPY price for validation, NOT features[-1] which is a different value!
        # The actual price will be passed via store_multi_timeframe_predictions_for_validation()
        # when called from generate_unified_signal() which has access to the real price.
        # Here we skip storing for validation - it will be done in generate_unified_signal()
        # with the correct current_price value.
        
        # Store predictions indexed by their TARGET time for aggregation
        # Note: We use a placeholder price here; the dashboard will use actual prices from the chart data
        try:
            current_time = self.get_market_time()
            # Use the last_known_price which is set by generate_unified_signal with real SPY price
            spy_price = getattr(self, '_last_known_spy_price', 0)
            if spy_price > 0:
                self._store_predictions_by_target_time(predictions, current_time, spy_price)
        except Exception as e:
            logger.debug(f"Could not store predictions by target time: {e}")
        
        return predictions
    
    def _store_predictions_by_target_time(self, predictions: Dict, current_time: datetime, current_price: float) -> None:
        """
        Store ALL predictions indexed by their TARGET time.
        
        Always stores new predictions - the aggregation logic uses ALL predictions
        to calculate agreement-adjusted confidence.
        """
        for tf_name, pred in predictions.items():
            tf_minutes = pred.get('timeframe_minutes', 15)
            
            # Calculate the target time this prediction is about
            target_time = current_time + timedelta(minutes=tf_minutes)
            # Round to nearest minute for aggregation
            target_key = target_time.strftime('%Y-%m-%d %H:%M')
            
            new_pred_data = {
                'prediction_time': current_time.isoformat(),
                'target_time': target_key,
                'timeframe': tf_name,
                'timeframe_minutes': tf_minutes,
                'predicted_return': pred.get('predicted_return', 0),
                'neural_confidence': pred.get('neural_confidence', 0.5),
                'direction': 'UP' if pred.get('predicted_return', 0) > 0.001 else 'DOWN' if pred.get('predicted_return', 0) < -0.001 else 'NEUTRAL',
                'entry_price': current_price,
                'weight': pred.get('weight', 0.1)
            }
            
            if target_key not in self.predictions_by_target_time:
                self.predictions_by_target_time[target_key] = []
            
            self.predictions_by_target_time[target_key].append(new_pred_data)
            
            # Keep max 10 predictions per target time (oldest get dropped)
            if len(self.predictions_by_target_time[target_key]) > 10:
                self.predictions_by_target_time[target_key] = self.predictions_by_target_time[target_key][-10:]
        
        # Clean up old predictions (keep memory bounded)
        self._cleanup_old_target_predictions()
    
    def _cleanup_old_target_predictions(self) -> None:
        """Remove predictions for target times that have passed."""
        current_time = self.get_market_time()
        cutoff_time = current_time - timedelta(minutes=30)  # Keep 30min history
        
        keys_to_remove = []
        for target_key in self.predictions_by_target_time:
            try:
                target_time = datetime.strptime(target_key, '%Y-%m-%d %H:%M')
                if target_time < cutoff_time:
                    keys_to_remove.append(target_key)
            except ValueError:
                keys_to_remove.append(target_key)
        
        for key in keys_to_remove:
            del self.predictions_by_target_time[key]
    
    def get_best_predictions_by_target_time(self, lookforward_minutes: int = 120) -> Dict[str, Dict]:
        """
        Get aggregated prediction using SMART ENSEMBLE:
        
        1. RECENCY WEIGHTING: More recent predictions get higher weight
        2. SHORTER TIMEFRAME BONUS: 5min predictions are more accurate than 60min
        3. PROXIMITY BONUS: As target approaches, predictions become more reliable
        4. AGREEMENT BOOST: When predictions agree, confidence increases
        5. VOLUME BONUS: More predictions = higher confidence
        
        Args:
            lookforward_minutes: How far into the future to include predictions
            
        Returns:
            Dict mapping target time -> smart ensemble prediction
        """
        import math
        
        current_time = self.get_market_time()
        best_predictions = {}
        
        for target_key, predictions in self.predictions_by_target_time.items():
            try:
                target_time = datetime.strptime(target_key, '%Y-%m-%d %H:%M')
            except ValueError:
                continue
            
            # Only include future predictions within lookforward window
            if target_time <= current_time:
                continue
            if target_time > current_time + timedelta(minutes=lookforward_minutes):
                continue
            
            if not predictions:
                continue
            
            total = len(predictions)
            time_to_target = (target_time - current_time).total_seconds() / 60
            
            # ===== CALCULATE SMART WEIGHTS FOR EACH PREDICTION =====
            weights = []
            for p in predictions:
                base_conf = p.get('neural_confidence', 0.5)
                
                # Parse prediction time
                pred_time_str = p.get('prediction_time', '')
                try:
                    pred_time = datetime.fromisoformat(pred_time_str)
                    pred_age = (current_time - pred_time).total_seconds() / 60
                except (ValueError, TypeError):
                    pred_age = 0
                
                # 1. RECENCY WEIGHT: Half-life of 10 minutes
                recency_weight = math.exp(-0.069 * max(0, pred_age))
                
                # 2. SHORTER TIMEFRAME BONUS: 5min=1.5x, 30min=1.0x, 60min=0.5x
                tf_minutes = p.get('timeframe_minutes', 30)
                timeframe_weight = max(0.5, 1.5 - (tf_minutes / 60))
                
                # 3. PROXIMITY BONUS: Weight increases as target approaches
                proximity_weight = min(2.0, 30 / max(1, time_to_target))
                
                # Combined weight
                weight = base_conf * recency_weight * timeframe_weight * proximity_weight
                weights.append(weight)
            
            total_weight = sum(weights) or 1
            
            # ===== WEIGHTED RETURN =====
            weighted_return = sum(p.get('predicted_return', 0) * w for p, w in zip(predictions, weights)) / total_weight
            
            # ===== DIRECTION VOTING (weighted) =====
            bullish_weight = sum(w for p, w in zip(predictions, weights) if p.get('direction') == 'UP')
            bearish_weight = sum(w for p, w in zip(predictions, weights) if p.get('direction') == 'DOWN')
            neutral_weight = total_weight - bullish_weight - bearish_weight
            
            # Count for consensus
            bullish_votes = sum(1 for p in predictions if p.get('direction') == 'UP')
            bearish_votes = sum(1 for p in predictions if p.get('direction') == 'DOWN')
            neutral_votes = total - bullish_votes - bearish_votes
            
            # Determine direction by weighted vote
            if bullish_weight > bearish_weight and bullish_weight > neutral_weight:
                direction = 'UP'
            elif bearish_weight > bullish_weight and bearish_weight > neutral_weight:
                direction = 'DOWN'
            else:
                direction = 'NEUTRAL'
            
            # ===== AGREEMENT-ADJUSTED CONFIDENCE =====
            base_confidence = sum(p.get('neural_confidence', 0.5) * w for p, w in zip(predictions, weights)) / total_weight
            
            # Agreement based on weighted votes
            max_vote_weight = max(bullish_weight, bearish_weight, neutral_weight)
            agreement_ratio = max_vote_weight / total_weight if total_weight > 0 else 0.5
            
            # Agreement multiplier: 100%=1.5x, 50%=1.1x
            agreement_multiplier = 0.7 + (0.8 * agreement_ratio)
            
            # Volume bonus: More predictions = higher confidence
            volume_bonus = min(1.3, 1.0 + (total - 1) * 0.03)
            
            adjusted_confidence = min(1.0, base_confidence * agreement_multiplier * volume_bonus)
            
            # Get entry price from most recent prediction
            latest_pred = max(predictions, key=lambda p: p.get('prediction_time', ''))
            
            best_predictions[target_key] = {
                'target_time': target_key,
                'predicted_return': weighted_return,
                'neural_confidence': adjusted_confidence,
                'base_confidence': base_confidence,
                'direction': direction,
                'entry_price': latest_pred.get('entry_price', 0),
                'time_to_target_minutes': int(time_to_target),
                'consensus': {
                    'bullish_votes': bullish_votes,
                    'bearish_votes': bearish_votes,
                    'neutral_votes': neutral_votes,
                    'total_predictions': total,
                    'agreement_ratio': agreement_ratio,
                    'agreement_multiplier': agreement_multiplier,
                    'volume_bonus': volume_bonus
                }
            }
        
        return best_predictions
    
    def get_aggregated_predictions_for_dashboard(self) -> Dict[str, Any]:
        """
        Get aggregated predictions formatted for the training dashboard.
        
        Returns predictions with:
        - Best (highest confidence) prediction per target time
        - Consensus info when multiple predictions overlap
        """
        current_time = self.get_market_time()
        current_price = 0
        if len(self.feature_buffer) > 0:
            try:
                current_price = float(list(self.feature_buffer)[-1][-1])
            except (IndexError, TypeError):
                pass
        
        best_preds = self.get_best_predictions_by_target_time()
        
        # Format for dashboard
        dashboard_data = {
            'timestamp': current_time.isoformat(),
            'current_price': current_price,
            'aggregated_predictions': [],
            'current_predictions': self.multi_timeframe_predictions
        }
        
        # Sort by target time and format
        sorted_targets = sorted(best_preds.keys())
        for target_key in sorted_targets:
            pred = best_preds[target_key]
            try:
                target_time = datetime.strptime(target_key, '%Y-%m-%d %H:%M')
                minutes_away = (target_time - current_time).total_seconds() / 60
            except ValueError:
                minutes_away = 0
            
            dashboard_data['aggregated_predictions'].append({
                'target_time': target_key,
                'minutes_away': int(minutes_away),
                'predicted_return': pred.get('predicted_return', 0),
                'confidence': pred.get('neural_confidence', 0),
                'direction': pred.get('direction', 'NEUTRAL'),
                'source_timeframe': pred.get('timeframe', 'unknown'),
                'prediction_made_at': pred.get('prediction_time', ''),
                'entry_price': pred.get('entry_price', 0),
                'consensus': pred.get('consensus', None)
            })
        
        return dashboard_data

    def _get_neural_prediction_for_timeframe(self, current_features: np.ndarray, minutes_ahead: int) -> Optional[Dict]:
        """Get neural network prediction scaled for a specific timeframe"""
        if self.model is None or len(self.feature_buffer) < self.sequence_length:
            # Don't spam warnings during warmup period (building feature buffer)
            # This is expected behavior for the first 30 cycles
            return None

        try:
            # ========================================================================
            # CRITICAL FIX: Use cached prediction if available!
            # This ensures multi-timeframe predictions match the neural_pred used for
            # signal generation (prevents sign flip due to Monte Carlo sampling)
            # ========================================================================
            base_prediction = getattr(self, '_cached_neural_pred', None)
            if base_prediction is None:
                # Fallback to fresh prediction if no cache (shouldn't happen normally)
                base_prediction = self.get_neural_prediction(current_features)
            if not base_prediction:
                return None

            # ===================================================================
            # IMPROVED PREDICTION SCALING (v3 - realistic, dampened)
            # ===================================================================
            # Key insight: Neural networks tend to OVERPREDICT magnitude!
            # We need to heavily dampen predictions to match real SPY volatility.
            
            # First, dampen the base prediction - NN outputs are typically 5-10x too large
            raw_return = base_prediction["predicted_return"]
            base_return = raw_return * 0.1  # Reduce to 10% of raw prediction
            
            if minutes_ahead <= 15:
                # Very short-term (5-15min): Use raw prediction, highest confidence
                # These are most reliable - momentum carries
                timeframe_multiplier = 1.0
                scaled_return = base_return
                confidence_penalty = 1.0
                
            elif minutes_ahead <= 30:
                # Short-term (15-30min): Slight dampening
                timeframe_multiplier = 0.85
                scaled_return = base_return * timeframe_multiplier
                confidence_penalty = 0.9
                
            elif minutes_ahead <= 60:
                # Medium-term (30-60min): More dampening, mean reversion starts
                timeframe_multiplier = 0.7
                scaled_return = base_return * timeframe_multiplier
                confidence_penalty = 0.75
                
            else:
                # Longer-term (60min+): Strong mean reversion
                # The further out, the more we expect price to mean-revert
                # Dampen prediction AND reduce confidence significantly
                reversion_factor = 0.5 * np.exp(-minutes_ahead / 120.0)  # Exponential decay
                timeframe_multiplier = max(0.2, reversion_factor)
                scaled_return = base_return * timeframe_multiplier
                confidence_penalty = max(0.3, 0.6 * np.exp(-minutes_ahead / 180.0))
            
            # REALISTIC SANITY CAPS: SPY typically moves 0.05-0.2% per 15min
            # Cap predictions to REALISTIC ranges based on timeframe
            # These are based on actual SPY intraday volatility!
            if minutes_ahead <= 5:
                max_return = 0.001  # ±0.1% for 5min (SPY ~$0.60 move)
            elif minutes_ahead <= 10:
                max_return = 0.0015  # ±0.15% for 10min
            elif minutes_ahead <= 15:
                max_return = 0.002  # ±0.2% for 15min
            elif minutes_ahead <= 30:
                max_return = 0.003  # ±0.3% for 30min
            elif minutes_ahead <= 60:
                max_return = 0.005  # ±0.5% for 1hr
            else:
                max_return = 0.008  # ±0.8% for 2hr (still conservative)
            
            scaled_return = max(-max_return, min(max_return, scaled_return))
            
            # Scale volatility prediction (less aggressive)
            scaled_volatility = base_prediction["predicted_volatility"] * min(timeframe_multiplier, 1.0)
            
            # Apply confidence penalty
            scaled_confidence = base_prediction["neural_confidence"] * confidence_penalty
            
            # Direction probabilities remain similar but with reduced confidence
            direction_probs = base_prediction["direction_probs"].copy()
            
            return {
                "predicted_return": float(scaled_return),
                "predicted_volatility": float(scaled_volatility), 
                "neural_confidence": float(scaled_confidence),
                "direction_probs": direction_probs,
                "timeframe_minutes": minutes_ahead,
                "timeframe_multiplier": float(timeframe_multiplier),
                "confidence_penalty": float(confidence_penalty),
                "base_prediction": base_prediction["predicted_return"]  # Store original for comparison
            }

        except Exception as e:
            logger.error(f"Error in neural prediction for {minutes_ahead}min timeframe: {e}")
            return None

    def combine_multi_timeframe_signals(self, use_aggregated: bool = True) -> Dict[str, Any]:
        """
        Combine multiple timeframe predictions into unified signal strength and confidence.
        
        SIMPLIFIED: Let HMM handle all regime detection!
        - HMM provides: Trend, Volatility, Liquidity states
        - No manual filters needed - NN learns from HMM features
        """
        if not self.multi_timeframe_predictions:
            return {"confidence": 0.5, "direction": "NEUTRAL", "strength": 0.0}
        
        # Count predictions for logging
        # Use 0.0001 (0.01%) threshold to match neural_return_threshold
        total_predictions = len(self.multi_timeframe_predictions)
        bullish_count = sum(1 for p in self.multi_timeframe_predictions.values() if p["predicted_return"] > 0.0001)
        bearish_count = sum(1 for p in self.multi_timeframe_predictions.values() if p["predicted_return"] < -0.0001)
        consensus_pct = max(bullish_count, bearish_count) / total_predictions if total_predictions > 0 else 0
        
        # Confidence-weighted voting
        weighted_bullish = 0.0
        weighted_bearish = 0.0
        total_weight = 0.0
        weighted_return = 0.0
        weighted_confidence = 0.0
        timeframe_details = {}
        
        for timeframe, pred in self.multi_timeframe_predictions.items():
            weight = pred["weight"]
            predicted_return = pred["predicted_return"]
            confidence = pred["neural_confidence"]
            
            # Use 0.0001 (0.01%) threshold to detect direction (matches neural_return_threshold)
            if predicted_return > 0.0001:
                weighted_bullish += confidence * weight
            elif predicted_return < -0.0001:
                weighted_bearish += confidence * weight
            
            weighted_return += predicted_return * weight
            weighted_confidence += confidence * weight
            total_weight += weight
            
            timeframe_details[timeframe] = {
                "return": predicted_return,
                "confidence": confidence,
                "direction": "UP" if predicted_return > 0.0001 else "DOWN" if predicted_return < -0.0001 else "NEUTRAL"
            }
        
        # Calculate final metrics
        if total_weight > 0:
            final_return = weighted_return / total_weight
            final_confidence = weighted_confidence / total_weight
        else:
            final_return = 0
            final_confidence = 0.5
        
        # Determine direction - REQUIRE CONSENSUS to avoid conflicting signals
        # Don't trade against the short-term (15min) if it disagrees
        total_tf = total_predictions  # FIX: Use the correct variable name
        
        # Check if 15min (short-term) agrees with the majority
        short_term_pred = self.multi_timeframe_predictions.get('15min', {})
        short_term_direction = short_term_pred.get('direction', 'NEUTRAL')
        
        # Require at least 75% of timeframes to agree (3 out of 4)
        # FIX: Use bullish_count/bearish_count instead of undefined bullish_votes/bearish_votes
        agreement_ratio = max(bullish_count, bearish_count) / total_tf if total_tf > 0 else 0
        
        if agreement_ratio >= 0.75:  # Strong consensus (75%+ agree)
            if weighted_bullish > weighted_bearish * 1.1:
                final_direction = "BULLISH"
            elif weighted_bearish > weighted_bullish * 1.1:
                final_direction = "BEARISH"
            else:
                final_direction = "NEUTRAL"
        elif short_term_direction != 'NEUTRAL':
            # No strong consensus - ONLY trade if ALL timeframes lean same way
            # Otherwise go NEUTRAL to avoid buying against the short-term trend
            if bullish_count == total_tf:  # FIX: bullish_count
                final_direction = "BULLISH"
            elif bearish_count == total_tf:  # FIX: bearish_count
                final_direction = "BEARISH"
            else:
                final_direction = "NEUTRAL"
                logger.info(f"⚠️ Mixed signals ({bullish_count}↑/{bearish_count}↓) - going NEUTRAL to avoid conflict")
        else:
            final_direction = "NEUTRAL"
        
        # Calculate agreement score
        total_votes = weighted_bullish + weighted_bearish
        direction_strength = max(weighted_bullish, weighted_bearish) / total_votes if total_votes > 0 else 0.5
        agreement_score = direction_strength * consensus_pct
        
        # Calculate weighted average timeframe
        weighted_timeframe = sum(
            pred["weight"] * pred.get("timeframe_minutes", 60) 
            for pred in self.multi_timeframe_predictions.values()
        ) / total_weight if total_weight > 0 else 60
        
        # Log (HMM handles regime - no manual filters!)
        logger.info(f"📊 Consensus: {bullish_count}↑/{bearish_count}↓ ({consensus_pct:.0%})")
        logger.info(f"💪 Weighted: Bull={weighted_bullish:.2f} vs Bear={weighted_bearish:.2f} → {final_direction}")
        
        # --- NEW: Prediction Stability & Momentum Check ---
        # Add current consensus to history
        self.consensus_history.append({
            'direction': final_direction,
            'weighted_return': final_return,
            'confidence': final_confidence,
            'timestamp': self.get_market_time()
        })
        
        stability_bonus = 0.0
        momentum_bonus = 0.0
        
        if len(self.consensus_history) >= 5:
            # 1. Stability: How many of the last 5 signals match the current direction?
            recent_dirs = [x['direction'] for x in list(self.consensus_history)[-5:]]
            match_count = recent_dirs.count(final_direction)
            stability_ratio = match_count / 5.0
            
            if stability_ratio >= 0.8:  # 4 or 5 out of 5 match
                stability_bonus = 0.05  # +5% confidence
                logger.info(f"🛡️ Signal Stability: HIGH ({stability_ratio:.0%}) -> +5% confidence")
            elif stability_ratio <= 0.4:  # Choppy signal
                stability_bonus = -0.05 # -5% confidence penalty
                logger.info(f"⚠️ Signal Stability: LOW ({stability_ratio:.0%}) -> -5% confidence")
                
            # 2. Momentum: Is the weighted return magnitude increasing?
            recent_returns = [abs(x['weighted_return']) for x in list(self.consensus_history)[-3:]]
            if len(recent_returns) == 3:
                # Check if strictly increasing (or close to it)
                if recent_returns[2] > recent_returns[1] > recent_returns[0]:
                    momentum_bonus = 0.03 # +3% confidence
                    logger.info(f"🚀 Signal Momentum: INCREASING -> +3% confidence")
                elif recent_returns[2] < recent_returns[1] < recent_returns[0]:
                    momentum_bonus = -0.02 # -2% confidence (fading signal)
                    logger.info(f"📉 Signal Momentum: FADING -> -2% confidence")

        # Apply bonuses to final confidence
        final_confidence = max(0.0, min(1.0, final_confidence + stability_bonus + momentum_bonus))
        
        return {
            "multi_timeframe_confidence": final_confidence,
            "multi_timeframe_return": final_return,
            "direction": final_direction,
            "agreement_score": agreement_score,
            "signal_strength": agreement_score * final_confidence,
            "timeframe_details": timeframe_details,
            "direction_votes": {"bullish": weighted_bullish, "bearish": weighted_bearish, "neutral": 0},
            "timeframe_minutes": int(weighted_timeframe),
            "consensus_pct": consensus_pct,
            "bullish_count": bullish_count,
            "bearish_count": bearish_count
        }

    def train_neural_network(
        self,
        features: np.ndarray,
        target_return: float,
        target_volatility: float = None,
        sequence_override: Optional[np.ndarray] = None
    ):
        """Train the neural network with multi-task loss (return + direction + volatility)
        
        Uses Mixed Precision Training (AMP) when GPU is available for ~2x speedup.
        """
        if not self.training_enabled or self.model is None or self.optimizer is None:
            return None

        try:
            self.model.train()
            self.optimizer.zero_grad()

            # Prepare input data
            if sequence_override is not None:
                buf = np.asarray(sequence_override)
            elif len(self.feature_buffer) >= self.sequence_length:
                buf = np.vstack(self.feature_buffer[-self.sequence_length:])
            else:
                buf = None
            
            if buf is not None and buf.shape[0] >= self.sequence_length:
                # Enforce correct window length
                buf = buf[-self.sequence_length:]
                mu, sd = buf.mean(axis=0), buf.std(axis=0) + 1e-6
                cf = (features - mu) / sd
                seq = (buf - mu) / sd

                x = self._safe_to_device(torch.from_numpy(cf).float().unsqueeze(0))
                s = self._safe_to_device(torch.from_numpy(seq).float().unsqueeze(0))

                # Mixed Precision Training: Use autocast for forward pass when GPU available
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    # Forward pass
                    output = self.model(x, s)

                    # Multi-task loss calculation
                    total_loss = torch.tensor(0.0, device=self.device)
                    loss_components = {}

                    # 1. Return regression loss (MSE)
                    target_return_tensor = self._safe_to_device(torch.tensor([target_return], dtype=torch.float32))
                    return_loss = self.loss_fn(output["return"].squeeze(), target_return_tensor)
                    total_loss = total_loss + return_loss
                    loss_components['return'] = return_loss.item()

                    # 2. Direction classification loss (Cross-entropy) - CRITICAL for trading!
                    # Convert return to direction class with SMALLER thresholds for sensitivity
                    # 0=DOWN (<-0.2%), 1=NEUTRAL (-0.2% to +0.2%), 2=UP (>+0.2%)
                    if target_return < -0.002:
                        target_direction = 0  # DOWN
                    elif target_return > 0.002:
                        target_direction = 2  # UP
                    else:
                        target_direction = 1  # NEUTRAL
                    
                    target_dir_tensor = self._safe_to_device(torch.tensor([target_direction], dtype=torch.long))
                    direction_loss = nn.CrossEntropyLoss()(output["direction"], target_dir_tensor)
                    # INCREASED direction weight - getting direction right matters MORE than exact magnitude!
                    total_loss = total_loss + direction_loss * 2.0  # Was 0.5, now 2.0 (4x increase)
                    loss_components['direction'] = direction_loss.item()

                    # 3. Volatility regression loss (if available)
                    if target_volatility is not None:
                        target_vol_tensor = self._safe_to_device(torch.tensor([target_volatility], dtype=torch.float32))
                        vol_loss = self.loss_fn(output["volatility"].squeeze(), target_vol_tensor)
                        total_loss = total_loss + vol_loss * 0.3  # Weight volatility loss
                        loss_components['volatility'] = vol_loss.item()

                # Backward pass with AMP scaling (outside autocast)
                if self.use_amp and self.grad_scaler is not None:
                    self.grad_scaler.scale(total_loss).backward()
                    # Gradient clipping with scaler
                    self.grad_scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.grad_scaler.step(self.optimizer)
                    self.grad_scaler.update()
                else:
                    # Standard backward pass for CPU
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    
                    # Step the learning rate scheduler
                    if hasattr(self, 'scheduler') and self.scheduler is not None:
                        self.scheduler.step()

                logger.info(f"Multi-task training: total_loss={total_loss.item():.6f}, "
                           f"return={loss_components['return']:.6f}, "
                           f"direction={loss_components['direction']:.6f}, "
                           f"vol={loss_components.get('volatility', 0):.6f}")

                # Save model after training
                self.save_model()

                return total_loss.item()

        except Exception as e:
            logger.error(f"Neural network training error: {e}")
            return None

    def get_actual_price_for_prediction(self, prediction_time: str, symbol: str = "BITX") -> Optional[float]:
        """Get the actual price at the prediction time"""
        try:
            # Parse prediction time
            pred_time = datetime.strptime(prediction_time, "%Y-%m-%d %H:%M:%S")
            
            # Get recent data around that time
            data = self.safe_get_data(symbol, period="1d", interval="1m")
            if data.empty:
                return None
                
            # Find the closest price to the prediction time
            data_times = pd.to_datetime(data.index)
            time_diffs = abs(data_times - pred_time)
            closest_idx = time_diffs.argmin()  # Use argmin instead of idxmin for TimedeltaIndex
            closest_idx = data.index[closest_idx]
            
            # Try both capitalized and lowercase column names
            if 'Close' in data.columns:
                actual_price = float(data.loc[closest_idx, 'Close'])
            elif 'close' in data.columns:
                actual_price = float(data.loc[closest_idx, 'close'])
            else:
                raise KeyError("Neither 'Close' nor 'close' column found")
            
            logger.info(f"Found actual price ${actual_price:.2f} for prediction time {prediction_time}")
            return actual_price
            
        except Exception as e:
            logger.debug(f"Could not get actual price for {prediction_time}: {e}")  # Changed to debug level
            return None

    def continuous_learning_cycle(self):
        """Perform continuous learning from past predictions"""
        if not self.learning_enabled or not self.model_initialized:
            return
            
        try:
            # Get predictions that are ready for learning (>15 minutes old)
            # Use market time (Eastern) for consistency with stored timestamps
            market_time = self.get_market_time()
            cutoff_time = market_time - timedelta(minutes=15)
            
            conn = sqlite3.connect("data/unified_options_bot.db")
            cursor = conn.cursor()
            
            query = """
            SELECT timestamp, prediction_time, predicted_price_15min, current_price, confidence, features_blob, sequence_blob
            FROM unified_signals 
            WHERE prediction_time IS NOT NULL 
            AND datetime(prediction_time) <= datetime(?)
            AND predicted_price_15min IS NOT NULL
            ORDER BY timestamp DESC
            LIMIT ?
            """
            
            cursor.execute(query, [cutoff_time.strftime('%Y-%m-%d %H:%M:%S'), self.learning_batch_size])
            predictions = cursor.fetchall()
            conn.close()
            
            if not predictions:
                logger.info("No predictions ready for learning")
                return
                
            learning_count = 0
            total_loss = 0.0
            
            for pred in predictions:
                timestamp, prediction_time, predicted_price, current_price, confidence, features_blob, sequence_blob = pred
                
                # Get actual price at prediction time
                actual_price = self.get_actual_price_for_prediction(prediction_time)
                if actual_price is None:
                    continue
                    
                # Calculate actual return
                actual_return = (actual_price - current_price) / current_price
                predicted_return = (predicted_price - current_price) / current_price
                
                # Skip extreme actual returns (likely data errors)
                if abs(actual_return) > 0.20:  # Skip >20% moves (likely errors)
                    logger.warning(f"Skipping extreme actual return: {actual_return:.4f}")
                    continue
                
                # Use the snapshot of features from decision time.
                # This prevents a serious bug: training on "current" (future) features
                # while labeling with past outcomes corrupts the online learner.
                features = None
                if features_blob:
                    try:
                        features = pickle.loads(features_blob)
                    except Exception as e:
                        logger.debug(f"Could not load features_blob for learning sample: {e}")
                        features = None
                
                sequence_snapshot = None
                if sequence_blob:
                    try:
                        sequence_snapshot = pickle.loads(sequence_blob)
                    except Exception as e:
                        logger.debug(f"Could not load sequence_blob for learning sample: {e}")
                        sequence_snapshot = None
                
                # Backward-compatible fallback for older DB rows (no features snapshot)
                if features is None:
                    data = self.safe_get_data("BITX", period="1d", interval="1m")
                    if not data.empty:
                        features = self.create_features(data)
                    
                    # Store experience for replay learning
                    experience = {
                        'features': features.copy(),
                        'sequence': sequence_snapshot.copy() if sequence_snapshot is not None else None,
                        'actual_return': actual_return,
                        'predicted_return': predicted_return,
                        'actual_volatility': abs(actual_return),  # Simple volatility approximation
                        'timestamp': timestamp
                    }
                    self.experience_buffer.append(experience)
                    
                    # Keep experience buffer size manageable
                    if len(self.experience_buffer) > self.max_experience_size:
                        self.experience_buffer.pop(0)
                    
                    # Train with experience replay (sample diverse experiences)
                    replay_experiences = self._sample_diverse_experiences(min(5, len(self.experience_buffer)))
                    
                    for exp in replay_experiences:
                        loss = self.train_neural_network(
                            exp['features'],
                            exp['actual_return'],
                            exp['actual_volatility'],
                            sequence_override=exp.get('sequence')
                        )
                        if loss is not None:
                            total_loss += loss
                            learning_count += 1
                    
                    # Track prediction accuracy for monitoring
                    self._track_prediction_accuracy(predicted_return, actual_return, confidence)
                    
                    logger.info(f"Experience replay: {len(replay_experiences)} samples, "
                               f"predicted={predicted_return:.4f}, actual={actual_return:.4f}")
            
            if learning_count > 0:
                avg_loss = total_loss / learning_count
                logger.info(f"Continuous learning completed: {learning_count} samples, avg_loss={avg_loss:.6f}")
                
                # Update learning time (use market time for consistency)
                self.last_learning_time = self.get_market_time()
            else:
                logger.info("No valid samples for learning this cycle")
            
            # 🆕 Train execution quality prediction heads
            # Runs alongside strategy learning to keep fillability predictor fresh
            self.train_execution_heads(batch_size=10)
                
        except Exception as e:
            logger.error(f"Error in continuous learning cycle: {e}")

    def _sample_diverse_experiences(self, num_samples: int) -> List[Dict]:
        """Sample diverse experiences for replay learning to prevent catastrophic forgetting"""
        if len(self.experience_buffer) <= num_samples:
            return self.experience_buffer.copy()
        
        try:
            import random
            
            # Strategy: Sample from different time periods and return ranges
            experiences = self.experience_buffer.copy()
            
            # Sort by return magnitude to ensure diversity
            experiences.sort(key=lambda x: abs(x['actual_return']))
            
            # Sample from different quartiles to get diverse experiences
            quartile_size = len(experiences) // 4
            samples = []
            
            # Sample from each quartile
            for i in range(4):
                start_idx = i * quartile_size
                end_idx = (i + 1) * quartile_size if i < 3 else len(experiences)
                quartile_samples = experiences[start_idx:end_idx]
                
                if quartile_samples:
                    num_from_quartile = max(1, num_samples // 4)
                    samples.extend(random.sample(quartile_samples, min(num_from_quartile, len(quartile_samples))))
            
            # Fill remaining slots with random samples
            while len(samples) < num_samples and len(samples) < len(experiences):
                remaining = []
                for exp in experiences:
                    # Fix array comparison issue
                    is_in_samples = False
                    for sample in samples:
                        try:
                            if np.array_equal(exp['features'], sample['features']):
                                is_in_samples = True
                                break
                        except (KeyError, TypeError, ValueError):
                            # Features might be missing or incompatible types
                            continue
                    if not is_in_samples:
                        remaining.append(exp)
                
                if remaining:
                    samples.append(random.choice(remaining))
                else:
                    break
            
            return samples[:num_samples]
            
        except Exception as e:
            logger.error(f"Error sampling diverse experiences: {e}")
            # Fallback: return random sample
            import random
            return random.sample(self.experience_buffer, min(num_samples, len(self.experience_buffer)))

    def _track_prediction_accuracy(self, predicted_return: float, actual_return: float, confidence: float, 
                                     horizon_minutes: int = None):
        """
        Track prediction accuracy for monitoring and calibration.
        
        IMPORTANT: For calibration, we track DIRECTION accuracy (not return error).
        This is what matters for trading decisions - did we get the direction right?
        
        Args:
            predicted_return: The predicted return (e.g., 0.005 = 0.5%)
            actual_return: The actual return that occurred
            confidence: The model's confidence (0-1)
            horizon_minutes: Which horizon this prediction was for (default: PRIMARY_HORIZON)
        """
        try:
            horizon = horizon_minutes or self.PRIMARY_HORIZON_MINUTES
            
            # Direction accuracy - this is what matters for trading!
            # Use small threshold (0.1%) to filter out noise near zero
            predicted_direction = 1 if predicted_return > 0.001 else (-1 if predicted_return < -0.001 else 0)
            actual_direction = 1 if actual_return > 0.001 else (-1 if actual_return < -0.001 else 0)
            
            direction_correct = (predicted_direction == actual_direction)
            
            # Update global direction stats
            self.accuracy_tracker['direction_correct'] += int(direction_correct)
            self.accuracy_tracker['direction_total'] += 1
            
            # Update per-horizon stats
            if horizon in self.accuracy_tracker['horizon_stats']:
                self.accuracy_tracker['horizon_stats'][horizon]['total'] += 1
                if direction_correct:
                    self.accuracy_tracker['horizon_stats'][horizon]['correct'] += 1
            
            # CALIBRATION: Only add to calibration buffer for PRIMARY_HORIZON
            # This ensures Platt scaling is trained on consistent data
            if horizon == self.PRIMARY_HORIZON_MINUTES:
                # Store (confidence, direction_correct) - NOT return error!
                self.accuracy_tracker['confidence_calibration'].append((confidence, direction_correct))
                
                # Keep rolling buffer at configured size
                if len(self.accuracy_tracker['confidence_calibration']) > self.CALIBRATION_BUFFER_SIZE:
                    self.accuracy_tracker['confidence_calibration'] = \
                        self.accuracy_tracker['confidence_calibration'][-self.CALIBRATION_BUFFER_SIZE:]
            
            # Track recent errors for monitoring (all horizons)
            return_error = abs(predicted_return - actual_return)
            self.accuracy_tracker['recent_errors'].append(return_error)
            if len(self.accuracy_tracker['recent_errors']) > 100:
                self.accuracy_tracker['recent_errors'] = self.accuracy_tracker['recent_errors'][-100:]
            
            # Log accuracy stats periodically
            if self.accuracy_tracker['direction_total'] % 20 == 0:
                direction_accuracy = self.accuracy_tracker['direction_correct'] / self.accuracy_tracker['direction_total']
                avg_error = np.mean(self.accuracy_tracker['recent_errors']) if self.accuracy_tracker['recent_errors'] else 0
                
                # Log per-horizon breakdown
                horizon_info = []
                for h, stats in self.accuracy_tracker['horizon_stats'].items():
                    if stats['total'] > 0:
                        h_acc = stats['correct'] / stats['total']
                        horizon_info.append(f"{h}m:{h_acc:.0%}")
                
                logger.info(f"📊 Direction accuracy: {direction_accuracy:.1%} | "
                           f"By horizon: {', '.join(horizon_info)} | "
                           f"Avg error: {avg_error:.4f}")
                
                # Log calibration stats every 50 samples
                if self.accuracy_tracker['direction_total'] % 50 == 0:
                    cal_buffer = self.accuracy_tracker['confidence_calibration']
                    if len(cal_buffer) >= 50:
                        cal_acc = sum(1 for _, correct in cal_buffer if correct) / len(cal_buffer)
                        logger.info(f"📐 Calibration buffer ({self.PRIMARY_HORIZON_MINUTES}m): "
                                   f"{len(cal_buffer)} samples, {cal_acc:.1%} direction accuracy")
                
        except Exception as e:
            logger.error(f"Error tracking prediction accuracy: {e}")

    def get_calibrated_confidence(self, raw_confidence: float) -> float:
        """
        Calibrate confidence using Platt scaling.
        
        Uses CalibrationTracker (preferred) or falls back to legacy buffer.
        Platt scaling fits: P(direction_correct) = 1 / (1 + exp(A*conf + B))
        
        Args:
            raw_confidence: The model's raw confidence output (0-1)
            
        Returns:
            Calibrated confidence based on Platt scaling
        """
        # Use CalibrationTracker if available (preferred)
        if self.calibration_tracker is not None:
            return self.calibration_tracker.calibrate(raw_confidence)
        
        # Fallback to legacy approach
        try:
            calibration_data = self.accuracy_tracker.get('confidence_calibration', [])
            
            # Need enough data points to fit Platt scaling
            min_samples = 50
            if len(calibration_data) < min_samples:
                return raw_confidence  # Not enough data, return raw
            
            # Check if we need to refit Platt parameters
            # Refit every 50 new samples or if not yet fitted
            platt_params = getattr(self, '_platt_params', None)
            last_fit_count = getattr(self, '_platt_last_fit_count', 0)
            
            if platt_params is None or len(calibration_data) - last_fit_count >= 50:
                platt_params = self._fit_platt_scaling(calibration_data)
                if platt_params:
                    self._platt_params = platt_params
                    self._platt_last_fit_count = len(calibration_data)
            
            # Apply Platt scaling if we have fitted parameters
            if platt_params:
                A, B = platt_params['A'], platt_params['B']
                z = A * raw_confidence + B
                # Sigmoid with clipping to avoid overflow
                calibrated = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
                calibrated = float(calibrated)
                
                # Log significant calibration adjustments
                if abs(calibrated - raw_confidence) > 0.1:
                    logger.debug(f"📐 Platt calibration: {raw_confidence:.1%} → {calibrated:.1%}")
                
                return calibrated
            
            # Fallback to bucket-based calibration if Platt fitting failed
            return self._bucket_calibration(raw_confidence, calibration_data)
            
        except Exception as e:
            logger.debug(f"Calibration error (using raw): {e}")
            return raw_confidence
    
    def _fit_platt_scaling(self, calibration_data: list) -> Optional[Dict]:
        """
        Fit Platt scaling parameters using gradient descent.
        
        Learns A, B such that: P(correct|confidence) = sigmoid(A * confidence + B)
        
        Args:
            calibration_data: List of (confidence, was_correct) tuples
            
        Returns:
            Dict with 'A' and 'B' parameters, or None if fitting failed
        """
        try:
            if len(calibration_data) < 50:
                return None
            
            # Extract arrays
            confidences = np.array([c for c, _ in calibration_data], dtype=np.float64)
            outcomes = np.array([1.0 if w else 0.0 for _, w in calibration_data], dtype=np.float64)
            
            # Initialize parameters
            A, B = -1.0, 0.0
            
            # Gradient descent with line search
            learning_rate = 0.1
            max_iters = 500
            tol = 1e-6
            
            for _ in range(max_iters):
                # Compute predictions
                z = A * confidences + B
                p = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
                
                # Compute gradients (negative log-likelihood)
                error = p - outcomes
                grad_A = np.dot(error, confidences) / len(confidences)
                grad_B = np.mean(error)
                
                # Update parameters
                A_new = A - learning_rate * grad_A
                B_new = B - learning_rate * grad_B
                
                # Check convergence
                if abs(A_new - A) < tol and abs(B_new - B) < tol:
                    break
                
                A, B = A_new, B_new
            
            logger.info(f"✅ Platt scaling fitted: A={A:.4f}, B={B:.4f} (n={len(calibration_data)})")
            return {'A': float(A), 'B': float(B), 'n_samples': len(calibration_data)}
            
        except Exception as e:
            logger.warning(f"Platt scaling fit failed: {e}")
            return None
    
    def _bucket_calibration(self, raw_confidence: float, calibration_data: list) -> float:
        """Fallback bucket-based calibration if Platt fails."""
        bucket_size = 0.1
        bucket_idx = max(0, min(9, int(raw_confidence / bucket_size)))
        bucket_min = bucket_idx * bucket_size
        bucket_max = (bucket_idx + 1) * bucket_size
        
        nearby_preds = [
            was_correct for conf, was_correct in calibration_data
            if bucket_min - 0.1 <= conf <= bucket_max + 0.1
        ]
        
        if len(nearby_preds) < 10:
            return raw_confidence
        
        # Simple empirical calibration
        return sum(nearby_preds) / len(nearby_preds)
    
    def get_calibration_metrics(self) -> Dict:
        """
        Calculate Brier score and ECE for DIRECTION-based calibration.
        
        Uses CalibrationTracker (preferred) or falls back to legacy buffer.
        
        Brier Score: Mean squared error of probability predictions (lower = better)
            - 0 = perfect predictions
            - 0.25 = random guessing (no skill)
            - 0.35 = our threshold for "acceptable"
            
        ECE (Expected Calibration Error): How well confidence matches actual accuracy
            - 0 = perfectly calibrated (70% confidence = 70% accuracy)
            - 0.15 = our threshold for "well calibrated"
        
        Returns:
            Dict with brier_score, ece, accuracy, sample_count, is_calibrated
        """
        # Use CalibrationTracker if available (preferred)
        if self.calibration_tracker is not None:
            return self.calibration_tracker.get_metrics()
        
        # Fallback to legacy approach
        try:
            calibration_data = self.accuracy_tracker.get('confidence_calibration', [])
            
            if len(calibration_data) < 20:
                return {
                    'brier_score': 1.0,
                    'ece': 1.0,
                    'accuracy': 0.0,
                    'sample_count': len(calibration_data),
                    'is_calibrated': False,
                    'horizon_minutes': self.PRIMARY_HORIZON_MINUTES,
                    'target': 'direction',
                    'buffer_max': self.CALIBRATION_BUFFER_SIZE,
                    'platt_params': None
                }
            
            # Use calibrated confidences for Brier/ECE calculation
            confidences = []
            outcomes = []
            for conf, was_correct in calibration_data:
                cal_conf = self.get_calibrated_confidence(conf)
                confidences.append(cal_conf)
                outcomes.append(1.0 if was_correct else 0.0)
            
            # Brier Score: mean squared error of probability predictions
            # Lower is better. 0 = perfect, 0.25 = random guessing
            brier = sum((c - o) ** 2 for c, o in zip(confidences, outcomes)) / len(confidences)
            
            # Expected Calibration Error (ECE)
            # Measures how well confidence matches actual accuracy per bucket
            n_bins = 10
            bin_counts = [0] * n_bins
            bin_correct = [0] * n_bins
            bin_confidence = [0.0] * n_bins
            
            for conf, outcome in zip(confidences, outcomes):
                bin_idx = min(int(conf * n_bins), n_bins - 1)
                bin_counts[bin_idx] += 1
                bin_correct[bin_idx] += outcome
                bin_confidence[bin_idx] += conf
            
            ece = 0.0
            total = len(confidences)
            for i in range(n_bins):
                if bin_counts[i] > 0:
                    avg_conf = bin_confidence[i] / bin_counts[i]
                    avg_acc = bin_correct[i] / bin_counts[i]
                    ece += (bin_counts[i] / total) * abs(avg_conf - avg_acc)
            
            # Overall accuracy
            accuracy = sum(outcomes) / len(outcomes)
            
            # Is calibration good enough for gating?
            is_calibrated = brier <= 0.35 and ece <= 0.15 and len(calibration_data) >= 50
            
            return {
                'brier_score': float(brier),
                'ece': float(ece),
                'accuracy': float(accuracy),  # Direction accuracy
                'sample_count': len(calibration_data),
                'is_calibrated': is_calibrated,
                'platt_params': getattr(self, '_platt_params', None),
                'horizon_minutes': self.PRIMARY_HORIZON_MINUTES,
                'target': 'direction',  # Clarify this is direction-based
                'buffer_max': self.CALIBRATION_BUFFER_SIZE
            }
            
        except Exception as e:
            logger.warning(f"Error calculating calibration metrics: {e}")
            return {
                'brier_score': 1.0,
                'ece': 1.0,
                'accuracy': 0.0,
                'sample_count': 0,
                'is_calibrated': False
            }
    
    def get_calibration_stats(self) -> Dict:
        """
        Get DIRECTION-based confidence calibration statistics by bucket.
        
        The calibration buffer stores (confidence, direction_correct) pairs for the
        PRIMARY_HORIZON only. This shows how well model confidence predicts direction accuracy.
        
        Returns:
            Dict with status, buckets (confidence range -> actual direction accuracy), total_samples
        """
        try:
            calibration_data = self.accuracy_tracker.get('confidence_calibration', [])
            
            if len(calibration_data) < 20:
                return {
                    'status': 'insufficient_data', 
                    'samples': len(calibration_data),
                    'horizon': self.PRIMARY_HORIZON_MINUTES,
                    'target': 'direction'
                }
            
            # Bin by confidence bucket
            buckets = {}
            for conf, direction_correct in calibration_data:
                bucket = f"{int(conf * 10) * 10}-{int(conf * 10) * 10 + 10}%"
                if bucket not in buckets:
                    buckets[bucket] = {'correct': 0, 'total': 0}
                buckets[bucket]['total'] += 1
                if direction_correct:
                    buckets[bucket]['correct'] += 1
            
            # Calculate direction accuracy per bucket
            stats = {}
            for bucket, data in sorted(buckets.items()):
                if data['total'] >= 5:  # Need at least 5 samples
                    accuracy = data['correct'] / data['total']
                    expected_conf = (int(bucket.split('-')[0]) + 5) / 100
                    stats[bucket] = {
                        'model_confidence': bucket,
                        'actual_direction_accuracy': f"{accuracy:.1%}",
                        'samples': data['total'],
                        # Well calibrated if actual accuracy is within 15% of stated confidence
                        'well_calibrated': abs(accuracy - expected_conf) < 0.15
                    }
            
            # Calculate overall direction accuracy
            total_correct = sum(1 for _, c in calibration_data if c)
            overall_accuracy = total_correct / len(calibration_data) if calibration_data else 0
            
            return {
                'status': 'ok', 
                'buckets': stats, 
                'total_samples': len(calibration_data),
                'overall_direction_accuracy': overall_accuracy,
                'horizon': self.PRIMARY_HORIZON_MINUTES,
                'target': 'direction'
            }
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}

    def get_prediction_consistency(self, lookback: int = 5) -> Dict:
        """
        Analyze recent predictions for temporal consistency.
        
        Returns:
            Dict with:
            - direction_consistency: 0-1 (how much recent predictions agree on direction)
            - avg_return: Average predicted return over lookback period
            - avg_confidence: Average confidence over lookback period  
            - trend: 'strengthening', 'weakening', or 'stable'
            - consensus_direction: 'BULLISH', 'BEARISH', or 'NEUTRAL'
        """
        try:
            if len(self.prediction_history) < 2:
                return {
                    'direction_consistency': 0.5,
                    'avg_return': 0.0,
                    'avg_confidence': 0.5,
                    'trend': 'stable',
                    'consensus_direction': 'NEUTRAL',
                    'samples': 0
                }
            
            # Get recent predictions
            recent = self.prediction_history[-lookback:] if len(self.prediction_history) >= lookback else self.prediction_history
            
            # Extract returns and confidences
            returns = []
            confidences = []
            for pred in recent:
                if 'predicted_return' in pred:
                    returns.append(pred['predicted_return'])
                if 'confidence' in pred:
                    confidences.append(pred['confidence'])
            
            if not returns:
                return {
                    'direction_consistency': 0.5,
                    'avg_return': 0.0,
                    'avg_confidence': 0.5,
                    'trend': 'stable',
                    'consensus_direction': 'NEUTRAL',
                    'samples': 0
                }
            
            # Calculate direction consistency (how many agree on direction)
            bullish_count = sum(1 for r in returns if r > 0.002)  # > 0.2%
            bearish_count = sum(1 for r in returns if r < -0.002)  # < -0.2%
            neutral_count = len(returns) - bullish_count - bearish_count
            
            max_count = max(bullish_count, bearish_count, neutral_count)
            direction_consistency = max_count / len(returns) if returns else 0.5
            
            # Determine consensus direction
            if bullish_count > bearish_count and bullish_count > neutral_count:
                consensus_direction = 'BULLISH'
            elif bearish_count > bullish_count and bearish_count > neutral_count:
                consensus_direction = 'BEARISH'
            else:
                consensus_direction = 'NEUTRAL'
            
            # Calculate averages
            avg_return = np.mean(returns)
            avg_confidence = np.mean(confidences) if confidences else 0.5
            
            # Detect trend (are predictions getting stronger in one direction?)
            if len(returns) >= 3:
                first_half = np.mean(returns[:len(returns)//2])
                second_half = np.mean(returns[len(returns)//2:])
                
                if abs(second_half) > abs(first_half) * 1.2:
                    trend = 'strengthening'
                elif abs(second_half) < abs(first_half) * 0.8:
                    trend = 'weakening'
                else:
                    trend = 'stable'
            else:
                trend = 'stable'
            
            return {
                'direction_consistency': direction_consistency,
                'avg_return': float(avg_return),
                'avg_confidence': float(avg_confidence),
                'trend': trend,
                'consensus_direction': consensus_direction,
                'samples': len(returns)
            }
            
        except Exception as e:
            logger.debug(f"Prediction consistency error: {e}")
            return {
                'direction_consistency': 0.5,
                'avg_return': 0.0,
                'avg_confidence': 0.5,
                'trend': 'stable',
                'consensus_direction': 'NEUTRAL',
                'samples': 0
            }

    def _monitor_system_resources(self):
        """Monitor system resources in background thread"""
        import time
        
        if not PSUTIL_AVAILABLE:
            return
            
        while True:
            try:
                # Get current process
                try:
                    process = psutil.Process()
                except AttributeError:
                    # Handle case where psutil.Process is not available
                    logger.warning("psutil.Process not available, skipping detailed process monitoring")
                    time.sleep(10)
                    continue
                
                # System-wide metrics
                self.system_monitor['cpu_percent'] = psutil.cpu_percent(interval=1)
                
                # Memory metrics
                memory = psutil.virtual_memory()
                self.system_monitor['memory_percent'] = memory.percent
                self.system_monitor['memory_used_gb'] = memory.used / (1024**3)
                self.system_monitor['memory_total_gb'] = memory.total / (1024**3)
                
                # Process-specific metrics
                self.system_monitor['process_cpu_percent'] = process.cpu_percent()
                self.system_monitor['process_memory_mb'] = process.memory_info().rss / (1024**2)
                
                # GPU metrics (if available)
                if GPUTIL_AVAILABLE:
                    try:
                        gpus = GPUtil.getGPUs()
                        if gpus:
                            gpu = gpus[0]  # Use first GPU
                            self.system_monitor['gpu_available'] = True
                            self.system_monitor['gpu_percent'] = gpu.load * 100
                            self.system_monitor['gpu_memory_percent'] = gpu.memoryUtil * 100
                            self.system_monitor['gpu_memory_used_gb'] = gpu.memoryUsed / 1024
                            self.system_monitor['gpu_memory_total_gb'] = gpu.memoryTotal / 1024
                        else:
                            self.system_monitor['gpu_available'] = False
                    except Exception:
                        self.system_monitor['gpu_available'] = False
                else:
                    self.system_monitor['gpu_available'] = False
                
                self.system_monitor['last_update'] = self.get_market_time()
                
                # Save system status to file for dashboard access
                try:
                    import json
                    status_data = {
                        'system_monitor': self.system_monitor.copy(),
                        'timestamp': self.system_monitor['last_update'].isoformat()
                    }
                    # Convert datetime to string for JSON serialization
                    status_data['system_monitor']['last_update'] = status_data['system_monitor']['last_update'].isoformat()
                    
                    with open('bot_status.json', 'w') as f:
                        json.dump(status_data, f, indent=2)
                except Exception as e:
                    logger.error(f"Error saving system status: {e}")
                
                # Sleep for 5 seconds before next update
                time.sleep(5)
                
            except Exception as e:
                logger.error(f"Error monitoring system resources: {e}")
                time.sleep(10)  # Wait longer on error

    def get_system_status(self) -> Dict:
        """Get current system resource usage"""
        return self.system_monitor.copy()

    def update_prediction_with_actual(self, prediction_time: str, actual_price: float):
        """Update a prediction with the actual outcome for training"""
        try:
            # Find the prediction by time
            for pred in self.prediction_history:
                if pred['prediction_time'] == prediction_time:
                    # Calculate actual return and volatility
                    actual_return = (
                        actual_price - pred['current_price']) / pred['current_price']
                    # Simple volatility approximation
                    actual_vol = abs(actual_return)

                    # Update the prediction record
                    pred['actual_price'] = actual_price
                    pred['actual_return'] = actual_return
                    pred['actual_volatility'] = actual_vol
                    pred['accuracy'] = 1.0 - \
                        abs(actual_return - pred['predicted_return'])

                    logger.info(
                        f"Updated prediction: actual_return={actual_return:.4f}, accuracy={pred['accuracy']:.4f}")
                    return True

            logger.warning(f"No prediction found for time {prediction_time}")
            return False

        except Exception as e:
            logger.error(f"Error updating prediction: {e}")
            return False

    def save_model(self):
        """Save complete model state for seamless restarts"""
        try:
            # Ensure model state directory exists
            import os
            os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)
            
            saved_components = []
            
            if self.model is not None and self.model_initialized:
                # Save model state
                torch.save(self.model.state_dict(), self.model_save_path)
                saved_components.append("model weights")

                # ALSO export a predictor checkpoint for ArchV2 / PredictorManager.
                # This prevents the "random predictor" regression when switching predictor.arch.
                try:
                    cfg = self.config.config if self.config else {}
                    predictor_ckpt_path = (
                        cfg.get("architecture", {})
                           .get("predictor", {})
                           .get("checkpoint_path", "models/predictor_v2.pt")
                    )
                    os.makedirs(os.path.dirname(predictor_ckpt_path) if os.path.dirname(predictor_ckpt_path) else ".", exist_ok=True)
                    torch.save(
                        {
                            "model_state_dict": self.model.state_dict(),
                            "config": {
                                "feature_dim": self.feature_dim,
                                "sequence_length": self.sequence_length,
                                "predictor_arch": (
                                    cfg.get("architecture", {})
                                       .get("predictor", {})
                                       .get("arch", "v1_original")
                                ),
                            },
                        },
                        predictor_ckpt_path,
                    )
                    saved_components.append(f"predictor checkpoint ({predictor_ckpt_path})")

                    # Also write/overwrite a stable "latest" checkpoint for live usage.
                    # This makes ArchV2 (default config checkpoint_path) work out-of-the-box.
                    latest_ckpt_path = os.path.join("models", "predictor_v2.pt")
                    if os.path.normpath(latest_ckpt_path) != os.path.normpath(predictor_ckpt_path):
                        os.makedirs(os.path.dirname(latest_ckpt_path), exist_ok=True)
                        torch.save(
                            {
                                "model_state_dict": self.model.state_dict(),
                                "config": {
                                    "feature_dim": self.feature_dim,
                                    "sequence_length": self.sequence_length,
                                    "predictor_arch": (
                                        cfg.get("architecture", {})
                                           .get("predictor", {})
                                           .get("arch", "v1_original")
                                    ),
                                },
                            },
                            latest_ckpt_path,
                        )
                        saved_components.append(f"predictor latest ({latest_ckpt_path})")
                except Exception as e:
                    logger.warning(f"Could not export predictor checkpoint: {e}")

                # Save optimizer state
                if self.optimizer is not None:
                    torch.save(self.optimizer.state_dict(), self.optimizer_save_path)
                    saved_components.append("optimizer state")
                
                # Save scheduler state
                if hasattr(self, 'scheduler') and self.scheduler is not None:
                    scheduler_path = self.optimizer_save_path.replace('optimizer', 'scheduler')
                    torch.save(self.scheduler.state_dict(), scheduler_path)
                    saved_components.append("scheduler state")

            # Save feature buffer
            if len(self.feature_buffer) > 0:
                with open(self.feature_buffer_save_path, 'wb') as f:
                    pickle.dump(self.feature_buffer, f)
                saved_components.append("feature buffer")

            # Save Gaussian processor state
            if self.gaussian_processor.is_fitted:
                with open(self.gaussian_processor_save_path, 'wb') as f:
                    pickle.dump(self.gaussian_processor, f)
                saved_components.append("gaussian processor")

            # Save learning state
            learning_state = {
                'last_learning_time': self.last_learning_time,
                'learning_enabled': self.learning_enabled,
                'learning_batch_size': self.learning_batch_size,
                'learning_interval': self.learning_interval,
                'model_initialized': self.model_initialized,
                'save_timestamp': self.get_market_time()
            }
            with open(self.learning_state_save_path, 'wb') as f:
                pickle.dump(learning_state, f)
            saved_components.append("learning state")

            # Save recent prediction history (keep last 50 for context)
            if len(self.prediction_history) > 0:
                recent_history = self.prediction_history[-50:]  # Keep last 50
                with open(self.prediction_history_save_path, 'wb') as f:
                    pickle.dump(recent_history, f)
                saved_components.append("prediction history")

            if saved_components:
                logger.info(f"💾 Comprehensive save completed: {', '.join(saved_components)}")
                return True
            else:
                logger.warning("No components to save")
                return False
                
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False

    def load_model(self):
        """Load complete model state for seamless restart"""
        try:
            loaded_components = []
            
            # Try to load BEST model first (from training)
            best_model_path = 'models/best/neural_network_full.pth'
            model_path_to_use = None
            
            if os.path.exists(best_model_path):
                model_path_to_use = best_model_path
                logger.info("🏆 Loading BEST trained model from training runs")
            elif os.path.exists(self.model_save_path):
                model_path_to_use = self.model_save_path
                logger.info("📂 Loading model from default location")
            
            # Load model state
            if model_path_to_use and self.model is not None:
                self.model.load_state_dict(torch.load(model_path_to_use, map_location=self.device))
                loaded_components.append(f"model weights from {model_path_to_use}")

                # Load optimizer state
                if os.path.exists(self.optimizer_save_path) and self.optimizer is not None:
                    self.optimizer.load_state_dict(torch.load(self.optimizer_save_path, map_location=self.device))
                    loaded_components.append("optimizer state")
                
                # Load scheduler state
                scheduler_path = self.optimizer_save_path.replace('optimizer', 'scheduler')
                if hasattr(self, 'scheduler') and self.scheduler is not None and os.path.exists(scheduler_path):
                    try:
                        self.scheduler.load_state_dict(torch.load(scheduler_path, map_location=self.device))
                        loaded_components.append("scheduler state")
                    except Exception as e:
                        logger.warning(f"Could not load scheduler state: {e}, using fresh scheduler")

            # Load feature buffer
            if os.path.exists(self.feature_buffer_save_path):
                with open(self.feature_buffer_save_path, 'rb') as f:
                    self.feature_buffer = pickle.load(f)
                loaded_components.append(f"feature buffer ({len(self.feature_buffer)} features)")

            # Load Gaussian processor state
            if os.path.exists(self.gaussian_processor_save_path):
                with open(self.gaussian_processor_save_path, 'rb') as f:
                    self.gaussian_processor = pickle.load(f)
                loaded_components.append("gaussian processor")

            # Load learning state
            if os.path.exists(self.learning_state_save_path):
                with open(self.learning_state_save_path, 'rb') as f:
                    learning_state = pickle.load(f)
                
                # Restore learning state
                saved_learning_time = learning_state.get('last_learning_time', self.get_market_time())
                
                # Ensure learning time is timezone-naive (to match get_market_time())
                if saved_learning_time.tzinfo is not None:
                    # Convert timezone-aware datetime to naive
                    self.last_learning_time = saved_learning_time.replace(tzinfo=None)
                else:
                    self.last_learning_time = saved_learning_time
                self.learning_enabled = learning_state.get('learning_enabled', True)
                self.learning_batch_size = learning_state.get('learning_batch_size', 5)
                self.learning_interval = learning_state.get('learning_interval', 60)
                
                save_time = learning_state.get('save_timestamp', self.get_market_time())
                time_since_save = (self.get_market_time() - save_time).total_seconds()
                loaded_components.append(f"learning state (saved {time_since_save:.0f}s ago)")

            # Load prediction history
            if os.path.exists(self.prediction_history_save_path):
                with open(self.prediction_history_save_path, 'rb') as f:
                    self.prediction_history = pickle.load(f)
                loaded_components.append(f"prediction history ({len(self.prediction_history)} records)")

            if loaded_components:
                logger.info(f"📂 Comprehensive load completed: {', '.join(loaded_components)}")
                logger.info("🔄 Bot will resume exactly where it left off")
                return True
            else:
                logger.info("📂 No saved state found - starting fresh")
                return False
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.warning("🔄 Continuing with fresh state")
            return False

    def load_feature_buffer_only(self):
        """Load just the feature buffer on startup to resume from where we left off"""
        try:
            if os.path.exists(self.feature_buffer_save_path):
                with open(self.feature_buffer_save_path, 'rb') as f:
                    self.feature_buffer = pickle.load(f)
                logger.info(
                    f"Feature buffer loaded: {len(self.feature_buffer)}/{self.sequence_length} features ready")
                return True
        except Exception as e:
            logger.error(f"Error loading feature buffer: {e}")
        return False

    def generate_15min_prediction(self, current_price: float, neural_pred: Optional[Dict], symbol: str) -> float:
        # Use network return if available; otherwise momentum fallback on correct symbol
        try:
            if neural_pred and "predicted_return" in neural_pred:
                predicted_return = neural_pred["predicted_return"]
                
                # Log extreme predictions for monitoring (but don't bound them - let learning fix it)
                if abs(predicted_return) > 0.10:
                    logger.warning(f"Extreme neural prediction: {predicted_return:.4f} ({predicted_return*100:.1f}%) - continuous learning will improve this")
                
                return float(current_price * (1.0 + predicted_return))
                
            data = self.safe_get_data(
                symbol, period="1d", interval="1m")
            if not data.empty and len(data) >= 15:
                # Normalize column names to lowercase
                data.columns = [col.lower() for col in data.columns]
                prices = data["close"].tail(15)
                momentum = (prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0]
                return float(current_price * (1.0 + momentum * 0.1))
            return float(current_price)
        except Exception as e:
            logger.error(f"15-minute prediction error: {e}")
            return float(current_price)

    # ------------------------ Signal & execution ------------------------
    def _combine_signals(self, options_metrics: Dict, neural_pred: Optional[Dict], current_price: float, multi_timeframe_signal: Optional[Dict] = None) -> Dict:
        signal = {
            "action": "HOLD",
            "strategy": "WAIT",
            "confidence": 0.5,
            # Important: downstream training / RL expects these fields to exist.
            # If they are missing, training gates may treat edge as 0 and skip nearly everything.
            "predicted_return": float(neural_pred.get("predicted_return", 0.0)) if neural_pred else 0.0,
            "predicted_volatility": float(neural_pred.get("predicted_volatility", 0.0)) if neural_pred else 0.0,
            "neural_confidence": float(neural_pred.get("neural_confidence", 0.0)) if neural_pred else 0.0,
            # Execution heads (world-model realism)
            "fillability": float(neural_pred.get("fillability_mean", 0.5)) if neural_pred else 0.5,
            "exp_slippage": float(neural_pred.get("exp_slippage_mean", 0.0)) if neural_pred else 0.0,
            "exp_ttf": float(neural_pred.get("exp_ttf_mean", 0.0)) if neural_pred else 0.0,
            # Prediction uncertainty (helps downstream gating / analysis)
            "prediction_uncertainty": float(
                ((neural_pred or {}).get("monte_carlo_stats") or {}).get("return_std", 0.0)
            ) if neural_pred else 0.0,
            "momentum_5min": float(options_metrics.get("momentum_5min", 0.0) or 0.0),
            "momentum_15min": float(options_metrics.get("momentum_15min", 0.0) or 0.0),
            "volume_spike": float(options_metrics.get("volume_spike", 1.0) or 1.0),
            "reasoning": [],
            "risk_level": "MEDIUM",
            "current_price": float(current_price),
            # Include HMM state for exit decisions
            "hmm_state": self.current_hmm_regime.get('state') if self.current_hmm_regime else None,
            "hmm_confidence": self.current_hmm_regime.get('confidence', 0) if self.current_hmm_regime else 0,
            "hmm_trend": self.current_hmm_regime.get('trend') if self.current_hmm_regime else None,
        }

        components: List[float] = []
        vol_pct = options_metrics["volatility_percentile"]
        mom15 = options_metrics["momentum_15min"]
        vol_spike = options_metrics["volume_spike"]
        
        # Debug logging to understand signal generation and adaptive thresholds
        logger.info(f"🔍 Signal conditions: vol_pct={vol_pct:.1f}%, mom15={mom15:.4f}, vol_spike={vol_spike:.2f}x")
        logger.info(f"🎯 Adaptive thresholds: neural_conf≥{self.neural_confidence_threshold:.1%}, return≥{self.neural_return_threshold:.1%}, momentum≥{self.momentum_threshold:.1%}")
        if neural_pred:
            logger.info(f"🧠 Neural: confidence={neural_pred['neural_confidence']:.3f}, return={neural_pred['predicted_return']:.4f}")

        # Volatility regimes (adjusted for normal market conditions)
        if vol_pct < 30.0:  # Lowered from 20% to 30%
            # Check HMM regime - avoid straddles if we are deeply stuck in low volatility
            hmm_stuck_in_low_vol = False
            if self.current_hmm_regime and self.current_hmm_regime.get('volatility') == 'Low':
                if self.current_hmm_regime.get('confidence', 0) > 0.6:  # 60% confidence
                    hmm_stuck_in_low_vol = True
            
            if abs(mom15) < 0.005 and not hmm_stuck_in_low_vol:
                if self.enable_straddles:
                    signal["action"] = "BUY_STRADDLE"
                    signal["strategy"] = "VOLATILITY_EXPANSION"
                    signal["reasoning"].append(
                        f"Low realized vol percentile ({vol_pct:.1f}%), looking for expansion.")
                    components.append(0.7)
                else:
                    signal["reasoning"].append(
                        "Straddle opportunity detected but straddles are disabled by config (trading.enable_straddles=false)."
                    )
            elif hmm_stuck_in_low_vol:
                 logger.info(f"🛑 Suppressed BUY_STRADDLE: HMM Confident Low Volatility ({self.current_hmm_regime.get('confidence', 0):.1%})")

        elif vol_pct > 70.0:  # Lowered from 80% to 70%
            # HIGH VOLATILITY REGIME: Don't trade blindly against the trend
            # Only act if we have a neural prediction to guide direction
            # Otherwise stay in HOLD and let neural prediction determine action below
            if neural_pred and abs(neural_pred.get("predicted_return", 0)) > 0.0005:
                # We have a directional signal - don't override it
                logger.info(f"📊 High vol ({vol_pct:.1f}%) but deferring to neural direction: {neural_pred.get('predicted_return', 0):.4f}")
            else:
                # No strong neural signal - this is a mean-reversion play only if momentum confirms
                if mom15 < -0.001:  # Only buy puts if momentum is actually bearish
                    signal["action"] = "BUY_PUTS"
                    signal["strategy"] = "VOLATILITY_CONTRACTION"
                    signal["reasoning"].append(
                        f"High vol ({vol_pct:.1f}%) + bearish momentum → puts.")
                    components.append(0.6)
                elif mom15 > 0.001:  # Bullish momentum - buy calls instead
                    signal["action"] = "BUY_CALLS"
                    signal["strategy"] = "VOLATILITY_CONTRACTION_CALLS"
                    signal["reasoning"].append(
                        f"High vol ({vol_pct:.1f}%) + bullish momentum → calls.")
                    components.append(0.6)
                else:
                    logger.info(f"📊 High vol ({vol_pct:.1f}%) but neutral momentum - staying HOLD")

        # Momentum (adaptive threshold based on performance)
        if abs(mom15) > self.momentum_threshold:
            near_resistance = options_metrics["near_resistance"]
            near_support = options_metrics["near_support"]
            price_position = options_metrics["price_position"]
            
            logger.info(f"🎯 Momentum check: mom15={mom15:.4f}, near_resistance={near_resistance}, near_support={near_support}, price_pos={price_position:.2f}")
            
            if mom15 > 0 and not near_resistance:
                signal["action"] = "BUY_CALLS"
                signal["strategy"] = "MOMENTUM_BULLISH"
                signal["reasoning"].append(
                    f"15m bullish momentum ({mom15*100:+.2f}%).")
                components.append(0.8)
                logger.info(f"✅ Bullish momentum signal triggered!")
            elif mom15 < 0 and not near_support:
                signal["action"] = "BUY_PUTS"
                signal["strategy"] = "MOMENTUM_BEARISH"
                signal["reasoning"].append(
                    f"15m bearish momentum ({mom15*100:+.2f}%).")
                components.append(0.8)
                logger.info(f"✅ Bearish momentum signal triggered!")
            else:
                logger.info(f"❌ Momentum signal blocked by support/resistance levels")

        # Neural (adjusted for normal market conditions)
        if neural_pred:
            nc = neural_pred["neural_confidence"]
            pr = neural_pred["predicted_return"]
            
            # DEBUG: Log neural prediction details to trace signal generation
            logger.info(f"🔬 NEURAL DEBUG: pr={pr:.6f} ({pr*100:.3f}%), nc={nc:.3f} ({nc*100:.1f}%)")
            logger.info(f"🔬 THRESHOLDS: return_thresh={self.neural_return_threshold:.6f}, conf_thresh={self.neural_confidence_threshold:.3f}")
            logger.info(f"🔬 CHECKS: pr>thresh={pr > self.neural_return_threshold}, pr<-thresh={pr < -self.neural_return_threshold}, nc>conf_thresh={nc > self.neural_confidence_threshold}")
            
            if components:
                components.append(min(nc * 0.3, 0.2))  # modest boost
                signal["reasoning"].append(f"Neural confidence {nc:.1%}.")
            if nc > self.neural_confidence_threshold:  # Adaptive neural confidence threshold
                if pr > self.neural_return_threshold:  # Adaptive return threshold
                    logger.info(f"✅ NEURAL: Setting BUY_CALLS (pr={pr:.4f} > thresh={self.neural_return_threshold:.4f})")
                    signal["action"] = "BUY_CALLS"
                    signal["strategy"] = "NEURAL_BULLISH"
                    components.append(nc)
                elif pr < -self.neural_return_threshold:  # Adaptive negative return threshold
                    logger.info(f"✅ NEURAL: Setting BUY_PUTS (pr={pr:.4f} < -thresh={-self.neural_return_threshold:.4f})")
                    signal["action"] = "BUY_PUTS"
                    signal["strategy"] = "NEURAL_BEARISH"
                    components.append(nc)
                else:
                    logger.info(f"⚠️ NEURAL: Return too small for signal (|{pr:.4f}| < {self.neural_return_threshold:.4f})")
            else:
                logger.info(f"⚠️ NEURAL: Confidence too low ({nc:.3f} < {self.neural_confidence_threshold:.3f})")

        # Multi-timeframe neural analysis (new!)
        if multi_timeframe_signal and multi_timeframe_signal.get("signal_strength", 0) > 0.1:
            mtf_confidence = multi_timeframe_signal["multi_timeframe_confidence"]
            mtf_direction = multi_timeframe_signal["direction"]
            mtf_strength = multi_timeframe_signal["signal_strength"]
            agreement_score = multi_timeframe_signal["agreement_score"]
            
            logger.info(f"📊 Multi-timeframe: {mtf_direction} | Confidence: {mtf_confidence:.1%} | Agreement: {agreement_score:.1%} | Strength: {mtf_strength:.3f}")
            
            # Log individual timeframe details
            for tf, details in multi_timeframe_signal.get("timeframe_details", {}).items():
                logger.info(f"   {tf}: {details['direction']} {details['return']:+.2%} (conf: {details['confidence']:.1%})")
            
            # Add multi-timeframe component if signal is strong enough
            if mtf_strength > 0.05:  # LOWERED: Let more signals through to RL
                if mtf_direction == "BULLISH" and mtf_confidence > self.neural_confidence_threshold:
                    if signal["action"] == "HOLD":  # Don't override existing signals
                        signal["action"] = "BUY_CALLS"
                        signal["strategy"] = "MULTI_TIMEFRAME_BULLISH"
                    signal["reasoning"].append(f"Multi-timeframe bullish consensus (strength: {mtf_strength:.2f})")
                    components.append(mtf_strength)
                    logger.info(f"✅ Multi-timeframe bullish signal triggered!")
                    
                elif mtf_direction == "BEARISH" and mtf_confidence > self.neural_confidence_threshold:
                    if signal["action"] == "HOLD":  # Don't override existing signals
                        signal["action"] = "BUY_PUTS"
                        signal["strategy"] = "MULTI_TIMEFRAME_BEARISH"
                    signal["reasoning"].append(f"Multi-timeframe bearish consensus (strength: {mtf_strength:.2f})")
                    components.append(mtf_strength)
                    logger.info(f"✅ Multi-timeframe bearish signal triggered!")
                else:
                    # Even if not strong enough for action, add confidence boost
                    components.append(mtf_strength * 0.5)  # Partial credit
                    signal["reasoning"].append(f"Multi-timeframe {mtf_direction.lower()} bias (weak)")

        # Volume confirmation (adaptive threshold)
        if vol_spike > self.volume_threshold:  # Adaptive volume spike threshold
            signal["reasoning"].append(f"Volume spike {vol_spike:.1f}×.")
            components.append(0.1)

        if components:
            raw_confidence = float(min(max(np.mean(components), 0.0), 0.95))
            
            # Apply prediction consistency boost/penalty
            consistency = self.get_prediction_consistency(lookback=5)
            consistency_factor = 1.0
            
            if consistency['samples'] >= 3:
                # Boost confidence if recent predictions agree on direction
                if consistency['direction_consistency'] > 0.7:
                    # Check if current signal aligns with consensus
                    current_direction = 'BULLISH' if signal.get('action') == 'BUY_CALLS' else (
                        'BEARISH' if signal.get('action') == 'BUY_PUTS' else 'NEUTRAL'
                    )
                    
                    if current_direction == consistency['consensus_direction'] and current_direction != 'NEUTRAL':
                        consistency_factor = 1.0 + (consistency['direction_consistency'] - 0.5) * 0.3  # Up to 15% boost
                        signal["reasoning"].append(f"Prediction consistency boost ({consistency['direction_consistency']:.0%} agreement)")
                    elif current_direction != 'NEUTRAL' and consistency['consensus_direction'] != 'NEUTRAL' and current_direction != consistency['consensus_direction']:
                        # Current signal contradicts recent consensus - be more cautious
                        consistency_factor = 0.85
                        signal["reasoning"].append(f"Signal contradicts recent trend ({consistency['consensus_direction']})")
                
                # Log consistency info
                if consistency['samples'] >= 5:
                    logger.info(f"📊 Prediction Consistency: {consistency['direction_consistency']:.0%} {consistency['consensus_direction']} | "
                               f"Avg Return: {consistency['avg_return']*100:+.2f}% | Trend: {consistency['trend']}")
            
            # Apply consistency factor
            adjusted_confidence = raw_confidence * consistency_factor
            
            # Apply Bayesian confidence calibration based on historical accuracy
            calibrated_confidence = self.get_calibrated_confidence(adjusted_confidence)
            signal["confidence"] = calibrated_confidence
            signal["raw_confidence"] = raw_confidence  # Keep raw for tracking
            signal["consistency_factor"] = consistency_factor
            signal["prediction_consistency"] = consistency
            
            if abs(calibrated_confidence - raw_confidence) > 0.05 or consistency_factor != 1.0:
                logger.info(f"✅ Confidence: raw={raw_confidence:.1%} × consistency={consistency_factor:.2f} → calibrated={calibrated_confidence:.1%}")
            else:
                logger.info(f"✅ Confidence calculated: {len(components)} components {components} → {calibrated_confidence:.3f}")
        else:
            logger.info(f"❌ No signal components met → confidence stays at default 50%")

        # Risk label
        if signal["confidence"] > 0.8:
            signal["risk_level"] = "LOW"
        elif signal["confidence"] < 0.6:
            signal["risk_level"] = "HIGH"

        # ✅ Add multi-timeframe analysis and timeframe to signal for RL exit policy
        if multi_timeframe_signal:
            signal["multi_timeframe_analysis"] = multi_timeframe_signal
            signal["timeframe_minutes"] = multi_timeframe_signal.get("timeframe_minutes", 60)
            signal["predicted_return"] = multi_timeframe_signal.get("multi_timeframe_return", 0.0)
        else:
            # Fallback to default timeframe
            signal["timeframe_minutes"] = 60
            signal["predicted_return"] = signal.get("predicted_return", 0.0)

        # If we are running "options-only" (no neural prediction), ensure predicted_return isn't stuck at 0.
        # Train-time bandit gates use predicted_return as "edge" and will skip almost everything if it's 0.
        try:
            if (not neural_pred) and abs(float(signal.get("predicted_return", 0.0) or 0.0)) < 1e-9:
                mom15_fallback = float(options_metrics.get("momentum_15min", 0.0) or 0.0)
                if signal.get("action") == "BUY_CALLS":
                    signal["predicted_return"] = abs(mom15_fallback)
                elif signal.get("action") == "BUY_PUTS":
                    signal["predicted_return"] = -abs(mom15_fallback)
        except Exception:
            pass

        # ------------------------------------------------------------------
        # 🛡️ State-Based Hysteresis (Prevent Signal Flickering)
        # ------------------------------------------------------------------
        if not hasattr(self, '_signal_state'):
            self._signal_state = "NEUTRAL"
        
        # Adaptive thresholds - LOWERED for training to generate more directional signals
        entry_thresh = max(0.10, self.min_confidence_threshold * 0.5)  # Very low for training
        exit_thresh = entry_thresh * 0.5                               # Even lower to exit
        
        current_action = signal["action"]
        current_conf = signal["confidence"]
        
        # DEBUG: Log state before hysteresis
        logger.info(f"🔬 PRE-HYSTERESIS: action={current_action}, conf={current_conf:.3f}, state={self._signal_state}, entry_thresh={entry_thresh:.3f}")
        
        # Logic: Sticky States
        # If NEUTRAL: Require high confidence to enter
        if self._signal_state == "NEUTRAL":
            if current_conf >= entry_thresh:
                if current_action == "BUY_CALLS":
                    self._signal_state = "BULLISH"
                    logger.info(f"🟢 Signal State: NEUTRAL -> BULLISH (Conf: {current_conf:.1%})")
                elif current_action == "BUY_PUTS":
                    self._signal_state = "BEARISH"
                    logger.info(f"🔴 Signal State: NEUTRAL -> BEARISH (Conf: {current_conf:.1%})")
        
        # If BULLISH: Stay BULLISH unless confidence drops or signal flips hard
        elif self._signal_state == "BULLISH":
            if current_action == "BUY_PUTS" and current_conf >= entry_thresh:
                self._signal_state = "BEARISH" # Hard flip
                logger.info(f"🔄 Signal State: BULLISH -> BEARISH (Hard Flip)")
            elif current_conf < exit_thresh:
                self._signal_state = "NEUTRAL" # Weak signal exit
                logger.info(f"⚪ Signal State: BULLISH -> NEUTRAL (Signal Faded: {current_conf:.1%})")
            else:
                # MAINTAIN BULLISH STATE even if current minute is weak/hold
                if current_action == "HOLD" and current_conf > exit_thresh:
                     # Override HOLD with weaker BUY_CALLS to maintain position
                     signal["action"] = "BUY_CALLS"
                     signal["reasoning"].append("Hysteresis: Maintaining BULLISH state")
        
        # If BEARISH: Stay BEARISH unless confidence drops or signal flips hard
        elif self._signal_state == "BEARISH":
            if current_action == "BUY_CALLS" and current_conf >= entry_thresh:
                self._signal_state = "BULLISH" # Hard flip
                logger.info(f"🔄 Signal State: BEARISH -> BULLISH (Hard Flip)")
            elif current_conf < exit_thresh:
                self._signal_state = "NEUTRAL" # Weak signal exit
                logger.info(f"⚪ Signal State: BEARISH -> NEUTRAL (Signal Faded: {current_conf:.1%})")
            else:
                # MAINTAIN BEARISH STATE even if current minute is weak/hold
                if current_action == "HOLD" and current_conf > exit_thresh:
                     # Override HOLD with weaker BUY_PUTS to maintain position
                     signal["action"] = "BUY_PUTS"
                     signal["reasoning"].append("Hysteresis: Maintaining BEARISH state")

        # Enforce State
        if self._signal_state == "NEUTRAL":
            signal["action"] = "HOLD"
            # We don't zero confidence, but action is HOLD
        
        # DEBUG: Log state after hysteresis
        logger.info(f"🔬 POST-HYSTERESIS: action={signal['action']}, state={self._signal_state}")
        
        # =====================================================================
        # CALIBRATION: Record prediction for Platt scaling
        # =====================================================================
        # Only record non-HOLD signals with valid direction
        if self.calibration_tracker is not None and signal["action"] != "HOLD":
            try:
                # Determine direction from action
                direction = 'UP' if 'CALL' in signal["action"] else 'DOWN'
                raw_conf = signal.get("raw_confidence", signal.get("confidence", 0.5))
                
                self.calibration_tracker.record_prediction(
                    confidence=raw_conf,
                    predicted_direction=direction,
                    entry_price=current_price
                )
            except Exception as e:
                logger.debug(f"Could not record calibration prediction: {e}")
        
        return signal

    def generate_time_travel_signal(
        self,
        symbol: str,
        data: pd.DataFrame,
        simulation_time: Optional[datetime] = None,
        interval: str = "1m",
    ) -> Optional[Dict]:
        """
        Simplified signal generator for time-travel training.
        Mirrors the original root behavior (no live-data gating, no RL/multi-timeframe overrides).
        """
        try:
            if data is None or data.empty:
                return None

            if simulation_time:
                self.simulated_time = simulation_time
                self.current_time = simulation_time

            data = data.copy()
            data.columns = data.columns.str.lower()
            data = data.loc[:, ~data.columns.duplicated()]
            data = data.dropna()

            if len(data) < 20:
                return None

            current_price = float(data["close"].iloc[-1])
            options_metrics = self.calculate_options_metrics(data, interval_hint=interval)
            if not options_metrics:
                return None

            # ========================================================================
            # Auto-train HMM regime detector (was missing from time-travel mode!)
            # ========================================================================
            self._maybe_train_hmm(data)
            
            current_features = self.create_features(data)
            if current_features is None:
                return None

            # For time-travel mode, estimate VIX from volatility percentile (Improvement #3)
            # VIX typically ranges 10-40, volatility percentile 0-100
            vol_pct = options_metrics.get("volatility_percentile", 50)
            # Rough mapping: vol_pct 20 → VIX 12, vol_pct 50 → VIX 17, vol_pct 80 → VIX 25
            self.current_vix = 10 + (vol_pct / 100) * 20  # Maps 0-100 to 10-30

            self.feature_buffer.append(current_features)
            if len(self.feature_buffer) > self.max_buffer_size:
                self.feature_buffer.pop(0)

            self._maybe_init_model()

            # ========================================================================
            # CRITICAL FIX: Get neural prediction ONCE and reuse for consistency!
            # The Bayesian NN with Monte Carlo sampling gives different results each
            # call, which was causing sign flips between signal generation and display.
            # ========================================================================
            neural_pred = self.get_neural_prediction(current_features)
            
            # Cache this prediction for multi-timeframe to use (avoids re-calling NN)
            self._cached_neural_pred = neural_pred
            
            multi_timeframe_preds = {}
            multi_timeframe_signal = None
            try:
                multi_timeframe_preds = self.get_multi_timeframe_predictions(current_features)
                multi_timeframe_signal = self.combine_multi_timeframe_signals()
                # Store for paper trader's exit decisions (dynamic hold time)
                self.last_multi_timeframe_predictions = multi_timeframe_preds
            except Exception as e:
                logger.debug(f"[SIM] Multi-timeframe prediction skipped: {e}")
            finally:
                # Clear cache after use
                self._cached_neural_pred = None

            signal = self._combine_signals(
                options_metrics,
                neural_pred,
                current_price,
                multi_timeframe_signal,
            )
            signal["current_price"] = current_price
            signal["simulation_mode"] = True
            if neural_pred:
                signal["neural_prediction"] = neural_pred
            if multi_timeframe_preds:
                signal["multi_timeframe_predictions"] = multi_timeframe_preds

            now = simulation_time or self.get_market_time()
            prediction_record = {
                "timestamp": now,
                "features": current_features.copy(),
                "predicted_return": neural_pred.get("predicted_return", 0.0) if neural_pred else 0.0,
                "predicted_volatility": neural_pred.get("predicted_volatility", 0.0) if neural_pred else 0.0,
                "current_price": current_price,
                "prediction_time": now.strftime("%Y-%m-%d %H:%M:%S"),
            }
            self.prediction_history.append(prediction_record)
            if len(self.prediction_history) > 100:
                self.prediction_history.pop(0)

            return signal
        except Exception as e:
            logger.error(f"[SIM] Signal generation error: {e}")
            return None

    def _ensure_multiframe_columns(self, cursor):
        """Ensure multi-timeframe columns exist in the database"""
        new_columns = [
            ("pred_15min_return", "REAL"),
            ("pred_15min_confidence", "REAL"),
            ("pred_30min_return", "REAL"),
            ("pred_30min_confidence", "REAL"),
            ("pred_1hour_return", "REAL"),
            ("pred_1hour_confidence", "REAL"),
            ("pred_4hour_return", "REAL"),
            ("pred_4hour_confidence", "REAL"),
            ("pred_12hour_return", "REAL"),
            ("pred_12hour_confidence", "REAL"),
            ("pred_24hour_return", "REAL"),
            ("pred_24hour_confidence", "REAL"),
            ("pred_48hour_return", "REAL"),
            ("pred_48hour_confidence", "REAL"),
        ]
        
        for column_name, column_type in new_columns:
            try:
                cursor.execute(f"ALTER TABLE unified_signals ADD COLUMN {column_name} {column_type}")
            except sqlite3.OperationalError:
                pass  # Column already exists

    def _log_signal(
        self,
        symbol: str,
        current_price: float,
        options_metrics: Dict,
        neural_pred: Optional[Dict],
        signal: Dict,
        prediction_time_str: str,
        current_features: Optional[np.ndarray] = None,
        current_sequence: Optional[np.ndarray] = None,
        multi_timeframe_preds: Optional[Dict] = None,
    ) -> Optional[int]:
        try:
            conn = sqlite3.connect("data/unified_options_bot.db")
            c = conn.cursor()
            
            # First, ensure multi-timeframe columns exist
            self._ensure_multiframe_columns(c)
            
            c.execute(
                """
                INSERT INTO unified_signals (
                    timestamp, symbol, current_price,
                    features_blob,
                    sequence_blob,
                    volatility_percentile, momentum_5min, momentum_15min, volume_spike, price_position,
                    predicted_return, predicted_volatility, direction_prob, temporal_confidence, lstm_context,
                    action, strategy, confidence, reasoning,
                    executed, position_size, entry_price, exit_price, pnl,
                    predicted_price_15min, prediction_time,
                    pred_15min_return, pred_15min_confidence,
                    pred_30min_return, pred_30min_confidence,
                    pred_1hour_return, pred_1hour_confidence,
                    pred_4hour_return, pred_4hour_confidence,
                    pred_12hour_return, pred_12hour_confidence,
                    pred_24hour_return, pred_24hour_confidence,
                    pred_48hour_return, pred_48hour_confidence
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self.get_market_time().strftime("%Y-%m-%d %H:%M:%S"),
                    symbol,
                    current_price,
                    sqlite3.Binary(pickle.dumps(current_features, protocol=pickle.HIGHEST_PROTOCOL)) if current_features is not None else None,
                    sqlite3.Binary(pickle.dumps(current_sequence, protocol=pickle.HIGHEST_PROTOCOL)) if current_sequence is not None else None,
                    options_metrics["volatility_percentile"],
                    options_metrics["momentum_5min"],
                    options_metrics["momentum_15min"],
                    options_metrics["volume_spike"],
                    options_metrics["price_position"],
                    (neural_pred or {}).get("predicted_return"),
                    (neural_pred or {}).get("predicted_volatility"),
                    json.dumps((neural_pred or {}).get(
                        "direction_probs")) if neural_pred else None,
                    (neural_pred or {}).get("temporal_confidence"),
                    json.dumps((neural_pred or {}).get("lstm_context", {})) if neural_pred else None,
                    signal["action"],
                    signal["strategy"],
                    signal["confidence"],
                    "; ".join(signal.get("reasoning", [])),
                    False,
                    0,
                    None,
                    None,
                    None,
                    signal.get("predicted_price_15min", current_price),
                    prediction_time_str,
                    # Multi-timeframe predictions (15min to 48hr)
                    multi_timeframe_preds.get("15min", {}).get("predicted_return") if multi_timeframe_preds else None,
                    multi_timeframe_preds.get("15min", {}).get("temporal_confidence") if multi_timeframe_preds else None,
                    multi_timeframe_preds.get("30min", {}).get("predicted_return") if multi_timeframe_preds else None,
                    multi_timeframe_preds.get("30min", {}).get("temporal_confidence") if multi_timeframe_preds else None,
                    multi_timeframe_preds.get("1hour", {}).get("predicted_return") if multi_timeframe_preds else None,
                    multi_timeframe_preds.get("1hour", {}).get("temporal_confidence") if multi_timeframe_preds else None,
                    multi_timeframe_preds.get("4hour", {}).get("predicted_return") if multi_timeframe_preds else None,
                    multi_timeframe_preds.get("4hour", {}).get("temporal_confidence") if multi_timeframe_preds else None,
                    multi_timeframe_preds.get("12hour", {}).get("predicted_return") if multi_timeframe_preds else None,
                    multi_timeframe_preds.get("12hour", {}).get("temporal_confidence") if multi_timeframe_preds else None,
                    multi_timeframe_preds.get("24hour", {}).get("predicted_return") if multi_timeframe_preds else None,
                    multi_timeframe_preds.get("24hour", {}).get("temporal_confidence") if multi_timeframe_preds else None,
                    multi_timeframe_preds.get("48hour", {}).get("predicted_return") if multi_timeframe_preds else None,
                    multi_timeframe_preds.get("48hour", {}).get("temporal_confidence") if multi_timeframe_preds else None,
                ),
            )
            row_id = c.lastrowid
            conn.commit()
            conn.close()
            return row_id
        except Exception as e:
            logger.error(f"Database logging error: {e}")
            return None

    def _update_execution_in_db(self, row_id: int, position_size: int, entry_price: float) -> None:
        try:
            conn = sqlite3.connect("data/unified_options_bot.db")
            c = conn.cursor()
            c.execute(
                "UPDATE unified_signals SET executed=?, position_size=?, entry_price=? WHERE id=?",
                (True, position_size, entry_price, row_id),
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"DB execution update error: {e}")

    def build_liquidity_features(self, contract_row: dict, underlying_features: np.ndarray) -> np.ndarray:
        """
        Build contract-level microstructure features for fillability prediction.
        
        Combines contract microstructure (bid/ask, sizes, OI, volume, greeks, moneyness, TTE)
        with underlying regime features (vol, momentum, RSI, HMM states).
        
        Args:
            contract_row: Option contract data with keys: bid, ask, bidsize, asksize,
                         open_interest, volume, delta, gamma, vega, strike, 
                         underlying_price, expiry (YYYY-MM-DD)
            underlying_features: Current feature vector from feature_buffer
        
        Returns:
            Combined feature array (~34 features)
        """
        # Extract contract microstructure
        bid = float(contract_row.get('bid', 0.0))
        ask = float(contract_row.get('ask', 0.0))
        bsz = float(contract_row.get('bidsize', 0.0))
        asz = float(contract_row.get('asksize', 0.0))
        oi = float(contract_row.get('open_interest', 0.0))
        vol = float(contract_row.get('volume', 0.0))
        delta = float(contract_row.get('delta', 0.0))
        gamma = float(contract_row.get('gamma', 0.0))
        vega = float(contract_row.get('vega', 0.0))
        strike = float(contract_row.get('strike', 0.0))
        und_px = float(contract_row.get('underlying_price', 0.0))
        
        # Spread and mid
        mid = (bid + ask) / 2.0 if (bid > 0 and ask > 0) else 0.0
        spread = (ask - bid) if (ask >= bid and bid > 0) else 9e9
        spread_pct = (spread / mid) if mid > 0 else 99.0
        
        # Book depth & imbalance
        depth_min = min(bsz, asz)
        imbalance = (bsz - asz) / (bsz + asz) if (bsz + asz) > 0 else 0.0
        
        # Moneyness & time to expiry (days)
        moneyness = (und_px - strike) / und_px if und_px > 0 else 0.0
        try:
            from datetime import datetime
            dte = (datetime.strptime(contract_row['expiry'], "%Y-%m-%d").date() - 
                   datetime.now().date()).days
        except (KeyError, ValueError, TypeError) as e:
            logger.debug(f"Could not parse expiry date: {e}. Using default DTE=30")
            dte = 30
        
        # Compact slice of underlying features (vol, momentum, RSI, HMM states)
        # Indices: [4-6: vol], [1-3: momentum], [10-14: technical], [20-29: HMM]
        uf = underlying_features.copy()
        if uf.shape[0] >= 30:
            uf_slice = uf[[4,5,6, 1,2,3, 10,11,12, 20,21,22,23,24,25,26,27,28,29]]
        else:
            uf_slice = uf[:19] if uf.shape[0] >= 19 else np.zeros(19)
        
        # Contract features (15) + underlying features (19) = 34 total
        feats = np.array([
            mid, spread, spread_pct, bsz, asz, depth_min, imbalance,
            oi, vol, delta, abs(delta), gamma, vega, moneyness, dte
        ], dtype=np.float32)
        
        return np.concatenate([feats, uf_slice.astype(np.float32)], axis=0)
    
    def calculate_position_size(self, confidence: float, current_price: float) -> int:
        """Simple risk-based sizing. (Still heuristic; replace with actual premium if available.)"""
        acct = self.paper_trader.get_account_summary()
        balance = float(acct.get("current_balance",
                        acct.get("cash_balance", 25000.0)))
        base_risk = balance * self.risk_per_trade
        confidence_mult = 0.5 + confidence  # 0.5× to 1.5×
        # rough premium proxy: 1.5% of underlying; floor at $0.20
        est_pct = 0.015
        est_option_cost = max(0.20, current_price * est_pct)
        contracts = int((base_risk * confidence_mult) /
                        (est_option_cost * 100.0))
        return max(1, min(contracts, 10))
    
    def make_rl_decision(
        self, 
        signal: Dict, 
        neural_pred: Optional[Dict],
        options_metrics: Dict,
        symbol: str
    ) -> Tuple[str, int]:
        """
        Use RL policy to make trading decision based on Bayesian NN predictions
        Returns: (action, position_size)
        
        UPDATED: Now supports unified RL policy for both entry and exit decisions.
        """
        if not self.use_rl_policy or neural_pred is None:
            # Fallback to old threshold-based logic
            return signal["action"], 1
        
        # =====================================================================
        # UNIFIED RL PATH (preferred)
        # =====================================================================
        if self.use_unified_rl and self.unified_rl is not None:
            return self._make_unified_rl_decision(signal, neural_pred, options_metrics, symbol)
        
        # =====================================================================
        # LEGACY RL PATH (fallback)
        # =====================================================================
        try:
            # Get current account state
            acct = self.paper_trader.get_account_summary()
            current_balance = float(acct.get("current_balance", acct.get("cash_balance", 1000.0)))
            num_open_positions = len(self.paper_trader.active_trades)
            total_pnl = current_balance - self.rl_episode_start_balance
            
            # Get recent performance
            recent_trades = self.paper_trader.get_recent_closed_trades(limit=10)
            if recent_trades:
                wins = sum(1 for t in recent_trades if t.get('pnl', 0) > 0)
                win_rate = wins / len(recent_trades)
                avg_return = np.mean([t.get('pnl', 0) / current_balance for t in recent_trades])
                
                # Calculate recent drawdown
                running_balance = current_balance
                max_balance = current_balance
                for t in reversed(recent_trades):
                    running_balance -= t.get('pnl', 0)
                    max_balance = max(max_balance, running_balance)
                drawdown = (max_balance - min(max_balance, running_balance)) / max_balance if max_balance > 0 else 0
            else:
                win_rate = 0.5
                avg_return = 0.0
                drawdown = 0.0
            
            # Get time features
            now = self.get_market_time()
            market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
            minutes_since_open = int((now - market_open).total_seconds() / 60)
            minutes_since_open = max(0, min(390, minutes_since_open))  # Cap at trading day length
            
            # Check if last hour
            market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
            is_last_hour = (market_close - now).total_seconds() < 3600
            
            # Get VIX data
            try:
                vix_data = self.safe_get_data("^VIX", period=self.historical_period, interval="5m")
                vix_level = vix_data['Close'].iloc[-1] if not vix_data.empty else 15.0
            except Exception as e:
                logger.debug(f"Could not fetch VIX: {e}. Using default 15.0")
                vix_level = 15.0
            
            # Extract features from neural prediction and signal
            predicted_return = neural_pred.get('predicted_return', 0.0)
            predicted_volatility = neural_pred.get('predicted_volatility', 0.01)
            confidence = signal.get('confidence', 0.5)
            
            # Direction probabilities (from neural network or derive from signal)
            direction_probs = np.array([0.33, 0.34, 0.33])  # Default: neutral [down, neutral, up]
            if 'direction_probs' in neural_pred:
                dp = neural_pred['direction_probs']
                # Convert dict to array if needed
                if isinstance(dp, dict):
                    direction_probs = np.array([dp.get('DOWN', 0.33), dp.get('NEUTRAL', 0.34), dp.get('UP', 0.33)])
                else:
                    direction_probs = np.array(dp) if not isinstance(dp, np.ndarray) else dp
            elif predicted_return > 0.01:
                direction_probs = np.array([0.1, 0.2, 0.7])  # Bullish
            elif predicted_return < -0.01:
                direction_probs = np.array([0.7, 0.2, 0.1])  # Bearish
            
            # Get current RSI
            rsi = options_metrics.get('rsi', 50.0)
            
            # Count trades today
            today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            # Use 2000-01-01 as safe default to avoid Pandas 0001-01-01 overflow errors
            safe_min_date = datetime(2000, 1, 1).replace(tzinfo=self.market_tz)
            trades_today = len([t for t in self.paper_trader.get_recent_closed_trades(limit=100) 
                               if t.get('entry_time', safe_min_date) > today_start])
            
            # Days until expiry (assume weekly options, nearest Friday)
            days_until_friday = (4 - now.weekday()) % 7  # 0-6 days until Friday
            days_until_expiry = max(1, days_until_friday)
            
            # Get HMM regime features for RL Trading Policy
            hmm_trend = 0.5
            hmm_vol = 0.5
            hmm_liq = 0.5
            hmm_regime = getattr(self, 'current_hmm_regime', None)
            if isinstance(hmm_regime, str):
                if 'bullish' in hmm_regime.lower():
                    hmm_trend = 1.0
                elif 'bearish' in hmm_regime.lower():
                    hmm_trend = 0.0
                if 'high_vol' in hmm_regime.lower():
                    hmm_vol = 1.0
                elif 'low_vol' in hmm_regime.lower():
                    hmm_vol = 0.0
                if 'high_liq' in hmm_regime.lower():
                    hmm_liq = 1.0
                elif 'low_liq' in hmm_regime.lower():
                    hmm_liq = 0.0
            
            # Get extended VIX features for RL Trading Policy
            vix_bb_pos = getattr(self, 'current_vix_bb_pos', 0.5)
            vix_roc = getattr(self, 'current_vix_roc', 0.0)
            vix_percentile = getattr(self, 'current_vix_percentile', 0.5)
            
            # Get price jerk for RL Trading Policy
            price_jerk = getattr(self, 'current_price_jerk', 0.0)
            
            # ==============================================================================
            # 🌍 NEW: CONTEXT & MACRO FEATURES (QQQ, NVDA, BTC, etc.)
            # ==============================================================================
            context_trend_alignment = 0.5
            tech_momentum = 0.0
            crypto_momentum = 0.0
            sector_rotation = 0.0
            
            try:
                # 1. Tech Momentum (QQQ/NVDA)
                tech_returns = []
                for s in ['QQQ', 'NVDA', 'MSFT', 'AAPL']:
                    if s in self.market_context_prices and s in self.market_context:
                        # Calculate rough 5-min momentum
                        df = self.market_context[s]
                        if len(df) >= 5:
                            # Use last 5 bars close
                            ret = (df['close'].iloc[-1] - df['close'].iloc[-5]) / df['close'].iloc[-5]
                            tech_returns.append(ret)
                
                if tech_returns:
                    tech_momentum = float(np.mean(tech_returns))
                
                # 2. Crypto Momentum (BTC-USD)
                if 'BTC-USD' in self.market_context:
                    df = self.market_context['BTC-USD']
                    if len(df) >= 5:
                        crypto_momentum = (df['close'].iloc[-1] - df['close'].iloc[-5]) / df['close'].iloc[-5]
                
                # 3. Context Trend Alignment
                # How many context symbols match our predicted direction?
                aligned_count = 0
                total_context = 0
                pred_dir = 1 if predicted_return > 0 else -1
                
                for s in ['QQQ', 'SPY', 'IWM', 'NVDA', 'BTC-USD']:
                    if s in self.market_context:
                        df = self.market_context[s]
                        if len(df) >= 5:
                            trend = (df['close'].iloc[-1] - df['close'].iloc[-5]) / df['close'].iloc[-5]
                            if np.sign(trend) == np.sign(pred_dir):
                                aligned_count += 1
                            total_context += 1
                
                if total_context > 0:
                    context_trend_alignment = aligned_count / total_context
                
                # 4. Sector Rotation (QQQ vs IWM)
                if 'QQQ' in self.market_context and 'IWM' in self.market_context:
                    q_df = self.market_context['QQQ']
                    i_df = self.market_context['IWM']
                    if len(q_df) >= 5 and len(i_df) >= 5:
                        q_ret = (q_df['close'].iloc[-1] - q_df['close'].iloc[-5]) / q_df['close'].iloc[-5]
                        i_ret = (i_df['close'].iloc[-1] - i_df['close'].iloc[-5]) / i_df['close'].iloc[-5]
                        sector_rotation = q_ret - i_ret # Positive = Tech Leading, Negative = Small Caps Leading

            except Exception as e:
                logger.debug(f"Context feature calculation error: {e}")

            # Build state vector (32 features)
            state = self.rl_policy.build_state_vector(
                predicted_return=predicted_return,
                predicted_volatility=predicted_volatility,
                confidence=confidence,
                direction_probs=direction_probs,
                current_balance=current_balance,
                num_open_positions=num_open_positions,
                max_positions=self.max_positions,
                total_pnl=total_pnl,
                vix_level=vix_level,
                momentum_5min=options_metrics.get('momentum_5m', 0.0),
                momentum_15min=options_metrics.get('momentum_15m', 0.0),
                volume_spike=options_metrics.get('volume_spike', 1.0),
                rsi=rsi,
                win_rate_last_10=win_rate,
                avg_return_last_10=avg_return,
                recent_drawdown=drawdown,
                trades_today=trades_today,
                minutes_since_market_open=minutes_since_open,
                days_until_expiry=days_until_expiry,
                is_last_hour=is_last_hour,
                # HMM regime features
                hmm_trend=hmm_trend,
                hmm_vol=hmm_vol,
                hmm_liq=hmm_liq,
                # Extended VIX features
                vix_bb_pos=vix_bb_pos,
                vix_roc=vix_roc,
                vix_percentile=vix_percentile,
                # Price jerk
                price_jerk=price_jerk,
                # NEW: Context features
                context_trend_alignment=context_trend_alignment,
                tech_momentum=tech_momentum,
                crypto_momentum=crypto_momentum,
                sector_rotation=sector_rotation
            )
            
            # Validate state vector
            if state is None or not isinstance(state, np.ndarray):
                logger.error(f"❌ RL state invalid: {type(state)}")
                return signal["action"], 1
            
            if np.any(np.isnan(state)) or np.any(np.isinf(state)):
                logger.error(f"❌ RL state contains NaN/Inf values")
                logger.error(f"   State: {state}")
                return signal["action"], 1
            
            if state.shape[0] != self.rl_policy.state_dim:
                logger.error(f"❌ RL state wrong size: {state.shape[0]} (expected {self.rl_policy.state_dim})")
                return signal["action"], 1
            
            # Select action using RL policy
            # UPDATED: PPO policy returns log_prob as 3rd value
            action_idx, action_prob, log_prob = self.rl_policy.select_action(
                state, 
                deterministic=False,  # Use stochastic policy for exploration
                temperature=self.rl_temperature
            )
            
            # Store state, action, and log_prob for later reward calculation
            self.rl_last_state = state
            self.rl_last_action = action_idx
            self.rl_last_log_prob = log_prob
            
            # Convert action index to trading action
            action_name = TradingAction.to_string(action_idx)
            base_position_size = TradingAction.get_position_size(action_idx)
            direction = TradingAction.get_direction(action_idx)
            
            # ==============================================================================
            # 🚀 DYNAMIC KELLY-STYLE SIZING
            # ==============================================================================
            # Scale position size based on RL confidence and trend alignment
            final_position_size = base_position_size
            
            if base_position_size > 0:
                # 1. Confidence Scaling
                if action_prob > 0.80:
                    final_position_size *= 1.5  # Boost size for very high confidence
                elif action_prob < 0.50:
                    final_position_size *= 0.5  # Cut size for low confidence
                
                # 2. Trend Alignment Scaling
                if context_trend_alignment > 0.8:
                     final_position_size *= 1.25 # Boost if market agrees
                elif context_trend_alignment < 0.2:
                     final_position_size *= 0.75 # Cut if market disagrees
                
                # Round to nearest sensible integer multiplier (e.g. 1, 2, 3)
                final_position_size = max(1, int(round(final_position_size)))
                # Cap at reasonable max (e.g. 3x or 4x)
                final_position_size = min(3, final_position_size)

            # Log RL decision
            action_dist = self.rl_policy.get_action_distribution(state)
            logger.info(f"🎯 RL Policy Decision: {action_name} ({action_prob:.1%} confidence)")
            logger.info(f"   Context: Tech={tech_momentum:.2%}, Crypto={crypto_momentum:.2%}, Align={context_trend_alignment:.0%}")
            logger.info(f"   Sizing: Base {base_position_size}x → Final {final_position_size}x (Kelly Adj)")
            
            # Update signal with RL decision
            if direction == 'CALL':
                signal["action"] = "BUY_CALLS"
                signal["strategy"] = f"RL_POLICY_BULLISH_{final_position_size}X"
            elif direction == 'PUT':
                signal["action"] = "BUY_PUTS"
                signal["strategy"] = f"RL_POLICY_BEARISH_{final_position_size}X"
            else:
                signal["action"] = "HOLD"
                signal["strategy"] = "RL_POLICY_HOLD"
            
            # Add RL reasoning
            signal["reasoning"].append(f"RL Policy ({action_prob:.1%} confidence)")
            signal["rl_action"] = action_name
            signal["rl_confidence"] = action_prob
            
            return signal["action"], final_position_size
            
        except Exception as e:
            logger.error(f"❌ RL policy decision error: {e}")
            logger.exception(e)
            # Fallback to old logic
            return signal["action"], 1

    def _make_unified_rl_decision(
        self, 
        signal: Dict, 
        neural_pred: Optional[Dict],
        options_metrics: Dict,
        symbol: str
    ) -> Tuple[str, int]:
        """
        Use UNIFIED RL policy for entry/exit decisions.
        
        This replaces the complex 3-system approach with a single clean policy.
        """
        try:
            # Get current account state
            acct = self.paper_trader.get_account_summary()
            current_balance = float(acct.get("current_balance", acct.get("cash_balance", 1000.0)))
            
            # Get VIX
            try:
                vix_data = self.safe_get_data("^VIX", period=self.historical_period, interval="5m")
                vix_level = vix_data['Close'].iloc[-1] if not vix_data.empty else 18.0
            except Exception:
                vix_level = 18.0
            
            # Extract key features
            predicted_return = neural_pred.get('predicted_return', 0.0) if neural_pred else 0.0
            predicted_volatility = neural_pred.get('predicted_volatility', 0.01) if neural_pred else 0.01
            confidence = signal.get('confidence', 0.5)
            momentum_5m = options_metrics.get('momentum_5m', 0.0)
            volume_spike = options_metrics.get('volume_spike', 1.0)
            
            # Estimate theta decay (rough approximation)
            # For 0DTE options, theta can be -5% to -20% daily
            # For 1-5 DTE, theta is typically -1% to -5% daily
            days_to_expiry = 1  # Assume near-term options
            estimated_theta = -0.03 if days_to_expiry <= 1 else -0.015  # -3% for 0DTE, -1.5% otherwise
            
            # ==========================================
            # GET HMM REGIME DATA (YOUR HMM MODEL)
            # ==========================================
            hmm_trend = 0.5      # Default: neutral
            hmm_volatility = 0.5  # Default: normal
            hmm_liquidity = 0.5   # Default: normal
            hmm_confidence = 0.5  # Default: medium confidence
            
            if hasattr(self, 'current_hmm_regime') and self.current_hmm_regime:
                hmm_regime = self.current_hmm_regime
                
                # Map trend names to 0-1 scale (0=bearish, 0.5=neutral, 1=bullish)
                trend_map = {
                    'Down': 0.0, 'Downtrend': 0.0, 'Bearish': 0.0,
                    'No Trend': 0.5, 'Neutral': 0.5, 'Sideways': 0.5,
                    'Up': 1.0, 'Uptrend': 1.0, 'Bullish': 1.0
                }
                trend_name = hmm_regime.get('trend', 'Neutral')
                hmm_trend = trend_map.get(trend_name, 0.5)
                
                # Map volatility names to 0-1 scale (0=low, 0.5=normal, 1=high)
                vol_map = {'Low': 0.0, 'Normal': 0.5, 'High': 1.0}
                vol_name = hmm_regime.get('volatility', 'Normal')
                hmm_volatility = vol_map.get(vol_name, 0.5)
                
                # Map liquidity names to 0-1 scale
                liq_map = {'Low': 0.0, 'Normal': 0.5, 'High': 1.0}
                liq_name = hmm_regime.get('liquidity', 'Normal')
                hmm_liquidity = liq_map.get(liq_name, 0.5)
                
                # HMM confidence (if available)
                hmm_confidence = hmm_regime.get('confidence', hmm_regime.get('combined_confidence', 0.5))
                
                logger.debug(f"HMM Regime: trend={trend_name}({hmm_trend:.1f}), vol={vol_name}({hmm_volatility:.1f}), conf={hmm_confidence:.1%}")
            
            # Build trade state for unified policy (NOW INCLUDES HMM!)
            trade_state = TradeState(
                is_in_trade=False,  # Entry decision
                is_call=predicted_return > 0,
                entry_price=0.0,
                current_price=0.0,
                position_size=0,
                unrealized_pnl_pct=0.0,
                max_pnl_seen=0.0,
                max_drawdown=0.0,
                minutes_held=0,
                minutes_to_expiry=days_to_expiry * 390,  # Convert to trading minutes
                # predicted_return is decimal (0.01 = 1%), scale to [-1, 1] where 1.0 = 1%
                # Neural network bounds output to ±2% anyway via tanh
                predicted_direction=np.clip(predicted_return * 100, -1, 1),
                prediction_confidence=confidence,  # YOUR MODEL'S CONFIDENCE
                vix_level=vix_level,
                momentum_5m=momentum_5m,
                volume_spike=volume_spike,
                # HMM REGIME DATA (YOUR HMM MODEL)
                hmm_trend=hmm_trend,
                hmm_volatility=hmm_volatility,
                hmm_liquidity=hmm_liquidity,
                hmm_confidence=hmm_confidence,
                # Greeks
                estimated_theta_decay=estimated_theta,
                estimated_delta=0.5,  # ATM assumption
            )
            
            # Get action from unified policy
            action, action_confidence, details = self.unified_rl.select_action(trade_state, deterministic=False)
            
            # DEBUG: Log comprehensive decision analysis (enable with DEBUG_RL_DECISIONS=1)
            import os
            if os.environ.get('DEBUG_RL_DECISIONS', '0') == '1':
                self.unified_rl.log_debug_state(trade_state, action, action_confidence, details)
            
            # Convert unified action to signal format
            if action == self.unified_rl.BUY_CALL:
                final_action = "BUY_CALLS"
                strategy = "UNIFIED_RL_BULLISH"
                position_size = 1
            elif action == self.unified_rl.BUY_PUT:
                final_action = "BUY_PUTS"
                strategy = "UNIFIED_RL_BEARISH"
                position_size = 1
            else:
                final_action = "HOLD"
                strategy = "UNIFIED_RL_HOLD"
                position_size = 0
            
            # Log decision
            mode = "BANDIT" if self.unified_rl.is_bandit_mode else "RL"
            logger.info(f"🎯 Unified RL ({mode}): {final_action} ({action_confidence:.1%} confidence)")
            logger.info(f"   Reason: {details.get('reason', 'N/A')}")
            
            # Update signal with unified RL decision
            signal["action"] = final_action
            signal["strategy"] = strategy
            signal["reasoning"].append(f"Unified RL ({mode}, {action_confidence:.1%})")
            signal["rl_action"] = final_action
            signal["rl_confidence"] = action_confidence
            
            # Store state for learning
            self.current_trade_state = trade_state
            
            return final_action, position_size
            
        except Exception as e:
            logger.error(f"❌ Unified RL decision error: {e}")
            logger.exception(e)
            # Fallback to signal's original action
            return signal["action"], 1
    
    def update_unified_rl_for_position(self, trade, current_price: float):
        """
        Update unified RL state for an open position.
        Called periodically to track P&L and potentially trigger exit.
        """
        if not self.use_unified_rl or self.unified_rl is None:
            return None
        
        try:
            entry_price = trade.get('entry_price', current_price)
            is_call = 'CALL' in str(trade.get('order_type', '')).upper()
            
            # Calculate P&L
            if is_call:
                # CALL profits when price goes up
                pnl_pct = (current_price - entry_price) / entry_price
            else:
                # PUT profits when price goes down
                pnl_pct = (entry_price - current_price) / entry_price
            
            # For options, multiply by approximate leverage (delta)
            estimated_delta = 0.5  # ATM assumption
            option_pnl_pct = pnl_pct * (1 / estimated_delta) if estimated_delta > 0 else pnl_pct
            
            # Track high water mark
            max_pnl = max(option_pnl_pct, getattr(trade, '_max_pnl', 0.0))
            trade['_max_pnl'] = max_pnl
            
            # Calculate drawdown from peak
            max_drawdown = max(0, max_pnl - option_pnl_pct)
            
            # Get time held
            entry_time = trade.get('entry_time', datetime.now())
            now = self.get_market_time()
            minutes_held = int((now - entry_time).total_seconds() / 60)
            
            # Calculate minutes to expiry (assume end of day if not specified)
            expiry_time = trade.get('expiry_time', now.replace(hour=16, minute=0))
            minutes_to_expiry = max(1, int((expiry_time - now).total_seconds() / 60))
            
            # Get current market state
            try:
                vix_data = self.safe_get_data("^VIX", period="1d", interval="5m")
                vix_level = vix_data['Close'].iloc[-1] if not vix_data.empty else 18.0
            except:
                vix_level = 18.0
            
            # Build exit decision state
            exit_state = TradeState(
                is_in_trade=True,
                is_call=is_call,
                entry_price=entry_price,
                current_price=current_price,
                position_size=trade.get('quantity', 1),
                unrealized_pnl_pct=option_pnl_pct,
                max_pnl_seen=max_pnl,
                max_drawdown=max_drawdown,
                minutes_held=minutes_held,
                minutes_to_expiry=minutes_to_expiry,
                predicted_direction=1.0 if is_call else -1.0,
                prediction_confidence=trade.get('entry_confidence', 0.5),
                vix_level=vix_level,
                momentum_5m=0.0,  # Would need live data
                volume_spike=1.0,
                estimated_theta_decay=-0.03 if minutes_to_expiry < 390 else -0.015,
                estimated_delta=estimated_delta,
            )
            
            # Get exit decision
            action, confidence, details = self.unified_rl.select_action(exit_state, deterministic=False)
            
            if action == self.unified_rl.EXIT:
                logger.info(f"🚪 Unified RL EXIT signal: {details.get('reason', 'RL decision')}")
                logger.info(f"   P&L: {option_pnl_pct:.1%}, Held: {minutes_held}m, Confidence: {confidence:.1%}")
                return {'should_exit': True, 'reason': details.get('reason', 'RL exit'), 'pnl_pct': option_pnl_pct}
            
            return {'should_exit': False, 'pnl_pct': option_pnl_pct, 'minutes_held': minutes_held}
            
        except Exception as e:
            logger.error(f"Error updating unified RL for position: {e}")
            return None
    
    def record_trade_outcome_unified(self, trade_result: Dict):
        """
        Record trade outcome for unified RL learning.
        Called when a trade closes.
        """
        if not self.use_unified_rl or self.unified_rl is None:
            return
        
        try:
            pnl = trade_result.get('pnl', 0.0)
            entry_price = trade_result.get('entry_price', 1.0)
            exit_price = trade_result.get('exit_price', entry_price)
            pnl_pct = pnl / max(1.0, entry_price * trade_result.get('quantity', 1) * 100)
            
            # Record as a completed episode for learning
            was_win = pnl > 0
            logger.info(f"📊 Unified RL recorded trade: {'WIN' if was_win else 'LOSS'} ${pnl:+.2f} ({pnl_pct:+.1%})")
            
            # Train the policy if we have enough data
            if len(self.unified_rl.experience_buffer) >= 32:
                train_result = self.unified_rl.train_step(batch_size=32)
                if train_result:
                    logger.info(f"🎓 Unified RL trained: loss={train_result.get('total_loss', 0):.4f}")
            
            # Save policy periodically
            if self.unified_rl.total_trades % 10 == 0:
                self.unified_rl.save(self.unified_rl_path)
                
        except Exception as e:
            logger.error(f"Error recording trade for unified RL: {e}")

    def execute_trade(self, signal: Dict, symbol: str, current_price: float) -> bool:
        # IMPORTANT: When called from simulation with RL-approved trades,
        # don't double-check confidence (RL already evaluated it)
        # Only block HOLD actions
        if signal["action"] == "HOLD":
            # NOTE: We do NOT store RL experiences for HOLD decisions here because
            # we don't know the actual_return (what would have happened if we traded).
            # Storing experiences with actual_return=0.0 would train the RL policy
            # on misleading data. Real RL learning happens in track_completed_trades_for_rl()
            # where we have actual P&L from completed trades.
            if self.use_rl_policy and self.rl_last_state is not None:
                logger.debug(f"🔄 HOLD decision - skipping RL experience storage (no actual return available)")
            return False
        
        # When running without RL simulation layer, check confidence threshold
        # Use adaptive threshold with a minimum floor of 10% for safety
        effective_threshold = max(0.10, getattr(self, 'min_confidence_threshold', 0.10))
        if signal["confidence"] < effective_threshold:
            logger.warning(f"⚠️ Blocking trade: confidence {signal['confidence']:.1%} < threshold {effective_threshold:.1%}")
            return False

        # Check minimum time between trades for algorithmic trading discipline
        # Skip this check in simulation mode to allow rapid testing
        if not self.simulation_mode and hasattr(self, 'last_trade_time'):
            time_since_last_trade = (self.get_market_time() - self.last_trade_time).total_seconds()
            if time_since_last_trade < self.min_time_between_trades:
                minutes_remaining = (self.min_time_between_trades - time_since_last_trade) / 60
                logger.info(f"[WAIT] Waiting {minutes_remaining:.1f} more minutes before next trade (algorithmic spacing)")
                return False

        # position limit (already checked by simulation, but double-check for safety)
        if len(self.paper_trader.active_trades) >= self.max_positions:
            logger.warning(f"[WARN] Max positions ({self.max_positions}) reached, cannot place trade")
            return False

        # Use RL position size if available, otherwise calculate
        if "rl_position_size" in signal and signal["rl_position_size"] > 0:
            position_size = signal["rl_position_size"]
            logger.info(f"[RL] Using RL policy position size: {position_size}x")
        else:
            position_size = self.calculate_position_size(
                signal["confidence"], current_price)

        # Best-effort mapping (backend may not support all legs)
        # Check for multi-leg order support and warn on fallbacks
        has_straddle_support = getattr(OrderType, "BUY_STRADDLE", None) is not None
        has_credit_spread_support = getattr(OrderType, "SELL_CREDIT_SPREAD", None) is not None
        
        order_type_map = {
            "BUY_CALLS": getattr(OrderType, "BUY_CALL", None),
            "BUY_CALL": getattr(OrderType, "BUY_CALL", None),
            "BUY_PUTS": getattr(OrderType, "BUY_PUT", None),
            "BUY_PUT": getattr(OrderType, "BUY_PUT", None),
            # SAFETY: never silently degrade multi-leg / short-premium actions to single-leg orders.
            "BUY_STRADDLE": getattr(OrderType, "BUY_STRADDLE", None),
            "SELL_PREMIUM": getattr(OrderType, "SELL_CREDIT_SPREAD", None),
        }
        order_type = order_type_map.get(signal["action"])
        
        # Block unsupported complex actions (safer than silent fallback).
        if signal["action"] == "BUY_STRADDLE" and not has_straddle_support:
            logger.warning("❌ BUY_STRADDLE requested but backend does not support BUY_STRADDLE. Blocking.")
            return False
        
        if signal["action"] == "SELL_PREMIUM" and not has_credit_spread_support:
            logger.warning("❌ SELL_PREMIUM requested but backend does not support SELL_CREDIT_SPREAD. Blocking.")
            return False
        
        if order_type is None:
            logger.warning(f"❌ Unknown or unsupported action: {signal['action']}")
            logger.warning(f"   Available actions: {list(order_type_map.keys())}")
            return False

        logger.info(f"📝 Preparing to place trade: {signal['action']} with confidence {signal['confidence']:.1%}")

        try:
            prediction_data = {
                "action": signal["action"],
                "confidence": signal["confidence"],
                "strategy": signal.get("strategy", "RL_APPROVED"),
                "reasoning": signal.get("reasoning", ["RL threshold learner approved"]),
                "risk_level": signal.get("risk_level", "MEDIUM"),
                "position_size": position_size,
                # Include HMM state for exit decisions
                "hmm_state": signal.get("hmm_state"),
                "hmm_confidence": signal.get("hmm_confidence", 0),
                "hmm_trend": signal.get("hmm_trend"),
            }

            logger.info(f"📞 Calling paper_trader.place_trade(symbol={symbol}, price={current_price:.2f})")
            trade = self.paper_trader.place_trade(
                symbol=symbol, prediction_data=prediction_data, current_price=current_price
            )
            success = trade is not None

            if success:
                logger.info(f"✅ Trade placed successfully! Trade ID: {trade.id if hasattr(trade, 'id') else 'unknown'}")
                
                # ========================================
                # 🎯 SIMONSAYS TRADE ANNOUNCEMENT 🎯
                # ========================================
                reasoning_text = " | ".join(signal.get("reasoning", ["Signal approved"]))
                neural_pred = signal.get("predicted_return", 0)
                timeframe = signal.get("timeframe_minutes", 60)
                
                logger.info("")
                logger.info("=" * 80)
                logger.info(f"🎯 SimonSays: {signal['action']} x{position_size} @ ${current_price:.2f}")
                logger.info(f"   💰 Trade Details:")
                logger.info(f"      • Symbol: {symbol}")
                logger.info(f"      • Position Size: {position_size} contracts")
                logger.info(f"      • Entry Price: ${current_price:.2f}")
                logger.info(f"      • Confidence: {signal['confidence']:.1%}")
                logger.info(f"      • Strategy: {signal.get('strategy', 'N/A')}")
                logger.info(f"      • Predicted Move: {neural_pred:+.2%} over {timeframe}min")
                logger.info(f"   📊 Why This Trade:")
                for i, reason in enumerate(signal.get("reasoning", ["RL approved"]), 1):
                    logger.info(f"      {i}. {reason}")
                if signal.get("multi_timeframe_analysis"):
                    mt_analysis = signal["multi_timeframe_analysis"]
                    logger.info(f"   🔮 Multi-Timeframe Consensus:")
                    logger.info(f"      • Direction: {mt_analysis.get('direction', 'N/A')}")
                    logger.info(f"      • Confidence: {mt_analysis.get('confidence', 0):.1%}")
                    logger.info(f"      • Agreement: {mt_analysis.get('agreement', 0):.1%}")
                logger.info("=" * 80)
                logger.info("")
                
                # Track last trade time for realistic spacing
                self.last_trade_time = self.get_market_time()
                
                # Update adaptive confidence system
                self.update_adaptive_confidence(signal["confidence"])
                
                logger.info(f"⏰ Next trade available in {self.min_time_between_trades/60:.0f} minutes (algorithmic spacing)")

                # Report trade entry to Data Manager
                if self.bot_reporter and self.bot_reporter.enabled:
                    try:
                        self.bot_reporter.report_trade(
                            symbol=symbol,
                            action=signal["action"],
                            quantity=position_size,
                            price=current_price,
                            metadata={
                                "confidence": signal.get("confidence", 0),
                                "strategy": signal.get("strategy", ""),
                                "predicted_return": signal.get("predicted_return", 0),
                                "timeframe_minutes": signal.get("timeframe_minutes", 60),
                                "hmm_trend": signal.get("hmm_trend"),
                            }
                        )
                        logger.debug(f"📡 Reported trade entry to Data Manager")
                    except Exception as e:
                        logger.debug(f"📡 Failed to report trade: {e}")

                if "_row_id" in signal and signal["_row_id"]:
                    self._update_execution_in_db(
                        signal["_row_id"], position_size, current_price)
                return True
            else:
                logger.error(f"❌ place_trade returned None - trade was rejected by paper trading system")
                logger.error(f"   Check paper_trading.db for any error messages")
                return False
        except Exception as e:
            logger.error(f"Trade execution error: {e}")
            return False

    def place_option_with_liquidity(self, 
                                   symbol: str, 
                                   expiry: str, 
                                   call_put: str, 
                                   side: str,
                                   desired_qty: int, 
                                   bias: str = "bull",
                                   max_debit: float = None, 
                                   min_credit: float = None) -> Dict:
        """
        Place option order through liquidity-aware execution layer
        
        This routes orders through sophisticated execution logic that:
        - Screens contracts by open interest, volume, spread, delta
        - Calculates liquidity scores
        - Sizes orders by displayed depth
        - Works midpoint-pegged orders with price ladder
        - Falls back to vertical spreads if single-leg has poor fill
        
        Args:
            symbol: Underlying symbol (e.g., 'BITX')
            expiry: Expiration date 'YYYY-MM-DD'
            call_put: 'call' or 'put'
            side: 'buy_to_open', 'sell_to_open', etc.
            desired_qty: Desired quantity
            bias: 'bull', 'bear', or 'neutral'
            max_debit: Maximum debit willing to pay (for buys)
            min_credit: Minimum credit to receive (for sells)
        
        Returns:
            Execution result dict with status and details
        """
        if not self.liquidity_exec:
            logger.warning("[LIQ-EXEC] Liquidity execution layer not available")
            return {
                "status": "unavailable",
                "reason": "Liquidity execution layer not initialized"
            }
        
        intent = OrderIntent(
            symbol=symbol,
            side=side,
            qty=desired_qty,
            expiry=expiry,
            call_put=call_put,
            bias=bias,
            max_debit=max_debit,
            min_credit=min_credit
        )
        
        logger.info(f"[LIQ-EXEC] Routing order through liquidity-aware execution layer")
        result = self.liquidity_exec.execute_with_fallback(intent)
        
        logger.info(f"[LIQ-EXEC] Execution result: {result.get('status')}")
        if result.get('reason'):
            logger.info(f"[LIQ-EXEC] Reason: {result.get('reason')}")
        
        return result
    
    def log_execution_outcome(self, outcome_data: dict):
        """
        Log execution outcome for training fillability predictor
        
        Args:
            outcome_data: Dict with execution details (flexible format)
        """
        try:
            conn = sqlite3.connect("data/unified_options_bot.db")
            c = conn.cursor()
            
            # Handle both full format and simplified paper trading format
            # Get features - if not provided, use empty array
            if 'features' in outcome_data:
                features_blob = pickle.dumps(outcome_data['features'])
            else:
                features_blob = pickle.dumps(np.array([]))
            
            # Map alternative key names for compatibility
            occ_symbol = outcome_data.get('occ_symbol', outcome_data.get('option_symbol', 'UNKNOWN'))
            symbol = outcome_data.get('symbol', 'SPY')
            side = outcome_data.get('side', outcome_data.get('option_type', 'UNKNOWN'))
            limit_price = outcome_data.get('limit_price', outcome_data.get('entry_price', 0))
            worked_secs = outcome_data.get('worked_secs', outcome_data.get('time_to_fill', 0))
            filled_qty = outcome_data.get('filled_qty', outcome_data.get('quantity', 1))
            desired_qty = outcome_data.get('desired_qty', outcome_data.get('quantity', 1))
            avg_fill_price = outcome_data.get('avg_fill_price', outcome_data.get('entry_price', 0))
            best_bid = outcome_data.get('best_bid', outcome_data.get('bid', 0))
            best_ask = outcome_data.get('best_ask', outcome_data.get('ask', 0))
            
            # Calculate labels if not provided
            label_fill = outcome_data.get('label_fill', 1 if filled_qty > 0 else 0)
            label_slip = outcome_data.get('label_slip', abs(avg_fill_price - limit_price) if limit_price > 0 else 0)
            label_ttf = outcome_data.get('label_ttf', worked_secs)
            
            c.execute("""
                INSERT INTO exec_outcomes 
                (ts, occ_symbol, symbol, side, limit_price, worked_secs, 
                 filled_qty, desired_qty, avg_fill_price, best_bid, best_ask,
                 features, label_fill, label_slip, label_ttf)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                occ_symbol,
                symbol,
                side,
                limit_price,
                worked_secs,
                filled_qty,
                desired_qty,
                avg_fill_price,
                best_bid,
                best_ask,
                features_blob,
                label_fill,
                label_slip,
                label_ttf
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"[XQ-LOG] Logged execution: fill={label_fill}, "
                       f"slip=${label_slip:.3f}, ttf={label_ttf:.1f}s")
        
        except Exception as e:
            logger.error(f"[XQ-LOG] Error logging execution outcome: {e}")
    
    def train_execution_heads(self, batch_size: int = 10):
        """
        Train fillability/slippage/TTF prediction heads from logged execution outcomes
        
        Samples recent execution outcomes and trains the neural heads with multi-task loss.
        Integrates seamlessly with existing continuous learning infrastructure.
        """
        if self.model is None or len(self.feature_buffer) < self.sequence_length:
            return
        
        try:
            conn = sqlite3.connect("data/unified_options_bot.db")
            c = conn.cursor()
            
            # Get recent outcomes (last 100, sample batch_size)
            c.execute("""
                SELECT features, label_fill, label_slip, label_ttf
                FROM exec_outcomes
                ORDER BY id DESC
                LIMIT 100
            """)
            
            rows = c.fetchall()
            conn.close()
            
            if len(rows) < 5:
                return  # Need at least 5 examples
            
            # Sample batch
            import random
            batch = random.sample(rows, min(batch_size, len(rows)))
            
            self.model.train()
            self.optimizer.zero_grad()
            losses = []
            
            # Get normalization stats
            buf = np.vstack(self.feature_buffer[-self.sequence_length:])
            mu, sd = buf.mean(axis=0), buf.std(axis=0) + 1e-6
            
            for row in batch:
                features_blob, label_fill, label_slip, label_ttf = row
                
                # Unpickle features (not currently used in forward pass,
                # but logged for potential future contract-specific models)
                # feats = pickle.loads(features_blob)
                
                # Use current underlying features (same as strategy predictions)
                cur_features = self.feature_buffer[-1].copy()
                cur = (cur_features - mu) / sd
                seq = (buf - mu) / sd
                
                x = torch.from_numpy(cur).float().unsqueeze(0).to(self.device)
                s = torch.from_numpy(seq).float().unsqueeze(0).to(self.device)
                
                out = self.model(x, s)
                
                # Targets (scalars for single-item loss computation)
                y_fill = torch.tensor(label_fill, dtype=torch.float32, device=self.device)
                y_slip = torch.tensor(label_slip, dtype=torch.float32, device=self.device)
                y_ttf = torch.tensor(label_ttf, dtype=torch.float32, device=self.device)
                
                # Losses (multi-task) - ensure both sides are scalars
                pred_fill = out["fillability"].view(-1)[0]  # Get scalar
                pred_slip = out["exp_slippage"].view(-1)[0]
                pred_ttf = out["exp_ttf"].view(-1)[0]
                
                bce_loss = nn.BCELoss()(pred_fill, y_fill)
                mse_slip = nn.MSELoss()(pred_slip, y_slip)
                mse_ttf = nn.MSELoss()(pred_ttf, y_ttf)
                
                # Combined loss (tune weights empirically)
                loss = 1.0 * bce_loss + 0.5 * mse_slip + 0.25 * mse_ttf
                losses.append(loss)
            
            if losses:
                total_loss = torch.stack(losses).mean()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                logger.info(f"[XQ-LRN] Execution heads trained: loss={total_loss.item():.5f} "
                           f"(n={len(batch)})")
        
        except Exception as e:
            logger.error(f"[XQ-LRN] Error training execution heads: {e}")
    
    def score_contract_by_nn(self, contract_row: dict, underlying_features: np.ndarray) -> dict:
        """
        Score contract using neural fillability predictor
        
        Args:
            contract_row: Option contract data
            underlying_features: Current feature vector
        
        Returns:
            Dict with: p_fill, exp_slip, exp_ttf, utility, features
        """
        # Build features (for logging, though not used in current forward pass)
        feats = self.build_liquidity_features(contract_row, underlying_features)
        
        # If model not ready, return conservative estimates
        if self.model is None or len(self.feature_buffer) < self.sequence_length:
            return {
                "p_fill": 0.0,
                "exp_slip": 0.05,
                "exp_ttf": 15.0,
                "utility": -1.0,
                "features": feats
            }
        
        try:
            # Get model predictions using current underlying features
            buf = np.vstack(self.feature_buffer[-self.sequence_length:])
            mu, sd = buf.mean(axis=0), buf.std(axis=0) + 1e-6
            
            cur = (self.feature_buffer[-1] - mu) / sd
            seq = (buf - mu) / sd
            
            x = torch.from_numpy(cur).float().unsqueeze(0).to(self.device)
            s = torch.from_numpy(seq).float().unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                out = self.model(x, s)
            
            p_fill = float(out["fillability"].cpu().item())
            exp_slip = float(out["exp_slippage"].cpu().item())
            exp_ttf = float(out["exp_ttf"].cpu().item())
            
            # Utility function: prefer high p_fill, penalize slippage and long TTF
            # Adjust weights based on your preferences
            utility = p_fill - 2.0 * max(0.0, exp_slip) - 0.02 * max(0.0, exp_ttf)
            
            return {
                "p_fill": p_fill,
                "exp_slip": exp_slip,
                "exp_ttf": exp_ttf,
                "utility": utility,
                "features": feats
            }
        
        except Exception as e:
            logger.error(f"[XQ-SCORE] Error scoring contract: {e}")
            return {
                "p_fill": 0.0,
                "exp_slip": 0.05,
                "exp_ttf": 15.0,
                "utility": -1.0,
                "features": feats
            }

    def update_confidence_from_closed_trades(self):
        """
        Check for recently closed trades and update adaptive confidence with their P&L
        """
        try:
            # Get recently closed trades from paper trading system
            recent_closed = self.paper_trader.get_recent_closed_trades(limit=10)
            
            for trade_dict in recent_closed:
                # Check if we've already processed this trade for confidence adjustment
                trade_id = trade_dict.get('id')
                if not hasattr(self, '_processed_confidence_trades'):
                    self._processed_confidence_trades = set()
                
                if trade_id not in self._processed_confidence_trades:
                    confidence = trade_dict.get('ml_confidence', 0)
                    pnl = trade_dict.get('profit_loss', 0)
                    
                    if confidence and confidence > 0:  # Valid confidence value
                        self.update_adaptive_confidence(confidence, pnl)
                        self._processed_confidence_trades.add(trade_id)
                        
                        # Log for visibility
                        logger.debug(f"📊 Confidence update: {confidence:.1%} → P&L: ${pnl:+.2f}")
                        
                    # Keep set size manageable
                    if len(self._processed_confidence_trades) > 100:
                        # Remove oldest entries (keep last 100)
                        self._processed_confidence_trades = set(list(self._processed_confidence_trades)[-100:])
                        
        except Exception as e:
            logger.error(f"Error updating confidence from closed trades: {e}")

    def track_completed_trades_for_rl(self):
        """Track completed trades and update RL policy with actual outcomes"""
        if not self.use_rl_policy:
            return
        
        try:
            # Get recent closed trades
            recent_closed = self.paper_trader.get_recent_closed_trades(limit=5)
            
            for trade in recent_closed:
                trade_id = trade.get('id')
                
                # Check if we've already processed this trade for RL
                if not hasattr(self, '_rl_processed_trades'):
                    self._rl_processed_trades = set()
                
                if trade_id in self._rl_processed_trades:
                    continue
                
                # Mark as processed
                self._rl_processed_trades.add(trade_id)
                
                # Calculate reward from actual trade outcome
                pnl = trade.get('pnl', 0.0)
                entry_time = trade.get('entry_time')
                exit_time = trade.get('exit_time')
                
                if entry_time and exit_time:
                    time_held_minutes = int((exit_time - entry_time).total_seconds() / 60)
                else:
                    time_held_minutes = 0
                
                current_balance = float(self.paper_trader.get_account_summary().get('current_balance', 1000.0))
                
                # Build trade result dict
                trade_result = {
                    'pnl': pnl,
                    'entry_price': trade.get('entry_price', 1.0),
                    'exit_price': trade.get('exit_price', 1.0),
                    'respected_stop_loss': trade.get('stop_loss_hit', False),
                    'took_profit_target': trade.get('take_profit_hit', False),
                    'held_to_expiry': time_held_minutes > 360,  # More than 6 hours
                    'max_drawdown': abs(min(0, pnl)) / current_balance
                }
                
                # Calculate final reward
                reward = self.rl_policy.calculate_reward(
                    trade_result=trade_result,
                    account_balance=current_balance,
                    time_held_minutes=time_held_minutes
                )
                
                # Store experience (episode complete)
                if hasattr(self, 'rl_last_state') and self.rl_last_state is not None:
                    self.rl_policy.store_experience(
                        state=self.rl_last_state,
                        action=self.rl_last_action if hasattr(self, 'rl_last_action') else 0,
                        reward=reward,
                        next_state=None,
                        done=True,  # Trade complete = episode done
                        log_prob=getattr(self, 'rl_last_log_prob', 0.0), # Pass log_prob for PPO
                        info={'trade_id': trade_id, 'pnl': pnl}
                    )
                    
                    logger.info(f"🎯 RL Reward: Trade #{trade_id} → P&L ${pnl:+.2f} → Reward {reward:+.3f}")

                    # Report completed trade to Data Manager
                    if self.bot_reporter and self.bot_reporter.enabled:
                        try:
                            exit_reason = "stop_loss" if trade.get('stop_loss_hit') else \
                                         "take_profit" if trade.get('take_profit_hit') else \
                                         "manual"
                            self.bot_reporter.report_trade(
                                symbol=trade.get('symbol', 'SPY'),
                                action="EXIT",
                                quantity=trade.get('quantity', 1),
                                price=trade.get('exit_price', 0),
                                pnl=pnl,
                                pnl_pct=trade.get('pnl_pct', 0) if 'pnl_pct' in trade else (pnl / current_balance * 100),
                                notes=exit_reason,
                                metadata={
                                    "trade_id": trade_id,
                                    "entry_price": trade.get('entry_price', 0),
                                    "hold_minutes": time_held_minutes,
                                    "exit_reason": exit_reason,
                                }
                            )
                            logger.debug(f"📡 Reported trade exit to Data Manager")
                        except Exception as e:
                            logger.debug(f"📡 Failed to report trade exit: {e}")

                    # Train the policy with recent experiences
                    if len(self.rl_policy.experience_buffer) >= 5:
                        metrics = self.rl_policy.train_step(batch_size=min(8, len(self.rl_policy.experience_buffer)))
                        if metrics:
                            logger.info(f"🧠 RL Training: Actor loss={metrics['actor_loss']:.4f}, "
                                      f"Critic loss={metrics['critic_loss']:.4f}, "
                                      f"Entropy={metrics['entropy']:.4f}")
                            
                            # Save RL policy periodically
                            if self.rl_policy.training_stats['total_updates'] % 10 == 0:
                                os.makedirs('models', exist_ok=True)
                                self.rl_policy.save(self.rl_policy_path)
                                logger.info(f"💾 RL Policy saved to {self.rl_policy_path}")
                    
            # Clean up old processed trades (keep last 100)
            if len(self._rl_processed_trades) > 100:
                oldest_trades = list(self._rl_processed_trades)[:50]
                for trade_id in oldest_trades:
                    self._rl_processed_trades.discard(trade_id)
                    
        except Exception as e:
            logger.error(f"Error tracking completed trades for RL: {e}")
            logger.exception(e)
    
    def store_multi_timeframe_predictions_for_validation(self, predictions: Dict, current_price: float):
        """
        Store predictions when made so we can validate them later and learn which timeframes are accurate.
        
        Args:
            predictions: Dict of multi-timeframe predictions
            current_price: Current market price when prediction was made
        """
        try:
            current_time = self.get_market_time()
            
            # Get current features for online tuning (store them with prediction)
            current_features = None
            if len(self.feature_buffer) >= self.sequence_length:
                current_features = np.array(list(self.feature_buffer)[-1])
            
            for tf_name, pred in predictions.items():
                # Store ALL timeframes for online tuning (not just 15min, 30min, etc.)
                tf_minutes = pred.get('timeframe_minutes', 15)
                
                # Store for online tuning - include confidence for calibration!
                self.store_prediction_for_online_tuning(
                    timeframe=tf_name,
                    timeframe_minutes=tf_minutes,
                    predicted_return=pred.get('predicted_return', 0),
                    current_price=current_price,
                    features=current_features,
                    confidence=pred.get('neural_confidence', 0.5)  # Pass confidence for calibration
                )
                
                # Also store for adaptive weights (existing behavior)
                if tf_name in ['15min', '30min', '1hour', '4hour', '5min', '10min']:
                    self.adaptive_weights.store_prediction(
                        timeframe=tf_name,
                        predicted_return=pred.get('predicted_return', 0),
                        predicted_direction=pred.get('direction', 'NEUTRAL'),
                        confidence=pred.get('neural_confidence', 0),
                        timestamp=current_time
                    )
                    
                    # Store price at prediction time for validation
                    if not hasattr(self, '_prediction_prices'):
                        self._prediction_prices = {}
                    self._prediction_prices[tf_name] = current_price
                    
        except Exception as e:
            logger.debug(f"Error storing predictions for validation: {e}")
    
    def validate_and_update_timeframe_accuracy(self, current_price: float):
        """
        Check past predictions to see if they were accurate and update weights accordingly.
        This is the key method that learns which timeframes are actually working!
        """
        try:
            current_time = self.get_market_time()
            
            # Validate predictions (check which ones have reached their timeframe)
            self.adaptive_weights.validate_predictions(current_price, current_time)
            
            # For simplified validation in simulation:
            # Check if any timeframes have reached their prediction window
            # and compare actual vs predicted
            
            # This is a simplified approach - in production you'd store actual prices
            # at prediction_time + timeframe and compare
            
        except Exception as e:
            logger.debug(f"Error validating predictions: {e}")
    
    def update_adaptive_timeframe_weights(self):
        """
        Update timeframe weights based on prediction accuracy.
        Called periodically to learn which timeframes are most accurate.
        """
        try:
            self.predictions_since_weight_update += 1
            
            # Only update every N prediction cycles
            if self.predictions_since_weight_update < self.weight_update_interval:
                return
            
            self.predictions_since_weight_update = 0
            
            logger.info("🎯 Checking if timeframe weights should be adjusted...")
            
            # Adjust weights based on learned accuracy
            weights_changed = self.adaptive_weights.adjust_weights_based_on_accuracy()
            
            if weights_changed:
                # Update the prediction_timeframes dict with new weights
                learned_weights = self.adaptive_weights.get_weights()
                
                for tf_name in self.prediction_timeframes:
                    if tf_name in learned_weights:
                        self.prediction_timeframes[tf_name]["weight"] = learned_weights[tf_name]
                
                # Save updated weights
                import os
                os.makedirs('models', exist_ok=True)
                self.adaptive_weights.save('models/adaptive_timeframe_weights.json')
                
                logger.info(f"✅ Timeframe weights updated based on accuracy!")
                logger.info(f"   New weights:")
                for tf, config in self.prediction_timeframes.items():
                    logger.info(f"     {tf}: {config['weight']:.1%}")
            else:
                logger.info("   Not enough data yet or weights are optimal")
            
            # Log current stats
            stats = self.adaptive_weights.get_stats()
            logger.info(f"📊 Timeframe accuracy stats:")
            for tf in ['15min', '30min', '1hour', '4hour']:
                accuracy = stats['accuracy_by_timeframe'].get(tf, 0)
                count = stats['prediction_counts'].get(tf, 0)
                correct = stats['correct_predictions'].get(tf, 0)
                if count > 0:
                    logger.info(f"     {tf}: {accuracy:.1%} ({correct}/{count} predictions)")
                
        except Exception as e:
            logger.error(f"Error updating adaptive timeframe weights: {e}")
    
    def online_tune_from_prediction_error(self, 
                                          predicted_return: float, 
                                          actual_return: float,
                                          features: np.ndarray,
                                          timeframe: str = "15min"):
        """
        ONLINE LEARNING: Tune the LSTM model based on prediction errors.
        
        Called when we validate a prediction (e.g., 5 minutes after a 5min prediction).
        This is the KEY method for making predictions more accurate over time!
        
        Args:
            predicted_return: What the model predicted (e.g., +0.1%)
            actual_return: What actually happened (e.g., -0.2%)
            features: The feature vector that was used for prediction
            timeframe: Which timeframe this was (for logging)
        """
        if self.model is None or self.optimizer is None or not self.training_enabled:
            return None
        
        try:
            # Calculate prediction error
            prediction_error = actual_return - predicted_return
            error_magnitude = abs(prediction_error)
            
            # Only tune if error is significant (>0.05% for short timeframes)
            if error_magnitude < 0.0005:  # Less than 0.05% error - good enough
                return None
            
            # Use a LOWER learning rate for online tuning (10% of normal)
            # This prevents overreacting to individual prediction errors
            online_lr = self.learning_rate * 0.1
            
            # Temporarily adjust learning rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = online_lr
            
            self.model.train()
            self.optimizer.zero_grad()
            
            # Prepare input
            if len(self.feature_buffer) >= self.sequence_length:
                # seq: sequence of historical features [batch, seq_len, feature_dim]
                sequence = torch.tensor(
                    list(self.feature_buffer)[-self.sequence_length:],
                    dtype=torch.float32
                ).unsqueeze(0).to(self.device)
                
                # cur: current feature vector [batch, feature_dim] - use FULL features, not just last 10!
                current_features = self._safe_to_device(
                    torch.tensor(features, dtype=torch.float32).unsqueeze(0)
                )
                
                # Forward pass: model(cur, seq) - current features first, then sequence
                output = self.model(current_features, sequence)
                
                # Loss based on ACTUAL return (what really happened)
                target_return_tensor = self._safe_to_device(
                    torch.tensor([actual_return], dtype=torch.float32)
                )
                
                # Simple MSE loss for return prediction
                loss = self.loss_fn(output["return"].squeeze(), target_return_tensor)
                
                # Also add direction loss if direction was wrong
                actual_direction = 2 if actual_return > 0.001 else (0 if actual_return < -0.001 else 1)
                target_dir_tensor = self._safe_to_device(
                    torch.tensor([actual_direction], dtype=torch.long)
                )
                direction_loss = nn.CrossEntropyLoss()(output["direction"], target_dir_tensor)
                
                total_loss = loss + direction_loss * 2.0  # Weight direction more
                
                # Backward pass
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)  # Tighter clip for online
                self.optimizer.step()
                
                # Restore normal learning rate
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.learning_rate
                
                # Track online learning stats
                if not hasattr(self, 'online_tune_count'):
                    self.online_tune_count = 0
                    self.online_tune_total_error = 0.0
                
                self.online_tune_count += 1
                self.online_tune_total_error += error_magnitude
                
                # Log significant corrections
                if error_magnitude > 0.002:  # >0.2% error
                    logger.info(f"🎯 ONLINE TUNE [{timeframe}]: pred={predicted_return*100:+.2f}% actual={actual_return*100:+.2f}% error={prediction_error*100:+.2f}%")
                
                return total_loss.item()
                
        except Exception as e:
            logger.debug(f"Online tuning error: {e}")
            # Restore learning rate on error
            if self.optimizer:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.learning_rate
            return None
    
    def validate_and_tune_predictions(self, current_price: float):
        """
        Validate past predictions and tune the model based on errors.
        
        This should be called every cycle to check if any predictions
        can now be validated (their timeframe has elapsed).
        
        IMPORTANT: This also feeds the calibration buffer for PRIMARY_HORIZON predictions.
        Calibration uses DIRECTION accuracy (not return error) for Platt scaling.
        """
        if not hasattr(self, '_pending_predictions'):
            self._pending_predictions = []
            return
        
        current_time = self.get_market_time()
        validated = []
        primary_horizon_validated = 0
        
        for pred in self._pending_predictions:
            pred_time = pred.get('timestamp')
            pred_price = pred.get('price')
            timeframe_min = pred.get('timeframe_minutes', 15)
            confidence = pred.get('confidence', 0.5)  # Get stored confidence
            
            if not pred_time or not pred_price:
                continue
            
            # SANITY CHECK: pred_price should be a valid stock price, not a feature value
            # SPY typically trades between 100-1000. If pred_price is tiny, it's a bug.
            if pred_price < 50 or pred_price > 2000:
                logger.warning(f"[ONLINE TUNE] Invalid pred_price={pred_price:.4f} (expected ~600 for SPY) - skipping")
                validated.append(pred)  # Remove this corrupted prediction
                continue
            
            # Check if enough time has passed to validate
            minutes_elapsed = (current_time - pred_time).total_seconds() / 60
            
            if minutes_elapsed >= timeframe_min:
                # This prediction can now be validated!
                predicted_return = pred.get('predicted_return', 0)
                actual_return = (current_price - pred_price) / pred_price
                features = pred.get('features')
                timeframe = pred.get('timeframe', '15min')
                
                # SANITY CHECK: Skip extreme actual returns (>20% move in minutes is impossible)
                if abs(actual_return) > 0.20:
                    logger.warning(f"[ONLINE TUNE] Skipping extreme return {actual_return*100:.1f}% (likely data error)")
                    validated.append(pred)
                    continue
                
                # Track prediction accuracy and UPDATE CALIBRATION BUFFER
                # This feeds Platt scaling with direction accuracy data
                self._track_prediction_accuracy(
                    predicted_return, actual_return, confidence, 
                    horizon_minutes=timeframe_min
                )
                
                if timeframe_min == self.PRIMARY_HORIZON_MINUTES:
                    primary_horizon_validated += 1
                
                # Tune the model based on this error
                if features is not None:
                    self.online_tune_from_prediction_error(
                        predicted_return, actual_return, features, timeframe
                    )
                
                validated.append(pred)
        
        # Remove validated predictions
        for pred in validated:
            if pred in self._pending_predictions:
                self._pending_predictions.remove(pred)
        
        # Log primary horizon validations (these feed calibration)
        if primary_horizon_validated > 0:
            cal_size = len(self.accuracy_tracker['confidence_calibration'])
            logger.debug(f"[CALIBRATION] Validated {primary_horizon_validated} primary ({self.PRIMARY_HORIZON_MINUTES}m) predictions | Buffer: {cal_size}")
        
        # Keep buffer manageable
        if len(self._pending_predictions) > 500:
            self._pending_predictions = self._pending_predictions[-500:]
    
    def store_prediction_for_online_tuning(self, 
                                           timeframe: str,
                                           timeframe_minutes: int,
                                           predicted_return: float,
                                           current_price: float,
                                           features: np.ndarray,
                                           confidence: float = 0.5):
        """
        Store a prediction so it can be validated and used for online tuning later.
        
        Args:
            timeframe: Name like '15min', '30min'
            timeframe_minutes: Minutes until prediction target
            predicted_return: The predicted return
            current_price: Price when prediction was made
            features: Feature vector for model tuning
            confidence: Model confidence (0-1) - needed for calibration!
        """
        if not hasattr(self, '_pending_predictions'):
            self._pending_predictions = []
        
        self._pending_predictions.append({
            'timeframe': timeframe,
            'timeframe_minutes': timeframe_minutes,
            'predicted_return': predicted_return,
            'price': current_price,
            'features': features.copy() if features is not None else None,
            'confidence': confidence,  # Store confidence for calibration
            'timestamp': self.get_market_time()
        })
    
    def _get_confidence_bracket(self, confidence: float) -> str:
        """Get the bracket key for a confidence value"""
        conf_pct = confidence * 100
        if conf_pct < 30:
            return '20-30'
        elif conf_pct < 40:
            return '30-40'
        elif conf_pct < 50:
            return '40-50'
        elif conf_pct < 60:
            return '50-60'
        elif conf_pct < 70:
            return '60-70'
        elif conf_pct < 80:
            return '70-80'
        else:
            return '80+'

    def update_adaptive_confidence(self, trade_confidence: float, trade_pnl: float = None):
        """Update the adaptive confidence threshold based on ACTUAL trading performance (P&L)"""
        try:
            # Track confidence history
            self.confidence_history.append(trade_confidence)
            self.trades_since_last_adjustment += 1
            
            # Track P&L by confidence bracket (NEW!)
            if trade_pnl is not None:
                bracket = self._get_confidence_bracket(trade_confidence)
                self.confidence_brackets[bracket]['trades'].append(trade_confidence)
                self.confidence_brackets[bracket]['pnl'].append(trade_pnl)
                
                # Keep only recent history per bracket
                if len(self.confidence_brackets[bracket]['trades']) > self.confidence_bracket_window:
                    self.confidence_brackets[bracket]['trades'] = self.confidence_brackets[bracket]['trades'][-self.confidence_bracket_window:]
                    self.confidence_brackets[bracket]['pnl'] = self.confidence_brackets[bracket]['pnl'][-self.confidence_bracket_window:]
            
            # Keep only recent history (last 50 trades)
            if len(self.confidence_history) > 50:
                self.confidence_history = self.confidence_history[-50:]
            
            # Adjust threshold every N trades
            if self.trades_since_last_adjustment >= self.adjustment_interval:
                self.trades_since_last_adjustment = 0
                
                # PERFORMANCE-BASED ADJUSTMENT (prioritize this!)
                best_bracket = None
                best_avg_pnl = float('-inf')
                best_bracket_confidence = 0
                
                # Find the most profitable confidence bracket
                for bracket_name, bracket_data in self.confidence_brackets.items():
                    if len(bracket_data['pnl']) >= 3:  # Need at least 3 trades
                        avg_pnl = sum(bracket_data['pnl']) / len(bracket_data['pnl'])
                        if avg_pnl > best_avg_pnl:
                            best_avg_pnl = avg_pnl
                            best_bracket = bracket_name
                            best_bracket_confidence = sum(bracket_data['trades']) / len(bracket_data['trades'])
                
                # If we found a profitable bracket, adjust toward it!
                if best_bracket and best_avg_pnl > 0:
                    old_threshold = self.min_confidence_threshold
                    
                    # Target the middle of the best performing bracket
                    if best_bracket_confidence > self.min_confidence_threshold + 0.05:
                        # Best bracket is above threshold - gradually raise toward it
                        self.min_confidence_threshold = min(
                            best_bracket_confidence,
                            self.min_confidence_threshold + self.confidence_adaptation_rate * self.adjustment_interval * 2
                        )
                    elif best_bracket_confidence < self.min_confidence_threshold - 0.05:
                        # Best bracket is below threshold - gradually lower toward it
                        self.min_confidence_threshold = max(
                            self.base_confidence_threshold,
                            self.min_confidence_threshold - self.confidence_adaptation_rate * self.adjustment_interval
                        )
                    
                    if self.min_confidence_threshold != old_threshold:
                        logger.info(f"[PERF] Performance-Based Adjustment: Threshold {old_threshold:.1%} -> {self.min_confidence_threshold:.1%}")
                        logger.info(f"[STATS] Best bracket: {best_bracket} (avg P&L: ${best_avg_pnl:+.2f}, {len(self.confidence_brackets[best_bracket]['pnl'])} trades)")
                
                # FALLBACK: Old confidence-based logic (if no P&L data yet)
                else:
                    # Calculate average confidence of recent trades
                    recent_trades = self.confidence_history[-self.adjustment_interval:]
                    avg_confidence = sum(recent_trades) / len(recent_trades)
                    
                    # Gradually increase threshold towards target
                    if avg_confidence > self.min_confidence_threshold + 0.05:
                        old_threshold = self.min_confidence_threshold
                        self.min_confidence_threshold = min(
                            self.target_confidence_threshold,
                            self.min_confidence_threshold + self.confidence_adaptation_rate * self.adjustment_interval
                        )
                        
                        if self.min_confidence_threshold != old_threshold:
                            logger.info(f"[ADAPT] Confidence-Based Adjustment: Threshold {old_threshold:.1%} -> {self.min_confidence_threshold:.1%}")
                            logger.info(f"[STATS] Recent avg confidence: {avg_confidence:.1%} (no P&L data yet)")
                    
                    # Decrease threshold if confidence is consistently low
                    elif avg_confidence < self.min_confidence_threshold - 0.02:
                        old_threshold = self.min_confidence_threshold
                        self.min_confidence_threshold = max(
                            self.base_confidence_threshold,
                            self.min_confidence_threshold - self.confidence_adaptation_rate * self.adjustment_interval * 0.5
                        )
                        
                        if self.min_confidence_threshold != old_threshold:
                            logger.info(f"[ADAPT] Confidence-Based Adjustment: Threshold {old_threshold:.1%} -> {self.min_confidence_threshold:.1%}")
                        
        except Exception as e:
            logger.error(f"Error updating adaptive confidence: {e}")
    
    def check_no_trade_activity(self):
        """
        Detect when signals are being generated but no trades are executing.
        Force threshold down if bot is stuck not trading.
        """
        try:
            # Count current trades (active + total completed)
            current_trade_count = len(self.paper_trader.active_trades) + self.paper_trader.total_trades
            
            # Check if we've made any new trades
            if current_trade_count == self.last_trade_count:
                self.signals_since_last_trade += 1
            else:
                # Reset counter when trades happen
                self.signals_since_last_trade = 0
                self.last_trade_count = current_trade_count
            
            # If we've generated many signals but no trades, force threshold down
            # BUT don't lower it too much - bad trades are worse than no trades!
            if self.signals_since_last_trade >= self.no_trade_threshold:
                old_threshold = self.min_confidence_threshold
                
                # Only lower threshold if above the safe minimum
                # 40% is the floor - don't trade on weak signals!
                self.min_confidence_threshold = max(
                    0.40,  # Absolute minimum 40% (was 20% - too low!)
                    self.min_confidence_threshold - 0.03  # Smaller drops (was 5%)
                )
                # CRITICAL: Also update base_confidence_threshold (used by simulation/trading logic)
                self.base_confidence_threshold = self.min_confidence_threshold
                
                if self.min_confidence_threshold != old_threshold:
                    logger.warning(f"⚠️ NO TRADE ACTIVITY DETECTED!")
                    logger.warning(f"📉 Forcing ALL thresholds down: {old_threshold:.1%} → {self.min_confidence_threshold:.1%}")
                    logger.warning(f"📊 {self.signals_since_last_trade} signals generated with 0 trades")
                    logger.info(f"💡 Bot will now be more aggressive (base_confidence_threshold={self.base_confidence_threshold:.1%})")
                    
                    # Reset counter after adjustment
                    self.signals_since_last_trade = 0
                
        except Exception as e:
            logger.error(f"Error checking no-trade activity: {e}")
    
    def auto_tune_thresholds_from_data(self):
        """
        Auto-tune thresholds based on recent signal patterns.
        Called every N signals to adapt to current market conditions.
        """
        try:
            import numpy as np
            
            # Get recent signals from database
            conn = sqlite3.connect("data/unified_options_bot.db")
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT confidence, predicted_price_15min, current_price
                FROM unified_signals
                WHERE confidence IS NOT NULL
                ORDER BY timestamp DESC
                LIMIT 200
            """)
            
            signals = cursor.fetchall()
            conn.close()
            
            if len(signals) < 50:
                logger.info("Not enough signals for auto-tuning yet (need 50+)")
                return
            
            # Extract metrics
            confidences = [s[0] for s in signals if s[0]]
            predicted_returns = []
            for conf, pred_price, current_price in signals:
                if pred_price and current_price and current_price > 0:
                    pred_return = abs((pred_price - current_price) / current_price)
                    predicted_returns.append(pred_return)
            
            # Calculate data-driven thresholds at 30-35th percentile
            # This allows 65-70% of signals to pass (balanced approach)
            
            old_thresholds = {
                'confidence': self.base_confidence_threshold,
                'return': self.neural_return_threshold,
                'momentum': self.momentum_threshold,
                'volume': self.volume_threshold
            }
            
            if confidences:
                # Use 30th percentile for confidence
                new_conf = np.percentile(confidences, 30)
                self.base_confidence_threshold = max(0.20, min(0.45, new_conf))
                self.min_confidence_threshold = self.base_confidence_threshold
            
            if predicted_returns:
                # Use 35th percentile for return
                new_ret = np.percentile(predicted_returns, 35)
                self.neural_return_threshold = max(0.0001, min(0.002, new_ret))  # Allow smaller thresholds (0.01% to 0.2%)
            
            # Momentum and volume would need market data analysis
            # For now, use conservative values that work with typical patterns
            self.momentum_threshold = 0.001  # Realistic for 1min intervals
            self.volume_threshold = 0.5  # Realistic for typical volume patterns
            
            logger.info("🎯 AUTO-TUNED THRESHOLDS FROM DATA:")
            logger.info(f"   Confidence: {old_thresholds['confidence']:.1%} → {self.base_confidence_threshold:.1%}")
            logger.info(f"   Return: {old_thresholds['return']:.3f} → {self.neural_return_threshold:.3f}")
            logger.info(f"   Momentum: {old_thresholds['momentum']:.4f} → {self.momentum_threshold:.4f}")
            logger.info(f"   Volume: {old_thresholds['volume']:.2f}x → {self.volume_threshold:.2f}x")
            logger.info(f"   Based on {len(signals)} recent signals (30-35th percentile)")
            
        except Exception as e:
            logger.error(f"Error auto-tuning thresholds: {e}")

    def simulate_learning_trade(self, signal: Dict, symbol: str) -> None:
        """Simulate a trade for learning purposes during off-market hours"""
        try:
            current_price = signal.get("current_price", 0.0)
            action = signal.get("action", "HOLD")
            confidence = signal.get("confidence", 0.0)
            strategy = signal.get("strategy", "N/A")
            reasoning = "; ".join(signal.get("reasoning", []))
            
            # Simulate trade outcome based on signal quality
            # For learning purposes, we'll simulate realistic outcomes
            entry_price = current_price
            
            # Simulate price movement based on signal confidence and action
            if action == "BUY":
                # Simulate positive movement for buy signals
                price_change_pct = (confidence - 0.5) * 0.02  # 0-2% movement based on confidence
                exit_price = entry_price * (1 + price_change_pct)
            elif action == "SELL":
                # Simulate negative movement for sell signals  
                price_change_pct = -(confidence - 0.5) * 0.02  # 0-2% movement based on confidence
                exit_price = entry_price * (1 + price_change_pct)
            else:
                # HOLD - no movement
                exit_price = entry_price
                price_change_pct = 0.0
            
            # Calculate simulated P&L (assuming 100 shares for simplicity)
            shares = 100
            simulated_pnl = shares * (exit_price - entry_price)
            
            # Simulate trade duration (15 minutes to 2 hours)
            duration_minutes = 15 + int(confidence * 105)  # 15-120 minutes
            
            # Log to learning database
            conn = sqlite3.connect("data/learning_trades.db")
            conn.execute(
                """
                INSERT INTO learning_trades (
                    timestamp, symbol, current_price,
                    signal_action, signal_confidence, signal_strategy, signal_reasoning,
                    simulated_entry_price, simulated_exit_price, simulated_pnl, simulated_duration_minutes,
                    market_hours, learning_mode
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self.get_market_time().isoformat(),
                    symbol,
                    current_price,
                    action,
                    confidence,
                    strategy,
                    reasoning,
                    entry_price,
                    exit_price,
                    simulated_pnl,
                    duration_minutes,
                    False,  # market_hours = False (off-hours learning)
                    "simulation"
                )
            )
            conn.commit()
            conn.close()
            
            logger.info(f"🧠 Learning trade simulated: {action} {symbol} @ ${entry_price:.2f} → ${exit_price:.2f} "
                       f"(P&L: ${simulated_pnl:.2f}, Duration: {duration_minutes}min)")
            
        except Exception as e:
            logger.error(f"Learning trade simulation error: {e}")

    def has_new_data_for_learning(self, symbol: str = "BITX") -> Dict:
        """Check if NEW data has arrived since last check - only learn on new data!"""
        try:
            current_time = self.get_market_time()
            
            # Check BITX data for NEW data
            bitx_data = self.safe_get_data(symbol, period="1d", interval="1m")
            bitx_has_new = False
            bitx_age_minutes = 999
            
            if not bitx_data.empty:
                try:
                    latest_time = bitx_data.index[-1]
                    # Normalize timezone for both age calculation and comparison
                    if hasattr(latest_time, 'tzinfo') and latest_time.tzinfo is not None:
                        latest_time = latest_time.replace(tzinfo=None)
                    market_time = current_time.replace(tzinfo=None) if hasattr(current_time, 'tzinfo') and current_time.tzinfo else current_time
                    
                    # Safe age calculation: Ensure years are valid (avoid 0001-01-01 overflow)
                    if latest_time.year > 2000 and market_time.year > 2000:
                        bitx_age_minutes = (market_time - latest_time).total_seconds() / 60
                    else:
                        bitx_age_minutes = 999
                        
                    # Ensure stored timestamp is also timezone-naive
                    last_bitx_ts_naive = self.last_bitx_timestamp
                    if last_bitx_ts_naive is not None and hasattr(last_bitx_ts_naive, 'tzinfo') and last_bitx_ts_naive.tzinfo is not None:
                        last_bitx_ts_naive = last_bitx_ts_naive.replace(tzinfo=None)

                    # Check if this is NEW data (different timestamp than last seen)
                    if last_bitx_ts_naive is None or latest_time > last_bitx_ts_naive:
                        bitx_has_new = True
                        # Always store timezone-naive timestamps
                        self.last_bitx_timestamp = latest_time
                except Exception as bitx_error:
                    logger.error(f"Error processing BITX data timestamp: {bitx_error}")
                    bitx_age_minutes = 999
            
            # Check VIX data for NEW data (always fetch for learning, even after hours)
            # We need VIX data for learning regardless of market hours
            vix_data = self.data_source.get_data("^VIX", period=self.historical_period, interval="1m")
            vix_has_new = False
            vix_age_minutes = 999
            
            if not vix_data.empty:
                try:
                    latest_time = vix_data.index[-1]
                    # Normalize timezone for both age calculation and comparison
                    if hasattr(latest_time, 'tzinfo') and latest_time.tzinfo is not None:
                        latest_time = latest_time.replace(tzinfo=None)
                    market_time = current_time.replace(tzinfo=None) if hasattr(current_time, 'tzinfo') and current_time.tzinfo else current_time
                    
                    # Safe age calculation: Ensure years are valid (avoid 0001-01-01 overflow)
                    if latest_time.year > 2000 and market_time.year > 2000:
                        vix_age_minutes = (market_time - latest_time).total_seconds() / 60
                    else:
                        vix_age_minutes = 999
                    
                    # Ensure stored timestamp is also timezone-naive
                    last_vix_ts_naive = self.last_vix_timestamp
                    if last_vix_ts_naive is not None and hasattr(last_vix_ts_naive, 'tzinfo') and last_vix_ts_naive.tzinfo is not None:
                        last_vix_ts_naive = last_vix_ts_naive.replace(tzinfo=None)
                    
                    # Check if this is NEW data (different timestamp than last seen)
                    if last_vix_ts_naive is None or latest_time > last_vix_ts_naive:
                        vix_has_new = True
                        # Always store timezone-naive timestamps
                        self.last_vix_timestamp = latest_time
                except Exception as vix_error:
                    logger.error(f"Error processing VIX data timestamp: {vix_error}")
                    vix_age_minutes = 999
            else:
                # No VIX data available - this prevents learning regardless of market state
                vix_age_minutes = 999  # Mark as stale
                vix_has_new = False
                logger.debug("No VIX data available - learning requires both BITX and VIX data")
            
            # Learning condition: Require BOTH BITX and VIX data to be fresh
            # This ensures we always have complete market context for learning
            is_bitx_fresh = bitx_age_minutes <= self.max_data_age_minutes
            is_vix_fresh = vix_age_minutes <= self.max_data_age_minutes

            # The bot should only proceed if both data streams are fresh and aligned in time.
            can_learn = is_bitx_fresh and is_vix_fresh

            return {
                "bitx_has_new": bitx_has_new,
                "vix_has_new": vix_has_new,
                "is_bitx_fresh": is_bitx_fresh,
                "is_vix_fresh": is_vix_fresh,
                "can_learn": can_learn,
                "bitx_age_minutes": bitx_age_minutes,
                "vix_age_minutes": vix_age_minutes
            }
            
        except Exception as e:
            import traceback
            logger.error(f"Error checking for new data: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                "bitx_has_new": False,
                "vix_has_new": False,
                "is_bitx_fresh": False,
                "is_vix_fresh": False,
                "can_learn": False,
                "bitx_age_minutes": 999,
                "vix_age_minutes": 999
            }

    def get_learning_trades_summary(self) -> Dict:
        """Get summary of learning trades for monitoring"""
        try:
            conn = sqlite3.connect("data/learning_trades.db")
            cursor = conn.cursor()
            
            # Get total learning trades
            cursor.execute("SELECT COUNT(*) FROM learning_trades")
            total_trades = cursor.fetchone()[0]
            
            # Get recent learning trades
            cursor.execute("""
                SELECT signal_action, COUNT(*) 
                FROM learning_trades 
                WHERE timestamp > datetime('now', '-1 hour')
                GROUP BY signal_action
            """)
            recent_trades = cursor.fetchall()
            
            conn.close()
            
            return {
                "total_learning_trades": total_trades,
                "recent_trades_last_hour": dict(recent_trades) if recent_trades else {}
            }
        except Exception as e:
            logger.error(f"Learning trades summary error: {e}")
            return {"total_learning_trades": 0, "recent_trades_last_hour": {}}

    def ensure_prediction_ahead(self, symbol: str) -> None:
        """Ensure we always have a prediction that extends 15 minutes ahead of current time"""
        try:
            now = self.get_market_time()
            target_time = now + pd.Timedelta(minutes=15)
            
            # Check if our latest prediction extends far enough ahead
            if self.prediction_history:
                latest_pred = self.prediction_history[-1]
                pred_time_str = latest_pred['prediction_time']
                pred_time = datetime.strptime(pred_time_str, "%Y-%m-%d %H:%M:%S")
                
                # If prediction is more than 10 minutes old, generate a new one
                # Handle timezone differences
                if target_time.tzinfo is not None:
                    target_time = target_time.replace(tzinfo=None)
                time_gap = (target_time - pred_time).total_seconds()
                if time_gap > 600:  # 10 minutes gap
                    logger.info(f"📈 Generating additional prediction to maintain 15-min ahead coverage")
                    # Generate a quick prediction without the full signal logic
                    self.generate_quick_prediction(symbol, now)
            
        except Exception as e:
            logger.error(f"Error ensuring prediction ahead: {e}")
    
    def generate_quick_prediction(self, symbol: str, current_time: datetime) -> None:
        """Generate a quick prediction to maintain 15-minute ahead coverage"""
        try:
            # Get current price
            data = self.data_source.get_data(symbol, period="1d", interval="1m")
            if data.empty:
                return
                
            current_price = float(data["close"].iloc[-1])
            
            # Simple prediction based on recent trend
            if len(data) >= 5:
                recent_prices = data["close"].tail(5)
                trend = (recent_prices.iloc[-1] - recent_prices.iloc[0]) / recent_prices.iloc[0]
                predicted_price = current_price * (1 + trend * 0.1)  # Conservative trend continuation
            else:
                predicted_price = current_price
            
            # Create prediction record
            prediction_time = (current_time + pd.Timedelta(minutes=15)).strftime("%Y-%m-%d %H:%M:%S")
            prediction_record = {
                'timestamp': current_time,
                'features': [],
                'predicted_return': (predicted_price - current_price) / current_price,
                'predicted_volatility': 0.0,
                'current_price': current_price,
                'predicted_price_15min': predicted_price,
                'prediction_time': prediction_time
            }
            
            self.prediction_history.append(prediction_record)
            logger.info(f"📈 Quick prediction: ${current_price:.2f} → ${predicted_price:.2f} at {prediction_time}")
            
        except Exception as e:
            logger.error(f"Error generating quick prediction: {e}")
    
    def check_for_new_data(self, data: pd.DataFrame, symbol: str) -> bool:
        """Check if we have new data since the last prediction"""
        try:
            if data.empty:
                return False
                
            # Get the latest data timestamp
            latest_time = data.index[-1]
            if latest_time.tzinfo is not None:
                latest_time = latest_time.tz_localize(None)
            
            # If we haven't made a prediction yet, we have "new" data
            if not self.last_signal_time:
                return True
                
            # Check if the latest data is newer than our last prediction
            market_time = self.get_market_time()
            if market_time.tzinfo:
                market_time = market_time.replace(tzinfo=None)
                
            # Consider data "new" if it's within the last 2 minutes and newer than last prediction
            time_diff = (market_time - latest_time).total_seconds()
            if time_diff < 120:  # Within 2 minutes
                return latest_time > self.last_signal_time
                
            return False
            
        except Exception as e:
            logger.error(f"Error checking for new data: {e}")
            return True  # Default to generating prediction if we can't check
    
    def generate_unified_signal(self, symbol: str = "SPY") -> Optional[Dict]:
        try:
            # Log market status but don't block signal generation
            if not self.is_market_open():
                logger.info(
                    f"Markets closed — {self.get_market_time().strftime('%Y-%m-%d %H:%M:%S %Z')} (Learning mode)"
                )

            # =====================================================================
            # CALIBRATION: Settle any predictions that have reached their horizon
            # =====================================================================
            # This must happen before we make new predictions so calibration is up-to-date
            if self.calibration_tracker is not None:
                try:
                    # We need current price to settle - fetch it first
                    data = self.data_source.get_data(symbol, period="1d", interval="1m")
                    if not data.empty:
                        settle_price = float(data["close"].iloc[-1])
                        settled = self.calibration_tracker.settle_predictions(settle_price)
                        if settled > 0:
                            metrics = self.calibration_tracker.get_metrics()
                            logger.info(f"📐 Settled {settled} predictions | Accuracy: {metrics['direction_accuracy']:.1%} | Buffer: {metrics['sample_count']}")
                except Exception as e:
                    logger.debug(f"Could not settle calibration predictions: {e}")

            # Check if we have new data since last prediction
            now = self.get_market_time()
            simulation_mode = getattr(self, "simulation_mode", False)
            
            # Always generate prediction if we haven't generated one yet
            if not self.last_signal_time:
                logger.info("Generating first prediction...")
            else:
                # Check if we have new data (data updates every minute typically)
                time_since_last = (now - self.last_signal_time).total_seconds()
                # In simulation/time‑travel mode, always allow predictions every cycle
                if (not simulation_mode) and time_since_last < 30:  # Minimum 30 seconds between predictions to avoid spam
                    logger.info(f"Skipping — only {time_since_last:.1f}s since last prediction.")
                    return None

            # Clear cache to force fresh data for all active symbols
            # Note: Cache clearing is optional - data is fetched fresh anyway
            try:
                import sqlite3
                # Check if cache_db_path exists on data_source
                if hasattr(self.data_source, 'cache_db_path'):
                    with sqlite3.connect(self.data_source.cache_db_path) as conn:
                        cursor = conn.cursor()
                        cursor.execute("DELETE FROM market_data_cache WHERE symbol = ? AND period = ? AND interval = ?",
                                       (symbol, "1d", "1m"))
                        cursor.execute("DELETE FROM market_data_cache WHERE symbol = ? AND period = ? AND interval = ?",
                                       ("VIX", "1d", "1m"))
                        conn.commit()
                    logger.info(f"Cleared data source cache for fresh {symbol} data")
            except Exception as e:
                # Silently ignore cache clearing errors - not critical
                pass

            # fetch fresh data (fallbacks) - only use fresh data
            interval = "1m"
            data = self.data_source.get_data(
                symbol, period=self.historical_period, interval=interval)
            
            # Check if we have new data since last prediction
            has_new_data = self.check_for_new_data(data, symbol)
            # In simulation/time‑travel mode, skip "new data" gating so every bar is processed
            if (not simulation_mode) and not has_new_data and self.last_signal_time:
                logger.info("No new data available since last prediction")
                return None

            # Check if data is fresh
            if not data.empty:
                latest_time = data.index[-1]
                # Convert to naive datetime for comparison
                if latest_time.tzinfo is not None:
                    latest_time = latest_time.tz_localize(None)
                market_time = self.get_market_time()
                if market_time.tzinfo is not None:
                    market_time = market_time.replace(tzinfo=None)
                time_diff = (market_time - latest_time).total_seconds()

                # Require fresh data (within 5 minutes) ONLY during market hours
                # During after-hours, allow predictions with most recent available data
                is_market_open = self.is_market_open()
                
                if is_market_open and (not simulation_mode):
                    # Market open: require fresh data (within 5 minutes)
                    max_age = 300  # 5 minutes
                    if time_diff > max_age:
                        logger.warning(
                            f"1m data too old ({time_diff/60:.1f} minutes), skipping prediction")
                        return None
                else:
                    # Market closed: allow predictions with most recent available data
                    # But warn if data is VERY old (more than 24 hours)
                    if not simulation_mode:
                        max_age = 86400  # 24 hours
                        if time_diff > max_age:
                            logger.warning(
                                f"Data too old ({time_diff/3600:.1f} hours), skipping prediction")
                            return None
                        elif time_diff > 300:
                            logger.info(
                                f"Using after-hours data ({time_diff/60:.0f} min old) for prediction")

            if data.empty or len(data) < 60:
                interval = "15m"
                data = self.data_source.get_data(
                    symbol, period=self.historical_period, interval=interval)

                # Check freshness for 15m data
                if not data.empty:
                    latest_time = data.index[-1]
                    # Convert to naive datetime for comparison
                    if latest_time.tzinfo is not None:
                        latest_time = latest_time.tz_localize(None)
                    market_time = self.get_market_time()
                    if market_time.tzinfo is not None:
                        market_time = market_time.replace(tzinfo=None)
                    time_diff = (market_time - latest_time).total_seconds()
                    # 15 minutes for 15m data during market hours
                    is_market_open = self.is_market_open()
                    max_age = 900 if is_market_open else 86400  # 15 min or 24 hours
                    if (not simulation_mode) and time_diff > max_age:
                        logger.warning(
                            f"15m data too old ({time_diff/60:.1f} minutes), skipping prediction")
                        return None
                    elif (not simulation_mode) and (not is_market_open) and time_diff > 900:
                        logger.info(f"Using after-hours 15m data ({time_diff/60:.0f} min old)")

                if data.empty or len(data) < 4:
                    interval = "5m"
                    data = self.data_source.get_data(
                        symbol, period=self.historical_period, interval=interval)

                    # Check freshness for 5m data
                    if not data.empty:
                        latest_time = data.index[-1]
                        # Convert to naive datetime for comparison
                        if latest_time.tzinfo is not None:
                            latest_time = latest_time.tz_localize(None)
                        market_time = self.get_market_time()
                        if market_time.tzinfo is not None:
                            market_time = market_time.replace(tzinfo=None)
                        time_diff = (market_time - latest_time).total_seconds()
                        # 5 minutes for 5m data during market hours
                        is_market_open = self.is_market_open()
                        max_age = 300 if is_market_open else 86400  # 5 min or 24 hours
                        if (not simulation_mode) and time_diff > max_age:
                            logger.warning(
                                f"5m data too old ({time_diff/60:.1f} minutes), skipping prediction")
                            return None
                        elif (not simulation_mode) and (not is_market_open) and time_diff > 300:
                            logger.info(f"Using after-hours 5m data ({time_diff/60:.0f} min old)")

                    if data.empty or len(data) < 12:
                        # Only log if not in simulation mode (reduce spam during warmup)
                        if not simulation_mode:
                            logger.warning("Insufficient data for analysis.")
                        return None

            # normalize
            data.columns = data.columns.str.lower()
            data = data.loc[:, ~data.columns.duplicated()]
            data = data.dropna()

            # Check data freshness for both BITX and VIX before making predictions
            # For predictions, we still check data freshness (not necessarily NEW data)
            current_time = self.get_market_time()
            
            # Quick freshness check for predictions (skip VIX if VIX market closed)
            # Use explicit conversion to avoid pandas casting errors with ancient dates
            try:
                bitx_data = self.data_source.get_data(symbol, period=self.historical_period, interval="1m")
            except Exception as e:
                logger.error(f"Error fetching BITX data: {e}")
                return None

            try:
                # Always fetch VIX data for signal generation (needed for learning)
                vix_data = self.data_source.get_data("^VIX", period=self.historical_period, interval="1m")
            except Exception as e:
                logger.error(f"Error fetching VIX data: {e}")
                vix_data = pd.DataFrame() # Empty DF fallback
            
            # Track current VIX for adaptive thresholds (Improvement #3)
            if not vix_data.empty:
                vix_col = 'close' if 'close' in vix_data.columns else 'Close'
                self.current_vix = float(vix_data[vix_col].iloc[-1])
            else:
                self.current_vix = getattr(self, 'current_vix', 17.0)  # Keep last value or default
            
            bitx_fresh = vix_fresh = False
            bitx_age = vix_age = 999
            
            # Market-aware freshness check (skip gating in simulation/time‑travel mode)
            is_market_open = self.is_market_open()
            max_age_minutes = self.max_data_age_minutes if is_market_open else 1440  # 5 min or 24 hours
            
            if not bitx_data.empty:
                try:
                    latest_time = bitx_data.index[-1]
                    if latest_time.tzinfo is not None:
                        latest_time = latest_time.tz_localize(None)
                    market_time = current_time.replace(tzinfo=None) if current_time.tzinfo else current_time
                    
                    # Safe age calculation
                    if market_time.year > 2000 and latest_time.year > 2000:
                        bitx_age = (market_time - latest_time).total_seconds() / 60
                        bitx_fresh = bitx_age <= max_age_minutes
                    else:
                        bitx_age = 0 # Assume fresh if dates are weird to prevent crash
                        bitx_fresh = True
                except Exception as e:
                    logger.error(f"Error checking BITX freshness: {e}")
                    bitx_fresh = True # Fallback
            
            if not vix_data.empty:
                try:
                    latest_time = vix_data.index[-1]
                    if latest_time.tzinfo is not None:
                        latest_time = latest_time.tz_localize(None)
                    market_time = current_time.replace(tzinfo=None) if current_time.tzinfo else current_time
                    
                    # Safe age calculation
                    if market_time.year > 2000 and latest_time.year > 2000:
                        vix_age = (market_time - latest_time).total_seconds() / 60
                        vix_fresh = vix_age <= max_age_minutes
                    else:
                        vix_age = 0 
                        vix_fresh = True
                except Exception as e:
                    logger.error(f"Error checking VIX freshness: {e}")
                    vix_fresh = True # Fallback
            else:
                # VIX data not available - allow predictions to continue with BITX data only
                # This is common during weekends, after-hours, or when VIX sources fail
                vix_age = 999  # Mark as missing
                vix_fresh = True  # But allow predictions to continue
                # Only log if not in simulation mode (reduce noise during warmup)
                if not getattr(self, 'simulation_mode', False):
                    logger.warning(f"⚠️ VIX data unavailable - proceeding with BITX data only")
            
            # Only require fresh BITX data (VIX is optional) in LIVE mode
            if (not simulation_mode) and (not bitx_fresh):
                if is_market_open:
                    logger.warning(f"Predictions require fresh BITX data - BITX: {bitx_age:.1f}min (max: {self.max_data_age_minutes}min)")
                    return None  # Stop prediction if BITX data is stale during market hours
                else:
                    logger.info(f"Using after-hours BITX data ({bitx_age:.0f} min old) for prediction")
                    bitx_fresh = True  # Allow predictions after hours

            # Also fetch VIX data for volatility analysis
            vix_data = self.data_source.get_data(
                "^VIX", period=self.historical_period, interval=interval)
            if not vix_data.empty:
                vix_data.columns = vix_data.columns.str.lower()
                vix_data = vix_data.loc[:, ~vix_data.columns.duplicated()]
                vix_data = vix_data.dropna()
                # logger.info(f"Got VIX data: {len(vix_data)} points")
            else:
                # Only log if not in simulation mode (reduce noise)
                if not getattr(self, 'simulation_mode', False):
                    logger.warning("No VIX data available - predictions require both BITX and VIX data")
                return None  # Stop prediction if VIX data is missing

            # ==============================================================================
            # 🌍 UPDATE MARKET CONTEXT (Critical for RL & Correlation Features)
            # ==============================================================================
            # Fetch context symbols (QQQ, SPY, NVDA, etc.) periodically to keep RL informed
            # Only needed every few minutes, not every cycle
            
            # Initialize check timestamp if missing (use 2000-01-01 as safe epoch for Pandas)
            if not hasattr(self, '_last_context_update'):
                self._last_context_update = datetime(2000, 1, 1)
            
            # Ensure _last_context_update is timezone-naive if current_time is naive
            if current_time.tzinfo is None and getattr(self._last_context_update, 'tzinfo', None) is not None:
                 self._last_context_update = self._last_context_update.replace(tzinfo=None)
            
            time_since_context = (current_time - self._last_context_update).total_seconds()
            
            # Update context every 5 minutes (or if never updated)
            # In simulation mode, update more frequently to match data flow
            context_update_interval = 60 if simulation_mode else 300
            
            if time_since_context > context_update_interval:
                logger.info("🌍 Updating market context symbols...")
                updated_count = 0
                
                # Filter useful symbols only to save time
                important_symbols = ['QQQ', 'SPY', 'IWM', 'NVDA', 'MSFT', 'AAPL', 'BTC-USD', 'UUP']
                target_symbols = [s for s in self.context_symbols if s in important_symbols]
                if not target_symbols: 
                    target_symbols = self.context_symbols[:5] # Fallback to first 5
                
                for ctx_symbol in target_symbols:
                    try:
                        # Use same interval as main symbol
                        ctx_data = self.data_source.get_data(ctx_symbol, period='5d', interval=interval)
                        if not ctx_data.empty:
                            self.update_market_context(ctx_symbol, ctx_data)
                            updated_count += 1
                    except Exception as e:
                        logger.debug(f"Failed to update context for {ctx_symbol}: {e}")
                
                self._last_context_update = current_time
                if updated_count > 0:
                    logger.info(f"🌍 Updated context for {updated_count} symbols")
            
            if data.empty:
                return None

            current_price = float(data["close"].iloc[-1])
            
            # Store the actual SPY price for use by prediction storage functions
            # This ensures online tuning uses the real price, not feature values
            self._last_known_spy_price = current_price

            # 1) options metrics
            options_metrics = self.calculate_options_metrics(
                data, interval_hint=interval)
            if not options_metrics:
                return None

            # 2) Auto-train/update HMM regime detector
            self._maybe_train_hmm(data)
            
            # 3) features (now includes HMM regime features if trained)
            current_features = self.create_features(data)
            if current_features is None:
                return None

            # buffer & maybe init model
            self.feature_buffer.append(current_features)
            if len(self.feature_buffer) > self.max_buffer_size:
                self.feature_buffer.pop(0)

            # Snapshot the temporal window that will be fed to the model for THIS decision.
            # This prevents online learning from pairing an old label with a different (current) window.
            sequence_snapshot = None
            if len(self.feature_buffer) >= self.sequence_length:
                try:
                    sequence_snapshot = np.vstack(self.feature_buffer[-self.sequence_length:]).astype(np.float32)
                except Exception:
                    sequence_snapshot = None

            # Debug logging for feature buffer with update tracking
            buffer_hash = hash(str(current_features[:5]))  # Hash first 5 features for tracking
            logger.info(
                f"Feature buffer size: {len(self.feature_buffer)}/{self.sequence_length} | New features hash: {buffer_hash}")

            self._maybe_init_model()
            
            # ============= ONLINE TUNING: Validate past predictions =============
            # Check if any past predictions can now be validated (timeframe elapsed)
            # This tunes the LSTM to be more accurate over time!
            try:
                self.validate_and_tune_predictions(current_price)
                
                # Log online tuning stats every 50 cycles
                if hasattr(self, 'online_tune_count') and self.online_tune_count > 0:
                    if self.online_tune_count % 50 == 0:
                        avg_error = self.online_tune_total_error / self.online_tune_count * 100
                        logger.info(f"📊 ONLINE TUNING STATS: {self.online_tune_count} corrections, avg error: {avg_error:.2f}%")
            except Exception as tune_err:
                logger.debug(f"Online tuning check: {tune_err}")

            # fit gaussian processor when we have enough (and re-fit periodically)
            should_refit = (
                (len(self.feature_buffer) >= 20 and not self.gaussian_processor.is_fitted) or
                (len(self.feature_buffer) >= 50 and len(self.feature_buffer) % 100 == 0)  # Re-fit every 100 samples
            )
            if should_refit:
                if self.gaussian_processor.fit(self.feature_buffer):
                    refit_type = "initial" if not self.gaussian_processor.is_fitted else "periodic"
                    logger.info(f"Gaussian kernel processor {refit_type} fit completed (buffer size: {len(self.feature_buffer)})")

            # 3) Multi-timeframe neural predictions
            # CRITICAL FIX: Get neural prediction ONCE and cache for multi-timeframe reuse
            # This prevents sign flips due to Monte Carlo sampling in Bayesian NN
            neural_pred = self.get_neural_prediction(current_features)
            self._cached_neural_pred = neural_pred  # Cache for multi-timeframe to reuse
            
            try:
                multi_timeframe_preds = self.get_multi_timeframe_predictions(current_features)
                multi_timeframe_signal = self.combine_multi_timeframe_signals()
            finally:
                self._cached_neural_pred = None  # Clear cache after use
            
            # Store for paper trader's exit decisions (dynamic hold time based on live predictions)
            self.last_multi_timeframe_predictions = multi_timeframe_preds
            
            # Store predictions for online tuning WITH the correct price (not features[-1] which is wrong!)
            try:
                self.store_multi_timeframe_predictions_for_validation(multi_timeframe_preds, current_price)
            except Exception as e:
                logger.debug(f"Could not store predictions for validation: {e}")

            # 3.5) Train neural network with historical data if available
            if len(self.prediction_history) > 0:
                # Get the most recent prediction to train on
                last_pred = self.prediction_history[-1]
                if 'actual_return' in last_pred and 'features' in last_pred:
                    actual_return = last_pred['actual_return']
                    actual_vol = last_pred.get('actual_volatility', None)
                    features = last_pred['features']

                    # Train the neural network
                    loss = self.train_neural_network(
                        features, actual_return, actual_vol)
                    if loss is not None:
                        logger.info(
                            f"Neural network training completed: loss={loss:.6f}")

            # 4) 15-minute forecast
            predicted_price_15min = self.generate_15min_prediction(
                current_price, neural_pred, symbol)
            prediction_time_str = (
                now + pd.Timedelta(minutes=15)).strftime("%Y-%m-%d %H:%M:%S")

            # 5) combine signals with multi-timeframe analysis
            signal = self._combine_signals(
                options_metrics, neural_pred, current_price, multi_timeframe_signal)
            signal["predicted_price_15min"] = float(predicted_price_15min)
            signal["prediction_time"] = prediction_time_str
            signal["current_price"] = float(current_price)
            # Expose predictor embedding for downstream controllers + offline training visibility
            if neural_pred and neural_pred.get("predictor_embedding") is not None:
                signal["predictor_embedding"] = neural_pred.get("predictor_embedding")

            # 5.2) Q_SCORER controller (offline supervised Q regression)
            # If active, it will also suppress entry RL refinement later.
            self._apply_q_entry_controller(signal, neural_pred, options_metrics, symbol)

            # 5.25) ENTRY CONTROLLER: Use world-model edge scorer (predictor + execution heads)
            # This reduces reliance on a tiny MLP policy for entries.
            self._apply_edge_entry_controller(signal, neural_pred, options_metrics, symbol)
            
            # 5.5) RL Policy Decision (if enabled)
            # If ENTRY_CONTROLLER is active, we intentionally do NOT let RL override entry actions.
            # Exits remain handled by the existing exit stack.
            import os
            entry_controller_mode = os.environ.get("ENTRY_CONTROLLER", "edge").strip().lower()
            if self.use_rl_policy and neural_pred and entry_controller_mode not in ("edge", "edge_scorer", "world_model", "q_scorer", "q", "qscorer"):
                logger.info("🎯 Applying RL policy to refine trading decision...")
                try:
                    rl_action, rl_position_size = self.make_rl_decision(
                        signal, neural_pred, options_metrics, symbol
                    )
                    signal["rl_position_size"] = rl_position_size
                    logger.info(f"✅ RL Policy applied: {signal['action']} x{rl_position_size}")
                except Exception as e:
                    logger.error(f"❌ RL policy error: {e}, using original signal")

            # update last signal time
            self.last_signal_time = now
            
            # CACHE signal for position updates (avoids regenerating for each position)
            self._last_generated_signal = signal.copy() if signal else None

            # 6) db log
            logger.info(f"💾 Saving signal with multi-timeframe data: {list(multi_timeframe_preds.keys()) if multi_timeframe_preds else 'None'}")
            row_id = self._log_signal(
                symbol=symbol,
                current_price=current_price,
                options_metrics=options_metrics,
                neural_pred=neural_pred,
                signal=signal,
                prediction_time_str=prediction_time_str,
                current_features=current_features,
                current_sequence=sequence_snapshot,
                multi_timeframe_preds=multi_timeframe_preds
            )
            signal["_row_id"] = row_id

            # 7) Store prediction for future training
            prediction_record = {
                'timestamp': now,
                'features': current_features.copy(),
                'predicted_return': neural_pred.get('predicted_return', 0.0) if neural_pred else 0.0,
                'predicted_volatility': neural_pred.get('predicted_volatility', 0.0) if neural_pred else 0.0,
                'current_price': current_price,
                'predicted_price_15min': predicted_price_15min,
                'prediction_time': prediction_time_str
            }
            self.prediction_history.append(prediction_record)

            # Keep only last 100 predictions to manage memory
            if len(self.prediction_history) > 100:
                self.prediction_history.pop(0)

            # Save model periodically to preserve progress (will be called from main loop)
            # Note: iteration counter is managed in the main loop
            
            # Check for no-trade activity and adjust thresholds if needed
            self.check_no_trade_activity()
            
            # Auto-tune thresholds from data every N signals
            self.signals_since_last_tune += 1
            if self.signals_since_last_tune >= self.auto_tune_interval:
                logger.info(f"🔧 Running periodic threshold auto-tuning ({self.signals_since_last_tune} signals since last tune)")
                self.auto_tune_thresholds_from_data()
                self.signals_since_last_tune = 0

            return signal

        except Exception as e:
            logger.error(f"Signal generation error: {e}")
            return None

    # ------------------------ Main loop ------------------------
    def run_continuous_trading(self, symbol: str = "BITX"):
        logger.info("Starting Starting Unified Options Trading Bot")
        acct = self.paper_trader.get_account_summary()
        bal = float(acct.get("current_balance", acct.get("cash_balance", 0.0)))
        logger.info(f"Balance: Initial Balance: ${bal:,.2f}")
        logger.info(f"Target: Min Confidence: {self.min_confidence_threshold:.1%} (Adaptive: {self.base_confidence_threshold:.1%} → {self.target_confidence_threshold:.1%})")
        logger.info(f"Stats: Max Positions: {self.max_positions}")

        # Optional safety: stop after N cycles (0 = run forever).
        import os
        try:
            max_cycles = int(os.environ.get("BOT_MAX_CYCLES", "0") or "0")
        except Exception:
            max_cycles = 0

        iteration = 0
        while True:
            try:
                iteration += 1
                if max_cycles > 0 and iteration > max_cycles:
                    logger.info(f"[STOP] BOT_MAX_CYCLES reached ({max_cycles}); exiting run loop.")
                    break
                mt = self.get_market_time()
                is_open = self.is_market_open("BITX")  # Use BITX hours for main market state

                # --- State Transition Logic ---
                if is_open and self.market_state == 'CLOSED':
                    logger.info("🌅 Market is now OPEN. Starting trading day.")
                    self.market_state = 'OPEN'
                    self.end_of_day_processed = False # Reset for the new day
                    self.last_close_price = None # Clear last closing price

                if not is_open and self.market_state == 'OPEN':
                    logger.info("🌇 Market is now CLOSED. Initiating end-of-day procedures.")
                    self.market_state = 'CLOSED'
                    # Fetch the final closing price for the day exactly once.
                    self.last_close_price = self.data_source.get_current_price(symbol)
                    if self.last_close_price:
                        logger.info(f"Final closing price for {symbol} captured: ${self.last_close_price:.2f}")
                    else:
                        logger.warning(f"Could not fetch final closing price for {symbol}. Will use fallback.")

                # --- Main Logic: Always run, but gate actions based on state ---
                logger.info("\n" + "=" * 60)
                logger.info(f" Cycle #{iteration} — {mt.strftime('%H:%M:%S %Z')} — Market State: {self.market_state} | BITX: {'OPEN' if self.is_symbol_trading('BITX') else 'CLOSED'} | VIX: {'OPEN' if self.is_symbol_trading('VIX') else 'CLOSED'}")

                # Always get the latest account balance
                acct = self.paper_trader.get_account_summary()
                bal = float(acct.get("current_balance", acct.get("cash_balance", 0.0)))

                # --- Unified Learning Logic: Always learn when fresh data is available ---
                data_freshness = self.has_new_data_for_learning(symbol)
                
                if data_freshness["can_learn"]:
                    # Fresh data available - proceed with learning and signal generation
                    mode_desc = "🌙 After-hours" if self.market_state == 'CLOSED' else "📈 Market hours"
                    logger.info(f"{mode_desc} learning: Fresh BITX & VIX data available, generating signal...")
                    
                    # Continuous learning cycle (same logic for both market states)
                    if (self.get_market_time() - self.last_learning_time).total_seconds() >= self.learning_interval:
                        logger.info("🧠 Starting continuous learning cycle with fresh data...")
                        self.continuous_learning_cycle()
                    
                    # Always update positions and generate signals when we have fresh data
                    self.paper_trader.update_positions(symbol)
                    
                    # Track completed trades for RL learning
                    if self.use_rl_policy:
                        self.track_completed_trades_for_rl()
                    
                    # Update adaptive timeframe weights based on prediction accuracy
                    # This learns which timeframes (15min, 30min, 1hr, 4hr) are most accurate!
                    self.update_adaptive_timeframe_weights()
                    
                    signal = self.generate_unified_signal(symbol)
                    
                    # Market-specific actions
                    if self.market_state == 'OPEN':
                        # During market hours: trade on signals
                        self.log_cycle_summary(iteration, signal, bal, symbol, is_after_hours=False, is_learning_mode=True)
                        executed_action = "HOLD"
                        trade_placed = False
                        rejection_reasons: List[str] = []
                        if signal is None:
                            rejection_reasons.append("no_signal")
                        else:
                            proposed_action = str(signal.get("action", "HOLD"))
                            if proposed_action == "HOLD":
                                rejection_reasons.append("policy_hold")
                            elif float(signal.get("confidence", 0.0)) < float(self.min_confidence_threshold):
                                rejection_reasons.append("confidence_below_threshold")
                            else:
                                trade_placed = bool(self.execute_trade(signal, symbol, signal.get("current_price", 0.0)))
                                if trade_placed:
                                    executed_action = proposed_action
                                else:
                                    # Best-effort surface why execution failed
                                    rej = getattr(self.paper_trader, "last_trade_rejection", None)
                                    if isinstance(rej, dict) and rej.get("reason"):
                                        reason = f"{rej.get('type', 'paper_reject')}: {rej.get('code', '')} {rej.get('reason', '')}".strip()
                                        rejection_reasons.append(f"PaperTrader: {reason}")
                                    else:
                                        rejection_reasons.append("execution_failed")

                        # Write DecisionRecord/MissedOpportunityRecord if enabled
                        try:
                            import os
                            if os.environ.get("DECISION_PIPELINE_ENABLED", "1").strip() == "1" and HAS_DECISION_PIPELINE:
                                if not hasattr(self, "decision_pipeline") or self.decision_pipeline is None:
                                    self.decision_pipeline = DecisionPipeline()  # type: ignore
                                if self.decision_pipeline:
                                    _sig = signal if signal is not None else {
                                        "action": "HOLD",
                                        "strategy": "NO_SIGNAL",
                                        "confidence": 0.0,
                                        "current_price": float(getattr(self, "last_close_price", 0.0) or 0.0),
                                        "reasoning": ["no_signal"],
                                    }
                                    self.decision_pipeline.record_decision(
                                        timestamp=mt,
                                        symbol=symbol,
                                        current_price=float(_sig.get("current_price", 0.0) or 0.0),
                                        signal=_sig,
                                        proposed_action=str(_sig.get("action", "HOLD")),
                                        executed_action=executed_action,
                                        trade_placed=trade_placed,
                                        rejection_reasons=rejection_reasons,
                                        paper_rejection=getattr(self.paper_trader, "last_trade_rejection", None),
                                        bot=self,
                                    )
                        except Exception:
                            pass
                    else:
                        # After hours: learn but don't trade
                        if not self.end_of_day_processed:
                            logger.info("Running end-of-day tasks...")
                            logger.info("🌙 Performing final end-of-day learning cycle...")
                            self.continuous_learning_cycle()
                            self.end_of_day_processed = True
                            logger.info("✅ End-of-day tasks complete.")
                        
                        self.log_cycle_summary(iteration, signal, bal, symbol, is_after_hours=True, is_learning_mode=True)
                        # Also log decisions after-hours as HOLD (useful for offline datasets)
                        try:
                            import os
                            if os.environ.get("DECISION_PIPELINE_ENABLED", "1").strip() == "1" and HAS_DECISION_PIPELINE:
                                if not hasattr(self, "decision_pipeline") or self.decision_pipeline is None:
                                    self.decision_pipeline = DecisionPipeline()  # type: ignore
                                if self.decision_pipeline:
                                    _sig = signal if signal is not None else {
                                        "action": "HOLD",
                                        "strategy": "AFTER_HOURS",
                                        "confidence": 0.0,
                                        "current_price": float(getattr(self, "last_close_price", 0.0) or 0.0),
                                        "reasoning": ["market_closed_after_hours"],
                                    }
                                    self.decision_pipeline.record_decision(
                                        timestamp=mt,
                                        symbol=symbol,
                                        current_price=float(_sig.get("current_price", 0.0) or 0.0),
                                        signal=_sig,
                                        proposed_action=str(_sig.get("action", "HOLD")),
                                        executed_action="HOLD",
                                        trade_placed=False,
                                        rejection_reasons=["market_closed_after_hours"],
                                        paper_rejection=None,
                                        bot=self,
                                    )
                        except Exception:
                            pass
                    
                    # Save model periodically (same for both market states)
                    if iteration % 10 == 0 and self.model_initialized:
                        self.save_model()
                        
                    # Check cache health every hour (60 cycles) - market hours only
                    if self.market_state == 'OPEN' and iteration % 60 == 0:
                        if not self.check_cache_health():
                            logger.warning("🔄 Cache corruption detected - will rebuild automatically")
                            
                else:
                    # No fresh data available - monitoring mode only
                    mode_desc = "⏸ After-hours monitoring" if self.market_state == 'CLOSED' else "⏸ MONITORING MODE"
                    is_after_hours = self.market_state == 'CLOSED'
                    
                    self.log_cycle_summary(iteration, None, bal, symbol, is_after_hours=is_after_hours, is_learning_mode=False)
                    # Log monitoring-mode decisions as HOLD (useful for offline datasets).
                    try:
                        import os
                        if os.environ.get("DECISION_PIPELINE_ENABLED", "1").strip() == "1" and HAS_DECISION_PIPELINE:
                            if not hasattr(self, "decision_pipeline") or self.decision_pipeline is None:
                                self.decision_pipeline = DecisionPipeline()  # type: ignore
                            if self.decision_pipeline:
                                _sig = {
                                    "action": "HOLD",
                                    "strategy": "MONITORING",
                                    "confidence": 0.0,
                                    "current_price": float(getattr(self, "last_close_price", 0.0) or 0.0),
                                    "reasoning": ["monitoring_mode_no_fresh_data"],
                                }
                                self.decision_pipeline.record_decision(
                                    timestamp=mt,
                                    symbol=symbol,
                                    current_price=float(_sig.get("current_price", 0.0) or 0.0),
                                    signal=_sig,
                                    proposed_action="HOLD",
                                    executed_action="HOLD",
                                    trade_placed=False,
                                    rejection_reasons=["monitoring_mode_no_fresh_data"],
                                    paper_rejection=None,
                                    bot=self,
                                )
                    except Exception:
                        pass
                    
                    if self.market_state == 'CLOSED':
                        # Handle end-of-day tasks even without fresh data
                        if not self.end_of_day_processed:
                            logger.info("Running end-of-day tasks...")
                            self.paper_trader.update_positions(symbol)
                            logger.info("🌙 Performing final end-of-day learning cycle...")
                            self.continuous_learning_cycle()
                            self.save_model()
                            self.end_of_day_processed = True
                            logger.info("✅ End-of-day tasks complete.")
                        logger.info(f"{mode_desc}: No fresh data - using cached data for summary")
                    else:
                        logger.info(f"{mode_desc}: Data is stale - no learning or trading until fresh data arrives")

                # Report metrics to Data Manager periodically (every 20 cycles ~5 min)
                if self.bot_reporter and self.bot_reporter.enabled and iteration % 20 == 0:
                    self.report_metrics_to_data_manager()

                # Consistent sleep interval to keep the bot responsive for the dashboard
                # Use signal_interval (15s) to enable rapid predictions on new data
                logger.info(f"⏰ Sleeping for {self.signal_interval} seconds...")
                time.sleep(self.signal_interval)

            except KeyboardInterrupt:
                logger.info("🛑 Bot stopped by user.")
                self.graceful_shutdown()
                break
            except Exception as e:
                import traceback

                logger.error(f"ERROR: Error in trading cycle: {e}")
                logger.error(f"ERROR: Traceback:\n{traceback.format_exc()}")

                time.sleep(30)
                
        # Ensure final save on exit
        self.graceful_shutdown()

    def log_cycle_summary(self, iteration: int, signal: Optional[Dict], balance: float, symbol: str, is_after_hours: bool = False, is_learning_mode: bool = False):
        """Log comprehensive cycle summary with price, VIX, signal, balance, and P&L"""
        try:
            # Get current market data
            if is_after_hours:
                # After hours, use fallback prices to avoid rate limiting
                current_price = self.last_close_price or 58.0  # Use stored closing price or fallback
                vix_price = 16.5  # Use reasonable VIX fallback to avoid API calls
            else:
                data = self.data_source.get_data(symbol, period=self.historical_period, interval="1m")
                current_price = float(data["close"].iloc[-1]) if not data.empty else 0.0
                vix_data = self.data_source.get_data("^VIX", period=self.historical_period, interval="1m")
                vix_price = float(vix_data["close"].iloc[-1]) if not vix_data.empty else 0.0
            
            # Calculate P&L from account summary
            acct = self.paper_trader.get_account_summary()
            total_pnl = float(acct.get("total_pnl", 0.0))
            unrealized_pnl = float(acct.get("unrealized_pnl", 0.0))
            
            # Get active positions info
            active_trades = len(self.paper_trader.active_trades)
            
            # Signal information
            if signal:
                action = signal.get("action", "UNKNOWN")
                strategy = signal.get("strategy", "UNKNOWN")
                confidence = signal.get("confidence", 0.0)
                predicted_price = signal.get("predicted_price_15min", current_price)
                risk_level = signal.get("risk_level", "UNKNOWN")
            else:
                action = "NO_SIGNAL"
                strategy = "N/A"
                confidence = 0.0
                predicted_price = current_price
                risk_level = "N/A"
            
            # Calculate price change prediction
            price_change_pred = predicted_price - current_price
            price_change_pct = (price_change_pred / current_price * 100) if current_price > 0 else 0.0
            
            # Get learning mode status
            if is_after_hours:
                market_mode = "AFTER HOURS"
            elif is_learning_mode:
                market_mode = "LEARNING"
            else:
                market_mode = "MONITORING"
            learning_summary = self.get_learning_trades_summary()
            
            # Get data freshness for summary
            data_freshness = self.has_new_data_for_learning(symbol)
            
            # Create comprehensive summary
            logger.info("🔍 " + "=" * 80)
            logger.info(f"📊 CYCLE #{iteration} SUMMARY ({market_mode} MODE)")
            logger.info("🔍 " + "=" * 80)
            logger.info(f"💰 BITX: ${current_price:.2f} | VIX: {vix_price:.2f}")
            logger.info(f"🎯 SIGNAL: {action} ({strategy}) | Confidence: {confidence:.1%} | Risk: {risk_level}")
            logger.info(f"📈 PREDICTION: ${predicted_price:.2f} ({price_change_pct:+.2f}% in 15min)")
            logger.info(f"💼 BALANCE: ${balance:,.2f} | P&L: ${total_pnl:+,.2f} | Unrealized: ${unrealized_pnl:+,.2f}")
            logger.info(f"📋 POSITIONS: {active_trades} active trades")
            
            # Add learning mode and data freshness information
            if not self.is_market_open():
                logger.info(f"🧠 LEARNING MODE: {learning_summary['total_learning_trades']} total simulated trades")
            
            # Always show data freshness status
            # For display purposes, consider data "fresh" if it's reasonably recent, not necessarily "new"
            bitx_fresh = data_freshness.get("bitx_age_minutes", 999) <= self.max_data_age_minutes
            vix_fresh = data_freshness.get("vix_age_minutes", 999) <= self.max_data_age_minutes
            freshness_status = "✅ FRESH" if (bitx_fresh and vix_fresh) else "⚠️ STALE"
            logger.info(f"📊 DATA STATUS: {freshness_status} | BITX: {data_freshness['bitx_age_minutes']:.1f}min | VIX: {data_freshness['vix_age_minutes']:.1f}min")
            
            # Add neural network status
            if self.model_initialized:
                buffer_status = f"{len(self.feature_buffer)}/{self.sequence_length}"
                learning_ready = "🧠 LEARNING" if len(self.feature_buffer) >= self.sequence_length else "⏳ BUILDING"
                logger.info(f"🤖 NEURAL NET: {learning_ready} | Buffer: {buffer_status} | Gaussian: {'✅' if self.gaussian_processor.is_fitted else '❌'}")
            else:
                logger.info(f"🤖 NEURAL NET: ⏳ INITIALIZING | Buffer: {len(self.feature_buffer)}/{self.sequence_length}")
            
            logger.info("🔍 " + "=" * 80)

        except Exception as e:
            logger.error(f"Error in cycle summary logging: {e}")

    def report_metrics_to_data_manager(self):
        """Report current performance metrics to Data Manager for leaderboard"""
        if not self.bot_reporter or not self.bot_reporter.enabled:
            return 0

        try:
            # Get account summary
            acct = self.paper_trader.get_account_summary()
            total_pnl = float(acct.get("total_pnl", 0.0))
            current_balance = float(acct.get("current_balance", acct.get("cash_balance", 0.0)))
            initial_balance = float(acct.get("initial_balance", 5000.0))

            # Calculate win rate from trade history
            completed_trades = self.paper_trader.get_recent_closed_trades(limit=100)
            total_trades = len(completed_trades)
            winning_trades = sum(1 for t in completed_trades if t.get('pnl', 0) > 0)
            win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

            # Calculate max drawdown (simplified)
            max_drawdown = float(acct.get("max_drawdown", 0.0))

            # Report metrics
            count = self.bot_reporter.report_metrics(
                total_pnl=total_pnl,
                win_rate=win_rate,
                total_trades=total_trades,
                current_balance=current_balance,
                max_drawdown=max_drawdown,
                pnl_pct=(total_pnl / initial_balance * 100) if initial_balance > 0 else 0.0,
            )

            if count > 0:
                logger.debug(f"📡 Reported {count} metrics to Data Manager")

            return count

        except Exception as e:
            logger.debug(f"📡 Failed to report metrics: {e}")
            return 0

    def graceful_shutdown(self):
        """Perform graceful shutdown with complete state save"""
        try:
            logger.info("🔄 Performing graceful shutdown...")

            # Report final metrics to Data Manager
            if self.bot_reporter and self.bot_reporter.enabled:
                self.report_metrics_to_data_manager()
                self.bot_reporter.close()
                logger.info("📡 Bot Reporter closed")

            # Save all model state
            if self.save_model():
                logger.info("✅ All model state saved successfully")
            else:
                logger.warning("⚠️ Some components may not have saved properly")

            # Log final statistics
            if len(self.feature_buffer) > 0:
                logger.info(f"📊 Final feature buffer size: {len(self.feature_buffer)}")
            if len(self.prediction_history) > 0:
                logger.info(f"📈 Total predictions made: {len(self.prediction_history)}")

            logger.info("👋 Shutdown complete - all state preserved for restart")

        except Exception as e:
            logger.error(f"Error during graceful shutdown: {e}")


# ------------------------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------------------------
def main():
    print("Starting UNIFIED OPTIONS TRADING BOT")
    print("Target: Features: Options Analysis + Bayesian NN + LSTM + Paper Trading")
    print(" Patched: Lazy NN init, interval-aware vol, safer RSI, better mapping/DB.")
    print()

    import os
    try:
        initial_balance = float(os.environ.get("BOT_INITIAL_BALANCE", "25000.0"))
    except Exception:
        initial_balance = 25000.0
    symbol = os.environ.get("BOT_SYMBOL", "BITX").strip() or "BITX"

    bot = UnifiedOptionsBot(initial_balance=initial_balance)

    # NOTE: We no longer pre-initialize the model here.
    # It will initialize automatically once the feature buffer reaches sequence_length.
    # This prevents the 'model never initializes' issue at startup.

    try:
        bot.run_continuous_trading(symbol)
    except KeyboardInterrupt:
        print("\n Bot stopped by user")
    except Exception as e:
        print(f"\nERROR: Bot error: {e}")
        logger.error(f"Bot error: {e}")


if __name__ == "__main__":
    main()
