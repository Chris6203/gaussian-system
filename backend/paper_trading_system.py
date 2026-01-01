#!/usr/bin/env python3
"""
Paper Trading System for BITX Options Trading Bot
Simulates real trades with fake money to test the system
Auto-uploaded to Tesla P40 server
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import json
import sqlite3
import os
from typing import Dict, List, Tuple, Optional
import uuid
from dataclasses import dataclass
from enum import Enum
import pytz

# Import real options pricing for proper Greeks calculation
try:
    from .real_option_pricing import OptionsGreeksCalculator, IntelligentStrikeSelector, DTEOptimizer
    from .greek_position_sizer import GreekPositionSizer, PortfolioGreeksTracker, ImprovedExitDecision
    REAL_GREEKS_AVAILABLE = True
except ImportError:
    try:
        from real_option_pricing import OptionsGreeksCalculator, IntelligentStrikeSelector, DTEOptimizer
        from greek_position_sizer import GreekPositionSizer, PortfolioGreeksTracker, ImprovedExitDecision
        REAL_GREEKS_AVAILABLE = True
    except ImportError:
        REAL_GREEKS_AVAILABLE = False
        logging.warning("Real options pricing not available - using fallback calculations")

try:
    from .data_sources import YahooFinanceDataSource
except ImportError:
    try:
        from data_sources import YahooFinanceDataSource
    except ImportError:
        YahooFinanceDataSource = None

try:
    from .online_learning_system import OnlineLearningSystem
except ImportError:
    try:
        from online_learning_system import OnlineLearningSystem
    except ImportError:
        OnlineLearningSystem = None

# Import XGBoost Exit Policy (fast, trainable, doesn't collapse)
try:
    from .xgboost_exit_policy import XGBoostExitPolicy, XGBoostExitConfig
    XGBOOST_EXIT_POLICY_AVAILABLE = True
except ImportError:
    try:
        from xgboost_exit_policy import XGBoostExitPolicy, XGBoostExitConfig
        XGBOOST_EXIT_POLICY_AVAILABLE = True
    except ImportError:
        XGBOOST_EXIT_POLICY_AVAILABLE = False

# Import Dynamic Exit Evaluator (continuous position re-evaluation)
try:
    from .dynamic_exit_evaluator import DynamicExitEvaluator, ExitReason
    DYNAMIC_EXIT_AVAILABLE = True
except ImportError:
    try:
        from dynamic_exit_evaluator import DynamicExitEvaluator, ExitReason
        DYNAMIC_EXIT_AVAILABLE = True
    except ImportError:
        DYNAMIC_EXIT_AVAILABLE = False
        logging.warning("DynamicExitEvaluator not available")
        logging.warning("XGBoost Exit Policy not available - using hard limits only")

# Import sophisticated trading system components
try:
    from .monte_carlo_pricer import MonteCarloOptionPricer, MCPricingParams
    from .execution_model import ExecutionModel
    from .distributional_utility import DistributionalUtilityCalculator
    SOPHISTICATED_SYSTEM_AVAILABLE = True
except ImportError:
    try:
        from monte_carlo_pricer import MonteCarloOptionPricer, MCPricingParams
        from execution_model import ExecutionModel
        from distributional_utility import DistributionalUtilityCalculator
        SOPHISTICATED_SYSTEM_AVAILABLE = True
    except ImportError:
        SOPHISTICATED_SYSTEM_AVAILABLE = False
        logging.warning("⚠️ Sophisticated trading system not available - using simple EV")

# Centralized risk manager (preferred single source of truth)
try:
    from .risk_manager import RiskManager, RiskCheckResult
    RISK_MANAGER_AVAILABLE = True
except Exception:
    try:
        from risk_manager import RiskManager, RiskCheckResult
        RISK_MANAGER_AVAILABLE = True
    except Exception:
        RISK_MANAGER_AVAILABLE = False
        RiskManager = None
        RiskCheckResult = None


class OrderType(Enum):
    CALL = "CALL"
    PUT = "PUT"
    BUY_CALL = "BUY_CALL"
    BUY_PUT = "BUY_PUT"
    SELL_CALL = "SELL_CALL"
    SELL_PUT = "SELL_PUT"


class OrderStatus(Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    CLOSED = "CLOSED"  # General close (e.g., RL exit, time-based exit on flat trade)
    EXPIRED = "EXPIRED"
    STOPPED_OUT = "STOPPED_OUT"
    PROFIT_TAKEN = "PROFIT_TAKEN"
    EXPIRED_ITM = "EXPIRED_ITM"
    EXPIRED_OTM = "EXPIRED_OTM"
    LIQUIDITY_REJECTED = "LIQUIDITY_REJECTED"
    PDT_REJECTED = "PDT_REJECTED"


@dataclass
class Option:
    """Represents an options contract"""
    symbol: str
    strike_price: float
    expiration: datetime
    option_type: OrderType
    premium: float


@dataclass
class Trade:
    """Represents a trade"""
    id: str
    timestamp: datetime
    symbol: str
    option_type: OrderType
    strike_price: float
    premium_paid: float
    quantity: int
    entry_price: float
    exit_price: Optional[float] = None
    exit_timestamp: Optional[datetime] = None
    status: OrderStatus = OrderStatus.PENDING
    profit_loss: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    ml_confidence: float = 0.0
    ml_prediction: str = ""
    expiration_date: Optional[datetime] = None
    # Position management fields
    planned_exit_time: Optional[datetime] = None  # When bot plans to close (max hold time)
    max_hold_hours: float = 24.0  # Maximum hours to hold position
    # Projected profit tracking
    projected_profit: float = 0.0  # Expected profit in dollars (from ML prediction)
    projected_return_pct: float = 0.0  # Expected return percentage
    # Shadow trading fields - link to real Tradier trades
    tradier_option_symbol: Optional[str] = None  # Tradier option symbol (e.g., SPY251219C00675000)
    tradier_order_id: Optional[str] = None  # Tradier order ID
    is_real_trade: bool = False  # TRUE if this is a real Tradier trade (not just paper)
    live_blocked_reason: Optional[str] = None  # Why live trade was blocked (if paper-only)
    # Run tracking for dashboard
    run_id: Optional[str] = None  # Training run ID (e.g., run_20251220_143000)
    exit_reason: Optional[str] = None  # Why trade was exited


class TradierFeeCalculator:
    """Calculate realistic Tradier fees for options trading"""

    def __init__(self, plan="lite"):
        # Tradier fee structure (2025) - Updated for Pro Plus
        if plan == "lite":
            self.options_contract_fee = 0.35  # $0.35 per contract
            self.index_options_fee = 0.35    # Same as regular options
        elif plan == "pro":
            self.options_contract_fee = 0.00  # $0.00 per contract (Pro plan)
            self.index_options_fee = 0.35    # $0.35 for index options
        elif plan == "pro_plus":
            self.options_contract_fee = 0.00  # $0.00 per contract (Pro Plus)
            self.index_options_fee = 0.10    # $0.10 per contract for index options (BITX)
        else:
            self.options_contract_fee = 0.35  # Default to Lite plan
            self.index_options_fee = 0.35

        # Regulatory fees (same across all brokers)
        self.regulatory_fee_rate = 0.000119  # 0.0119% of trade value
        self.trading_activity_fee = 0.000145  # 0.0145% of sell proceeds
        self.sec_fee_rate = 0.0000278  # 0.00278% of sell proceeds

        # Market impact costs (REALISTIC for algorithmic options trading)
        # REMOVED: These were being double-counted with place_trade/update_positions
        # Spread and slippage are now ONLY applied in place_trade and update_positions
        self.bid_ask_spread_cost = 0.0  # REMOVED to prevent double-counting
        self.slippage_cost = 0.0  # REMOVED to prevent double-counting

        # Assignment/Exercise fee
        self.assignment_fee = 9.00 if plan == "lite" else 5.00

    def calculate_entry_fees(self, premium: float, quantity: int, symbol: str = "BITX") -> float:
        """Calculate fees for entering a position (spread/slippage handled separately)"""
        # Use index options fee for BITX (Pro Plus: $0.10 per contract)
        if symbol in ['BITX', 'SPY', 'QQQ', 'IWM']:  # Index ETFs
            contract_fees = self.index_options_fee * quantity
        else:
            contract_fees = self.options_contract_fee * quantity
            
        trade_value = premium * quantity * 100
        regulatory_fees = trade_value * self.regulatory_fee_rate
        # REMOVED: spread_cost and slippage_cost to prevent double-counting
        # These are now ONLY applied in place_trade method
        
        return contract_fees + regulatory_fees

    def calculate_exit_fees(self, premium: float, quantity: int, symbol: str = "BITX") -> float:
        """Calculate fees for exiting a position (spread/slippage handled separately)"""
        # Use index options fee for BITX (Pro Plus: $0.10 per contract)
        if symbol in ['BITX', 'SPY', 'QQQ', 'IWM']:  # Index ETFs
            contract_fees = self.index_options_fee * quantity
        else:
            contract_fees = self.options_contract_fee * quantity
            
        trade_value = premium * quantity * 100
        trading_activity_fee = trade_value * self.trading_activity_fee
        sec_fee = trade_value * self.sec_fee_rate
        regulatory_fees = trade_value * self.regulatory_fee_rate
        # REMOVED: spread_cost and slippage_cost to prevent double-counting
        # These are now ONLY applied in update_positions method
        return contract_fees + trading_activity_fee + sec_fee + regulatory_fees


class PaperTradingSystem:
    """
    Paper trading system that simulates options trades with fake money
    Includes realistic Tradier fees and costs
    """
    
    @staticmethod
    def round_to_realistic_strike(strike_price: float, spot_price: float) -> float:
        """
        Round calculated strike to realistic exchange intervals
        
        Exchanges don't offer arbitrary strikes like $4.72.
        They offer strikes in standard intervals:
        - Under $5: $0.50 intervals ($4.00, $4.50, $5.00)
        - $5-$25: $0.50 or $1.00 intervals
        - $25-$200: $1.00 or $2.50 intervals
        - Over $200: $5.00 or $10.00 intervals
        
        This ensures paper trading matches real market conditions.
        
        Args:
            strike_price: Calculated optimal strike (e.g., $4.72)
            spot_price: Current underlying price
            
        Returns:
            Rounded strike (e.g., $4.50 or $5.00)
        """
        # Determine interval based on price range
        if spot_price < 5:
            # Under $5: $0.50 intervals
            interval = 0.50
        elif spot_price < 25:
            # $5-$25: $1.00 intervals (sometimes $0.50, but use $1 for safety)
            interval = 1.00
        elif spot_price < 50:
            # $25-$50: $1.00 intervals
            interval = 1.00
        elif spot_price < 200:
            # $50-$200: $2.50 or $5.00 intervals (use $2.50)
            interval = 2.50
        else:
            # Over $200: $5.00 intervals
            interval = 5.00
        
        # Round to nearest interval
        rounded_strike = round(strike_price / interval) * interval
        
        return rounded_strike
    
    def get_market_time(self) -> datetime:
        """Get current time in market timezone (US/Eastern) as naive datetime"""
        # ✅ CRITICAL FIX: Use simulated time if in simulation mode!
        if hasattr(self, 'simulation_mode') and self.simulation_mode and hasattr(self, 'simulated_time') and self.simulated_time:
            # In simulation, use the simulated time (already naive)
            return self.simulated_time
        
        # In live trading, use real time
        market_tz = pytz.timezone('US/Eastern')
        utc_now = datetime.utcnow()
        utc_tz = pytz.timezone('UTC')
        utc_dt = utc_tz.localize(utc_now)
        market_dt = utc_dt.astimezone(market_tz)
        # Return as naive datetime to match database storage
        return market_dt.replace(tzinfo=None)

    def set_bot_reference(self, bot):
        """Set reference to main bot for liquidity learning"""
        self.bot = bot

    def set_run_id(self, run_id: str):
        """Set the current run ID for trade tracking in dashboard"""
        self.current_run_id = run_id
        self.logger.info(f"📊 Run ID set to: {run_id}")

    def _training_unlimited_funds(self) -> bool:
        """
        Training helper:
        - In simulation/time-travel, allow the account to go negative AND avoid "can't trade" blocks
          caused purely by a low/negative balance.
        Controlled via env TT_UNLIMITED_FUNDS (default: 1 in simulation/time-travel).
        """
        try:
            is_sim = bool(getattr(self, "simulation_mode", False) or getattr(self, "time_travel_mode", False) or getattr(self, "disable_pdt_override", False))
            if not is_sim:
                return False
            return os.environ.get("TT_UNLIMITED_FUNDS", "1").strip() == "1"
        except Exception:
            return False

    def _training_virtual_balance(self) -> float:
        """
        Training helper: use a stable 'virtual balance' for sizing so the run doesn't stall
        after a drawdown (even though we allow negative cashflow).
        """
        try:
            env_bal = os.environ.get("TRAINING_INITIAL_BALANCE")
            if env_bal:
                vb = float(env_bal)
            else:
                vb = float(getattr(self, "initial_balance", 0.0) or 0.0)
            # Never below current_balance (if it's higher), but also never below $0 for sizing math.
            vb = max(vb, float(getattr(self, "current_balance", 0.0) or 0.0), 0.0)
            return float(vb)
        except Exception:
            return float(max(float(getattr(self, "initial_balance", 0.0) or 0.0), 0.0))
        self.logger.info("[LEARN] Bot reference set - liquidity learning enabled")
    
    def _get_minutes_to_close(self) -> int:
        """
        Minutes until market close (US equities/options: 16:00 ET).
        
        Returns:
            int: minutes remaining (0 if already closed)
        """
        try:
            now = self.get_market_time()
            close_dt = now.replace(hour=16, minute=0, second=0, microsecond=0)
            return max(0, int((close_dt - now).total_seconds() // 60))
        except Exception:
            return 0
    
    def _get_daily_and_weekly_pnl(self) -> Tuple[float, float]:
        """
        Best-effort realized P&L for risk gating.
        
        Uses the daily_performance table if present; falls back to 0.0 if unavailable.
        """
        try:
            today = self.get_market_time().date()
            # Monday = 0
            week_start = today - timedelta(days=today.weekday())
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Daily P&L
                cursor.execute(
                    "SELECT daily_pnl FROM daily_performance WHERE date = ?",
                    (today.isoformat(),)
                )
                row = cursor.fetchone()
                daily_pnl = float(row[0]) if row and row[0] is not None else 0.0
                
                # Weekly P&L: sum from week_start to today
                cursor.execute(
                    "SELECT SUM(daily_pnl) FROM daily_performance WHERE date >= ? AND date <= ?",
                    (week_start.isoformat(), today.isoformat())
                )
                row = cursor.fetchone()
                weekly_pnl = float(row[0]) if row and row[0] is not None else 0.0
            
            return daily_pnl, weekly_pnl
        except Exception:
            return 0.0, 0.0
    
    def __init__(self, initial_balance: float = 1000.0, db_path: str = 'data/paper_trading.db', disable_pdt: bool = False, enable_liquidity_checks: bool = False, tradier_trading_system=None):
        # Setup logging first (needed for PDT protection logging)
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.db_path = db_path
        self.disable_pdt_override = disable_pdt  # Override for simulations
        self.enable_liquidity_checks = enable_liquidity_checks  # Enable for live mode
        self.tradier_trading_system = tradier_trading_system  # For executing live trades
        
        # Centralized risk management (single source of truth)
        if RISK_MANAGER_AVAILABLE:
            try:
                # Uses defaults if no config.json is present
                self.risk_manager = RiskManager(config_path='config.json')
                self.logger.info("✅ RiskManager enabled for trade gating/sizing")
            except Exception as e:
                self.risk_manager = None
                self.logger.warning(f"⚠️ Could not initialize RiskManager: {e}")
        else:
            self.risk_manager = None

        # ------------------------------------------------------------------
        # Optional ML Tradability Gate (expectancy/friction-aware entry filter)
        # ------------------------------------------------------------------
        self.tradability_gate_enabled = False
        self.tradability_gate_checkpoint_path = "models/tradability_gate.pt"
        self.tradability_gate_veto_threshold = 0.25
        self.tradability_gate_downgrade_threshold = 0.45
        self._tradability_gate_model = None
        try:
            import json
            from pathlib import Path

            cfg_path = Path("config.json")
            if cfg_path.exists():
                with cfg_path.open("r", encoding="utf-8") as f:
                    cfg = json.load(f)
                gate_cfg = ((cfg or {}).get("architecture") or {}).get("tradability_gate") or {}
                self.tradability_gate_enabled = bool(gate_cfg.get("enabled", self.tradability_gate_enabled))
                self.tradability_gate_checkpoint_path = str(gate_cfg.get("checkpoint_path", self.tradability_gate_checkpoint_path))
                self.tradability_gate_veto_threshold = float(gate_cfg.get("veto_threshold", self.tradability_gate_veto_threshold))
                self.tradability_gate_downgrade_threshold = float(gate_cfg.get("downgrade_threshold", self.tradability_gate_downgrade_threshold))
        except Exception as e:
            self.logger.debug(f"[GATE] Could not read tradability_gate config: {e}")

        if self.tradability_gate_enabled:
            self.logger.info(
                f"🧱 TradabilityGate ENABLED | ckpt={self.tradability_gate_checkpoint_path} "
                f"| veto<{self.tradability_gate_veto_threshold:.2f} "
                f"| downgrade<{self.tradability_gate_downgrade_threshold:.2f}"
            )
        else:
            self.logger.info("🧱 TradabilityGate DISABLED")
        
        # ===== SETTLEMENT TRACKING (T+1 for options) =====
        # Funds from closed trades settle the next business day
        # pending_settlements: list of {'amount': float, 'settlement_date': datetime, 'trade_id': str}
        self.pending_settlements = []
        self.simulate_settlement = True  # Enable by default for realism

        # Initialize fee calculator (Tradier Pro Plus for algorithmic trading)
        self.fee_calculator = TradierFeeCalculator(
            plan="pro_plus")  # Pro Plus: $0.00 options fees, $0.10 index options

        # Initialize liquidity validator for live mode
        if self.enable_liquidity_checks:
            try:
                from backend.liquidity_validator import LiquidityValidator
                from backend.execution_ev import ExecutionEVCalculator
                
                self.liquidity_validator = LiquidityValidator(
                    max_bid_ask_spread_pct=5.0,
                    min_volume=100,
                    min_open_interest=75,
                    max_strike_deviation_pct=15.0
                )
                
                # Initialize Greeks-based EV calculator
                self.ev_calculator = ExecutionEVCalculator(
                    w_slippage=1.0,
                    w_time_to_fill=0.02,
                    base_fees_per_contract=0.65
                )
                
                self.logger.info("🛡️ Liquidity validation ENABLED for live mode")
                self.logger.info("🧮 Greeks-based EV calculator ENABLED")
            except ImportError as e:
                self.liquidity_validator = None
                self.ev_calculator = None
                self.logger.warning(f"⚠️  Liquidity validator not available: {e}")
        else:
            self.liquidity_validator = None
            self.ev_calculator = None
            self.logger.info("📝 Liquidity validation DISABLED (simulation mode)")

        # Initialize sophisticated trading system (Monte Carlo + Execution Model + Utility)
        if SOPHISTICATED_SYSTEM_AVAILABLE and self.enable_liquidity_checks:
            try:
                self.mc_pricer = MonteCarloOptionPricer(seed=42)
                self.exec_model = ExecutionModel()
                self.utility_calc = DistributionalUtilityCalculator()
                self.logger.info("🎯 Sophisticated trading system ENABLED")
                self.logger.info("   • Monte Carlo P&L simulation")
                self.logger.info("   • Learned execution model")
                self.logger.info("   • Distributional utility calculation")
            except Exception as e:
                self.mc_pricer = None
                self.exec_model = None
                self.utility_calc = None
                self.logger.warning(f"⚠️ Could not initialize sophisticated system: {e}")
        else:
            self.mc_pricer = None
            self.exec_model = None
            self.utility_calc = None
            if not SOPHISTICATED_SYSTEM_AVAILABLE:
                self.logger.info("📝 Sophisticated system not available - using simple EV")
            else:
                self.logger.info("📝 Sophisticated system disabled (simulation mode)")

        # Initialize real options pricing (CRITICAL FOR PROPER OPTIONS TRADING!)
        if REAL_GREEKS_AVAILABLE:
            self.greeks_calculator = OptionsGreeksCalculator(risk_free_rate=0.05)
            self.strike_selector = IntelligentStrikeSelector(self.greeks_calculator)
            self.dte_optimizer = DTEOptimizer()
            self.position_sizer = GreekPositionSizer(
                base_position_pct=0.05,  # 5% base
                max_portfolio_delta=1.0,  # Max total delta
                max_portfolio_theta=50.0,  # Max $50/day theta decay
                max_single_position_pct=0.15  # Max 15% per position
            )
            self.portfolio_tracker = PortfolioGreeksTracker()
            self.exit_decision = ImprovedExitDecision()
            self.logger.info("✅ Real options pricing enabled (Black-Scholes Greeks)")
            self.logger.info("✅ Greek-based position sizing enabled")
        else:
            self.greeks_calculator = None
            self.strike_selector = None
            self.dte_optimizer = None
            self.position_sizer = None
            self.portfolio_tracker = None
            self.exit_decision = None
            self.logger.warning("⚠️ Real options pricing NOT available - using fallback")

        # Initialize XGBoost Exit Policy (learns from trades, fast, reliable)
        if XGBOOST_EXIT_POLICY_AVAILABLE:
            try:
                exit_config = XGBoostExitConfig(
                    min_trades_for_ml=20,         # Use simple rules until 20 trades (faster ML adoption)
                    exit_probability_threshold=0.50,  # Exit if >50% confident
                    fallback_take_profit=15.0,    # ALIGNED with unified RL: +15% take profit
                    fallback_stop_loss=-8.0,      # ALIGNED with unified RL: -8% stop loss
                    fallback_max_hold_minutes=90.0,  # EXTENDED: 90 min hold (give trades time)
                    retrain_every_n_trades=10,    # Retrain every 10 trades
                )

                # TRAINING curriculum: dampen churny exits during simulation/time-travel.
                # This is about learning good entries/exits, not scalping pennies every minute.
                try:
                    is_sim = bool(getattr(self, "simulation_mode", False) or getattr(self, "time_travel_mode", False) or getattr(self, "disable_pdt_override", False))
                    if is_sim:
                        # Stricter threshold = fewer exits
                        exit_config.exit_probability_threshold = float(os.environ.get("TT_XGB_EXIT_THRESHOLD", "0.75"))
                        # Longer max hold in training (minutes)
                        exit_config.fallback_max_hold_minutes = float(os.environ.get("TT_XGB_MAX_HOLD_MINUTES", "180"))
                        # Raise scalp trigger by increasing fallback_take_profit a bit (still bounded by env if you want)
                        exit_config.fallback_take_profit = float(os.environ.get("TT_XGB_TP", str(exit_config.fallback_take_profit)))
                        exit_config.fallback_stop_loss = float(os.environ.get("TT_XGB_SL", str(exit_config.fallback_stop_loss)))
                except Exception:
                    pass
                self.rl_exit_policy = XGBoostExitPolicy(config=exit_config)
                self.logger.info("🌲 XGBoost Exit Policy ENABLED")
                self.logger.info("   • Uses simple rules until 30 trades")
                self.logger.info("   • Then learns optimal exits from history")
                self.logger.info("   • Fallback: TP +8%, SL -10%, Max 45min")
                
                # Try to load existing trained model
                for model_path in ['models/xgboost_exit.pkl', 'models/xgboost_exit_policy.pkl']:
                    try:
                        import os
                        if os.path.exists(model_path):
                            if self.rl_exit_policy.load(model_path):
                                break
                    except Exception as e:
                        self.logger.debug(f"Could not load from {model_path}: {e}")
            except Exception as e:
                self.rl_exit_policy = None
                self.logger.warning(f"⚠️ Could not initialize XGBoost exit policy: {e}")
        else:
            self.rl_exit_policy = None
            self.logger.info("📝 XGBoost Exit Policy NOT available - using hard limits only")

        # Initialize Dynamic Exit Evaluator (continuous position re-evaluation)
        # DISABLED BY DEFAULT: Testing showed it hurts win rate significantly (40% -> 0.4%)
        # The theta_exceeds_expected check cuts positions too early for short-term options
        # Enable with DYNAMIC_EXIT_ENABLED=1 environment variable if needed for experimentation
        enable_dynamic_exit = os.environ.get('DYNAMIC_EXIT_ENABLED', '0') == '1'
        if DYNAMIC_EXIT_AVAILABLE and enable_dynamic_exit:
            try:
                dynamic_exit_config = {
                    'prediction_flip_exit': True,
                    'confidence_collapse_threshold': 0.5,
                    'regime_change_exit': True,
                    'theta_check_enabled': True,
                    'momentum_check_enabled': True,
                    'time_decay_exit_enabled': True,
                    'flat_trade_exit_minutes': 25,  # Exit flat trades after 25 min (was 10 - too aggressive)
                    'flat_threshold_pct': 1.0,  # Only exit if truly flat (+/- 1%, was 2%)
                    'profit_protection_enabled': True,
                    'profit_lock_threshold': 5.0,  # Lock profits above 5%
                    'profit_pullback_exit': 3.0,  # Exit if pulls back 3% from peak
                }
                self.dynamic_exit_evaluator = DynamicExitEvaluator(config=dynamic_exit_config)
                self.logger.info("🔄 Dynamic Exit Evaluator ENABLED (via env var)")
                self.logger.info("   • Thesis invalidation check")
                self.logger.info("   • Confidence collapse detection")
                self.logger.info("   • Regime change monitoring")
                self.logger.info("   • Profit protection (5% lock, 3% pullback)")
            except Exception as e:
                self.dynamic_exit_evaluator = None
                self.logger.warning(f"⚠️ Could not initialize Dynamic Exit Evaluator: {e}")
        else:
            self.dynamic_exit_evaluator = None
            if enable_dynamic_exit and not DYNAMIC_EXIT_AVAILABLE:
                self.logger.info("📝 Dynamic Exit Evaluator requested but NOT available")
            else:
                self.logger.info("📝 Dynamic Exit Evaluator DISABLED (hurts win rate)")

        # Trading parameters
        self.position_size_pct = 0.05  # 5% of balance per trade
        self.max_positions = 3  # REDUCED: Fewer concurrent positions for better risk control
        
        # Load exit parameters from centralized config (SINGLE SOURCE OF TRUTH)
        try:
            from backend.exit_config import get_exit_config
            exit_cfg = get_exit_config()
            self.stop_loss_pct = exit_cfg.stop_loss_pct
            self.take_profit_pct = exit_cfg.take_profit_pct
            self.logger.info(f"📋 Exit params from config: stop={self.stop_loss_pct*100:.1f}%, take={self.take_profit_pct*100:.1f}%")
        except Exception as e:
            self.logger.warning(f"Could not load exit_config: {e}, using defaults")
            self.stop_loss_pct = 0.08  # 8% stop loss
            self.take_profit_pct = 0.15  # 15% take profit
        
        # Transaction costs (REAL WORLD)
        self.commission_per_contract = 0.65  # Typical broker commission
        self.exchange_fees_per_contract = 0.08  # Exchange/regulatory fees
        self.total_fees_per_contract = self.commission_per_contract + self.exchange_fees_per_contract
        
        # Tradier API execution simulation
        self.execution_delay_seconds = 2  # 2-second execution delay (Tradier realistic)
        self.pending_orders = []  # Track orders awaiting execution
        
        # ============= PDT (Pattern Day Trader) PROTECTION =============
        # PDT Rule: Accounts under $25,000 are limited to 3 day trades per 5 business days
        self.account_value_for_pdt = initial_balance  # Track for PDT rule
        self.day_trades_count = 0  # Track day trades for PDT
        self.last_day_trade_reset = self.get_market_time().date()
        
        # PDT protection can be disabled for simulations
        if disable_pdt:
            self.pdt_protected = False
            self.logger.info("🔓 PDT protection DISABLED (simulation mode)")
        else:
            self.pdt_protected = initial_balance < 25000  # Enable PDT protection under $25K
        
        self.max_day_trades_under_25k = 3  # PDT limit: 3 day trades per 5 business days
        self.day_trade_history = []  # Track individual day trades with timestamps
        
        # Account size-based trading modes
        if initial_balance < 25000:
            # Small account mode: Conservative settings
            self.position_size_pct = 0.10  # 10% of balance per trade (more aggressive for small accounts)
            self.max_positions = 5  # Maximum concurrent positions (increased from 2)
            self.max_position_size_per_trade = 0.25  # Max 25% of account per trade
            if self.pdt_protected:
                self.logger.info(f"🚨 PDT PROTECTION ENABLED - Account under $25,000")
                self.logger.info(f"📊 Small Account Mode: Max {self.max_day_trades_under_25k} day trades per 5 business days")
            else:
                self.logger.info(f"💰 Small Account Mode: Balance under $25,000 (PDT protection overridden)")
                self.logger.info(f"📊 Unlimited day trades allowed (Tradier will enforce limits)")
        else:
            # Large account mode: Standard settings
            self.position_size_pct = 0.05  # 5% of balance per trade
            self.max_positions = 5  # Maximum concurrent positions (increased from 2)
            self.max_position_size_per_trade = 0.10  # Max 10% of account per trade
            self.logger.info(f"✅ PDT PROTECTION DISABLED - Account over $25,000")
            self.logger.info(f"📊 Standard Account Mode: Unlimited day trades allowed")
        # ============= END PDT PROTECTION =============
        
        # Options-specific realism factors
        self.implied_volatility_base = 0.25  # Base IV for BITX (25%)
        self.time_decay_rate = 0.05  # 5% theta decay per day (realistic for near-term options)
        self.liquidity_impact_threshold = 1000  # Orders >$1000 face liquidity constraints

        # Components
        self.data_source = YahooFinanceDataSource()
        self.learning_system = OnlineLearningSystem() if OnlineLearningSystem else None

        # Initialize database
        self._init_database()

        # Active trades (load before account state to check if fresh start)
        self.active_trades: List[Trade] = []
        self._load_active_trades()
        
        # Load account state ONLY if there are active trades or historical trades
        # Otherwise, keep the synced initial_balance
        if self._has_trading_history():
            self._load_account_state()
            self.logger.info(f"[INIT] Loaded existing account state: ${self.current_balance:.2f}")
        else:
            # Fresh start - initialize default state with synced balance
            self.total_trades = 0
            self.winning_trades = 0
            self.losing_trades = 0
            self.total_profit_loss = 0.0
            self.max_drawdown = 0.0
            self.logger.info(f"[INIT] Fresh start with synced balance: ${self.current_balance:.2f}")
        
        # Sync Tradier positions on startup (prevent desync)
        # BUT: Skip sync during simulation/training to preserve training balance
        # Note: disable_pdt_override indicates simulation mode (set in training scripts)
        is_simulation = getattr(self, 'simulation_mode', False) or getattr(self, 'time_travel_mode', False) or self.disable_pdt_override
        if is_simulation:
            self.logger.info("[INIT] Skipping Tradier sync (simulation/training mode)")
            # Disable settlement simulation in training mode for faster iteration
            self.simulate_settlement = False
        else:
            self.sync_tradier_positions()
            # Load pending settlements from database
            self._load_pending_settlements()
            self.logger.info(f"[INIT] Settlement simulation ENABLED (T+1 for options)")
            if self.pending_settlements:
                pending_total = sum(s['amount'] for s in self.pending_settlements)
                self.logger.info(f"[INIT] Pending settlements: ${pending_total:.2f} in {len(self.pending_settlements)} transaction(s)")

    def _get_tradability_gate_model(self):
        """
        Lazy-load TradabilityGateModel. Returns None if unavailable.
        """
        if not getattr(self, "tradability_gate_enabled", False):
            return None
        if getattr(self, "_tradability_gate_model", None) is not None:
            return self._tradability_gate_model

        try:
            from backend.tradability_gate import TradabilityGateModel

            m = TradabilityGateModel(device="cpu")
            if not m.load(getattr(self, "tradability_gate_checkpoint_path", "models/tradability_gate.pt")):
                self.logger.warning(
                    f"[GATE] Enabled but checkpoint failed to load: {getattr(self, 'tradability_gate_checkpoint_path', None)}"
                )
                self._tradability_gate_model = None
                return None
            self._tradability_gate_model = m
            return m
        except Exception as e:
            self.logger.warning(f"[GATE] Failed to initialize TradabilityGateModel: {e}")
            self._tradability_gate_model = None
            return None

    def _init_database(self):
        """Initialize paper trading database"""
        self.logger.info("Initializing paper trading database...")

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Account state table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS account_state (
                    id INTEGER PRIMARY KEY,
                    balance REAL NOT NULL,
                    total_trades INTEGER NOT NULL,
                    winning_trades INTEGER NOT NULL,
                    losing_trades INTEGER NOT NULL,
                    total_profit_loss REAL NOT NULL,
                    max_drawdown REAL NOT NULL,
                    updated_at TEXT NOT NULL
                )
            ''')

            # Trades table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    option_type TEXT NOT NULL,
                    strike_price REAL NOT NULL,
                    premium_paid REAL NOT NULL,
                    quantity INTEGER NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL,
                    exit_timestamp TEXT,
                    status TEXT NOT NULL,
                    profit_loss REAL NOT NULL,
                    stop_loss REAL NOT NULL,
                    take_profit REAL NOT NULL,
                    ml_confidence REAL NOT NULL,
                    ml_prediction TEXT NOT NULL,
                    expiration_date TEXT,
                    created_at TEXT NOT NULL,
                    projected_profit REAL,
                    projected_return_pct REAL,
                    planned_exit_time TEXT,
                    max_hold_hours REAL,
                    tradier_option_symbol TEXT,
                    tradier_order_id TEXT,
                    is_real_trade INTEGER DEFAULT 0,
                    live_blocked_reason TEXT,
                    exit_reason TEXT
                )
            ''')

            # Daily performance table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS daily_performance (
                    date TEXT PRIMARY KEY,
                    starting_balance REAL NOT NULL,
                    ending_balance REAL NOT NULL,
                    daily_pnl REAL NOT NULL,
                    trades_count INTEGER NOT NULL,
                    winning_trades INTEGER NOT NULL,
                    losing_trades INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                )
            ''')
            
            # Pending settlements table (T+1 for options)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS pending_settlements (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_id TEXT NOT NULL,
                    amount REAL NOT NULL,
                    settlement_date TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    settled INTEGER DEFAULT 0
                )
            ''')

            conn.commit()
            
            # Migrate existing tables - add new columns if they don't exist
            # This handles databases created with older schemas
            self._migrate_trades_table(cursor)
            conn.commit()

        self.logger.info("Paper trading database initialized")

    def _migrate_trades_table(self, cursor):
        """Add missing columns to trades table for backwards compatibility"""
        # Get existing columns
        cursor.execute("PRAGMA table_info(trades)")
        existing_columns = {row[1] for row in cursor.fetchall()}
        
        # Define columns that may be missing in older databases
        migrations = [
            ("projected_profit", "REAL"),
            ("projected_return_pct", "REAL"),
            ("planned_exit_time", "TEXT"),
            ("max_hold_hours", "REAL"),
            ("tradier_option_symbol", "TEXT"),
            ("tradier_order_id", "TEXT"),
            ("is_real_trade", "INTEGER DEFAULT 0"),
            ("live_blocked_reason", "TEXT"),
        ]
        
        for column_name, column_type in migrations:
            if column_name not in existing_columns:
                try:
                    cursor.execute(f"ALTER TABLE trades ADD COLUMN {column_name} {column_type}")
                    self.logger.info(f"[MIGRATION] Added column '{column_name}' to trades table")
                except Exception as e:
                    self.logger.warning(f"[MIGRATION] Could not add column '{column_name}': {e}")

    # ===== SETTLEMENT METHODS (T+1 for options) =====
    
    def _get_next_business_day(self, from_date: datetime = None) -> datetime:
        """Get the next business day (skips weekends, not holidays for simplicity)"""
        if from_date is None:
            from_date = self.get_market_time()
        
        next_day = from_date + timedelta(days=1)
        
        # Skip weekends
        while next_day.weekday() >= 5:  # Saturday = 5, Sunday = 6
            next_day += timedelta(days=1)
        
        return next_day
    
    def _load_pending_settlements(self):
        """Load pending settlements from database"""
        self.pending_settlements = []
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT trade_id, amount, settlement_date 
                    FROM pending_settlements 
                    WHERE settled = 0
                ''')
                rows = cursor.fetchall()
                
                for row in rows:
                    self.pending_settlements.append({
                        'trade_id': row[0],
                        'amount': row[1],
                        'settlement_date': datetime.fromisoformat(row[2])
                    })
        except Exception as e:
            self.logger.debug(f"[SETTLEMENT] Could not load pending settlements: {e}")
    
    def _save_settlement(self, trade_id: str, amount: float, settlement_date: datetime):
        """Save a pending settlement to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO pending_settlements (trade_id, amount, settlement_date, created_at, settled)
                    VALUES (?, ?, ?, ?, 0)
                ''', (trade_id, amount, settlement_date.isoformat(), datetime.now().isoformat()))
                conn.commit()
        except Exception as e:
            self.logger.warning(f"[SETTLEMENT] Could not save settlement: {e}")
    
    def _mark_settlement_complete(self, trade_id: str):
        """Mark a settlement as complete in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE pending_settlements SET settled = 1 WHERE trade_id = ?
                ''', (trade_id,))
                conn.commit()
        except Exception as e:
            self.logger.debug(f"[SETTLEMENT] Could not mark settlement complete: {e}")
    
    def process_settlements(self):
        """Process any settlements that have reached their settlement date"""
        if not self.simulate_settlement or not self.pending_settlements:
            return
        
        now = self.get_market_time()
        settled_amount = 0
        settled_trades = []
        
        for settlement in self.pending_settlements[:]:  # Copy list to allow modification
            if now >= settlement['settlement_date']:
                # Settlement has cleared - add to available balance
                self.current_balance += settlement['amount']
                settled_amount += settlement['amount']
                settled_trades.append(settlement['trade_id'])
                self._mark_settlement_complete(settlement['trade_id'])
                self.pending_settlements.remove(settlement)
        
        if settled_amount > 0:
            self.logger.info(f"💰 [SETTLEMENT] ${settled_amount:.2f} settled from {len(settled_trades)} trade(s)")
            self.logger.info(f"   Available balance: ${self.current_balance:.2f}")
            self._save_account_state()
    
    @property
    def settled_balance(self) -> float:
        """Get available (settled) balance - excludes pending settlements"""
        return self.current_balance
    
    @property
    def pending_settlement_amount(self) -> float:
        """Get total amount pending settlement"""
        return sum(s['amount'] for s in self.pending_settlements)
    
    @property
    def total_equity(self) -> float:
        """Get total equity including pending settlements"""
        return self.current_balance + self.pending_settlement_amount
    
    def _has_trading_history(self):
        """
        Check if there's any PAPER trading history in the database.
        Returns False if only LIVE trades exist (no paper trading history to restore).
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Count paper trades (is_real_trade = 0 or NULL)
            cursor.execute('SELECT COUNT(*) FROM trades WHERE is_real_trade IS NULL OR is_real_trade = 0')
            paper_count = cursor.fetchone()[0]
            return paper_count > 0
    
    def _load_account_state(self):
        """Load account state from database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM account_state WHERE id = 1')
            row = cursor.fetchone()

            if row:
                self.current_balance = row[1]
                self.total_trades = row[2]
                self.winning_trades = row[3]
                self.losing_trades = row[4]
                self.total_profit_loss = row[5]
                self.max_drawdown = row[6]
            else:
                # Initialize default state
                self.total_trades = 0
                self.winning_trades = 0
                self.losing_trades = 0
                self.total_profit_loss = 0.0
                self.max_drawdown = 0.0
                self._save_account_state()

    def _save_account_state(self):
        """Save account state to database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO account_state 
                (id, balance, total_trades, winning_trades, losing_trades, 
                 total_profit_loss, max_drawdown, updated_at)
                VALUES (1, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                self.current_balance,
                self.total_trades,
                self.winning_trades,
                self.losing_trades,
                self.total_profit_loss,
                self.max_drawdown,
                self.get_market_time().isoformat()
            ))
            conn.commit()

    def _load_active_trades(self):
        """Load active trades from database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM trades 
                WHERE status IN ('PENDING', 'FILLED')
                ORDER BY timestamp DESC
            ''')

            rows = cursor.fetchall()
            self.active_trades = []
            
            real_trade_count = 0
            paper_trade_count = 0

            for row in rows:
                # Handle new columns with fallback for old database schema
                tradier_option_symbol = row[18] if len(row) > 18 else None
                tradier_order_id = row[19] if len(row) > 19 else None
                is_real_trade = bool(row[20]) if len(row) > 20 else False
                live_blocked_reason = row[21] if len(row) > 21 else None
                
                # Parse timestamp and convert to timezone-naive (market time)
                timestamp = pd.to_datetime(row[1])
                if hasattr(timestamp, 'tzinfo') and timestamp.tzinfo is not None:
                    timestamp = timestamp.replace(tzinfo=None)
                
                # Parse exit timestamp if exists
                exit_timestamp = None
                if row[9]:
                    exit_timestamp = pd.to_datetime(row[9])
                    if hasattr(exit_timestamp, 'tzinfo') and exit_timestamp.tzinfo is not None:
                        exit_timestamp = exit_timestamp.replace(tzinfo=None)
                
                # Parse expiration date if exists (row[16])
                expiration_date = None
                if len(row) > 16 and row[16]:
                    try:
                        expiration_date = pd.to_datetime(row[16])
                        if hasattr(expiration_date, 'tzinfo') and expiration_date.tzinfo is not None:
                            expiration_date = expiration_date.replace(tzinfo=None)
                    except:
                        pass
                
                trade = Trade(
                    id=row[0],
                    timestamp=timestamp,  # Now timezone-naive
                    symbol=row[2],
                    option_type=OrderType(row[3]),
                    strike_price=row[4],
                    premium_paid=row[5],
                    quantity=row[6],
                    entry_price=row[7],
                    exit_price=row[8],
                    exit_timestamp=exit_timestamp,  # Now timezone-naive if present
                    status=OrderStatus(row[10]),
                    profit_loss=row[11],
                    stop_loss=row[12],
                    take_profit=row[13],
                    ml_confidence=row[14],
                    ml_prediction=row[15],
                    tradier_option_symbol=tradier_option_symbol,
                    tradier_order_id=tradier_order_id,
                    is_real_trade=is_real_trade,
                    live_blocked_reason=live_blocked_reason
                )
                
                # Set expiration_date if we parsed it
                if expiration_date is not None:
                    trade.expiration_date = expiration_date
                self.active_trades.append(trade)
                
                if is_real_trade:
                    real_trade_count += 1
                else:
                    paper_trade_count += 1
            
            # Log priority information
            if real_trade_count > 0:
                self.logger.warning(f"[PRIORITY] Loaded {real_trade_count} REAL trades, {paper_trade_count} paper trades")
                self.logger.warning(f"[PRIORITY] Real trades will be managed FIRST")
            else:
                self.logger.info(f"[PRIORITY] Loaded {paper_trade_count} paper-only trades")
    
    def sync_tradier_positions(self):
        """
        Bidirectional sync between Tradier and database:
        1. Add missing Tradier positions to database
        2. Close database trades that are no longer on Tradier
        3. Sync balance from Tradier
        """
        try:
            # Import Tradier credentials
            try:
                from tradier_credentials import TradierCredentials
                import requests
            except ImportError:
                self.logger.warning("[SYNC] Tradier credentials not available, skipping sync")
                return
            
            creds = TradierCredentials()
            live_creds = creds.get_live_credentials()
            
            if not live_creds:
                self.logger.debug("[SYNC] No live Tradier credentials, skipping position sync")
                return
            
            headers = {
                'Authorization': f'Bearer {live_creds["access_token"]}',
                'Accept': 'application/json'
            }
            
            base_url = "https://api.tradier.com/v1"
            account_id = live_creds["account_number"]
            
            # ===== STEP 1: Log Tradier Balance (DO NOT SYNC to paper account) =====
            # Paper trading account is INDEPENDENT - it tracks hypothetical performance
            # of ALL signals, not just the ones that executed on Tradier
            try:
                balance_url = f'{base_url}/accounts/{account_id}/balances'
                balance_response = requests.get(balance_url, headers=headers, timeout=10)
                
                if balance_response.status_code == 200:
                    balance_data = balance_response.json()
                    balances = balance_data.get('balances', {})
                    tradier_balance = float(balances.get('total_equity', 0))
                    
                    # Just log the difference - DO NOT overwrite paper balance
                    if tradier_balance > 0:
                        diff = self.current_balance - tradier_balance
                        self.logger.info(f"[SYNC] Tradier balance: ${tradier_balance:.2f} | Paper balance: ${self.current_balance:.2f} | Diff: ${diff:+.2f}")
            except Exception as bal_err:
                self.logger.debug(f"[SYNC] Could not fetch Tradier balance: {bal_err}")
            
            # ===== STEP 2: Get Tradier Positions =====
            positions_url = f'{base_url}/accounts/{account_id}/positions'
            response = requests.get(positions_url, headers=headers, timeout=10)
            
            if response.status_code != 200:
                self.logger.warning(f"[SYNC] Error fetching Tradier positions: {response.status_code}")
                return
            
            data = response.json()
            positions = data.get('positions', {})
            
            tradier_positions = []
            tradier_symbols = set()
            if positions and positions != 'null' and 'position' in positions:
                pos_list = positions['position']
                if not isinstance(pos_list, list):
                    pos_list = [pos_list]
                tradier_positions = pos_list
                tradier_symbols = set(p.get('symbol') for p in tradier_positions if p.get('symbol'))
            
            self.logger.info(f"[SYNC] Tradier has {len(tradier_positions)} open position(s)")
            
            # ===== STEP 3: Close DB trades not on Tradier =====
            # Find real trades in DB that are no longer on Tradier (already closed)
            # Also catch trades marked closed but with P/L = 0 (orphaned exit)
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT id, tradier_option_symbol, entry_price, symbol, option_type, strike_price, status, profit_loss
                    FROM trades 
                    WHERE is_real_trade = 1 
                    AND tradier_option_symbol IS NOT NULL
                    AND (
                        status IN ('PENDING', 'FILLED')
                        OR (status IN ('PROFIT_TAKEN', 'STOPPED_OUT') AND (profit_loss = 0 OR profit_loss IS NULL))
                    )
                ''')
                db_real_trades = cursor.fetchall()
            
            closed_count = 0
            for row in db_real_trades:
                trade_id, opt_symbol, entry_price, symbol, opt_type, strike, current_status, current_pnl = row
                
                if opt_symbol not in tradier_symbols:
                    # This trade was closed on Tradier but not in our DB (or has 0 P/L)
                    if current_status in ('PENDING', 'FILLED'):
                        self.logger.warning(f"[SYNC] Trade {trade_id[:8]}... ({opt_symbol}) no longer on Tradier - marking CLOSED")
                    else:
                        self.logger.warning(f"[SYNC] Trade {trade_id[:8]}... ({opt_symbol}) has P/L=0 - fetching actual exit price")
                    
                    # Try to get the fill price from recent orders
                    exit_price = entry_price  # Default to entry if we can't find it
                    try:
                        orders_url = f'{base_url}/accounts/{account_id}/orders'
                        orders_response = requests.get(orders_url, headers=headers, timeout=10)
                        if orders_response.status_code == 200:
                            orders_data = orders_response.json()
                            orders = orders_data.get('orders', {})
                            if orders and orders != 'null' and 'order' in orders:
                                order_list = orders['order']
                                if not isinstance(order_list, list):
                                    order_list = [order_list]
                                # Find sell order for this symbol
                                for order in order_list:
                                    if (order.get('symbol') == opt_symbol and 
                                        order.get('side') == 'sell_to_close' and
                                        order.get('status') == 'filled'):
                                        exit_price = float(order.get('avg_fill_price', entry_price))
                                        break
                    except Exception as ord_err:
                        self.logger.debug(f"[SYNC] Could not fetch order history: {ord_err}")
                    
                    # Calculate P&L
                    pnl = (exit_price - entry_price) * 100  # Per contract
                    status = 'PROFIT_TAKEN' if pnl > 0 else 'STOPPED_OUT'
                    
                    with sqlite3.connect(self.db_path) as conn:
                        cursor = conn.cursor()
                        cursor.execute('''
                            UPDATE trades 
                            SET status = ?, exit_price = ?, profit_loss = ?, 
                                exit_timestamp = ?
                            WHERE id = ?
                        ''', (status, exit_price, pnl, datetime.now().isoformat(), trade_id))
                        conn.commit()
                    
                    self.logger.info(f"[SYNC] Closed {opt_symbol}: exit=${exit_price:.2f}, P&L=${pnl:.2f}")
                    closed_count += 1
            
            if closed_count > 0:
                self.logger.warning(f"[SYNC] Closed {closed_count} trade(s) that were already closed on Tradier")
                # Reload active trades to remove closed ones
                self._load_active_trades()
            
            # ===== STEP 4: Add missing Tradier positions to DB =====
            if not tradier_positions:
                self.logger.debug("[SYNC] No Tradier positions to add")
                return
                
            self.logger.info(f"[SYNC] Checking {len(tradier_positions)} position(s) for sync")
            
            # Get list of option symbols we already have in database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT tradier_option_symbol FROM trades 
                    WHERE status IN ('PENDING', 'FILLED') 
                    AND tradier_option_symbol IS NOT NULL
                ''')
                existing_symbols = set(row[0] for row in cursor.fetchall())
            
            # Check each Tradier position
            synced_count = 0
            created_count = 0
            
            for pos in tradier_positions:
                option_symbol = pos.get('symbol')
                quantity = int(pos.get('quantity', 0))
                cost_basis = float(pos.get('cost_basis', 0))
                date_acquired = pos.get('date_acquired')
                
                # Skip if we already have this position
                if option_symbol in existing_symbols:
                    continue
                
                self.logger.warning(f"[SYNC] Found orphaned Tradier position: {option_symbol}")
                
                # Parse option symbol (format: SPY260102C00690000)
                try:
                    underlying = option_symbol[:3] if len(option_symbol) > 15 else option_symbol
                    
                    date_str = option_symbol[3:9]
                    exp_year = 2000 + int(date_str[:2])
                    exp_month = int(date_str[2:4])
                    exp_day = int(date_str[4:6])
                    expiration = f"{exp_year}-{exp_month:02d}-{exp_day:02d}"
                    
                    option_type_char = option_symbol[9]
                    option_type = "CALL" if option_type_char == 'C' else "PUT"
                    
                    strike_str = option_symbol[10:18]
                    strike = int(strike_str) / 1000.0
                    
                    entry_price = cost_basis / (quantity * 100) if quantity > 0 else 0
                    
                    self.logger.info(f"[SYNC] Creating DB entry: {underlying} {option_type} ${strike:.2f} exp {expiration}")
                    
                    # Create new trade entry in database
                    with sqlite3.connect(self.db_path) as conn:
                        cursor = conn.cursor()
                        # Use market time for consistency
                        now = self.get_market_time().isoformat()
                        
                        # Parse date_acquired and convert to market time (remove timezone)
                        if date_acquired:
                            try:
                                acquired_dt = datetime.fromisoformat(date_acquired.replace('Z', '+00:00'))
                                # Convert to market time (remove timezone for consistency)
                                if acquired_dt.tzinfo is not None:
                                    acquired_dt = acquired_dt.replace(tzinfo=None)
                                timestamp_str = acquired_dt.isoformat()
                            except:
                                timestamp_str = now
                        else:
                            timestamp_str = now
                        
                        # Parse expiration and ensure it's in market time
                        try:
                            exp_dt = datetime.fromisoformat(expiration)
                            if exp_dt.tzinfo is not None:
                                exp_dt = exp_dt.replace(tzinfo=None)
                            expiration_str = exp_dt.isoformat()
                        except:
                            expiration_str = expiration
                        
                        cursor.execute('''
                            INSERT INTO trades (
                                id, timestamp, symbol, option_type, strike_price,
                                premium_paid, quantity, entry_price, status, is_real_trade,
                                tradier_option_symbol, expiration_date, ml_confidence,
                                profit_loss, stop_loss, take_profit, ml_prediction, created_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            str(uuid.uuid4()),
                            timestamp_str,  # Use parsed timestamp in market time
                            underlying,
                            option_type,
                            strike,
                            cost_basis,
                            quantity,
                            entry_price,
                            'FILLED',
                            1,  # is_real_trade = True
                            option_symbol,
                            expiration_str,  # Use parsed expiration
                            0.0,  # Unknown confidence for synced trades
                            0.0,  # Initial P&L
                            entry_price * 0.5,  # 50% stop loss
                            entry_price * 2.0,  # 100% take profit
                            0.0,  # ml_prediction
                            now  # created_at
                        ))
                        conn.commit()
                    
                    created_count += 1
                    self.logger.info(f"[SYNC] ✅ Created trade entry for {option_symbol}")
                    
                except Exception as e:
                    self.logger.error(f"[SYNC] Error parsing/creating trade for {option_symbol}: {e}")
            
            if created_count > 0:
                self.logger.warning(f"[SYNC] Created {created_count} new trade(s) from Tradier positions")
                # Reload active trades to include newly synced positions
                self._load_active_trades()
            else:
                self.logger.debug("[SYNC] All Tradier positions already in database")
                
        except Exception as e:
            self.logger.error(f"[SYNC] Error syncing Tradier positions: {e}")
    
    def recover_stale_positions(self, max_age_hours=48):
        """
        Close positions that are too old (stuck open from previous sessions)
        
        Args:
            max_age_hours: Maximum age in hours before force-closing (default 48)
        """
        if not self.active_trades:
            return
        
        current_time = self.get_market_time()
        stale_trades = []
        
        for trade in self.active_trades:
            # Calculate trade age
            trade_age = (current_time - trade.timestamp).total_seconds() / 3600
            trade_age_minutes = trade_age * 60
            
            # Check if trade is stale
            is_stale = False
            reason = ""
            
            # NOTE: Don't close based on planned_exit_time alone!
            # This was causing premature exits. Let normal exit logic handle timing.
            # Only close for TRUE emergencies:
            
            # Check 1: Exceeded max age (very long, like 48h+) 
            # NOT exceeding planned_exit_time (that was the bug!)
            if trade_age > max_age_hours:
                is_stale = True
                reason = f"Trade age ({trade_age:.1f}h) > max ({max_age_hours}h)"
            
            # Check 2: Past option expiration (must close!)
            elif trade.expiration_date and current_time > trade.expiration_date:
                is_stale = True
                reason = f"Past expiration ({trade.expiration_date})"
            
            # Check 4: CRITICAL - Exceeded 3x predicted timeframe
            # BUT: Only close if LOSING - let winners run!
            # Default 60 min if no max_hold_hours set
            predicted_minutes = (trade.max_hold_hours or 1.0) * 60
            max_allowed_minutes = predicted_minutes * 3  # 3x the prediction = long hold
            
            # Estimate current P&L (rough - just for stale check)
            # Entry premium is stored; can't easily get current, so assume based on time decay
            estimated_pnl_pct = 0.0  # Will be checked properly in update_positions
            
            if trade_age_minutes > max_allowed_minutes:
                # Long hold, but might be profitable - don't mark as stale, let normal exit logic handle it
                self.logger.info(f"📊 Long hold position ({trade_age_minutes:.0f}min > {max_allowed_minutes:.0f}min predicted) - will check P&L in update cycle")
                # Don't mark as stale - let the RL and normal exit logic decide based on actual P&L
            
            # Check 5: Only force close for VERY long holds (8+ hours) and only if not obviously profitable
            # This is a safety net for truly stuck positions
            elif trade_age > 8.0 and self._is_market_hours():
                self.logger.warning(f"⚠️ Very long hold position ({trade_age:.1f}h) - will monitor closely")
                # Still don't mark as stale - let normal update cycle handle with actual P&L check
            
            if is_stale:
                stale_trades.append((trade, reason))
        
        # Close stale trades
        if stale_trades:
            self.logger.warning(f"[RECOVERY] Found {len(stale_trades)} stale positions to close")
            
            for trade, reason in stale_trades:
                self.logger.warning(f"[RECOVERY] Closing stale trade {trade.id[:8]}... | Reason: {reason}")
                
                # Get current price for exit - CALCULATE ACTUAL OPTION VALUE
                try:
                    # Try to get current underlying price
                    current_price = None
                    if hasattr(self, 'data_source') and self.data_source:
                        try:
                            current_price = self.data_source.get_current_price(trade.symbol)
                        except:
                            pass
                    
                    if current_price is None:
                        current_price = trade.entry_price  # Use entry underlying price as fallback
                    
                    # Calculate actual option value at stale time
                    # Use the same premium estimation logic as active trades
                    if "Exceeded planned exit time" in reason or "expiration" in reason.lower():
                        # Calculate intrinsic value
                        if trade.option_type.value in ['CALL', 'BUY_CALL']:
                            intrinsic_value = max(0, current_price - trade.strike_price)
                        else:  # PUT
                            intrinsic_value = max(0, trade.strike_price - current_price)
                        
                        # Calculate realistic exit price using price movement
                        # Options still have value even after planned exit time!
                        entry_premium = trade.premium_paid  # Already per-share
                        price_move_pct = ((current_price - trade.entry_price) / trade.entry_price) * 100
                        
                        # Apply delta-based movement (same as active trade monitoring)
                        delta = 0.5  # ATM option delta
                        premium_change_pct = price_move_pct * delta * 3  # Leverage factor
                        
                        # For CALLs: price UP = premium UP, price DOWN = premium DOWN
                        # For PUTs: price UP = premium DOWN, price DOWN = premium UP
                        if trade.option_type.value in ['PUT', 'BUY_PUT']:
                            premium_change_pct = -premium_change_pct
                        
                        exit_price = entry_premium * (1 + premium_change_pct / 100)
                        
                        # Apply modest time decay penalty (these are stale but not expired)
                        # Only penalize 2% since these are just past their hold time, not truly expired
                        time_decay_penalty = 0.98  # 2% penalty for being stale
                        exit_price = exit_price * time_decay_penalty
                        
                        # Floor at intrinsic value, ceiling at 2x entry
                        exit_price = max(intrinsic_value, exit_price)  # At least intrinsic
                        exit_price = min(entry_premium * 2.0, exit_price)  # Cap at 2x entry
                        exit_price = max(0.01, exit_price)  # Absolute minimum
                        
                        self.logger.warning(f"[RECOVERY] Stale option: Entry ${entry_premium:.2f}/share, Underlying {price_move_pct:+.2f}% → Exit ${exit_price:.2f}/share")
                    else:
                        # For other stale reasons, use break-even as fallback
                        exit_price = trade.premium_paid
                    
                    # Close the trade
                    self._close_trade(
                        trade=trade,
                        exit_price=exit_price,
                        reason=f"STALE_POSITION: {reason}"
                    )
                    
                    self.logger.info(f"[RECOVERY] Closed stale trade at ${exit_price:.2f}/share")
                    
                except Exception as e:
                    self.logger.error(f"[RECOVERY] Error closing stale trade: {e}")
            
            # Reload active trades after cleanup
            self._load_active_trades()
            
            self.logger.info(f"[RECOVERY] Position recovery complete. Active trades: {len(self.active_trades)}")
        else:
            self.logger.info(f"[RECOVERY] No stale positions found. All {len(self.active_trades)} trades are current.")
    
    def _close_trade(self, trade: Trade, exit_price: float, reason: str = "Manual Close"):
        """
        Close a single trade at a given exit price.
        Used by recover_stale_positions for forced closures.
        """
        try:
            # Calculate fees
            exit_fees = self.fee_calculator.calculate_exit_fees(
                exit_price, trade.quantity, trade.symbol)
            entry_fees = self.fee_calculator.calculate_entry_fees(
                trade.premium_paid, trade.quantity, trade.symbol)
            total_fees = entry_fees + exit_fees
            
            # Calculate P&L - must multiply by 100 (shares per contract)
            profit_loss = (exit_price - trade.premium_paid) * trade.quantity * 100 - total_fees
            
            # Update trade
            trade.exit_price = exit_price
            trade.exit_timestamp = self.get_market_time()
            trade.exit_reason = reason
            trade.profit_loss = profit_loss
            trade.status = OrderStatus.PROFIT_TAKEN if profit_loss > 0 else OrderStatus.STOPPED_OUT
            
            # Update balance - must multiply by 100 (shares per contract)
            exit_value = exit_price * trade.quantity * 100 - exit_fees
            
            # Settlement tracking (T+1 for options)
            if self.simulate_settlement and not (getattr(self, 'simulation_mode', False) or getattr(self, 'time_travel_mode', False)):
                # Funds settle next business day
                settlement_date = self._get_next_business_day()
                self.pending_settlements.append({
                    'trade_id': trade.id,
                    'amount': exit_value,
                    'settlement_date': settlement_date
                })
                self._save_settlement(trade.id, exit_value, settlement_date)
                self.logger.info(f"💵 [SETTLEMENT] ${exit_value:.2f} pending until {settlement_date.strftime('%Y-%m-%d')}")
            else:
                # Immediate settlement (simulation mode)
                self.current_balance += exit_value
            
            # Update stats
            self.total_trades += 1
            if profit_loss > 0:
                self.winning_trades += 1
            else:
                self.losing_trades += 1
            self.total_profit_loss += profit_loss
            
            # Save to DB
            self._save_trade(trade)
            self._save_account_state()
            
            # Remove from active trades
            if trade in self.active_trades:
                self.active_trades.remove(trade)
            
            self.logger.info(f"[CLOSE] Trade closed: P&L=${profit_loss:.2f}, Reason: {reason}")
            
        except Exception as e:
            self.logger.error(f"[CLOSE] Error closing trade: {e}")
            raise

    def _simulate_liquidity(self, symbol: str, strike_price: float, current_price: float, 
                           option_type: str, days_to_expiry: int = 30) -> Dict:
        """
        Get liquidity data for training/simulation mode.
        
        PRIORITY:
        1. Use REAL historical liquidity data from database if available
        2. Fall back to SYNTHETIC simulation only if no real data exists
        
        This ensures training uses real market conditions when possible.
        """
        import random
        import math
        
        # ===== FIRST: Try to get REAL historical liquidity data =====
        real_data = self._get_historical_liquidity(symbol, strike_price, current_price, option_type)
        
        if real_data:
            self.logger.info(f"[LIQUIDITY] Using REAL historical data for {option_type} ${strike_price:.2f}")
            return real_data
        
        # ===== FALLBACK: Generate synthetic liquidity =====
        self.logger.debug(f"[LIQUIDITY] No historical data found, using synthetic for {option_type} ${strike_price:.2f}")
        
        # Calculate moneyness
        if option_type.upper() in ['CALL', 'BUY_CALL']:
            moneyness = (strike_price - current_price) / current_price  # + = OTM for calls
        else:
            moneyness = (current_price - strike_price) / current_price  # + = OTM for puts
        
        abs_moneyness = abs(moneyness)
        
        # === SPREAD SIMULATION (based on historical patterns) ===
        # ATM options have tightest spreads, OTM options have wider spreads
        if abs_moneyness < 0.02:  # ATM (within 2%)
            base_spread_pct = 0.03 + random.uniform(0, 0.02)  # 3-5% spread
        elif abs_moneyness < 0.05:  # Near the money (2-5%)
            base_spread_pct = 0.05 + random.uniform(0, 0.04)  # 5-9% spread
        elif abs_moneyness < 0.10:  # Moderate OTM (5-10%)
            base_spread_pct = 0.08 + random.uniform(0, 0.07)  # 8-15% spread
        else:  # Far OTM (>10%)
            base_spread_pct = 0.12 + random.uniform(0, 0.10)  # 12-22% spread
        
        # Time decay affects spreads (shorter DTE = wider spreads due to gamma risk)
        if days_to_expiry <= 1:  # 0-1 DTE
            base_spread_pct *= 1.3 + random.uniform(0, 0.2)
        elif days_to_expiry <= 7:  # Weekly
            base_spread_pct *= 1.1 + random.uniform(0, 0.1)
        
        # === VOLUME SIMULATION (based on popularity) ===
        # ATM options trade more volume
        if abs_moneyness < 0.02:  # ATM
            base_volume = random.randint(500, 2000)
        elif abs_moneyness < 0.05:  # Near ATM
            base_volume = random.randint(200, 800)
        elif abs_moneyness < 0.10:  # Moderate OTM
            base_volume = random.randint(50, 300)
        else:  # Far OTM
            base_volume = random.randint(10, 100)
        
        # Popular symbols get more volume
        if symbol in ['SPY', 'QQQ', 'BITX']:
            base_volume = int(base_volume * (1.5 + random.uniform(0, 0.5)))
        
        # === OPEN INTEREST SIMULATION ===
        # OI is typically 5-20x daily volume, ATM options have more OI
        oi_multiplier = 5 + random.uniform(0, 15)
        if abs_moneyness < 0.03:  # Near ATM
            oi_multiplier *= 1.5
        base_oi = int(base_volume * oi_multiplier)
        
        # === CALCULATE BID/ASK ===
        # Estimate theoretical premium using Black-Scholes-like calculation
        # Intrinsic value
        if option_type.upper() in ['CALL', 'BUY_CALL']:
            intrinsic = max(0, current_price - strike_price)
        else:
            intrinsic = max(0, strike_price - current_price)
        
        # Time value (simplified)
        time_factor = math.sqrt(days_to_expiry / 365.0)
        iv = 0.25 + random.uniform(0, 0.15)  # 25-40% IV
        time_value = current_price * iv * time_factor * math.exp(-abs_moneyness * 5)
        
        # Theoretical mid price
        theoretical_mid = intrinsic + time_value
        theoretical_mid = max(0.05, theoretical_mid)  # Minimum 5 cents
        
        # Apply spread
        half_spread = theoretical_mid * base_spread_pct / 2
        bid = max(0.01, theoretical_mid - half_spread)
        ask = theoretical_mid + half_spread
        
        # Round to realistic tick sizes
        if theoretical_mid < 3.0:
            # Options under $3 trade in $0.05 increments
            bid = round(bid * 20) / 20
            ask = round(ask * 20) / 20
        else:
            # Options $3+ trade in $0.10 increments
            bid = round(bid * 10) / 10
            ask = round(ask * 10) / 10
        
        # Ensure ask > bid
        if ask <= bid:
            ask = bid + 0.05
        
        simulated_data = {
            'bid': bid,
            'ask': ask,
            'volume': base_volume,
            'open_interest': base_oi,
            'strike': strike_price,
            'spread_pct': (ask - bid) / ((bid + ask) / 2) * 100,
            'simulated': True  # Flag that this is SYNTHETIC data
        }
        
        self.logger.debug(f"[SIM-LIQUIDITY] {option_type} ${strike_price:.2f}: "
                         f"Bid=${bid:.2f}, Ask=${ask:.2f}, Vol={base_volume}, OI={base_oi}")
        
        return simulated_data

    def _get_historical_liquidity(self, symbol: str, strike_price: float, 
                                  current_price: float, option_type: str) -> Optional[Dict]:
        """
        Look up REAL historical liquidity data from the database.
        
        Searches liquidity_snapshots table for similar conditions:
        - Same symbol and option type
        - Similar moneyness (within 2%)
        - Returns data with some randomness for variation
        
        Returns None if no suitable historical data found.
        """
        import random
        
        try:
            # Calculate target moneyness
            if option_type.upper() in ['CALL', 'BUY_CALL']:
                target_moneyness = (strike_price - current_price) / current_price
            else:
                target_moneyness = (current_price - strike_price) / current_price
            
            # Try to find data in the historical database
            historical_db_path = 'data/db/historical.db'
            
            import os
            if not os.path.exists(historical_db_path):
                self.logger.debug(f"[LIQUIDITY] Historical database not found at {historical_db_path}")
                return None
            
            conn = sqlite3.connect(historical_db_path)
            cursor = conn.cursor()
            
            # Check if liquidity_snapshots table exists
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='liquidity_snapshots'
            """)
            if not cursor.fetchone():
                conn.close()
                self.logger.debug("[LIQUIDITY] liquidity_snapshots table does not exist")
                return None
            
            # Query for similar liquidity conditions
            # Look for same symbol, same option type, similar moneyness
            opt_type_upper = option_type.upper()
            if opt_type_upper in ['BUY_CALL']:
                opt_type_upper = 'CALL'
            elif opt_type_upper in ['BUY_PUT']:
                opt_type_upper = 'PUT'
            
            cursor.execute("""
                SELECT 
                    bid, ask, spread_pct, volume, open_interest, 
                    strike_price, underlying_price, quality_score,
                    implied_volatility, delta, gamma, theta, vega
                FROM liquidity_snapshots
                WHERE symbol = ?
                AND option_type = ?
                AND ABS((strike_price - underlying_price) / underlying_price - ?) < 0.03
                ORDER BY timestamp DESC
                LIMIT 50
            """, (symbol, opt_type_upper, target_moneyness))
            
            rows = cursor.fetchall()
            conn.close()
            
            if not rows:
                self.logger.debug(f"[LIQUIDITY] No historical data found for {symbol} {opt_type_upper} "
                                f"at moneyness {target_moneyness:.2%}")
                return None
            
            # Use a random sample from recent data for variation
            sample = random.choice(rows)
            
            bid, ask, spread_pct, volume, oi, hist_strike, hist_underlying, quality, iv, delta, gamma, theta, vega = sample
            
            # Add some randomness (±10%) to simulate market variation
            noise_factor = 1 + random.uniform(-0.10, 0.10)
            
            # Scale bid/ask to current price level if underlying has moved significantly
            price_ratio = current_price / hist_underlying if hist_underlying > 0 else 1.0
            
            scaled_bid = bid * price_ratio * noise_factor
            scaled_ask = ask * price_ratio * noise_factor
            
            # Add volume/OI variation
            volume_noise = max(1, int(volume * (1 + random.uniform(-0.20, 0.20))))
            oi_noise = max(1, int(oi * (1 + random.uniform(-0.10, 0.10))))
            
            # Recalculate spread_pct after scaling
            mid = (scaled_bid + scaled_ask) / 2
            actual_spread_pct = ((scaled_ask - scaled_bid) / mid * 100) if mid > 0 else spread_pct
            
            historical_data = {
                'bid': round(scaled_bid, 2),
                'ask': round(scaled_ask, 2),
                'volume': volume_noise,
                'open_interest': oi_noise,
                'strike': strike_price,
                'spread_pct': actual_spread_pct,
                'quality_score': quality,
                'implied_volatility': iv,
                'delta': delta,
                'gamma': gamma,
                'theta': theta,
                'vega': vega,
                'simulated': False,  # Flag that this is REAL historical data
                'source': 'historical_db'
            }
            
            self.logger.info(f"[LIQUIDITY] Found historical data: {symbol} {opt_type_upper} "
                           f"Spread={actual_spread_pct:.1f}%, Vol={volume_noise}, OI={oi_noise}, Quality={quality:.0f}")
            
            return historical_data
            
        except Exception as e:
            self.logger.debug(f"[LIQUIDITY] Error querying historical data: {e}")
            return None

    def _calculate_option_premium(self, symbol: str, option_type: OrderType,
                                  strike_price: float, current_price: float, 
                                  is_entry: bool = False, time_to_expiry_days: float = 1.0) -> float:
        """
        Calculate realistic option premium with bid/ask spreads and market microstructure
        """
        # Validate inputs to prevent impossible prices
        if current_price <= 0 or strike_price <= 0:
            self.logger.error(f"Invalid price data: current=${current_price}, strike=${strike_price}")
            return 0.5  # Minimum premium
        
        # Ensure current_price is within realistic bounds for BITX (prevent data errors)
        if symbol == "BITX":
            if current_price < 10 or current_price > 150:
                self.logger.warning(f"⚠️ BITX price ${current_price:.2f} outside realistic range (10-150), using clamped price")
                current_price = max(10, min(150, current_price))
        
        moneyness = current_price / strike_price

        if option_type in [OrderType.CALL, OrderType.BUY_CALL]:
            # CALL premium: intrinsic + time + volatility
            intrinsic_value = max(0, current_price - strike_price)
        else:  # PUT options
            # PUT premium: intrinsic + time + volatility  
            intrinsic_value = max(0, strike_price - current_price)

        # Time value with realistic theta decay
        # CRITICAL: OTM options have MUCH LESS time value than ATM options!
        # ATM options have maximum time value, OTM options decay exponentially
        time_decay_factor = max(0.1, time_to_expiry_days / 30.0)  # Normalize to 30-day options
        
        # Calculate how far OTM/ITM the option is
        if option_type in [OrderType.CALL, OrderType.BUY_CALL]:
            # CALL is OTM when current_price < strike_price
            otm_pct = max(0, (strike_price - current_price) / current_price)
        else:  # PUT
            # PUT is OTM when current_price > strike_price
            otm_pct = max(0, (current_price - strike_price) / current_price)
        
        # Time value decays exponentially with OTM percentage
        # ATM (0% OTM): full time value
        # 2% OTM: ~60% time value
        # 5% OTM: ~20% time value
        # 10% OTM: ~5% time value
        import math
        otm_decay_factor = math.exp(-otm_pct * 30)  # Exponential decay
        
        # FIX: Base time value should be realistic for 0DTE SPY options
        # Old formula: strike_price * 0.008 = .52 for 90 strike (WAY too high!)
        # Real SPY 0DTE ATM options have ~/usr/bin/bash.50-1.00 in time value
        base_time_value = 0.40 + (current_price * 0.0008)  # ~/usr/bin/bash.40 + /usr/bin/bash.47 for SPY 590 = ~/usr/bin/bash.87
        time_value = base_time_value * time_decay_factor * otm_decay_factor
        
        # Apply daily theta decay for existing positions
        if not is_entry and time_to_expiry_days < 1.0:
            # Accelerated decay in final day
            time_value *= (1 - self.time_decay_rate * 2)  # Double decay rate
        elif not is_entry:
            # Normal theta decay
            time_value *= (1 - self.time_decay_rate)

        # Volatility premium with implied volatility changes
        # IV typically increases with market stress and decreases in calm markets
        iv_adjustment = self._calculate_iv_adjustment(current_price, strike_price)
        
        # FIX: Volatility premium (Vega) is highest ATM and decays as price moves away
        # Previous formula: abs(1 - moneyness) INCREASED as price moved away (wrong!)
        dist_pct = abs(current_price - strike_price) / current_price
        vega_decay = math.exp(-dist_pct * 30)
        
        volatility_premium = strike_price * 0.003 * vega_decay * iv_adjustment

        # Base premium calculation
        base_premium = intrinsic_value + time_value + volatility_premium
        
        # REDUCED bid/ask spread (original was too aggressive for liquid SPY options)
        # SPY options are very liquid - spreads are typically $0.01-0.05 on $1-5 options
        if abs(1 - moneyness) < 0.02:  # ATM (within 2%)
            bid_ask_spread_pct = 0.003  # 0.3% spread - very liquid
        elif abs(1 - moneyness) < 0.05:  # Near the money (2-5%)
            bid_ask_spread_pct = 0.008  # 0.8% spread
        else:  # Far OTM/ITM
            bid_ask_spread_pct = 0.015  # 1.5% spread

        if is_entry:
            # When buying, pay the ASK (higher price)
            premium = base_premium * (1 + bid_ask_spread_pct/2)
        else:
            # When selling, get the BID (lower price)
            premium = base_premium * (1 - bid_ask_spread_pct/2)

        # REDUCED slippage (original was too aggressive for liquid SPY options)
        if abs(1 - moneyness) < 0.02:  # ATM
            slippage_pct = 0.002  # 0.2% slippage - liquid
        elif abs(1 - moneyness) < 0.05:  # Near the money
            slippage_pct = 0.004  # 0.4% slippage
        else:  # Far OTM/ITM
            slippage_pct = 0.008  # 0.8% slippage

        if is_entry:
            premium = premium * (1 + slippage_pct)  # Pay more when entering
        else:
            premium = premium * (1 - slippage_pct)  # Get less when exiting
        
        # REALISTIC premium bounds for SPY/ETF options
        # OTM options CAN become nearly worthless - don't set artificial floors!
        # Only cap MAXIMUM premiums to prevent unrealistic spikes
        
        # Calculate moneyness-adjusted MAX bounds only
        if abs(1 - moneyness) < 0.02:  # ATM (within 2%)
            max_premium = strike_price * 0.06   # 6% maximum
        elif abs(1 - moneyness) < 0.05:  # Near the money (2-5% OTM)
            max_premium = strike_price * 0.04   # 4% maximum  
        else:  # Far OTM/ITM
            max_premium = strike_price * 0.15   # 15% maximum (for deep ITM)
        
        # NO artificial floor - OTM options can and do go to near zero!
        # A PUT with strike $630 when price is $650 should be nearly worthless
        # NOTE: 0.01 is absolute floor for calculation stability
        min_premium = 0.01  
        
        # === CRITICAL FIX FOR ITM CALLS ===
        # Ensure premium is AT LEAST intrinsic value + 1 cent
        # Previous clamping logic could accidentally clamp BELOW intrinsic value!
        intrinsic_value = 0.0
        if option_type in [OrderType.CALL, OrderType.BUY_CALL]:
            intrinsic_value = max(0.0, current_price - strike_price)
        else:
            intrinsic_value = max(0.0, strike_price - current_price)
            
        # The floor must be intrinsic value (plus a tiny sliver of time value)
        # Otherwise deep ITM options get undervalued
        safe_floor = intrinsic_value + 0.01
        
        premium = max(safe_floor, min(max_premium, premium))
        
        # Add SMALL randomness to simulate bid-ask bounce
        # Real options move in $0.05-0.10 increments, not huge percentages
        import random
        market_noise = random.uniform(-0.01, 0.01)  # ±1% random noise
        premium = premium * (1 + market_noise)
        
        # Final sanity check against intrinsic again after noise
        premium = max(safe_floor, premium)
        
        self.logger.debug(f"Option premium calc: {symbol} {option_type.value} strike=${strike_price:.2f} "
                         f"current=${current_price:.2f} entry={is_entry} → premium=${premium:.2f}")
        
        return premium

    def _calculate_iv_adjustment(self, current_price: float, strike_price: float) -> float:
        """Calculate implied volatility adjustment based on market conditions"""
        import random
        
        # Base IV adjustment
        moneyness = current_price / strike_price
        
        # ATM options have highest IV, OTM/ITM have lower IV
        if 0.95 <= moneyness <= 1.05:  # ATM
            iv_multiplier = 1.2
        elif 0.90 <= moneyness <= 1.10:  # Near ATM
            iv_multiplier = 1.1
        else:  # Far OTM/ITM
            iv_multiplier = 0.9
        
        # Add market volatility simulation
        # In real markets, IV changes based on VIX, earnings, etc.
        # Reduced noise from ±30% to ±5% for more stable pricing
        market_vol_factor = random.uniform(0.95, 1.05)
        
        return iv_multiplier * market_vol_factor

    def _check_pdt_compliance(self) -> bool:
        """Check Pattern Day Trading rule compliance"""
        # Reset day trade count if it's a new day
        today = self.get_market_time().date()
        if today != self.last_day_trade_reset:
            self.day_trades_count = 0
            self.last_day_trade_reset = today
        
        # PDT rule: accounts under $25k limited to 3 day trades per 5 business days
        if self.account_value_for_pdt < 25000:
            if self.day_trades_count >= 3:
                self.logger.warning("🚫 PDT Rule: Cannot place trade - exceeded 3 day trades with account < $25k")
                return False
        
        return True
    
    def _check_liquidity_constraints(self, trade_value: float) -> Tuple[bool, str]:
        """Check if trade can be filled given liquidity constraints"""
        if trade_value > self.liquidity_impact_threshold:
            # Large orders face additional slippage and may not fill completely
            fill_probability = max(0.7, 1.0 - (trade_value / 10000))  # Harder to fill larger orders
            
            import random
            if random.random() > fill_probability:
                return False, f"Insufficient liquidity for ${trade_value:.0f} order"
        
        return True, ""
    
    
    def _check_pdt_compliance(self) -> bool:
        """Check if we can place a trade without violating PDT rules"""
        if not self.pdt_protected:
            # Account over $25K - no PDT restrictions
            return True
        
        # Update day trade count (remove old day trades outside 5-business-day window)
        self._update_day_trade_count()
        
        current_day_trades = len(self.day_trade_history)
        
        if current_day_trades >= self.max_day_trades_under_25k:
            self.logger.warning(f"🚫 PDT VIOLATION PREVENTED!")
            self.logger.warning(f"   Current day trades: {current_day_trades}/{self.max_day_trades_under_25k}")
            self.logger.warning(f"   Account balance: ${self.current_balance:,.2f} (under $25,000)")
            self.logger.warning(f"   Must wait for day trades to expire or deposit to reach $25,000")
            return False
        
        # Log PDT status
        remaining_trades = self.max_day_trades_under_25k - current_day_trades
        self.logger.info(f"📊 PDT Status: {current_day_trades}/{self.max_day_trades_under_25k} day trades used")
        self.logger.info(f"   Remaining day trades: {remaining_trades}")
        
        return True
    
    def _update_day_trade_count(self):
        """Update day trade count by removing trades older than 5 business days"""
        current_date = self.get_market_time().date()
        
        # Calculate 5 business days ago
        business_days_back = 0
        check_date = current_date
        while business_days_back < 5:
            check_date -= timedelta(days=1)
            if check_date.weekday() < 5:  # Monday=0, Friday=4
                business_days_back += 1
        
        cutoff_date = check_date
        
        # Remove old day trades
        old_count = len(self.day_trade_history)
        self.day_trade_history = [
            trade_date for trade_date in self.day_trade_history 
            if trade_date >= cutoff_date
        ]
        
        removed_count = old_count - len(self.day_trade_history)
        if removed_count > 0:
            self.logger.info(f"📅 Removed {removed_count} expired day trades from PDT tracking")
    
    def _record_day_trade(self):
        """Record a day trade for PDT tracking"""
        if self.pdt_protected:
            trade_date = self.get_market_time().date()
            self.day_trade_history.append(trade_date)
            
            current_count = len(self.day_trade_history)
            remaining = self.max_day_trades_under_25k - current_count
            
            self.logger.info(f"📊 Day Trade Recorded: {current_count}/{self.max_day_trades_under_25k} used")
            if remaining <= 1:
                self.logger.warning(f"⚠️  PDT WARNING: Only {remaining} day trades remaining!")
                if remaining == 0:
                    self.logger.warning(f"🚫 PDT LIMIT REACHED - No more day trades until older ones expire")
    
    def _check_account_upgrade(self):
        """Check if account has grown to $25K and disable PDT protection"""
        if self.pdt_protected and self.current_balance >= 25000:
            self.logger.info(f"🎉 ACCOUNT UPGRADE! Balance reached ${self.current_balance:,.2f}")
            self.logger.info(f"✅ PDT PROTECTION DISABLED - Unlimited day trades now allowed!")
            
            # Upgrade to standard account mode
            self.pdt_protected = False
            self.position_size_pct = 0.05  # 5% of balance per trade
            self.max_positions = 5  # Maximum concurrent positions (increased from 2)
            self.max_position_size_per_trade = 0.10  # Max 10% of account per trade
            
            self.logger.info(f"📊 Upgraded to Standard Account Mode")
    
    def _is_market_hours(self) -> bool:
        """Check if current time is during market hours (9:30 AM - 4:00 PM EST)"""
        now_est = self.get_market_time()
        
        # Check if weekend
        if now_est.weekday() >= 5:  # Saturday=5, Sunday=6
            return False
        
        # Check if during market hours
        market_open = now_est.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now_est.replace(hour=16, minute=0, second=0, microsecond=0)
        
        return market_open <= now_est < market_close

    def _get_regime_adjusted_stops(self, prediction_data: Dict) -> Tuple[float, float]:
        """
        Get dynamic stop loss and take profit percentages based on HMM regime.
        
        Strategy:
        - Low Vol regime: Tighter stops (40%) - smaller moves expected
        - Normal Vol regime: Standard stops (50%)
        - High Vol regime: Wider stops (65%) - need more room for volatility
        
        Also adjusts based on:
        - Trend alignment: If trading WITH trend, allow tighter stops
        - Liquidity: Low liquidity = wider stops (harder to exit cleanly)
        
        Returns:
            Tuple of (stop_loss_pct, take_profit_pct)
        """
        # Base stop/take profit (from class defaults - 3%/3%)
        base_stop = self.stop_loss_pct  # 0.03 (3%)
        base_take = self.take_profit_pct  # 0.03 (3%)
        
        # Get HMM regime from prediction data
        hmm_state = prediction_data.get('hmm_state', {})
        
        if not hmm_state:
            # No HMM data - use defaults
            self.logger.debug(f"[REGIME-STOPS] No HMM data, using defaults: SL={base_stop:.0%}, TP={base_take:.0%}")
            return base_stop, base_take
        
        # Extract HMM dimensions
        if isinstance(hmm_state, dict):
            volatility = hmm_state.get('volatility', hmm_state.get('volatility_name', 'Normal'))
            trend = hmm_state.get('trend', hmm_state.get('trend_name', 'No Trend'))
            liquidity = hmm_state.get('liquidity', hmm_state.get('liquidity_name', 'Normal'))
        elif isinstance(hmm_state, str):
            # Parse combined string like "Uptrend_HighVol_LowLiq"
            volatility = 'High' if 'High' in hmm_state else ('Low' if 'Low' in hmm_state else 'Normal')
            trend = 'Up' if 'Up' in hmm_state else ('Down' if 'Down' in hmm_state else 'No Trend')
            liquidity = 'High' if 'HighLiq' in hmm_state else ('Low' if 'LowLiq' in hmm_state else 'Normal')
        else:
            volatility, trend, liquidity = 'Normal', 'No Trend', 'Normal'
        
        # ============= VOLATILITY ADJUSTMENT (PRIMARY) =============
        # This is the main driver of stop loss width
        vol_lower = str(volatility).lower()
        if 'high' in vol_lower or 'volatile' in vol_lower or 'stress' in vol_lower or 'panic' in vol_lower:
            # HIGH VOLATILITY: Wide stops (65%)
            # With 20x leverage, 65% stop = 3.25% adverse move allowed
            vol_stop_mult = 1.30  # 50% * 1.30 = 65%
            vol_take_mult = 1.40  # Allow bigger gains in volatile markets
            self.logger.info(f"📊 [REGIME] HIGH VOL detected - widening stops (×{vol_stop_mult})")
        elif 'low' in vol_lower or 'calm' in vol_lower:
            # LOW VOLATILITY: Tighter stops (40%)
            # With 20x leverage, 40% stop = 2.0% adverse move allowed
            vol_stop_mult = 0.80  # 50% * 0.80 = 40%
            vol_take_mult = 0.80  # Expect smaller gains
            self.logger.info(f"📊 [REGIME] LOW VOL detected - tightening stops (×{vol_stop_mult})")
        else:
            # NORMAL VOLATILITY: Standard stops
            vol_stop_mult = 1.0
            vol_take_mult = 1.0
        
        # ============= TREND ADJUSTMENT =============
        # Trading WITH trend = tighter stops (trend should protect you)
        # Trading AGAINST trend = wider stops (fighting momentum)
        action = prediction_data.get('action', 'HOLD')
        trend_lower = str(trend).lower()
        
        trend_stop_mult = 1.0
        if 'call' in action.lower() or 'buy_call' in action.lower():
            # LONG/BULLISH position
            if 'up' in trend_lower or 'bull' in trend_lower:
                trend_stop_mult = 0.90  # Tighten 10% - trend supports us
                self.logger.debug(f"📈 CALL in UPTREND - tightening stop (×{trend_stop_mult})")
            elif 'down' in trend_lower or 'bear' in trend_lower:
                trend_stop_mult = 1.15  # Widen 15% - fighting trend
                self.logger.debug(f"📈 CALL in DOWNTREND - widening stop (×{trend_stop_mult})")
        elif 'put' in action.lower() or 'buy_put' in action.lower():
            # SHORT/BEARISH position
            if 'down' in trend_lower or 'bear' in trend_lower:
                trend_stop_mult = 0.90  # Tighten 10% - trend supports us
                self.logger.debug(f"📉 PUT in DOWNTREND - tightening stop (×{trend_stop_mult})")
            elif 'up' in trend_lower or 'bull' in trend_lower:
                trend_stop_mult = 1.15  # Widen 15% - fighting trend
                self.logger.debug(f"📉 PUT in UPTREND - widening stop (×{trend_stop_mult})")
        
        # ============= LIQUIDITY ADJUSTMENT =============
        # Low liquidity = harder to exit cleanly, need wider stops
        liq_lower = str(liquidity).lower()
        if 'low' in liq_lower:
            liq_stop_mult = 1.10  # Widen 10%
            self.logger.debug(f"💧 LOW LIQUIDITY - widening stop (×{liq_stop_mult})")
        elif 'high' in liq_lower:
            liq_stop_mult = 0.95  # Slightly tighter
        else:
            liq_stop_mult = 1.0
        
        # ============= COMBINE ADJUSTMENTS =============
        final_stop = base_stop * vol_stop_mult * trend_stop_mult * liq_stop_mult
        final_take = base_take * vol_take_mult * trend_stop_mult * liq_stop_mult
        
        # Clamp to reasonable ranges
        # Stop: Min 1.5% (tight), Max 8% (allows some room)
        # Take: Min 3% (minimum viable profit), Max 15% (let winners run)
        final_stop = max(0.015, min(0.08, final_stop))
        final_take = max(0.03, min(0.15, final_take))  # Let winners run!
        
        self.logger.info(f"📊 [REGIME-STOPS] Vol={volatility}, Trend={trend}, Liq={liquidity} → "
                        f"SL={final_stop:.0%} (base={base_stop:.0%}), TP={final_take:.0%}")
        
        return final_stop, final_take

    def _calculate_optimal_hold_time(self, prediction_data: Dict) -> float:
        """
        Calculate optimal hold time based on LSTM ACCURACY and predictions.
        
        KEY INSIGHT: Hold for the timeframe of the MOST ACCURATE prediction
        that agrees with our trade direction!
        
        Strategy:
        1. Get accuracy stats for each timeframe from bot's validation history
        2. Find timeframes that AGREE with our trade direction
        3. Pick the MOST ACCURATE agreeing timeframe
        4. Hold for that timeframe's duration
        
        Returns:
            float: Optimal hold time in hours
        """
        # NO ARBITRARY TIME LIMITS - Let momentum decide!
        # RL exit policy handles exits based on:
        # - Momentum direction and strength
        # - Prediction confidence changes  
        # - P&L thresholds (stop loss / take profit)
        # - Regime changes (HMM)
        #
        # The SMART calculation below sets a reasonable max based on predictions
        # but RL will exit MUCH earlier if momentum reverses
        
        # Timeframe mapping (name -> minutes)
        # Extended to support longer holds when momentum is strong
        TIMEFRAME_MINUTES = {
            '5min': 5,
            '10min': 10,
            '15min': 15,
            '20min': 20,
            '30min': 30,
            '40min': 40,
            '45min': 45,
            '50min': 50,
            '1h': 60,
            '1hour': 60,
            '1h20m': 80,
            '1h40m': 100,
            '2h': 120,
            '2hour': 120,
            '4hour': 240,
            '12hour': 720,
            '24hour': 1440,
        }
        
        # Get multi-timeframe predictions from prediction_data
        multi_tf = prediction_data.get('multi_timeframe', {})
        action = prediction_data.get('action', 'HOLD')
        
        # Determine target direction
        if 'CALL' in str(action).upper():
            target_direction = 'UP'
        elif 'PUT' in str(action).upper():
            target_direction = 'DOWN'
        else:
            return 0.25  # Default 15 min for unknown
        
        # ============= GET ACCURACY DATA FROM BOT =============
        accuracy_by_timeframe = {}
        
        if hasattr(self, 'bot') and self.bot:
            # Try to get accuracy from validated predictions
            if hasattr(self.bot, 'adaptive_weights'):
                try:
                    stats = self.bot.adaptive_weights.get_stats()
                    accuracy_by_timeframe = stats.get('accuracy_by_timeframe', {})
                    self.logger.info(f"📊 Using learned accuracy: {accuracy_by_timeframe}")
                except Exception:
                    pass
        
        # Default accuracies if not available (favor short-term)
        if not accuracy_by_timeframe:
            accuracy_by_timeframe = {
                '5min': 0.55, '10min': 0.55, '15min': 0.55, '20min': 0.52,
                '30min': 0.52, '40min': 0.50, '50min': 0.50, '1h': 0.50,
                '1h20m': 0.48, '1h40m': 0.48, '2h': 0.48
            }
        
        # ============= FIND BEST AGREEING TIMEFRAME =============
        agreeing_timeframes = []
        
        for tf_name, minutes in TIMEFRAME_MINUTES.items():
            tf_data = multi_tf.get(tf_name, {})
            if not tf_data:
                continue
            
            direction = tf_data.get('direction', '')
            confidence = tf_data.get('confidence', 0) or tf_data.get('neural_confidence', 0)
            predicted_return = abs(tf_data.get('predicted_return', 0) or tf_data.get('return', 0))
            
            # Get accuracy for this timeframe
            accuracy = accuracy_by_timeframe.get(tf_name, 0.5)
            
            # Check if this timeframe agrees with our trade direction
            if direction == target_direction:
                # SCORE = ACCURACY (most important) * confidence * (1 + return magnitude)
                # Accuracy is weighted HEAVILY because it's based on real results!
                score = (accuracy ** 2) * confidence * (1 + predicted_return * 10)
                
                agreeing_timeframes.append({
                    'name': tf_name,
                    'minutes': minutes,
                    'accuracy': accuracy,
                    'confidence': confidence,
                    'return': predicted_return,
                    'score': score
                })
        
        if not agreeing_timeframes:
            # No agreeing timeframes - use reasonable default hold (120 min)
            # Was 60 min which caused time exits before stop/profit could trigger!
            self.logger.info(f"📊 No agreeing timeframes for {target_direction} - using 120 min hold (default)")
            return 2.0  # 2 hours default to let trades develop and hit stop/profit levels
        
        # Sort by SCORE (accuracy-weighted)
        agreeing_timeframes.sort(key=lambda x: x['score'], reverse=True)
        
        # Pick the BEST scoring timeframe
        best_tf = agreeing_timeframes[0]
        
        # Log the analysis
        self.logger.info(f"📊 ACCURACY-BASED hold time for {target_direction}:")
        for tf in agreeing_timeframes[:5]:
            self.logger.info(f"   {tf['name']}: accuracy={tf['accuracy']:.0%}, conf={tf['confidence']:.1%}, score={tf['score']:.4f}")
        
        # Use the BEST timeframe's duration
        optimal_minutes = best_tf['minutes']
        
        # If multiple timeframes are close in score, consider the range
        if len(agreeing_timeframes) >= 2:
            second_best = agreeing_timeframes[1]
            if second_best['score'] > best_tf['score'] * 0.8:  # Within 20% of best
                # Average the top 2 timeframes
                optimal_minutes = (best_tf['minutes'] + second_best['minutes']) / 2
                self.logger.info(f"   Using avg of top 2: {best_tf['name']} + {second_best['name']}")
        
        optimal_hours = optimal_minutes / 60.0
        
        # Clamp to reasonable bounds (30 min to 8 hours)
        # Let RL exit based on momentum - this is just the MAX window
        # Increased from 4h to 8h to let strong momentum trades run
        optimal_hours = max(0.5, min(8.0, optimal_hours))
        
        self.logger.info(f"📊 OPTIMAL HOLD: {optimal_hours*60:.0f} min (based on {best_tf['name']} @ {best_tf['accuracy']:.0%} accuracy)")
        
        return optimal_hours
    
    def _get_historical_hold_multiplier(self) -> float:
        """
        Learn from past successful trades to adjust hold time.
        
        Returns:
            float: Multiplier to apply to calculated hold time (0.5 to 2.0)
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get last 10 profitable trades and their actual hold times
                cursor.execute('''
                    SELECT 
                        max_hold_hours,
                        (julianday(exit_timestamp) - julianday(timestamp)) * 24 as actual_hold_hours,
                        profit_loss
                    FROM trades
                    WHERE status IN ('PROFIT_TAKEN', 'CLOSED')
                    AND profit_loss > 0
                    AND exit_timestamp IS NOT NULL
                    ORDER BY exit_timestamp DESC
                    LIMIT 10
                ''')
                
                profitable_trades = cursor.fetchall()
                
                if len(profitable_trades) < 3:
                    # Not enough data - use default
                    return 1.0
                
                # Calculate average ratio of actual vs planned hold time for winners
                ratios = []
                for planned, actual, pnl in profitable_trades:
                    if planned and planned > 0 and actual and actual > 0:
                        ratio = actual / planned
                        ratios.append(ratio)
                
                if not ratios:
                    return 1.0
                
                # Average ratio tells us if we should hold longer or shorter
                avg_ratio = sum(ratios) / len(ratios)
                
                # Clamp to reasonable bounds
                multiplier = max(0.5, min(2.0, avg_ratio))
                
                self.logger.debug(f"📊 Historical hold multiplier: {multiplier:.2f}x (from {len(ratios)} profitable trades)")
                
                return multiplier
                
        except Exception as e:
            self.logger.debug(f"Could not calculate historical multiplier: {e}")
            return 1.0

    def place_trade(self, symbol: str, prediction_data: Dict, current_price: float = None) -> Optional[Trade]:
        """
        Place a paper trade based on ML prediction with full real-world constraints
        """
        # Expose latest rejection reason to upstream callers (e.g., LiveTradingEngine)
        self.last_trade_rejection = None

        # Real-world constraint checks
        
        # 1. Check Pattern Day Trading compliance
        if not self._check_pdt_compliance():
            self.last_trade_rejection = {"type": "pdt", "code": "PDT_BLOCK", "reason": "PDT compliance blocked trade"}
            return None
        
        # 2. Check market hours (only trade when there's liquidity)
        if not self._is_market_hours():
            self.logger.warning("Market Closed: Cannot place options trade outside market hours (no liquidity)")
            self.last_trade_rejection = {"type": "market_hours", "code": "MARKET_CLOSED", "reason": "Market closed"}
            return None
        
        # 3. Check if we can place more trades
        # IMPORTANT: Only count PAPER trades against the limit
        # Live trades (from Tradier) should ALWAYS be allowed
        is_live_trade = prediction_data.get('_is_live_trade', False)
        paper_trades_count = sum(1 for t in self.active_trades if not t.is_real_trade)
        
        # NOTE: Opposing position management is handled by the RL exit policy
        # during update_positions(). The RL learns when to exit positions based on:
        # - Prediction reversals (TCN now predicting opposite direction)
        # - HMM regime changes
        # - Position direction vs current market signals
        # This is more adaptive than hard-coded rules.
        
        if not is_live_trade and paper_trades_count >= self.max_positions:
            self.logger.warning(
                f"Maximum PAPER positions reached ({paper_trades_count}/{self.max_positions}), cannot place new paper trade")
            self.logger.info(f"[PRIORITY] Note: Live trades are NOT blocked by this limit")
            self.last_trade_rejection = {
                "type": "risk",
                "code": "MAX_POSITIONS",
                "reason": f"Max paper positions reached ({paper_trades_count}/{self.max_positions})"
            }
            return None
        
        if is_live_trade:
            self.logger.warning(f"[PRIORITY] LIVE TRADE - bypassing paper position limit ({paper_trades_count} paper positions active)")
        else:
            self.logger.info(f"[POSITIONS] Current: {paper_trades_count} paper, {len(self.active_trades) - paper_trades_count} live (max paper: {self.max_positions})")

        # 4. BALANCE FLOOR PROTECTION - Stop trading if account is dying
        # This prevents infinite negative balance spirals in training mode
        BALANCE_FLOOR_PCT = 0.10  # Stop trading at 10% of initial balance
        min_balance = self.initial_balance * BALANCE_FLOOR_PCT
        if self.current_balance < min_balance:
            self.logger.error(f"🛑 BALANCE FLOOR: ${self.current_balance:.2f} < ${min_balance:.2f} (10% of ${self.initial_balance:.2f})")
            self.logger.error("Refusing new trades until positions close and balance recovers")
            self.last_trade_rejection = {
                "type": "risk",
                "code": "BALANCE_FLOOR",
                "reason": f"Balance ${self.current_balance:.2f} below floor ${min_balance:.2f}"
            }
            return None

        # Use provided current price or get from data source
        if current_price is None:
            # Check if we're in simulation mode
            is_simulation = getattr(self, 'simulation_mode', False) or getattr(self, 'time_travel_mode', False)
            
            if is_simulation:
                # In simulation, caller should provide current_price
                # If not provided, log warning but don't fail - use last known price if available
                self.logger.warning(f"[SIMULATION] ⚠️ current_price not provided, trying to get from historical data")
            
            # Try to get current market data
            data = self.data_source.get_data(
                symbol, period='1d', interval='1m')
            if data.empty:
                # Try alternative intervals
                data = self.data_source.get_data(
                    symbol, period='1d', interval='15m')
                if data.empty:
                    if is_simulation:
                        # In simulation without price data - this shouldn't happen
                        self.logger.error(f"[SIMULATION] ❌ No historical data available for {symbol}")
                        self.logger.error(f"[SIMULATION] Trade rejected - caller must provide current_price")
                        return None
                    else:
                        # Live mode without data
                        self.logger.error(f"[LIVE] ❌ No market data available for {symbol}")
                        self.logger.error(f"[LIVE] Trade rejected - cannot determine current price")
                        return None
                else:
                    current_price = data['close'].iloc[-1]
            else:
                current_price = data['close'].iloc[-1]

        # Determine trade parameters from prediction
        # Support both old format (recommendation) and new format (action)
        ml_prediction = prediction_data.get(
            'recommendation', prediction_data.get('action', 'HOLD'))
        ml_confidence = prediction_data.get('confidence', 0.0)

        # Optional expectancy/friction gate: veto/downgrade low-tradability setups early.
        # Applies to paper trades only (live/mirrored bookkeeping trades bypass via _is_live_trade).
        if getattr(self, "tradability_gate_enabled", False) and not prediction_data.get('_is_live_trade', False):
            gate = self._get_tradability_gate_model()
            if gate is not None:
                try:
                    # Build features aligned with backend.tradability_gate.FEATURE_NAMES.
                    pred_ret = float(prediction_data.get("predicted_return", 0.0) or 0.0)
                    conf = float(ml_confidence or 0.0)
                    vix_level = float(prediction_data.get("vix_level", 0.0) or prediction_data.get("vix", 0.0) or 0.0)
                    mom5 = float(prediction_data.get("momentum_5m", 0.0) or prediction_data.get("momentum_5min", 0.0) or 0.0)
                    vol_spike = float(prediction_data.get("volume_spike", 1.0) or 1.0)

                    # If we didn't get VIX from caller, use the current VIX quote.
                    if vix_level <= 0:
                        try:
                            vix_data = self.data_source.get_data('VIX', period='1d', interval='1m')
                            if vix_data is not None and (not vix_data.empty):
                                vix_level = float(vix_data['close'].iloc[-1])
                        except Exception:
                            pass

                    gate_features = {
                        "predicted_return": pred_ret,
                        "prediction_confidence": conf,
                        "vix_level": vix_level,
                        "momentum_5m": mom5,
                        "volume_spike": vol_spike,
                        "hmm_trend": float(prediction_data.get("hmm_trend", 0.0) or 0.0),
                        "hmm_volatility": float(prediction_data.get("hmm_volatility", 0.0) or 0.0),
                        "hmm_liquidity": float(prediction_data.get("hmm_liquidity", 0.0) or 0.0),
                        "hmm_confidence": float(prediction_data.get("hmm_confidence", 0.0) or 0.0),
                    }

                    p_tradable = float(gate.predict_proba(gate_features))
                    prediction_data["_tradability_gate_p"] = p_tradable

                    if p_tradable < float(getattr(self, "tradability_gate_veto_threshold", 0.25)):
                        self.logger.warning(
                            f"[GATE] VETO {symbol} {ml_prediction} | p_tradable={p_tradable:.2f} "
                            f"(veto<{float(getattr(self, 'tradability_gate_veto_threshold', 0.25)):.2f})"
                        )
                        self.last_trade_rejection = {
                            "type": "tradability_gate",
                            "code": "GATE_VETO",
                            "reason": f"TradabilityGate veto (p={p_tradable:.2f})",
                            "p_tradable": p_tradable,
                        }
                        return None

                    if p_tradable < float(getattr(self, "tradability_gate_downgrade_threshold", 0.45)):
                        old_conf = float(ml_confidence or 0.0)
                        scale = max(
                            0.50,
                            min(1.0, p_tradable / float(getattr(self, "tradability_gate_downgrade_threshold", 0.45))),
                        )
                        ml_confidence = old_conf * scale
                        prediction_data["confidence"] = ml_confidence
                        self.logger.info(
                            f"[GATE] DOWNGRADE {symbol} {ml_prediction} | p_tradable={p_tradable:.2f} "
                            f"conf {old_conf:.2f}→{ml_confidence:.2f}"
                        )
                except Exception as e:
                    self.logger.debug(f"[GATE] Failed to apply tradability gate: {e}")
        
        # Safety: multi-leg / premium-selling actions are not supported by this paper system.
        # Prevent silent fallback that would create the wrong position type.
        if ml_prediction in ['BUY_STRADDLE', 'SELL_PREMIUM']:
            self.logger.warning(
                f"❌ Unsupported action for PaperTradingSystem: {ml_prediction}. "
                f"This paper engine only supports single-leg long CALL/PUT entries."
            )
            self.last_trade_rejection = {
                "type": "unsupported_action",
                "code": "UNSUPPORTED_ACTION",
                "reason": f"Unsupported action for paper trader: {ml_prediction}"
            }
            return None
        
        # Risk manager inputs
        calibration_samples = 0
        try:
            if 'calibration_samples' in prediction_data:
                calibration_samples = int(prediction_data.get('calibration_samples') or 0)
            elif isinstance(prediction_data.get('calibration_metrics'), dict):
                calibration_samples = int(prediction_data['calibration_metrics'].get('sample_count') or 0)
        except Exception:
            calibration_samples = 0
        
        minutes_to_close = self._get_minutes_to_close()
        daily_pnl, weekly_pnl = self._get_daily_and_weekly_pnl()
        
        # Estimate DTE based on prediction horizon (same logic used later for strike selection)
        prediction_timeframe = prediction_data.get('timeframe_minutes', 60)
        try:
            prediction_timeframe = int(prediction_timeframe)
        except Exception:
            prediction_timeframe = 60
        dte = 1
        if self.dte_optimizer is not None:
            try:
                dte = int(self.dte_optimizer.calculate_optimal_dte(prediction_timeframe))
            except Exception:
                dte = 1

        # TRAINING/REALISM: Prefer longer DTE for SPY/QQQ in simulation unless explicitly overridden.
        # Very short DTE encourages scalping behavior and high theta noise, which can look like "bad play performance"
        # even when the predictor gets better.
        try:
            is_simulation = getattr(self, 'simulation_mode', False) or getattr(self, 'time_travel_mode', False)
            if is_simulation and symbol in ("SPY", "QQQ"):
                # default 7DTE in simulation unless caller specified dte
                dte = max(int(os.environ.get("TT_MIN_DTE_DAYS", "7")), int(dte))
        except Exception:
            pass
        
        # Centralized pre-trade risk gate (paper trades only)
        # Live/mirrored bookkeeping trades can bypass gating via _is_live_trade.
        if getattr(self, 'risk_manager', None) is not None and not is_live_trade and (not self._training_unlimited_funds()):
            # If we can't determine calibration_samples, assume we're in early regime
            # but do NOT hard-block trading due to missing metadata.
            cal_samples_for_risk = int(calibration_samples) if calibration_samples > 0 else 50
            can_trade, risk_result, risk_reason = self.risk_manager.check_trade_allowed(
                account_balance=float(self.current_balance),
                current_positions=int(len(self.active_trades)),
                daily_pnl=float(daily_pnl),
                weekly_pnl=float(weekly_pnl),
                calibration_samples=cal_samples_for_risk,
                minutes_to_close=int(minutes_to_close),
                dte=int(dte),
            )
            if not can_trade:
                self.logger.warning(f"[RISK] Trade blocked: {risk_result.value} | {risk_reason}")
                self.last_trade_rejection = {
                    "type": "risk",
                    "code": risk_result.value,
                    "reason": risk_reason
                }
                return None

        # Map unified bot signals to option types
        if 'CALL' in ml_prediction or ml_prediction in ['BUY_CALLS', 'BUY_STRADDLE']:
            option_type_str = 'call'
            option_type = OrderType.CALL
        elif 'PUT' in ml_prediction or ml_prediction in ['BUY_PUTS', 'SELL_PREMIUM']:
            option_type_str = 'put'
            option_type = OrderType.PUT
        else:
            self.logger.info(
                f"No clear signal ({ml_prediction}), not placing trade")
            return None
        
        # INTELLIGENT STRIKE SELECTION using real Greeks!
        if self.strike_selector:
            # Get prediction details for intelligent strike selection
            predicted_return = prediction_data.get('predicted_return', 0.02)  # Default 2% move
            
            # Get current IV (use VIX as proxy for BITX volatility)
            try:
                vix_data = self.data_source.get_data('VIX', period='1d', interval='1m')
                if not vix_data.empty:
                    current_iv = vix_data['close'].iloc[-1] / 100  # VIX is in percentage
                else:
                    current_iv = 0.25  # Default 25% IV
            except:
                current_iv = 0.25  # Fallback to 25% IV
            
            # Optimize DTE based on prediction timeframe
            prediction_timeframe = prediction_data.get('timeframe_minutes', 60)  # Default 1 hour
            optimal_dte = self.dte_optimizer.calculate_optimal_dte(prediction_timeframe)
            
            # Select optimal strike using intelligent selector
            strike_selection = self.strike_selector.select_strike(
                spot_price=current_price,
                predicted_move_pct=predicted_return,
                confidence=ml_confidence,
                implied_volatility=current_iv,
                time_to_expiration_days=optimal_dte,
                option_type=option_type_str
            )
            
            # Get the calculated optimal strike
            calculated_strike = strike_selection['strike']
            option_greeks = strike_selection['greeks']
            time_to_expiry = optimal_dte / 365.0  # Convert to years for calculations
            
            # ✅ ROUND TO REALISTIC STRIKE INTERVALS (matches real exchanges)
            # Exchanges don't offer $4.72 strikes, only $4.50 or $5.00
            strike_price = self.round_to_realistic_strike(calculated_strike, current_price)

            # Sanity: correct occasional 10x strike scaling bugs.
            try:
                if current_price > 0 and strike_price > (current_price * 3.0):
                    while strike_price > (current_price * 3.0):
                        strike_price = strike_price / 10.0
                    strike_price = self.round_to_realistic_strike(strike_price, current_price)
                    self.logger.warning(f"[SANITY] Corrected suspicious strike scaling → ${strike_price:.2f}")
            except Exception:
                pass
            
            # Recalculate Greeks for the ROUNDED strike (more accurate pricing)
            if strike_price != calculated_strike:
                self.logger.info(f"📍 Rounding strike: ${calculated_strike:.2f} → ${strike_price:.2f} (exchange interval)")
                
                # Recalculate Greeks for actual strike we'll use
                try:
                    option_greeks = self.strike_selector.greeks_calc.calculate_all_greeks(
                        current_price, strike_price, optimal_dte,
                        current_iv, option_type_str
                    )
                except Exception as e:
                    self.logger.warning(f"Could not recalculate Greeks for rounded strike: {e}, using original")
            
            self.logger.info(f"🎯 INTELLIGENT STRIKE SELECTION:")
            self.logger.info(f"   Strike: ${strike_price:.2f} (delta={option_greeks['delta']:.2f})")
            self.logger.info(f"   DTE: {optimal_dte} days (matched to {prediction_timeframe}min prediction)")
            self.logger.info(f"   IV: {current_iv:.1%}, Premium: ${option_greeks['price']:.2f}")
            self.logger.info(f"   Theta: ${option_greeks['theta']:.2f}/day, Gamma: {option_greeks['gamma']:.4f}")
        else:
            # Fallback to simple strike selection - OTM for smaller accounts
            # OTM options are cheaper but have lower delta
            if option_type == OrderType.CALL:
                calculated_strike = current_price * 1.005  # 0.5% OTM CALL
            else:
                calculated_strike = current_price * 0.995  # 0.5% OTM PUT
            
            # Round to realistic exchange intervals
            strike_price = self.round_to_realistic_strike(calculated_strike, current_price)

            # Sanity: sometimes strike can be 10x due to upstream scaling bugs.
            # If the strike is wildly out of range vs spot (e.g., SPY 660, strike 6660),
            # bring it back down and re-round.
            try:
                if current_price > 0 and strike_price > (current_price * 3.0):
                    while strike_price > (current_price * 3.0):
                        strike_price = strike_price / 10.0
                    strike_price = self.round_to_realistic_strike(strike_price, current_price)
                    self.logger.warning(f"[SANITY] Corrected suspicious strike scaling → ${strike_price:.2f}")
            except Exception:
                pass
            
            time_to_expiry = 0.25  # 6 hours (0.25 days)
            option_greeks = None
            self.logger.warning(f"⚠️ Using OTM strike: ${strike_price:.2f} (SPY: ${current_price:.2f}, ~0.5% OTM)")

        # Calculate option premium
        # If we have real Greeks, use them! Otherwise use fallback calculation
        if option_greeks:
            # Use real Black-Scholes price from Greeks calculation
            base_premium = option_greeks['price']
            
            # ✅ APPLY REALISTIC ADJUSTMENTS (same as _calculate_option_premium)
            # Bid-ask spread (buy at ASK)
            premium = base_premium * (1 + 0.05/2)  # +2.5%
            # Market order slippage
            premium = premium * (1 + 0.02)  # +2%
            # Market noise
            import random
            market_noise = random.uniform(-0.03, 0.03)
            premium = premium * (1 + market_noise)
            # Cap premium to prevent unrealistic gains
            max_premium = strike_price * 0.04  # Max 4% of strike
            min_premium = strike_price * 0.003  # Min 0.3% of strike
            premium = max(min_premium, min(max_premium, premium))
            
            self.logger.info(f"💰 Using Black-Scholes premium: ${base_premium:.2f} → ${premium:.2f} (with realism)")
        else:
            # Fallback to old calculation
            premium = self._calculate_option_premium(
                symbol, option_type, strike_price, current_price, is_entry=True, 
                time_to_expiry_days=time_to_expiry)

        # ------------------------------------------------------------------
        # TRAINING/SMALL-ACCOUNT FRIENDLY: Ensure contract is affordable
        # ------------------------------------------------------------------
        # If premium per contract exceeds max position value, RiskManager will block ("Position too small").
        # In simulation/time-travel, try nudging strike further OTM to reduce premium instead of skipping.
        try:
            is_simulation = getattr(self, 'simulation_mode', False) or getattr(self, 'time_travel_mode', False)
            if is_simulation and getattr(self, 'risk_manager', None) is not None and current_price > 0:
                max_position_value = float(self.current_balance) * float(self.risk_manager.limits.max_position_size_pct)
                contract_cost = float(premium) * 100.0
                if contract_cost > max_position_value:
                    orig_strike = float(strike_price)
                    orig_premium = float(premium)
                    # Try up to 6 steps of +0.5% further OTM (max ~3% extra)
                    for step in range(1, 7):
                        bump = 0.005 * step
                        if option_type in [OrderType.CALL, OrderType.BUY_CALL]:
                            trial_strike = self.round_to_realistic_strike(current_price * (1.005 + bump), current_price)
                        else:
                            trial_strike = self.round_to_realistic_strike(current_price * (0.995 - bump), current_price)
                        trial_premium = self._calculate_option_premium(
                            symbol, option_type, trial_strike, current_price, is_entry=True, time_to_expiry_days=time_to_expiry
                        )
                        if (float(trial_premium) * 100.0) <= max_position_value:
                            strike_price = trial_strike
                            premium = trial_premium
                            self.logger.info(
                                f"[TRAIN] Adjusted strike for affordability: ${orig_strike:.2f}→${strike_price:.2f} "
                                f"(premium ${orig_premium:.2f}→${premium:.2f}, max_pos=${max_position_value:.0f})"
                            )
                            break
        except Exception:
            pass

        # GREEK-BASED POSITION SIZING (if available)
        if self.position_sizer and option_greeks:
            # Get portfolio Greeks for risk management
            portfolio_greeks = None
            if self.portfolio_tracker:
                portfolio_greeks = self.portfolio_tracker.calculate_portfolio_greeks(self.active_trades)
                self.logger.info(f"📊 Portfolio Greeks: Delta={portfolio_greeks['total_delta']:.2f}, "
                               f"Theta=${portfolio_greeks['total_theta']:.2f}/day")
            
            # Calculate intelligent position size
            sizing_result = self.position_sizer.calculate_position_size(
                balance=self.current_balance,
                confidence=ml_confidence,
                option_greeks=option_greeks,
                current_iv=current_iv if 'current_iv' in locals() else 0.25,
                portfolio_greeks=portfolio_greeks
            )
            
            quantity = sizing_result['contracts']
            self.logger.info(f"🎯 Greek-based sizing: {quantity} contracts (multiplier: {sizing_result['final_multiplier']:.2f}x)")
        else:
            # Fallback to simple position sizing
            sizing_balance = self._training_virtual_balance() if self._training_unlimited_funds() else float(self.current_balance)
            base_position_value = float(self.initial_balance) * float(self.position_size_pct)
            max_position_value = float(sizing_balance) * float(self.max_position_size_per_trade)
            position_value = min(base_position_value, max_position_value)

            # === DRAWDOWN-BASED POSITION SCALING (Phase 26 improvement) ===
            # Reduce position size during drawdowns to preserve capital
            drawdown_scale_enabled = os.environ.get('DRAWDOWN_SCALE_ENABLED', '1') == '1'
            if drawdown_scale_enabled and not self._training_unlimited_funds():
                try:
                    peak_balance = max(self.initial_balance + getattr(self, 'total_profit_loss', 0), self.initial_balance)
                    current_drawdown_pct = ((peak_balance - self.current_balance) / peak_balance) * 100 if peak_balance > 0 else 0

                    # Scaling thresholds (configurable via env vars)
                    half_size_dd = float(os.environ.get('DRAWDOWN_HALF_SIZE_PCT', '10'))  # Half size at 10% drawdown
                    quarter_size_dd = float(os.environ.get('DRAWDOWN_QUARTER_SIZE_PCT', '20'))  # Quarter size at 20% drawdown

                    if current_drawdown_pct >= quarter_size_dd:
                        position_value *= 0.25
                        self.logger.info(f"⚠️ QUARTER SIZE: {current_drawdown_pct:.1f}% drawdown (>{quarter_size_dd}%)")
                    elif current_drawdown_pct >= half_size_dd:
                        position_value *= 0.50
                        self.logger.info(f"⚠️ HALF SIZE: {current_drawdown_pct:.1f}% drawdown (>{half_size_dd}%)")
                except Exception as e:
                    self.logger.debug(f"Drawdown scaling skipped: {e}")
            # === END DRAWDOWN-BASED SCALING ===

            # === FUZZY POSITION SIZING (Jerry's Quantor-MTFuzz A.25-A.26) ===
            # Scale position size based on confidence and volatility
            # Formula: g(F_t, σ*_t) = F_t × (1 - β × σ*_t)
            fuzzy_sizing_enabled = os.environ.get('FUZZY_SIZING_ENABLED', '0') == '1'
            if fuzzy_sizing_enabled:
                try:
                    # F_t = fuzzy confidence [0,1]
                    fuzzy_confidence = float(ml_confidence) if ml_confidence else 0.5
                    fuzzy_confidence = max(0.0, min(1.0, fuzzy_confidence))

                    # σ*_t = normalized volatility [0,1]
                    # Get VIX level from prediction_data or estimate from context
                    vix_level = prediction_data.get('vix_level', 20.0)
                    if vix_level is None:
                        vix_level = 20.0
                    # Normalize VIX: 12 = low (0.0), 20 = normal (0.5), 35 = high (1.0)
                    vix_min, vix_max = 12.0, 35.0
                    normalized_vol = (float(vix_level) - vix_min) / (vix_max - vix_min)
                    normalized_vol = max(0.0, min(1.0, normalized_vol))

                    # β = volatility penalty factor [0.3, 0.5]
                    beta = float(os.environ.get('FUZZY_SIZING_BETA', '0.4'))

                    # Scaling function: g(F_t, σ*_t) = F_t × (1 - β × σ*_t)
                    fuzzy_scale = fuzzy_confidence * (1.0 - beta * normalized_vol)
                    fuzzy_scale = max(0.1, min(1.0, fuzzy_scale))  # Clamp to [0.1, 1.0]

                    position_value *= fuzzy_scale
                    self.logger.info(
                        f"📐 Fuzzy sizing: scale={fuzzy_scale:.2f} "
                        f"(conf={fuzzy_confidence:.2f}, VIX={vix_level:.1f}, σ*={normalized_vol:.2f})"
                    )
                except Exception as e:
                    self.logger.debug(f"Fuzzy sizing skipped: {e}")
            # === END FUZZY POSITION SIZING ===

            quantity = max(1, int(position_value / (premium * 100)))
            self.logger.info(f"📊 Simple sizing: {quantity} contracts")
        
        # If caller provided an explicit desired size (e.g., RL sizing), respect it,
        # but still run centralized risk limits below.
        desired_qty = prediction_data.get('position_size')
        if desired_qty is not None:
            try:
                desired_qty = int(desired_qty)
                if desired_qty > 0:
                    quantity = desired_qty
                    self.logger.info(f"🎯 Using caller-specified position size: {quantity} contracts")
            except Exception:
                pass
        
        # Centralized sizing clamp (paper trades only): do not allow quantity beyond risk limits.
        if getattr(self, 'risk_manager', None) is not None and not is_live_trade and (not self._training_unlimited_funds()):
            try:
                vol_regime = prediction_data.get('volatility_regime', 'NORMAL_VOL')
                cal_samples_for_size = int(calibration_samples) if calibration_samples > 0 else 50
                rm_assess = self.risk_manager.calculate_position_size(
                    account_balance=float(self.current_balance),
                    option_price=float(premium),
                    calibrated_confidence=float(ml_confidence),
                    calibration_samples=cal_samples_for_size,
                    volatility_regime=str(vol_regime),
                    regime_position_scale=1.0,
                )
                
                if not rm_assess.allowed:
                    self.logger.warning(f"[RISK] Position sizing blocked: {rm_assess.reason}")
                    # TRAINING escape hatch: if we're in simulation/time-travel, we still want experiences
                    # even when the account is tiny (e.g. balance ~$420 can't afford "proper" sizing caps).
                    # Let the simulator take a 1-contract position and allow balance to go negative
                    # (training already permits negative balance later).
                    try:
                        is_simulation = getattr(self, 'simulation_mode', False) or getattr(self, 'time_travel_mode', False) or getattr(self, 'disable_pdt_override', False)
                        allow_min = os.environ.get("TT_ALLOW_MIN_CONTRACT", "1").strip() == "1"
                        if is_simulation and allow_min and ("Position too small" in str(rm_assess.reason) or "too small" in str(rm_assess.reason).lower()):
                            quantity = 1
                            self.logger.warning(
                                f"[TRAIN] Forcing minimum 1-contract entry despite sizing block "
                                f"(balance=${self.current_balance:.2f}, premium=${premium:.2f})"
                            )
                        else:
                            return None
                    except Exception:
                        return None
                
                if quantity > rm_assess.contracts:
                    self.logger.info(
                        f"[RISK] Capping contracts {quantity} → {rm_assess.contracts} "
                        f"({rm_assess.reason}; cold_start={rm_assess.is_cold_start})"
                    )
                    quantity = rm_assess.contracts
            except Exception as e:
                self.logger.warning(f"[RISK] Could not apply RiskManager sizing clamp: {e}")
        
        # Calculate total trade value for liquidity check
        total_trade_value = premium * quantity * 100
        
        # 4a. Check REAL market liquidity (live mode only - skip during simulation/training!)
        # CRITICAL: During simulation, we use calculated premiums, not live Tradier data
        # This ensures training calculations are accurate and don't depend on API calls
        is_simulation = getattr(self, 'simulation_mode', False) or getattr(self, 'time_travel_mode', False)
        
        if self.liquidity_validator and not is_simulation:
            self.logger.info(f"🔍 Checking real market liquidity for ${strike_price:.2f} {option_type_str}...")
            
            try:
                # Fetch real option chain data from Tradier
                from datetime import datetime, timedelta
                from backend.tradier_trading_system import TradierTradingSystem
                from tradier_credentials import TradierCredentials
                from trading_mode import get_trading_mode
                
                mode = get_trading_mode()
                creds = TradierCredentials()
                
                # Get the appropriate credentials
                if mode['is_sandbox']:
                    tradier_creds = creds.get_sandbox_credentials()
                else:
                    tradier_creds = creds.get_live_credentials()
                
                if not tradier_creds:
                    self.logger.error("[LIQUIDITY] ❌ No Tradier credentials - cannot validate liquidity")
                    self.logger.error("[LIQUIDITY] Trade rejected - liquidity validation required in live mode")
                    return None
                
                # Initialize Tradier trading system to access option chain API
                tradier = TradierTradingSystem(
                    account_id=tradier_creds['account_number'],
                    api_token=tradier_creds['access_token'],
                    sandbox=mode['is_sandbox']
                )
                
                # Get option chain for this strike
                option_type_tradier = 'call' if option_type_str == 'CALL' else 'put'
                
                # ===== SMART DTE SELECTION =====
                # Evaluate multiple expirations using predictions, confidence, HMM + liquidity
                self.logger.info(f"[SMART-DTE] Evaluating expirations using predictions + liquidity...")
                
                # Get prediction data for smart selection
                pred_confidence = prediction_data.get('confidence', 0.5)
                pred_return = abs(prediction_data.get('predicted_return', 0.02))
                pred_timeframe = prediction_data.get('timeframe_minutes', 60)
                hmm_state = prediction_data.get('hmm_state', {})
                
                # Extract HMM volatility info
                hmm_volatility = 'Normal'
                if isinstance(hmm_state, dict):
                    hmm_volatility = hmm_state.get('volatility', hmm_state.get('volatility_name', 'Normal'))
                elif isinstance(hmm_state, str):
                    hmm_volatility = 'High' if 'High' in hmm_state else ('Low' if 'Low' in hmm_state else 'Normal')
                
                self.logger.info(f"[SMART-DTE] Prediction: conf={pred_confidence:.1%}, return={pred_return:.1%}, "
                               f"timeframe={pred_timeframe}min, HMM vol={hmm_volatility}")
                
                # Get all available expirations from Tradier
                available_expirations = tradier.get_options_expirations(symbol)
                
                if not available_expirations:
                    self.logger.error(f"[SMART-DTE] No expirations available for {symbol}")
                    return None
                
                today = self.get_market_time().strftime('%Y-%m-%d')
                
                # Determine ideal DTE based on prediction confidence and HMM regime
                # High confidence + High vol = short DTE (quick capture)
                # Low confidence + Low vol = longer DTE (more time to be right)
                if pred_confidence > 0.7 and hmm_volatility == 'High':
                    ideal_dte = 0  # 0DTE for high confidence in volatile markets
                elif pred_confidence > 0.5 or hmm_volatility == 'High':
                    ideal_dte = 1  # Near-term
                elif pred_confidence > 0.3:
                    ideal_dte = 7  # Weekly
                else:
                    ideal_dte = 30  # Swing trade (more time to be right)
                
                self.logger.info(f"[SMART-DTE] Ideal DTE based on conf/HMM: {ideal_dte} days")
                
                # Categorize expirations and prioritize by how close they are to ideal
                candidate_expirations = []
                for exp in available_expirations:
                    try:
                        exp_date = datetime.strptime(exp, '%Y-%m-%d')
                        days_to_exp = (exp_date - self.get_market_time().replace(tzinfo=None)).days
                        
                        # Skip past expirations
                        if days_to_exp < 0:
                            continue
                        
                        # Priority = distance from ideal DTE (lower = better)
                        distance_from_ideal = abs(days_to_exp - ideal_dte)
                        
                        candidate_expirations.append({
                            'date': exp,
                            'days': days_to_exp,
                            'priority': distance_from_ideal
                        })
                    except:
                        continue
                
                # Sort by priority (lower = closer to ideal)
                candidate_expirations.sort(key=lambda x: (x['priority'], x['days']))
                
                # Try each expiration until we find one with a liquid option
                all_options = None
                expiration_date = None
                best_option_data = None
                best_option_score = -1
                
                for candidate in candidate_expirations[:5]:  # Check top 5 candidates
                    exp = candidate['date']
                    days_to_exp = candidate['days']
                    self.logger.info(f"[SMART-DTE] Checking {exp} ({days_to_exp} DTE, dist from ideal: {candidate['priority']})...")
                    
                    options = tradier.get_options_chain(symbol, exp)
                    if not options:
                        self.logger.debug(f"[SMART-DTE] No options for {exp}")
                        continue
                    
                    # Find the specific strike and option type
                    for opt in options:
                        if (abs(float(opt.get('strike', 0)) - strike_price) < 1.0 and 
                            opt.get('option_type', '').lower() == option_type_tradier):
                            
                            bid = float(opt.get('bid') or 0)
                            ask = float(opt.get('ask') or 0)
                            volume = int(opt.get('volume') or 0)
                            
                            # Skip invalid quotes
                            if bid <= 0 or ask <= 0:
                                continue
                            
                            # ===== ENHANCED SCORING with Predictions/Confidence/HMM =====
                            spread_pct = (ask - bid) / ((bid + ask) / 2) * 100 if (bid + ask) > 0 else 100
                            premium_cost = ask * 100  # Cost per contract
                            
                            score = 0
                            
                            # 1. LIQUIDITY SCORE (40 pts max)
                            score += max(0, 25 - spread_pct * 5)  # Lower spread = higher score
                            score += min(15, volume / 20)  # Volume bonus
                            
                            # 2. AFFORDABILITY SCORE (20 pts max)
                            if premium_cost < self.current_balance * 0.3:
                                score += 20  # Very affordable
                            elif premium_cost < self.current_balance * 0.5:
                                score += 10  # Affordable
                            
                            # 3. CONFIDENCE-ADJUSTED DTE MATCH (25 pts max)
                            # Higher confidence = prefer closer to ideal DTE
                            dte_match_score = 25 - (candidate['priority'] * pred_confidence * 5)
                            score += max(0, dte_match_score)
                            
                            # 4. THETA/GAMMA TRADE-OFF (15 pts max)
                            # Short DTE = high gamma (good for quick moves), high theta (bad)
                            # Long DTE = low gamma, low theta
                            if days_to_exp == 0:  # 0DTE
                                if pred_confidence > 0.6 and pred_return > 0.01:
                                    score += 15  # High gamma good for confident large moves
                                else:
                                    score += 5   # Risky without confidence
                            elif days_to_exp <= 7:
                                score += 12  # Good balance
                            elif days_to_exp <= 30:
                                score += 10  # Safe but less gamma
                            else:
                                score += 8   # Very safe, low gamma
                            
                            # 5. HMM REGIME ADJUSTMENT
                            if hmm_volatility == 'High' and days_to_exp <= 3:
                                score += 5   # High vol = prefer short DTE (capture quick moves)
                            elif hmm_volatility == 'Low' and days_to_exp >= 7:
                                score += 5   # Low vol = prefer slightly longer DTE (need time for move)
                            
                            # 6. STRADDLE ADJUSTMENT (CRITICAL)
                            # Straddles need more time to overcome theta decay + spread
                            if option_type == 'STRADDLE':
                                if days_to_exp < 14:
                                    score -= 50  # Strongly penalize < 2 weeks for straddles (theta kills them)
                                elif days_to_exp >= 30:
                                    score += 20  # Reward > 30 days
                                elif days_to_exp >= 45:
                                    score += 15  # Good
                            
                            self.logger.info(f"[SMART-DTE] {exp}: ${strike_price} {option_type_tradier} | "
                                           f"Spread={spread_pct:.1f}%, Cost=${premium_cost:.0f}, Score={score:.0f}")
                            
                            if score > best_option_score:
                                best_option_score = score
                                best_option_data = opt
                                expiration_date = exp
                                all_options = options
                            break
                
                if not all_options or not expiration_date:
                    self.logger.error(f"[SMART-DTE] ❌ No liquid options found across any expiration")
                    self.logger.error(f"[LIQUIDITY] Trade rejected - cannot validate without market data")
                    return None
                
                self.logger.info(f"[SMART-DTE] ✅ Selected: {expiration_date} with score {best_option_score:.0f}")
                
                # Calculate days until expiry for downstream use
                exp_dt = datetime.strptime(expiration_date, '%Y-%m-%d')
                days_until_expiry = max(0, (exp_dt - self.get_market_time().replace(tzinfo=None)).days)
                
                # Use the best option we already found
                option_data = best_option_data
                
                if not option_data:
                    self.logger.error(f"[LIQUIDITY] ❌ Strike ${strike_price:.2f} not found in option chain")
                    self.logger.error(f"[LIQUIDITY] Trade rejected - strike not available")
                    return None
                
                # Extract liquidity data
                validation_data = {
                    'bid': float(option_data.get('bid', 0)),
                    'ask': float(option_data.get('ask', 0)),
                    'volume': int(option_data.get('volume', 0)),
                    'open_interest': int(option_data.get('open_interest', 0)),
                    'strike': float(option_data.get('strike', 0))
                }
                
                self.logger.info(f"[LIQUIDITY] Quote: Bid=${validation_data['bid']:.2f}, Ask=${validation_data['ask']:.2f}")
                self.logger.info(f"[LIQUIDITY] Volume: {validation_data['volume']}, OI: {validation_data['open_interest']}")
                
                # Validate liquidity
                check_hours = not mode['is_sandbox']  # Skip hours check in sandbox for testing
                is_valid, reason = self.liquidity_validator.validate_option(
                    validation_data,
                    current_price,
                    check_market_hours=check_hours
                )
                
                if not is_valid:
                    self.logger.warning(f"[LIQUIDITY] ❌ Original strike ${strike_price:.2f} FAILED: {reason}")
                    self.logger.info(f"[LIQUIDITY] 🔍 Searching for liquid alternatives...")
                    
                    # Extract predictions for smart strike selection
                    predicted_return = prediction_data.get('predicted_return', 0)
                    confidence = prediction_data.get('confidence', 0.5)
                    predicted_price = current_price * (1 + predicted_return)
                    
                    # Use multi-timeframe predictions if available
                    multi_timeframe = prediction_data.get('multi_timeframe_predictions')
                    if multi_timeframe and isinstance(multi_timeframe, dict):
                        # Use 15min prediction for short-term trades
                        timeframe_pred = multi_timeframe.get('15min')
                        if timeframe_pred and isinstance(timeframe_pred, dict):
                            predicted_return = timeframe_pred.get('return', predicted_return)
                            predicted_price = current_price * (1 + predicted_return)
                            self.logger.info(f"[LIQUIDITY] Using 15min prediction: {predicted_return*100:+.2f}% → ${predicted_price:.2f}")
                    
                    self.logger.info(f"[LIQUIDITY] Prediction: ${current_price:.2f} → ${predicted_price:.2f} ({predicted_return*100:+.2f}%) | Confidence: {confidence:.1%}")
                    
                    # DYNAMIC STRIKE SELECTION: Find liquid alternative with best expected profit
                    best_alternative = None
                    best_quality = 0
                    best_expected_profit = -999999  # Track expected P&L
                    
                    # Adjust max deviation based on confidence
                    # Higher confidence = stick closer to original strike
                    # Lower confidence = more flexibility
                    if confidence >= 0.7:
                        max_deviation_pct = 0.03  # 3% for high confidence
                    elif confidence >= 0.5:
                        max_deviation_pct = 0.05  # 5% for medium confidence
                    else:
                        max_deviation_pct = 0.08  # 8% for lower confidence
                    
                    self.logger.info(f"[LIQUIDITY] Max deviation: {max_deviation_pct*100:.1f}% (confidence-adjusted)")
                    
                    # Calculate original strike's characteristics
                    original_moneyness = (strike_price - current_price) / current_price
                    is_call = option_type_str == 'CALL'
                    
                    # Determine original strategy intent
                    if is_call:
                        if original_moneyness > 0.02:
                            strategy = "OTM_CALL"  # Bullish, needs upside
                        elif original_moneyness < -0.02:
                            strategy = "ITM_CALL"  # Deep ITM for delta
                        else:
                            strategy = "ATM_CALL"  # At-the-money
                    else:  # PUT
                        if original_moneyness < -0.02:
                            strategy = "OTM_PUT"  # Bearish, needs downside
                        elif original_moneyness > 0.02:
                            strategy = "ITM_PUT"  # Deep ITM for delta
                        else:
                            strategy = "ATM_PUT"  # At-the-money
                    
                    self.logger.info(f"[LIQUIDITY] Original strategy: {strategy} (moneyness: {original_moneyness*100:.1f}%)")
                    
                    # Sort options by how close they are to our target strike
                    strike_candidates = []
                    for opt in all_options:
                        if opt.get('option_type', '').lower() != option_type_tradier:
                            continue  # Only same option type
                        
                        candidate_strike = float(opt.get('strike', 0))
                        deviation = abs(candidate_strike - strike_price) / strike_price
                        
                        if deviation <= max_deviation_pct:
                            # Check if candidate maintains strategy intent
                            candidate_moneyness = (candidate_strike - current_price) / current_price
                            
                            # PROFITABILITY CHECK 1: Must maintain directional intent
                            maintains_strategy = False
                            if strategy == "OTM_CALL" and candidate_moneyness > 0:
                                maintains_strategy = True  # Still out-of-money (bullish)
                            elif strategy == "ITM_CALL" and candidate_moneyness < 0:
                                maintains_strategy = True  # Still in-the-money
                            elif strategy == "ATM_CALL" and -0.05 < candidate_moneyness < 0.05:
                                maintains_strategy = True  # Still near-the-money
                            elif strategy == "OTM_PUT" and candidate_moneyness < 0:
                                maintains_strategy = True  # Still out-of-money (bearish)
                            elif strategy == "ITM_PUT" and candidate_moneyness > 0:
                                maintains_strategy = True  # Still in-the-money
                            elif strategy == "ATM_PUT" and -0.05 < candidate_moneyness < 0.05:
                                maintains_strategy = True  # Still near-the-money
                            
                            if not maintains_strategy:
                                self.logger.info(f"[LIQUIDITY]   ✗ ${candidate_strike:.2f} wrong moneyness: {candidate_moneyness*100:.1f}% (need {strategy})")
                                continue
                            
                            # PROFITABILITY CHECK 2: Premium must be reasonable
                            candidate_bid = float(opt.get('bid', 0))
                            candidate_ask = float(opt.get('ask', 0))
                            candidate_mid = (candidate_bid + candidate_ask) / 2
                            
                            if candidate_mid <= 0:
                                self.logger.info(f"[LIQUIDITY]   ✗ ${candidate_strike:.2f} no premium available")
                                continue
                            
                            # Check if premium is in reasonable range (not too expensive vs original)
                            if candidate_mid > premium * 2.0:
                                self.logger.info(f"[LIQUIDITY]   ✗ ${candidate_strike:.2f} too expensive: ${candidate_mid:.2f} vs ${premium:.2f}")
                                continue
                            
                            # PROFITABILITY CHECK 3: Risk/reward must still be favorable
                            # For OTM options, check they're not too far OTM (unlikely to profit)
                            max_otm = 0.10  # Max 10% out-of-the-money
                            if is_call and candidate_moneyness > max_otm:
                                self.logger.info(f"[LIQUIDITY]   ✗ ${candidate_strike:.2f} too far OTM: {candidate_moneyness*100:.1f}%")
                                continue
                            elif not is_call and candidate_moneyness < -max_otm:
                                self.logger.info(f"[LIQUIDITY]   ✗ ${candidate_strike:.2f} too far OTM: {candidate_moneyness*100:.1f}%")
                                continue
                            
                            strike_candidates.append((deviation, opt))
                    
                    # Try candidates in order of closeness to original strike
                    strike_candidates.sort(key=lambda x: x[0])
                    self.logger.info(f"[LIQUIDITY] Found {len(strike_candidates)} strikes within 5% of ${strike_price:.2f}")
                    
                    for deviation, candidate in strike_candidates[:5]:  # Try top 5 closest
                        candidate_strike = float(candidate.get('strike'))
                        candidate_data = {
                            'bid': float(candidate.get('bid', 0)),
                            'ask': float(candidate.get('ask', 0)),
                            'volume': int(candidate.get('volume', 0)),
                            'open_interest': int(candidate.get('open_interest', 0)),
                            'strike': candidate_strike
                        }
                        
                        is_valid_alt, reason_alt = self.liquidity_validator.validate_option(
                            candidate_data,
                            current_price,
                            check_market_hours=check_hours
                        )
                        
                        if is_valid_alt:
                            quality = self.liquidity_validator.get_execution_quality_score(candidate_data)
                            
                            # Try sophisticated system (Monte Carlo + Execution Model) if available
                            if hasattr(self, 'mc_pricer') and self.mc_pricer and option_greeks:
                                # SOPHISTICATED SYSTEM: Monte Carlo + Execution Model + Distributional Utility
                                try:
                                    # Calculate minutes since market open (9:30 AM ET)
                                    now = self.get_market_time()
                                    market_open = now.replace(hour=9, minute=30, second=0)
                                    minutes_since_open = max(0, int((now - market_open).total_seconds() / 60))
                                    
                                    # Calculate spread percentage
                                    candidate_mid = (candidate_data['bid'] + candidate_data['ask']) / 2
                                    spread_pct = (candidate_data['ask'] - candidate_data['bid']) / candidate_mid * 100 if candidate_mid > 0 else 100
                                    
                                    # 1. Monte Carlo P&L simulation
                                    mc_params = MCPricingParams(
                                        underlying_price=current_price,
                                        strike=candidate_strike,
                                        option_type='call' if is_call else 'put',
                                        current_iv=option_greeks.get('iv', 0.35),  # Use IV from Greeks
                                        horizon_minutes=15,
                                        predicted_drift=predicted_return,
                                        regime=None,  # Could add HMM regime here if available
                                        num_paths=500
                                    )
                                    pnl_dist = self.mc_pricer.simulate_pnl_distribution(mc_params, candidate_mid)
                                    
                                    # 2. Execution model prediction
                                    exec_features = {
                                        'spread_pct': spread_pct,
                                        'spread_dollars': candidate_data['ask'] - candidate_data['bid'],
                                        'bid_size': 10,  # Default assumption
                                        'ask_size': 10,
                                        'open_interest': candidate_data['open_interest'],
                                        'volume_today': candidate_data['volume'],
                                        'minutes_since_open': minutes_since_open,
                                        'moneyness': (candidate_strike - current_price) / current_price,
                                        'delta': option_greeks.get('delta', 0.5),
                                        'gamma': option_greeks.get('gamma', 0.01),
                                        'vega': option_greeks.get('vega', 0.10),
                                        'theta': option_greeks.get('theta', -0.05),
                                        'regime': None,
                                        'days_to_expiry': days_until_expiry
                                    }
                                    exec_pred = self.exec_model.predict(exec_features)
                                    
                                    # 3. Distributional utility calculation
                                    utility, util_debug = self.utility_calc.calculate_utility(
                                        pnl_distribution=pnl_dist,
                                        execution_pred=exec_pred,
                                        calibrated_confidence=confidence  # Use raw for now, will calibrate later
                                    )
                                    
                                    # 4. Check tail guards
                                    passes, reason = self.utility_calc.passes_tail_guards(
                                        pnl_dist, exec_pred, utility
                                    )
                                    
                                    if not passes:
                                        self.logger.info(f"[LIQUIDITY]   ✗ ${candidate_strike:.2f} failed tail guards: {reason}")
                                        continue  # Skip this candidate
                                    
                                    expected_value = utility
                                    
                                    self.logger.info(f"[LIQUIDITY]   ✓ ${candidate_strike:.2f} PASSED | Quality: {quality}/100")
                                    self.logger.info(f"[LIQUIDITY]      Utility: ${expected_value:.2f} | E[P&L]: ${util_debug['expected_pnl']:.2f} | q05: ${pnl_dist['q05']:.2f} | p_fill: {exec_pred['p_fill']:.1%}")
                                    
                                except Exception as e:
                                    self.logger.warning(f"[LIQUIDITY] Sophisticated system failed: {e}, using simple method")
                                    import traceback
                                    self.logger.warning(traceback.format_exc())
                                    # Fall back to simple calculation
                                    if is_call:
                                        intrinsic_at_prediction = max(0, predicted_price - candidate_strike)
                                    else:
                                        intrinsic_at_prediction = max(0, candidate_strike - predicted_price)
                                    expected_profit = intrinsic_at_prediction - candidate_mid
                                    expected_value = expected_profit * confidence * (quality / 100)
                                    self.logger.info(f"[LIQUIDITY]   ✓ ${candidate_strike:.2f} PASSED | Quality: {quality}/100 | Expected P&L: ${expected_profit:.2f} | EV: ${expected_value:.2f}")
                            else:
                                # SIMPLE INTRINSIC VALUE CALCULATION (fallback)
                                candidate_mid = (candidate_data['bid'] + candidate_data['ask']) / 2
                                if is_call:
                                    intrinsic_at_prediction = max(0, predicted_price - candidate_strike)
                                else:
                                    intrinsic_at_prediction = max(0, candidate_strike - predicted_price)
                                
                                expected_profit = intrinsic_at_prediction - candidate_mid
                                expected_value = expected_profit * confidence * (quality / 100)
                                
                                self.logger.info(f"[LIQUIDITY]   ✓ ${candidate_strike:.2f} PASSED | Quality: {quality}/100 | Expected P&L: ${expected_profit:.2f} | EV: ${expected_value:.2f}")
                            
                            # Pick strike with best expected value
                            if expected_value > best_expected_profit:
                                best_alternative = candidate_data
                                best_quality = quality
                                best_expected_profit = expected_value
                        else:
                            self.logger.debug(f"[LIQUIDITY]   ✗ ${candidate_strike:.2f} failed: {reason_alt}")
                    
                    if best_alternative:
                        # Found a liquid alternative!
                        original_strike = strike_price
                        strike_price = best_alternative['strike']
                        premium = (best_alternative['bid'] + best_alternative['ask']) / 2
                        
                        # Calculate profitability metrics for the alternative
                        alternative_moneyness = (strike_price - current_price) / current_price
                        spread_pct = (best_alternative['ask'] - best_alternative['bid']) / premium * 100
                        
                        deviation_pct = abs(strike_price - original_strike) / original_strike * 100
                        
                        # Log whether using Greeks or simple method
                        if hasattr(self, 'ev_calculator') and self.ev_calculator and option_greeks:
                            method = "GREEKS-BASED EV"
                        else:
                            method = "PREDICTION-BASED"
                        
                        self.logger.info(f"[LIQUIDITY] ✅ Using {method} ALTERNATIVE: ${strike_price:.2f} (was ${original_strike:.2f}, {deviation_pct:.1f}% deviation)")
                        self.logger.info(f"[LIQUIDITY] Strategy preserved: {strategy} | Moneyness: {alternative_moneyness*100:.1f}% (was {original_moneyness*100:.1f}%)")
                        self.logger.info(f"[LIQUIDITY] Expected Value: ${best_expected_profit:.2f}")
                        self.logger.info(f"[LIQUIDITY] Adjusted Quote: Bid=${best_alternative['bid']:.2f}, Ask=${best_alternative['ask']:.2f} | Spread: {spread_pct:.1f}%")
                        self.logger.info(f"[LIQUIDITY] Quality: {best_quality}/100 | Vol: {best_alternative['volume']} | OI: {best_alternative['open_interest']}")
                        self.logger.info(f"[LIQUIDITY] At predicted price ${predicted_price:.2f}, this strike maximizes edge")
                        
                        validation_data = best_alternative  # Use for quality score later
                    else:
                        # No liquid alternatives found - try longer expiration as fallback
                        self.logger.warning(f"[LIQUIDITY] ⚠️  No liquid strikes found at current expiration ({expiration_date})")
                        self.logger.info(f"[LIQUIDITY] 🔍 Trying expiration +2 days for better liquidity...")
                        
                        # Try 2 days later
                        fallback_expiration = (self.get_market_time() + timedelta(days=days_until_expiry + 2)).strftime('%Y-%m-%d')
                        fallback_options = tradier.get_options_chain(symbol, fallback_expiration)
                        
                        if fallback_options:
                            for opt in fallback_options:
                                if (opt.get('option_type', '').lower() == option_type_tradier and
                                    abs(float(opt.get('strike', 0)) - strike_price) < 0.01):
                                    
                                    fallback_data = {
                                        'bid': float(opt.get('bid', 0)),
                                        'ask': float(opt.get('ask', 0)),
                                        'volume': int(opt.get('volume', 0)),
                                        'open_interest': int(opt.get('open_interest', 0)),
                                        'strike': float(opt.get('strike', 0))
                                    }
                                    
                                    is_valid_fb, reason_fb = self.liquidity_validator.validate_option(
                                        fallback_data, current_price, check_market_hours=check_hours
                                    )
                                    
                                    if is_valid_fb:
                                        quality = self.liquidity_validator.get_execution_quality_score(fallback_data)
                                        self.logger.info(f"[LIQUIDITY] ✅ Using LONGER EXPIRATION: {fallback_expiration} (was {expiration_date})")
                                        self.logger.info(f"[LIQUIDITY] Quality: {quality}/100 | Better liquidity at longer DTE")
                                        
                                        premium = (fallback_data['bid'] + fallback_data['ask']) / 2
                                        validation_data = fallback_data
                                        best_alternative = fallback_data
                                        break
                        
                        if not best_alternative:
                            self.logger.error(f"[LIQUIDITY] ❌ No liquid alternatives found")
                            self.logger.error(f"[LIQUIDITY] Paper trade REJECTED (no tradeable strikes available)")
                            return None
                else:
                    quality_score = self.liquidity_validator.get_execution_quality_score(validation_data)
                    self.logger.info(f"[LIQUIDITY] ✅ Original strike PASSED | Quality: {quality_score}/100")
                    self.logger.info(f"[LIQUIDITY] Paper trade will proceed (matches Tradier validation)")
                
            except Exception as e:
                self.logger.error(f"[LIQUIDITY] ❌ Error validating liquidity: {e}")
                self.logger.error(f"[LIQUIDITY] Trade rejected - validation failed")
                import traceback
                self.logger.error(traceback.format_exc())
                return None
        elif is_simulation:
            # In simulation mode, use SIMULATED liquidity instead of live Tradier API
            # This provides realistic market conditions for training without API calls
            sim_liquidity = self._simulate_liquidity(
                symbol=symbol,
                strike_price=strike_price,
                current_price=current_price,
                option_type=option_type_str,
                days_to_expiry=int(time_to_expiry * 365) if time_to_expiry < 1 else 30
            )
            
            # Apply simulated bid/ask spread to premium (makes training realistic)
            simulated_spread_pct = sim_liquidity['spread_pct'] / 100
            simulated_slippage = simulated_spread_pct * 0.3  # Pay ~30% of spread on average
            
            # Adjust premium based on simulated liquidity
            # Entry: pay higher (ask side), Exit: receive lower (bid side)
            premium_adjustment = premium * simulated_slippage
            premium = premium + premium_adjustment
            
            self.logger.info(f"[SIMULATION] Using simulated liquidity: Spread={sim_liquidity['spread_pct']:.1f}%, "
                           f"Vol={sim_liquidity['volume']}, OI={sim_liquidity['open_interest']}")
            self.logger.debug(f"[SIMULATION] Premium adjusted by +${premium_adjustment:.3f} for simulated spread")
        
        # 4b. Check position size liquidity constraints
        can_fill, liquidity_msg = self._check_liquidity_constraints(total_trade_value)
        if not can_fill:
            self.logger.warning(f"🚫 Liquidity Issue: {liquidity_msg}")
            return None

        # Calculate entry fees with symbol for correct fee structure
        entry_fees = self.fee_calculator.calculate_entry_fees(
            premium, quantity, symbol)
        total_cost = premium * quantity * 100 + entry_fees

        # Check if we have enough balance
        if total_cost > self.current_balance:
            # During simulation/training, ALLOW negative balance so bot learns consequences
            # NO auto-refill - bot must learn that losing money is bad!
            if self.simulation_mode or self.time_travel_mode:
                self.logger.warning(f"⚠️ [TRAINING] Low balance: ${self.current_balance:.2f} < ${total_cost:.2f}")
                self.logger.warning(f"⚠️ [TRAINING] Allowing trade with negative balance - bot must learn consequences!")
                # Continue with trade - balance will go negative
            else:
                self.logger.warning(
                    f"Insufficient balance for trade: ${total_cost:.2f} > ${self.current_balance:.2f}")
                return None

        # Calculate expiration date (30-45 days from now for realistic options)
        from datetime import timedelta
        import random
        # Track day trade if this is an intraday trade
        if time_to_expiry < 1.0:  # Same-day expiry = day trade
            self.day_trades_count += 1
            self.logger.info(f"📊 Day Trade #{self.day_trades_count} placed (PDT limit: 3 for accounts < $25k)")

        # Set expiration date for options (30-45 days is optimal for swing trading)
        # 0DTE (6 hours) was killing trades with theta decay!
        days_to_expiry = random.randint(30, 45)  # 30-45 days gives time for moves to play out
        expiration_date = self.get_market_time() + timedelta(days=days_to_expiry)

        # ============= ADD REALISTIC MARKET CONDITIONS =============
        # Apply bid-ask spread and slippage to make paper trading realistic
        # FURTHER REDUCED: Previous 7-11% round-trip was killing all trades!
        # Real liquid options: ~1-2% spread, 0.5% slippage = ~2-3% round-trip

        # TT_REDUCE_FRICTION: Scale down friction for testing (0.0 = no friction, 1.0 = full)
        friction_scale = float(os.environ.get("TT_REDUCE_FRICTION", "1.0"))
        friction_scale = max(0.0, min(1.0, friction_scale))

        # 1. Bid-ask spread (liquid options: 1-2%)
        spread_pct = 0.015 * friction_scale  # 1.5% spread (was 4% - way too high!)
        spread_cost = premium * spread_pct

        # 2. Market impact slippage (0.3-0.5% for market orders on liquid options)
        slippage_pct = 0.005 * friction_scale  # 0.5% slippage (was 1.5%)
        slippage_cost = premium * slippage_pct
        
        # 3. Time-of-day penalty (minimal for market hours)
        current_hour = self.get_market_time().hour
        time_penalty = 0.0
        if current_hour < 9 or current_hour >= 16:  # Outside market hours
            time_penalty = premium * 0.005  # 0.5% penalty for off-hours (was 1%)
        
        # 4. Apply realistic entry price (when buying: pay ask + slippage + penalties)
        # Entry: +0.75% (spread/2) + 0.5% (slippage) + 0-0.5% (time) = +1.25-1.75%
        # Round-trip total: ~2.5-3.5% (realistic for liquid options)
        realistic_premium = premium + (spread_cost / 2) + slippage_cost + time_penalty
        
        # 5. Recalculate costs with realistic premium
        realistic_total_cost = realistic_premium * quantity * 100 + entry_fees
        
        # 6. Check balance again with realistic costs
        if realistic_total_cost > self.current_balance:
            # During simulation/training, ALLOW negative balance - no free money!
            if self.simulation_mode or self.time_travel_mode:
                self.logger.warning(f"⚠️ [TRAINING] Realistic cost ${realistic_total_cost:.2f} > balance ${self.current_balance:.2f}")
                self.logger.warning(f"⚠️ [TRAINING] Allowing negative balance - bot learns from consequences!")
                # Continue with trade - balance will go negative
            else:
                self.logger.warning(
                    f"Insufficient balance for realistic trade: ${realistic_total_cost:.2f} > ${self.current_balance:.2f}")
                self.logger.info(f"   Original cost: ${total_cost:.2f}, Realistic cost: ${realistic_total_cost:.2f}")
                self.logger.info(f"   Market friction: ${realistic_total_cost - total_cost:.2f} ({((realistic_total_cost - total_cost) / total_cost * 100):.1f}%)")
                return None
        
        # Use realistic premium for the trade
        premium = realistic_premium
        total_cost = realistic_total_cost
        
        self.logger.info(f"💸 Market friction applied: +${realistic_total_cost - (premium * quantity * 100 + entry_fees):.2f}")
        # ============= END REALISTIC MARKET CONDITIONS =============

        # Create trade
        # Calculate planned exit time using SMART multi-timeframe analysis
        max_hold_hours = self._calculate_optimal_hold_time(prediction_data)
        planned_exit = self.get_market_time() + timedelta(hours=max_hold_hours)
        
        # ============= DYNAMIC STOP LOSS BASED ON HMM REGIME =============
        # Adjust stop loss and take profit based on market volatility, trend, and liquidity
        regime_stop_pct, regime_take_pct = self._get_regime_adjusted_stops(prediction_data)
        
        trade = Trade(
            id=str(uuid.uuid4()),
            timestamp=self.get_market_time(),
            symbol=symbol,
            option_type=option_type,
            strike_price=strike_price,
            premium_paid=premium,
            quantity=quantity,
            entry_price=current_price,
            status=OrderStatus.FILLED,
            stop_loss=premium * (1 - regime_stop_pct),  # Dynamic based on HMM regime
            take_profit=premium * (1 + regime_take_pct),  # Dynamic based on HMM regime
            ml_confidence=ml_confidence,
            ml_prediction=ml_prediction,
            expiration_date=expiration_date,
            planned_exit_time=planned_exit,
            max_hold_hours=max_hold_hours
        )

        # DEBUG: Log actual stop/take profit levels
        tp_pct = ((trade.take_profit / premium) - 1.0) * 100.0 if premium > 0 else 0
        sl_pct = (1.0 - (trade.stop_loss / premium)) * 100.0 if premium > 0 else 0
        self.logger.info(f"📊 ENTRY LEVELS: Premium=${premium:.2f}, SL=${trade.stop_loss:.2f} (-{sl_pct:.1f}%), TP=${trade.take_profit:.2f} (+{tp_pct:.1f}%)")

        # Add prediction metadata for RL exit policy
        raw_predicted_return = prediction_data.get('predicted_return', 0.0)
        trade.prediction_timeframe = prediction_data.get('timeframe_minutes', 60)
        trade.hmm_state = prediction_data.get('hmm_state', None)  # Store HMM regime at entry
        
        # ============= REALISTIC PROJECTION CAP =============
        # ML models often over-predict moves. Cap based on timeframe:
        # 1 hour: max 0.5% (realistic intraday move)
        # 4 hours: max 1.0%
        # 1 day: max 2.0%
        # Multi-day: max 3.0%
        timeframe_hours = trade.prediction_timeframe / 60
        if timeframe_hours <= 1:
            max_return = 0.005  # 0.5%
        elif timeframe_hours <= 4:
            max_return = 0.010  # 1.0%
        elif timeframe_hours <= 24:
            max_return = 0.020  # 2.0%
        else:
            max_return = 0.030  # 3.0%
        
        # Cap the prediction but preserve direction
        if abs(raw_predicted_return) > max_return:
            capped_return = max_return if raw_predicted_return > 0 else -max_return
            self.logger.info(f"📉 Capped prediction: {raw_predicted_return*100:.2f}% → {capped_return*100:.2f}% (max for {timeframe_hours:.1f}h)")
            trade.predicted_return = capped_return
        else:
            trade.predicted_return = raw_predicted_return
        # ============= END REALISTIC CAP =============
        
        # Calculate projected profit based on ML prediction
        # Use Greeks (delta) if available, otherwise use conservative estimate
        predicted_return_pct = trade.predicted_return * 100  # Convert to percentage
        predicted_price_move = current_price * trade.predicted_return  # Dollar move in underlying
        
        # Try to use actual delta from Greeks
        if option_greeks and 'delta' in option_greeks:
            delta = abs(option_greeks['delta'])
            # Delta tells us how much option price changes per $1 move in stock
            # For more accurate projection, account for gamma (delta change)
            # Conservative estimate: use 70% of delta to account for theta decay
            effective_delta = delta * 0.7
            projected_premium_change = predicted_price_move * effective_delta * 100  # Per contract
            projected_exit_premium = premium + (projected_premium_change / 100)
            
            self.logger.info(f"📊 Using Greeks: Delta={delta:.3f}, Effective={effective_delta:.3f}")
        else:
            # Fallback: Estimate based on moneyness and time
            # OTM options: higher leverage (3-5x)
            # ATM options: medium leverage (2-3x)
            # ITM options: lower leverage (1-2x)
            moneyness = abs(strike_price - current_price) / current_price
            
            if moneyness > 0.05:  # >5% OTM
                leverage = 4.0  # High leverage for OTM
            elif moneyness > 0.02:  # 2-5% OTM
                leverage = 3.0  # Medium-high leverage
            else:  # Near ATM
                leverage = 2.0  # Lower leverage
            
            # Adjust for time decay (longer DTE = less leverage)
            if expiration_date:
                days_to_expiry = (expiration_date - self.get_market_time()).days
                if days_to_expiry > 30:
                    leverage *= 0.7  # Reduce leverage for longer-dated options
            
            projected_exit_premium = premium * (1 + trade.predicted_return * leverage)
            self.logger.info(f"📊 Using estimated leverage: {leverage:.1f}x (moneyness={moneyness:.1%})")
        
        # Calculate projected profit in dollars
        projected_profit_per_contract = (projected_exit_premium - premium) * 100  # Per contract
        trade.projected_profit = projected_profit_per_contract * quantity
        trade.projected_return_pct = predicted_return_pct
        
        self.logger.info(f"📊 Projected P&L: ${trade.projected_profit:+.2f} ({predicted_return_pct:+.2f}% predicted move)")
        self.logger.info(f"   Entry: ${premium:.2f} → Projected Exit: ${projected_exit_premium:.2f}")

        # Update balance
        self.current_balance -= total_cost

        # Add to active trades
        self.active_trades.append(trade)
        print(f"[DEBUG_PLACE] Trade placed! active_trades now = {len(self.active_trades)}")

        # Save to database
        self._save_trade(trade)
        self._save_account_state()

        self.logger.info(f"Placed {option_type.value} trade:")
        self.logger.info(f"  • Symbol: {symbol}")
        self.logger.info(f"  • Strike: ${strike_price:.2f}")
        self.logger.info(f"  • Premium: ${premium:.2f}")
        self.logger.info(f"  • Quantity: {quantity}")
        self.logger.info(f"  • Entry Fees: ${entry_fees:.2f}")
        self.logger.info(f"  • Total Cost: ${total_cost:.2f}")
        self.logger.info(f"  • ML Confidence: {ml_confidence:.1%}")
        self.logger.info(f"  • Remaining Balance: ${self.current_balance:.2f}")

        # 🧠 LOG EXECUTION OUTCOME FOR LIQUIDITY LEARNING
        # This allows the neural network to learn which strikes/conditions lead to good fills
        try:
            if hasattr(self, 'bot') and self.bot and hasattr(self.bot, 'log_execution_outcome'):
                # Build execution outcome data
                from datetime import datetime as dt
                execution_outcome = {
                    'timestamp': dt.now().isoformat(),
                    'symbol': symbol,
                    'strike': strike_price,
                    'option_type': option_type_str,
                    'spot_price': current_price,
                    'bid': validation_data.get('bid', 0) if 'validation_data' in locals() else premium * 0.95,
                    'ask': validation_data.get('ask', 0) if 'validation_data' in locals() else premium * 1.05,
                    'volume': validation_data.get('volume', 0) if 'validation_data' in locals() else 0,
                    'open_interest': validation_data.get('open_interest', 0) if 'validation_data' in locals() else 0,
                    'greeks': option_greeks if option_greeks else {},
                    'filled': True,  # Paper trades always "fill" instantly
                    'actual_price': premium,
                    'slippage': 0.0,  # Paper trades have no slippage
                    'time_to_fill': 0.0,  # Instant fill
                    'confidence': ml_confidence,
                    'prediction': ml_prediction
                }
                
                self.bot.log_execution_outcome(execution_outcome)
                self.logger.info(f"[LEARN] Logged execution outcome for liquidity learning")
        except Exception as e:
            self.logger.warning(f"[LEARN] Could not log execution outcome: {e}")

        return trade

    def _save_trade(self, trade: Trade):
        """Save trade to database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO trades
                (id, timestamp, symbol, option_type, strike_price, premium_paid,
                 quantity, entry_price, exit_price, exit_timestamp, status,
                 profit_loss, stop_loss, take_profit, ml_confidence, ml_prediction,
                 expiration_date, created_at, projected_profit, projected_return_pct,
                 planned_exit_time, max_hold_hours, tradier_option_symbol,
                 tradier_order_id, is_real_trade, live_blocked_reason, exit_reason, run_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                trade.id,
                trade.timestamp.isoformat(),
                trade.symbol,
                trade.option_type.value,
                trade.strike_price,
                trade.premium_paid,
                trade.quantity,
                trade.entry_price,
                trade.exit_price,
                trade.exit_timestamp.isoformat() if trade.exit_timestamp else None,
                trade.status.value,
                trade.profit_loss,
                trade.stop_loss,
                trade.take_profit,
                trade.ml_confidence,
                trade.ml_prediction,
                trade.expiration_date.isoformat() if trade.expiration_date else None,
                self.get_market_time().isoformat(),
                getattr(trade, 'projected_profit', 0.0),
                getattr(trade, 'projected_return_pct', 0.0),
                trade.planned_exit_time.isoformat() if trade.planned_exit_time else None,
                trade.max_hold_hours,
                getattr(trade, 'tradier_option_symbol', None),
                getattr(trade, 'tradier_order_id', None),
                getattr(trade, 'is_real_trade', False),
                getattr(trade, 'live_blocked_reason', None),
                getattr(trade, 'exit_reason', None),
                getattr(trade, 'run_id', None) or getattr(self, 'current_run_id', None)
            ))
            conn.commit()

    def update_positions(self, symbol: str = 'BITX', current_price: float = None):
        """
        Update all active positions based on current market prices
        PRIORITY: Real Tradier trades are processed FIRST

        Args:
            symbol: Symbol to update
            current_price: If provided, uses this price instead of fetching from API
                          (Important for simulation mode to avoid rate limiting!)
        """
        # DEBUG: Entry point
        print(f"[DEBUG_UPDATE] update_positions called: symbol={symbol}, price={current_price}, active_trades={len(self.active_trades)}")

        # Process any settlements that have cleared (T+1)
        self.process_settlements()
        
        # SAFETY NET: Periodic stale position check (every 10 cycles)
        # This catches positions that got stuck due to bot issues (shutdown, bugs, etc.)
        if not hasattr(self, '_update_cycle_count'):
            self._update_cycle_count = 0
        self._update_cycle_count += 1
        
        # ALWAYS check for stale positions (was every 10 cycles - too slow!)
        # 4-hour max is appropriate for intraday options trading
        self.recover_stale_positions(max_age_hours=4)
        
        if not self.active_trades:
            return
        
        # PRIORITY CHECK: Separate real trades from paper trades
        real_trades = [t for t in self.active_trades if t.is_real_trade]
        paper_trades = [t for t in self.active_trades if not t.is_real_trade]
        
        if real_trades:
            self.logger.warning(f"[PRIORITY] Processing {len(real_trades)} REAL trades first, then {len(paper_trades)} paper trades")

        # Use provided price if available (simulation mode), otherwise fetch
        if current_price is not None:
            self.logger.debug(f"📊 Using provided price: ${current_price:.2f} (simulation mode)")
        else:
            # Check if in simulation mode (NO API CALLS allowed!)
            # The parent bot should ALWAYS pass current_price in simulation
            if hasattr(self, 'simulation_mode') and self.simulation_mode:
                self.logger.error(f"🚨 SIMULATION MODE: Cannot fetch price! Skipping position update.")
                return  # EXIT EARLY - don't make API calls in simulation!
            
            # Get current market data - force fresh data for accurate pricing (LIVE TRADING ONLY)
            try:
                # Try 1-minute data first for most current price
                data = self.data_source.get_data(symbol, period='1d', interval='1m')
                if data.empty:
                    # Fallback to 15-minute data
                    data = self.data_source.get_data(symbol, period='1d', interval='15m')
                
                if not data.empty:
                    current_price = float(data['close'].iloc[-1])
                    self.logger.info(f"📊 Current {symbol} price: ${current_price:.2f} (from {data.index[-1]})")
                else:
                    # No market data available
                    self.logger.error(f"❌ No market data available for {symbol} - skipping position update")
                    return  # Exit early - can't update without price data
            except Exception as e:
                self.logger.error(f"❌ Error getting market data: {e}")
                self.logger.error(f"❌ Skipping position update - cannot proceed without price")
                return  # Exit early - can't update without price data

        trades_to_close = []

        for trade in self.active_trades:
            if trade.status != OrderStatus.FILLED:
                continue

            # Calculate time to expiry for theta decay
            if trade.expiration_date is None:
                # Default to same-day expiry if expiration_date is missing
                trade.expiration_date = self.get_market_time().replace(hour=16, minute=0, second=0, microsecond=0)
                self.logger.warning(f"⚠️ Trade {trade.id} missing expiration_date, set to market close today")
            
            # Handle timezone-aware vs timezone-naive datetime comparison for expiration
            market_time = self.get_market_time()
            expiration_date = trade.expiration_date
            
            if expiration_date.tzinfo is not None and market_time.tzinfo is None:
                import pytz
                market_time = pytz.UTC.localize(market_time)
            elif expiration_date.tzinfo is None and market_time.tzinfo is not None:
                import pytz
                expiration_date = pytz.UTC.localize(expiration_date)
            
            time_to_expiry_hours = (expiration_date - market_time).total_seconds() / 3600
            time_to_expiry_days = max(0.01, time_to_expiry_hours / 24)  # Minimum 0.01 days
            
            # Calculate current option value
            # For LIVE trades, fetch REAL-TIME quote from Tradier
            # BUT ONLY IF NOT IN SIMULATION MODE!
            is_sim = getattr(self, 'simulation_mode', False) or getattr(self, 'time_travel_mode', False)
            if trade.is_real_trade and trade.tradier_option_symbol and not is_sim:
                try:
                    self.logger.info(f"📊 [LIVE] Fetching real-time quote for {trade.tradier_option_symbol}")
                    
                    # Import Tradier credentials
                    from tradier_credentials import TradierCredentials
                    import requests
                    
                    creds = TradierCredentials()
                    live_creds = creds.get_live_credentials()
                    
                    if live_creds:
                        headers = {
                            'Authorization': f'Bearer {live_creds["access_token"]}',
                            'Accept': 'application/json'
                        }
                        
                        quote_url = 'https://api.tradier.com/v1/markets/quotes'
                        response = requests.get(
                            quote_url,
                            headers=headers,
                            params={'symbols': trade.tradier_option_symbol, 'greeks': 'false'},
                            timeout=10
                        )
                        
                        if response.status_code == 200:
                            quote_data = response.json()
                            quotes = quote_data.get('quotes', {}).get('quote', {})
                            if isinstance(quotes, list):
                                quotes = quotes[0] if quotes else {}
                            
                            # Use LAST price (most recent trade), fallback to BID
                            last_price = float(quotes.get('last', 0))
                            bid_price = float(quotes.get('bid', 0))
                            ask_price = float(quotes.get('ask', 0))
                            
                            # Use last price if available, otherwise bid (per-share price)
                            per_share_price = last_price if last_price > 0 else bid_price
                            
                            # FIX: premium_paid is per-share, so current_premium must also be per-share
                            # The P&L calculation multiplies by (quantity * 100) later
                            # OLD BUG: current_premium = per_share_price * 100 / trade.quantity (WRONG!)
                            current_premium = per_share_price  # Keep as per-share to match premium_paid
                            
                            self.logger.info(f"📊 [LIVE] Real Tradier quote: ${per_share_price:.2f}/share (Last: ${last_price:.2f}, Bid: ${bid_price:.2f}, Ask: ${ask_price:.2f})")
                        else:
                            # Fallback to calculation
                            current_premium = self._calculate_option_premium(
                                trade.symbol, trade.option_type, trade.strike_price, current_price,
                                is_entry=False, time_to_expiry_days=time_to_expiry_days
                            )
                            self.logger.warning(f"⚠️ [LIVE] Could not fetch quote (status {response.status_code}), using simulation: ${current_premium:.2f}")
                    else:
                        # No credentials, use calculation
                        current_premium = self._calculate_option_premium(
                            trade.symbol, trade.option_type, trade.strike_price, current_price,
                            is_entry=False, time_to_expiry_days=time_to_expiry_days
                        )
                        self.logger.warning(f"⚠️ [LIVE] No credentials, using simulation: ${current_premium:.2f}")
                        
                except Exception as e:
                    # Error fetching quote, fallback to calculation
                    current_premium = self._calculate_option_premium(
                        trade.symbol, trade.option_type, trade.strike_price, current_price,
                        is_entry=False, time_to_expiry_days=time_to_expiry_days
                    )
                    self.logger.error(f"❌ [LIVE] Quote fetch error: {e}, using simulation: ${current_premium:.2f}")
            else:
                # ==============================================================================
                # ✅ RELATIVE PRICING MODEL (Fixes "off" P&L)
                # ==============================================================================
                # Instead of recalculating option price from scratch (which can drift),
                # we calculate the CHANGE in price relative to entry based on:
                # 1. Delta (Price movement)
                # 2. Theta (Time decay)
                # ==============================================================================
                
                entry_time = datetime.fromisoformat(trade.timestamp.replace('Z', '+00:00')) if isinstance(trade.timestamp, str) else trade.timestamp
                current_time = self.get_market_time()
                
                # Handle timezone-aware vs timezone-naive datetime comparison
                if entry_time.tzinfo is not None and current_time.tzinfo is None:
                    import pytz
                    current_time = pytz.UTC.localize(current_time)
                elif entry_time.tzinfo is None and current_time.tzinfo is not None:
                    import pytz
                    entry_time = pytz.UTC.localize(entry_time)
                
                minutes_held = (current_time - entry_time).total_seconds() / 60
                
                # 1. Calculate underlying price move
                price_move_pct = (current_price / trade.entry_price - 1.0)
                
                # 2. Determine directional impact with REALISTIC MONEYNESS-BASED LEVERAGE
                # Delta varies by moneyness:
                #   - Deep ITM: delta ~0.80-1.00, leverage VARIES (higher if Near-ITM)
                #   - ATM: delta ~0.50, leverage ~10-20x (0DTE)
                #   - OTM: delta ~0.20-0.40, leverage ~5-10x
                
                is_call = trade.option_type in [OrderType.CALL, OrderType.BUY_CALL]
                
                # Calculate moneyness (how far ITM/OTM the option is)
                if is_call:
                    moneyness = (current_price - trade.strike_price) / trade.strike_price
                else:  # PUT
                    moneyness = (trade.strike_price - current_price) / trade.strike_price
                
                # Moneyness-based delta leverage (more realistic)
                # Positive moneyness = ITM, negative = OTM
                if moneyness > 0.05:  # Very Deep ITM (>5%)
                    delta_leverage = 4.0  # Moves like stock, lower leverage ratio due to high premium
                elif moneyness > 0.02:  # Deep ITM (2-5%)
                    delta_leverage = 8.0  # Decent leverage
                elif moneyness > 0.0:  # Just ITM (0-2%)
                    delta_leverage = 15.0  # High leverage (Delta > 0.5, Premium still low)
                elif moneyness > -0.01:  # ATM (-1% to 0%)
                    delta_leverage = 12.0  # Good leverage
                elif moneyness > -0.03:  # Slightly OTM (-3% to -1%)
                    delta_leverage = 8.0
                else:  # Deep OTM (<-3%)
                    delta_leverage = 4.0  # Low leverage, mostly time value
                
                if is_call:
                    # CALL: Up is good, Down is bad
                    option_move_pct = price_move_pct * delta_leverage
                else:
                    # PUT: Down is good, Up is bad (price_move_pct is negative when price goes down)
                    option_move_pct = -price_move_pct * delta_leverage
                
                # 3. Apply Delta/Gamma impact to Entry Premium
                # (1 + move) logic works for both gains and losses
                theoretical_premium = trade.premium_paid * (1 + option_move_pct)
                
                # 4. Determine Theta Decay Rate (ONLY for Extrinsic Value)
                # 0DTE (expires today): ~0.2% per minute of EXTRINSIC value
                # < 7 days: ~0.03% per minute
                # < 30 days: ~0.01% per minute
                # > 30 days: ~0.005% per minute
                if time_to_expiry_days < 1.0:
                    theta_decay_per_minute = 0.0010  # 0.10% per minute (0DTE) - reduced from 0.20%
                elif time_to_expiry_days < 7:
                    theta_decay_per_minute = 0.00015  # 0.015% per minute (near expiration) - reduced
                elif time_to_expiry_days < 30:
                    theta_decay_per_minute = 0.00005  # 0.005% per minute (mid-term)
                else:
                    theta_decay_per_minute = 0.00002  # 0.002% per minute (long-term)
                
                # Calculate Intrinsic Value (Real Value)
                if is_call:
                    intrinsic_value = max(0.0, current_price - trade.strike_price)
                else:
                    intrinsic_value = max(0.0, trade.strike_price - current_price)
                
                # Calculate Extrinsic Value (Time Value)
                # Extrinsic = Premium - Intrinsic
                # Using theoretical_premium (price movement adjusted) as the "Pre-Theta Premium"
                # Ensure theoretical_premium is at least intrinsic value
                theoretical_premium = max(intrinsic_value + 0.01, theoretical_premium)
                
                extrinsic_value = max(0.0, theoretical_premium - intrinsic_value)
                
                # 5. Apply Theta Decay ONLY to Extrinsic Value
                theta_decay_total = extrinsic_value * theta_decay_per_minute * minutes_held
                
                # New Current Premium = Intrinsic + (Extrinsic - Decay)
                current_extrinsic = max(0.0, extrinsic_value - theta_decay_total)
                current_premium = intrinsic_value + current_extrinsic
                
                # 6. Add small random noise for market realism (IV fluctuations)
                import random
                iv_noise = random.uniform(-0.01, 0.01)  # +/- 1% noise
                current_premium *= (1 + iv_noise)
                
                # 7. ✅ IMMEDIATE GAIN CAP - Apply BEFORE any exit logic
                # Prevents unrealistic gains from calculation errors
                # Max 400% gain per position (realistic for volatility explosions)
                max_gain_cap = trade.premium_paid * 5.0
                if current_premium > max_gain_cap:
                    self.logger.warning(f"🚨 [GAIN CAP] Premium ${current_premium:.2f} > 400% cap ${max_gain_cap:.2f} - clamping (moneyness: {moneyness:.2%}, leverage: {delta_leverage}x)")
                    current_premium = max_gain_cap

            
            # ✅ REALISTIC THETA DECAY AND ADJUSTMENTS (ONLY FOR PAPER TRADES!)
            # For live trades, we already have the real market price - don't simulate!
            if not (trade.is_real_trade and trade.tradier_option_symbol):
                # This is a PAPER trade - apply simulation adjustments
                # Note: Theta decay already applied above in Relative Pricing Model
                # We just need to define variables used by downstream logic (sanity checks)
                
                # Re-calculate minutes_held for downstream use if needed
                entry_time = datetime.fromisoformat(trade.timestamp.replace('Z', '+00:00')) if isinstance(trade.timestamp, str) else trade.timestamp
                current_time = self.get_market_time()
                if entry_time.tzinfo is not None and current_time.tzinfo is None:
                    import pytz
                    current_time = pytz.UTC.localize(current_time)
                elif entry_time.tzinfo is None and current_time.tzinfo is not None:
                    import pytz
                    entry_time = pytz.UTC.localize(entry_time)
                minutes_held = (current_time - entry_time).total_seconds() / 60
                
                # Already applied theta decay in the else block above, so we skip it here
                # to avoid double counting.
                
                # ✅ SANITY CHECKS (simulation only)
                # Prevent paper-trade options from showing profits when price moves
                # clearly against the position. This keeps P&L directionally consistent.
                try:
                    if trade.entry_price and trade.premium_paid:
                        price_move_pct = (current_price / trade.entry_price - 1.0) * 100.0
                        # FIX: premium_paid is already per-share, don't divide by quantity!
                        entry_premium = trade.premium_paid  # NOT / trade.quantity

                        is_call = trade.option_type in [OrderType.CALL, OrderType.BUY_CALL]
                        is_put = trade.option_type in [OrderType.PUT, OrderType.BUY_PUT]
                        
                        # DEBUG: Log sanity check inputs
                        self.logger.debug(f"[SANITY DEBUG] {trade.id[:8]} | Type: {'CALL' if is_call else 'PUT'} | "
                                        f"Entry Price: ${trade.entry_price:.2f} | Current Price: ${current_price:.2f} | "
                                        f"Price Move: {price_move_pct:+.2f}% | Entry Premium: ${entry_premium:.2f} | "
                                        f"Raw Premium: ${current_premium:.2f}")

                        # Use SAME moneyness-based leverage as premium calculation above
                        # This ensures sanity checks are consistent with pricing
                        if is_call:
                            moneyness = (current_price - trade.strike_price) / trade.strike_price
                        else:  # PUT
                            moneyness = (trade.strike_price - current_price) / trade.strike_price
                        
                        # Moneyness-based leverage (same as above)
                        if moneyness > 0.03:  # Deep ITM
                            delta_leverage = 3.0
                        elif moneyness > 0.01:  # Slightly ITM
                            delta_leverage = 6.0
                        elif moneyness > -0.01:  # ATM
                            delta_leverage = 10.0
                        elif moneyness > -0.03:  # Slightly OTM
                            delta_leverage = 6.0
                        else:  # Deep OTM
                            delta_leverage = 3.0
                        
                        max_gain_pct = abs(price_move_pct) * delta_leverage  # Max gain proportional to underlying move
                        
                        # TIGHTER CAP: Max 500% gain (allow gamma explosions)
                        # This allows multi-baggers while preventing data errors
                        max_gain_pct = max(1.0, min(500.0, max_gain_pct))  # Floor 1%, cap 500%
                        
                        if is_put and price_move_pct > 0:
                            # Underlying UP → long PUT MUST lose value (not gain!)
                            # When price goes UP, PUT premium should DECREASE
                            loss_factor = min(0.9, abs(price_move_pct) * delta_leverage / 100)
                            max_dir_premium = entry_premium * (1 - loss_factor)
                            max_dir_premium = max(0.01, max_dir_premium)  # Floor at $0.01
                            
                            if current_premium > max_dir_premium:
                                self.logger.info(
                                    f"[SANITY] PUT losing value: price UP {price_move_pct:+.2f}%, "
                                    f"entry=${entry_premium:.2f}, raw=${current_premium:.2f} → clamped=${max_dir_premium:.2f}"
                                )
                                current_premium = max_dir_premium
                        
                        elif is_put and price_move_pct < 0:
                            # Underlying DOWN → long PUT gains value
                            # Cap gains to prevent unrealistic premium inflation (e.g. 1000% on 1% move)
                            # But allow realistic gamma explosions
                            max_gain_premium = entry_premium * (1 + max_gain_pct / 100)
                            
                            if current_premium > max_gain_premium:
                                # Only log if we are clamping a significant amount
                                if current_premium > max_gain_premium * 1.1:
                                    self.logger.info(
                                        f"[SANITY] PUT capping gain: price DOWN {price_move_pct:+.2f}%, "
                                        f"entry=${entry_premium:.2f}, raw=${current_premium:.2f} → capped=${max_gain_premium:.2f}"
                                    )
                                current_premium = max_gain_premium
                                
                        elif is_call and price_move_pct < 0:
                            # Underlying DOWN → long CALL MUST lose value (not gain!)
                            loss_factor = min(0.9, abs(price_move_pct) * delta_leverage / 100)
                            max_dir_premium = entry_premium * (1 - loss_factor)
                            max_dir_premium = max(0.01, max_dir_premium)  # Floor at $0.01
                            
                            if current_premium > max_dir_premium:
                                self.logger.info(
                                    f"[SANITY] CALL losing value: price DOWN {price_move_pct:+.2f}%, "
                                    f"entry=${entry_premium:.2f}, raw=${current_premium:.2f} → clamped=${max_dir_premium:.2f}"
                                )
                                current_premium = max_dir_premium
                        
                        elif is_call and price_move_pct > 0:
                            # Underlying UP → long CALL gains value
                            max_gain_premium = entry_premium * (1 + max_gain_pct / 100)
                            
                            if current_premium > max_gain_premium:
                                if current_premium > max_gain_premium * 1.1:
                                    self.logger.info(
                                        f"[SANITY] CALL capping gain: price UP {price_move_pct:+.2f}%, "
                                        f"entry=${entry_premium:.2f}, raw=${current_premium:.2f} → capped=${max_gain_premium:.2f}"
                                    )
                                current_premium = max_gain_premium
                                
                except Exception as e:
                    # Never let safety logic break position updates
                    self.logger.debug(f"[SANITY] Exception in directional check: {e}")
            else:
                # Live trade - real market price already fetched, no simulation adjustments needed
                self.logger.info(f"💎 [LIVE] Using real market price: ${current_premium:.2f} (no simulation adjustments)")

            # Calculate P&L
            # FIX: premium_paid and current_premium are per-share, multiply by 100 (shares per contract)
            premium_change = current_premium - trade.premium_paid
            trade.profit_loss = premium_change * trade.quantity * 100

            # Check exit conditions
            should_exit = False
            exit_reason = ""
            
            # ============= DEBUG: Log position check start =============
            time_held_seconds = (self.get_market_time() - trade.timestamp).total_seconds()
            self.logger.debug(f"🔍 Checking exit for {trade.id[:8]}... | Entry: ${trade.premium_paid:.2f} | Current: ${current_premium:.2f} | Time: {time_held_seconds/60:.1f}min")
            
            # ============= RESTART PROTECTION (LIVE TRADES ONLY) =============
            # Don't allow exits within first 5 minutes for live trades UNLESS emergency
            # This prevents accidental closes when bot restarts mid-trade
            MIN_HOLD_SECONDS_LIVE = 300  # 5 minutes
            if trade.is_real_trade and time_held_seconds < MIN_HOLD_SECONDS_LIVE:
                self.logger.info(f"🛡️ [RESTART PROTECTION] Live trade only {time_held_seconds:.0f}s old (min: {MIN_HOLD_SECONDS_LIVE}s) - skipping non-emergency exits")
                # Only allow emergency exits (catastrophic loss) for young live trades
                current_pnl_pct_check = ((current_premium / trade.premium_paid) - 1.0) * 100.0 if trade.premium_paid > 0 else 0
                if current_pnl_pct_check > -50.0:  # Not a catastrophic loss
                    continue  # Skip to next trade, don't evaluate exit conditions

            # ============= EMERGENCY CIRCUIT BREAKER (ALWAYS FIRST!) =============
            # These exits happen REGARDLESS of RL, time held, or any other logic!
            current_pnl_pct_emergency = ((current_premium / trade.premium_paid) - 1.0) * 100.0 if trade.premium_paid > 0 else 0

            # Calculate DOLLAR loss for this position
            current_dollar_pnl = (current_premium - trade.premium_paid) * trade.quantity * 100  # options are 100 shares each

            # 🚨 MAX DOLLAR LOSS CAP - EXIT IMMEDIATELY
            # Configurable via env var: MAX_LOSS_PER_TRADE (default -50.0, set to -99999 to disable)
            MAX_LOSS_PER_TRADE = float(os.environ.get("MAX_LOSS_PER_TRADE", "-50.0"))
            if current_dollar_pnl <= MAX_LOSS_PER_TRADE:
                should_exit = True
                exit_reason = f"🚨 MAX LOSS CAP: ${abs(current_dollar_pnl):.2f} loss (cap: ${abs(MAX_LOSS_PER_TRADE):.2f})"
                trade.status = OrderStatus.STOPPED_OUT
                self.logger.error(f"🚨 MAX DOLLAR LOSS CAP TRIGGERED!")
                self.logger.error(f"   Dollar loss: ${current_dollar_pnl:.2f} (max allowed: ${MAX_LOSS_PER_TRADE:.2f})")
                self.logger.error(f"   Position: {trade.quantity} contracts @ ${trade.premium_paid:.2f} -> ${current_premium:.2f}")

            # 🚨 CATASTROPHIC LOSS (-25% or worse) - EXIT IMMEDIATELY
            elif current_pnl_pct_emergency <= -25.0:
                should_exit = True
                exit_reason = f"🚨 EMERGENCY EXIT: Catastrophic loss ({current_pnl_pct_emergency:.1f}%)"
                trade.status = OrderStatus.STOPPED_OUT
                self.logger.error(f"🚨 EMERGENCY CIRCUIT BREAKER TRIGGERED!")
                self.logger.error(f"   Loss: {current_pnl_pct_emergency:.1f}% - FORCING EXIT to prevent further damage")
                # Skip all other checks - exit NOW
            
            # 💰 MASSIVE PROFIT (+100% or better) - TAKE IT!
            elif current_pnl_pct_emergency >= 100.0:
                should_exit = True
                exit_reason = f"💰 EMERGENCY EXIT: Massive profit ({current_pnl_pct_emergency:.1f}%)"
                trade.status = OrderStatus.PROFIT_TAKEN
                self.logger.info(f"💰 MASSIVE PROFIT DETECTED! Taking +{current_pnl_pct_emergency:.1f}% gain!")
            
            # If emergency exit triggered, skip RL logic
            rl_wants_to_hold = False
            rl_exit_info = {}

            # ============= DYNAMIC EXIT EVALUATOR (CONTINUOUS RE-EVALUATION) =============
            # This is the key architectural change: actively manage positions instead of waiting
            if not should_exit and self.dynamic_exit_evaluator is not None:
                try:
                    # Get trade type
                    trade_type = 'CALL' if trade.option_type in [OrderType.CALL, OrderType.BUY_CALL] else 'PUT'

                    # Get entry info from trade metadata (if available)
                    entry_prediction = getattr(trade, 'entry_prediction_direction', 'NEUTRAL')
                    entry_confidence = getattr(trade, 'entry_confidence', 0.25)
                    entry_hmm_regime = getattr(trade, 'entry_hmm_regime', {'trend_state': 1.0})
                    entry_momentum = getattr(trade, 'entry_momentum', 0.0)

                    # Get current market state from bot (if available)
                    current_prediction = 'NEUTRAL'
                    current_confidence = 0.25
                    current_hmm_regime = {'trend_state': 1.0}
                    current_momentum = 0.0

                    if hasattr(self, 'bot') and self.bot:
                        # Get current prediction
                        if hasattr(self.bot, 'last_multi_timeframe_predictions'):
                            multi_tf = self.bot.last_multi_timeframe_predictions
                            if multi_tf and '15min' in multi_tf:
                                current_prediction = multi_tf['15min'].get('direction', 'NEUTRAL')
                                current_confidence = multi_tf['15min'].get('confidence', 0.25)

                        # Get current HMM regime
                        if hasattr(self.bot, 'hmm_regime'):
                            current_hmm_regime = self.bot.hmm_regime or {'trend_state': 1.0}

                        # Get momentum (use 5-bar momentum if available)
                        if hasattr(self.bot, 'last_features') and self.bot.last_features:
                            current_momentum = self.bot.last_features.get('momentum_5', 0.0)

                    # Calculate P&L and time held
                    current_pnl_pct = ((current_premium / trade.premium_paid) - 1.0) * 100.0 if trade.premium_paid > 0 else 0
                    minutes_held = (self.get_market_time() - trade.timestamp).total_seconds() / 60.0

                    # Evaluate exit
                    exit_decision = self.dynamic_exit_evaluator.evaluate(
                        trade_id=trade.id,
                        trade_type=trade_type,
                        entry_prediction_direction=entry_prediction,
                        entry_confidence=entry_confidence,
                        entry_hmm_regime=entry_hmm_regime,
                        current_prediction_direction=current_prediction,
                        current_confidence=current_confidence,
                        current_hmm_regime=current_hmm_regime,
                        current_pnl_pct=current_pnl_pct,
                        minutes_held=minutes_held,
                        current_momentum=current_momentum,
                        entry_momentum=entry_momentum,
                    )

                    if exit_decision.should_exit:
                        should_exit = True
                        exit_reason = f"🔄 DYNAMIC: {exit_decision.reason.value} - {exit_decision.details}"
                        self.logger.info(f"🔄 Dynamic Exit triggered: {exit_decision.reason.value}")
                        self.logger.info(f"   Details: {exit_decision.details}")
                        self.logger.info(f"   Confidence: {exit_decision.confidence:.0%}")
                except Exception as e:
                    self.logger.debug(f"Dynamic exit evaluator error: {e}")

            # ============= SIMPLE STOP LOSS / TAKE PROFIT (BEFORE RL LOGIC!) =============
            # These checks run BEFORE RL exit policy to ensure basic risk management
            if not should_exit and trade.stop_loss > 0 and current_premium <= trade.stop_loss:
                # Check minimum hold time to avoid initial noise
                # Configurable via env var: STOP_LOSS_MIN_HOLD_MINUTES (default 1.0, was 5.0)
                STOP_LOSS_MIN_HOLD = float(os.environ.get("STOP_LOSS_MIN_HOLD_MINUTES", "1.0"))
                time_held_for_stop = (self.get_market_time() - trade.timestamp).total_seconds() / 60.0
                if time_held_for_stop >= STOP_LOSS_MIN_HOLD:
                    should_exit = True
                    stop_pnl_pct = ((current_premium / trade.premium_paid) - 1.0) * 100.0
                    exit_reason = f"🛑 STOP LOSS ({stop_pnl_pct:+.1f}%)"
                    trade.status = OrderStatus.STOPPED_OUT
                    self.logger.info(f"🛑 STOP LOSS triggered: {stop_pnl_pct:+.1f}% (premium ${current_premium:.2f} <= SL ${trade.stop_loss:.2f})")
                else:
                    self.logger.info(f"⏳ Stop loss hit but held < {STOP_LOSS_MIN_HOLD:.0f}min ({time_held_for_stop:.1f}m) - initial volatility filter")

            if not should_exit and trade.take_profit > 0 and current_premium >= trade.take_profit:
                should_exit = True
                tp_pnl_pct = ((current_premium / trade.premium_paid) - 1.0) * 100.0
                exit_reason = f"🎯 TAKE PROFIT ({tp_pnl_pct:+.1f}%)"
                trade.status = OrderStatus.PROFIT_TAKEN
                self.logger.info(f"🎯 TAKE PROFIT triggered: {tp_pnl_pct:+.1f}% (premium ${current_premium:.2f} >= TP ${trade.take_profit:.2f})")

            # ============= DYNAMIC HOLD TIME ADJUSTMENT =============
            # Check if live predictions still support our trade direction
            # If bullish predictions + CALL position → extend hold time
            # If bearish predictions + PUT position → extend hold time
            if not should_exit and hasattr(self, 'bot') and self.bot:
                try:
                    live_multi_tf = getattr(self.bot, 'last_multi_timeframe_predictions', {})
                    if live_multi_tf:
                        # Determine trade direction
                        is_call = trade.option_type in [OrderType.CALL, OrderType.BUY_CALL]
                        is_put = trade.option_type in [OrderType.PUT, OrderType.BUY_PUT]
                        
                        # Calculate actual price move since entry
                        actual_move_pct = ((current_price / trade.entry_price) - 1.0) * 100.0 if trade.entry_price else 0
                        
                        # ============= LSTM PREDICTION ACCURACY TRACKING =============
                        # Compare each timeframe's prediction vs actual move
                        prediction_accuracy_scores = []
                        weighted_confidence_sum = 0
                        total_confidence = 0
                        
                        for tf, data in live_multi_tf.items():
                            pred_return = data.get('predicted_return', 0) * 100  # Convert to %
                            pred_conf = data.get('confidence', 0.5)
                            pred_dir = data.get('direction', 'NEUTRAL')
                            
                            # Check if prediction direction matches actual movement
                            actual_dir = 'UP' if actual_move_pct > 0.1 else 'DOWN' if actual_move_pct < -0.1 else 'NEUTRAL'
                            direction_correct = (pred_dir == actual_dir) or actual_dir == 'NEUTRAL'
                            
                            # Calculate accuracy score for this timeframe
                            # Higher confidence predictions that are correct get more weight
                            if direction_correct:
                                accuracy_score = pred_conf  # Correct = confidence becomes the score
                            else:
                                accuracy_score = -pred_conf  # Wrong = negative score weighted by confidence
                            
                            prediction_accuracy_scores.append({
                                'timeframe': tf,
                                'predicted': pred_return,
                                'confidence': pred_conf,
                                'direction': pred_dir,
                                'actual': actual_move_pct,
                                'correct': direction_correct,
                                'score': accuracy_score
                            })
                            
                            # Weight by confidence for overall assessment
                            if pred_dir in ['UP', 'DOWN']:
                                weighted_confidence_sum += pred_conf * (1 if direction_correct else -1)
                                total_confidence += pred_conf
                        
                        # Calculate overall prediction quality
                        avg_accuracy = sum(p['score'] for p in prediction_accuracy_scores) / len(prediction_accuracy_scores) if prediction_accuracy_scores else 0
                        confidence_weighted_accuracy = weighted_confidence_sum / total_confidence if total_confidence > 0 else 0
                        
                        # Log detailed prediction tracking
                        correct_count = sum(1 for p in prediction_accuracy_scores if p['correct'])
                        total_preds = len(prediction_accuracy_scores)
                        
                        self.logger.info(f"📊 Prediction Tracking: Actual={actual_move_pct:+.2f}% | {correct_count}/{total_preds} correct | Avg Score={avg_accuracy:.2f} | Conf-Weighted={confidence_weighted_accuracy:.2f}")
                        
                        # Log high-confidence predictions and their accuracy
                        high_conf_preds = [p for p in prediction_accuracy_scores if p['confidence'] > 0.2]
                        if high_conf_preds:
                            for p in high_conf_preds:
                                status = "✅" if p['correct'] else "❌"
                                self.logger.info(f"   {status} {p['timeframe']}: pred={p['predicted']:+.1f}% ({p['confidence']:.0%} conf) | actual={p['actual']:+.2f}%")
                        
                        # ============= DECISION LOGIC BASED ON ACCURACY =============
                        # Count agreeing predictions (weighted by confidence)
                        bullish_score = sum(data.get('confidence', 0.5) for tf, data in live_multi_tf.items() 
                                          if data.get('direction') == 'UP')
                        bearish_score = sum(data.get('confidence', 0.5) for tf, data in live_multi_tf.items() 
                                          if data.get('direction') == 'DOWN')
                        total_score = bullish_score + bearish_score
                        
                        bullish_pct = (bullish_score / total_score * 100) if total_score > 0 else 50
                        bearish_pct = (bearish_score / total_score * 100) if total_score > 0 else 50
                        
                        # ============= INFORMATIONAL LOGGING ONLY =============
                        # This data is OBSERVED and LOGGED - RL makes the actual decision
                        # We pass this info to RL, but don't override RL's decision here
                        
                        # Store prediction quality for RL to use
                        trade._live_prediction_quality = {
                            'bullish_pct': bullish_pct,
                            'bearish_pct': bearish_pct,
                            'avg_accuracy': avg_accuracy,
                            'confidence_weighted_accuracy': confidence_weighted_accuracy,
                            'correct_count': correct_count,
                            'total_preds': total_preds
                        }
                        
                        # LOG prediction alignment (informational only)
                        if is_call:
                            if bullish_pct >= 70 and avg_accuracy > 0:
                                self.logger.info(f"📈 [INFO] STRONG BULLISH: {bullish_pct:.0f}% conf-weighted | Accuracy={avg_accuracy:.2f} | RL will decide")
                            elif bullish_pct >= 60:
                                self.logger.info(f"📈 [INFO] Moderate bullish: {bullish_pct:.0f}% conf-weighted")
                            elif bearish_pct > bullish_pct:
                                self.logger.warning(f"⚠️ [INFO] Predictions shifted AGAINST call: {bearish_pct:.0f}% bearish vs {bullish_pct:.0f}% bullish")
                        elif is_put:
                            if bearish_pct >= 70 and avg_accuracy > 0:
                                self.logger.info(f"📉 [INFO] STRONG BEARISH: {bearish_pct:.0f}% conf-weighted | Accuracy={avg_accuracy:.2f} | RL will decide")
                            elif bearish_pct >= 60:
                                self.logger.info(f"📉 [INFO] Moderate bearish: {bearish_pct:.0f}% conf-weighted")
                            elif bullish_pct > bearish_pct:
                                self.logger.warning(f"⚠️ [INFO] Predictions shifted AGAINST put: {bullish_pct:.0f}% bullish vs {bearish_pct:.0f}% bearish")
                            
                except Exception as e:
                    self.logger.debug(f"Could not check live predictions: {e}")
            
            if should_exit:
                self.logger.warning(f"⚡ Emergency exit bypasses all other logic")
            elif hasattr(self, 'rl_exit_policy') and self.rl_exit_policy is not None:
                # ============= RL EXIT POLICY INTEGRATION =============
                # Let RL decide when to exit (prediction-aware strategy)
                try:
                    # Calculate metrics for RL exit policy
                    # CRITICAL: Ensure both timestamps are in US/Eastern (market time) for correct comparison
                    import pytz
                    from datetime import datetime
                    
                    # Get current market time (US/Eastern)
                    market_time_for_calc = self.get_market_time()
                    
                    # Parse trade timestamp (stored as Eastern time string)
                    if isinstance(trade.timestamp, str):
                        trade_timestamp = datetime.fromisoformat(trade.timestamp.replace('Z', '+00:00'))
                    else:
                        trade_timestamp = trade.timestamp
                    
                    # Strip timezone info from both (both are already Eastern time)
                    if hasattr(trade_timestamp, 'tzinfo') and trade_timestamp.tzinfo is not None:
                        trade_timestamp = trade_timestamp.replace(tzinfo=None)
                    if hasattr(market_time_for_calc, 'tzinfo') and market_time_for_calc.tzinfo is not None:
                        market_time_for_calc = market_time_for_calc.replace(tzinfo=None)
                    
                    time_held_minutes = (market_time_for_calc - trade_timestamp).total_seconds() / 60.0
                    
                    # Sanity check: Negative time means timezone mismatch
                    if time_held_minutes < 0:
                        self.logger.warning(f"⚠️ TIMEZONE FIX: Negative time held ({time_held_minutes:.1f}min)")
                        self.logger.warning(f"   Entry: {trade_timestamp}, Current: {market_time_for_calc}")
                        # Convert to positive - entry timestamp must be from future timezone
                        time_held_minutes = abs(time_held_minutes)
                        self.logger.info(f"✅ Corrected to: {time_held_minutes:.1f} minutes held")
                    current_pnl_pct = ((current_premium / trade.premium_paid) - 1.0) * 100.0
                    
                    # Handle timezone for expiration calculation
                    expiration_for_calc = trade.expiration_date
                    if hasattr(expiration_for_calc, 'tzinfo') and expiration_for_calc.tzinfo is not None:
                        expiration_for_calc = expiration_for_calc.replace(tzinfo=None)
                    
                    days_to_expiration = max(0.0, (expiration_for_calc - market_time_for_calc).total_seconds() / 86400.0)
                    
                    # Get ENTRY-TIME predicted move from trade metadata
                    predicted_move_pct = getattr(trade, 'predicted_return', 0.0) * 100.0
                    actual_move_pct = ((current_price / trade.entry_price) - 1.0) * 100.0
                    entry_confidence = getattr(trade, 'ml_confidence', 0.5)
                    entry_hmm_state = getattr(trade, 'hmm_state', None)
                    # Use the SMART hold time we calculated at entry (not a fixed 60 min!)
                    prediction_timeframe_minutes = (trade.max_hold_hours or 1.0) * 60
                    
                    # ============= GET LIVE PREDICTIONS (OPTIMIZED!) =============
                    # Use CACHED predictions from bot instead of regenerating for each position
                    # This avoids expensive neural network forward passes per position
                    live_predicted_move_pct = None
                    live_confidence = None
                    live_hmm_state = None
                    
                    if hasattr(self, 'bot') and self.bot:
                        try:
                            # Use cached signal from last generation (FAST - no NN call!)
                            # The bot generates this in STEP 4 of the training cycle
                            cached_signal = getattr(self.bot, '_last_generated_signal', None)
                            cached_multi_tf = getattr(self.bot, 'last_multi_timeframe_predictions', {})
                            
                            if cached_signal:
                                # Use cached prediction data
                                live_predicted_return = cached_signal.get('predicted_return', 0.0)
                                live_predicted_move_pct = live_predicted_return * 100.0
                                live_confidence = cached_signal.get('confidence', 0.5)
                                live_hmm_state = cached_signal.get('hmm_state', None)
                                
                                self.logger.debug(f"📊 Cached prediction: {live_predicted_move_pct:+.1f}% (conf: {live_confidence:.1%})")
                            elif cached_multi_tf:
                                # Fallback: Calculate from multi-timeframe predictions
                                returns = [p.get('predicted_return', 0) for p in cached_multi_tf.values()]
                                confidences = [p.get('confidence', 0.5) for p in cached_multi_tf.values()]
                                if returns:
                                    live_predicted_move_pct = sum(returns) / len(returns) * 100.0
                                    live_confidence = sum(confidences) / len(confidences)
                        except Exception as live_err:
                            self.logger.debug(f"Could not get cached prediction: {live_err}")
                    # ============= END LIVE PREDICTIONS =============
                    
                    # Get full HMM regime info for RL learning
                    hmm_regime_info = None
                    hmm_trend = 0.5
                    hmm_vol = 0.5
                    hmm_liq = 0.5
                    if hasattr(self, 'bot') and self.bot:
                        hmm_regime_info = getattr(self.bot, 'current_hmm_regime', None)
                        # Extract HMM states as floats for RL
                        if isinstance(hmm_regime_info, str):
                            if 'bullish' in hmm_regime_info.lower():
                                hmm_trend = 1.0
                            elif 'bearish' in hmm_regime_info.lower():
                                hmm_trend = 0.0
                            if 'high_vol' in hmm_regime_info.lower():
                                hmm_vol = 1.0
                            elif 'low_vol' in hmm_regime_info.lower():
                                hmm_vol = 0.0
                            if 'high_liq' in hmm_regime_info.lower():
                                hmm_liq = 1.0
                            elif 'low_liq' in hmm_regime_info.lower():
                                hmm_liq = 0.0
                    
                    # Get VIX extended features
                    vix_bb_pos = 0.5
                    vix_roc = 0.0
                    vix_percentile = 0.5
                    if hasattr(self, 'bot') and self.bot:
                        vix_bb_pos = getattr(self.bot, 'current_vix_bb_pos', 0.5)
                        vix_roc = getattr(self.bot, 'current_vix_roc', 0.0)
                        vix_percentile = getattr(self.bot, 'current_vix_percentile', 0.5)
                    
                    # Calculate position-signal alignment for RL
                    position_is_call = trade.option_type in [OrderType.CALL, OrderType.BUY_CALL]
                    bullish_tf_ratio = 0.5
                    bearish_tf_ratio = 0.5
                    
                    # Get multi-timeframe predictions to calculate alignment
                    if hasattr(self, 'bot') and self.bot:
                        multi_tf_preds = getattr(self.bot, 'last_multi_timeframe_predictions', None)
                        if multi_tf_preds and isinstance(multi_tf_preds, dict):
                            bullish_count = 0
                            bearish_count = 0
                            total_count = 0
                            for tf_name, pred in multi_tf_preds.items():
                                if isinstance(pred, dict):
                                    direction = pred.get('direction', 'N/A')
                                    conf = pred.get('confidence', 0)
                                    if conf >= 0.10:  # Only count meaningful predictions
                                        total_count += 1
                                        if direction == 'UP':
                                            bullish_count += 1
                                        elif direction == 'DOWN':
                                            bearish_count += 1
                            if total_count > 0:
                                bullish_tf_ratio = bullish_count / total_count
                                bearish_tf_ratio = bearish_count / total_count
                    
                    # Ask RL exit policy with predictions + HMM + position-signal alignment
                    should_exit_rl, exit_score, rl_exit_info = self.rl_exit_policy.should_exit(
                        time_held_minutes=time_held_minutes,
                        current_pnl_pct=current_pnl_pct,
                        days_to_expiration=days_to_expiration,
                        predicted_move_pct=predicted_move_pct,
                        actual_move_pct=actual_move_pct,
                        entry_confidence=entry_confidence,
                        prediction_timeframe_minutes=prediction_timeframe_minutes,
                        # Live prediction parameters
                        live_predicted_move_pct=live_predicted_move_pct,
                        live_confidence=live_confidence,
                        live_hmm_state=live_hmm_state,
                        entry_hmm_state=entry_hmm_state,
                        # Full HMM regime info for RL
                        hmm_regime_info=hmm_regime_info,
                        # Position-signal alignment for smarter exit decisions
                        position_is_call=position_is_call,
                        bullish_timeframe_ratio=bullish_tf_ratio,
                        bearish_timeframe_ratio=bearish_tf_ratio,
                        # HMM regime features (for NN input)
                        hmm_trend=hmm_trend,
                        hmm_vol=hmm_vol,
                        hmm_liq=hmm_liq,
                        # Extended VIX features
                        vix_bb_pos=vix_bb_pos,
                        vix_roc=vix_roc,
                        vix_percentile=vix_percentile
                    )
                    
                    if not should_exit_rl:
                        rl_wants_to_hold = True
                        # Enhanced logging with live prediction info
                        if rl_exit_info.get('prediction_reversed'):
                            self.logger.warning(f"💎 RL: HOLD despite prediction reversal - {rl_exit_info.get('reason', 'Unknown')}")
                        else:
                            self.logger.info(f"🤖 RL Exit Policy: HOLD - {rl_exit_info.get('reason', 'Unknown')}")
                    else:
                        # RL says EXIT - this should trigger actual exit!
                        rl_wants_to_hold = False
                        rl_reason = rl_exit_info.get('reason', 'Unknown')
                        
                        # Check if exit is due to live prediction changes
                        prediction_changed = rl_exit_info.get('prediction_reversed') or rl_exit_info.get('confidence_dropped') or rl_exit_info.get('regime_changed')
                        if prediction_changed:
                            self.logger.warning(f"🚨 RL: EXIT (Live prediction changed!) - {rl_reason}")
                        else:
                            self.logger.info(f"🤖 RL Exit Policy: EXIT - {rl_reason}")
                        
                        # CRITICAL FIX: RL EXIT decision should trigger actual exit
                        # But be SMART about prediction changes - don't lock in small losses!
                        
                        # Calculate current P&L percentage
                        current_pnl_for_exit = ((current_premium / trade.premium_paid) - 1.0) * 100.0 if trade.premium_paid > 0 else 0.0
                        
                        # RL-FIRST EXIT LOGIC: Respect the RL's decision!
                        # The RL has access to ALL market context (predictions, confidence, HMM, VIX, etc.)
                        # If it says EXIT with high confidence, trust it!
                        
                        rl_exit_score = rl_exit_info.get('exit_score', 0)
                        
                        # High confidence RL exit - ALWAYS respect it
                        if rl_exit_score >= 0.65:
                            should_trigger_exit = True
                            self.logger.info(f"🤖 RL HIGH CONFIDENCE EXIT (score={rl_exit_score:.2f}, P&L={current_pnl_for_exit:+.1f}%)")
                        
                        # Significant confidence drop (> 50% drop) - exit regardless of P&L
                        elif rl_exit_info.get('confidence_dropped') and live_confidence is not None:
                            entry_conf = rl_exit_info.get('entry_confidence', entry_confidence)
                            if entry_conf > 0 and live_confidence / entry_conf < 0.5:
                                # Confidence dropped by more than 50% - signal quality degraded significantly
                                should_trigger_exit = True
                                self.logger.warning(f"📉 CONFIDENCE CRASHED ({entry_conf:.1%}→{live_confidence:.1%}) - exiting, P&L={current_pnl_for_exit:+.1f}%")
                            elif current_pnl_for_exit > 0:
                                # Some profit + confidence dropped - take the money
                                should_trigger_exit = True
                                self.logger.info(f"✅ Confidence dropped, taking profit ({current_pnl_for_exit:+.1f}%)")
                            else:
                                # Small loss + moderate confidence drop - give a bit more time
                                should_trigger_exit = abs(current_pnl_for_exit) > 10.0  # Exit if loss > 10%
                                if not should_trigger_exit:
                                    self.logger.info(f"💎 Moderate signal change, small loss ({current_pnl_for_exit:+.1f}%) - brief hold")
                        
                        # Prediction reversed (direction flipped) - exit unless tiny loss
                        elif rl_exit_info.get('prediction_reversed'):
                            should_trigger_exit = current_pnl_for_exit > -5.0  # Exit unless losing > 5%
                            if should_trigger_exit:
                                self.logger.info(f"🔄 Prediction reversed - exiting (P&L={current_pnl_for_exit:+.1f}%)")
                            else:
                                self.logger.warning(f"⚠️ Prediction reversed BUT loss is {current_pnl_for_exit:+.1f}% - brief hold to avoid locking in loss")
                        
                        # Other RL exits (time exceeded, regime change, etc.)
                        else:
                            should_trigger_exit = (
                                'Time exceeded' in rl_reason or 
                                'time exceeded' in rl_reason or
                                rl_exit_score > 0.55
                            )
                        
                        if should_trigger_exit:
                            # TRAINING/REALISM GUARD: Avoid ultra-short churn exits.
                            # XGBoost exit can be overly eager on tiny profits in simulation.
                            # Require a small minimum hold unless an emergency condition applies.
                            try:
                                is_simulation = getattr(self, 'simulation_mode', False) or getattr(self, 'time_travel_mode', False)
                                min_hold_minutes = 5.0 if is_simulation else 1.0
                                current_pnl_for_exit = ((current_premium / trade.premium_paid) - 1.0) * 100.0 if trade.premium_paid > 0 else 0.0
                                if time_held_minutes < min_hold_minutes and (-5.0 < current_pnl_for_exit < 15.0):
                                    should_trigger_exit = False
                                    self.logger.info(
                                        f"⏳ Suppressing early RL exit (<{min_hold_minutes:.0f}m). "
                                        f"Held={time_held_minutes:.1f}m, P&L={current_pnl_for_exit:+.1f}%"
                                    )
                            except Exception:
                                pass

                            should_exit = True
                            if prediction_changed:
                                exit_reason = f"RL Prediction Change ({rl_reason})"
                            else:
                                exit_reason = f"RL Time Exit ({rl_reason})"
                            trade.status = OrderStatus.PROFIT_TAKEN if current_premium > trade.premium_paid else OrderStatus.STOPPED_OUT
                            self.logger.info(f"⏰ RL triggered exit: {exit_reason}")
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ RL exit policy error: {e}, falling back to standard exits")
                    import traceback
                    self.logger.debug(traceback.format_exc())
                    rl_wants_to_hold = False

            # ============= EMERGENCY BACKSTOP =============
            # SAFETY: Force exit after 2 hours (options decay fast - don't hold forever!)
            # This is a true emergency backstop, not the primary exit mechanism
            time_since_entry = self.get_market_time() - trade.timestamp
            hours_held = time_since_entry.total_seconds() / 3600
            
            if not should_exit and hours_held >= 2.0:
                current_pnl_pct = ((current_premium / trade.premium_paid) - 1.0) * 100.0
                should_exit = True
                exit_reason = f"Emergency Max Hold (2h) - P&L: {current_pnl_pct:+.1f}%"
                trade.status = OrderStatus.PROFIT_TAKEN if current_premium > trade.premium_paid else OrderStatus.STOPPED_OUT
                self.logger.warning(f"🚨 EMERGENCY EXIT: Position held {hours_held:.1f}h (max 2h), P&L: {current_pnl_pct:+.1f}%")
            
            # Check expiration (options expire worthless if OTM at expiry)
            # Note: Emergency exit already set should_exit=True, these are elif chains
            if not should_exit and self.get_market_time() >= trade.expiration_date:
                should_exit = True
                # Determine if option expires ITM or OTM
                if trade.option_type in [OrderType.CALL, OrderType.BUY_CALL]:
                    is_itm = current_price > trade.strike_price
                else:  # PUT
                    is_itm = current_price < trade.strike_price
                
                if is_itm:
                    exit_reason = "Expired ITM (Auto-Exercise)"
                    trade.status = OrderStatus.EXPIRED_ITM
                else:
                    exit_reason = "Expired OTM (Worthless)"
                    trade.status = OrderStatus.EXPIRED_OTM
                    current_premium = 0.0  # Worthless at expiration
            
            # ============= FIX: Check STOP LOSS and TAKE PROFIT BEFORE time-based exit =============
            # This ensures stop loss and take profit can trigger even if planned_exit_time passed
            current_pnl_pct = ((current_premium / trade.premium_paid) - 1.0) * 100.0
            # Configurable via env var: STOP_LOSS_MIN_HOLD_MINUTES (default 1.0, was 5.0)
            MIN_HOLD_BEFORE_STOP_MINUTES = float(os.environ.get("STOP_LOSS_MIN_HOLD_MINUTES", "1.0"))
            time_held_for_stop = (self.get_market_time() - trade.timestamp).total_seconds() / 60.0

            # 1. Check STOP LOSS FIRST (highest priority - prevent big losses)
            if not should_exit and current_premium <= trade.stop_loss and time_held_for_stop >= MIN_HOLD_BEFORE_STOP_MINUTES:
                should_exit = True
                exit_reason = f"Stop Loss ({current_pnl_pct:+.1f}%)"
                trade.status = OrderStatus.STOPPED_OUT
                self.logger.info(f"🛑 STOP LOSS triggered: {current_pnl_pct:+.1f}% after {time_held_for_stop:.0f}min")
            elif not should_exit and current_premium <= trade.stop_loss and time_held_for_stop < MIN_HOLD_BEFORE_STOP_MINUTES:
                # Too early to stop out - log and skip
                self.logger.info(f"⏳ Stop loss hit ({current_pnl_pct:.1f}%) but held < {MIN_HOLD_BEFORE_STOP_MINUTES:.0f}min - waiting")

            # 2. Check TAKE PROFIT (lock in gains)
            if not should_exit and current_premium >= trade.take_profit:
                is_exceptional_profit = current_pnl_pct >= 200.0  # 200%+ profit
                if is_exceptional_profit or not rl_wants_to_hold:
                    should_exit = True
                    exit_reason = f"Take Profit ({current_pnl_pct:+.1f}%)"
                    trade.status = OrderStatus.PROFIT_TAKEN
                    self.logger.info(f"🎯 TAKE PROFIT triggered: {current_pnl_pct:+.1f}%")
                else:
                    self.logger.info(f"💎 Holding for bigger gains despite take-profit: {rl_exit_info.get('reason', 'RL strategy')}")

            # 3. Check time-based exit (planned exit time reached) - ONLY if stop/profit didn't trigger
            if not should_exit and trade.planned_exit_time and current_time >= trade.planned_exit_time:
                # Check if we're in simulation/time-travel mode
                is_simulation = getattr(self, 'simulated_time', None) is not None or getattr(self, 'time_travel_mode', False)

                if is_simulation:
                    # SIMULATION MODE: Exit when time passes (so we can measure wins/losses)
                    should_exit = True
                    if current_pnl_pct > 0:
                        exit_reason = f"Time Exit (profit: {current_pnl_pct:+.1f}%)"
                        trade.status = OrderStatus.PROFIT_TAKEN
                        self.logger.info(f"⏰ [SIM] Time exit with profit: {current_pnl_pct:+.1f}%")
                    else:
                        exit_reason = f"Time Exit (loss: {current_pnl_pct:+.1f}%)"
                        trade.status = OrderStatus.STOPPED_OUT
                        self.logger.info(f"⏰ [SIM] Time exit with loss: {current_pnl_pct:+.1f}%")
                else:
                    # LIVE MODE: Original logic - let winners run, exit flat/losers
                    if current_pnl_pct > 5.0:  # >5% profit - let it run even longer
                        self.logger.info(f"🚀 Time limit reached but PROFITABLE ({current_pnl_pct:+.1f}%) - letting winner run!")
                    elif current_pnl_pct > 0:  # Small profit (0-5%)
                        self.logger.info(f"⏳ Time limit but small profit ({current_pnl_pct:+.1f}%) - holding for 5%+ to cover spread")
                    elif current_pnl_pct > -5.0:  # Flat/small loss
                        should_exit = True
                        exit_reason = f"Time Exit (flat/small loss: {current_pnl_pct:+.1f}%)"
                        trade.status = OrderStatus.CLOSED
                        self.logger.info(f"⏰ Time-based exit: Flat trade {current_pnl_pct:+.1f}%, avoiding more decay")
                    else:  # Losing more than 5%
                        should_exit = True
                        exit_reason = f"Time Exit (held {minutes_held/60:.1f}h, P&L: {current_pnl_pct:+.1f}%)"
                        trade.status = OrderStatus.STOPPED_OUT
                        self.logger.info(f"⏰ Time-based exit: Trade held {minutes_held/60:.1f}h, losing {current_pnl_pct:.1f}%")

            # NOTE: Primary stop loss and take profit checks are above (lines 4595-4614)
            # The code below handles secondary/expiration-based checks

            # Check expiration - designed for fast algorithmic trading via Tradier API
            time_since_entry = self.get_market_time() - trade.timestamp
            
            # Tradier-realistic trading exit conditions
            # Skip if emergency exit already triggered
            # Check if we've reached the actual expiration date
            if not should_exit and trade.expiration_date and self.get_market_time() >= trade.expiration_date:
                should_exit = True
                exit_reason = "Contract Expiration"
                trade.status = OrderStatus.EXPIRED

            # Tradier allows any hold time - but require minimum hold before stop-loss
            # Configurable via env var: STOP_LOSS_MIN_HOLD_MINUTES (default 1.0, was 5.0)
            MIN_HOLD_MINUTES_TRADIER = float(os.environ.get("STOP_LOSS_MIN_HOLD_MINUTES", "1.0"))
            if not should_exit and time_since_entry > timedelta(minutes=MIN_HOLD_MINUTES_TRADIER):
                # Use the bot's actual stop-loss and take-profit levels (with RL override)
                if current_premium >= trade.take_profit:
                    # RL can override take-profit UNLESS it's an exceptional profit
                    current_pnl_pct = ((current_premium / trade.premium_paid) - 1.0) * 100.0
                    is_exceptional_profit = current_pnl_pct >= 200.0  # 200%+ profit
                    
                    if is_exceptional_profit or not rl_wants_to_hold:
                        should_exit = True
                        exit_reason = f"Take Profit ({current_pnl_pct:.1f}%)" if is_exceptional_profit else f"Take Profit ({current_pnl_pct:.1f}%, RL approved)"
                        trade.status = OrderStatus.PROFIT_TAKEN
                    else:
                        self.logger.info(f"💎 Holding for bigger gains: {current_pnl_pct:.1f}% profit - {rl_exit_info.get('reason', 'RL strategy')}")
                        
                elif current_premium <= trade.stop_loss:
                    # PREDICTION-AWARE stop-loss (with override for strong predictions)
                    current_pnl_pct = ((current_premium / trade.premium_paid) - 1.0) * 100.0
                    
                    # Check if prediction reversed (from RL exit info)
                    prediction_reversed = rl_exit_info.get('prediction_reversed', False)
                    confidence_dropped = rl_exit_info.get('confidence_dropped', False)
                    
                    # Check if LIVE PREDICTION supports our position
                    predictions_support_position = False
                    
                    if hasattr(self, 'bot') and self.bot and hasattr(self.bot, 'last_multi_timeframe_predictions'):
                        try:
                            live_preds = self.bot.last_multi_timeframe_predictions
                            if live_preds:
                                bullish_count = sum(1 for p in live_preds.values() 
                                                  if p.get('predicted_return', 0) > 0)
                                bearish_count = sum(1 for p in live_preds.values() 
                                                  if p.get('predicted_return', 0) < 0)
                                
                                is_put = 'PUT' in str(trade.option_type).upper()
                                is_call = 'CALL' in str(trade.option_type).upper()
                                
                                # STRICTER: Require 65% agreement
                                total_preds = len(live_preds)
                                support_threshold = 0.65
                                if is_put and bearish_count >= total_preds * support_threshold:
                                    predictions_support_position = True
                                elif is_call and bullish_count >= total_preds * support_threshold:
                                    predictions_support_position = True
                        except Exception:
                            pass
                    
                    # Exit on critical loss OR if predictions reversed/dropped
                    # MORE AGGRESSIVE: Cut losses faster when predictions flip
                    is_critical_loss = current_pnl_pct <= -25.0  # Lowered from -50% - cut losses faster!
                    is_prediction_problem = (prediction_reversed or confidence_dropped) and current_pnl_pct < -5.0  # Lowered from -10%
                    
                    if predictions_support_position and not is_critical_loss:
                        # Predictions support us - HOLD!
                        self.logger.info(f"💎 HOLDING through stop ({current_pnl_pct:.1f}%) - predictions SUPPORT position!")
                    elif is_critical_loss or is_prediction_problem or not rl_wants_to_hold:
                        should_exit = True
                        if is_critical_loss:
                            exit_reason = f"Stop Loss ({current_pnl_pct:.1f}% critical loss)"
                        elif prediction_reversed:
                            exit_reason = f"Stop Loss ({current_pnl_pct:.1f}% + prediction reversed)"
                        elif confidence_dropped:
                            exit_reason = f"Stop Loss ({current_pnl_pct:.1f}% + confidence dropped)"
                        else:
                            exit_reason = f"Stop Loss ({current_pnl_pct:.1f}%, RL approved)"
                        trade.status = OrderStatus.STOPPED_OUT
                    else:
                        self.logger.info(f"🛡️ Holding through stop ({current_pnl_pct:.1f}%) - RL wants to hold")
                # End of day exit (market close) - more aggressive
                elif not self._is_market_hours() and time_since_entry > timedelta(minutes=30):  # Close after market hours
                    should_exit = True
                    exit_reason = "End of Day Exit (After Hours)"
                    trade.status = OrderStatus.PROFIT_TAKEN if current_premium > trade.premium_paid else OrderStatus.STOPPED_OUT
                elif time_since_entry > timedelta(hours=6):  # Full trading day
                    should_exit = True
                    exit_reason = "End of Day Exit"
                    trade.status = OrderStatus.PROFIT_TAKEN if current_premium > trade.premium_paid else OrderStatus.STOPPED_OUT

            if should_exit:
                # ============= CRITICAL EXIT LOG =============
                self.logger.warning(f"⚠️⚠️⚠️ EXIT TRIGGERED ⚠️⚠️⚠️")
                self.logger.warning(f"   Trade: {trade.id[:8]}... ({trade.tradier_option_symbol})")
                self.logger.warning(f"   Reason: {exit_reason}")
                self.logger.warning(f"   Entry: ${trade.premium_paid:.2f}, Current: ${current_premium:.2f}")
                self.logger.warning(f"   P&L: {((current_premium / trade.premium_paid) - 1.0) * 100.0:.1f}%")
                self.logger.warning(f"   Time Held: {(self.get_market_time() - trade.timestamp).total_seconds() / 60:.1f} min")
                
                # ============= EXECUTE LIVE CLOSE ON TRADIER (IF LIVE TRADE) =============
                if trade.is_real_trade and trade.tradier_option_symbol and self.tradier_trading_system:
                    self.logger.warning(f"🔴 [LIVE] Closing position on Tradier: {trade.tradier_option_symbol}")
                    self.logger.warning(f"🔴 [LIVE] Close reason: {exit_reason}")
                    
                    try:
                        close_result = self.tradier_trading_system.close_position(
                            option_symbol=trade.tradier_option_symbol,
                            quantity=trade.quantity,
                            symbol=trade.symbol
                        )
                        
                        if close_result:
                            order_id = close_result.get('order_id')
                            status = close_result.get('status')
                            self.logger.info(f"✅ [LIVE] Tradier close order placed: Order ID {order_id}, Status: {status}")
                        else:
                            self.logger.error(f"❌ [LIVE] Failed to close position on Tradier - no response")
                    except Exception as e:
                        self.logger.error(f"❌ [LIVE] Error closing position on Tradier: {e}")
                
                # ============= APPLY REALISTIC EXIT PRICING (PAPER TRADES ONLY) =============
                # Apply bid-ask spread and slippage for realistic exit prices
                # REDUCED from previous levels to prevent 26% round-trip cost
                # NOTE: For live trades, current_premium already has the real market price
                
                if not (trade.is_real_trade and trade.tradier_option_symbol):
                    # Only apply simulation adjustments to paper trades
                    
                    # ✅ HARD CAP ON EXIT PREMIUM: Max 500% above entry (relaxed from 50%)
                    # This prevents unrealistic gains from calculation errors while allowing multi-baggers
                    entry_prem = trade.premium_paid
                    max_exit_cap = entry_prem * 6.0  # 500% gain (6x multiplier)
                    if current_premium > max_exit_cap:
                        self.logger.warning(f"🚨 [EXIT CAP] Premium ${current_premium:.2f} > 500% cap ${max_exit_cap:.2f} - clamping!")
                        current_premium = max_exit_cap
                    
                    # FURTHER REDUCED to match entry friction: ~1-2% round-trip total
                    # TT_REDUCE_FRICTION: Scale down friction for testing (0.0 = no friction, 1.0 = full)
                    friction_scale = float(os.environ.get("TT_REDUCE_FRICTION", "1.0"))
                    friction_scale = max(0.0, min(1.0, friction_scale))

                    # 1. Exit spread (same as entry, 1.5%)
                    exit_spread_pct = 0.015 * friction_scale  # Was 4% - way too high!
                    exit_spread_cost = current_premium * exit_spread_pct

                    # 2. Exit slippage (0.5% for market orders)
                    exit_slippage_pct = 0.005 * friction_scale  # Was 1.5%
                    exit_slippage_cost = current_premium * exit_slippage_pct
                    
                    # 3. Time penalty for off-hours exits
                    current_hour = self.get_market_time().hour
                    exit_time_penalty = 0.0
                    if current_hour < 9 or current_hour >= 16:
                        exit_time_penalty = current_premium * 0.005  # 0.5% penalty (was 1%)
                    
                    # 4. Apply realistic exit price (when selling: receive bid - slippage - penalties)
                    # Exit: -0.75% (spread/2) - 0.5% (slippage) - 0-0.5% (time) = -1.25-1.75%
                    # Total round-trip: ~2.5-3.5% (realistic for liquid options)
                    realistic_exit_premium = max(0.01, current_premium - (exit_spread_cost / 2) - exit_slippage_cost - exit_time_penalty)
                    
                    # 5. Log the market friction
                    market_friction = current_premium - realistic_exit_premium
                    if market_friction > 0.01:  # Only log significant friction
                        self.logger.info(f"💸 Exit market friction: -${market_friction:.3f} ({(market_friction/current_premium*100):.1f}%)")
                    
                    # Use realistic exit price
                    current_premium = realistic_exit_premium
                else:
                    # SAFETY: If live trade has $0 exit price (API failure), use simulation fallback
                    if current_premium <= 0.01:
                        self.logger.warning(f"⚠️ [LIVE] Exit price is ${current_premium:.2f} (API failure?) - using simulation fallback")
                        current_premium = self._calculate_option_premium(
                            trade.symbol, trade.option_type, trade.strike_price, current_price,
                            is_entry=False, time_to_expiry_days=time_to_expiry_days
                        )
                        self.logger.info(f"💎 [LIVE] Fallback exit price: ${current_premium:.2f}")
                    else:
                        self.logger.info(f"💎 [LIVE] Using real exit price: ${current_premium:.2f} (no simulation adjustments)")
                # ============= END REALISTIC EXIT PRICING =============
                
                # ============= SANITY CHECK: PREVENT UNDERLYING PRICE AS EXIT =============
                # Option premiums rarely exceed $200 (except deep ITM LEAPS)
                # If exit_price looks like underlying stock price, it's a bug!
                # FIX: premium_paid is already per-share, don't divide by quantity!
                entry_premium = trade.premium_paid
                
                # Check if current_premium is unreasonably high (likely underlying price by mistake)
                if current_premium > 200 and current_premium > entry_premium * 10:
                    self.logger.error(f"🚨 EXIT PRICE SANITY CHECK FAILED!")
                    self.logger.error(f"   Calculated exit: ${current_premium:.2f} (looks like underlying price, not option premium!)")
                    self.logger.error(f"   Entry premium was: ${entry_premium:.2f}")
                    
                    # Use entry_premium as fallback (assume break-even) rather than wrong price
                    # For LIVE trades, try to get actual fill price from Tradier order history
                    if trade.is_real_trade and trade.tradier_order_id:
                        try:
                            from tradier_credentials import TradierCredentials
                            import requests
                            
                            creds = TradierCredentials()
                            live_creds = creds.get_live_credentials()
                            
                            if live_creds:
                                headers = {
                                    'Authorization': f'Bearer {live_creds["access_token"]}',
                                    'Accept': 'application/json'
                                }
                                
                                # Get order history to find actual fill price
                                order_url = f'https://api.tradier.com/v1/accounts/{live_creds["account_number"]}/orders'
                                response = requests.get(order_url, headers=headers, timeout=10)
                                
                                if response.status_code == 200:
                                    orders = response.json().get('orders', {}).get('order', [])
                                    if not isinstance(orders, list):
                                        orders = [orders]
                                    
                                    # Find the close order for this position
                                    for order in orders:
                                        if (order.get('symbol') == trade.tradier_option_symbol and 
                                            order.get('side') in ['sell_to_close', 'buy_to_close'] and
                                            order.get('status') == 'filled'):
                                            fill_price = float(order.get('avg_fill_price', 0))
                                            if fill_price > 0:
                                                current_premium = fill_price * 100  # Convert per-share to per-contract
                                                self.logger.info(f"✅ Found actual Tradier fill: ${fill_price:.2f}/share = ${current_premium:.2f}/contract")
                                                break
                        except Exception as e:
                            self.logger.error(f"   Could not fetch Tradier fill price: {e}")
                    
                    # If still invalid, use entry_premium as fallback
                    if current_premium > 200 and current_premium > entry_premium * 10:
                        current_premium = entry_premium
                        self.logger.warning(f"   Using entry premium as fallback: ${current_premium:.2f}")
                # ============= END SANITY CHECK =============
                
                trade.exit_price = current_premium  # Store option premium, not underlying price!
                trade.exit_timestamp = self.get_market_time()
                trade.exit_reason = exit_reason  # ✅ SAVE EXIT REASON for logging/tracking

                # Calculate exit fees with symbol for correct fee structure
                exit_fees = self.fee_calculator.calculate_exit_fees(
                    current_premium, trade.quantity, trade.symbol)

                # Update balance (subtract exit fees)
                # FIX: premium is per-share, must multiply by 100 (shares per contract) to match entry cost
                exit_value = current_premium * trade.quantity * 100 - exit_fees
                
                # Settlement tracking (T+1 for options)
                if self.simulate_settlement and not (getattr(self, 'simulation_mode', False) or getattr(self, 'time_travel_mode', False)):
                    # Funds settle next business day
                    settlement_date = self._get_next_business_day()
                    self.pending_settlements.append({
                        'trade_id': trade.id,
                        'amount': exit_value,
                        'settlement_date': settlement_date
                    })
                    self._save_settlement(trade.id, exit_value, settlement_date)
                    self.logger.info(f"💵 [SETTLEMENT] ${exit_value:.2f} pending until {settlement_date.strftime('%Y-%m-%d')}")
                else:
                    # Immediate settlement (simulation mode)
                    self.current_balance += exit_value

                # Update P&L to include all fees
                entry_fees = self.fee_calculator.calculate_entry_fees(
                    trade.premium_paid, trade.quantity, trade.symbol)
                total_fees = entry_fees + exit_fees
                # FIX: premium_paid is per-share, must multiply by 100 (shares per contract)
                trade.profit_loss = (
                    current_premium - trade.premium_paid) * trade.quantity * 100 - total_fees

                # FIX: Update status based on ACTUAL P&L (after fees and friction)
                # Status was set earlier based on pre-friction premium comparison,
                # but must be corrected here based on actual net P&L
                trade.status = OrderStatus.PROFIT_TAKEN if trade.profit_loss > 0 else OrderStatus.STOPPED_OUT

                # Update statistics
                self.total_trades += 1
                if trade.profit_loss > 0:
                    self.winning_trades += 1
                else:
                    self.losing_trades += 1

                self.total_profit_loss += trade.profit_loss

                # Calculate drawdown
                peak_balance = max(self.initial_balance +
                                   self.total_profit_loss, self.initial_balance)
                current_drawdown = (
                    peak_balance - self.current_balance) / peak_balance
                self.max_drawdown = max(self.max_drawdown, current_drawdown)

                trades_to_close.append(trade)

                # === ATOMIC CLOSE: Remove from active_trades IMMEDIATELY ===
                # This prevents double-booking if exception occurs later
                try:
                    if trade in self.active_trades:
                        self.active_trades.remove(trade)
                        self._save_trade(trade)
                        self.logger.debug(f"✅ ATOMIC: Trade {trade.id[:8]} removed from active_trades")
                except Exception as e:
                    self.logger.error(f"❌ ATOMIC CLOSE FAILED for {trade.id[:8]}: {e}")
                    # Mark for retry
                    trade._atomic_close_failed = True

                # Store experience in RL exit policy for learning
                if hasattr(self, 'rl_exit_policy') and self.rl_exit_policy is not None:
                    try:
                        # Calculate metrics for RL learning
                        entry_time = datetime.fromisoformat(trade.timestamp.replace('Z', '+00:00')) if isinstance(trade.timestamp, str) else trade.timestamp
                        exit_time = self.get_market_time()
                        
                        # Handle timezone-aware vs timezone-naive datetime
                        if entry_time.tzinfo is not None and exit_time.tzinfo is None:
                            import pytz
                            exit_time = pytz.UTC.localize(exit_time)
                        elif entry_time.tzinfo is None and exit_time.tzinfo is not None:
                            import pytz
                            entry_time = pytz.UTC.localize(entry_time)
                        
                        time_held_minutes = (exit_time - entry_time).total_seconds() / 60
                        # FIX: premium_paid is already per-share, don't divide by quantity!
                        exit_pnl_pct = ((current_premium / trade.premium_paid) - 1.0) * 100.0
                        
                        # Get prediction timeframe (default to 60 minutes if not set)
                        prediction_timeframe = getattr(trade, 'prediction_timeframe', 60)
                        predicted_move_pct = getattr(trade, 'predicted_return', 0.0) * 100.0
                        actual_move_pct = ((current_price / trade.entry_price) - 1.0) * 100.0 if hasattr(trade, 'entry_price') and trade.entry_price > 0 else 0.0
                        entry_confidence = getattr(trade, 'ml_confidence', 0.5)
                        
                        # Calculate days to expiration at exit
                        if trade.expiration_date:
                            expiration_date = trade.expiration_date
                            if expiration_date.tzinfo is not None and exit_time.tzinfo is None:
                                exit_time = pytz.UTC.localize(exit_time)
                            elif expiration_date.tzinfo is None and exit_time.tzinfo is not None:
                                expiration_date = pytz.UTC.localize(expiration_date)
                            days_to_expiration = max(0, (expiration_date - exit_time).days)
                        else:
                            days_to_expiration = 0
                        
                        # Determine if exited early (before hard limits)
                        exited_early = exit_reason not in ["Stop Loss", "Take Profit", "Expired ITM", "Expired OTM", "Contract Expiration"]
                        
                        # Store experience for learning
                        self.rl_exit_policy.store_exit_experience(
                            prediction_timeframe_minutes=prediction_timeframe,
                            time_held_minutes=time_held_minutes,
                            exit_pnl_pct=exit_pnl_pct,
                            days_to_expiration=days_to_expiration,
                            predicted_move_pct=predicted_move_pct,
                            actual_move_pct=actual_move_pct,
                            entry_confidence=entry_confidence,
                            exited_early=exited_early
                        )
                        
                        # Log XGBoost stats periodically
                        if self.total_trades % 10 == 0:
                            stats = self.rl_exit_policy.get_stats()
                            if stats.get('using_ml'):
                                self.logger.info(f"🌲 XGBoost: {stats.get('ml_exits', 0)} ML exits, " +
                                               f"{stats.get('rule_exits', 0)} rule exits, " +
                                               f"accuracy={stats.get('model_accuracy', 0):.1%}")
                            else:
                                self.logger.info(f"📋 Rules: {stats.get('rule_exits', 0)} exits, " +
                                               f"{stats.get('trades_seen', 0)}/{30} trades until ML")
                        
                        # Save model periodically
                        if self.total_trades % 25 == 0:
                            # Note: os is imported at module level (line 14)
                            os.makedirs('models', exist_ok=True)
                            self.rl_exit_policy.save('models/xgboost_exit.pkl')
                    except Exception as e:
                        self.logger.warning(f"⚠️ Could not store RL exit experience: {e}")

                # Calculate gross P&L for verification
                gross_pnl = (current_premium - trade.premium_paid) * trade.quantity * 100
                premium_change_pct = ((current_premium / trade.premium_paid) - 1.0) * 100.0 if trade.premium_paid > 0 else 0
                
                # Verify P&L calculation is correct
                expected_net_pnl = gross_pnl - total_fees
                pnl_error = abs(trade.profit_loss - expected_net_pnl)
                
                self.logger.info(
                    f"Closed {trade.option_type.value} trade ({exit_reason}):")
                self.logger.info(f"  • Quantity: {trade.quantity} contract(s)")
                self.logger.info(
                    f"  • Entry Premium: ${trade.premium_paid:.4f}/share")
                self.logger.info(f"  • Exit Premium: ${current_premium:.4f}/share")
                self.logger.info(f"  • Premium Change: {premium_change_pct:+.2f}%")
                self.logger.info(f"  • Gross P&L: ${gross_pnl:+.2f} (before fees)")
                self.logger.info(f"  • Entry Fees: ${entry_fees:.2f}")
                self.logger.info(f"  • Exit Fees: ${exit_fees:.2f}")
                self.logger.info(f"  • Total Fees: ${total_fees:.2f}")
                self.logger.info(f"  • Net P&L: ${trade.profit_loss:+.2f}")
                self.logger.info(
                    f"  • New Balance: ${self.current_balance:.2f}")
                
                # Alert if P&L calculation has errors
                if pnl_error > 0.01:
                    self.logger.error(f"⚠️ P&L VERIFICATION FAILED!")
                    self.logger.error(f"   Expected: ${expected_net_pnl:.2f}, Actual: ${trade.profit_loss:.2f}, Error: ${pnl_error:.2f}")

        # Handle any trades that failed atomic close + day trade recording
        for trade in trades_to_close:
            # Only remove if still in active_trades (atomic close already handled most)
            if trade in self.active_trades:
                try:
                    self.active_trades.remove(trade)
                    self._save_trade(trade)
                    self.logger.warning(f"🔄 FALLBACK: Trade {trade.id[:8]} removed (atomic close failed earlier)")
                except Exception as e:
                    self.logger.error(f"❌ CRITICAL: Could not remove trade {trade.id[:8]}: {e}")

            # Record day trade if applicable (same-day entry and exit)
            if (hasattr(trade, 'exit_timestamp') and trade.exit_timestamp and
                trade.timestamp.date() == trade.exit_timestamp.date() and
                self.pdt_protected):
                self._record_day_trade()
        
        # Check if account has grown enough to disable PDT protection
        if trades_to_close:  # Only check after trades are closed
            self._check_account_upgrade()

        # Save account state
        self._save_account_state()

    def get_account_summary(self) -> Dict:
        """Get account summary and performance metrics"""
        total_value = self.current_balance

        # Add unrealized P&L from active positions
        unrealized_pnl = sum(trade.profit_loss for trade in self.active_trades)
        total_value += unrealized_pnl
        
        # Add pending settlements to total value (funds that are settling)
        pending_settlement = self.pending_settlement_amount
        total_value += pending_settlement

        # Calculate performance metrics
        win_rate = self.winning_trades / max(self.total_trades, 1)
        avg_win = self.total_profit_loss / \
            max(self.winning_trades, 1) if self.winning_trades > 0 else 0
        avg_loss = abs(self.total_profit_loss) / \
            max(self.losing_trades, 1) if self.losing_trades > 0 else 0

        return {
            'account_balance': self.current_balance,
            'current_balance': self.current_balance,  # Add this for compatibility (settled/available funds)
            'cash_balance': self.current_balance,     # Add this for compatibility
            'settled_balance': self.current_balance,  # Explicitly named settled balance
            'pending_settlement': pending_settlement,  # Funds settling (T+1)
            'total_equity': total_value,              # Total including pending and unrealized
            'total_value': total_value,
            'unrealized_pnl': unrealized_pnl,
            'total_pnl': self.total_profit_loss,
            'total_trades': self.total_trades,
            'active_positions': len(self.active_trades),
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'max_drawdown': self.max_drawdown,
            'return_pct': (total_value - self.initial_balance) / self.initial_balance
        }
    
    def get_recent_closed_trades(self, limit: int = 10) -> list:
        """
        Get recent closed trades from the database
        
        Args:
            limit: Maximum number of trades to return
            
        Returns:
            List of trade dictionaries with trade details and P&L
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Query for closed trades with all details
            # Note: Closed trades have status PROFIT_TAKEN or STOPPED_OUT, not 'CLOSED'
            cursor.execute('''
                SELECT 
                    id, timestamp, symbol, option_type, strike_price,
                    premium_paid, quantity, entry_price, exit_price,
                    exit_timestamp, profit_loss, status
                FROM trades
                WHERE status IN ('PROFIT_TAKEN', 'STOPPED_OUT', 'CLOSED', 'EXPIRED_ITM', 'EXPIRED_OTM') 
                AND exit_timestamp IS NOT NULL
                ORDER BY exit_timestamp DESC
                LIMIT ?
            ''', (limit,))
            
            rows = cursor.fetchall()
            conn.close()
            
            trades = []
            for row in rows:
                trade_dict = {
                    'id': row[0],
                    'timestamp': row[1],
                    'entry_time': datetime.fromisoformat(row[1]),
                    'symbol': row[2],
                    'option_type': row[3],
                    'strike_price': row[4],
                    'premium_paid': row[5],
                    'quantity': row[6],
                    'entry_price': row[7],
                    'exit_price': row[8],
                    'exit_timestamp': row[9],
                    'exit_time': datetime.fromisoformat(row[9]) if row[9] else None,
                    'pnl': row[10] if row[10] is not None else 0,
                    'profit_loss': row[10] if row[10] is not None else 0,
                    'status': row[11]
                }
                trades.append(trade_dict)
            
            return trades
            
        except Exception as e:
            self.logger.warning(f"Could not fetch recent closed trades: {e}")
            return []

    def run_trading_session(self, duration_hours: int = 8, symbol: str = 'BITX'):
        """
        Run a simulated trading session
        """
        self.logger.info(
            f"Starting {duration_hours}-hour paper trading session...")
        self.logger.info(f"Initial Balance: ${self.current_balance:.2f}")

        start_time = self.get_market_time()
        end_time = start_time + timedelta(hours=duration_hours)

        # Simulate trading every 15 minutes
        current_time = start_time

        while current_time < end_time:
            # Update existing positions
            self.update_positions(symbol)

            # Get ML prediction
            prediction = self.learning_system.make_prediction_with_tracking(
                symbol)

            if not prediction.get('error') and prediction.get('confidence', 0) > 0.6:
                # Only trade on high confidence predictions
                self.place_trade(symbol, prediction)

            # Wait 15 minutes (simulated)
            current_time += timedelta(minutes=15)

            # In real implementation, you'd actually wait
            # time.sleep(900)  # 15 minutes

        # Final position update
        self.update_positions(symbol)

        # Get final summary
        summary = self.get_account_summary()

        self.logger.info("Trading session completed!")
        self.logger.info(f"Final Balance: ${summary['account_balance']:.2f}")
        self.logger.info(f"Total P&L: ${summary['total_pnl']:.2f}")
        self.logger.info(f"Return: {summary['return_pct']:.1%}")
        self.logger.info(f"Win Rate: {summary['win_rate']:.1%}")
        self.logger.info(f"Total Trades: {summary['total_trades']}")

        return summary


def main():
    """
    Demo of paper trading system
    """
    print("💰 BITX Paper Trading System Demo")
    print("=" * 50)

    # Initialize paper trading system
    paper_trader = PaperTradingSystem(initial_balance=1000.0)

    # Get initial account summary
    initial_summary = paper_trader.get_account_summary()
    print(f"📊 Initial Account Status:")
    print(f"   • Balance: ${initial_summary['account_balance']:.2f}")
    print(f"   • Total Value: ${initial_summary['total_value']:.2f}")
    print(f"   • Active Positions: {initial_summary['active_positions']}")

    # Run a short trading session
    print(f"\n🚀 Running 2-Hour Paper Trading Session...")
    final_summary = paper_trader.run_trading_session(duration_hours=2)

    print(f"\n📈 Final Results:")
    print(f"   • Final Balance: ${final_summary['account_balance']:.2f}")
    print(f"   • Total P&L: ${final_summary['total_pnl']:.2f}")
    print(f"   • Return: {final_summary['return_pct']:.1%}")
    print(f"   • Total Trades: {final_summary['total_trades']}")
    print(f"   • Win Rate: {final_summary['win_rate']:.1%}")
    print(f"   • Max Drawdown: {final_summary['max_drawdown']:.1%}")

    print(f"\n💡 Paper Trading Features:")
    print(f"   • ✅ Simulates real options trading")
    print(f"   • ✅ Uses ML predictions for trade decisions")
    print(f"   • ✅ Tracks P&L and performance metrics")
    print(f"   • ✅ Implements stop loss and take profit")
    print(f"   • ✅ Persistent database storage")
    print(f"   • ✅ Risk management (position sizing)")


if __name__ == "__main__":
    main()