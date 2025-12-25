"""
Trade Execution Module
======================

Handles trade execution logic, order routing, and execution tracking.

Extracted from UnifiedOptionsBot to provide:
- Order type mapping and validation
- Position sizing calculations
- Liquidity-aware order routing
- Execution outcome logging

Usage:
    from bot_modules.trade_execution import (
        TradeExecutor,
        OrderTypeMapper,
        PositionSizer
    )
"""

import logging
import sqlite3
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# Order Type Mapping
# ------------------------------------------------------------------------------
class TradeAction(Enum):
    """Supported trading actions."""
    HOLD = "HOLD"
    BUY_CALLS = "BUY_CALLS"
    BUY_CALL = "BUY_CALL"
    BUY_PUTS = "BUY_PUTS"
    BUY_PUT = "BUY_PUT"
    BUY_STRADDLE = "BUY_STRADDLE"
    SELL_PREMIUM = "SELL_PREMIUM"


class OrderTypeMapper:
    """
    Maps signal actions to backend order types with fallback handling.
    
    Handles cases where the backend doesn't support certain order types
    (e.g., straddles, credit spreads) by mapping to simpler alternatives.
    """
    
    def __init__(self, order_type_enum):
        """
        Initialize with the OrderType enum from the trading system.
        
        Args:
            order_type_enum: The OrderType enum class (e.g., from paper_trading_system)
        """
        self.OrderType = order_type_enum
        self._build_mapping()
    
    def _build_mapping(self):
        """Build the action to order type mapping."""
        self.mapping = {
            "BUY_CALLS": getattr(self.OrderType, "BUY_CALL", None),
            "BUY_CALL": getattr(self.OrderType, "BUY_CALL", None),
            "BUY_PUTS": getattr(self.OrderType, "BUY_PUT", None),
            "BUY_PUT": getattr(self.OrderType, "BUY_PUT", None),
            # IMPORTANT SAFETY:
            # Do NOT silently fall back multi-leg / short premium strategies to single-leg orders.
            # If the backend doesn't support the strategy, return None and let the caller block.
            "BUY_STRADDLE": getattr(self.OrderType, "BUY_STRADDLE", None),
            "SELL_PREMIUM": getattr(self.OrderType, "SELL_CREDIT_SPREAD", None),
        }
        
        # Track supported features
        self.has_straddle = getattr(self.OrderType, "BUY_STRADDLE", None) is not None
        self.has_credit_spread = getattr(self.OrderType, "SELL_CREDIT_SPREAD", None) is not None
    
    def get_order_type(self, action: str) -> Tuple[Optional[Any], List[str]]:
        """
        Map action string to order type with warnings for fallbacks.
        
        Args:
            action: Signal action string (e.g., "BUY_CALLS")
            
        Returns:
            Tuple of (order_type, list_of_warnings)
        """
        warnings = []
        order_type = self.mapping.get(action)
        
        # Warn about fallbacks
        if action == "BUY_STRADDLE" and not self.has_straddle:
            warnings.append(
                "Unsupported: BUY_STRADDLE requested but backend has no BUY_STRADDLE order type. "
                "Blocking to avoid accidentally placing a single-leg order."
            )
        
        if action == "SELL_PREMIUM" and not self.has_credit_spread:
            warnings.append(
                "Unsupported: SELL_PREMIUM requested but backend has no SELL_CREDIT_SPREAD order type. "
                "Blocking to avoid accidentally placing a naked short position."
            )
        
        if order_type is None:
            warnings.append(f"Unknown action: {action}. Available: {list(self.mapping.keys())}")
        
        return order_type, warnings


# ------------------------------------------------------------------------------
# Position Sizing
# ------------------------------------------------------------------------------
@dataclass
class PositionSizeConfig:
    """Configuration for position sizing."""
    base_position_size: float = 1.0      # Base contracts per trade
    max_position_size: float = 5.0       # Maximum contracts
    min_position_size: float = 1.0       # Minimum contracts
    confidence_scale: float = 1.0        # Scale factor for confidence
    volatility_scale: float = 1.0        # Scale factor for volatility
    max_account_risk_pct: float = 0.02   # Max 2% account risk per trade


class PositionSizer:
    """
    Calculates appropriate position sizes based on confidence and risk.
    
    Supports multiple sizing strategies:
    - Fixed: Always use base_position_size
    - Confidence-scaled: Scale by signal confidence
    - Kelly-inspired: Scale by edge estimate
    - Risk-parity: Scale by volatility
    """
    
    def __init__(self, config: PositionSizeConfig = None):
        self.config = config or PositionSizeConfig()
    
    def calculate(
        self,
        confidence: float,
        account_balance: float,
        current_price: float,
        predicted_volatility: float = None,
        regime: str = None
    ) -> int:
        """
        Calculate position size.
        
        Args:
            confidence: Signal confidence (0-1)
            account_balance: Current account balance
            current_price: Current asset price
            predicted_volatility: Optional predicted volatility
            regime: Optional market regime
            
        Returns:
            Number of contracts to trade
        """
        # Start with base size
        size = self.config.base_position_size
        
        # Scale by confidence
        if confidence > 0.6:
            size *= 1 + (confidence - 0.6) * self.config.confidence_scale
        elif confidence < 0.5:
            size *= max(0.5, confidence / 0.5)
        
        # Scale by volatility (reduce size in high vol)
        if predicted_volatility is not None:
            if predicted_volatility > 0.02:  # High volatility
                size *= 0.7
            elif predicted_volatility < 0.01:  # Low volatility
                size *= 1.2
        
        # Regime adjustments
        if regime:
            regime_scales = {
                'extreme_vol': 0.5,
                'high_vol': 0.7,
                'normal_vol': 1.0,
                'low_vol': 1.2,
                'ultra_low_vol': 1.3,
            }
            size *= regime_scales.get(regime, 1.0)
        
        # Apply limits
        size = max(self.config.min_position_size, min(self.config.max_position_size, size))
        
        # Risk check: don't risk more than max_account_risk_pct
        max_risk_dollars = account_balance * self.config.max_account_risk_pct
        estimated_contract_cost = current_price * 100 * 0.02  # Assume ~2% of spot for ATM options
        max_contracts_by_risk = max(1, int(max_risk_dollars / estimated_contract_cost))
        
        return int(min(size, max_contracts_by_risk))


# ------------------------------------------------------------------------------
# Execution Tracking
# ------------------------------------------------------------------------------
@dataclass
class ExecutionOutcome:
    """Record of a trade execution outcome for learning."""
    timestamp: datetime
    symbol: str
    option_symbol: str
    side: str
    limit_price: float
    fill_price: float
    filled_qty: int
    desired_qty: int
    time_to_fill_secs: float
    bid: float
    ask: float
    features: np.ndarray = None
    
    @property
    def was_filled(self) -> bool:
        return self.filled_qty > 0
    
    @property
    def slippage(self) -> float:
        return abs(self.fill_price - self.limit_price)
    
    @property
    def fill_rate(self) -> float:
        return self.filled_qty / self.desired_qty if self.desired_qty > 0 else 0


class ExecutionTracker:
    """
    Tracks execution outcomes for learning and analysis.
    
    Logs execution data to database for training fillability predictors.
    """
    
    def __init__(self, db_path: str = "data/unified_options_bot.db"):
        self.db_path = db_path
        self._ensure_table()
    
    def _ensure_table(self):
        """Ensure the exec_outcomes table exists."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS exec_outcomes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT,
                    occ_symbol TEXT,
                    symbol TEXT,
                    side TEXT,
                    limit_price REAL,
                    worked_secs REAL,
                    filled_qty INTEGER,
                    desired_qty INTEGER,
                    avg_fill_price REAL,
                    best_bid REAL,
                    best_ask REAL,
                    features BLOB,
                    label_fill INTEGER,
                    label_slip REAL,
                    label_ttf REAL
                )
            """)
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"Could not create exec_outcomes table: {e}")
    
    def log_outcome(self, outcome: ExecutionOutcome):
        """Log an execution outcome to the database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            features_blob = pickle.dumps(outcome.features) if outcome.features is not None else pickle.dumps(np.array([]))
            
            cursor.execute("""
                INSERT INTO exec_outcomes 
                (ts, occ_symbol, symbol, side, limit_price, worked_secs, 
                 filled_qty, desired_qty, avg_fill_price, best_bid, best_ask,
                 features, label_fill, label_slip, label_ttf)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                outcome.timestamp.isoformat(),
                outcome.option_symbol,
                outcome.symbol,
                outcome.side,
                outcome.limit_price,
                outcome.time_to_fill_secs,
                outcome.filled_qty,
                outcome.desired_qty,
                outcome.fill_price,
                outcome.bid,
                outcome.ask,
                features_blob,
                1 if outcome.was_filled else 0,
                outcome.slippage,
                outcome.time_to_fill_secs
            ))
            
            conn.commit()
            conn.close()
            logger.debug(f"Logged execution outcome for {outcome.option_symbol}")
            
        except Exception as e:
            logger.error(f"Error logging execution outcome: {e}")
    
    def get_recent_outcomes(self, limit: int = 100) -> List[Dict]:
        """Get recent execution outcomes for analysis."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT ts, occ_symbol, symbol, side, limit_price, worked_secs,
                       filled_qty, desired_qty, avg_fill_price, best_bid, best_ask,
                       label_fill, label_slip, label_ttf
                FROM exec_outcomes
                ORDER BY ts DESC
                LIMIT ?
            """, (limit,))
            
            rows = cursor.fetchall()
            conn.close()
            
            outcomes = []
            for row in rows:
                outcomes.append({
                    'timestamp': row[0],
                    'option_symbol': row[1],
                    'symbol': row[2],
                    'side': row[3],
                    'limit_price': row[4],
                    'time_to_fill': row[5],
                    'filled_qty': row[6],
                    'desired_qty': row[7],
                    'fill_price': row[8],
                    'bid': row[9],
                    'ask': row[10],
                    'was_filled': row[11] == 1,
                    'slippage': row[12],
                    'ttf': row[13]
                })
            
            return outcomes
            
        except Exception as e:
            logger.error(f"Error getting execution outcomes: {e}")
            return []
    
    def get_fill_rate_stats(self) -> Dict:
        """Get aggregate fill rate statistics."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT 
                    COUNT(*) as total,
                    SUM(label_fill) as filled,
                    AVG(label_slip) as avg_slippage,
                    AVG(label_ttf) as avg_ttf
                FROM exec_outcomes
            """)
            
            row = cursor.fetchone()
            conn.close()
            
            total = row[0] or 0
            filled = row[1] or 0
            
            return {
                'total_orders': total,
                'filled_orders': filled,
                'fill_rate': filled / total if total > 0 else 0,
                'avg_slippage': row[2] or 0,
                'avg_time_to_fill': row[3] or 0
            }
            
        except Exception as e:
            logger.error(f"Error getting fill rate stats: {e}")
            return {}


# ------------------------------------------------------------------------------
# Trade Executor
# ------------------------------------------------------------------------------
class TradeExecutor:
    """
    Main trade execution orchestrator.
    
    Coordinates:
    - Order type mapping
    - Position sizing
    - Pre-trade validation
    - Execution tracking
    """
    
    def __init__(
        self,
        order_type_enum,
        position_config: PositionSizeConfig = None,
        db_path: str = "data/unified_options_bot.db",
        min_time_between_trades: int = 300  # 5 minutes
    ):
        self.order_mapper = OrderTypeMapper(order_type_enum)
        self.position_sizer = PositionSizer(position_config)
        self.execution_tracker = ExecutionTracker(db_path)
        self.min_time_between_trades = min_time_between_trades
        self.last_trade_time: Optional[datetime] = None
    
    def validate_trade(
        self,
        signal: Dict,
        current_time: datetime,
        active_positions: int,
        max_positions: int,
        min_confidence: float = 0.1,
        simulation_mode: bool = False
    ) -> Tuple[bool, str]:
        """
        Validate if a trade should be executed.
        
        Args:
            signal: Trading signal dictionary
            current_time: Current market time
            active_positions: Number of active positions
            max_positions: Maximum allowed positions
            min_confidence: Minimum confidence threshold
            simulation_mode: Whether running in simulation
            
        Returns:
            Tuple of (is_valid, rejection_reason)
        """
        # Check for HOLD action
        if signal.get("action") == "HOLD":
            return False, "HOLD action"
        
        # Check confidence threshold
        confidence = signal.get("confidence", 0)
        if confidence < min_confidence:
            return False, f"Confidence {confidence:.1%} below threshold {min_confidence:.1%}"
        
        # Check position limit
        if active_positions >= max_positions:
            return False, f"Max positions ({max_positions}) reached"
        
        # Check time between trades (skip in simulation)
        if not simulation_mode and self.last_trade_time:
            time_since_last = (current_time - self.last_trade_time).total_seconds()
            if time_since_last < self.min_time_between_trades:
                remaining = (self.min_time_between_trades - time_since_last) / 60
                return False, f"Waiting {remaining:.1f} more minutes (algorithmic spacing)"
        
        # Validate order type mapping
        order_type, warnings = self.order_mapper.get_order_type(signal.get("action", ""))
        if order_type is None:
            return False, f"Unknown action: {signal.get('action')}"
        
        return True, "OK"
    
    def prepare_trade(
        self,
        signal: Dict,
        account_balance: float,
        current_price: float
    ) -> Dict:
        """
        Prepare trade details including position size and order type.
        
        Args:
            signal: Trading signal
            account_balance: Current account balance
            current_price: Current asset price
            
        Returns:
            Dict with trade preparation details
        """
        # Get position size
        if "rl_position_size" in signal and signal["rl_position_size"] > 0:
            position_size = signal["rl_position_size"]
            size_source = "rl_policy"
        else:
            position_size = self.position_sizer.calculate(
                confidence=signal.get("confidence", 0.5),
                account_balance=account_balance,
                current_price=current_price,
                predicted_volatility=signal.get("predicted_volatility"),
                regime=signal.get("regime")
            )
            size_source = "calculated"
        
        # Get order type
        order_type, warnings = self.order_mapper.get_order_type(signal.get("action", ""))
        
        return {
            "position_size": position_size,
            "size_source": size_source,
            "order_type": order_type,
            "warnings": warnings,
            "action": signal.get("action"),
            "confidence": signal.get("confidence", 0),
            "strategy": signal.get("strategy", "ML_SIGNAL"),
            "reasoning": signal.get("reasoning", []),
            "risk_level": signal.get("risk_level", "MEDIUM"),
            "hmm_state": signal.get("hmm_state"),
            "hmm_confidence": signal.get("hmm_confidence", 0),
            "hmm_trend": signal.get("hmm_trend"),
        }
    
    def record_trade_execution(self, success: bool, current_time: datetime):
        """Record that a trade was executed (or attempted)."""
        if success:
            self.last_trade_time = current_time
    
    def format_trade_announcement(
        self,
        signal: Dict,
        position_size: int,
        current_price: float,
        symbol: str
    ) -> str:
        """Format a trade announcement message."""
        lines = [
            "",
            "=" * 80,
            f"🎯 SimonSays: {signal['action']} x{position_size} @ ${current_price:.2f}",
            f"   💰 Trade Details:",
            f"      • Symbol: {symbol}",
            f"      • Position Size: {position_size} contracts",
            f"      • Entry Price: ${current_price:.2f}",
            f"      • Confidence: {signal.get('confidence', 0):.1%}",
            f"      • Strategy: {signal.get('strategy', 'N/A')}",
        ]
        
        predicted_return = signal.get("predicted_return", 0)
        timeframe = signal.get("timeframe_minutes", 60)
        lines.append(f"      • Predicted Move: {predicted_return:+.2%} over {timeframe}min")
        
        lines.append(f"   📊 Why This Trade:")
        for i, reason in enumerate(signal.get("reasoning", ["Signal approved"]), 1):
            lines.append(f"      {i}. {reason}")
        
        if signal.get("multi_timeframe_analysis"):
            mt = signal["multi_timeframe_analysis"]
            lines.extend([
                f"   🔮 Multi-Timeframe Consensus:",
                f"      • Direction: {mt.get('direction', 'N/A')}",
                f"      • Confidence: {mt.get('confidence', 0):.1%}",
                f"      • Agreement: {mt.get('agreement', 0):.1%}",
            ])
        
        lines.extend(["=" * 80, ""])
        
        return "\n".join(lines)


__all__ = [
    'TradeAction',
    'OrderTypeMapper',
    'PositionSizeConfig',
    'PositionSizer',
    'ExecutionOutcome',
    'ExecutionTracker',
    'TradeExecutor',
]








