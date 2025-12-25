#!/usr/bin/env python3
"""
Trading Environment Abstraction
===============================

Issue 7 Fix: Unified environment for simulation and live trading.

The Problem:
- run_simulation.py and train_then_go_live.py have separate code paths
- Simulation may bypass gating, liquidity checks, risk limits
- Different P&L formulas, missing spread/commission simulation
- Results in rosier backtest than live performance

The Solution:
- Single TradingEnvironment abstraction (like RL Gym)
- step(action) returns (obs, reward, done, info)
- Same config, gating, and risk logic for both sim and live
- Only difference: data source + order execution adapter

Ensures:
- Bid/ask spread simulation (don't fill at mid)
- Commission and fees simulation
- Slippage on large spread / low volume
- Exact same P&L formula in sim and live

Usage:
    # Simulation
    env = TradingEnvironment(
        broker=PaperTradingSystem(...),
        data_source=HistoricalDataSource(...),
        config=config,
        mode='simulation'
    )
    
    # Live
    env = TradingEnvironment(
        broker=TradierTradingSystem(...),
        data_source=TradierDataSource(...),
        config=config,
        mode='live'
    )
    
    # Same interface for both!
    obs = env.reset()
    while not done:
        action = model.predict(obs)
        obs, reward, done, info = env.step(action)
"""

import logging
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np

logger = logging.getLogger(__name__)


class TradingMode(Enum):
    """Trading environment mode."""
    SIMULATION = "simulation"
    PAPER = "paper"
    LIVE = "live"


class ActionType(Enum):
    """Possible trading actions."""
    HOLD = "HOLD"
    BUY_CALLS = "BUY_CALLS"
    BUY_PUTS = "BUY_PUTS"
    CLOSE_POSITION = "CLOSE_POSITION"


@dataclass
class MarketState:
    """Current market state observation."""
    timestamp: datetime
    symbol: str
    underlying_price: float
    
    # OHLCV
    open: float
    high: float
    low: float
    close: float
    volume: int
    
    # Market context
    vix_level: float = 17.0
    minutes_to_close: int = 390
    is_market_open: bool = True
    
    # Technical indicators (computed from data)
    features: Optional[np.ndarray] = None
    
    # Regime info
    regime: Optional[str] = None
    regime_confidence: float = 0.5


@dataclass
class Position:
    """Open position."""
    id: str
    symbol: str
    option_type: str  # 'CALL' or 'PUT'
    strike: float
    expiry: datetime
    quantity: int
    entry_price: float  # Per share (option premium)
    entry_time: datetime
    current_price: float
    unrealized_pnl: float
    
    # Stop/target
    stop_loss: float
    take_profit: float
    max_hold_minutes: int


@dataclass
class TradeResult:
    """Result of a trade action."""
    success: bool
    action: ActionType
    position_id: Optional[str] = None
    fill_price: Optional[float] = None
    quantity: int = 0
    commission: float = 0.0
    slippage: float = 0.0
    reason: str = ""


@dataclass
class StepResult:
    """Result of environment step (RL Gym-like interface)."""
    observation: MarketState
    reward: float
    done: bool
    truncated: bool  # Episode truncated (time limit, not terminal)
    info: Dict = field(default_factory=dict)


class ExecutionSimulator:
    """
    Simulates realistic trade execution for backtesting.
    
    Ensures simulation doesn't get rosier fills than live:
    - Fills at worse side of spread (buy at ask, sell at bid)
    - Adds slippage for low liquidity
    - Includes commission/fees
    
    NOTE: Settings are realistic for liquid SPY options.
    SPY options typically have 0.5-1.5% spreads for ATM options.
    """
    
    def __init__(
        self,
        spread_pct: float = 0.01,  # REDUCED: 1% bid-ask spread (SPY is liquid)
        slippage_pct: float = 0.002,  # REDUCED: 0.2% slippage (SPY is liquid)
        commission_per_contract: float = 0.65,  # Per contract fee
        volume_impact_threshold: int = 50,  # LOWERED: SPY has high volume
    ):
        """
        Args:
            spread_pct: Default bid-ask spread as percent of price
            slippage_pct: Additional slippage percent
            commission_per_contract: Commission per contract
            volume_impact_threshold: Volume threshold for impact
        """
        self.spread_pct = spread_pct
        self.slippage_pct = slippage_pct
        self.commission_per_contract = commission_per_contract
        self.volume_impact_threshold = volume_impact_threshold
    
    def simulate_fill(
        self,
        mid_price: float,
        is_buy: bool,
        quantity: int,
        volume: int = 1000,
        spread_override: Optional[float] = None
    ) -> Tuple[float, float, float]:
        """
        Simulate realistic fill price.
        
        Args:
            mid_price: Mid-market price
            is_buy: True if buying, False if selling
            quantity: Number of contracts
            volume: Current option volume
            spread_override: Override spread percent
            
        Returns:
            (fill_price, slippage_cost, commission)
        """
        spread = spread_override or self.spread_pct
        half_spread = spread / 2
        
        # Fill at worse side of spread
        if is_buy:
            # Buy at ask (above mid)
            fill_price = mid_price * (1 + half_spread)
        else:
            # Sell at bid (below mid)
            fill_price = mid_price * (1 - half_spread)
        
        # Additional slippage for low volume
        if volume < self.volume_impact_threshold:
            volume_factor = self.volume_impact_threshold / max(1, volume)
            additional_slip = self.slippage_pct * volume_factor
            if is_buy:
                fill_price *= (1 + additional_slip)
            else:
                fill_price *= (1 - additional_slip)
        
        # Calculate costs
        slippage_cost = abs(fill_price - mid_price) * quantity * 100
        commission = self.commission_per_contract * quantity
        
        return fill_price, slippage_cost, commission


class BrokerAdapter(ABC):
    """Abstract interface for broker operations."""
    
    @abstractmethod
    def place_order(
        self,
        symbol: str,
        option_type: str,
        strike: float,
        expiry: datetime,
        quantity: int,
        is_buy: bool,
        order_type: str = 'market'
    ) -> TradeResult:
        """Place an order."""
        pass
    
    @abstractmethod
    def get_positions(self) -> List[Position]:
        """Get current positions."""
        pass
    
    @abstractmethod
    def close_position(self, position_id: str) -> TradeResult:
        """Close a position."""
        pass
    
    @abstractmethod
    def get_account_balance(self) -> float:
        """Get current account balance."""
        pass


class SimulatedBroker(BrokerAdapter):
    """
    Simulated broker for backtesting.
    
    Uses ExecutionSimulator for realistic fills.
    
    Training Mode: When allow_negative_balance=True, the account can go negative
    so the model learns from all market conditions (drawdowns, losing streaks, etc.)
    True P&L is tracked separately so performance metrics remain accurate.
    """
    
    def __init__(
        self,
        initial_balance: float,
        execution_sim: ExecutionSimulator = None,
        allow_negative_balance: bool = False,
        replenish_threshold: float = 500.0
    ):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.execution_sim = execution_sim or ExecutionSimulator()
        
        # Training mode settings
        self.allow_negative_balance = allow_negative_balance
        self.replenish_threshold = replenish_threshold
        
        # True P&L tracking (unaffected by auto-replenishment)
        self.true_pnl = 0.0  # Running total of actual gains/losses
        self.total_replenished = 0.0  # Track how much was auto-replenished
        self.replenish_count = 0
        
        self._positions: Dict[str, Position] = {}
        self._position_counter = 0
        self._trade_history: List[Dict] = []
        
        logger = logging.getLogger(__name__)
        if allow_negative_balance:
            logger.info(f"🎓 [TRAINING MODE] Negative balance allowed - model will learn from all conditions")
            logger.info(f"   Auto-replenish below ${replenish_threshold:.2f} to keep realistic position sizing")
    
    def place_order(
        self,
        symbol: str,
        option_type: str,
        strike: float,
        expiry: datetime,
        quantity: int,
        is_buy: bool,
        order_type: str = 'market',
        mid_price: float = 0.0,
        volume: int = 100
    ) -> TradeResult:
        """Place a simulated order."""
        if mid_price <= 0:
            return TradeResult(
                success=False,
                action=ActionType.BUY_CALLS if option_type == 'CALL' else ActionType.BUY_PUTS,
                reason="Invalid price"
            )
        
        # Simulate fill
        fill_price, slippage, commission = self.execution_sim.simulate_fill(
            mid_price, is_buy, quantity, volume
        )
        
        total_cost = fill_price * quantity * 100 + commission
        
        if is_buy:
            # Check balance (with training mode override)
            if total_cost > self.balance:
                if self.allow_negative_balance:
                    # Training mode: auto-replenish to keep learning
                    replenish_amount = self.initial_balance
                    self.balance += replenish_amount
                    self.total_replenished += replenish_amount
                    self.replenish_count += 1
                    logger = logging.getLogger(__name__)
                    logger.warning(f"💰 [TRAINING] Auto-replenished ${replenish_amount:.2f} (#{self.replenish_count}) to continue learning")
                    logger.info(f"   Balance: ${self.balance:.2f} | True P&L: ${self.true_pnl:.2f}")
                else:
                    return TradeResult(
                        success=False,
                        action=ActionType.BUY_CALLS if option_type == 'CALL' else ActionType.BUY_PUTS,
                        reason=f"Insufficient balance: need ${total_cost:.2f}, have ${self.balance:.2f}"
                    )
            
            self.balance -= total_cost
            
            # Create position
            self._position_counter += 1
            pos_id = f"sim_{self._position_counter}"
            
            position = Position(
                id=pos_id,
                symbol=symbol,
                option_type=option_type,
                strike=strike,
                expiry=expiry,
                quantity=quantity,
                entry_price=fill_price,
                entry_time=datetime.now(),
                current_price=fill_price,
                unrealized_pnl=0.0,
                stop_loss=fill_price * 0.80,  # Default 20% stop
                take_profit=fill_price * 1.40,  # Default 40% target
                max_hold_minutes=120
            )
            
            self._positions[pos_id] = position
            
            return TradeResult(
                success=True,
                action=ActionType.BUY_CALLS if option_type == 'CALL' else ActionType.BUY_PUTS,
                position_id=pos_id,
                fill_price=fill_price,
                quantity=quantity,
                commission=commission,
                slippage=slippage
            )
        else:
            # Selling (closing position handled separately)
            return TradeResult(
                success=False,
                action=ActionType.CLOSE_POSITION,
                reason="Use close_position() to close"
            )
    
    def get_positions(self) -> List[Position]:
        """Get current positions."""
        return list(self._positions.values())
    
    def close_position(
        self,
        position_id: str,
        exit_price: float = 0.0,
        volume: int = 100
    ) -> TradeResult:
        """Close a position."""
        if position_id not in self._positions:
            return TradeResult(
                success=False,
                action=ActionType.CLOSE_POSITION,
                reason=f"Position {position_id} not found"
            )
        
        position = self._positions[position_id]
        
        if exit_price <= 0:
            exit_price = position.current_price
        
        # Simulate fill (selling)
        fill_price, slippage, commission = self.execution_sim.simulate_fill(
            exit_price, is_buy=False, quantity=position.quantity, volume=volume
        )
        
        # Calculate P&L
        proceeds = fill_price * position.quantity * 100 - commission
        cost_basis = position.entry_price * position.quantity * 100
        pnl = proceeds - cost_basis
        
        self.balance += proceeds
        
        # Track true P&L (unaffected by auto-replenishment for training metrics)
        self.true_pnl += pnl
        
        # Record trade
        self._trade_history.append({
            'position_id': position_id,
            'symbol': position.symbol,
            'option_type': position.option_type,
            'entry_price': position.entry_price,
            'exit_price': fill_price,
            'quantity': position.quantity,
            'pnl': pnl,
            'true_pnl_running': self.true_pnl,  # Track running true P&L
            'commission': commission,
            'slippage': slippage,
            'hold_time': (datetime.now() - position.entry_time).total_seconds() / 60
        })
        
        del self._positions[position_id]
        
        return TradeResult(
            success=True,
            action=ActionType.CLOSE_POSITION,
            position_id=position_id,
            fill_price=fill_price,
            quantity=position.quantity,
            commission=commission,
            slippage=slippage,
            reason=f"P&L: ${pnl:.2f}"
        )
    
    def get_account_balance(self) -> float:
        """Get current balance."""
        return self.balance
    
    def get_equity(self) -> float:
        """Get total equity (balance + unrealized P&L)."""
        unrealized = sum(p.unrealized_pnl for p in self._positions.values())
        return self.balance + unrealized
    
    def get_trade_history(self) -> List[Dict]:
        """Get trade history."""
        return self._trade_history.copy()
    
    def get_training_stats(self) -> Dict:
        """
        Get training-specific statistics.
        
        Returns true P&L (unaffected by auto-replenishment) for accurate
        performance metrics during training.
        """
        trades = len(self._trade_history)
        wins = sum(1 for t in self._trade_history if t['pnl'] > 0)
        
        return {
            'true_pnl': self.true_pnl,
            'display_balance': self.balance,
            'total_replenished': self.total_replenished,
            'replenish_count': self.replenish_count,
            'true_return_pct': (self.true_pnl / self.initial_balance) * 100 if self.initial_balance > 0 else 0,
            'total_trades': trades,
            'win_rate': (wins / trades * 100) if trades > 0 else 0,
            'training_mode': self.allow_negative_balance,
            'note': 'true_pnl reflects actual performance; display_balance may include replenishment'
        }


class TradingEnvironment:
    """
    Unified trading environment for simulation and live trading.
    
    Provides RL Gym-like interface:
    - reset() -> observation
    - step(action) -> (observation, reward, done, truncated, info)
    
    Ensures same logic runs in sim and live:
    - Gating checks
    - Risk limits
    - Liquidity checks
    - P&L calculation
    """
    
    def __init__(
        self,
        broker: BrokerAdapter,
        config: Optional[Dict] = None,
        mode: TradingMode = TradingMode.SIMULATION,
        execution_sim: ExecutionSimulator = None,
        risk_manager = None,  # RiskManager instance
        symbol: str = 'SPY'
    ):
        """
        Initialize trading environment.
        
        Args:
            broker: Broker adapter (simulated or real)
            config: Config dict
            mode: Trading mode
            execution_sim: Execution simulator (for realistic fills)
            risk_manager: RiskManager instance (from risk_manager.py)
            symbol: Trading symbol
        """
        self.broker = broker
        self.config = config or {}
        self.mode = mode
        self.execution_sim = execution_sim or ExecutionSimulator()
        self.risk_manager = risk_manager
        self.symbol = symbol
        
        # Episode state
        self._current_step = 0
        self._episode_start_balance: float = 0.0
        self._episode_pnl: float = 0.0
        self._last_state: Optional[MarketState] = None
        
        # Gates (same as live)
        self._min_confidence = config.get('trading', {}).get('base_confidence_threshold', 0.55)
        self._max_trades_per_hour = config.get('calibration', {}).get('max_trades_per_hour', 5)
        self._trades_this_hour = 0
        self._last_trade_hour = -1
        
        logger.info(f"✅ TradingEnvironment initialized: mode={mode.value}, symbol={symbol}")
    
    def reset(
        self,
        initial_state: Optional[MarketState] = None,
        seed: Optional[int] = None
    ) -> MarketState:
        """
        Reset environment for new episode.
        
        Args:
            initial_state: Starting market state
            seed: Random seed for reproducibility
            
        Returns:
            Initial observation
        """
        self._current_step = 0
        self._episode_start_balance = self.broker.get_account_balance()
        self._episode_pnl = 0.0
        self._trades_this_hour = 0
        self._last_trade_hour = -1
        
        if initial_state:
            self._last_state = initial_state
        else:
            # Create default state
            self._last_state = MarketState(
                timestamp=datetime.now(),
                symbol=self.symbol,
                underlying_price=0.0,
                open=0.0, high=0.0, low=0.0, close=0.0,
                volume=0
            )
        
        return self._last_state
    
    def step(
        self,
        action: Union[ActionType, Dict],
        market_state: Optional[MarketState] = None
    ) -> StepResult:
        """
        Execute one step in the environment.
        
        Args:
            action: Trading action (ActionType or dict with details)
            market_state: Current market state (required for simulation)
            
        Returns:
            StepResult with observation, reward, done, truncated, info
        """
        self._current_step += 1
        
        if market_state:
            self._last_state = market_state
        
        state = self._last_state
        
        # Parse action
        if isinstance(action, dict):
            action_type = ActionType(action.get('action', 'HOLD'))
            confidence = action.get('confidence', 0.5)
            signal_details = action
        else:
            action_type = action
            confidence = 0.5
            signal_details = {}
        
        # Initialize step result
        info = {
            'action': action_type.value,
            'confidence': confidence,
            'step': self._current_step,
            'balance': self.broker.get_account_balance(),
        }
        
        reward = 0.0
        done = False
        truncated = False
        
        # Apply gating (same as live)
        trade_allowed, gate_reason = self._check_gates(action_type, confidence, state)
        info['gate_result'] = 'allowed' if trade_allowed else gate_reason
        
        if action_type == ActionType.HOLD:
            # No trade, just update positions
            reward = self._update_positions(state)
            info['action_taken'] = 'hold'
            
        elif action_type in [ActionType.BUY_CALLS, ActionType.BUY_PUTS]:
            if trade_allowed:
                result = self._execute_entry(action_type, signal_details, state)
                info['trade_result'] = result.reason
                info['fill_price'] = result.fill_price
                info['slippage'] = result.slippage
                info['commission'] = result.commission
                
                if result.success:
                    info['action_taken'] = 'entered'
                    self._trades_this_hour += 1
                else:
                    info['action_taken'] = 'entry_failed'
            else:
                info['action_taken'] = 'blocked'
            
            reward = self._update_positions(state)
            
        elif action_type == ActionType.CLOSE_POSITION:
            position_id = signal_details.get('position_id')
            if position_id:
                result = self.broker.close_position(position_id, state.underlying_price)
                info['trade_result'] = result.reason
                info['fill_price'] = result.fill_price
                reward = float(result.reason.split('$')[1].replace(',', '')) if result.success else 0
            
            reward += self._update_positions(state)
        
        # Check episode termination
        if not state.is_market_open:
            truncated = True
            info['termination'] = 'market_closed'
        
        # Check risk limits
        if self.risk_manager:
            status = self.risk_manager.get_risk_status(self.broker.get_account_balance())
            if status['is_daily_stop'] or status['is_weekly_stop']:
                done = True
                info['termination'] = 'risk_limit'
        
        # Update P&L tracking
        current_equity = self.broker.get_account_balance()
        if hasattr(self.broker, 'get_equity'):
            current_equity = self.broker.get_equity()
        self._episode_pnl = current_equity - self._episode_start_balance
        info['episode_pnl'] = self._episode_pnl
        
        return StepResult(
            observation=state,
            reward=reward,
            done=done,
            truncated=truncated,
            info=info
        )
    
    def _check_gates(
        self,
        action: ActionType,
        confidence: float,
        state: MarketState
    ) -> Tuple[bool, str]:
        """Check trading gates (same as live)."""
        if action == ActionType.HOLD:
            return True, ""
        
        # Gate 1: Confidence threshold
        if confidence < self._min_confidence:
            return False, f"confidence_too_low_{confidence:.2f}"
        
        # Gate 2: Trade frequency
        current_hour = state.timestamp.hour
        if current_hour != self._last_trade_hour:
            self._trades_this_hour = 0
            self._last_trade_hour = current_hour
        
        if self._trades_this_hour >= self._max_trades_per_hour:
            return False, "max_trades_per_hour"
        
        # Gate 3: Time to close (Issue 5)
        if state.minutes_to_close < 90:
            return False, "too_close_to_market_close"
        
        # Gate 4: Risk manager (if available)
        if self.risk_manager:
            allowed, result, reason = self.risk_manager.check_trade_allowed(
                account_balance=self.broker.get_account_balance(),
                current_positions=len(self.broker.get_positions()),
                minutes_to_close=state.minutes_to_close
            )
            if not allowed:
                return False, f"risk_{result.value}"
        
        return True, ""
    
    def _execute_entry(
        self,
        action: ActionType,
        signal: Dict,
        state: MarketState
    ) -> TradeResult:
        """Execute trade entry."""
        option_type = 'CALL' if action == ActionType.BUY_CALLS else 'PUT'
        
        # Get option parameters from signal
        strike = signal.get('strike', state.underlying_price)
        expiry = signal.get('expiry', datetime.now() + timedelta(days=1))
        quantity = signal.get('quantity', 1)
        option_price = signal.get('option_price', state.underlying_price * 0.01)  # Default 1% of underlying
        
        return self.broker.place_order(
            symbol=self.symbol,
            option_type=option_type,
            strike=strike,
            expiry=expiry,
            quantity=quantity,
            is_buy=True,
            mid_price=option_price
        )
    
    def _update_positions(self, state: MarketState) -> float:
        """Update positions and return realized P&L from exits.
        
        Uses a more realistic option pricing model:
        - Delta-based price changes (not linear to strike)
        - Time decay (theta) based on hold time
        - Gamma effect for larger moves
        """
        realized_pnl = 0.0
        
        for position in self.broker.get_positions():
            # Calculate underlying price move as percentage
            underlying_move_pct = (state.underlying_price - position.strike) / state.underlying_price
            
            # Estimate delta based on moneyness (simplified Black-Scholes approximation)
            # ATM options have ~0.50 delta, ITM higher, OTM lower
            moneyness = state.underlying_price / position.strike
            if position.option_type == 'CALL':
                # CALL delta: increases as stock goes up
                if moneyness > 1.05:  # ITM
                    delta = min(0.90, 0.50 + (moneyness - 1) * 2)
                elif moneyness < 0.95:  # OTM
                    delta = max(0.10, 0.50 - (1 - moneyness) * 2)
                else:  # ATM
                    delta = 0.50
                
                # Option price change = delta * underlying_move + gamma * move^2
                price_change_pct = delta * underlying_move_pct + 0.5 * abs(underlying_move_pct) * underlying_move_pct
            else:
                # PUT delta: negative (increases as stock goes down)
                if moneyness < 0.95:  # ITM for put
                    delta = min(0.90, 0.50 + (1 - moneyness) * 2)
                elif moneyness > 1.05:  # OTM for put
                    delta = max(0.10, 0.50 - (moneyness - 1) * 2)
                else:  # ATM
                    delta = 0.50
                
                # For puts, we profit when price goes down
                price_change_pct = delta * (-underlying_move_pct) + 0.5 * abs(underlying_move_pct) * (-underlying_move_pct)
            
            # Apply time decay (theta) - options lose value over time
            hold_minutes = (datetime.now() - position.entry_time).total_seconds() / 60
            # Theta decay: ~0.5-2% per hour for short-dated options
            theta_decay_per_hour = 0.01  # 1% per hour
            time_decay = 1 - (theta_decay_per_hour * hold_minutes / 60)
            time_decay = max(0.5, time_decay)  # Cap at 50% loss from theta alone
            
            # Calculate new price
            new_price = position.entry_price * (1 + price_change_pct) * time_decay
            position.current_price = max(0.01, new_price)  # Floor at $0.01
            
            # Update unrealized P&L
            position.unrealized_pnl = (position.current_price - position.entry_price) * position.quantity * 100
            
            # Check stop loss / take profit
            if position.current_price <= position.stop_loss:
                result = self.broker.close_position(position.id, position.current_price)
                if result.success:
                    pnl = (position.current_price - position.entry_price) * position.quantity * 100
                    realized_pnl += pnl
                    
            elif position.current_price >= position.take_profit:
                result = self.broker.close_position(position.id, position.current_price)
                if result.success:
                    pnl = (position.current_price - position.entry_price) * position.quantity * 100
                    realized_pnl += pnl
            
            # Check max hold time
            if hold_minutes >= position.max_hold_minutes:
                result = self.broker.close_position(position.id, position.current_price)
                if result.success:
                    pnl = (position.current_price - position.entry_price) * position.quantity * 100
                    realized_pnl += pnl
        
        return realized_pnl
    
    def get_episode_stats(self) -> Dict:
        """Get statistics for current episode."""
        positions = self.broker.get_positions()
        balance = self.broker.get_account_balance()
        equity = balance + sum(p.unrealized_pnl for p in positions)
        
        return {
            'steps': self._current_step,
            'start_balance': self._episode_start_balance,
            'current_balance': balance,
            'current_equity': equity,
            'episode_pnl': self._episode_pnl,
            'return_pct': (equity / self._episode_start_balance - 1) * 100 if self._episode_start_balance > 0 else 0,
            'open_positions': len(positions),
            'trades_this_hour': self._trades_this_hour,
        }


def create_trading_environment(
    mode: str = 'simulation',
    initial_balance: float = 5000.0,
    config: Optional[Dict] = None,
    config_path: str = 'config.json'
) -> TradingEnvironment:
    """
    Factory function to create TradingEnvironment.
    
    Args:
        mode: 'simulation', 'paper', or 'live'
        initial_balance: Starting balance
        config: Config dict (loads from file if None)
        config_path: Path to config file
        
    Returns:
        Configured TradingEnvironment
    """
    if config is None:
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
        except Exception:
            config = {}
    
    trading_mode = TradingMode(mode)
    
    # Create broker based on mode
    if trading_mode == TradingMode.SIMULATION:
        broker = SimulatedBroker(initial_balance=initial_balance)
    else:
        # For paper/live, would use real broker adapter
        # For now, use simulated
        broker = SimulatedBroker(initial_balance=initial_balance)
    
    # Create risk manager
    try:
        from backend.risk_manager import RiskManager
        risk_manager = RiskManager(config=config)
    except ImportError:
        risk_manager = None
    
    return TradingEnvironment(
        broker=broker,
        config=config,
        mode=trading_mode,
        risk_manager=risk_manager,
        symbol=config.get('trading', {}).get('symbol', 'SPY')
    )

