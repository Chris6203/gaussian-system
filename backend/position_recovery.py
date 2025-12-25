#!/usr/bin/env python3
"""
Position Recovery Module
========================

Handles evaluation and cleanup of existing positions on bot restart.
Decides whether to keep or close stale/problematic positions.

This module extracts position recovery logic from train_then_go_live.py
for better separation of concerns and testability.
"""

import logging
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RecoveryConfig:
    """Configuration for position recovery logic."""
    max_stale_hours: float = 24.0
    stale_loss_threshold_pct: float = -5.0
    extreme_loss_threshold_pct: float = -50.0
    profit_near_expiry_threshold_pct: float = 10.0
    near_profit_threshold_pct: float = 15.0
    recent_trade_hours: float = 1.0
    recent_trade_max_loss_pct: float = -20.0
    
    @classmethod
    def from_config(cls, config: Any) -> 'RecoveryConfig':
        """Create from BotConfig instance."""
        recovery_cfg = config.get('live_trading', 'position_recovery', default={})
        return cls(
            max_stale_hours=recovery_cfg.get('max_stale_hours', 24.0),
            stale_loss_threshold_pct=recovery_cfg.get('stale_loss_threshold_pct', -5.0),
            extreme_loss_threshold_pct=recovery_cfg.get('extreme_loss_threshold_pct', -50.0),
            profit_near_expiry_threshold_pct=recovery_cfg.get('profit_near_expiry_threshold_pct', 10.0),
            near_profit_threshold_pct=recovery_cfg.get('near_profit_threshold_pct', 15.0),
            recent_trade_hours=recovery_cfg.get('recent_trade_hours', 1.0),
            recent_trade_max_loss_pct=recovery_cfg.get('recent_trade_max_loss_pct', -20.0)
        )


@dataclass
class PositionEvaluation:
    """Result of evaluating a single position."""
    trade_id: str
    should_close: bool
    reason: Optional[str] = None
    hours_held: float = 0.0
    pnl_percent: float = 0.0
    current_value: float = 0.0


class PositionRecovery:
    """
    Evaluates and manages existing positions on bot restart.
    
    Responsibilities:
    - Detect stale positions that should be closed
    - Identify positions with extreme losses
    - Preserve profitable or recent positions
    - Execute position cleanup
    """
    
    def __init__(self, config: RecoveryConfig = None):
        """
        Initialize position recovery.
        
        Args:
            config: RecoveryConfig instance, or None for defaults
        """
        self.config = config or RecoveryConfig()
    
    def evaluate_positions(
        self,
        paper_trader: Any,
        current_price: float,
        calculate_option_value_fn: callable = None
    ) -> Tuple[List[PositionEvaluation], List[PositionEvaluation]]:
        """
        Evaluate all active positions and categorize them.
        
        Args:
            paper_trader: Paper trading system with active_trades
            current_price: Current underlying price
            calculate_option_value_fn: Function to calculate current option value
                                      (defaults to paper_trader._calculate_option_premium)
            
        Returns:
            Tuple of (positions_to_close, positions_to_keep)
        """
        active_trades = [
            t for t in paper_trader.active_trades 
            if t.status.value == 'FILLED'
        ]
        
        if not active_trades:
            logger.info("[RECOVERY] No open positions to evaluate")
            return [], []
        
        logger.info(f"[RECOVERY] Evaluating {len(active_trades)} existing position(s)...")
        
        # Use paper trader's method if not provided
        if calculate_option_value_fn is None:
            calculate_option_value_fn = paper_trader._calculate_option_premium
        
        positions_to_close = []
        positions_to_keep = []
        
        for trade in active_trades:
            evaluation = self._evaluate_single_position(
                trade, 
                current_price,
                calculate_option_value_fn
            )
            
            if evaluation.should_close:
                positions_to_close.append(evaluation)
            else:
                positions_to_keep.append(evaluation)
        
        return positions_to_close, positions_to_keep
    
    def _evaluate_single_position(
        self,
        trade: Any,
        current_price: float,
        calculate_option_value_fn: callable
    ) -> PositionEvaluation:
        """
        Evaluate a single position for keep/close decision.
        
        Args:
            trade: Trade object to evaluate
            current_price: Current underlying price
            calculate_option_value_fn: Function to calculate option value
            
        Returns:
            PositionEvaluation with decision and reason
        """
        # Calculate time held
        entry_time = trade.timestamp
        if isinstance(entry_time, str):
            from dateutil import parser
            entry_time = parser.parse(entry_time)
        
        time_held = datetime.now() - entry_time
        hours_held = time_held.total_seconds() / 3600
        
        # Calculate current value and P&L
        try:
            days_to_expiry = self._get_days_to_expiry(trade)
            
            current_value = calculate_option_value_fn(
                trade.option_type.upper(),
                current_price,
                trade.strike_price,
                days_to_expiry
            )
            
            pnl_percent = ((current_value - trade.entry_price) / trade.entry_price) * 100
            
        except Exception as e:
            # If we can't calculate, keep the position (safer)
            logger.warning(f"[RECOVERY] Could not calculate value for {trade.id}: {e}, keeping position")
            return PositionEvaluation(
                trade_id=trade.id,
                should_close=False,
                reason=f"Could not evaluate: {e}",
                hours_held=hours_held
            )
        
        # Apply decision rules
        should_close, reason = self._apply_decision_rules(
            trade, hours_held, pnl_percent, days_to_expiry
        )
        
        # Log decision
        self._log_evaluation(trade, hours_held, pnl_percent, should_close, reason)
        
        return PositionEvaluation(
            trade_id=trade.id,
            should_close=should_close,
            reason=reason,
            hours_held=hours_held,
            pnl_percent=pnl_percent,
            current_value=current_value
        )
    
    def _get_days_to_expiry(self, trade: Any) -> int:
        """Get days to expiry from trade, with fallback."""
        if trade.expiration_date:
            if isinstance(trade.expiration_date, str):
                from dateutil import parser
                exp_date = parser.parse(trade.expiration_date)
            else:
                exp_date = trade.expiration_date
            return max(0, (exp_date - datetime.now()).days)
        else:
            # Default if no expiration date available
            return 30
    
    def _apply_decision_rules(
        self,
        trade: Any,
        hours_held: float,
        pnl_percent: float,
        days_to_expiry: int
    ) -> Tuple[bool, Optional[str]]:
        """
        Apply decision rules to determine if position should be closed.
        
        Returns:
            Tuple of (should_close, reason)
        """
        cfg = self.config
        
        # Rule 1: Close stale positions with loss
        if hours_held > cfg.max_stale_hours and pnl_percent < cfg.stale_loss_threshold_pct:
            return True, f"stale position ({hours_held:.1f}h old, {pnl_percent:.1f}% loss)"
        
        # Rule 2: Close near-expiry positions with profit (lock in gains)
        if days_to_expiry <= 1 and pnl_percent > cfg.profit_near_expiry_threshold_pct:
            return True, f"near expiry with profit ({pnl_percent:.1f}%, {days_to_expiry}d left)"
        
        # Rule 3: Close extreme losses (safety check)
        if pnl_percent < cfg.extreme_loss_threshold_pct:
            return True, f"extreme loss ({pnl_percent:.1f}%)"
        
        # Rule 4: Keep positions near take profit
        if pnl_percent > cfg.near_profit_threshold_pct:
            return False, f"near profit target (+{pnl_percent:.1f}%)"
        
        # Rule 5: Keep recent positions with manageable loss
        if hours_held < cfg.recent_trade_hours and pnl_percent > cfg.recent_trade_max_loss_pct:
            return False, f"recent trade ({hours_held*60:.0f}min old, {pnl_percent:+.1f}%)"
        
        # Rule 6: Keep positions in normal range
        if cfg.recent_trade_max_loss_pct < pnl_percent < cfg.near_profit_threshold_pct:
            return False, f"normal range ({pnl_percent:+.1f}%, {hours_held:.1f}h old)"
        
        # Default: keep position
        return False, "default keep"
    
    def _log_evaluation(
        self,
        trade: Any,
        hours_held: float,
        pnl_percent: float,
        should_close: bool,
        reason: str
    ) -> None:
        """Log position evaluation decision."""
        symbol = "✗" if should_close else "✓"
        action = "Closing" if should_close else "Keeping"
        
        logger.info(
            f"[RECOVERY] {symbol} {action} {trade.option_type} @ ${trade.entry_price:.2f}: {reason}"
        )
    
    def execute_recovery(
        self,
        paper_trader: Any,
        positions_to_close: List[PositionEvaluation]
    ) -> int:
        """
        Execute recovery by closing identified positions.
        
        Args:
            paper_trader: Paper trading system
            positions_to_close: List of PositionEvaluation objects to close
            
        Returns:
            Number of positions successfully closed
        """
        if not positions_to_close:
            return 0
        
        logger.info(f"[RECOVERY] Closing {len(positions_to_close)} position(s)")
        
        closed_count = 0
        for evaluation in positions_to_close:
            try:
                paper_trader.close_trade(
                    evaluation.trade_id,
                    exit_price=evaluation.current_value,
                    reason=f"Auto-closed on restart: {evaluation.reason}"
                )
                logger.info(f"[RECOVERY] Closed {evaluation.trade_id}")
                closed_count += 1
            except Exception as e:
                logger.error(f"[RECOVERY] Error closing {evaluation.trade_id}: {e}")
        
        return closed_count


def evaluate_existing_positions(
    bot: Any,
    symbol: str,
    config: Any = None,
    data_source: Any = None,
    historical_period: str = '7d'
) -> None:
    """
    Convenience function to evaluate existing positions on restart.
    
    Args:
        bot: UnifiedOptionsBot instance
        symbol: Trading symbol
        config: BotConfig instance (optional)
        data_source: Data source for current price (uses bot.data_source if None)
        historical_period: Period for fetching current price
    """
    print(f"\n[*] Evaluating existing positions...")
    
    # Get current price
    source = data_source or bot.data_source
    
    try:
        data = source.get_data(symbol, period=historical_period, interval='1m')
        if data.empty:
            logger.warning("[RECOVERY] Could not get current price, keeping all positions")
            return
        
        data.columns = [col.lower() for col in data.columns]
        current_price = float(data['close'].iloc[-1])
        logger.info(f"[RECOVERY] Current price: ${current_price:.2f}")
        
    except Exception as e:
        logger.error(f"[RECOVERY] Error getting price: {e}")
        return
    
    # Create recovery config from bot config
    if config:
        recovery_config = RecoveryConfig.from_config(config)
    else:
        recovery_config = RecoveryConfig()
    
    # Run recovery
    recovery = PositionRecovery(recovery_config)
    positions_to_close, positions_to_keep = recovery.evaluate_positions(
        bot.paper_trader,
        current_price
    )
    
    # Execute closures
    if positions_to_close:
        for eval_result in positions_to_close:
            print(f"  [!] Closing bad position: {eval_result.reason}")
        
        closed = recovery.execute_recovery(bot.paper_trader, positions_to_close)
        logger.info(f"[RECOVERY] Closed {closed} position(s)")
    
    # Log final state
    final_count = len([
        t for t in bot.paper_trader.active_trades 
        if t.status.value == 'FILLED'
    ])
    logger.info(f"[RECOVERY] Complete: {final_count} position(s) remain active")
    print(f"  [✓] Recovery complete: {final_count} positions active")





