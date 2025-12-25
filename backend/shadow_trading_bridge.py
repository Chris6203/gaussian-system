#!/usr/bin/env python3
"""
Shadow Trading Bridge
=====================

Bridges paper trades to Tradier shadow/live orders.
Handles entry mirroring, exit syncing, and order verification.

This module extracts shadow trading logic from train_then_go_live.py
for better separation of concerns and testability.
"""

import logging
from typing import Dict, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ShadowTradeResult:
    """Result of a shadow trade operation."""
    success: bool
    order_id: Optional[str] = None
    option_symbol: Optional[str] = None
    status: Optional[str] = None
    reason: Optional[str] = None
    verified: bool = False


class ShadowTradingBridge:
    """
    Bridges paper trades to Tradier shadow/live orders.
    
    Responsibilities:
    - Mirror paper trade entries to Tradier
    - Mirror paper trade exits to Tradier
    - Verify order fills
    - Handle errors gracefully without affecting paper trading
    """
    
    def __init__(
        self,
        tradier_system: Any,
        paper_only_mode: bool = True,
        verify_entry_fill: bool = True,
        verify_exit_fill: bool = True,
        entry_fill_timeout: int = 5,
        exit_fill_timeout: int = 10
    ):
        """
        Initialize the shadow trading bridge.
        
        Args:
            tradier_system: TradierTradingSystem instance (or None if disabled)
            paper_only_mode: If True, no real trades are placed
            verify_entry_fill: Whether to verify entry orders are filled
            verify_exit_fill: Whether to verify exit orders are filled
            entry_fill_timeout: Seconds to wait for entry fill verification
            exit_fill_timeout: Seconds to wait for exit fill verification
        """
        self.tradier = tradier_system
        self.paper_only_mode = paper_only_mode
        self.verify_entry_fill = verify_entry_fill
        self.verify_exit_fill = verify_exit_fill
        self.entry_fill_timeout = entry_fill_timeout
        self.exit_fill_timeout = exit_fill_timeout
        
        self._log_init_status()
    
    def _log_init_status(self) -> None:
        """Log initialization status."""
        if self.paper_only_mode:
            logger.info("[SHADOW] Paper-only mode - no live trades will be placed")
        elif self.tradier is None:
            logger.warning("[SHADOW] No Tradier system provided - shadow trading disabled")
        else:
            mode = "SANDBOX" if getattr(self.tradier, 'sandbox', True) else "LIVE"
            logger.info(f"[SHADOW] Bridge initialized in {mode} mode")
            logger.info(f"[SHADOW] Entry fill verification: {self.verify_entry_fill}")
            logger.info(f"[SHADOW] Exit fill verification: {self.verify_exit_fill}")
    
    @property
    def is_active(self) -> bool:
        """Check if shadow trading is active (not paper-only and has tradier)."""
        return not self.paper_only_mode and self.tradier is not None
    
    def mirror_entry(
        self,
        paper_trade: Any,
        signal: Dict,
        symbol: str,
        current_price: float
    ) -> ShadowTradeResult:
        """
        Mirror a paper trade entry to Tradier.
        
        Args:
            paper_trade: The paper trade object that was just placed
            signal: The signal dict that triggered the trade
            symbol: The underlying symbol (e.g., 'SPY')
            current_price: Current price of the underlying
            
        Returns:
            ShadowTradeResult with success status and details
        """
        action = signal.get('action', 'HOLD')
        
        # Paper-only mode - just log and return
        if self.paper_only_mode:
            logger.info(f"[SHADOW] Paper-only mode - {action} recorded for learning")
            return ShadowTradeResult(
                success=True,
                reason="Paper-only mode - no live trade placed"
            )
        
        # No tradier system available
        if self.tradier is None:
            logger.error("[SHADOW] No Tradier system available!")
            logger.error(f"[SHADOW] Paper trade placed but NOT mirrored: {action}")
            return ShadowTradeResult(
                success=False,
                reason="No Tradier system available"
            )
        
        # Attempt to mirror the trade
        try:
            logger.info(f"[SHADOW] Mirroring {action} to Tradier...")
            
            # Build shadow signal with paper trade parameters
            shadow_signal = signal.copy()
            shadow_signal['quantity'] = paper_trade.quantity
            shadow_signal['strike_price'] = paper_trade.strike_price
            
            logger.info(
                f"[SHADOW] Using paper trade params: {paper_trade.quantity} contracts "
                f"@ ${paper_trade.strike_price:.2f} strike"
            )
            
            # Place the shadow trade
            shadow_trade = self.tradier.place_trade(symbol, shadow_signal, current_price)
            
            if shadow_trade is None:
                logger.warning("[SHADOW] Shadow trader place_trade() returned None")
                logger.warning("[SHADOW] Trade blocked by safety checks (market hours, liquidity, etc.)")
                return ShadowTradeResult(
                    success=False,
                    reason="Blocked by safety checks"
                )
            
            # Check if trade was blocked
            if shadow_trade.get('blocked'):
                reason = shadow_trade.get('reason', 'Unknown')
                logger.warning(f"[SHADOW] Trade blocked: {reason}")
                return ShadowTradeResult(
                    success=False,
                    reason=f"Blocked: {reason}"
                )
            
            order_id = shadow_trade.get('order_id')
            order_status = shadow_trade.get('status')
            option_symbol = shadow_trade.get('option_symbol')
            
            logger.info(f"[SHADOW] Trade mirrored: Order ID={order_id}, Status={order_status}")
            
            # Verify fill if enabled
            verified = False
            if self.verify_entry_fill and order_id:
                verified = self._verify_order_fill(
                    order_id, 
                    self.entry_fill_timeout,
                    "entry"
                )
            
            # Link paper trade to Tradier position for auto-close later
            if option_symbol:
                paper_trade.tradier_option_symbol = option_symbol
                paper_trade.tradier_order_id = order_id
                logger.info(f"[SHADOW] Linked paper trade {paper_trade.id} → Tradier {option_symbol}")
            
            return ShadowTradeResult(
                success=True,
                order_id=order_id,
                option_symbol=option_symbol,
                status=order_status,
                verified=verified
            )
            
        except Exception as e:
            logger.error(f"[SHADOW] Error mirroring trade: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return ShadowTradeResult(
                success=False,
                reason=f"Exception: {e}"
            )
    
    def mirror_exit(
        self,
        paper_trade: Any,
        symbol: str,
        exit_reason: str = "Unknown"
    ) -> ShadowTradeResult:
        """
        Mirror a paper trade exit to Tradier.
        
        Args:
            paper_trade: The paper trade that was just closed
            symbol: The underlying symbol (e.g., 'SPY')
            exit_reason: Reason for closing (e.g., "Stop loss", "Take profit")
            
        Returns:
            ShadowTradeResult with success status and details
        """
        tradier_option_symbol = getattr(paper_trade, 'tradier_option_symbol', None)
        
        # No Tradier position to close
        if not tradier_option_symbol:
            logger.warning(f"[SHADOW] No Tradier symbol tracked for closed trade {paper_trade.id}")
            return ShadowTradeResult(
                success=False,
                reason="No Tradier position linked"
            )
        
        # Paper-only mode
        if self.paper_only_mode:
            logger.info(f"[SHADOW] Paper-only mode - exit not mirrored: {tradier_option_symbol}")
            return ShadowTradeResult(
                success=True,
                option_symbol=tradier_option_symbol,
                reason="Paper-only mode - no live close"
            )
        
        # No tradier system
        if self.tradier is None:
            logger.error(f"[SHADOW] Cannot close Tradier position - no system available")
            return ShadowTradeResult(
                success=False,
                option_symbol=tradier_option_symbol,
                reason="No Tradier system available"
            )
        
        # Log exit details
        entry_price = getattr(paper_trade, 'entry_price', 0)
        exit_price = getattr(paper_trade, 'exit_price', 0)
        pnl = ((exit_price - entry_price) / entry_price * 100) if entry_price > 0 else 0
        
        logger.info(f"[SHADOW] Auto-closing Tradier position: {tradier_option_symbol}")
        logger.info(f"[SHADOW] Close reason: {exit_reason}")
        logger.info(f"[SHADOW] P&L: {pnl:+.1f}%")
        
        try:
            close_result = self.tradier.close_position(
                option_symbol=tradier_option_symbol,
                quantity=paper_trade.quantity,
                symbol=symbol
            )
            
            if not close_result:
                logger.warning(f"[SHADOW] Failed to close Tradier position: {tradier_option_symbol}")
                return ShadowTradeResult(
                    success=False,
                    option_symbol=tradier_option_symbol,
                    reason="Close order failed"
                )
            
            order_id = close_result.get('order_id')
            logger.info(f"[SHADOW] Close order placed: Order #{order_id}")
            
            # Verify fill if enabled
            verified = False
            if self.verify_exit_fill and order_id:
                verified = self._verify_order_fill(
                    order_id,
                    self.exit_fill_timeout,
                    "exit"
                )
                
                if not verified:
                    logger.error(f"[SHADOW] Position {tradier_option_symbol} may still be open!")
                    logger.error("[SHADOW] MANUAL INTERVENTION MAY BE REQUIRED")
            
            return ShadowTradeResult(
                success=True,
                order_id=order_id,
                option_symbol=tradier_option_symbol,
                verified=verified
            )
            
        except Exception as e:
            logger.error(f"[SHADOW] Error closing Tradier position: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return ShadowTradeResult(
                success=False,
                option_symbol=tradier_option_symbol,
                reason=f"Exception: {e}"
            )
    
    def _verify_order_fill(
        self,
        order_id: str,
        timeout_seconds: int,
        order_type: str
    ) -> bool:
        """
        Verify an order was filled.
        
        Args:
            order_id: The order ID to verify
            timeout_seconds: How long to wait for fill
            order_type: "entry" or "exit" for logging
            
        Returns:
            True if order was confirmed filled, False otherwise
        """
        logger.info(f"[SHADOW] Verifying {order_type} order fill...")
        
        verification = self.tradier.verify_order_filled(
            order_id, 
            max_wait_seconds=timeout_seconds
        )
        
        if verification['success'] is True:
            logger.info(f"[SHADOW] ✅ {order_type.capitalize()} order {order_id} CONFIRMED FILLED")
            return True
        elif verification['success'] is False:
            reason = verification.get('reason', 'Unknown')
            logger.error(f"[SHADOW] ❌ {order_type.capitalize()} order {order_id} REJECTED: {reason}")
            return False
        else:
            # Still pending
            logger.warning(f"[SHADOW] ⏳ {order_type.capitalize()} order {order_id} still pending after {timeout_seconds}s")
            return False
    
    def sync_closed_positions(
        self,
        closed_positions: list,
        symbol: str
    ) -> Dict[str, ShadowTradeResult]:
        """
        Sync multiple closed paper positions to Tradier.
        
        Args:
            closed_positions: List of dicts with 'trade' and metadata
            symbol: The underlying symbol
            
        Returns:
            Dict mapping trade_id to ShadowTradeResult
        """
        results = {}
        
        if not closed_positions:
            return results
        
        logger.info(f"[SHADOW] Syncing {len(closed_positions)} closed position(s) to Tradier")
        
        for meta in closed_positions:
            trade = meta.get('trade')
            if not trade:
                continue
            
            exit_reason = getattr(trade, 'exit_reason', 'Unknown')
            exit_status = trade.status.value if hasattr(trade, 'status') else 'Unknown'
            
            logger.info(f"[SHADOW] Processing closed trade: {exit_status} ({exit_reason})")
            
            result = self.mirror_exit(trade, symbol, exit_reason)
            results[trade.id] = result
            
            # Log result for user
            if result.success:
                if result.verified:
                    print(f"  [✓] SHADOW: Closed on Tradier: {result.option_symbol} ({exit_status})")
                else:
                    print(f"  [!] SHADOW: Close pending - monitor {result.option_symbol}")
            else:
                print(f"  [!] SHADOW: Close failed - {result.reason}")
        
        return results





