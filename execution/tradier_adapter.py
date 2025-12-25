"""
Tradier API Adapter for Liquidity Execution Layer

Maps liquidity executor API calls to Tradier REST API
"""

import pandas as pd
import logging
from typing import Dict, Optional, List
from backend.tradier_trading_system import TradierTradingSystem, OrderType

logger = logging.getLogger(__name__)


class TradierAdapter:
    """
    Adapter between LiquidityExecutor and Tradier API
    
    Maps the execution layer's interface to Tradier REST API calls
    """
    
    def __init__(self, tradier_system: TradierTradingSystem):
        """
        Initialize adapter with existing TradierTradingSystem
        
        Args:
            tradier_system: Initialized TradierTradingSystem instance
        """
        self.system = tradier_system
        self.account_id = tradier_system.account_id
        self.headers = tradier_system.headers
        self.base_url = tradier_system.base_url
        logger.info("[TRADIER-ADAPTER] Initialized")
    
    def get_option_chain(self, symbol: str, expiry: str) -> pd.DataFrame:
        """
        Get options chain as DataFrame
        
        Args:
            symbol: Underlying symbol (e.g., 'BITX')
            expiry: Expiration date 'YYYY-MM-DD'
        
        Returns:
            DataFrame with columns: symbol, occ_symbol, strike, type, bid, ask, 
            bidsize, asksize, open_interest, volume, delta
        """
        logger.info(f"[TRADIER-ADAPTER] Fetching chain for {symbol} {expiry}")
        
        # Use existing system method
        chain = self.system.get_options_chain(symbol, expiry)
        
        if not chain:
            logger.warning(f"[TRADIER-ADAPTER] No chain data for {symbol} {expiry}")
            return pd.DataFrame()
        
        # Convert to DataFrame with standardized columns
        rows = []
        for opt in chain:
            # Extract Greeks if available
            greeks = opt.get('greeks', {}) or {}
            delta = greeks.get('delta', 0.0) if greeks else 0.0
            
            rows.append({
                'symbol': opt.get('symbol'),  # Underlying
                'occ_symbol': opt.get('symbol'),  # Option symbol (OCC format)
                'strike': float(opt.get('strike', 0)),
                'type': opt.get('option_type', ''),  # 'call' or 'put'
                'bid': float(opt.get('bid', 0)),
                'ask': float(opt.get('ask', 0)),
                'bidsize': int(opt.get('bidsize', 0)),
                'asksize': int(opt.get('asksize', 0)),
                'open_interest': int(opt.get('open_interest', 0)),
                'volume': int(opt.get('volume', 0)),
                'delta': float(delta),
                'last': float(opt.get('last', 0)),
                'change': float(opt.get('change', 0)),
                'change_percentage': float(opt.get('change_percentage', 0))
            })
        
        df = pd.DataFrame(rows)
        logger.info(f"[TRADIER-ADAPTER] Retrieved {len(df)} options")
        
        return df
    
    def get_quote(self, occ_symbol: str) -> Dict:
        """
        Get real-time quote for option symbol
        
        Args:
            occ_symbol: Option symbol in OCC format
        
        Returns:
            Dict with: {bid, ask, bidsize, asksize, last, volume}
        """
        logger.info(f"[TRADIER-ADAPTER] Getting quote for {occ_symbol}")
        
        # Call Tradier quotes API
        response = self.system._make_request(
            'GET', 
            '/markets/quotes',
            params={'symbols': occ_symbol, 'greeks': 'false'}
        )
        
        if not response or 'quotes' not in response:
            logger.error(f"[TRADIER-ADAPTER] Failed to get quote for {occ_symbol}")
            return {
                'bid': 0.0,
                'ask': 0.0,
                'bidsize': 0,
                'asksize': 0,
                'last': 0.0,
                'volume': 0
            }
        
        # Extract quote data
        quotes = response['quotes']
        quote = quotes.get('quote', {})
        
        # Handle both single quote dict and list
        if isinstance(quote, list):
            quote = quote[0] if quote else {}
        
        result = {
            'bid': float(quote.get('bid', 0)),
            'ask': float(quote.get('ask', 0)),
            'bidsize': int(quote.get('bidsize', 0)),
            'asksize': int(quote.get('asksize', 0)),
            'last': float(quote.get('last', 0)),
            'volume': int(quote.get('volume', 0))
        }
        
        logger.info(f"[TRADIER-ADAPTER] Quote: bid=${result['bid']:.2f} x{result['bidsize']}, "
                   f"ask=${result['ask']:.2f} x{result['asksize']}")
        
        return result
    
    def place_limit(self, occ_symbol: str, side: str, qty: int, 
                   limit_price: float, tif: str) -> str:
        """
        Place limit order
        
        Args:
            occ_symbol: Option symbol
            side: 'buy_to_open', 'sell_to_open', etc.
            qty: Quantity
            limit_price: Limit price
            tif: Time in force ('day', 'gtc', 'ioc')
        
        Returns:
            Order ID string
        """
        logger.info(f"[TRADIER-ADAPTER] Placing limit order: {side} {qty} {occ_symbol} @ ${limit_price:.2f}")
        
        # Map side string to OrderType
        side_map = {
            'buy_to_open': OrderType.BUY_TO_OPEN,
            'buy_to_close': OrderType.BUY_TO_CLOSE,
            'sell_to_open': OrderType.SELL_TO_OPEN,
            'sell_to_close': OrderType.SELL_TO_CLOSE
        }
        
        order_side = side_map.get(side.lower())
        if not order_side:
            logger.error(f"[TRADIER-ADAPTER] Invalid side: {side}")
            return None
        
        # Extract underlying symbol from option symbol
        # OCC format: BITX241219C00046000 -> BITX
        underlying = occ_symbol[:4] if len(occ_symbol) > 4 else occ_symbol
        
        # Place order using existing system
        order_type = 'limit'
        if tif.lower() == 'ioc':
            # IOC orders are market orders with immediate-or-cancel
            order_type = 'market'
            tif = 'day'  # Tradier doesn't support IOC, use market order
        
        # Build order data
        order_data = {
            'class': 'option',
            'symbol': underlying,
            'option_symbol': occ_symbol,
            'side': side.lower(),
            'quantity': str(qty),
            'type': order_type,
            'duration': tif.lower()
        }
        
        if order_type == 'limit':
            order_data['price'] = f"{limit_price:.2f}"
        
        response = self.system._make_request(
            'POST', 
            f"/accounts/{self.account_id}/orders", 
            data=order_data
        )
        
        if response and 'order' in response:
            order = response['order']
            order_id = order.get('id')
            status = order.get('status')
            
            logger.info(f"[TRADIER-ADAPTER] Order placed: ID={order_id}, Status={status}")
            return str(order_id)
        
        logger.error(f"[TRADIER-ADAPTER] Failed to place order")
        return None
    
    def modify_order(self, order_id: str, limit_price: float) -> None:
        """
        Modify existing order price
        
        Args:
            order_id: Order ID
            limit_price: New limit price
        """
        logger.info(f"[TRADIER-ADAPTER] Modifying order {order_id} to ${limit_price:.2f}")
        
        # Tradier modify order API
        order_data = {
            'type': 'limit',
            'price': f"{limit_price:.2f}",
            'duration': 'day'
        }
        
        response = self.system._make_request(
            'PUT',
            f"/accounts/{self.account_id}/orders/{order_id}",
            data=order_data
        )
        
        if response:
            logger.info(f"[TRADIER-ADAPTER] Order {order_id} modified")
        else:
            logger.error(f"[TRADIER-ADAPTER] Failed to modify order {order_id}")
    
    def cancel_order(self, order_id: str) -> None:
        """
        Cancel order
        
        Args:
            order_id: Order ID
        """
        logger.info(f"[TRADIER-ADAPTER] Cancelling order {order_id}")
        self.system.cancel_order(order_id)
    
    def place_spread_limit(self, legs: List[Dict], side: str, qty: int, 
                          net_price: float, tif: str) -> str:
        """
        Place multi-leg spread order
        
        Args:
            legs: List of leg dicts: [{"occ": "...", "side": "buy_to_open", "ratio": 1}, ...]
            side: 'debit' or 'credit'
            qty: Quantity
            net_price: Net debit/credit
            tif: Time in force
        
        Returns:
            Order ID string
        """
        logger.info(f"[TRADIER-ADAPTER] Placing spread: {side} spread @ ${net_price:.2f}")
        logger.info(f"[TRADIER-ADAPTER] Legs: {legs}")
        
        # Extract underlying from first leg
        first_leg = legs[0]
        occ = first_leg['occ']
        underlying = occ[:4] if len(occ) > 4 else occ
        
        # Build multileg order
        order_data = {
            'class': 'multileg',
            'symbol': underlying,
            'type': 'limit',
            'duration': tif.lower(),
            'price': f"{net_price:.2f}"
        }
        
        # Add legs
        for i, leg in enumerate(legs):
            order_data[f'option_symbol[{i}]'] = leg['occ']
            order_data[f'side[{i}]'] = leg['side']
            order_data[f'quantity[{i}]'] = str(qty * leg.get('ratio', 1))
        
        response = self.system._make_request(
            'POST',
            f"/accounts/{self.account_id}/orders",
            data=order_data
        )
        
        if response and 'order' in response:
            order = response['order']
            order_id = order.get('id')
            logger.info(f"[TRADIER-ADAPTER] Spread placed: ID={order_id}")
            return str(order_id)
        
        logger.error(f"[TRADIER-ADAPTER] Failed to place spread")
        return None
    
    def filled_qty(self, order_id: str) -> int:
        """
        Get filled quantity for order
        
        Args:
            order_id: Order ID
        
        Returns:
            Filled quantity
        """
        # Get order status
        status = self.system.get_order_status(order_id)
        
        if not status:
            logger.warning(f"[TRADIER-ADAPTER] Could not get status for order {order_id}")
            return 0
        
        # Check status
        order_status = status.get('status', '').lower()
        
        if order_status in ['filled', 'completely_filled']:
            qty = int(status.get('quantity', 0))
            logger.info(f"[TRADIER-ADAPTER] Order {order_id} filled: {qty}")
            return qty
        
        elif order_status == 'partially_filled':
            exec_qty = int(status.get('exec_quantity', 0))
            logger.info(f"[TRADIER-ADAPTER] Order {order_id} partially filled: {exec_qty}")
            return exec_qty
        
        else:
            logger.info(f"[TRADIER-ADAPTER] Order {order_id} not filled (status: {order_status})")
            return 0


