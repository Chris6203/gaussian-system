#!/usr/bin/env python3
"""
Tradier Sandbox Trading System
================================

Full integration with Tradier API for live sandbox trading.
Supports all bot signals and includes realistic fee calculations.

Supported Operations:
- BUY_CALLS: Buy call options
- BUY_PUTS: Buy put options
- BUY_STRADDLE: Buy call + put at same strike
- SELL_PREMIUM: Sell put options (credit spreads)

API Endpoints:
- POST /v1/accounts/{id}/orders - Place orders
- GET /v1/accounts/{id}/positions - Get positions
- GET /v1/accounts/{id}/orders - Get order history
- GET /v1/accounts/{id}/balances - Get account balance
- GET /v1/markets/options/chains - Get options chain
- GET /v1/markets/options/expirations - Get expiration dates
"""

import requests
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import pytz

logger = logging.getLogger(__name__)

# Import liquidity validator
try:
    from backend.liquidity_validator import LiquidityValidator
    HAS_LIQUIDITY_VALIDATOR = True
except ImportError:
    logger.warning("[TRADIER] Liquidity validator not available - trades will not be checked for liquidity")
    HAS_LIQUIDITY_VALIDATOR = False


class OrderType(Enum):
    """Tradier order types"""
    BUY_TO_OPEN = "buy_to_open"
    BUY_TO_CLOSE = "buy_to_close"
    SELL_TO_OPEN = "sell_to_open"
    SELL_TO_CLOSE = "sell_to_close"


class OrderClass(Enum):
    """Tradier order classes"""
    EQUITY = "equity"
    OPTION = "option"
    MULTILEG = "multileg"
    COMBO = "combo"


class OrderDuration(Enum):
    """Order duration types"""
    DAY = "day"
    GTC = "gtc"
    PRE = "pre"
    POST = "post"


@dataclass
class TradierFees:
    """Tradier fee structure (accurate as of 2024)"""
    
    # Commission (Tradier charges per-contract for options)
    option_contract_fee: float = 0.35  # $0.35 per contract (Tradier Select)
    
    # Regulatory fees (passed through to customer)
    options_regulatory_fee: float = 0.04  # $0.04 per contract (ORF)
    finra_trading_activity_fee: float = 0.002  # $0.002 per contract (TAF)
    sec_fee_rate: float = 0.000008  # 0.0008% of principal (rounded)
    
    # Exchange fees (Tradier Select passes through)
    exchange_fee: float = 0.01  # ~$0.01 per contract (varies by exchange)
    
    # Total per-contract cost (typical)
    @property
    def total_per_contract(self) -> float:
        """Total cost per options contract"""
        return (
            self.option_contract_fee +
            self.options_regulatory_fee +
            self.finra_trading_activity_fee +
            self.exchange_fee
        )  # ~$0.42 per contract


class TradierTradingSystem:
    """
    Full Tradier Sandbox trading integration
    Places real orders via Tradier API with proper fee calculations
    """
    
    def __init__(self, 
                 account_id: str,
                 api_token: str,
                 sandbox: bool = True,
                 initial_balance: float = 100000.0,
                 db_path: str = 'data/unified_options_bot.db'):
        """
        Initialize Tradier trading system
        
        Args:
            account_id: Tradier account number (e.g., VA92393335)
            api_token: Tradier API access token
            sandbox: If True, use sandbox API (fake money)
            initial_balance: Starting balance (sandbox only)
            db_path: SQLite database path for local tracking
        """
        self.account_id = account_id
        self.api_token = api_token
        self.sandbox = sandbox
        self.sandbox_mode = sandbox  # Alias for compatibility
        self.initial_balance = initial_balance
        self.db_path = db_path
        
        # API configuration
        if sandbox:
            self.base_url = "https://sandbox.tradier.com/v1"
            logger.info(f"[SANDBOX] Tradier Sandbox mode enabled")
        else:
            self.base_url = "https://api.tradier.com/v1"
            logger.warning(f"[LIVE] Tradier LIVE mode - REAL MONEY!")
        
        self.headers = {
            'Authorization': f'Bearer {api_token}',
            'Accept': 'application/json'
        }
        
        # Fee calculator
        self.fees = TradierFees()
        
        # Active trades (tracked locally)
        self.active_trades = []
        
        # Position tracking
        self.current_balance = initial_balance
        self.max_positions = 5
        
        # Initialize liquidity validator
        if HAS_LIQUIDITY_VALIDATOR:
            self.liquidity_validator = LiquidityValidator(
                max_bid_ask_spread_pct=5.0,    # Aligned with LiquidityExecutor (4%) + 1% buffer
                min_volume=100,                 # Aligned with LiquidityExecutor (150) - 50 buffer
                min_open_interest=75,           # Aligned with LiquidityExecutor (100) - 25 buffer
                max_strike_deviation_pct=15.0   # Within 15% of spot (unchanged)
            )
            logger.info(f"[TRADIER] Liquidity validation enabled")
        else:
            self.liquidity_validator = None
            logger.warning(f"[TRADIER] Liquidity validation DISABLED (validator not available)")
        
        logger.info(f"[TRADIER] Trading System initialized")
        logger.info(f"[TRADIER] Account: {account_id}")
        
        # Fetch real balance from Tradier API (not the default initial_balance)
        try:
            real_balance = self.get_account_balance()
            logger.info(f"[TRADIER] Balance: ${real_balance:,.2f} (from Tradier API)")
        except Exception as e:
            logger.warning(f"[TRADIER] Could not fetch balance on init: {e}")
            logger.info(f"[TRADIER] Balance: ${initial_balance:,.2f} (default)")
        
        logger.info(f"[TRADIER] Max positions: {self.max_positions}")
    
    def _make_request(self, method: str, endpoint: str, params: Dict = None, data: Dict = None) -> Optional[Dict]:
        """Make API request to Tradier"""
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=self.headers, params=params)
            elif method == 'POST':
                response = requests.post(url, headers=self.headers, data=data)
            elif method == 'PUT':
                response = requests.put(url, headers=self.headers, data=data)
            elif method == 'DELETE':
                response = requests.delete(url, headers=self.headers)
            else:
                logger.error(f"Unsupported HTTP method: {method}")
                return None
            
            if response.status_code == 200 or response.status_code == 201:
                return response.json()
            else:
                logger.error(f"[TRADIER] API error: {response.status_code}")
                logger.error(f"[TRADIER] Response: {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"[TRADIER] Request failed: {e}")
            return None
    
    def get_account_balance(self) -> float:
        """Get current account balance from Tradier"""
        response = self._make_request('GET', f"/accounts/{self.account_id}/balances")
        
        if response and 'balances' in response:
            balances = response['balances']
            account_type = balances.get('account_type', 'unknown')
            
            # Get top-level fields
            total_equity = float(balances.get('total_equity') or 0)
            total_cash = float(balances.get('total_cash') or 0)
            
            # For CASH accounts, buying power is nested in 'cash' object
            # For MARGIN accounts, it's in 'margin' object or top-level
            cash_available = 0.0
            option_buying_power = 0.0
            stock_buying_power = 0.0
            
            if 'cash' in balances:
                # Cash account - funds are in nested 'cash' object
                cash_obj = balances['cash']
                cash_available = float(cash_obj.get('cash_available') or 0)
                logger.info(f"[TRADIER] Cash account detected")
            
            if 'margin' in balances:
                # Margin account - buying power in nested 'margin' object
                margin_obj = balances['margin']
                option_buying_power = float(margin_obj.get('option_buying_power') or 0)
                stock_buying_power = float(margin_obj.get('stock_buying_power') or 0)
                logger.info(f"[TRADIER] Margin account detected")
            
            # Also check top-level (some account types have it here)
            if cash_available == 0:
                cash_available = float(balances.get('cash_available') or 0)
            if option_buying_power == 0:
                option_buying_power = float(balances.get('option_buying_power') or 0)
            if stock_buying_power == 0:
                stock_buying_power = float(balances.get('stock_buying_power') or 0)
            
            # Determine buying power: cash_available > option_bp > stock_bp > total_cash
            buying_power = cash_available if cash_available > 0 else option_buying_power
            if buying_power == 0:
                buying_power = stock_buying_power
            if buying_power == 0:
                buying_power = total_cash  # Last resort for cash accounts
            
            logger.info(f"[TRADIER] Account Type: {account_type}")
            logger.info(f"[TRADIER] Total Equity: ${total_equity:,.2f}")
            logger.info(f"[TRADIER] Cash Available: ${cash_available:,.2f}")
            logger.info(f"[TRADIER] Buying Power: ${buying_power:,.2f}")
            
            self.current_balance = buying_power  # Use buying power for trading decisions
            return total_equity
        
        logger.warning("[TRADIER] Could not fetch balance, using cached value")
        return self.current_balance
    
    def get_options_expirations(self, symbol: str) -> List[str]:
        """Get available options expiration dates"""
        response = self._make_request('GET', "/markets/options/expirations",
                                     params={'symbol': symbol})
        
        if response and 'expirations' in response:
            expirations = response['expirations'].get('date', [])
            logger.info(f"[TRADIER] Found {len(expirations)} expiration dates for {symbol}")
            return expirations
        
        return []
    
    def get_options_chain(self, symbol: str, expiration: str) -> List[Dict]:
        """Get options chain for symbol and expiration"""
        response = self._make_request('GET', "/markets/options/chains",
                                     params={
                                         'symbol': symbol,
                                         'expiration': expiration
                                     })
        
        if response and 'options' in response and response['options'] is not None and 'option' in response['options']:
            options = response['options']['option']
            logger.info(f"[TRADIER] Found {len(options)} options for {symbol} exp {expiration}")
            return options
        
        logger.warning(f"[TRADIER] No options found for {symbol} exp {expiration}")
        return []
    
    def get_option_quote(self, option_symbol: str) -> Optional[Dict]:
        """
        Get real-time quote for a specific option symbol
        
        Args:
            option_symbol: The OCC option symbol (e.g., 'SPY240126C00500000')
        
        Returns:
            Quote dict with bid, ask, last, volume, etc. or None if not found
        """
        response = self._make_request('GET', "/markets/quotes",
                                     params={'symbols': option_symbol})
        
        if response and 'quotes' in response:
            quotes = response['quotes']
            if quotes and 'quote' in quotes:
                quote = quotes['quote']
                # Handle single quote (dict) or list of quotes
                if isinstance(quote, list):
                    quote = quote[0] if quote else None
                if quote:
                    logger.debug(f"[TRADIER] Quote for {option_symbol}: bid=${quote.get('bid', 0)}, ask=${quote.get('ask', 0)}")
                    return quote
        
        logger.warning(f"[TRADIER] No quote found for {option_symbol}")
        return None
    
    def select_strike_and_expiration(self, symbol: str, option_type: str, current_price: float, target_strike: Optional[float] = None) -> Tuple[Optional[str], Optional[float], Optional[str]]:
        """
        Select optimal strike and expiration
        
        Args:
            target_strike: If provided, find the closest available strike to this value
        
        Returns:
            (option_symbol, strike, expiration) or (None, None, None)
        """
        # Get expirations (aim for ~45 days out)
        expirations = self.get_options_expirations(symbol)
        if not expirations:
            logger.error(f"[TRADIER] No expirations available for {symbol}")
            return None, None, None
        
        # Find expiration ~30-60 days out
        target_date = datetime.now() + timedelta(days=45)
        best_exp = min(expirations, 
                      key=lambda x: abs((datetime.strptime(x, '%Y-%m-%d') - target_date).days))
        
        logger.info(f"[TRADIER] Selected expiration: {best_exp}")
        
        # Get chain
        chain = self.get_options_chain(symbol, best_exp)
        if not chain:
            logger.error(f"[TRADIER] No options chain for {symbol} {best_exp}")
            return None, None, None
        
        # Filter by option type
        filtered = [opt for opt in chain if opt.get('option_type') == option_type.lower()]
        
        if not filtered:
            logger.error(f"[TRADIER] No {option_type} options found")
            return None, None, None
        
        # CRITICAL: Filter strikes to only reasonable range (within 20% of current price)
        # This prevents selecting deep ITM/OTM strikes with no liquidity
        min_strike = current_price * 0.85  # 15% below current
        max_strike = current_price * 1.15  # 15% above current
        
        filtered = [opt for opt in filtered 
                   if min_strike <= float(opt.get('strike') or 0) <= max_strike]
        
        if not filtered:
            logger.error(f"[TRADIER] No {option_type} options in reasonable strike range ${min_strike:.2f}-${max_strike:.2f}")
            return None, None, None
        
        logger.info(f"[TRADIER] Filtered to {len(filtered)} options in range ${min_strike:.2f}-${max_strike:.2f}")
        
        # CRITICAL: Pre-filter for liquidity BEFORE selecting strike
        # This prevents wasting time on illiquid options
        if self.liquidity_validator:
            logger.info(f"[TRADIER] Pre-filtering for liquid strikes...")
            liquid_options = []
            for opt in filtered:
                is_valid, reason = self._validate_option_liquidity(opt, current_price)
                if is_valid:
                    liquid_options.append(opt)
            
            if not liquid_options:
                logger.error(f"[TRADIER] No liquid {option_type} options found in range")
                logger.error(f"[TRADIER] Checked {len(filtered)} strikes, none passed liquidity requirements")
                return None, None, None
            
            logger.info(f"[TRADIER] Found {len(liquid_options)}/{len(filtered)} liquid strikes")
            filtered = liquid_options  # Only consider liquid options
        
        # Now select best strike from LIQUID options only
        if target_strike:
            logger.info(f"[TRADIER] Looking for closest liquid strike to target: ${target_strike:.2f}")
            best_option = min(filtered, 
                             key=lambda x: abs(float(x.get('strike') or 0) - target_strike))
            
            actual_strike = float(best_option.get('strike') or 0)
            strike_difference_pct = abs(actual_strike - target_strike) / target_strike * 100
            
            logger.info(f"[TRADIER] Matched target strike ${target_strike:.2f} → actual strike ${actual_strike:.2f} ({strike_difference_pct:.1f}% difference)")
            
            # ✅ SAFETY CHECK: Reject if strike is too far from target
            MAX_STRIKE_DIFFERENCE_PCT = 5.0
            
            if strike_difference_pct > MAX_STRIKE_DIFFERENCE_PCT:
                logger.error(f"[TRADIER] ❌ STRIKE MISMATCH: Available strike ${actual_strike:.2f} is {strike_difference_pct:.1f}% away from target ${target_strike:.2f}")
                logger.error(f"[TRADIER] ❌ Maximum allowed difference: {MAX_STRIKE_DIFFERENCE_PCT}% - REJECTING TRADE FOR SAFETY")
                return None, None, None
            
            logger.info(f"[TRADIER] ✅ Strike within acceptable range ({strike_difference_pct:.1f}% < {MAX_STRIKE_DIFFERENCE_PCT}%)")
        else:
            # SMALL ACCOUNT FIX: Adjust target delta based on buying power
            # Small accounts need cheaper OTM options, large accounts can afford ATM
            max_affordable_premium = self.current_balance * 0.80  # Use 80% of balance max
            
            # FIXED: Higher deltas = closer to ATM = less theta damage!
            # OLD deltas (0.15-0.30) = too far OTM → theta ate all profits
            # NEW deltas (0.35-0.50) = near ATM → profits when direction is right
            if self.current_balance < 1000:
                # Very small account: slightly OTM but not too far
                target_delta = 0.35 if option_type.lower() == 'call' else -0.35
                logger.info(f"[TRADIER] Small account (${self.current_balance:.2f}) - targeting near-ATM options (delta ~0.35)")
            elif self.current_balance < 5000:
                # Small-medium account: near ATM for better Greeks
                target_delta = 0.40 if option_type.lower() == 'call' else -0.40
                logger.info(f"[TRADIER] Medium account (${self.current_balance:.2f}) - targeting near-ATM options (delta ~0.40)")
            else:
                # Larger account: ATM or slightly ITM for best delta capture
                target_delta = 0.50 if option_type.lower() == 'call' else -0.50
                logger.info(f"[TRADIER] Standard account (${self.current_balance:.2f}) - targeting ATM options (delta ~0.50)")
            
            # Check if Greeks are available
            has_greeks = any(opt.get('greeks') for opt in filtered)
            
            if has_greeks:
                # Find closest delta from liquid options
                best_option = min(filtered, 
                                 key=lambda x: abs(float(x.get('greeks', {}).get('delta') or 0) - target_delta)
                                 if x.get('greeks') else float('inf'))
            else:
                # Fallback: select by strike proximity to current price
                # FIXED: Use tighter OTM (closer to ATM) to reduce theta damage
                logger.warning(f"[TRADIER] No Greeks available, using strike proximity fallback")
                if self.current_balance < 1000:
                    otm_pct = 0.015  # 1.5% OTM for small accounts (near ATM)
                elif self.current_balance < 5000:
                    otm_pct = 0.01  # 1% OTM for medium accounts
                else:
                    otm_pct = 0.005  # 0.5% OTM for standard accounts (essentially ATM)
                
                if option_type.lower() == 'call':
                    fallback_strike = current_price * (1 + otm_pct)
                else:
                    fallback_strike = current_price * (1 - otm_pct)
                
                logger.info(f"[TRADIER] Current price: ${current_price:.2f}, Target strike: ${fallback_strike:.2f}")
                best_option = min(filtered,
                                 key=lambda x: abs(float(x.get('strike') or 0) - fallback_strike))
        
        option_symbol = best_option.get('symbol')
        strike = float(best_option.get('strike') or 0)
        
        logger.info(f"[TRADIER] Selected liquid strike: {option_symbol} @ ${strike:.2f}")
        
        return option_symbol, strike, best_exp
    
    def _validate_option_liquidity(self, option_data: Dict, current_price: float) -> Tuple[bool, str]:
        """
        Validate option has sufficient liquidity
        
        Args:
            option_data: Option from chain with bid, ask, volume, open_interest, strike
            current_price: Current underlying price
        
        Returns:
            (is_valid, reason)
        """
        if not self.liquidity_validator:
            return True, "Validation disabled"
        
        # Extract data from Tradier format
        # Use 'or 0' to handle None values from API
        validation_data = {
            'bid': float(option_data.get('bid') or 0),
            'ask': float(option_data.get('ask') or 0),
            'volume': int(option_data.get('volume') or 0),
            'open_interest': int(option_data.get('open_interest') or 0),
            'strike': float(option_data.get('strike') or 0)
        }
        
        # Log quote data
        logger.info(f"[LIQUIDITY] Quote: Bid=${validation_data['bid']:.2f}, Ask=${validation_data['ask']:.2f}")
        logger.info(f"[LIQUIDITY] Volume: {validation_data['volume']}, OI: {validation_data['open_interest']}")
        
        # Validate (check market hours unless in sandbox)
        check_hours = not self.sandbox  # Skip hours check in sandbox for testing
        is_valid, reason = self.liquidity_validator.validate_option(
            validation_data,
            current_price,
            check_market_hours=check_hours
        )
        
        # Calculate quality score
        quality_score = self.liquidity_validator.get_execution_quality_score(validation_data)
        logger.info(f"[LIQUIDITY] Execution quality score: {quality_score}/100")
        
        # Save liquidity snapshot for historical training data
        self._save_liquidity_snapshot(option_data, current_price, quality_score, is_valid, reason)
        
        return is_valid, f"{reason} | Quality: {quality_score}/100"
    
    def _save_liquidity_snapshot(self, option_data: Dict, current_price: float, 
                                  quality_score: float, passed: bool, reason: str):
        """Save liquidity snapshot to database for model retraining"""
        try:
            from src.data.persistence import get_persistence
            persistence = get_persistence()
            
            bid = float(option_data.get('bid') or 0)
            ask = float(option_data.get('ask') or 0)
            mid_price = (bid + ask) / 2 if bid > 0 and ask > 0 else 0
            spread_pct = ((ask - bid) / mid_price * 100) if mid_price > 0 else 100
            
            snapshot = {
                'timestamp': datetime.now().isoformat(),
                'symbol': option_data.get('underlying', option_data.get('root_symbol', 'SPY')),
                'option_symbol': option_data.get('symbol'),
                'underlying_price': current_price,
                'strike_price': float(option_data.get('strike') or 0),
                'option_type': option_data.get('option_type', '').upper(),
                'expiration_date': option_data.get('expiration_date'),
                'bid': bid,
                'ask': ask,
                'spread_pct': spread_pct,
                'mid_price': mid_price,
                'volume': int(option_data.get('volume') or 0),
                'open_interest': int(option_data.get('open_interest') or 0),
                'quality_score': quality_score,
                'implied_volatility': float(option_data.get('implied_volatility') or 0),
                'delta': float(option_data.get('delta') or 0),
                'gamma': float(option_data.get('gamma') or 0),
                'theta': float(option_data.get('theta') or 0),
                'vega': float(option_data.get('vega') or 0),
                'signal_action': None,  # Will be updated by caller if available
                'signal_confidence': None,
                'trade_executed': 1 if passed else 0,
                'trade_blocked_reason': reason if not passed else None,
                'hmm_regime': None,
                'vix_value': None
            }
            
            persistence.save_liquidity_snapshot(snapshot)
            logger.debug(f"[LIQUIDITY] Saved snapshot: {option_data.get('symbol')} quality={quality_score}")
        except Exception as e:
            logger.debug(f"[LIQUIDITY] Could not save snapshot: {e}")
    
    def calculate_fees(self, quantity: int, premium: float) -> Dict[str, float]:
        """Calculate all fees for an options trade"""
        contract_fees = quantity * self.fees.option_contract_fee
        regulatory_fees = quantity * self.fees.options_regulatory_fee
        trading_fees = quantity * self.fees.finra_trading_activity_fee
        exchange_fees = quantity * self.fees.exchange_fee
        
        # SEC fee (based on premium amount)
        notional_value = quantity * premium * 100  # 100 shares per contract
        sec_fees = notional_value * self.fees.sec_fee_rate
        
        total = contract_fees + regulatory_fees + trading_fees + exchange_fees + sec_fees
        
        return {
            'contract_fees': contract_fees,
            'regulatory_fees': regulatory_fees,
            'trading_fees': trading_fees,
            'exchange_fees': exchange_fees,
            'sec_fees': sec_fees,
            'total_fees': total
        }
    
    def place_order(self, 
                   symbol: str,
                   option_symbol: str,
                   side: OrderType,
                   quantity: int = 1,
                   order_type: str = "market",
                   limit_price: float = None) -> Optional[Dict]:
        """
        Place an options order via Tradier API
        
        Args:
            symbol: Underlying symbol (e.g., 'BITX')
            option_symbol: OSI format option symbol
            side: OrderType (BUY_TO_OPEN, SELL_TO_OPEN, etc.)
            quantity: Number of contracts
            order_type: 'market' or 'limit'
            limit_price: Price for limit orders
        
        Returns:
            Order response dict or None
        """
        logger.info(f"[TRADIER] Placing order: {side.value} {quantity} {option_symbol}")
        
        order_data = {
            'class': 'option',
            'symbol': symbol,
            'option_symbol': option_symbol,
            'side': side.value,
            'quantity': str(quantity),
            'type': order_type,
            'duration': 'day'
        }
        
        if order_type == 'limit' and limit_price:
            order_data['price'] = f"{limit_price:.2f}"
        
        response = self._make_request('POST', f"/accounts/{self.account_id}/orders", data=order_data)
        
        if response and 'order' in response:
            order = response['order']
            order_id = order.get('id')
            status = order.get('status')
            
            # Check if order was rejected
            if status and status.lower() == 'rejected':
                reason = order.get('reason_description') or order.get('reject_reason') or 'Unknown reason'
                logger.error(f"[TRADIER] Order REJECTED: ID={order_id}")
                logger.error(f"[TRADIER] Rejection reason: {reason}")
                logger.error(f"[TRADIER] Order details: {side.value} {quantity} {option_symbol}")
                return None
            
            # Valid statuses: 'ok', 'open', 'filled', 'partially_filled', 'pending'
            # Note: 'ok' means order was accepted and queued for processing
            # It may still be rejected later (e.g., insufficient buying power)
            if status and status.lower() in ['ok', 'open', 'filled', 'partially_filled', 'pending']:
                logger.info(f"[TRADIER] Order accepted: ID={order_id}, Status={status}")
                
                # Note: Status='ok' means queued, not necessarily filled
                # The order may still be rejected during processing
                if status.lower() == 'ok':
                    logger.info(f"[TRADIER] Order queued for processing (may be rejected later)")
                
                return {
                    'order_id': order_id,
                    'status': status,
                    'symbol': symbol,
                    'option_symbol': option_symbol,
                    'side': side.value,
                    'quantity': quantity
                }
            else:
                # Unknown or unexpected status - log full response for debugging
                logger.warning(f"[TRADIER] Order unexpected status: ID={order_id}, Status={status}")
                logger.warning(f"[TRADIER] Full response: {order}")
                return None
        
        logger.error(f"[TRADIER] Order failed - no response from API")
        return None
    
    def is_market_open(self) -> bool:
        """Check if US options market is currently open"""
        et = pytz.timezone('US/Eastern')
        now = datetime.now(et)
        
        # Check if weekend
        if now.weekday() >= 5:  # Saturday=5, Sunday=6
            logger.warning(f"[TRADIER] Market closed: Weekend ({now.strftime('%A')})")
            return False
        
        # Check if within market hours (9:30 AM - 4:00 PM ET)
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        
        if now < market_open:
            logger.warning(f"[TRADIER] Market closed: Before open (opens at 9:30 AM ET)")
            return False
        
        if now > market_close:
            logger.warning(f"[TRADIER] Market closed: After close (closed at 4:00 PM ET)")
            return False
        
        return True
    
    def get_buying_power(self) -> float:
        """Get current buying power for options trading"""
        response = self._make_request('GET', f"/accounts/{self.account_id}/balances")
        
        if response and 'balances' in response:
            balances = response['balances']
            total_cash = float(balances.get('total_cash') or 0)
            
            # For CASH accounts, buying power is nested in 'cash' object
            # For MARGIN accounts, it's in 'margin' object
            cash_available = 0.0
            option_buying_power = 0.0
            stock_buying_power = 0.0
            
            if 'cash' in balances:
                cash_obj = balances['cash']
                cash_available = float(cash_obj.get('cash_available') or 0)
            
            if 'margin' in balances:
                margin_obj = balances['margin']
                option_buying_power = float(margin_obj.get('option_buying_power') or 0)
                stock_buying_power = float(margin_obj.get('stock_buying_power') or 0)
            
            # Also check top-level
            if cash_available == 0:
                cash_available = float(balances.get('cash_available') or 0)
            if option_buying_power == 0:
                option_buying_power = float(balances.get('option_buying_power') or 0)
            if stock_buying_power == 0:
                stock_buying_power = float(balances.get('stock_buying_power') or 0)
            
            # Priority: cash_available > option_buying_power > stock_buying_power > total_cash
            buying_power = cash_available if cash_available > 0 else option_buying_power
            if buying_power == 0:
                buying_power = stock_buying_power
            if buying_power == 0:
                buying_power = total_cash
            
            logger.info(f"[TRADIER] Current buying power: ${buying_power:,.2f}")
            return buying_power
        return 0.0
    
    def estimate_trade_cost(self, symbol: str, action: str, quantity: int, current_price: float) -> Dict:
        """
        Estimate the total cost of a trade before execution.
        Returns dict with estimated costs and whether we have enough buying power.
        """
        buying_power = self.get_buying_power()
        
        result = {
            'buying_power': buying_power,
            'estimated_cost': 0.0,
            'has_enough': False,
            'reason': None,
            'legs': []
        }
        
        # For straddles, we need to check both legs
        if action == 'BUY_STRADDLE':
            # Get call option price
            call_option, call_strike, exp = self.select_strike_and_expiration(symbol, 'call', current_price)
            call_cost = 0.0
            if call_option:
                quote = self.get_option_quote(call_option)
                if quote:
                    call_cost = float(quote.get('ask', 0)) * 100 * quantity
                    result['legs'].append({'type': 'CALL', 'symbol': call_option, 'cost': call_cost})
            
            # Get put option price  
            put_option, put_strike, _ = self.select_strike_and_expiration(symbol, 'put', current_price)
            put_cost = 0.0
            if put_option:
                quote = self.get_option_quote(put_option)
                if quote:
                    put_cost = float(quote.get('ask', 0)) * 100 * quantity
                    result['legs'].append({'type': 'PUT', 'symbol': put_option, 'cost': put_cost})
            
            result['estimated_cost'] = call_cost + put_cost
            
            if not call_option or not put_option:
                result['reason'] = 'Could not find options for straddle'
            elif result['estimated_cost'] > buying_power:
                result['reason'] = f'Insufficient funds: need ${result["estimated_cost"]:.2f}, have ${buying_power:.2f}'
            else:
                result['has_enough'] = True
                
        elif action in ['BUY_CALLS', 'BUY_CALL']:
            option, strike, exp = self.select_strike_and_expiration(symbol, 'call', current_price)
            if option:
                quote = self.get_option_quote(option)
                if quote:
                    result['estimated_cost'] = float(quote.get('ask', 0)) * 100 * quantity
                    result['legs'].append({'type': 'CALL', 'symbol': option, 'cost': result['estimated_cost']})
                    
            if result['estimated_cost'] > buying_power:
                result['reason'] = f'Insufficient funds: need ${result["estimated_cost"]:.2f}, have ${buying_power:.2f}'
            elif result['estimated_cost'] == 0:
                result['reason'] = 'Could not get option quote'
            else:
                result['has_enough'] = True
                
        elif action in ['BUY_PUTS', 'BUY_PUT']:
            option, strike, exp = self.select_strike_and_expiration(symbol, 'put', current_price)
            if option:
                quote = self.get_option_quote(option)
                if quote:
                    result['estimated_cost'] = float(quote.get('ask', 0)) * 100 * quantity
                    result['legs'].append({'type': 'PUT', 'symbol': option, 'cost': result['estimated_cost']})
                    
            if result['estimated_cost'] > buying_power:
                result['reason'] = f'Insufficient funds: need ${result["estimated_cost"]:.2f}, have ${buying_power:.2f}'
            elif result['estimated_cost'] == 0:
                result['reason'] = 'Could not get option quote'
            else:
                result['has_enough'] = True
        
        return result
    
    def place_trade(self, symbol: str, prediction_data: Dict, current_price: float = None) -> Optional[Dict]:
        """
        Main entry point: Place trade based on bot's signal
        
        Maps bot signals to Tradier orders:
        - BUY_CALLS → buy_to_open call
        - BUY_PUTS → buy_to_open put
        - BUY_STRADDLE → buy_to_open call + put
        - SELL_PREMIUM → sell_to_open put
        
        Returns dict with order info, or None if trade was blocked.
        If blocked, returns {'blocked': True, 'reason': '...'} 
        """
        action = prediction_data.get('action', 'HOLD')
        quantity = prediction_data.get('quantity', 1)
        
        if action == 'HOLD':
            return None
        
        # Check market hours (skip in sandbox for testing)
        if not self.sandbox and not self.is_market_open():
            logger.warning(f"[TRADIER] Market closed - order may not fill or be rejected")
            logger.warning(f"[TRADIER] Skipping trade: {action}")
            return {'blocked': True, 'reason': 'Market closed'}
        
        # Check buying power before attempting trade
        current_balance = self.get_account_balance()
        logger.info(f"[TRADIER] Current buying power: ${current_balance:,.2f}")
        
        # Minimum balance check (need at least $100 to trade options)
        if current_balance < 100:
            logger.error(f"[TRADIER] Insufficient buying power: ${current_balance:.2f} < $100")
            logger.error(f"[TRADIER] Skipping trade: {action}")
            return {'blocked': True, 'reason': f'Insufficient buying power: ${current_balance:.2f}'}
        
        # Pre-check estimated cost for ALL legs before placing any orders
        # This is critical for straddles - don't place half a straddle!
        cost_check = self.estimate_trade_cost(symbol, action, quantity, current_price)
        logger.info(f"[TRADIER] Estimated cost: ${cost_check['estimated_cost']:.2f}")
        logger.info(f"[TRADIER] Available: ${cost_check['buying_power']:.2f}")
        
        if not cost_check['has_enough']:
            logger.error(f"[TRADIER] {cost_check['reason']}")
            if action == 'BUY_STRADDLE':
                logger.error(f"[TRADIER] Cannot execute partial straddle - blocking entire trade")
            return {'blocked': True, 'reason': cost_check['reason']}
        
        logger.info(f"[TRADIER] ✓ Buying power check passed")
        logger.info(f"[TRADIER] Processing signal: {action}")
        
        # Map signal to option type
        if action in ['BUY_CALLS', 'BUY_CALL']:
            return self._execute_buy_calls(symbol, prediction_data, current_price)
        
        elif action in ['BUY_PUTS', 'BUY_PUT']:
            return self._execute_buy_puts(symbol, prediction_data, current_price)
        
        elif action == 'BUY_STRADDLE':
            return self._execute_straddle(symbol, prediction_data, current_price)
        
        elif action == 'SELL_PREMIUM':
            return self._execute_sell_premium(symbol, prediction_data, current_price)
        
        else:
            logger.warning(f"[TRADIER] Unknown action: {action}")
            return None
    
    def _execute_buy_calls(self, symbol: str, prediction_data: Dict, current_price: float) -> Optional[Dict]:
        """Execute BUY_CALLS strategy"""
        logger.info(f"[TRADIER] Executing BUY_CALLS for {symbol}")
        
        # Get quantity from prediction_data if available (from paper trading system)
        quantity = prediction_data.get('quantity', 1)
        logger.info(f"[TRADIER] Using quantity: {quantity} contracts")
        
        # Check if paper trading provided a specific strike to use
        preferred_strike = prediction_data.get('strike_price')
        
        if preferred_strike:
            logger.info(f"[TRADIER] Using paper trade's strike: ${preferred_strike:.2f}")
            # Select option matching the paper trade's strike
            option_symbol, strike, expiration = self.select_strike_and_expiration(
                symbol, 'call', current_price, target_strike=preferred_strike
            )
        else:
            # Select strike and expiration normally
            option_symbol, strike, expiration = self.select_strike_and_expiration(
                symbol, 'call', current_price
            )
        
        if not option_symbol:
            logger.error(f"[TRADIER] Could not find suitable call option")
            return None
        
        # Get option quote to estimate cost
        chain = self.get_options_chain(symbol, expiration)
        if chain:
            option_data = next((opt for opt in chain if opt.get('symbol') == option_symbol), None)
            if option_data:
                ask_price = float(option_data.get('ask') or 0)
                estimated_cost = (ask_price * 100 * quantity) + (quantity * 0.65)  # Premium + fees
                
                logger.info(f"[TRADIER] Estimated cost: ${estimated_cost:.2f} (${ask_price:.2f}/contract × {quantity})")
                
                # Check if we have enough buying power
                if estimated_cost > self.current_balance:
                    logger.error(f"[TRADIER] Insufficient buying power for trade")
                    logger.error(f"[TRADIER] Need: ${estimated_cost:.2f}, Have: ${self.current_balance:.2f}")
                    return None
                
                logger.info(f"[TRADIER] Buying power OK: ${self.current_balance:.2f} >= ${estimated_cost:.2f}")
        
        # Place order
        order = self.place_order(
            symbol=symbol,
            option_symbol=option_symbol,
            side=OrderType.BUY_TO_OPEN,
            quantity=quantity,
            order_type='market'
        )
        
        return order
    
    def _execute_buy_puts(self, symbol: str, prediction_data: Dict, current_price: float) -> Optional[Dict]:
        """Execute BUY_PUTS strategy"""
        logger.info(f"[TRADIER] Executing BUY_PUTS for {symbol}")
        
        # Get quantity from prediction_data if available (from paper trading system)
        quantity = prediction_data.get('quantity', 1)
        logger.info(f"[TRADIER] Using quantity: {quantity} contracts")
        
        # Check if paper trading provided a specific strike to use
        preferred_strike = prediction_data.get('strike_price')
        
        if preferred_strike:
            logger.info(f"[TRADIER] Using paper trade's strike: ${preferred_strike:.2f}")
            # Select option matching the paper trade's strike
            option_symbol, strike, expiration = self.select_strike_and_expiration(
                symbol, 'put', current_price, target_strike=preferred_strike
            )
        else:
            # Select strike and expiration normally
            option_symbol, strike, expiration = self.select_strike_and_expiration(
                symbol, 'put', current_price
            )
        
        if not option_symbol:
            logger.error(f"[TRADIER] Could not find suitable put option")
            return None
        
        # Get option quote to estimate cost
        chain = self.get_options_chain(symbol, expiration)
        if chain:
            option_data = next((opt for opt in chain if opt.get('symbol') == option_symbol), None)
            if option_data:
                ask_price = float(option_data.get('ask') or 0)
                estimated_cost = (ask_price * 100 * quantity) + (quantity * 0.65)  # Premium + fees
                
                logger.info(f"[TRADIER] Estimated cost: ${estimated_cost:.2f} (${ask_price:.2f}/contract × {quantity})")
                
                # Check if we have enough buying power
                if estimated_cost > self.current_balance:
                    logger.error(f"[TRADIER] Insufficient buying power for trade")
                    logger.error(f"[TRADIER] Need: ${estimated_cost:.2f}, Have: ${self.current_balance:.2f}")
                    return None
                
                logger.info(f"[TRADIER] Buying power OK: ${self.current_balance:.2f} >= ${estimated_cost:.2f}")
        
        # Place order
        order = self.place_order(
            symbol=symbol,
            option_symbol=option_symbol,
            side=OrderType.BUY_TO_OPEN,
            quantity=quantity,
            order_type='market'
        )
        
        return order
    
    def _execute_straddle(self, symbol: str, prediction_data: Dict, current_price: float) -> Optional[Dict]:
        """Execute BUY_STRADDLE strategy (buy call + put at same strike)"""
        logger.info(f"[TRADIER] Executing BUY_STRADDLE for {symbol}")
        
        # Get quantity from prediction_data if available (from paper trading system)
        quantity = prediction_data.get('quantity', 1)
        logger.info(f"[TRADIER] Using quantity: {quantity} contracts per leg")
        
        # For straddle, select ATM options
        # Get both call and put with same strike
        
        call_option, call_strike, expiration = self.select_strike_and_expiration(
            symbol, 'call', current_price
        )
        
        if not call_option:
            logger.error(f"[TRADIER] Could not find call for straddle")
            return None
        
        put_option, put_strike, _ = self.select_strike_and_expiration(
            symbol, 'put', current_price
        )
        
        if not put_option:
            logger.error(f"[TRADIER] Could not find put for straddle")
            return None
        
        # Place call order
        call_order = self.place_order(
            symbol=symbol,
            option_symbol=call_option,
            side=OrderType.BUY_TO_OPEN,
            quantity=quantity
        )
        
        # Place put order
        put_order = self.place_order(
            symbol=symbol,
            option_symbol=put_option,
            side=OrderType.BUY_TO_OPEN,
            quantity=quantity
        )
        
        return {
            'strategy': 'STRADDLE',
            'call_order': call_order,
            'put_order': put_order
        }
    
    def _execute_sell_premium(self, symbol: str, prediction_data: Dict, current_price: float) -> Optional[Dict]:
        """Execute SELL_PREMIUM strategy (sell puts for credit)"""
        logger.info(f"[TRADIER] Executing SELL_PREMIUM for {symbol}")
        
        # Get quantity from prediction_data if available (from paper trading system)
        quantity = prediction_data.get('quantity', 1)
        logger.info(f"[TRADIER] Using quantity: {quantity} contracts")
        
        # Select OTM put to sell
        option_symbol, strike, expiration = self.select_strike_and_expiration(
            symbol, 'put', current_price
        )
        
        if not option_symbol:
            logger.error(f"[TRADIER] Could not find put to sell")
            return None
        
        # Place sell-to-open order (collect premium)
        order = self.place_order(
            symbol=symbol,
            option_symbol=option_symbol,
            side=OrderType.SELL_TO_OPEN,  # Selling puts for credit
            quantity=quantity,
            order_type='market'
        )
        
        return order
    
    def close_position(self, option_symbol: str, quantity: int, symbol: str = 'BITX') -> Optional[Dict]:
        """
        Close an open option position by placing a SELL_TO_CLOSE or BUY_TO_CLOSE order
        
        Args:
            option_symbol: The OSI option symbol to close (e.g., 'BITX251226C00043000')
            quantity: Number of contracts to close (should be absolute value)
            symbol: Underlying symbol (default: 'BITX')
            
        Returns:
            Order response dict or None
        """
        logger.info(f"[TRADIER] Closing position: {quantity} contracts of {option_symbol}")
        
        # Get current positions to determine if it's LONG or SHORT
        response = self._make_request('GET', f"/accounts/{self.account_id}/positions")
        is_short = False
        
        if response and 'positions' in response:
            positions = response['positions'].get('position', [])
            if not isinstance(positions, list):
                positions = [positions]
            
            # Find this specific position
            for pos in positions:
                if pos.get('symbol') == option_symbol:
                    pos_qty = float(pos.get('quantity') or 0)
                    is_short = pos_qty < 0
                    logger.info(f"[TRADIER] Position {option_symbol}: qty={pos_qty}, is_short={is_short}")
                    break
        
        # Use correct order type based on position direction
        order_side = OrderType.BUY_TO_CLOSE if is_short else OrderType.SELL_TO_CLOSE
        logger.info(f"[TRADIER] Using {order_side} for {'SHORT' if is_short else 'LONG'} position")
        
        # Place close order
        result = self.place_order(
            symbol=symbol,
            option_symbol=option_symbol,
            side=order_side,
            quantity=abs(int(quantity)),  # Ensure positive quantity
            order_type='market'
        )
        
        if result:
            order_id = result.get('order_id')  # Fixed: was 'id', should be 'order_id'
            status = result.get('status')
            logger.info(f"[TRADIER] ✅ Position close order placed: Order ID {order_id}, Status: {status}")
            
            # Verify the close order was accepted
            if status and status.lower() == 'rejected':
                logger.error(f"[TRADIER] ❌ Close order was REJECTED for {option_symbol}")
                return None
                
            logger.info(f"[TRADIER] Close order accepted, waiting for fill...")
        else:
            logger.error(f"[TRADIER] ❌ Failed to close position {option_symbol} - no response from API")
            
        return result
    
    def update_positions(self, symbol: str, current_price: float = None):
        """
        Update positions from Tradier API
        Gets real position data from broker
        """
        response = self._make_request('GET', f"/accounts/{self.account_id}/positions")
        
        if response and 'positions' in response:
            positions = response['positions'].get('position', [])
            
            # Ensure it's a list
            if isinstance(positions, dict):
                positions = [positions]
            
            logger.info(f"[TRADIER] Active positions synced: {len(positions)} open trades")
            
            # Update local tracking
            self.active_trades = positions
            
            return positions
        
        return []
    
    def get_order_status(self, order_id: str) -> Optional[Dict]:
        """Get status of a specific order"""
        response = self._make_request('GET', f"/accounts/{self.account_id}/orders/{order_id}")
        
        if response and 'order' in response:
            return response['order']
        
        return None
    
    def get_order_history(self, limit: int = 20) -> List[Dict]:
        """Get recent order history"""
        response = self._make_request('GET', f"/accounts/{self.account_id}/orders")
        
        if response and 'orders' in response:
            orders = response['orders'].get('order', [])
            # Ensure it's a list (API returns single dict if only one order)
            if isinstance(orders, dict):
                orders = [orders]
            return orders[:limit]
        
        return []
    
    def verify_order_filled(self, order_id: str, max_wait_seconds: int = 5) -> Dict:
        """
        Verify if an order was actually filled or rejected after processing
        
        Args:
            order_id: Order ID to check
            max_wait_seconds: Maximum seconds to wait for processing
            
        Returns:
            Dict with 'success' (bool), 'status' (str), 'reason' (str if rejected)
        """
        import time
        
        waited = 0
        check_interval = 1  # Check every 1 second
        
        while waited < max_wait_seconds:
            order = self.get_order_status(order_id)
            
            if not order:
                return {
                    'success': False,
                    'status': 'unknown',
                    'reason': 'Could not fetch order status'
                }
            
            status = order.get('status', '').lower()
            
            # Order was rejected
            if status == 'rejected':
                reason = order.get('reason_description') or order.get('reject_reason') or 'Unknown'
                logger.warning(f"[TRADIER] Order {order_id} was rejected: {reason}")
                return {
                    'success': False,
                    'status': 'rejected',
                    'reason': reason
                }
            
            # Order was filled
            if status in ['filled', 'partially_filled']:
                logger.info(f"[TRADIER] Order {order_id} was filled")
                return {
                    'success': True,
                    'status': status,
                    'reason': None
                }
            
            # Order still processing
            if status in ['ok', 'open', 'pending']:
                time.sleep(check_interval)
                waited += check_interval
                continue
            
            # Unknown status
            logger.warning(f"[TRADIER] Order {order_id} has unexpected status: {status}")
            return {
                'success': False,
                'status': status,
                'reason': f'Unexpected status: {status}'
            }
        
        # Timeout - order still processing
        logger.warning(f"[TRADIER] Order {order_id} still processing after {max_wait_seconds}s")
        return {
            'success': None,  # None means "still processing"
            'status': 'pending',
            'reason': 'Order still processing'
        }
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order"""
        response = self._make_request('DELETE', f"/accounts/{self.account_id}/orders/{order_id}")
        
        if response:
            logger.info(f"[TRADIER] Order {order_id} cancelled")
            return True
        
        return False
    
    def get_account_summary(self) -> Dict:
        """Get comprehensive account summary"""
        return {
            'account_id': self.account_id,
            'current_balance': self.get_account_balance(),
            'active_positions': len(self.active_trades),
            'max_positions': self.max_positions,
            'sandbox_mode': self.sandbox
        }


def test_tradier_connection():
    """Test script to verify Tradier connection"""
    import os
    from tradier_credentials import TradierCredentials
    
    print("\n" + "="*70)
    print("TRADIER API CONNECTION TEST")
    print("="*70)
    
    # Get credentials
    creds = TradierCredentials()
    account_id, api_token = creds.get_sandbox_credentials()
    
    if not account_id or not api_token:
        print("❌ No sandbox credentials found")
        print("   Run: python setup_credentials.py")
        return
    
    # Initialize system
    system = TradierTradingSystem(
        account_id=account_id,
        api_token=api_token,
        sandbox=True
    )
    
    # Test 1: Get balance
    print("\n[TEST 1] Get Account Balance")
    balance = system.get_account_balance()
    print(f"✓ Balance: ${balance:,.2f}")
    
    # Test 2: Get expirations
    print("\n[TEST 2] Get Options Expirations")
    expirations = system.get_options_expirations('BITX')
    print(f"✓ Found {len(expirations)} expiration dates")
    if expirations:
        print(f"   Next 3: {expirations[:3]}")
    
    # Test 3: Get options chain
    if expirations:
        print("\n[TEST 3] Get Options Chain")
        chain = system.get_options_chain('BITX', expirations[0])
        print(f"✓ Found {len(chain)} options")
        if chain:
            sample = chain[0]
            print(f"   Sample: {sample.get('symbol')} ${sample.get('strike')}")
    
    # Test 4: Fee calculation
    print("\n[TEST 4] Calculate Fees")
    fees = system.calculate_fees(quantity=1, premium=0.50)
    print(f"✓ Total fees for 1 contract @ $0.50:")
    for key, value in fees.items():
        print(f"   {key}: ${value:.4f}")
    
    print("\n" + "="*70)
    print("✓ All tests passed!")
    print("="*70)


if __name__ == "__main__":
    test_tradier_connection()




