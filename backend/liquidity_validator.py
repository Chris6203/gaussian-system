#!/usr/bin/env python3
"""
Liquidity Validation System
============================

Ensures trades will execute successfully in production by checking:
1. Bid-ask spread (tight spreads = liquid market)
2. Volume (daily trading activity)
3. Open interest (total outstanding contracts)
4. Market hours (only trade when market is open)
5. Strike price reasonableness
6. Quote freshness

Usage:
    validator = LiquidityValidator()
    is_valid, reason = validator.validate_option(option_data, current_price)
    if not is_valid:
        logger.warning(f"Trade rejected: {reason}")
"""

import logging
from typing import Dict, Tuple, Optional
from datetime import datetime, time
import pytz

logger = logging.getLogger(__name__)


class LiquidityValidator:
    """Validates options liquidity before placing trades"""
    
    def __init__(self,
                 max_bid_ask_spread_pct: float = 10.0,
                 min_volume: int = 10,
                 min_open_interest: int = 50,
                 max_strike_deviation_pct: float = 15.0):
        """
        Initialize liquidity validator
        
        Args:
            max_bid_ask_spread_pct: Maximum bid-ask spread as % of mid price (default 10%)
            min_volume: Minimum daily volume required (default 10 contracts)
            min_open_interest: Minimum open interest required (default 50 contracts)
            max_strike_deviation_pct: Maximum strike deviation from spot (default 15%)
        """
        self.max_bid_ask_spread_pct = max_bid_ask_spread_pct
        self.min_volume = min_volume
        self.min_open_interest = min_open_interest
        self.max_strike_deviation_pct = max_strike_deviation_pct
        
        # Market hours (US Eastern Time)
        self.market_open = time(9, 30)  # 9:30 AM ET
        self.market_close = time(16, 0)  # 4:00 PM ET
        self.eastern_tz = pytz.timezone('US/Eastern')
        
        logger.info(f"[LIQUIDITY] Validator initialized:")
        logger.info(f"  Max bid-ask spread: {max_bid_ask_spread_pct}%")
        logger.info(f"  Min volume: {min_volume}")
        logger.info(f"  Min open interest: {min_open_interest}")
        logger.info(f"  Max strike deviation: {max_strike_deviation_pct}%")
    
    def is_market_open(self) -> Tuple[bool, str]:
        """Check if market is currently open"""
        now_et = datetime.now(self.eastern_tz)
        current_time = now_et.time()
        
        # Check if weekend
        if now_et.weekday() >= 5:  # Saturday=5, Sunday=6
            return False, f"Market closed: Weekend ({now_et.strftime('%A')})"
        
        # Check if within market hours
        if current_time < self.market_open:
            return False, f"Market closed: Before open (opens at {self.market_open.strftime('%H:%M')} ET)"
        
        if current_time > self.market_close:
            return False, f"Market closed: After close (closed at {self.market_close.strftime('%H:%M')} ET)"
        
        return True, "Market open"
    
    def validate_bid_ask_spread(self, bid: float, ask: float) -> Tuple[bool, str]:
        """
        Validate bid-ask spread is reasonable
        
        Args:
            bid: Bid price
            ask: Ask price
        
        Returns:
            (is_valid, reason)
        """
        if bid <= 0 or ask <= 0:
            return False, f"Invalid quote: bid=${bid:.2f}, ask=${ask:.2f}"
        
        if ask < bid:
            return False, f"Invalid quote: ask (${ask:.2f}) < bid (${bid:.2f})"
        
        mid_price = (bid + ask) / 2
        spread = ask - bid
        spread_pct = (spread / mid_price) * 100
        
        if spread_pct > self.max_bid_ask_spread_pct:
            return False, f"Spread too wide: {spread_pct:.1f}% (max {self.max_bid_ask_spread_pct}%)"
        
        logger.info(f"[LIQUIDITY] Spread OK: {spread_pct:.1f}% (bid=${bid:.2f}, ask=${ask:.2f})")
        return True, f"Spread acceptable: {spread_pct:.1f}%"
    
    def infer_volume_from_spread(self, bid: float, ask: float) -> int:
        """
        Infer likely volume from bid-ask spread
        
        Empirical relationship:
        - Tight spread (< 5%) → High volume (100+)
        - Medium spread (5-10%) → Medium volume (20-100)
        - Wide spread (> 10%) → Low volume (< 20)
        
        Args:
            bid: Bid price
            ask: Ask price
        
        Returns:
            Estimated volume
        """
        if bid <= 0 or ask <= 0 or ask < bid:
            return 0
        
        mid_price = (bid + ask) / 2
        spread_pct = ((ask - bid) / mid_price) * 100
        
        # Exponential decay: tighter spread = higher estimated volume
        if spread_pct < 3.0:
            estimated_volume = 200  # Very liquid
        elif spread_pct < 5.0:
            estimated_volume = 100  # Liquid
        elif spread_pct < 8.0:
            estimated_volume = 50   # Moderate
        elif spread_pct < 12.0:
            estimated_volume = 20   # Low
        else:
            estimated_volume = 5    # Very low
        
        logger.info(f"[LIQUIDITY] Inferred volume from {spread_pct:.1f}% spread: ~{estimated_volume} contracts")
        return estimated_volume
    
    def validate_volume(self, volume: int, bid: float = None, ask: float = None) -> Tuple[bool, str]:
        """
        Validate daily trading volume is sufficient
        If volume is 0 or missing, infer from spread
        
        Args:
            volume: Daily trading volume
            bid: Bid price (for inference if volume missing)
            ask: Ask price (for inference if volume missing)
        
        Returns:
            (is_valid, reason)
        """
        # If volume data is missing or zero, try to infer from spread
        if volume == 0 and bid and ask:
            logger.info(f"[LIQUIDITY] Volume data missing, inferring from spread...")
            volume = self.infer_volume_from_spread(bid, ask)
        
        if volume < self.min_volume:
            return False, f"Insufficient volume: {volume} (min {self.min_volume})"
        
        logger.info(f"[LIQUIDITY] Volume OK: {volume} contracts")
        return True, f"Volume sufficient: {volume} contracts"
    
    def validate_open_interest(self, open_interest: int, bid: float = None, ask: float = None) -> Tuple[bool, str]:
        """
        Validate open interest is sufficient
        If OI is 0 but spread is tight, assume sufficient OI (common in early morning)
        
        Args:
            open_interest: Total outstanding contracts
            bid: Bid price (for inference if OI is 0)
            ask: Ask price (for inference if OI is 0)
        
        Returns:
            (is_valid, reason)
        """
        # If OI is 0 but we have a tight spread, infer OI is likely sufficient
        # This handles early morning when OI data isn't populated yet
        if open_interest == 0 and bid and ask and bid > 0 and ask > 0:
            mid_price = (bid + ask) / 2
            spread_pct = ((ask - bid) / mid_price) * 100
            
            # Tight spread indicates market maker presence = liquidity exists
            if spread_pct < 2.0:
                logger.info(f"[LIQUIDITY] OI=0 but tight spread ({spread_pct:.1f}%) - inferring sufficient OI")
                return True, f"OI inferred from tight spread: {spread_pct:.1f}%"
            elif spread_pct < 5.0:
                logger.info(f"[LIQUIDITY] OI=0, moderate spread ({spread_pct:.1f}%) - allowing with caution")
                return True, f"OI inferred from spread: {spread_pct:.1f}%"
        
        if open_interest < self.min_open_interest:
            return False, f"Low open interest: {open_interest} (min {self.min_open_interest})"
        
        logger.info(f"[LIQUIDITY] Open interest OK: {open_interest}")
        return True, f"Open interest sufficient: {open_interest}"
    
    def validate_strike_price(self, strike: float, current_price: float) -> Tuple[bool, str]:
        """
        Validate strike price is reasonable (not too far from spot)
        
        Args:
            strike: Option strike price
            current_price: Current underlying price
        
        Returns:
            (is_valid, reason)
        """
        deviation_pct = abs(strike - current_price) / current_price * 100
        
        if deviation_pct > self.max_strike_deviation_pct:
            return False, f"Strike too far from spot: ${strike:.2f} vs ${current_price:.2f} ({deviation_pct:.1f}%)"
        
        logger.info(f"[LIQUIDITY] Strike OK: ${strike:.2f} ({deviation_pct:.1f}% from spot)")
        return True, f"Strike reasonable: {deviation_pct:.1f}% from spot"
    
    def validate_quote_freshness(self, last_trade_time: Optional[datetime], 
                                  max_age_seconds: int = 300) -> Tuple[bool, str]:
        """
        Validate quote is recent
        
        Args:
            last_trade_time: Time of last trade
            max_age_seconds: Maximum age in seconds (default 5 minutes)
        
        Returns:
            (is_valid, reason)
        """
        if last_trade_time is None:
            return False, "No recent trades (stale quote)"
        
        age_seconds = (datetime.now(pytz.UTC) - last_trade_time).total_seconds()
        
        if age_seconds > max_age_seconds:
            return False, f"Quote too old: {age_seconds:.0f}s (max {max_age_seconds}s)"
        
        logger.info(f"[LIQUIDITY] Quote fresh: {age_seconds:.0f}s old")
        return True, f"Quote fresh: {age_seconds:.0f}s old"
    
    def validate_option(self, 
                        option_data: Dict, 
                        current_price: float,
                        check_market_hours: bool = True) -> Tuple[bool, str]:
        """
        Comprehensive validation of option liquidity
        
        Args:
            option_data: Option data from Tradier API with keys:
                - bid: Bid price
                - ask: Ask price
                - volume: Daily volume
                - open_interest: Open interest
                - strike: Strike price
                - last_trade_time: Last trade timestamp (optional)
            current_price: Current underlying price
            check_market_hours: Whether to check market hours (default True)
        
        Returns:
            (is_valid, reason)
        """
        validations = []
        
        # 1. Market hours check
        if check_market_hours:
            is_open, msg = self.is_market_open()
            if not is_open:
                return False, msg
            validations.append(msg)
        
        # 2. Bid-ask spread check
        bid = float(option_data.get('bid', 0))
        ask = float(option_data.get('ask', 0))
        is_valid, msg = self.validate_bid_ask_spread(bid, ask)
        if not is_valid:
            return False, msg
        validations.append(msg)
        
        # 3. Volume check (with spread-based inference if needed)
        volume = int(option_data.get('volume', 0))
        is_valid, msg = self.validate_volume(volume, bid=bid, ask=ask)
        if not is_valid:
            return False, msg
        validations.append(msg)
        
        # 4. Open interest check (pass bid/ask for inference when OI=0)
        open_interest = int(option_data.get('open_interest', 0))
        is_valid, msg = self.validate_open_interest(open_interest, bid=bid, ask=ask)
        if not is_valid:
            return False, msg
        validations.append(msg)
        
        # 5. Strike price check
        strike = float(option_data.get('strike', 0))
        is_valid, msg = self.validate_strike_price(strike, current_price)
        if not is_valid:
            return False, msg
        validations.append(msg)
        
        # 6. Quote freshness check (optional, if last_trade_time available)
        if 'last_trade_time' in option_data:
            last_trade_time = option_data['last_trade_time']
            is_valid, msg = self.validate_quote_freshness(last_trade_time)
            if not is_valid:
                logger.warning(f"[LIQUIDITY] {msg} (non-fatal)")
            validations.append(msg)
        
        summary = " | ".join(validations)
        logger.info(f"[LIQUIDITY] ✅ All checks passed: {summary}")
        return True, f"Liquidity OK: {summary}"
    
    def get_execution_quality_score(self, option_data: Dict) -> float:
        """
        Calculate execution quality score (0-100)
        Higher scores indicate better liquidity and execution probability
        
        Args:
            option_data: Option data with bid, ask, volume, open_interest
        
        Returns:
            Quality score (0-100)
        """
        score = 0.0
        
        # Bid-ask spread component (40 points)
        bid = float(option_data.get('bid', 0))
        ask = float(option_data.get('ask', 0))
        if bid > 0 and ask > bid:
            mid_price = (bid + ask) / 2
            spread_pct = ((ask - bid) / mid_price) * 100
            # Score: 40 points for 0% spread, 0 points for 10%+ spread
            spread_score = max(0, 40 * (1 - spread_pct / 10))
            score += spread_score
        
        # Volume component (30 points)
        volume = int(option_data.get('volume', 0))
        # Score: 30 points for 100+ volume, scaled down to min_volume
        volume_score = min(30, 30 * (volume / 100))
        score += volume_score
        
        # Open interest component (30 points)
        open_interest = int(option_data.get('open_interest', 0))
        # Score: 30 points for 500+ OI, scaled down to min_open_interest
        oi_score = min(30, 30 * (open_interest / 500))
        score += oi_score
        
        return round(score, 1)


def main():
    """Test liquidity validator"""
    validator = LiquidityValidator()
    
    # Test case 1: Good liquidity
    good_option = {
        'bid': 1.45,
        'ask': 1.55,
        'volume': 250,
        'open_interest': 1500,
        'strike': 58.0
    }
    
    current_price = 57.5
    
    is_valid, reason = validator.validate_option(good_option, current_price, check_market_hours=False)
    quality_score = validator.get_execution_quality_score(good_option)
    
    print(f"\nTest 1 - Good Liquidity:")
    print(f"  Valid: {is_valid}")
    print(f"  Reason: {reason}")
    print(f"  Quality Score: {quality_score}/100")
    
    # Test case 2: Wide spread
    wide_spread = {
        'bid': 1.00,
        'ask': 1.50,  # 50% spread!
        'volume': 250,
        'open_interest': 1500,
        'strike': 58.0
    }
    
    is_valid, reason = validator.validate_option(wide_spread, current_price, check_market_hours=False)
    quality_score = validator.get_execution_quality_score(wide_spread)
    
    print(f"\nTest 2 - Wide Spread:")
    print(f"  Valid: {is_valid}")
    print(f"  Reason: {reason}")
    print(f"  Quality Score: {quality_score}/100")
    
    # Test case 3: Low volume
    low_volume = {
        'bid': 1.45,
        'ask': 1.55,
        'volume': 5,  # Below minimum
        'open_interest': 1500,
        'strike': 58.0
    }
    
    is_valid, reason = validator.validate_option(low_volume, current_price, check_market_hours=False)
    quality_score = validator.get_execution_quality_score(low_volume)
    
    print(f"\nTest 3 - Low Volume:")
    print(f"  Valid: {is_valid}")
    print(f"  Reason: {reason}")
    print(f"  Quality Score: {quality_score}/100")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

