#!/usr/bin/env python3
"""
Option Pricing
==============

Option premium calculation for dashboard position tracking.
Uses relative pricing model to match paper_trading_system.py.
"""


class OptionPricing:
    """
    Calculates option premiums using relative pricing model.
    
    This MUST match paper_trading_system.py update_positions()
    for accurate P&L display on dashboard.
    """
    
    # Delta leverage for SPY ATM options
    DELTA_LEVERAGE = 20.0
    
    # Theta decay rates (per minute)
    THETA_0DTE = 0.0020      # 0.20% per minute (0 DTE)
    THETA_WEEKLY = 0.0003    # 0.03% per minute (< 7 days)
    THETA_MONTHLY = 0.0001   # 0.01% per minute (< 30 days)
    THETA_LONG = 0.00005     # 0.005% per minute (30+ days)
    
    @classmethod
    def calculate_premium(
        cls,
        option_type: str,
        entry_price: float,
        entry_premium: float,
        current_price: float,
        minutes_held: float = 0,
        days_to_expiry: float = 1.0
    ) -> float:
        """
        Calculate current option premium using RELATIVE PRICING MODEL.
        
        Args:
            option_type: 'CALL' or 'PUT'
            entry_price: Underlying price at entry
            entry_premium: Option premium paid at entry
            current_price: Current underlying price
            minutes_held: Minutes since trade entry
            days_to_expiry: Days until option expires
            
        Returns:
            Current option premium estimate
        """
        if entry_price <= 0 or entry_premium <= 0 or current_price <= 0:
            return entry_premium
        
        # 1. Calculate underlying price move
        price_move_pct = (current_price / entry_price - 1.0)
        
        # 2. Determine direction
        is_call = 'CALL' in option_type.upper()
        
        # 3. Apply delta leverage
        if is_call:
            # CALL: Up is good, Down is bad
            option_move_pct = price_move_pct * cls.DELTA_LEVERAGE
        else:
            # PUT: Down is good, Up is bad
            option_move_pct = -price_move_pct * cls.DELTA_LEVERAGE
        
        # 4. Apply delta impact to entry premium
        theoretical_premium = entry_premium * (1 + option_move_pct)
        
        # 5. Apply theta decay
        theta_rate = cls._get_theta_rate(days_to_expiry)
        theta_decay = entry_premium * theta_rate * minutes_held
        current_premium = max(0.01, theoretical_premium - theta_decay)
        
        # 6. Apply sanity checks
        current_premium = cls._apply_sanity_checks(
            current_premium,
            entry_premium,
            price_move_pct,
            is_call
        )
        
        return max(0.01, current_premium)
    
    @classmethod
    def _get_theta_rate(cls, days_to_expiry: float) -> float:
        """Get theta decay rate based on days to expiry."""
        if days_to_expiry < 1.0:
            return cls.THETA_0DTE
        elif days_to_expiry < 7:
            return cls.THETA_WEEKLY
        elif days_to_expiry < 30:
            return cls.THETA_MONTHLY
        else:
            return cls.THETA_LONG
    
    @classmethod
    def _apply_sanity_checks(
        cls,
        current_premium: float,
        entry_premium: float,
        price_move_pct: float,
        is_call: bool
    ) -> float:
        """Apply sanity checks to prevent unrealistic values."""
        price_move_pct_100 = price_move_pct * 100.0
        max_gain_pct = abs(price_move_pct_100) * cls.DELTA_LEVERAGE
        max_gain_pct = max(1.0, min(400.0, max_gain_pct))  # Floor 1%, cap 400%
        
        if not is_call and price_move_pct > 0:
            # Underlying UP → long PUT MUST lose value
            loss_factor = min(0.9, abs(price_move_pct) * cls.DELTA_LEVERAGE)
            max_dir_premium = entry_premium * (1 - loss_factor)
            current_premium = min(current_premium, max(0.01, max_dir_premium))
        
        elif not is_call and price_move_pct < 0:
            # Underlying DOWN → long PUT gains (cap gains)
            max_gain_premium = entry_premium * (1 + max_gain_pct / 100)
            current_premium = min(current_premium, max_gain_premium)
        
        elif is_call and price_move_pct < 0:
            # Underlying DOWN → long CALL MUST lose value
            loss_factor = min(0.9, abs(price_move_pct) * cls.DELTA_LEVERAGE)
            max_dir_premium = entry_premium * (1 - loss_factor)
            current_premium = min(current_premium, max(0.01, max_dir_premium))
        
        elif is_call and price_move_pct > 0:
            # Underlying UP → long CALL gains (cap gains)
            max_gain_premium = entry_premium * (1 + max_gain_pct / 100)
            current_premium = min(current_premium, max_gain_premium)
        
        return current_premium
    
    @classmethod
    def calculate_unrealized_pnl(
        cls,
        option_type: str,
        entry_price: float,
        entry_premium: float,
        current_price: float,
        quantity: int,
        minutes_held: float = 0,
        days_to_expiry: float = 1.0
    ) -> tuple:
        """
        Calculate unrealized P&L for a position.
        
        Args:
            option_type: 'CALL' or 'PUT'
            entry_price: Underlying price at entry
            entry_premium: Option premium paid at entry
            current_price: Current underlying price
            quantity: Number of contracts
            minutes_held: Minutes since entry
            days_to_expiry: Days until expiry
            
        Returns:
            Tuple of (current_value, unrealized_pnl, unrealized_pnl_pct)
        """
        current_premium = cls.calculate_premium(
            option_type,
            entry_price,
            entry_premium,
            current_price,
            minutes_held,
            days_to_expiry
        )
        
        invested = entry_premium * quantity * 100  # Options are 100 shares
        current_value = current_premium * quantity * 100
        unrealized_pnl = current_value - invested
        unrealized_pnl_pct = (unrealized_pnl / invested * 100) if invested > 0 else 0
        
        return current_value, unrealized_pnl, unrealized_pnl_pct





