"""
Options Pricing Module
======================

Black-Scholes option pricing and Greeks calculations.

Provides:
- Black-Scholes pricing for calls and puts
- Greek calculations (delta, gamma, theta, vega)
- IV estimation from option prices
- Strike price utilities

Usage:
    from backend.options_pricing import (
        BlackScholesPricer,
        calculate_option_premium,
        estimate_iv_from_premium
    )
"""

import logging
import math
from dataclasses import dataclass
from typing import Tuple, Optional
from datetime import datetime, timedelta

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# Data Classes
# ------------------------------------------------------------------------------
@dataclass
class OptionGreeks:
    """Container for option Greeks."""
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            'delta': self.delta,
            'gamma': self.gamma,
            'theta': self.theta,
            'vega': self.vega,
            'rho': self.rho
        }


@dataclass
class OptionPrice:
    """Container for option pricing results."""
    premium: float
    intrinsic: float
    time_value: float
    greeks: OptionGreeks
    iv: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            'premium': self.premium,
            'intrinsic': self.intrinsic,
            'time_value': self.time_value,
            'iv': self.iv,
            **self.greeks.to_dict()
        }


# ------------------------------------------------------------------------------
# Strike Price Utilities
# ------------------------------------------------------------------------------
def round_to_realistic_strike(strike_price: float, spot_price: float) -> float:
    """
    Round calculated strike to realistic exchange intervals.
    
    Exchanges don't offer arbitrary strikes like $4.72.
    They offer strikes in standard intervals:
    - Under $5: $0.50 intervals ($4.00, $4.50, $5.00)
    - $5-$25: $0.50 or $1.00 intervals
    - $25-$200: $1.00 or $2.50 intervals
    - Over $200: $5.00 or $10.00 intervals
    
    Args:
        strike_price: Calculated optimal strike
        spot_price: Current underlying price
        
    Returns:
        Rounded strike matching exchange intervals
    """
    if spot_price < 5:
        interval = 0.50
    elif spot_price < 25:
        interval = 1.00
    elif spot_price < 50:
        interval = 1.00
    elif spot_price < 200:
        interval = 2.50
    else:
        interval = 5.00
    
    return round(strike_price / interval) * interval


def get_atm_strike(spot_price: float) -> float:
    """Get the at-the-money strike for a given spot price."""
    return round_to_realistic_strike(spot_price, spot_price)


def get_otm_strikes(spot_price: float, num_strikes: int = 5, is_call: bool = True) -> list:
    """
    Get out-of-the-money strikes.
    
    Args:
        spot_price: Current underlying price
        num_strikes: Number of strikes to return
        is_call: If True, return strikes above spot (OTM calls), else below (OTM puts)
        
    Returns:
        List of OTM strike prices
    """
    atm = get_atm_strike(spot_price)
    
    # Determine interval
    if spot_price < 5:
        interval = 0.50
    elif spot_price < 25:
        interval = 1.00
    elif spot_price < 200:
        interval = 2.50
    else:
        interval = 5.00
    
    strikes = []
    for i in range(1, num_strikes + 1):
        if is_call:
            strikes.append(atm + i * interval)
        else:
            strikes.append(atm - i * interval)
    
    return strikes


# ------------------------------------------------------------------------------
# Black-Scholes Pricing
# ------------------------------------------------------------------------------
class BlackScholesPricer:
    """
    Black-Scholes option pricing with Greeks.
    
    Supports both call and put options with:
    - Analytical pricing
    - Greek calculations
    - IV solving
    """
    
    def __init__(self, risk_free_rate: float = 0.05):
        """
        Initialize pricer.
        
        Args:
            risk_free_rate: Annual risk-free rate (default 5%)
        """
        self.risk_free_rate = risk_free_rate
    
    def price(
        self,
        spot: float,
        strike: float,
        time_to_expiry: float,
        iv: float,
        is_call: bool = True
    ) -> OptionPrice:
        """
        Calculate option price and Greeks.
        
        Args:
            spot: Current underlying price
            strike: Strike price
            time_to_expiry: Time to expiry in years (e.g., 0.0027 for 1 day)
            iv: Implied volatility (annualized, e.g., 0.30 for 30%)
            is_call: True for call, False for put
            
        Returns:
            OptionPrice with premium and Greeks
        """
        # Handle edge cases
        if time_to_expiry <= 0:
            intrinsic = max(0, spot - strike) if is_call else max(0, strike - spot)
            return OptionPrice(
                premium=intrinsic,
                intrinsic=intrinsic,
                time_value=0,
                greeks=OptionGreeks(
                    delta=1.0 if is_call and spot > strike else (0.0 if is_call else -1.0 if spot < strike else 0.0),
                    gamma=0,
                    theta=0,
                    vega=0
                ),
                iv=iv
            )
        
        if iv <= 0:
            iv = 0.001  # Minimum IV to avoid division by zero
        
        # Black-Scholes d1 and d2
        d1 = (math.log(spot / strike) + (self.risk_free_rate + 0.5 * iv ** 2) * time_to_expiry) / (iv * math.sqrt(time_to_expiry))
        d2 = d1 - iv * math.sqrt(time_to_expiry)
        
        # Standard normal CDF and PDF
        n_d1 = stats.norm.cdf(d1)
        n_d2 = stats.norm.cdf(d2)
        n_neg_d1 = stats.norm.cdf(-d1)
        n_neg_d2 = stats.norm.cdf(-d2)
        pdf_d1 = stats.norm.pdf(d1)
        
        # Option price
        if is_call:
            premium = spot * n_d1 - strike * math.exp(-self.risk_free_rate * time_to_expiry) * n_d2
            intrinsic = max(0, spot - strike)
            delta = n_d1
        else:
            premium = strike * math.exp(-self.risk_free_rate * time_to_expiry) * n_neg_d2 - spot * n_neg_d1
            intrinsic = max(0, strike - spot)
            delta = n_d1 - 1
        
        time_value = premium - intrinsic
        
        # Greeks
        gamma = pdf_d1 / (spot * iv * math.sqrt(time_to_expiry))
        
        theta_part1 = -(spot * pdf_d1 * iv) / (2 * math.sqrt(time_to_expiry))
        if is_call:
            theta_part2 = -self.risk_free_rate * strike * math.exp(-self.risk_free_rate * time_to_expiry) * n_d2
        else:
            theta_part2 = self.risk_free_rate * strike * math.exp(-self.risk_free_rate * time_to_expiry) * n_neg_d2
        theta = (theta_part1 + theta_part2) / 365  # Daily theta
        
        vega = spot * math.sqrt(time_to_expiry) * pdf_d1 / 100  # Per 1% IV change
        
        rho = 0
        if is_call:
            rho = strike * time_to_expiry * math.exp(-self.risk_free_rate * time_to_expiry) * n_d2 / 100
        else:
            rho = -strike * time_to_expiry * math.exp(-self.risk_free_rate * time_to_expiry) * n_neg_d2 / 100
        
        return OptionPrice(
            premium=max(0, premium),
            intrinsic=intrinsic,
            time_value=max(0, time_value),
            greeks=OptionGreeks(
                delta=delta,
                gamma=gamma,
                theta=theta,
                vega=vega,
                rho=rho
            ),
            iv=iv
        )
    
    def solve_iv(
        self,
        market_price: float,
        spot: float,
        strike: float,
        time_to_expiry: float,
        is_call: bool = True,
        tolerance: float = 0.0001,
        max_iterations: int = 100
    ) -> float:
        """
        Solve for implied volatility using Newton-Raphson.
        
        Args:
            market_price: Observed market price
            spot: Current underlying price
            strike: Strike price
            time_to_expiry: Time to expiry in years
            is_call: True for call, False for put
            tolerance: Convergence tolerance
            max_iterations: Maximum iterations
            
        Returns:
            Implied volatility (annualized)
        """
        if time_to_expiry <= 0:
            return 0.0
        
        # Initial guess based on ATM approximation
        iv = math.sqrt(2 * math.pi / time_to_expiry) * market_price / spot
        iv = max(0.01, min(iv, 5.0))  # Bound initial guess
        
        for _ in range(max_iterations):
            price_result = self.price(spot, strike, time_to_expiry, iv, is_call)
            diff = price_result.premium - market_price
            
            if abs(diff) < tolerance:
                return iv
            
            # Vega for Newton-Raphson step
            vega = price_result.greeks.vega * 100  # Undo per-1% scaling
            if abs(vega) < 0.0001:
                # Vega too small, use bisection step
                if diff > 0:
                    iv *= 0.9
                else:
                    iv *= 1.1
            else:
                iv = iv - diff / vega
            
            # Bound IV
            iv = max(0.01, min(iv, 5.0))
        
        return iv


# ------------------------------------------------------------------------------
# IV Adjustment
# ------------------------------------------------------------------------------
def calculate_iv_adjustment(current_price: float, strike_price: float) -> float:
    """
    Calculate IV adjustment based on moneyness (volatility smile/skew).
    
    OTM options typically have higher IV due to:
    - Put skew (crash protection)
    - Call skew (FOMO premium)
    
    Args:
        current_price: Current underlying price
        strike_price: Strike price
        
    Returns:
        IV adjustment multiplier (e.g., 1.15 for 15% higher IV)
    """
    if current_price <= 0:
        return 1.0
    
    moneyness = strike_price / current_price
    
    # Skew parameters
    if moneyness < 1.0:  # OTM puts / ITM calls
        # Put skew: IV increases as we go further OTM
        distance = 1.0 - moneyness
        adjustment = 1.0 + distance * 0.5  # +50% per 100% OTM
    else:  # OTM calls / ITM puts
        # Call skew: Slight IV increase for far OTM calls
        distance = moneyness - 1.0
        adjustment = 1.0 + distance * 0.3  # +30% per 100% OTM
    
    # Cap adjustment
    return min(2.0, max(0.5, adjustment))


# ------------------------------------------------------------------------------
# Convenience Functions
# ------------------------------------------------------------------------------
def calculate_option_premium(
    spot: float,
    strike: float,
    time_to_expiry_days: float,
    iv: float,
    is_call: bool = True,
    risk_free_rate: float = 0.05
) -> float:
    """
    Calculate option premium using Black-Scholes.
    
    Convenience function for simple pricing needs.
    
    Args:
        spot: Current underlying price
        strike: Strike price
        time_to_expiry_days: Time to expiry in days
        iv: Implied volatility (annualized)
        is_call: True for call, False for put
        risk_free_rate: Annual risk-free rate
        
    Returns:
        Option premium
    """
    pricer = BlackScholesPricer(risk_free_rate)
    time_years = time_to_expiry_days / 365
    result = pricer.price(spot, strike, time_years, iv, is_call)
    return result.premium


def estimate_iv_from_premium(
    premium: float,
    spot: float,
    strike: float,
    time_to_expiry_days: float,
    is_call: bool = True
) -> float:
    """
    Estimate IV from option premium.
    
    Args:
        premium: Option premium
        spot: Current underlying price
        strike: Strike price
        time_to_expiry_days: Time to expiry in days
        is_call: True for call, False for put
        
    Returns:
        Estimated implied volatility
    """
    pricer = BlackScholesPricer()
    time_years = time_to_expiry_days / 365
    return pricer.solve_iv(premium, spot, strike, time_years, is_call)


def get_time_to_expiry_years(expiry_date: datetime, current_time: datetime = None) -> float:
    """
    Calculate time to expiry in years.
    
    Args:
        expiry_date: Option expiration date
        current_time: Current time (defaults to now)
        
    Returns:
        Time to expiry in years
    """
    if current_time is None:
        current_time = datetime.now()
    
    time_diff = expiry_date - current_time
    return max(0, time_diff.total_seconds() / (365.25 * 24 * 3600))


__all__ = [
    'OptionGreeks',
    'OptionPrice',
    'round_to_realistic_strike',
    'get_atm_strike',
    'get_otm_strikes',
    'BlackScholesPricer',
    'calculate_iv_adjustment',
    'calculate_option_premium',
    'estimate_iv_from_premium',
    'get_time_to_expiry_years',
]








