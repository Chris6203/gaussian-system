"""
Real Options Pricing and Greeks Calculation
Using Black-Scholes model for accurate option valuation
"""

import numpy as np
from scipy.stats import norm
from typing import Dict, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class OptionsGreeksCalculator:
    """
    Calculate real option prices and Greeks using Black-Scholes model
    Critical for proper options trading!
    """
    
    def __init__(self, risk_free_rate: float = 0.05):
        """
        Initialize calculator
        
        Args:
            risk_free_rate: Risk-free interest rate (default 5% = 0.05)
        """
        self.risk_free_rate = risk_free_rate
        self.logger = logging.getLogger(__name__)
    
    def calculate_all_greeks(
        self,
        spot_price: float,
        strike_price: float,
        time_to_expiration_days: float,
        implied_volatility: float,
        option_type: str = 'call'
    ) -> Dict[str, float]:
        """
        Calculate option price and all Greeks using Black-Scholes
        
        Args:
            spot_price: Current underlying price
            strike_price: Option strike price
            time_to_expiration_days: Days until expiration
            implied_volatility: Implied volatility (e.g., 0.25 for 25%)
            option_type: 'call' or 'put'
        
        Returns:
            Dictionary with price, delta, gamma, theta, vega, rho
        """
        S = spot_price
        K = strike_price
        T = time_to_expiration_days / 365.0  # Convert to years
        sigma = implied_volatility
        r = self.risk_free_rate
        
        # Handle edge cases
        if T <= 0:
            # Expired option
            if option_type.lower() == 'call':
                intrinsic = max(0, S - K)
            else:
                intrinsic = max(0, K - S)
            return {
                'price': intrinsic,
                'delta': 1.0 if intrinsic > 0 else 0.0,
                'gamma': 0.0,
                'theta': 0.0,
                'vega': 0.0,
                'rho': 0.0
            }
        
        # Black-Scholes calculation
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # Standard normal CDF and PDF
        N_d1 = norm.cdf(d1)
        N_d2 = norm.cdf(d2)
        n_d1 = norm.pdf(d1)  # PDF for gamma/vega
        
        if option_type.lower() == 'call':
            # Call option
            price = S * N_d1 - K * np.exp(-r * T) * N_d2
            delta = N_d1
            theta_base = -(S * n_d1 * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * N_d2
            rho = K * T * np.exp(-r * T) * N_d2
        else:
            # Put option
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            delta = N_d1 - 1
            theta_base = -(S * n_d1 * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
        
        # Greeks (same for calls and puts)
        gamma = n_d1 / (S * sigma * np.sqrt(T))
        theta = theta_base / 365  # Convert to per-day theta
        vega = S * n_d1 * np.sqrt(T) / 100  # Per 1% change in IV
        
        return {
            'price': price,
            'delta': delta,
            'gamma': gamma,
            'theta': theta,  # Per day
            'vega': vega,    # Per 1% IV change
            'rho': rho
        }
    
    def calculate_strike_for_delta(
        self,
        spot_price: float,
        target_delta: float,
        time_to_expiration_days: float,
        implied_volatility: float,
        option_type: str = 'call'
    ) -> float:
        """
        Find the strike price that gives a target delta
        Useful for intelligent strike selection!
        
        Args:
            spot_price: Current underlying price
            target_delta: Desired delta (e.g., 0.30, 0.50, 0.70)
            time_to_expiration_days: Days until expiration
            implied_volatility: Implied volatility
            option_type: 'call' or 'put'
        
        Returns:
            Strike price that achieves target delta
        """
        # For puts, adjust target (put deltas are negative)
        # ✅ FIX: Just flip sign, don't subtract 1!
        # For a "0.20 delta put" we want -0.20, not -0.80
        if option_type.lower() == 'put':
            target_delta = -abs(target_delta)  # Always negative for puts
        
        # Binary search for strike
        low_strike = spot_price * 0.70  # 30% OTM
        high_strike = spot_price * 1.30  # 30% ITM
        
        for _ in range(50):  # Max 50 iterations
            mid_strike = (low_strike + high_strike) / 2
            
            greeks = self.calculate_all_greeks(
                spot_price, mid_strike, time_to_expiration_days,
                implied_volatility, option_type
            )
            
            delta = greeks['delta']
            
            if abs(delta - target_delta) < 0.01:  # Within 1% of target
                return mid_strike
            
            # ✅ FIX: Corrected binary search logic for puts
            # For CALLS: Higher strike → Lower delta (more OTM)
            # For PUTS: Higher strike → More negative delta (more ITM)
            
            if option_type.lower() == 'call':
                # Call logic: delta decreases as strike increases
                if delta > target_delta:
                    # Delta too high, need higher strike (more OTM)
                    low_strike = mid_strike
                else:
                    # Delta too low, need lower strike (more ITM)
                    high_strike = mid_strike
            else:
                # Put logic: delta gets more negative as strike increases
                if delta < target_delta:
                    # Delta too negative, need lower strike (less ITM / more OTM)
                    high_strike = mid_strike
                else:
                    # Delta not negative enough, need higher strike (more ITM)
                    low_strike = mid_strike
        
        # If we didn't converge, return ATM
        return spot_price
    
    def optimal_exit_score(
        self,
        greeks: Dict[str, float],
        current_pnl_pct: float,
        days_held: float,
        days_to_expiration: float
    ) -> Tuple[float, str]:
        """
        Calculate an exit score based on Greeks and P&L
        Higher score = should exit
        
        Args:
            greeks: Current option Greeks
            current_pnl_pct: Current P&L as percentage
            days_held: Days position has been held
            days_to_expiration: Days until expiration
        
        Returns:
            (exit_score, reason) where score > 0.7 suggests exit
        """
        score = 0.0
        reasons = []
        
        # 1. Profit target (most important)
        if current_pnl_pct > 0.50:  # 50% profit
            score += 0.5
            reasons.append(f"Strong profit: {current_pnl_pct:.1%}")
        elif current_pnl_pct > 0.25:  # 25% profit
            score += 0.3
            reasons.append(f"Good profit: {current_pnl_pct:.1%}")
        
        # 2. Theta decay risk
        remaining_theta = abs(greeks['theta']) * days_to_expiration
        if current_pnl_pct > 0 and remaining_theta > (current_pnl_pct * 0.5):
            # Theta will eat half your profit
            score += 0.3
            reasons.append(f"Theta risk: ${remaining_theta:.2f} decay ahead")
        
        # 3. Time-based (approaching expiration)
        if days_to_expiration < 2:
            score += 0.3
            reasons.append(f"Near expiration: {days_to_expiration:.1f} days left")
        elif days_to_expiration < 1:
            score += 0.5
            reasons.append(f"Same-day expiration!")
        
        # 4. Loss management
        if current_pnl_pct < -0.30:  # 30% loss
            score += 0.4
            reasons.append(f"Stop loss: {current_pnl_pct:.1%}")
        elif current_pnl_pct < -0.50:  # 50% loss
            score += 0.6
            reasons.append(f"Hard stop: {current_pnl_pct:.1%}")
        
        # 5. Delta erosion (option going OTM)
        if abs(greeks['delta']) < 0.15:  # Very low delta
            score += 0.2
            reasons.append(f"Low delta: {greeks['delta']:.2f}")
        
        reason = " | ".join(reasons) if reasons else "Hold - no exit criteria met"
        return min(score, 1.0), reason


class IntelligentStrikeSelector:
    """
    Select optimal strikes based on prediction and market conditions
    """
    
    def __init__(self, greeks_calculator: OptionsGreeksCalculator):
        self.greeks_calc = greeks_calculator
        self.logger = logging.getLogger(__name__)
    
    def select_strike(
        self,
        spot_price: float,
        predicted_move_pct: float,
        confidence: float,
        implied_volatility: float,
        time_to_expiration_days: float,
        option_type: str
    ) -> Dict[str, float]:
        """
        Intelligently select strike based on prediction strength
        
        Args:
            spot_price: Current price
            predicted_move_pct: Expected move as percentage (e.g., 0.02 for 2%)
            confidence: Prediction confidence (0-1)
            implied_volatility: Current IV
            time_to_expiration_days: DTE
            option_type: 'call' or 'put'
        
        Returns:
            Dict with strike, expected_delta, greeks
        """
        # Determine target delta based on confidence
        if confidence > 0.75:
            # Very high confidence → ATM (delta ~0.50)
            target_delta = 0.50
            self.logger.info(f"High confidence ({confidence:.1%}) → ATM strike (delta 0.50)")
        elif confidence > 0.55:
            # Medium-high confidence → Slight OTM (delta ~0.40)
            target_delta = 0.40
            self.logger.info(f"Medium confidence ({confidence:.1%}) → Slight OTM (delta 0.40)")
        elif confidence > 0.40:
            # Medium confidence → OTM (delta ~0.30)
            target_delta = 0.30
            self.logger.info(f"Lower confidence ({confidence:.1%}) → OTM (delta 0.30)")
        else:
            # Low confidence → Far OTM (delta ~0.20) - lottery ticket
            target_delta = 0.20
            self.logger.info(f"Low confidence ({confidence:.1%}) → Far OTM (delta 0.20)")
        
        # Find strike for target delta
        strike = self.greeks_calc.calculate_strike_for_delta(
            spot_price, target_delta, time_to_expiration_days,
            implied_volatility, option_type
        )
        
        # Calculate Greeks for selected strike
        greeks = self.greeks_calc.calculate_all_greeks(
            spot_price, strike, time_to_expiration_days,
            implied_volatility, option_type
        )
        
        self.logger.info(f"Selected strike: ${strike:.2f} with delta={greeks['delta']:.2f}, "
                        f"theta=${greeks['theta']:.2f}/day, price=${greeks['price']:.2f}")
        
        return {
            'strike': strike,
            'target_delta': target_delta,
            'actual_delta': greeks['delta'],
            'greeks': greeks
        }


class DTEOptimizer:
    """
    Optimize Days To Expiration based on prediction timeframe
    """
    
    @staticmethod
    def calculate_optimal_dte(prediction_timeframe_minutes: int) -> int:
        """
        Match DTE to prediction timeframe + buffer for theta decay
        
        Args:
            prediction_timeframe_minutes: Prediction horizon in minutes
        
        Returns:
            Optimal days to expiration
        """
        pred_hours = prediction_timeframe_minutes / 60
        
        if pred_hours < 1:  # Intraday scalp (< 1 hour)
            dte = 0  # Same day expiration (0DTE)
        elif pred_hours < 4:  # Short-term (< 4 hours)
            dte = 1  # Next day
        elif pred_hours < 24:  # Intraday to daily
            dte = 2
        else:  # Multi-day
            dte = min(7, int(pred_hours / 24) + 1)
        
        # Add 1-day buffer for safety and theta management
        dte += 1
        
        logger.info(f"Prediction timeframe: {pred_hours:.1f}h → Optimal DTE: {dte} days")
        
        return dte
    
    @staticmethod
    def get_next_expiration(dte_target: int, current_date: datetime) -> datetime:
        """
        Get the next available expiration date
        
        In practice, would check actual option chain
        For now, just adds days
        """
        return current_date + timedelta(days=dte_target)
