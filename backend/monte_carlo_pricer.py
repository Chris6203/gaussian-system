"""
Monte Carlo Options Pricing with Distribution-Aware P&L

Simulates option P&L distributions considering:
- Greeks (delta, gamma, theta, vega)
- IV surface and regime-dependent shocks
- Path-dependent effects
- Realistic time decay

This provides honest expectancy that respects market dynamics.
"""

import numpy as np
import logging
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MCPricingParams:
    """Parameters for Monte Carlo simulation"""
    underlying_price: float
    strike: float
    option_type: str  # 'call' or 'put'
    current_iv: float
    horizon_minutes: int
    predicted_drift: float  # From neural network
    regime: Optional[str] = None
    num_paths: int = 500
    risk_free_rate: float = 0.05
    

class MonteCarloOptionPricer:
    """
    Monte Carlo pricer for option P&L distributions
    
    Uses Black-Scholes framework with:
    - Regime-dependent IV shocks
    - Antithetic sampling for variance reduction
    - Greeks-aware P&L calculation
    """
    
    def __init__(self, seed: int = 42):
        """
        Args:
            seed: Random seed for reproducibility
        """
        self.rng = np.random.default_rng(seed)
        logger.info("[MC] Monte Carlo pricer initialized")
    
    def estimate_iv_shock(self, predicted_drift: float, regime: Optional[str] = None) -> float:
        """
        Estimate IV change based on predicted move and regime
        
        Args:
            predicted_drift: Predicted return from neural network
            regime: HMM regime if available
            
        Returns:
            Expected IV change in points (e.g., 0.01 for +1%)
        """
        # Base IV response to price moves
        if predicted_drift < -0.02:
            # Significant selloff → IV spike
            iv_shock = 0.015  # +1.5%
        elif predicted_drift < -0.01:
            # Moderate selloff → IV rise
            iv_shock = 0.01  # +1.0%
        elif predicted_drift > 0.02:
            # Strong rally → IV crush
            iv_shock = -0.005  # -0.5%
        elif predicted_drift > 0.01:
            # Moderate rally → slight IV decline
            iv_shock = -0.003  # -0.3%
        else:
            # Small moves → minimal IV change
            iv_shock = 0.0
        
        # Adjust based on regime
        if regime:
            if 'stress' in regime.lower() or 'volatile' in regime.lower():
                iv_shock += 0.005  # Add +0.5% in stress
            elif 'panic' in regime.lower():
                iv_shock += 0.01   # Add +1.0% in panic
            elif 'calm' in regime.lower() or 'low_vol' in regime.lower():
                iv_shock -= 0.003  # Subtract 0.3% in calm
        
        logger.debug(f"[MC] IV shock estimate: {iv_shock*100:+.1f}% (drift={predicted_drift*100:.1f}%, regime={regime})")
        return iv_shock
    
    def black_scholes_price(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str
    ) -> float:
        """
        Black-Scholes option pricing
        
        Args:
            S: Underlying price
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free rate
            sigma: Implied volatility
            option_type: 'call' or 'put'
            
        Returns:
            Option price
        """
        from scipy.stats import norm
        
        if T <= 0:
            # At expiration, return intrinsic value
            if option_type == 'call':
                return max(0, S - K)
            else:
                return max(0, K - S)
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type == 'call':
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        
        return max(0, price)
    
    def simulate_pnl_distribution(
        self,
        params: MCPricingParams,
        current_option_price: float
    ) -> Dict[str, float]:
        """
        Simulate option P&L distribution using Monte Carlo
        
        Args:
            params: Pricing parameters
            current_option_price: Current market price of option
            
        Returns:
            Dictionary with:
                - mean: Expected P&L
                - std: Standard deviation
                - q05: 5th percentile (downside risk)
                - q25: 25th percentile
                - q50: Median
                - q75: 75th percentile
                - q95: 95th percentile (upside potential)
                - sharpe: Sharpe-like ratio (mean/std)
        """
        num_paths = params.num_paths
        
        # Convert horizon to years
        T_current = 30 / 365.0  # Assume 30 DTE currently (could be passed in)
        T_horizon = (30 * 24 * 60 - params.horizon_minutes) / (365 * 24 * 60)
        dt = (params.horizon_minutes / (24 * 60)) / 365.0  # Time step in years
        
        # Estimate IV shock
        iv_shock = self.estimate_iv_shock(params.predicted_drift, params.regime)
        future_iv = params.current_iv + iv_shock
        
        # Generate price paths with antithetic sampling
        half_paths = num_paths // 2
        
        # Random normal samples
        Z = self.rng.standard_normal(half_paths)
        
        # Antithetic paths (variance reduction)
        Z_anti = -Z
        Z_all = np.concatenate([Z, Z_anti])
        
        # Simulate underlying price at horizon
        # dS = mu*S*dt + sigma*S*sqrt(dt)*Z
        drift = params.predicted_drift
        vol = params.current_iv
        
        S_future = params.underlying_price * np.exp(
            (drift - 0.5 * vol**2) * dt + vol * np.sqrt(dt) * Z_all
        )
        
        # Calculate option prices at horizon
        option_prices_future = np.array([
            self.black_scholes_price(
                S=S,
                K=params.strike,
                T=T_horizon,
                r=params.risk_free_rate,
                sigma=future_iv,
                option_type=params.option_type
            )
            for S in S_future
        ])
        
        # Calculate P&L per contract (× 100 shares)
        pnl_per_contract = (option_prices_future - current_option_price) * 100
        
        # Calculate statistics
        results = {
            'mean': float(np.mean(pnl_per_contract)),
            'std': float(np.std(pnl_per_contract)),
            'q05': float(np.percentile(pnl_per_contract, 5)),
            'q25': float(np.percentile(pnl_per_contract, 25)),
            'q50': float(np.percentile(pnl_per_contract, 50)),
            'q75': float(np.percentile(pnl_per_contract, 75)),
            'q95': float(np.percentile(pnl_per_contract, 95)),
        }
        
        # Sharpe-like ratio
        if results['std'] > 0:
            results['sharpe'] = results['mean'] / results['std']
        else:
            results['sharpe'] = 0.0
        
        logger.debug(f"[MC] P&L distribution: mean=${results['mean']:.2f}, "
                    f"q05=${results['q05']:.2f}, q95=${results['q95']:.2f}")
        
        return results
    
    def simulate_vertical_pnl_distribution(
        self,
        long_params: MCPricingParams,
        short_params: MCPricingParams,
        long_price: float,
        short_price: float
    ) -> Dict[str, float]:
        """
        Simulate P&L distribution for a vertical spread
        
        Args:
            long_params: Parameters for long leg
            short_params: Parameters for short leg
            long_price: Current price of long leg
            short_price: Current price of short leg
            
        Returns:
            P&L distribution statistics
        """
        num_paths = long_params.num_paths
        
        # Must use same random paths for both legs
        half_paths = num_paths // 2
        Z = self.rng.standard_normal(half_paths)
        Z_anti = -Z
        Z_all = np.concatenate([Z, Z_anti])
        
        # Time parameters
        T_current = 30 / 365.0
        T_horizon = (30 * 24 * 60 - long_params.horizon_minutes) / (365 * 24 * 60)
        dt = (long_params.horizon_minutes / (24 * 60)) / 365.0
        
        # Simulate underlying (same for both legs)
        drift = long_params.predicted_drift
        vol = long_params.current_iv
        S_future = long_params.underlying_price * np.exp(
            (drift - 0.5 * vol**2) * dt + vol * np.sqrt(dt) * Z_all
        )
        
        # IV shocks
        iv_shock = self.estimate_iv_shock(long_params.predicted_drift, long_params.regime)
        future_iv = long_params.current_iv + iv_shock
        
        # Long leg prices at horizon
        long_prices_future = np.array([
            self.black_scholes_price(
                S=S, K=long_params.strike, T=T_horizon,
                r=long_params.risk_free_rate, sigma=future_iv,
                option_type=long_params.option_type
            )
            for S in S_future
        ])
        
        # Short leg prices at horizon
        short_prices_future = np.array([
            self.black_scholes_price(
                S=S, K=short_params.strike, T=T_horizon,
                r=short_params.risk_free_rate, sigma=future_iv,
                option_type=short_params.option_type
            )
            for S in S_future
        ])
        
        # Vertical P&L = (long gain - long cost) - (short gain - short cost)
        # We bought long at long_price, sell at long_prices_future
        # We sold short at short_price, buy back at short_prices_future
        pnl_per_contract = (
            (long_prices_future - long_price) - (short_prices_future - short_price)
        ) * 100
        
        # Calculate statistics
        results = {
            'mean': float(np.mean(pnl_per_contract)),
            'std': float(np.std(pnl_per_contract)),
            'q05': float(np.percentile(pnl_per_contract, 5)),
            'q25': float(np.percentile(pnl_per_contract, 25)),
            'q50': float(np.percentile(pnl_per_contract, 50)),
            'q75': float(np.percentile(pnl_per_contract, 75)),
            'q95': float(np.percentile(pnl_per_contract, 95)),
        }
        
        if results['std'] > 0:
            results['sharpe'] = results['mean'] / results['std']
        else:
            results['sharpe'] = 0.0
        
        logger.debug(f"[MC] Vertical P&L: mean=${results['mean']:.2f}, "
                    f"q05=${results['q05']:.2f}, q95=${results['q95']:.2f}")
        
        return results

