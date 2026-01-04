#!/usr/bin/env python3
"""
SyntheticOptionsEngine.py - Generate realistic synthetic options data using Black-Scholes
Provides option chain snapshots with accurate Greeks for backtesting
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
import datetime as dt
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.config import RunConfig, StrategyConfig

# ============================================================================
# BLACK-SCHOLES IMPLEMENTATION
# ============================================================================

class BlackScholesCalculator:
    """Calculate option prices and Greeks using Black-Scholes model"""
    
    def __init__(self, risk_free_rate: float = 0.05):
        self.r = risk_free_rate
    
    def d1(self, S: float, K: float, T: float, sigma: float) -> float:
        """Calculate d1 component of Black-Scholes"""
        if T <= 0:
            return float('inf') if S > K else float('-inf')
        return (np.log(S / K) + (self.r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    
    def d2(self, S: float, K: float, T: float, sigma: float) -> float:
        """Calculate d2 component of Black-Scholes"""
        if T <= 0:
            return float('inf') if S > K else float('-inf')
        d_1 = self.d1(S, K, T, sigma)
        return d_1 - sigma * np.sqrt(T)
    
    def call_price(self, S: float, K: float, T: float, sigma: float) -> float:
        """Calculate European call option price"""
        if T <= 0:
            return max(0, S - K)
        d_1 = self.d1(S, K, T, sigma)
        d_2 = self.d2(S, K, T, sigma)
        call = S * norm.cdf(d_1) - K * np.exp(-self.r * T) * norm.cdf(d_2)
        return max(0, call)
    
    def put_price(self, S: float, K: float, T: float, sigma: float) -> float:
        """Calculate European put option price"""
        if T <= 0:
            return max(0, K - S)
        d_1 = self.d1(S, K, T, sigma)
        d_2 = self.d2(S, K, T, sigma)
        put = K * np.exp(-self.r * T) * norm.cdf(-d_2) - S * norm.cdf(-d_1)
        return max(0, put)
    
    def delta_call(self, S: float, K: float, T: float, sigma: float) -> float:
        """Calculate call delta"""
        if T <= 0:
            return 1.0 if S > K else 0.0
        d_1 = self.d1(S, K, T, sigma)
        return norm.cdf(d_1)
    
    def delta_put(self, S: float, K: float, T: float, sigma: float) -> float:
        """Calculate put delta"""
        if T <= 0:
            return -1.0 if S > K else 0.0
        d_1 = self.d1(S, K, T, sigma)
        return norm.cdf(d_1) - 1.0
    
    def gamma(self, S: float, K: float, T: float, sigma: float) -> float:
        """Calculate gamma (same for calls and puts)"""
        if T <= 0:
            return 0.0
        d_1 = self.d1(S, K, T, sigma)
        return norm.pdf(d_1) / (S * sigma * np.sqrt(T))
    
    def theta_call(self, S: float, K: float, T: float, sigma: float) -> float:
        """Calculate call theta (per day)"""
        if T <= 0:
            return 0.0
        d_1 = self.d1(S, K, T, sigma)
        d_2 = self.d2(S, K, T, sigma)
        theta = (-S * norm.pdf(d_1) * sigma / (2 * np.sqrt(T)) - 
                 self.r * K * np.exp(-self.r * T) * norm.cdf(d_2))
        return theta / 365.0  # Convert to daily theta
    
    def theta_put(self, S: float, K: float, T: float, sigma: float) -> float:
        """Calculate put theta (per day)"""
        if T <= 0:
            return 0.0
        d_1 = self.d1(S, K, T, sigma)
        d_2 = self.d2(S, K, T, sigma)
        theta = (-S * norm.pdf(d_1) * sigma / (2 * np.sqrt(T)) + 
                 self.r * K * np.exp(-self.r * T) * norm.cdf(-d_2))
        return theta / 365.0  # Convert to daily theta
    
    def vega(self, S: float, K: float, T: float, sigma: float) -> float:
        """Calculate vega (per 1% change in volatility, per day basis)"""
        if T <= 0:
            return 0.0
        d_1 = self.d1(S, K, T, sigma)
        vega = S * norm.pdf(d_1) * np.sqrt(T) / 100.0  # Per 1% IV
        return vega / 365.0  # Convert to daily vega

# ============================================================================
# SYNTHETIC OPTIONS ENGINE
# ============================================================================

class SyntheticOptionsEngine:
    """Generate realistic synthetic options data based on spot prices"""
    
    def __init__(self, config: RunConfig):
        self.config = config
        self.bs = BlackScholesCalculator(risk_free_rate=config.risk_free_rate)
        self.iv_annual = config.iv_annual_volatility
    
    def generate_strikes(self, spot_price: float, atm_distance: float = 10.0) -> np.ndarray:
        """Generate strike prices around current spot"""
        # Create strikes from -$50 to +$50 around spot, in $1 increments
        min_strike = max(10, spot_price - 50)
        max_strike = spot_price + 50
        strikes = np.arange(np.ceil(min_strike), np.floor(max_strike) + 1, 1.0)
        return strikes
    
    def generate_expirations(self, current_date: dt.date) -> List[dt.date]:
        """Generate relevant expiration dates (weekly and monthly)"""
        expirations = []
        
        # Find next Friday (weekly expiration)
        days_to_friday = (4 - current_date.weekday()) % 7
        if days_to_friday == 0:
            days_to_friday = 7
        
        next_friday = current_date + dt.timedelta(days=days_to_friday)
        
        # Add weekly expirations for next 8 weeks
        for week in range(8):
            exp_date = next_friday + dt.timedelta(weeks=week)
            expirations.append(exp_date)
        
        return expirations
    
    def calculate_bid_ask_spread(self, mid_price: float, moneyness: float, 
                                dte: int) -> Tuple[float, float]:
        """
        Generate realistic bid-ask spread based on moneyness and DTE
        Moneyness: ratio of (K - S) / S (negative = OTM call, positive = ITM call)
        """
        
        # ATM options have tighter spreads
        # OTM options have wider spreads (as % of price)
        otm_factor = 1.0 + 2.0 * abs(moneyness)  # Wider for OTM
        
        # Shorter DTE = tighter spreads (more liquid)
        dte_factor = 1.0 + (60.0 / max(dte, 1.0)) * 0.5
        
        # Minimum spread floor
        spread_pct = max(0.01, 0.01 * otm_factor * dte_factor)  # 1-3% spread
        spread = mid_price * spread_pct
        
        bid = mid_price - spread / 2.0
        ask = mid_price + spread / 2.0
        
        return max(0.01, bid), ask
    
    def generate_option_chain_snapshot(self, spot_price: float, current_date: dt.date,
                                      underlying: str = "SPY") -> pd.DataFrame:
        """
        Generate complete option chain snapshot for a given spot price and date
        """
        
        records = []
        strikes = self.generate_strikes(spot_price)
        expirations = self.generate_expirations(current_date)
        
        for exp_date in expirations:
            dte = (exp_date - current_date).days
            
            # Skip expired contracts
            if dte < 0:
                continue
            
            # Annualized time to expiration
            T = max(dte / 365.0, 0.001)
            
            for strike in strikes:
                # Skip extreme strikes to reduce data size
                moneyness = (strike - spot_price) / spot_price
                if abs(moneyness) > 0.30:  # Within 30% OTM/ITM
                    continue
                
                # Calculate call option
                call_price = self.bs.call_price(spot_price, strike, T, self.iv_annual)
                call_delta = self.bs.delta_call(spot_price, strike, T, self.iv_annual)
                call_bid, call_ask = self.calculate_bid_ask_spread(call_price, moneyness, dte)
                call_mid = (call_bid + call_ask) / 2.0
                
                call_gamma = self.bs.gamma(spot_price, strike, T, self.iv_annual)
                call_theta = self.bs.theta_call(spot_price, strike, T, self.iv_annual)
                call_vega = self.bs.vega(spot_price, strike, T, self.iv_annual)
                
                records.append({
                    'timestamp': dt.datetime.combine(current_date, dt.time(16, 0)),
                    'date': current_date,
                    'symbol': underlying,
                    'option_symbol': f"{underlying}{exp_date.strftime('%y%m%d')}C{strike:08.2f}",
                    'strike': strike,
                    'expiration': exp_date,
                    'contract_type': 'call',
                    'bid': round(call_bid, 2),
                    'ask': round(call_ask, 2),
                    'last_price': round(call_mid, 2),
                    'bid_size': 100,
                    'ask_size': 100,
                    'volume': np.random.randint(10, 1000),
                    'open_interest': np.random.randint(100, 10000),
                    'delta': round(call_delta, 4),
                    'gamma': round(call_gamma, 6),
                    'theta': round(call_theta, 4),
                    'vega': round(call_vega, 4),
                    'implied_volatility': self.iv_annual,
                })
                
                # Calculate put option
                put_price = self.bs.put_price(spot_price, strike, T, self.iv_annual)
                put_delta = self.bs.delta_put(spot_price, strike, T, self.iv_annual)
                put_bid, put_ask = self.calculate_bid_ask_spread(put_price, -moneyness, dte)
                put_mid = (put_bid + put_ask) / 2.0
                
                put_gamma = call_gamma  # Same as call
                put_theta = self.bs.theta_put(spot_price, strike, T, self.iv_annual)
                put_vega = call_vega  # Same as call
                
                records.append({
                    'timestamp': dt.datetime.combine(current_date, dt.time(16, 0)),
                    'date': current_date,
                    'symbol': underlying,
                    'option_symbol': f"{underlying}{exp_date.strftime('%y%m%d')}P{strike:08.2f}",
                    'strike': strike,
                    'expiration': exp_date,
                    'contract_type': 'put',
                    'bid': round(put_bid, 2),
                    'ask': round(put_ask, 2),
                    'last_price': round(put_mid, 2),
                    'bid_size': 100,
                    'ask_size': 100,
                    'volume': np.random.randint(10, 1000),
                    'open_interest': np.random.randint(100, 10000),
                    'delta': round(put_delta, 4),
                    'gamma': round(put_gamma, 6),
                    'theta': round(put_theta, 4),
                    'vega': round(put_vega, 4),
                    'implied_volatility': self.iv_annual,
                })
        
        df = pd.DataFrame(records)
        return df
    
    def generate_full_backtest_data(self, spot_data: pd.DataFrame, 
                                    underlying: str = "SPY") -> pd.DataFrame:
        """
        Generate synthetic options data for entire backtest period based on spot prices
        
        spot_data: DataFrame with columns ['date', 'close']
        """
        
        all_options = []
        
        for idx, row in spot_data.iterrows():
            date = row['date']
            spot = row['close']
            
            # Skip weekends
            if date.weekday() >= 5:
                continue
            
            # Generate option chain for this date
            chain = self.generate_option_chain_snapshot(spot, date, underlying)
            all_options.append(chain)
        
        # Combine all snapshots
        combined = pd.concat(all_options, ignore_index=True)
        return combined

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def save_synthetic_options(df: pd.DataFrame, underlying: str, timeframe: str = "D") -> str:
    """Save synthetic options data to CSV"""
    
    data_dir = Path("data/synthetic_options")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    filename = data_dir / f"{underlying}_{timeframe}.csv"
    df.to_csv(filename, index=False)
    
    return str(filename)

def validate_greeks(df: pd.DataFrame) -> bool:
    """Validate that Greeks are within reasonable ranges"""
    
    issues = []
    
    # Delta should be between -1 and 1
    calls = df[df['contract_type'] == 'call']
    puts = df[df['contract_type'] == 'put']
    
    if (calls['delta'] < 0).any() or (calls['delta'] > 1).any():
        issues.append("Call deltas outside [0, 1] range")
    
    if (puts['delta'] < -1).any() or (puts['delta'] > 0).any():
        issues.append("Put deltas outside [-1, 0] range")
    
    # Gamma should be positive
    if (df['gamma'] < 0).any():
        issues.append("Negative gamma values found")
    
    # Theta should generally be negative (except deep ITM calls)
    if (df['theta'] > 0.1).any():
        issues.append("Unexpectedly high theta values")
    
    # Vega should be positive
    if (df['vega'] < 0).any():
        issues.append("Negative vega values found")
    
    if issues:
        print("[WARNING] Greeks Validation Issues:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    
    print("[OK] Greeks validation passed")
    return True

# ============================================================================
# MAIN EXECUTION (FOR TESTING)
# ============================================================================

if __name__ == "__main__":
    config = RunConfig()
    engine = SyntheticOptionsEngine(config)
    
    # Test with sample spot prices
    test_dates = pd.date_range('2025-12-01', '2025-12-27', freq='B')  # Business days
    spot_prices = 600 + np.cumsum(np.random.randn(len(test_dates)) * 2)
    
    test_data = pd.DataFrame({
        'date': test_dates.date,
        'close': spot_prices
    })
    
    print("Generating synthetic options data...")
    options_df = engine.generate_full_backtest_data(test_data)
    
    print(f"Generated {len(options_df)} option records")
    print(f"Date range: {options_df['date'].min()} to {options_df['date'].max()}")
    print(f"Strikes range: ${options_df['strike'].min():.2f} to ${options_df['strike'].max():.2f}")
    
    # Validate
    validate_greeks(options_df)
    
    # Save
    filepath = save_synthetic_options(options_df)
    print(f"Saved to: {filepath}")