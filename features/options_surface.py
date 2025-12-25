#!/usr/bin/env python3
"""
Volatility & Options Surface Features

Computes features from SPY options chain data:
- ATM implied volatility
- IV skew (OTM puts vs OTM calls)
- Term structure (near vs far expiries)
- Put/call open interest and volume ratios
- Dealer gamma exposure (GEX) approximation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, date
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def find_atm_options(
    chain: pd.DataFrame,
    spot_price: float,
    n_strikes: int = 3
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Find ATM call and put options.
    
    Args:
        chain: Options chain DataFrame
        spot_price: Current underlying price
        n_strikes: Number of strikes around ATM to include
        
    Returns:
        Tuple of (atm_calls, atm_puts) DataFrames
    """
    if chain.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    # Split by type
    calls = chain[chain['type'] == 'call'].copy() if 'type' in chain.columns else pd.DataFrame()
    puts = chain[chain['type'] == 'put'].copy() if 'type' in chain.columns else pd.DataFrame()
    
    if calls.empty and puts.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    # Find strikes closest to spot
    all_strikes = chain['strike'].unique() if 'strike' in chain.columns else []
    if len(all_strikes) == 0:
        return pd.DataFrame(), pd.DataFrame()
    
    # Sort by distance from spot
    strike_distances = [(s, abs(s - spot_price)) for s in all_strikes]
    strike_distances.sort(key=lambda x: x[1])
    atm_strikes = [s[0] for s in strike_distances[:n_strikes]]
    
    atm_calls = calls[calls['strike'].isin(atm_strikes)] if not calls.empty else pd.DataFrame()
    atm_puts = puts[puts['strike'].isin(atm_strikes)] if not puts.empty else pd.DataFrame()
    
    return atm_calls, atm_puts


def find_otm_options(
    chain: pd.DataFrame,
    spot_price: float,
    moneyness_pct: float = 5.0
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Find OTM options by moneyness threshold.
    
    Args:
        chain: Options chain DataFrame
        spot_price: Current underlying price
        moneyness_pct: Percentage OTM threshold (e.g., 5 = 5% OTM)
        
    Returns:
        Tuple of (otm_calls, otm_puts) DataFrames
    """
    if chain.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    calls = chain[chain['type'] == 'call'].copy() if 'type' in chain.columns else pd.DataFrame()
    puts = chain[chain['type'] == 'put'].copy() if 'type' in chain.columns else pd.DataFrame()
    
    threshold = spot_price * (moneyness_pct / 100.0)
    
    # OTM calls: strike > spot + threshold
    otm_calls = calls[calls['strike'] > spot_price + threshold] if not calls.empty else pd.DataFrame()
    
    # OTM puts: strike < spot - threshold
    otm_puts = puts[puts['strike'] < spot_price - threshold] if not puts.empty else pd.DataFrame()
    
    return otm_calls, otm_puts


def compute_gex(
    chain: pd.DataFrame,
    spot_price: float
) -> Dict[str, float]:
    """
    Compute Gamma Exposure (GEX) approximation.
    
    GEX = gamma * spot^2 * open_interest * contract_multiplier
    
    Sign convention:
    - Calls: Dealers are short -> negative GEX when gamma > 0
    - Puts: Dealers are long -> positive GEX when gamma > 0
    
    Args:
        chain: Options chain with gamma and open_interest
        spot_price: Current underlying price
        
    Returns:
        Dict with GEX metrics
    """
    if chain.empty or 'gamma' not in chain.columns or 'open_interest' not in chain.columns:
        return {'gex_total': 0.0, 'gex_near_atm': 0.0}
    
    contract_mult = 100  # Standard options contract multiplier
    
    calls = chain[chain['type'] == 'call'] if 'type' in chain.columns else pd.DataFrame()
    puts = chain[chain['type'] == 'put'] if 'type' in chain.columns else pd.DataFrame()
    
    gex_total = 0.0
    gex_near_atm = 0.0
    
    # Near ATM threshold (within 2% of spot)
    atm_threshold = spot_price * 0.02
    
    # Calls (dealers short -> negative sign)
    if not calls.empty:
        calls_gex = -calls['gamma'] * (spot_price ** 2) * calls['open_interest'] * contract_mult
        gex_total += calls_gex.sum()
        
        near_atm_calls = calls[abs(calls['strike'] - spot_price) < atm_threshold]
        if not near_atm_calls.empty:
            gex_near_atm += (-near_atm_calls['gamma'] * (spot_price ** 2) * 
                           near_atm_calls['open_interest'] * contract_mult).sum()
    
    # Puts (dealers long -> positive sign)
    if not puts.empty:
        puts_gex = puts['gamma'] * (spot_price ** 2) * puts['open_interest'] * contract_mult
        gex_total += puts_gex.sum()
        
        near_atm_puts = puts[abs(puts['strike'] - spot_price) < atm_threshold]
        if not near_atm_puts.empty:
            gex_near_atm += (near_atm_puts['gamma'] * (spot_price ** 2) * 
                           near_atm_puts['open_interest'] * contract_mult).sum()
    
    # Normalize to millions for readability
    return {
        'gex_total': gex_total / 1e6,
        'gex_near_atm': gex_near_atm / 1e6
    }


# =============================================================================
# MAIN FEATURE COMPUTATION FUNCTION
# =============================================================================

def compute_options_surface_features(
    chain: pd.DataFrame,
    spot_price: float,
    expiration_date: Optional[str] = None
) -> Dict[str, float]:
    """
    Compute volatility and options surface features from chain data.
    
    Args:
        chain: Options chain DataFrame with columns:
            - type: 'call' or 'put'
            - strike: Strike price
            - expiration: Expiration date
            - bid, ask, last: Prices
            - volume, open_interest: Activity metrics
            - delta, gamma, theta, vega: Greeks
            - bid_iv, mid_iv, ask_iv: Implied volatilities
        spot_price: Current underlying price
        expiration_date: Filter to specific expiration (optional)
        
    Returns:
        Dict of feature name -> value:
        
        ATM IV:
        - atm_iv_call: ATM call mid IV
        - atm_iv_put: ATM put mid IV
        - atm_iv_avg: Average ATM IV
        
        Skew:
        - iv_skew_25d: 25-delta put IV - 25-delta call IV (approx)
        - iv_skew_otm: OTM put IV - OTM call IV
        
        Term Structure:
        - iv_term_slope: IV slope across expirations
        
        Put/Call Metrics:
        - oi_total_calls: Total call open interest
        - oi_total_puts: Total put open interest
        - oi_put_call_ratio: Put/call OI ratio
        - vol_total_calls: Total call volume
        - vol_total_puts: Total put volume
        - vol_put_call_ratio: Put/call volume ratio
        
        Near-Money Positioning:
        - oi_near_money_calls: Near-ATM call OI
        - oi_near_money_puts: Near-ATM put OI
        - vol_near_money_calls: Near-ATM call volume
        - vol_near_money_puts: Near-ATM put volume
        - near_money_put_bias: Put OI / (Put OI + Call OI) near ATM
        
        GEX:
        - gex_total: Total gamma exposure (millions)
        - gex_near_atm: ATM gamma exposure (millions)
    """
    features = {
        # ATM IV
        'atm_iv_call': 0.0,
        'atm_iv_put': 0.0,
        'atm_iv_avg': 0.0,
        
        # Skew
        'iv_skew_25d': 0.0,
        'iv_skew_otm': 0.0,
        
        # Term Structure
        'iv_term_slope': 0.0,
        
        # Put/Call OI
        'oi_total_calls': 0.0,
        'oi_total_puts': 0.0,
        'oi_put_call_ratio': 1.0,
        
        # Put/Call Volume
        'vol_total_calls': 0.0,
        'vol_total_puts': 0.0,
        'vol_put_call_ratio': 1.0,
        
        # Near-Money
        'oi_near_money_calls': 0.0,
        'oi_near_money_puts': 0.0,
        'vol_near_money_calls': 0.0,
        'vol_near_money_puts': 0.0,
        'near_money_put_bias': 0.5,
        
        # GEX
        'gex_total': 0.0,
        'gex_near_atm': 0.0,
    }
    
    if chain.empty or spot_price <= 0:
        return features
    
    # Filter by expiration if specified
    if expiration_date and 'expiration' in chain.columns:
        chain = chain[chain['expiration'] == expiration_date]
    
    if chain.empty:
        return features
    
    try:
        # =====================================================================
        # ATM IV
        # =====================================================================
        atm_calls, atm_puts = find_atm_options(chain, spot_price, n_strikes=3)
        
        iv_col = 'mid_iv' if 'mid_iv' in chain.columns else 'bid_iv'
        
        if not atm_calls.empty and iv_col in atm_calls.columns:
            valid_iv = atm_calls[iv_col][atm_calls[iv_col] > 0]
            if not valid_iv.empty:
                features['atm_iv_call'] = float(valid_iv.mean())
        
        if not atm_puts.empty and iv_col in atm_puts.columns:
            valid_iv = atm_puts[iv_col][atm_puts[iv_col] > 0]
            if not valid_iv.empty:
                features['atm_iv_put'] = float(valid_iv.mean())
        
        if features['atm_iv_call'] > 0 and features['atm_iv_put'] > 0:
            features['atm_iv_avg'] = (features['atm_iv_call'] + features['atm_iv_put']) / 2
        elif features['atm_iv_call'] > 0:
            features['atm_iv_avg'] = features['atm_iv_call']
        elif features['atm_iv_put'] > 0:
            features['atm_iv_avg'] = features['atm_iv_put']
        
        # =====================================================================
        # IV SKEW
        # =====================================================================
        otm_calls, otm_puts = find_otm_options(chain, spot_price, moneyness_pct=5.0)
        
        otm_call_iv = 0.0
        otm_put_iv = 0.0
        
        if not otm_calls.empty and iv_col in otm_calls.columns:
            valid_iv = otm_calls[iv_col][otm_calls[iv_col] > 0]
            if not valid_iv.empty:
                otm_call_iv = float(valid_iv.mean())
        
        if not otm_puts.empty and iv_col in otm_puts.columns:
            valid_iv = otm_puts[iv_col][otm_puts[iv_col] > 0]
            if not valid_iv.empty:
                otm_put_iv = float(valid_iv.mean())
        
        if otm_put_iv > 0 and otm_call_iv > 0:
            features['iv_skew_otm'] = otm_put_iv - otm_call_iv
        
        # Approximate 25-delta skew using delta column if available
        if 'delta' in chain.columns:
            calls = chain[chain['type'] == 'call'] if 'type' in chain.columns else pd.DataFrame()
            puts = chain[chain['type'] == 'put'] if 'type' in chain.columns else pd.DataFrame()
            
            # 25-delta call (delta ~ 0.25)
            if not calls.empty:
                calls_25d = calls[(calls['delta'] > 0.20) & (calls['delta'] < 0.30)]
                if not calls_25d.empty and iv_col in calls_25d.columns:
                    call_25d_iv = calls_25d[iv_col].mean()
                else:
                    call_25d_iv = otm_call_iv
            else:
                call_25d_iv = otm_call_iv
            
            # 25-delta put (delta ~ -0.25)
            if not puts.empty:
                puts_25d = puts[(puts['delta'] > -0.30) & (puts['delta'] < -0.20)]
                if not puts_25d.empty and iv_col in puts_25d.columns:
                    put_25d_iv = puts_25d[iv_col].mean()
                else:
                    put_25d_iv = otm_put_iv
            else:
                put_25d_iv = otm_put_iv
            
            if put_25d_iv > 0 and call_25d_iv > 0:
                features['iv_skew_25d'] = put_25d_iv - call_25d_iv
        else:
            features['iv_skew_25d'] = features['iv_skew_otm']
        
        # =====================================================================
        # PUT/CALL METRICS
        # =====================================================================
        calls = chain[chain['type'] == 'call'] if 'type' in chain.columns else pd.DataFrame()
        puts = chain[chain['type'] == 'put'] if 'type' in chain.columns else pd.DataFrame()
        
        # Open Interest
        if 'open_interest' in chain.columns:
            features['oi_total_calls'] = float(calls['open_interest'].sum()) if not calls.empty else 0.0
            features['oi_total_puts'] = float(puts['open_interest'].sum()) if not puts.empty else 0.0
            
            total_oi = features['oi_total_calls'] + features['oi_total_puts']
            if features['oi_total_calls'] > 0:
                features['oi_put_call_ratio'] = features['oi_total_puts'] / features['oi_total_calls']
        
        # Volume
        if 'volume' in chain.columns:
            features['vol_total_calls'] = float(calls['volume'].sum()) if not calls.empty else 0.0
            features['vol_total_puts'] = float(puts['volume'].sum()) if not puts.empty else 0.0
            
            if features['vol_total_calls'] > 0:
                features['vol_put_call_ratio'] = features['vol_total_puts'] / features['vol_total_calls']
        
        # =====================================================================
        # NEAR-MONEY POSITIONING
        # =====================================================================
        # Near money = within 3% of spot
        near_threshold = spot_price * 0.03
        
        near_calls = calls[abs(calls['strike'] - spot_price) < near_threshold] if not calls.empty else pd.DataFrame()
        near_puts = puts[abs(puts['strike'] - spot_price) < near_threshold] if not puts.empty else pd.DataFrame()
        
        if 'open_interest' in chain.columns:
            features['oi_near_money_calls'] = float(near_calls['open_interest'].sum()) if not near_calls.empty else 0.0
            features['oi_near_money_puts'] = float(near_puts['open_interest'].sum()) if not near_puts.empty else 0.0
            
            total_near = features['oi_near_money_calls'] + features['oi_near_money_puts']
            if total_near > 0:
                features['near_money_put_bias'] = features['oi_near_money_puts'] / total_near
        
        if 'volume' in chain.columns:
            features['vol_near_money_calls'] = float(near_calls['volume'].sum()) if not near_calls.empty else 0.0
            features['vol_near_money_puts'] = float(near_puts['volume'].sum()) if not near_puts.empty else 0.0
        
        # =====================================================================
        # GAMMA EXPOSURE (GEX)
        # =====================================================================
        gex = compute_gex(chain, spot_price)
        features['gex_total'] = gex['gex_total']
        features['gex_near_atm'] = gex['gex_near_atm']
        
    except Exception as e:
        logger.warning(f"Error computing options surface features: {e}")
    
    return features


def compute_term_structure_features(
    chains: Dict[str, pd.DataFrame],
    spot_price: float,
    short_dte_max: int = 3,
    medium_dte_max: int = 10
) -> Dict[str, float]:
    """
    Compute IV term structure features across multiple expirations.
    
    Args:
        chains: Dict mapping expiration date -> chain DataFrame
        spot_price: Current underlying price
        short_dte_max: Max DTE for short-term bucket
        medium_dte_max: Max DTE for medium-term bucket
        
    Returns:
        Dict with term structure features:
        - iv_short_term: Average IV for short-term options
        - iv_medium_term: Average IV for medium-term options
        - iv_long_term: Average IV for long-term options
        - iv_term_slope_short: Short to medium term IV slope
        - iv_term_slope_long: Medium to long term IV slope
    """
    features = {
        'iv_short_term': 0.0,
        'iv_medium_term': 0.0,
        'iv_long_term': 0.0,
        'iv_term_slope_short': 0.0,
        'iv_term_slope_long': 0.0,
    }
    
    if not chains or spot_price <= 0:
        return features
    
    today = datetime.now().date()
    short_ivs = []
    medium_ivs = []
    long_ivs = []
    
    for exp_str, chain in chains.items():
        try:
            exp_date = datetime.strptime(exp_str, '%Y-%m-%d').date()
            dte = (exp_date - today).days
            
            if dte < 0:
                continue
            
            # Get ATM IV for this expiration
            surface_features = compute_options_surface_features(chain, spot_price)
            atm_iv = surface_features.get('atm_iv_avg', 0)
            
            if atm_iv > 0:
                if dte <= short_dte_max:
                    short_ivs.append(atm_iv)
                elif dte <= medium_dte_max:
                    medium_ivs.append(atm_iv)
                else:
                    long_ivs.append(atm_iv)
                    
        except Exception as e:
            logger.debug(f"Error processing expiration {exp_str}: {e}")
            continue
    
    # Compute averages
    if short_ivs:
        features['iv_short_term'] = np.mean(short_ivs)
    if medium_ivs:
        features['iv_medium_term'] = np.mean(medium_ivs)
    if long_ivs:
        features['iv_long_term'] = np.mean(long_ivs)
    
    # Compute slopes
    if features['iv_short_term'] > 0 and features['iv_medium_term'] > 0:
        features['iv_term_slope_short'] = features['iv_medium_term'] - features['iv_short_term']
    
    if features['iv_medium_term'] > 0 and features['iv_long_term'] > 0:
        features['iv_term_slope_long'] = features['iv_long_term'] - features['iv_medium_term']
    
    return features


if __name__ == "__main__":
    """Test options surface features with mock data."""
    import logging
    
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 70)
    print("OPTIONS SURFACE FEATURE TEST")
    print("=" * 70)
    
    # Create mock options chain
    spot = 450.0
    strikes = np.arange(430, 470, 2.5)
    
    records = []
    for strike in strikes:
        for opt_type in ['call', 'put']:
            moneyness = (strike - spot) / spot
            
            # Mock IV (higher for OTM puts - skew)
            base_iv = 0.15
            if opt_type == 'put' and strike < spot:
                iv = base_iv + 0.05 * abs(moneyness)
            elif opt_type == 'call' and strike > spot:
                iv = base_iv + 0.02 * abs(moneyness)
            else:
                iv = base_iv
            
            # Mock delta
            if opt_type == 'call':
                delta = max(0.0, min(1.0, 0.5 - moneyness * 5))
            else:
                delta = -max(0.0, min(1.0, 0.5 + moneyness * 5))
            
            records.append({
                'symbol': f'SPY{strike:.0f}{opt_type[0].upper()}',
                'type': opt_type,
                'strike': strike,
                'expiration': '2024-01-19',
                'bid': max(0.01, (spot - strike if opt_type == 'call' else strike - spot) + 2),
                'ask': max(0.05, (spot - strike if opt_type == 'call' else strike - spot) + 2.5),
                'volume': np.random.randint(100, 5000),
                'open_interest': np.random.randint(1000, 50000),
                'delta': delta,
                'gamma': 0.02 * np.exp(-abs(moneyness) * 10),
                'theta': -0.05,
                'vega': 0.15,
                'mid_iv': iv,
                'bid_iv': iv * 0.95,
                'ask_iv': iv * 1.05,
            })
    
    chain = pd.DataFrame(records)
    
    # Compute features
    features = compute_options_surface_features(chain, spot)
    
    print(f"\nSpot Price: ${spot}")
    print(f"Chain Size: {len(chain)} options")
    print("\nOptions Surface Features:")
    for k, v in sorted(features.items()):
        print(f"  {k}: {v:.4f}")
    
    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)




