#!/usr/bin/env python3
"""
Breadth / Cross-Section Approximations

Computes market breadth features using cross-section of ETFs:
- Fraction of symbols with positive returns
- Fraction above moving averages
- Cross-section return mean and std (dispersion)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


def compute_breadth_features(
    data: Dict[str, pd.DataFrame],
    symbols: Optional[List[str]] = None,
    return_horizon: int = 5,
    ma_window: int = 20
) -> Dict[str, float]:
    """
    Compute market breadth features from cross-section of ETFs.
    
    Since we don't have direct access to breadth metrics like $ADD/$TICK,
    this approximates breadth using a basket of ETFs.
    
    Args:
        data: Dict mapping symbol -> OHLCV DataFrame
        symbols: List of symbols to include (default: all in data)
        return_horizon: Horizon for return calculation (bars)
        ma_window: Moving average window for price comparison
        
    Returns:
        Dict of breadth features:
        
        Direction-Based:
        - breadth_up_frac_short: Fraction with positive short-term return
        - breadth_up_frac_long: Fraction with positive long-term return
        - breadth_above_ma_frac: Fraction with price above MA
        
        Cross-Section Stats:
        - xsec_ret_mean: Cross-section mean return
        - xsec_ret_std: Cross-section return std (dispersion)
        - xsec_ret_skew: Cross-section return skewness
        
        Derived:
        - breadth_momentum: Direction of breadth change
        - breadth_divergence: SPY return vs breadth divergence
    """
    features = {
        # Direction-based
        'breadth_up_frac_short': 0.5,
        'breadth_up_frac_long': 0.5,
        'breadth_above_ma_frac': 0.5,
        
        # Cross-section stats
        'xsec_ret_mean': 0.0,
        'xsec_ret_std': 0.0,
        'xsec_ret_skew': 0.0,
        
        # Derived
        'breadth_momentum': 0.0,
        'breadth_divergence': 0.0,
    }
    
    symbols = symbols or list(data.keys())
    if not symbols:
        return features
    
    # Collect returns and MA comparisons across symbols
    short_returns = []  # return_horizon bars
    long_returns = []   # ma_window bars
    above_ma = []
    current_prices = {}
    
    for symbol in symbols:
        if symbol not in data or data[symbol].empty:
            continue
        
        df = data[symbol]
        
        # Get close column
        close_col = 'close' if 'close' in df.columns else 'Close'
        if close_col not in df.columns:
            continue
        
        close = df[close_col].values
        
        if len(close) < max(return_horizon + 1, ma_window + 1):
            continue
        
        try:
            # Short-term return
            short_ret = (close[-1] - close[-return_horizon - 1]) / (close[-return_horizon - 1] + 1e-10)
            short_returns.append(short_ret)
            
            # Long-term return (using ma_window as horizon)
            long_ret = (close[-1] - close[-ma_window - 1]) / (close[-ma_window - 1] + 1e-10)
            long_returns.append(long_ret)
            
            # Price vs MA
            ma = np.mean(close[-ma_window:])
            above_ma.append(1.0 if close[-1] > ma else 0.0)
            
            # Store current price
            current_prices[symbol] = close[-1]
            
        except Exception as e:
            logger.debug(f"Error processing {symbol} for breadth: {e}")
            continue
    
    n_symbols = len(short_returns)
    
    if n_symbols == 0:
        return features
    
    try:
        # =====================================================================
        # DIRECTION-BASED BREADTH
        # =====================================================================
        
        # Fraction with positive short-term return
        features['breadth_up_frac_short'] = sum(1 for r in short_returns if r > 0) / n_symbols
        
        # Fraction with positive long-term return
        features['breadth_up_frac_long'] = sum(1 for r in long_returns if r > 0) / n_symbols
        
        # Fraction above moving average
        features['breadth_above_ma_frac'] = sum(above_ma) / n_symbols
        
        # =====================================================================
        # CROSS-SECTION STATISTICS
        # =====================================================================
        
        # Mean return
        features['xsec_ret_mean'] = float(np.mean(short_returns))
        
        # Std of returns (dispersion indicator)
        features['xsec_ret_std'] = float(np.std(short_returns))
        
        # Skewness
        if n_symbols >= 3:
            mean = features['xsec_ret_mean']
            std = features['xsec_ret_std']
            if std > 1e-10:
                skew = np.mean([(r - mean) ** 3 for r in short_returns]) / (std ** 3)
                features['xsec_ret_skew'] = float(np.clip(skew, -3, 3))
        
        # =====================================================================
        # DERIVED FEATURES
        # =====================================================================
        
        # Breadth momentum: compare short vs long breadth
        features['breadth_momentum'] = features['breadth_up_frac_short'] - features['breadth_up_frac_long']
        
        # Breadth divergence: compare SPY return to cross-section
        if 'SPY' in current_prices and 'SPY' in data and not data['SPY'].empty:
            spy_close = data['SPY']['close' if 'close' in data['SPY'].columns else 'Close'].values
            if len(spy_close) > return_horizon:
                spy_ret = (spy_close[-1] - spy_close[-return_horizon - 1]) / (spy_close[-return_horizon - 1] + 1e-10)
                # Divergence: SPY outperforming average = positive, underperforming = negative
                features['breadth_divergence'] = spy_ret - features['xsec_ret_mean']
        
    except Exception as e:
        logger.warning(f"Error computing breadth features: {e}")
    
    return features


def compute_sector_breadth(
    data: Dict[str, pd.DataFrame],
    sector_symbols: Optional[Dict[str, List[str]]] = None,
    return_horizon: int = 5
) -> Dict[str, float]:
    """
    Compute sector-specific breadth features.
    
    Args:
        data: Dict mapping symbol -> OHLCV DataFrame
        sector_symbols: Dict mapping sector name -> list of symbols
        return_horizon: Horizon for return calculation
        
    Returns:
        Dict of sector breadth features
    """
    # Default sector mapping
    if sector_symbols is None:
        sector_symbols = {
            'tech': ['QQQ', 'XLK'],
            'finance': ['XLF'],
            'energy': ['XLE'],
            'defensive': ['XLU', 'XLP'],
            'cyclical': ['XLY', 'IWM'],
        }
    
    features = {}
    sector_returns = {}
    
    for sector, symbols in sector_symbols.items():
        returns = []
        
        for symbol in symbols:
            if symbol not in data or data[symbol].empty:
                continue
            
            df = data[symbol]
            close_col = 'close' if 'close' in df.columns else 'Close'
            
            if close_col not in df.columns:
                continue
            
            close = df[close_col].values
            
            if len(close) > return_horizon:
                ret = (close[-1] - close[-return_horizon - 1]) / (close[-return_horizon - 1] + 1e-10)
                returns.append(ret)
        
        if returns:
            sector_ret = float(np.mean(returns))
            sector_returns[sector] = sector_ret
            features[f'sector_{sector}_ret'] = sector_ret
            features[f'sector_{sector}_up'] = 1.0 if sector_ret > 0 else 0.0
    
    # Sector rotation indicators
    if 'tech' in sector_returns and 'defensive' in sector_returns:
        # Tech vs defensive spread (risk-on indicator)
        features['sector_risk_on'] = sector_returns['tech'] - sector_returns['defensive']
    
    if 'cyclical' in sector_returns and 'defensive' in sector_returns:
        # Cyclical vs defensive spread
        features['sector_cyclical_spread'] = sector_returns['cyclical'] - sector_returns['defensive']
    
    return features


# Default symbols for breadth computation
BREADTH_SYMBOLS = [
    'SPY',  # S&P 500
    'QQQ',  # Nasdaq-100
    'IWM',  # Russell 2000
    'DIA',  # Dow Jones
    'XLF',  # Financials
    'XLK',  # Technology
    'XLE',  # Energy
    'XLU',  # Utilities
    'XLY',  # Consumer Discretionary
    'XLP',  # Consumer Staples
]


if __name__ == "__main__":
    """Test breadth features with mock data."""
    import logging
    
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 70)
    print("BREADTH FEATURE TEST")
    print("=" * 70)
    
    # Create mock data
    np.random.seed(42)
    n_bars = 100
    
    data = {}
    base_trend = np.cumsum(np.random.randn(n_bars) * 0.002)
    
    for symbol in BREADTH_SYMBOLS:
        # Add symbol-specific noise
        noise = np.random.randn(n_bars) * 0.005
        prices = 100 * np.exp(base_trend + noise)
        
        df = pd.DataFrame({
            'open': prices * (1 + np.random.randn(n_bars) * 0.001),
            'high': prices * (1 + np.abs(np.random.randn(n_bars) * 0.002)),
            'low': prices * (1 - np.abs(np.random.randn(n_bars) * 0.002)),
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, n_bars),
        })
        data[symbol] = df
    
    # Compute features
    features = compute_breadth_features(data, symbols=BREADTH_SYMBOLS)
    
    print(f"\nSymbols: {len(data)}")
    print("\nBreadth Features:")
    for k, v in sorted(features.items()):
        print(f"  {k}: {v:.4f}")
    
    # Sector features
    sector_features = compute_sector_breadth(data)
    print("\nSector Features:")
    for k, v in sorted(sector_features.items()):
        print(f"  {k}: {v:.4f}")
    
    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)




