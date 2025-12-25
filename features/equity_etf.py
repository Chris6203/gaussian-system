#!/usr/bin/env python3
"""
Cross-Asset Equity & ETF Features

Computes features from multiple equity/ETF symbols for cross-asset analysis:
- Returns at multiple horizons
- Rolling realized volatility
- Rolling z-scores
- Moving average ratios
- Relative volume
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# CORE FEATURE COMPUTATION HELPERS
# =============================================================================

def compute_returns(
    df: pd.DataFrame,
    col: str = 'close',
    horizons: List[int] = [1, 5, 15, 60]
) -> pd.DataFrame:
    """
    Compute simple and log returns at multiple horizons.
    
    Args:
        df: DataFrame with OHLCV data
        col: Column to compute returns from
        horizons: List of lookback periods (in bars)
        
    Returns:
        DataFrame with return columns
    """
    result = pd.DataFrame(index=df.index)
    
    # Normalize column name
    close_col = col if col in df.columns else col.title()
    if close_col not in df.columns:
        close_col = 'Close' if 'Close' in df.columns else 'close'
    
    if close_col not in df.columns:
        return result
    
    prices = df[close_col].values
    
    for h in horizons:
        if len(prices) > h:
            # Simple returns
            simple_ret = np.zeros(len(prices))
            simple_ret[h:] = (prices[h:] - prices[:-h]) / (prices[:-h] + 1e-10)
            result[f'ret_{h}'] = simple_ret
            
            # Log returns
            log_ret = np.zeros(len(prices))
            log_ret[h:] = np.log(prices[h:] / (prices[:-h] + 1e-10))
            result[f'log_ret_{h}'] = log_ret
    
    return result


def compute_rolling_volatility(
    df: pd.DataFrame,
    col: str = 'close',
    windows: List[int] = [20, 60, 120]
) -> pd.DataFrame:
    """
    Compute rolling realized volatility at multiple windows.
    
    Args:
        df: DataFrame with OHLCV data
        col: Column to compute volatility from
        windows: List of rolling window sizes
        
    Returns:
        DataFrame with volatility columns
    """
    result = pd.DataFrame(index=df.index)
    
    # Normalize column name
    close_col = col if col in df.columns else col.title()
    if close_col not in df.columns:
        close_col = 'Close' if 'Close' in df.columns else 'close'
    
    if close_col not in df.columns:
        return result
    
    # Calculate returns for volatility
    prices = df[close_col]
    returns = prices.pct_change()
    
    for w in windows:
        if len(df) >= w:
            result[f'rvol_{w}'] = returns.rolling(window=w, min_periods=max(1, w//2)).std()
    
    return result


def compute_zscore(
    df: pd.DataFrame,
    col: str = 'close',
    windows: List[int] = [20, 60]
) -> pd.DataFrame:
    """
    Compute rolling z-score of prices and returns.
    
    Args:
        df: DataFrame with OHLCV data
        col: Column to compute z-score from
        windows: List of rolling window sizes
        
    Returns:
        DataFrame with z-score columns
    """
    result = pd.DataFrame(index=df.index)
    
    # Normalize column name
    close_col = col if col in df.columns else col.title()
    if close_col not in df.columns:
        close_col = 'Close' if 'Close' in df.columns else 'close'
    
    if close_col not in df.columns:
        return result
    
    prices = df[close_col]
    returns = prices.pct_change()
    
    for w in windows:
        if len(df) >= w:
            # Price z-score
            roll_mean = prices.rolling(window=w, min_periods=max(1, w//2)).mean()
            roll_std = prices.rolling(window=w, min_periods=max(1, w//2)).std()
            result[f'price_zscore_{w}'] = (prices - roll_mean) / (roll_std + 1e-10)
            
            # Returns z-score
            ret_mean = returns.rolling(window=w, min_periods=max(1, w//2)).mean()
            ret_std = returns.rolling(window=w, min_periods=max(1, w//2)).std()
            result[f'ret_zscore_{w}'] = (returns - ret_mean) / (ret_std + 1e-10)
    
    return result


def compute_ma_features(
    df: pd.DataFrame,
    col: str = 'close',
    short_windows: List[int] = [10, 20],
    long_windows: List[int] = [50, 200]
) -> pd.DataFrame:
    """
    Compute moving average features and ratios.
    
    Args:
        df: DataFrame with OHLCV data
        col: Column to compute MAs from
        short_windows: Short-term MA windows
        long_windows: Long-term MA windows
        
    Returns:
        DataFrame with MA features
    """
    result = pd.DataFrame(index=df.index)
    
    # Normalize column name
    close_col = col if col in df.columns else col.title()
    if close_col not in df.columns:
        close_col = 'Close' if 'Close' in df.columns else 'close'
    
    if close_col not in df.columns:
        return result
    
    prices = df[close_col]
    
    # Compute all MAs
    mas = {}
    all_windows = set(short_windows) | set(long_windows)
    for w in all_windows:
        if len(df) >= w:
            mas[w] = prices.rolling(window=w, min_periods=max(1, w//2)).mean()
    
    # Price vs MA
    for w in all_windows:
        if w in mas:
            result[f'price_vs_ma{w}'] = (prices - mas[w]) / (mas[w] + 1e-10)
    
    # Short vs long MA ratios
    for sw in short_windows:
        for lw in long_windows:
            if sw in mas and lw in mas and sw < lw:
                result[f'ma{sw}_over_ma{lw}'] = mas[sw] / (mas[lw] + 1e-10) - 1.0
    
    return result


def compute_relative_volume(
    df: pd.DataFrame,
    windows: List[int] = [10, 20, 60]
) -> pd.DataFrame:
    """
    Compute relative volume (current volume / average volume).
    
    Args:
        df: DataFrame with volume column
        windows: Rolling windows for average volume
        
    Returns:
        DataFrame with relative volume columns
    """
    result = pd.DataFrame(index=df.index)
    
    vol_col = 'volume' if 'volume' in df.columns else 'Volume'
    if vol_col not in df.columns:
        return result
    
    volume = df[vol_col]
    
    for w in windows:
        if len(df) >= w:
            avg_vol = volume.rolling(window=w, min_periods=max(1, w//2)).mean()
            result[f'rel_vol_{w}'] = volume / (avg_vol + 1.0)  # +1 to avoid div by zero
    
    return result


# =============================================================================
# MAIN FEATURE COMPUTATION FUNCTION
# =============================================================================

def compute_equity_etf_features(
    data: Dict[str, pd.DataFrame],
    primary_symbol: str = 'SPY',
    symbols: Optional[List[str]] = None,
    return_horizons: List[int] = [1, 5, 15, 60],
    vol_windows: List[int] = [20, 60, 120],
    ma_short: List[int] = [10, 20],
    ma_long: List[int] = [50, 200]
) -> pd.DataFrame:
    """
    Compute cross-asset equity/ETF features for all symbols.
    
    Args:
        data: Dict mapping symbol -> OHLCV DataFrame
        primary_symbol: Primary trading symbol (features aligned to this)
        symbols: List of symbols to compute features for (default: all in data)
        return_horizons: Horizons for return computation
        vol_windows: Windows for volatility computation
        ma_short: Short MA windows
        ma_long: Long MA windows
        
    Returns:
        DataFrame with all equity/ETF features, indexed by timestamp
        Columns prefixed by symbol (e.g., SPY_ret_1, QQQ_rvol_20)
    """
    symbols = symbols or list(data.keys())
    
    # Get primary index
    if primary_symbol not in data or data[primary_symbol].empty:
        logger.warning(f"Primary symbol {primary_symbol} not in data")
        return pd.DataFrame()
    
    primary_df = data[primary_symbol]
    result = pd.DataFrame(index=primary_df.index)
    
    for symbol in symbols:
        if symbol not in data or data[symbol].empty:
            logger.debug(f"No data for {symbol}")
            continue
        
        df = data[symbol]
        prefix = symbol.replace('^', '').replace('-', '_')
        
        try:
            # Returns
            returns_df = compute_returns(df, horizons=return_horizons)
            for col in returns_df.columns:
                result[f'{prefix}_{col}'] = returns_df[col]
            
            # Volatility
            vol_df = compute_rolling_volatility(df, windows=vol_windows)
            for col in vol_df.columns:
                result[f'{prefix}_{col}'] = vol_df[col]
            
            # Z-scores
            zscore_df = compute_zscore(df, windows=vol_windows[:2])  # Use first 2 windows
            for col in zscore_df.columns:
                result[f'{prefix}_{col}'] = zscore_df[col]
            
            # MA features
            ma_df = compute_ma_features(df, short_windows=ma_short, long_windows=ma_long)
            for col in ma_df.columns:
                result[f'{prefix}_{col}'] = ma_df[col]
            
            # Relative volume
            rvol_df = compute_relative_volume(df)
            for col in rvol_df.columns:
                result[f'{prefix}_{col}'] = rvol_df[col]
            
        except Exception as e:
            logger.warning(f"Error computing features for {symbol}: {e}")
            continue
    
    # Align to primary symbol's index
    result = result.reindex(primary_df.index)
    
    # Fill NaNs with 0 for numeric stability
    result = result.fillna(0.0)
    
    return result


def get_latest_equity_etf_features(
    data: Dict[str, pd.DataFrame],
    primary_symbol: str = 'SPY',
    **kwargs
) -> Dict[str, float]:
    """
    Get the latest (most recent) equity/ETF features as a dict.
    
    Useful for live inference where you need a single feature row.
    
    Args:
        data: Dict mapping symbol -> OHLCV DataFrame
        primary_symbol: Primary trading symbol
        **kwargs: Additional arguments passed to compute_equity_etf_features
        
    Returns:
        Dict mapping feature name -> value
    """
    features_df = compute_equity_etf_features(data, primary_symbol, **kwargs)
    
    if features_df.empty:
        return {}
    
    latest = features_df.iloc[-1]
    return latest.to_dict()


# =============================================================================
# SYMBOL CONFIGURATION
# =============================================================================

# Default symbols for equity/ETF feature computation
EQUITY_ETF_SYMBOLS = {
    # Core indices/ETFs
    'SPY': 'S&P 500 ETF (primary)',
    'QQQ': 'Nasdaq-100 ETF (tech)',
    'IWM': 'Russell 2000 ETF (small cap)',
    'DIA': 'Dow Jones ETF',
    
    # Sector ETFs
    'XLF': 'Financial Select Sector',
    'XLK': 'Technology Select Sector',
    'XLE': 'Energy Select Sector',
    'XLU': 'Utilities Select Sector',
    'XLY': 'Consumer Discretionary',
    'XLP': 'Consumer Staples',
    
    # Credit/Bond proxies
    'HYG': 'High Yield Corporate Bond ETF',
    'LQD': 'Investment Grade Corporate Bond ETF',
}


if __name__ == "__main__":
    """Test equity/ETF feature computation."""
    import yfinance as yf
    
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 70)
    print("EQUITY/ETF FEATURE TEST")
    print("=" * 70)
    
    # Fetch test data
    symbols = ['SPY', 'QQQ', 'IWM', 'XLF']
    data = {}
    
    for sym in symbols:
        print(f"Fetching {sym}...")
        ticker = yf.Ticker(sym)
        df = ticker.history(period='1mo', interval='1h')
        if not df.empty:
            df.columns = [c.lower() for c in df.columns]
            data[sym] = df
            print(f"  Got {len(df)} bars")
    
    # Compute features
    print("\nComputing features...")
    features = compute_equity_etf_features(data, primary_symbol='SPY')
    
    print(f"\nGenerated {len(features.columns)} features:")
    for col in sorted(features.columns)[:20]:
        print(f"  {col}")
    print(f"  ... and {len(features.columns) - 20} more" if len(features.columns) > 20 else "")
    
    print(f"\nLatest features:")
    latest = get_latest_equity_etf_features(data, primary_symbol='SPY')
    for k, v in list(latest.items())[:10]:
        print(f"  {k}: {v:.6f}")
    
    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)




