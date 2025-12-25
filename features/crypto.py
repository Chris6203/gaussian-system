#!/usr/bin/env python3
"""
Crypto Risk-On/Off Features

Computes crypto-based risk sentiment features:
- BTC and ETH returns and volatility
- Crypto momentum indicators
- Crypto-equity correlation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


def compute_crypto_features(
    data: Dict[str, pd.DataFrame],
    return_horizons: List[int] = [1, 5, 24, 168],  # 1h, 5h, 24h, 1w (if hourly)
    vol_windows: List[int] = [24, 168]
) -> Dict[str, float]:
    """
    Compute crypto risk-on/off features from BTC and ETH data.
    
    Crypto assets often lead equity risk sentiment:
    - Strong crypto = risk-on environment
    - Weak crypto = risk-off environment
    
    Args:
        data: Dict mapping symbol -> OHLCV DataFrame
              Expected symbols: BTC-USD, ETH-USD (or BTCUSD, ETHUSD)
        return_horizons: Horizons for return calculation (in bars)
        vol_windows: Windows for volatility calculation
        
    Returns:
        Dict of crypto features:
        
        BTC Features:
        - btc_ret_{h}: BTC return at horizon h
        - btc_vol_{w}: BTC realized volatility
        - btc_ma_ratio: BTC short/long MA ratio
        - btc_zscore: BTC price z-score
        
        ETH Features (if available):
        - eth_ret_{h}: ETH return at horizon h
        - eth_btc_ratio: ETH/BTC relative strength
        
        Risk Sentiment:
        - crypto_risk_on: Combined crypto risk-on indicator
        - crypto_momentum: Crypto momentum score
        - crypto_vol_regime: High/low vol regime indicator
    """
    features = {
        # BTC
        'btc_ret_1': 0.0,
        'btc_ret_5': 0.0,
        'btc_ret_24': 0.0,
        'btc_vol_24': 0.0,
        'btc_ma_ratio': 0.0,
        'btc_zscore': 0.0,
        
        # ETH
        'eth_ret_5': 0.0,
        'eth_btc_ratio': 0.0,
        
        # Risk sentiment
        'crypto_risk_on': 0.0,
        'crypto_momentum': 0.0,
        'crypto_vol_regime': 0.0,
    }
    
    # Find BTC symbol (various formats)
    btc_symbol = None
    for sym in ['BTC-USD', 'BTCUSD', 'BTC', 'X:BTCUSD']:
        if sym in data and not data[sym].empty:
            btc_symbol = sym
            break
    
    # Find ETH symbol
    eth_symbol = None
    for sym in ['ETH-USD', 'ETHUSD', 'ETH', 'X:ETHUSD']:
        if sym in data and not data[sym].empty:
            eth_symbol = sym
            break
    
    # Helper functions
    def get_close_series(symbol: str) -> Optional[np.ndarray]:
        if symbol not in data or data[symbol].empty:
            return None
        
        df = data[symbol]
        close_col = 'close' if 'close' in df.columns else 'Close'
        
        if close_col not in df.columns:
            return None
        
        return df[close_col].values
    
    def compute_return(close: np.ndarray, horizon: int) -> float:
        if len(close) <= horizon:
            return 0.0
        return (close[-1] - close[-horizon - 1]) / (close[-horizon - 1] + 1e-10)
    
    def compute_volatility(close: np.ndarray, window: int) -> float:
        if len(close) < window + 1:
            return 0.0
        returns = np.diff(close[-window - 1:]) / close[-window - 1:-1]
        return float(np.std(returns))
    
    try:
        # =====================================================================
        # BTC FEATURES
        # =====================================================================
        
        if btc_symbol:
            btc_close = get_close_series(btc_symbol)
            
            if btc_close is not None and len(btc_close) > 1:
                # Returns at various horizons
                for h in return_horizons:
                    if h <= len(btc_close):
                        key = f'btc_ret_{h}'
                        if key in features:
                            features[key] = compute_return(btc_close, h)
                
                # Volatility
                for w in vol_windows:
                    if w <= len(btc_close):
                        key = f'btc_vol_{w}'
                        if key in features:
                            features[key] = compute_volatility(btc_close, w)
                
                # MA ratio (short/long)
                if len(btc_close) >= 50:
                    ma_short = np.mean(btc_close[-10:])
                    ma_long = np.mean(btc_close[-50:])
                    features['btc_ma_ratio'] = (ma_short / ma_long) - 1.0
                
                # Z-score
                if len(btc_close) >= 20:
                    mean = np.mean(btc_close[-20:])
                    std = np.std(btc_close[-20:])
                    if std > 0:
                        features['btc_zscore'] = (btc_close[-1] - mean) / std
        
        # =====================================================================
        # ETH FEATURES
        # =====================================================================
        
        if eth_symbol:
            eth_close = get_close_series(eth_symbol)
            
            if eth_close is not None and len(eth_close) > 5:
                # ETH return
                features['eth_ret_5'] = compute_return(eth_close, 5)
                
                # ETH/BTC relative strength
                if btc_symbol and btc_close is not None and len(btc_close) > 5:
                    eth_ret = compute_return(eth_close, 5)
                    btc_ret = compute_return(btc_close, 5)
                    features['eth_btc_ratio'] = eth_ret - btc_ret
        
        # =====================================================================
        # RISK SENTIMENT INDICATORS
        # =====================================================================
        
        # Crypto risk-on: combination of momentum and volatility
        btc_ret_5 = features.get('btc_ret_5', 0)
        btc_ma_ratio = features.get('btc_ma_ratio', 0)
        
        # Risk-on = positive momentum + trending up
        features['crypto_risk_on'] = float(np.clip(
            (btc_ret_5 * 10) + (btc_ma_ratio * 5), -1, 1
        ))
        
        # Momentum score: combine short and longer-term returns
        btc_ret_1 = features.get('btc_ret_1', 0)
        btc_ret_24 = features.get('btc_ret_24', 0)
        
        features['crypto_momentum'] = float(np.clip(
            (btc_ret_1 * 20) + (btc_ret_5 * 10) + (btc_ret_24 * 5), -2, 2
        ))
        
        # Vol regime: high vol = elevated risk, low vol = complacent
        btc_vol = features.get('btc_vol_24', 0)
        
        # Typical BTC daily vol is 2-5%, annualized ~40-80%
        if btc_vol > 0.05:  # High vol
            features['crypto_vol_regime'] = 1.0
        elif btc_vol > 0.02:  # Normal vol
            features['crypto_vol_regime'] = 0.0
        else:  # Low vol
            features['crypto_vol_regime'] = -1.0
        
    except Exception as e:
        logger.warning(f"Error computing crypto features: {e}")
    
    return features


def compute_crypto_equity_correlation(
    crypto_data: Dict[str, pd.DataFrame],
    equity_data: Dict[str, pd.DataFrame],
    correlation_window: int = 20
) -> Dict[str, float]:
    """
    Compute correlation between crypto and equity returns.
    
    Args:
        crypto_data: Dict with crypto DataFrames (BTC-USD, etc.)
        equity_data: Dict with equity DataFrames (SPY, QQQ)
        correlation_window: Rolling window for correlation
        
    Returns:
        Dict with correlation features
    """
    features = {
        'btc_spy_corr': 0.0,
        'btc_qqq_corr': 0.0,
    }
    
    def get_returns(data: Dict, symbol_options: List[str], length: int) -> Optional[np.ndarray]:
        for sym in symbol_options:
            if sym in data and not data[sym].empty:
                df = data[sym]
                close_col = 'close' if 'close' in df.columns else 'Close'
                if close_col in df.columns:
                    close = df[close_col].values
                    if len(close) >= length + 1:
                        return np.diff(close[-length - 1:]) / close[-length - 1:-1]
        return None
    
    btc_returns = get_returns(crypto_data, ['BTC-USD', 'BTCUSD', 'BTC'], correlation_window)
    spy_returns = get_returns(equity_data, ['SPY'], correlation_window)
    qqq_returns = get_returns(equity_data, ['QQQ'], correlation_window)
    
    if btc_returns is not None and spy_returns is not None:
        if len(btc_returns) == len(spy_returns):
            corr = np.corrcoef(btc_returns, spy_returns)[0, 1]
            if not np.isnan(corr):
                features['btc_spy_corr'] = float(corr)
    
    if btc_returns is not None and qqq_returns is not None:
        if len(btc_returns) == len(qqq_returns):
            corr = np.corrcoef(btc_returns, qqq_returns)[0, 1]
            if not np.isnan(corr):
                features['btc_qqq_corr'] = float(corr)
    
    return features


# Symbols used for crypto features
CRYPTO_SYMBOLS = {
    'BTC-USD': 'Bitcoin (primary crypto risk indicator)',
    'ETH-USD': 'Ethereum (secondary crypto indicator)',
}


if __name__ == "__main__":
    """Test crypto features with mock data."""
    import logging
    
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 70)
    print("CRYPTO FEATURE TEST")
    print("=" * 70)
    
    # Create mock data
    np.random.seed(42)
    n_bars = 200
    
    data = {}
    
    # Mock BTC (volatile, trending)
    btc_trend = np.cumsum(np.random.randn(n_bars) * 0.02 + 0.001)  # Slight upward bias
    btc_prices = 50000 * np.exp(btc_trend)
    data['BTC-USD'] = pd.DataFrame({
        'open': btc_prices * (1 + np.random.randn(n_bars) * 0.005),
        'high': btc_prices * (1 + np.abs(np.random.randn(n_bars) * 0.02)),
        'low': btc_prices * (1 - np.abs(np.random.randn(n_bars) * 0.02)),
        'close': btc_prices,
        'volume': np.random.randint(10000, 100000, n_bars),
    })
    
    # Mock ETH (correlated with BTC but more volatile)
    eth_trend = btc_trend * 1.2 + np.cumsum(np.random.randn(n_bars) * 0.01)
    eth_prices = 3000 * np.exp(eth_trend)
    data['ETH-USD'] = pd.DataFrame({
        'open': eth_prices * (1 + np.random.randn(n_bars) * 0.005),
        'high': eth_prices * (1 + np.abs(np.random.randn(n_bars) * 0.025)),
        'low': eth_prices * (1 - np.abs(np.random.randn(n_bars) * 0.025)),
        'close': eth_prices,
        'volume': np.random.randint(5000, 50000, n_bars),
    })
    
    # Mock SPY for correlation
    spy_trend = np.cumsum(np.random.randn(n_bars) * 0.005 + btc_trend * 0.1)
    spy_prices = 450 * np.exp(spy_trend)
    equity_data = {
        'SPY': pd.DataFrame({
            'close': spy_prices,
            'volume': np.random.randint(1000000, 10000000, n_bars),
        })
    }
    
    # Compute features
    features = compute_crypto_features(data)
    
    print(f"\nCrypto Symbols: {list(data.keys())}")
    print("\nCrypto Features:")
    for k, v in sorted(features.items()):
        print(f"  {k}: {v:.4f}")
    
    # Correlation features
    corr_features = compute_crypto_equity_correlation(data, equity_data)
    print("\nCorrelation Features:")
    for k, v in sorted(corr_features.items()):
        print(f"  {k}: {v:.4f}")
    
    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)




