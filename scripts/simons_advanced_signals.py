#!/usr/bin/env python3
"""
Simons-Style Advanced Signal Analysis

Based on Renaissance Technologies' known approaches:
1. Cross-asset correlations (SPY/QQQ divergence, VIX term structure)
2. Market microstructure (volume patterns, price momentum)
3. Statistical anomalies (autocorrelation, regime detection)
4. Time-based patterns (intraday seasonality, day-of-week)
5. Options flow (put/call ratios, skew)

Philosophy: Many weak signals combined = strong edge
"""

import os
import sys
import numpy as np
import pandas as pd
import sqlite3
from scipy import stats
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_multi_asset_data(start_date='2025-06-01', limit=20000):
    """Load SPY, QQQ, VIX data for cross-asset analysis."""
    conn = sqlite3.connect('data/db/historical.db')

    assets = {}
    for symbol in ['SPY', 'QQQ', 'VIX', '^VIX']:
        df = pd.read_sql_query("""
            SELECT timestamp,
                   open_price as open,
                   high_price as high,
                   low_price as low,
                   close_price as close,
                   volume
            FROM historical_data
            WHERE symbol = ?
            AND timestamp >= ?
            ORDER BY timestamp
            LIMIT ?
        """, conn, params=(symbol, start_date, limit))

        if len(df) > 0:
            df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601', utc=True)
            df = df.set_index('timestamp')
            df = df.sort_index()  # Ensure monotonic index
            df = df[~df.index.duplicated(keep='first')]  # Remove duplicates
            assets[symbol] = df

    conn.close()

    # Merge VIX variants
    if '^VIX' in assets and 'VIX' not in assets:
        assets['VIX'] = assets['^VIX']
    elif '^VIX' in assets and 'VIX' in assets:
        # Combine both
        assets['VIX'] = pd.concat([assets['VIX'], assets['^VIX']]).drop_duplicates()

    return assets


def calculate_simons_features(assets: dict) -> pd.DataFrame:
    """
    Calculate Simons-style features based on RenTech approaches.
    """
    if 'SPY' not in assets:
        raise ValueError("SPY data required")

    spy = assets['SPY'].copy()
    features = pd.DataFrame(index=spy.index)

    # =========================================
    # 1. CROSS-ASSET DIVERGENCE (RenTech Core)
    # =========================================

    if 'QQQ' in assets:
        qqq = assets['QQQ'].reindex(spy.index, method='ffill')

        # SPY-QQQ spread (tech vs broad market)
        spy_ret = spy['close'].pct_change(5)
        qqq_ret = qqq['close'].pct_change(5)
        features['spy_qqq_divergence'] = spy_ret - qqq_ret

        # Rolling correlation breakdown (regime signal)
        features['spy_qqq_corr_20'] = spy_ret.rolling(20).corr(qqq_ret)
        features['spy_qqq_corr_60'] = spy_ret.rolling(60).corr(qqq_ret)
        features['corr_breakdown'] = features['spy_qqq_corr_20'] - features['spy_qqq_corr_60']

    if 'VIX' in assets:
        vix = assets['VIX'].reindex(spy.index, method='ffill')

        # VIX level and changes
        features['vix'] = vix['close']
        features['vix_change_5m'] = vix['close'].pct_change(5)
        features['vix_change_15m'] = vix['close'].pct_change(15)

        # VIX regime (high vol vs low vol)
        features['vix_regime'] = (vix['close'] > 20).astype(int)
        features['vix_sma_ratio'] = vix['close'] / vix['close'].rolling(20).mean()

        # SPY-VIX correlation (should be negative, breakdown = regime change)
        spy_ret = spy['close'].pct_change(5)
        vix_ret = vix['close'].pct_change(5)
        features['spy_vix_corr'] = spy_ret.rolling(20).corr(vix_ret)

    # =========================================
    # 2. MARKET MICROSTRUCTURE
    # =========================================

    # Volume analysis
    features['volume'] = spy['volume']
    features['volume_sma_20'] = spy['volume'].rolling(20).mean()
    features['volume_ratio'] = spy['volume'] / features['volume_sma_20']
    features['volume_trend'] = spy['volume'].rolling(5).mean() / spy['volume'].rolling(20).mean()

    # Price-volume divergence (smart money signal)
    price_change = spy['close'].pct_change(5)
    volume_change = spy['volume'].pct_change(5)
    features['price_volume_divergence'] = price_change - volume_change.clip(-0.5, 0.5)

    # Range analysis (volatility proxy)
    features['range_pct'] = (spy['high'] - spy['low']) / spy['close']
    features['range_sma'] = features['range_pct'].rolling(20).mean()
    features['range_expansion'] = features['range_pct'] / features['range_sma']

    # =========================================
    # 3. STATISTICAL ANOMALIES
    # =========================================

    # Autocorrelation (mean reversion signal)
    returns = spy['close'].pct_change(1)
    features['autocorr_1'] = returns.rolling(20).apply(lambda x: x.autocorr(lag=1), raw=False)
    features['autocorr_5'] = returns.rolling(20).apply(lambda x: x.autocorr(lag=5) if len(x) > 5 else 0, raw=False)

    # Skewness and kurtosis (tail risk)
    features['return_skew'] = returns.rolling(60).skew()
    features['return_kurtosis'] = returns.rolling(60).kurt()

    # Z-score of price (how many std from mean)
    features['price_zscore'] = (spy['close'] - spy['close'].rolling(20).mean()) / spy['close'].rolling(20).std()

    # Hurst exponent proxy (trending vs mean-reverting)
    # Simplified: variance ratio test
    ret_1 = spy['close'].pct_change(1)
    ret_5 = spy['close'].pct_change(5)
    var_1 = ret_1.rolling(60).var()
    var_5 = ret_5.rolling(60).var()
    features['variance_ratio'] = var_5 / (5 * var_1 + 1e-8)  # =1 random, <1 mean revert, >1 trend

    # =========================================
    # 4. TIME-BASED PATTERNS (RenTech Edge)
    # =========================================

    # Extract time components
    features['hour'] = spy.index.hour
    features['minute'] = spy.index.minute
    features['day_of_week'] = spy.index.dayofweek

    # Time of day signals
    features['is_open_30m'] = ((features['hour'] == 9) & (features['minute'] >= 30)) | \
                              ((features['hour'] == 10) & (features['minute'] < 0))
    features['is_close_30m'] = (features['hour'] == 15) & (features['minute'] >= 30)
    features['is_lunch'] = (features['hour'] >= 11) & (features['hour'] < 14)

    # Day of week effects
    features['is_monday'] = (features['day_of_week'] == 0).astype(int)
    features['is_friday'] = (features['day_of_week'] == 4).astype(int)

    # Time since open (minutes)
    features['mins_since_open'] = (features['hour'] - 9) * 60 + features['minute'] - 30
    features['mins_since_open'] = features['mins_since_open'].clip(0, 390)

    # =========================================
    # 5. MOMENTUM & MEAN REVERSION COMBINED
    # =========================================

    # Multi-timeframe momentum
    for period in [5, 15, 30, 60]:
        features[f'momentum_{period}m'] = spy['close'].pct_change(period)

    # Momentum divergence (short vs long term)
    features['momentum_divergence'] = features['momentum_5m'] - features['momentum_30m']

    # RSI
    delta = spy['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-8)
    features['rsi_14'] = 100 - (100 / (1 + rs))

    # Bollinger Band position
    bb_middle = spy['close'].rolling(20).mean()
    bb_std = spy['close'].rolling(20).std()
    features['bb_position'] = (spy['close'] - bb_middle) / (2 * bb_std + 1e-8)

    # =========================================
    # 6. OPTIONS-DERIVED SIGNALS (if available)
    # =========================================

    # Put/Call ratio proxy using VIX
    if 'VIX' in assets:
        # VIX spike = fear = puts being bought
        features['fear_signal'] = (features['vix'] > features['vix'].rolling(20).mean() +
                                   features['vix'].rolling(20).std()).astype(int)

        # VIX crush = complacency = calls dominating
        features['greed_signal'] = (features['vix'] < features['vix'].rolling(20).mean() -
                                    features['vix'].rolling(20).std()).astype(int)

    return features


def calculate_forward_returns(spy: pd.DataFrame, horizons=[15, 30, 60]) -> pd.DataFrame:
    """Calculate forward returns."""
    forward = pd.DataFrame(index=spy.index)
    for h in horizons:
        forward[f'fwd_return_{h}m'] = spy['close'].shift(-h) / spy['close'] - 1
    return forward


def analyze_simons_signals(features: pd.DataFrame, forward: pd.DataFrame) -> pd.DataFrame:
    """Analyze correlation of Simons features with forward returns."""
    results = []

    target = 'fwd_return_15m'
    if target not in forward.columns:
        return pd.DataFrame()

    for feat_col in features.columns:
        if feat_col in ['hour', 'minute', 'day_of_week', 'mins_since_open']:
            continue

        mask = ~(features[feat_col].isna() | forward[target].isna())
        x = features.loc[mask, feat_col].values
        y = forward.loc[mask, target].values

        if len(x) < 100:
            continue

        # Remove infinities
        valid = np.isfinite(x) & np.isfinite(y)
        x, y = x[valid], y[valid]

        if len(x) < 100:
            continue

        corr, p_value = stats.spearmanr(x, y)

        results.append({
            'feature': feat_col,
            'correlation': corr,
            'p_value': p_value,
            'n_samples': len(x),
            'significant': p_value < 0.01 and abs(corr) > 0.02
        })

    df = pd.DataFrame(results)
    df['abs_corr'] = df['correlation'].abs()
    return df.sort_values('abs_corr', ascending=False)


def main():
    print("="*70)
    print("SIMONS-STYLE ADVANCED SIGNAL ANALYSIS")
    print("="*70)
    print("\nBased on Renaissance Technologies approaches:")
    print("1. Cross-asset correlations (SPY/QQQ divergence)")
    print("2. Market microstructure (volume patterns)")
    print("3. Statistical anomalies (autocorrelation)")
    print("4. Time-based patterns (intraday seasonality)")
    print("5. Options-derived signals (VIX-based fear/greed)\n")

    # Load multi-asset data
    print("[1] Loading multi-asset data...")
    assets = load_multi_asset_data(limit=20000)
    print(f"    Loaded assets: {list(assets.keys())}")
    for symbol, df in assets.items():
        print(f"    {symbol}: {len(df)} bars")

    # Calculate features
    print("\n[2] Calculating Simons-style features...")
    features = calculate_simons_features(assets)
    print(f"    Generated {len(features.columns)} features")

    # Calculate forward returns
    print("\n[3] Calculating forward returns...")
    forward = calculate_forward_returns(assets['SPY'])

    # Analyze correlations
    print("\n[4] Analyzing signal correlations...")
    results = analyze_simons_signals(features, forward)

    # Display results
    print("\n" + "="*70)
    print("RESULTS: Feature Correlations with 15-min Forward Returns")
    print("="*70)
    print(f"\n{'Feature':<30} {'Corr':>10} {'p-value':>12} {'Significant':>12}")
    print("-"*70)

    significant_count = 0
    for _, row in results.head(30).iterrows():
        sig = "✓ YES" if row['significant'] else "✗ no"
        if row['significant']:
            significant_count += 1
        print(f"{row['feature']:<30} {row['correlation']:>+10.4f} {row['p_value']:>12.2e} {sig:>12}")

    # Summary by category
    print("\n" + "="*70)
    print("ANALYSIS BY CATEGORY")
    print("="*70)

    categories = {
        'Cross-Asset': ['spy_qqq', 'spy_vix', 'corr_'],
        'Volume/Microstructure': ['volume', 'range_', 'price_volume'],
        'Statistical': ['autocorr', 'skew', 'kurtosis', 'zscore', 'variance'],
        'Time-Based': ['is_', 'hour', 'minute', 'mins_since', 'monday', 'friday'],
        'Momentum/MR': ['momentum', 'rsi', 'bb_position'],
        'VIX/Fear': ['vix', 'fear', 'greed']
    }

    for cat_name, patterns in categories.items():
        cat_features = results[results['feature'].apply(
            lambda x: any(p in x.lower() for p in patterns)
        )]
        if len(cat_features) > 0:
            avg_corr = cat_features['correlation'].abs().mean()
            max_corr = cat_features['correlation'].abs().max()
            sig_count = cat_features['significant'].sum()
            best_feat = cat_features.iloc[0]['feature'] if len(cat_features) > 0 else 'N/A'
            print(f"\n{cat_name}:")
            print(f"  Avg |corr|: {avg_corr:.4f}")
            print(f"  Max |corr|: {max_corr:.4f}")
            print(f"  Significant: {sig_count}/{len(cat_features)}")
            print(f"  Best: {best_feat}")

    # Simons' verdict
    print("\n" + "="*70)
    print("SIMONS' VERDICT")
    print("="*70)

    if significant_count >= 10:
        print("✓ MULTIPLE WEAK SIGNALS FOUND")
        print("  Renaissance approach: Combine many weak signals into strong edge")
        print("  Recommended: Ensemble model using top 10-15 features")
    elif significant_count >= 5:
        print("⚠ SOME SIGNALS FOUND")
        print("  Moderate predictive power exists")
        print("  Recommended: Focus on strongest signals, add more data")
    else:
        print("✗ FEW SIGNALS")
        print("  Limited predictive power at this timeframe")
        print("  Recommended: Try different timeframes or data sources")

    # Specific recommendations
    print("\n" + "="*70)
    print("RECOMMENDATIONS FOR IMPROVEMENT")
    print("="*70)

    # Find best positive and negative correlations
    pos_corr = results[results['correlation'] > 0].head(3)
    neg_corr = results[results['correlation'] < 0].head(3)

    print("\nBest POSITIVE correlations (high feature = market UP):")
    for _, row in pos_corr.iterrows():
        print(f"  {row['feature']}: {row['correlation']:+.4f}")

    print("\nBest NEGATIVE correlations (high feature = market DOWN):")
    for _, row in neg_corr.iterrows():
        print(f"  {row['feature']}: {row['correlation']:+.4f}")

    print("\nTrading implications:")
    print("  - Use negative correlations for MEAN REVERSION")
    print("  - Use positive correlations for TREND FOLLOWING")
    print("  - Combine both with regime detection (HMM)")

    return results


if __name__ == "__main__":
    results = main()
