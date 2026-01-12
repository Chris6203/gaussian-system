#!/usr/bin/env python3
"""
Simons Feature Correlation Analysis

Philosophy: Before building ANY model, verify there's a signal to learn.
If no feature correlates with forward returns, the model has nothing to learn.

This script:
1. Loads features from the feature pipeline
2. Calculates 15-minute forward returns
3. Computes correlation of each feature with forward returns
4. Identifies which features (if any) have predictive power
"""

import os
import sys
import numpy as np
import pandas as pd
import sqlite3
from scipy import stats
from datetime import datetime, timedelta

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_price_data(start_date='2025-06-01', limit=50000):
    """Load SPY price data."""
    conn = sqlite3.connect('data/db/historical.db')
    df = pd.read_sql_query("""
        SELECT timestamp,
               open_price as open,
               high_price as high,
               low_price as low,
               close_price as close,
               volume
        FROM historical_data
        WHERE symbol = 'SPY'
        AND timestamp >= ?
        ORDER BY timestamp
        LIMIT ?
    """, conn, params=(start_date, limit))
    conn.close()

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


def load_vix_data(start_date='2025-06-01', limit=50000):
    """Load VIX data."""
    conn = sqlite3.connect('data/db/historical.db')
    df = pd.read_sql_query("""
        SELECT timestamp, close_price as vix
        FROM historical_data
        WHERE symbol IN ('VIX', '^VIX')
        AND timestamp >= ?
        ORDER BY timestamp
        LIMIT ?
    """, conn, params=(start_date, limit))
    conn.close()

    if len(df) > 0:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate common technical features.
    These are the same features the model uses.
    """
    features = pd.DataFrame(index=df.index)

    # Price features
    features['close'] = df['close']
    features['return_1m'] = df['close'].pct_change(1)
    features['return_5m'] = df['close'].pct_change(5)
    features['return_15m'] = df['close'].pct_change(15)
    features['return_30m'] = df['close'].pct_change(30)
    features['return_60m'] = df['close'].pct_change(60)

    # Momentum
    features['momentum_5m'] = df['close'].diff(5) / df['close'].shift(5)
    features['momentum_15m'] = df['close'].diff(15) / df['close'].shift(15)
    features['momentum_30m'] = df['close'].diff(30) / df['close'].shift(30)

    # Moving averages
    features['sma_5'] = df['close'].rolling(5).mean()
    features['sma_20'] = df['close'].rolling(20).mean()
    features['sma_50'] = df['close'].rolling(50).mean()
    features['ma_cross_5_20'] = (features['sma_5'] - features['sma_20']) / features['sma_20']
    features['ma_cross_20_50'] = (features['sma_20'] - features['sma_50']) / features['sma_50']

    # Volatility
    features['volatility_5m'] = df['close'].rolling(5).std() / df['close']
    features['volatility_20m'] = df['close'].rolling(20).std() / df['close']
    features['volatility_60m'] = df['close'].rolling(60).std() / df['close']

    # Volume features
    features['volume'] = df['volume']
    features['volume_sma_20'] = df['volume'].rolling(20).mean()
    features['volume_ratio'] = df['volume'] / features['volume_sma_20']

    # High-Low range
    features['range_pct'] = (df['high'] - df['low']) / df['close']
    features['range_5m_avg'] = features['range_pct'].rolling(5).mean()

    # Price position within day's range
    features['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)

    # RSI-like feature (14-period)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-8)
    features['rsi_14'] = 100 - (100 / (1 + rs))

    # Bollinger Band position
    bb_middle = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    features['bb_position'] = (df['close'] - bb_middle) / (2 * bb_std + 1e-8)

    # MACD-like
    ema_12 = df['close'].ewm(span=12).mean()
    ema_26 = df['close'].ewm(span=26).mean()
    features['macd'] = (ema_12 - ema_26) / df['close']

    # Time features (if timestamp available)
    if 'timestamp' in df.columns:
        ts = pd.to_datetime(df['timestamp'])
        features['hour'] = ts.dt.hour
        features['minute'] = ts.dt.minute
        features['day_of_week'] = ts.dt.dayofweek
        # Morning (9-11), Midday (11-14), Afternoon (14-16)
        features['is_morning'] = ((ts.dt.hour >= 9) & (ts.dt.hour < 11)).astype(int)
        features['is_lunch'] = ((ts.dt.hour >= 11) & (ts.dt.hour < 14)).astype(int)
        features['is_afternoon'] = ((ts.dt.hour >= 14) & (ts.dt.hour < 16)).astype(int)

    return features


def calculate_forward_returns(df: pd.DataFrame, horizons=[15, 30, 60]) -> pd.DataFrame:
    """Calculate forward returns at various horizons."""
    forward = pd.DataFrame(index=df.index)

    for h in horizons:
        forward[f'fwd_return_{h}m'] = df['close'].shift(-h) / df['close'] - 1

    return forward


def analyze_correlations(features: pd.DataFrame, forward: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate correlations between features and forward returns.
    Returns sorted by absolute correlation.
    """
    results = []

    for target_col in forward.columns:
        for feat_col in features.columns:
            if feat_col == 'close':  # Skip price itself
                continue

            # Get aligned data
            mask = ~(features[feat_col].isna() | forward[target_col].isna())
            x = features.loc[mask, feat_col].values
            y = forward.loc[mask, target_col].values

            if len(x) < 100:
                continue

            # Spearman correlation (rank-based, more robust)
            corr, p_value = stats.spearmanr(x, y)

            # Also compute Pearson for comparison
            pearson_corr, pearson_p = stats.pearsonr(x, y)

            results.append({
                'feature': feat_col,
                'target': target_col,
                'spearman_corr': corr,
                'spearman_p': p_value,
                'pearson_corr': pearson_corr,
                'pearson_p': pearson_p,
                'n_samples': len(x),
                'significant': p_value < 0.01 and abs(corr) > 0.02
            })

    df_results = pd.DataFrame(results)
    df_results['abs_corr'] = df_results['spearman_corr'].abs()
    df_results = df_results.sort_values('abs_corr', ascending=False)

    return df_results


def main():
    print("="*70)
    print("SIMONS FEATURE CORRELATION ANALYSIS")
    print("="*70)
    print("\nPhilosophy: Before building a model, verify there's a signal to learn.")
    print("If no feature correlates with forward returns, the model is doomed.\n")

    # Load data
    print("[1] Loading SPY price data...")
    df = load_price_data(limit=20000)
    print(f"    Loaded {len(df)} bars")
    print(f"    Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    # Calculate features
    print("\n[2] Calculating technical features...")
    features = calculate_features(df)
    print(f"    Calculated {len(features.columns)} features")

    # Calculate forward returns
    print("\n[3] Calculating forward returns...")
    forward = calculate_forward_returns(df, horizons=[15, 30, 60])
    print(f"    Horizons: 15m, 30m, 60m")

    # Analyze correlations
    print("\n[4] Computing correlations...")
    corr_results = analyze_correlations(features, forward)

    # Show results for 15-minute horizon (main trading horizon)
    print("\n" + "="*70)
    print("RESULTS: Feature Correlations with 15-min Forward Returns")
    print("="*70)
    print(f"\n{'Feature':<25} {'Spearman':>10} {'p-value':>12} {'Significant':>12}")
    print("-"*70)

    fwd_15m = corr_results[corr_results['target'] == 'fwd_return_15m'].head(25)
    significant_count = 0

    for _, row in fwd_15m.iterrows():
        sig_marker = "✓ YES" if row['significant'] else "✗ no"
        if row['significant']:
            significant_count += 1
        print(f"{row['feature']:<25} {row['spearman_corr']:>10.4f} {row['spearman_p']:>12.2e} {sig_marker:>12}")

    # Summary statistics
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    max_corr = fwd_15m['spearman_corr'].abs().max()
    avg_corr = fwd_15m['spearman_corr'].abs().mean()

    print(f"\nTotal features analyzed: {len(fwd_15m)}")
    print(f"Significant features (p<0.01, |corr|>0.02): {significant_count}")
    print(f"Max |correlation|: {max_corr:.4f}")
    print(f"Avg |correlation|: {avg_corr:.4f}")

    # Simons' verdict
    print("\n" + "="*70)
    print("SIMONS' VERDICT")
    print("="*70)

    if max_corr > 0.10:
        print("✓ STRONG SIGNAL EXISTS")
        print(f"  Max correlation {max_corr:.4f} suggests predictable patterns.")
        print("  A well-designed model should be able to exploit this.")
    elif max_corr > 0.05:
        print("⚠ WEAK SIGNAL EXISTS")
        print(f"  Max correlation {max_corr:.4f} is marginal.")
        print("  Model needs careful feature engineering and risk management.")
    elif max_corr > 0.02:
        print("⚠ VERY WEAK SIGNAL")
        print(f"  Max correlation {max_corr:.4f} is barely above noise.")
        print("  Model will struggle. Consider alternative features.")
    else:
        print("✗ NO SIGNAL DETECTED")
        print(f"  Max correlation {max_corr:.4f} is indistinguishable from random.")
        print("  Building a model on this data is futile.")
        print("  RECOMMENDATION: Find better features or different data.")

    # Show best features for different horizons
    print("\n" + "="*70)
    print("BEST FEATURES BY HORIZON")
    print("="*70)

    for horizon in ['fwd_return_15m', 'fwd_return_30m', 'fwd_return_60m']:
        horizon_data = corr_results[corr_results['target'] == horizon].head(5)
        print(f"\n{horizon}:")
        for _, row in horizon_data.iterrows():
            print(f"  {row['feature']:<20}: {row['spearman_corr']:+.4f}")

    # Analyze which feature TYPES are most predictive
    print("\n" + "="*70)
    print("FEATURE TYPE ANALYSIS")
    print("="*70)

    feature_types = {
        'momentum': ['return_', 'momentum_'],
        'volatility': ['volatility_', 'range_'],
        'moving_avg': ['sma_', 'ma_cross', 'ema_', 'macd'],
        'volume': ['volume'],
        'oscillators': ['rsi_', 'bb_'],
        'time': ['hour', 'minute', 'day_of_week', 'is_']
    }

    fwd_15m_all = corr_results[corr_results['target'] == 'fwd_return_15m']

    for ftype, patterns in feature_types.items():
        matching = fwd_15m_all[fwd_15m_all['feature'].apply(
            lambda x: any(p in x for p in patterns)
        )]
        if len(matching) > 0:
            avg_abs_corr = matching['spearman_corr'].abs().mean()
            max_abs_corr = matching['spearman_corr'].abs().max()
            print(f"{ftype:<15}: avg |corr| = {avg_abs_corr:.4f}, max |corr| = {max_abs_corr:.4f}")

    return corr_results


if __name__ == "__main__":
    results = main()
