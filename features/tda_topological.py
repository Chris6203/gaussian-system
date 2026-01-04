#!/usr/bin/env python3
"""
Topological Data Analysis (TDA) Features Module

Implements persistent homology features for time series analysis using giotto-tda.
These features capture "shape" information that traditional indicators miss:
- Regime transitions
- Crash-like geometry
- Structural changes in price dynamics

Based on: https://giotto-ai.github.io/gtda-docs/latest/notebooks/topology_time_series.html

For trading, TDA features can detect:
1. Loop structures (cyclical behavior) - H1 homology
2. Void structures (regime changes) - H2 homology
3. Connected components (trend segments) - H0 homology
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
import os

logger = logging.getLogger(__name__)

# Check if giotto-tda is available
try:
    from gtda.time_series import SingleTakensEmbedding, SlidingWindow
    from gtda.homology import VietorisRipsPersistence
    from gtda.diagrams import (
        PersistenceEntropy,
        Amplitude,
        NumberOfPoints,
        PersistenceLandscape,
        BettiCurve
    )
    GTDA_AVAILABLE = True
    logger.info("giotto-tda available - TDA features enabled")
except ImportError:
    GTDA_AVAILABLE = False
    logger.warning("giotto-tda not installed. Install with: pip install giotto-tda")


# =============================================================================
# CONFIGURATION
# =============================================================================

# Window sizes for TDA computation (in bars/minutes)
TDA_WINDOW_SIZE = int(os.environ.get('TDA_WINDOW_SIZE', '60'))  # 60 minutes
TDA_STRIDE = int(os.environ.get('TDA_STRIDE', '1'))  # Compute every bar

# Takens embedding parameters
TAKENS_TIME_DELAY = int(os.environ.get('TAKENS_TIME_DELAY', '3'))  # 3 bars delay
TAKENS_DIMENSION = int(os.environ.get('TAKENS_DIMENSION', '3'))  # 3D embedding

# Homology dimensions to compute (0=components, 1=loops, 2=voids)
HOMOLOGY_DIMENSIONS = [0, 1]  # H0 and H1 are most useful for 1D time series

# Feature extraction settings
N_BINS = 50  # For Betti curves and landscapes
N_LAYERS = 3  # For persistence landscapes


# =============================================================================
# TDA FEATURE COMPUTATION
# =============================================================================

class TDAFeatureExtractor:
    """
    Extract topological features from time series using persistent homology.

    Pipeline:
    1. Takens embedding (time delay embedding)
    2. Vietoris-Rips persistence (compute persistent homology)
    3. Feature extraction (entropy, amplitude, Betti curves, etc.)
    """

    def __init__(
        self,
        window_size: int = TDA_WINDOW_SIZE,
        time_delay: int = TAKENS_TIME_DELAY,
        dimension: int = TAKENS_DIMENSION,
        homology_dimensions: List[int] = None
    ):
        self.window_size = window_size
        self.time_delay = time_delay
        self.dimension = dimension
        self.homology_dimensions = homology_dimensions or HOMOLOGY_DIMENSIONS

        if not GTDA_AVAILABLE:
            logger.warning("TDAFeatureExtractor initialized but giotto-tda not available")
            return

        # Initialize persistence and feature extractors
        self.persistence = VietorisRipsPersistence(
            homology_dimensions=self.homology_dimensions,
            n_jobs=1  # Single thread to avoid overhead
        )

        # Feature extractors
        self.entropy = PersistenceEntropy()
        self.amplitude = Amplitude(metric='wasserstein')
        self.n_points = NumberOfPoints()

    def _create_point_cloud(self, ts: np.ndarray) -> np.ndarray:
        """Create point cloud from time series using time delay embedding."""
        n = len(ts)
        n_points = n - (self.dimension - 1) * self.time_delay
        if n_points < 10:  # Need at least 10 points for meaningful topology
            return None
        cloud = np.zeros((n_points, self.dimension))
        for i in range(self.dimension):
            cloud[:, i] = ts[i * self.time_delay:i * self.time_delay + n_points]
        return cloud

    def _prepare_window(self, data: np.ndarray) -> np.ndarray:
        """Prepare data window for TDA computation."""
        # Normalize to [0, 1] range for stable computation
        data = np.array(data).flatten()
        if len(data) < self.window_size:
            # Pad with last value if too short
            padding = np.full(self.window_size - len(data), data[-1] if len(data) > 0 else 0)
            data = np.concatenate([padding, data])
        elif len(data) > self.window_size:
            # Take most recent window
            data = data[-self.window_size:]

        # Normalize
        data_min, data_max = data.min(), data.max()
        if data_max - data_min > 1e-10:
            data = (data - data_min) / (data_max - data_min)
        else:
            data = np.zeros_like(data) + 0.5

        return data  # Shape: (window_size,) - 1D array for Takens embedding

    def compute_features(self, returns: np.ndarray) -> Dict[str, float]:
        """
        Compute TDA features from returns time series.

        Args:
            returns: 1D array of returns (at least window_size points)

        Returns:
            Dict of TDA features
        """
        if not GTDA_AVAILABLE:
            return self._get_default_features()

        try:
            # Prepare and normalize data
            data = self._prepare_window(returns)

            # Create point cloud via time delay embedding
            cloud = self._create_point_cloud(data)
            if cloud is None:
                return self._get_default_features()

            # Reshape for gtda (expects 3D: n_samples, n_points, n_dims)
            clouds = cloud.reshape(1, *cloud.shape)

            # Compute persistence diagram
            diagrams = self.persistence.fit_transform(clouds)

            # Extract features
            features = {}

            # Persistence entropy (measure of topological complexity)
            entropy = self.entropy.fit_transform(diagrams)
            for i, dim in enumerate(self.homology_dimensions):
                features[f'tda_entropy_h{dim}'] = float(entropy[0, i])

            # Amplitude (Wasserstein distance - measure of persistence strength)
            amplitude = self.amplitude.fit_transform(diagrams)
            for i, dim in enumerate(self.homology_dimensions):
                features[f'tda_amplitude_h{dim}'] = float(amplitude[0, i])

            # Number of points (count of topological features)
            n_points = self.n_points.fit_transform(diagrams)
            for i, dim in enumerate(self.homology_dimensions):
                features[f'tda_n_points_h{dim}'] = float(n_points[0, i])

            # Derived features
            # High H1 entropy = cyclical/oscillating behavior
            # High H0 amplitude = strong trend segments
            # Low H0 n_points = smooth trend
            # High H1 n_points = many loops = choppy/ranging

            # Regime indicator: H1/H0 ratio
            h0_amp = features.get('tda_amplitude_h0', 1.0)
            h1_amp = features.get('tda_amplitude_h1', 0.0)
            features['tda_loop_trend_ratio'] = h1_amp / max(h0_amp, 0.01)

            # Complexity score
            features['tda_complexity'] = np.mean([
                features.get('tda_entropy_h0', 0),
                features.get('tda_entropy_h1', 0)
            ])

            # Normalize features
            for key in features:
                if np.isnan(features[key]) or np.isinf(features[key]):
                    features[key] = 0.0

            return features

        except Exception as e:
            logger.warning(f"TDA computation failed: {e}")
            return self._get_default_features()

    def _get_default_features(self) -> Dict[str, float]:
        """Return default features when TDA is unavailable."""
        features = {}
        for dim in self.homology_dimensions:
            features[f'tda_entropy_h{dim}'] = 0.0
            features[f'tda_amplitude_h{dim}'] = 0.0
            features[f'tda_n_points_h{dim}'] = 0.0
        features['tda_loop_trend_ratio'] = 0.0
        features['tda_complexity'] = 0.0
        return features


# =============================================================================
# MULTI-SYMBOL TDA FEATURES
# =============================================================================

def compute_tda_features(
    price_data: Dict[str, pd.DataFrame],
    symbols: List[str] = None,
    window_size: int = TDA_WINDOW_SIZE
) -> Dict[str, float]:
    """
    Compute TDA features for multiple symbols.

    Args:
        price_data: Dict mapping symbol -> OHLCV DataFrame
        symbols: List of symbols to compute TDA for (default: SPY, QQQ, VIX)
        window_size: Rolling window size

    Returns:
        Dict of TDA features with symbol prefix
    """
    if symbols is None:
        symbols = ['SPY', 'QQQ', 'VIX']

    extractor = TDAFeatureExtractor(window_size=window_size)
    all_features = {}

    for symbol in symbols:
        if symbol not in price_data:
            continue

        df = price_data[symbol]
        if df.empty or len(df) < window_size:
            continue

        # Get close prices and compute returns
        close_col = 'close' if 'close' in df.columns else 'Close'
        if close_col not in df.columns:
            continue

        prices = df[close_col].values
        returns = np.diff(prices) / prices[:-1]

        # Compute TDA features
        features = extractor.compute_features(returns)

        # Add symbol prefix
        for key, value in features.items():
            all_features[f'{symbol.lower()}_{key}'] = value

    # Cross-symbol features
    if 'spy_tda_complexity' in all_features and 'qqq_tda_complexity' in all_features:
        # Divergence between SPY and QQQ topology
        all_features['tda_spy_qqq_divergence'] = abs(
            all_features['spy_tda_complexity'] - all_features['qqq_tda_complexity']
        )

    if 'vix_tda_loop_trend_ratio' in all_features:
        # VIX topology as fear indicator
        all_features['tda_vix_fear_signal'] = all_features['vix_tda_loop_trend_ratio']

    return all_features


def compute_regime_tda_features(
    returns: np.ndarray,
    window_sizes: List[int] = None
) -> Dict[str, float]:
    """
    Compute multi-scale TDA features for regime detection.

    Uses multiple window sizes to capture different time scales:
    - Short (30 bars): Intraday patterns
    - Medium (60 bars): 1-hour patterns
    - Long (120 bars): 2-hour patterns

    Args:
        returns: 1D array of returns
        window_sizes: List of window sizes to use

    Returns:
        Dict of multi-scale TDA features
    """
    if window_sizes is None:
        window_sizes = [30, 60, 120]

    all_features = {}

    for window in window_sizes:
        if len(returns) < window:
            continue

        extractor = TDAFeatureExtractor(window_size=window)
        features = extractor.compute_features(returns[-window:])

        # Add window size suffix
        for key, value in features.items():
            all_features[f'{key}_{window}m'] = value

    # Multi-scale regime indicators
    if all(f'tda_complexity_{w}m' in all_features for w in window_sizes[:2]):
        # Complexity trend (increasing = regime change)
        short = all_features[f'tda_complexity_{window_sizes[0]}m']
        medium = all_features[f'tda_complexity_{window_sizes[1]}m']
        all_features['tda_complexity_trend'] = short - medium

    return all_features


# =============================================================================
# FEATURE NAMES (for pipeline integration)
# =============================================================================

def get_tda_feature_names(symbols: List[str] = None) -> List[str]:
    """Get list of TDA feature names for pipeline integration."""
    if symbols is None:
        symbols = ['spy', 'qqq', 'vix']

    base_features = [
        'tda_entropy_h0', 'tda_entropy_h1',
        'tda_amplitude_h0', 'tda_amplitude_h1',
        'tda_n_points_h0', 'tda_n_points_h1',
        'tda_loop_trend_ratio', 'tda_complexity'
    ]

    features = []
    for symbol in symbols:
        for feat in base_features:
            features.append(f'{symbol}_{feat}')

    # Cross-symbol features
    features.extend([
        'tda_spy_qqq_divergence',
        'tda_vix_fear_signal'
    ])

    return features


# =============================================================================
# TEST / DEMO
# =============================================================================

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("Testing TDA Features Module")
    print("=" * 60)

    if not GTDA_AVAILABLE:
        print("\ngiotto-tda not installed. Install with:")
        print("  pip install giotto-tda")
        print("\nShowing default features:")
        extractor = TDAFeatureExtractor()
        features = extractor._get_default_features()
        for k, v in features.items():
            print(f"  {k}: {v}")
    else:
        print("\nGenerating test data (simulated returns)...")

        # Generate test returns with regime change
        np.random.seed(42)
        n_points = 120

        # Trending regime (first half)
        trend_returns = np.random.normal(0.001, 0.005, n_points // 2)
        trend_returns = np.cumsum(trend_returns) * 0.01  # Cumulative for trend

        # Choppy regime (second half)
        choppy_returns = np.random.normal(0, 0.01, n_points // 2)

        returns = np.concatenate([trend_returns, choppy_returns])

        print(f"\nTest data: {len(returns)} points")
        print(f"  Trending regime: first {n_points//2} points")
        print(f"  Choppy regime: last {n_points//2} points")

        # Compute features
        extractor = TDAFeatureExtractor(window_size=60)

        print("\n--- Trending Regime Features ---")
        trend_features = extractor.compute_features(returns[:60])
        for k, v in trend_features.items():
            print(f"  {k}: {v:.4f}")

        print("\n--- Choppy Regime Features ---")
        choppy_features = extractor.compute_features(returns[60:])
        for k, v in choppy_features.items():
            print(f"  {k}: {v:.4f}")

        print("\n--- Interpretation ---")
        trend_ratio_diff = choppy_features['tda_loop_trend_ratio'] - trend_features['tda_loop_trend_ratio']
        print(f"Loop/Trend ratio difference: {trend_ratio_diff:+.4f}")
        print(f"  (Positive = more loops/oscillations in choppy regime)")

        complexity_diff = choppy_features['tda_complexity'] - trend_features['tda_complexity']
        print(f"Complexity difference: {complexity_diff:+.4f}")
        print(f"  (Higher complexity = more topological features)")
