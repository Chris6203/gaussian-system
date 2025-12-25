#!/usr/bin/env python3
"""
Multi-Dimensional HMM for Market Regime Detection

Uses 3 separate HMMs to capture different market dimensions:
1. TREND: Discovers optimal states (2-7) from data using BIC
2. VOLATILITY: Discovers optimal states (2-7) from data using BIC
3. LIQUIDITY: Discovers optimal states (2-7) from data using BIC

Auto-Discovery Feature:
- Uses Bayesian Information Criterion (BIC) to find optimal number of hidden states
- Each dimension can have different numbers of states based on data structure
- Total regimes = trend_states × vol_states × liq_states (up to 343 with max 7 each)

Example discovered regimes:
- 5 trend × 4 vol × 3 liq = 60 nuanced market states
- The data tells us what the market structure actually is!
"""

import numpy as np
import pandas as pd
from hmmlearn import hmm
import pickle
import logging
from typing import Dict, Tuple, Optional, List
from datetime import datetime
import warnings

logger = logging.getLogger(__name__)


class MultiDimensionalHMM:
    """
    Multi-dimensional market regime detector using 3 independent HMMs
    with AUTO-DISCOVERY of optimal number of hidden states using BIC.
    """
    
    def __init__(self, lookback_window: int = 50, n_iter: int = 200, tol: float = 1e-4,
                 auto_discover: bool = True, min_states: int = 2, max_states: int = 7,
                 trend_states: int = None, vol_states: int = None, liq_states: int = None):
        """
        Args:
            lookback_window: Number of bars for feature calculation
            n_iter: Maximum iterations for Baum-Welch algorithm (default 200)
            tol: Convergence threshold for Baum-Welch (default 1e-4)
            auto_discover: If True, use BIC to find optimal number of states
            min_states: Minimum states to try during auto-discovery (default 2)
            max_states: Maximum states to try during auto-discovery (default 7)
            trend_states: Override trend states (if None and not auto_discover, uses 3)
            vol_states: Override volatility states (if None and not auto_discover, uses 3)
            liq_states: Override liquidity states (if None and not auto_discover, uses 3)
        """
        self.lookback_window = lookback_window
        self.n_iter = n_iter
        self.tol = tol
        self.auto_discover = auto_discover
        self.min_states = min_states
        self.max_states = max_states
        
        # Number of states per dimension (will be set during fit if auto_discover=True)
        self.n_trend_states = trend_states or 3
        self.n_vol_states = vol_states or 3
        self.n_liq_states = liq_states or 3
        
        # HMMs will be created during fit() with optimal state counts
        self.trend_hmm = None
        self.volatility_hmm = None
        self.liquidity_hmm = None
        
        # Track fitting status
        self.is_fitted = False
        self.trend_fitted = False
        self.volatility_fitted = False
        self.liquidity_fitted = False
        
        # Feature normalization parameters
        self.trend_means = None
        self.trend_stds = None
        self.vol_means = None
        self.vol_stds = None
        self.liq_means = None
        self.liq_stds = None
        
        # State labels (will be populated dynamically based on discovered states)
        self.trend_labels = {}
        self.vol_labels = {}
        self.liq_labels = {}
        
        # BIC scores for model selection transparency
        self.trend_bic_scores = {}
        self.vol_bic_scores = {}
        self.liq_bic_scores = {}
        
        if auto_discover:
            logger.info(f"🎰 Multi-Dimensional HMM initialized with AUTO-DISCOVERY")
            logger.info(f"   Will test {min_states}-{max_states} states per dimension using BIC")
        else:
            total = self.n_trend_states * self.n_vol_states * self.n_liq_states
            logger.info(f"🎰 Multi-Dimensional HMM initialized ({self.n_trend_states}×{self.n_vol_states}×{self.n_liq_states} = {total} regimes)")
        logger.info(f"   Training algorithm: Baum-Welch EM (max_iter={n_iter}, tol={tol})")
    
    def _create_hmm(self, n_components: int, random_state: int = 42) -> hmm.GaussianHMM:
        """Create a GaussianHMM with specified number of components"""
        return hmm.GaussianHMM(
            n_components=n_components,
            covariance_type='full',
            n_iter=self.n_iter,
            tol=self.tol,
            verbose=False,
            algorithm='viterbi',
            random_state=random_state,
            init_params='stmc',
            params='stmc'
        )
    
    def _calculate_bic(self, model: hmm.GaussianHMM, data: np.ndarray) -> float:
        """
        Calculate Bayesian Information Criterion (BIC) for model selection.
        
        BIC = -2 * log_likelihood + k * log(n)
        
        Where:
            - log_likelihood: Model's log-likelihood on the data
            - k: Number of free parameters
            - n: Number of data points
            
        Lower BIC is better (balances fit vs. complexity)
        """
        n_samples, n_features = data.shape
        n_components = model.n_components
        
        # Number of free parameters for GaussianHMM with full covariance:
        # - Initial state distribution: (n_components - 1)
        # - Transition matrix: n_components * (n_components - 1)
        # - Means: n_components * n_features
        # - Covariances (full): n_components * n_features * (n_features + 1) / 2
        n_params = (
            (n_components - 1) +  # Initial probs
            n_components * (n_components - 1) +  # Transition matrix
            n_components * n_features +  # Means
            n_components * n_features * (n_features + 1) // 2  # Covariances
        )
        
        log_likelihood = model.score(data)
        bic = -2 * log_likelihood + n_params * np.log(n_samples)
        
        return bic
    
    def _find_optimal_states(self, data: np.ndarray, dimension_name: str, 
                             random_state: int = 42) -> Tuple[int, Dict[int, float]]:
        """
        Find optimal number of hidden states using BIC.
        
        Args:
            data: Normalized feature array (n_samples, n_features)
            dimension_name: Name for logging ("Trend", "Volatility", "Liquidity")
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (optimal_n_states, dict of {n_states: bic_score})
        """
        logger.info(f"   🔍 {dimension_name}: Testing {self.min_states}-{self.max_states} states...")
        
        bic_scores = {}
        best_bic = float('inf')
        best_n = self.min_states
        
        for n in range(self.min_states, self.max_states + 1):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    
                    model = self._create_hmm(n, random_state=random_state + n)
                    model.fit(data)
                    
                    bic = self._calculate_bic(model, data)
                    bic_scores[n] = bic
                    
                    if bic < best_bic:
                        best_bic = bic
                        best_n = n
                    
                    # Convergence info
                    converged = "✓" if hasattr(model, 'monitor_') and model.monitor_.converged else "~"
                    logger.debug(f"      n={n}: BIC={bic:.1f} {converged}")
                    
            except Exception as e:
                logger.debug(f"      n={n}: Failed ({e})")
                bic_scores[n] = float('inf')
        
        # Log results
        bic_range = max(bic_scores.values()) - min(bic_scores.values()) if bic_scores else 0
        logger.info(f"      → Optimal: {best_n} states (BIC={best_bic:.1f}, range={bic_range:.1f})")
        
        return best_n, bic_scores
    
    def extract_trend_features(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Extract features for trend detection
        
        Returns array of shape (n_samples, 5) with features:
        - Returns (short, medium, long term)
        - Momentum
        - Price vs moving average
        """
        if len(df) < self.lookback_window:
            return None
        
        df = df.copy()
        df.columns = [col.lower() for col in df.columns]
        
        features = []
        
        # Returns at different horizons
        returns_5 = df['close'].pct_change(5).fillna(0)
        returns_10 = df['close'].pct_change(10).fillna(0)
        returns_20 = df['close'].pct_change(20).fillna(0)
        features.extend([returns_5.values, returns_10.values, returns_20.values])
        
        # Momentum (rate of change of returns)
        momentum = returns_10.diff().fillna(0)
        features.append(momentum.values)
        
        # Price relative to SMA
        sma_20 = df['close'].rolling(20).mean()
        price_vs_sma = (df['close'] - sma_20) / (sma_20 + 1e-8)
        features.append(price_vs_sma.fillna(0).values)
        
        return np.column_stack(features)
    
    def extract_volatility_features(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Extract features for volatility detection
        
        Returns array of shape (n_samples, 5) with features:
        - Realized volatility (multiple windows)
        - Volatility of volatility
        - True Range
        """
        if len(df) < self.lookback_window:
            return None
        
        df = df.copy()
        df.columns = [col.lower() for col in df.columns]
        
        features = []
        
        # Log returns
        log_returns = np.log(df['close'] / df['close'].shift(1)).fillna(0)
        
        # Realized volatility at different windows
        #
        # IMPORTANT: Never use backfill (bfill) here — it leaks *future* information
        # into earlier timestamps (lookahead bias). In live trading this will make
        # the regime detector appear better than it is.
        #
        # Use forward-fill (ffill) for continuity and then fill remaining leading
        # NaNs with 0.
        vol_5 = log_returns.rolling(5).std().fillna(method='ffill').fillna(0)
        vol_10 = log_returns.rolling(10).std().fillna(method='ffill').fillna(0)
        vol_20 = log_returns.rolling(20).std().fillna(method='ffill').fillna(0)
        features.extend([vol_5.values, vol_10.values, vol_20.values])
        
        # Volatility of volatility (same leakage caveat as above)
        vol_of_vol = vol_10.rolling(10).std().fillna(method='ffill').fillna(0)
        features.append(vol_of_vol.values)
        
        # True Range (normalized)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift(1))
        low_close = np.abs(df['low'] - df['close'].shift(1))
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        tr_norm = true_range / df['close']
        features.append(tr_norm.fillna(0).values)
        
        return np.column_stack(features)
    
    def extract_liquidity_features(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Extract features for liquidity detection
        
        Returns array of shape (n_samples, 5) with features:
        - Volume patterns
        - Spread proxies
        - Price impact measures
        """
        if len(df) < self.lookback_window:
            return None
        
        df = df.copy()
        df.columns = [col.lower() for col in df.columns]
        
        features = []
        
        # Volume z-scores at different windows
        vol_ma_10 = df['volume'].rolling(10).mean()
        vol_std_10 = df['volume'].rolling(10).std()
        vol_zscore_10 = ((df['volume'] - vol_ma_10) / (vol_std_10 + 1e-8)).fillna(0)
        features.append(vol_zscore_10.values)
        
        vol_ma_20 = df['volume'].rolling(20).mean()
        vol_std_20 = df['volume'].rolling(20).std()
        vol_zscore_20 = ((df['volume'] - vol_ma_20) / (vol_std_20 + 1e-8)).fillna(0)
        features.append(vol_zscore_20.values)
        
        # Volume trend
        volume_trend = df['volume'].pct_change(10).fillna(0)
        features.append(volume_trend.values)
        
        # Spread proxy: (high-low) / close
        spread_proxy = (df['high'] - df['low']) / df['close']
        features.append(spread_proxy.fillna(0).values)
        
        # Price impact proxy: |return| / volume (normalized)
        price_change = np.abs(df['close'].pct_change()).fillna(0)
        volume_norm = df['volume'] / df['volume'].rolling(20).mean()
        price_impact = price_change / (volume_norm + 1e-8)
        features.append(price_impact.fillna(0).values)
        
        return np.column_stack(features)
    
    def fit(self, df: pd.DataFrame) -> bool:
        """
        Fit all 3 HMMs to historical data.
        
        If auto_discover=True, uses BIC to find optimal number of states for each dimension.
        
        Args:
            df: Historical OHLCV DataFrame
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"📊 Fitting Multi-Dimensional HMM to {len(df)} samples...")
            
            # Extract features for each dimension
            trend_features = self.extract_trend_features(df)
            vol_features = self.extract_volatility_features(df)
            liq_features = self.extract_liquidity_features(df)
            
            if trend_features is None or vol_features is None or liq_features is None:
                logger.error("❌ Insufficient data for feature extraction")
                return False
            
            # Normalize features
            self.trend_means = np.mean(trend_features, axis=0)
            self.trend_stds = np.std(trend_features, axis=0) + 1e-8
            trend_norm = (trend_features - self.trend_means) / self.trend_stds
            
            self.vol_means = np.mean(vol_features, axis=0)
            self.vol_stds = np.std(vol_features, axis=0) + 1e-8
            vol_norm = (vol_features - self.vol_means) / self.vol_stds
            
            self.liq_means = np.mean(liq_features, axis=0)
            self.liq_stds = np.std(liq_features, axis=0) + 1e-8
            liq_norm = (liq_features - self.liq_means) / self.liq_stds
            
            # ============= AUTO-DISCOVER OPTIMAL STATES =============
            if self.auto_discover:
                logger.info("🔬 AUTO-DISCOVERING optimal number of hidden states using BIC...")
                
                # Find optimal states for each dimension
                self.n_trend_states, self.trend_bic_scores = self._find_optimal_states(
                    trend_norm, "Trend", random_state=42
                )
                self.n_vol_states, self.vol_bic_scores = self._find_optimal_states(
                    vol_norm, "Volatility", random_state=43
                )
                self.n_liq_states, self.liq_bic_scores = self._find_optimal_states(
                    liq_norm, "Liquidity", random_state=44
                )
                
                total_regimes = self.n_trend_states * self.n_vol_states * self.n_liq_states
                logger.info(f"   📊 DISCOVERED: {self.n_trend_states}×{self.n_vol_states}×{self.n_liq_states} = {total_regimes} regimes")
            
            # ============= CREATE AND FIT HMMs =============
            # Create HMMs with optimal state counts
            logger.info(f"   Training Trend HMM ({self.n_trend_states} states)...")
            self.trend_hmm = self._create_hmm(self.n_trend_states, random_state=42)
            self.trend_hmm.fit(trend_norm)
            self.trend_fitted = True
            if hasattr(self.trend_hmm, 'monitor_'):
                logger.info(f"      Converged: {self.trend_hmm.monitor_.converged}, "
                          f"Iterations: {self.trend_hmm.monitor_.iter}, "
                          f"Log-likelihood: {self.trend_hmm.score(trend_norm):.2f}")
            
            logger.info(f"   Training Volatility HMM ({self.n_vol_states} states)...")
            self.volatility_hmm = self._create_hmm(self.n_vol_states, random_state=43)
            self.volatility_hmm.fit(vol_norm)
            self.volatility_fitted = True
            if hasattr(self.volatility_hmm, 'monitor_'):
                logger.info(f"      Converged: {self.volatility_hmm.monitor_.converged}, "
                          f"Iterations: {self.volatility_hmm.monitor_.iter}, "
                          f"Log-likelihood: {self.volatility_hmm.score(vol_norm):.2f}")
            
            logger.info(f"   Training Liquidity HMM ({self.n_liq_states} states)...")
            self.liquidity_hmm = self._create_hmm(self.n_liq_states, random_state=44)
            self.liquidity_hmm.fit(liq_norm)
            self.liquidity_fitted = True
            if hasattr(self.liquidity_hmm, 'monitor_'):
                logger.info(f"      Converged: {self.liquidity_hmm.monitor_.converged}, "
                          f"Iterations: {self.liquidity_hmm.monitor_.iter}, "
                          f"Log-likelihood: {self.liquidity_hmm.score(liq_norm):.2f}")
            
            self.is_fitted = True
            
            # Characterize states (now handles variable state counts)
            self._characterize_states(df, trend_norm, vol_norm, liq_norm)
            
            total = self.n_trend_states * self.n_vol_states * self.n_liq_states
            logger.info(f"✅ Multi-Dimensional HMM fitted successfully! ({total} total regimes)")
            return True
            
        except Exception as e:
            logger.error(f"❌ Multi-Dimensional HMM fitting failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return False
    
    def _characterize_states(self, df: pd.DataFrame, trend_norm, vol_norm, liq_norm):
        """
        Analyze and label each state based on characteristics.
        Handles variable numbers of states per dimension (auto-discovered).
        """
        # Predict states
        trend_states = self.trend_hmm.predict(trend_norm)
        vol_states = self.volatility_hmm.predict(vol_norm)
        liq_states = self.liquidity_hmm.predict(liq_norm)
        
        # Clear existing labels
        self.trend_labels = {}
        self.vol_labels = {}
        self.liq_labels = {}
        
        # ===== CHARACTERIZE TREND STATES =====
        trend_features_raw = self.extract_trend_features(df)
        trend_returns = []
        
        for state in range(self.n_trend_states):
            mask = (trend_states == state)
            if np.sum(mask) > 0:
                avg_return = np.mean(trend_features_raw[mask, 0])  # 5-bar return
                trend_returns.append((state, avg_return))
        
        # Sort states by return to assign labels (most bearish to most bullish)
        trend_returns.sort(key=lambda x: x[1])
        
        # Assign labels based on relative position
        for i, (state, avg_return) in enumerate(trend_returns):
            n = len(trend_returns)
            if n <= 2:
                labels = ["Downtrend", "Uptrend"]
            elif n == 3:
                labels = ["Strong Downtrend", "Sideways", "Strong Uptrend"]
            elif n == 4:
                labels = ["Strong Downtrend", "Mild Downtrend", "Mild Uptrend", "Strong Uptrend"]
            elif n == 5:
                labels = ["Crash", "Downtrend", "Sideways", "Uptrend", "Rally"]
            elif n == 6:
                labels = ["Crash", "Strong Down", "Mild Down", "Mild Up", "Strong Up", "Rally"]
            else:
                labels = [f"Trend_{j}" for j in range(n)]
            
            self.trend_labels[state] = labels[i]
            freq = np.sum(trend_states == state) / len(trend_states)
            logger.info(f"   Trend State {state}: {self.trend_labels[state]} "
                       f"(return={avg_return:.4f}, freq={freq:.1%})")
        
        # ===== CHARACTERIZE VOLATILITY STATES =====
        vol_features_raw = self.extract_volatility_features(df)
        vol_levels = []
        
        for state in range(self.n_vol_states):
            mask = (vol_states == state)
            if np.sum(mask) > 0:
                avg_vol = np.mean(vol_features_raw[mask, 1])  # 10-period realized vol
                vol_levels.append((state, avg_vol))
        
        # Sort states by volatility level (lowest to highest)
        vol_levels.sort(key=lambda x: x[1])
        
        # Assign labels based on relative position
        for i, (state, avg_vol) in enumerate(vol_levels):
            n = len(vol_levels)
            if n <= 2:
                labels = ["Low Vol", "High Vol"]
            elif n == 3:
                labels = ["Low Vol", "Normal Vol", "High Vol"]
            elif n == 4:
                labels = ["Very Low Vol", "Low Vol", "High Vol", "Extreme Vol"]
            elif n == 5:
                labels = ["Dead Calm", "Low Vol", "Normal Vol", "Elevated Vol", "Panic Vol"]
            elif n == 6:
                labels = ["Dead Calm", "Very Low", "Low", "Elevated", "High", "Panic"]
            else:
                labels = [f"Vol_{j}" for j in range(n)]
            
            self.vol_labels[state] = labels[i]
            freq = np.sum(vol_states == state) / len(vol_states)
            logger.info(f"   Vol State {state}: {self.vol_labels[state]} "
                       f"(vol={avg_vol:.4f}, freq={freq:.1%})")
        
        # ===== CHARACTERIZE LIQUIDITY STATES =====
        liq_features_raw = self.extract_liquidity_features(df)
        liq_levels = []
        
        for state in range(self.n_liq_states):
            mask = (liq_states == state)
            if np.sum(mask) > 0:
                avg_vol_zscore = np.mean(liq_features_raw[mask, 0])  # Volume z-score
                avg_spread = np.mean(liq_features_raw[mask, 3])  # Spread proxy
                # Higher volume z-score and lower spread = higher liquidity
                liq_score = avg_vol_zscore - avg_spread * 10
                liq_levels.append((state, liq_score, avg_vol_zscore, avg_spread))
        
        # Sort states by liquidity score (lowest to highest)
        liq_levels.sort(key=lambda x: x[1])
        
        # Assign labels based on relative position
        for i, (state, _, avg_vol_zscore, avg_spread) in enumerate(liq_levels):
            n = len(liq_levels)
            if n <= 2:
                labels = ["Low Liquidity", "High Liquidity"]
            elif n == 3:
                labels = ["Low Liquidity", "Normal Liquidity", "High Liquidity"]
            elif n == 4:
                labels = ["Illiquid", "Low Liq", "Normal Liq", "High Liq"]
            elif n == 5:
                labels = ["Illiquid", "Low Liq", "Normal Liq", "High Liq", "Very Liquid"]
            else:
                labels = [f"Liq_{j}" for j in range(n)]
            
            self.liq_labels[state] = labels[i]
            freq = np.sum(liq_states == state) / len(liq_states)
            logger.info(f"   Liq State {state}: {self.liq_labels[state]} "
                       f"(vol_z={avg_vol_zscore:.2f}, spread={avg_spread:.4f}, freq={freq:.1%})")
    
    def predict_current_regime(self, df: pd.DataFrame) -> Dict:
        """
        Predict current market regime across all 3 dimensions
        
        Returns:
            Dict with regime information for each dimension + combined
        """
        if not self.is_fitted:
            return self._get_default_regime()
        
        try:
            # Extract features
            trend_features = self.extract_trend_features(df)
            vol_features = self.extract_volatility_features(df)
            liq_features = self.extract_liquidity_features(df)
            
            if trend_features is None or vol_features is None or liq_features is None:
                return self._get_default_regime()
            
            # Normalize
            trend_norm = (trend_features - self.trend_means) / self.trend_stds
            vol_norm = (vol_features - self.vol_means) / self.vol_stds
            liq_norm = (liq_features - self.liq_means) / self.liq_stds
            
            # Predict states
            trend_state = self.trend_hmm.predict(trend_norm)[-1]
            vol_state = self.volatility_hmm.predict(vol_norm)[-1]
            liq_state = self.liquidity_hmm.predict(liq_norm)[-1]
            
            # Get probabilities
            trend_probs = self.trend_hmm.predict_proba(trend_norm)[-1]
            vol_probs = self.volatility_hmm.predict_proba(vol_norm)[-1]
            liq_probs = self.liquidity_hmm.predict_proba(liq_norm)[-1]
            
            # Combined confidence (geometric mean)
            combined_confidence = (trend_probs[trend_state] * 
                                 vol_probs[vol_state] * 
                                 liq_probs[liq_state]) ** (1/3)
            
            # Combined regime ID (handles variable state counts)
            # Formula: trend * (vol_states * liq_states) + vol * liq_states + liq
            combined_id = (trend_state * (self.n_vol_states * self.n_liq_states) + 
                          vol_state * self.n_liq_states + 
                          liq_state)
            
            # Generate combined name
            combined_name = (f"{self.trend_labels[trend_state]} + "
                           f"{self.vol_labels[vol_state]} + "
                           f"{self.liq_labels[liq_state]}")
            
            # Calculate total possible regimes
            total_regimes = self.n_trend_states * self.n_vol_states * self.n_liq_states
            
            regime_info = {
                # Combined regime
                "combined_id": int(combined_id),
                "combined_name": combined_name,
                "combined_confidence": float(combined_confidence),
                "total_regimes": total_regimes,
                
                # Trend dimension
                "trend_state": int(trend_state),
                "trend_name": self.trend_labels.get(trend_state, f"Trend_{trend_state}"),
                "trend_confidence": float(trend_probs[trend_state]),
                "trend_probabilities": trend_probs.tolist(),
                "n_trend_states": self.n_trend_states,
                
                # Volatility dimension
                "volatility_state": int(vol_state),
                "volatility_name": self.vol_labels.get(vol_state, f"Vol_{vol_state}"),
                "volatility_confidence": float(vol_probs[vol_state]),
                "volatility_probabilities": vol_probs.tolist(),
                "n_vol_states": self.n_vol_states,
                
                # Liquidity dimension
                "liquidity_state": int(liq_state),
                "liquidity_name": self.liq_labels.get(liq_state, f"Liq_{liq_state}"),
                "liquidity_confidence": float(liq_probs[liq_state]),
                "liquidity_probabilities": liq_probs.tolist(),
                "n_liq_states": self.n_liq_states,
                
                "timestamp": datetime.now().isoformat()
            }
            
            logger.debug(f"🎰 Multi-D Regime: {combined_name} (conf: {combined_confidence:.1%})")
            
            return regime_info
            
        except Exception as e:
            logger.error(f"❌ Regime prediction failed: {e}")
            return self._get_default_regime()
    
    def _get_default_regime(self) -> Dict:
        """Return default regime when prediction fails"""
        return {
            "combined_id": -1,
            "combined_name": "Unknown",
            "combined_confidence": 0.0,
            "total_regimes": self.n_trend_states * self.n_vol_states * self.n_liq_states,
            "trend_state": -1,
            "trend_name": "Unknown",
            "trend_confidence": 0.0,
            "trend_probabilities": [1.0 / self.n_trend_states] * self.n_trend_states,
            "n_trend_states": self.n_trend_states,
            "volatility_state": -1,
            "volatility_name": "Unknown",
            "volatility_confidence": 0.0,
            "volatility_probabilities": [1.0 / self.n_vol_states] * self.n_vol_states,
            "n_vol_states": self.n_vol_states,
            "liquidity_state": -1,
            "liquidity_name": "Unknown",
            "liquidity_confidence": 0.0,
            "liquidity_probabilities": [1.0 / self.n_liq_states] * self.n_liq_states,
            "n_liq_states": self.n_liq_states
        }
    
    def save(self, filepath: str = 'models/multi_dimensional_hmm.pkl'):
        """Save fitted model"""
        try:
            import os
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'version': 'v2_auto_discover',  # Version for compatibility
                    'trend_hmm': self.trend_hmm,
                    'volatility_hmm': self.volatility_hmm,
                    'liquidity_hmm': self.liquidity_hmm,
                    'is_fitted': self.is_fitted,
                    'trend_means': self.trend_means,
                    'trend_stds': self.trend_stds,
                    'vol_means': self.vol_means,
                    'vol_stds': self.vol_stds,
                    'liq_means': self.liq_means,
                    'liq_stds': self.liq_stds,
                    'trend_labels': self.trend_labels,
                    'vol_labels': self.vol_labels,
                    'liq_labels': self.liq_labels,
                    'lookback_window': self.lookback_window,
                    # NEW: State counts (for variable regime support)
                    'n_trend_states': self.n_trend_states,
                    'n_vol_states': self.n_vol_states,
                    'n_liq_states': self.n_liq_states,
                    # NEW: BIC scores for transparency
                    'trend_bic_scores': self.trend_bic_scores,
                    'vol_bic_scores': self.vol_bic_scores,
                    'liq_bic_scores': self.liq_bic_scores,
                    'auto_discover': self.auto_discover,
                }, f)
            
            total = self.n_trend_states * self.n_vol_states * self.n_liq_states
            logger.info(f"💾 Multi-Dimensional HMM saved to {filepath} ({total} regimes: {self.n_trend_states}×{self.n_vol_states}×{self.n_liq_states})")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to save Multi-D HMM: {e}")
            return False
    
    def load(self, filepath: str = 'models/multi_dimensional_hmm.pkl') -> bool:
        """Load fitted model"""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            self.trend_hmm = data['trend_hmm']
            self.volatility_hmm = data['volatility_hmm']
            self.liquidity_hmm = data['liquidity_hmm']
            self.is_fitted = data['is_fitted']
            self.trend_means = data['trend_means']
            self.trend_stds = data['trend_stds']
            self.vol_means = data['vol_means']
            self.vol_stds = data['vol_stds']
            self.liq_means = data['liq_means']
            self.liq_stds = data['liq_stds']
            self.trend_labels = data['trend_labels']
            self.vol_labels = data['vol_labels']
            self.liq_labels = data['liq_labels']
            self.lookback_window = data['lookback_window']
            
            # Load state counts (with backward compatibility for old models)
            self.n_trend_states = data.get('n_trend_states', 3)
            self.n_vol_states = data.get('n_vol_states', 3)
            self.n_liq_states = data.get('n_liq_states', 3)
            
            # Load BIC scores if available
            self.trend_bic_scores = data.get('trend_bic_scores', {})
            self.vol_bic_scores = data.get('vol_bic_scores', {})
            self.liq_bic_scores = data.get('liq_bic_scores', {})
            self.auto_discover = data.get('auto_discover', False)
            
            self.trend_fitted = True
            self.volatility_fitted = True
            self.liquidity_fitted = True
            
            total = self.n_trend_states * self.n_vol_states * self.n_liq_states
            version = data.get('version', 'v1_legacy')
            logger.info(f"✅ Multi-Dimensional HMM loaded from {filepath} ({version}, {total} regimes: {self.n_trend_states}×{self.n_vol_states}×{self.n_liq_states})")
            return True
        except Exception as e:
            logger.warning(f"⚠️ Could not load Multi-D HMM: {e}")
            return False


# ==============================================================================
# NEURAL REGIME CLASSIFIER - Complement to statistical HMM
# ==============================================================================

class NeuralRegimeClassifier:
    """
    Neural network complement to HMM for regime detection.
    
    Advantages over pure HMM:
    - Can learn non-linear regime patterns
    - Faster inference (no Viterbi)
    - Can incorporate more features
    - Better at capturing regime transitions
    
    Works as ensemble with HMM for more robust predictions.
    """
    
    def __init__(self, feature_dim: int = 15, n_regimes: int = 27, hidden_dim: int = 64, device: str = 'cpu'):
        """
        Args:
            feature_dim: Number of input features
            n_regimes: Number of combined regimes (3x3x3 = 27)
            hidden_dim: Hidden layer dimension
            device: 'cpu' or 'cuda'
        """
        import torch
        import torch.nn as nn
        
        self.device = device
        self.n_regimes = n_regimes
        self.feature_dim = feature_dim
        
        # Encoder network
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        ).to(device)
        
        # Regime classification head
        self.regime_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, n_regimes)
        ).to(device)
        
        # Trend/Vol/Liq individual heads (for interpretability)
        self.trend_head = nn.Linear(hidden_dim, 3).to(device)
        self.vol_head = nn.Linear(hidden_dim, 3).to(device)
        self.liq_head = nn.Linear(hidden_dim, 3).to(device)
        
        # Transition prediction (given previous regime, predict next)
        self.transition_head = nn.Sequential(
            nn.Linear(hidden_dim + n_regimes, 32),
            nn.ReLU(),
            nn.Linear(32, n_regimes)
        ).to(device)
        
        # Optimizer
        all_params = list(self.encoder.parameters()) + \
                    list(self.regime_head.parameters()) + \
                    list(self.trend_head.parameters()) + \
                    list(self.vol_head.parameters()) + \
                    list(self.liq_head.parameters()) + \
                    list(self.transition_head.parameters())
        self.optimizer = torch.optim.AdamW(all_params, lr=1e-3, weight_decay=1e-4)
        
        # Training state
        self.is_trained = False
        self.training_samples = 0
        
        logger.info(f"🧠 Neural Regime Classifier initialized ({feature_dim} features -> {n_regimes} regimes)")
    
    def predict(self, features: np.ndarray, prev_regime_probs: np.ndarray = None) -> Dict:
        """
        Predict regime from features.
        
        Args:
            features: Input features array
            prev_regime_probs: Optional previous regime probabilities for transition
            
        Returns:
            Dict with regime predictions and probabilities
        """
        import torch
        import torch.nn.functional as F
        
        # Convert to tensor
        x = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Encode features
            encoded = self.encoder(x)
            
            # Get regime probabilities
            regime_logits = self.regime_head(encoded)
            regime_probs = F.softmax(regime_logits, dim=-1).squeeze().cpu().numpy()
            
            # Get individual dimension predictions
            trend_probs = F.softmax(self.trend_head(encoded), dim=-1).squeeze().cpu().numpy()
            vol_probs = F.softmax(self.vol_head(encoded), dim=-1).squeeze().cpu().numpy()
            liq_probs = F.softmax(self.liq_head(encoded), dim=-1).squeeze().cpu().numpy()
            
            # If we have previous regime, blend with transition prediction
            if prev_regime_probs is not None:
                prev_tensor = torch.FloatTensor(prev_regime_probs).unsqueeze(0).to(self.device)
                trans_input = torch.cat([encoded, prev_tensor], dim=-1)
                trans_logits = self.transition_head(trans_input)
                trans_probs = F.softmax(trans_logits, dim=-1).squeeze().cpu().numpy()
                
                # Blend: 70% current, 30% transition
                regime_probs = 0.7 * regime_probs + 0.3 * trans_probs
        
        # Get predicted regime
        predicted_regime = int(np.argmax(regime_probs))
        confidence = float(regime_probs[predicted_regime])
        
        # Decode combined regime ID to individual states
        trend_state = predicted_regime // 9
        vol_state = (predicted_regime % 9) // 3
        liq_state = predicted_regime % 3
        
        return {
            'predicted_regime': predicted_regime,
            'regime_confidence': confidence,
            'regime_probabilities': regime_probs.tolist(),
            'trend_state': trend_state,
            'trend_probabilities': trend_probs.tolist(),
            'volatility_state': vol_state,
            'volatility_probabilities': vol_probs.tolist(),
            'liquidity_state': liq_state,
            'liquidity_probabilities': liq_probs.tolist()
        }
    
    def train_step(self, features: np.ndarray, target_regime: int, 
                   target_trend: int, target_vol: int, target_liq: int) -> Dict:
        """
        Train on a single sample.
        
        Args:
            features: Input features
            target_regime: Combined regime ID (0-26)
            target_trend: Trend state (0-2)
            target_vol: Volatility state (0-2)
            target_liq: Liquidity state (0-2)
            
        Returns:
            Dict with loss values
        """
        import torch
        import torch.nn.functional as F
        
        x = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        
        # Forward pass
        encoded = self.encoder(x)
        regime_logits = self.regime_head(encoded)
        trend_logits = self.trend_head(encoded)
        vol_logits = self.vol_head(encoded)
        liq_logits = self.liq_head(encoded)
        
        # Losses
        regime_loss = F.cross_entropy(regime_logits, torch.LongTensor([target_regime]).to(self.device))
        trend_loss = F.cross_entropy(trend_logits, torch.LongTensor([target_trend]).to(self.device))
        vol_loss = F.cross_entropy(vol_logits, torch.LongTensor([target_vol]).to(self.device))
        liq_loss = F.cross_entropy(liq_logits, torch.LongTensor([target_liq]).to(self.device))
        
        # Combined loss (regime is primary)
        total_loss = regime_loss + 0.3 * (trend_loss + vol_loss + liq_loss)
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.0)
        self.optimizer.step()
        
        self.training_samples += 1
        if self.training_samples >= 100:
            self.is_trained = True
        
        return {
            'total_loss': total_loss.item(),
            'regime_loss': regime_loss.item(),
            'dimension_loss': (trend_loss + vol_loss + liq_loss).item() / 3
        }
    
    def save(self, filepath: str = 'models/neural_regime_classifier.pt'):
        """Save model"""
        import torch
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        torch.save({
            'encoder_state': self.encoder.state_dict(),
            'regime_head_state': self.regime_head.state_dict(),
            'trend_head_state': self.trend_head.state_dict(),
            'vol_head_state': self.vol_head.state_dict(),
            'liq_head_state': self.liq_head.state_dict(),
            'transition_head_state': self.transition_head.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'is_trained': self.is_trained,
            'training_samples': self.training_samples,
            'feature_dim': self.feature_dim,
            'n_regimes': self.n_regimes
        }, filepath)
        logger.info(f"💾 Neural Regime Classifier saved to {filepath}")
    
    def load(self, filepath: str = 'models/neural_regime_classifier.pt') -> bool:
        """Load model"""
        import torch
        try:
            checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
            self.encoder.load_state_dict(checkpoint['encoder_state'])
            self.regime_head.load_state_dict(checkpoint['regime_head_state'])
            self.trend_head.load_state_dict(checkpoint['trend_head_state'])
            self.vol_head.load_state_dict(checkpoint['vol_head_state'])
            self.liq_head.load_state_dict(checkpoint['liq_head_state'])
            self.transition_head.load_state_dict(checkpoint['transition_head_state'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            self.is_trained = checkpoint.get('is_trained', False)
            self.training_samples = checkpoint.get('training_samples', 0)
            logger.info(f"✅ Neural Regime Classifier loaded from {filepath}")
            return True
        except Exception as e:
            logger.warning(f"⚠️ Could not load Neural Regime Classifier: {e}")
            return False


class EnsembleRegimeDetector:
    """
    Ensemble of HMM + Neural Classifier for robust regime detection.
    
    Combines statistical (HMM) and neural approaches for best of both worlds:
    - HMM: Interpretable, works well with limited data
    - Neural: Can capture complex patterns, faster inference
    """
    
    def __init__(self, hmm: MultiDimensionalHMM, neural: NeuralRegimeClassifier = None,
                 hmm_weight: float = 0.6):
        """
        Args:
            hmm: Fitted MultiDimensionalHMM instance
            neural: Optional NeuralRegimeClassifier instance
            hmm_weight: Weight for HMM predictions (0-1), neural gets 1-hmm_weight
        """
        self.hmm = hmm
        self.neural = neural
        self.hmm_weight = hmm_weight
        self.neural_weight = 1.0 - hmm_weight
        
        logger.info(f"🎯 Ensemble Regime Detector initialized (HMM: {hmm_weight:.0%}, Neural: {1-hmm_weight:.0%})")
    
    def predict(self, df: pd.DataFrame, features: np.ndarray = None) -> Dict:
        """
        Get ensemble regime prediction.
        
        Args:
            df: OHLCV DataFrame for HMM
            features: Feature array for neural classifier (optional)
            
        Returns:
            Combined regime prediction dict
        """
        # Get HMM prediction
        hmm_result = self.hmm.predict_current_regime(df)
        
        # If no neural classifier or features, return HMM only
        if self.neural is None or features is None or not self.neural.is_trained:
            hmm_result['source'] = 'hmm_only'
            return hmm_result
        
        # Get neural prediction
        neural_result = self.neural.predict(features)
        
        # Ensemble: Weighted average of probabilities
        hmm_probs = np.array(hmm_result.get('trend_probabilities', [0.33, 0.34, 0.33]))
        neural_probs = np.array(neural_result.get('trend_probabilities', [0.33, 0.34, 0.33]))
        
        ensemble_trend_probs = self.hmm_weight * hmm_probs + self.neural_weight * neural_probs
        ensemble_trend = int(np.argmax(ensemble_trend_probs))
        
        hmm_vol_probs = np.array(hmm_result.get('volatility_probabilities', [0.33, 0.34, 0.33]))
        neural_vol_probs = np.array(neural_result.get('volatility_probabilities', [0.33, 0.34, 0.33]))
        ensemble_vol_probs = self.hmm_weight * hmm_vol_probs + self.neural_weight * neural_vol_probs
        ensemble_vol = int(np.argmax(ensemble_vol_probs))
        
        hmm_liq_probs = np.array(hmm_result.get('liquidity_probabilities', [0.33, 0.34, 0.33]))
        neural_liq_probs = np.array(neural_result.get('liquidity_probabilities', [0.33, 0.34, 0.33]))
        ensemble_liq_probs = self.hmm_weight * hmm_liq_probs + self.neural_weight * neural_liq_probs
        ensemble_liq = int(np.argmax(ensemble_liq_probs))
        
        # Combined regime ID (handles variable state counts)
        n_vol = self.hmm.n_vol_states
        n_liq = self.hmm.n_liq_states
        combined_id = ensemble_trend * (n_vol * n_liq) + ensemble_vol * n_liq + ensemble_liq
        
        # Combined confidence
        combined_conf = (ensemble_trend_probs[ensemble_trend] * 
                        ensemble_vol_probs[ensemble_vol] * 
                        ensemble_liq_probs[ensemble_liq]) ** (1/3)
        
        return {
            'combined_id': combined_id,
            'combined_name': f"{self.hmm.trend_labels.get(ensemble_trend, 'Unknown')} + "
                           f"{self.hmm.vol_labels.get(ensemble_vol, 'Unknown')} + "
                           f"{self.hmm.liq_labels.get(ensemble_liq, 'Unknown')}",
            'combined_confidence': float(combined_conf),
            'trend_state': ensemble_trend,
            'trend_name': self.hmm.trend_labels.get(ensemble_trend, 'Unknown'),
            'trend_confidence': float(ensemble_trend_probs[ensemble_trend]),
            'trend_probabilities': ensemble_trend_probs.tolist(),
            'volatility_state': ensemble_vol,
            'volatility_name': self.hmm.vol_labels.get(ensemble_vol, 'Unknown'),
            'volatility_confidence': float(ensemble_vol_probs[ensemble_vol]),
            'volatility_probabilities': ensemble_vol_probs.tolist(),
            'liquidity_state': ensemble_liq,
            'liquidity_name': self.hmm.liq_labels.get(ensemble_liq, 'Unknown'),
            'liquidity_confidence': float(ensemble_liq_probs[ensemble_liq]),
            'liquidity_probabilities': ensemble_liq_probs.tolist(),
            'source': 'ensemble',
            'hmm_weight': self.hmm_weight,
            'neural_weight': self.neural_weight
        }


# Helper function for bot integration
def add_multid_regime_to_features(features: np.ndarray, regime_info: Dict) -> np.ndarray:
    """
    Add multi-dimensional regime information as features
    
    Args:
        features: Existing feature array (n_features,)
        regime_info: Regime dict from predict_current_regime()
    
    Returns:
        Extended feature array with 10 regime features
    """
    # Get state counts (with defaults for backward compatibility)
    n_trend = regime_info.get('n_trend_states', 3)
    n_vol = regime_info.get('n_vol_states', 3)
    n_liq = regime_info.get('n_liq_states', 3)
    total_regimes = regime_info.get('total_regimes', n_trend * n_vol * n_liq)
    
    # Normalize by max possible values (handles variable state counts)
    regime_features = np.array([
        regime_info['combined_id'] / max(total_regimes - 1, 1),  # Normalize to [0, 1]
        regime_info['combined_confidence'],
        regime_info['trend_state'] / max(n_trend - 1, 1),  # Normalize by max trend state
        regime_info['trend_confidence'],
        regime_info['volatility_state'] / max(n_vol - 1, 1),  # Normalize by max vol state
        regime_info['volatility_confidence'],
        regime_info['liquidity_state'] / max(n_liq - 1, 1),  # Normalize by max liq state
        regime_info['liquidity_confidence'],
        # Additional context (relative to discovered states)
        1.0 if regime_info['trend_state'] == n_trend - 1 else 0.0,  # Is max uptrend?
        1.0 if regime_info['volatility_state'] == 0 else 0.0  # Is lowest vol?
    ])
    
    return np.concatenate([features, regime_features])


if __name__ == "__main__":
    """Test Multi-Dimensional HMM with Auto-Discovery"""
    
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data with clear regime structure
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=2000, freq='1min')
    
    # Simulate different market regimes (5 distinct phases)
    prices = [100]
    for i in range(1999):
        if i < 400:  # Strong uptrend + low vol
            change = np.random.normal(0.002, 0.008)
        elif i < 700:  # Mild uptrend + normal vol
            change = np.random.normal(0.0005, 0.015)
        elif i < 1000:  # Crash + high vol
            change = np.random.normal(-0.003, 0.04)
        elif i < 1400:  # Recovery + moderate vol
            change = np.random.normal(0.001, 0.02)
        else:  # Sideways + low vol
            change = np.random.normal(0, 0.006)
        prices.append(prices[-1] * (1 + change))
    
    df = pd.DataFrame({
        'open': np.array(prices) * 0.999,
        'high': np.array(prices) * 1.002,
        'low': np.array(prices) * 0.998,
        'close': prices,
        'volume': np.random.randint(1000, 10000, len(prices))
    }, index=dates)
    
    print("\n" + "=" * 70)
    print("TEST 1: AUTO-DISCOVERY (BIC-based state selection)")
    print("=" * 70)
    
    # Test with auto-discovery enabled
    md_hmm = MultiDimensionalHMM(
        lookback_window=50, 
        auto_discover=True,  # Let the data tell us optimal state count!
        min_states=2, 
        max_states=6
    )
    
    # Fit
    success = md_hmm.fit(df.iloc[:-50])
    
    if success:
        # Predict current regime
        regime = md_hmm.predict_current_regime(df)
        
        total = md_hmm.n_trend_states * md_hmm.n_vol_states * md_hmm.n_liq_states
        print(f"\n[AUTO-DISCOVERED MARKET REGIME]")
        print(f"=" * 70)
        print(f"Discovered Structure: {md_hmm.n_trend_states}×{md_hmm.n_vol_states}×{md_hmm.n_liq_states} = {total} regimes")
        print(f"\nCurrent Regime: {regime['combined_name']}")
        print(f"Confidence: {regime['combined_confidence']:.1%}")
        print(f"\nDimensions:")
        print(f"  Trend:      {regime['trend_name']} (state {regime['trend_state']}/{md_hmm.n_trend_states-1}, conf: {regime['trend_confidence']:.1%})")
        print(f"  Volatility: {regime['volatility_name']} (state {regime['volatility_state']}/{md_hmm.n_vol_states-1}, conf: {regime['volatility_confidence']:.1%})")
        print(f"  Liquidity:  {regime['liquidity_name']} (state {regime['liquidity_state']}/{md_hmm.n_liq_states-1}, conf: {regime['liquidity_confidence']:.1%})")
        
        # Show BIC scores
        print(f"\nBIC Scores (lower = better):")
        print(f"  Trend:      {md_hmm.trend_bic_scores}")
        print(f"  Volatility: {md_hmm.vol_bic_scores}")
        print(f"  Liquidity:  {md_hmm.liq_bic_scores}")
        
        # Save model
        md_hmm.save()
        
    print("\n" + "=" * 70)
    print("TEST 2: FIXED STATES (3×3×3 = 27 regimes, like before)")
    print("=" * 70)
    
    # Test with fixed states (backward compatible)
    md_hmm_fixed = MultiDimensionalHMM(
        lookback_window=50, 
        auto_discover=False,  # Use fixed 3×3×3
        trend_states=3,
        vol_states=3,
        liq_states=3
    )
    
    success = md_hmm_fixed.fit(df.iloc[:-50])
    if success:
        regime = md_hmm_fixed.predict_current_regime(df)
        print(f"\nCurrent Regime: {regime['combined_name']}")
        print(f"Total possible: {regime.get('total_regimes', 27)} regimes")
