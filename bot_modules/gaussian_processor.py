"""
Gaussian Kernel Processor
=========================

Feature-space Gaussian similarity statistics over rolling reference data.

Computes RBF kernel similarities between current features and a reference 
window, providing statistics like mean, std, max, and percentiles across
multiple gamma values.

Usage:
    from bot_modules.gaussian_processor import GaussianKernelProcessor
    
    processor = GaussianKernelProcessor()
    processor.fit(feature_history)
    gaussian_features = processor.transform(current_features)
"""

import logging
from typing import List, Optional

import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class GaussianKernelProcessor:
    """
    Feature-space Gaussian similarity statistics over rolling reference data.
    
    This processor maintains a reference window of recent feature vectors and
    computes RBF kernel similarities at multiple gamma values. The resulting
    statistics (mean, std, max, percentiles) provide a measure of how similar
    the current market conditions are to recent patterns.
    
    Attributes:
        gamma_values: List of RBF kernel gamma values to use
        reference_data: The fitted reference feature window
        scaler: StandardScaler for normalizing features
        is_fitted: Whether the processor has been fitted
        
    Feature indices (for 20+ feature vectors):
        - volatility_indices: [4, 5, 6] - ATR, volatility measures
        - momentum_indices: [1, 2, 3] - Momentum at different horizons
        - volume_indices: [7, 8, 9] - Volume-related features
        - technical_indices: [10, 11, 12, 13, 14] - RSI, BB, MACD, etc.
        - jerk_indices: [30, 31, 32] - Rate of change of acceleration (if available)
        - vix_indices: [33, 34, 35] - VIX-related features (if available)
    """
    
    def __init__(self, gamma_values: List[float] = None):
        """
        Initialize the Gaussian kernel processor.
        
        Args:
            gamma_values: List of RBF kernel gamma values. 
                         Smaller gamma = wider kernels (smooth similarity).
                         Larger gamma = narrower kernels (sharp similarity).
                         Default: [0.1, 0.5, 1.0, 2.0, 5.0]
        """
        self.gamma_values = gamma_values or [0.1, 0.5, 1.0, 2.0, 5.0]
        self.reference_data: Optional[np.ndarray] = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        # Feature indices for different categories
        # These assume the feature layout from create_features()
        self.volatility_indices = [4, 5, 6]
        self.momentum_indices = [1, 2, 3]
        self.volume_indices = [7, 8, 9]
        self.technical_indices = [10, 11, 12, 13, 14]  # RSI, BB_pos, BB_std, MACD, range features
        self.jerk_indices = [30, 31, 32]  # Jerk (rate of change of acceleration)
        self.vix_indices = [33, 34, 35]   # VIX-related features

    def fit(self, features_history: List[np.ndarray], reference_window: int = 50) -> bool:
        """
        Fit the processor on historical feature data.
        
        Args:
            features_history: List of feature vectors (most recent at end)
            reference_window: Number of recent samples to use as reference
            
        Returns:
            True if fitting succeeded, False if insufficient data
        """
        if len(features_history) < 10:
            logger.debug(f"Insufficient data for Gaussian processor: {len(features_history)} < 10")
            return False
        
        # Stack recent features
        ref = np.vstack(features_history[-reference_window:])
        
        # Select relevant feature indices
        idx = self._get_feature_indices(ref.shape[1])
        
        self.reference_data = ref[:, idx]
        self.scaler.fit(self.reference_data)
        self.is_fitted = True
        
        logger.debug(f"Gaussian processor fitted with {len(self.reference_data)} reference samples, "
                    f"{len(idx)} features")
        return True

    def transform(self, features: np.ndarray) -> np.ndarray:
        """
        Transform features into Gaussian kernel similarity statistics.
        
        Args:
            features: Current feature vector
            
        Returns:
            Array of similarity statistics for each gamma value:
            [mean, std, max, 75th percentile, 25th percentile] per gamma
            Total length: 5 * len(gamma_values) = 25 features (default)
        """
        if not self.is_fitted or self.reference_data is None:
            return np.array([])
        
        # Get indices matching those used in fit()
        idx = self._get_feature_indices(len(features))
        
        # Filter to valid indices
        idx = [i for i in idx if i < len(features)]
        
        if len(idx) == 0:
            logger.warning("No valid feature indices for Gaussian transform")
            return np.array([])
            
        cur = features[idx].reshape(1, -1)
        
        try:
            cur_s = self.scaler.transform(cur)
            ref_s = self.scaler.transform(self.reference_data)
        except ValueError as e:
            logger.warning(f"Scaler transform failed: {e}")
            return np.array([])
        
        feats = []
        for gamma in self.gamma_values:
            sims = rbf_kernel(cur_s, ref_s, gamma=gamma)
            feats.extend([
                sims.mean(), 
                sims.std(), 
                sims.max(),
                np.percentile(sims, 75), 
                np.percentile(sims, 25)
            ])
        return np.array(feats)
    
    def _get_feature_indices(self, feature_dim: int) -> List[int]:
        """
        Get the feature indices to use based on feature dimension.
        
        Args:
            feature_dim: Total number of features
            
        Returns:
            List of indices to use for kernel computation
        """
        # Base indices
        idx = (self.volatility_indices + 
               self.momentum_indices + 
               self.volume_indices + 
               self.technical_indices)
        
        # Add jerk and VIX indices if features have enough dimensions
        if feature_dim >= 35:
            idx = idx + self.jerk_indices + self.vix_indices
        elif feature_dim >= 33:
            idx = idx + self.jerk_indices
        
        # Filter to valid indices
        return [i for i in idx if i < feature_dim]
    
    def reset(self):
        """Reset the processor to unfitted state."""
        self.reference_data = None
        self.is_fitted = False
        self.scaler = StandardScaler()
        logger.debug("Gaussian processor reset")
    
    def get_info(self) -> dict:
        """
        Get processor information.
        
        Returns:
            Dict with processor state information
        """
        return {
            'is_fitted': self.is_fitted,
            'gamma_values': self.gamma_values,
            'reference_samples': len(self.reference_data) if self.reference_data is not None else 0,
            'output_dim': 5 * len(self.gamma_values),
        }


# Convenience function
def compute_gaussian_features(
    features: np.ndarray,
    features_history: List[np.ndarray],
    gamma_values: List[float] = None
) -> np.ndarray:
    """
    Compute Gaussian kernel features in one call.
    
    Args:
        features: Current feature vector
        features_history: Historical feature vectors
        gamma_values: RBF kernel gamma values
        
    Returns:
        Gaussian similarity statistics array
    """
    processor = GaussianKernelProcessor(gamma_values)
    if processor.fit(features_history):
        return processor.transform(features)
    return np.array([])


__all__ = [
    'GaussianKernelProcessor',
    'compute_gaussian_features',
]








