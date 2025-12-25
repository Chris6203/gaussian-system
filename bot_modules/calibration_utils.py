"""
Calibration Utilities
=====================

Confidence calibration methods for trading signals.

Provides:
- Platt scaling (logistic calibration)
- Bucket-based calibration
- Calibration metrics (Brier score, ECE)

Usage:
    from bot_modules.calibration_utils import (
        PlattCalibrator,
        BucketCalibrator,
        CalibrationMetrics
    )
"""

import logging
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CalibrationConfig:
    """Configuration for calibration."""
    buffer_size: int = 1000          # Maximum samples to keep
    min_samples: int = 50            # Minimum samples for calibration
    refit_interval: int = 50         # Refit after this many new samples
    max_brier_score: float = 0.35    # Maximum acceptable Brier score
    max_ece: float = 0.15            # Maximum acceptable ECE
    platt_weight: float = 0.4        # Weight for Platt scaling
    isotonic_weight: float = 0.6     # Weight for isotonic regression


class PlattCalibrator:
    """
    Platt scaling for probability calibration.
    
    Learns parameters A, B such that:
        P(correct|confidence) = sigmoid(A * confidence + B)
    
    This maps raw confidence scores to calibrated probabilities.
    """
    
    def __init__(self, learning_rate: float = 0.1, max_iters: int = 500):
        """
        Initialize Platt calibrator.
        
        Args:
            learning_rate: Gradient descent learning rate
            max_iters: Maximum iterations for fitting
        """
        self.learning_rate = learning_rate
        self.max_iters = max_iters
        self.A: Optional[float] = None
        self.B: Optional[float] = None
        self.n_samples: int = 0
        self.is_fitted: bool = False
    
    def fit(self, confidences: np.ndarray, outcomes: np.ndarray) -> bool:
        """
        Fit Platt scaling parameters.
        
        Args:
            confidences: Array of raw confidence scores
            outcomes: Array of binary outcomes (0 or 1)
            
        Returns:
            True if fitting succeeded
        """
        if len(confidences) < 50:
            return False
        
        try:
            # Initialize parameters
            A, B = -1.0, 0.0
            tol = 1e-6
            
            for _ in range(self.max_iters):
                # Compute predictions
                z = A * confidences + B
                p = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
                
                # Compute gradients (negative log-likelihood)
                error = p - outcomes
                grad_A = np.dot(error, confidences) / len(confidences)
                grad_B = np.mean(error)
                
                # Update parameters
                A_new = A - self.learning_rate * grad_A
                B_new = B - self.learning_rate * grad_B
                
                # Check convergence
                if abs(A_new - A) < tol and abs(B_new - B) < tol:
                    break
                
                A, B = A_new, B_new
            
            self.A = float(A)
            self.B = float(B)
            self.n_samples = len(confidences)
            self.is_fitted = True
            
            logger.debug(f"Platt scaling fitted: A={self.A:.4f}, B={self.B:.4f} (n={self.n_samples})")
            return True
            
        except Exception as e:
            logger.warning(f"Platt scaling fit failed: {e}")
            return False
    
    def calibrate(self, raw_confidence: float) -> float:
        """
        Calibrate a raw confidence score.
        
        Args:
            raw_confidence: Raw confidence (0-1)
            
        Returns:
            Calibrated confidence (0-1)
        """
        if not self.is_fitted:
            return raw_confidence
        
        z = self.A * raw_confidence + self.B
        calibrated = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
        return float(calibrated)
    
    def get_params(self) -> Dict:
        """Get current parameters."""
        return {
            'A': self.A,
            'B': self.B,
            'n_samples': self.n_samples,
            'is_fitted': self.is_fitted
        }


class BucketCalibrator:
    """
    Bucket-based (empirical) calibration.
    
    Divides confidence into buckets and tracks actual accuracy per bucket.
    Simple but effective fallback when Platt scaling fails.
    """
    
    def __init__(self, n_buckets: int = 10, min_samples_per_bucket: int = 10):
        """
        Initialize bucket calibrator.
        
        Args:
            n_buckets: Number of confidence buckets
            min_samples_per_bucket: Minimum samples to use bucket
        """
        self.n_buckets = n_buckets
        self.min_samples = min_samples_per_bucket
        self.bucket_size = 1.0 / n_buckets
        self.buckets: Dict[int, List[bool]] = {i: [] for i in range(n_buckets)}
    
    def add_sample(self, confidence: float, correct: bool):
        """
        Add a sample to the calibration data.
        
        Args:
            confidence: Raw confidence score
            correct: Whether the prediction was correct
        """
        bucket_idx = min(self.n_buckets - 1, int(confidence / self.bucket_size))
        self.buckets[bucket_idx].append(correct)
    
    def calibrate(self, raw_confidence: float) -> float:
        """
        Calibrate a raw confidence score.
        
        Args:
            raw_confidence: Raw confidence (0-1)
            
        Returns:
            Calibrated confidence based on empirical accuracy
        """
        bucket_idx = min(self.n_buckets - 1, int(raw_confidence / self.bucket_size))
        
        # Look at nearby buckets for smoother calibration
        samples = []
        for i in range(max(0, bucket_idx - 1), min(self.n_buckets, bucket_idx + 2)):
            samples.extend(self.buckets[i])
        
        if len(samples) < self.min_samples:
            return raw_confidence
        
        return sum(samples) / len(samples)
    
    def get_bucket_stats(self) -> Dict[str, Dict]:
        """Get statistics for each bucket."""
        stats = {}
        for i in range(self.n_buckets):
            if len(self.buckets[i]) > 0:
                bucket_name = f"{i * 10}-{(i + 1) * 10}%"
                accuracy = sum(self.buckets[i]) / len(self.buckets[i])
                stats[bucket_name] = {
                    'samples': len(self.buckets[i]),
                    'accuracy': accuracy
                }
        return stats
    
    def clear(self):
        """Clear all bucket data."""
        self.buckets = {i: [] for i in range(self.n_buckets)}


class CalibrationMetrics:
    """
    Calculates calibration quality metrics.
    
    Metrics:
    - Brier Score: Mean squared error of probability predictions
    - ECE (Expected Calibration Error): Gap between confidence and accuracy
    - Direction Accuracy: Overall prediction accuracy
    """
    
    @staticmethod
    def brier_score(confidences: np.ndarray, outcomes: np.ndarray) -> float:
        """
        Calculate Brier score.
        
        Lower is better:
        - 0 = perfect predictions
        - 0.25 = random guessing
        - 0.35 = typical threshold for "acceptable"
        
        Args:
            confidences: Calibrated confidence scores
            outcomes: Binary outcomes (0 or 1)
            
        Returns:
            Brier score
        """
        return float(np.mean((confidences - outcomes) ** 2))
    
    @staticmethod
    def expected_calibration_error(
        confidences: np.ndarray, 
        outcomes: np.ndarray, 
        n_bins: int = 10
    ) -> float:
        """
        Calculate Expected Calibration Error (ECE).
        
        Measures how well confidence matches actual accuracy per bucket.
        Lower is better:
        - 0 = perfectly calibrated
        - 0.15 = typical threshold for "well calibrated"
        
        Args:
            confidences: Calibrated confidence scores
            outcomes: Binary outcomes
            n_bins: Number of bins for calibration
            
        Returns:
            ECE score
        """
        bin_counts = np.zeros(n_bins)
        bin_correct = np.zeros(n_bins)
        bin_confidence = np.zeros(n_bins)
        
        for conf, outcome in zip(confidences, outcomes):
            bin_idx = min(int(conf * n_bins), n_bins - 1)
            bin_counts[bin_idx] += 1
            bin_correct[bin_idx] += outcome
            bin_confidence[bin_idx] += conf
        
        ece = 0.0
        total = len(confidences)
        for i in range(n_bins):
            if bin_counts[i] > 0:
                avg_conf = bin_confidence[i] / bin_counts[i]
                avg_acc = bin_correct[i] / bin_counts[i]
                ece += (bin_counts[i] / total) * abs(avg_conf - avg_acc)
        
        return float(ece)
    
    @staticmethod
    def accuracy(outcomes: np.ndarray) -> float:
        """Calculate overall accuracy."""
        return float(np.mean(outcomes))
    
    @staticmethod
    def compute_all(
        confidences: np.ndarray, 
        outcomes: np.ndarray,
        config: CalibrationConfig = None
    ) -> Dict:
        """
        Compute all calibration metrics.
        
        Args:
            confidences: Calibrated confidence scores
            outcomes: Binary outcomes
            config: Calibration configuration
            
        Returns:
            Dict with all metrics and calibration status
        """
        if config is None:
            config = CalibrationConfig()
        
        n_samples = len(confidences)
        
        if n_samples < 20:
            return {
                'brier_score': 1.0,
                'ece': 1.0,
                'accuracy': 0.0,
                'sample_count': n_samples,
                'is_calibrated': False
            }
        
        brier = CalibrationMetrics.brier_score(confidences, outcomes)
        ece = CalibrationMetrics.expected_calibration_error(confidences, outcomes)
        accuracy = CalibrationMetrics.accuracy(outcomes)
        
        is_calibrated = (
            brier <= config.max_brier_score and 
            ece <= config.max_ece and 
            n_samples >= config.min_samples
        )
        
        return {
            'brier_score': brier,
            'ece': ece,
            'accuracy': accuracy,
            'sample_count': n_samples,
            'is_calibrated': is_calibrated
        }


class CalibrationBuffer:
    """
    Rolling buffer for calibration data.
    
    Stores (confidence, outcome) pairs for training calibrators.
    Uses a fixed-size deque to limit memory usage.
    """
    
    def __init__(self, max_size: int = 1000):
        """
        Initialize calibration buffer.
        
        Args:
            max_size: Maximum number of samples to keep
        """
        self.max_size = max_size
        self.buffer: deque = deque(maxlen=max_size)
    
    def add(self, confidence: float, correct: bool):
        """Add a sample to the buffer."""
        self.buffer.append((confidence, correct))
    
    def get_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get confidence and outcome arrays.
        
        Returns:
            (confidences, outcomes) arrays
        """
        if len(self.buffer) == 0:
            return np.array([]), np.array([])
        
        data = list(self.buffer)
        confidences = np.array([c for c, _ in data])
        outcomes = np.array([1.0 if o else 0.0 for _, o in data])
        return confidences, outcomes
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    def clear(self):
        """Clear the buffer."""
        self.buffer.clear()


class HybridCalibrator:
    """
    Hybrid calibrator combining Platt scaling and bucket calibration.
    
    Uses a weighted combination of both methods for robust calibration.
    Automatically refits when enough new samples are collected.
    """
    
    def __init__(self, config: CalibrationConfig = None):
        """
        Initialize hybrid calibrator.
        
        Args:
            config: Calibration configuration
        """
        self.config = config or CalibrationConfig()
        self.buffer = CalibrationBuffer(self.config.buffer_size)
        self.platt = PlattCalibrator()
        self.bucket = BucketCalibrator()
        self.samples_since_refit = 0
    
    def add_sample(self, confidence: float, correct: bool):
        """
        Add a sample and potentially refit calibrators.
        
        Args:
            confidence: Raw confidence score
            correct: Whether prediction was correct
        """
        self.buffer.add(confidence, correct)
        self.bucket.add_sample(confidence, correct)
        self.samples_since_refit += 1
        
        # Refit Platt scaling periodically
        if self.samples_since_refit >= self.config.refit_interval:
            self._refit()
            self.samples_since_refit = 0
    
    def _refit(self):
        """Refit Platt calibrator."""
        confidences, outcomes = self.buffer.get_arrays()
        if len(confidences) >= self.config.min_samples:
            self.platt.fit(confidences, outcomes)
    
    def calibrate(self, raw_confidence: float) -> float:
        """
        Calibrate a raw confidence score.
        
        Uses weighted combination of Platt and bucket calibration.
        
        Args:
            raw_confidence: Raw confidence (0-1)
            
        Returns:
            Calibrated confidence (0-1)
        """
        if len(self.buffer) < self.config.min_samples:
            return raw_confidence
        
        # Get both calibrations
        platt_conf = self.platt.calibrate(raw_confidence)
        bucket_conf = self.bucket.calibrate(raw_confidence)
        
        # Weighted combination
        if self.platt.is_fitted:
            calibrated = (
                self.config.platt_weight * platt_conf +
                self.config.isotonic_weight * bucket_conf
            )
        else:
            calibrated = bucket_conf
        
        return float(np.clip(calibrated, 0.0, 1.0))
    
    def get_metrics(self) -> Dict:
        """Get current calibration metrics."""
        confidences, outcomes = self.buffer.get_arrays()
        
        if len(confidences) < 20:
            return {
                'brier_score': 1.0,
                'ece': 1.0,
                'accuracy': 0.0,
                'sample_count': len(confidences),
                'is_calibrated': False,
                'platt_params': None
            }
        
        # Calculate metrics using calibrated confidences
        calibrated = np.array([self.calibrate(c) for c in confidences])
        metrics = CalibrationMetrics.compute_all(calibrated, outcomes, self.config)
        metrics['platt_params'] = self.platt.get_params()
        
        return metrics
    
    def get_stats(self) -> Dict:
        """Get detailed calibration statistics."""
        return {
            'sample_count': len(self.buffer),
            'platt_fitted': self.platt.is_fitted,
            'platt_params': self.platt.get_params(),
            'bucket_stats': self.bucket.get_bucket_stats(),
            'samples_since_refit': self.samples_since_refit
        }


__all__ = [
    'CalibrationConfig',
    'PlattCalibrator',
    'BucketCalibrator',
    'CalibrationMetrics',
    'CalibrationBuffer',
    'HybridCalibrator',
]








