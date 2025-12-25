#!/usr/bin/env python3
"""
Signal Calibration & Gating Layer
==================================

Improves trading win rate by:
1. Calibrating raw confidence scores using Platt scaling or isotonic regression
2. Tracking rolling Brier score and Expected Calibration Error (ECE)
3. Gating trades based on calibration quality
4. Filtering by multi-horizon agreement
5. Regime-aware filtering (volatility, time-of-day)

The goal is to filter out false positives and only trade when the model
is well-calibrated and multiple signals agree.
"""

import logging
import json
import sqlite3
import math
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class PredictionRecord:
    """Record of a prediction for calibration tracking."""
    timestamp: str
    symbol: str
    horizon_minutes: int  # e.g., 1, 5, 15, 30
    predicted_direction: str  # 'UP' or 'DOWN'
    raw_confidence: float  # 0.0 to 1.0
    calibrated_confidence: float  # 0.0 to 1.0 (after Platt scaling)
    predicted_return: float  # percentage
    entry_price: float
    # Filled after settlement
    actual_direction: Optional[str] = None
    actual_return: Optional[float] = None
    exit_price: Optional[float] = None
    settled: bool = False
    is_correct: Optional[bool] = None
    settlement_time: Optional[str] = None


@dataclass
class CalibrationMetrics:
    """Rolling calibration metrics."""
    brier_score: float = 1.0  # 0 = perfect, 1 = worst
    ece: float = 1.0  # Expected Calibration Error (0 = perfect)
    accuracy: float = 0.0  # Overall accuracy
    sample_count: int = 0
    last_update: Optional[str] = None
    # Per-confidence-bucket metrics for ECE
    bucket_accuracies: Dict[str, float] = field(default_factory=dict)
    bucket_counts: Dict[str, int] = field(default_factory=dict)


@dataclass 
class TradeGateResult:
    """Result of trade gating decision."""
    should_trade: bool
    calibrated_confidence: float
    raw_confidence: float
    rejection_reasons: List[str] = field(default_factory=list)
    gating_metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# PLATT SCALING CALIBRATOR
# =============================================================================

class PlattScalingCalibrator:
    """
    Platt scaling for confidence calibration.
    
    Maps raw confidence scores to calibrated probabilities using
    logistic regression: P(correct) = 1 / (1 + exp(A*x + B))
    
    This is trained on rolling historical predictions.
    """
    
    def __init__(self, min_samples: int = 50):
        self.min_samples = min_samples
        self.A = 0.0  # Scale parameter
        self.B = 0.0  # Bias parameter
        self.is_fitted = False
        self.fit_sample_count = 0
    
    def fit(self, confidences: np.ndarray, outcomes: np.ndarray) -> bool:
        """
        Fit Platt scaling parameters using gradient descent.
        
        Args:
            confidences: Raw confidence scores (0 to 1)
            outcomes: Binary outcomes (1 = correct, 0 = incorrect)
            
        Returns:
            True if fitting succeeded
        """
        if len(confidences) < self.min_samples:
            logger.debug(f"Platt scaling: insufficient samples ({len(confidences)} < {self.min_samples})")
            return False
        
        try:
            # Convert to numpy arrays
            x = np.array(confidences, dtype=np.float64)
            y = np.array(outcomes, dtype=np.float64)
            
            # Initialize parameters
            A, B = -1.0, 0.0
            
            # Gradient descent with line search
            learning_rate = 0.1
            max_iters = 1000
            tol = 1e-6
            
            for _ in range(max_iters):
                # Compute predictions
                z = A * x + B
                p = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
                
                # Compute gradients
                error = p - y
                grad_A = np.dot(error, x) / len(x)
                grad_B = np.mean(error)
                
                # Update parameters
                A_new = A - learning_rate * grad_A
                B_new = B - learning_rate * grad_B
                
                # Check convergence
                if abs(A_new - A) < tol and abs(B_new - B) < tol:
                    break
                
                A, B = A_new, B_new
            
            self.A = float(A)
            self.B = float(B)
            self.is_fitted = True
            self.fit_sample_count = len(confidences)
            
            logger.info(f"✅ Platt scaling fitted: A={self.A:.4f}, B={self.B:.4f} (n={len(confidences)})")
            return True
            
        except Exception as e:
            logger.error(f"Platt scaling fit error: {e}")
            return False
    
    def calibrate(self, raw_confidence: float) -> float:
        """
        Calibrate a raw confidence score.
        
        Args:
            raw_confidence: Raw confidence (0 to 1)
            
        Returns:
            Calibrated confidence (0 to 1)
        """
        if not self.is_fitted:
            return raw_confidence
        
        try:
            z = self.A * raw_confidence + self.B
            calibrated = 1.0 / (1.0 + math.exp(-max(-500, min(500, z))))
            return float(calibrated)
        except Exception:
            return raw_confidence


# =============================================================================
# ISOTONIC REGRESSION CALIBRATOR
# =============================================================================

class IsotonicCalibrator:
    """
    Isotonic regression for confidence calibration.
    
    Learns a monotonic mapping from raw confidence to calibrated probability.
    More flexible than Platt scaling but requires more data.
    """
    
    def __init__(self, min_samples: int = 100, n_bins: int = 10):
        self.min_samples = min_samples
        self.n_bins = n_bins
        self.bin_edges: List[float] = []
        self.bin_values: List[float] = []
        self.is_fitted = False
    
    def fit(self, confidences: np.ndarray, outcomes: np.ndarray) -> bool:
        """Fit isotonic regression."""
        if len(confidences) < self.min_samples:
            return False
        
        try:
            x = np.array(confidences, dtype=np.float64)
            y = np.array(outcomes, dtype=np.float64)
            
            # Sort by confidence
            sorted_indices = np.argsort(x)
            x_sorted = x[sorted_indices]
            y_sorted = y[sorted_indices]
            
            # Bin the data
            bin_size = max(1, len(x) // self.n_bins)
            self.bin_edges = [0.0]
            self.bin_values = []
            
            for i in range(0, len(x), bin_size):
                end = min(i + bin_size, len(x))
                if end > i:
                    self.bin_edges.append(float(x_sorted[end - 1]))
                    self.bin_values.append(float(np.mean(y_sorted[i:end])))
            
            # Ensure monotonicity (pool adjacent violators)
            for i in range(1, len(self.bin_values)):
                if self.bin_values[i] < self.bin_values[i - 1]:
                    # Pool with previous bin
                    pooled = (self.bin_values[i] + self.bin_values[i - 1]) / 2
                    self.bin_values[i] = pooled
                    self.bin_values[i - 1] = pooled
            
            self.is_fitted = True
            logger.info(f"✅ Isotonic calibrator fitted (n={len(confidences)}, bins={len(self.bin_values)})")
            return True
            
        except Exception as e:
            logger.error(f"Isotonic fit error: {e}")
            return False
    
    def calibrate(self, raw_confidence: float) -> float:
        """Calibrate using piecewise linear interpolation."""
        if not self.is_fitted or not self.bin_values:
            return raw_confidence
        
        try:
            # Find the bin
            for i in range(len(self.bin_edges) - 1):
                if raw_confidence <= self.bin_edges[i + 1]:
                    return self.bin_values[min(i, len(self.bin_values) - 1)]
            return self.bin_values[-1]
        except Exception:
            return raw_confidence


# =============================================================================
# CALIBRATION METRICS CALCULATOR
# =============================================================================

class CalibrationMetricsCalculator:
    """
    Calculate calibration metrics (Brier score, ECE) from prediction history.
    """
    
    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
    
    def calculate_brier_score(self, confidences: List[float], outcomes: List[bool]) -> float:
        """
        Calculate Brier score: mean squared error of probability predictions.
        
        Lower is better. 0 = perfect, 0.25 = random guessing, 1 = worst.
        """
        if not confidences or not outcomes:
            return 1.0
        
        brier = sum((c - (1.0 if o else 0.0)) ** 2 for c, o in zip(confidences, outcomes))
        return brier / len(confidences)
    
    def calculate_ece(self, confidences: List[float], outcomes: List[bool]) -> Tuple[float, Dict, Dict]:
        """
        Calculate Expected Calibration Error.
        
        ECE measures how well confidence scores match actual accuracy.
        Lower is better. 0 = perfectly calibrated.
        
        Returns:
            (ece, bucket_accuracies, bucket_counts)
        """
        if not confidences or not outcomes:
            return 1.0, {}, {}
        
        # Create bins
        bin_edges = np.linspace(0, 1, self.n_bins + 1)
        bucket_accs = {}
        bucket_counts = {}
        
        total_ece = 0.0
        total_samples = len(confidences)
        
        for i in range(self.n_bins):
            low, high = bin_edges[i], bin_edges[i + 1]
            bin_name = f"{low:.1f}-{high:.1f}"
            
            # Find samples in this bin
            mask = [(low <= c < high) or (i == self.n_bins - 1 and c == high) 
                    for c in confidences]
            bin_confs = [c for c, m in zip(confidences, mask) if m]
            bin_outcomes = [o for o, m in zip(outcomes, mask) if m]
            
            if bin_confs:
                avg_conf = sum(bin_confs) / len(bin_confs)
                avg_acc = sum(1 for o in bin_outcomes if o) / len(bin_outcomes)
                
                bucket_accs[bin_name] = avg_acc
                bucket_counts[bin_name] = len(bin_confs)
                
                # Weighted contribution to ECE
                total_ece += len(bin_confs) * abs(avg_conf - avg_acc)
            else:
                bucket_accs[bin_name] = 0.0
                bucket_counts[bin_name] = 0
        
        ece = total_ece / total_samples if total_samples > 0 else 1.0
        return ece, bucket_accs, bucket_counts
    
    def calculate_all(self, predictions: List[PredictionRecord]) -> CalibrationMetrics:
        """Calculate all calibration metrics from prediction records."""
        # Filter to settled predictions
        settled = [p for p in predictions if p.settled and p.is_correct is not None]
        
        if not settled:
            return CalibrationMetrics()
        
        confidences = [p.calibrated_confidence for p in settled]
        outcomes = [p.is_correct for p in settled]
        
        brier = self.calculate_brier_score(confidences, outcomes)
        ece, bucket_accs, bucket_counts = self.calculate_ece(confidences, outcomes)
        accuracy = sum(1 for o in outcomes if o) / len(outcomes)
        
        return CalibrationMetrics(
            brier_score=brier,
            ece=ece,
            accuracy=accuracy,
            sample_count=len(settled),
            last_update=datetime.now().isoformat(),
            bucket_accuracies=bucket_accs,
            bucket_counts=bucket_counts
        )


# =============================================================================
# SIGNAL CALIBRATION & GATING SYSTEM
# =============================================================================

class SignalCalibrationGate:
    """
    Main calibration and gating system.
    
    Responsibilities:
    1. Track prediction outcomes in rolling window
    2. Calibrate raw confidence using Platt scaling
    3. Calculate and monitor Brier score / ECE
    4. Gate trades based on calibration quality
    5. Filter by multi-horizon agreement
    6. Apply regime filters (volatility, time-of-day)
    """
    
    def __init__(
        self,
        rolling_window_size: int = 1000,
        min_calibration_samples: int = 50,
        max_brier_score: float = 0.35,  # Reject if Brier > this
        max_ece: float = 0.15,  # Reject if ECE > this
        min_confidence_threshold: float = 0.55,  # Minimum calibrated confidence
        require_horizon_agreement: bool = True,
        agreement_horizons: List[int] = None,  # e.g., [1, 5] for 1m and 5m
        db_path: str = "data/calibration.db"
    ):
        self.rolling_window_size = rolling_window_size
        self.min_calibration_samples = min_calibration_samples
        self.max_brier_score = max_brier_score
        self.max_ece = max_ece
        self.min_confidence_threshold = min_confidence_threshold
        self.require_horizon_agreement = require_horizon_agreement
        self.agreement_horizons = agreement_horizons or [1, 5]
        self.db_path = db_path
        
        # Rolling prediction history
        self.prediction_history: deque = deque(maxlen=rolling_window_size)
        
        # Calibrators per horizon
        self.platt_calibrators: Dict[int, PlattScalingCalibrator] = {}
        self.isotonic_calibrators: Dict[int, IsotonicCalibrator] = {}
        
        # Current calibration metrics per horizon
        self.calibration_metrics: Dict[int, CalibrationMetrics] = {}
        
        # Metrics calculator
        self.metrics_calculator = CalibrationMetricsCalculator()
        
        # Regime tracking
        self.recent_volatility: deque = deque(maxlen=60)  # Last 60 minutes
        self.hourly_hit_rates: Dict[int, float] = {}  # Hour -> hit rate
        
        # Trade limits
        self.trades_this_hour: int = 0
        self.last_trade_hour: int = -1
        self.max_trades_per_hour: int = 5
        
        # Initialize DB
        self._init_db()
        self._load_history()
    
    def _init_db(self):
        """Initialize SQLite database for persistence."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                symbol TEXT,
                horizon_minutes INTEGER,
                predicted_direction TEXT,
                raw_confidence REAL,
                calibrated_confidence REAL,
                predicted_return REAL,
                entry_price REAL,
                actual_direction TEXT,
                actual_return REAL,
                exit_price REAL,
                settled INTEGER DEFAULT 0,
                is_correct INTEGER,
                settlement_time TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS calibration_params (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                horizon_minutes INTEGER UNIQUE,
                platt_A REAL,
                platt_B REAL,
                updated_at TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_predictions_settled 
            ON predictions(settled, timestamp)
        ''')
        
        conn.commit()
        conn.close()
        
        logger.info(f"✅ Calibration database initialized: {self.db_path}")
    
    def _load_history(self):
        """Load recent prediction history from database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Load recent predictions
            cursor.execute('''
                SELECT timestamp, symbol, horizon_minutes, predicted_direction,
                       raw_confidence, calibrated_confidence, predicted_return,
                       entry_price, actual_direction, actual_return, exit_price,
                       settled, is_correct, settlement_time
                FROM predictions
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (self.rolling_window_size,))
            
            rows = cursor.fetchall()
            
            for row in reversed(rows):
                pred = PredictionRecord(
                    timestamp=row[0],
                    symbol=row[1],
                    horizon_minutes=row[2],
                    predicted_direction=row[3],
                    raw_confidence=row[4],
                    calibrated_confidence=row[5],
                    predicted_return=row[6],
                    entry_price=row[7],
                    actual_direction=row[8],
                    actual_return=row[9],
                    exit_price=row[10],
                    settled=bool(row[11]),
                    is_correct=bool(row[12]) if row[12] is not None else None,
                    settlement_time=row[13]
                )
                self.prediction_history.append(pred)
            
            # Load calibration params
            cursor.execute('SELECT horizon_minutes, platt_A, platt_B FROM calibration_params')
            for row in cursor.fetchall():
                horizon = row[0]
                self.platt_calibrators[horizon] = PlattScalingCalibrator()
                self.platt_calibrators[horizon].A = row[1]
                self.platt_calibrators[horizon].B = row[2]
                self.platt_calibrators[horizon].is_fitted = True
            
            conn.close()
            
            logger.info(f"📊 Loaded {len(self.prediction_history)} predictions from history")
            
            # Refit calibrators with loaded data
            self._refit_calibrators()
            
        except Exception as e:
            logger.warning(f"Could not load calibration history: {e}")
    
    def _save_prediction(self, pred: PredictionRecord):
        """Save prediction to database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO predictions (
                    timestamp, symbol, horizon_minutes, predicted_direction,
                    raw_confidence, calibrated_confidence, predicted_return,
                    entry_price, actual_direction, actual_return, exit_price,
                    settled, is_correct, settlement_time
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                pred.timestamp, pred.symbol, pred.horizon_minutes,
                pred.predicted_direction, pred.raw_confidence,
                pred.calibrated_confidence, pred.predicted_return,
                pred.entry_price, pred.actual_direction, pred.actual_return,
                pred.exit_price, int(pred.settled),
                int(pred.is_correct) if pred.is_correct is not None else None,
                pred.settlement_time
            ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"Could not save prediction: {e}")
    
    def _save_calibration_params(self, horizon: int):
        """Save calibration parameters to database."""
        if horizon not in self.platt_calibrators:
            return
        
        cal = self.platt_calibrators[horizon]
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO calibration_params (horizon_minutes, platt_A, platt_B, updated_at)
                VALUES (?, ?, ?, ?)
            ''', (horizon, cal.A, cal.B, datetime.now().isoformat()))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"Could not save calibration params: {e}")
    
    def _refit_calibrators(self):
        """Refit calibrators from prediction history."""
        # Group by horizon
        by_horizon: Dict[int, List[PredictionRecord]] = {}
        for pred in self.prediction_history:
            if pred.settled and pred.is_correct is not None:
                if pred.horizon_minutes not in by_horizon:
                    by_horizon[pred.horizon_minutes] = []
                by_horizon[pred.horizon_minutes].append(pred)
        
        # Fit calibrators for each horizon
        for horizon, preds in by_horizon.items():
            if len(preds) >= self.min_calibration_samples:
                confidences = np.array([p.raw_confidence for p in preds])
                outcomes = np.array([1.0 if p.is_correct else 0.0 for p in preds])
                
                # Platt scaling
                if horizon not in self.platt_calibrators:
                    self.platt_calibrators[horizon] = PlattScalingCalibrator(self.min_calibration_samples)
                self.platt_calibrators[horizon].fit(confidences, outcomes)
                self._save_calibration_params(horizon)
                
                # Isotonic (optional, needs more data)
                if len(preds) >= 100:
                    if horizon not in self.isotonic_calibrators:
                        self.isotonic_calibrators[horizon] = IsotonicCalibrator()
                    self.isotonic_calibrators[horizon].fit(confidences, outcomes)
                
                # Update metrics
                self.calibration_metrics[horizon] = self.metrics_calculator.calculate_all(preds)
    
    def record_prediction(
        self,
        symbol: str,
        horizon_minutes: int,
        predicted_direction: str,
        raw_confidence: float,
        predicted_return: float,
        entry_price: float,
        timestamp: Optional[str] = None
    ) -> PredictionRecord:
        """
        Record a new prediction for tracking.
        
        Args:
            symbol: Trading symbol
            horizon_minutes: Prediction horizon (1, 5, 15, 30)
            predicted_direction: 'UP' or 'DOWN'
            raw_confidence: Raw confidence score (0-1)
            predicted_return: Predicted return percentage
            entry_price: Price at prediction time
            timestamp: Optional timestamp override
            
        Returns:
            PredictionRecord
        """
        ts = timestamp or datetime.now().isoformat()
        
        # Calibrate confidence
        calibrated = self.calibrate_confidence(raw_confidence, horizon_minutes)
        
        pred = PredictionRecord(
            timestamp=ts,
            symbol=symbol,
            horizon_minutes=horizon_minutes,
            predicted_direction=predicted_direction,
            raw_confidence=raw_confidence,
            calibrated_confidence=calibrated,
            predicted_return=predicted_return,
            entry_price=entry_price
        )
        
        self.prediction_history.append(pred)
        self._save_prediction(pred)
        
        logger.debug(f"📝 Recorded {horizon_minutes}m prediction: {predicted_direction} @ {raw_confidence:.1%} -> {calibrated:.1%}")
        
        return pred
    
    def settle_predictions(self, current_price: float, current_time: Optional[datetime] = None):
        """
        Settle all predictions that have reached their horizon.
        
        Args:
            current_price: Current price for settlement
            current_time: Current time (uses now if not provided)
        """
        now = current_time or datetime.now()
        settled_count = 0
        
        for pred in self.prediction_history:
            if pred.settled:
                continue
            
            try:
                pred_time = datetime.fromisoformat(pred.timestamp)
                elapsed_minutes = (now - pred_time).total_seconds() / 60
                
                # Check if horizon has elapsed
                if elapsed_minutes >= pred.horizon_minutes:
                    # Settle the prediction
                    pred.exit_price = current_price
                    pred.actual_return = ((current_price - pred.entry_price) / pred.entry_price) * 100
                    pred.actual_direction = 'UP' if current_price > pred.entry_price else 'DOWN'
                    pred.is_correct = (pred.predicted_direction == pred.actual_direction)
                    pred.settled = True
                    pred.settlement_time = now.isoformat()
                    
                    settled_count += 1
                    
                    # Update DB
                    self._update_settled_prediction(pred)
                    
            except Exception as e:
                logger.debug(f"Could not settle prediction: {e}")
        
        if settled_count > 0:
            logger.info(f"✅ Settled {settled_count} predictions at ${current_price:.2f}")
            # Refit calibrators periodically
            if len([p for p in self.prediction_history if p.settled]) % 50 == 0:
                self._refit_calibrators()
    
    def _update_settled_prediction(self, pred: PredictionRecord):
        """Update a settled prediction in the database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE predictions SET
                    actual_direction = ?,
                    actual_return = ?,
                    exit_price = ?,
                    settled = 1,
                    is_correct = ?,
                    settlement_time = ?
                WHERE timestamp = ? AND horizon_minutes = ?
            ''', (
                pred.actual_direction, pred.actual_return, pred.exit_price,
                int(pred.is_correct) if pred.is_correct is not None else None,
                pred.settlement_time, pred.timestamp, pred.horizon_minutes
            ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"Could not update settled prediction: {e}")
    
    def calibrate_confidence(self, raw_confidence: float, horizon_minutes: int = 15) -> float:
        """
        Calibrate a raw confidence score.
        
        Args:
            raw_confidence: Raw confidence (0-1)
            horizon_minutes: Prediction horizon
            
        Returns:
            Calibrated confidence (0-1)
        """
        # Try Platt scaling first
        if horizon_minutes in self.platt_calibrators and self.platt_calibrators[horizon_minutes].is_fitted:
            return self.platt_calibrators[horizon_minutes].calibrate(raw_confidence)
        
        # Fall back to nearest horizon
        for h in sorted(self.platt_calibrators.keys(), key=lambda x: abs(x - horizon_minutes)):
            if self.platt_calibrators[h].is_fitted:
                return self.platt_calibrators[h].calibrate(raw_confidence)
        
        # No calibration available - return raw
        return raw_confidence
    
    def get_calibration_quality(self, horizon_minutes: int = 15) -> Dict[str, float]:
        """
        Get current calibration quality metrics.
        
        Returns:
            Dict with brier_score, ece, accuracy, sample_count
        """
        if horizon_minutes in self.calibration_metrics:
            m = self.calibration_metrics[horizon_minutes]
            return {
                'brier_score': m.brier_score,
                'ece': m.ece,
                'accuracy': m.accuracy,
                'sample_count': m.sample_count,
                'is_good': m.brier_score <= self.max_brier_score and m.ece <= self.max_ece
            }
        
        # Calculate from history
        preds = [p for p in self.prediction_history 
                 if p.settled and p.horizon_minutes == horizon_minutes]
        if preds:
            metrics = self.metrics_calculator.calculate_all(preds)
            self.calibration_metrics[horizon_minutes] = metrics
            return {
                'brier_score': metrics.brier_score,
                'ece': metrics.ece,
                'accuracy': metrics.accuracy,
                'sample_count': metrics.sample_count,
                'is_good': metrics.brier_score <= self.max_brier_score and metrics.ece <= self.max_ece
            }
        
        return {
            'brier_score': 1.0,
            'ece': 1.0,
            'accuracy': 0.0,
            'sample_count': 0,
            'is_good': False
        }
    
    def check_horizon_agreement(
        self,
        multi_timeframe_predictions: Dict[str, Dict]
    ) -> Tuple[bool, str, Dict]:
        """
        Check if multiple horizons agree on direction.
        
        Args:
            multi_timeframe_predictions: Dict mapping timeframe names to prediction dicts
                                         e.g., {'5min': {'direction': 'UP', ...}, ...}
        
        Returns:
            (agrees, dominant_direction, details)
        """
        if not self.require_horizon_agreement:
            return True, 'N/A', {}
        
        # Map timeframe names to minutes
        tf_to_minutes = {
            '1min': 1, '5min': 5, '10min': 10, '15min': 15,
            '20min': 20, '30min': 30, '1h': 60, '1hour': 60
        }
        
        directions = {}
        for tf_name, pred in multi_timeframe_predictions.items():
            if tf_name in tf_to_minutes:
                minutes = tf_to_minutes[tf_name]
                if minutes in self.agreement_horizons:
                    direction = pred.get('direction', 'N/A')
                    if direction in ['UP', 'DOWN']:
                        directions[minutes] = direction
        
        if len(directions) < len(self.agreement_horizons):
            return False, 'INSUFFICIENT', {'found': list(directions.keys()), 'required': self.agreement_horizons}
        
        # Check agreement
        unique_directions = set(directions.values())
        if len(unique_directions) == 1:
            dominant = list(unique_directions)[0]
            return True, dominant, {'directions': directions, 'agreement': 'FULL'}
        
        return False, 'MIXED', {'directions': directions, 'agreement': 'NONE'}
    
    def update_volatility(self, current_volatility: float):
        """Track recent volatility for regime filtering."""
        self.recent_volatility.append(current_volatility)
    
    def get_volatility_regime(self) -> str:
        """
        Get current volatility regime.
        
        Returns:
            'LOW', 'NORMAL', or 'HIGH'
        """
        if len(self.recent_volatility) < 10:
            return 'UNKNOWN'
        
        avg_vol = sum(self.recent_volatility) / len(self.recent_volatility)
        
        if avg_vol < 12:  # VIX < 12
            return 'LOW'
        elif avg_vol > 25:  # VIX > 25
            return 'HIGH'
        return 'NORMAL'
    
    def get_time_of_day_hit_rate(self, hour: int) -> float:
        """Get historical hit rate for a specific hour."""
        if hour in self.hourly_hit_rates:
            return self.hourly_hit_rates[hour]
        
        # Calculate from history
        hour_preds = [p for p in self.prediction_history 
                      if p.settled and datetime.fromisoformat(p.timestamp).hour == hour]
        
        if len(hour_preds) >= 10:
            hit_rate = sum(1 for p in hour_preds if p.is_correct) / len(hour_preds)
            self.hourly_hit_rates[hour] = hit_rate
            return hit_rate
        
        return 0.5  # Default
    
    def should_trade(
        self,
        signal: Dict,
        multi_timeframe_predictions: Dict[str, Dict],
        current_price: float,
        current_vix: float = 17.0,
        expected_return_pct: float = 0.0,
        fees_slippage_pct: float = 0.1
    ) -> TradeGateResult:
        """
        Main gating function: Should we take this trade?
        
        Args:
            signal: Trading signal dict with 'action', 'confidence', etc.
            multi_timeframe_predictions: Multi-horizon predictions
            current_price: Current price
            current_vix: Current VIX level
            expected_return_pct: Expected return percentage
            fees_slippage_pct: Fees + slippage as percentage
            
        Returns:
            TradeGateResult with decision and reasons
        """
        action = signal.get('action', 'HOLD')
        raw_confidence = signal.get('confidence', 0)
        rejection_reasons = []
        metadata = {}
        
        # 1. Basic action check
        if action == 'HOLD':
            return TradeGateResult(
                should_trade=False,
                calibrated_confidence=raw_confidence,
                raw_confidence=raw_confidence,
                rejection_reasons=['Signal is HOLD']
            )
        
        # 2. Calibrate confidence
        calibrated_confidence = self.calibrate_confidence(raw_confidence)
        metadata['raw_confidence'] = raw_confidence
        metadata['calibrated_confidence'] = calibrated_confidence
        
        # 3. Minimum confidence check
        if calibrated_confidence < self.min_confidence_threshold:
            rejection_reasons.append(
                f'Calibrated confidence too low ({calibrated_confidence:.1%} < {self.min_confidence_threshold:.1%})'
            )
        
        # 4. Calibration quality check
        quality = self.get_calibration_quality()
        metadata['calibration_quality'] = quality
        
        if quality['sample_count'] >= self.min_calibration_samples:
            if quality['brier_score'] > self.max_brier_score:
                rejection_reasons.append(
                    f'Poor Brier score ({quality["brier_score"]:.3f} > {self.max_brier_score:.3f})'
                )
            if quality['ece'] > self.max_ece:
                rejection_reasons.append(
                    f'High ECE ({quality["ece"]:.3f} > {self.max_ece:.3f})'
                )
        
        # 5. Multi-horizon agreement
        if multi_timeframe_predictions:
            agrees, dominant_dir, agreement_details = self.check_horizon_agreement(multi_timeframe_predictions)
            metadata['horizon_agreement'] = agreement_details
            
            if not agrees:
                rejection_reasons.append(f'Horizons disagree: {agreement_details}')
        
        # 6. Expected edge check
        if expected_return_pct < fees_slippage_pct + 0.05:  # Need 5bps buffer
            rejection_reasons.append(
                f'Expected return too low ({expected_return_pct:.2%} < {fees_slippage_pct + 0.05:.2%})'
            )
        
        # 7. Volatility regime filter
        self.update_volatility(current_vix)
        vol_regime = self.get_volatility_regime()
        metadata['volatility_regime'] = vol_regime
        
        # Skip trades in extreme volatility regimes (optional)
        if vol_regime == 'HIGH' and calibrated_confidence < 0.70:
            rejection_reasons.append(f'High vol regime requires higher confidence')
        
        # 8. Time-of-day filter
        current_hour = datetime.now().hour
        hour_hit_rate = self.get_time_of_day_hit_rate(current_hour)
        metadata['hour_hit_rate'] = hour_hit_rate
        
        if hour_hit_rate < 0.35:  # Skip hours with <35% historical hit rate
            rejection_reasons.append(f'Poor historical hit rate for hour {current_hour} ({hour_hit_rate:.1%})')
        
        # 9. Trades per hour limit
        if current_hour != self.last_trade_hour:
            self.trades_this_hour = 0
            self.last_trade_hour = current_hour
        
        if self.trades_this_hour >= self.max_trades_per_hour:
            rejection_reasons.append(f'Max trades per hour reached ({self.max_trades_per_hour})')
        
        # Decision
        should_trade = len(rejection_reasons) == 0
        
        if should_trade:
            self.trades_this_hour += 1
        
        return TradeGateResult(
            should_trade=should_trade,
            calibrated_confidence=calibrated_confidence,
            raw_confidence=raw_confidence,
            rejection_reasons=rejection_reasons,
            gating_metadata=metadata
        )
    
    def log_diagnostics(
        self,
        gate_result: TradeGateResult,
        signal: Dict,
        current_price: float
    ):
        """
        Log detailed diagnostics for analysis.
        
        This helps identify where the system fails and why.
        """
        now = datetime.now()
        
        log_entry = {
            'timestamp': now.isoformat(),
            'price': current_price,
            'action': signal.get('action', 'UNKNOWN'),
            'raw_confidence': gate_result.raw_confidence,
            'calibrated_confidence': gate_result.calibrated_confidence,
            'should_trade': gate_result.should_trade,
            'rejection_reasons': gate_result.rejection_reasons,
            'metadata': gate_result.gating_metadata
        }
        
        # Log level based on outcome
        if gate_result.should_trade:
            logger.info(f"✅ TRADE APPROVED: {signal.get('action')} @ {gate_result.calibrated_confidence:.1%}")
        else:
            reasons_str = '; '.join(gate_result.rejection_reasons[:3])  # First 3 reasons
            logger.info(f"🚫 TRADE REJECTED: {reasons_str}")
        
        # Detailed debug log
        logger.debug(f"Gate diagnostics: {json.dumps(log_entry, default=str)}")
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for monitoring."""
        settled = [p for p in self.prediction_history if p.settled]
        
        if not settled:
            return {
                'total_predictions': len(self.prediction_history),
                'settled_predictions': 0,
                'overall_accuracy': 0.0,
                'calibration_quality': {},
                'trades_this_hour': self.trades_this_hour
            }
        
        correct = sum(1 for p in settled if p.is_correct)
        
        return {
            'total_predictions': len(self.prediction_history),
            'settled_predictions': len(settled),
            'overall_accuracy': correct / len(settled),
            'calibration_quality': {h: asdict(m) for h, m in self.calibration_metrics.items()},
            'hourly_hit_rates': dict(self.hourly_hit_rates),
            'volatility_regime': self.get_volatility_regime(),
            'trades_this_hour': self.trades_this_hour,
            'platt_fitted': {h: c.is_fitted for h, c in self.platt_calibrators.items()}
        }


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_calibration_gate(config: Optional[Dict] = None) -> SignalCalibrationGate:
    """
    Create a SignalCalibrationGate with configuration.
    
    Args:
        config: Optional config dict (loads from config.json if not provided)
        
    Returns:
        Configured SignalCalibrationGate instance
    """
    # Default settings
    settings = {
        'rolling_window_size': 1000,
        'min_calibration_samples': 50,
        'max_brier_score': 0.35,
        'max_ece': 0.15,
        'min_confidence_threshold': 0.55,
        'require_horizon_agreement': True,
        'agreement_horizons': [1, 5],
        'max_trades_per_hour': 5,
        'db_path': 'data/calibration.db'
    }
    
    # Load from config.json if available
    try:
        config_path = Path('config.json')
        if config_path.exists():
            with open(config_path, 'r') as f:
                cfg = json.load(f)
            
            cal_cfg = cfg.get('calibration', {})
            settings.update(cal_cfg)
    except Exception as e:
        logger.warning(f"Could not load calibration config: {e}")
    
    # Override with provided config
    if config:
        settings.update(config)
    
    return SignalCalibrationGate(
        rolling_window_size=settings['rolling_window_size'],
        min_calibration_samples=settings['min_calibration_samples'],
        max_brier_score=settings['max_brier_score'],
        max_ece=settings['max_ece'],
        min_confidence_threshold=settings['min_confidence_threshold'],
        require_horizon_agreement=settings['require_horizon_agreement'],
        agreement_horizons=settings['agreement_horizons'],
        db_path=settings['db_path']
    )

