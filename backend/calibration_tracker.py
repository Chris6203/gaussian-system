#!/usr/bin/env python3
"""
Calibration Tracker
===================

Enhanced module for confidence calibration using Platt scaling.

Design principles:
1. DUAL CALIBRATION - Track both direction accuracy AND option PnL outcomes
2. HORIZON AWARENESS - Different horizons for direction (15min) vs PnL (actual hold time)
3. PNL-GATED - Primary trading gate uses PnL probability, not just directional
4. MINIMAL STATE - Buffers of (confidence, outcome) pairs for each calibrator

CRITICAL FIX: Issue 1 - Objective/Horizon Misalignment
-------------------------------------------------------
The neural net predicts direction over 15 minutes, but options are held 45-120 minutes.
A correct direction prediction can still lose money due to IV crush, time decay, gamma.

Solution:
- direction_calibrator: Calibrates P(direction correct at 15min)
- pnl_calibrator: Calibrates P(option PnL > 0 at actual exit time)
- Use pnl_calibrator as PRIMARY gate for trade decisions

Usage:
    tracker = CalibrationTracker(direction_horizon=15, pnl_horizon=60)
    
    # At signal generation time:
    tracker.record_prediction(
        confidence=0.65,
        predicted_direction='UP',
        entry_price=598.50
    )
    
    # Every cycle:
    tracker.settle_predictions(current_price=599.20)
    
    # When trade closes, record actual PnL outcome:
    tracker.record_pnl_outcome(
        trade_id=trade.id,
        option_pnl=actual_pnl,
        hold_minutes=elapsed_minutes
    )
    
    # When making trade decisions, use PnL-calibrated confidence:
    pnl_calibrated = tracker.calibrate_pnl(raw_confidence=0.65)
    direction_calibrated = tracker.calibrate(raw_confidence=0.65)
    metrics = tracker.get_metrics()
"""

import logging
import math
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class PendingPrediction:
    """A prediction waiting to be settled."""
    timestamp: datetime
    confidence: float
    predicted_direction: str  # 'UP' or 'DOWN'
    entry_price: float
    trade_id: Optional[str] = None  # Link to trade for PnL tracking


@dataclass
class PendingPnLOutcome:
    """A trade waiting for PnL outcome."""
    trade_id: str
    timestamp: datetime
    confidence: float
    predicted_direction: str
    entry_price: float
    entry_option_price: Optional[float] = None


@dataclass 
class PlattParams:
    """Platt scaling parameters."""
    A: float = 0.0
    B: float = 0.0
    n_samples: int = 0
    last_fit: Optional[datetime] = None


@dataclass
class IsotonicParams:
    """Isotonic regression parameters."""
    x_points: List[float] = field(default_factory=list)
    y_points: List[float] = field(default_factory=list)
    n_samples: int = 0
    last_fit: Optional[datetime] = None


class CalibrationTracker:
    """
    Tracks confidence calibration for BOTH direction accuracy AND option PnL outcomes.
    
    Uses HYBRID calibration combining:
    1. Platt scaling (parametric, good for small datasets)
    2. Isotonic regression (non-parametric, flexible)
    3. Ensemble voting for final calibrated confidence
    
    TWO calibrators:
    - direction_calibrator: Based on direction accuracy (15 min horizon)
    - pnl_calibrator: Based on actual option P&L (actual hold time)
    
    PRIMARY GATE should use pnl_calibrator for trade decisions.
    """
    
    def __init__(
        self,
        horizon_minutes: int = 15,  # Direction horizon (backward compat alias)
        direction_horizon: int = None,  # Explicit direction horizon
        pnl_horizon: int = 60,  # Typical hold time for PnL evaluation
        buffer_size: int = 1000,
        min_samples_for_calibration: int = 50,
        refit_interval: int = 50,
        use_hybrid: bool = True,
        ensemble_weights: Tuple[float, float] = (0.4, 0.6)  # (platt, isotonic)
    ):
        """
        Args:
            horizon_minutes: Legacy alias for direction_horizon
            direction_horizon: How long to wait before settling direction predictions
            pnl_horizon: Expected hold time for PnL-based calibration
            buffer_size: Max number of (confidence, outcome) pairs to keep
            min_samples_for_calibration: Minimum settled predictions before calibration
            refit_interval: Refit calibration every N new settlements
            use_hybrid: If True, use Platt + Isotonic ensemble; if False, Platt only
            ensemble_weights: (platt_weight, isotonic_weight) for ensemble voting
        """
        # Handle legacy parameter
        self.direction_horizon = direction_horizon or horizon_minutes
        self.pnl_horizon = pnl_horizon
        self.horizon_minutes = self.direction_horizon  # Backward compat
        
        self.buffer_size = buffer_size
        self.min_samples_for_calibration = min_samples_for_calibration
        self.refit_interval = refit_interval
        self.use_hybrid = use_hybrid
        self.ensemble_weights = ensemble_weights
        
        # =====================================================================
        # DIRECTION CALIBRATOR (legacy behavior)
        # =====================================================================
        self._pending: List[PendingPrediction] = []
        self._calibration_buffer: deque = deque(maxlen=buffer_size)
        self._platt = PlattParams()
        self._isotonic = IsotonicParams()
        self._settlements_since_fit = 0
        
        # =====================================================================
        # PNL CALIBRATOR (NEW - Issue 1 fix)
        # =====================================================================
        self._pending_pnl: Dict[str, PendingPnLOutcome] = {}  # trade_id -> outcome
        self._pnl_calibration_buffer: deque = deque(maxlen=buffer_size)
        self._pnl_platt = PlattParams()
        self._pnl_isotonic = IsotonicParams()
        self._pnl_settlements_since_fit = 0
        
        # Validation set for preventing overfitting (last 20% of data)
        self._validation_split = 0.2
        
        # Stats
        self._total_recorded = 0
        self._total_settled = 0
        self._total_pnl_recorded = 0
        self._total_pnl_settled = 0
        self._calibration_method = 'none'  # 'platt', 'isotonic', 'hybrid'
        self._pnl_calibration_method = 'none'
        
        logger.info(f"✅ CalibrationTracker initialized: direction={self.direction_horizon}m, pnl={pnl_horizon}m, hybrid={use_hybrid}")
        logger.info(f"   PRIMARY GATE: Use pnl-calibrated confidence (calibrate_pnl method)")
    
    # =========================================================================
    # DIRECTION CALIBRATION (legacy)
    # =========================================================================
    
    def record_prediction(
        self,
        confidence: float,
        predicted_direction: str,
        entry_price: float,
        timestamp: Optional[datetime] = None,
        trade_id: Optional[str] = None
    ) -> None:
        """
        Record a prediction to be settled later.
        
        Call this at signal generation time.
        
        Args:
            confidence: Model confidence (0-1)
            predicted_direction: 'UP' or 'DOWN'
            entry_price: Price when prediction was made
            timestamp: When prediction was made (default: now)
            trade_id: Optional trade ID for PnL tracking
        """
        if predicted_direction not in ('UP', 'DOWN'):
            return  # Ignore HOLD/NEUTRAL
        
        pred = PendingPrediction(
            timestamp=timestamp or datetime.now(),
            confidence=confidence,
            predicted_direction=predicted_direction,
            entry_price=entry_price,
            trade_id=trade_id
        )
        self._pending.append(pred)
        self._total_recorded += 1
        
        # Keep pending buffer reasonable
        if len(self._pending) > 500:
            self._pending = self._pending[-500:]
    
    def settle_predictions(self, current_price: float, current_time: Optional[datetime] = None) -> int:
        """
        Settle any predictions that have reached their horizon.
        
        Call this every cycle.
        
        Args:
            current_price: Current market price
            current_time: Current time (default: now)
            
        Returns:
            Number of predictions settled this call
        """
        now = current_time or datetime.now()
        settled_count = 0
        to_remove = []
        
        for pred in self._pending:
            elapsed = (now - pred.timestamp).total_seconds() / 60
            
            if elapsed >= self.direction_horizon:
                # Settle this prediction
                actual_direction = 'UP' if current_price > pred.entry_price else 'DOWN'
                direction_correct = (pred.predicted_direction == actual_direction)
                
                # Add to calibration buffer
                self._calibration_buffer.append((pred.confidence, direction_correct))
                
                to_remove.append(pred)
                settled_count += 1
                self._total_settled += 1
                self._settlements_since_fit += 1
        
        # Remove settled predictions
        for pred in to_remove:
            self._pending.remove(pred)
        
        # Refit calibration if needed
        if self._settlements_since_fit >= self.refit_interval:
            if len(self._calibration_buffer) >= self.min_samples_for_calibration:
                self._fit_calibration()
                self._settlements_since_fit = 0
        
        if settled_count > 0:
            logger.debug(f"[CALIBRATION] Settled {settled_count} direction predictions | Buffer: {len(self._calibration_buffer)}")
        
        return settled_count
    
    def calibrate(self, raw_confidence: float) -> float:
        """
        Apply hybrid calibration (Platt + Isotonic ensemble) for DIRECTION.
        
        Args:
            raw_confidence: Model's raw confidence (0-1)
            
        Returns:
            Calibrated confidence (0-1)
        """
        # Not enough data yet
        if self._platt.n_samples == 0 and self._isotonic.n_samples == 0:
            return raw_confidence
        
        platt_conf = None
        isotonic_conf = None
        
        # Get Platt calibration
        if self._platt.n_samples > 0:
            try:
                z = self._platt.A * raw_confidence + self._platt.B
                platt_conf = 1.0 / (1.0 + math.exp(-max(-500, min(500, z))))
            except (OverflowError, ValueError):
                pass  # Numerical issues, fall back to isotonic or raw
        
        # Get Isotonic calibration
        if self._isotonic.n_samples > 0 and self._isotonic.x_points:
            try:
                isotonic_conf = self._interpolate_isotonic(raw_confidence)
            except (IndexError, ValueError):
                pass  # Interpolation failed, use Platt or raw
        
        # Ensemble voting
        if self.use_hybrid and platt_conf is not None and isotonic_conf is not None:
            w_platt, w_isotonic = self.ensemble_weights
            calibrated = w_platt * platt_conf + w_isotonic * isotonic_conf
            self._calibration_method = 'hybrid'
        elif platt_conf is not None:
            calibrated = platt_conf
            self._calibration_method = 'platt'
        elif isotonic_conf is not None:
            calibrated = isotonic_conf
            self._calibration_method = 'isotonic'
        else:
            return raw_confidence
        
        # Clamp to valid probability range
        return float(max(0.0, min(1.0, calibrated)))
    
    # =========================================================================
    # PNL CALIBRATION (NEW - Issue 1 fix)
    # =========================================================================
    
    def record_trade_entry(
        self,
        trade_id: str,
        confidence: float,
        predicted_direction: str,
        entry_price: float,
        entry_option_price: Optional[float] = None,
        timestamp: Optional[datetime] = None
    ) -> None:
        """
        Record a trade entry for PnL-based calibration.
        
        Call this when a trade is ACTUALLY placed (not just a signal).
        
        Args:
            trade_id: Unique trade identifier
            confidence: Model confidence at entry (0-1)
            predicted_direction: 'UP' or 'DOWN'
            entry_price: Underlying price at entry
            entry_option_price: Option price at entry (for PnL calculation)
            timestamp: Entry time (default: now)
        """
        if predicted_direction not in ('UP', 'DOWN'):
            return
        
        outcome = PendingPnLOutcome(
            trade_id=trade_id,
            timestamp=timestamp or datetime.now(),
            confidence=confidence,
            predicted_direction=predicted_direction,
            entry_price=entry_price,
            entry_option_price=entry_option_price
        )
        self._pending_pnl[trade_id] = outcome
        self._total_pnl_recorded += 1
        
        logger.debug(f"[PNL_CAL] Recorded trade entry: {trade_id} ({predicted_direction} @ conf={confidence:.1%})")
    
    def record_pnl_outcome(
        self,
        trade_id: str,
        option_pnl: float,
        hold_minutes: Optional[float] = None,
        exit_option_price: Optional[float] = None
    ) -> bool:
        """
        Record the actual P&L outcome of a trade.
        
        Call this when a trade EXITS (regardless of reason).
        
        Args:
            trade_id: Trade identifier (must match record_trade_entry)
            option_pnl: Actual option P&L (positive = profit, negative = loss)
            hold_minutes: How long the trade was held
            exit_option_price: Option price at exit
            
        Returns:
            True if outcome was recorded, False if trade_id not found
        """
        if trade_id not in self._pending_pnl:
            logger.debug(f"[PNL_CAL] Trade {trade_id} not found in pending PnL tracker")
            return False
        
        entry = self._pending_pnl.pop(trade_id)
        
        # Record to PnL calibration buffer: (confidence, was_profitable)
        was_profitable = option_pnl > 0
        self._pnl_calibration_buffer.append((entry.confidence, was_profitable))
        
        self._total_pnl_settled += 1
        self._pnl_settlements_since_fit += 1
        
        # Log for debugging
        outcome_str = "✓ WIN" if was_profitable else "✗ LOSS"
        logger.debug(f"[PNL_CAL] Settled: {trade_id} {outcome_str} | PnL: ${option_pnl:.2f} | Conf: {entry.confidence:.1%}")
        
        # Refit PnL calibration if needed
        if self._pnl_settlements_since_fit >= self.refit_interval:
            if len(self._pnl_calibration_buffer) >= self.min_samples_for_calibration:
                self._fit_pnl_calibration()
                self._pnl_settlements_since_fit = 0
        
        return True
    
    def calibrate_pnl(self, raw_confidence: float) -> float:
        """
        Apply hybrid calibration for PNL PROBABILITY.
        
        THIS IS THE PRIMARY GATE for trade decisions.
        Returns P(option PnL > 0 | confidence).
        
        Args:
            raw_confidence: Model's raw confidence (0-1)
            
        Returns:
            Calibrated probability of profit (0-1)
        """
        # Not enough PnL data yet - fall back to direction calibration
        if self._pnl_platt.n_samples == 0 and self._pnl_isotonic.n_samples == 0:
            # Use direction calibration as fallback
            return self.calibrate(raw_confidence)
        
        platt_conf = None
        isotonic_conf = None
        
        # Get Platt calibration
        if self._pnl_platt.n_samples > 0:
            try:
                z = self._pnl_platt.A * raw_confidence + self._pnl_platt.B
                platt_conf = 1.0 / (1.0 + math.exp(-max(-500, min(500, z))))
            except (OverflowError, ValueError):
                pass
        
        # Get Isotonic calibration
        if self._pnl_isotonic.n_samples > 0 and self._pnl_isotonic.x_points:
            try:
                isotonic_conf = self._interpolate_isotonic(raw_confidence, use_pnl=True)
            except (IndexError, ValueError):
                pass
        
        # Ensemble voting
        if self.use_hybrid and platt_conf is not None and isotonic_conf is not None:
            w_platt, w_isotonic = self.ensemble_weights
            calibrated = w_platt * platt_conf + w_isotonic * isotonic_conf
            self._pnl_calibration_method = 'hybrid'
        elif platt_conf is not None:
            calibrated = platt_conf
            self._pnl_calibration_method = 'platt'
        elif isotonic_conf is not None:
            calibrated = isotonic_conf
            self._pnl_calibration_method = 'isotonic'
        else:
            return self.calibrate(raw_confidence)  # Fallback to direction
        
        return float(max(0.0, min(1.0, calibrated)))
    
    def _interpolate_isotonic(self, raw_confidence: float, use_pnl: bool = False) -> float:
        """Interpolate isotonic regression for a given confidence."""
        iso = self._pnl_isotonic if use_pnl else self._isotonic
        x_pts = iso.x_points
        y_pts = iso.y_points
        
        if not x_pts or not y_pts:
            return raw_confidence
        
        # Edge cases
        if raw_confidence <= x_pts[0]:
            return y_pts[0]
        if raw_confidence >= x_pts[-1]:
            return y_pts[-1]
        
        # Linear interpolation between points
        for i in range(len(x_pts) - 1):
            if x_pts[i] <= raw_confidence <= x_pts[i + 1]:
                # Linear interpolation
                t = (raw_confidence - x_pts[i]) / (x_pts[i + 1] - x_pts[i] + 1e-10)
                return y_pts[i] + t * (y_pts[i + 1] - y_pts[i])
        
        return raw_confidence
    
    # =========================================================================
    # METRICS
    # =========================================================================
    
    def get_metrics(self) -> Dict:
        """
        Calculate calibration metrics for BOTH direction and PnL calibrators.
        
        Returns:
            Dict with brier_score, ece, direction_accuracy, pnl_metrics, etc.
        """
        result = {
            # Direction calibrator metrics
            'brier_score': 1.0,
            'ece': 1.0,
            'direction_accuracy': 0.0,
            'sample_count': len(self._calibration_buffer),
            'is_calibrated': False,
            'horizon_minutes': self.direction_horizon,
            'pending_count': len(self._pending),
            'calibration_method': self._calibration_method,
            'use_hybrid': self.use_hybrid,
            'platt_fitted': self._platt.n_samples > 0,
            'platt_A': self._platt.A,
            'platt_B': self._platt.B,
            'isotonic_fitted': self._isotonic.n_samples > 0,
            'isotonic_points': len(self._isotonic.x_points),
            'total_recorded': self._total_recorded,
            'total_settled': self._total_settled,
            
            # PnL calibrator metrics (NEW)
            'pnl_sample_count': len(self._pnl_calibration_buffer),
            'pnl_win_rate': 0.0,
            'pnl_brier_score': 1.0,
            'pnl_ece': 1.0,
            'pnl_is_calibrated': False,
            'pnl_horizon_minutes': self.pnl_horizon,
            'pnl_pending_count': len(self._pending_pnl),
            'pnl_calibration_method': self._pnl_calibration_method,
            'pnl_platt_fitted': self._pnl_platt.n_samples > 0,
            'pnl_platt_A': self._pnl_platt.A,
            'pnl_platt_B': self._pnl_platt.B,
            'pnl_isotonic_fitted': self._pnl_isotonic.n_samples > 0,
            'pnl_isotonic_points': len(self._pnl_isotonic.x_points),
            'pnl_total_recorded': self._total_pnl_recorded,
            'pnl_total_settled': self._total_pnl_settled,
        }
        
        # Calculate direction metrics
        if len(self._calibration_buffer) >= 20:
            data = list(self._calibration_buffer)
            confidences = [self.calibrate(c) for c, _ in data]
            outcomes = [1.0 if correct else 0.0 for _, correct in data]
            
            result['brier_score'] = sum((c - o) ** 2 for c, o in zip(confidences, outcomes)) / len(data)
            result['ece'] = self._calculate_ece(confidences, outcomes)
            result['direction_accuracy'] = sum(outcomes) / len(outcomes)
            result['is_calibrated'] = (
                result['brier_score'] <= 0.35 and 
                result['ece'] <= 0.15 and 
                len(data) >= self.min_samples_for_calibration
            )
        
        # Calculate PnL metrics
        if len(self._pnl_calibration_buffer) >= 20:
            pnl_data = list(self._pnl_calibration_buffer)
            pnl_confidences = [self.calibrate_pnl(c) for c, _ in pnl_data]
            pnl_outcomes = [1.0 if profitable else 0.0 for _, profitable in pnl_data]
            
            result['pnl_win_rate'] = sum(pnl_outcomes) / len(pnl_outcomes)
            result['pnl_brier_score'] = sum((c - o) ** 2 for c, o in zip(pnl_confidences, pnl_outcomes)) / len(pnl_data)
            result['pnl_ece'] = self._calculate_ece(pnl_confidences, pnl_outcomes)
            result['pnl_is_calibrated'] = (
                result['pnl_brier_score'] <= 0.35 and 
                result['pnl_ece'] <= 0.15 and 
                len(pnl_data) >= self.min_samples_for_calibration
            )
        
        return result
    
    def _calculate_ece(self, confidences: List[float], outcomes: List[float], n_bins: int = 10) -> float:
        """Calculate Expected Calibration Error."""
        bin_counts = [0] * n_bins
        bin_correct = [0.0] * n_bins
        bin_confidence = [0.0] * n_bins
        
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
        
        return ece
    
    # =========================================================================
    # FITTING
    # =========================================================================
    
    def _fit_calibration(self) -> bool:
        """Fit both Platt and Isotonic calibration for DIRECTION."""
        if len(self._calibration_buffer) < self.min_samples_for_calibration:
            return False
        
        try:
            data = list(self._calibration_buffer)
            
            # Split into train/validation (chronological split)
            split_idx = int(len(data) * (1 - self._validation_split))
            train_data = data[:split_idx]
            valid_data = data[split_idx:]
            
            if len(train_data) < 30:
                train_data = data
                valid_data = []
            
            x_train = np.array([c for c, _ in train_data], dtype=np.float64)
            y_train = np.array([1.0 if correct else 0.0 for _, correct in train_data], dtype=np.float64)
            
            # Fit Platt scaling
            platt_success = self._fit_platt_internal(x_train, y_train, len(data), use_pnl=False)
            
            # Fit Isotonic regression
            isotonic_success = self._fit_isotonic_internal(x_train, y_train, len(data), use_pnl=False)
            
            # Log validation performance
            if valid_data and (platt_success or isotonic_success):
                x_valid = np.array([c for c, _ in valid_data], dtype=np.float64)
                y_valid = np.array([1.0 if correct else 0.0 for _, correct in valid_data], dtype=np.float64)
                self._log_validation_performance(x_valid, y_valid, "Direction")
            
            return platt_success or isotonic_success
            
        except Exception as e:
            logger.warning(f"Direction calibration fitting failed: {e}")
            return False
    
    def _fit_pnl_calibration(self) -> bool:
        """Fit both Platt and Isotonic calibration for PNL."""
        if len(self._pnl_calibration_buffer) < self.min_samples_for_calibration:
            return False
        
        try:
            data = list(self._pnl_calibration_buffer)
            
            split_idx = int(len(data) * (1 - self._validation_split))
            train_data = data[:split_idx]
            valid_data = data[split_idx:]
            
            if len(train_data) < 30:
                train_data = data
                valid_data = []
            
            x_train = np.array([c for c, _ in train_data], dtype=np.float64)
            y_train = np.array([1.0 if profitable else 0.0 for _, profitable in train_data], dtype=np.float64)
            
            # Fit Platt scaling for PnL
            platt_success = self._fit_platt_internal(x_train, y_train, len(data), use_pnl=True)
            
            # Fit Isotonic regression for PnL
            isotonic_success = self._fit_isotonic_internal(x_train, y_train, len(data), use_pnl=True)
            
            # Log validation performance
            if valid_data and (platt_success or isotonic_success):
                x_valid = np.array([c for c, _ in valid_data], dtype=np.float64)
                y_valid = np.array([1.0 if profitable else 0.0 for _, profitable in valid_data], dtype=np.float64)
                self._log_validation_performance(x_valid, y_valid, "PnL")
            
            return platt_success or isotonic_success
            
        except Exception as e:
            logger.warning(f"PnL calibration fitting failed: {e}")
            return False
    
    def _fit_platt_internal(self, x: np.ndarray, y: np.ndarray, total_samples: int, use_pnl: bool = False) -> bool:
        """Fit Platt scaling parameters using gradient descent."""
        try:
            # Gradient descent
            A, B = -1.0, 0.0
            lr = 0.1
            
            for _ in range(500):
                z = A * x + B
                p = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
                
                grad_A = np.dot(p - y, x) / len(x)
                grad_B = np.mean(p - y)
                
                A_new = A - lr * grad_A
                B_new = B - lr * grad_B
                
                if abs(A_new - A) < 1e-6 and abs(B_new - B) < 1e-6:
                    break
                A, B = A_new, B_new
            
            params = PlattParams(
                A=float(A),
                B=float(B),
                n_samples=total_samples,
                last_fit=datetime.now()
            )
            
            if use_pnl:
                self._pnl_platt = params
                logger.info(f"✅ PnL Platt fitted: A={A:.4f}, B={B:.4f} (n={total_samples})")
            else:
                self._platt = params
                logger.info(f"✅ Direction Platt fitted: A={A:.4f}, B={B:.4f} (n={total_samples})")
            
            return True
            
        except Exception as e:
            logger.warning(f"Platt fitting failed: {e}")
            return False
    
    def _fit_isotonic_internal(self, x: np.ndarray, y: np.ndarray, total_samples: int, use_pnl: bool = False) -> bool:
        """Fit Isotonic regression (Pool Adjacent Violators algorithm)."""
        try:
            # Sort by confidence
            sort_idx = np.argsort(x)
            x_sorted = x[sort_idx]
            y_sorted = y[sort_idx]
            
            # Pool Adjacent Violators (PAV) algorithm
            n = len(x_sorted)
            y_calibrated = y_sorted.copy()
            
            # Forward pass
            i = 0
            while i < n - 1:
                if y_calibrated[i] > y_calibrated[i + 1]:
                    j = i + 1
                    pool_sum = y_calibrated[i] + y_calibrated[j]
                    pool_count = 2
                    
                    while j < n - 1 and pool_sum / pool_count > y_calibrated[j + 1]:
                        j += 1
                        pool_sum += y_calibrated[j]
                        pool_count += 1
                    
                    avg = pool_sum / pool_count
                    for k in range(i, j + 1):
                        y_calibrated[k] = avg
                    
                    i = j + 1
                else:
                    i += 1
            
            # Create interpolation points (subsample if too many)
            max_points = 50
            if len(x_sorted) > max_points:
                indices = np.linspace(0, len(x_sorted) - 1, max_points, dtype=int)
                x_points = x_sorted[indices].tolist()
                y_points = y_calibrated[indices].tolist()
            else:
                x_points = x_sorted.tolist()
                y_points = y_calibrated.tolist()
            
            params = IsotonicParams(
                x_points=x_points,
                y_points=y_points,
                n_samples=total_samples,
                last_fit=datetime.now()
            )
            
            if use_pnl:
                self._pnl_isotonic = params
                logger.info(f"✅ PnL Isotonic fitted: {len(x_points)} points (n={total_samples})")
            else:
                self._isotonic = params
                logger.info(f"✅ Direction Isotonic fitted: {len(x_points)} points (n={total_samples})")
            
            return True
            
        except Exception as e:
            logger.warning(f"Isotonic fitting failed: {e}")
            return False
    
    def _log_validation_performance(self, x_valid: np.ndarray, y_valid: np.ndarray, cal_type: str = "Direction") -> None:
        """Log validation set performance for monitoring overfitting."""
        try:
            # Calculate raw vs calibrated Brier on validation
            raw_brier = np.mean((x_valid - y_valid) ** 2)
            
            if cal_type == "PnL":
                cal_preds = [self.calibrate_pnl(x) for x in x_valid]
            else:
                cal_preds = [self.calibrate(x) for x in x_valid]
            
            cal_brier = np.mean((np.array(cal_preds) - y_valid) ** 2)
            
            logger.info(f"📊 {cal_type} Validation Brier: raw={raw_brier:.4f}, calibrated={cal_brier:.4f}")
            
        except Exception as e:
            logger.debug(f"Validation logging failed: {e}")
    
    # Keep old method name for compatibility
    def _fit_platt(self) -> bool:
        """Backward compatible method - calls _fit_calibration."""
        return self._fit_calibration()
    
    def get_bucket_stats(self) -> Dict[str, Dict]:
        """Get per-confidence-bucket statistics for direction calibration."""
        if len(self._calibration_buffer) < 20:
            return {}
        
        buckets = {}
        for conf, correct in self._calibration_buffer:
            bucket = f"{int(conf * 10) * 10}-{int(conf * 10) * 10 + 10}%"
            if bucket not in buckets:
                buckets[bucket] = {'correct': 0, 'total': 0}
            buckets[bucket]['total'] += 1
            if correct:
                buckets[bucket]['correct'] += 1
        
        result = {}
        for bucket, data in sorted(buckets.items()):
            if data['total'] >= 5:
                accuracy = data['correct'] / data['total']
                expected = (int(bucket.split('-')[0]) + 5) / 100
                result[bucket] = {
                    'actual_accuracy': accuracy,
                    'expected': expected,
                    'samples': data['total'],
                    'well_calibrated': abs(accuracy - expected) < 0.15
                }
        
        return result
    
    def get_pnl_bucket_stats(self) -> Dict[str, Dict]:
        """Get per-confidence-bucket statistics for PnL calibration."""
        if len(self._pnl_calibration_buffer) < 20:
            return {}
        
        buckets = {}
        for conf, profitable in self._pnl_calibration_buffer:
            bucket = f"{int(conf * 10) * 10}-{int(conf * 10) * 10 + 10}%"
            if bucket not in buckets:
                buckets[bucket] = {'profitable': 0, 'total': 0}
            buckets[bucket]['total'] += 1
            if profitable:
                buckets[bucket]['profitable'] += 1
        
        result = {}
        for bucket, data in sorted(buckets.items()):
            if data['total'] >= 5:
                win_rate = data['profitable'] / data['total']
                expected = (int(bucket.split('-')[0]) + 5) / 100
                result[bucket] = {
                    'actual_win_rate': win_rate,
                    'expected': expected,
                    'samples': data['total'],
                    'well_calibrated': abs(win_rate - expected) < 0.15
                }
        
        return result


def create_calibration_tracker(config: Optional[Dict] = None) -> CalibrationTracker:
    """Factory function to create CalibrationTracker with config."""
    defaults = {
        'direction_horizon': 15,
        'pnl_horizon': 60,
        'buffer_size': 1000,
        'min_samples_for_calibration': 50,
        'refit_interval': 50,
        'use_hybrid': True,
        'ensemble_weights': (0.4, 0.6)  # (platt, isotonic)
    }
    
    if config:
        # Map legacy 'horizon_minutes' to 'direction_horizon'
        if 'horizon_minutes' in config and 'direction_horizon' not in config:
            config['direction_horizon'] = config.pop('horizon_minutes')
        defaults.update(config)
    
    return CalibrationTracker(**defaults)
