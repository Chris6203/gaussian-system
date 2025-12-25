#!/usr/bin/env python3
"""
Model Health & Drift Detection
==============================

Monitors model health and detects distribution drift that signals
retraining is needed.

Features:
1. Prediction error tracking with rolling statistics
2. Feature distribution drift detection (Kolmogorov-Smirnov test)
3. Calibration drift monitoring
4. Health score computation
5. Automatic alerts and retraining triggers

Usage:
    health = ModelHealthMonitor()
    
    # Every prediction:
    health.record_prediction_error(predicted=0.003, actual=0.002)
    health.record_feature_sample(features)
    
    # Check health:
    status = health.get_health_status()
    if status['needs_retraining']:
        trigger_retraining()
"""

import logging
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import deque
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class DriftAlert:
    """Alert for detected drift."""
    timestamp: datetime
    alert_type: str  # 'feature_drift', 'error_spike', 'calibration_drift'
    severity: str  # 'warning', 'critical'
    message: str
    details: Dict = field(default_factory=dict)


class ModelHealthMonitor:
    """
    Monitors model health and detects drift.
    
    Tracks:
    - Prediction errors (rolling mean/std)
    - Feature distributions (KS test for drift)
    - Calibration quality (Brier/ECE trends)
    """
    
    def __init__(
        self,
        error_window: int = 500,
        feature_window: int = 1000,
        error_threshold: float = 0.02,  # 2% error threshold
        drift_threshold: float = 0.05,  # KS p-value threshold
        calibration_drift_threshold: float = 0.1,  # Brier increase threshold
        alert_cooldown_minutes: int = 60
    ):
        """
        Args:
            error_window: Rolling window for error statistics
            feature_window: Window for feature distribution baseline
            error_threshold: Mean abs error that triggers alert
            drift_threshold: KS test p-value below which drift is detected
            calibration_drift_threshold: Brier score increase that triggers alert
            alert_cooldown_minutes: Minimum time between same-type alerts
        """
        self.error_window = error_window
        self.feature_window = feature_window
        self.error_threshold = error_threshold
        self.drift_threshold = drift_threshold
        self.calibration_drift_threshold = calibration_drift_threshold
        self.alert_cooldown_minutes = alert_cooldown_minutes
        
        # Error tracking
        self._errors: deque = deque(maxlen=error_window)
        self._direction_results: deque = deque(maxlen=error_window)
        
        # Feature distribution tracking
        self._baseline_features: Optional[np.ndarray] = None
        self._recent_features: deque = deque(maxlen=feature_window)
        self._feature_names: List[str] = []
        
        # Calibration tracking
        self._brier_history: deque = deque(maxlen=100)
        self._baseline_brier: Optional[float] = None
        
        # Alerts
        self._alerts: List[DriftAlert] = []
        self._last_alert_times: Dict[str, datetime] = {}
        
        # Health metrics
        self._last_health_check: Optional[datetime] = None
        self._health_score: float = 1.0
        
        logger.info("✅ ModelHealthMonitor initialized")
    
    def record_prediction_error(
        self,
        predicted_return: float,
        actual_return: float,
        predicted_direction: Optional[str] = None,
        actual_direction: Optional[str] = None
    ) -> None:
        """
        Record a prediction error for tracking.
        
        Args:
            predicted_return: Predicted return value
            actual_return: Actual return value
            predicted_direction: 'UP' or 'DOWN'
            actual_direction: 'UP' or 'DOWN'
        """
        error = abs(predicted_return - actual_return)
        self._errors.append(error)
        
        # Track direction accuracy
        if predicted_direction and actual_direction:
            correct = predicted_direction == actual_direction
            self._direction_results.append(correct)
        
        # Check for error spike
        if len(self._errors) >= 50:
            recent_errors = list(self._errors)[-50:]
            mean_error = np.mean(recent_errors)
            
            if mean_error > self.error_threshold:
                self._create_alert(
                    alert_type='error_spike',
                    severity='warning' if mean_error < self.error_threshold * 1.5 else 'critical',
                    message=f"Prediction error spike: {mean_error:.4f} (threshold: {self.error_threshold})",
                    details={'mean_error': mean_error, 'recent_count': len(recent_errors)}
                )
    
    def record_feature_sample(self, features: np.ndarray, feature_names: Optional[List[str]] = None) -> None:
        """
        Record a feature sample for drift detection.
        
        Args:
            features: Feature vector
            feature_names: Optional names for features
        """
        if feature_names:
            self._feature_names = feature_names
        
        self._recent_features.append(features.copy())
        
        # Set baseline if not set
        if self._baseline_features is None and len(self._recent_features) >= self.feature_window // 2:
            self._baseline_features = np.array(list(self._recent_features))
            logger.info(f"📊 Feature baseline established ({len(self._baseline_features)} samples)")
    
    def record_calibration_metric(self, brier_score: float) -> None:
        """
        Record calibration metric for drift tracking.
        
        Args:
            brier_score: Current Brier score
        """
        self._brier_history.append(brier_score)
        
        # Set baseline if not set
        if self._baseline_brier is None and len(self._brier_history) >= 20:
            self._baseline_brier = np.mean(list(self._brier_history)[:20])
            logger.info(f"📊 Calibration baseline established: Brier={self._baseline_brier:.4f}")
        
        # Check for calibration drift
        if self._baseline_brier is not None and len(self._brier_history) >= 20:
            recent_brier = np.mean(list(self._brier_history)[-20:])
            drift = recent_brier - self._baseline_brier
            
            if drift > self.calibration_drift_threshold:
                self._create_alert(
                    alert_type='calibration_drift',
                    severity='warning' if drift < self.calibration_drift_threshold * 1.5 else 'critical',
                    message=f"Calibration drift detected: Brier increased by {drift:.4f}",
                    details={'baseline_brier': self._baseline_brier, 'current_brier': recent_brier}
                )
    
    def check_feature_drift(self) -> Dict[str, float]:
        """
        Check for feature distribution drift using Kolmogorov-Smirnov test.
        
        Returns:
            Dict mapping feature index/name to drift p-value
        """
        if self._baseline_features is None or len(self._recent_features) < 100:
            return {}
        
        try:
            from scipy import stats
        except ImportError:
            logger.debug("scipy not available for KS test")
            return {}
        
        drift_results = {}
        recent = np.array(list(self._recent_features)[-100:])
        
        # Test each feature dimension
        n_features = min(self._baseline_features.shape[1], recent.shape[1])
        drifted_features = []
        
        for i in range(n_features):
            baseline_dist = self._baseline_features[:, i]
            recent_dist = recent[:, i]
            
            # Remove any NaN/inf
            baseline_dist = baseline_dist[np.isfinite(baseline_dist)]
            recent_dist = recent_dist[np.isfinite(recent_dist)]
            
            if len(baseline_dist) < 10 or len(recent_dist) < 10:
                continue
            
            # Kolmogorov-Smirnov test
            ks_stat, p_value = stats.ks_2samp(baseline_dist, recent_dist)
            
            feature_name = self._feature_names[i] if i < len(self._feature_names) else f"feature_{i}"
            drift_results[feature_name] = {'ks_stat': ks_stat, 'p_value': p_value}
            
            if p_value < self.drift_threshold:
                drifted_features.append((feature_name, p_value))
        
        # Alert if significant drift
        if len(drifted_features) > 3:  # More than 3 features drifted
            self._create_alert(
                alert_type='feature_drift',
                severity='warning' if len(drifted_features) < 10 else 'critical',
                message=f"Feature drift detected in {len(drifted_features)} features",
                details={'drifted_features': drifted_features[:10]}
            )
        
        return drift_results
    
    def get_health_status(self) -> Dict:
        """
        Get comprehensive health status.
        
        Returns:
            Dict with health metrics and recommendations
        """
        self._last_health_check = datetime.now()
        
        # Calculate error stats
        error_stats = {}
        if len(self._errors) >= 10:
            errors = list(self._errors)
            error_stats = {
                'mean_error': float(np.mean(errors)),
                'std_error': float(np.std(errors)),
                'max_error': float(np.max(errors)),
                'sample_count': len(errors)
            }
        
        # Calculate direction accuracy
        direction_accuracy = None
        if len(self._direction_results) >= 10:
            direction_accuracy = sum(self._direction_results) / len(self._direction_results)
        
        # Calculate calibration stats
        calibration_stats = {}
        if len(self._brier_history) >= 10:
            brier_values = list(self._brier_history)
            calibration_stats = {
                'current_brier': float(brier_values[-1]) if brier_values else None,
                'mean_brier': float(np.mean(brier_values)),
                'baseline_brier': self._baseline_brier,
                'brier_trend': float(np.mean(brier_values[-10:]) - np.mean(brier_values[:10])) if len(brier_values) >= 20 else 0
            }
        
        # Check for drift
        drift_results = self.check_feature_drift()
        drift_count = sum(1 for d in drift_results.values() if d['p_value'] < self.drift_threshold)
        
        # Calculate health score (0-1, 1 is healthy)
        health_score = 1.0
        
        # Penalize for high errors
        if error_stats.get('mean_error', 0) > self.error_threshold:
            health_score -= 0.3
        
        # Penalize for low direction accuracy
        if direction_accuracy is not None and direction_accuracy < 0.45:
            health_score -= 0.3
        
        # Penalize for calibration drift
        if calibration_stats.get('brier_trend', 0) > self.calibration_drift_threshold:
            health_score -= 0.2
        
        # Penalize for feature drift
        if drift_count > 5:
            health_score -= 0.2
        
        health_score = max(0.0, health_score)
        self._health_score = health_score
        
        # Determine if retraining needed
        needs_retraining = (
            health_score < 0.5 or
            (direction_accuracy is not None and direction_accuracy < 0.40) or
            drift_count > 10 or
            len(self._alerts) > 3  # Multiple recent alerts
        )
        
        # Build recommendations
        recommendations = []
        if error_stats.get('mean_error', 0) > self.error_threshold:
            recommendations.append("High prediction errors - consider model retraining")
        if direction_accuracy is not None and direction_accuracy < 0.45:
            recommendations.append("Low direction accuracy - review feature engineering")
        if drift_count > 5:
            recommendations.append(f"Feature drift detected in {drift_count} features - update baseline or retrain")
        if calibration_stats.get('brier_trend', 0) > self.calibration_drift_threshold:
            recommendations.append("Calibration degrading - refit Platt scaling")
        
        return {
            'health_score': health_score,
            'status': 'healthy' if health_score > 0.7 else ('degraded' if health_score > 0.4 else 'unhealthy'),
            'needs_retraining': needs_retraining,
            'error_stats': error_stats,
            'direction_accuracy': direction_accuracy,
            'calibration_stats': calibration_stats,
            'feature_drift_count': drift_count,
            'recent_alerts': [
                {'type': a.alert_type, 'severity': a.severity, 'message': a.message}
                for a in self._alerts[-5:]
            ],
            'recommendations': recommendations,
            'last_check': self._last_health_check.isoformat() if self._last_health_check else None
        }
    
    def _create_alert(self, alert_type: str, severity: str, message: str, details: Dict = None) -> None:
        """Create an alert if not in cooldown."""
        now = datetime.now()
        
        # Check cooldown
        last_alert = self._last_alert_times.get(alert_type)
        if last_alert and (now - last_alert).total_seconds() < self.alert_cooldown_minutes * 60:
            return
        
        alert = DriftAlert(
            timestamp=now,
            alert_type=alert_type,
            severity=severity,
            message=message,
            details=details or {}
        )
        
        self._alerts.append(alert)
        self._last_alert_times[alert_type] = now
        
        # Keep only recent alerts
        if len(self._alerts) > 50:
            self._alerts = self._alerts[-50:]
        
        # Log alert
        log_func = logger.warning if severity == 'warning' else logger.error
        log_func(f"🚨 [{severity.upper()}] {alert_type}: {message}")
    
    def reset_baseline(self) -> None:
        """Reset baselines after retraining."""
        if len(self._recent_features) >= 100:
            self._baseline_features = np.array(list(self._recent_features))
            logger.info(f"📊 Feature baseline reset ({len(self._baseline_features)} samples)")
        
        if len(self._brier_history) >= 10:
            self._baseline_brier = np.mean(list(self._brier_history)[-20:])
            logger.info(f"📊 Calibration baseline reset: Brier={self._baseline_brier:.4f}")
        
        self._alerts.clear()
        self._health_score = 1.0
        logger.info("✅ Model health baselines reset")


class HealthCheckRunner:
    """
    Runs periodic health checks and can trigger actions.
    """
    
    def __init__(
        self,
        monitor: ModelHealthMonitor,
        check_interval_cycles: int = 50,
        on_unhealthy: Optional[callable] = None,
        on_retraining_needed: Optional[callable] = None
    ):
        """
        Args:
            monitor: ModelHealthMonitor instance
            check_interval_cycles: Run health check every N cycles
            on_unhealthy: Callback when health is unhealthy
            on_retraining_needed: Callback when retraining is needed
        """
        self.monitor = monitor
        self.check_interval_cycles = check_interval_cycles
        self.on_unhealthy = on_unhealthy
        self.on_retraining_needed = on_retraining_needed
        
        self._cycle_count = 0
        self._last_status: Optional[Dict] = None
    
    def on_cycle(self) -> Optional[Dict]:
        """
        Called every trading cycle. Returns health status if check was run.
        """
        self._cycle_count += 1
        
        if self._cycle_count % self.check_interval_cycles == 0:
            status = self.monitor.get_health_status()
            self._last_status = status
            
            # Trigger callbacks
            if status['status'] == 'unhealthy' and self.on_unhealthy:
                self.on_unhealthy(status)
            
            if status['needs_retraining'] and self.on_retraining_needed:
                self.on_retraining_needed(status)
            
            return status
        
        return None


def create_health_monitor(config: Optional[Dict] = None) -> ModelHealthMonitor:
    """Factory function to create ModelHealthMonitor."""
    defaults = {
        'error_window': 500,
        'feature_window': 1000,
        'error_threshold': 0.02,
        'drift_threshold': 0.05,
        'calibration_drift_threshold': 0.1,
        'alert_cooldown_minutes': 60
    }
    
    if config:
        defaults.update(config)
    
    return ModelHealthMonitor(**defaults)









