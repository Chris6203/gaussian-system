#!/usr/bin/env python3
"""
Health Checks & Fail-Safes
==========================

Pre-cycle health checks to prevent trading with unhealthy components.

Features:
1. Data freshness checks
2. Model health validation
3. Calibration quality gates
4. System resource checks
5. Rate limit guards
6. Alert notifications (email/Slack/webhook)

Usage:
    health = HealthCheckSystem()
    
    # Before each trading cycle:
    result = health.run_pre_cycle_checks(
        data_freshness_seconds=120,
        model_health_score=0.8,
        calibration_samples=50
    )
    
    if not result['can_trade']:
        print(f"Trading blocked: {result['blocking_issues']}")
"""

import logging
import os
import time
import json
import requests
import smtplib
from email.mime.text import MIMEText
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from threading import Lock
import psutil

logger = logging.getLogger(__name__)


@dataclass
class HealthCheckResult:
    """Result of a single health check."""
    name: str
    passed: bool
    message: str
    severity: str = 'info'  # 'info', 'warning', 'critical'
    details: Dict = field(default_factory=dict)


@dataclass
class PreCycleResult:
    """Result of all pre-cycle checks."""
    can_trade: bool
    checks: List[HealthCheckResult]
    blocking_issues: List[str]
    warnings: List[str]
    timestamp: datetime = field(default_factory=datetime.now)


class AlertManager:
    """
    Manages alerts via multiple channels.
    
    Supports:
    - Slack webhooks
    - Email (SMTP)
    - Custom webhooks
    - Console logging
    """
    
    def __init__(
        self,
        slack_webhook: Optional[str] = None,
        email_config: Optional[Dict] = None,
        webhook_url: Optional[str] = None,
        alert_cooldown_minutes: int = 30
    ):
        """
        Args:
            slack_webhook: Slack webhook URL
            email_config: Dict with smtp_server, port, sender, password, recipients
            webhook_url: Custom webhook URL
            alert_cooldown_minutes: Minimum time between same alerts
        """
        self.slack_webhook = slack_webhook or os.environ.get('SLACK_WEBHOOK_URL')
        self.email_config = email_config
        self.webhook_url = webhook_url
        self.alert_cooldown_minutes = alert_cooldown_minutes
        
        self._last_alerts: Dict[str, datetime] = {}
        self._lock = Lock()
    
    def send_alert(
        self,
        title: str,
        message: str,
        severity: str = 'warning',
        details: Optional[Dict] = None,
        force: bool = False
    ) -> bool:
        """
        Send alert to configured channels.
        
        Args:
            title: Alert title
            message: Alert message
            severity: 'info', 'warning', 'critical'
            details: Additional details dict
            force: Skip cooldown check
            
        Returns:
            True if alert was sent
        """
        alert_key = f"{title}:{severity}"
        
        with self._lock:
            # Check cooldown
            if not force:
                last_alert = self._last_alerts.get(alert_key)
                if last_alert:
                    elapsed = (datetime.now() - last_alert).total_seconds() / 60
                    if elapsed < self.alert_cooldown_minutes:
                        logger.debug(f"Alert cooldown: {alert_key} (last sent {elapsed:.1f}m ago)")
                        return False
            
            self._last_alerts[alert_key] = datetime.now()
        
        # Build alert payload
        payload = {
            'title': title,
            'message': message,
            'severity': severity,
            'timestamp': datetime.now().isoformat(),
            'details': details or {}
        }
        
        sent = False
        
        # Console (always)
        emoji = {'info': 'ℹ️', 'warning': '⚠️', 'critical': '🚨'}.get(severity, '📢')
        logger.warning(f"{emoji} [{severity.upper()}] {title}: {message}")
        
        # Slack
        if self.slack_webhook:
            sent |= self._send_slack(title, message, severity, details)
        
        # Email
        if self.email_config:
            sent |= self._send_email(title, message, severity, details)
        
        # Webhook
        if self.webhook_url:
            sent |= self._send_webhook(payload)
        
        return sent
    
    def _send_slack(self, title: str, message: str, severity: str, details: Optional[Dict]) -> bool:
        """Send Slack notification."""
        try:
            color = {'info': '#36a64f', 'warning': '#ff9800', 'critical': '#dc3545'}.get(severity, '#808080')
            
            payload = {
                'attachments': [{
                    'color': color,
                    'title': title,
                    'text': message,
                    'footer': f"Trading Bot Alert | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    'fields': [{'title': k, 'value': str(v), 'short': True} for k, v in (details or {}).items()][:8]
                }]
            }
            
            response = requests.post(self.slack_webhook, json=payload, timeout=10)
            return response.status_code == 200
            
        except Exception as e:
            logger.warning(f"Slack alert failed: {e}")
            return False
    
    def _send_email(self, title: str, message: str, severity: str, details: Optional[Dict]) -> bool:
        """Send email notification."""
        try:
            cfg = self.email_config
            
            body = f"""
Trading Bot Alert: {title}
Severity: {severity.upper()}
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{message}

Details:
{json.dumps(details or {}, indent=2)}
"""
            
            msg = MIMEText(body)
            msg['Subject'] = f"[{severity.upper()}] {title}"
            msg['From'] = cfg['sender']
            msg['To'] = ', '.join(cfg.get('recipients', [cfg['sender']]))
            
            with smtplib.SMTP(cfg['smtp_server'], cfg.get('port', 587)) as server:
                server.starttls()
                server.login(cfg['sender'], cfg['password'])
                server.send_message(msg)
            
            return True
            
        except Exception as e:
            logger.warning(f"Email alert failed: {e}")
            return False
    
    def _send_webhook(self, payload: Dict) -> bool:
        """Send to custom webhook."""
        try:
            response = requests.post(self.webhook_url, json=payload, timeout=10)
            return response.status_code in (200, 201, 202)
        except Exception as e:
            logger.warning(f"Webhook alert failed: {e}")
            return False


class HealthCheckSystem:
    """
    Comprehensive health check system for trading.
    """
    
    def __init__(
        self,
        alert_manager: Optional[AlertManager] = None,
        max_data_age_seconds: int = 300,
        min_health_score: float = 0.5,
        min_calibration_samples: int = 50,
        max_memory_percent: float = 90.0,
        max_cpu_percent: float = 95.0
    ):
        """
        Args:
            alert_manager: AlertManager for sending alerts
            max_data_age_seconds: Max age of data before blocking
            min_health_score: Minimum model health score
            min_calibration_samples: Minimum calibration samples
            max_memory_percent: Max memory usage before warning
            max_cpu_percent: Max CPU usage before warning
        """
        self.alert_manager = alert_manager or AlertManager()
        self.max_data_age_seconds = max_data_age_seconds
        self.min_health_score = min_health_score
        self.min_calibration_samples = min_calibration_samples
        self.max_memory_percent = max_memory_percent
        self.max_cpu_percent = max_cpu_percent
        
        # Track consecutive failures
        self._consecutive_failures = 0
        self._last_check_time: Optional[datetime] = None
        
        # Custom checks
        self._custom_checks: List[Callable[[], HealthCheckResult]] = []
    
    def add_custom_check(self, check_func: Callable[[], HealthCheckResult]) -> None:
        """Add a custom health check function."""
        self._custom_checks.append(check_func)
    
    def run_pre_cycle_checks(
        self,
        data_timestamp: Optional[datetime] = None,
        model_health: Optional[Dict] = None,
        calibration_metrics: Optional[Dict] = None,
        current_positions: int = 0
    ) -> PreCycleResult:
        """
        Run all pre-cycle health checks.
        
        Args:
            data_timestamp: Timestamp of latest data
            model_health: Dict from ModelHealthMonitor.get_health_status()
            calibration_metrics: Dict from CalibrationTracker.get_metrics()
            current_positions: Number of open positions
            
        Returns:
            PreCycleResult with can_trade and details
        """
        checks: List[HealthCheckResult] = []
        
        # 1. System resources check
        checks.append(self._check_system_resources())
        
        # 2. Data freshness check
        if data_timestamp:
            checks.append(self._check_data_freshness(data_timestamp))
        
        # 3. Model health check
        if model_health:
            checks.append(self._check_model_health(model_health))
        
        # 4. Calibration check
        if calibration_metrics:
            checks.append(self._check_calibration(calibration_metrics))
        
        # 5. Position limits check
        checks.append(self._check_position_limits(current_positions))
        
        # 6. Run custom checks
        for check_func in self._custom_checks:
            try:
                checks.append(check_func())
            except Exception as e:
                checks.append(HealthCheckResult(
                    name='custom_check',
                    passed=False,
                    message=f"Custom check failed: {e}",
                    severity='warning'
                ))
        
        # Determine if trading is allowed
        blocking_issues = [c.message for c in checks if not c.passed and c.severity == 'critical']
        warnings = [c.message for c in checks if not c.passed and c.severity == 'warning']
        
        can_trade = len(blocking_issues) == 0
        
        # Track consecutive failures
        if not can_trade:
            self._consecutive_failures += 1
        else:
            self._consecutive_failures = 0
        
        # Send alerts for critical issues
        if blocking_issues:
            self.alert_manager.send_alert(
                title="Trading Blocked",
                message="; ".join(blocking_issues),
                severity='critical',
                details={'checks': [c.__dict__ for c in checks if not c.passed]}
            )
        
        # Alert on consecutive failures
        if self._consecutive_failures >= 5:
            self.alert_manager.send_alert(
                title="Consecutive Health Check Failures",
                message=f"{self._consecutive_failures} consecutive failed checks",
                severity='critical',
                force=True
            )
        
        self._last_check_time = datetime.now()
        
        result = PreCycleResult(
            can_trade=can_trade,
            checks=checks,
            blocking_issues=blocking_issues,
            warnings=warnings
        )
        
        # Log summary
        status = "✅ PASS" if can_trade else "❌ BLOCKED"
        logger.info(f"Health Check: {status} | {len(checks)} checks | {len(blocking_issues)} blocking | {len(warnings)} warnings")
        
        return result
    
    def _check_system_resources(self) -> HealthCheckResult:
        """Check CPU and memory usage."""
        try:
            memory = psutil.virtual_memory()
            cpu = psutil.cpu_percent(interval=0.1)
            
            issues = []
            severity = 'info'
            
            if memory.percent > self.max_memory_percent:
                issues.append(f"High memory: {memory.percent:.1f}%")
                severity = 'critical' if memory.percent > 95 else 'warning'
            
            if cpu > self.max_cpu_percent:
                issues.append(f"High CPU: {cpu:.1f}%")
                severity = 'critical' if cpu > 99 else 'warning'
            
            passed = len(issues) == 0 or severity != 'critical'
            message = "; ".join(issues) if issues else f"Resources OK (CPU: {cpu:.0f}%, Memory: {memory.percent:.0f}%)"
            
            return HealthCheckResult(
                name='system_resources',
                passed=passed,
                message=message,
                severity=severity,
                details={'cpu_percent': cpu, 'memory_percent': memory.percent}
            )
            
        except Exception as e:
            return HealthCheckResult(
                name='system_resources',
                passed=True,  # Don't block on monitoring failure
                message=f"Resource check error: {e}",
                severity='warning'
            )
    
    def _check_data_freshness(self, data_timestamp: datetime) -> HealthCheckResult:
        """Check if data is fresh enough."""
        age_seconds = (datetime.now() - data_timestamp).total_seconds()
        
        if age_seconds > self.max_data_age_seconds:
            return HealthCheckResult(
                name='data_freshness',
                passed=False,
                message=f"Stale data: {age_seconds:.0f}s old (max: {self.max_data_age_seconds}s)",
                severity='critical',
                details={'age_seconds': age_seconds, 'max_age': self.max_data_age_seconds}
            )
        
        return HealthCheckResult(
            name='data_freshness',
            passed=True,
            message=f"Data fresh ({age_seconds:.0f}s old)",
            severity='info',
            details={'age_seconds': age_seconds}
        )
    
    def _check_model_health(self, health: Dict) -> HealthCheckResult:
        """Check model health status."""
        score = health.get('health_score', 0)
        status = health.get('status', 'unknown')
        
        if status == 'unhealthy' or score < self.min_health_score:
            return HealthCheckResult(
                name='model_health',
                passed=False,
                message=f"Model unhealthy: score={score:.2f}, status={status}",
                severity='critical',
                details=health
            )
        
        if status == 'degraded':
            return HealthCheckResult(
                name='model_health',
                passed=True,
                message=f"Model degraded: score={score:.2f}",
                severity='warning',
                details=health
            )
        
        return HealthCheckResult(
            name='model_health',
            passed=True,
            message=f"Model healthy: score={score:.2f}",
            severity='info',
            details={'health_score': score}
        )
    
    def _check_calibration(self, metrics: Dict) -> HealthCheckResult:
        """Check calibration quality."""
        sample_count = metrics.get('sample_count', 0)
        is_calibrated = metrics.get('is_calibrated', False)
        brier = metrics.get('brier_score', 1.0)
        
        if sample_count < self.min_calibration_samples:
            return HealthCheckResult(
                name='calibration',
                passed=True,  # Don't block, just warn
                message=f"Low calibration samples: {sample_count}/{self.min_calibration_samples}",
                severity='warning',
                details=metrics
            )
        
        if not is_calibrated:
            return HealthCheckResult(
                name='calibration',
                passed=True,  # Don't block, just warn
                message=f"Poor calibration: Brier={brier:.3f}",
                severity='warning',
                details=metrics
            )
        
        return HealthCheckResult(
            name='calibration',
            passed=True,
            message=f"Calibration OK: Brier={brier:.3f}, samples={sample_count}",
            severity='info',
            details=metrics
        )
    
    def _check_position_limits(self, current_positions: int, max_positions: int = 10) -> HealthCheckResult:
        """Check position limits."""
        if current_positions >= max_positions:
            return HealthCheckResult(
                name='position_limits',
                passed=False,
                message=f"Max positions reached: {current_positions}/{max_positions}",
                severity='critical',
                details={'current': current_positions, 'max': max_positions}
            )
        
        if current_positions >= max_positions * 0.8:
            return HealthCheckResult(
                name='position_limits',
                passed=True,
                message=f"Near position limit: {current_positions}/{max_positions}",
                severity='warning',
                details={'current': current_positions, 'max': max_positions}
            )
        
        return HealthCheckResult(
            name='position_limits',
            passed=True,
            message=f"Positions OK: {current_positions}/{max_positions}",
            severity='info',
            details={'current': current_positions, 'max': max_positions}
        )


def create_health_check_system(config: Optional[Dict] = None) -> HealthCheckSystem:
    """Factory function to create HealthCheckSystem."""
    alert_config = config.get('alerts', {}) if config else {}
    
    alert_manager = AlertManager(
        slack_webhook=alert_config.get('slack_webhook'),
        email_config=alert_config.get('email'),
        webhook_url=alert_config.get('webhook_url'),
        alert_cooldown_minutes=alert_config.get('cooldown_minutes', 30)
    )
    
    defaults = {
        'max_data_age_seconds': 300,
        'min_health_score': 0.5,
        'min_calibration_samples': 50,
        'max_memory_percent': 90.0,
        'max_cpu_percent': 95.0
    }
    
    if config:
        defaults.update({k: v for k, v in config.items() if k != 'alerts'})
    
    return HealthCheckSystem(alert_manager=alert_manager, **defaults)









