#!/usr/bin/env python3
"""
Trading Bot Monitoring Dashboard
================================

Real-time monitoring for:
- Position status (stuck, profitable, at risk)
- Signal rejection rates by reason
- Regime-specific performance
- Calibration health
- Trade frequency analysis

Usage (as Flask app):
    python -m backend.monitoring_dashboard
    
Or integrate with existing dashboard:
    from backend.monitoring_dashboard import TradingMonitor
    
    monitor = TradingMonitor()
    stats = monitor.get_dashboard_data()
"""

import logging
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from pathlib import Path
import threading

logger = logging.getLogger(__name__)


@dataclass
class PositionStatus:
    """Status tracking for a single position."""
    symbol: str
    option_symbol: str
    entry_time: datetime
    entry_price: float
    current_price: float
    quantity: int
    direction: str  # 'CALL' or 'PUT'
    
    # Calculated fields
    pnl_pct: float = 0.0
    hold_duration_minutes: int = 0
    is_stuck: bool = False
    is_at_risk: bool = False
    status: str = "OPEN"  # OPEN, STUCK, AT_RISK, PROFITABLE
    alerts: List[str] = field(default_factory=list)


@dataclass
class RejectionStats:
    """Statistics about signal rejections."""
    total_signals: int = 0
    total_rejections: int = 0
    rejections_by_reason: Dict[str, int] = field(default_factory=dict)
    rejections_by_hour: Dict[int, int] = field(default_factory=dict)
    rejections_by_regime: Dict[str, int] = field(default_factory=dict)
    
    @property
    def rejection_rate(self) -> float:
        return self.total_rejections / max(1, self.total_signals)


@dataclass
class CalibrationHealth:
    """Calibration system health metrics."""
    brier_score: float = 1.0
    ece: float = 1.0
    sample_count: int = 0
    direction_accuracy: float = 0.0
    is_healthy: bool = False
    last_refit: Optional[datetime] = None
    trend: str = "unknown"  # improving, degrading, stable


class TradingMonitor:
    """
    Central monitoring hub for trading bot health and performance.
    """
    
    # Thresholds for alerts
    STUCK_POSITION_MINUTES = 120
    AT_RISK_PNL_PCT = -0.15
    HIGH_REJECTION_RATE = 0.7
    
    def __init__(
        self,
        db_path: str = "data/paper_trades.db",
        log_buffer_size: int = 1000
    ):
        """
        Args:
            db_path: Path to trades database
            log_buffer_size: Max recent events to keep in memory
        """
        self.db_path = Path(db_path)
        self.log_buffer_size = log_buffer_size
        
        # In-memory buffers
        self._signals_log: List[Dict] = []
        self._rejections_log: List[Dict] = []
        self._positions: Dict[str, PositionStatus] = {}
        self._calibration_history: List[Dict] = []
        
        # Aggregated stats
        self._rejection_stats = RejectionStats()
        self._regime_performance: Dict[str, Dict] = defaultdict(
            lambda: {'trades': 0, 'wins': 0, 'total_pnl': 0}
        )
        
        # Lock for thread safety
        self._lock = threading.Lock()
        
        logger.info("✅ TradingMonitor initialized")
    
    # =========================================================================
    # EVENT LOGGING
    # =========================================================================
    
    def log_signal(
        self,
        signal: Dict,
        executed: bool,
        rejection_reasons: List[str] = None,
        regime: str = None
    ):
        """
        Log a trading signal and its outcome.
        
        Args:
            signal: Signal dict with action, confidence, etc.
            executed: Whether trade was executed
            rejection_reasons: List of reasons if rejected
            regime: Current regime name
        """
        with self._lock:
            event = {
                'timestamp': datetime.now().isoformat(),
                'action': signal.get('action', 'UNKNOWN'),
                'confidence': signal.get('confidence', 0),
                'calibrated_confidence': signal.get('calibrated_confidence', 0),
                'executed': executed,
                'rejection_reasons': rejection_reasons or [],
                'regime': regime,
                'hour': datetime.now().hour
            }
            
            self._signals_log.append(event)
            if len(self._signals_log) > self.log_buffer_size:
                self._signals_log = self._signals_log[-self.log_buffer_size:]
            
            # Update rejection stats
            self._rejection_stats.total_signals += 1
            
            if not executed and rejection_reasons:
                self._rejection_stats.total_rejections += 1
                hour = datetime.now().hour
                self._rejection_stats.rejections_by_hour[hour] = \
                    self._rejection_stats.rejections_by_hour.get(hour, 0) + 1
                
                if regime:
                    self._rejection_stats.rejections_by_regime[regime] = \
                        self._rejection_stats.rejections_by_regime.get(regime, 0) + 1
                
                for reason in rejection_reasons:
                    # Normalize reason key
                    reason_key = self._normalize_reason(reason)
                    self._rejection_stats.rejections_by_reason[reason_key] = \
                        self._rejection_stats.rejections_by_reason.get(reason_key, 0) + 1
    
    def _normalize_reason(self, reason: str) -> str:
        """Normalize rejection reason to a category key."""
        reason_lower = reason.lower()
        
        if 'confidence' in reason_lower:
            return 'low_confidence'
        elif 'brier' in reason_lower:
            return 'poor_brier'
        elif 'ece' in reason_lower:
            return 'poor_ece'
        elif 'calibration' in reason_lower or 'sample' in reason_lower:
            return 'insufficient_calibration'
        elif 'horizon' in reason_lower or 'agreement' in reason_lower or 'timeframe' in reason_lower:
            return 'multi_tf_disagreement'
        elif 'trades' in reason_lower or 'hour' in reason_lower or 'frequency' in reason_lower:
            return 'trade_limit'
        elif 'regime' in reason_lower or 'vol' in reason_lower:
            return 'regime_filter'
        elif 'time' in reason_lower or 'hour' in reason_lower:
            return 'time_filter'
        elif 'trend' in reason_lower or 'counter' in reason_lower:
            return 'counter_trend'
        else:
            return 'other'
    
    def log_trade_result(
        self,
        trade: Dict,
        regime: str = None
    ):
        """
        Log completed trade for regime performance tracking.
        
        Args:
            trade: Trade result with pnl, win/loss, etc.
            regime: Regime during trade
        """
        with self._lock:
            regime_key = regime or 'unknown'
            stats = self._regime_performance[regime_key]
            
            stats['trades'] += 1
            stats['total_pnl'] += trade.get('pnl', 0)
            if trade.get('pnl', 0) > 0:
                stats['wins'] += 1
    
    def log_calibration_update(
        self,
        metrics: Dict
    ):
        """
        Log calibration metrics for health tracking.
        
        Args:
            metrics: Dict with brier_score, ece, sample_count, etc.
        """
        with self._lock:
            event = {
                'timestamp': datetime.now().isoformat(),
                **metrics
            }
            
            self._calibration_history.append(event)
            if len(self._calibration_history) > 200:
                self._calibration_history = self._calibration_history[-200:]
    
    # =========================================================================
    # POSITION MONITORING
    # =========================================================================
    
    def update_position(
        self,
        symbol: str,
        option_symbol: str,
        entry_time: datetime,
        entry_price: float,
        current_price: float,
        quantity: int,
        direction: str
    ):
        """
        Update or add position for monitoring.
        
        Args:
            symbol: Underlying symbol
            option_symbol: Option contract symbol
            entry_time: When position was opened
            entry_price: Entry price
            current_price: Current market price
            quantity: Number of contracts
            direction: 'CALL' or 'PUT'
        """
        with self._lock:
            # Calculate metrics
            pnl_pct = (current_price / entry_price - 1) if entry_price > 0 else 0
            hold_minutes = int((datetime.now() - entry_time).total_seconds() / 60)
            
            # Determine status
            is_stuck = hold_minutes > self.STUCK_POSITION_MINUTES and abs(pnl_pct) < 0.05
            is_at_risk = pnl_pct < self.AT_RISK_PNL_PCT
            
            if is_stuck:
                status = "STUCK"
            elif is_at_risk:
                status = "AT_RISK"
            elif pnl_pct > 0.05:
                status = "PROFITABLE"
            else:
                status = "OPEN"
            
            # Generate alerts
            alerts = []
            if is_stuck:
                alerts.append(f"Position stuck for {hold_minutes}min with {pnl_pct:.1%} PnL")
            if is_at_risk:
                alerts.append(f"Position at risk: {pnl_pct:.1%} PnL")
            if hold_minutes > 180:
                alerts.append(f"Long hold time: {hold_minutes}min")
            
            position = PositionStatus(
                symbol=symbol,
                option_symbol=option_symbol,
                entry_time=entry_time,
                entry_price=entry_price,
                current_price=current_price,
                quantity=quantity,
                direction=direction,
                pnl_pct=pnl_pct,
                hold_duration_minutes=hold_minutes,
                is_stuck=is_stuck,
                is_at_risk=is_at_risk,
                status=status,
                alerts=alerts
            )
            
            self._positions[option_symbol] = position
    
    def close_position(self, option_symbol: str):
        """Remove closed position from monitoring."""
        with self._lock:
            if option_symbol in self._positions:
                del self._positions[option_symbol]
    
    def get_position_alerts(self) -> List[Dict]:
        """Get all position alerts."""
        alerts = []
        with self._lock:
            for symbol, pos in self._positions.items():
                if pos.alerts:
                    alerts.append({
                        'symbol': symbol,
                        'status': pos.status,
                        'alerts': pos.alerts,
                        'pnl_pct': pos.pnl_pct,
                        'hold_minutes': pos.hold_duration_minutes
                    })
        return alerts
    
    # =========================================================================
    # DASHBOARD DATA
    # =========================================================================
    
    def get_dashboard_data(self) -> Dict:
        """
        Get comprehensive dashboard data.
        
        Returns:
            Dict with all monitoring metrics
        """
        with self._lock:
            # Position summary
            positions_list = list(self._positions.values())
            stuck_count = sum(1 for p in positions_list if p.is_stuck)
            at_risk_count = sum(1 for p in positions_list if p.is_at_risk)
            profitable_count = sum(1 for p in positions_list if p.pnl_pct > 0)
            
            # Rejection summary
            rejection_summary = {
                'total_signals': self._rejection_stats.total_signals,
                'total_rejections': self._rejection_stats.total_rejections,
                'rejection_rate': self._rejection_stats.rejection_rate,
                'by_reason': dict(self._rejection_stats.rejections_by_reason),
                'by_hour': dict(self._rejection_stats.rejections_by_hour),
                'by_regime': dict(self._rejection_stats.rejections_by_regime)
            }
            
            # Calibration health
            cal_health = self._get_calibration_health()
            
            # Regime performance
            regime_perf = {}
            for regime, stats in self._regime_performance.items():
                regime_perf[regime] = {
                    'trades': stats['trades'],
                    'wins': stats['wins'],
                    'win_rate': stats['wins'] / max(1, stats['trades']),
                    'total_pnl': stats['total_pnl'],
                    'avg_pnl': stats['total_pnl'] / max(1, stats['trades'])
                }
            
            # System health
            health_score = self._calculate_health_score(
                rejection_rate=self._rejection_stats.rejection_rate,
                stuck_count=stuck_count,
                cal_health=cal_health
            )
            
            return {
                'timestamp': datetime.now().isoformat(),
                'positions': {
                    'total': len(positions_list),
                    'stuck': stuck_count,
                    'at_risk': at_risk_count,
                    'profitable': profitable_count,
                    'details': [self._position_to_dict(p) for p in positions_list]
                },
                'rejections': rejection_summary,
                'calibration': {
                    'brier_score': cal_health.brier_score,
                    'ece': cal_health.ece,
                    'sample_count': cal_health.sample_count,
                    'direction_accuracy': cal_health.direction_accuracy,
                    'is_healthy': cal_health.is_healthy,
                    'trend': cal_health.trend
                },
                'regime_performance': regime_perf,
                'health_score': health_score,
                'alerts': self.get_position_alerts()
            }
    
    def _position_to_dict(self, pos: PositionStatus) -> Dict:
        """Convert PositionStatus to dict."""
        return {
            'symbol': pos.symbol,
            'option_symbol': pos.option_symbol,
            'entry_time': pos.entry_time.isoformat() if pos.entry_time else None,
            'entry_price': pos.entry_price,
            'current_price': pos.current_price,
            'quantity': pos.quantity,
            'direction': pos.direction,
            'pnl_pct': pos.pnl_pct,
            'hold_duration_minutes': pos.hold_duration_minutes,
            'status': pos.status,
            'alerts': pos.alerts
        }
    
    def _get_calibration_health(self) -> CalibrationHealth:
        """Calculate calibration health from history."""
        if not self._calibration_history:
            return CalibrationHealth()
        
        recent = self._calibration_history[-1]
        
        brier = recent.get('brier_score', 1.0)
        ece = recent.get('ece', 1.0)
        samples = recent.get('sample_count', 0)
        accuracy = recent.get('direction_accuracy', 0)
        
        is_healthy = brier < 0.35 and ece < 0.15 and samples >= 50
        
        # Determine trend from last 10 readings
        if len(self._calibration_history) >= 10:
            recent_briers = [h.get('brier_score', 1) for h in self._calibration_history[-10:]]
            first_half = sum(recent_briers[:5]) / 5
            second_half = sum(recent_briers[5:]) / 5
            
            if second_half < first_half - 0.02:
                trend = "improving"
            elif second_half > first_half + 0.02:
                trend = "degrading"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"
        
        return CalibrationHealth(
            brier_score=brier,
            ece=ece,
            sample_count=samples,
            direction_accuracy=accuracy,
            is_healthy=is_healthy,
            trend=trend
        )
    
    def _calculate_health_score(
        self,
        rejection_rate: float,
        stuck_count: int,
        cal_health: CalibrationHealth
    ) -> float:
        """Calculate overall system health score (0-1)."""
        score = 1.0
        
        # Penalize high rejection rate
        if rejection_rate > 0.8:
            score -= 0.3
        elif rejection_rate > 0.6:
            score -= 0.15
        
        # Penalize stuck positions
        score -= min(0.3, stuck_count * 0.1)
        
        # Penalize poor calibration
        if not cal_health.is_healthy:
            if cal_health.brier_score > 0.5:
                score -= 0.2
            elif cal_health.brier_score > 0.35:
                score -= 0.1
        
        # Bonus for good calibration trend
        if cal_health.trend == "improving":
            score += 0.05
        elif cal_health.trend == "degrading":
            score -= 0.1
        
        return max(0, min(1, score))
    
    # =========================================================================
    # REPORTING
    # =========================================================================
    
    def get_rejection_report(self) -> str:
        """Generate human-readable rejection report."""
        stats = self._rejection_stats
        
        lines = [
            "=" * 60,
            "SIGNAL REJECTION REPORT",
            "=" * 60,
            f"Total Signals: {stats.total_signals}",
            f"Total Rejections: {stats.total_rejections}",
            f"Rejection Rate: {stats.rejection_rate:.1%}",
            "",
            "REJECTIONS BY REASON:",
        ]
        
        for reason, count in sorted(
            stats.rejections_by_reason.items(),
            key=lambda x: x[1],
            reverse=True
        ):
            pct = count / max(1, stats.total_rejections) * 100
            lines.append(f"  {reason:25s}: {count:4d} ({pct:.1f}%)")
        
        lines.append("")
        lines.append("REJECTIONS BY HOUR (ET):")
        for hour in sorted(stats.rejections_by_hour.keys()):
            count = stats.rejections_by_hour[hour]
            bar = "█" * int(count / max(stats.rejections_by_hour.values()) * 20)
            lines.append(f"  {hour:02d}:00  {count:4d} {bar}")
        
        lines.append("")
        lines.append("REJECTIONS BY REGIME:")
        for regime, count in sorted(
            stats.rejections_by_regime.items(),
            key=lambda x: x[1],
            reverse=True
        ):
            lines.append(f"  {regime:20s}: {count:4d}")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def get_position_report(self) -> str:
        """Generate human-readable position report."""
        positions = list(self._positions.values())
        
        lines = [
            "=" * 60,
            "POSITION STATUS REPORT",
            "=" * 60,
            f"Total Positions: {len(positions)}",
            f"Stuck: {sum(1 for p in positions if p.is_stuck)}",
            f"At Risk: {sum(1 for p in positions if p.is_at_risk)}",
            f"Profitable: {sum(1 for p in positions if p.pnl_pct > 0)}",
            "",
        ]
        
        for pos in sorted(positions, key=lambda x: x.pnl_pct):
            status_emoji = {
                'STUCK': '🔒',
                'AT_RISK': '⚠️',
                'PROFITABLE': '✅',
                'OPEN': '📊'
            }.get(pos.status, '❓')
            
            lines.append(
                f"{status_emoji} {pos.option_symbol[:20]:20s} "
                f"{pos.pnl_pct:+7.1%} | {pos.hold_duration_minutes:3d}min | {pos.status}"
            )
            
            if pos.alerts:
                for alert in pos.alerts:
                    lines.append(f"   └─ {alert}")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    # =========================================================================
    # DATABASE LOADING
    # =========================================================================
    
    def load_positions_from_db(self):
        """Load open positions from database."""
        if not self.db_path.exists():
            return
        
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT symbol, option_symbol, entry_time, entry_price,
                       quantity, option_type
                FROM positions
                WHERE status = 'OPEN'
            """)
            
            for row in cursor.fetchall():
                symbol, opt_symbol, entry_time, entry_price, qty, opt_type = row
                
                # Parse entry time
                try:
                    entry_dt = datetime.fromisoformat(entry_time)
                except Exception:
                    entry_dt = datetime.now()
                
                self.update_position(
                    symbol=symbol,
                    option_symbol=opt_symbol,
                    entry_time=entry_dt,
                    entry_price=float(entry_price),
                    current_price=float(entry_price),  # Will be updated
                    quantity=int(qty),
                    direction='CALL' if opt_type == 'call' else 'PUT'
                )
            
            conn.close()
            logger.info(f"📊 Loaded {len(self._positions)} positions from database")
            
        except Exception as e:
            logger.error(f"Error loading positions from DB: {e}")


# =============================================================================
# FLASK APP (Optional Web Interface)
# =============================================================================

def create_app(monitor: TradingMonitor = None) -> Any:
    """
    Create Flask app for web dashboard.
    
    Usage:
        from backend.monitoring_dashboard import create_app, TradingMonitor
        
        monitor = TradingMonitor()
        app = create_app(monitor)
        app.run(port=5001)
    """
    try:
        from flask import Flask, jsonify, render_template_string
    except ImportError:
        logger.warning("Flask not installed, web dashboard unavailable")
        return None
    
    app = Flask(__name__)
    monitor = monitor or TradingMonitor()
    
    DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Trading Bot Monitor</title>
    <style>
        body {
            font-family: 'SF Mono', 'Monaco', 'Inconsolata', monospace;
            background: #0d1117;
            color: #c9d1d9;
            margin: 0;
            padding: 20px;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        h1 { color: #58a6ff; border-bottom: 1px solid #30363d; padding-bottom: 10px; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; }
        .card {
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 6px;
            padding: 16px;
        }
        .card h2 { color: #58a6ff; margin-top: 0; font-size: 14px; text-transform: uppercase; }
        .metric { font-size: 32px; font-weight: bold; color: #58a6ff; }
        .metric.good { color: #3fb950; }
        .metric.bad { color: #f85149; }
        .metric.warn { color: #d29922; }
        .stat-row { display: flex; justify-content: space-between; padding: 4px 0; }
        .label { color: #8b949e; }
        .bar-container { height: 8px; background: #30363d; border-radius: 4px; margin: 4px 0; }
        .bar { height: 100%; border-radius: 4px; }
        .bar.blue { background: #58a6ff; }
        .bar.green { background: #3fb950; }
        .bar.red { background: #f85149; }
        .alert { background: #f8514922; border: 1px solid #f85149; border-radius: 4px; padding: 8px; margin: 4px 0; }
        table { width: 100%; border-collapse: collapse; font-size: 12px; }
        th, td { padding: 8px; text-align: left; border-bottom: 1px solid #30363d; }
        th { color: #8b949e; font-weight: normal; }
        .status-stuck { color: #f85149; }
        .status-at_risk { color: #d29922; }
        .status-profitable { color: #3fb950; }
        .refresh-btn { background: #238636; color: white; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer; }
        .refresh-btn:hover { background: #2ea043; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 Trading Bot Monitor</h1>
        <button class="refresh-btn" onclick="refresh()">Refresh</button>
        <div id="content">Loading...</div>
    </div>
    <script>
        async function refresh() {
            const resp = await fetch('/api/dashboard');
            const data = await resp.json();
            document.getElementById('content').innerHTML = renderDashboard(data);
        }
        
        function renderDashboard(d) {
            const healthClass = d.health_score > 0.7 ? 'good' : d.health_score > 0.4 ? 'warn' : 'bad';
            const rejRate = (d.rejections.rejection_rate * 100).toFixed(1);
            
            return `
            <div class="grid">
                <div class="card">
                    <h2>System Health</h2>
                    <div class="metric ${healthClass}">${(d.health_score * 100).toFixed(0)}%</div>
                    <div class="bar-container"><div class="bar ${healthClass}" style="width:${d.health_score*100}%"></div></div>
                </div>
                
                <div class="card">
                    <h2>Positions</h2>
                    <div class="stat-row"><span class="label">Total</span><span>${d.positions.total}</span></div>
                    <div class="stat-row"><span class="label">Stuck</span><span class="status-stuck">${d.positions.stuck}</span></div>
                    <div class="stat-row"><span class="label">At Risk</span><span class="status-at_risk">${d.positions.at_risk}</span></div>
                    <div class="stat-row"><span class="label">Profitable</span><span class="status-profitable">${d.positions.profitable}</span></div>
                </div>
                
                <div class="card">
                    <h2>Signal Rejections</h2>
                    <div class="stat-row"><span class="label">Total Signals</span><span>${d.rejections.total_signals}</span></div>
                    <div class="stat-row"><span class="label">Rejections</span><span>${d.rejections.total_rejections}</span></div>
                    <div class="stat-row"><span class="label">Rate</span><span class="${rejRate > 70 ? 'status-stuck' : rejRate > 50 ? 'status-at_risk' : ''}">${rejRate}%</span></div>
                </div>
                
                <div class="card">
                    <h2>Calibration</h2>
                    <div class="stat-row"><span class="label">Brier Score</span><span>${d.calibration.brier_score.toFixed(3)}</span></div>
                    <div class="stat-row"><span class="label">ECE</span><span>${d.calibration.ece.toFixed(3)}</span></div>
                    <div class="stat-row"><span class="label">Samples</span><span>${d.calibration.sample_count}</span></div>
                    <div class="stat-row"><span class="label">Accuracy</span><span>${(d.calibration.direction_accuracy*100).toFixed(1)}%</span></div>
                    <div class="stat-row"><span class="label">Trend</span><span>${d.calibration.trend}</span></div>
                </div>
                
                <div class="card" style="grid-column: span 2;">
                    <h2>Rejection Breakdown</h2>
                    ${Object.entries(d.rejections.by_reason).sort((a,b)=>b[1]-a[1]).map(([k,v]) => `
                        <div class="stat-row">
                            <span class="label">${k}</span>
                            <span>${v} (${(v/d.rejections.total_rejections*100).toFixed(1)}%)</span>
                        </div>
                    `).join('')}
                </div>
                
                <div class="card" style="grid-column: span 2;">
                    <h2>Open Positions</h2>
                    <table>
                        <tr><th>Symbol</th><th>Direction</th><th>PnL %</th><th>Hold Time</th><th>Status</th></tr>
                        ${d.positions.details.map(p => `
                            <tr>
                                <td>${p.option_symbol}</td>
                                <td>${p.direction}</td>
                                <td class="${p.pnl_pct > 0 ? 'status-profitable' : p.pnl_pct < -0.1 ? 'status-stuck' : ''}">${(p.pnl_pct*100).toFixed(1)}%</td>
                                <td>${p.hold_duration_minutes}m</td>
                                <td class="status-${p.status.toLowerCase()}">${p.status}</td>
                            </tr>
                        `).join('')}
                    </table>
                </div>
                
                ${d.alerts.length > 0 ? `
                <div class="card" style="grid-column: span 2;">
                    <h2>⚠️ Alerts</h2>
                    ${d.alerts.map(a => `
                        <div class="alert">
                            <strong>${a.symbol}</strong>: ${a.alerts.join(', ')}
                        </div>
                    `).join('')}
                </div>
                ` : ''}
                
                <div class="card" style="grid-column: span 2;">
                    <h2>Regime Performance</h2>
                    <table>
                        <tr><th>Regime</th><th>Trades</th><th>Win Rate</th><th>Total PnL</th><th>Avg PnL</th></tr>
                        ${Object.entries(d.regime_performance).map(([k,v]) => `
                            <tr>
                                <td>${k}</td>
                                <td>${v.trades}</td>
                                <td>${(v.win_rate*100).toFixed(1)}%</td>
                                <td>$${v.total_pnl.toFixed(2)}</td>
                                <td>$${v.avg_pnl.toFixed(2)}</td>
                            </tr>
                        `).join('')}
                    </table>
                </div>
            </div>
            `;
        }
        
        refresh();
        setInterval(refresh, 10000);  // Auto-refresh every 10s
    </script>
</body>
</html>
    """
    
    @app.route('/')
    def index():
        return render_template_string(DASHBOARD_HTML)
    
    @app.route('/api/dashboard')
    def api_dashboard():
        return jsonify(monitor.get_dashboard_data())
    
    @app.route('/api/rejections')
    def api_rejections():
        return monitor.get_rejection_report()
    
    @app.route('/api/positions')
    def api_positions():
        return monitor.get_position_report()
    
    return app


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    # Run web dashboard
    monitor = TradingMonitor()
    
    # Load from DB
    monitor.load_positions_from_db()
    
    # Create and run Flask app
    app = create_app(monitor)
    if app:
        print("🌐 Starting monitoring dashboard on http://localhost:5001")
        app.run(port=5001, debug=True)
    else:
        # Print reports to console if Flask not available
        print(monitor.get_rejection_report())
        print(monitor.get_position_report())









