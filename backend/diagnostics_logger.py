#!/usr/bin/env python3
"""
Diagnostics Logger
==================

Comprehensive logging for debugging low win rates.

Tracks per-minute:
- Hit/miss outcomes
- Raw vs calibrated confidence
- Horizon returns
- Rejection reason codes
- Time-of-day and volatility regime

Provides analysis tools to identify:
- Which hours/regimes fail
- Which confidence levels are miscalibrated
- Which horizons are most accurate
"""

import json
import sqlite3
import logging
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class DiagnosticRecord:
    """Single diagnostic record for a trading cycle."""
    timestamp: str
    cycle_number: int
    symbol: str
    
    # Price info
    current_price: float
    vix_level: float
    
    # Signal info
    action: str
    raw_confidence: float
    calibrated_confidence: float
    predicted_return: float
    
    # Multi-timeframe predictions
    pred_1m_direction: Optional[str] = None
    pred_1m_confidence: Optional[float] = None
    pred_5m_direction: Optional[str] = None
    pred_5m_confidence: Optional[float] = None
    pred_15m_direction: Optional[str] = None
    pred_15m_confidence: Optional[float] = None
    
    # Decision
    trade_placed: bool = False
    rejection_reasons: str = ""  # JSON list
    
    # Regime
    volatility_regime: str = "UNKNOWN"
    hour_of_day: int = 0
    minute_of_hour: int = 0
    day_of_week: int = 0
    
    # Calibration quality at decision time
    brier_score: float = 1.0
    ece: float = 1.0
    recent_accuracy: float = 0.0
    
    # Outcome (filled later after settlement)
    actual_return_1m: Optional[float] = None
    actual_return_5m: Optional[float] = None
    actual_return_15m: Optional[float] = None
    is_correct_1m: Optional[bool] = None
    is_correct_5m: Optional[bool] = None
    is_correct_15m: Optional[bool] = None
    settled: bool = False


class DiagnosticsLogger:
    """
    Comprehensive diagnostics logging and analysis.
    
    Saves detailed records to SQLite for post-hoc analysis.
    """
    
    def __init__(self, db_path: str = "data/diagnostics.db"):
        self.db_path = db_path
        self._init_db()
        
        # In-memory cache of recent records for quick analysis
        self.recent_records: List[DiagnosticRecord] = []
        self.max_recent = 1000
    
    def _init_db(self):
        """Initialize SQLite database."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS diagnostics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                cycle_number INTEGER,
                symbol TEXT,
                current_price REAL,
                vix_level REAL,
                action TEXT,
                raw_confidence REAL,
                calibrated_confidence REAL,
                predicted_return REAL,
                pred_1m_direction TEXT,
                pred_1m_confidence REAL,
                pred_5m_direction TEXT,
                pred_5m_confidence REAL,
                pred_15m_direction TEXT,
                pred_15m_confidence REAL,
                trade_placed INTEGER,
                rejection_reasons TEXT,
                volatility_regime TEXT,
                hour_of_day INTEGER,
                minute_of_hour INTEGER,
                day_of_week INTEGER,
                brier_score REAL,
                ece REAL,
                recent_accuracy REAL,
                actual_return_1m REAL,
                actual_return_5m REAL,
                actual_return_15m REAL,
                is_correct_1m INTEGER,
                is_correct_5m INTEGER,
                is_correct_15m INTEGER,
                settled INTEGER DEFAULT 0
            )
        ''')
        
        # Indices for fast queries
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_diag_timestamp ON diagnostics(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_diag_hour ON diagnostics(hour_of_day)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_diag_settled ON diagnostics(settled)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_diag_action ON diagnostics(action)')
        
        conn.commit()
        conn.close()
        
        logger.info(f"✅ Diagnostics database initialized: {self.db_path}")
    
    def log_cycle(
        self,
        cycle_number: int,
        symbol: str,
        current_price: float,
        signal: Dict,
        gate_result: Optional[Any] = None,
        multi_tf_preds: Optional[Dict] = None,
        calibration_quality: Optional[Dict] = None,
        vix_level: float = 17.0
    ) -> DiagnosticRecord:
        """
        Log a complete cycle's diagnostic information.
        
        Args:
            cycle_number: Current cycle number
            symbol: Trading symbol
            current_price: Current price
            signal: Signal dict with action, confidence, etc.
            gate_result: TradeGateResult from calibration gate
            multi_tf_preds: Multi-timeframe predictions
            calibration_quality: Current calibration metrics
            vix_level: Current VIX
            
        Returns:
            DiagnosticRecord
        """
        now = datetime.now()
        
        # Extract gate result info
        raw_conf = signal.get('confidence', 0)
        cal_conf = raw_conf
        rejection_reasons = []
        trade_placed = False
        vol_regime = "UNKNOWN"
        
        if gate_result:
            raw_conf = gate_result.raw_confidence
            cal_conf = gate_result.calibrated_confidence
            rejection_reasons = gate_result.rejection_reasons
            trade_placed = gate_result.should_trade
            vol_regime = gate_result.gating_metadata.get('volatility_regime', 'UNKNOWN')
        
        # Extract multi-timeframe predictions
        pred_1m = multi_tf_preds.get('1min', {}) if multi_tf_preds else {}
        pred_5m = multi_tf_preds.get('5min', {}) if multi_tf_preds else {}
        pred_15m = multi_tf_preds.get('15min', {}) if multi_tf_preds else {}
        
        # Calibration quality
        brier = calibration_quality.get('brier_score', 1.0) if calibration_quality else 1.0
        ece = calibration_quality.get('ece', 1.0) if calibration_quality else 1.0
        accuracy = calibration_quality.get('accuracy', 0.0) if calibration_quality else 0.0
        
        record = DiagnosticRecord(
            timestamp=now.isoformat(),
            cycle_number=cycle_number,
            symbol=symbol,
            current_price=current_price,
            vix_level=vix_level,
            action=signal.get('action', 'UNKNOWN'),
            raw_confidence=raw_conf,
            calibrated_confidence=cal_conf,
            predicted_return=signal.get('predicted_return', 0),
            pred_1m_direction=pred_1m.get('direction'),
            pred_1m_confidence=pred_1m.get('confidence'),
            pred_5m_direction=pred_5m.get('direction'),
            pred_5m_confidence=pred_5m.get('confidence'),
            pred_15m_direction=pred_15m.get('direction'),
            pred_15m_confidence=pred_15m.get('confidence'),
            trade_placed=trade_placed,
            rejection_reasons=json.dumps(rejection_reasons),
            volatility_regime=vol_regime,
            hour_of_day=now.hour,
            minute_of_hour=now.minute,
            day_of_week=now.weekday(),
            brier_score=brier,
            ece=ece,
            recent_accuracy=accuracy
        )
        
        # Save to DB
        self._save_record(record)
        
        # Cache in memory
        self.recent_records.append(record)
        if len(self.recent_records) > self.max_recent:
            self.recent_records.pop(0)
        
        return record
    
    def _save_record(self, record: DiagnosticRecord):
        """Save record to database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO diagnostics (
                    timestamp, cycle_number, symbol, current_price, vix_level,
                    action, raw_confidence, calibrated_confidence, predicted_return,
                    pred_1m_direction, pred_1m_confidence,
                    pred_5m_direction, pred_5m_confidence,
                    pred_15m_direction, pred_15m_confidence,
                    trade_placed, rejection_reasons,
                    volatility_regime, hour_of_day, minute_of_hour, day_of_week,
                    brier_score, ece, recent_accuracy
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                record.timestamp, record.cycle_number, record.symbol,
                record.current_price, record.vix_level,
                record.action, record.raw_confidence, record.calibrated_confidence,
                record.predicted_return,
                record.pred_1m_direction, record.pred_1m_confidence,
                record.pred_5m_direction, record.pred_5m_confidence,
                record.pred_15m_direction, record.pred_15m_confidence,
                int(record.trade_placed), record.rejection_reasons,
                record.volatility_regime, record.hour_of_day,
                record.minute_of_hour, record.day_of_week,
                record.brier_score, record.ece, record.recent_accuracy
            ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"Could not save diagnostic record: {e}")
    
    def settle_records(self, current_price: float, settlement_time: Optional[datetime] = None):
        """
        Settle unsettled records by calculating actual returns.
        
        Args:
            current_price: Current price for settlement
            settlement_time: Time of settlement (uses now if not provided)
        """
        now = settlement_time or datetime.now()
        settled_count = 0
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Find unsettled records older than 15 minutes
            cutoff = (now - timedelta(minutes=15)).isoformat()
            
            cursor.execute('''
                SELECT id, timestamp, current_price, pred_1m_direction,
                       pred_5m_direction, pred_15m_direction
                FROM diagnostics
                WHERE settled = 0 AND timestamp < ?
            ''', (cutoff,))
            
            rows = cursor.fetchall()
            
            for row in rows:
                record_id, ts, entry_price, dir_1m, dir_5m, dir_15m = row
                
                # Calculate actual returns (simplified - uses current price)
                actual_return = ((current_price - entry_price) / entry_price) * 100
                actual_direction = 'UP' if current_price > entry_price else 'DOWN'
                
                # Check if predictions were correct
                is_correct_1m = (dir_1m == actual_direction) if dir_1m else None
                is_correct_5m = (dir_5m == actual_direction) if dir_5m else None
                is_correct_15m = (dir_15m == actual_direction) if dir_15m else None
                
                cursor.execute('''
                    UPDATE diagnostics SET
                        actual_return_1m = ?,
                        actual_return_5m = ?,
                        actual_return_15m = ?,
                        is_correct_1m = ?,
                        is_correct_5m = ?,
                        is_correct_15m = ?,
                        settled = 1
                    WHERE id = ?
                ''', (
                    actual_return, actual_return, actual_return,
                    int(is_correct_1m) if is_correct_1m is not None else None,
                    int(is_correct_5m) if is_correct_5m is not None else None,
                    int(is_correct_15m) if is_correct_15m is not None else None,
                    record_id
                ))
                
                settled_count += 1
            
            conn.commit()
            conn.close()
            
            if settled_count > 0:
                logger.info(f"📊 Settled {settled_count} diagnostic records")
                
        except Exception as e:
            logger.warning(f"Could not settle diagnostic records: {e}")
    
    def get_hourly_analysis(self) -> Dict[int, Dict]:
        """
        Analyze performance by hour of day.
        
        Returns:
            Dict mapping hour -> {signals, trades, hit_rate, avg_confidence}
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT hour_of_day,
                       COUNT(*) as total,
                       SUM(CASE WHEN trade_placed = 1 THEN 1 ELSE 0 END) as trades,
                       SUM(CASE WHEN is_correct_15m = 1 THEN 1 ELSE 0 END) as correct,
                       SUM(CASE WHEN is_correct_15m IS NOT NULL THEN 1 ELSE 0 END) as settled,
                       AVG(calibrated_confidence) as avg_conf
                FROM diagnostics
                WHERE action != 'HOLD'
                GROUP BY hour_of_day
                ORDER BY hour_of_day
            ''')
            
            result = {}
            for row in cursor.fetchall():
                hour, total, trades, correct, settled, avg_conf = row
                hit_rate = correct / settled if settled > 0 else 0
                result[hour] = {
                    'signals': total,
                    'trades': trades,
                    'settled': settled,
                    'correct': correct,
                    'hit_rate': hit_rate,
                    'avg_confidence': avg_conf or 0
                }
            
            conn.close()
            return result
            
        except Exception as e:
            logger.warning(f"Could not get hourly analysis: {e}")
            return {}
    
    def get_regime_analysis(self) -> Dict[str, Dict]:
        """
        Analyze performance by volatility regime.
        
        Returns:
            Dict mapping regime -> {signals, trades, hit_rate}
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT volatility_regime,
                       COUNT(*) as total,
                       SUM(CASE WHEN trade_placed = 1 THEN 1 ELSE 0 END) as trades,
                       SUM(CASE WHEN is_correct_15m = 1 THEN 1 ELSE 0 END) as correct,
                       SUM(CASE WHEN is_correct_15m IS NOT NULL THEN 1 ELSE 0 END) as settled
                FROM diagnostics
                WHERE action != 'HOLD'
                GROUP BY volatility_regime
            ''')
            
            result = {}
            for row in cursor.fetchall():
                regime, total, trades, correct, settled = row
                hit_rate = correct / settled if settled > 0 else 0
                result[regime] = {
                    'signals': total,
                    'trades': trades,
                    'settled': settled,
                    'correct': correct,
                    'hit_rate': hit_rate
                }
            
            conn.close()
            return result
            
        except Exception as e:
            logger.warning(f"Could not get regime analysis: {e}")
            return {}
    
    def get_rejection_analysis(self) -> Dict[str, int]:
        """
        Analyze rejection reasons to identify common issues.
        
        Returns:
            Dict mapping rejection reason -> count
        """
        reason_counts = defaultdict(int)
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT rejection_reasons
                FROM diagnostics
                WHERE trade_placed = 0 AND action != 'HOLD' AND rejection_reasons != '[]'
            ''')
            
            for (reasons_json,) in cursor.fetchall():
                try:
                    reasons = json.loads(reasons_json)
                    for reason in reasons:
                        # Normalize reason (take first part before numbers)
                        normalized = reason.split('(')[0].strip()
                        reason_counts[normalized] += 1
                except json.JSONDecodeError:
                    pass
            
            conn.close()
            
        except Exception as e:
            logger.warning(f"Could not get rejection analysis: {e}")
        
        return dict(sorted(reason_counts.items(), key=lambda x: -x[1]))
    
    def get_confidence_calibration_analysis(self, n_bins: int = 10) -> Dict[str, Any]:
        """
        Analyze how well confidence scores predict outcomes.
        
        Returns:
            Dict with per-bin accuracy vs confidence
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT calibrated_confidence, is_correct_15m
                FROM diagnostics
                WHERE is_correct_15m IS NOT NULL AND trade_placed = 1
            ''')
            
            rows = cursor.fetchall()
            conn.close()
            
            if not rows:
                return {}
            
            # Bin by confidence
            bins = defaultdict(list)
            for conf, correct in rows:
                bin_idx = min(int(conf * n_bins), n_bins - 1)
                bin_name = f"{bin_idx * 10}-{(bin_idx + 1) * 10}%"
                bins[bin_name].append(correct)
            
            result = {}
            for bin_name, outcomes in sorted(bins.items()):
                accuracy = sum(outcomes) / len(outcomes) if outcomes else 0
                result[bin_name] = {
                    'count': len(outcomes),
                    'accuracy': accuracy,
                    'expected_accuracy': (int(bin_name.split('-')[0]) + 5) / 100
                }
            
            return result
            
        except Exception as e:
            logger.warning(f"Could not get calibration analysis: {e}")
            return {}
    
    def print_summary_report(self):
        """Print a comprehensive diagnostic report."""
        logger.info("=" * 70)
        logger.info("📊 DIAGNOSTIC SUMMARY REPORT")
        logger.info("=" * 70)
        
        # Hourly analysis
        hourly = self.get_hourly_analysis()
        if hourly:
            logger.info("\n📅 HOURLY PERFORMANCE:")
            best_hour = max(hourly.items(), key=lambda x: x[1]['hit_rate']) if hourly else None
            worst_hour = min(hourly.items(), key=lambda x: x[1]['hit_rate']) if hourly else None
            
            if best_hour:
                logger.info(f"   Best hour: {best_hour[0]}:00 ({best_hour[1]['hit_rate']:.1%} hit rate)")
            if worst_hour:
                logger.info(f"   Worst hour: {worst_hour[0]}:00 ({worst_hour[1]['hit_rate']:.1%} hit rate)")
        
        # Regime analysis
        regimes = self.get_regime_analysis()
        if regimes:
            logger.info("\n🌡️ VOLATILITY REGIME PERFORMANCE:")
            for regime, stats in regimes.items():
                logger.info(f"   {regime}: {stats['hit_rate']:.1%} ({stats['trades']} trades)")
        
        # Rejection analysis
        rejections = self.get_rejection_analysis()
        if rejections:
            logger.info("\n🚫 TOP REJECTION REASONS:")
            for reason, count in list(rejections.items())[:5]:
                logger.info(f"   {reason}: {count}")
        
        # Calibration analysis
        calibration = self.get_confidence_calibration_analysis()
        if calibration:
            logger.info("\n📈 CONFIDENCE CALIBRATION:")
            for bin_name, stats in calibration.items():
                expected = stats['expected_accuracy']
                actual = stats['accuracy']
                gap = actual - expected
                gap_indicator = "✅" if abs(gap) < 0.1 else "⚠️" if gap < 0 else "📈"
                logger.info(f"   {bin_name}: {actual:.1%} actual vs {expected:.1%} expected {gap_indicator}")
        
        logger.info("=" * 70)


# Factory function
def create_diagnostics_logger(config: Optional[Dict] = None) -> DiagnosticsLogger:
    """Create a DiagnosticsLogger with configuration."""
    db_path = "data/diagnostics.db"
    
    if config:
        db_path = config.get('db_path', db_path)
    else:
        try:
            config_path = Path('config.json')
            if config_path.exists():
                with open(config_path, 'r') as f:
                    cfg = json.load(f)
                db_path = cfg.get('diagnostics', {}).get('db_path', db_path)
        except Exception:
            pass
    
    return DiagnosticsLogger(db_path=db_path)

