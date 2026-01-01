#!/usr/bin/env python3
"""
Training Dashboard API Server
=============================

Serves real-time training data for the web dashboard.
Run this WHILE train_then_go_live.py is running.

Usage:
    python training_dashboard_server.py

Configuration is loaded from config.json under the "dashboard" section.

Version: 2.4 - Fixed prediction history unbounded growth causing chart slowdown
"""

# Version for debugging
DASHBOARD_VERSION = "2.4-fix-prediction-history-growth"

import json
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional

from flask import Flask, jsonify, send_file
from flask_cors import CORS

from backend.dashboard.state import TrainingState, TradeRecord, ValidatedPrediction
from backend.dashboard.log_parser import LogParser
from backend.dashboard.db_loader import DatabaseLoader

# Import centralized time utilities for consistent market time
try:
    from backend.time_utils import get_market_time, format_timestamp
except ImportError:
    # Fallback if time_utils not available
    import pytz
    def get_market_time():
        market_tz = pytz.timezone('US/Eastern')
        return pytz.UTC.localize(datetime.utcnow()).astimezone(market_tz).replace(tzinfo=None)
    def format_timestamp(dt):
        return dt.strftime('%Y-%m-%dT%H:%M:%S') if dt else ""

# =============================================================================
# CONFIGURATION
# =============================================================================

def load_config() -> Dict[str, Any]:
    """Load dashboard configuration from config.json."""
    config_path = Path('config.json')
    
    defaults = {
        'port': 5001,
        'host': '0.0.0.0',
        'paper_trading_db': 'data/paper_trading.db',
        'historical_db': 'data/db/historical.db',
        'log_pattern': 'logs/real_bot_simulation*.log',
        'refresh_interval_ms': 2000,
        'chart': {
            'max_price_points': 10000,
            'max_trades': 200,
            'max_lstm_predictions': 500,
            'max_validated_predictions': 200,
            'lookback_days': 3
        },
        'state': {
            'max_recent_trades': 10,
            'max_log_trades': 50,
            'max_archived_sessions': 5,
            'initial_log_lines': 500000  # Parse entire log for completed runs
        }
    }
    
    try:
        if config_path.exists():
            with open(config_path, 'r') as f:
                cfg = json.load(f)
            dashboard_cfg = cfg.get('training_dashboard', cfg.get('dashboard', {}))
            
            # Merge with defaults
            for key, value in dashboard_cfg.items():
                if isinstance(value, dict) and key in defaults:
                    defaults[key].update(value)
                else:
                    defaults[key] = value
            
            # Get initial balance from trading config
            defaults['initial_balance'] = cfg.get('trading', {}).get('initial_balance', 1000.0)
    except Exception as e:
        print(f"Warning: Could not load config: {e}")
        defaults['initial_balance'] = 1000.0
    
    return defaults


# Load config
CONFIG = load_config()

# =============================================================================
# FLASK APP
# =============================================================================

app = Flask(__name__)
CORS(app)

# =============================================================================
# STATE MANAGEMENT
# =============================================================================

# Global state
state = TrainingState()
state_lock = threading.Lock()

# Initialize components
log_parser = LogParser(
    log_pattern=CONFIG['log_pattern'],
    fallback_pattern='../logs/real_bot_simulation*.log',
    initial_lines=CONFIG['state']['initial_log_lines']
)

db_loader = DatabaseLoader(
    paper_db_path=CONFIG['paper_trading_db'],
    historical_db_path=CONFIG['historical_db'],
    trade_filter='paper'  # Training dashboard shows only paper trades
)


def aggregate_predictions_by_target_time(predictions_history: list, current_time_str: str) -> dict:
    """
    Aggregate overlapping predictions using SMART ENSEMBLE:
    
    1. RECENCY WEIGHTING: Predictions made closer to target time get more weight
       (a 5min prediction is more reliable than a 30min prediction made 25min ago)
    
    2. SHORTER TIMEFRAME BONUS: Shorter timeframe predictions are inherently more accurate
    
    3. AGREEMENT BOOST: When predictions agree, confidence increases
    
    4. DECAY: Old predictions matter less as new information arrives
    """
    from datetime import datetime, timedelta
    import math
    
    # Timeframe to minutes mapping
    tf_minutes = {
        '5min': 5, '10min': 10, '15min': 15, '20min': 20, '30min': 30,
        '40min': 40, '50min': 50, '1h': 60, '1hour': 60, '1h20m': 80, 
        '1h40m': 100, '2h': 120, '4hour': 240
    }
    
    # Parse current time
    try:
        if current_time_str:
            current_time = datetime.fromisoformat(current_time_str.replace('Z', '+00:00').replace(' ', 'T'))
            # Strip timezone if present (we work with naive market time)
            if current_time.tzinfo is not None:
                current_time = current_time.replace(tzinfo=None)
        else:
            current_time = get_market_time()
    except Exception:
        current_time = get_market_time()
    
    # Group predictions by their target time
    predictions_by_target = {}
    
    for pred in predictions_history:
        if isinstance(pred, dict):
            pred_timestamp = pred.get('timestamp')
            predictions = pred.get('predictions', {})
            spy_price = pred.get('spy_price', 0)
        else:
            pred_timestamp = getattr(pred, 'timestamp', None)
            predictions = getattr(pred, 'predictions', {})
            spy_price = getattr(pred, 'spy_price', 0)
        
        if not pred_timestamp or not predictions:
            continue
        
        try:
            pred_time = datetime.fromisoformat(pred_timestamp.replace('Z', '+00:00'))
        except Exception:
            continue
        
        for tf_name, tf_data in predictions.items():
            if tf_name not in tf_minutes:
                continue
            
            if isinstance(tf_data, dict):
                direction = tf_data.get('direction', 'N/A')
                pred_return = tf_data.get('return', 0)
                confidence = tf_data.get('confidence', 0)
            else:
                continue
            
            if direction == 'N/A' or confidence <= 0:
                continue
            
            minutes = tf_minutes[tf_name]
            target_time = pred_time + timedelta(minutes=minutes)
            target_key = target_time.strftime('%Y-%m-%d %H:%M')
            
            # Only include future predictions (within next 2 hours)
            if target_time <= current_time:
                continue
            if target_time > current_time + timedelta(hours=2):
                continue
            
            # Calculate time until target (in minutes)
            time_to_target = (target_time - current_time).total_seconds() / 60
            
            # Calculate age of prediction (how long ago it was made)
            pred_age = (current_time - pred_time).total_seconds() / 60
            
            # CRITICAL: Skip predictions that are too old - they cause drift!
            # Old predictions were made at different price levels and will pull the average
            # away from current predictions. Only use predictions made within last 30 minutes.
            if pred_age > 30:
                continue
            
            if target_key not in predictions_by_target:
                predictions_by_target[target_key] = []
            
            # Calculate the ABSOLUTE target price (not relative return)
            # This is what the model actually predicted the price would be
            target_price = spy_price * (1 + pred_return / 100) if spy_price > 0 else 0
            
            predictions_by_target[target_key].append({
                'target_time': target_key,
                'target_time_dt': target_time,
                'prediction_made_at': pred_timestamp,
                'pred_time_dt': pred_time,
                'source_timeframe': tf_name,
                'minutes_ahead': minutes,
                'time_to_target': time_to_target,
                'pred_age': pred_age,
                'direction': direction,
                'predicted_return': pred_return,
                'confidence': confidence,
                'entry_price': spy_price,
                'target_price': target_price  # NEW: Absolute price prediction
            })
    
    # Calculate SMART ENSEMBLE predictions for each target time
    best_predictions = {}
    for target_key, preds in predictions_by_target.items():
        if not preds:
            continue
        
        total = len(preds)
        
        # ===== CALCULATE SMART WEIGHTS FOR EACH PREDICTION =====
        weights = []
        for p in preds:
            base_conf = p.get('confidence', 25) / 100  # Normalize to 0-1
            
            # 1. RECENCY WEIGHT: More recent predictions get higher weight
            # Exponential decay based on prediction age
            # Half-life of 10 minutes: predictions from 10min ago have 50% weight
            pred_age = max(0, p.get('pred_age', 0))
            recency_weight = math.exp(-0.069 * pred_age)  # 0.069 = ln(2)/10 for 10min half-life
            
            # 2. SHORTER TIMEFRAME BONUS: 5min predictions are more accurate than 60min
            # Scale: 5min=1.5x, 15min=1.2x, 30min=1.0x, 60min=0.8x
            tf_minutes_val = p.get('minutes_ahead', 30)
            timeframe_weight = max(0.5, 1.5 - (tf_minutes_val / 60))
            
            # 3. PROXIMITY BONUS: As target approaches, predictions become more reliable
            # Weight increases as time_to_target decreases
            time_to_target = max(1, p.get('time_to_target', 30))
            proximity_weight = min(2.0, 30 / time_to_target)  # 2x weight when 15min away
            
            # Combined weight
            weight = base_conf * recency_weight * timeframe_weight * proximity_weight
            weights.append(weight)
        
        total_weight = sum(weights)
        if total_weight == 0:
            total_weight = 1
        
        # ===== WEIGHTED TARGET PRICE (absolute, not relative!) =====
        # Use target_price instead of predicted_return to avoid drift
        weighted_target_price = sum(p.get('target_price', 0) * w for p, w in zip(preds, weights)) / total_weight
        
        # Also calculate weighted return for compatibility (but this can drift)
        weighted_return = sum(p.get('predicted_return', 0) * w for p, w in zip(preds, weights)) / total_weight
        
        # ===== DIRECTION VOTING (weighted) =====
        bullish_weight = sum(w for p, w in zip(preds, weights) if p.get('direction') == 'UP')
        bearish_weight = sum(w for p, w in zip(preds, weights) if p.get('direction') == 'DOWN')
        neutral_weight = total_weight - bullish_weight - bearish_weight
        
        # Count for consensus display
        bullish = sum(1 for p in preds if p.get('direction') == 'UP')
        bearish = sum(1 for p in preds if p.get('direction') == 'DOWN')
        neutral = total - bullish - bearish
        
        # Determine direction by weighted vote
        if bullish_weight > bearish_weight and bullish_weight > neutral_weight:
            direction = 'UP'
        elif bearish_weight > bullish_weight and bearish_weight > neutral_weight:
            direction = 'DOWN'
        else:
            direction = 'NEUTRAL'
        
        # ===== AGREEMENT-ADJUSTED CONFIDENCE =====
        # Base confidence is weighted average
        base_confidence = sum(p.get('confidence', 25) * w for p, w in zip(preds, weights)) / total_weight
        
        # Agreement ratio based on weighted votes
        max_vote_weight = max(bullish_weight, bearish_weight, neutral_weight)
        agreement_ratio = max_vote_weight / total_weight if total_weight > 0 else 0.5
        
        # Agreement multiplier: 100% agreement = 1.5x, 50% = 1.1x, 33% = 0.97x
        agreement_multiplier = 0.7 + (0.8 * agreement_ratio)
        
        # Volume bonus: More predictions = higher confidence (up to 1.3x for 10+ predictions)
        volume_bonus = min(1.3, 1.0 + (total - 1) * 0.03)
        
        adjusted_confidence = min(100.0, base_confidence * agreement_multiplier * volume_bonus)
        
        # Get entry price from most recent prediction
        latest = max(preds, key=lambda p: p.get('prediction_made_at', ''))
        
        best_predictions[target_key] = {
            'target_time': target_key,
            'predicted_return': weighted_return,
            'target_price': weighted_target_price,  # NEW: Use this for charting!
            'confidence': adjusted_confidence,
            'base_confidence': base_confidence,
            'direction': direction,
            'entry_price': latest.get('entry_price', 0),
            'consensus': {
                'bullish_votes': bullish,
                'bearish_votes': bearish,
                'neutral_votes': neutral,
                'total_predictions': total,
                'agreement_ratio': agreement_ratio,
                'agreement_multiplier': agreement_multiplier,
                'volume_bonus': volume_bonus
            }
        }
    
    return best_predictions


def validate_lstm_predictions() -> None:
    """
    Check past TCN predictions against actual price movement.
    Marks predictions as correct/incorrect once enough time has passed.
    (Function named lstm_* for API compatibility)
    """
    global state
    
    if not state.lstm_predictions_history or state.market.spy_price <= 0:
        return
    
    if not state.session.simulated_datetime:
        return
    
    try:
        current_time = datetime.strptime(
            f"{state.session.simulated_date} {state.session.simulated_time}",
            '%Y-%m-%d %H:%M:%S'
        )
    except Exception:
        return
    
    # Timeframes to validate
    timeframes = [('5min', 5), ('10min', 10), ('15min', 15), ('30min', 30)]
    
    for pred in state.lstm_predictions_history:
        # Handle both dict and dataclass
        if isinstance(pred, dict):
            if pred.get('validated'):
                continue
            pred_timestamp = pred.get('timestamp')
            pred_price = pred.get('spy_price', 0)
            predictions = pred.get('predictions', {})
        else:
            if getattr(pred, 'validated', False):
                continue
            pred_timestamp = getattr(pred, 'timestamp', None)
            pred_price = getattr(pred, 'spy_price', 0)
            predictions = getattr(pred, 'predictions', {})
        
        if not pred_timestamp or not pred_price or pred_price <= 0:
            continue
        
        try:
            pred_time = datetime.fromisoformat(pred_timestamp)
        except Exception:
            continue
        
        minutes_elapsed = (current_time - pred_time).total_seconds() / 60
        
        # Validate each timeframe
        for tf_name, tf_minutes in timeframes:
            validated_key = f'{tf_name}_validated'
            
            # Check if already validated
            if isinstance(pred, dict):
                already_validated = pred.get(validated_key, False)
            else:
                already_validated = getattr(pred, validated_key, False)
            
            if already_validated or minutes_elapsed < tf_minutes:
                continue
            
            if tf_name not in predictions:
                continue
            
            pred_data = predictions[tf_name]
            direction = pred_data.get('direction') if isinstance(pred_data, dict) else getattr(pred_data, 'direction', None)
            
            if not direction or direction == 'N/A':
                continue
            
            predicted_up = direction == 'UP'
            actual_up = state.market.spy_price > pred_price
            is_correct = predicted_up == actual_up
            actual_move = ((state.market.spy_price - pred_price) / pred_price) * 100
            
            # Mark as validated
            if isinstance(pred, dict):
                pred[validated_key] = True
                pred[f'{tf_name}_correct'] = is_correct
                pred[f'{tf_name}_actual_price'] = state.market.spy_price
                pred[f'{tf_name}_actual_move'] = actual_move
            
            # Store validated prediction
            confidence = pred_data.get('confidence', 0) if isinstance(pred_data, dict) else getattr(pred_data, 'confidence', 0)
            pred_return = pred_data.get('return', 0) if isinstance(pred_data, dict) else getattr(pred_data, 'return', 0)
            
            validated = ValidatedPrediction(
                timestamp=pred_timestamp,
                validation_time=current_time.isoformat(),
                timeframe=tf_name,
                predicted_direction=direction,
                predicted_return=pred_return,
                confidence=confidence,
                actual_direction='UP' if actual_up else 'DOWN',
                actual_move_pct=actual_move,
                is_correct=is_correct,
                entry_price=pred_price,
                exit_price=state.market.spy_price
            )
            state.lstm_validated_predictions.append(validated)
        
        # Mark fully validated if key timeframes checked
        if isinstance(pred, dict):
            if pred.get('5min_validated') and pred.get('15min_validated') and pred.get('30min_validated'):
                pred['validated'] = True
    
    # Keep only last N validated predictions
    max_validated = CONFIG['chart']['max_validated_predictions']
    if len(state.lstm_validated_predictions) > max_validated:
        state.lstm_validated_predictions = state.lstm_validated_predictions[-max_validated:]
    
    # Keep only last N predictions in history to prevent unbounded growth
    max_predictions = CONFIG['chart']['max_lstm_predictions']
    if len(state.lstm_predictions_history) > max_predictions:
        state.lstm_predictions_history = state.lstm_predictions_history[-max_predictions:]


def refresh_data() -> None:
    """Refresh all dashboard data."""
    global state
    
    with state_lock:
        # Initialize start time
        if state.session.start_time == 0:
            state.session.start_time = time.time()
        
        # Set initial balance
        state.account.initial_balance = CONFIG['initial_balance']
        if state.account.current_balance == 1000.0:
            state.account.current_balance = state.account.initial_balance
        
        # Parse log file
        log_path = log_parser.find_latest_log()
        if log_path:
            log_parser.parse_log_file(log_path, state)
        
        # Load from database
        db_loader.load_all(state)
        
        # Fallback: Get simulated time from database if not in log
        if (state.session.simulated_date == "Not started" or not state.session.simulated_time) and state.recent_trades:
            # Use most recent trade timestamp as simulated time
            for trade in reversed(state.recent_trades):
                trade_time = getattr(trade, 'time', None) or (trade.get('time') if isinstance(trade, dict) else None)
                trade_date = getattr(trade, 'date', None) or (trade.get('date') if isinstance(trade, dict) else None)
                if trade_time and trade_date and trade_time != '--' and trade_date != '--':
                    # Convert MM/DD to YYYY-MM-DD (assume current year)
                    try:
                        year = get_market_time().year
                        state.session.simulated_date = f"{year}-{trade_date.replace('/', '-')}"
                        state.session.simulated_time = f"{trade_time}:00" if len(trade_time) == 5 else trade_time
                        break
                    except Exception:
                        pass
        
        # Validate TCN predictions
        validate_lstm_predictions()
        
        # Populate recent_trades from log_trades if DB is empty
        if not state.recent_trades and state.log_trades:
            for trade in state.log_trades[-CONFIG['state']['max_recent_trades']:]:
                try:
                    if hasattr(trade, 'entry_time'):
                        entry_time = trade.entry_time
                    else:
                        entry_time = trade.get('entry_time', '')
                    
                    if entry_time:
                        dt = datetime.fromisoformat(entry_time)
                        trade_date = dt.strftime('%m/%d')
                        trade_time = dt.strftime('%H:%M')
                    else:
                        trade_date = '--'
                        trade_time = '--'
                    
                    exit_time_str = '--'
                    exit_time = trade.exit_time if hasattr(trade, 'exit_time') else trade.get('exit_time')
                    if exit_time:
                        try:
                            exit_dt = datetime.fromisoformat(exit_time)
                            exit_time_str = exit_dt.strftime('%H:%M')
                        except Exception:
                            pass
                    
                    record = TradeRecord(
                        date=trade_date,
                        time=trade_time,
                        action=trade.type if hasattr(trade, 'type') else trade.get('type', 'UNKNOWN'),
                        strike=trade.strike if hasattr(trade, 'strike') else trade.get('strike', 0) or 0,
                        entry_price=trade.entry_price if hasattr(trade, 'entry_price') else trade.get('entry_price', 0),
                        exit_price=trade.exit_price if hasattr(trade, 'exit_price') else trade.get('exit_price', 0),
                        exit_time=exit_time_str,
                        pnl=trade.pnl if hasattr(trade, 'pnl') else trade.get('pnl', 0) or 0,
                        status=trade.status if hasattr(trade, 'status') else trade.get('status', 'UNKNOWN')
                    )
                    state.recent_trades.append(record)
                except Exception as e:
                    print(f"Error converting log trade: {e}")
        
        state.session.last_update = format_timestamp(get_market_time())


# =============================================================================
# API ROUTES
# =============================================================================

@app.route('/')
def index():
    """Serve the training dashboard HTML."""
    return send_file('training_dashboard.html')


@app.route('/api/data')
def get_data():
    """Get current training dashboard data."""
    refresh_data()
    
    with state_lock:
        data = state.to_api_dict()
        
        # Calculate derived metrics
        elapsed = time.time() - state.session.start_time if state.session.start_time > 0 else 0
        data['elapsed_seconds'] = elapsed
        data['signal_rate'] = (state.progress.signals / max(state.progress.cycles, 1)) * 100
        data['win_rate'] = state.account.win_rate
        
        # TCN prediction accuracy metrics (named lstm_* for API compatibility)
        for tf in ['5min', '10min', '15min', '30min']:
            correct = sum(
                1 for p in state.lstm_validated_predictions
                if (p.timeframe if hasattr(p, 'timeframe') else p.get('timeframe')) == tf
                and (p.is_correct if hasattr(p, 'is_correct') else p.get('is_correct'))
            )
            total = sum(
                1 for p in state.lstm_validated_predictions
                if (p.timeframe if hasattr(p, 'timeframe') else p.get('timeframe')) == tf
            )
            data[f'lstm_accuracy_{tf}'] = (correct / max(total, 1)) * 100
            data[f'lstm_correct_{tf}'] = correct
            data[f'lstm_total_{tf}'] = total
        
        # Overall accuracy
        total_correct = sum(
            1 for p in state.lstm_validated_predictions
            if (p.is_correct if hasattr(p, 'is_correct') else p.get('is_correct'))
        )
        total_all = len(state.lstm_validated_predictions)
        data['lstm_accuracy_overall'] = (total_correct / max(total_all, 1)) * 100
        data['lstm_total_predictions'] = total_all
    
    return jsonify(data)


@app.route('/api/health')
def health():
    """Health check endpoint."""
    return jsonify({'status': 'ok', 'time': format_timestamp(get_market_time()), 'timezone': 'US/Eastern'})


@app.route('/api/runs')
def list_runs():
    """List all active/recent training runs based on log files."""
    import glob
    from pathlib import Path

    runs = []

    # Find all recent log files (modified in last 2 hours)
    log_pattern = CONFIG.get('log_pattern', 'logs/real_bot_simulation*.log')
    cutoff_time = time.time() - (2 * 60 * 60)  # 2 hours ago

    for log_path in glob.glob(log_pattern):
        try:
            path = Path(log_path)
            if not path.exists() or path.stat().st_size == 0:
                continue

            mtime = path.stat().st_mtime
            if mtime < cutoff_time:
                continue

            # Extract run name from log file
            run_name = path.stem.replace('real_bot_simulation_', '')

            # Check if actively being written (modified in last 30 seconds)
            is_active = (time.time() - mtime) < 30

            runs.append({
                'run_name': run_name,
                'log_file': str(path),
                'size_mb': round(path.stat().st_size / (1024 * 1024), 1),
                'last_modified': datetime.fromtimestamp(mtime).isoformat(),
                'is_active': is_active
            })
        except Exception as e:
            continue

    # Sort by last modified (most recent first)
    runs.sort(key=lambda x: x['last_modified'], reverse=True)

    return jsonify({
        'success': True,
        'runs': runs,
        'active_count': sum(1 for r in runs if r['is_active'])
    })


@app.route('/api/reset-session', methods=['POST'])
def reset_session():
    """Manually reset the training session."""
    try:
        with state_lock:
            state.reset_for_new_session(None, CONFIG['initial_balance'])
        return jsonify({'success': True, 'message': 'Session reset'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/chart')
def get_chart_data():
    """Get SPY price history + trades for chart."""
    try:
        # Refresh state to get latest simulated time
        refresh_data()
        
        # Get reference time
        simulated_datetime = state.session.simulated_datetime
        
        # Find trade time range as fallback
        trade_time_range = None
        if state.log_trades:
            trade_times = []
            for t in state.log_trades:
                entry_time = t.entry_time if hasattr(t, 'entry_time') else t.get('entry_time')
                if entry_time:
                    trade_times.append(entry_time)
            if trade_times:
                trade_time_range = max(trade_times)
        
        reference_time = simulated_datetime or trade_time_range

        # For live/paper trading, use current market time if no reference
        if not reference_time:
            from datetime import datetime
            import pytz
            market_tz = pytz.timezone('US/Eastern')
            reference_time = datetime.now(market_tz).strftime('%Y-%m-%dT%H:%M:%S')

        # Load data
        chart_cfg = CONFIG['chart']
        
        print(f"[CHART] Loading prices for ref_time={reference_time}")
        
        spy_prices = db_loader.load_spy_prices(
            reference_time=reference_time,
            lookback_days=chart_cfg['lookback_days'],
            max_points=chart_cfg['max_price_points']
        )
        
        # Load VIX with Bollinger Bands
        vix_data = db_loader.load_vix_prices(
            reference_time=reference_time,
            lookback_days=chart_cfg['lookback_days'],
            max_points=chart_cfg['max_price_points'],
            bb_period=20
        )
        
        print(f"[CHART] Loaded {len(spy_prices)} SPY prices, {len(vix_data.get('vix_prices', []))} VIX prices")
        
        trades, annotations = db_loader.load_trades_for_chart(
            reference_time=reference_time,
            lookback_days=chart_cfg['lookback_days'],
            max_trades=chart_cfg['max_trades']
        )
        
        # Fallback to log trades
        if not trades and state.log_trades:
            for trade in state.log_trades:
                entry_time = trade.entry_time if hasattr(trade, 'entry_time') else trade.get('entry_time')
                exit_time = trade.exit_time if hasattr(trade, 'exit_time') else trade.get('exit_time')
                trade_type = trade.type if hasattr(trade, 'type') else trade.get('type', 'UNKNOWN')
                entry_price = trade.entry_price if hasattr(trade, 'entry_price') else trade.get('entry_price', 0)
                pnl = trade.pnl if hasattr(trade, 'pnl') else trade.get('pnl')
                status = trade.status if hasattr(trade, 'status') else trade.get('status', 'OPEN')
                strike = trade.strike if hasattr(trade, 'strike') else trade.get('strike')
                
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': exit_time,
                    'type': trade_type,
                    'entry_price': entry_price,
                    'strike': strike,
                    'pnl': pnl,
                    'status': status,
                    'color': '#22c55e' if (pnl and pnl > 0) else '#ef4444' if (pnl and pnl < 0) else '#a855f7'
                })
                
                if entry_time:
                    strike_str = f" ${strike:.0f}" if strike else ""
                    annotations.append({
                        'timestamp': entry_time,
                        'type': 'entry',
                        'option_type': trade_type,
                        'strike': strike,
                        'label': f"📚 {trade_type}{strike_str}"
                    })
        
        # Build open positions details for API
        open_positions = []
        for pos in state.open_positions_details:
            if hasattr(pos, '__dataclass_fields__'):
                from dataclasses import asdict
                open_positions.append(asdict(pos))
            else:
                open_positions.append(pos)
        
        # Build TCN predictions history for API (named lstm_* for compatibility)
        lstm_history = []
        for pred in state.lstm_predictions_history:
            if hasattr(pred, '__dataclass_fields__'):
                from dataclasses import asdict
                lstm_history.append(asdict(pred))
            else:
                lstm_history.append(pred)
        
        # Build validated predictions for API
        validated = []
        for pred in state.lstm_validated_predictions:
            if hasattr(pred, '__dataclass_fields__'):
                from dataclasses import asdict
                validated.append(asdict(pred))
            else:
                validated.append(pred)
        
        # Aggregate predictions by target time (NEW!)
        # This shows the highest confidence prediction for each future minute
        aggregated_by_target = aggregate_predictions_by_target_time(
            lstm_history,
            simulated_datetime
        )
        
        # Convert to sorted list for frontend
        aggregated_list = []
        for target_key in sorted(aggregated_by_target.keys()):
            pred = aggregated_by_target[target_key]
            aggregated_list.append(pred)
        
        return jsonify({
            'spy_prices': spy_prices,
            'trades': trades,
            'annotations': annotations,
            'simulated_time': simulated_datetime,
            'cycles': state.progress.cycles,
            'lstm_predictions': lstm_history,
            'current_lstm': state.signal.lstm_predictions,
            'open_positions': open_positions,
            'current_price': state.market.spy_price,
            'lstm_validated': validated,
            'aggregated_predictions': aggregated_list,  # Best prediction per target time
            # VIX with Bollinger Bands
            'vix_prices': vix_data.get('vix_prices', []),
            'vix_bb_upper': vix_data.get('bb_upper', []),
            'vix_bb_lower': vix_data.get('bb_lower', []),
            'vix_bb_middle': vix_data.get('bb_middle', [])
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'spy_prices': [],
            'trades': [],
            'annotations': [],
            'lstm_predictions': [],
            'current_lstm': {},
            'open_positions': [],
            'current_price': 0,
            'lstm_validated': [],
            'aggregated_predictions': [],
            'vix_prices': [],
            'vix_bb_upper': [],
            'vix_bb_lower': [],
            'vix_bb_middle': []
        })


@app.route('/api/missed-opportunities')
def get_missed_opportunities():
    """Get counterfactual learning data - signals that were rejected and their outcomes."""
    try:
        # Try to get analyzed signals from the running bot's RL threshold learner
        # Look for a shared state file or import from the running process
        missed_opps_path = Path('data/missed_opportunities.json')
        
        if missed_opps_path.exists():
            import json
            with open(missed_opps_path, 'r') as f:
                data = json.load(f)
            return jsonify(data)
        
        # Fallback: return empty list if no data file exists yet
        return jsonify({
            'signals': [],
            'stats': {
                'total_missed_winners': 0,
                'total_avoided_losers': 0,
                'miss_rate': 0.0
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting missed opportunities: {e}")
        return jsonify({'signals': [], 'stats': {}, 'error': str(e)})


@app.route('/api/debug')
def debug():
    """Debug endpoint."""
    log_path = log_parser.find_latest_log()
    db_exists = Path(CONFIG['paper_trading_db']).exists()
    
    log_lines = 0
    if log_path:
        try:
            with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                log_lines = sum(1 for _ in f)
        except Exception:
            pass
    
    return jsonify({
        'log_file': log_path,
        'log_lines': log_lines,
        'db_exists': db_exists,
        'db_path': CONFIG['paper_trading_db'],
        'config': {k: v for k, v in CONFIG.items() if k != 'initial_balance'},
        'state': {
            'cycles': state.progress.cycles,
            'spy_price': state.market.spy_price,
            'signal': state.signal.last_signal,
            'phase': state.session.phase,
            'run_id': state.session.current_run_id,
            'archived_count': len(state.archived_sessions)
        }
    })


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print(f"[*] Training Dashboard Server v{DASHBOARD_VERSION}")
    print("=" * 60)
    print()
    print(f"Dashboard URL: http://localhost:{CONFIG['port']}")
    print(f"API Endpoint:  http://localhost:{CONFIG['port']}/api/data")
    print()
    print("Configuration:")
    print(f"  - Paper DB: {CONFIG['paper_trading_db']}")
    print(f"  - Historical DB: {CONFIG['historical_db']}")
    print(f"  - Log Pattern: {CONFIG['log_pattern']}")
    print(f"  - Initial Balance: ${CONFIG['initial_balance']:,.2f}")
    print()
    print("Run this WHILE train_then_go_live.py is running")
    print()
    print("Press Ctrl+C to stop")
    print()
    
    # Initial data load
    refresh_data()
    
    app.run(
        host=CONFIG['host'],
        port=CONFIG['port'],
        debug=False,
        threaded=True
    )
