#!/usr/bin/env python3
"""
Live Trading Dashboard API Server
==================================

Serves real-time trading data for the web dashboard.
Run this WHILE go_live_only.py is running.

Usage:
    python dashboard_server.py

Configuration is loaded from config.json under the "dashboard" section.

Version: 2.0 - Unified architecture with training_dashboard_server.py
"""

# Version for debugging
DASHBOARD_VERSION = "2.0-unified"

import json
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

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
        'port': 5000,
        'host': '0.0.0.0',
        'paper_trading_db': 'data/paper_trading.db',
        'historical_db': 'data/db/historical.db',
        'log_pattern': 'logs/trading_bot.log*',
        'fallback_log_pattern': 'logs/real_bot_*.log',
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
            'initial_log_lines': 5000
        },
        'tradier': {
            'sync_interval_seconds': 60,
            'enable_live_sync': True
        }
    }
    
    try:
        if config_path.exists():
            with open(config_path, 'r') as f:
                cfg = json.load(f)
            dashboard_cfg = cfg.get('dashboard', {})
            
            # Merge with defaults
            for key, value in dashboard_cfg.items():
                if isinstance(value, dict) and key in defaults:
                    defaults[key].update(value)
                else:
                    defaults[key] = value
            
            # Get initial balance from trading config
            defaults['initial_balance'] = cfg.get('trading', {}).get('initial_balance', 1000.0)
            defaults['trading_symbol'] = cfg.get('trading', {}).get('symbol', 'SPY')
    except Exception as e:
        print(f"Warning: Could not load config: {e}")
        defaults['initial_balance'] = 1000.0
        defaults['trading_symbol'] = 'SPY'
    
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

# Tradier-specific state (extends base state)
tradier_state = {
    'balance': 0.0,
    'positions': 0,
    'orders': [],
    'account_id': '',
    'is_sandbox': True,
    'last_sync': 0
}

# Initialize components
log_parser = LogParser(
    log_pattern=CONFIG['log_pattern'],
    fallback_pattern=CONFIG['fallback_log_pattern'],
    initial_lines=CONFIG['state']['initial_log_lines']
)

db_loader = DatabaseLoader(
    paper_db_path=CONFIG['paper_trading_db'],
    historical_db_path=CONFIG['historical_db']
)


def aggregate_predictions_by_target_time(predictions_history: list, current_time_str: str) -> dict:
    """
    Aggregate overlapping predictions using SMART ENSEMBLE:
    
    1. RECENCY WEIGHTING: Predictions made closer to target time get more weight
    2. SHORTER TIMEFRAME BONUS: Shorter timeframe predictions are inherently more accurate
    3. AGREEMENT BOOST: When predictions agree, confidence increases
    4. DECAY: Old predictions matter less as new information arrives
    """
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
            
            # Calculate age of prediction
            pred_age = (current_time - pred_time).total_seconds() / 60
            
            # Skip predictions that are too old (>30 minutes)
            if pred_age > 30:
                continue
            
            if target_key not in predictions_by_target:
                predictions_by_target[target_key] = []
            
            # Calculate absolute target price
            target_price = spy_price * (1 + pred_return / 100) if spy_price > 0 else 0
            
            predictions_by_target[target_key].append({
                'target_time': target_key,
                'target_time_dt': target_time,
                'prediction_made_at': pred_timestamp,
                'pred_time_dt': pred_time,
                'source_timeframe': tf_name,
                'minutes_ahead': minutes,
                'time_to_target': (target_time - current_time).total_seconds() / 60,
                'pred_age': pred_age,
                'direction': direction,
                'predicted_return': pred_return,
                'confidence': confidence,
                'entry_price': spy_price,
                'target_price': target_price
            })
    
    # Calculate SMART ENSEMBLE predictions for each target time
    best_predictions = {}
    for target_key, preds in predictions_by_target.items():
        if not preds:
            continue
        
        total = len(preds)
        
        # Calculate weights
        weights = []
        for p in preds:
            base_conf = p.get('confidence', 25) / 100
            pred_age = max(0, p.get('pred_age', 0))
            recency_weight = math.exp(-0.069 * pred_age)
            tf_minutes_val = p.get('minutes_ahead', 30)
            timeframe_weight = max(0.5, 1.5 - (tf_minutes_val / 60))
            time_to_target = max(1, p.get('time_to_target', 30))
            proximity_weight = min(2.0, 30 / time_to_target)
            weight = base_conf * recency_weight * timeframe_weight * proximity_weight
            weights.append(weight)
        
        total_weight = sum(weights) or 1
        
        # Weighted target price
        weighted_target_price = sum(p.get('target_price', 0) * w for p, w in zip(preds, weights)) / total_weight
        weighted_return = sum(p.get('predicted_return', 0) * w for p, w in zip(preds, weights)) / total_weight
        
        # Direction voting
        bullish_weight = sum(w for p, w in zip(preds, weights) if p.get('direction') == 'UP')
        bearish_weight = sum(w for p, w in zip(preds, weights) if p.get('direction') == 'DOWN')
        neutral_weight = total_weight - bullish_weight - bearish_weight
        
        bullish = sum(1 for p in preds if p.get('direction') == 'UP')
        bearish = sum(1 for p in preds if p.get('direction') == 'DOWN')
        neutral = total - bullish - bearish
        
        if bullish_weight > bearish_weight and bullish_weight > neutral_weight:
            direction = 'UP'
        elif bearish_weight > bullish_weight and bearish_weight > neutral_weight:
            direction = 'DOWN'
        else:
            direction = 'NEUTRAL'
        
        # Confidence calculation
        base_confidence = sum(p.get('confidence', 25) * w for p, w in zip(preds, weights)) / total_weight
        max_vote_weight = max(bullish_weight, bearish_weight, neutral_weight)
        agreement_ratio = max_vote_weight / total_weight if total_weight > 0 else 0.5
        agreement_multiplier = 0.7 + (0.8 * agreement_ratio)
        volume_bonus = min(1.3, 1.0 + (total - 1) * 0.03)
        adjusted_confidence = min(100.0, base_confidence * agreement_multiplier * volume_bonus)
        
        latest = max(preds, key=lambda p: p.get('prediction_made_at', ''))
        
        best_predictions[target_key] = {
            'target_time': target_key,
            'predicted_return': weighted_return,
            'target_price': weighted_target_price,
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
    """Check past TCN predictions against actual price movement."""
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
    
    timeframes = [('5min', 5), ('10min', 10), ('15min', 15), ('30min', 30)]
    
    for pred in state.lstm_predictions_history:
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
        
        for tf_name, tf_minutes in timeframes:
            validated_key = f'{tf_name}_validated'
            
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
            
            if isinstance(pred, dict):
                pred[validated_key] = True
                pred[f'{tf_name}_correct'] = is_correct
                pred[f'{tf_name}_actual_price'] = state.market.spy_price
                pred[f'{tf_name}_actual_move'] = actual_move
            
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
        
        if isinstance(pred, dict):
            if pred.get('5min_validated') and pred.get('15min_validated') and pred.get('30min_validated'):
                pred['validated'] = True
    
    max_validated = CONFIG['chart']['max_validated_predictions']
    if len(state.lstm_validated_predictions) > max_validated:
        state.lstm_validated_predictions = state.lstm_validated_predictions[-max_validated:]


def load_live_spy_prices(lookback_days: int = 3, max_points: int = 10000) -> List[Dict[str, Any]]:
    """
    Load live SPY prices from market data cache or Tradier API.
    Used as fallback when historical DB doesn't have recent data.
    """
    prices = []
    
    # Try market_data_cache.db first
    cache_db_path = Path('data/market_data_cache.db')
    if cache_db_path.exists():
        try:
            import sqlite3
            conn = sqlite3.connect(str(cache_db_path))
            cursor = conn.cursor()
            
            # Get schema info to find the right table
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [t[0] for t in cursor.fetchall()]
            
            # Try different possible table names
            for table in ['spy_prices', 'price_cache', 'ohlcv', 'prices', 'candles']:
                if table in tables:
                    try:
                        cursor.execute(f'''
                            SELECT timestamp, close, high, low, open
                            FROM {table}
                            WHERE symbol = 'SPY' OR symbol IS NULL
                            ORDER BY timestamp DESC
                            LIMIT ?
                        ''', (max_points,))
                        rows = cursor.fetchall()
                        for row in rows:
                            prices.append({
                                'timestamp': row[0],
                                'close': row[1],
                                'high': row[2] or row[1],
                                'low': row[3] or row[1],
                                'open': row[4] or row[1]
                            })
                        break
                    except:
                        continue
            
            conn.close()
        except Exception as e:
            print(f"Error loading from cache DB: {e}")
    
    # Reverse to get chronological order
    prices.reverse()
    
    # If still no data, try to get from Tradier API (intraday history)
    if not prices:
        prices = fetch_tradier_intraday_prices(lookback_days, max_points)
    
    return prices


def fetch_tradier_intraday_prices(lookback_days: int = 3, max_points: int = 10000) -> List[Dict[str, Any]]:
    """Fetch intraday price history from Tradier API."""
    prices = []
    
    try:
        import requests
        from tradier_credentials import TradierCredentials
        from trading_mode import get_trading_mode
        
        mode = get_trading_mode()
        creds = TradierCredentials()
        
        if mode['is_sandbox']:
            if not creds.has_sandbox_credentials():
                return prices
            cred_data = creds.get_sandbox_credentials()
            base_url = "https://sandbox.tradier.com/v1"
        else:
            if not creds.has_live_credentials():
                return prices
            cred_data = creds.get_live_credentials()
            base_url = "https://api.tradier.com/v1"
        
        headers = {
            'Authorization': f'Bearer {cred_data["access_token"]}',
            'Accept': 'application/json'
        }
        
        # Get time series data
        # Tradier timesales endpoint gives intraday minute data
        start_date = (get_market_time() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
        
        resp = requests.get(
            f'{base_url}/markets/timesales',
            params={
                'symbol': CONFIG['trading_symbol'],
                'interval': '1min',
                'start': f'{start_date} 09:30',
                'session_filter': 'all'
            },
            headers=headers,
            timeout=10
        )
        
        if resp.status_code == 200:
            data = resp.json()
            series = data.get('series', {})
            if series and 'data' in series:
                candles = series['data']
                if not isinstance(candles, list):
                    candles = [candles]
                
                for candle in candles[-max_points:]:
                    prices.append({
                        'timestamp': candle.get('time', candle.get('timestamp', '')),
                        'close': float(candle.get('close', candle.get('price', 0))),
                        'high': float(candle.get('high', candle.get('close', 0))),
                        'low': float(candle.get('low', candle.get('close', 0))),
                        'open': float(candle.get('open', candle.get('close', 0)))
                    })
        
    except Exception as e:
        print(f"Error fetching Tradier intraday prices: {e}")
    
    return prices


def load_tradier_account() -> None:
    """Load Tradier account info."""
    global tradier_state
    
    # Rate limit syncs
    if time.time() - tradier_state['last_sync'] < CONFIG['tradier']['sync_interval_seconds']:
        return
    
    if not CONFIG['tradier']['enable_live_sync']:
        return
    
    try:
        import requests
        from tradier_credentials import TradierCredentials
        from trading_mode import get_trading_mode
        
        mode = get_trading_mode()
        creds = TradierCredentials()
        
        if mode['is_sandbox']:
            if not creds.has_sandbox_credentials():
                return
            cred_data = creds.get_sandbox_credentials()
            base_url = "https://sandbox.tradier.com/v1"
        else:
            if not creds.has_live_credentials():
                return
            cred_data = creds.get_live_credentials()
            base_url = "https://api.tradier.com/v1"
        
        headers = {
            'Authorization': f'Bearer {cred_data["access_token"]}',
            'Accept': 'application/json'
        }
        account_id = cred_data["account_number"]
        tradier_state['account_id'] = account_id
        tradier_state['is_sandbox'] = mode['is_sandbox']
        
        # Get balance
        resp = requests.get(f'{base_url}/accounts/{account_id}/balances', headers=headers, timeout=5)
        if resp.status_code == 200:
            bal = resp.json().get('balances', {})
            tradier_state['balance'] = float(bal.get('total_equity', 0)) or float(bal.get('cash_available', 0))
        
        # Get positions
        resp = requests.get(f'{base_url}/accounts/{account_id}/positions', headers=headers, timeout=5)
        if resp.status_code == 200:
            pos = resp.json().get('positions', {})
            if pos and 'position' in pos:
                pos_list = pos['position']
                tradier_state['positions'] = len(pos_list) if isinstance(pos_list, list) else (1 if pos_list else 0)
            else:
                tradier_state['positions'] = 0
        
        # Get recent orders
        resp = requests.get(f'{base_url}/accounts/{account_id}/orders', headers=headers, timeout=5)
        if resp.status_code == 200:
            orders_data = resp.json().get('orders', {})
            if orders_data and 'order' in orders_data:
                orders = orders_data['order']
                if not isinstance(orders, list):
                    orders = [orders]
                
                tradier_orders = []
                for order in orders[:10]:  # Last 10 orders
                    if order.get('status') == 'filled':
                        symbol = order.get('symbol', '')
                        side = order.get('side', '')
                        avg_price = float(order.get('avg_fill_price', 0))
                        create_date = order.get('create_date', '')
                        
                        # Determine action
                        if 'C' in symbol[-15:]:
                            action = 'CALL'
                        elif 'P' in symbol[-15:]:
                            action = 'PUT'
                        else:
                            action = side.upper()
                        
                        try:
                            dt = datetime.fromisoformat(create_date.replace('Z', '+00:00'))
                            trade_datetime = dt.strftime('%m/%d %H:%M')
                        except:
                            trade_datetime = create_date[:16] if create_date else '--'
                        
                        tradier_orders.append({
                            'datetime': trade_datetime,
                            'action': f"{side.upper()} {action}",
                            'price': avg_price,
                            'status': 'FILLED',
                            'is_real': True,
                            'source': 'TRADIER'
                        })
                
                tradier_state['orders'] = tradier_orders
        
        tradier_state['last_sync'] = time.time()
        
    except Exception as e:
        print(f"Tradier sync error: {e}")


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
        
        # Set phase to Live Paper (this is live dashboard)
        state.session.phase = "Live Paper"
        
        # Parse log file
        log_path = log_parser.find_latest_log()
        if log_path:
            log_parser.parse_log_file(log_path, state)
        
        # Load from database
        db_loader.load_all(state)
        
        # Load Tradier account info
        load_tradier_account()
        
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
    """Serve the dashboard HTML."""
    return send_file('dashboard.html')


@app.route('/api/data')
def get_data():
    """Get current dashboard data."""
    refresh_data()
    
    with state_lock:
        data = state.to_api_dict()
        
        # Calculate derived metrics
        elapsed = time.time() - state.session.start_time if state.session.start_time > 0 else 0
        data['elapsed_seconds'] = elapsed
        data['signal_rate'] = (state.progress.signals / max(state.progress.cycles, 1)) * 100
        data['win_rate'] = state.account.win_rate
        
        # Add Tradier-specific data
        data['tradier_balance'] = tradier_state['balance']
        data['tradier_positions'] = tradier_state['positions']
        data['tradier_account_id'] = tradier_state['account_id']
        data['tradier_is_sandbox'] = tradier_state['is_sandbox']
        data['tradier_orders'] = tradier_state['orders']
        
        # Trading symbol
        data['trading_symbol'] = CONFIG['trading_symbol']
        
        # TCN prediction accuracy metrics
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


@app.route('/api/reset-session', methods=['POST'])
def reset_session():
    """Manually reset the session."""
    try:
        with state_lock:
            state.reset_for_new_session(None, CONFIG['initial_balance'])
        return jsonify({'success': True, 'message': 'Session reset'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/sync-paper', methods=['POST'])
def sync_paper_account():
    """Sync paper account balance to match Tradier balance."""
    try:
        # Get Tradier balance
        live_balance = tradier_state['balance']
        if live_balance <= 0:
            live_balance = CONFIG['initial_balance']
        
        # Update config file
        config_path = Path('config.json')
        if config_path.exists():
            with open(config_path, 'r') as f:
                cfg = json.load(f)
            cfg['trading']['initial_balance'] = live_balance
            with open(config_path, 'w') as f:
                json.dump(cfg, f, indent=4)
        
        # Update database
        db_path = Path(CONFIG['paper_trading_db'])
        if db_path.exists():
            import sqlite3
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            cursor.execute('UPDATE account_state SET balance = ?, total_profit_loss = 0.0', (live_balance,))
            conn.commit()
            conn.close()
        
        # Update state
        with state_lock:
            state.account.initial_balance = live_balance
            state.account.current_balance = live_balance
            state.account.pnl = 0.0
            state.account.pnl_pct = 0.0
        
        return jsonify({
            'success': True, 
            'new_balance': live_balance,
            'message': f'Paper account synced to ${live_balance:.2f}'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/chart')
def get_chart_data():
    """Get SPY price history + trades + predictions for chart."""
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
        
        # In live mode, use current time if no simulated time
        reference_time = simulated_datetime or trade_time_range or format_timestamp(get_market_time())
        
        # Load data
        chart_cfg = CONFIG['chart']
        
        spy_prices = db_loader.load_spy_prices(
            reference_time=reference_time,
            lookback_days=chart_cfg['lookback_days'],
            max_points=chart_cfg['max_price_points']
        )
        
        # If no historical data, try to get live prices from market data cache
        if not spy_prices:
            spy_prices = load_live_spy_prices(chart_cfg['lookback_days'], chart_cfg['max_price_points'])

        # Load VIX with Bollinger Bands
        vix_data = db_loader.load_vix_prices(
            reference_time=reference_time,
            lookback_days=chart_cfg['lookback_days'],
            bb_period=20
        )

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
        
        # Build TCN predictions history for API
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
        
        # Aggregate predictions by target time
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
            'aggregated_predictions': aggregated_list,
            'tradier_orders': tradier_state['orders'],
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
            'tradier_orders': [],
            'vix_prices': [],
            'vix_bb_upper': [],
            'vix_bb_lower': [],
            'vix_bb_middle': []
        })


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
        'version': DASHBOARD_VERSION,
        'log_file': log_path,
        'log_lines': log_lines,
        'db_exists': db_exists,
        'db_path': CONFIG['paper_trading_db'],
        'config': {k: v for k, v in CONFIG.items() if k not in ['initial_balance', 'tradier']},
        'state': {
            'cycles': state.progress.cycles,
            'spy_price': state.market.spy_price,
            'signal': state.signal.last_signal,
            'phase': state.session.phase,
            'run_id': state.session.current_run_id,
            'archived_count': len(state.archived_sessions)
        },
        'tradier': {
            'balance': tradier_state['balance'],
            'positions': tradier_state['positions'],
            'account_id': tradier_state['account_id'],
            'is_sandbox': tradier_state['is_sandbox'],
            'orders_count': len(tradier_state['orders'])
        }
    })


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print(f"[*] Live Trading Dashboard Server v{DASHBOARD_VERSION}")
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
    print(f"  - Trading Symbol: {CONFIG['trading_symbol']}")
    print()
    print("Run this WHILE go_live_only.py is running")
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
