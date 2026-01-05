#!/usr/bin/env python3
"""
Unified Dashboard Server
========================

Single server combining all dashboard functionality:
- Hub/Overview (scoreboard, trade browser, agent API)
- Live Trading (Tradier integration, real-time monitoring)
- Training Monitor (log parsing, predictions, experiments)
- Model History (past runs, SUMMARY.txt parsing)

Features:
- Tab-based navigation at single URL
- Remote data ingestion for multi-machine setups
- All APIs under one roof

Usage:
    python unified_dashboard_server.py

Environment Variables:
    DASHBOARD_PORT - Server port (default: 5003)
    DASHBOARD_HOST - Server host (default: 0.0.0.0)
"""

import os
import json
import time
import re
import glob
import sqlite3
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import asdict

from flask import Flask, jsonify, render_template, send_from_directory, request
from flask_cors import CORS

# Import existing blueprints
from backend.dashboard.agent_api import agent_bp
from backend.dashboard.scoreboard_api import scoreboard_bp
from backend.dashboard.trades_api import trades_bp

# Import shared state classes
from backend.dashboard.state import TrainingState, TradeRecord, ValidatedPrediction
from backend.dashboard.log_parser import LogParser
from backend.dashboard.db_loader import DatabaseLoader

# Time utilities
try:
    from backend.time_utils import get_market_time, format_timestamp
except ImportError:
    import pytz
    def get_market_time():
        market_tz = pytz.timezone('US/Eastern')
        return pytz.UTC.localize(datetime.utcnow()).astimezone(market_tz).replace(tzinfo=None)
    def format_timestamp(dt):
        return dt.strftime('%Y-%m-%dT%H:%M:%S') if dt else ""

# =============================================================================
# VERSION
# =============================================================================
DASHBOARD_VERSION = "1.0.0-unified"

# =============================================================================
# CONFIGURATION
# =============================================================================

def load_config() -> Dict[str, Any]:
    """Load unified dashboard configuration."""
    config_path = Path('config.json')

    defaults = {
        'port': 5003,
        'host': '0.0.0.0',
        'paper_trading_db': 'data/paper_trading.db',
        'historical_db': 'data/db/historical.db',
        'models_dir': 'models',
        'log_patterns': {
            'training': 'logs/real_bot_simulation*.log',
            'live': 'logs/trading_bot.log*'
        },
        'initial_balance': 5000.0,
        'trading_symbol': 'SPY',
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
            'initial_log_lines': 500000
        },
        'tradier': {
            'sync_interval_seconds': 60,
            'enable_live_sync': True
        },
        'remote': {
            'enabled': True,
            'api_key': None  # Set via config for security
        }
    }

    try:
        if config_path.exists():
            with open(config_path, 'r') as f:
                cfg = json.load(f)

            # Merge dashboard configs
            for key in ['dashboard', 'training_dashboard', 'dashboard_hub', 'unified_dashboard']:
                if key in cfg:
                    for k, v in cfg[key].items():
                        if isinstance(v, dict) and k in defaults:
                            defaults[k].update(v)
                        else:
                            defaults[k] = v

            # Get trading config
            defaults['initial_balance'] = cfg.get('trading', {}).get('initial_balance', 5000.0)
            defaults['trading_symbol'] = cfg.get('trading', {}).get('symbol', 'SPY')

    except Exception as e:
        print(f"Warning: Could not load config: {e}")

    return defaults

CONFIG = load_config()

# =============================================================================
# FLASK APP
# =============================================================================

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# Register existing blueprints
app.register_blueprint(agent_bp)
app.register_blueprint(scoreboard_bp)
app.register_blueprint(trades_bp)

# =============================================================================
# STATE MANAGEMENT
# =============================================================================

# Training state
training_state = TrainingState()
training_state_lock = threading.Lock()

# Live trading state
live_state = TrainingState()
live_state_lock = threading.Lock()

# Tradier-specific state
tradier_state = {
    'balance': 0.0,
    'positions': 0,
    'orders': [],
    'account_id': '',
    'is_sandbox': True,
    'last_sync': 0
}

# Remote machines state (for multi-machine setups)
remote_machines = {}
remote_machines_lock = threading.Lock()

# Initialize log parsers
training_log_parser = LogParser(
    log_pattern=CONFIG['log_patterns']['training'],
    fallback_pattern='../logs/real_bot_simulation*.log',
    initial_lines=CONFIG['state']['initial_log_lines']
)

live_log_parser = LogParser(
    log_pattern=CONFIG['log_patterns']['live'],
    fallback_pattern='logs/real_bot_*.log',
    initial_lines=CONFIG['state']['initial_log_lines']
)

# Initialize DB loaders
training_db_loader = DatabaseLoader(
    paper_db_path=CONFIG['paper_trading_db'],
    historical_db_path=CONFIG['historical_db'],
    trade_filter='paper'
)

live_db_loader = DatabaseLoader(
    paper_db_path=CONFIG['paper_trading_db'],
    historical_db_path=CONFIG['historical_db'],
    trade_filter='live'
)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def parse_summary_txt(summary_path: Path) -> Optional[Dict[str, Any]]:
    """Parse a SUMMARY.txt file into a structured dict."""
    if not summary_path.exists():
        return None

    try:
        with open(summary_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        result = {
            'run_id': summary_path.parent.name,
            'path': str(summary_path),
            'timestamp': '',
            'initial_balance': 0.0,
            'final_balance': 0.0,
            'pnl': 0.0,
            'pnl_pct': 0.0,
            'total_trades': 0,
            'total_signals': 0,
            'total_cycles': 0,
            'win_rate': 0.0,
            'wins': 0,
            'losses': 0,
            'raw_content': content
        }

        # Parse fields
        patterns = {
            'timestamp': r'Timestamp:\s*(\S+)',
            'initial_balance': r'Initial Balance:\s*\$?([\d,]+\.?\d*)',
            'final_balance': r'Final Balance:\s*\$?([\d,]+\.?\d*)',
            'total_trades': r'Total Trades:\s*(\d+)',
            'total_signals': r'Total Signals:\s*(\d+)',
            'total_cycles': r'Total Cycles:\s*(\d+)',
            'win_rate': r'Win Rate:\s*([\d.]+)%',
            'wins': r'Wins:\s*(\d+)',
            'losses': r'Losses:\s*(\d+)'
        }

        for field, pattern in patterns.items():
            match = re.search(pattern, content)
            if match:
                val = match.group(1).replace(',', '')
                if field in ['initial_balance', 'final_balance', 'win_rate']:
                    result[field] = float(val)
                elif field == 'timestamp':
                    result[field] = val
                else:
                    result[field] = int(val)

        # Parse P&L
        match = re.search(r'P&L:\s*\$?([+-]?[\d,]+\.?\d*)\s*\(([+-]?[\d.]+)%\)', content)
        if match:
            result['pnl'] = float(match.group(1).replace(',', ''))
            result['pnl_pct'] = float(match.group(2))

        return result
    except Exception as e:
        print(f"Error parsing {summary_path}: {e}")
        return None


def list_all_models() -> List[Dict[str, Any]]:
    """List all models with SUMMARY.txt."""
    models_dir = Path(CONFIG['models_dir'])
    models = []

    if not models_dir.exists():
        return models

    for item in models_dir.iterdir():
        if item.is_dir():
            summary_path = item / 'SUMMARY.txt'
            if summary_path.exists():
                summary = parse_summary_txt(summary_path)
                if summary:
                    summary['mtime'] = summary_path.stat().st_mtime
                    summary['mtime_str'] = datetime.fromtimestamp(
                        summary_path.stat().st_mtime
                    ).strftime('%Y-%m-%d %H:%M')
                    models.append(summary)

    models.sort(key=lambda x: x.get('mtime', 0), reverse=True)
    return models


def refresh_training_data(selected_log_file: str = None) -> None:
    """Refresh training dashboard data."""
    global training_state

    with training_state_lock:
        if training_state.session.start_time == 0:
            training_state.session.start_time = time.time()

        training_state.account.initial_balance = CONFIG['initial_balance']
        if training_state.account.current_balance == 1000.0:
            training_state.account.current_balance = training_state.account.initial_balance

        # Parse log file
        if selected_log_file and Path(selected_log_file).exists():
            log_path = selected_log_file
        else:
            log_path = training_log_parser.find_latest_log()

        if log_path:
            training_log_parser.parse_log_file(log_path, training_state)

        training_db_loader.load_all(training_state)
        training_state.session.last_update = format_timestamp(get_market_time())


def refresh_live_data() -> None:
    """Refresh live dashboard data."""
    global live_state

    with live_state_lock:
        if live_state.session.start_time == 0:
            live_state.session.start_time = time.time()

        live_state.account.initial_balance = CONFIG['initial_balance']
        if live_state.account.current_balance == 1000.0:
            live_state.account.current_balance = live_state.account.initial_balance

        live_state.session.phase = "Live Paper"

        log_path = live_log_parser.find_latest_log()
        if log_path:
            live_log_parser.parse_log_file(log_path, live_state)

        live_db_loader.load_all(live_state)
        load_tradier_account()

        live_state.session.last_update = format_timestamp(get_market_time())


def load_tradier_account() -> None:
    """Load Tradier account info."""
    global tradier_state

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

        tradier_state['last_sync'] = time.time()

    except Exception as e:
        print(f"Tradier sync error: {e}")


# =============================================================================
# ROUTES - PAGES
# =============================================================================

@app.route('/')
@app.route('/hub')
def hub_page():
    """Hub/overview page."""
    return render_template('unified.html', active_tab='hub')


@app.route('/live')
def live_page():
    """Live trading page."""
    return render_template('unified.html', active_tab='live')


@app.route('/training')
def training_page():
    """Training monitor page."""
    return render_template('unified.html', active_tab='training')


@app.route('/history')
def history_page():
    """Model history page."""
    return render_template('unified.html', active_tab='history')


# =============================================================================
# ROUTES - TRAINING API
# =============================================================================

@app.route('/api/training/data')
def api_training_data():
    """Get training dashboard data."""
    selected_log = request.args.get('log_file')
    refresh_training_data(selected_log)

    with training_state_lock:
        data = training_state.to_api_dict()
        elapsed = time.time() - training_state.session.start_time if training_state.session.start_time > 0 else 0
        data['elapsed_seconds'] = elapsed
        data['signal_rate'] = (training_state.progress.signals / max(training_state.progress.cycles, 1)) * 100
        data['win_rate'] = training_state.account.win_rate

    return jsonify(data)


@app.route('/api/training/chart')
def api_training_chart():
    """Get training chart data."""
    selected_log = request.args.get('log_file')
    refresh_training_data(selected_log)

    try:
        simulated_datetime = training_state.session.simulated_datetime
        reference_time = simulated_datetime or format_timestamp(get_market_time())
        chart_cfg = CONFIG['chart']

        spy_prices = training_db_loader.load_spy_prices(
            reference_time=reference_time,
            lookback_days=chart_cfg['lookback_days'],
            max_points=chart_cfg['max_price_points']
        )

        vix_data = training_db_loader.load_vix_prices(
            reference_time=reference_time,
            lookback_days=chart_cfg['lookback_days'],
            bb_period=20
        )

        trades, annotations = training_db_loader.load_trades_for_chart(
            reference_time=reference_time,
            lookback_days=chart_cfg['lookback_days'],
            max_trades=chart_cfg['max_trades']
        )

        return jsonify({
            'spy_prices': spy_prices,
            'trades': trades,
            'annotations': annotations,
            'simulated_time': simulated_datetime,
            'cycles': training_state.progress.cycles,
            'current_price': training_state.market.spy_price,
            'vix_prices': vix_data.get('vix_prices', []),
            'vix_bb_upper': vix_data.get('bb_upper', []),
            'vix_bb_lower': vix_data.get('bb_lower', []),
            'vix_bb_middle': vix_data.get('bb_middle', [])
        })
    except Exception as e:
        return jsonify({'error': str(e), 'spy_prices': [], 'trades': []})


@app.route('/api/training/runs')
def api_training_runs():
    """List training runs."""
    runs = []
    log_pattern = CONFIG['log_patterns']['training']
    cutoff_time = time.time() - (24 * 60 * 60)  # Last 24 hours

    for log_path in glob.glob(log_pattern):
        try:
            path = Path(log_path)
            if not path.exists() or path.stat().st_size == 0:
                continue

            mtime = path.stat().st_mtime
            if mtime < cutoff_time:
                continue

            run_name = path.stem.replace('real_bot_simulation_', '')
            is_active = (time.time() - mtime) < 30

            runs.append({
                'run_name': run_name,
                'log_file': str(path),
                'size_mb': round(path.stat().st_size / (1024 * 1024), 1),
                'last_modified': datetime.fromtimestamp(mtime).isoformat(),
                'is_active': is_active,
                'source': 'local'
            })
        except Exception:
            continue

    # Add remote machines' runs
    with remote_machines_lock:
        for machine_id, machine_data in remote_machines.items():
            for run in machine_data.get('runs', []):
                run['source'] = f'remote:{machine_id}'
                runs.append(run)

    runs.sort(key=lambda x: x['last_modified'], reverse=True)

    return jsonify({
        'success': True,
        'runs': runs,
        'active_count': sum(1 for r in runs if r.get('is_active'))
    })


# =============================================================================
# ROUTES - LIVE API
# =============================================================================

@app.route('/api/live/data')
def api_live_data():
    """Get live trading data."""
    refresh_live_data()

    with live_state_lock:
        data = live_state.to_api_dict()
        elapsed = time.time() - live_state.session.start_time if live_state.session.start_time > 0 else 0
        data['elapsed_seconds'] = elapsed
        data['signal_rate'] = (live_state.progress.signals / max(live_state.progress.cycles, 1)) * 100
        data['win_rate'] = live_state.account.win_rate

        # Add Tradier data
        data['tradier_balance'] = tradier_state['balance']
        data['tradier_positions'] = tradier_state['positions']
        data['tradier_account_id'] = tradier_state['account_id']
        data['tradier_is_sandbox'] = tradier_state['is_sandbox']
        data['tradier_orders'] = tradier_state['orders']
        data['trading_symbol'] = CONFIG['trading_symbol']

    return jsonify(data)


@app.route('/api/live/chart')
def api_live_chart():
    """Get live chart data."""
    refresh_live_data()

    try:
        simulated_datetime = live_state.session.simulated_datetime
        reference_time = simulated_datetime or format_timestamp(get_market_time())
        chart_cfg = CONFIG['chart']

        spy_prices = live_db_loader.load_spy_prices(
            reference_time=reference_time,
            lookback_days=chart_cfg['lookback_days'],
            max_points=chart_cfg['max_price_points']
        )

        trades, annotations = live_db_loader.load_trades_for_chart(
            reference_time=reference_time,
            lookback_days=chart_cfg['lookback_days'],
            max_trades=chart_cfg['max_trades']
        )

        return jsonify({
            'spy_prices': spy_prices,
            'trades': trades,
            'annotations': annotations,
            'simulated_time': simulated_datetime,
            'current_price': live_state.market.spy_price,
            'tradier_orders': tradier_state['orders']
        })
    except Exception as e:
        return jsonify({'error': str(e), 'spy_prices': [], 'trades': []})


# =============================================================================
# ROUTES - HISTORY API
# =============================================================================

@app.route('/api/history/models')
def api_history_models():
    """List all models."""
    models = list_all_models()
    return jsonify({'models': models, 'count': len(models)})


@app.route('/api/history/model/<run_id>')
def api_history_model(run_id: str):
    """Get model details."""
    models_dir = Path(CONFIG['models_dir'])
    summary_path = models_dir / run_id / 'SUMMARY.txt'

    summary = parse_summary_txt(summary_path)
    if not summary:
        return jsonify({'error': f'Model {run_id} not found'}), 404

    return jsonify(summary)


@app.route('/api/history/model/<run_id>/trades')
def api_history_model_trades(run_id: str):
    """Get trades for a model."""
    db_path = Path(CONFIG['paper_trading_db'])
    if not db_path.exists():
        return jsonify({'trades': [], 'error': 'Database not found'})

    trades = []
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Try to get trades by run_id first
        cursor.execute("""
            SELECT * FROM trades
            WHERE run_id = ? AND status IS NOT NULL AND status != 'OPEN'
            ORDER BY timestamp ASC
        """, (run_id,))

        rows = cursor.fetchall()

        # If no trades found by run_id, this is legacy data
        if not rows:
            cursor.execute("""
                SELECT * FROM trades
                WHERE status IS NOT NULL AND status != 'OPEN'
                ORDER BY timestamp ASC
                LIMIT 500
            """)
            rows = cursor.fetchall()

        for row in rows:
            t = dict(row)
            entry_price = t.get('premium_paid', 0) or t.get('entry_price', 0) or 0
            profit_loss = t.get('profit_loss', 0) or 0
            pnl_pct = (profit_loss / entry_price * 100) if entry_price > 0 else 0

            trades.append({
                'id': t.get('id', ''),
                'entry_time': t.get('timestamp', ''),
                'exit_time': t.get('exit_timestamp', ''),
                'option_type': t.get('option_type', ''),
                'entry_price': entry_price,
                'exit_price': t.get('exit_price', 0),
                'profit_loss': profit_loss,
                'pnl_pct': pnl_pct,
                'exit_reason': t.get('status', ''),
                'strike': t.get('strike_price', 0)
            })

        conn.close()
    except Exception as e:
        return jsonify({'trades': [], 'error': str(e)})

    return jsonify({'trades': trades, 'trade_count': len(trades)})


@app.route('/api/history/spy_prices')
def api_history_spy_prices():
    """Get SPY prices for date range."""
    start = request.args.get('start', '')
    end = request.args.get('end', '')

    db_path = Path(CONFIG['historical_db'])
    if not db_path.exists():
        db_path = Path('data/historical.db')

    if not db_path.exists():
        return jsonify({'prices': [], 'error': 'No historical database'})

    prices = []
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        if start and end:
            cursor.execute("""
                SELECT timestamp, close_price, high_price, low_price, open_price
                FROM historical_data
                WHERE symbol = 'SPY' AND timestamp >= ? AND timestamp <= ?
                ORDER BY timestamp ASC
            """, (start, end))
        else:
            cursor.execute("""
                SELECT timestamp, close_price, high_price, low_price, open_price
                FROM historical_data
                WHERE symbol = 'SPY'
                ORDER BY timestamp DESC
                LIMIT 500
            """)

        for row in cursor.fetchall():
            prices.append({
                'timestamp': row[0],
                'close': row[1],
                'high': row[2],
                'low': row[3],
                'open': row[4]
            })

        if not start:
            prices.reverse()

        conn.close()
    except Exception as e:
        return jsonify({'prices': [], 'error': str(e)})

    return jsonify({'prices': prices, 'count': len(prices)})


# =============================================================================
# ROUTES - REMOTE DATA INGESTION
# =============================================================================

@app.route('/api/remote/register', methods=['POST'])
def api_remote_register():
    """Register a remote training machine."""
    if not CONFIG['remote']['enabled']:
        return jsonify({'success': False, 'error': 'Remote ingestion disabled'}), 403

    data = request.get_json()
    if not data:
        return jsonify({'success': False, 'error': 'No data provided'}), 400

    machine_id = data.get('machine_id')
    if not machine_id:
        return jsonify({'success': False, 'error': 'machine_id required'}), 400

    with remote_machines_lock:
        remote_machines[machine_id] = {
            'registered_at': datetime.now().isoformat(),
            'last_seen': datetime.now().isoformat(),
            'hostname': data.get('hostname', 'unknown'),
            'ip': request.remote_addr,
            'runs': [],
            'status': 'active'
        }

    return jsonify({'success': True, 'machine_id': machine_id})


@app.route('/api/remote/heartbeat', methods=['POST'])
def api_remote_heartbeat():
    """Heartbeat from remote machine with current runs."""
    data = request.get_json()
    if not data:
        return jsonify({'success': False, 'error': 'No data provided'}), 400

    machine_id = data.get('machine_id')
    if not machine_id:
        return jsonify({'success': False, 'error': 'machine_id required'}), 400

    with remote_machines_lock:
        if machine_id not in remote_machines:
            remote_machines[machine_id] = {
                'registered_at': datetime.now().isoformat(),
                'hostname': data.get('hostname', 'unknown'),
                'ip': request.remote_addr
            }

        remote_machines[machine_id]['last_seen'] = datetime.now().isoformat()
        remote_machines[machine_id]['runs'] = data.get('runs', [])
        remote_machines[machine_id]['status'] = 'active'

        # Include training stats if provided
        if 'training_state' in data:
            remote_machines[machine_id]['training_state'] = data['training_state']

    return jsonify({'success': True})


@app.route('/api/remote/submit_result', methods=['POST'])
def api_remote_submit_result():
    """Submit experiment result from remote machine."""
    data = request.get_json()
    if not data:
        return jsonify({'success': False, 'error': 'No data provided'}), 400

    machine_id = data.get('machine_id')
    run_id = data.get('run_id')
    result = data.get('result', {})

    if not all([machine_id, run_id, result]):
        return jsonify({'success': False, 'error': 'machine_id, run_id, and result required'}), 400

    # Store result in experiments database
    try:
        db_path = Path('data/experiments.db')
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO experiments
            (run_id, machine_id, pnl, pnl_pct, win_rate, total_trades,
             total_cycles, env_vars, timestamp, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run_id,
            machine_id,
            result.get('pnl', 0),
            result.get('pnl_pct', 0),
            result.get('win_rate', 0),
            result.get('total_trades', 0),
            result.get('total_cycles', 0),
            json.dumps(result.get('env_vars', {})),
            datetime.now().isoformat(),
            f'remote:{machine_id}'
        ))

        conn.commit()
        conn.close()

        return jsonify({'success': True, 'run_id': run_id})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/remote/machines')
def api_remote_machines():
    """List registered remote machines."""
    with remote_machines_lock:
        machines = []
        for mid, mdata in remote_machines.items():
            # Mark inactive if no heartbeat in 5 minutes
            last_seen = datetime.fromisoformat(mdata['last_seen'])
            if datetime.now() - last_seen > timedelta(minutes=5):
                mdata['status'] = 'inactive'

            machines.append({
                'machine_id': mid,
                **mdata
            })

    return jsonify({'machines': machines, 'count': len(machines)})


# =============================================================================
# ROUTES - UTILITY
# =============================================================================

@app.route('/api/health')
def api_health():
    """Health check."""
    return jsonify({
        'status': 'ok',
        'version': DASHBOARD_VERSION,
        'timestamp': datetime.now().isoformat(),
        'remote_machines': len(remote_machines)
    })


@app.route('/api/config')
def api_config():
    """Get safe config values."""
    return jsonify({
        'trading_symbol': CONFIG['trading_symbol'],
        'initial_balance': CONFIG['initial_balance'],
        'models_dir': CONFIG['models_dir'],
        'remote_enabled': CONFIG['remote']['enabled']
    })


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    port = int(os.environ.get('DASHBOARD_PORT', CONFIG['port']))
    host = os.environ.get('DASHBOARD_HOST', CONFIG['host'])

    print(f"""
+===============================================================+
|          Unified Trading Dashboard v{DASHBOARD_VERSION}              |
+===============================================================+
|                                                               |
|  Dashboard:     http://{host}:{port}                    |
|                                                               |
|  Pages:                                                       |
|    /           Hub (scoreboard, trades, overview)             |
|    /live       Live trading monitor                           |
|    /training   Training experiments monitor                   |
|    /history    Past model runs                                |
|                                                               |
|  APIs:                                                        |
|    /api/training/*   Training data & runs                     |
|    /api/live/*       Live trading data                        |
|    /api/history/*    Model history                            |
|    /api/scoreboard/* Experiment scoreboard                    |
|    /api/trades/*     Trade browser                            |
|    /api/agent/*      AI agent integration                     |
|    /api/remote/*     Remote machine ingestion                 |
|                                                               |
+===============================================================+
""")

    # Initial data refresh
    refresh_training_data()
    refresh_live_data()

    app.run(host=host, port=port, debug=False, threaded=True)
