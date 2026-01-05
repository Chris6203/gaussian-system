#!/usr/bin/env python3
"""
Historical Model Viewer Dashboard
==================================

Browse and view performance of past training runs.
Displays SUMMARY.txt data and trade history from models/ directory.

Usage:
    python history_dashboard_server.py

Port: 5002 (configurable in config.json)
"""

import json
import os
import re
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional

from flask import Flask, jsonify, send_file, send_from_directory
from flask_cors import CORS

# Version
DASHBOARD_VERSION = "1.0"

# =============================================================================
# CONFIGURATION
# =============================================================================

def load_config() -> Dict[str, Any]:
    """Load dashboard configuration from config.json."""
    config_path = Path('config.json')

    defaults = {
        'port': 5002,
        'host': '0.0.0.0',
        'models_dir': 'models'
    }

    try:
        if config_path.exists():
            with open(config_path, 'r') as f:
                cfg = json.load(f)
            history_cfg = cfg.get('history_dashboard', {})

            for key, value in history_cfg.items():
                defaults[key] = value

    except Exception as e:
        print(f"Warning: Could not load config: {e}")

    return defaults


CONFIG = load_config()

# =============================================================================
# FLASK APP
# =============================================================================

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

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

        # Parse timestamp
        match = re.search(r'Timestamp:\s*(\S+)', content)
        if match:
            result['timestamp'] = match.group(1)

        # Parse Initial Balance
        match = re.search(r'Initial Balance:\s*\$?([\d,]+\.?\d*)', content)
        if match:
            result['initial_balance'] = float(match.group(1).replace(',', ''))

        # Parse Final Balance
        match = re.search(r'Final Balance:\s*\$?([\d,]+\.?\d*)', content)
        if match:
            result['final_balance'] = float(match.group(1).replace(',', ''))

        # Parse P&L
        match = re.search(r'P&L:\s*\$?([+-]?[\d,]+\.?\d*)\s*\(([+-]?[\d.]+)%\)', content)
        if match:
            result['pnl'] = float(match.group(1).replace(',', ''))
            result['pnl_pct'] = float(match.group(2))

        # Parse Total Trades
        match = re.search(r'Total Trades:\s*(\d+)', content)
        if match:
            result['total_trades'] = int(match.group(1))

        # Parse Total Signals
        match = re.search(r'Total Signals:\s*(\d+)', content)
        if match:
            result['total_signals'] = int(match.group(1))

        # Parse Total Cycles
        match = re.search(r'Total Cycles:\s*(\d+)', content)
        if match:
            result['total_cycles'] = int(match.group(1))

        # Parse Win Rate
        match = re.search(r'Win Rate:\s*([\d.]+)%', content)
        if match:
            result['win_rate'] = float(match.group(1))

        # Parse Wins
        match = re.search(r'Wins:\s*(\d+)', content)
        if match:
            result['wins'] = int(match.group(1))

        # Parse Losses
        match = re.search(r'Losses:\s*(\d+)', content)
        if match:
            result['losses'] = int(match.group(1))

        return result

    except Exception as e:
        print(f"Error parsing {summary_path}: {e}")
        return None


def list_all_models() -> List[Dict[str, Any]]:
    """List all models in the models directory with their summary data."""
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
                    # Add modification time for sorting
                    summary['mtime'] = summary_path.stat().st_mtime
                    summary['mtime_str'] = datetime.fromtimestamp(
                        summary_path.stat().st_mtime
                    ).strftime('%Y-%m-%d %H:%M')
                    models.append(summary)

    # Sort by modification time (newest first)
    models.sort(key=lambda x: x.get('mtime', 0), reverse=True)

    return models


def load_decision_records(run_id: str, limit: int = 100) -> List[Dict[str, Any]]:
    """Load decision records from a model's state directory."""
    models_dir = Path(CONFIG['models_dir'])
    records_path = models_dir / run_id / 'state' / 'decision_records.jsonl'

    if not records_path.exists():
        # Try root of model directory
        records_path = models_dir / run_id / 'decision_records.jsonl'

    if not records_path.exists():
        return []

    records = []
    try:
        with open(records_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        record = json.loads(line)
                        records.append(record)
                    except json.JSONDecodeError:
                        continue

        # Return last N records
        return records[-limit:]

    except Exception as e:
        print(f"Error loading decision records: {e}")
        return []


def load_trades_from_db(run_id: str, summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Load trades from paper_trading.db for a given model run.

    Currently loads all trades. In the future, we could link trades to model runs
    via a run_id column in the trades table.
    """
    db_path = Path('data/paper_trading.db')
    if not db_path.exists():
        return []

    trades = []
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # For now, load all completed trades
        # TODO: Add run_id column to trades table to link trades to specific runs
        cursor.execute("""
            SELECT * FROM trades
            WHERE status IS NOT NULL AND status != 'OPEN'
            ORDER BY timestamp ASC
        """)

        rows = cursor.fetchall()

        for row in rows:
            trade = dict(row)
            trades.append(trade)

        conn.close()

    except Exception as e:
        print(f"Error loading trades from DB: {e}")

    return trades


# =============================================================================
# API ROUTES
# =============================================================================

@app.route('/')
def index():
    """Serve the dashboard HTML."""
    return send_from_directory('templates', 'history.html')


@app.route('/api/models')
def get_models():
    """Get list of all models with summary data."""
    models = list_all_models()
    return jsonify({
        'models': models,
        'count': len(models)
    })


@app.route('/api/model/<run_id>')
def get_model(run_id: str):
    """Get detailed data for a specific model."""
    models_dir = Path(CONFIG['models_dir'])
    summary_path = models_dir / run_id / 'SUMMARY.txt'

    summary = parse_summary_txt(summary_path)
    if not summary:
        return jsonify({'error': f'Model {run_id} not found'}), 404

    # Check what files exist
    state_dir = models_dir / run_id / 'state'
    files = []
    if state_dir.exists():
        files = [f.name for f in state_dir.iterdir() if f.is_file()]

    summary['available_files'] = files

    return jsonify(summary)


@app.route('/api/model/<run_id>/trades')
def get_model_trades(run_id: str):
    """Get trades for a model from paper_trading.db."""
    # First get the model summary to get timestamp
    models_dir = Path(CONFIG['models_dir'])
    summary_path = models_dir / run_id / 'SUMMARY.txt'
    summary = parse_summary_txt(summary_path)

    if not summary:
        return jsonify({'error': f'Model {run_id} not found'}), 404

    # Load trades from database
    db_trades = load_trades_from_db(run_id, summary)

    # Format trades for the frontend
    trades = []
    for t in db_trades:
        entry_price = t.get('premium_paid', 0) or t.get('entry_price', 0) or 0
        exit_price = t.get('exit_price', 0) or 0
        profit_loss = t.get('profit_loss', 0) or 0

        # Calculate P&L percentage
        pnl_pct = 0
        if entry_price and entry_price > 0:
            pnl_pct = (profit_loss / entry_price) * 100

        trade = {
            'id': t.get('id', ''),
            'timestamp': t.get('timestamp', ''),
            'entry_time': t.get('timestamp', ''),
            'exit_time': t.get('exit_timestamp', ''),
            'option_type': t.get('option_type', ''),
            'entry_price': entry_price,
            'exit_price': exit_price,
            'profit_loss': profit_loss,
            'pnl_pct': pnl_pct,
            'exit_reason': t.get('status', ''),
            'contracts': t.get('quantity', 1),
            'symbol': t.get('symbol', 'SPY'),
            'strike': t.get('strike_price', 0),
            'confidence': t.get('ml_confidence', 0)
        }

        trades.append(trade)

    return jsonify({
        'trades': trades,
        'trade_count': len(trades),
        'model_timestamp': summary.get('timestamp', ''),
        'source': 'paper_trading.db'
    })


@app.route('/api/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok',
        'version': DASHBOARD_VERSION,
        'models_dir': CONFIG['models_dir']
    })


@app.route('/api/spy_prices')
def get_spy_prices():
    """Get SPY price data for a date range."""
    from flask import request

    start = request.args.get('start', '')
    end = request.args.get('end', '')

    # Try historical database
    db_path = Path('data/db/historical.db')
    if not db_path.exists():
        db_path = Path('data/historical.db')

    if not db_path.exists():
        return jsonify({'prices': [], 'error': 'No historical database found'})

    prices = []
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Get SPY prices from historical_data table
        if start and end:
            cursor.execute("""
                SELECT timestamp, close_price, high_price, low_price, open_price, volume
                FROM historical_data
                WHERE symbol = 'SPY' AND timestamp >= ? AND timestamp <= ?
                ORDER BY timestamp ASC
            """, (start, end))
        else:
            # Get last day of data
            cursor.execute("""
                SELECT timestamp, close_price, high_price, low_price, open_price, volume
                FROM historical_data
                WHERE symbol = 'SPY'
                ORDER BY timestamp DESC
                LIMIT 500
            """)

        rows = cursor.fetchall()

        for row in rows:
            prices.append({
                'timestamp': row[0],
                'close': row[1],
                'high': row[2],
                'low': row[3],
                'open': row[4],
                'volume': row[5]
            })

        # Reverse if we got DESC order
        if not start:
            prices = prices[::-1]

        conn.close()

    except Exception as e:
        print(f"Error loading SPY prices: {e}")
        return jsonify({'prices': [], 'error': str(e)})

    return jsonify({
        'prices': prices,
        'count': len(prices)
    })


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print(f"[*] Historical Model Viewer v{DASHBOARD_VERSION}")
    print("=" * 60)
    print()
    print(f"Dashboard URL: http://localhost:{CONFIG['port']}")
    print(f"Models Directory: {CONFIG['models_dir']}")
    print()

    # Count models
    models = list_all_models()
    print(f"Found {len(models)} models with SUMMARY.txt")
    print()
    print("Press Ctrl+C to stop")
    print()

    app.run(
        host=CONFIG['host'],
        port=CONFIG['port'],
        debug=False,
        threaded=True
    )
