#!/usr/bin/env python3
"""
Dashboard Hub Server
====================

Central dashboard that provides:
1. Scoreboard - Experiment leaderboard with filtering/comparison
2. Trade Browser - Deep trade analysis across all runs
3. Agent API - REST endpoints for AI collaboration
4. Links to existing dashboards (training, live)

Run this alongside existing dashboards:
    python dashboard_hub_server.py  # Port 5003 (hub)
    python training_dashboard_server.py  # Port 5001 (existing)
    python dashboard_server.py  # Port 5000 (existing)

Or run standalone with all features.
"""

import os
import json
from pathlib import Path
from datetime import datetime

import requests
from flask import Flask, jsonify, render_template, send_from_directory, request
from flask_cors import CORS

# Import blueprints
from backend.dashboard.agent_api import agent_bp
from backend.dashboard.scoreboard_api import scoreboard_bp
from backend.dashboard.trades_api import trades_bp

# Configuration
def load_config():
    config_path = Path('config.json')
    defaults = {
        'port': 5003,
        'host': '0.0.0.0',
        'existing_dashboards': {
            'training': 'http://localhost:5001',
            'live': 'http://localhost:5000'
        }
    }

    if config_path.exists():
        try:
            with open(config_path) as f:
                cfg = json.load(f)
            hub_cfg = cfg.get('dashboard_hub', {})
            for k, v in hub_cfg.items():
                defaults[k] = v
        except Exception as e:
            print(f"Warning: Could not load config: {e}")

    return defaults

CONFIG = load_config()

# Create Flask app with static and template folders
app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# Register blueprints
app.register_blueprint(agent_bp)
app.register_blueprint(scoreboard_bp)
app.register_blueprint(trades_bp)

@app.route('/')
def index():
    return render_template(
        'hub.html',
        training_url=CONFIG['existing_dashboards']['training'],
        live_url=CONFIG['existing_dashboards']['live']
    )


@app.route('/health')
def health():
    return jsonify({
        "status": "ok",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    })


# ===== TRAINING DASHBOARD PROXY =====
# Proxy requests to training dashboard so browser doesn't need direct access to port 5001
TRAINING_SERVER = os.environ.get('TRAINING_SERVER', '192.168.20.235')
TRAINING_PORT = os.environ.get('TRAINING_PORT', '5001')

@app.route('/api/training/runs')
def proxy_training_runs():
    """Proxy request to training dashboard /api/runs"""
    try:
        resp = requests.get(
            f'http://{TRAINING_SERVER}:{TRAINING_PORT}/api/runs',
            timeout=5
        )
        return jsonify(resp.json())
    except requests.RequestException as e:
        return jsonify({
            'success': False,
            'error': f'Training dashboard not reachable: {e}',
            'runs': [],
            'active_count': 0
        })


@app.route('/api/training/data')
def proxy_training_data():
    """Proxy request to training dashboard /api/data"""
    try:
        # Forward any query params
        params = request.args.to_dict()
        resp = requests.get(
            f'http://{TRAINING_SERVER}:{TRAINING_PORT}/api/data',
            params=params,
            timeout=20  # Longer timeout for parsing large log files
        )
        return jsonify(resp.json())
    except requests.RequestException as e:
        return jsonify({
            'success': False,
            'error': f'Training dashboard not reachable: {e}'
        })


@app.route('/api/training/chart')
def proxy_training_chart():
    """Proxy request to training dashboard /api/chart (SPY prices, trades, predictions)"""
    try:
        # Forward any query params (like log_file)
        params = request.args.to_dict()
        resp = requests.get(
            f'http://{TRAINING_SERVER}:{TRAINING_PORT}/api/chart',
            params=params,
            timeout=10
        )
        return jsonify(resp.json())
    except requests.RequestException as e:
        return jsonify({
            'success': False,
            'error': f'Training dashboard not reachable: {e}',
            'spy_prices': [],
            'trades': [],
            'lstm_predictions': []
        })


if __name__ == '__main__':
    port = int(os.environ.get('HUB_PORT', CONFIG['port']))
    host = os.environ.get('HUB_HOST', CONFIG['host'])

    print(f"""
+===============================================================+
|            Trading Dashboard Hub v1.0.0                       |
+===============================================================+
|  Hub Dashboard:     http://{host}:{port}
|  Agent API:         http://{host}:{port}/api/agent/summary
|  Scoreboard API:    http://{host}:{port}/api/scoreboard
|  Trades API:        http://{host}:{port}/api/trades
+---------------------------------------------------------------+
|  Existing Dashboards (external links):                        |
|  Training:          {CONFIG['existing_dashboards']['training']}
|  Live:              {CONFIG['existing_dashboards']['live']}
+===============================================================+
""")

    app.run(host=host, port=port, debug=False, threaded=True)
