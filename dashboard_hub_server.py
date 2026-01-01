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

from flask import Flask, jsonify, render_template_string, send_from_directory
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

# Create Flask app
app = Flask(__name__)
CORS(app)

# Register blueprints
app.register_blueprint(agent_bp)
app.register_blueprint(scoreboard_bp)
app.register_blueprint(trades_bp)

# HTML Template for Hub
HUB_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Trading Dashboard Hub</title>
    <style>
        :root {
            --bg: #0d1117;
            --bg-card: #161b22;
            --bg-elevated: #21262d;
            --border: #30363d;
            --text: #c9d1d9;
            --text-muted: #8b949e;
            --green: #3fb950;
            --red: #f85149;
            --cyan: #58a6ff;
            --yellow: #d29922;
            --accent: #58a6ff;
        }
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg);
            color: var(--text);
            min-height: 100vh;
        }
        .header {
            background: var(--bg-card);
            border-bottom: 1px solid var(--border);
            padding: 16px 24px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .header h1 {
            font-size: 20px;
            font-weight: 600;
        }
        .header .status {
            display: flex;
            gap: 16px;
            align-items: center;
        }
        .status-badge {
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 500;
        }
        .status-badge.running {
            background: rgba(63, 185, 80, 0.2);
            color: var(--green);
        }
        .tabs {
            background: var(--bg-card);
            border-bottom: 1px solid var(--border);
            display: flex;
            gap: 0;
            padding: 0 24px;
        }
        .tab {
            padding: 12px 20px;
            color: var(--text-muted);
            text-decoration: none;
            border-bottom: 2px solid transparent;
            transition: all 0.2s;
            cursor: pointer;
        }
        .tab:hover {
            color: var(--text);
        }
        .tab.active {
            color: var(--cyan);
            border-bottom-color: var(--cyan);
        }
        .tab.external {
            color: var(--yellow);
        }
        .content {
            padding: 24px;
            max-width: 1600px;
            margin: 0 auto;
        }
        .card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 8px;
            margin-bottom: 16px;
        }
        .card-header {
            padding: 16px;
            border-bottom: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .card-header h2 {
            font-size: 16px;
            font-weight: 600;
        }
        .card-body {
            padding: 16px;
        }
        .filters {
            display: flex;
            gap: 12px;
            flex-wrap: wrap;
            margin-bottom: 16px;
        }
        .filters input, .filters select {
            background: var(--bg-elevated);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 8px 12px;
            color: var(--text);
            font-size: 13px;
        }
        .filters button {
            background: var(--accent);
            border: none;
            border-radius: 6px;
            padding: 8px 16px;
            color: white;
            cursor: pointer;
            font-weight: 500;
        }
        .filters button:hover {
            opacity: 0.9;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }
        th {
            text-align: left;
            padding: 10px 12px;
            color: var(--text-muted);
            font-weight: 500;
            font-size: 11px;
            text-transform: uppercase;
            border-bottom: 1px solid var(--border);
            cursor: pointer;
            user-select: none;
        }
        th:hover {
            color: var(--cyan);
        }
        th.sortable::after {
            content: ' ↕';
            opacity: 0.3;
        }
        th.sort-asc::after {
            content: ' ↑';
            opacity: 1;
        }
        th.sort-desc::after {
            content: ' ↓';
            opacity: 1;
        }
        td {
            padding: 10px 12px;
            border-bottom: 1px solid var(--border);
            font-family: 'JetBrains Mono', 'Consolas', monospace;
        }
        tr:hover {
            background: var(--bg-elevated);
        }
        .positive { color: var(--green); }
        .negative { color: var(--red); }
        .running-indicator {
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--green);
            margin-right: 6px;
            animation: pulse 1.5s infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.4; }
        }
        .pagination {
            display: flex;
            justify-content: center;
            gap: 8px;
            margin-top: 16px;
        }
        .pagination button {
            background: var(--bg-elevated);
            border: 1px solid var(--border);
            color: var(--text);
            padding: 6px 12px;
            border-radius: 4px;
            cursor: pointer;
        }
        .pagination button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        .stats-row {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 16px;
            margin-bottom: 16px;
        }
        .stat-card {
            background: var(--bg-elevated);
            border-radius: 8px;
            padding: 16px;
            text-align: center;
        }
        .stat-card .value {
            font-size: 24px;
            font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
        }
        .stat-card .label {
            font-size: 11px;
            color: var(--text-muted);
            text-transform: uppercase;
            margin-top: 4px;
        }
        .tab-content {
            display: none;
        }
        .tab-content.active {
            display: block;
        }
        .api-docs {
            background: var(--bg-elevated);
            border-radius: 8px;
            padding: 16px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 13px;
            overflow-x: auto;
        }
        .api-docs code {
            color: var(--cyan);
        }
        .external-links {
            display: flex;
            gap: 16px;
        }
        .external-link {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 16px 24px;
            background: var(--bg-elevated);
            border-radius: 8px;
            text-decoration: none;
            color: var(--text);
            transition: all 0.2s;
        }
        .external-link:hover {
            background: var(--border);
        }
        .compare-selection {
            display: flex;
            gap: 8px;
            align-items: center;
            flex-wrap: wrap;
        }
        .compare-chip {
            background: var(--accent);
            color: white;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 12px;
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .compare-chip .remove {
            cursor: pointer;
            opacity: 0.7;
        }
        .compare-chip .remove:hover {
            opacity: 1;
        }
        .hidden { display: none; }

        /* Modal styles */
        .modal-overlay {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.7);
            z-index: 1000;
            justify-content: center;
            align-items: center;
        }
        .modal-overlay.active {
            display: flex;
        }
        .modal {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            width: 90%;
            max-width: 1200px;
            max-height: 90vh;
            overflow: auto;
        }
        .modal-header {
            padding: 16px 20px;
            border-bottom: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .modal-header h2 {
            font-size: 18px;
            margin: 0;
        }
        .modal-close {
            background: none;
            border: none;
            color: var(--text-muted);
            font-size: 24px;
            cursor: pointer;
            padding: 0;
            line-height: 1;
        }
        .modal-close:hover {
            color: var(--text);
        }
        .modal-body {
            padding: 20px;
        }
        .chart-container {
            background: var(--bg-elevated);
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 20px;
            height: 300px;
            position: relative;
        }
        .chart-svg {
            width: 100%;
            height: 100%;
        }
        .chart-line {
            fill: none;
            stroke-width: 2;
        }
        .chart-line.positive { stroke: var(--green); }
        .chart-line.negative { stroke: var(--red); }
        .chart-area {
            opacity: 0.2;
        }
        .chart-area.positive { fill: var(--green); }
        .chart-area.negative { fill: var(--red); }
        .chart-zero-line {
            stroke: var(--border);
            stroke-dasharray: 4;
        }
        .chart-tooltip {
            position: absolute;
            background: var(--bg-card);
            border: 1px solid var(--border);
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 12px;
            pointer-events: none;
            display: none;
        }
        .run-stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 12px;
            margin-bottom: 20px;
        }
        .run-stat {
            background: var(--bg-elevated);
            border-radius: 8px;
            padding: 12px;
            text-align: center;
        }
        .run-stat .value {
            font-size: 20px;
            font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
        }
        .run-stat .label {
            font-size: 10px;
            color: var(--text-muted);
            text-transform: uppercase;
            margin-top: 4px;
        }
        .trades-mini-table {
            max-height: 300px;
            overflow-y: auto;
        }
    </style>
</head>
<body>
    <header class="header">
        <h1>Trading Dashboard Hub</h1>
        <div class="status">
            <span id="running-status" class="status-badge">Loading...</span>
            <span id="total-experiments">-- experiments</span>
        </div>
    </header>

    <nav class="tabs">
        <a class="tab" data-tab="training">Live Training</a>
        <a class="tab active" data-tab="scoreboard">Scoreboard</a>
        <a class="tab" data-tab="trades">Trade Browser</a>
        <a class="tab" data-tab="api">Agent API</a>
        <a class="tab external" href="{{ training_url }}" target="_blank">Full Training ↗</a>
        <a class="tab external" href="{{ live_url }}" target="_blank">Live Dashboard ↗</a>
    </nav>

    <main class="content">
        <!-- Live Training Tab -->
        <div id="training-tab" class="tab-content">
            <div class="card" style="margin-bottom: 16px;">
                <div class="card-body" style="display: flex; align-items: center; gap: 16px;">
                    <label style="color: var(--text-muted);">Active Runs:</label>
                    <select id="run-selector" onchange="switchRun()" style="flex: 1; max-width: 400px; background: var(--bg-elevated); border: 1px solid var(--border); border-radius: 6px; padding: 8px 12px; color: var(--text);">
                        <option value="">Loading runs...</option>
                    </select>
                    <span id="run-status" style="font-size: 12px; color: var(--text-muted);"></span>
                </div>
            </div>

            <div class="stats-row" id="training-stats">
                <div class="stat-card">
                    <div class="value" id="train-pnl">--</div>
                    <div class="label">P&L</div>
                </div>
                <div class="stat-card">
                    <div class="value" id="train-pnl-pct">--</div>
                    <div class="label">Return %</div>
                </div>
                <div class="stat-card">
                    <div class="value" id="train-win-rate">--</div>
                    <div class="label">Win Rate</div>
                </div>
                <div class="stat-card">
                    <div class="value" id="train-trades">--</div>
                    <div class="label">Trades</div>
                </div>
                <div class="stat-card">
                    <div class="value" id="train-cycles">--</div>
                    <div class="label">Cycles</div>
                </div>
                <div class="stat-card">
                    <div class="value" id="train-positions">--</div>
                    <div class="label">Open Positions</div>
                </div>
            </div>

            <div class="card">
                <div class="card-header">
                    <h2>Training Status</h2>
                    <span id="train-sim-time" style="color: var(--text-muted); font-size: 13px;">--</span>
                </div>
                <div class="card-body">
                    <div class="stats-row">
                        <div class="stat-card">
                            <div class="value" id="train-balance">--</div>
                            <div class="label">Balance</div>
                        </div>
                        <div class="stat-card">
                            <div class="value" id="train-signal">--</div>
                            <div class="label">Last Signal</div>
                        </div>
                        <div class="stat-card">
                            <div class="value" id="train-confidence">--</div>
                            <div class="label">Confidence</div>
                        </div>
                        <div class="stat-card">
                            <div class="value" id="train-regime">--</div>
                            <div class="label">HMM Regime</div>
                        </div>
                        <div class="stat-card">
                            <div class="value" id="train-spy">--</div>
                            <div class="label">SPY Price</div>
                        </div>
                        <div class="stat-card">
                            <div class="value" id="train-vix">--</div>
                            <div class="label">VIX</div>
                        </div>
                    </div>
                </div>
            </div>

            <div class="card">
                <div class="card-header">
                    <h2>Recent Trades</h2>
                    <span id="train-update-time" style="color: var(--text-muted); font-size: 12px;">Last update: --</span>
                </div>
                <div class="card-body">
                    <table>
                        <thead>
                            <tr>
                                <th>Time</th>
                                <th>Type</th>
                                <th>Strike</th>
                                <th>Entry</th>
                                <th>Exit</th>
                                <th>P&L</th>
                                <th>Exit Reason</th>
                            </tr>
                        </thead>
                        <tbody id="train-trades-body"></tbody>
                    </table>
                </div>
            </div>

            <div id="train-error" class="card" style="display: none; border-color: var(--red);">
                <div class="card-body" style="color: var(--red); text-align: center;">
                    Training dashboard not running. Start with: <code>python training_dashboard_server.py</code>
                </div>
            </div>
        </div>

        <!-- Scoreboard Tab -->
        <div id="scoreboard-tab" class="tab-content active">
            <div class="stats-row" id="scoreboard-stats"></div>

            <div class="card">
                <div class="card-header">
                    <h2>Experiment Leaderboard</h2>
                    <div class="compare-selection">
                        <span id="compare-label" class="hidden">Compare:</span>
                        <div id="compare-chips"></div>
                        <button id="compare-btn" class="hidden" onclick="compareSelected()">Compare</button>
                    </div>
                </div>
                <div class="card-body">
                    <div class="filters">
                        <input type="text" id="search" placeholder="Search runs...">
                        <input type="number" id="min-trades" placeholder="Min trades" value="10">
                        <select id="sort-by">
                            <option value="pnl_pct">Sort by P&L %</option>
                            <option value="win_rate">Sort by Win Rate</option>
                            <option value="per_trade_pnl">Sort by $/Trade</option>
                            <option value="trades">Sort by Trade Count</option>
                            <option value="timestamp">Sort by Date</option>
                        </select>
                        <button onclick="loadScoreboard()">Apply</button>
                    </div>

                    <table id="scoreboard-table">
                        <thead>
                            <tr>
                                <th></th>
                                <th class="sortable" data-sort="run_name">Run Name</th>
                                <th class="sortable" data-sort="timestamp">Date</th>
                                <th class="sortable sort-desc" data-sort="pnl_pct">P&L %</th>
                                <th class="sortable" data-sort="win_rate">Win Rate</th>
                                <th class="sortable" data-sort="trades">Trades</th>
                                <th class="sortable" data-sort="per_trade_pnl">$/Trade</th>
                                <th class="sortable" data-sort="cycles">Cycles</th>
                            </tr>
                        </thead>
                        <tbody id="scoreboard-body"></tbody>
                    </table>

                    <div class="pagination">
                        <button id="prev-btn" onclick="prevPage()" disabled>&lt; Prev</button>
                        <span id="page-info">Page 1</span>
                        <button id="next-btn" onclick="nextPage()">Next &gt;</button>
                    </div>
                </div>
            </div>
        </div>

        <!-- Trade Browser Tab -->
        <div id="trades-tab" class="tab-content">
            <div class="card">
                <div class="card-header">
                    <h2>Trade Browser</h2>
                </div>
                <div class="card-body">
                    <div class="filters">
                        <select id="trade-run-filter">
                            <option value="">All Runs</option>
                        </select>
                        <select id="trade-type-filter">
                            <option value="">All Types</option>
                            <option value="CALL">CALL</option>
                            <option value="PUT">PUT</option>
                        </select>
                        <input type="date" id="trade-date-from" placeholder="From date">
                        <input type="date" id="trade-date-to" placeholder="To date">
                        <button onclick="loadTrades()">Apply</button>
                    </div>

                    <div class="stats-row" id="trade-stats"></div>

                    <table id="trades-table">
                        <thead>
                            <tr>
                                <th class="sortable" data-sort="timestamp" data-table="trades">Time</th>
                                <th class="sortable" data-sort="option_type" data-table="trades">Type</th>
                                <th class="sortable" data-sort="strike_price" data-table="trades">Strike</th>
                                <th class="sortable" data-sort="premium_paid" data-table="trades">Entry</th>
                                <th class="sortable" data-sort="exit_price" data-table="trades">Exit</th>
                                <th class="sortable" data-sort="profit_loss" data-table="trades">P&L</th>
                                <th class="sortable" data-sort="exit_reason" data-table="trades">Exit Reason</th>
                                <th class="sortable" data-sort="run_id" data-table="trades">Run</th>
                            </tr>
                        </thead>
                        <tbody id="trades-body"></tbody>
                    </table>

                    <div class="pagination">
                        <button id="trades-prev-btn" onclick="prevTradePage()" disabled>&lt; Prev</button>
                        <span id="trades-page-info">Page 1</span>
                        <button id="trades-next-btn" onclick="nextTradePage()">Next &gt;</button>
                    </div>
                </div>
            </div>
        </div>

        <!-- API Tab -->
        <div id="api-tab" class="tab-content">
            <div class="card">
                <div class="card-header">
                    <h2>Agent API Documentation</h2>
                </div>
                <div class="card-body">
                    <p style="margin-bottom: 16px; color: var(--text-muted);">
                        REST API for AI agents (Claude, Codex, Gemini) to query experiments and suggest improvements.
                    </p>

                    <div class="api-docs">
<pre>
<code># Get AI-friendly summary</code>
GET /api/agent/summary

<code># Get all experiments</code>
GET /api/agent/experiments?limit=100&min_trades=50

<code># Get top performers</code>
GET /api/agent/experiments/best?limit=10&metric=pnl_pct

<code># Compare experiments</code>
GET /api/agent/experiments/compare?runs=run1,run2,run3

<code># Get suggestion context</code>
GET /api/agent/suggest

<code># Submit experiment idea</code>
POST /api/agent/ideas
{
    "title": "Test wider stop loss",
    "hypothesis": "Wider stops may reduce premature exits",
    "env_vars": {"HARD_STOP_LOSS_PCT": "12"},
    "source": "gemini"
}

<code># Get trades for a run</code>
GET /api/agent/trades/{run_id}

<code># Get running experiments</code>
GET /api/agent/status
</pre>
                    </div>

                    <h3 style="margin: 24px 0 16px;">Quick Test</h3>
                    <div class="external-links">
                        <a class="external-link" href="/api/agent/summary" target="_blank">
                            Test /api/agent/summary
                        </a>
                        <a class="external-link" href="/api/agent/experiments/best?limit=5" target="_blank">
                            Test /api/agent/experiments/best
                        </a>
                        <a class="external-link" href="/api/agent/suggest" target="_blank">
                            Test /api/agent/suggest
                        </a>
                    </div>
                </div>
            </div>
        </div>
    </main>

    <!-- Run Detail Modal -->
    <div id="run-modal" class="modal-overlay" onclick="if(event.target===this) closeRunModal()">
        <div class="modal">
            <div class="modal-header">
                <h2 id="modal-run-name">Run Details</h2>
                <button class="modal-close" onclick="closeRunModal()">&times;</button>
            </div>
            <div class="modal-body">
                <div id="modal-stats" class="run-stats-grid"></div>
                <div class="chart-container">
                    <svg id="pnl-chart" class="chart-svg"></svg>
                    <div id="chart-tooltip" class="chart-tooltip"></div>
                </div>
                <h3 style="margin-bottom: 12px;">Trade History</h3>
                <div class="trades-mini-table">
                    <table style="width: 100%;">
                        <thead>
                            <tr>
                                <th>Time</th>
                                <th>Type</th>
                                <th>P&L</th>
                                <th>Exit Reason</th>
                            </tr>
                        </thead>
                        <tbody id="modal-trades-body"></tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>

    <script>
        // State
        let currentPage = 1;
        let totalPages = 1;
        let tradesPage = 1;
        let tradesTotalPages = 1;
        let selectedForCompare = [];
        let scoreboardSort = { column: 'pnl_pct', direction: 'desc' };
        let tradesSort = { column: 'timestamp', direction: 'desc' };
        let scoreboardData = [];
        let tradesData = [];

        // Format date as YYYY-MM-DD
        function formatDate(dateStr) {
            if (!dateStr) return '-';
            try {
                const d = new Date(dateStr);
                if (isNaN(d.getTime())) return dateStr.split('T')[0] || '-';
                return d.toISOString().split('T')[0];
            } catch (e) {
                return dateStr.split('T')[0] || '-';
            }
        }

        // Format datetime as YYYY-MM-DD HH:MM
        function formatDateTime(dateStr) {
            if (!dateStr) return '-';
            try {
                const d = new Date(dateStr);
                if (isNaN(d.getTime())) return dateStr;
                return d.toISOString().slice(0, 16).replace('T', ' ');
            } catch (e) {
                return dateStr;
            }
        }

        // Sort data by column
        function sortData(data, column, direction) {
            return [...data].sort((a, b) => {
                let aVal = a[column];
                let bVal = b[column];

                // Handle nulls
                if (aVal == null) aVal = direction === 'asc' ? Infinity : -Infinity;
                if (bVal == null) bVal = direction === 'asc' ? Infinity : -Infinity;

                // Numeric comparison for numbers
                if (typeof aVal === 'number' && typeof bVal === 'number') {
                    return direction === 'asc' ? aVal - bVal : bVal - aVal;
                }

                // String comparison
                aVal = String(aVal).toLowerCase();
                bVal = String(bVal).toLowerCase();
                if (direction === 'asc') {
                    return aVal.localeCompare(bVal);
                } else {
                    return bVal.localeCompare(aVal);
                }
            });
        }

        // Handle column header click for sorting
        function setupColumnSorting() {
            document.querySelectorAll('th.sortable').forEach(th => {
                th.addEventListener('click', () => {
                    const column = th.dataset.sort;
                    const table = th.dataset.table || 'scoreboard';

                    if (table === 'trades') {
                        // Toggle direction if same column
                        if (tradesSort.column === column) {
                            tradesSort.direction = tradesSort.direction === 'asc' ? 'desc' : 'asc';
                        } else {
                            tradesSort.column = column;
                            tradesSort.direction = 'desc';
                        }
                        updateTradesSortUI();
                        renderTrades();
                    } else {
                        // Toggle direction if same column
                        if (scoreboardSort.column === column) {
                            scoreboardSort.direction = scoreboardSort.direction === 'asc' ? 'desc' : 'asc';
                        } else {
                            scoreboardSort.column = column;
                            scoreboardSort.direction = 'desc';
                        }
                        updateScoreboardSortUI();
                        renderScoreboard();
                    }
                });
            });
        }

        function updateScoreboardSortUI() {
            document.querySelectorAll('#scoreboard-table th.sortable').forEach(th => {
                th.classList.remove('sort-asc', 'sort-desc');
                if (th.dataset.sort === scoreboardSort.column) {
                    th.classList.add(scoreboardSort.direction === 'asc' ? 'sort-asc' : 'sort-desc');
                }
            });
        }

        function updateTradesSortUI() {
            document.querySelectorAll('#trades-table th.sortable').forEach(th => {
                th.classList.remove('sort-asc', 'sort-desc');
                if (th.dataset.sort === tradesSort.column) {
                    th.classList.add(tradesSort.direction === 'asc' ? 'sort-asc' : 'sort-desc');
                }
            });
        }

        // Tab switching
        document.querySelectorAll('.tab[data-tab]').forEach(tab => {
            tab.addEventListener('click', () => {
                document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
                document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
                tab.classList.add('active');
                document.getElementById(tab.dataset.tab + '-tab').classList.add('active');
            });
        });

        // Load scoreboard stats
        async function loadStats() {
            try {
                const res = await fetch('/api/scoreboard/stats');
                const data = await res.json();
                if (data.success) {
                    document.getElementById('total-experiments').textContent =
                        `${data.stats.total_experiments} experiments`;

                    document.getElementById('running-status').textContent =
                        data.running_experiments > 0 ?
                        `${data.running_experiments} running` : 'No runs active';
                    document.getElementById('running-status').className =
                        'status-badge ' + (data.running_experiments > 0 ? 'running' : '');

                    document.getElementById('scoreboard-stats').innerHTML = `
                        <div class="stat-card">
                            <div class="value positive">+${data.stats.best_pnl_pct}%</div>
                            <div class="label">Best P&L</div>
                        </div>
                        <div class="stat-card">
                            <div class="value">${data.stats.avg_pnl_pct}%</div>
                            <div class="label">Avg P&L</div>
                        </div>
                        <div class="stat-card">
                            <div class="value">${data.stats.best_win_rate_pct}%</div>
                            <div class="label">Best Win Rate</div>
                        </div>
                        <div class="stat-card">
                            <div class="value">$${data.stats.avg_per_trade_pnl}</div>
                            <div class="label">Avg $/Trade</div>
                        </div>
                    `;
                }
            } catch (e) {
                console.error('Failed to load stats:', e);
            }
        }

        // Load scoreboard
        async function loadScoreboard() {
            const search = document.getElementById('search').value;
            const minTrades = document.getElementById('min-trades').value || 10;

            try {
                const res = await fetch(
                    `/api/scoreboard?page=${currentPage}&limit=100&min_trades=${minTrades}&search=${search}`
                );
                const data = await res.json();

                if (data.success) {
                    scoreboardData = data.experiments;
                    totalPages = data.pagination.total_pages;
                    renderScoreboard();
                }
            } catch (e) {
                console.error('Failed to load scoreboard:', e);
            }
        }

        function renderScoreboard() {
            const sorted = sortData(scoreboardData, scoreboardSort.column, scoreboardSort.direction);
            const tbody = document.getElementById('scoreboard-body');
            tbody.innerHTML = sorted.map(exp => `
                <tr>
                    <td>
                        <input type="checkbox"
                               onchange="toggleCompare('${exp.run_name}')"
                               ${selectedForCompare.includes(exp.run_name) ? 'checked' : ''}>
                    </td>
                    <td>
                        ${exp.is_running ? '<span class="running-indicator"></span>' : ''}
                        <a href="#" onclick="openRunModal('${exp.run_name}'); return false;" style="color: var(--cyan)">
                            ${exp.run_name}
                        </a>
                    </td>
                    <td>${formatDate(exp.timestamp)}</td>
                    <td class="${exp.pnl_pct > 0 ? 'positive' : 'negative'}">
                        ${exp.pnl_pct > 0 ? '+' : ''}${exp.pnl_pct?.toFixed(1) || 0}%
                    </td>
                    <td>${exp.win_rate_pct || 0}%</td>
                    <td>${exp.trades || 0}</td>
                    <td class="${(exp.per_trade_pnl || 0) > 0 ? 'positive' : 'negative'}">
                        $${exp.per_trade_pnl?.toFixed(2) || '0.00'}
                    </td>
                    <td>${exp.cycles || 0}</td>
                </tr>
            `).join('');

            document.getElementById('page-info').textContent =
                `Page ${currentPage} of ${totalPages}`;
            document.getElementById('prev-btn').disabled = currentPage <= 1;
            document.getElementById('next-btn').disabled = currentPage >= totalPages;
        }

        function prevPage() {
            if (currentPage > 1) {
                currentPage--;
                loadScoreboard();
            }
        }

        function nextPage() {
            if (currentPage < totalPages) {
                currentPage++;
                loadScoreboard();
            }
        }

        // Compare functionality
        function toggleCompare(runName) {
            const idx = selectedForCompare.indexOf(runName);
            if (idx >= 0) {
                selectedForCompare.splice(idx, 1);
            } else {
                selectedForCompare.push(runName);
            }
            updateCompareUI();
        }

        function updateCompareUI() {
            const chips = document.getElementById('compare-chips');
            const label = document.getElementById('compare-label');
            const btn = document.getElementById('compare-btn');

            if (selectedForCompare.length > 0) {
                label.classList.remove('hidden');
                btn.classList.remove('hidden');
                chips.innerHTML = selectedForCompare.map(name => `
                    <span class="compare-chip">
                        ${name}
                        <span class="remove" onclick="toggleCompare('${name}')">&times;</span>
                    </span>
                `).join('');
            } else {
                label.classList.add('hidden');
                btn.classList.add('hidden');
                chips.innerHTML = '';
            }
        }

        async function compareSelected() {
            if (selectedForCompare.length < 2) {
                alert('Select at least 2 experiments to compare');
                return;
            }
            // Open comparison in new tab
            const url = `/api/scoreboard/compare`;
            const response = await fetch(url, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({runs: selectedForCompare})
            });
            const data = await response.json();
            // For now, show in alert - could make a nice modal
            console.log('Comparison:', data);
            alert('Comparison logged to console. Check for config differences.');
        }

        // Load trades
        async function loadTradeRuns() {
            try {
                const res = await fetch('/api/trades/runs');
                const data = await res.json();
                if (data.success) {
                    const select = document.getElementById('trade-run-filter');
                    select.innerHTML = '<option value="">All Runs</option>' +
                        data.runs.map(r => `<option value="${r.run_id}">${r.run_id} (${r.trade_count})</option>`).join('');
                }
            } catch (e) {
                console.error('Failed to load trade runs:', e);
            }
        }

        async function loadTrades() {
            const runId = document.getElementById('trade-run-filter').value;
            const optionType = document.getElementById('trade-type-filter').value;
            const dateFrom = document.getElementById('trade-date-from').value;
            const dateTo = document.getElementById('trade-date-to').value;

            let url = `/api/trades?page=${tradesPage}&limit=100`;
            if (runId) url += `&run_id=${runId}`;
            if (optionType) url += `&option_type=${optionType}`;
            if (dateFrom) url += `&start_date=${dateFrom}`;
            if (dateTo) url += `&end_date=${dateTo}`;

            try {
                const res = await fetch(url);
                const data = await res.json();

                if (data.success) {
                    tradesData = data.trades;
                    tradesTotalPages = data.pagination.total_pages;
                    renderTrades();
                    document.getElementById('trades-page-info').textContent =
                        `Page ${tradesPage} of ${tradesTotalPages} (${data.pagination.total} trades)`;

                    // Load aggregations for stats
                    loadTradeStats(runId);
                }
            } catch (e) {
                console.error('Failed to load trades:', e);
            }
        }

        function renderTrades() {
            const sorted = sortData(tradesData, tradesSort.column, tradesSort.direction);
            const tbody = document.getElementById('trades-body');
            tbody.innerHTML = sorted.map(t => `
                <tr>
                    <td>${formatDateTime(t.timestamp)}</td>
                    <td>${t.option_type || '-'}</td>
                    <td>$${t.strike_price || '-'}</td>
                    <td>$${t.premium_paid?.toFixed(2) || '-'}</td>
                    <td>$${t.exit_price?.toFixed(2) || '-'}</td>
                    <td class="${(t.profit_loss || 0) > 0 ? 'positive' : 'negative'}">
                        ${t.profit_loss > 0 ? '+' : ''}$${t.profit_loss?.toFixed(2) || '0.00'}
                        (${t.pnl_pct || 0}%)
                    </td>
                    <td>${t.exit_reason || '-'}</td>
                    <td>${t.run_id || '-'}</td>
                </tr>
            `).join('');

            document.getElementById('trades-prev-btn').disabled = tradesPage <= 1;
            document.getElementById('trades-next-btn').disabled = tradesPage >= tradesTotalPages;
        }

        async function loadTradeStats(runId) {
            let url = '/api/trades/aggregations?group_by=exit_reason';
            if (runId) url += `&run_id=${runId}`;

            try {
                const res = await fetch(url);
                const data = await res.json();
                if (data.success && data.summary) {
                    document.getElementById('trade-stats').innerHTML = `
                        <div class="stat-card">
                            <div class="value">${data.summary.total_trades}</div>
                            <div class="label">Total Trades</div>
                        </div>
                        <div class="stat-card">
                            <div class="value">${data.summary.win_rate}%</div>
                            <div class="label">Win Rate</div>
                        </div>
                        <div class="stat-card">
                            <div class="value ${data.summary.total_pnl > 0 ? 'positive' : 'negative'}">
                                $${data.summary.total_pnl?.toFixed(2) || 0}
                            </div>
                            <div class="label">Total P&L</div>
                        </div>
                        <div class="stat-card">
                            <div class="value">$${data.summary.avg_pnl?.toFixed(2) || 0}</div>
                            <div class="label">Avg P&L</div>
                        </div>
                    `;
                }
            } catch (e) {
                console.error('Failed to load trade stats:', e);
            }
        }

        function prevTradePage() {
            if (tradesPage > 1) {
                tradesPage--;
                loadTrades();
            }
        }

        function nextTradePage() {
            if (tradesPage < tradesTotalPages) {
                tradesPage++;
                loadTrades();
            }
        }

        // ===== LIVE TRAINING =====
        let trainingInterval = null;
        let runsInterval = null;
        let selectedRun = null;

        async function loadActiveRuns() {
            try {
                const res = await fetch('{{ training_url }}/api/runs');
                const data = await res.json();

                if (data.success && data.runs.length > 0) {
                    const selector = document.getElementById('run-selector');
                    const currentValue = selector.value;

                    selector.innerHTML = data.runs.map(r => {
                        const activeIcon = r.is_active ? '🟢 ' : '⚪ ';
                        return `<option value="${r.log_file}" ${r.log_file === currentValue ? 'selected' : ''}>
                            ${activeIcon}${r.run_name} (${r.size_mb}MB)
                        </option>`;
                    }).join('');

                    document.getElementById('run-status').textContent =
                        `${data.active_count} active of ${data.runs.length} runs`;

                    // Auto-select first active run if nothing selected
                    if (!currentValue && data.runs.length > 0) {
                        const activeRun = data.runs.find(r => r.is_active) || data.runs[0];
                        selector.value = activeRun.log_file;
                        selectedRun = activeRun.log_file;
                    }
                }
            } catch (e) {
                console.error('Failed to load runs:', e);
            }
        }

        function switchRun() {
            const selector = document.getElementById('run-selector');
            selectedRun = selector.value;
            loadTrainingData();
        }

        async function loadTrainingData() {
            try {
                const res = await fetch('{{ training_url }}/api/data');
                const d = await res.json();

                document.getElementById('train-error').style.display = 'none';

                // Main stats
                const pnlClass = d.pnl >= 0 ? 'positive' : 'negative';
                document.getElementById('train-pnl').className = 'value ' + pnlClass;
                document.getElementById('train-pnl').textContent = (d.pnl >= 0 ? '+' : '') + '$' + d.pnl?.toFixed(2);

                document.getElementById('train-pnl-pct').className = 'value ' + pnlClass;
                document.getElementById('train-pnl-pct').textContent = (d.pnl_pct >= 0 ? '+' : '') + d.pnl_pct?.toFixed(1) + '%';

                document.getElementById('train-win-rate').textContent = d.win_rate?.toFixed(1) + '%';
                document.getElementById('train-trades').textContent = d.trades || 0;
                document.getElementById('train-cycles').textContent = d.cycles || 0;
                document.getElementById('train-positions').textContent = d.current_positions || 0;

                // Status
                document.getElementById('train-balance').textContent = '$' + d.current_balance?.toFixed(0);
                document.getElementById('train-signal').textContent = d.last_signal || '--';
                document.getElementById('train-confidence').textContent = d.signal_confidence?.toFixed(0) + '%';
                document.getElementById('train-regime').textContent = d.hmm_regime || '--';
                document.getElementById('train-spy').textContent = '$' + d.spy_price?.toFixed(2);
                document.getElementById('train-vix').textContent = d.vix_value?.toFixed(2);
                document.getElementById('train-sim-time').textContent = 'Simulated: ' + d.simulated_date + ' ' + d.simulated_time;
                document.getElementById('train-update-time').textContent = 'Last update: ' + new Date().toLocaleTimeString();

                // Recent trades
                if (d.recent_trades && d.recent_trades.length > 0) {
                    const tbody = document.getElementById('train-trades-body');
                    tbody.innerHTML = d.recent_trades.slice(0, 20).map(t => {
                        const pnl = t.profit_loss || t.pnl || 0;
                        const pnlClass = pnl >= 0 ? 'positive' : 'negative';
                        return `<tr>
                            <td>${t.exit_time || t.timestamp || '--'}</td>
                            <td>${t.option_type || t.type || '--'}</td>
                            <td>$${t.strike || t.strike_price || '--'}</td>
                            <td>$${t.entry_price?.toFixed(2) || '--'}</td>
                            <td>$${t.exit_price?.toFixed(2) || '--'}</td>
                            <td class="${pnlClass}">${pnl >= 0 ? '+' : ''}$${pnl.toFixed(2)}</td>
                            <td>${t.exit_reason || '--'}</td>
                        </tr>`;
                    }).join('');
                }
            } catch (e) {
                document.getElementById('train-error').style.display = 'block';
                console.error('Training data fetch failed:', e);
            }
        }

        function startTrainingUpdates() {
            loadActiveRuns();
            loadTrainingData();
            if (!trainingInterval) {
                trainingInterval = setInterval(loadTrainingData, 2000); // Update every 2 seconds
            }
            if (!runsInterval) {
                runsInterval = setInterval(loadActiveRuns, 10000); // Refresh runs list every 10 seconds
            }
        }

        function stopTrainingUpdates() {
            if (trainingInterval) {
                clearInterval(trainingInterval);
                trainingInterval = null;
            }
            if (runsInterval) {
                clearInterval(runsInterval);
                runsInterval = null;
            }
        }

        // Start/stop training updates based on active tab
        document.querySelectorAll('.tab[data-tab]').forEach(tab => {
            tab.addEventListener('click', () => {
                if (tab.dataset.tab === 'training') {
                    startTrainingUpdates();
                } else {
                    stopTrainingUpdates();
                }
            });
        });

        // Initial load
        setupColumnSorting();
        loadStats();
        loadScoreboard();
        loadTradeRuns();
        loadTrades();

        // Refresh stats periodically
        setInterval(loadStats, 30000);

        // ===== RUN DETAIL MODAL =====
        function openRunModal(runName) {
            document.getElementById('modal-run-name').textContent = runName;
            document.getElementById('run-modal').classList.add('active');
            loadRunDetails(runName);
        }

        function closeRunModal() {
            document.getElementById('run-modal').classList.remove('active');
        }

        async function loadRunDetails(runName) {
            // Load P&L curve
            try {
                const curveRes = await fetch(`/api/trades/pnl-curve?run_id=${runName}`);
                const curveData = await curveRes.json();

                if (curveData.success) {
                    renderPnLChart(curveData.curve, curveData.stats);
                    renderModalStats(curveData.stats, runName);
                }
            } catch (e) {
                console.error('Failed to load P&L curve:', e);
            }

            // Load trades
            try {
                const tradesRes = await fetch(`/api/trades?run_id=${runName}&limit=100`);
                const tradesData = await tradesRes.json();

                if (tradesData.success) {
                    renderModalTrades(tradesData.trades);
                }
            } catch (e) {
                console.error('Failed to load trades:', e);
            }
        }

        function renderModalStats(stats, runName) {
            document.getElementById('modal-stats').innerHTML = `
                <div class="run-stat">
                    <div class="value ${stats.total_pnl >= 0 ? 'positive' : 'negative'}">
                        ${stats.total_pnl >= 0 ? '+' : ''}$${stats.total_pnl?.toFixed(2) || 0}
                    </div>
                    <div class="label">Total P&L</div>
                </div>
                <div class="run-stat">
                    <div class="value ${stats.total_pnl_pct >= 0 ? 'positive' : 'negative'}">
                        ${stats.total_pnl_pct >= 0 ? '+' : ''}${stats.total_pnl_pct?.toFixed(1) || 0}%
                    </div>
                    <div class="label">Return</div>
                </div>
                <div class="run-stat">
                    <div class="value">${stats.total_trades || 0}</div>
                    <div class="label">Total Trades</div>
                </div>
                <div class="run-stat">
                    <div class="value negative">-${stats.max_drawdown_pct?.toFixed(1) || 0}%</div>
                    <div class="label">Max Drawdown</div>
                </div>
                <div class="run-stat">
                    <div class="value">$${stats.initial_balance?.toFixed(0) || 0}</div>
                    <div class="label">Starting</div>
                </div>
                <div class="run-stat">
                    <div class="value ${stats.final_balance >= stats.initial_balance ? 'positive' : 'negative'}">
                        $${stats.final_balance?.toFixed(0) || 0}
                    </div>
                    <div class="label">Ending</div>
                </div>
            `;
        }

        function renderPnLChart(curve, stats) {
            const svg = document.getElementById('pnl-chart');
            const container = svg.parentElement;
            const width = container.clientWidth - 32;
            const height = container.clientHeight - 32;
            const padding = { top: 20, right: 20, bottom: 30, left: 60 };
            const chartWidth = width - padding.left - padding.right;
            const chartHeight = height - padding.top - padding.bottom;

            if (!curve || curve.length === 0) {
                svg.innerHTML = '<text x="50%" y="50%" text-anchor="middle" fill="#8b949e">No trade data</text>';
                return;
            }

            // Find min/max for scaling
            const pnls = curve.map(d => d.cumulative_pnl);
            const minPnl = Math.min(0, ...pnls);
            const maxPnl = Math.max(0, ...pnls);
            const range = maxPnl - minPnl || 1;

            // Scale functions
            const xScale = (i) => padding.left + (i / (curve.length - 1 || 1)) * chartWidth;
            const yScale = (pnl) => padding.top + chartHeight - ((pnl - minPnl) / range) * chartHeight;
            const zeroY = yScale(0);

            // Build path
            let linePath = `M ${xScale(0)} ${yScale(curve[0].cumulative_pnl)}`;
            let areaPath = `M ${xScale(0)} ${zeroY} L ${xScale(0)} ${yScale(curve[0].cumulative_pnl)}`;
            for (let i = 1; i < curve.length; i++) {
                linePath += ` L ${xScale(i)} ${yScale(curve[i].cumulative_pnl)}`;
                areaPath += ` L ${xScale(i)} ${yScale(curve[i].cumulative_pnl)}`;
            }
            areaPath += ` L ${xScale(curve.length - 1)} ${zeroY} Z`;

            const finalPnl = curve[curve.length - 1].cumulative_pnl;
            const colorClass = finalPnl >= 0 ? 'positive' : 'negative';

            // Y-axis labels
            const yLabels = [minPnl, minPnl + range/2, maxPnl].map(v => `
                <text x="${padding.left - 8}" y="${yScale(v)}" text-anchor="end"
                      fill="#8b949e" font-size="10" dominant-baseline="middle">
                    $${v.toFixed(0)}
                </text>
                <line x1="${padding.left}" y1="${yScale(v)}" x2="${width - padding.right}" y2="${yScale(v)}"
                      stroke="#30363d" stroke-width="1" stroke-dasharray="2"/>
            `).join('');

            svg.innerHTML = `
                ${yLabels}
                <line class="chart-zero-line" x1="${padding.left}" y1="${zeroY}"
                      x2="${width - padding.right}" y2="${zeroY}"/>
                <path class="chart-area ${colorClass}" d="${areaPath}"/>
                <path class="chart-line ${colorClass}" d="${linePath}"/>
                ${curve.map((d, i) => `
                    <circle cx="${xScale(i)}" cy="${yScale(d.cumulative_pnl)}" r="3"
                            fill="${d.pnl >= 0 ? '#3fb950' : '#f85149'}"
                            opacity="0.7"
                            onmouseover="showChartTooltip(event, '${formatDateTime(d.timestamp)}', ${d.pnl}, ${d.cumulative_pnl}, '${d.option_type}')"
                            onmouseout="hideChartTooltip()"/>
                `).join('')}
            `;
        }

        function showChartTooltip(event, time, pnl, cumPnl, type) {
            const tooltip = document.getElementById('chart-tooltip');
            tooltip.innerHTML = `
                <div><strong>${time}</strong></div>
                <div>${type}: ${pnl >= 0 ? '+' : ''}$${pnl.toFixed(2)}</div>
                <div>Cumulative: ${cumPnl >= 0 ? '+' : ''}$${cumPnl.toFixed(2)}</div>
            `;
            tooltip.style.display = 'block';
            tooltip.style.left = (event.offsetX + 10) + 'px';
            tooltip.style.top = (event.offsetY - 10) + 'px';
        }

        function hideChartTooltip() {
            document.getElementById('chart-tooltip').style.display = 'none';
        }

        function renderModalTrades(trades) {
            const tbody = document.getElementById('modal-trades-body');
            if (!trades || trades.length === 0) {
                tbody.innerHTML = '<tr><td colspan="4" style="text-align: center; color: var(--text-muted);">No trades found</td></tr>';
                return;
            }

            tbody.innerHTML = trades.map(t => `
                <tr>
                    <td>${formatDateTime(t.timestamp)}</td>
                    <td>${t.option_type || '-'}</td>
                    <td class="${(t.profit_loss || 0) >= 0 ? 'positive' : 'negative'}">
                        ${(t.profit_loss || 0) >= 0 ? '+' : ''}$${(t.profit_loss || 0).toFixed(2)}
                    </td>
                    <td>${t.exit_reason || '-'}</td>
                </tr>
            `).join('');
        }

        // Close modal on Escape key
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') closeRunModal();
        });
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(
        HUB_HTML,
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


if __name__ == '__main__':
    port = int(os.environ.get('HUB_PORT', CONFIG['port']))
    host = os.environ.get('HUB_HOST', CONFIG['host'])

    print(f"""
╔═══════════════════════════════════════════════════════════════╗
║            Trading Dashboard Hub v1.0.0                      ║
╠═══════════════════════════════════════════════════════════════╣
║  Hub Dashboard:     http://{host}:{port}
║  Agent API:         http://{host}:{port}/api/agent/summary
║  Scoreboard API:    http://{host}:{port}/api/scoreboard
║  Trades API:        http://{host}:{port}/api/trades
╠═══════════════════════════════════════════════════════════════╣
║  Existing Dashboards (external links):                       ║
║  Training:          {CONFIG['existing_dashboards']['training']}
║  Live:              {CONFIG['existing_dashboards']['live']}
╚═══════════════════════════════════════════════════════════════╝
""")

    app.run(host=host, port=port, debug=False, threaded=True)
