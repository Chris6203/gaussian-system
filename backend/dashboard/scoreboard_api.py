#!/usr/bin/env python3
"""
Scoreboard API - Experiment Leaderboard Endpoints

Provides endpoints for viewing, sorting, filtering, and comparing experiments.
Can be integrated into existing dashboards as a blueprint.

Endpoints:
    GET  /api/scoreboard              - Paginated experiment list with sorting/filtering
    GET  /api/scoreboard/<run_name>   - Single experiment details
    POST /api/scoreboard/compare      - Compare 2+ experiments
    GET  /api/scoreboard/running      - Currently running experiments
    GET  /api/scoreboard/stats        - Overall statistics
"""

import sqlite3
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from flask import Blueprint, jsonify, request

# Create blueprint
scoreboard_bp = Blueprint('scoreboard', __name__, url_prefix='/api/scoreboard')

# Database paths
EXPERIMENTS_DB = os.environ.get('EXPERIMENTS_DB', 'data/experiments.db')
MODELS_DIR = os.environ.get('MODELS_DIR', 'models')


def get_db():
    """Get database connection with row factory"""
    conn = sqlite3.connect(EXPERIMENTS_DB)
    conn.row_factory = lambda c, r: {col[0]: r[idx] for idx, col in enumerate(c.description)}
    return conn


@scoreboard_bp.route('', methods=['GET'])
def get_scoreboard():
    """
    Get paginated experiment leaderboard with sorting and filtering.

    Query Parameters:
        page: Page number (default 1)
        limit: Results per page (default 50, max 200)
        sort: Sort field (pnl_pct, win_rate, trades, per_trade_pnl, timestamp)
        order: asc or desc (default desc)
        min_trades: Minimum trade count (default 10)
        min_cycles: Minimum cycles (default 0)
        min_win_rate: Minimum win rate (0-1)
        search: Search in run_name
        date_from: Start date filter (YYYY-MM-DD)
        date_to: End date filter (YYYY-MM-DD)
    """
    # Parse parameters
    page = max(1, request.args.get('page', 1, type=int))
    limit = min(200, max(1, request.args.get('limit', 50, type=int)))
    offset = (page - 1) * limit

    sort_by = request.args.get('sort', 'pnl_pct')
    order = request.args.get('order', 'desc').upper()

    min_trades = request.args.get('min_trades', 10, type=int)
    min_cycles = request.args.get('min_cycles', 0, type=int)
    min_win_rate = request.args.get('min_win_rate', type=float)
    search = request.args.get('search', '')
    date_from = request.args.get('date_from', '')
    date_to = request.args.get('date_to', '')

    # Validate sort field
    valid_sorts = {'pnl_pct', 'win_rate', 'trades', 'per_trade_pnl', 'timestamp', 'cycles'}
    if sort_by not in valid_sorts:
        sort_by = 'pnl_pct'
    if order not in ('ASC', 'DESC'):
        order = 'DESC'

    conn = get_db()
    cursor = conn.cursor()

    # Build WHERE clause
    conditions = ["trades >= ?"]
    params = [min_trades]

    if min_cycles > 0:
        conditions.append("cycles >= ?")
        params.append(min_cycles)

    if min_win_rate is not None:
        conditions.append("win_rate >= ?")
        params.append(min_win_rate)

    if search:
        conditions.append("run_name LIKE ?")
        params.append(f"%{search}%")

    if date_from:
        conditions.append("timestamp >= ?")
        params.append(date_from)

    if date_to:
        conditions.append("timestamp <= ?")
        params.append(date_to + " 23:59:59")

    where_clause = " AND ".join(conditions)

    # Check for running experiments
    cursor.execute("""
        SELECT run_name FROM running_experiments
        WHERE last_heartbeat > datetime('now', '-5 minutes')
    """)
    running_names = {row['run_name'] for row in cursor.fetchall()}

    # Get total count
    cursor.execute(f"SELECT COUNT(*) as total FROM experiments WHERE {where_clause}", params)
    total = cursor.fetchone()['total']

    # Get experiments
    query = f"""
        SELECT
            run_name, timestamp, pnl, pnl_pct, cycles, trades,
            win_rate, wins, losses, per_trade_pnl, config_hash
        FROM experiments
        WHERE {where_clause}
        ORDER BY {sort_by} {order}
        LIMIT ? OFFSET ?
    """
    cursor.execute(query, params + [limit, offset])
    experiments = cursor.fetchall()

    conn.close()

    # Mark running experiments
    for exp in experiments:
        exp['is_running'] = exp['run_name'] in running_names
        # Format win_rate as percentage
        if exp['win_rate']:
            exp['win_rate_pct'] = round(exp['win_rate'] * 100, 1)

    return jsonify({
        "success": True,
        "experiments": experiments,
        "pagination": {
            "page": page,
            "limit": limit,
            "total": total,
            "total_pages": (total + limit - 1) // limit
        },
        "filters": {
            "sort": sort_by,
            "order": order,
            "min_trades": min_trades,
            "min_cycles": min_cycles,
            "min_win_rate": min_win_rate,
            "search": search,
            "date_from": date_from,
            "date_to": date_to
        },
        "running_count": len(running_names)
    })


@scoreboard_bp.route('/<run_name>', methods=['GET'])
def get_experiment_detail(run_name: str):
    """Get detailed information for a single experiment."""
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT * FROM experiments WHERE run_name = ?
    """, (run_name,))
    exp = cursor.fetchone()

    if not exp:
        conn.close()
        return jsonify({"error": f"Experiment '{run_name}' not found"}), 404

    # Check if running
    cursor.execute("""
        SELECT * FROM running_experiments WHERE run_name = ?
    """, (run_name,))
    running_info = cursor.fetchone()

    conn.close()

    # Parse env_vars
    if exp.get('env_vars'):
        try:
            exp['env_vars'] = json.loads(exp['env_vars'])
        except:
            pass

    # Check for model files
    model_dir = Path(MODELS_DIR) / run_name
    exp['files'] = {
        'exists': model_dir.exists(),
        'has_run_info': (model_dir / 'run_info.json').exists() if model_dir.exists() else False,
        'has_summary': (model_dir / 'SUMMARY.txt').exists() if model_dir.exists() else False,
        'has_rl_model': (model_dir / 'unified_rl.pth').exists() if model_dir.exists() else False,
        'has_decision_records': (model_dir / 'state' / 'decision_records.jsonl').exists() if model_dir.exists() else False
    }

    if exp['files']['has_decision_records']:
        try:
            exp['files']['decision_records_size_mb'] = round(
                (model_dir / 'state' / 'decision_records.jsonl').stat().st_size / (1024 * 1024), 2
            )
        except:
            pass

    exp['is_running'] = running_info is not None
    exp['running_info'] = running_info

    return jsonify({
        "success": True,
        "experiment": exp
    })


@scoreboard_bp.route('/compare', methods=['POST'])
def compare_experiments():
    """
    Compare multiple experiments side-by-side.

    Request Body:
    {
        "runs": ["run1", "run2", "run3"]
    }
    """
    data = request.get_json() or {}
    run_names = data.get('runs', [])

    if not run_names or len(run_names) < 2:
        return jsonify({"error": "At least 2 runs required for comparison"}), 400

    conn = get_db()
    cursor = conn.cursor()

    placeholders = ','.join(['?' for _ in run_names])
    cursor.execute(f"""
        SELECT
            run_name, pnl_pct, win_rate, trades,
            per_trade_pnl, cycles, env_vars, timestamp
        FROM experiments
        WHERE run_name IN ({placeholders})
    """, run_names)

    results = cursor.fetchall()
    conn.close()

    if not results:
        return jsonify({"error": "No experiments found"}), 404

    # Parse env_vars and collect all keys
    all_keys = set()
    for r in results:
        if r.get('env_vars'):
            try:
                r['env_vars'] = json.loads(r['env_vars'])
                all_keys.update(r['env_vars'].keys())
            except:
                r['env_vars'] = {}

    # Build comparison matrix
    config_comparison = []
    for key in sorted(all_keys):
        row = {"key": key}
        values = []
        for r in results:
            val = r.get('env_vars', {}).get(key, '-')
            row[r['run_name']] = val
            values.append(str(val))
        row['is_different'] = len(set(values)) > 1
        config_comparison.append(row)

    # Sort: different values first
    config_comparison.sort(key=lambda x: (not x['is_different'], x['key']))

    return jsonify({
        "success": True,
        "runs": results,
        "config_comparison": config_comparison,
        "metrics_summary": {
            "best_pnl": max(r['pnl_pct'] or 0 for r in results),
            "best_win_rate": max(r['win_rate'] or 0 for r in results),
            "total_runs": len(results)
        }
    })


@scoreboard_bp.route('/running', methods=['GET'])
def get_running_experiments():
    """Get currently running experiments with live progress."""
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            run_name, started_at, pid, machine_id,
            last_heartbeat, current_cycle, current_pnl,
            target_cycles, env_vars
        FROM running_experiments
        WHERE last_heartbeat > datetime('now', '-5 minutes')
        ORDER BY started_at DESC
    """)
    running = cursor.fetchall()
    conn.close()

    # Calculate progress and elapsed time
    for r in running:
        if r.get('target_cycles') and r.get('current_cycle'):
            r['progress_pct'] = round(r['current_cycle'] / r['target_cycles'] * 100, 1)
        if r.get('started_at'):
            try:
                started = datetime.fromisoformat(r['started_at'])
                elapsed = datetime.now() - started
                r['elapsed_minutes'] = round(elapsed.total_seconds() / 60, 1)
            except:
                pass
        if r.get('env_vars'):
            try:
                r['env_vars'] = json.loads(r['env_vars'])
            except:
                pass

    # Also check heartbeat file
    heartbeat = None
    heartbeat_path = Path('data/running_experiment.json')
    if heartbeat_path.exists():
        try:
            with open(heartbeat_path) as f:
                heartbeat = json.load(f)
        except:
            pass

    return jsonify({
        "success": True,
        "running": running,
        "count": len(running),
        "heartbeat_file": heartbeat
    })


@scoreboard_bp.route('/stats', methods=['GET'])
def get_stats():
    """Get overall experiment statistics."""
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            COUNT(*) as total_experiments,
            COUNT(DISTINCT date(timestamp)) as unique_days,
            SUM(trades) as total_trades,
            AVG(pnl_pct) as avg_pnl_pct,
            MAX(pnl_pct) as best_pnl_pct,
            MIN(pnl_pct) as worst_pnl_pct,
            AVG(win_rate) as avg_win_rate,
            MAX(win_rate) as best_win_rate,
            AVG(per_trade_pnl) as avg_per_trade_pnl
        FROM experiments
        WHERE trades >= 10
    """)
    stats = cursor.fetchone()

    # Get category breakdown (by env var patterns)
    cursor.execute("""
        SELECT
            CASE
                WHEN env_vars LIKE '%PREDICTOR_ARCH%' THEN 'architecture'
                WHEN env_vars LIKE '%STOP_LOSS%' OR env_vars LIKE '%TAKE_PROFIT%' THEN 'exit_strategy'
                WHEN env_vars LIKE '%HMM%' THEN 'entry_strategy'
                WHEN env_vars LIKE '%LOAD_PRETRAINED%' THEN 'pretrained'
                ELSE 'other'
            END as category,
            COUNT(*) as count,
            AVG(pnl_pct) as avg_pnl
        FROM experiments
        WHERE trades >= 10
        GROUP BY category
        ORDER BY count DESC
    """)
    categories = cursor.fetchall()

    # Get running count
    cursor.execute("""
        SELECT COUNT(*) as running FROM running_experiments
        WHERE last_heartbeat > datetime('now', '-5 minutes')
    """)
    running = cursor.fetchone()['running']

    conn.close()

    return jsonify({
        "success": True,
        "stats": {
            "total_experiments": stats['total_experiments'],
            "unique_days": stats['unique_days'],
            "total_trades": stats['total_trades'],
            "avg_pnl_pct": round(stats['avg_pnl_pct'], 2) if stats['avg_pnl_pct'] else 0,
            "best_pnl_pct": round(stats['best_pnl_pct'], 2) if stats['best_pnl_pct'] else 0,
            "worst_pnl_pct": round(stats['worst_pnl_pct'], 2) if stats['worst_pnl_pct'] else 0,
            "avg_win_rate_pct": round(stats['avg_win_rate'] * 100, 1) if stats['avg_win_rate'] else 0,
            "best_win_rate_pct": round(stats['best_win_rate'] * 100, 1) if stats['best_win_rate'] else 0,
            "avg_per_trade_pnl": round(stats['avg_per_trade_pnl'], 2) if stats['avg_per_trade_pnl'] else 0
        },
        "categories": categories,
        "running_experiments": running
    })
