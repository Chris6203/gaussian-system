#!/usr/bin/env python3
"""
Agent-Accessible API for AI Collaboration

Provides REST endpoints optimized for consumption by AI agents (Claude, Codex, Gemini, etc.)
to analyze experiments, compare configurations, and suggest improvements.

Endpoints:
    GET  /api/agent/experiments          - All experiments with metrics
    GET  /api/agent/experiments/best     - Top N performers
    GET  /api/agent/experiments/compare  - Compare runs side-by-side
    GET  /api/agent/trades/<run_id>      - All trades for a run
    GET  /api/agent/config/<run_id>      - Config/env vars for a run
    GET  /api/agent/suggest              - AI-friendly context for suggestions
    POST /api/agent/ideas                - Submit new experiment ideas
    GET  /api/agent/status               - Current running experiments
    GET  /api/agent/summary              - High-level system summary
"""

import sqlite3
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from flask import Blueprint, jsonify, request

# Create blueprint
agent_bp = Blueprint('agent', __name__, url_prefix='/api/agent')

# Database paths
EXPERIMENTS_DB = os.environ.get('EXPERIMENTS_DB', 'data/experiments.db')
PAPER_TRADING_DB = os.environ.get('PAPER_TRADING_DB', 'data/paper_trading.db')
MODELS_DIR = os.environ.get('MODELS_DIR', 'models')
COLLAB_DIR = os.environ.get('COLLAB_DIR', '.claude/collab')


def get_experiments_db():
    """Get experiments database connection"""
    return sqlite3.connect(EXPERIMENTS_DB)


def get_paper_trading_db():
    """Get paper trading database connection"""
    return sqlite3.connect(PAPER_TRADING_DB)


def dict_factory(cursor, row):
    """Convert sqlite row to dict"""
    return {col[0]: row[idx] for idx, col in enumerate(cursor.description)}


@agent_bp.route('/summary', methods=['GET'])
def get_summary():
    """
    High-level system summary for AI agents.

    Returns context about the trading system, best configurations,
    and key insights to help AI agents understand the current state.
    """
    conn = get_experiments_db()
    conn.row_factory = dict_factory
    cursor = conn.cursor()

    # Get overall stats
    cursor.execute("""
        SELECT
            COUNT(*) as total_experiments,
            AVG(pnl_pct) as avg_pnl_pct,
            MAX(pnl_pct) as best_pnl_pct,
            AVG(win_rate) as avg_win_rate,
            MAX(win_rate) as best_win_rate,
            SUM(trades) as total_trades
        FROM experiments
        WHERE trades >= 10
    """)
    stats = cursor.fetchone()

    # Get best run
    cursor.execute("""
        SELECT run_name, pnl_pct, win_rate, trades, per_trade_pnl, env_vars
        FROM experiments
        WHERE trades >= 50
        ORDER BY pnl_pct DESC
        LIMIT 1
    """)
    best_run = cursor.fetchone()

    # Get recent experiments
    cursor.execute("""
        SELECT run_name, pnl_pct, win_rate, trades, timestamp
        FROM experiments
        ORDER BY timestamp DESC
        LIMIT 10
    """)
    recent = cursor.fetchall()

    # Get running experiments
    cursor.execute("""
        SELECT run_name, current_cycle, current_pnl, started_at
        FROM running_experiments
        WHERE last_heartbeat > datetime('now', '-5 minutes')
    """)
    running = cursor.fetchall()

    conn.close()

    # Parse best run env_vars
    best_config = {}
    if best_run and best_run.get('env_vars'):
        try:
            best_config = json.loads(best_run['env_vars'])
        except:
            pass

    return jsonify({
        "context": "SPY options trading bot using Bayesian neural networks and HMM regime detection",
        "system_info": {
            "total_experiments": stats['total_experiments'] if stats else 0,
            "avg_pnl_pct": round(stats['avg_pnl_pct'], 2) if stats and stats['avg_pnl_pct'] else 0,
            "best_pnl_pct": round(stats['best_pnl_pct'], 2) if stats and stats['best_pnl_pct'] else 0,
            "avg_win_rate": round(stats['avg_win_rate'] * 100, 1) if stats and stats['avg_win_rate'] else 0,
            "total_trades_analyzed": stats['total_trades'] if stats else 0
        },
        "best_run": {
            "name": best_run['run_name'] if best_run else None,
            "pnl_pct": best_run['pnl_pct'] if best_run else None,
            "win_rate": round(best_run['win_rate'] * 100, 1) if best_run and best_run['win_rate'] else None,
            "trades": best_run['trades'] if best_run else None,
            "per_trade_pnl": best_run['per_trade_pnl'] if best_run else None,
            "config": best_config
        },
        "recent_experiments": recent,
        "running_experiments": running,
        "key_parameters": {
            "exit_strategy": "HARD_STOP_LOSS_PCT, HARD_TAKE_PROFIT_PCT",
            "entry_filters": "HMM_STRONG_BULLISH, HMM_STRONG_BEARISH, HMM_MIN_CONFIDENCE",
            "architecture": "PREDICTOR_ARCH, TEMPORAL_ENCODER",
            "training": "TT_MAX_CYCLES, LOAD_PRETRAINED"
        },
        "documentation": {
            "claude_md": "CLAUDE.md",
            "architecture": "docs/SYSTEM_ARCHITECTURE.md",
            "results": "RESULTS_TRACKER.md"
        }
    })


@agent_bp.route('/experiments', methods=['GET'])
def get_experiments():
    """
    Get all experiments with filtering and sorting.

    Query Parameters:
        limit: Max results (default 100)
        offset: Pagination offset
        min_trades: Minimum trade count filter
        min_pnl: Minimum P&L % filter
        sort: Sort field (pnl_pct, win_rate, trades, timestamp)
        order: asc or desc (default desc)
    """
    limit = request.args.get('limit', 100, type=int)
    offset = request.args.get('offset', 0, type=int)
    min_trades = request.args.get('min_trades', 10, type=int)
    min_pnl = request.args.get('min_pnl', type=float)
    sort_by = request.args.get('sort', 'pnl_pct')
    order = request.args.get('order', 'desc').upper()

    if order not in ('ASC', 'DESC'):
        order = 'DESC'

    valid_sorts = {'pnl_pct', 'win_rate', 'trades', 'timestamp', 'per_trade_pnl'}
    if sort_by not in valid_sorts:
        sort_by = 'pnl_pct'

    conn = get_experiments_db()
    conn.row_factory = dict_factory
    cursor = conn.cursor()

    query = f"""
        SELECT
            run_name, timestamp, start_date, end_date,
            initial_balance, final_balance, pnl, pnl_pct,
            cycles, trades, win_rate, wins, losses,
            per_trade_pnl, config_hash, env_vars, notes
        FROM experiments
        WHERE trades >= ?
        {"AND pnl_pct >= ?" if min_pnl is not None else ""}
        ORDER BY {sort_by} {order}
        LIMIT ? OFFSET ?
    """

    params = [min_trades]
    if min_pnl is not None:
        params.append(min_pnl)
    params.extend([limit, offset])

    cursor.execute(query, params)
    experiments = cursor.fetchall()

    # Get total count
    count_query = f"""
        SELECT COUNT(*) as total FROM experiments
        WHERE trades >= ?
        {"AND pnl_pct >= ?" if min_pnl is not None else ""}
    """
    count_params = [min_trades]
    if min_pnl is not None:
        count_params.append(min_pnl)
    cursor.execute(count_query, count_params)
    total = cursor.fetchone()['total']

    conn.close()

    # Parse env_vars JSON for each experiment
    for exp in experiments:
        if exp.get('env_vars'):
            try:
                exp['env_vars'] = json.loads(exp['env_vars'])
            except:
                pass

    return jsonify({
        "experiments": experiments,
        "total": total,
        "limit": limit,
        "offset": offset,
        "filters_applied": {
            "min_trades": min_trades,
            "min_pnl": min_pnl,
            "sort": sort_by,
            "order": order
        }
    })


@agent_bp.route('/experiments/best', methods=['GET'])
def get_best_experiments():
    """
    Get top N performing experiments for AI analysis.

    Query Parameters:
        limit: Number of top runs to return (default 10)
        metric: Metric to rank by (pnl_pct, per_trade_pnl, win_rate)
        min_trades: Minimum trades to qualify (default 50)
    """
    limit = request.args.get('limit', 10, type=int)
    metric = request.args.get('metric', 'pnl_pct')
    min_trades = request.args.get('min_trades', 50, type=int)

    valid_metrics = {'pnl_pct', 'per_trade_pnl', 'win_rate'}
    if metric not in valid_metrics:
        metric = 'pnl_pct'

    conn = get_experiments_db()
    conn.row_factory = dict_factory
    cursor = conn.cursor()

    cursor.execute(f"""
        SELECT
            run_name, pnl_pct, win_rate, trades,
            per_trade_pnl, cycles, env_vars, timestamp
        FROM experiments
        WHERE trades >= ?
        ORDER BY {metric} DESC
        LIMIT ?
    """, (min_trades, limit))

    results = cursor.fetchall()
    conn.close()

    # Parse env_vars and extract key differences
    all_env_keys = set()
    for r in results:
        if r.get('env_vars'):
            try:
                r['env_vars'] = json.loads(r['env_vars'])
                all_env_keys.update(r['env_vars'].keys())
            except:
                r['env_vars'] = {}

    return jsonify({
        "best_experiments": results,
        "ranked_by": metric,
        "min_trades_filter": min_trades,
        "config_keys_observed": list(all_env_keys)
    })


@agent_bp.route('/experiments/compare', methods=['GET', 'POST'])
def compare_experiments():
    """
    Compare multiple experiments side-by-side.

    GET: ?runs=run1,run2,run3
    POST: {"runs": ["run1", "run2", "run3"]}
    """
    if request.method == 'POST':
        data = request.get_json() or {}
        run_names = data.get('runs', [])
    else:
        run_names = request.args.get('runs', '').split(',')

    run_names = [r.strip() for r in run_names if r.strip()]

    if not run_names:
        return jsonify({"error": "No runs specified"}), 400

    conn = get_experiments_db()
    conn.row_factory = dict_factory
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

    # Parse env_vars and find differences
    all_keys = set()
    parsed_results = []
    for r in results:
        env = {}
        if r.get('env_vars'):
            try:
                env = json.loads(r['env_vars'])
            except:
                pass
        r['env_vars'] = env
        all_keys.update(env.keys())
        parsed_results.append(r)

    # Build config comparison matrix
    config_matrix = {}
    for key in sorted(all_keys):
        config_matrix[key] = {}
        for r in parsed_results:
            config_matrix[key][r['run_name']] = r['env_vars'].get(key)

    # Find what's different vs same
    differences = {}
    common = {}
    for key, values in config_matrix.items():
        unique_values = set(str(v) for v in values.values())
        if len(unique_values) > 1:
            differences[key] = values
        else:
            common[key] = list(values.values())[0] if values else None

    return jsonify({
        "runs": parsed_results,
        "config_differences": differences,
        "config_common": common,
        "all_config_keys": sorted(all_keys)
    })


@agent_bp.route('/trades/<run_id>', methods=['GET'])
def get_trades_for_run(run_id: str):
    """
    Get all trades for a specific run.

    Path Parameters:
        run_id: The run name/ID

    Query Parameters:
        limit: Max trades (default 500)
        offset: Pagination offset
    """
    limit = request.args.get('limit', 500, type=int)
    offset = request.args.get('offset', 0, type=int)

    conn = get_paper_trading_db()
    conn.row_factory = dict_factory
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            id, timestamp, option_type, strike_price,
            premium_paid, exit_price, profit_loss,
            exit_reason, ml_confidence, ml_prediction,
            status, exit_timestamp
        FROM trades
        WHERE run_id = ?
        ORDER BY timestamp DESC
        LIMIT ? OFFSET ?
    """, (run_id, limit, offset))

    trades = cursor.fetchall()

    # Get stats
    cursor.execute("""
        SELECT
            COUNT(*) as total_trades,
            SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN profit_loss <= 0 THEN 1 ELSE 0 END) as losses,
            SUM(profit_loss) as total_pnl,
            AVG(profit_loss) as avg_pnl
        FROM trades
        WHERE run_id = ?
    """, (run_id,))
    stats = cursor.fetchone()

    conn.close()

    return jsonify({
        "run_id": run_id,
        "trades": trades,
        "stats": stats,
        "limit": limit,
        "offset": offset
    })


@agent_bp.route('/config/<run_id>', methods=['GET'])
def get_config_for_run(run_id: str):
    """
    Get configuration/environment variables for a specific run.
    """
    conn = get_experiments_db()
    conn.row_factory = dict_factory
    cursor = conn.cursor()

    cursor.execute("""
        SELECT run_name, env_vars, config_hash, notes
        FROM experiments
        WHERE run_name = ?
    """, (run_id,))

    result = cursor.fetchone()
    conn.close()

    if not result:
        return jsonify({"error": f"Run '{run_id}' not found"}), 404

    env_vars = {}
    if result.get('env_vars'):
        try:
            env_vars = json.loads(result['env_vars'])
        except:
            pass

    return jsonify({
        "run_id": run_id,
        "env_vars": env_vars,
        "config_hash": result.get('config_hash'),
        "notes": result.get('notes')
    })


@agent_bp.route('/suggest', methods=['GET'])
def get_suggestions_context():
    """
    Get AI-friendly context for generating experiment suggestions.

    Returns comprehensive data about what's been tried, what worked,
    and what patterns have been observed.
    """
    conn = get_experiments_db()
    conn.row_factory = dict_factory
    cursor = conn.cursor()

    # Get top 10 best runs
    cursor.execute("""
        SELECT run_name, pnl_pct, win_rate, trades, per_trade_pnl, env_vars
        FROM experiments
        WHERE trades >= 50
        ORDER BY pnl_pct DESC
        LIMIT 10
    """)
    best_runs = cursor.fetchall()

    # Get worst 10 runs (to learn what NOT to do)
    cursor.execute("""
        SELECT run_name, pnl_pct, win_rate, trades, per_trade_pnl, env_vars
        FROM experiments
        WHERE trades >= 50
        ORDER BY pnl_pct ASC
        LIMIT 10
    """)
    worst_runs = cursor.fetchall()

    # Get all tested ideas from idea_queue
    ideas_tested = []
    idea_queue_path = Path(COLLAB_DIR) / 'idea_queue.json'
    if idea_queue_path.exists():
        try:
            with open(idea_queue_path) as f:
                idea_data = json.load(f)
            ideas_tested = [
                {
                    'id': i.get('id'),
                    'title': i.get('title'),
                    'status': i.get('status'),
                    'category': i.get('category'),
                    'env_vars': i.get('env_vars'),
                    'last_result': i.get('last_result')
                }
                for i in idea_data.get('ideas', [])
            ]
        except Exception as e:
            pass

    conn.close()

    # Parse env_vars for analysis
    for runs in [best_runs, worst_runs]:
        for r in runs:
            if r.get('env_vars'):
                try:
                    r['env_vars'] = json.loads(r['env_vars'])
                except:
                    r['env_vars'] = {}

    # Extract patterns from best vs worst
    patterns = analyze_patterns(best_runs, worst_runs)

    return jsonify({
        "context": "Options trading bot optimization - analyze these experiments to suggest new configurations",
        "goal": "Maximize P&L while maintaining reasonable win rate (>30%)",
        "best_runs": best_runs,
        "worst_runs": worst_runs,
        "patterns_observed": patterns,
        "ideas_already_tested": ideas_tested,
        "untested_directions": [
            "Combine best exit params with different entry filters",
            "Test time-of-day restrictions",
            "Try momentum-based position sizing",
            "Experiment with VIX-adaptive thresholds"
        ],
        "key_parameters_to_tune": {
            "HARD_STOP_LOSS_PCT": "Stop loss percentage (default 8)",
            "HARD_TAKE_PROFIT_PCT": "Take profit percentage (default 12)",
            "HMM_STRONG_BULLISH": "HMM threshold for bullish (default 0.65)",
            "HMM_STRONG_BEARISH": "HMM threshold for bearish (default 0.35)",
            "HMM_MIN_CONFIDENCE": "Minimum HMM confidence (default 0.60)",
            "PREDICTOR_ARCH": "v2_slim_bayesian, v3_multi_horizon",
            "TEMPORAL_ENCODER": "tcn, transformer, lstm",
            "LOAD_PRETRAINED": "1 to load pretrained model"
        }
    })


def analyze_patterns(best_runs: List[Dict], worst_runs: List[Dict]) -> List[str]:
    """Analyze patterns between best and worst runs"""
    patterns = []

    # Extract numeric params from best runs
    best_stops = []
    best_tps = []
    worst_stops = []
    worst_tps = []

    for r in best_runs:
        env = r.get('env_vars', {})
        if 'HARD_STOP_LOSS_PCT' in env:
            try:
                best_stops.append(float(env['HARD_STOP_LOSS_PCT']))
            except:
                pass
        if 'HARD_TAKE_PROFIT_PCT' in env:
            try:
                best_tps.append(float(env['HARD_TAKE_PROFIT_PCT']))
            except:
                pass

    for r in worst_runs:
        env = r.get('env_vars', {})
        if 'HARD_STOP_LOSS_PCT' in env:
            try:
                worst_stops.append(float(env['HARD_STOP_LOSS_PCT']))
            except:
                pass
        if 'HARD_TAKE_PROFIT_PCT' in env:
            try:
                worst_tps.append(float(env['HARD_TAKE_PROFIT_PCT']))
            except:
                pass

    if best_stops and worst_stops:
        avg_best_stop = sum(best_stops) / len(best_stops)
        avg_worst_stop = sum(worst_stops) / len(worst_stops)
        if avg_best_stop > avg_worst_stop:
            patterns.append(f"Wider stop losses perform better (best avg: {avg_best_stop:.1f}% vs worst avg: {avg_worst_stop:.1f}%)")
        elif avg_best_stop < avg_worst_stop:
            patterns.append(f"Tighter stop losses perform better (best avg: {avg_best_stop:.1f}% vs worst avg: {avg_worst_stop:.1f}%)")

    if best_tps and worst_tps:
        avg_best_tp = sum(best_tps) / len(best_tps)
        avg_worst_tp = sum(worst_tps) / len(worst_tps)
        if avg_best_tp > avg_worst_tp:
            patterns.append(f"Higher take profits perform better (best avg: {avg_best_tp:.1f}% vs worst avg: {avg_worst_tp:.1f}%)")

    # Check for pretrained usage
    best_pretrained = sum(1 for r in best_runs if r.get('env_vars', {}).get('LOAD_PRETRAINED') == '1')
    worst_pretrained = sum(1 for r in worst_runs if r.get('env_vars', {}).get('LOAD_PRETRAINED') == '1')
    if best_pretrained > worst_pretrained:
        patterns.append("Using pretrained model (LOAD_PRETRAINED=1) correlates with better performance")

    if not patterns:
        patterns.append("Insufficient data to detect clear patterns - more experiments needed")

    return patterns


@agent_bp.route('/ideas', methods=['POST'])
def submit_idea():
    """
    Submit a new experiment idea for testing.

    Request Body:
    {
        "title": "Test wider stop loss",
        "hypothesis": "Wider stops may reduce premature exits",
        "category": "exit_strategy",
        "env_vars": {"HARD_STOP_LOSS_PCT": "12"},
        "source": "gemini"  # or "codex", "claude", etc.
    }
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    required = ['title', 'hypothesis', 'env_vars']
    for field in required:
        if field not in data:
            return jsonify({"error": f"Missing required field: {field}"}), 400

    idea_queue_path = Path(COLLAB_DIR) / 'idea_queue.json'

    # Load existing queue
    if idea_queue_path.exists():
        with open(idea_queue_path) as f:
            queue_data = json.load(f)
    else:
        queue_data = {"ideas": []}

    # Generate new idea ID
    existing_ids = [i.get('id', '') for i in queue_data.get('ideas', [])]
    max_id = 0
    for id_str in existing_ids:
        if id_str.startswith('IDEA-'):
            try:
                num = int(id_str.split('-')[1])
                max_id = max(max_id, num)
            except:
                pass
    new_id = f"IDEA-{max_id + 1:03d}"

    new_idea = {
        "id": new_id,
        "source": data.get('source', 'api'),
        "timestamp": datetime.now().isoformat(),
        "category": data.get('category', 'general'),
        "title": data['title'],
        "description": data.get('description', ''),
        "hypothesis": data['hypothesis'],
        "priority": data.get('priority', 'medium'),
        "status": "pending",
        "env_vars": data['env_vars']
    }

    queue_data['ideas'].append(new_idea)

    # Save updated queue
    idea_queue_path.parent.mkdir(parents=True, exist_ok=True)
    with open(idea_queue_path, 'w') as f:
        json.dump(queue_data, f, indent=2)

    return jsonify({
        "success": True,
        "idea_id": new_id,
        "message": f"Idea '{new_id}' queued for testing"
    })


@agent_bp.route('/status', methods=['GET'])
def get_status():
    """
    Get current running experiments and system status.
    """
    conn = get_experiments_db()
    conn.row_factory = dict_factory
    cursor = conn.cursor()

    # Get running experiments
    cursor.execute("""
        SELECT
            run_name, started_at, pid, machine_id,
            last_heartbeat, current_cycle, current_pnl, target_cycles
        FROM running_experiments
        WHERE last_heartbeat > datetime('now', '-5 minutes')
    """)
    running = cursor.fetchall()

    # Get recent completions
    cursor.execute("""
        SELECT run_name, pnl_pct, win_rate, trades, timestamp
        FROM experiments
        ORDER BY timestamp DESC
        LIMIT 5
    """)
    recent = cursor.fetchall()

    conn.close()

    # Check heartbeat file too
    heartbeat_path = Path('data/running_experiment.json')
    heartbeat_data = None
    if heartbeat_path.exists():
        try:
            with open(heartbeat_path) as f:
                heartbeat_data = json.load(f)
        except:
            pass

    return jsonify({
        "running_experiments": running,
        "heartbeat_file": heartbeat_data,
        "recent_completions": recent,
        "system_time": datetime.now().isoformat()
    })


# Error handlers
@agent_bp.errorhandler(400)
def bad_request(e):
    return jsonify({"error": "Bad request", "message": str(e)}), 400


@agent_bp.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not found", "message": str(e)}), 404


@agent_bp.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error", "message": str(e)}), 500
