#!/usr/bin/env python3
"""
Trade Browser API - Deep Trade Analysis Endpoints

Provides endpoints for browsing, filtering, and analyzing trades across all runs.
Can be integrated into existing dashboards as a blueprint.

Endpoints:
    GET  /api/trades                   - Paginated trade list with filtering
    GET  /api/trades/<trade_id>        - Single trade details
    GET  /api/trades/by-run/<run_id>   - All trades for a specific run
    GET  /api/trades/aggregations      - Trade statistics by various dimensions
    GET  /api/trades/pnl-curve         - Cumulative P&L curve data
    GET  /api/trades/runs              - List of runs with trade counts
"""

import sqlite3
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from flask import Blueprint, jsonify, request

# Create blueprint
trades_bp = Blueprint('trades', __name__, url_prefix='/api/trades')

# Database paths
PAPER_TRADING_DB = os.environ.get('PAPER_TRADING_DB', 'data/paper_trading.db')
EXPERIMENTS_DB = os.environ.get('EXPERIMENTS_DB', 'data/experiments.db')
MODELS_DIR = os.environ.get('MODELS_DIR', 'models')


def get_trades_db():
    """Get paper trading database connection"""
    conn = sqlite3.connect(PAPER_TRADING_DB)
    conn.row_factory = lambda c, r: {col[0]: r[idx] for idx, col in enumerate(c.description)}
    return conn


def get_experiments_db():
    """Get experiments database connection"""
    conn = sqlite3.connect(EXPERIMENTS_DB)
    conn.row_factory = lambda c, r: {col[0]: r[idx] for idx, col in enumerate(c.description)}
    return conn


@trades_bp.route('', methods=['GET'])
def get_trades():
    """
    Get paginated trades with filtering.

    Query Parameters:
        page: Page number (default 1)
        limit: Results per page (default 50, max 200)
        run_id: Filter by run ID
        start_date: Start date filter (YYYY-MM-DD)
        end_date: End date filter (YYYY-MM-DD)
        option_type: CALL or PUT
        pnl_min: Minimum P&L
        pnl_max: Maximum P&L
        exit_reason: Filter by exit reason (supports LIKE patterns)
        status: Trade status filter
        min_confidence: Minimum ML confidence
        sort: Sort field (timestamp, profit_loss, ml_confidence)
        order: asc or desc (default desc)
    """
    # Parse parameters
    page = max(1, request.args.get('page', 1, type=int))
    limit = min(200, max(1, request.args.get('limit', 50, type=int)))
    offset = (page - 1) * limit

    run_id = request.args.get('run_id', '')
    start_date = request.args.get('start_date', '')
    end_date = request.args.get('end_date', '')
    option_type = request.args.get('option_type', '')
    pnl_min = request.args.get('pnl_min', type=float)
    pnl_max = request.args.get('pnl_max', type=float)
    exit_reason = request.args.get('exit_reason', '')
    status = request.args.get('status', '')
    min_confidence = request.args.get('min_confidence', type=float)
    sort_by = request.args.get('sort', 'timestamp')
    order = request.args.get('order', 'desc').upper()

    # Validate sort
    valid_sorts = {'timestamp', 'profit_loss', 'ml_confidence', 'strike_price'}
    if sort_by not in valid_sorts:
        sort_by = 'timestamp'
    if order not in ('ASC', 'DESC'):
        order = 'DESC'

    conn = get_trades_db()
    cursor = conn.cursor()

    # Build WHERE clause
    conditions = ["1=1"]
    params = []

    if run_id:
        conditions.append("run_id = ?")
        params.append(run_id)

    if start_date:
        conditions.append("timestamp >= ?")
        params.append(start_date)

    if end_date:
        conditions.append("timestamp <= ?")
        params.append(end_date + " 23:59:59")

    if option_type:
        conditions.append("option_type = ?")
        params.append(option_type.upper())

    if pnl_min is not None:
        conditions.append("profit_loss >= ?")
        params.append(pnl_min)

    if pnl_max is not None:
        conditions.append("profit_loss <= ?")
        params.append(pnl_max)

    if exit_reason:
        conditions.append("exit_reason LIKE ?")
        params.append(f"%{exit_reason}%")

    if status:
        conditions.append("status = ?")
        params.append(status)

    if min_confidence is not None:
        conditions.append("ml_confidence >= ?")
        params.append(min_confidence)

    where_clause = " AND ".join(conditions)

    # Get total count
    cursor.execute(f"SELECT COUNT(*) as total FROM trades WHERE {where_clause}", params)
    total = cursor.fetchone()['total']

    # Get trades
    query = f"""
        SELECT
            id, timestamp, symbol, option_type, strike_price,
            premium_paid, quantity, entry_price, exit_price,
            exit_timestamp, status, profit_loss, stop_loss,
            take_profit, ml_confidence, ml_prediction,
            exit_reason, run_id, is_real_trade,
            CASE WHEN premium_paid > 0 THEN
                ROUND((profit_loss / premium_paid) * 100, 2)
            ELSE 0 END as pnl_pct,
            CASE WHEN exit_timestamp IS NOT NULL THEN
                ROUND((julianday(exit_timestamp) - julianday(timestamp)) * 24 * 60, 1)
            ELSE NULL END as hold_minutes
        FROM trades
        WHERE {where_clause}
        ORDER BY {sort_by} {order}
        LIMIT ? OFFSET ?
    """
    cursor.execute(query, params + [limit, offset])
    trades = cursor.fetchall()

    conn.close()

    return jsonify({
        "success": True,
        "trades": trades,
        "pagination": {
            "page": page,
            "limit": limit,
            "total": total,
            "total_pages": (total + limit - 1) // limit if limit > 0 else 0
        },
        "filters": {
            "run_id": run_id,
            "start_date": start_date,
            "end_date": end_date,
            "option_type": option_type,
            "pnl_min": pnl_min,
            "pnl_max": pnl_max,
            "exit_reason": exit_reason,
            "status": status,
            "min_confidence": min_confidence,
            "sort": sort_by,
            "order": order
        }
    })


@trades_bp.route('/<trade_id>', methods=['GET'])
def get_trade_detail(trade_id: str):
    """Get detailed information for a single trade."""
    conn = get_trades_db()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT * FROM trades WHERE id = ?
    """, (trade_id,))
    trade = cursor.fetchone()

    if not trade:
        conn.close()
        return jsonify({"error": f"Trade '{trade_id}' not found"}), 404

    conn.close()

    # Calculate additional metrics
    if trade.get('premium_paid') and trade['premium_paid'] > 0:
        trade['pnl_pct'] = round((trade['profit_loss'] or 0) / trade['premium_paid'] * 100, 2)

    if trade.get('timestamp') and trade.get('exit_timestamp'):
        try:
            entry = datetime.fromisoformat(trade['timestamp'].replace(' ', 'T'))
            exit_dt = datetime.fromisoformat(trade['exit_timestamp'].replace(' ', 'T'))
            trade['hold_minutes'] = round((exit_dt - entry).total_seconds() / 60, 1)
        except:
            pass

    # Try to get decision context from decision_records.jsonl
    decision_context = None
    if trade.get('run_id'):
        decision_path = Path(MODELS_DIR) / trade['run_id'] / 'state' / 'decision_records.jsonl'
        if decision_path.exists():
            trade['decision_records_available'] = True
            # Note: Full context loading would require indexing - placeholder for now
        else:
            trade['decision_records_available'] = False

    return jsonify({
        "success": True,
        "trade": trade,
        "decision_context": decision_context
    })


@trades_bp.route('/by-run/<run_id>', methods=['GET'])
def get_trades_by_run(run_id: str):
    """Get all trades for a specific run with summary stats."""
    limit = request.args.get('limit', 500, type=int)
    offset = request.args.get('offset', 0, type=int)

    conn = get_trades_db()
    cursor = conn.cursor()

    # Get trades
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

    # Get summary stats
    cursor.execute("""
        SELECT
            COUNT(*) as total_trades,
            SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN profit_loss <= 0 THEN 1 ELSE 0 END) as losses,
            SUM(profit_loss) as total_pnl,
            AVG(profit_loss) as avg_pnl,
            MIN(profit_loss) as worst_trade,
            MAX(profit_loss) as best_trade,
            AVG(ml_confidence) as avg_confidence
        FROM trades
        WHERE run_id = ?
    """, (run_id,))
    stats = cursor.fetchone()

    conn.close()

    # Calculate win rate
    if stats['total_trades'] and stats['total_trades'] > 0:
        stats['win_rate'] = round(stats['wins'] / stats['total_trades'] * 100, 1)
    else:
        stats['win_rate'] = 0

    return jsonify({
        "success": True,
        "run_id": run_id,
        "trades": trades,
        "stats": stats,
        "pagination": {
            "limit": limit,
            "offset": offset,
            "returned": len(trades)
        }
    })


@trades_bp.route('/aggregations', methods=['GET'])
def get_aggregations():
    """
    Get trade statistics aggregated by various dimensions.

    Query Parameters:
        run_id: Filter by run ID
        group_by: Dimension to group by (exit_reason, option_type, hour, day, run_id)
    """
    run_id = request.args.get('run_id', '')
    group_by = request.args.get('group_by', 'exit_reason')

    valid_groups = {'exit_reason', 'option_type', 'hour', 'day', 'run_id'}
    if group_by not in valid_groups:
        group_by = 'exit_reason'

    conn = get_trades_db()
    cursor = conn.cursor()

    # Base WHERE clause
    where = "status NOT IN ('OPEN', 'FILLED', 'PENDING')"
    params = []
    if run_id:
        where += " AND run_id = ?"
        params.append(run_id)

    # Build GROUP BY based on dimension
    if group_by == 'exit_reason':
        group_col = "COALESCE(exit_reason, 'Unknown')"
        group_alias = "exit_reason"
    elif group_by == 'option_type':
        group_col = "option_type"
        group_alias = "option_type"
    elif group_by == 'hour':
        group_col = "strftime('%H', timestamp)"
        group_alias = "hour"
    elif group_by == 'day':
        group_col = "strftime('%w', timestamp)"  # 0=Sunday
        group_alias = "day_of_week"
    elif group_by == 'run_id':
        group_col = "COALESCE(run_id, 'Unknown')"
        group_alias = "run_id"

    query = f"""
        SELECT
            {group_col} as {group_alias},
            COUNT(*) as trade_count,
            SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN profit_loss <= 0 THEN 1 ELSE 0 END) as losses,
            SUM(profit_loss) as total_pnl,
            AVG(profit_loss) as avg_pnl,
            ROUND(CAST(SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END) AS FLOAT) /
                  NULLIF(COUNT(*), 0) * 100, 1) as win_rate
        FROM trades
        WHERE {where}
        GROUP BY {group_col}
        ORDER BY trade_count DESC
    """
    cursor.execute(query, params)
    aggregations = cursor.fetchall()

    # Get overall stats
    cursor.execute(f"""
        SELECT
            COUNT(*) as total_trades,
            SUM(profit_loss) as total_pnl,
            AVG(profit_loss) as avg_pnl,
            ROUND(CAST(SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END) AS FLOAT) /
                  NULLIF(COUNT(*), 0) * 100, 1) as win_rate
        FROM trades
        WHERE {where}
    """, params)
    summary = cursor.fetchone()

    conn.close()

    # Add day names if grouping by day
    if group_by == 'day':
        day_names = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
        for agg in aggregations:
            try:
                agg['day_name'] = day_names[int(agg['day_of_week'])]
            except:
                pass

    return jsonify({
        "success": True,
        "group_by": group_by,
        "aggregations": aggregations,
        "summary": summary,
        "run_id_filter": run_id or None
    })


@trades_bp.route('/pnl-curve', methods=['GET'])
def get_pnl_curve():
    """
    Get cumulative P&L curve data for charting.

    Query Parameters:
        run_id: Filter by run ID
        initial_balance: Starting balance (default 5000)
    """
    run_id = request.args.get('run_id', '')
    initial_balance = request.args.get('initial_balance', 5000, type=float)

    conn = get_trades_db()
    cursor = conn.cursor()

    where = "status NOT IN ('OPEN', 'FILLED', 'PENDING') AND profit_loss IS NOT NULL"
    params = []
    if run_id:
        where += " AND run_id = ?"
        params.append(run_id)

    cursor.execute(f"""
        SELECT
            timestamp,
            profit_loss,
            option_type,
            exit_reason
        FROM trades
        WHERE {where}
        ORDER BY timestamp ASC
    """, params)
    trades = cursor.fetchall()
    conn.close()

    # Build cumulative curve
    curve = []
    cumulative_pnl = 0
    balance = initial_balance
    max_balance = initial_balance
    max_drawdown = 0

    for t in trades:
        cumulative_pnl += t['profit_loss'] or 0
        balance = initial_balance + cumulative_pnl

        if balance > max_balance:
            max_balance = balance

        drawdown = (max_balance - balance) / max_balance * 100 if max_balance > 0 else 0
        if drawdown > max_drawdown:
            max_drawdown = drawdown

        curve.append({
            "timestamp": t['timestamp'],
            "pnl": round(t['profit_loss'] or 0, 2),
            "cumulative_pnl": round(cumulative_pnl, 2),
            "balance": round(balance, 2),
            "option_type": t['option_type']
        })

    return jsonify({
        "success": True,
        "curve": curve,
        "stats": {
            "initial_balance": initial_balance,
            "final_balance": round(balance, 2),
            "total_pnl": round(cumulative_pnl, 2),
            "total_pnl_pct": round(cumulative_pnl / initial_balance * 100, 2) if initial_balance > 0 else 0,
            "max_drawdown_pct": round(max_drawdown, 2),
            "total_trades": len(curve)
        },
        "run_id_filter": run_id or None
    })


@trades_bp.route('/runs', methods=['GET'])
def get_runs_with_trades():
    """Get list of runs that have trades, with trade counts."""
    conn = get_trades_db()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            COALESCE(run_id, 'Unknown') as run_id,
            COUNT(*) as trade_count,
            SUM(profit_loss) as total_pnl,
            MIN(timestamp) as first_trade,
            MAX(timestamp) as last_trade,
            ROUND(CAST(SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END) AS FLOAT) /
                  NULLIF(COUNT(*), 0) * 100, 1) as win_rate
        FROM trades
        WHERE status NOT IN ('OPEN', 'FILLED', 'PENDING')
        GROUP BY run_id
        ORDER BY last_trade DESC
    """)
    runs = cursor.fetchall()
    conn.close()

    return jsonify({
        "success": True,
        "runs": runs,
        "total_runs": len(runs)
    })


@trades_bp.route('/exit-reasons', methods=['GET'])
def get_exit_reasons():
    """Get list of all unique exit reasons for filtering."""
    conn = get_trades_db()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT DISTINCT exit_reason, COUNT(*) as count
        FROM trades
        WHERE exit_reason IS NOT NULL
        GROUP BY exit_reason
        ORDER BY count DESC
    """)
    reasons = cursor.fetchall()
    conn.close()

    return jsonify({
        "success": True,
        "exit_reasons": reasons
    })
