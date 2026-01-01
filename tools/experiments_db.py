#!/usr/bin/env python3
"""
Experiments Database - Track and query all model runs

Usage:
    python tools/experiments_db.py scan      # Scan models/ and import all runs
    python tools/experiments_db.py leaderboard [--limit 20]
    python tools/experiments_db.py search --win-rate-min 0.40
    python tools/experiments_db.py show <run_name>
"""

import sqlite3
import json
import os
from pathlib import Path
from datetime import datetime
import argparse

DB_PATH = "data/experiments.db"
MODELS_DIR = "models"

def init_db():
    """Create the experiments database schema."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_name TEXT UNIQUE NOT NULL,
            timestamp TEXT,
            start_date TEXT,
            end_date TEXT,
            initial_balance REAL,
            final_balance REAL,
            pnl REAL,
            pnl_pct REAL,
            cycles INTEGER,
            signals INTEGER,
            trades INTEGER,
            win_rate REAL,
            wins INTEGER,
            losses INTEGER,
            per_trade_pnl REAL,
            training_time_seconds REAL,
            config_hash TEXT,
            env_vars TEXT,
            notes TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Create indexes for common queries
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_pnl_pct ON experiments(pnl_pct DESC)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_win_rate ON experiments(win_rate DESC)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON experiments(timestamp DESC)")

    conn.commit()
    conn.close()
    print(f"[OK] Database initialized at {DB_PATH}")

def parse_summary_txt(summary_path):
    """Parse SUMMARY.txt file for additional metrics."""
    metrics = {}
    try:
        with open(summary_path, 'r') as f:
            content = f.read()

        for line in content.split('\n'):
            if 'Win Rate:' in line:
                val = line.split(':')[1].strip().replace('%', '')
                metrics['win_rate'] = float(val) / 100 if float(val) > 1 else float(val)
            elif 'Wins:' in line and 'Losses' not in line:
                metrics['wins'] = int(line.split(':')[1].strip())
            elif 'Losses:' in line:
                metrics['losses'] = int(line.split(':')[1].strip())
    except Exception as e:
        pass
    return metrics

def scan_models():
    """Scan models/ directory and import all run_info.json files."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    models_path = Path(MODELS_DIR)
    imported = 0
    skipped = 0

    for run_dir in models_path.iterdir():
        if not run_dir.is_dir():
            continue

        run_info_path = run_dir / "run_info.json"
        summary_path = run_dir / "SUMMARY.txt"

        if not run_info_path.exists():
            continue

        try:
            with open(run_info_path, 'r') as f:
                info = json.load(f)

            run_name = run_dir.name

            # Parse SUMMARY.txt for additional metrics
            summary_metrics = parse_summary_txt(summary_path) if summary_path.exists() else {}

            # Calculate per-trade P&L
            trades = info.get('trades', 0)
            pnl = info.get('pnl', 0)
            per_trade_pnl = pnl / trades if trades > 0 else 0

            # Check if already exists
            cursor.execute("SELECT id FROM experiments WHERE run_name = ?", (run_name,))
            if cursor.fetchone():
                skipped += 1
                continue

            cursor.execute("""
                INSERT INTO experiments (
                    run_name, timestamp, start_date, end_date,
                    initial_balance, final_balance, pnl, pnl_pct,
                    cycles, signals, trades, win_rate, wins, losses,
                    per_trade_pnl, training_time_seconds
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                run_name,
                info.get('timestamp'),
                info.get('start_date'),
                info.get('end_date'),
                info.get('initial_balance'),
                info.get('final_balance'),
                pnl,
                info.get('pnl_pct'),
                info.get('cycles'),
                info.get('signals'),
                trades,
                summary_metrics.get('win_rate'),
                summary_metrics.get('wins'),
                summary_metrics.get('losses'),
                per_trade_pnl,
                info.get('training_time_seconds')
            ))
            imported += 1

        except Exception as e:
            print(f"[WARN] Failed to import {run_dir.name}: {e}")
            continue

    conn.commit()
    conn.close()
    print(f"[OK] Imported {imported} runs, skipped {skipped} existing")

def leaderboard(limit=20, metric='pnl_pct'):
    """Show top runs by metric."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    order_col = {
        'pnl': 'pnl DESC',
        'pnl_pct': 'pnl_pct DESC',
        'win_rate': 'win_rate DESC',
        'per_trade': 'per_trade_pnl DESC'
    }.get(metric, 'pnl_pct DESC')

    cursor.execute(f"""
        SELECT run_name, timestamp, pnl_pct, win_rate, trades, per_trade_pnl, cycles
        FROM experiments
        WHERE trades > 10
        ORDER BY {order_col}
        LIMIT ?
    """, (limit,))

    rows = cursor.fetchall()
    conn.close()

    print("\n" + "="*100)
    print(f"{'Rank':<5} {'Run Name':<40} {'P&L %':<12} {'Win Rate':<10} {'Trades':<8} {'$/Trade':<12} {'Cycles':<8}")
    print("="*100)

    for i, row in enumerate(rows, 1):
        pnl = f"{row['pnl_pct']:.2f}%" if row['pnl_pct'] else "N/A"
        wr = f"{row['win_rate']*100:.1f}%" if row['win_rate'] else "N/A"
        ppt = f"${row['per_trade_pnl']:.2f}" if row['per_trade_pnl'] else "N/A"
        print(f"{i:<5} {row['run_name']:<40} {pnl:<12} {wr:<10} {row['trades'] or 0:<8} {ppt:<12} {row['cycles'] or 0:<8}")

    print("="*100)

def search(win_rate_min=None, pnl_min=None, date_after=None):
    """Search experiments by criteria."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    conditions = ["trades > 10"]
    params = []

    if win_rate_min:
        conditions.append("win_rate >= ?")
        params.append(win_rate_min)
    if pnl_min:
        conditions.append("pnl_pct >= ?")
        params.append(pnl_min)
    if date_after:
        conditions.append("timestamp >= ?")
        params.append(date_after)

    query = f"""
        SELECT run_name, timestamp, pnl_pct, win_rate, trades
        FROM experiments
        WHERE {' AND '.join(conditions)}
        ORDER BY pnl_pct DESC
        LIMIT 50
    """

    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()

    print(f"\nFound {len(rows)} matching experiments:")
    for row in rows:
        pnl = f"{row['pnl_pct']:.2f}%" if row['pnl_pct'] else "N/A"
        wr = f"{row['win_rate']*100:.1f}%" if row['win_rate'] else "N/A"
        print(f"  {row['run_name']}: P&L={pnl}, WR={wr}, Trades={row['trades']}")

def show(run_name):
    """Show details for a specific run."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM experiments WHERE run_name = ?", (run_name,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        print(f"Run '{run_name}' not found")
        return

    print(f"\n{'='*60}")
    print(f"Run: {row['run_name']}")
    print(f"{'='*60}")
    print(f"Timestamp:      {row['timestamp']}")
    print(f"Date Range:     {row['start_date']} to {row['end_date']}")
    print(f"Cycles:         {row['cycles']}")
    print(f"Signals:        {row['signals']}")
    print(f"Trades:         {row['trades']}")
    print(f"")
    print(f"Initial:        ${row['initial_balance']:.2f}")
    print(f"Final:          ${row['final_balance']:.2f}")
    print(f"P&L:            ${row['pnl']:.2f} ({row['pnl_pct']:.2f}%)")
    print(f"Per-Trade:      ${row['per_trade_pnl']:.2f}")
    print(f"")
    print(f"Win Rate:       {row['win_rate']*100:.1f}%" if row['win_rate'] else "Win Rate: N/A")
    print(f"Wins/Losses:    {row['wins']}/{row['losses']}")
    print(f"Training Time:  {row['training_time_seconds']:.1f}s" if row['training_time_seconds'] else "")

def main():
    parser = argparse.ArgumentParser(description='Experiments Database')
    subparsers = parser.add_subparsers(dest='command')

    # scan command
    subparsers.add_parser('scan', help='Scan models/ and import all runs')

    # leaderboard command
    lb_parser = subparsers.add_parser('leaderboard', help='Show leaderboard')
    lb_parser.add_argument('--limit', type=int, default=20)
    lb_parser.add_argument('--metric', choices=['pnl', 'pnl_pct', 'win_rate', 'per_trade'], default='pnl_pct')

    # search command
    search_parser = subparsers.add_parser('search', help='Search experiments')
    search_parser.add_argument('--win-rate-min', type=float)
    search_parser.add_argument('--pnl-min', type=float)
    search_parser.add_argument('--date-after', type=str)

    # show command
    show_parser = subparsers.add_parser('show', help='Show run details')
    show_parser.add_argument('run_name', type=str)

    args = parser.parse_args()

    if args.command == 'scan':
        scan_models()
    elif args.command == 'leaderboard':
        leaderboard(limit=args.limit, metric=args.metric)
    elif args.command == 'search':
        search(win_rate_min=args.win_rate_min, pnl_min=args.pnl_min, date_after=args.date_after)
    elif args.command == 'show':
        show(args.run_name)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
