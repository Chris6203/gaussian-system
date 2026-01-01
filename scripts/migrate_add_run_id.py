#!/usr/bin/env python3
"""
Database Migration: Add run_id to trades and create running_experiments table

Usage:
    python scripts/migrate_add_run_id.py
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime

PAPER_DB = "data/paper_trading.db"
EXPERIMENTS_DB = "data/experiments.db"
MODELS_DIR = "models"


def migrate_paper_trading_db():
    """Add run_id column and indexes to paper_trading.db"""
    print(f"[*] Migrating {PAPER_DB}...")

    conn = sqlite3.connect(PAPER_DB)
    cursor = conn.cursor()

    # Check if run_id column already exists
    cursor.execute("PRAGMA table_info(trades)")
    columns = [col[1] for col in cursor.fetchall()]

    if 'run_id' not in columns:
        print("    Adding run_id column to trades table...")
        cursor.execute("ALTER TABLE trades ADD COLUMN run_id TEXT")
        conn.commit()
        print("    [OK] run_id column added")
    else:
        print("    [SKIP] run_id column already exists")

    # Create indexes if they don't exist
    indexes = [
        ("idx_trades_run_id", "CREATE INDEX IF NOT EXISTS idx_trades_run_id ON trades(run_id)"),
        ("idx_trades_timestamp", "CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp)"),
        ("idx_trades_exit_reason", "CREATE INDEX IF NOT EXISTS idx_trades_exit_reason ON trades(exit_reason)"),
        ("idx_trades_option_type", "CREATE INDEX IF NOT EXISTS idx_trades_option_type ON trades(option_type)"),
        ("idx_trades_profit_loss", "CREATE INDEX IF NOT EXISTS idx_trades_profit_loss ON trades(profit_loss)"),
    ]

    for idx_name, sql in indexes:
        print(f"    Creating index {idx_name}...")
        cursor.execute(sql)

    conn.commit()
    conn.close()
    print(f"[OK] {PAPER_DB} migration complete")


def create_running_experiments_table():
    """Create running_experiments table in experiments.db"""
    print(f"[*] Creating running_experiments table in {EXPERIMENTS_DB}...")

    conn = sqlite3.connect(EXPERIMENTS_DB)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS running_experiments (
            run_name TEXT PRIMARY KEY,
            started_at TEXT NOT NULL,
            pid INTEGER,
            machine_id TEXT,
            last_heartbeat TEXT,
            current_cycle INTEGER DEFAULT 0,
            current_pnl REAL DEFAULT 0,
            target_cycles INTEGER,
            env_vars TEXT
        )
    """)

    conn.commit()
    conn.close()
    print(f"[OK] running_experiments table created")


def load_run_date_ranges():
    """Load date ranges from all run_info.json files"""
    print(f"[*] Loading run date ranges from {MODELS_DIR}...")

    runs = []
    models_path = Path(MODELS_DIR)

    for run_dir in models_path.iterdir():
        if not run_dir.is_dir():
            continue

        run_info_path = run_dir / "run_info.json"
        if not run_info_path.exists():
            continue

        try:
            with open(run_info_path) as f:
                info = json.load(f)

            start_date = info.get('start_date')
            end_date = info.get('end_date')

            if start_date and end_date:
                runs.append({
                    'run_id': run_dir.name,
                    'start_date': start_date,
                    'end_date': end_date
                })
        except Exception as e:
            print(f"    [WARN] Could not parse {run_info_path}: {e}")

    # Sort by start_date descending (newest first)
    runs.sort(key=lambda x: x['start_date'], reverse=True)

    print(f"    Found {len(runs)} runs with date ranges")
    return runs


def backfill_run_ids():
    """Backfill run_id for existing trades based on timestamp matching"""
    print(f"[*] Backfilling run_id for trades...")

    runs = load_run_date_ranges()
    if not runs:
        print("    [SKIP] No runs found with date ranges")
        return

    conn = sqlite3.connect(PAPER_DB)
    cursor = conn.cursor()

    # Get trades without run_id
    cursor.execute("""
        SELECT id, timestamp FROM trades
        WHERE run_id IS NULL AND timestamp IS NOT NULL
        ORDER BY timestamp
    """)
    trades = cursor.fetchall()

    print(f"    Found {len(trades)} trades without run_id")

    updated = 0
    for trade_id, timestamp in trades:
        # Find matching run by date range
        for run in runs:
            if run['start_date'] <= timestamp <= run['end_date']:
                cursor.execute(
                    "UPDATE trades SET run_id = ? WHERE id = ?",
                    (run['run_id'], trade_id)
                )
                updated += 1
                break

    conn.commit()
    conn.close()

    print(f"[OK] Updated {updated} trades with run_id")


def create_training_runs_cache():
    """Create a training_runs table for quick run info lookups"""
    print(f"[*] Creating training_runs cache table...")

    conn = sqlite3.connect(EXPERIMENTS_DB)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS training_runs (
            run_id TEXT PRIMARY KEY,
            start_date TEXT,
            end_date TEXT,
            initial_balance REAL,
            final_balance REAL,
            pnl REAL,
            pnl_pct REAL,
            total_trades INTEGER,
            cycles INTEGER,
            win_rate REAL,
            summary_path TEXT,
            decision_records_path TEXT,
            decision_records_size INTEGER,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_training_runs_dates
        ON training_runs(start_date, end_date)
    """)

    conn.commit()

    # Populate from run_info.json files
    print("    Populating from run_info.json files...")
    models_path = Path(MODELS_DIR)
    populated = 0

    for run_dir in models_path.iterdir():
        if not run_dir.is_dir():
            continue

        run_info_path = run_dir / "run_info.json"
        if not run_info_path.exists():
            continue

        try:
            with open(run_info_path) as f:
                info = json.load(f)

            decision_path = run_dir / "state" / "decision_records.jsonl"
            decision_size = decision_path.stat().st_size if decision_path.exists() else 0

            cursor.execute("""
                INSERT OR REPLACE INTO training_runs
                (run_id, start_date, end_date, initial_balance, final_balance,
                 pnl, pnl_pct, total_trades, cycles, win_rate,
                 summary_path, decision_records_path, decision_records_size)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                run_dir.name,
                info.get('start_date'),
                info.get('end_date'),
                info.get('initial_balance'),
                info.get('final_balance'),
                info.get('pnl'),
                info.get('pnl_pct'),
                info.get('trades'),
                info.get('cycles'),
                info.get('win_rate'),
                str(run_dir / "SUMMARY.txt") if (run_dir / "SUMMARY.txt").exists() else None,
                str(decision_path) if decision_path.exists() else None,
                decision_size
            ))
            populated += 1
        except Exception as e:
            print(f"    [WARN] Could not process {run_dir.name}: {e}")

    conn.commit()
    conn.close()

    print(f"[OK] Populated {populated} runs in training_runs cache")


def main():
    print("=" * 60)
    print("Database Migration: Unified Dashboard Support")
    print("=" * 60)
    print()

    # Ensure data directory exists
    Path("data").mkdir(exist_ok=True)

    # Run migrations
    migrate_paper_trading_db()
    print()

    create_running_experiments_table()
    print()

    create_training_runs_cache()
    print()

    backfill_run_ids()
    print()

    print("=" * 60)
    print("Migration complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
