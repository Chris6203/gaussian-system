#!/usr/bin/env python3
"""
Migration: Add tuning fields to trades table
============================================

Adds columns for capturing entry/exit context to help tune trading strategies.

Run: python scripts/migrate_add_tuning_fields.py
"""

import sqlite3
from pathlib import Path

DB_PATH = Path('data/paper_trading.db')

# New columns to add
NEW_COLUMNS = [
    # Calibration (Phase 14)
    ("calibrated_confidence", "REAL"),  # P(profit) from calibration tracker

    # Entry context
    ("spy_price", "REAL"),           # SPY price at entry
    ("vix_level", "REAL"),           # VIX at entry
    ("hmm_trend", "REAL"),           # HMM trend state (0=bearish, 1=bullish)
    ("hmm_volatility", "REAL"),      # HMM volatility state
    ("hmm_liquidity", "REAL"),       # HMM liquidity state
    ("hmm_confidence", "REAL"),      # HMM confidence
    ("predicted_return", "REAL"),    # Model's predicted return
    ("prediction_timeframe", "TEXT"), # Which timeframe (5min, 15min, etc.)
    ("entry_controller", "TEXT"),    # bandit, rl, consensus, q_scorer
    ("signal_strategy", "TEXT"),     # NEURAL_BULLISH, HMM_TREND, etc.
    ("signal_reasoning", "TEXT"),    # Full reasoning chain
    ("momentum_5m", "REAL"),         # 5-min momentum at entry
    ("momentum_15m", "REAL"),        # 15-min momentum at entry
    ("volume_spike", "REAL"),        # Volume spike indicator
    ("direction_probs", "TEXT"),     # JSON: [down, neutral, up]

    # Exit context
    ("exit_spy_price", "REAL"),      # SPY price at exit
    ("exit_vix_level", "REAL"),      # VIX at exit
    ("hold_minutes", "REAL"),        # Actual hold duration in minutes
    ("max_drawdown_pct", "REAL"),    # Maximum drawdown during trade
    ("max_gain_pct", "REAL"),        # Maximum gain during trade
    ("exit_hmm_trend", "REAL"),      # HMM trend at exit (regime change?)
]


def migrate():
    if not DB_PATH.exists():
        print(f"Database not found: {DB_PATH}")
        return False

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Get existing columns
    cursor.execute("PRAGMA table_info(trades)")
    existing_cols = {row[1] for row in cursor.fetchall()}

    added = 0
    for col_name, col_type in NEW_COLUMNS:
        if col_name not in existing_cols:
            try:
                cursor.execute(f"ALTER TABLE trades ADD COLUMN {col_name} {col_type}")
                print(f"  Added: {col_name} ({col_type})")
                added += 1
            except sqlite3.OperationalError as e:
                print(f"  Skip {col_name}: {e}")
        else:
            print(f"  Exists: {col_name}")

    conn.commit()
    conn.close()

    print(f"\nMigration complete: {added} columns added")
    return True


if __name__ == "__main__":
    print("Migrating trades table with tuning fields...\n")
    migrate()
