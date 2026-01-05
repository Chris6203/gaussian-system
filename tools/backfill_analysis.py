#!/usr/bin/env python3
"""
Backfill Analysis Tool

Runs post_experiment_analysis.py on all experiments that have trade data
but don't yet have an ANALYSIS.md file.

Usage:
    python tools/backfill_analysis.py           # Analyze all missing
    python tools/backfill_analysis.py --force   # Re-analyze all
    python tools/backfill_analysis.py --limit 10  # Limit to N experiments
"""

import os
import sys
import sqlite3
import subprocess
from pathlib import Path
from datetime import datetime
import argparse

BASE_DIR = Path(__file__).parent.parent.resolve()


def get_experiments_with_trades():
    """Get all experiment run_ids that have trade data."""
    db_path = BASE_DIR / "data" / "paper_trading.db"
    if not db_path.exists():
        return []

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT DISTINCT run_id, COUNT(*) as trade_count
        FROM trades
        WHERE run_id IS NOT NULL AND run_id != ''
        GROUP BY run_id
        HAVING trade_count >= 5
        ORDER BY run_id
    """)

    experiments = cursor.fetchall()
    conn.close()
    return experiments


def get_experiments_needing_analysis(force=False):
    """Find experiments that need analysis."""
    experiments = get_experiments_with_trades()
    needs_analysis = []

    for run_id, trade_count in experiments:
        model_dir = BASE_DIR / "models" / run_id
        analysis_file = model_dir / "ANALYSIS.md"

        if force or not analysis_file.exists():
            needs_analysis.append((run_id, trade_count, model_dir.exists()))

    return needs_analysis


def run_analysis(run_id):
    """Run post_experiment_analysis.py on a single experiment."""
    model_dir = BASE_DIR / "models" / run_id
    script_path = BASE_DIR / "tools" / "post_experiment_analysis.py"

    # Create model dir if it doesn't exist (for experiments without saved models)
    if not model_dir.exists():
        model_dir.mkdir(parents=True, exist_ok=True)

    try:
        result = subprocess.run(
            [sys.executable, str(script_path), str(model_dir)],
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode == 0:
            return True, "Success"
        else:
            return False, result.stderr[:200]

    except subprocess.TimeoutExpired:
        return False, "Timeout"
    except Exception as e:
        return False, str(e)


def main():
    parser = argparse.ArgumentParser(description="Backfill analysis for experiments")
    parser.add_argument("--force", action="store_true", help="Re-analyze all experiments")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of experiments")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be analyzed")
    args = parser.parse_args()

    print("=" * 60)
    print("Backfill Analysis Tool")
    print("=" * 60)

    # Get experiments needing analysis
    needs_analysis = get_experiments_needing_analysis(force=args.force)

    if args.limit > 0:
        needs_analysis = needs_analysis[:args.limit]

    print(f"Found {len(needs_analysis)} experiments needing analysis")

    if args.dry_run:
        print("\nDry run - would analyze:")
        for run_id, trade_count, has_dir in needs_analysis[:20]:
            status = "has_dir" if has_dir else "no_dir"
            print(f"  {run_id}: {trade_count} trades ({status})")
        if len(needs_analysis) > 20:
            print(f"  ... and {len(needs_analysis) - 20} more")
        return

    # Run analysis
    success_count = 0
    fail_count = 0

    for i, (run_id, trade_count, has_dir) in enumerate(needs_analysis):
        print(f"[{i+1}/{len(needs_analysis)}] Analyzing {run_id} ({trade_count} trades)...", end=" ")

        success, message = run_analysis(run_id)

        if success:
            print("✅")
            success_count += 1
        else:
            print(f"❌ {message[:50]}")
            fail_count += 1

    print("\n" + "=" * 60)
    print(f"Complete: {success_count} succeeded, {fail_count} failed")
    print("=" * 60)


if __name__ == "__main__":
    main()
