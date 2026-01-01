#!/usr/bin/env python3
"""
Decision Records Indexer

Creates index files for decision_records.jsonl files to enable fast trade lookups.
Index allows seeking directly to specific trades without reading entire 45MB+ file.

Usage:
    python scripts/index_decision_records.py              # Index all runs
    python scripts/index_decision_records.py --run v3_10k # Index specific run
    python scripts/index_decision_records.py --check      # Check index status
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime


MODELS_DIR = "models"


def build_index(jsonl_path: Path) -> dict:
    """
    Build an index for fast trade lookup in decision_records.jsonl.

    The index maps trade timestamps to byte offsets and line numbers,
    allowing direct seeking to specific trades.
    """
    index = {
        "version": 1,
        "created_at": datetime.now().isoformat(),
        "file_path": str(jsonl_path),
        "file_size_bytes": jsonl_path.stat().st_size,
        "total_records": 0,
        "trade_records": 0,
        "hold_records": 0,
        "trade_offsets": {}
    }

    print(f"    Indexing {jsonl_path.name} ({index['file_size_bytes'] / (1024*1024):.1f} MB)...")

    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            byte_offset = 0
            for line_num, line in enumerate(f):
                index["total_records"] += 1

                try:
                    record = json.loads(line.strip())

                    # Check if this is a trade record
                    trade_placed = record.get("trade_placed", False)
                    action = record.get("action", "")

                    if trade_placed or action in ("BUY_CALLS", "BUY_PUTS"):
                        index["trade_records"] += 1

                        # Get timestamp
                        timestamp = record.get("timestamp") or record.get("sim_time")
                        if timestamp:
                            # Normalize timestamp format
                            if isinstance(timestamp, str):
                                ts_key = timestamp
                            else:
                                ts_key = str(timestamp)

                            index["trade_offsets"][ts_key] = {
                                "byte_offset": byte_offset,
                                "line_number": line_num,
                                "action": action
                            }
                    elif action == "HOLD":
                        index["hold_records"] += 1

                except json.JSONDecodeError:
                    pass

                byte_offset += len(line.encode('utf-8'))

                # Progress indicator
                if line_num > 0 and line_num % 100000 == 0:
                    print(f"      Processed {line_num:,} records...")

    except Exception as e:
        print(f"    Error indexing: {e}")
        return None

    return index


def save_index(index: dict, jsonl_path: Path) -> Path:
    """Save index file alongside the jsonl file."""
    index_path = jsonl_path.parent / 'decision_records_index.json'
    with open(index_path, 'w') as f:
        json.dump(index, f, indent=2)
    return index_path


def get_decision_by_timestamp(jsonl_path: Path, index_path: Path, timestamp: str) -> dict:
    """
    Get decision record by timestamp using index for fast lookup.

    Args:
        jsonl_path: Path to decision_records.jsonl
        index_path: Path to decision_records_index.json
        timestamp: Timestamp to look up

    Returns:
        The decision record dict, or None if not found
    """
    try:
        with open(index_path) as f:
            index = json.load(f)

        offset_info = index.get("trade_offsets", {}).get(timestamp)
        if not offset_info:
            return None

        with open(jsonl_path, 'r', encoding='utf-8') as f:
            f.seek(offset_info["byte_offset"])
            line = f.readline()
            return json.loads(line.strip())

    except Exception as e:
        print(f"Error looking up decision: {e}")
        return None


def index_all_runs():
    """Index all decision_records.jsonl files in models/"""
    models_path = Path(MODELS_DIR)
    indexed = 0
    skipped = 0
    failed = 0

    for run_dir in sorted(models_path.iterdir()):
        if not run_dir.is_dir():
            continue

        jsonl_path = run_dir / 'state' / 'decision_records.jsonl'
        index_path = run_dir / 'state' / 'decision_records_index.json'

        if not jsonl_path.exists():
            continue

        # Skip if index is newer than jsonl
        if index_path.exists():
            if index_path.stat().st_mtime > jsonl_path.stat().st_mtime:
                print(f"[SKIP] {run_dir.name} - index up to date")
                skipped += 1
                continue

        print(f"[INDEX] {run_dir.name}")
        index = build_index(jsonl_path)

        if index:
            save_index(index, jsonl_path)
            print(f"    Trades: {index['trade_records']:,}, Total: {index['total_records']:,}")
            indexed += 1
        else:
            failed += 1

    print(f"\nSummary: {indexed} indexed, {skipped} skipped, {failed} failed")
    return indexed, skipped, failed


def index_single_run(run_name: str):
    """Index a specific run."""
    run_dir = Path(MODELS_DIR) / run_name
    if not run_dir.exists():
        print(f"Run not found: {run_name}")
        return False

    jsonl_path = run_dir / 'state' / 'decision_records.jsonl'
    if not jsonl_path.exists():
        print(f"No decision_records.jsonl in {run_name}")
        return False

    print(f"[INDEX] {run_name}")
    index = build_index(jsonl_path)

    if index:
        index_path = save_index(index, jsonl_path)
        print(f"    Trades: {index['trade_records']:,}")
        print(f"    Total records: {index['total_records']:,}")
        print(f"    Index saved: {index_path}")
        return True

    return False


def check_index_status():
    """Check which runs have indexes and their status."""
    models_path = Path(MODELS_DIR)

    print(f"{'Run Name':<40} {'JSONL Size':<12} {'Index':<8} {'Status'}")
    print("-" * 75)

    for run_dir in sorted(models_path.iterdir()):
        if not run_dir.is_dir():
            continue

        jsonl_path = run_dir / 'state' / 'decision_records.jsonl'
        index_path = run_dir / 'state' / 'decision_records_index.json'

        if not jsonl_path.exists():
            continue

        jsonl_size = jsonl_path.stat().st_size / (1024 * 1024)

        if index_path.exists():
            index_newer = index_path.stat().st_mtime > jsonl_path.stat().st_mtime
            status = "Up to date" if index_newer else "STALE"
            has_index = "Yes"
        else:
            has_index = "No"
            status = "MISSING"

        print(f"{run_dir.name:<40} {jsonl_size:>8.1f} MB  {has_index:<8} {status}")


def main():
    parser = argparse.ArgumentParser(description="Index decision records for fast lookups")
    parser.add_argument('--run', type=str, help="Index specific run only")
    parser.add_argument('--check', action='store_true', help="Check index status")
    args = parser.parse_args()

    if args.check:
        check_index_status()
    elif args.run:
        index_single_run(args.run)
    else:
        index_all_runs()


if __name__ == "__main__":
    main()
