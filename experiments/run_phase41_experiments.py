#!/usr/bin/env python3
"""
Phase 41: Win Rate Improvement Experiments

Tests new quality filters to achieve 60%+ win rate:
1. Time Zone Filter - avoid first hour and last 30 min
2. Inverted Confidence - only trade low confidence (<0.25)
3. VIX Goldilocks - only trade VIX 15-22
4. Momentum Confirmation - require price moving in trade direction
5. Day of Week - skip Monday and Friday

Run: python experiments/run_phase41_experiments.py
"""

import os
import json
import subprocess
from pathlib import Path
from datetime import datetime
import time

EXPERIMENTS = [
    {
        "name": "baseline",
        "desc": "Baseline (no new filters)",
        "env": {}
    },
    {
        "name": "time_zone",
        "desc": "Time Zone Filter (skip first 60m, last 30m)",
        "env": {
            "TIME_ZONE_FILTER": "1",
            "TIME_ZONE_START_MINUTES": "60",
            "TIME_ZONE_END_MINUTES": "30"
        }
    },
    {
        "name": "inv_conf_25",
        "desc": "Inverted Confidence (max 0.25)",
        "env": {"TRAIN_MAX_CONF": "0.25"}
    },
    {
        "name": "inv_conf_timezone",
        "desc": "Inverted Conf + Time Zone",
        "env": {
            "TRAIN_MAX_CONF": "0.25",
            "TIME_ZONE_FILTER": "1",
            "TIME_ZONE_START_MINUTES": "60",
            "TIME_ZONE_END_MINUTES": "30"
        }
    },
    {
        "name": "vix_goldilocks",
        "desc": "VIX Goldilocks (15-22)",
        "env": {
            "VIX_GOLDILOCKS_FILTER": "1",
            "VIX_GOLDILOCKS_MIN": "15.0",
            "VIX_GOLDILOCKS_MAX": "22.0"
        }
    },
    {
        "name": "momentum",
        "desc": "Momentum Confirmation",
        "env": {
            "MOMENTUM_CONFIRM_FILTER": "1",
            "MOMENTUM_MIN_STRENGTH": "0.001"
        }
    },
    {
        "name": "day_of_week",
        "desc": "Day of Week (Tue-Thu only)",
        "env": {
            "DAY_OF_WEEK_FILTER": "1",
            "SKIP_MONDAY": "1",
            "SKIP_FRIDAY": "1"
        }
    },
    {
        "name": "ultra_selective",
        "desc": "Ultra-Selective (ALL filters)",
        "env": {
            "TRAIN_MAX_CONF": "0.30",
            "TIME_ZONE_FILTER": "1",
            "TIME_ZONE_START_MINUTES": "60",
            "TIME_ZONE_END_MINUTES": "30",
            "VIX_GOLDILOCKS_FILTER": "1",
            "VIX_GOLDILOCKS_MIN": "15.0",
            "VIX_GOLDILOCKS_MAX": "22.0",
            "MOMENTUM_CONFIRM_FILTER": "1",
            "MOMENTUM_MIN_STRENGTH": "0.001",
            "DAY_OF_WEEK_FILTER": "1",
            "SKIP_MONDAY": "1",
            "SKIP_FRIDAY": "1"
        }
    }
]

def check_results():
    """Check results of Phase 41 experiments"""
    results = []
    models_dir = Path("models")

    for exp in EXPERIMENTS:
        run_dir = models_dir / f"phase41_{exp['name']}"
        run_info = run_dir / "run_info.json"

        if run_info.exists():
            with open(run_info) as f:
                info = json.load(f)

            trades = info.get("trades", 0)
            # Calculate win rate from decision records if available
            decision_file = run_dir / "state" / "decision_records.jsonl"
            wins = 0
            if decision_file.exists():
                # Count wins from closed trades
                pass  # Would need to parse JSONL

            results.append({
                "name": exp["name"],
                "desc": exp["desc"],
                "pnl_pct": info.get("pnl_pct", 0),
                "trades": trades,
                "cycles": info.get("cycles", 0),
                "per_trade_pnl": info.get("pnl", 0) / max(1, trades)
            })
        else:
            results.append({
                "name": exp["name"],
                "desc": exp["desc"],
                "status": "not_found"
            })

    return results

if __name__ == "__main__":
    results = check_results()
    print("\n=== Phase 41 Results ===\n")
    for r in results:
        if "status" in r:
            print(f"  {r['name']}: {r['status']}")
        else:
            print(f"  {r['name']}: P&L={r['pnl_pct']:+.1f}%, trades={r['trades']}, $/trade=${r['per_trade_pnl']:.2f}")
