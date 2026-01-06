#!/usr/bin/env python3
"""
SKIP_MONDAY + Quantor Improvements Test

Tests whether adding Quantor-MTFuzz components improves the best model (SKIP_MONDAY).

Baseline: COMBO_SKIP_MONDAY_20K = +1630% P&L

Tests:
1. SKIP_MONDAY baseline (5K) - establish 5K baseline
2. SKIP_MONDAY + Quantor regime filter - adds direction filtering
3. SKIP_MONDAY + Data alignment - adds freshness tracking
4. SKIP_MONDAY + Full Quantor - all improvements combined

Usage:
    python experiments/run_skip_monday_quantor.py           # Run all tests
    python experiments/run_skip_monday_quantor.py --quick   # 2K cycles each
"""

import os
import sys
import subprocess
from datetime import datetime

# ============================================================
# BASE CONFIGURATION (from BEST_CONFIG_SKIP_MONDAY.md)
# ============================================================
SKIP_MONDAY_BASE = {
    # Day-of-week filtering
    "DAY_OF_WEEK_FILTER": "1",
    "SKIP_MONDAY": "1",
    "SKIP_FRIDAY": "0",

    # Trailing stop
    "USE_TRAILING_STOP": "1",
    "TRAILING_ACTIVATION_PCT": "10",
    "TRAILING_STOP_PCT": "5",

    # Trade Direction Analysis
    "ENABLE_TDA": "1",
    "TDA_REGIME_FILTER": "1",

    # Low confidence filter (counter-intuitive best performer)
    "TRAIN_MAX_CONF": "0.25",

    # Standard settings
    "PAPER_TRADING": "True",
}

# ============================================================
# TEST CONFIGURATIONS
# ============================================================

# Test 1: Baseline (5K cycles to compare with 20K proportionally)
BASELINE_5K = {
    **SKIP_MONDAY_BASE,
    "MODEL_RUN_DIR": "models/SKIPMONDAYQ_BASELINE",
    "TT_MAX_CYCLES": "5000",
    "TT_PRINT_EVERY": "500",
}

# Test 2: + Quantor Regime Filter (direction alignment)
PLUS_REGIME = {
    **SKIP_MONDAY_BASE,
    "QUANTOR_REGIME_FILTER": "1",
    "QUANTOR_CRASH_BLOCK": "1",
    "QUANTOR_DIRECTION_FILTER": "1",
    "MODEL_RUN_DIR": "models/SKIPMONDAYQ_REGIME",
    "TT_MAX_CYCLES": "5000",
    "TT_PRINT_EVERY": "500",
}

# Test 3: + Data Alignment (freshness tracking)
PLUS_ALIGNMENT = {
    **SKIP_MONDAY_BASE,
    "ALIGNMENT_ENABLED": "1",
    "ALIGNMENT_MAX_LAG_SEC": "300",  # 5 min max lag for training
    "ALIGNMENT_IV_DECAY_HALF_LIFE": "150",  # 2.5 min half-life
    "MODEL_RUN_DIR": "models/SKIPMONDAYQ_ALIGNMENT",
    "TT_MAX_CYCLES": "5000",
    "TT_PRINT_EVERY": "500",
}

# Test 4: Full Combined (Regime + Alignment)
FULL_COMBINED = {
    **SKIP_MONDAY_BASE,
    # Quantor regime filter
    "QUANTOR_REGIME_FILTER": "1",
    "QUANTOR_CRASH_BLOCK": "1",
    "QUANTOR_DIRECTION_FILTER": "1",
    # Data alignment
    "ALIGNMENT_ENABLED": "1",
    "ALIGNMENT_MAX_LAG_SEC": "300",
    "ALIGNMENT_IV_DECAY_HALF_LIFE": "150",
    "MODEL_RUN_DIR": "models/SKIPMONDAYQ_FULL",
    "TT_MAX_CYCLES": "5000",
    "TT_PRINT_EVERY": "500",
}

# Quick versions (2K cycles for faster iteration)
QUICK_CYCLES = "2000"
QUICK_PRINT = "400"


def run_experiment(name, config, quick=False):
    """Run a single experiment with given configuration."""
    if quick:
        config = config.copy()
        config["TT_MAX_CYCLES"] = QUICK_CYCLES
        config["TT_PRINT_EVERY"] = QUICK_PRINT
        config["MODEL_RUN_DIR"] = config["MODEL_RUN_DIR"] + "_QUICK"

    print("=" * 70)
    print(f"EXPERIMENT: {name}")
    print("=" * 70)
    print("\nKey Configuration:")
    key_vars = ["MODEL_RUN_DIR", "TT_MAX_CYCLES", "QUANTOR_REGIME_FILTER",
                "ALIGNMENT_ENABLED", "SKIP_MONDAY", "TRAIN_MAX_CONF"]
    for key in key_vars:
        if key in config:
            print(f"  {key}={config[key]}")
    print()

    env = os.environ.copy()
    env.update(config)

    cmd = [sys.executable, "scripts/train_time_travel.py"]

    print(f"Starting at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 70)

    try:
        result = subprocess.run(cmd, env=env, capture_output=False, text=True)
        print(f"\nCompleted at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        return result.returncode == 0
    except KeyboardInterrupt:
        print("\n[INTERRUPTED]")
        return False


def run_all(quick=False):
    """Run all experiments sequentially."""
    configs = [
        ("SKIP_MONDAY Baseline (5K)", BASELINE_5K),
        ("SKIP_MONDAY + Quantor Regime Filter", PLUS_REGIME),
        ("SKIP_MONDAY + Data Alignment", PLUS_ALIGNMENT),
        ("SKIP_MONDAY + Full Combined", FULL_COMBINED),
    ]

    results = {}
    for name, config in configs:
        success = run_experiment(name, config, quick=quick)
        results[name] = "PASS" if success else "FAIL"
        print("\n")

    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print("\nExperiment Status:")
    for name, status in results.items():
        print(f"  [{status}] {name}")

    print("\nCheck these files for detailed results:")
    print("  models/SKIPMONDAYQ_BASELINE/SUMMARY.txt")
    print("  models/SKIPMONDAYQ_REGIME/SUMMARY.txt")
    print("  models/SKIPMONDAYQ_ALIGNMENT/SUMMARY.txt")
    print("  models/SKIPMONDAYQ_FULL/SUMMARY.txt")

    return results


def main():
    quick = "--quick" in sys.argv

    if "--baseline" in sys.argv:
        run_experiment("SKIP_MONDAY Baseline", BASELINE_5K, quick=quick)
    elif "--regime" in sys.argv:
        run_experiment("SKIP_MONDAY + Regime", PLUS_REGIME, quick=quick)
    elif "--alignment" in sys.argv:
        run_experiment("SKIP_MONDAY + Alignment", PLUS_ALIGNMENT, quick=quick)
    elif "--full" in sys.argv:
        run_experiment("SKIP_MONDAY + Full", FULL_COMBINED, quick=quick)
    else:
        run_all(quick=quick)


if __name__ == "__main__":
    main()
