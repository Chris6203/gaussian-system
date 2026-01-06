#!/usr/bin/env python3
"""
Quantor-MTFuzz Improvements Test

Integrates Jerry Mahabub & John Draper's Quantor-MTFuzz components:
1. REGIME FILTER - Block trades during CRASH mode, wrong direction
2. FUZZY POSITION SIZING - Size based on 9-factor fuzzy confidence
3. VOLATILITY ANALYTICS - VRP-aware trading, dynamic stops
4. DATA ALIGNMENT - Already integrated for backtest quality

Usage:
    python experiments/run_quantor_improvements.py
    python experiments/run_quantor_improvements.py --baseline  # Run without Quantor
    python experiments/run_quantor_improvements.py --compare   # Run both
"""

import os
import sys
import subprocess
from datetime import datetime

# ============================================================
# QUANTOR IMPROVEMENT CONFIGURATIONS
# ============================================================

# Level 1: Regime Filter Only
QUANTOR_REGIME_ONLY = {
    "QUANTOR_REGIME_FILTER": "1",        # Enable Quantor regime filter
    "QUANTOR_CRASH_BLOCK": "1",          # Block trades in CRASH mode (VIX > 35)
    "QUANTOR_DIRECTION_FILTER": "1",     # Block wrong direction (CALL in bear, PUT in bull)
    "MODEL_RUN_DIR": "models/QUANTOR_REGIME_ONLY",
    "TT_MAX_CYCLES": "5000",
    "TT_PRINT_EVERY": "500",
    "PAPER_TRADING": "True",
}

# Level 2: Regime + Fuzzy Sizing
QUANTOR_REGIME_FUZZY = {
    "QUANTOR_REGIME_FILTER": "1",
    "QUANTOR_CRASH_BLOCK": "1",
    "QUANTOR_DIRECTION_FILTER": "1",
    "QUANTOR_FUZZY_SIZING": "1",         # Enable fuzzy position sizing
    "QUANTOR_MIN_FUZZY_CONF": "0.4",     # Minimum fuzzy confidence to trade
    "QUANTOR_FUZZY_SCALE": "1",          # Scale position by fuzzy confidence
    "MODEL_RUN_DIR": "models/QUANTOR_REGIME_FUZZY",
    "TT_MAX_CYCLES": "5000",
    "TT_PRINT_EVERY": "500",
    "PAPER_TRADING": "True",
}

# Level 3: Full Quantor (Regime + Fuzzy + Vol Analytics)
QUANTOR_FULL = {
    "QUANTOR_REGIME_FILTER": "1",
    "QUANTOR_CRASH_BLOCK": "1",
    "QUANTOR_DIRECTION_FILTER": "1",
    "QUANTOR_FUZZY_SIZING": "1",
    "QUANTOR_MIN_FUZZY_CONF": "0.4",
    "QUANTOR_FUZZY_SCALE": "1",
    "QUANTOR_VOL_ANALYTICS": "1",        # Enable volatility analytics
    "QUANTOR_VRP_FILTER": "1",           # Only trade when VRP is favorable
    "QUANTOR_DYNAMIC_STOPS": "1",        # ATR-based stop distances
    "QUANTOR_ATR_STOP_MULT": "2.0",      # 2x ATR for stop distance
    "MODEL_RUN_DIR": "models/QUANTOR_FULL",
    "TT_MAX_CYCLES": "5000",
    "TT_PRINT_EVERY": "500",
    "PAPER_TRADING": "True",
}

# Baseline (current best config without Quantor)
BASELINE = {
    "MODEL_RUN_DIR": "models/QUANTOR_BASELINE",
    "TT_MAX_CYCLES": "5000",
    "TT_PRINT_EVERY": "500",
    "PAPER_TRADING": "True",
}

# 20K Validation (for promising configs)
VALIDATION_20K = {
    "QUANTOR_REGIME_FILTER": "1",
    "QUANTOR_CRASH_BLOCK": "1",
    "QUANTOR_DIRECTION_FILTER": "1",
    "QUANTOR_FUZZY_SIZING": "1",
    "QUANTOR_MIN_FUZZY_CONF": "0.4",
    "MODEL_RUN_DIR": "models/QUANTOR_20K_VALIDATION",
    "TT_MAX_CYCLES": "20000",
    "TT_PRINT_EVERY": "2000",
    "PAPER_TRADING": "True",
}


def run_experiment(name, config):
    """Run a single experiment with given configuration."""
    print("=" * 70)
    print(f"EXPERIMENT: {name}")
    print("=" * 70)
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}={value}")
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


def run_comparison():
    """Run baseline and all Quantor levels for comparison."""
    results = {}

    configs = [
        ("Baseline (No Quantor)", BASELINE),
        ("Quantor Level 1: Regime Filter", QUANTOR_REGIME_ONLY),
        ("Quantor Level 2: Regime + Fuzzy", QUANTOR_REGIME_FUZZY),
        ("Quantor Level 3: Full Integration", QUANTOR_FULL),
    ]

    for name, config in configs:
        success = run_experiment(name, config)
        results[name] = "PASS" if success else "FAIL"
        print("\n" + "=" * 70 + "\n")

    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)
    for name, status in results.items():
        print(f"  {name}: {status}")

    print("\nCheck models/ directory for SUMMARY.txt files to compare P&L and win rates.")
    return results


def main():
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg == "--baseline":
            run_experiment("Baseline", BASELINE)
        elif arg == "--regime":
            run_experiment("Quantor Regime Only", QUANTOR_REGIME_ONLY)
        elif arg == "--fuzzy":
            run_experiment("Quantor Regime + Fuzzy", QUANTOR_REGIME_FUZZY)
        elif arg == "--full":
            run_experiment("Quantor Full Integration", QUANTOR_FULL)
        elif arg == "--validate":
            run_experiment("Quantor 20K Validation", VALIDATION_20K)
        elif arg == "--compare":
            run_comparison()
        else:
            print(f"Unknown argument: {arg}")
            print("Usage: python run_quantor_improvements.py [--baseline|--regime|--fuzzy|--full|--validate|--compare]")
    else:
        # Default: run regime + fuzzy (best balance of improvement vs complexity)
        run_experiment("Quantor Regime + Fuzzy Sizing", QUANTOR_REGIME_FUZZY)


if __name__ == "__main__":
    main()
