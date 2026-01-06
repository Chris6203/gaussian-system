#!/usr/bin/env python3
"""
Day Trading Optimization Experiments

Target: 3-4 profitable trades per day

Based on analysis:
1. skew_tighter_tp: 59.1% WR on 15-60 min holds
2. Best hour: 15:00, worst: 12:00 (noon)
3. Calls significantly outperform puts
4. 30 min force close works well
5. Low confidence (22-30%) performs better

Tests:
1. BASELINE - Current day trading baseline
2. CALLS_ONLY - Skip PUT trades entirely
3. SKIP_NOON - Skip 11:30-12:30 trades
4. SHORT_HOLD - 30 min max hold
5. AFTERNOON_ONLY - Trade 13:00-16:00 only
6. COMBINED - All improvements
"""

import os
import sys
import subprocess
from datetime import datetime

# ============================================================
# BASE CONFIGURATION (day trading focused)
# ============================================================
DAY_TRADING_BASE = {
    # Short hold times for day trading
    "TT_MAX_HOLD_MINUTES": "45",

    # Standard exit settings
    "HARD_STOP_LOSS_PCT": "8",
    "HARD_TAKE_PROFIT_PCT": "12",

    # Trailing stop
    "USE_TRAILING_STOP": "1",
    "TRAILING_ACTIVATION_PCT": "8",
    "TRAILING_STOP_PCT": "4",

    # Low confidence filter (proven effective)
    "TRAIN_MAX_CONF": "0.30",

    # TDA for direction filtering
    "ENABLE_TDA": "1",

    # Standard settings
    "PAPER_TRADING": "True",
    "TT_MAX_CYCLES": "10000",
    "TT_PRINT_EVERY": "1000",
}

# ============================================================
# DAY TRADING TESTS
# ============================================================

# Test 1: Baseline day trading
BASELINE = {
    **DAY_TRADING_BASE,
    "MODEL_RUN_DIR": "models/DAYTRADE_BASELINE",
}

# Test 2: Calls only (puts lose money in bull market)
CALLS_ONLY = {
    **DAY_TRADING_BASE,
    "CALLS_ONLY": "1",
    "MODEL_RUN_DIR": "models/DAYTRADE_CALLS_ONLY",
}

# Test 3: Skip first 90 min (analysis shows 28% WR morning)
SKIP_MORNING = {
    **DAY_TRADING_BASE,
    "SKIP_FIRST_90_MIN": "1",
    "MODEL_RUN_DIR": "models/DAYTRADE_SKIP_MORNING",
}

# Test 4: Shorter hold (30 min - proven best from skew_tighter_tp)
SHORT_HOLD = {
    **DAY_TRADING_BASE,
    "TT_MAX_HOLD_MINUTES": "30",
    "MODEL_RUN_DIR": "models/DAYTRADE_SHORT_HOLD",
}

# Test 5: Midday only (13:00-14:59 - analysis shows 57% WR vs 33%)
MIDDAY = {
    **DAY_TRADING_BASE,
    "MIDDAY_ONLY": "1",
    "MODEL_RUN_DIR": "models/DAYTRADE_MIDDAY",
}

# Test 6: Tighter TP (6% instead of 12%)
TIGHT_TP = {
    **DAY_TRADING_BASE,
    "HARD_TAKE_PROFIT_PCT": "6",
    "MODEL_RUN_DIR": "models/DAYTRADE_TIGHT_TP",
}

# Test 7: Combined best settings
COMBINED = {
    **DAY_TRADING_BASE,
    "TT_MAX_HOLD_MINUTES": "30",
    "CALLS_ONLY": "1",
    "SKIP_FIRST_90_MIN": "1",
    "HARD_TAKE_PROFIT_PCT": "8",
    "MODEL_RUN_DIR": "models/DAYTRADE_COMBINED",
}

# Test 8: Combined + Midday only
COMBINED_MIDDAY = {
    **DAY_TRADING_BASE,
    "TT_MAX_HOLD_MINUTES": "30",
    "CALLS_ONLY": "1",
    "MIDDAY_ONLY": "1",
    "HARD_TAKE_PROFIT_PCT": "8",
    "MODEL_RUN_DIR": "models/DAYTRADE_COMBINED_MIDDAY",
}


def run_experiment(name, config, quick=False):
    """Run a single experiment with given configuration."""
    if quick:
        config = config.copy()
        config["TT_MAX_CYCLES"] = "5000"
        config["TT_PRINT_EVERY"] = "500"
        config["MODEL_RUN_DIR"] = config["MODEL_RUN_DIR"] + "_QUICK"

    print("=" * 70)
    print(f"EXPERIMENT: {name}")
    print("=" * 70)
    print("\nKey Configuration:")
    key_vars = ["MODEL_RUN_DIR", "TT_MAX_CYCLES", "TT_MAX_HOLD_MINUTES",
                "CALLS_ONLY", "SKIP_FIRST_90_MIN", "MIDDAY_ONLY",
                "HARD_TAKE_PROFIT_PCT", "TRAIN_MAX_CONF"]
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
        ("Baseline (day trading)", BASELINE),
        ("+ Calls Only", CALLS_ONLY),
        ("+ Skip Morning 90min", SKIP_MORNING),
        ("+ 30min Hold", SHORT_HOLD),
        ("+ Midday Only", MIDDAY),
        ("+ Tight TP (6%)", TIGHT_TP),
        ("+ Combined (Calls+Skip+8%TP)", COMBINED),
        ("+ Combined Midday", COMBINED_MIDDAY),
    ]

    results = {}
    for name, config in configs:
        success = run_experiment(name, config, quick=quick)
        results[name] = "PASS" if success else "FAIL"
        print("\n")

    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY - Check models/DAYTRADE_*/SUMMARY.txt for details")
    print("=" * 70)
    for name, status in results.items():
        print(f"  [{status}] {name}")

    return results


def main():
    quick = "--quick" in sys.argv

    if "--baseline" in sys.argv:
        run_experiment("Baseline", BASELINE, quick=quick)
    elif "--calls" in sys.argv:
        run_experiment("Calls Only", CALLS_ONLY, quick=quick)
    elif "--skip-morning" in sys.argv:
        run_experiment("Skip Morning", SKIP_MORNING, quick=quick)
    elif "--short" in sys.argv:
        run_experiment("Short Hold", SHORT_HOLD, quick=quick)
    elif "--midday" in sys.argv:
        run_experiment("Midday Only", MIDDAY, quick=quick)
    elif "--tight-tp" in sys.argv:
        run_experiment("Tight TP", TIGHT_TP, quick=quick)
    elif "--combined" in sys.argv:
        run_experiment("Combined", COMBINED, quick=quick)
    elif "--combined-midday" in sys.argv:
        run_experiment("Combined Midday", COMBINED_MIDDAY, quick=quick)
    else:
        run_all(quick=quick)


if __name__ == "__main__":
    main()
