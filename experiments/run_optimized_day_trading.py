#!/usr/bin/env python3
"""
Optimized Day Trading Configuration

Target: ~6 trades per day with improved win rate

Based on analysis of HIGH_MIN_RET trades (281 trades, 38% WR, +423% P&L):
- Best hours: 10:xx (58.6% WR), 12:xx (50% WR)
- Worst hour: 15:xx (9.5% WR) - SKIP THIS
- Best days: Wed/Thu (51-52% WR)
- Worst days: Mon (24% WR) - SKIP THIS
- FAST CUT exit: 0% WR, -$712 loss - DISABLE THIS
- FORCE_CLOSE 45min: 50% WR, +$68 - KEEP THIS

This config applies all discovered improvements:
1. CALLS_ONLY (CALLs 42% WR vs PUTs 34% WR)
2. SKIP_MONDAY (24% WR on Mondays)
3. SKIP_LAST_HOUR (9.5% WR in 15:xx)
4. FAST_LOSS_CUT_ENABLED=0 (disable -2% early exit with 0% WR)
5. Keep 45min max hold (profitable exit type)
6. Use HIGH_MIN_RET thresholds (5.6 trades/day)
"""

import os
import sys
import subprocess
from datetime import datetime

# ============================================================
# OPTIMIZED DAY TRADING CONFIG
# Target: ~6 trades/day with improved win rate
# ============================================================

OPTIMIZED_CONFIG = {
    # Base settings from HIGH_MIN_RET (best P&L performer)
    "CALLS_ONLY": "1",
    "TT_MAX_HOLD_MINUTES": "45",
    "HARD_STOP_LOSS_PCT": "8",
    "HARD_TAKE_PROFIT_PCT": "12",
    "USE_TRAILING_STOP": "1",
    "TRAILING_ACTIVATION_PCT": "8",
    "TRAILING_STOP_PCT": "4",
    "ENABLE_TDA": "1",

    # Confidence thresholds from HIGH_MIN_RET
    "TT_TRAIN_MIN_CONF": "0.15",
    "TRAIN_MAX_CONF": "0.50",
    "TRAIN_MIN_ABS_RET": "0.002",  # 0.2% minimum edge

    # NEW: Day/Time filters based on analysis
    "DAY_OF_WEEK_FILTER": "1",
    "SKIP_MONDAY": "1",           # Mon = 24% WR (worst day)
    "SKIP_FRIDAY": "0",           # Keep Friday (37% WR)
    "SKIP_LAST_HOUR": "1",        # 15:xx = 9.5% WR (worst hour)

    # NEW: Disable FAST CUT exit (0% WR, -$712 loss)
    "FAST_LOSS_CUT_ENABLED": "0",

    # Training settings
    "PAPER_TRADING": "True",
    "TT_MAX_CYCLES": "5000",
    "TT_PRINT_EVERY": "500",
    "MODEL_RUN_DIR": "models/OPTIMIZED_DAY_TRADING",
}

# Variant: Add midday focus (10:xx-14:xx only)
MIDDAY_FOCUS_CONFIG = {
    **OPTIMIZED_CONFIG,
    "MIDDAY_ONLY": "1",  # Only trade 13:00-14:59 (best hours)
    "MODEL_RUN_DIR": "models/OPTIMIZED_MIDDAY_FOCUS",
}

# Variant: Skip Friday too (worst day after Monday)
SKIP_MON_FRI_CONFIG = {
    **OPTIMIZED_CONFIG,
    "SKIP_FRIDAY": "1",  # Also skip Friday
    "MODEL_RUN_DIR": "models/OPTIMIZED_SKIP_MON_FRI",
}

# Variant: Disable all RL-based exits, use only hard rules
HARD_RULES_ONLY_CONFIG = {
    **OPTIMIZED_CONFIG,
    "USE_XGBOOST_EXIT": "0",  # Disable XGBoost exit entirely
    "FAST_LOSS_CUT_ENABLED": "0",
    "MODEL_RUN_DIR": "models/OPTIMIZED_HARD_RULES",
}


def run_experiment(name, config):
    """Run a single experiment with given configuration."""
    print("=" * 70)
    print(f"EXPERIMENT: {name}")
    print("=" * 70)
    print("\nKey Configuration:")
    key_vars = ["MODEL_RUN_DIR", "CALLS_ONLY", "SKIP_MONDAY", "SKIP_LAST_HOUR",
                "FAST_LOSS_CUT_ENABLED", "TRAIN_MIN_ABS_RET", "TT_MAX_HOLD_MINUTES"]
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


def run_all():
    """Run all experiments sequentially."""
    configs = [
        ("Optimized Day Trading", OPTIMIZED_CONFIG),
        ("Midday Focus (10-14h)", MIDDAY_FOCUS_CONFIG),
        ("Skip Mon+Fri", SKIP_MON_FRI_CONFIG),
        ("Hard Rules Only", HARD_RULES_ONLY_CONFIG),
    ]

    results = {}
    for name, config in configs:
        success = run_experiment(name, config)
        results[name] = "PASS" if success else "FAIL"
        print("\n")

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY - Check models/OPTIMIZED_*/SUMMARY.txt")
    print("=" * 70)
    for name, status in results.items():
        print(f"  [{status}] {name}")

    return results


def main():
    if "--optimized" in sys.argv:
        run_experiment("Optimized Day Trading", OPTIMIZED_CONFIG)
    elif "--midday" in sys.argv:
        run_experiment("Midday Focus", MIDDAY_FOCUS_CONFIG)
    elif "--skip-mon-fri" in sys.argv:
        run_experiment("Skip Mon+Fri", SKIP_MON_FRI_CONFIG)
    elif "--hard-rules" in sys.argv:
        run_experiment("Hard Rules Only", HARD_RULES_ONLY_CONFIG)
    else:
        run_all()


if __name__ == "__main__":
    main()
