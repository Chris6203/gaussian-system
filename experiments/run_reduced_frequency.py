#!/usr/bin/env python3
"""
Reduced Frequency Day Trading Tests

Target: 3-4 trades per day (vs current 18-24/day)

CALLS_ONLY had 232 trades in 5K cycles (18/day)
To get 3-4/day, need to reduce by ~5x

Tests different confidence thresholds and filters to reduce frequency
while maintaining profitability.
"""

import os
import sys
import subprocess
from datetime import datetime

# ============================================================
# BASE: CALLS_ONLY config (best performer at +434%)
# ============================================================
CALLS_ONLY_BASE = {
    "CALLS_ONLY": "1",
    "TT_MAX_HOLD_MINUTES": "45",
    "HARD_STOP_LOSS_PCT": "8",
    "HARD_TAKE_PROFIT_PCT": "12",
    "USE_TRAILING_STOP": "1",
    "TRAILING_ACTIVATION_PCT": "8",
    "TRAILING_STOP_PCT": "4",
    "ENABLE_TDA": "1",
    "PAPER_TRADING": "True",
    "TT_MAX_CYCLES": "5000",
    "TT_PRINT_EVERY": "500",
}

# ============================================================
# FREQUENCY REDUCTION TESTS
# ============================================================

# Test 1: Disable min confidence gate entirely, keep max at 30%
NO_MIN_CONF = {
    **CALLS_ONLY_BASE,
    "TT_TRAIN_MIN_CONF": "0.0",  # No minimum
    "TRAIN_MAX_CONF": "0.30",
    "MODEL_RUN_DIR": "models/FREQ_NO_MIN_CONF",
}

# Test 2: Min 20%, Max 35% (narrower band)
NARROW_BAND = {
    **CALLS_ONLY_BASE,
    "TT_TRAIN_MIN_CONF": "0.20",
    "TRAIN_MAX_CONF": "0.35",
    "MODEL_RUN_DIR": "models/FREQ_NARROW_BAND",
}

# Test 3: Very narrow band 22-28%
VERY_NARROW = {
    **CALLS_ONLY_BASE,
    "TT_TRAIN_MIN_CONF": "0.22",
    "TRAIN_MAX_CONF": "0.28",
    "MODEL_RUN_DIR": "models/FREQ_VERY_NARROW",
}

# Test 4: Higher minimum return threshold (0.15%)
MIN_RETURN = {
    **CALLS_ONLY_BASE,
    "TT_TRAIN_MIN_CONF": "0.20",
    "TRAIN_MAX_CONF": "0.40",
    "TRAIN_MIN_ABS_RET": "0.0015",  # 0.15% minimum edge
    "MODEL_RUN_DIR": "models/FREQ_MIN_RET",
}

# Test 5: Skip first 90 min + calls only (reduces by ~30%)
SKIP_MORNING = {
    **CALLS_ONLY_BASE,
    "TT_TRAIN_MIN_CONF": "0.20",
    "TRAIN_MAX_CONF": "0.40",
    "SKIP_FIRST_90_MIN": "1",
    "MODEL_RUN_DIR": "models/FREQ_SKIP_MORNING",
}

# Test 6: Skip morning + afternoon only (midday)
MIDDAY_CALLS = {
    **CALLS_ONLY_BASE,
    "TT_TRAIN_MIN_CONF": "0.20",
    "TRAIN_MAX_CONF": "0.40",
    "MIDDAY_ONLY": "1",
    "MODEL_RUN_DIR": "models/FREQ_MIDDAY_CALLS",
}

# Test 7: Higher min return (0.2%) - should dramatically reduce trades
HIGH_MIN_RET = {
    **CALLS_ONLY_BASE,
    "TT_TRAIN_MIN_CONF": "0.15",
    "TRAIN_MAX_CONF": "0.50",
    "TRAIN_MIN_ABS_RET": "0.002",  # 0.2% minimum edge
    "MODEL_RUN_DIR": "models/FREQ_HIGH_MIN_RET",
}

# Test 8: Combined: Min return + skip morning + narrow confidence
COMBINED = {
    **CALLS_ONLY_BASE,
    "TT_TRAIN_MIN_CONF": "0.18",
    "TRAIN_MAX_CONF": "0.35",
    "TRAIN_MIN_ABS_RET": "0.0012",  # 0.12% minimum edge
    "SKIP_FIRST_90_MIN": "1",
    "MODEL_RUN_DIR": "models/FREQ_COMBINED",
}


def run_experiment(name, config):
    """Run a single experiment with given configuration."""
    print("=" * 70)
    print(f"EXPERIMENT: {name}")
    print("=" * 70)
    print("\nKey Configuration:")
    key_vars = ["MODEL_RUN_DIR", "TRAIN_MAX_CONF", "TRAIN_MIN_ABS_RET",
                "SKIP_FIRST_90_MIN", "HMM_STRONG_BULLISH", "CALLS_ONLY"]
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
        ("No Min Conf", NO_MIN_CONF),
        ("Narrow Band (20-35%)", NARROW_BAND),
        ("Very Narrow (22-28%)", VERY_NARROW),
        ("Min Return 0.15%", MIN_RETURN),
        ("Skip Morning", SKIP_MORNING),
        ("Midday Calls", MIDDAY_CALLS),
        ("High Min Return 0.2%", HIGH_MIN_RET),
        ("Combined", COMBINED),
    ]

    results = {}
    for name, config in configs:
        success = run_experiment(name, config)
        results[name] = "PASS" if success else "FAIL"
        print("\n")

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY - Check models/FREQ_*/SUMMARY.txt")
    print("=" * 70)
    for name, status in results.items():
        print(f"  [{status}] {name}")

    return results


def main():
    if "--no-min" in sys.argv:
        run_experiment("No Min Conf", NO_MIN_CONF)
    elif "--narrow" in sys.argv:
        run_experiment("Narrow Band", NARROW_BAND)
    elif "--very-narrow" in sys.argv:
        run_experiment("Very Narrow", VERY_NARROW)
    elif "--min-ret" in sys.argv:
        run_experiment("Min Return", MIN_RETURN)
    elif "--skip-morning" in sys.argv:
        run_experiment("Skip Morning", SKIP_MORNING)
    elif "--midday" in sys.argv:
        run_experiment("Midday Calls", MIDDAY_CALLS)
    elif "--high-ret" in sys.argv:
        run_experiment("High Min Return", HIGH_MIN_RET)
    elif "--combined" in sys.argv:
        run_experiment("Combined", COMBINED)
    else:
        run_all()


if __name__ == "__main__":
    main()
