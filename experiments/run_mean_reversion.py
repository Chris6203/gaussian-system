#!/usr/bin/env python3
"""
Mean Reversion Strategy Test

KEY FINDING: The neural network's high confidence signals are NOISE!

Analysis of 4638 CALL signals showed:
- High confidence (>=25%) + Positive momentum: 1.6% win rate (TERRIBLE)
- Low confidence (<22%) + High volume + Mean reversion: 9.8% win rate (BEST)

The bot should trade AGAINST the neural network's confidence:
- Trade when model is UNCERTAIN (low confidence)
- Trade when there's REAL activity (high volume)
- Trade AGAINST recent momentum (mean reversion)

This flips the intuitive logic on its head!
"""

import os
import sys
import subprocess
from datetime import datetime

# ============================================================
# MEAN REVERSION CONFIG
# Key insight: Low confidence + Mean reversion = better trades
# ============================================================

BASE_CONFIG = {
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

# Test 1: Inverse confidence - trade when model is LESS confident
# This catches mean reversion opportunities the model misses
MEAN_REVERSION_V1 = {
    **BASE_CONFIG,
    # FLIP THE CONFIDENCE LOGIC
    # Instead of requiring MIN confidence, require MAX confidence
    # (trade when model is uncertain, not when it's sure)
    "TT_TRAIN_MIN_CONF": "0.0",   # No minimum - allow uncertain signals
    "TRAIN_MAX_CONF": "0.22",     # But cap at 22% - avoid overconfident signals!

    # Higher minimum return to filter noise
    "TRAIN_MIN_ABS_RET": "0.001",  # 0.1% minimum edge

    # Require volume spike (real market activity)
    "MIN_VOLUME_SPIKE": "1.0",

    "MODEL_RUN_DIR": "models/MEAN_REVERSION_V1",
}

# Test 2: Moderate settings
MEAN_REVERSION_V2 = {
    **BASE_CONFIG,
    "TT_TRAIN_MIN_CONF": "0.0",
    "TRAIN_MAX_CONF": "0.25",     # Slightly wider
    "TRAIN_MIN_ABS_RET": "0.0008",
    "MIN_VOLUME_SPIKE": "0.8",
    "MODEL_RUN_DIR": "models/MEAN_REVERSION_V2",
}

# Test 3: Add momentum filter (mean reversion: buy on dips)
MEAN_REVERSION_MOMENTUM = {
    **BASE_CONFIG,
    "TT_TRAIN_MIN_CONF": "0.0",
    "TRAIN_MAX_CONF": "0.22",
    "TRAIN_MIN_ABS_RET": "0.001",
    "MIN_VOLUME_SPIKE": "1.0",
    # Buy calls when momentum is negative (price dipping - mean reversion)
    "MEAN_REVERSION_MODE": "1",   # Enable mean reversion: buy calls on dips
    "MODEL_RUN_DIR": "models/MEAN_REVERSION_MOMENTUM",
}


def run_experiment(name, config):
    """Run a single experiment with given configuration."""
    print("=" * 70)
    print(f"EXPERIMENT: {name}")
    print("=" * 70)
    print("\nKey Configuration (MEAN REVERSION):")
    key_vars = ["MODEL_RUN_DIR", "TRAIN_MAX_CONF", "TT_TRAIN_MIN_CONF",
                "TRAIN_MIN_ABS_RET", "MIN_VOLUME_SPIKE", "MEAN_REVERSION_MODE"]
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
        ("Mean Reversion V1 (Strict)", MEAN_REVERSION_V1),
        ("Mean Reversion V2 (Moderate)", MEAN_REVERSION_V2),
        ("Mean Reversion + Momentum", MEAN_REVERSION_MOMENTUM),
    ]

    results = {}
    for name, config in configs:
        success = run_experiment(name, config)
        results[name] = "PASS" if success else "FAIL"
        print("\n")

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    for name, status in results.items():
        print(f"  [{status}] {name}")

    return results


def main():
    if "--v1" in sys.argv:
        run_experiment("Mean Reversion V1", MEAN_REVERSION_V1)
    elif "--v2" in sys.argv:
        run_experiment("Mean Reversion V2", MEAN_REVERSION_V2)
    elif "--momentum" in sys.argv:
        run_experiment("Mean Reversion + Momentum", MEAN_REVERSION_MOMENTUM)
    else:
        run_all()


if __name__ == "__main__":
    main()
