#!/usr/bin/env python3
"""
Phase 26: Combined Improvements Test

Tests three key improvements simultaneously:
1. FASTER LOSS CUTTING - Exit losers earlier (they're held 8x longer than winners!)
2. REGIME FILTERING - Only trade in favorable market conditions
3. ADAPTIVE POSITION SIZING - Reduce size during drawdowns

Usage:
    python experiments/run_phase26_improvements.py
"""

import os
import sys
import subprocess
from datetime import datetime

# Configuration for Phase 26 improvements
IMPROVEMENTS = {
    # 1. FASTER LOSS CUTTING
    # Current: Losers held 240min, Winners held 30min
    # Fix: Cut losses at 45min if losing, or if prediction flips
    "FAST_LOSS_CUT_MINUTES": "45",  # Exit losing trades after 45 min max
    "LOSS_CUT_THRESHOLD_PCT": "-3",  # Start considering exit if down 3%+

    # 2. REGIME FILTERING
    # Only trade when regime quality is good
    "TT_REGIME_FILTER": "3",  # Enable regime filter level 3
    "REGIME_MIN_QUALITY": "0.40",  # Minimum quality score to trade
    "REGIME_FULL_SIZE_QUALITY": "0.60",  # Quality for full position

    # 3. ADAPTIVE POSITION SIZING
    # Reduce size during drawdowns
    "DRAWDOWN_SCALE_ENABLED": "1",  # Enable drawdown-based scaling
    "DRAWDOWN_HALF_SIZE_PCT": "10",  # Half size after 10% drawdown
    "DRAWDOWN_QUARTER_SIZE_PCT": "20",  # Quarter size after 20% drawdown

    # Exit policy improvements
    "HARD_MAX_HOLD_MINUTES": "45",  # Reduced from default
    "TRAILING_STOP_ACTIVATION": "3",  # Activate trailing at +3%
    "TRAILING_STOP_DISTANCE": "1.5",  # 1.5% trailing distance

    # Standard test settings
    "MODEL_RUN_DIR": "models/phase26_combined_improvements",
    "TT_MAX_CYCLES": "10000",  # 10K cycles for validation
    "TT_PRINT_EVERY": "1000",
    "PAPER_TRADING": "True",
}

def run_test():
    """Run the Phase 26 test with all improvements enabled."""

    print("=" * 70)
    print("PHASE 26: Combined Improvements Test")
    print("=" * 70)
    print("\nImprovements being tested:")
    print("  1. FASTER LOSS CUTTING - Exit losers at 45min max")
    print("  2. REGIME FILTERING - Only trade in favorable conditions")
    print("  3. ADAPTIVE POSITION SIZING - Reduce size in drawdowns")
    print("\nEnvironment variables:")
    for key, value in IMPROVEMENTS.items():
        print(f"  {key}={value}")
    print()

    # Build environment
    env = os.environ.copy()
    env.update(IMPROVEMENTS)

    # Run training script
    cmd = [sys.executable, "scripts/train_time_travel.py"]

    print(f"Starting test at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 70)

    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=False,
            text=True
        )
        return result.returncode
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Test stopped by user")
        return 1


def run_baseline():
    """Run baseline test without improvements for comparison."""

    print("=" * 70)
    print("PHASE 26: Baseline Test (No Improvements)")
    print("=" * 70)

    env = os.environ.copy()
    env.update({
        "MODEL_RUN_DIR": "models/phase26_baseline",
        "TT_MAX_CYCLES": "10000",
        "TT_PRINT_EVERY": "1000",
        "PAPER_TRADING": "True",
    })

    cmd = [sys.executable, "scripts/train_time_travel.py"]

    print(f"Starting baseline at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 70)

    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=False,
            text=True
        )
        return result.returncode
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Test stopped by user")
        return 1


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Phase 26 Combined Improvements Test")
    parser.add_argument("--baseline", action="store_true", help="Run baseline instead of improvements")
    args = parser.parse_args()

    if args.baseline:
        sys.exit(run_baseline())
    else:
        sys.exit(run_test())
