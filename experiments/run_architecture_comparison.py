#!/usr/bin/env python3
"""
Architecture Comparison Tests

Run multiple configurations in parallel to find improvements.
"""

import subprocess
import sys
import os
from datetime import datetime

# Base configuration (common to all tests)
BASE_ENV = {
    "PAPER_TRADING": "True",
    "TT_MAX_CYCLES": "5000",
    "TT_PRINT_EVERY": "500",
}

# Test configurations to compare
TESTS = {
    # Test 1: Current best (baseline for comparison)
    "baseline_v3": {
        "PREDICTOR_ARCH": "v3_multi_horizon",
        "MODEL_RUN_DIR": "models/arch_compare_baseline",
    },

    # Test 2: V3 with reduced features (disable extended macro)
    "v3_core_features": {
        "PREDICTOR_ARCH": "v3_multi_horizon",
        "ENABLE_EXTENDED_MACRO": "0",
        "MODEL_RUN_DIR": "models/arch_compare_core_features",
    },

    # Test 3: Single 5m horizon (faster predictions)
    "v3_5m_only": {
        "PREDICTOR_ARCH": "v3_multi_horizon",
        "V3_DEFAULT_HORIZON": "5",
        "MODEL_RUN_DIR": "models/arch_compare_5m_only",
    },

    # Test 4: V3 + RSI/MACD filter
    "v3_rsi_macd": {
        "PREDICTOR_ARCH": "v3_multi_horizon",
        "RSI_MACD_FILTER": "1",
        "RSI_OVERSOLD": "30",
        "RSI_OVERBOUGHT": "70",
        "MACD_CONFIRM": "1",
        "MODEL_RUN_DIR": "models/arch_compare_rsi_macd",
    },

    # Test 5: Tighter stop loss (5% instead of 8%)
    "v3_tight_stop": {
        "PREDICTOR_ARCH": "v3_multi_horizon",
        "TT_STOP_LOSS_PCT": "5",
        "MODEL_RUN_DIR": "models/arch_compare_tight_stop",
    },

    # Test 6: Wider take profit (15% instead of 12%)
    "v3_wide_tp": {
        "PREDICTOR_ARCH": "v3_multi_horizon",
        "TT_TAKE_PROFIT_PCT": "15",
        "MODEL_RUN_DIR": "models/arch_compare_wide_tp",
    },
}

def run_test(name: str, config: dict):
    """Run a single test configuration."""
    env = os.environ.copy()
    env.update(BASE_ENV)
    env.update(config)

    print(f"\n{'='*60}")
    print(f"Starting test: {name}")
    print(f"Config: {config}")
    print(f"{'='*60}")

    result = subprocess.run(
        [sys.executable, "scripts/train_time_travel.py"],
        env=env,
        capture_output=False,
    )

    return result.returncode

def main():
    """Run all tests sequentially."""
    print(f"\nArchitecture Comparison Tests - {datetime.now()}")
    print(f"Running {len(TESTS)} test configurations...")

    results = {}
    for name, config in TESTS.items():
        returncode = run_test(name, config)
        results[name] = "SUCCESS" if returncode == 0 else "FAILED"

    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    for name, status in results.items():
        print(f"  {name}: {status}")

    print("\nCheck models/arch_compare_*/SUMMARY.txt for results")
    print("Run: python tools/experiments_db.py scan && python tools/experiments_db.py leaderboard")

if __name__ == "__main__":
    main()
