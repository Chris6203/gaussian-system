#!/usr/bin/env python3
"""
Win Rate Improvement Tests for SKIP_MONDAY

Based on Codex learnings and historical analysis, tests combining
the best SKIP_MONDAY config with win rate improvements.

Baseline SKIP_MONDAY (20K): 43.0% WR, +1630% P&L

Improvements to test:
1. Tight TP (8%) - +2.4pp improvement in Phase 30
2. RSI+MACD filter (30/70) - 40.1% WR in Phase 31
3. Transformer encoder - best OOS generalization
4. Combined: All improvements together

Usage:
    python experiments/run_winrate_improvements.py           # Run all
    python experiments/run_winrate_improvements.py --quick   # 5K each
"""

import os
import sys
import subprocess
from datetime import datetime

# ============================================================
# BASE CONFIGURATION (SKIP_MONDAY best config)
# ============================================================
SKIP_MONDAY_BASE = {
    # Day-of-week filtering (skip 23% WR Monday)
    "DAY_OF_WEEK_FILTER": "1",
    "SKIP_MONDAY": "1",
    "SKIP_FRIDAY": "0",

    # Trailing stop (locks in profits)
    "USE_TRAILING_STOP": "1",
    "TRAILING_ACTIVATION_PCT": "10",
    "TRAILING_STOP_PCT": "5",

    # Trade Direction Analysis
    "ENABLE_TDA": "1",
    "TDA_REGIME_FILTER": "1",

    # Low confidence filter (counter-intuitive winner)
    "TRAIN_MAX_CONF": "0.25",

    # Standard settings
    "PAPER_TRADING": "True",
}

# ============================================================
# WIN RATE IMPROVEMENT TESTS
# ============================================================

# Test 1: Baseline (for comparison)
BASELINE = {
    **SKIP_MONDAY_BASE,
    "MODEL_RUN_DIR": "models/WR_BASELINE",
    "TT_MAX_CYCLES": "10000",
    "TT_PRINT_EVERY": "1000",
}

# Test 2: + Tight Take Profit (8%) - Phase 30 showed +2.4pp WR improvement
TIGHT_TP = {
    **SKIP_MONDAY_BASE,
    "TT_TAKE_PROFIT_PCT": "8",
    "MODEL_RUN_DIR": "models/WR_TIGHT_TP",
    "TT_MAX_CYCLES": "10000",
    "TT_PRINT_EVERY": "1000",
}

# Test 3: + RSI+MACD filter (30/70) - Phase 31 showed 40.1% WR
RSI_MACD = {
    **SKIP_MONDAY_BASE,
    "RSI_MACD_FILTER": "1",
    "RSI_OVERSOLD": "30",
    "RSI_OVERBOUGHT": "70",
    "MACD_CONFIRM": "1",
    "MODEL_RUN_DIR": "models/WR_RSI_MACD",
    "TT_MAX_CYCLES": "10000",
    "TT_PRINT_EVERY": "1000",
}

# Test 4: + Transformer encoder - best OOS generalization (LEARN-008)
TRANSFORMER = {
    **SKIP_MONDAY_BASE,
    "TEMPORAL_ENCODER": "transformer",
    "MODEL_RUN_DIR": "models/WR_TRANSFORMER",
    "TT_MAX_CYCLES": "10000",
    "TT_PRINT_EVERY": "1000",
}

# Test 5: + Early TP (6%) - Phase 30 best WR at 39.6%
EARLY_TP = {
    **SKIP_MONDAY_BASE,
    "TT_TAKE_PROFIT_PCT": "6",
    "MODEL_RUN_DIR": "models/WR_EARLY_TP",
    "TT_MAX_CYCLES": "10000",
    "TT_PRINT_EVERY": "1000",
}

# Test 6: Combined Best (Tight TP + RSI+MACD + Transformer)
COMBINED = {
    **SKIP_MONDAY_BASE,
    "TT_TAKE_PROFIT_PCT": "8",
    "RSI_MACD_FILTER": "1",
    "RSI_OVERSOLD": "30",
    "RSI_OVERBOUGHT": "70",
    "MACD_CONFIRM": "1",
    "TEMPORAL_ENCODER": "transformer",
    "MODEL_RUN_DIR": "models/WR_COMBINED",
    "TT_MAX_CYCLES": "10000",
    "TT_PRINT_EVERY": "1000",
}

# Test 7: Skip Tuesday too (Monday 23%, Tuesday lower than Wed-Thu)
SKIP_MON_TUE = {
    **SKIP_MONDAY_BASE,
    "SKIP_TUESDAY": "1",
    "MODEL_RUN_DIR": "models/WR_SKIP_MON_TUE",
    "TT_MAX_CYCLES": "10000",
    "TT_PRINT_EVERY": "1000",
}

# Test 8: Higher confidence threshold (35% instead of 25%)
HIGHER_CONF = {
    **SKIP_MONDAY_BASE,
    "TRAIN_MAX_CONF": "0.35",
    "MODEL_RUN_DIR": "models/WR_HIGHER_CONF",
    "TT_MAX_CYCLES": "10000",
    "TT_PRINT_EVERY": "1000",
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
    key_vars = ["MODEL_RUN_DIR", "TT_MAX_CYCLES", "TT_TAKE_PROFIT_PCT",
                "RSI_MACD_FILTER", "TEMPORAL_ENCODER", "TRAIN_MAX_CONF"]
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
        ("Baseline (SKIP_MONDAY)", BASELINE),
        ("+ Tight TP (8%)", TIGHT_TP),
        ("+ Early TP (6%)", EARLY_TP),
        ("+ RSI+MACD (30/70)", RSI_MACD),
        ("+ Transformer Encoder", TRANSFORMER),
        ("+ Combined (TP+RSI+Transformer)", COMBINED),
        ("+ Skip Mon+Tue", SKIP_MON_TUE),
        ("+ Higher Conf (35%)", HIGHER_CONF),
    ]

    results = {}
    for name, config in configs:
        success = run_experiment(name, config, quick=quick)
        results[name] = "PASS" if success else "FAIL"
        print("\n")

    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY - Check models/WR_*/SUMMARY.txt for details")
    print("=" * 70)
    for name, status in results.items():
        print(f"  [{status}] {name}")

    return results


def main():
    quick = "--quick" in sys.argv

    if "--baseline" in sys.argv:
        run_experiment("Baseline", BASELINE, quick=quick)
    elif "--tight-tp" in sys.argv:
        run_experiment("Tight TP", TIGHT_TP, quick=quick)
    elif "--early-tp" in sys.argv:
        run_experiment("Early TP", EARLY_TP, quick=quick)
    elif "--rsi" in sys.argv:
        run_experiment("RSI+MACD", RSI_MACD, quick=quick)
    elif "--transformer" in sys.argv:
        run_experiment("Transformer", TRANSFORMER, quick=quick)
    elif "--combined" in sys.argv:
        run_experiment("Combined", COMBINED, quick=quick)
    else:
        run_all(quick=quick)


if __name__ == "__main__":
    main()
