#!/usr/bin/env python3
"""
Phase 22: Controlled Exit Experiments

Tests single changes against Phase 16 baseline:
- A1: TP 10% (was 12%)
- A2: max_hold 30min (was 20min)
- B1: min_confidence 0.65 (was 0.55)
- B2: HMM threshold 0.75 (was 0.70)

Usage:
    python experiments/run_phase22_controlled.py A1    # Test lower TP
    python experiments/run_phase22_controlled.py A2    # Test longer hold
    python experiments/run_phase22_controlled.py B1    # Test higher confidence
    python experiments/run_phase22_controlled.py B2    # Test stricter HMM
    python experiments/run_phase22_controlled.py all   # Run all experiments
"""

import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).parent.parent

# Phase 16 baseline settings (proven stable: -4.34% over 20K cycles)
# Environment variables from backend/exit_config.py and train_time_travel.py
BASELINE = {
    "MODEL_RUN_DIR": "models/phase22_baseline",
    "TT_MAX_CYCLES": "5000",
    "TT_PRINT_EVERY": "500",
    "PAPER_TRADING": "True",
    "PNL_CAL_GATE": "1",
    "PNL_CAL_MIN_PROB": "0.40",
    # Default exits (Phase 16) - these env vars are read by exit_config.py
    "TT_STOP_LOSS_PCT": "-8",      # backend/exit_config.py line 60
    "TT_TAKE_PROFIT_PCT": "12",    # backend/exit_config.py line 63
    "TT_MAX_HOLD_MINUTES": "45",   # backend/exit_config.py line 66, train_time_travel.py line 80
}

EXPERIMENTS = {
    "baseline": {
        "description": "Phase 16 baseline (PnL Cal Gate)",
        "env": {}  # No changes
    },
    "A1": {
        "description": "TP 10% (from 12%) - more reachable target",
        "env": {"TT_TAKE_PROFIT_PCT": "10"}
    },
    "A2": {
        "description": "max_hold 30min (from 45min) - align with best win rate window",
        "env": {"TT_MAX_HOLD_MINUTES": "30"}
    },
    "B1": {
        "description": "min_confidence 0.65 (from 0.55) - higher quality signals",
        "env": {"MIN_CONFIDENCE_TO_TRADE": "0.65"}
    },
    "B2": {
        "description": "HMM threshold 0.75/0.25 (from 0.70/0.30) - stronger trend",
        "env": {"HMM_STRONG_BULLISH": "0.75", "HMM_STRONG_BEARISH": "0.25"}
    },
}


def run_experiment(exp_name: str):
    """Run a single experiment."""
    if exp_name not in EXPERIMENTS:
        print(f"Unknown experiment: {exp_name}")
        print(f"Available: {list(EXPERIMENTS.keys())}")
        return None

    exp = EXPERIMENTS[exp_name]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"models/phase22_{exp_name}_{timestamp}"

    print("=" * 70)
    print(f"PHASE 22 EXPERIMENT: {exp_name}")
    print(f"Description: {exp['description']}")
    print(f"Run Directory: {run_dir}")
    print("=" * 70)

    # Build environment
    env = os.environ.copy()
    for key, value in BASELINE.items():
        env[key] = value
    env["MODEL_RUN_DIR"] = str(BASE_DIR / run_dir)

    # Apply experiment-specific changes
    for key, value in exp.get("env", {}).items():
        env[key] = str(value)
        print(f"  Override: {key}={value}")

    print("\nStarting experiment...")

    # Run training
    result = subprocess.run(
        [sys.executable, str(BASE_DIR / "scripts" / "train_time_travel.py")],
        env=env,
        cwd=str(BASE_DIR),
    )

    return result.returncode


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    exp_name = sys.argv[1]

    if exp_name == "all":
        for name in EXPERIMENTS:
            run_experiment(name)
    else:
        run_experiment(exp_name)


if __name__ == "__main__":
    main()
