#!/usr/bin/env python3
"""Baseline + PnL Calibration Gate Test (20K Validation)

Combines:
1. Bandit baseline (proven +85% over 23K cycles - BEST VERIFIED)
2. PnL Calibration Gate (reduced losses from -87% to -2% in Phase 13)

Goal: Test if PnL calibration gate can improve upon the +85% baseline.
"""

import os
import sys
import subprocess

# Set environment variables for the subprocess
env = os.environ.copy()

# Use bandit (default) entry controller - the ONLY verified profitable config
# Bandit mode uses HMM-only entry strategy with strict trend thresholds

# Enable PnL Calibration Gate (Phase 13 improvement)
env['PNL_CAL_GATE'] = '1'
env['PNL_CAL_MIN_PROB'] = '0.40'  # Minimum 40% P(profit) to trade
env['PNL_CAL_MIN_SAMPLES'] = '30'  # Learn from first 30 trades

# RLThresholdLearner (Phase 9) - enabled via config.json

# Extended test for validation
env['MODEL_RUN_DIR'] = 'models/baseline_pnl_gate_20k'
env['TT_MAX_CYCLES'] = '20000'  # Full 20K validation
env['TT_PRINT_EVERY'] = '1000'
env['PAPER_TRADING'] = 'True'

print("=" * 70)
print("BASELINE + PnL CALIBRATION GATE - 20K VALIDATION")
print("=" * 70)
print()
print("Configuration:")
print(f"  Entry Controller: bandit (HMM-only, verified +85% over 23K)")
print(f"  PnL Calibration Gate: ENABLED")
print(f"    - Min P(profit): 40%")
print(f"    - Learning samples: 30")
print(f"  RLThresholdLearner: ENABLED (via config.json)")
print()
print(f"  Test Duration: 20,000 cycles")
print(f"  Output: models/baseline_pnl_gate_20k")
print()
print("Hypothesis: PnL gate will filter out losing trades while preserving")
print("the +85% profit potential of the bandit baseline.")
print("=" * 70)
sys.stdout.flush()

# Run the training script as a subprocess with proper environment
result = subprocess.run(
    [sys.executable, 'scripts/train_time_travel.py'],
    cwd='E:/gaussian/output3',
    env=env
)
sys.exit(result.returncode)
