#!/usr/bin/env python3
"""Baseline + PnL Calibration Gate - 20K Extended Validation

Phase 16 showed excellent results (5K):
- P&L: +$19.66 (+0.39%)
- Win Rate: 46.7% (HIGHEST EVER!)
- Per-Trade P&L: +$0.07

This test validates over 20K cycles to confirm robustness.
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

# Extended test for validation - uses config.json max_cycles (20000)
env['MODEL_RUN_DIR'] = 'models/baseline_pnl_gate_20k_validation'
env['TT_PRINT_EVERY'] = '2000'
env['PAPER_TRADING'] = 'True'

print("=" * 70)
print("PHASE 16 VALIDATION: Baseline + PnL Calibration Gate - 20K Cycles")
print("=" * 70)
print()
print("Phase 16 (5K) Results to Validate:")
print("  - P&L: +$19.66 (+0.39%)")
print("  - Win Rate: 46.7% (HIGHEST EVER)")
print("  - Per-Trade P&L: +$0.07")
print()
print("Configuration:")
print("  Entry Controller: bandit (HMM-only)")
print("  PnL Calibration Gate: ENABLED (40% min P(profit))")
print("  RLThresholdLearner: ENABLED")
print()
print("  Test Duration: 20,000 cycles (from config.json)")
print("  Output: models/baseline_pnl_gate_20k_validation")
print()
print("Goal: Confirm the +0.39% P&L and 46.7% win rate hold over extended period.")
print("=" * 70)
sys.stdout.flush()

# Run the training script as a subprocess with proper environment
result = subprocess.run(
    [sys.executable, 'scripts/train_time_travel.py'],
    cwd='E:/gaussian/output3',
    env=env
)
sys.exit(result.returncode)
