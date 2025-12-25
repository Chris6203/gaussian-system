#!/usr/bin/env python3
"""Test V2 RL Architecture

This test compares the new V2 architecture with baseline:
- V2 uses 40-dim state (embedding + direction probs + execution heads)
- V2 uses GRU for temporal awareness
- V2 uses EV-based entry gating
- V2 has EXIT_FAST and EXIT_PATIENT actions

Test Configuration:
- 5000 cycles for initial validation
- Uses V3 multi-horizon predictor
- Compares against bandit baseline
"""

import os
import sys
import subprocess

# Set environment variables
env = os.environ.copy()

# Use V2 architecture
env['USE_RL_POLICY_V2'] = '1'
env['USE_PREDICTOR_V3'] = '1'  # Multi-horizon predictor

# Test configuration
env['MODEL_RUN_DIR'] = 'models/v2_arch_test'
env['TT_MAX_CYCLES'] = '5000'
env['TT_PRINT_EVERY'] = '500'
env['PAPER_TRADING'] = 'True'

# Keep standard 1m config
# interval: 1m, stop: -8%, TP: +12%, hold: 45min

print("=" * 70)
print("V2 ARCHITECTURE TEST")
print("=" * 70)
print()
print("Architecture Changes:")
print("  1. State: 40 features (was 18)")
print("     - Includes 16-dim compressed predictor embedding")
print("     - Includes direction probs [DOWN, NEUTRAL, UP]")
print("     - Includes execution heads (fillability, slippage, ttf)")
print()
print("  2. Temporal: GRU over last 10 states")
print("     - Can learn patterns like 'confidence dropping for 3 steps'")
print()
print("  3. Entry: EV-based gating (was confidence threshold)")
print("     - EV = P(up)*return - P(down)*return - friction")
print("     - Only trades when EV > 0")
print()
print("  4. Exit: EXIT_FAST + EXIT_PATIENT (was single EXIT)")
print("     - EXIT_FAST: Market order, immediate")
print("     - EXIT_PATIENT: Limit order, wait for better fill")
print()
print("  5. Predictor: Multi-horizon (5m, 15m, 30m, 45m)")
print("     - Can pick horizon with best edge")
print()
print("Test Duration: 5000 cycles")
print("Output: models/v2_arch_test")
print("=" * 70)
sys.stdout.flush()

# Run the training script
result = subprocess.run(
    [sys.executable, 'scripts/train_time_travel.py'],
    cwd='E:/gaussian/output3',
    env=env
)
sys.exit(result.returncode)
