#!/usr/bin/env python3
"""Phase 17: 5-Minute Bars + Aligned Horizons + Tighter Risk/Reward

This test combines three architectural improvements:
1. 5-minute bars (reduces noise vs 1-minute)
2. 20-minute max hold (aligns with prediction horizon)
3. -5%/+15% exits (3:1 risk/reward ratio, break-even at 25% win rate)

Expected improvements:
- Less noise from 5m bars should improve signal quality
- Aligned horizons mean positions don't drift after prediction expires
- 3:1 ratio means we can profit even with 25-30% win rate
"""

import os
import sys
import subprocess

# Set environment variables for the subprocess
env = os.environ.copy()

# Enable PnL Calibration Gate (proven to reduce losses)
env['PNL_CAL_GATE'] = '1'
env['PNL_CAL_MIN_PROB'] = '0.40'
env['PNL_CAL_MIN_SAMPLES'] = '30'

# Test configuration
env['MODEL_RUN_DIR'] = 'models/phase17_5m_test'
env['TT_PRINT_EVERY'] = '500'
env['PAPER_TRADING'] = 'True'

print("=" * 70)
print("PHASE 17: 5-Minute Bars + Aligned Horizons + Tighter Exits")
print("=" * 70)
print()
print("Configuration Changes (from config.json):")
print("  Data Interval: 5m (was 1m) - reduces noise")
print("  Max Hold: 20 min (was 45 min) - aligns with prediction horizon")
print("  Stop Loss: -5% (was -8%) - tighter risk control")
print("  Take Profit: +15% (was +12%) - better risk/reward")
print("  Exit Ratio: 3:1 (was 1.5:1) - only need 25% win rate to break even")
print("  Sequence Length: 12 (was 30) - 12 5m bars = 60min lookback")
print()
print("  PnL Calibration Gate: ENABLED (40% min P(profit))")
print()
print("  Test Duration: 20,000 cycles (uses config.json)")
print("  Output: models/phase17_5m_test")
print()
print("Hypothesis: These changes should address the three major issues:")
print("  1. Horizon misalignment (prediction expires before exit)")
print("  2. Exit ratio doesn't work at current win rate")
print("  3. 1-minute data too noisy for prediction")
print("=" * 70)
sys.stdout.flush()

# Run the training script
result = subprocess.run(
    [sys.executable, 'scripts/train_time_travel.py'],
    cwd='E:/gaussian/output3',
    env=env
)
sys.exit(result.returncode)
