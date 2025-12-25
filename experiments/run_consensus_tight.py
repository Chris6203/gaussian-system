#!/usr/bin/env python
"""
Run consensus controller test with TIGHTER exit conditions.
- Shorter max hold time (30 min instead of 90 min)
- Tighter stop loss (5% instead of 8%)
- More frequent position checks
"""
import os
import sys
import subprocess

# Set working directory
os.chdir(r"E:\gaussian\output3")

# Set environment
env = os.environ.copy()
env["ENTRY_CONTROLLER"] = "consensus"
env["TT_MAX_CYCLES"] = "1000"  # Moderate test
env["TT_PRINT_EVERY"] = "50"   # Print every 50 cycles
env["PAPER_TRADING"] = "True"
env["MODEL_RUN_DIR"] = "models/consensus_tight"

# Tighter exit settings
env["TT_XGB_MAX_HOLD_MINUTES"] = "30"  # Max 30 min hold (force exits)
env["TT_XGB_SL"] = "-5.0"  # Tighter stop loss: -5%
env["TT_XGB_TP"] = "10.0"  # Tighter take profit: +10%

print("=" * 60)
print("CONSENSUS CONTROLLER - TIGHT EXITS TEST")
print("=" * 60)
print(f"ENTRY_CONTROLLER = {env.get('ENTRY_CONTROLLER')}")
print(f"TT_MAX_CYCLES = {env.get('TT_MAX_CYCLES')}")
print(f"TT_XGB_MAX_HOLD_MINUTES = {env.get('TT_XGB_MAX_HOLD_MINUTES')}")
print(f"TT_XGB_SL = {env.get('TT_XGB_SL')}%")
print(f"TT_XGB_TP = {env.get('TT_XGB_TP')}%")
print("=" * 60)
print("\nEntry thresholds (consensus):")
print("  - Timeframe: 2/3 must agree, confidence >= 15%")
print("  - HMM alignment required")
print("  - Momentum: at least ONE of (5m, 15m) must align")
print("  - RSI: only block extreme (>80 or <20)")
print("=" * 60)
sys.stdout.flush()

# Run the training script as a subprocess with the correct env
result = subprocess.run(
    [sys.executable, "scripts/train_time_travel.py"],
    env=env,
    cwd=r"E:\gaussian\output3",
)

sys.exit(result.returncode)
