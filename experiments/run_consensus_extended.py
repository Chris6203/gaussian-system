#!/usr/bin/env python
"""
Run extended consensus controller test (2500 cycles) with detailed stats.
"""
import os
import sys
import subprocess

# Set working directory
os.chdir(r"E:\gaussian\output3")

# Set environment
env = os.environ.copy()
env["ENTRY_CONTROLLER"] = "consensus"
env["TT_MAX_CYCLES"] = "2500"  # Extended test
env["TT_PRINT_EVERY"] = "100"  # Print every 100 cycles
env["PAPER_TRADING"] = "True"
env["MODEL_RUN_DIR"] = "models/consensus_extended"

print("=" * 60)
print("CONSENSUS CONTROLLER EXTENDED TEST (2500 cycles)")
print("=" * 60)
print(f"ENTRY_CONTROLLER = {env.get('ENTRY_CONTROLLER')}")
print(f"TT_MAX_CYCLES = {env.get('TT_MAX_CYCLES')}")
print(f"TT_PRINT_EVERY = {env.get('TT_PRINT_EVERY')}")
print("=" * 60)
print("\nThresholds (tuned):")
print("  - Timeframe: 2/3 must agree, confidence >= 15%")
print("  - HMM: bullish > 0.55, bearish < 0.45, confidence >= 10%")
print("  - Momentum: at least ONE of (5m, 15m) must align (OR logic)")
print("  - RSI: only block extreme (>80 or <20)")
print("  - VIX: 10-35 range, volume spike < 5x")
print("=" * 60)
sys.stdout.flush()

# Run the training script as a subprocess with the correct env
result = subprocess.run(
    [sys.executable, "scripts/train_time_travel.py"],
    env=env,
    cwd=r"E:\gaussian\output3",
)

sys.exit(result.returncode)
