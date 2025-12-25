#!/usr/bin/env python
"""
Run consensus controller V2 test with new technical indicators:
- MACD confirmation
- Bollinger Band position
- HMA trend alignment
- Market breadth (SPY/QQQ)
- RSI properly wired
"""
import os
import sys
import subprocess

# Set working directory
os.chdir(r"E:\gaussian\output3")

# Set environment
env = os.environ.copy()
env["ENTRY_CONTROLLER"] = "consensus"
env["TT_MAX_CYCLES"] = "1000"
env["TT_PRINT_EVERY"] = "50"
env["PAPER_TRADING"] = "True"
env["MODEL_RUN_DIR"] = "models/consensus_v2"

# Exit settings (same as tight test)
env["TT_XGB_MAX_HOLD_MINUTES"] = "30"
env["TT_XGB_SL"] = "-5.0"
env["TT_XGB_TP"] = "10.0"

print("=" * 60)
print("CONSENSUS CONTROLLER V2 - WITH TECHNICAL INDICATORS")
print("=" * 60)
print(f"ENTRY_CONTROLLER = {env.get('ENTRY_CONTROLLER')}")
print(f"TT_MAX_CYCLES = {env.get('TT_MAX_CYCLES')}")
print(f"TT_XGB_MAX_HOLD_MINUTES = {env.get('TT_XGB_MAX_HOLD_MINUTES')}")
print(f"TT_XGB_SL = {env.get('TT_XGB_SL')}%")
print(f"TT_XGB_TP = {env.get('TT_XGB_TP')}%")
print("=" * 60)
print("\n5 Signals Required (ALL must pass):")
print("  1. Timeframe: 2/3 must agree, confidence >= 15%")
print("  2. HMM: Trend must align with direction")
print("  3. Momentum: At least ONE of (5m, 15m) must align")
print("  4. Volatility: VIX 10-35, volume spike < 5x")
print("  5. Technical: MACD, BB position, HMA trend must confirm")
print("=" * 60)
sys.stdout.flush()

# Run the training script as a subprocess with the correct env
result = subprocess.run(
    [sys.executable, "scripts/train_time_travel.py"],
    env=env,
    cwd=r"E:\gaussian\output3",
)

sys.exit(result.returncode)
