#!/usr/bin/env python
"""
Run consensus controller test with proper environment.
"""
import os
import sys
import subprocess

# Set working directory
os.chdir(r"E:\gaussian\output3")

# Set environment
env = os.environ.copy()
env["ENTRY_CONTROLLER"] = "consensus"
env["TT_MAX_CYCLES"] = "500"
env["TT_PRINT_EVERY"] = "50"
env["PAPER_TRADING"] = "True"
env["MODEL_RUN_DIR"] = "models/consensus_test"

print("=" * 60)
print("CONSENSUS CONTROLLER TEST")
print("=" * 60)
print(f"ENTRY_CONTROLLER = {env.get('ENTRY_CONTROLLER')}")
print(f"TT_MAX_CYCLES = {env.get('TT_MAX_CYCLES')}")
print("=" * 60)
sys.stdout.flush()

# Run the training script as a subprocess with the correct env
result = subprocess.run(
    [sys.executable, "scripts/train_time_travel.py"],
    env=env,
    cwd=r"E:\gaussian\output3",
)

sys.exit(result.returncode)
