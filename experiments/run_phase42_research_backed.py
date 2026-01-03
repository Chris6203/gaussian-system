#!/usr/bin/env python3
"""Phase 42b: Research-backed experiments based on Codex findings"""
import subprocess
import os
import time

experiments = [
    # WIDER STOPS (50% vs current 8%) - research shows this is critical
    {
        "name": "phase42_wide_stop_50",
        "env": {
            "MODEL_RUN_DIR": "models/phase42_wide_stop_50",
            "HARD_STOP_LOSS_PCT": "50",
            "PAPER_TRADING": "True",
            "TT_MAX_CYCLES": "5000",
            "TT_PRINT_EVERY": "1000"
        }
    },
    # SMALLER TAKE PROFIT (10%) - research shows highest win rate
    {
        "name": "phase42_tp_10",
        "env": {
            "MODEL_RUN_DIR": "models/phase42_tp_10",
            "HARD_TAKE_PROFIT_PCT": "10",
            "PAPER_TRADING": "True",
            "TT_MAX_CYCLES": "5000",
            "TT_PRINT_EVERY": "1000"
        }
    },
    # WIDE STOP + SMALL TP (50% stop, 10% TP)
    {
        "name": "phase42_wide_stop_small_tp",
        "env": {
            "MODEL_RUN_DIR": "models/phase42_wide_stop_small_tp",
            "HARD_STOP_LOSS_PCT": "50",
            "HARD_TAKE_PROFIT_PCT": "10",
            "PAPER_TRADING": "True",
            "TT_MAX_CYCLES": "5000",
            "TT_PRINT_EVERY": "1000"
        }
    },
    # FOCUS ON FIRST 2 HOURS (opposite of my filter!)
    {
        "name": "phase42_first_2h",
        "env": {
            "MODEL_RUN_DIR": "models/phase42_first_2h",
            "TIME_ZONE_FILTER": "1",
            "TIME_ZONE_START_MINUTES": "0",  # Trade from open
            "TIME_ZONE_END_MINUTES": "210",  # Skip last 3.5 hours (only first 2h)
            "PAPER_TRADING": "True",
            "TT_MAX_CYCLES": "5000",
            "TT_PRINT_EVERY": "1000"
        }
    },
    # COMBO: Wide stop + small TP + inverted confidence
    {
        "name": "phase42_research_combo",
        "env": {
            "MODEL_RUN_DIR": "models/phase42_research_combo",
            "HARD_STOP_LOSS_PCT": "50",
            "HARD_TAKE_PROFIT_PCT": "10",
            "TRAIN_MAX_CONF": "0.25",
            "PAPER_TRADING": "True",
            "TT_MAX_CYCLES": "5000",
            "TT_PRINT_EVERY": "1000"
        }
    },
    # MODERATE: 25% stop, 15% TP (balanced approach)
    {
        "name": "phase42_balanced",
        "env": {
            "MODEL_RUN_DIR": "models/phase42_balanced",
            "HARD_STOP_LOSS_PCT": "25",
            "HARD_TAKE_PROFIT_PCT": "15",
            "PAPER_TRADING": "True",
            "TT_MAX_CYCLES": "5000",
            "TT_PRINT_EVERY": "1000"
        }
    },
]

def run_experiment(exp):
    """Run a single experiment"""
    env = os.environ.copy()
    env.update(exp["env"])
    
    print(f"Starting {exp['name']}...")
    proc = subprocess.Popen(
        ["python", "scripts/train_time_travel.py"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    return proc

if __name__ == "__main__":
    print("Phase 42b: Research-backed experiments")
    print("=" * 50)
    
    procs = []
    for exp in experiments:
        proc = run_experiment(exp)
        procs.append((exp["name"], proc))
        time.sleep(3)  # Stagger starts
    
    print(f"\nStarted {len(procs)} experiments in parallel")
    print("Monitoring progress...")
    
    while procs:
        for name, proc in procs[:]:
            ret = proc.poll()
            if ret is not None:
                print(f"\n{name} completed (exit code {ret})")
                procs.remove((name, proc))
        time.sleep(30)
    
    print("\nAll experiments complete!")
