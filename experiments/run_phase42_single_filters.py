#!/usr/bin/env python3
"""Phase 42: Single-filter experiments to find best individual filter"""
import subprocess
import os
import time

experiments = [
    # Baseline - no filters
    {
        "name": "phase42_baseline",
        "env": {
            "MODEL_RUN_DIR": "models/phase42_baseline",
            "PAPER_TRADING": "True",
            "TT_MAX_CYCLES": "5000",
            "TT_PRINT_EVERY": "1000"
        }
    },
    # Inverted confidence - proven best from Phase 40
    {
        "name": "phase42_inv_conf_25",
        "env": {
            "MODEL_RUN_DIR": "models/phase42_inv_conf_25",
            "TRAIN_MAX_CONF": "0.25",
            "PAPER_TRADING": "True",
            "TT_MAX_CYCLES": "5000",
            "TT_PRINT_EVERY": "1000"
        }
    },
    # VIX Goldilocks with WIDER range (12-28)
    {
        "name": "phase42_vix_wide",
        "env": {
            "MODEL_RUN_DIR": "models/phase42_vix_wide",
            "VIX_GOLDILOCKS_FILTER": "1",
            "VIX_GOLDILOCKS_MIN": "12.0",
            "VIX_GOLDILOCKS_MAX": "28.0",
            "PAPER_TRADING": "True",
            "TT_MAX_CYCLES": "5000",
            "TT_PRINT_EVERY": "1000"
        }
    },
    # Time zone RELAXED (30m start, 15m end)
    {
        "name": "phase42_tz_relaxed",
        "env": {
            "MODEL_RUN_DIR": "models/phase42_tz_relaxed",
            "TIME_ZONE_FILTER": "1",
            "TIME_ZONE_START_MINUTES": "30",
            "TIME_ZONE_END_MINUTES": "15",
            "PAPER_TRADING": "True",
            "TT_MAX_CYCLES": "5000",
            "TT_PRINT_EVERY": "1000"
        }
    },
    # Day of week (Tue-Thu only)
    {
        "name": "phase42_dow_only",
        "env": {
            "MODEL_RUN_DIR": "models/phase42_dow_only",
            "DAY_OF_WEEK_FILTER": "1",
            "SKIP_MONDAY": "1",
            "SKIP_FRIDAY": "1",
            "PAPER_TRADING": "True",
            "TT_MAX_CYCLES": "5000",
            "TT_PRINT_EVERY": "1000"
        }
    },
    # Momentum confirmation only
    {
        "name": "phase42_momentum",
        "env": {
            "MODEL_RUN_DIR": "models/phase42_momentum",
            "MOMENTUM_CONFIRM_FILTER": "1",
            "MOMENTUM_MIN_STRENGTH": "0.0005",
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
    print("Phase 42: Single-filter experiments")
    print("=" * 50)
    
    procs = []
    for exp in experiments:
        proc = run_experiment(exp)
        procs.append((exp["name"], proc))
        time.sleep(2)  # Stagger starts
    
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
