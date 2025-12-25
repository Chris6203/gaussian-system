#!/usr/bin/env python3
"""
Gate Rejection Analysis Script

Runs training and tracks which gates reject signals most often.
Also tests different threshold configurations.
"""

import os
import sys
import subprocess
import json
from datetime import datetime

def run_test(name, env_vars, cycles=5000):
    """Run a single test with given environment variables"""
    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"Cycles: {cycles}")
    print(f"{'='*60}\n")

    # Set up environment
    env = os.environ.copy()
    env.update(env_vars)
    env['MODEL_RUN_DIR'] = f'models/gate_analysis_{name}'
    env['TT_MAX_CYCLES'] = str(cycles)
    env['TT_PRINT_EVERY'] = '500'
    env['PAPER_TRADING'] = 'True'

    # Run the training script
    cmd = ['python', 'scripts/train_time_travel.py']
    result = subprocess.run(cmd, env=env, capture_output=False)

    return result.returncode == 0

def main():
    """Run all gate analysis tests"""

    tests = {
        # A) Current setup with gate tracking (baseline)
        'baseline': {
            'HMM_STRONG_BULLISH': '0.70',
            'HMM_STRONG_BEARISH': '0.30',
            'HMM_MIN_CONFIDENCE': '0.70',
            'HMM_MAX_VOLATILITY': '0.70',
        },

        # B) Relaxed thresholds to capture more trades
        'relaxed_trend': {
            'HMM_STRONG_BULLISH': '0.65',  # Was 0.70
            'HMM_STRONG_BEARISH': '0.35',  # Was 0.30
            'HMM_MIN_CONFIDENCE': '0.70',
            'HMM_MAX_VOLATILITY': '0.70',
        },

        'relaxed_confidence': {
            'HMM_STRONG_BULLISH': '0.70',
            'HMM_STRONG_BEARISH': '0.30',
            'HMM_MIN_CONFIDENCE': '0.60',  # Was 0.70
            'HMM_MAX_VOLATILITY': '0.70',
        },

        'relaxed_volatility': {
            'HMM_STRONG_BULLISH': '0.70',
            'HMM_STRONG_BEARISH': '0.30',
            'HMM_MIN_CONFIDENCE': '0.70',
            'HMM_MAX_VOLATILITY': '0.80',  # Was 0.70
        },

        'relaxed_all': {
            'HMM_STRONG_BULLISH': '0.65',
            'HMM_STRONG_BEARISH': '0.35',
            'HMM_MIN_CONFIDENCE': '0.60',
            'HMM_MAX_VOLATILITY': '0.80',
        },
    }

    # Parse command line args
    if len(sys.argv) > 1:
        if sys.argv[1] == 'baseline':
            tests = {'baseline': tests['baseline']}
        elif sys.argv[1] == 'relaxed':
            tests = {k: v for k, v in tests.items() if k.startswith('relaxed')}
        elif sys.argv[1] in tests:
            tests = {sys.argv[1]: tests[sys.argv[1]]}

    cycles = int(sys.argv[2]) if len(sys.argv) > 2 else 5000

    print(f"Running {len(tests)} tests with {cycles} cycles each")

    results = {}
    for name, env_vars in tests.items():
        success = run_test(name, env_vars, cycles)
        results[name] = 'success' if success else 'failed'

        # Read summary if available
        summary_path = f'models/gate_analysis_{name}/SUMMARY.txt'
        if os.path.exists(summary_path):
            with open(summary_path) as f:
                print(f"\n--- {name} Summary ---")
                print(f.read())

    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    for name, status in results.items():
        print(f"  {name}: {status}")

if __name__ == '__main__':
    main()
