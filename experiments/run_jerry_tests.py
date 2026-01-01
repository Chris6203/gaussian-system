#!/usr/bin/env python3
"""
Jerry Features Experiment Runner

Tests four configurations:
1. Baseline (no Jerry features, no filter)
2. Jerry Features Only (feed features to NN)
3. Jerry Filter Only (use F_t as confirmation)
4. Both Features + Filter (combined)

Usage:
    python experiments/run_jerry_tests.py
    python experiments/run_jerry_tests.py --cycles 5000
"""

import os
import sys
import subprocess
import time
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configuration
DEFAULT_CYCLES = 5000
PRINT_EVERY = 500

# Test configurations
TESTS = {
    'baseline': {
        'name': 'Baseline (No Jerry)',
        'env': {
            'JERRY_FEATURES': '0',
            'JERRY_FILTER': '0',
        },
        'description': 'Standard V3 predictor without Jerry enhancements',
    },
    'features_only': {
        'name': 'Jerry Features Only',
        'env': {
            'JERRY_FEATURES': '1',
            'JERRY_FILTER': '0',
        },
        'description': 'Jerry fuzzy scores fed as additional NN inputs',
    },
    'filter_only': {
        'name': 'Jerry Filter Only',
        'env': {
            'JERRY_FEATURES': '0',
            'JERRY_FILTER': '1',
            'JERRY_FILTER_THRESHOLD': '0.5',
        },
        'description': 'Jerry F_t score used as confirmation filter (>0.5)',
    },
    'both': {
        'name': 'Features + Filter',
        'env': {
            'JERRY_FEATURES': '1',
            'JERRY_FILTER': '1',
            'JERRY_FILTER_THRESHOLD': '0.5',
        },
        'description': 'Both Jerry features and filter enabled',
    },
}


def run_test(test_id: str, config: dict, cycles: int) -> dict:
    """Run a single test configuration."""
    print(f"\n{'='*70}")
    print(f"TEST: {config['name']}")
    print(f"Description: {config['description']}")
    print(f"{'='*70}\n")

    # Set up environment
    env = os.environ.copy()
    env.update(config['env'])
    env['MODEL_RUN_DIR'] = f"models/jerry_test_{test_id}"
    env['TT_MAX_CYCLES'] = str(cycles)
    env['TT_PRINT_EVERY'] = str(PRINT_EVERY)
    env['PAPER_TRADING'] = 'True'

    # Run training
    start_time = time.time()
    result = subprocess.run(
        ['python', 'scripts/train_time_travel.py'],
        env=env,
        capture_output=True,
        text=True,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    elapsed = time.time() - start_time

    # Parse results from output
    output = result.stdout + result.stderr

    # Extract key metrics
    metrics = {
        'test_id': test_id,
        'name': config['name'],
        'elapsed_seconds': elapsed,
        'return_code': result.returncode,
    }

    # Parse P&L
    for line in output.split('\n'):
        if 'Final P&L' in line or 'P&L:' in line:
            try:
                # Look for percentage
                if '%' in line:
                    import re
                    match = re.search(r'([+-]?\d+\.?\d*)%', line)
                    if match:
                        metrics['pnl_pct'] = float(match.group(1))
            except:
                pass

        if 'Win Rate' in line or 'win rate' in line.lower():
            try:
                import re
                match = re.search(r'(\d+\.?\d*)%', line)
                if match:
                    metrics['win_rate'] = float(match.group(1))
            except:
                pass

        if 'Total Trades' in line or 'trades:' in line.lower():
            try:
                import re
                match = re.search(r'(\d+)', line)
                if match:
                    metrics['total_trades'] = int(match.group(1))
            except:
                pass

    # Print summary
    print(f"\n--- {config['name']} Results ---")
    print(f"Elapsed: {elapsed:.1f}s")
    print(f"P&L: {metrics.get('pnl_pct', 'N/A')}%")
    print(f"Win Rate: {metrics.get('win_rate', 'N/A')}%")
    print(f"Trades: {metrics.get('total_trades', 'N/A')}")

    if result.returncode != 0:
        print(f"WARNING: Test exited with code {result.returncode}")
        print("STDERR:", result.stderr[-500:] if result.stderr else "None")

    return metrics


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Run Jerry features experiments')
    parser.add_argument('--cycles', type=int, default=DEFAULT_CYCLES,
                       help=f'Number of cycles per test (default: {DEFAULT_CYCLES})')
    parser.add_argument('--test', type=str, choices=list(TESTS.keys()),
                       help='Run only a specific test')
    args = parser.parse_args()

    print("="*70)
    print("JERRY FEATURES EXPERIMENT")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Cycles per test: {args.cycles}")
    print("="*70)

    # Run tests
    results = []
    tests_to_run = [args.test] if args.test else list(TESTS.keys())

    for test_id in tests_to_run:
        config = TESTS[test_id]
        metrics = run_test(test_id, config, args.cycles)
        results.append(metrics)

    # Summary table
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)
    print(f"\n{'Test':<25} {'P&L %':<12} {'Win Rate':<12} {'Trades':<10}")
    print("-"*60)

    for r in results:
        pnl = f"{r.get('pnl_pct', 'N/A'):.1f}%" if isinstance(r.get('pnl_pct'), (int, float)) else 'N/A'
        wr = f"{r.get('win_rate', 'N/A'):.1f}%" if isinstance(r.get('win_rate'), (int, float)) else 'N/A'
        trades = r.get('total_trades', 'N/A')
        print(f"{r['name']:<25} {pnl:<12} {wr:<12} {trades:<10}")

    print("\n" + "="*70)

    # Determine winner
    valid_results = [r for r in results if isinstance(r.get('pnl_pct'), (int, float))]
    if valid_results:
        best = max(valid_results, key=lambda x: x['pnl_pct'])
        print(f"BEST PERFORMER: {best['name']} with {best['pnl_pct']:.1f}% P&L")

    print("="*70)

    # Save results
    results_file = Path('models') / 'jerry_experiment_results.txt'
    results_file.parent.mkdir(exist_ok=True)
    with open(results_file, 'w') as f:
        f.write(f"Jerry Features Experiment\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Cycles: {args.cycles}\n\n")
        for r in results:
            f.write(f"{r['name']}: P&L={r.get('pnl_pct', 'N/A')}%, ")
            f.write(f"WinRate={r.get('win_rate', 'N/A')}%, ")
            f.write(f"Trades={r.get('total_trades', 'N/A')}\n")

    print(f"\nResults saved to: {results_file}")


if __name__ == '__main__':
    main()
