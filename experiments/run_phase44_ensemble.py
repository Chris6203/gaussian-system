#!/usr/bin/env python3
"""
Phase 44: Test all improvement techniques for win rate

Techniques tested:
1. Ensemble Stacking (TCN + LSTM + XGBoost)
2. LSTM-XGBoost Hybrid
3. GEX/Gamma Signals
4. Order Flow Imbalance
5. Multi-Indicator Stacking
6. Combined approach
"""

import subprocess
import os
import sys
import time
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_experiment(name: str, env_vars: dict, cycles: int = 5000) -> dict:
    """Run a single experiment and return results."""
    env = os.environ.copy()
    env.update({
        "MODEL_RUN_DIR": f"models/phase44_{name}",
        "PAPER_TRADING": "True",
        "TT_MAX_CYCLES": str(cycles),
        "TT_PRINT_EVERY": "1000",
        # Base config from combo_dow (65% baseline)
        "HARD_STOP_LOSS_PCT": "50",
        "HARD_TAKE_PROFIT_PCT": "10",
        "TRAIN_MAX_CONF": "0.25",
        "DAY_OF_WEEK_FILTER": "1",
        "SKIP_MONDAY": "1",
        "SKIP_FRIDAY": "1",
    })
    env.update(env_vars)

    print(f"\n{'='*60}")
    print(f"Starting experiment: {name}")
    print(f"{'='*60}")

    proc = subprocess.run(
        ["python", "scripts/train_time_travel.py"],
        env=env,
        capture_output=True,
        text=True,
        timeout=3600  # 1 hour timeout
    )

    # Read results
    result = {"name": name, "success": False}
    run_info_path = Path(f"models/phase44_{name}/run_info.json")
    summary_path = Path(f"models/phase44_{name}/SUMMARY.txt")

    if run_info_path.exists():
        with open(run_info_path) as f:
            info = json.load(f)
            result.update({
                "success": True,
                "pnl_pct": info.get("pnl_pct", 0),
                "trades": info.get("trades", 0),
                "cycles": info.get("cycles", 0),
            })

    if summary_path.exists():
        with open(summary_path) as f:
            content = f.read()
            for line in content.split('\n'):
                if 'Win Rate:' in line:
                    try:
                        result["win_rate"] = float(line.split(':')[1].strip().replace('%', ''))
                    except:
                        pass
                elif 'Wins:' in line:
                    try:
                        result["wins"] = int(line.split(':')[1].strip())
                    except:
                        pass
                elif 'Losses:' in line:
                    try:
                        result["losses"] = int(line.split(':')[1].strip())
                    except:
                        pass

    return result


def main():
    experiments = [
        # 1. Baseline (combo_dow)
        {
            "name": "baseline",
            "env": {},
            "description": "Baseline combo_dow (65% target)"
        },

        # 2. Ensemble Stacking enabled
        {
            "name": "ensemble_stack",
            "env": {
                "ENSEMBLE_ENABLED": "1",
                "ENSEMBLE_TYPE": "stacking",
            },
            "description": "Ensemble stacking (TCN+LSTM+XGBoost)"
        },

        # 3. LSTM-XGBoost Hybrid
        {
            "name": "lstm_xgb_hybrid",
            "env": {
                "ENSEMBLE_ENABLED": "1",
                "ENSEMBLE_TYPE": "hybrid",
            },
            "description": "LSTM-XGBoost hybrid"
        },

        # 4. GEX/Gamma Signals
        {
            "name": "gex_signals",
            "env": {
                "GEX_SIGNALS_ENABLED": "1",
            },
            "description": "Gamma exposure signals"
        },

        # 5. Order Flow Imbalance
        {
            "name": "order_flow",
            "env": {
                "ORDER_FLOW_ENABLED": "1",
            },
            "description": "Order flow imbalance signals"
        },

        # 6. Multi-Indicator Stacking
        {
            "name": "multi_indicator",
            "env": {
                "MULTI_INDICATOR_ENABLED": "1",
            },
            "description": "Multi-indicator stacking (6+ indicators)"
        },

        # 7. Combined: All techniques
        {
            "name": "combined_all",
            "env": {
                "ENSEMBLE_ENABLED": "1",
                "ENSEMBLE_TYPE": "stacking",
                "GEX_SIGNALS_ENABLED": "1",
                "ORDER_FLOW_ENABLED": "1",
                "MULTI_INDICATOR_ENABLED": "1",
            },
            "description": "All techniques combined"
        },
    ]

    results = []
    print("\n" + "="*80)
    print("PHASE 44: TESTING ALL IMPROVEMENT TECHNIQUES")
    print("="*80)

    for exp in experiments:
        print(f"\nRunning: {exp['description']}")
        try:
            result = run_experiment(exp["name"], exp["env"])
            result["description"] = exp["description"]
            results.append(result)

            if result["success"]:
                win_rate = result.get("win_rate", 0)
                pnl = result.get("pnl_pct", 0)
                trades = result.get("trades", 0)
                print(f"  ✓ Win Rate: {win_rate:.1f}%, P&L: {pnl:+.1f}%, Trades: {trades}")
            else:
                print(f"  ✗ Failed to complete")
        except Exception as e:
            print(f"  ✗ Error: {e}")
            results.append({
                "name": exp["name"],
                "description": exp["description"],
                "success": False,
                "error": str(e)
            })

    # Print summary
    print("\n" + "="*80)
    print("PHASE 44 RESULTS SUMMARY")
    print("="*80)
    print(f"{'Experiment':<25} {'Win Rate':>10} {'P&L':>10} {'Trades':>8} {'$/Trade':>10}")
    print("-"*80)

    for r in sorted(results, key=lambda x: x.get("win_rate", 0), reverse=True):
        if r["success"]:
            name = r["name"][:24]
            win_rate = r.get("win_rate", 0)
            pnl = r.get("pnl_pct", 0)
            trades = r.get("trades", 0)
            per_trade = (pnl * 50) / trades if trades > 0 else 0
            marker = " ★" if win_rate > 65 else ""
            print(f"{name:<25} {win_rate:>9.1f}% {pnl:>+9.1f}% {trades:>8} {per_trade:>+9.2f}{marker}")
        else:
            print(f"{r['name']:<25} {'FAILED':>10}")

    print("-"*80)

    # Save results
    with open("models/phase44_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to models/phase44_results.json")


if __name__ == "__main__":
    main()
