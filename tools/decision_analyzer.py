#!/usr/bin/env python3
"""
Decision Analyzer - Learn from recorded trade decisions to improve entry/exit.

Analyzes decision_records.jsonl to find:
1. Which features predict winning vs losing trades
2. Which rejection reasons block good trades
3. Optimal thresholds for confidence, HMM, etc.
"""

import json
import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timedelta
import argparse


def load_decision_records(run_dir: str) -> list:
    """Load decision records from a run directory."""
    path = Path(run_dir) / "state" / "decision_records.jsonl"
    if not path.exists():
        print(f"No decision records found at {path}")
        return []

    records = []
    with open(path, 'r') as f:
        for line in f:
            try:
                records.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    return records


def load_trades(run_id: str = None) -> pd.DataFrame:
    """Load trades from paper_trading.db."""
    db_path = Path("data/paper_trading.db")
    if not db_path.exists():
        print("No paper_trading.db found")
        return pd.DataFrame()

    conn = sqlite3.connect(db_path)
    query = "SELECT * FROM trades"
    if run_id:
        query += f" WHERE run_id = '{run_id}'"
    df = pd.read_sql(query, conn)
    conn.close()
    return df


def match_decisions_to_outcomes(records: list, trades: pd.DataFrame) -> list:
    """Match decision records to their trade outcomes."""
    if trades.empty:
        return []

    # Convert trades to dict for fast lookup
    trades['timestamp'] = pd.to_datetime(trades['timestamp'])
    trade_outcomes = {}
    for _, t in trades.iterrows():
        key = t['timestamp'].strftime('%Y-%m-%dT%H:%M')
        trade_outcomes[key] = {
            'pnl': t['profit_loss'],
            'pnl_pct': t.get('projected_return_pct', 0),
            'exit_reason': t.get('exit_reason', 'unknown'),
            'option_type': t.get('option_type', 'unknown'),
            'won': t['profit_loss'] > 0,
            'ml_confidence': t.get('ml_confidence', 0),
            'calibrated_confidence': t.get('calibrated_confidence', 0),
            'hmm_trend': t.get('hmm_trend', 0),
            'signal_reasoning': t.get('signal_reasoning', '')
        }

    matched = []
    for rec in records:
        if not rec.get('trade_placed'):
            continue

        ts = rec['timestamp'][:16]  # YYYY-MM-DDTHH:MM
        if ts in trade_outcomes:
            matched.append({
                **rec,
                'outcome': trade_outcomes[ts]
            })

    return matched


def analyze_feature_importance(matched: list) -> dict:
    """Find which features predict winning trades."""
    if not matched:
        return {}

    winners = [m for m in matched if m['outcome']['won']]
    losers = [m for m in matched if not m['outcome']['won']]

    if not winners or not losers:
        return {}

    feature_names = matched[0].get('feature_names', [])
    if not feature_names:
        return {}

    results = {}
    for i, name in enumerate(feature_names):
        try:
            win_vals = [m['feature_vector'][i] for m in winners if i < len(m.get('feature_vector', []))]
            lose_vals = [m['feature_vector'][i] for m in losers if i < len(m.get('feature_vector', []))]

            if win_vals and lose_vals:
                win_mean = np.mean(win_vals)
                lose_mean = np.mean(lose_vals)
                diff = win_mean - lose_mean

                # Normalize by std to get effect size
                pooled_std = np.std(win_vals + lose_vals)
                effect_size = diff / pooled_std if pooled_std > 0 else 0

                results[name] = {
                    'win_mean': float(win_mean),
                    'lose_mean': float(lose_mean),
                    'difference': float(diff),
                    'effect_size': float(effect_size)
                }
        except (IndexError, TypeError):
            continue

    # Sort by absolute effect size
    results = dict(sorted(results.items(), key=lambda x: abs(x[1]['effect_size']), reverse=True))
    return results


def analyze_rejection_impact(records: list, trades: pd.DataFrame) -> dict:
    """Analyze which rejection reasons block profitable trades."""
    rejection_stats = defaultdict(lambda: {'blocked': 0, 'would_have_won': 0, 'missed_pnl': 0})

    # Look at rejected trades and see if similar conditions later were profitable
    rejected = [r for r in records if not r.get('trade_placed') and r.get('rejection_reasons')]

    for rec in rejected:
        for reason in rec.get('rejection_reasons', []):
            rejection_stats[reason]['blocked'] += 1

    return dict(rejection_stats)


def find_optimal_thresholds(matched: list) -> dict:
    """Find optimal thresholds for various features."""
    if not matched:
        return {}

    thresholds = {}

    # Calibrated confidence threshold
    conf_vals = [(m['calibrated_confidence'], m['outcome']['won']) for m in matched]
    if conf_vals:
        for threshold in [0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7]:
            above = [won for conf, won in conf_vals if conf >= threshold]
            if above:
                win_rate = sum(above) / len(above)
                thresholds[f'conf_{threshold}'] = {
                    'win_rate': win_rate,
                    'trades': len(above),
                    'threshold': threshold
                }

    # HMM trend thresholds
    feature_names = matched[0].get('feature_names', [])
    if 'hmm_trend' in feature_names:
        idx = feature_names.index('hmm_trend')
        hmm_vals = [(m['feature_vector'][idx], m['outcome']['won']) for m in matched if idx < len(m.get('feature_vector', []))]

        for threshold in [0.6, 0.65, 0.7, 0.75, 0.8]:
            # Bullish trades with HMM > threshold
            bullish = [won for hmm, won in hmm_vals if hmm >= threshold]
            if bullish:
                thresholds[f'hmm_bullish_{threshold}'] = {
                    'win_rate': sum(bullish) / len(bullish),
                    'trades': len(bullish),
                    'threshold': threshold
                }

    return thresholds


def generate_experiment_ideas(analysis: dict) -> list:
    """Generate experiment ideas based on analysis."""
    ideas = []

    # From feature importance
    if 'feature_importance' in analysis:
        top_features = list(analysis['feature_importance'].items())[:5]
        for name, stats in top_features:
            if abs(stats['effect_size']) > 0.3:
                direction = "higher" if stats['difference'] > 0 else "lower"
                ideas.append({
                    'title': f'Filter by {name}',
                    'description': f'Winners have {direction} {name} (effect size: {stats["effect_size"]:.2f})',
                    'hypothesis': f'{name} {direction} correlates with winning trades',
                    'priority': 'high' if abs(stats['effect_size']) > 0.5 else 'medium'
                })

    # From optimal thresholds
    if 'thresholds' in analysis:
        best_conf = max(
            [(k, v) for k, v in analysis['thresholds'].items() if k.startswith('conf_')],
            key=lambda x: x[1]['win_rate'] if x[1]['trades'] >= 20 else 0,
            default=(None, None)
        )
        if best_conf[0]:
            ideas.append({
                'title': f'Calibrated confidence >= {best_conf[1]["threshold"]}',
                'description': f'Win rate: {best_conf[1]["win_rate"]:.1%} with {best_conf[1]["trades"]} trades',
                'env_vars': {
                    'PNL_CAL_GATE': '1',
                    'PNL_CAL_MIN_PROB': str(best_conf[1]['threshold'])
                },
                'priority': 'critical'
            })

    return ideas


def main():
    parser = argparse.ArgumentParser(description='Analyze decision records')
    parser.add_argument('--run-dir', type=str, help='Model run directory')
    parser.add_argument('--all-runs', action='store_true', help='Analyze all recent runs')
    parser.add_argument('--output', type=str, help='Output JSON file')
    args = parser.parse_args()

    if args.all_runs:
        # Find recent runs with decision records
        models_dir = Path('models')
        run_dirs = sorted(
            [d for d in models_dir.iterdir() if (d / 'state' / 'decision_records.jsonl').exists()],
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )[:10]
    elif args.run_dir:
        run_dirs = [Path(args.run_dir)]
    else:
        # Default to most recent
        models_dir = Path('models')
        run_dirs = sorted(
            [d for d in models_dir.iterdir() if (d / 'state' / 'decision_records.jsonl').exists()],
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )[:1]

    all_records = []
    for run_dir in run_dirs:
        print(f"Loading {run_dir.name}...")
        records = load_decision_records(str(run_dir))
        all_records.extend(records)

    print(f"Loaded {len(all_records)} decision records")

    # Load trades
    trades = load_trades()
    print(f"Loaded {len(trades)} trades")

    # Match decisions to outcomes
    matched = match_decisions_to_outcomes(all_records, trades)
    print(f"Matched {len(matched)} decisions to outcomes")

    if not matched:
        print("No matched decisions found. Trades may be from different runs.")
        return

    # Run analysis
    analysis = {
        'total_records': len(all_records),
        'total_trades': len(trades),
        'matched': len(matched),
        'win_rate': sum(1 for m in matched if m['outcome']['won']) / len(matched) if matched else 0,
        'feature_importance': analyze_feature_importance(matched),
        'thresholds': find_optimal_thresholds(matched),
        'rejection_impact': analyze_rejection_impact(all_records, trades)
    }

    # Generate ideas
    ideas = generate_experiment_ideas(analysis)
    analysis['experiment_ideas'] = ideas

    # Print summary
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    print(f"Win rate: {analysis['win_rate']:.1%}")

    print("\nTop predictive features (effect size):")
    for name, stats in list(analysis['feature_importance'].items())[:10]:
        direction = "+" if stats['difference'] > 0 else ""
        print(f"  {name}: {direction}{stats['effect_size']:.3f}")

    print("\nOptimal thresholds:")
    for name, stats in analysis['thresholds'].items():
        if stats['trades'] >= 20:
            print(f"  {name}: {stats['win_rate']:.1%} WR ({stats['trades']} trades)")

    print("\nRejection reasons:")
    for reason, stats in sorted(analysis['rejection_impact'].items(), key=lambda x: x[1]['blocked'], reverse=True)[:10]:
        print(f"  {reason}: {stats['blocked']} blocked")

    print("\nExperiment ideas:")
    for idea in ideas[:5]:
        print(f"  [{idea.get('priority', 'medium')}] {idea['title']}")

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(analysis, f, indent=2)
        print(f"\nSaved analysis to {args.output}")


if __name__ == '__main__':
    main()
