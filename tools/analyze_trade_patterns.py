#!/usr/bin/env python3
"""
Analyze trade patterns from 20K validation decision records.
Looks for patterns beyond what we already found:
- VIX level correlations
- Momentum strength at entry
- HMM regime states
- Volume spike correlations
- Confidence level buckets
- Sequential patterns
- Price trend vs trade direction
"""

import json
import sys
from collections import defaultdict
from datetime import datetime
import statistics

def load_trades(filepath):
    """Load only the actual trades (not HOLDs) from decision records."""
    trades = []
    all_records = []

    with open(filepath, 'r') as f:
        for line in f:
            try:
                record = json.loads(line.strip())
                all_records.append(record)

                # Only look at actual trades
                if record.get('trade_placed', False):
                    trades.append(record)
            except json.JSONDecodeError:
                continue

    return trades, all_records

def get_feature_value(record, feature_name):
    """Get a feature value from the record."""
    # Try trade_state first
    ts = record.get('trade_state', {})
    if feature_name in ts:
        return ts[feature_name]

    # Try feature_names + feature_vector
    feature_names = record.get('feature_names', [])
    feature_vector = record.get('feature_vector', [])

    if feature_name in feature_names:
        idx = feature_names.index(feature_name)
        if idx < len(feature_vector):
            return feature_vector[idx]

    return None

def calculate_pnl(record):
    """Calculate if trade was a win or loss based on unrealized_pnl_pct."""
    # We need to look at the exit - find when position closed
    # For now, check if we have pnl info in record
    pnl = record.get('pnl', None) or record.get('realized_pnl', None)
    if pnl is not None:
        return pnl

    # Check extras
    extras = record.get('trade_state', {}).get('extras', {})
    if 'pnl' in extras:
        return extras['pnl']

    return None

def main():
    filepath = '/var/ai/simon/gaussian-system/models/COMBO_BEST_20K_VALIDATION/state/decision_records.jsonl'

    print("Loading trades...")
    trades, all_records = load_trades(filepath)
    print(f"Total records: {len(all_records)}")
    print(f"Actual trades placed: {len(trades)}")

    # First, let's understand the trade structure better
    if trades:
        print("\n=== Sample Trade Record ===")
        sample = trades[0]
        print(f"Keys: {list(sample.keys())}")
        print(f"trade_state keys: {list(sample.get('trade_state', {}).keys())}")
        print(f"Feature names: {sample.get('feature_names', [])[:10]}...")

    # Now let's find entries and exits to pair them
    # Look for BUY_CALL, BUY_PUT actions and their corresponding exits
    entries = []
    in_trade = False
    current_entry = None

    for record in all_records:
        action = record.get('executed_action') or record.get('proposed_action')
        ts = record.get('trade_state', {})
        is_in_trade = ts.get('is_in_trade', False)

        if action in ['BUY_CALL', 'BUY_PUT'] and record.get('trade_placed', False):
            current_entry = {
                'record': record,
                'action': action,
                'timestamp': record.get('timestamp'),
                'entry_price': ts.get('current_price'),
                'vix_level': get_feature_value(record, 'vix_level'),
                'momentum_5m': get_feature_value(record, 'momentum_5m'),
                'momentum_15m': get_feature_value(record, 'momentum_15m'),
                'volume_spike': get_feature_value(record, 'volume_spike'),
                'hmm_trend': get_feature_value(record, 'hmm_trend'),
                'hmm_volatility': get_feature_value(record, 'hmm_volatility'),
                'hmm_liquidity': get_feature_value(record, 'hmm_liquidity'),
                'hmm_confidence': get_feature_value(record, 'hmm_confidence'),
                'confidence': get_feature_value(record, 'calibrated_confidence'),
                'raw_confidence': get_feature_value(record, 'raw_confidence'),
            }
            in_trade = True

        elif action == 'EXIT' and in_trade and current_entry:
            # Found the exit for this entry
            exit_price = ts.get('current_price')
            unrealized_pnl = ts.get('unrealized_pnl_pct', 0)

            # Get exit reason from extras
            extras = ts.get('extras', {})
            exit_reason = extras.get('exit_reason', 'UNKNOWN')

            current_entry['exit_price'] = exit_price
            current_entry['pnl_pct'] = unrealized_pnl
            current_entry['exit_reason'] = exit_reason
            current_entry['win'] = unrealized_pnl > 0
            entries.append(current_entry)

            in_trade = False
            current_entry = None

    print(f"\nMatched entry/exit pairs: {len(entries)}")

    if not entries:
        print("No matched trades found. Trying alternative approach...")
        # Try to find trades with pnl info
        for record in all_records:
            ts = record.get('trade_state', {})
            action = record.get('executed_action')

            if action == 'EXIT':
                pnl = ts.get('unrealized_pnl_pct', 0)
                entries.append({
                    'record': record,
                    'pnl_pct': pnl,
                    'win': pnl > 0,
                    'vix_level': get_feature_value(record, 'vix_level'),
                    'momentum_5m': get_feature_value(record, 'momentum_5m'),
                    'momentum_15m': get_feature_value(record, 'momentum_15m'),
                    'volume_spike': get_feature_value(record, 'volume_spike'),
                    'hmm_trend': get_feature_value(record, 'hmm_trend'),
                    'hmm_volatility': get_feature_value(record, 'hmm_volatility'),
                    'hmm_liquidity': get_feature_value(record, 'hmm_liquidity'),
                    'hmm_confidence': get_feature_value(record, 'hmm_confidence'),
                    'confidence': get_feature_value(record, 'calibrated_confidence'),
                    'timestamp': record.get('timestamp'),
                    'exit_reason': ts.get('extras', {}).get('exit_reason', 'UNKNOWN'),
                })

        print(f"Found {len(entries)} EXIT records with P&L")

    # Now analyze patterns
    analyze_patterns(entries)

def analyze_patterns(entries):
    """Analyze all patterns in the entries."""
    if not entries:
        print("No entries to analyze!")
        return

    total = len(entries)
    wins = sum(1 for e in entries if e.get('win', False))
    overall_wr = wins / total * 100 if total > 0 else 0

    print(f"\n{'='*60}")
    print(f"PATTERN ANALYSIS - {total} trades, {wins} wins ({overall_wr:.1f}% WR)")
    print(f"{'='*60}")

    results = []

    # 1. VIX Level Analysis
    print("\n=== 1. VIX LEVEL CORRELATIONS ===")
    vix_buckets = {
        'Low (<15)': lambda v: v is not None and v < 15,
        'Med (15-20)': lambda v: v is not None and 15 <= v < 20,
        'High (20-25)': lambda v: v is not None and 20 <= v < 25,
        'Very High (>25)': lambda v: v is not None and v >= 25,
    }

    for bucket_name, condition in vix_buckets.items():
        bucket_entries = [e for e in entries if condition(e.get('vix_level'))]
        if bucket_entries:
            bucket_wins = sum(1 for e in bucket_entries if e.get('win', False))
            wr = bucket_wins / len(bucket_entries) * 100
            diff = wr - overall_wr
            print(f"  {bucket_name}: {len(bucket_entries)} trades, {bucket_wins} wins ({wr:.1f}% WR) [{diff:+.1f}% vs avg]")
            results.append(('VIX ' + bucket_name, len(bucket_entries), wr, diff))

    # 2. Momentum Strength at Entry
    print("\n=== 2. MOMENTUM STRENGTH (5min) ===")
    mom_buckets = {
        'Strong Down (<-0.001)': lambda m: m is not None and m < -0.001,
        'Weak Down (-0.001 to 0)': lambda m: m is not None and -0.001 <= m < 0,
        'Neutral (~0)': lambda m: m is not None and -0.0001 <= m <= 0.0001,
        'Weak Up (0 to 0.001)': lambda m: m is not None and 0 < m < 0.001,
        'Strong Up (>0.001)': lambda m: m is not None and m >= 0.001,
    }

    for bucket_name, condition in mom_buckets.items():
        bucket_entries = [e for e in entries if condition(e.get('momentum_5m'))]
        if bucket_entries:
            bucket_wins = sum(1 for e in bucket_entries if e.get('win', False))
            wr = bucket_wins / len(bucket_entries) * 100
            diff = wr - overall_wr
            print(f"  {bucket_name}: {len(bucket_entries)} trades, {wr:.1f}% WR [{diff:+.1f}% vs avg]")
            results.append(('Mom5m ' + bucket_name, len(bucket_entries), wr, diff))

    # 3. HMM Regime States
    print("\n=== 3. HMM REGIME STATES ===")

    # HMM Trend
    hmm_trend_buckets = {
        'Bearish (<0.4)': lambda t: t is not None and t < 0.4,
        'Neutral (0.4-0.6)': lambda t: t is not None and 0.4 <= t <= 0.6,
        'Bullish (>0.6)': lambda t: t is not None and t > 0.6,
    }

    print("  HMM Trend:")
    for bucket_name, condition in hmm_trend_buckets.items():
        bucket_entries = [e for e in entries if condition(e.get('hmm_trend'))]
        if bucket_entries:
            bucket_wins = sum(1 for e in bucket_entries if e.get('win', False))
            wr = bucket_wins / len(bucket_entries) * 100
            diff = wr - overall_wr
            print(f"    {bucket_name}: {len(bucket_entries)} trades, {wr:.1f}% WR [{diff:+.1f}% vs avg]")
            results.append(('HMM Trend ' + bucket_name, len(bucket_entries), wr, diff))

    # HMM Volatility
    hmm_vol_buckets = {
        'Low (<0.4)': lambda v: v is not None and v < 0.4,
        'Med (0.4-0.6)': lambda v: v is not None and 0.4 <= v <= 0.6,
        'High (>0.6)': lambda v: v is not None and v > 0.6,
    }

    print("  HMM Volatility:")
    for bucket_name, condition in hmm_vol_buckets.items():
        bucket_entries = [e for e in entries if condition(e.get('hmm_volatility'))]
        if bucket_entries:
            bucket_wins = sum(1 for e in bucket_entries if e.get('win', False))
            wr = bucket_wins / len(bucket_entries) * 100
            diff = wr - overall_wr
            print(f"    {bucket_name}: {len(bucket_entries)} trades, {wr:.1f}% WR [{diff:+.1f}% vs avg]")
            results.append(('HMM Vol ' + bucket_name, len(bucket_entries), wr, diff))

    # HMM Confidence
    print("  HMM Confidence:")
    hmm_conf_buckets = {
        'Low (<0.3)': lambda c: c is not None and c < 0.3,
        'Med (0.3-0.6)': lambda c: c is not None and 0.3 <= c <= 0.6,
        'High (>0.6)': lambda c: c is not None and c > 0.6,
    }

    for bucket_name, condition in hmm_conf_buckets.items():
        bucket_entries = [e for e in entries if condition(e.get('hmm_confidence'))]
        if bucket_entries:
            bucket_wins = sum(1 for e in bucket_entries if e.get('win', False))
            wr = bucket_wins / len(bucket_entries) * 100
            diff = wr - overall_wr
            print(f"    {bucket_name}: {len(bucket_entries)} trades, {wr:.1f}% WR [{diff:+.1f}% vs avg]")
            results.append(('HMM Conf ' + bucket_name, len(bucket_entries), wr, diff))

    # 4. Volume Spike
    print("\n=== 4. VOLUME SPIKE ===")
    vol_spike_buckets = {
        'Low (<0.8)': lambda v: v is not None and v < 0.8,
        'Normal (0.8-1.2)': lambda v: v is not None and 0.8 <= v < 1.2,
        'Elevated (1.2-2.0)': lambda v: v is not None and 1.2 <= v < 2.0,
        'High (>2.0)': lambda v: v is not None and v >= 2.0,
    }

    for bucket_name, condition in vol_spike_buckets.items():
        bucket_entries = [e for e in entries if condition(e.get('volume_spike'))]
        if bucket_entries:
            bucket_wins = sum(1 for e in bucket_entries if e.get('win', False))
            wr = bucket_wins / len(bucket_entries) * 100
            diff = wr - overall_wr
            print(f"  {bucket_name}: {len(bucket_entries)} trades, {wr:.1f}% WR [{diff:+.1f}% vs avg]")
            results.append(('VolSpike ' + bucket_name, len(bucket_entries), wr, diff))

    # 5. Confidence Level Buckets
    print("\n=== 5. CONFIDENCE LEVELS ===")
    conf_buckets = {
        'Low (0.0-0.4)': lambda c: c is not None and c < 0.4,
        'Med (0.4-0.6)': lambda c: c is not None and 0.4 <= c < 0.6,
        'High (0.6-0.8)': lambda c: c is not None and 0.6 <= c < 0.8,
        'Very High (0.8-1.0)': lambda c: c is not None and c >= 0.8,
    }

    for bucket_name, condition in conf_buckets.items():
        bucket_entries = [e for e in entries if condition(e.get('confidence'))]
        if bucket_entries:
            bucket_wins = sum(1 for e in bucket_entries if e.get('win', False))
            wr = bucket_wins / len(bucket_entries) * 100
            diff = wr - overall_wr
            print(f"  {bucket_name}: {len(bucket_entries)} trades, {wr:.1f}% WR [{diff:+.1f}% vs avg]")
            results.append(('Conf ' + bucket_name, len(bucket_entries), wr, diff))

    # 6. Sequential Patterns
    print("\n=== 6. SEQUENTIAL PATTERNS ===")
    prev_win = None
    after_win = {'wins': 0, 'total': 0}
    after_loss = {'wins': 0, 'total': 0}

    for entry in entries:
        if prev_win is not None:
            if prev_win:
                after_win['total'] += 1
                if entry.get('win', False):
                    after_win['wins'] += 1
            else:
                after_loss['total'] += 1
                if entry.get('win', False):
                    after_loss['wins'] += 1
        prev_win = entry.get('win', False)

    if after_win['total'] > 0:
        wr = after_win['wins'] / after_win['total'] * 100
        diff = wr - overall_wr
        print(f"  After WIN: {after_win['total']} trades, {wr:.1f}% WR [{diff:+.1f}% vs avg]")
        results.append(('After WIN', after_win['total'], wr, diff))

    if after_loss['total'] > 0:
        wr = after_loss['wins'] / after_loss['total'] * 100
        diff = wr - overall_wr
        print(f"  After LOSS: {after_loss['total']} trades, {wr:.1f}% WR [{diff:+.1f}% vs avg]")
        results.append(('After LOSS', after_loss['total'], wr, diff))

    # 7. Exit Reason Analysis
    print("\n=== 7. EXIT REASON ANALYSIS ===")
    exit_reasons = defaultdict(lambda: {'wins': 0, 'total': 0, 'pnls': []})

    for entry in entries:
        reason = entry.get('exit_reason', 'UNKNOWN')
        exit_reasons[reason]['total'] += 1
        exit_reasons[reason]['pnls'].append(entry.get('pnl_pct', 0))
        if entry.get('win', False):
            exit_reasons[reason]['wins'] += 1

    for reason, stats in sorted(exit_reasons.items(), key=lambda x: x[1]['total'], reverse=True):
        if stats['total'] > 0:
            wr = stats['wins'] / stats['total'] * 100
            avg_pnl = statistics.mean(stats['pnls']) if stats['pnls'] else 0
            diff = wr - overall_wr
            print(f"  {reason}: {stats['total']} trades, {wr:.1f}% WR, avg P&L: {avg_pnl:.2f}% [{diff:+.1f}% vs avg]")
            results.append(('Exit ' + reason, stats['total'], wr, diff))

    # 8. P&L Distribution
    print("\n=== 8. P&L DISTRIBUTION ===")
    pnls = [e.get('pnl_pct', 0) for e in entries if e.get('pnl_pct') is not None]
    if pnls:
        print(f"  Mean P&L: {statistics.mean(pnls):.2f}%")
        print(f"  Median P&L: {statistics.median(pnls):.2f}%")
        print(f"  Std Dev: {statistics.stdev(pnls):.2f}%")
        print(f"  Min P&L: {min(pnls):.2f}%")
        print(f"  Max P&L: {max(pnls):.2f}%")

    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE - SORTED BY WIN RATE DIFFERENCE")
    print("="*80)
    print(f"{'Pattern':<35} {'Count':>8} {'WR%':>8} {'Diff':>10} {'Action':<15}")
    print("-"*80)

    # Sort by absolute difference
    results.sort(key=lambda x: abs(x[3]), reverse=True)

    for pattern, count, wr, diff in results:
        if count >= 5:  # Only show patterns with enough samples
            action = "AVOID" if diff < -5 else ("PREFER" if diff > 5 else "neutral")
            print(f"{pattern:<35} {count:>8} {wr:>7.1f}% {diff:>+9.1f}% {action:<15}")

    # Print recommended filters
    print("\n" + "="*60)
    print("RECOMMENDED FILTERS (>5% difference, >10 samples)")
    print("="*60)

    for pattern, count, wr, diff in results:
        if count >= 10 and abs(diff) > 5:
            if diff < 0:
                print(f"AVOID: {pattern} ({wr:.1f}% WR, {diff:+.1f}% vs avg)")
            else:
                print(f"PREFER: {pattern} ({wr:.1f}% WR, {diff:+.1f}% vs avg)")

if __name__ == "__main__":
    main()
