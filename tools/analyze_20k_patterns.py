#!/usr/bin/env python3
"""
Analyze trade patterns from 20K validation run using the paper_trading.db database.
Looks for NEW patterns beyond what we already found:
- VIX level correlations
- Momentum strength at entry
- HMM regime states
- Volume spike correlations
- Confidence level buckets
- Sequential patterns (win after win, loss after loss)
- Price trend vs trade direction
"""

import sqlite3
import json
from collections import defaultdict
from datetime import datetime
import statistics

def get_trades():
    """Load trades from database for the COMBO_BEST_20K_VALIDATION run."""
    conn = sqlite3.connect('data/paper_trading.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
        SELECT *
        FROM trades
        WHERE run_id = 'COMBO_BEST_20K_VALIDATION'
        ORDER BY timestamp
    """)

    trades = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return trades

def analyze_patterns(trades):
    """Analyze all patterns in the trades."""
    if not trades:
        print("No trades to analyze!")
        return

    total = len(trades)
    wins = sum(1 for t in trades if t['profit_loss'] > 0)
    overall_wr = wins / total * 100 if total > 0 else 0
    total_pnl = sum(t['profit_loss'] for t in trades)

    print(f"\n{'='*60}")
    print(f"PATTERN ANALYSIS - {total} trades, {wins} wins ({overall_wr:.1f}% WR)")
    print(f"Total P&L: ${total_pnl:.2f}")
    print(f"{'='*60}")

    results = []

    # 1. VIX Level Analysis
    print("\n=== 1. VIX LEVEL CORRELATIONS ===")
    vix_buckets = {
        'Very Low (<12)': lambda v: v is not None and v < 12,
        'Low (12-15)': lambda v: v is not None and 12 <= v < 15,
        'Medium (15-18)': lambda v: v is not None and 15 <= v < 18,
        'Elevated (18-22)': lambda v: v is not None and 18 <= v < 22,
        'High (22-28)': lambda v: v is not None and 22 <= v < 28,
        'Very High (>28)': lambda v: v is not None and v >= 28,
    }

    for bucket_name, condition in vix_buckets.items():
        bucket_trades = [t for t in trades if condition(t.get('vix_level'))]
        if bucket_trades:
            bucket_wins = sum(1 for t in bucket_trades if t['profit_loss'] > 0)
            bucket_pnl = sum(t['profit_loss'] for t in bucket_trades)
            wr = bucket_wins / len(bucket_trades) * 100
            diff = wr - overall_wr
            avg_pnl = bucket_pnl / len(bucket_trades)
            print(f"  {bucket_name}: {len(bucket_trades)} trades, {wr:.1f}% WR, avg P&L ${avg_pnl:.2f} [{diff:+.1f}% vs avg]")
            results.append(('VIX ' + bucket_name, len(bucket_trades), wr, diff, bucket_pnl))

    # 2. Momentum Strength at Entry
    print("\n=== 2. MOMENTUM STRENGTH (5min) ===")
    mom_buckets = {
        'Strong Negative (<-0.002)': lambda m: m is not None and m < -0.002,
        'Moderate Negative (-0.002 to -0.0005)': lambda m: m is not None and -0.002 <= m < -0.0005,
        'Weak Negative (-0.0005 to 0)': lambda m: m is not None and -0.0005 <= m < 0,
        'Weak Positive (0 to 0.0005)': lambda m: m is not None and 0 <= m < 0.0005,
        'Moderate Positive (0.0005 to 0.002)': lambda m: m is not None and 0.0005 <= m < 0.002,
        'Strong Positive (>0.002)': lambda m: m is not None and m >= 0.002,
    }

    for bucket_name, condition in mom_buckets.items():
        bucket_trades = [t for t in trades if condition(t.get('momentum_5m'))]
        if bucket_trades:
            bucket_wins = sum(1 for t in bucket_trades if t['profit_loss'] > 0)
            bucket_pnl = sum(t['profit_loss'] for t in bucket_trades)
            wr = bucket_wins / len(bucket_trades) * 100
            diff = wr - overall_wr
            avg_pnl = bucket_pnl / len(bucket_trades)
            print(f"  {bucket_name}: {len(bucket_trades)} trades, {wr:.1f}% WR, avg ${avg_pnl:.2f} [{diff:+.1f}%]")
            results.append(('Mom5m ' + bucket_name, len(bucket_trades), wr, diff, bucket_pnl))

    # 3. Momentum 15m Analysis
    print("\n=== 3. MOMENTUM STRENGTH (15min) ===")
    for bucket_name, condition in mom_buckets.items():
        bucket_trades = [t for t in trades if condition(t.get('momentum_15m'))]
        if bucket_trades:
            bucket_wins = sum(1 for t in bucket_trades if t['profit_loss'] > 0)
            bucket_pnl = sum(t['profit_loss'] for t in bucket_trades)
            wr = bucket_wins / len(bucket_trades) * 100
            diff = wr - overall_wr
            avg_pnl = bucket_pnl / len(bucket_trades)
            print(f"  {bucket_name}: {len(bucket_trades)} trades, {wr:.1f}% WR, avg ${avg_pnl:.2f} [{diff:+.1f}%]")
            results.append(('Mom15m ' + bucket_name, len(bucket_trades), wr, diff, bucket_pnl))

    # 4. HMM Regime States
    print("\n=== 4. HMM REGIME STATES ===")

    # HMM Trend
    hmm_trend_buckets = {
        'Bearish (<0.35)': lambda t: t is not None and t < 0.35,
        'Slightly Bearish (0.35-0.45)': lambda t: t is not None and 0.35 <= t < 0.45,
        'Neutral (0.45-0.55)': lambda t: t is not None and 0.45 <= t <= 0.55,
        'Slightly Bullish (0.55-0.65)': lambda t: t is not None and 0.55 < t <= 0.65,
        'Bullish (>0.65)': lambda t: t is not None and t > 0.65,
    }

    print("  -- HMM Trend --")
    for bucket_name, condition in hmm_trend_buckets.items():
        bucket_trades = [t for t in trades if condition(t.get('hmm_trend'))]
        if bucket_trades:
            bucket_wins = sum(1 for t in bucket_trades if t['profit_loss'] > 0)
            bucket_pnl = sum(t['profit_loss'] for t in bucket_trades)
            wr = bucket_wins / len(bucket_trades) * 100
            diff = wr - overall_wr
            avg_pnl = bucket_pnl / len(bucket_trades)
            print(f"    {bucket_name}: {len(bucket_trades)} trades, {wr:.1f}% WR, avg ${avg_pnl:.2f} [{diff:+.1f}%]")
            results.append(('HMM Trend ' + bucket_name, len(bucket_trades), wr, diff, bucket_pnl))

    # HMM Volatility
    hmm_vol_buckets = {
        'Low Vol (<0.35)': lambda v: v is not None and v < 0.35,
        'Mod Low (0.35-0.45)': lambda v: v is not None and 0.35 <= v < 0.45,
        'Neutral (0.45-0.55)': lambda v: v is not None and 0.45 <= v <= 0.55,
        'Mod High (0.55-0.65)': lambda v: v is not None and 0.55 < v <= 0.65,
        'High Vol (>0.65)': lambda v: v is not None and v > 0.65,
    }

    print("  -- HMM Volatility --")
    for bucket_name, condition in hmm_vol_buckets.items():
        bucket_trades = [t for t in trades if condition(t.get('hmm_volatility'))]
        if bucket_trades:
            bucket_wins = sum(1 for t in bucket_trades if t['profit_loss'] > 0)
            bucket_pnl = sum(t['profit_loss'] for t in bucket_trades)
            wr = bucket_wins / len(bucket_trades) * 100
            diff = wr - overall_wr
            avg_pnl = bucket_pnl / len(bucket_trades)
            print(f"    {bucket_name}: {len(bucket_trades)} trades, {wr:.1f}% WR, avg ${avg_pnl:.2f} [{diff:+.1f}%]")
            results.append(('HMM Vol ' + bucket_name, len(bucket_trades), wr, diff, bucket_pnl))

    # HMM Confidence
    print("  -- HMM Confidence --")
    hmm_conf_buckets = {
        'Very Low (<0.2)': lambda c: c is not None and c < 0.2,
        'Low (0.2-0.4)': lambda c: c is not None and 0.2 <= c < 0.4,
        'Medium (0.4-0.6)': lambda c: c is not None and 0.4 <= c < 0.6,
        'High (0.6-0.8)': lambda c: c is not None and 0.6 <= c < 0.8,
        'Very High (>0.8)': lambda c: c is not None and c >= 0.8,
    }

    for bucket_name, condition in hmm_conf_buckets.items():
        bucket_trades = [t for t in trades if condition(t.get('hmm_confidence'))]
        if bucket_trades:
            bucket_wins = sum(1 for t in bucket_trades if t['profit_loss'] > 0)
            bucket_pnl = sum(t['profit_loss'] for t in bucket_trades)
            wr = bucket_wins / len(bucket_trades) * 100
            diff = wr - overall_wr
            avg_pnl = bucket_pnl / len(bucket_trades)
            print(f"    {bucket_name}: {len(bucket_trades)} trades, {wr:.1f}% WR, avg ${avg_pnl:.2f} [{diff:+.1f}%]")
            results.append(('HMM Conf ' + bucket_name, len(bucket_trades), wr, diff, bucket_pnl))

    # 5. Volume Spike
    print("\n=== 5. VOLUME SPIKE ===")
    vol_spike_buckets = {
        'Very Low (<0.5)': lambda v: v is not None and v < 0.5,
        'Low (0.5-0.8)': lambda v: v is not None and 0.5 <= v < 0.8,
        'Normal (0.8-1.2)': lambda v: v is not None and 0.8 <= v < 1.2,
        'Elevated (1.2-1.8)': lambda v: v is not None and 1.2 <= v < 1.8,
        'High (1.8-3.0)': lambda v: v is not None and 1.8 <= v < 3.0,
        'Very High (>3.0)': lambda v: v is not None and v >= 3.0,
    }

    for bucket_name, condition in vol_spike_buckets.items():
        bucket_trades = [t for t in trades if condition(t.get('volume_spike'))]
        if bucket_trades:
            bucket_wins = sum(1 for t in bucket_trades if t['profit_loss'] > 0)
            bucket_pnl = sum(t['profit_loss'] for t in bucket_trades)
            wr = bucket_wins / len(bucket_trades) * 100
            diff = wr - overall_wr
            avg_pnl = bucket_pnl / len(bucket_trades)
            print(f"  {bucket_name}: {len(bucket_trades)} trades, {wr:.1f}% WR, avg ${avg_pnl:.2f} [{diff:+.1f}%]")
            results.append(('VolSpike ' + bucket_name, len(bucket_trades), wr, diff, bucket_pnl))

    # 6. Confidence Level Buckets
    print("\n=== 6. ML CONFIDENCE LEVELS ===")
    conf_buckets = {
        'Very Low (0-15%)': lambda c: c is not None and c < 0.15,
        'Low (15-25%)': lambda c: c is not None and 0.15 <= c < 0.25,
        'Medium (25-40%)': lambda c: c is not None and 0.25 <= c < 0.40,
        'High (40-60%)': lambda c: c is not None and 0.40 <= c < 0.60,
        'Very High (60-80%)': lambda c: c is not None and 0.60 <= c < 0.80,
        'Ultra High (>80%)': lambda c: c is not None and c >= 0.80,
    }

    for bucket_name, condition in conf_buckets.items():
        bucket_trades = [t for t in trades if condition(t.get('ml_confidence'))]
        if bucket_trades:
            bucket_wins = sum(1 for t in bucket_trades if t['profit_loss'] > 0)
            bucket_pnl = sum(t['profit_loss'] for t in bucket_trades)
            wr = bucket_wins / len(bucket_trades) * 100
            diff = wr - overall_wr
            avg_pnl = bucket_pnl / len(bucket_trades)
            print(f"  {bucket_name}: {len(bucket_trades)} trades, {wr:.1f}% WR, avg ${avg_pnl:.2f} [{diff:+.1f}%]")
            results.append(('MLConf ' + bucket_name, len(bucket_trades), wr, diff, bucket_pnl))

    # 6b. Calibrated Confidence
    print("\n=== 6b. CALIBRATED CONFIDENCE ===")
    for bucket_name, condition in conf_buckets.items():
        bucket_trades = [t for t in trades if condition(t.get('calibrated_confidence'))]
        if bucket_trades:
            bucket_wins = sum(1 for t in bucket_trades if t['profit_loss'] > 0)
            bucket_pnl = sum(t['profit_loss'] for t in bucket_trades)
            wr = bucket_wins / len(bucket_trades) * 100
            diff = wr - overall_wr
            avg_pnl = bucket_pnl / len(bucket_trades)
            print(f"  {bucket_name}: {len(bucket_trades)} trades, {wr:.1f}% WR, avg ${avg_pnl:.2f} [{diff:+.1f}%]")
            results.append(('CalConf ' + bucket_name, len(bucket_trades), wr, diff, bucket_pnl))

    # 7. Sequential Patterns
    print("\n=== 7. SEQUENTIAL PATTERNS ===")
    prev_win = None
    after_win = {'wins': 0, 'total': 0, 'pnl': 0}
    after_loss = {'wins': 0, 'total': 0, 'pnl': 0}
    after_2_wins = {'wins': 0, 'total': 0, 'pnl': 0}
    after_2_losses = {'wins': 0, 'total': 0, 'pnl': 0}

    prev_prev_win = None
    for trade in trades:
        curr_win = trade['profit_loss'] > 0

        if prev_win is not None:
            if prev_win:
                after_win['total'] += 1
                after_win['pnl'] += trade['profit_loss']
                if curr_win:
                    after_win['wins'] += 1
            else:
                after_loss['total'] += 1
                after_loss['pnl'] += trade['profit_loss']
                if curr_win:
                    after_loss['wins'] += 1

            # Two in a row
            if prev_prev_win is not None:
                if prev_win and prev_prev_win:
                    after_2_wins['total'] += 1
                    after_2_wins['pnl'] += trade['profit_loss']
                    if curr_win:
                        after_2_wins['wins'] += 1
                elif not prev_win and not prev_prev_win:
                    after_2_losses['total'] += 1
                    after_2_losses['pnl'] += trade['profit_loss']
                    if curr_win:
                        after_2_losses['wins'] += 1

        prev_prev_win = prev_win
        prev_win = curr_win

    if after_win['total'] > 0:
        wr = after_win['wins'] / after_win['total'] * 100
        diff = wr - overall_wr
        avg_pnl = after_win['pnl'] / after_win['total']
        print(f"  After 1 WIN: {after_win['total']} trades, {wr:.1f}% WR, avg ${avg_pnl:.2f} [{diff:+.1f}%]")
        results.append(('After 1 WIN', after_win['total'], wr, diff, after_win['pnl']))

    if after_loss['total'] > 0:
        wr = after_loss['wins'] / after_loss['total'] * 100
        diff = wr - overall_wr
        avg_pnl = after_loss['pnl'] / after_loss['total']
        print(f"  After 1 LOSS: {after_loss['total']} trades, {wr:.1f}% WR, avg ${avg_pnl:.2f} [{diff:+.1f}%]")
        results.append(('After 1 LOSS', after_loss['total'], wr, diff, after_loss['pnl']))

    if after_2_wins['total'] > 0:
        wr = after_2_wins['wins'] / after_2_wins['total'] * 100
        diff = wr - overall_wr
        avg_pnl = after_2_wins['pnl'] / after_2_wins['total']
        print(f"  After 2 WINS in a row: {after_2_wins['total']} trades, {wr:.1f}% WR, avg ${avg_pnl:.2f} [{diff:+.1f}%]")
        results.append(('After 2 WINS', after_2_wins['total'], wr, diff, after_2_wins['pnl']))

    if after_2_losses['total'] > 0:
        wr = after_2_losses['wins'] / after_2_losses['total'] * 100
        diff = wr - overall_wr
        avg_pnl = after_2_losses['pnl'] / after_2_losses['total']
        print(f"  After 2 LOSSES in a row: {after_2_losses['total']} trades, {wr:.1f}% WR, avg ${avg_pnl:.2f} [{diff:+.1f}%]")
        results.append(('After 2 LOSSES', after_2_losses['total'], wr, diff, after_2_losses['pnl']))

    # 8. Exit Reason Analysis
    print("\n=== 8. EXIT REASON ANALYSIS ===")
    exit_reasons = defaultdict(lambda: {'wins': 0, 'total': 0, 'pnls': []})

    for trade in trades:
        reason = trade.get('exit_reason') or 'UNKNOWN'
        exit_reasons[reason]['total'] += 1
        exit_reasons[reason]['pnls'].append(trade['profit_loss'])
        if trade['profit_loss'] > 0:
            exit_reasons[reason]['wins'] += 1

    for reason, stats in sorted(exit_reasons.items(), key=lambda x: x[1]['total'], reverse=True):
        if stats['total'] > 0:
            wr = stats['wins'] / stats['total'] * 100
            avg_pnl = sum(stats['pnls']) / len(stats['pnls'])
            total_pnl = sum(stats['pnls'])
            diff = wr - overall_wr
            print(f"  {reason}: {stats['total']} trades, {wr:.1f}% WR, avg ${avg_pnl:.2f}, total ${total_pnl:.2f} [{diff:+.1f}%]")
            results.append(('Exit ' + reason, stats['total'], wr, diff, total_pnl))

    # 9. Predicted Return at Entry
    print("\n=== 9. PREDICTED RETURN AT ENTRY ===")
    pred_ret_buckets = {
        'Negative (<0%)': lambda r: r is not None and r < 0,
        'Small (0-0.1%)': lambda r: r is not None and 0 <= r < 0.001,
        'Medium (0.1-0.2%)': lambda r: r is not None and 0.001 <= r < 0.002,
        'Good (0.2-0.3%)': lambda r: r is not None and 0.002 <= r < 0.003,
        'Strong (>0.3%)': lambda r: r is not None and r >= 0.003,
    }

    for bucket_name, condition in pred_ret_buckets.items():
        bucket_trades = [t for t in trades if condition(t.get('predicted_return'))]
        if bucket_trades:
            bucket_wins = sum(1 for t in bucket_trades if t['profit_loss'] > 0)
            bucket_pnl = sum(t['profit_loss'] for t in bucket_trades)
            wr = bucket_wins / len(bucket_trades) * 100
            diff = wr - overall_wr
            avg_pnl = bucket_pnl / len(bucket_trades)
            print(f"  {bucket_name}: {len(bucket_trades)} trades, {wr:.1f}% WR, avg ${avg_pnl:.2f} [{diff:+.1f}%]")
            results.append(('PredRet ' + bucket_name, len(bucket_trades), wr, diff, bucket_pnl))

    # 10. Trade Direction vs Momentum Alignment
    print("\n=== 10. MOMENTUM-DIRECTION ALIGNMENT ===")
    aligned_trades = []
    misaligned_trades = []

    for t in trades:
        mom = t.get('momentum_5m')
        if mom is None:
            continue
        is_call = t.get('option_type') == 'CALL'
        mom_up = mom > 0

        if (is_call and mom_up) or (not is_call and not mom_up):
            aligned_trades.append(t)
        else:
            misaligned_trades.append(t)

    if aligned_trades:
        wins = sum(1 for t in aligned_trades if t['profit_loss'] > 0)
        pnl = sum(t['profit_loss'] for t in aligned_trades)
        wr = wins / len(aligned_trades) * 100
        diff = wr - overall_wr
        avg_pnl = pnl / len(aligned_trades)
        print(f"  Aligned (CALL+up / PUT+down): {len(aligned_trades)} trades, {wr:.1f}% WR, avg ${avg_pnl:.2f} [{diff:+.1f}%]")
        results.append(('Mom Aligned', len(aligned_trades), wr, diff, pnl))

    if misaligned_trades:
        wins = sum(1 for t in misaligned_trades if t['profit_loss'] > 0)
        pnl = sum(t['profit_loss'] for t in misaligned_trades)
        wr = wins / len(misaligned_trades) * 100
        diff = wr - overall_wr
        avg_pnl = pnl / len(misaligned_trades)
        print(f"  Contrarian (CALL+down / PUT+up): {len(misaligned_trades)} trades, {wr:.1f}% WR, avg ${avg_pnl:.2f} [{diff:+.1f}%]")
        results.append(('Mom Contrarian', len(misaligned_trades), wr, diff, pnl))

    # 11. Max Drawdown during trade
    print("\n=== 11. MAX DRAWDOWN DURING TRADE ===")
    dd_buckets = {
        'Minimal (<2%)': lambda d: d is not None and d < 2,
        'Small (2-4%)': lambda d: d is not None and 2 <= d < 4,
        'Medium (4-6%)': lambda d: d is not None and 4 <= d < 6,
        'Large (6-8%)': lambda d: d is not None and 6 <= d < 8,
        'Severe (>8%)': lambda d: d is not None and d >= 8,
    }

    for bucket_name, condition in dd_buckets.items():
        bucket_trades = [t for t in trades if condition(t.get('max_drawdown_pct'))]
        if bucket_trades:
            bucket_wins = sum(1 for t in bucket_trades if t['profit_loss'] > 0)
            bucket_pnl = sum(t['profit_loss'] for t in bucket_trades)
            wr = bucket_wins / len(bucket_trades) * 100
            diff = wr - overall_wr
            avg_pnl = bucket_pnl / len(bucket_trades)
            print(f"  {bucket_name}: {len(bucket_trades)} trades, {wr:.1f}% WR, avg ${avg_pnl:.2f} [{diff:+.1f}%]")
            results.append(('MaxDD ' + bucket_name, len(bucket_trades), wr, diff, bucket_pnl))

    # 12. Hold Time Analysis
    print("\n=== 12. HOLD TIME ANALYSIS ===")
    hold_buckets = {
        'Quick (<5 min)': lambda h: h is not None and h < 5,
        'Short (5-10 min)': lambda h: h is not None and 5 <= h < 10,
        'Medium (10-20 min)': lambda h: h is not None and 10 <= h < 20,
        'Long (20-30 min)': lambda h: h is not None and 20 <= h < 30,
        'Extended (>30 min)': lambda h: h is not None and h >= 30,
    }

    for bucket_name, condition in hold_buckets.items():
        bucket_trades = [t for t in trades if condition(t.get('hold_minutes'))]
        if bucket_trades:
            bucket_wins = sum(1 for t in bucket_trades if t['profit_loss'] > 0)
            bucket_pnl = sum(t['profit_loss'] for t in bucket_trades)
            wr = bucket_wins / len(bucket_trades) * 100
            diff = wr - overall_wr
            avg_pnl = bucket_pnl / len(bucket_trades)
            print(f"  {bucket_name}: {len(bucket_trades)} trades, {wr:.1f}% WR, avg ${avg_pnl:.2f} [{diff:+.1f}%]")
            results.append(('Hold ' + bucket_name, len(bucket_trades), wr, diff, bucket_pnl))

    # 13. Signal Strategy Analysis
    print("\n=== 13. SIGNAL STRATEGY ===")
    strategies = defaultdict(lambda: {'wins': 0, 'total': 0, 'pnl': 0})

    for t in trades:
        strategy = t.get('signal_strategy') or 'UNKNOWN'
        strategies[strategy]['total'] += 1
        strategies[strategy]['pnl'] += t['profit_loss']
        if t['profit_loss'] > 0:
            strategies[strategy]['wins'] += 1

    for strategy, stats in sorted(strategies.items(), key=lambda x: x[1]['total'], reverse=True):
        if stats['total'] > 0:
            wr = stats['wins'] / stats['total'] * 100
            avg_pnl = stats['pnl'] / stats['total']
            diff = wr - overall_wr
            print(f"  {strategy}: {stats['total']} trades, {wr:.1f}% WR, avg ${avg_pnl:.2f}, total ${stats['pnl']:.2f} [{diff:+.1f}%]")
            results.append(('Strategy ' + strategy, stats['total'], wr, diff, stats['pnl']))

    # 14. Combination analysis: VIX + Momentum
    print("\n=== 14. VIX + MOMENTUM COMBINATIONS ===")
    combinations = {
        'Low VIX + Pos Mom': lambda t: (t.get('vix_level') or 99) < 15 and (t.get('momentum_5m') or 0) > 0,
        'Low VIX + Neg Mom': lambda t: (t.get('vix_level') or 99) < 15 and (t.get('momentum_5m') or 0) < 0,
        'High VIX + Pos Mom': lambda t: (t.get('vix_level') or 0) >= 20 and (t.get('momentum_5m') or 0) > 0,
        'High VIX + Neg Mom': lambda t: (t.get('vix_level') or 0) >= 20 and (t.get('momentum_5m') or 0) < 0,
    }

    for name, condition in combinations.items():
        bucket_trades = [t for t in trades if condition(t)]
        if bucket_trades:
            bucket_wins = sum(1 for t in bucket_trades if t['profit_loss'] > 0)
            bucket_pnl = sum(t['profit_loss'] for t in bucket_trades)
            wr = bucket_wins / len(bucket_trades) * 100
            diff = wr - overall_wr
            avg_pnl = bucket_pnl / len(bucket_trades)
            print(f"  {name}: {len(bucket_trades)} trades, {wr:.1f}% WR, avg ${avg_pnl:.2f} [{diff:+.1f}%]")
            results.append(('Combo ' + name, len(bucket_trades), wr, diff, bucket_pnl))

    # Print summary table
    print("\n" + "="*90)
    print("SUMMARY TABLE - SORTED BY WIN RATE DIFFERENCE (min 10 samples)")
    print("="*90)
    print(f"{'Pattern':<40} {'Count':>8} {'WR%':>8} {'Diff':>10} {'Total P&L':>12} {'Action':<10}")
    print("-"*90)

    # Sort by WR difference
    results.sort(key=lambda x: x[3], reverse=True)

    for pattern, count, wr, diff, pnl in results:
        if count >= 10:  # Only show patterns with enough samples
            action = "AVOID" if diff < -5 else ("PREFER" if diff > 5 else "neutral")
            print(f"{pattern:<40} {count:>8} {wr:>7.1f}% {diff:>+9.1f}% ${pnl:>10.2f} {action:<10}")

    # Print recommended filters
    print("\n" + "="*70)
    print("RECOMMENDED FILTERS (>8% difference, >10 samples)")
    print("="*70)

    positive_filters = []
    negative_filters = []

    for pattern, count, wr, diff, pnl in results:
        if count >= 10 and abs(diff) > 8:
            if diff > 0:
                positive_filters.append((pattern, count, wr, diff, pnl))
            else:
                negative_filters.append((pattern, count, wr, diff, pnl))

    print("\n🟢 PREFER (Higher Win Rate):")
    for pattern, count, wr, diff, pnl in sorted(positive_filters, key=lambda x: x[3], reverse=True):
        print(f"  {pattern}: {count} trades, {wr:.1f}% WR ({diff:+.1f}% vs avg), P&L ${pnl:.2f}")

    print("\n🔴 AVOID (Lower Win Rate):")
    for pattern, count, wr, diff, pnl in sorted(negative_filters, key=lambda x: x[3]):
        print(f"  {pattern}: {count} trades, {wr:.1f}% WR ({diff:+.1f}% vs avg), P&L ${pnl:.2f}")

if __name__ == "__main__":
    trades = get_trades()
    print(f"Loaded {len(trades)} trades from COMBO_BEST_20K_VALIDATION run")

    if trades:
        # Print sample trade to understand schema
        print("\n=== SAMPLE TRADE FIELDS ===")
        sample = trades[0]
        for key in sorted(sample.keys()):
            val = sample[key]
            if val is not None and val != '':
                print(f"  {key}: {val}")

    analyze_patterns(trades)
