#!/usr/bin/env python3
"""
Simons Predictor Backtest

Test the Simons-style predictor using multiple weak signals
combined into a strong edge.
"""

import os
import sys
import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.simons_predictor import SimonsPredictor, SimonsPrediction


def load_data(start_date='2025-06-01', limit=10000):
    """Load SPY and VIX data."""
    conn = sqlite3.connect('data/db/historical.db')

    # Load SPY
    spy = pd.read_sql_query("""
        SELECT timestamp,
               open_price as open,
               high_price as high,
               low_price as low,
               close_price as close,
               volume
        FROM historical_data
        WHERE symbol = 'SPY'
        AND timestamp >= ?
        ORDER BY timestamp
        LIMIT ?
    """, conn, params=(start_date, limit))

    # Load VIX
    vix = pd.read_sql_query("""
        SELECT timestamp, close_price as vix
        FROM historical_data
        WHERE symbol IN ('VIX', '^VIX')
        AND timestamp >= ?
        ORDER BY timestamp
        LIMIT ?
    """, conn, params=(start_date, limit))

    conn.close()

    spy['timestamp'] = pd.to_datetime(spy['timestamp'], format='ISO8601', utc=True)
    vix['timestamp'] = pd.to_datetime(vix['timestamp'], format='ISO8601', utc=True)

    # Merge
    df = spy.merge(vix, on='timestamp', how='left')
    df['vix'] = df['vix'].ffill()

    return df


def run_backtest(max_cycles=5000, config=None):
    """Run backtest of Simons predictor."""

    if config is None:
        config = {
            'name': 'Default',
            'min_signal': 0.3,
            'option_cost': 50,
            'stop_loss': -0.08,
            'take_profit': 0.12,
            'max_hold': 30,  # Shorter holds
        }

    print(f"\n{'='*60}")
    print(f"SIMONS PREDICTOR BACKTEST: {config['name']}")
    print(f"{'='*60}")

    # Create predictor (bypass env var)
    predictor = SimonsPredictor()
    predictor.enabled = True
    predictor.min_signal = config['min_signal']

    # Load data
    df = load_data(limit=max_cycles + 200)
    print(f"Loaded {len(df)} bars")

    # Simulation state
    balance = 5000.0
    initial = balance
    position = None
    trades = []
    wins = 0
    losses = 0

    OPTION_COST = config['option_cost']
    STOP_LOSS = config['stop_loss']
    TAKE_PROFIT = config['take_profit']
    MAX_HOLD = config['max_hold']

    for i in range(60, min(len(df), max_cycles + 60)):
        row = df.iloc[i]
        spy_price = row['close']
        vix_price = row['vix'] if pd.notna(row['vix']) else None
        volume = row['volume']
        timestamp = row['timestamp']

        # Update predictor
        predictor.update(spy_price, vix_price, volume=volume)

        # Check existing position
        if position:
            bars_held = i - position['entry_bar']
            price_change = (spy_price - position['entry_price']) / position['entry_price']

            if position['type'] == 'CALL':
                pnl_pct = price_change * 5  # 5x leverage
            else:
                pnl_pct = -price_change * 5

            should_exit = False
            exit_reason = ""

            if pnl_pct <= STOP_LOSS:
                should_exit = True
                exit_reason = "stop_loss"
            elif pnl_pct >= TAKE_PROFIT:
                should_exit = True
                exit_reason = "take_profit"
            elif bars_held >= MAX_HOLD:
                should_exit = True
                exit_reason = "max_hold"

            if should_exit:
                pnl = OPTION_COST * pnl_pct
                balance += pnl + OPTION_COST  # Return cost + P&L

                if pnl > 0:
                    wins += 1
                else:
                    losses += 1

                trades.append({
                    'entry_time': position['entry_time'],
                    'exit_time': timestamp,
                    'type': position['type'],
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'exit_reason': exit_reason,
                    'signal': position['signal'],
                    'components': position['components']
                })
                position = None

        # Get prediction
        if position is None and balance >= OPTION_COST:
            pred = predictor.predict(timestamp)

            if pred.action in ('BUY_CALLS', 'BUY_PUTS'):
                position = {
                    'type': 'CALL' if pred.action == 'BUY_CALLS' else 'PUT',
                    'entry_price': spy_price,
                    'entry_bar': i,
                    'entry_time': timestamp,
                    'signal': pred.signal_strength,
                    'components': pred.components.copy()
                }
                balance -= OPTION_COST

        # Progress
        if (i - 60) % 1000 == 0:
            total = wins + losses
            wr = 100 * wins / total if total > 0 else 0
            pnl_pct = 100 * (balance - initial) / initial
            print(f"  Cycle {i-60}: ${balance:.2f} ({pnl_pct:+.1f}%), Trades: {total}, WR: {wr:.1f}%")

    # Final results
    total = wins + losses
    win_rate = 100 * wins / total if total > 0 else 0
    pnl = balance - initial
    pnl_pct = 100 * pnl / initial

    print(f"\n{'='*50}")
    print(f"RESULTS: {config['name']}")
    print(f"{'='*50}")
    print(f"Final Balance: ${balance:.2f}")
    print(f"P&L: ${pnl:.2f} ({pnl_pct:+.2f}%)")
    print(f"Trades: {total}")
    print(f"Win Rate: {win_rate:.1f}% ({wins}W/{losses}L)")

    if trades:
        avg_win = np.mean([t['pnl'] for t in trades if t['pnl'] > 0]) if wins else 0
        avg_loss = np.mean([t['pnl'] for t in trades if t['pnl'] < 0]) if losses else 0
        print(f"Avg Win: ${avg_win:.2f}")
        print(f"Avg Loss: ${avg_loss:.2f}")

        if avg_loss != 0:
            pf = abs(avg_win * wins / (avg_loss * losses)) if losses else float('inf')
            print(f"Profit Factor: {pf:.2f}")

        # Exit analysis
        print(f"\nExit Reasons:")
        for reason in ['stop_loss', 'take_profit', 'max_hold']:
            count = sum(1 for t in trades if t['exit_reason'] == reason)
            reason_pnl = sum(t['pnl'] for t in trades if t['exit_reason'] == reason)
            if count > 0:
                print(f"  {reason}: {count} trades, ${reason_pnl:.2f}")

        # Signal strength analysis
        print(f"\nSignal Analysis:")
        strong_signals = [t for t in trades if abs(t['signal']) > 0.4]
        weak_signals = [t for t in trades if abs(t['signal']) <= 0.4]

        if strong_signals:
            strong_wr = 100 * sum(1 for t in strong_signals if t['pnl'] > 0) / len(strong_signals)
            strong_pnl = sum(t['pnl'] for t in strong_signals)
            print(f"  Strong signals (|s|>0.4): {len(strong_signals)} trades, WR={strong_wr:.1f}%, P&L=${strong_pnl:.2f}")

        if weak_signals:
            weak_wr = 100 * sum(1 for t in weak_signals if t['pnl'] > 0) / len(weak_signals)
            weak_pnl = sum(t['pnl'] for t in weak_signals)
            print(f"  Weak signals (|s|<=0.4): {len(weak_signals)} trades, WR={weak_wr:.1f}%, P&L=${weak_pnl:.2f}")

        # Component analysis
        print(f"\nComponent Contribution Analysis:")
        for comp in ['spy_vix_corr', 'bb_position', 'rsi', 'momentum', 'vix_regime']:
            comp_trades = [t for t in trades if comp in t['components']]
            if comp_trades:
                # Split by component direction
                pos_comp = [t for t in comp_trades if t['components'].get(comp, 0) > 0]
                neg_comp = [t for t in comp_trades if t['components'].get(comp, 0) < 0]

                if pos_comp:
                    pos_wr = 100 * sum(1 for t in pos_comp if t['pnl'] > 0) / len(pos_comp)
                    print(f"  {comp} > 0: {len(pos_comp)} trades, WR={pos_wr:.1f}%")
                if neg_comp:
                    neg_wr = 100 * sum(1 for t in neg_comp if t['pnl'] > 0) / len(neg_comp)
                    print(f"  {comp} < 0: {len(neg_comp)} trades, WR={neg_wr:.1f}%")

    return {
        'balance': balance,
        'pnl': pnl,
        'pnl_pct': pnl_pct,
        'trades': total,
        'win_rate': win_rate,
        'config': config['name']
    }


def main():
    print("="*70)
    print("SIMONS PREDICTOR BACKTEST SUITE")
    print("="*70)
    print("\nTesting different configurations...")

    configs = [
        {
            'name': 'Default',
            'min_signal': 0.3,
            'option_cost': 50,
            'stop_loss': -0.08,
            'take_profit': 0.12,
            'max_hold': 30,
        },
        {
            'name': 'Conservative (higher threshold)',
            'min_signal': 0.4,
            'option_cost': 50,
            'stop_loss': -0.06,
            'take_profit': 0.10,
            'max_hold': 25,
        },
        {
            'name': 'Aggressive (lower threshold)',
            'min_signal': 0.25,
            'option_cost': 50,
            'stop_loss': -0.10,
            'take_profit': 0.15,
            'max_hold': 35,
        },
        {
            'name': 'VIX-Focused (stronger signal req)',
            'min_signal': 0.35,
            'option_cost': 50,
            'stop_loss': -0.07,
            'take_profit': 0.12,
            'max_hold': 30,
        },
    ]

    results = []
    for config in configs:
        result = run_backtest(max_cycles=5000, config=config)
        results.append(result)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\n{'Config':<35} {'P&L':>10} {'Trades':>8} {'Win Rate':>10}")
    print("-"*70)

    for r in sorted(results, key=lambda x: x['pnl'], reverse=True):
        print(f"{r['config']:<35} ${r['pnl']:>+9.2f} {r['trades']:>8} {r['win_rate']:>9.1f}%")

    best = max(results, key=lambda x: x['pnl'])
    print(f"\nBest config: {best['config']} (P&L: ${best['pnl']:.2f})")


if __name__ == "__main__":
    main()
