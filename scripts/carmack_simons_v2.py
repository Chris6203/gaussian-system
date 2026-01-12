#!/usr/bin/env python3
"""
Carmack-Simons V2: Risk-Managed Mean Reversion

Key Fixes:
1. Reduced position size: $25 per trade (not $100)
2. Shorter holds: 15 min max (not 45)
3. Higher entry threshold: 0.5 (not 0.25)
4. Tighter stops: -5% stop, +8% target
5. Only trade during high-probability windows
"""

import os
import sys
import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class RiskManagedPredictor:
    """
    Mean reversion with proper risk management.
    """

    def __init__(self, config=None):
        self.name = "CARMACK_SIMONS_V2"
        self.price_history = []
        self.lookback = 60

        # Configuration with defaults
        config = config or {}
        self.entry_threshold = config.get('entry_threshold', 0.5)
        self.min_rsi_extreme = config.get('min_rsi_extreme', 30)  # RSI < 30 or > 70
        self.max_rsi_extreme = config.get('max_rsi_extreme', 70)

    def calculate_rsi(self, prices, period=14):
        if len(prices) < period + 1:
            return 50.0
        deltas = np.diff(prices[-period-1:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def calculate_bb_position(self, prices, period=20):
        if len(prices) < period:
            return 0.0
        recent = prices[-period:]
        middle = np.mean(recent)
        std = np.std(recent)
        if std == 0:
            return 0.0
        current = prices[-1]
        return np.clip((current - middle) / (2 * std), -2, 2)

    def calculate_momentum(self, prices, period=15):
        if len(prices) < period + 1:
            return 0.0
        return (prices[-1] - prices[-period-1]) / prices[-period-1]

    def predict(self, data: dict) -> dict:
        price = data.get('price', data.get('close', 0))
        self.price_history.append(price)

        if len(self.price_history) < self.lookback:
            return {'action': 'HOLD', 'confidence': 0.0, 'reason': 'warming_up'}

        self.price_history = self.price_history[-200:]
        prices = np.array(self.price_history)

        # Calculate signals
        rsi = self.calculate_rsi(prices, 14)
        bb_pos = self.calculate_bb_position(prices, 20)
        momentum = self.calculate_momentum(prices, 15)

        # Time filter - avoid lunch (11-14) and last 30 min
        timestamp = data.get('timestamp')
        hour = 12
        minute = 0
        if timestamp:
            if isinstance(timestamp, str):
                try:
                    timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                except:
                    pass
            if hasattr(timestamp, 'hour'):
                hour = timestamp.hour
                minute = timestamp.minute

        # Skip low-probability times
        if 11 <= hour < 14:  # Lunch doldrums
            return {'action': 'HOLD', 'confidence': 0.0, 'reason': 'lunch_avoid'}
        if hour >= 15 and minute >= 30:  # Last 30 min volatility
            return {'action': 'HOLD', 'confidence': 0.0, 'reason': 'eod_avoid'}

        # Composite overbought score (higher = more overbought)
        overbought_score = (
            0.40 * (bb_pos / 1.5) +
            0.35 * ((rsi - 50) / 50) +
            0.25 * np.clip(momentum * 20, -1, 1)
        )

        # STRICT entry criteria - only trade extremes
        if overbought_score > self.entry_threshold and rsi > self.max_rsi_extreme:
            confidence = min(0.85, 0.6 + abs(overbought_score) * 0.2)
            return {
                'action': 'BUY_PUTS',
                'confidence': confidence,
                'reason': f'overbought_extreme (RSI={rsi:.0f}, BB={bb_pos:.2f})',
                'signal_strength': overbought_score
            }
        elif overbought_score < -self.entry_threshold and rsi < self.min_rsi_extreme:
            confidence = min(0.85, 0.6 + abs(overbought_score) * 0.2)
            return {
                'action': 'BUY_CALLS',
                'confidence': confidence,
                'reason': f'oversold_extreme (RSI={rsi:.0f}, BB={bb_pos:.2f})',
                'signal_strength': overbought_score
            }

        return {'action': 'HOLD', 'confidence': 0.3, 'reason': 'not_extreme'}


def run_backtest(config=None, max_cycles=5000):
    """Run risk-managed backtest."""

    # Default config with Carmack/Simons fixes
    default_config = {
        'position_size': 25,       # $25 per trade (not $100)
        'stop_loss_pct': -0.05,    # -5% stop (tighter)
        'take_profit_pct': 0.08,   # +8% target
        'max_hold_bars': 15,       # 15 min max (not 45)
        'entry_threshold': 0.5,    # Higher threshold
        'leverage': 5,             # Options leverage
    }

    config = {**default_config, **(config or {})}

    print("="*70)
    print("CARMACK-SIMONS V2: Risk-Managed Mean Reversion")
    print("="*70)
    print("\nRisk Management Fixes:")
    print(f"  Position size:  ${config['position_size']} (was $100)")
    print(f"  Stop loss:      {config['stop_loss_pct']*100:.0f}% (was -8%)")
    print(f"  Take profit:    {config['take_profit_pct']*100:.0f}% (was +12%)")
    print(f"  Max hold:       {config['max_hold_bars']} bars (was 45)")
    print(f"  Entry threshold: {config['entry_threshold']} (was 0.25)")
    print()

    predictor = RiskManagedPredictor({
        'entry_threshold': config['entry_threshold']
    })

    # Load data
    conn = sqlite3.connect('data/db/historical.db')
    df = pd.read_sql_query("""
        SELECT timestamp,
               open_price as open,
               high_price as high,
               low_price as low,
               close_price as close,
               volume
        FROM historical_data
        WHERE symbol = 'SPY'
        AND timestamp >= '2025-06-01'
        ORDER BY timestamp
        LIMIT ?
    """, conn, params=(max_cycles * 2,))
    conn.close()

    print(f"Loaded {len(df)} bars of data")
    print(f"Date range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")

    # Simulation state
    balance = 5000.0
    initial_balance = balance
    position = None
    trades = []
    wins = 0
    losses = 0
    max_balance = balance
    max_drawdown = 0

    POSITION_SIZE = config['position_size']
    STOP_LOSS = config['stop_loss_pct']
    TAKE_PROFIT = config['take_profit_pct']
    MAX_HOLD = config['max_hold_bars']
    LEVERAGE = config['leverage']

    for i in range(60, min(len(df), max_cycles + 60)):
        row = df.iloc[i]
        price = row['close']
        timestamp = row['timestamp']

        data = {
            'price': price,
            'close': price,
            'open': row['open'],
            'high': row['high'],
            'low': row['low'],
            'volume': row['volume'],
            'timestamp': timestamp
        }

        # Track drawdown
        if balance > max_balance:
            max_balance = balance
        current_dd = (max_balance - balance) / max_balance
        if current_dd > max_drawdown:
            max_drawdown = current_dd

        # Check existing position
        if position:
            bars_held = i - position['entry_bar']
            price_change = (price - position['entry_price']) / position['entry_price']

            # Option P&L with leverage
            if position['type'] == 'CALL':
                option_pnl_pct = price_change * LEVERAGE
            else:
                option_pnl_pct = -price_change * LEVERAGE

            should_exit = False
            exit_reason = ""

            if option_pnl_pct <= STOP_LOSS:
                should_exit = True
                exit_reason = "stop_loss"
            elif option_pnl_pct >= TAKE_PROFIT:
                should_exit = True
                exit_reason = "take_profit"
            elif bars_held >= MAX_HOLD:
                should_exit = True
                exit_reason = "max_hold"

            if should_exit:
                pnl = POSITION_SIZE * option_pnl_pct
                balance += POSITION_SIZE + pnl  # Return position + P&L

                if pnl > 0:
                    wins += 1
                else:
                    losses += 1

                trades.append({
                    'entry_time': position['entry_time'],
                    'exit_time': timestamp,
                    'type': position['type'],
                    'pnl': pnl,
                    'pnl_pct': option_pnl_pct * 100,
                    'exit_reason': exit_reason,
                    'reason': position['reason'],
                    'bars_held': bars_held
                })
                position = None

        # Get prediction (only if no position and have enough capital)
        if position is None and balance >= POSITION_SIZE:
            pred = predictor.predict(data)

            if pred['action'] in ('BUY_CALLS', 'BUY_PUTS'):
                position = {
                    'type': 'CALL' if pred['action'] == 'BUY_CALLS' else 'PUT',
                    'entry_price': price,
                    'entry_bar': i,
                    'entry_time': timestamp,
                    'reason': pred['reason']
                }
                balance -= POSITION_SIZE

        # Progress update
        if (i - 60) % 500 == 0:
            total_trades = wins + losses
            wr = 100 * wins / total_trades if total_trades > 0 else 0
            pnl_pct = 100 * (balance - initial_balance) / initial_balance
            print(f"  Cycle {i-60}: Balance ${balance:.2f} ({pnl_pct:+.2f}%), "
                  f"Trades: {total_trades}, WR: {wr:.1f}%, DD: {max_drawdown*100:.1f}%")

    # Close any open position
    if position:
        balance += POSITION_SIZE  # Return position at cost
        trades.append({
            'entry_time': position['entry_time'],
            'exit_time': df.iloc[-1]['timestamp'],
            'type': position['type'],
            'pnl': 0,
            'exit_reason': 'end_of_test'
        })

    # Final results
    total_trades = wins + losses
    win_rate = 100 * wins / total_trades if total_trades > 0 else 0
    pnl = balance - initial_balance
    pnl_pct = 100 * pnl / initial_balance

    print(f"\n{'='*60}")
    print(f"FINAL RESULTS: CARMACK_SIMONS_V2")
    print(f"{'='*60}")
    print(f"Initial Balance: ${initial_balance:.2f}")
    print(f"Final Balance:   ${balance:.2f}")
    print(f"P&L:             ${pnl:.2f} ({pnl_pct:+.2f}%)")
    print(f"Max Drawdown:    {max_drawdown*100:.1f}%")
    print(f"Total Trades:    {total_trades}")
    print(f"Win Rate:        {win_rate:.1f}% ({wins}W/{losses}L)")

    if trades:
        winning_trades = [t for t in trades if t['pnl'] > 0]
        losing_trades = [t for t in trades if t['pnl'] < 0]

        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0

        print(f"\nAvg Win:         ${avg_win:.2f}")
        print(f"Avg Loss:        ${avg_loss:.2f}")

        if avg_loss != 0:
            profit_factor = abs(sum(t['pnl'] for t in winning_trades) /
                              sum(t['pnl'] for t in losing_trades)) if losing_trades else 0
            print(f"Profit Factor:   {profit_factor:.2f}")

        # Exit reason analysis
        print(f"\nExit Analysis:")
        for reason in ['stop_loss', 'take_profit', 'max_hold']:
            reason_trades = [t for t in trades if t.get('exit_reason') == reason]
            if reason_trades:
                count = len(reason_trades)
                reason_pnl = sum(t['pnl'] for t in reason_trades)
                reason_wr = 100 * sum(1 for t in reason_trades if t['pnl'] > 0) / count
                avg_bars = np.mean([t.get('bars_held', 0) for t in reason_trades])
                print(f"  {reason:12}: {count:3} trades, ${reason_pnl:>7.2f}, "
                      f"WR: {reason_wr:.0f}%, Avg hold: {avg_bars:.0f} bars")

        # Trade type analysis
        print(f"\nTrade Type Analysis:")
        for trade_type in ['CALL', 'PUT']:
            type_trades = [t for t in trades if t['type'] == trade_type]
            if type_trades:
                count = len(type_trades)
                type_pnl = sum(t['pnl'] for t in type_trades)
                type_wr = 100 * sum(1 for t in type_trades if t['pnl'] > 0) / count
                print(f"  {trade_type:4}: {count:3} trades, ${type_pnl:>7.2f}, WR: {type_wr:.0f}%")

    return {
        'final_balance': balance,
        'pnl': pnl,
        'pnl_pct': pnl_pct,
        'trades': total_trades,
        'win_rate': win_rate,
        'max_drawdown': max_drawdown
    }


def run_parameter_sweep():
    """Test different configurations to find optimal."""
    print("\n" + "="*70)
    print("PARAMETER SWEEP: Finding Optimal Configuration")
    print("="*70 + "\n")

    configs = [
        {'name': 'Baseline (Original)', 'position_size': 100, 'stop_loss_pct': -0.08,
         'take_profit_pct': 0.12, 'max_hold_bars': 45, 'entry_threshold': 0.25},

        {'name': 'Small Position', 'position_size': 25, 'stop_loss_pct': -0.08,
         'take_profit_pct': 0.12, 'max_hold_bars': 45, 'entry_threshold': 0.25},

        {'name': 'Tight Stops', 'position_size': 25, 'stop_loss_pct': -0.05,
         'take_profit_pct': 0.08, 'max_hold_bars': 45, 'entry_threshold': 0.25},

        {'name': 'Quick Exit', 'position_size': 25, 'stop_loss_pct': -0.05,
         'take_profit_pct': 0.08, 'max_hold_bars': 15, 'entry_threshold': 0.25},

        {'name': 'High Threshold', 'position_size': 25, 'stop_loss_pct': -0.05,
         'take_profit_pct': 0.08, 'max_hold_bars': 15, 'entry_threshold': 0.5},

        {'name': 'Very Strict', 'position_size': 25, 'stop_loss_pct': -0.04,
         'take_profit_pct': 0.06, 'max_hold_bars': 10, 'entry_threshold': 0.6},
    ]

    results = []
    for cfg in configs:
        name = cfg.pop('name')
        print(f"\n--- Testing: {name} ---")
        result = run_backtest(cfg, max_cycles=5000)
        result['name'] = name
        results.append(result)

    # Summary
    print("\n" + "="*70)
    print("PARAMETER SWEEP RESULTS")
    print("="*70)
    print(f"\n{'Config':<25} {'P&L':>10} {'P&L%':>8} {'Trades':>8} {'WR':>6} {'MaxDD':>8}")
    print("-"*70)

    for r in sorted(results, key=lambda x: x['pnl_pct'], reverse=True):
        print(f"{r['name']:<25} ${r['pnl']:>9.2f} {r['pnl_pct']:>7.2f}% "
              f"{r['trades']:>8} {r['win_rate']:>5.1f}% {r['max_drawdown']*100:>7.1f}%")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--sweep', action='store_true', help='Run parameter sweep')
    parser.add_argument('--cycles', type=int, default=5000)
    args = parser.parse_args()

    if args.sweep:
        run_parameter_sweep()
    else:
        run_backtest(max_cycles=args.cycles)
