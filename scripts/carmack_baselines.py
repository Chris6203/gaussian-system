#!/usr/bin/env python3
"""
Carmack Baseline Predictors

Philosophy: Start with the simplest thing that could possibly work.
If random does as well as your fancy ML model, your ML model is broken.

Baselines:
1. RANDOM - Pure random (50% calls, 50% puts) - establishes true floor
2. MOMENTUM - Buy direction of last N bars momentum
3. MA_CROSS - Classic moving average crossover
4. VIX_MEAN_REVERSION - Buy calls when VIX high, puts when VIX low
5. TIME_OF_DAY - Trade based on historically profitable times
"""

import os
import sys
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sqlite3

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class BaselinePredictor:
    """Base class for simple predictors."""

    def __init__(self, name: str):
        self.name = name
        self.trades = []

    def predict(self, data: dict) -> dict:
        """
        Returns prediction dict with:
        - action: 'BUY_CALLS', 'BUY_PUTS', or 'HOLD'
        - confidence: 0.0 to 1.0
        - reason: why this prediction
        """
        raise NotImplementedError


class RandomPredictor(BaselinePredictor):
    """
    Pure random baseline - 50% calls, 50% puts.
    If your ML model can't beat this, it's worthless.
    """

    def __init__(self, trade_probability: float = 0.1):
        super().__init__("RANDOM")
        self.trade_probability = trade_probability

    def predict(self, data: dict) -> dict:
        if random.random() > self.trade_probability:
            return {'action': 'HOLD', 'confidence': 0.5, 'reason': 'random_hold'}

        action = 'BUY_CALLS' if random.random() > 0.5 else 'BUY_PUTS'
        return {
            'action': action,
            'confidence': 0.5,  # Always 50% - we're guessing
            'reason': 'random_coin_flip'
        }


class MomentumPredictor(BaselinePredictor):
    """
    Simple momentum: if price went up last N bars, buy calls.
    Classic trend-following.
    """

    def __init__(self, lookback: int = 5, min_move_pct: float = 0.001):
        super().__init__("MOMENTUM")
        self.lookback = lookback
        self.min_move_pct = min_move_pct
        self.price_history = []

    def predict(self, data: dict) -> dict:
        price = data.get('price', data.get('close', 0))
        self.price_history.append(price)

        if len(self.price_history) < self.lookback + 1:
            return {'action': 'HOLD', 'confidence': 0.0, 'reason': 'warming_up'}

        # Keep only what we need
        self.price_history = self.price_history[-(self.lookback + 10):]

        # Calculate momentum
        old_price = self.price_history[-self.lookback - 1]
        momentum = (price - old_price) / old_price

        if abs(momentum) < self.min_move_pct:
            return {'action': 'HOLD', 'confidence': 0.3, 'reason': 'no_momentum'}

        # Trend following: go with the momentum
        if momentum > 0:
            return {
                'action': 'BUY_CALLS',
                'confidence': min(0.8, 0.5 + abs(momentum) * 10),
                'reason': f'momentum_up_{momentum:.4f}'
            }
        else:
            return {
                'action': 'BUY_PUTS',
                'confidence': min(0.8, 0.5 + abs(momentum) * 10),
                'reason': f'momentum_down_{momentum:.4f}'
            }


class MACrossPredictor(BaselinePredictor):
    """
    Classic moving average crossover.
    Fast MA crosses above slow MA = bullish.
    """

    def __init__(self, fast_period: int = 5, slow_period: int = 20):
        super().__init__("MA_CROSS")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.price_history = []
        self.prev_fast_above = None

    def predict(self, data: dict) -> dict:
        price = data.get('price', data.get('close', 0))
        self.price_history.append(price)

        if len(self.price_history) < self.slow_period + 1:
            return {'action': 'HOLD', 'confidence': 0.0, 'reason': 'warming_up'}

        # Keep only what we need
        self.price_history = self.price_history[-(self.slow_period + 10):]

        # Calculate MAs
        fast_ma = np.mean(self.price_history[-self.fast_period:])
        slow_ma = np.mean(self.price_history[-self.slow_period:])

        fast_above = fast_ma > slow_ma

        # Detect crossover
        if self.prev_fast_above is not None and fast_above != self.prev_fast_above:
            self.prev_fast_above = fast_above
            if fast_above:
                return {
                    'action': 'BUY_CALLS',
                    'confidence': 0.6,
                    'reason': 'golden_cross'
                }
            else:
                return {
                    'action': 'BUY_PUTS',
                    'confidence': 0.6,
                    'reason': 'death_cross'
                }

        self.prev_fast_above = fast_above
        return {'action': 'HOLD', 'confidence': 0.4, 'reason': 'no_crossover'}


class VIXMeanReversionPredictor(BaselinePredictor):
    """
    VIX mean reversion: high VIX = buy calls (fear overdone),
    low VIX = buy puts (complacency).
    """

    def __init__(self, high_vix: float = 25, low_vix: float = 15):
        super().__init__("VIX_REVERSION")
        self.high_vix = high_vix
        self.low_vix = low_vix

    def predict(self, data: dict) -> dict:
        vix = data.get('vix', 18)

        if vix > self.high_vix:
            return {
                'action': 'BUY_CALLS',
                'confidence': min(0.8, 0.5 + (vix - self.high_vix) / 20),
                'reason': f'vix_high_{vix:.1f}'
            }
        elif vix < self.low_vix:
            return {
                'action': 'BUY_PUTS',
                'confidence': min(0.8, 0.5 + (self.low_vix - vix) / 10),
                'reason': f'vix_low_{vix:.1f}'
            }
        else:
            return {'action': 'HOLD', 'confidence': 0.4, 'reason': 'vix_neutral'}


class TimeOfDayPredictor(BaselinePredictor):
    """
    Trade based on time of day patterns.
    Morning momentum, lunch reversal, power hour trend.
    """

    def __init__(self):
        super().__init__("TIME_OF_DAY")
        self.morning_momentum_end = 10  # 10:00 AM
        self.lunch_start = 12
        self.lunch_end = 14
        self.power_hour_start = 15

    def predict(self, data: dict) -> dict:
        timestamp = data.get('timestamp')
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))

        hour = timestamp.hour if timestamp else 12
        price = data.get('price', data.get('close', 0))
        open_price = data.get('open', price)

        # Morning momentum (9:30-10:00): follow overnight gap
        if hour < self.morning_momentum_end:
            if price > open_price * 1.001:
                return {'action': 'BUY_CALLS', 'confidence': 0.55, 'reason': 'morning_gap_up'}
            elif price < open_price * 0.999:
                return {'action': 'BUY_PUTS', 'confidence': 0.55, 'reason': 'morning_gap_down'}

        # Lunch doldrums (12-2): don't trade
        if self.lunch_start <= hour < self.lunch_end:
            return {'action': 'HOLD', 'confidence': 0.3, 'reason': 'lunch_avoid'}

        # Power hour (3-4): trend continuation
        if hour >= self.power_hour_start:
            day_change = (price - open_price) / open_price
            if day_change > 0.002:
                return {'action': 'BUY_CALLS', 'confidence': 0.6, 'reason': 'power_hour_bull'}
            elif day_change < -0.002:
                return {'action': 'BUY_PUTS', 'confidence': 0.6, 'reason': 'power_hour_bear'}

        return {'action': 'HOLD', 'confidence': 0.4, 'reason': 'no_pattern'}


class CombinedPredictor(BaselinePredictor):
    """
    Combine multiple predictors with voting.
    Only trade when multiple agree.
    """

    def __init__(self, predictors: list, min_agreement: int = 2):
        super().__init__("COMBINED")
        self.predictors = predictors
        self.min_agreement = min_agreement

    def predict(self, data: dict) -> dict:
        votes = {'BUY_CALLS': 0, 'BUY_PUTS': 0, 'HOLD': 0}
        reasons = []

        for pred in self.predictors:
            result = pred.predict(data)
            action = result['action']
            votes[action] += 1
            if action != 'HOLD':
                reasons.append(f"{pred.name}:{result['reason']}")

        # Find winner
        max_votes = max(votes['BUY_CALLS'], votes['BUY_PUTS'])

        if max_votes >= self.min_agreement:
            if votes['BUY_CALLS'] > votes['BUY_PUTS']:
                return {
                    'action': 'BUY_CALLS',
                    'confidence': 0.5 + 0.1 * max_votes,
                    'reason': '|'.join(reasons)
                }
            elif votes['BUY_PUTS'] > votes['BUY_CALLS']:
                return {
                    'action': 'BUY_PUTS',
                    'confidence': 0.5 + 0.1 * max_votes,
                    'reason': '|'.join(reasons)
                }

        return {'action': 'HOLD', 'confidence': 0.4, 'reason': 'no_consensus'}


def run_baseline_backtest(predictor: BaselinePredictor, max_cycles: int = 5000):
    """
    Run a baseline predictor through proper backtesting.
    Uses actual historical data and realistic P&L calculation.
    """
    print(f"\n{'='*60}")
    print(f"CARMACK BASELINE: {predictor.name}")
    print(f"{'='*60}")

    # Load historical data
    conn = sqlite3.connect('data/db/historical.db')

    # Get SPY minute data from historical_data table
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

    if len(df) < 100:
        print(f"ERROR: Only {len(df)} rows of data available")
        return None

    print(f"Loaded {len(df)} bars of data")

    # Simulation state
    balance = 5000.0
    initial_balance = balance
    position = None
    trades = []
    wins = 0
    losses = 0

    # Options parameters
    OPTION_COST = 100  # $1 option × 100 multiplier
    STOP_LOSS_PCT = -0.08  # -8%
    TAKE_PROFIT_PCT = 0.12  # +12%
    MAX_HOLD_BARS = 45  # 45 minutes

    for i in range(60, min(len(df), max_cycles + 60)):
        row = df.iloc[i]
        price = row['close']
        timestamp = row['timestamp']

        # Build data dict for predictor
        data = {
            'price': price,
            'close': price,
            'open': row['open'],
            'high': row['high'],
            'low': row['low'],
            'volume': row['volume'],
            'timestamp': timestamp,
            'vix': 18  # Default VIX
        }

        # Check existing position
        if position:
            bars_held = i - position['entry_bar']

            # Calculate P&L based on price movement
            price_change = (price - position['entry_price']) / position['entry_price']

            # Options have leverage (~5x for ATM)
            if position['type'] == 'CALL':
                option_pnl_pct = price_change * 5  # Calls profit when price goes up
            else:
                option_pnl_pct = -price_change * 5  # Puts profit when price goes down

            # Check exit conditions
            should_exit = False
            exit_reason = ""

            if option_pnl_pct <= STOP_LOSS_PCT:
                should_exit = True
                exit_reason = "stop_loss"
            elif option_pnl_pct >= TAKE_PROFIT_PCT:
                should_exit = True
                exit_reason = "take_profit"
            elif bars_held >= MAX_HOLD_BARS:
                should_exit = True
                exit_reason = "max_hold"

            if should_exit:
                pnl = OPTION_COST * option_pnl_pct
                balance += pnl

                if pnl > 0:
                    wins += 1
                else:
                    losses += 1

                trades.append({
                    'entry_time': position['entry_time'],
                    'exit_time': timestamp,
                    'type': position['type'],
                    'pnl': pnl,
                    'exit_reason': exit_reason
                })
                position = None

        # Get prediction (only if no position)
        if position is None:
            pred = predictor.predict(data)

            if pred['action'] in ('BUY_CALLS', 'BUY_PUTS') and balance >= OPTION_COST:
                position = {
                    'type': 'CALL' if pred['action'] == 'BUY_CALLS' else 'PUT',
                    'entry_price': price,
                    'entry_bar': i,
                    'entry_time': timestamp,
                    'reason': pred['reason']
                }
                balance -= OPTION_COST  # Pay for option

        # Progress
        if (i - 60) % 500 == 0:
            total_trades = wins + losses
            wr = 100 * wins / total_trades if total_trades > 0 else 0
            pnl_pct = 100 * (balance - initial_balance) / initial_balance
            print(f"  Cycle {i-60}: Balance ${balance:.2f} ({pnl_pct:+.2f}%), "
                  f"Trades: {total_trades}, WR: {wr:.1f}%")

    # Close any open position at end
    if position:
        pnl = 0  # Assume flat exit
        balance += OPTION_COST  # Return option cost
        trades.append({
            'entry_time': position['entry_time'],
            'exit_time': df.iloc[-1]['timestamp'],
            'type': position['type'],
            'pnl': pnl,
            'exit_reason': 'end_of_test'
        })

    # Results
    total_trades = wins + losses
    win_rate = 100 * wins / total_trades if total_trades > 0 else 0
    pnl = balance - initial_balance
    pnl_pct = 100 * pnl / initial_balance

    print(f"\n{'='*40}")
    print(f"RESULTS: {predictor.name}")
    print(f"{'='*40}")
    print(f"Initial Balance: ${initial_balance:.2f}")
    print(f"Final Balance:   ${balance:.2f}")
    print(f"P&L:             ${pnl:.2f} ({pnl_pct:+.2f}%)")
    print(f"Total Trades:    {total_trades}")
    print(f"Win Rate:        {win_rate:.1f}% ({wins}W/{losses}L)")

    if trades:
        avg_win = np.mean([t['pnl'] for t in trades if t['pnl'] > 0]) if wins > 0 else 0
        avg_loss = np.mean([t['pnl'] for t in trades if t['pnl'] < 0]) if losses > 0 else 0
        print(f"Avg Win:         ${avg_win:.2f}")
        print(f"Avg Loss:        ${avg_loss:.2f}")

    return {
        'name': predictor.name,
        'final_balance': balance,
        'pnl': pnl,
        'pnl_pct': pnl_pct,
        'trades': total_trades,
        'win_rate': win_rate,
        'wins': wins,
        'losses': losses
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Carmack baseline predictors")
    parser.add_argument('--predictor', type=str, default='all',
                       choices=['random', 'momentum', 'ma_cross', 'vix', 'time', 'combined', 'all'])
    parser.add_argument('--cycles', type=int, default=5000)
    args = parser.parse_args()

    predictors = {
        'random': RandomPredictor(trade_probability=0.05),
        'momentum': MomentumPredictor(lookback=5, min_move_pct=0.0005),
        'ma_cross': MACrossPredictor(fast_period=5, slow_period=20),
        'vix': VIXMeanReversionPredictor(high_vix=22, low_vix=14),
        'time': TimeOfDayPredictor(),
    }

    # Combined uses all others
    predictors['combined'] = CombinedPredictor(
        [MomentumPredictor(), MACrossPredictor(), VIXMeanReversionPredictor()],
        min_agreement=2
    )

    results = []

    if args.predictor == 'all':
        for name, pred in predictors.items():
            result = run_baseline_backtest(pred, args.cycles)
            if result:
                results.append(result)
    else:
        result = run_baseline_backtest(predictors[args.predictor], args.cycles)
        if result:
            results.append(result)

    # Summary table
    if results:
        print(f"\n{'='*70}")
        print("CARMACK BASELINE COMPARISON")
        print(f"{'='*70}")
        print(f"{'Predictor':<15} {'P&L':>10} {'P&L%':>8} {'Trades':>8} {'Win Rate':>10}")
        print("-" * 70)
        for r in sorted(results, key=lambda x: x['pnl_pct'], reverse=True):
            print(f"{r['name']:<15} ${r['pnl']:>9.2f} {r['pnl_pct']:>7.2f}% {r['trades']:>8} {r['win_rate']:>9.1f}%")
