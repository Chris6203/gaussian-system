#!/usr/bin/env python3
"""
Mean Reversion Predictor

Based on Simons' correlation analysis:
- All features show NEGATIVE correlation with forward returns
- This means the market MEAN REVERTS at 15-minute timeframe
- Strategy: Fade the current momentum

Key signals (from correlation analysis):
- bb_position: -0.058 (overbought → sell, oversold → buy)
- rsi_14: -0.054 (high RSI → sell, low RSI → buy)
- momentum_15m: -0.050 (up momentum → sell, down momentum → buy)
"""

import os
import sys
import numpy as np
import pandas as pd
import sqlite3

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class MeanReversionPredictor:
    """
    Mean reversion strategy based on Simons' correlation analysis.
    FADE the current momentum.
    """

    def __init__(self):
        self.name = "MEAN_REVERSION"
        self.price_history = []
        self.lookback = 60

    def calculate_rsi(self, prices, period=14):
        """Calculate RSI."""
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
        """Calculate Bollinger Band position (-1 to +1)."""
        if len(prices) < period:
            return 0.0

        recent = prices[-period:]
        middle = np.mean(recent)
        std = np.std(recent)

        if std == 0:
            return 0.0

        current = prices[-1]
        bb_position = (current - middle) / (2 * std)
        return np.clip(bb_position, -1.5, 1.5)

    def calculate_momentum(self, prices, period=15):
        """Calculate momentum as percentage change."""
        if len(prices) < period + 1:
            return 0.0

        return (prices[-1] - prices[-period-1]) / prices[-period-1]

    def predict(self, data: dict) -> dict:
        """
        Generate mean-reversion prediction.
        FADE the current market condition.
        """
        price = data.get('price', data.get('close', 0))
        self.price_history.append(price)

        if len(self.price_history) < self.lookback:
            return {'action': 'HOLD', 'confidence': 0.0, 'reason': 'warming_up'}

        # Keep limited history
        self.price_history = self.price_history[-200:]

        prices = np.array(self.price_history)

        # Calculate signals
        rsi = self.calculate_rsi(prices, 14)
        bb_pos = self.calculate_bb_position(prices, 20)
        momentum = self.calculate_momentum(prices, 15)

        # Composite score: higher = more overbought = expect DOWN
        # Weight by correlation strength from analysis
        overbought_score = (
            0.40 * (bb_pos / 1.5) +           # BB position (normalized to -1/+1)
            0.35 * ((rsi - 50) / 50) +        # RSI (normalized to -1/+1)
            0.25 * np.clip(momentum * 20, -1, 1)  # Momentum (scaled)
        )

        # Time of day adjustment
        timestamp = data.get('timestamp')
        hour = 12
        if timestamp:
            if isinstance(timestamp, str):
                from datetime import datetime
                try:
                    timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                except:
                    pass
            if hasattr(timestamp, 'hour'):
                hour = timestamp.hour

        # Afternoon is bearish (from correlation: -0.041)
        if hour >= 14:
            overbought_score += 0.1

        # Decision thresholds
        THRESHOLD = 0.25  # Minimum signal strength to trade

        if overbought_score > THRESHOLD:
            # Market is overbought → FADE → BUY PUTS
            confidence = min(0.8, 0.5 + abs(overbought_score) * 0.3)
            return {
                'action': 'BUY_PUTS',
                'confidence': confidence,
                'reason': f'overbought (score={overbought_score:.2f}, RSI={rsi:.0f}, BB={bb_pos:.2f})'
            }
        elif overbought_score < -THRESHOLD:
            # Market is oversold → FADE → BUY CALLS
            confidence = min(0.8, 0.5 + abs(overbought_score) * 0.3)
            return {
                'action': 'BUY_CALLS',
                'confidence': confidence,
                'reason': f'oversold (score={overbought_score:.2f}, RSI={rsi:.0f}, BB={bb_pos:.2f})'
            }
        else:
            return {
                'action': 'HOLD',
                'confidence': 0.4,
                'reason': f'neutral (score={overbought_score:.2f})'
            }


def run_backtest(max_cycles=5000):
    """Run backtest of mean reversion strategy."""
    print("="*70)
    print("MEAN REVERSION BACKTEST")
    print("="*70)
    print("\nStrategy: FADE overbought/oversold conditions")
    print("Based on Simons' correlation analysis showing mean reversion\n")

    predictor = MeanReversionPredictor()

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

    # Simulation state
    balance = 5000.0
    initial_balance = balance
    position = None
    trades = []
    wins = 0
    losses = 0

    OPTION_COST = 100
    STOP_LOSS_PCT = -0.08
    TAKE_PROFIT_PCT = 0.12
    MAX_HOLD_BARS = 45

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

        # Check existing position
        if position:
            bars_held = i - position['entry_bar']
            price_change = (price - position['entry_price']) / position['entry_price']

            if position['type'] == 'CALL':
                option_pnl_pct = price_change * 5
            else:
                option_pnl_pct = -price_change * 5

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
                    'exit_reason': exit_reason,
                    'reason': position['reason']
                })
                position = None

        # Get prediction
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
                balance -= OPTION_COST

        if (i - 60) % 500 == 0:
            total_trades = wins + losses
            wr = 100 * wins / total_trades if total_trades > 0 else 0
            pnl_pct = 100 * (balance - initial_balance) / initial_balance
            print(f"  Cycle {i-60}: Balance ${balance:.2f} ({pnl_pct:+.2f}%), "
                  f"Trades: {total_trades}, WR: {wr:.1f}%")

    # Results
    total_trades = wins + losses
    win_rate = 100 * wins / total_trades if total_trades > 0 else 0
    pnl = balance - initial_balance
    pnl_pct = 100 * pnl / initial_balance

    print(f"\n{'='*50}")
    print(f"FINAL RESULTS: MEAN_REVERSION")
    print(f"{'='*50}")
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

        # Analyze by exit reason
        print(f"\nExit Reasons:")
        for reason in ['stop_loss', 'take_profit', 'max_hold']:
            count = sum(1 for t in trades if t['exit_reason'] == reason)
            reason_pnl = sum(t['pnl'] for t in trades if t['exit_reason'] == reason)
            if count > 0:
                print(f"  {reason}: {count} trades, ${reason_pnl:.2f}")

    return {
        'final_balance': balance,
        'pnl': pnl,
        'pnl_pct': pnl_pct,
        'trades': total_trades,
        'win_rate': win_rate
    }


if __name__ == "__main__":
    run_backtest(5000)
