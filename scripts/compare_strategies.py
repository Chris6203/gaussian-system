#!/usr/bin/env python3
"""
Strategy Comparison: Gaussian Directional vs Jerry's Iron Condor

Runs both strategies through the same market data to compare:
1. Win rate
2. P&L
3. Risk-adjusted returns (Sharpe)
4. Drawdown
5. Trade frequency

Usage:
    python scripts/compare_strategies.py
    python scripts/compare_strategies.py --cycles 5000
"""

import os
import sys
import argparse
import logging
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

# Setup path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.iron_condor_controller import IronCondorEntryController

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


class StrategySimulator:
    """Simulate a trading strategy through historical data"""

    def __init__(self, name: str, starting_balance: float = 5000.0):
        self.name = name
        self.starting_balance = starting_balance
        self.balance = starting_balance
        self.peak_balance = starting_balance

        self.trades: List[Dict] = []
        self.equity_curve: List[float] = [starting_balance]
        self.drawdowns: List[float] = [0.0]

        self.wins = 0
        self.losses = 0
        self.total_pnl = 0.0

    def record_trade(self, pnl: float, trade_type: str, reason: str):
        """Record a completed trade"""
        self.balance += pnl
        self.total_pnl += pnl

        if pnl > 0:
            self.wins += 1
        else:
            self.losses += 1

        self.trades.append({
            'pnl': pnl,
            'type': trade_type,
            'reason': reason,
            'balance': self.balance
        })

        # Update equity curve and drawdown
        self.equity_curve.append(self.balance)
        self.peak_balance = max(self.peak_balance, self.balance)
        self.drawdowns.append(self.peak_balance - self.balance)

    def get_stats(self) -> Dict:
        """Calculate performance statistics"""
        total_trades = len(self.trades)
        win_rate = self.wins / max(1, total_trades)

        # Calculate Sharpe ratio from returns
        if len(self.equity_curve) > 1:
            returns = np.diff(self.equity_curve) / np.array(self.equity_curve[:-1])
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252 * 78) if np.std(returns) > 0 else 0
        else:
            sharpe = 0

        max_dd = max(self.drawdowns) if self.drawdowns else 0
        max_dd_pct = max_dd / self.peak_balance * 100 if self.peak_balance > 0 else 0

        return {
            'strategy': self.name,
            'starting_balance': self.starting_balance,
            'final_balance': self.balance,
            'total_pnl': self.total_pnl,
            'pnl_pct': (self.balance - self.starting_balance) / self.starting_balance * 100,
            'total_trades': total_trades,
            'wins': self.wins,
            'losses': self.losses,
            'win_rate': win_rate * 100,
            'avg_win': np.mean([t['pnl'] for t in self.trades if t['pnl'] > 0]) if self.wins > 0 else 0,
            'avg_loss': np.mean([t['pnl'] for t in self.trades if t['pnl'] < 0]) if self.losses > 0 else 0,
            'sharpe': sharpe,
            'max_drawdown': max_dd,
            'max_drawdown_pct': max_dd_pct,
            'profit_factor': abs(sum(t['pnl'] for t in self.trades if t['pnl'] > 0)) /
                           max(1, abs(sum(t['pnl'] for t in self.trades if t['pnl'] < 0)))
        }


def generate_market_scenarios(n_cycles: int, seed: int = 42) -> List[Dict]:
    """
    Generate synthetic market scenarios for comparison.
    Each scenario represents a 5-minute bar with features.
    """
    np.random.seed(seed)

    scenarios = []
    spot_price = 580.0  # Starting SPY price
    vix = 18.0

    for i in range(n_cycles):
        # Random walk for price
        spot_return = np.random.normal(0, 0.001)  # ~0.1% per bar
        spot_price *= (1 + spot_return)

        # Mean-reverting VIX
        vix += np.random.normal(0, 0.5) + (18 - vix) * 0.05
        vix = np.clip(vix, 10, 40)

        # Calculate features
        momentum_5m = spot_return
        momentum_15m = np.random.normal(0, 0.002)

        # HMM trend (somewhat persistent)
        if i == 0:
            hmm_trend = 0.5
        else:
            hmm_trend = scenarios[-1]['hmm_trend'] + np.random.normal(0, 0.05)
            hmm_trend = np.clip(hmm_trend, 0.1, 0.9)

        # Model confidence (noisy)
        confidence = np.clip(np.random.beta(2, 3), 0.2, 0.9)

        # Predicted direction based on recent momentum
        predicted_direction = np.clip(momentum_15m * 50, -0.5, 0.5)

        scenarios.append({
            'cycle': i,
            'spot_price': spot_price,
            'spot_return': spot_return,
            'vix_level': vix,
            'hmm_trend': hmm_trend,
            'hmm_volatility': np.clip(vix / 30, 0.2, 0.8),
            'hmm_confidence': np.random.uniform(0.5, 0.9),
            'momentum_5m': momentum_5m,
            'momentum_15m': momentum_15m,
            'confidence': confidence,
            'predicted_direction': predicted_direction,
            'volume_spike': np.random.uniform(0.8, 1.5)
        })

    return scenarios


def simulate_directional_trade(entry_features: Dict, exit_features: Dict,
                               direction: str, hold_bars: int) -> Tuple[float, str]:
    """
    Simulate a directional (single option) trade.

    Returns (pnl_dollars, exit_reason)
    """
    # Price movement during hold
    entry_spot = entry_features['spot_price']
    exit_spot = exit_features['spot_price']
    spot_move_pct = (exit_spot - entry_spot) / entry_spot

    # Option premium (simplified)
    entry_vix = entry_features['vix_level']
    premium = entry_spot * 0.005 * (entry_vix / 18)  # ~$3 for ATM option

    # Theta decay (~5% per day for short-dated options)
    theta_decay = premium * 0.05 * (hold_bars / 78)  # 78 bars per day

    # Delta P&L
    delta = 0.5  # ATM option
    if direction == "CALL":
        delta_pnl = spot_move_pct * entry_spot * delta
    else:  # PUT
        delta_pnl = -spot_move_pct * entry_spot * delta

    # Total P&L
    pnl = delta_pnl - theta_decay

    # Determine exit reason
    pnl_pct = pnl / (premium * 100)
    if pnl_pct >= 0.12:  # 12% take profit
        return pnl, "TAKE_PROFIT"
    elif pnl_pct <= -0.08:  # 8% stop loss
        return max(pnl, -premium * 100 * 0.08), "STOP_LOSS"
    elif hold_bars >= 45:  # Max hold
        return pnl, "MAX_HOLD"
    else:
        return pnl, "TIME_EXIT"


def simulate_condor_trade(entry_features: Dict, exit_features: Dict,
                          signal, hold_bars: int) -> Tuple[float, str]:
    """
    Simulate an Iron Condor trade.

    Returns (pnl_dollars, exit_reason)
    """
    entry_spot = entry_features['spot_price']
    exit_spot = exit_features['spot_price']
    spot_move_pct = (exit_spot - entry_spot) / entry_spot

    entry_vix = entry_features['vix_level']
    exit_vix = exit_features['vix_level']

    # Use the controller's P&L simulation
    controller = IronCondorEntryController()
    pnl, reason = controller.simulate_condor_pnl(
        entry_credit=signal.expected_credit,
        wing_width=signal.wing_width,
        spot_move_pct=spot_move_pct,
        days_held=hold_bars / 78,  # Convert bars to days
        entry_vix=entry_vix,
        exit_vix=exit_vix
    )

    return pnl, reason


def run_comparison(n_cycles: int = 5000, verbose: bool = True) -> pd.DataFrame:
    """
    Run both strategies through the same market data and compare.
    """
    logger.info(f"Generating {n_cycles} market scenarios...")
    scenarios = generate_market_scenarios(n_cycles)

    # Initialize simulators
    directional_sim = StrategySimulator("Gaussian Directional")
    condor_sim = StrategySimulator("Iron Condor")

    # Initialize controllers
    condor_controller = IronCondorEntryController()

    # Simulation state
    directional_position = None
    condor_position = None
    hold_period = 15  # Bars between checks

    logger.info("Running strategy comparison...")

    for i, scenario in enumerate(scenarios):
        # === DIRECTIONAL STRATEGY ===
        if directional_position is None:
            # Entry logic: Use HMM + confidence
            hmm_trend = scenario['hmm_trend']
            confidence = scenario['confidence']
            pred_dir = scenario['predicted_direction']

            should_enter = False
            direction = "HOLD"

            # Simplified directional entry: Strong HMM signal + adequate confidence
            if hmm_trend > 0.65 and confidence > 0.3 and pred_dir > 0.1:
                should_enter = True
                direction = "CALL"
            elif hmm_trend < 0.35 and confidence > 0.3 and pred_dir < -0.1:
                should_enter = True
                direction = "PUT"

            if should_enter:
                directional_position = {
                    'entry_idx': i,
                    'entry_features': scenario.copy(),
                    'direction': direction
                }

        elif i - directional_position['entry_idx'] >= hold_period:
            # Exit directional trade
            pnl, reason = simulate_directional_trade(
                directional_position['entry_features'],
                scenario,
                directional_position['direction'],
                i - directional_position['entry_idx']
            )
            directional_sim.record_trade(pnl, directional_position['direction'], reason)
            directional_position = None

        # === IRON CONDOR STRATEGY ===
        if condor_position is None:
            signal = condor_controller.should_enter_condor(scenario)

            if signal.should_enter:
                condor_position = {
                    'entry_idx': i,
                    'entry_features': scenario.copy(),
                    'signal': signal
                }

        elif i - condor_position['entry_idx'] >= hold_period * 2:  # Condors held longer
            # Exit condor trade
            pnl, reason = simulate_condor_trade(
                condor_position['entry_features'],
                scenario,
                condor_position['signal'],
                i - condor_position['entry_idx']
            )
            condor_sim.record_trade(pnl, "CONDOR", reason)
            condor_position = None

        # Progress
        if verbose and i % 1000 == 0:
            logger.info(f"  Cycle {i}/{n_cycles}")

    # Close any remaining positions at end
    if directional_position is not None:
        pnl, reason = simulate_directional_trade(
            directional_position['entry_features'],
            scenarios[-1],
            directional_position['direction'],
            len(scenarios) - directional_position['entry_idx']
        )
        directional_sim.record_trade(pnl, directional_position['direction'], reason)

    if condor_position is not None:
        pnl, reason = simulate_condor_trade(
            condor_position['entry_features'],
            scenarios[-1],
            condor_position['signal'],
            len(scenarios) - condor_position['entry_idx']
        )
        condor_sim.record_trade(pnl, "CONDOR", reason)

    # Collect results
    directional_stats = directional_sim.get_stats()
    condor_stats = condor_sim.get_stats()

    # Create comparison dataframe
    results = pd.DataFrame([directional_stats, condor_stats])

    return results, condor_controller.get_stats()


def print_comparison(results: pd.DataFrame, condor_stats: Dict):
    """Pretty print the comparison results"""
    print("\n" + "=" * 80)
    print("STRATEGY COMPARISON: Gaussian Directional vs Iron Condor")
    print("=" * 80)

    print("\n{:<25} {:>20} {:>20}".format("Metric", "Directional", "Iron Condor"))
    print("-" * 65)

    for _, row in results.iterrows():
        pass  # We'll print manually for better control

    d = results.iloc[0]  # Directional
    c = results.iloc[1]  # Condor

    metrics = [
        ("Starting Balance", f"${d['starting_balance']:,.2f}", f"${c['starting_balance']:,.2f}"),
        ("Final Balance", f"${d['final_balance']:,.2f}", f"${c['final_balance']:,.2f}"),
        ("Total P&L", f"${d['total_pnl']:,.2f}", f"${c['total_pnl']:,.2f}"),
        ("P&L %", f"{d['pnl_pct']:.2f}%", f"{c['pnl_pct']:.2f}%"),
        ("Total Trades", f"{int(d['total_trades'])}", f"{int(c['total_trades'])}"),
        ("Win Rate", f"{d['win_rate']:.1f}%", f"{c['win_rate']:.1f}%"),
        ("Avg Win", f"${d['avg_win']:.2f}", f"${c['avg_win']:.2f}"),
        ("Avg Loss", f"${d['avg_loss']:.2f}", f"${c['avg_loss']:.2f}"),
        ("Sharpe Ratio", f"{d['sharpe']:.2f}", f"{c['sharpe']:.2f}"),
        ("Max Drawdown", f"${d['max_drawdown']:.2f}", f"${c['max_drawdown']:.2f}"),
        ("Max DD %", f"{d['max_drawdown_pct']:.2f}%", f"{c['max_drawdown_pct']:.2f}%"),
        ("Profit Factor", f"{d['profit_factor']:.2f}", f"{c['profit_factor']:.2f}"),
    ]

    for metric, d_val, c_val in metrics:
        print(f"{metric:<25} {d_val:>20} {c_val:>20}")

    print("\n" + "-" * 65)
    print("\nIRON CONDOR CONTROLLER STATS:")
    print(f"  Decisions: {condor_stats['decisions']}")
    print(f"  Entries: {condor_stats['entries']}")
    print(f"  Entry Rate: {condor_stats['entry_rate']:.2%}")
    print("\n  Rejection Reasons:")
    for reason, count in sorted(condor_stats['rejections'].items(), key=lambda x: -x[1])[:5]:
        print(f"    {reason}: {count}")

    print("\n" + "=" * 80)

    # Determine winner
    if c['pnl_pct'] > d['pnl_pct'] and c['max_drawdown_pct'] <= d['max_drawdown_pct']:
        winner = "IRON CONDOR"
        reason = "Higher returns with equal or lower drawdown"
    elif d['pnl_pct'] > c['pnl_pct'] and d['sharpe'] > c['sharpe']:
        winner = "DIRECTIONAL"
        reason = "Higher returns with better risk-adjusted performance"
    elif c['sharpe'] > d['sharpe']:
        winner = "IRON CONDOR"
        reason = "Better risk-adjusted returns (Sharpe)"
    else:
        winner = "TIE / DEPENDS ON GOALS"
        reason = "Each strategy has different strengths"

    print(f"\nWINNER: {winner}")
    print(f"Reason: {reason}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Compare Gaussian vs Iron Condor strategies")
    parser.add_argument('--cycles', type=int, default=5000, help='Number of market cycles to simulate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()

    results, condor_stats = run_comparison(n_cycles=args.cycles)
    print_comparison(results, condor_stats)

    # Save results
    results.to_csv('reports/strategy_comparison.csv', index=False)
    logger.info("Results saved to reports/strategy_comparison.csv")


if __name__ == "__main__":
    main()
