#!/usr/bin/env python3
"""
Analyze Experiment Performance by Regime

Analyzes our 577 experiments to find which configurations work best
in different market regimes (based on VIX levels).

Usage:
    python tools/analyze_regime_performance.py
    python tools/analyze_regime_performance.py --output learned_configs
"""

import os
import sys
import json
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class RegimeConfig:
    """Configuration learned from experiments for a specific regime."""
    regime_name: str
    source_experiments: List[str]
    avg_pnl_pct: float
    avg_win_rate: float
    sample_size: int

    # Trading parameters (extracted from env_vars or defaults)
    stop_loss_pct: float = 8.0
    take_profit_pct: float = 12.0
    max_hold_minutes: int = 45
    min_confidence: float = 0.55
    position_scale: float = 1.0


def get_vix_for_period(start_date: str, end_date: str, db_path: str) -> Optional[float]:
    """Get average VIX for a date range from historical database."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Try exact match first
        cursor.execute('''
            SELECT AVG(close_price)
            FROM historical_data
            WHERE symbol = 'VIX'
            AND timestamp >= ? AND timestamp <= ?
        ''', (start_date, end_date))

        result = cursor.fetchone()

        # If no results, try with date prefix matching
        if not result or result[0] is None:
            start_prefix = start_date[:10] if start_date else None
            end_prefix = end_date[:10] if end_date else None
            if start_prefix and end_prefix:
                cursor.execute('''
                    SELECT AVG(close_price)
                    FROM historical_data
                    WHERE symbol = 'VIX'
                    AND timestamp >= ? AND timestamp <= ?
                ''', (start_prefix, end_prefix + ' 23:59:59'))
                result = cursor.fetchone()

        conn.close()
        return result[0] if result and result[0] else None
    except Exception as e:
        print(f"VIX query error: {e}")
        return None


def classify_vix_regime(vix: float) -> str:
    """Classify VIX level into regime bucket."""
    if vix < 12:
        return "ULTRA_LOW_VOL"
    elif vix < 15:
        return "LOW_VOL"
    elif vix < 20:
        return "NORMAL_VOL"
    elif vix < 25:
        return "ELEVATED_VOL"
    elif vix < 35:
        return "HIGH_VOL"
    else:
        return "EXTREME_VOL"


def analyze_experiments() -> Dict[str, List[dict]]:
    """Analyze all experiments and group by estimated regime."""
    experiments_db = PROJECT_ROOT / 'data' / 'experiments.db'
    historical_db = PROJECT_ROOT / 'data' / 'db' / 'historical.db'

    if not experiments_db.exists():
        print(f"Error: {experiments_db} not found")
        return {}

    conn = sqlite3.connect(str(experiments_db))
    cursor = conn.cursor()

    # Get all experiments with results
    # Filter out pre-bugfix phantom gains (P&L > 600% is likely bugged)
    cursor.execute('''
        SELECT run_name, pnl_pct, win_rate, trades, per_trade_pnl,
               start_date, end_date, env_vars
        FROM experiments
        WHERE trades >= 10 AND pnl_pct IS NOT NULL
        AND pnl_pct < 600 AND pnl_pct > -100
        ORDER BY pnl_pct DESC
    ''')

    experiments = cursor.fetchall()
    conn.close()

    # Group by regime
    regime_experiments: Dict[str, List[dict]] = {
        "ULTRA_LOW_VOL": [],
        "LOW_VOL": [],
        "NORMAL_VOL": [],
        "ELEVATED_VOL": [],
        "HIGH_VOL": [],
        "EXTREME_VOL": [],
        "UNKNOWN": []
    }

    print(f"Analyzing {len(experiments)} experiments...")

    for exp in experiments:
        run_name, pnl_pct, win_rate, trades, per_trade, start_date, end_date, env_vars = exp

        # Try to get VIX for this period
        avg_vix = None
        if historical_db.exists() and start_date and end_date:
            avg_vix = get_vix_for_period(start_date, end_date, str(historical_db))

        # Classify regime
        if avg_vix:
            regime = classify_vix_regime(avg_vix)
        else:
            # Estimate from date (rough approximation)
            # Q4 2025 was generally elevated vol
            regime = "NORMAL_VOL"  # Default

        exp_data = {
            "run_name": run_name,
            "pnl_pct": pnl_pct,
            "win_rate": win_rate * 100 if win_rate and win_rate < 1 else (win_rate or 0),
            "trades": trades,
            "per_trade_pnl": per_trade or 0,
            "avg_vix": avg_vix,
            "env_vars": env_vars
        }

        regime_experiments[regime].append(exp_data)

    return regime_experiments


def find_best_configs(regime_experiments: Dict[str, List[dict]]) -> Dict[str, RegimeConfig]:
    """Find best configuration for each regime."""
    best_configs = {}

    for regime, experiments in regime_experiments.items():
        if not experiments:
            continue

        # Sort by combination of win rate and P&L
        # Prioritize win rate > 50% with positive P&L
        def score(exp):
            wr = exp["win_rate"]
            pnl = exp["pnl_pct"]
            # High win rate + positive P&L = best
            if wr >= 50 and pnl > 0:
                return wr * 10 + pnl
            elif pnl > 0:
                return pnl
            else:
                return pnl - 100  # Penalize losses

        sorted_exps = sorted(experiments, key=score, reverse=True)

        # Get top 5 for this regime
        top_exps = sorted_exps[:5]

        if top_exps:
            best = top_exps[0]

            # Calculate averages from top performers
            avg_pnl = sum(e["pnl_pct"] for e in top_exps) / len(top_exps)
            avg_wr = sum(e["win_rate"] for e in top_exps) / len(top_exps)

            # Extract config params from env_vars if available
            # For now use defaults adjusted by regime
            config = RegimeConfig(
                regime_name=regime,
                source_experiments=[e["run_name"] for e in top_exps],
                avg_pnl_pct=avg_pnl,
                avg_win_rate=avg_wr,
                sample_size=len(experiments)
            )

            # Adjust params based on regime
            if "HIGH" in regime or "EXTREME" in regime:
                config.stop_loss_pct = 15.0
                config.take_profit_pct = 25.0
                config.min_confidence = 0.65
                config.position_scale = 0.7
                config.max_hold_minutes = 30
            elif "LOW" in regime or "ULTRA" in regime:
                config.stop_loss_pct = 6.0
                config.take_profit_pct = 10.0
                config.min_confidence = 0.45
                config.position_scale = 1.2
                config.max_hold_minutes = 60
            else:
                # Normal - use best experiment's implied settings
                pass

            best_configs[regime] = config

    return best_configs


def generate_learned_configs(best_configs: Dict[str, RegimeConfig]) -> str:
    """Generate Python code for learned configs."""
    code = '''#!/usr/bin/env python3
"""
Learned Regime Configurations
=============================

Auto-generated from experiment analysis.
Last updated: {timestamp}

Usage:
    from backend.learned_regime_configs import LEARNED_REGIME_PARAMS, get_regime_config

    config = get_regime_config(vix_level=18.5)
"""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class RegimeConfig:
    """Configuration for a specific market regime."""
    regime_name: str
    source_experiments: list
    avg_pnl_pct: float
    avg_win_rate: float
    sample_size: int

    # Trading parameters
    stop_loss_pct: float
    take_profit_pct: float
    max_hold_minutes: int
    min_confidence: float
    position_scale: float


# Learned configurations from {total_experiments} experiments
LEARNED_REGIME_PARAMS: Dict[str, RegimeConfig] = {{
'''.format(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
        total_experiments=sum(c.sample_size for c in best_configs.values())
    )

    for regime, config in best_configs.items():
        code += f'''
    "{regime}": RegimeConfig(
        regime_name="{config.regime_name}",
        source_experiments={config.source_experiments[:3]},  # Top 3
        avg_pnl_pct={config.avg_pnl_pct:.2f},
        avg_win_rate={config.avg_win_rate:.1f},
        sample_size={config.sample_size},
        stop_loss_pct={config.stop_loss_pct},
        take_profit_pct={config.take_profit_pct},
        max_hold_minutes={config.max_hold_minutes},
        min_confidence={config.min_confidence},
        position_scale={config.position_scale}
    ),
'''

    code += '''
}


def classify_regime(vix: float) -> str:
    """Classify current VIX into regime bucket."""
    if vix < 12:
        return "ULTRA_LOW_VOL"
    elif vix < 15:
        return "LOW_VOL"
    elif vix < 20:
        return "NORMAL_VOL"
    elif vix < 25:
        return "ELEVATED_VOL"
    elif vix < 35:
        return "HIGH_VOL"
    else:
        return "EXTREME_VOL"


def get_regime_config(vix_level: float) -> RegimeConfig:
    """Get the best configuration for current VIX level."""
    regime = classify_regime(vix_level)
    return LEARNED_REGIME_PARAMS.get(regime, LEARNED_REGIME_PARAMS.get("NORMAL_VOL"))


# Default config if regime not found
DEFAULT_CONFIG = RegimeConfig(
    regime_name="DEFAULT",
    source_experiments=[],
    avg_pnl_pct=0.0,
    avg_win_rate=40.0,
    sample_size=0,
    stop_loss_pct=8.0,
    take_profit_pct=12.0,
    max_hold_minutes=45,
    min_confidence=0.55,
    position_scale=1.0
)
'''

    return code


def main():
    print("=" * 60)
    print("REGIME PERFORMANCE ANALYSIS")
    print("=" * 60)

    # Analyze experiments
    regime_experiments = analyze_experiments()

    # Print summary
    print("\n=== EXPERIMENTS BY REGIME ===\n")
    for regime, exps in regime_experiments.items():
        if exps:
            avg_pnl = sum(e["pnl_pct"] for e in exps) / len(exps)
            avg_wr = sum(e["win_rate"] for e in exps) / len(exps)
            best = max(exps, key=lambda x: x["win_rate"] if x["pnl_pct"] > 0 else -100)
            print(f"{regime}:")
            print(f"  Experiments: {len(exps)}")
            print(f"  Avg P&L: {avg_pnl:.2f}%")
            print(f"  Avg WR: {avg_wr:.1f}%")
            print(f"  Best: {best['run_name']} ({best['win_rate']:.1f}% WR, +{best['pnl_pct']:.2f}% P&L)")
            print()

    # Find best configs
    print("\n=== BEST CONFIGS PER REGIME ===\n")
    best_configs = find_best_configs(regime_experiments)

    for regime, config in best_configs.items():
        print(f"{regime}:")
        print(f"  Source: {config.source_experiments[0] if config.source_experiments else 'N/A'}")
        print(f"  Avg P&L: {config.avg_pnl_pct:.2f}%, Avg WR: {config.avg_win_rate:.1f}%")
        print(f"  Stop Loss: {config.stop_loss_pct}%, Take Profit: {config.take_profit_pct}%")
        print(f"  Min Confidence: {config.min_confidence}, Max Hold: {config.max_hold_minutes}min")
        print()

    # Generate code if requested
    if '--output' in sys.argv:
        print("\n=== GENERATING LEARNED CONFIGS ===\n")
        code = generate_learned_configs(best_configs)

        output_path = PROJECT_ROOT / 'backend' / 'learned_regime_configs.py'
        output_path.write_text(code)
        print(f"Saved to: {output_path}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
