# core/main.py - Integrated CLI with Full Parameter Control

import argparse
import datetime as dt
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import StrategyConfig, RunConfig
from core.backtest_engine import run_backtest_and_report
from data_factory.polygon_client import PolygonClient
from core.broker import PaperBroker
from strategies.options_strategy import run_strategy
from analytics.metrics import plot_metrics


# ==========================================================
# ARGUMENT PARSER
# ==========================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Quantor-MTFuzz: Iron Condor Trading with Multi-Timeframe Intelligence",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python core/main.py --mode backtest
  python core/main.py --mode backtest --use-mtf
  python core/main.py --mode backtest --no-mtf-filter
        """
    )

    # --- Mode ---
    parser.add_argument("--mode", choices=["live", "backtest"], default="backtest")

    # --- API Keys ---
    parser.add_argument("--polygon-key", type=str, default=None)
    parser.add_argument("--alpaca-key", type=str, default=None)
    parser.add_argument("--alpaca-secret", type=str, default=None)

    # --- Strategy ---
    parser.add_argument("--underlying", type=str, default="SPY")
    parser.add_argument("--quantity", type=int, default=1)

    # --- DTE ---
    parser.add_argument("--dte-min", type=int, default=30)
    parser.add_argument("--dte-max", type=int, default=45)

    # --- Delta ---
    parser.add_argument("--delta-low", type=float, default=0.15)
    parser.add_argument("--delta-high", type=float, default=0.20)

    # --- Wings ---
    parser.add_argument("--wing-min", type=float, default=5.0)
    parser.add_argument("--wing-max", type=float, default=10.0)
    parser.add_argument("--min-credit-ratio", type=float, default=0.25)

    # --- Filters ---
    parser.add_argument("--ivr-min", type=float, default=30.0)
    parser.add_argument("--vix-max", type=float, default=25.0)

    # --- Exits ---
    parser.add_argument("--profit-pct", type=float, default=0.50)
    parser.add_argument("--loss-multiple", type=float, default=1.5)
    parser.add_argument("--max-hold-days", type=int, default=14)

    # --- Position limits ---
    parser.add_argument("--max-positions", type=int, default=3)
    parser.add_argument("--max-alloc", type=float, default=0.15)

    # --- Regime ---
    parser.add_argument("--regime-ivr-widen", type=float, default=40.0)
    parser.add_argument("--regime-vix-widen", type=float, default=22.0)
    parser.add_argument("--width-increment", type=float, default=5.0)

    # --- Hedging ---
    parser.add_argument("--enable-hedge", action="store_true")
    parser.add_argument("--hedge-threshold", type=float, default=0.10)
    parser.add_argument("--hedge-unit", type=int, default=10)

    # --- MTF ---
    parser.add_argument("--use-mtf", action="store_true")
    parser.add_argument("--no-mtf-filter", action="store_true")
    parser.add_argument("--mtf-consensus-min", type=float, default=0.40)
    parser.add_argument("--mtf-consensus-max", type=float, default=0.60)
    parser.add_argument("--mtf-timeframes", type=str, default="5,15,60")

    # --- Liquidity ---
    parser.add_argument("--no-liquidity-gate", action="store_true")

    # --- Backtest ---
    parser.add_argument("--bt-cash", type=float, default=25000.0)
    parser.add_argument("--bt-start", type=str, default="2024-01-01")
    parser.add_argument("--bt-end", type=str, default="2024-12-31")
    parser.add_argument("--bt-samples", type=int, default=500)
    parser.add_argument("--no-plot", action="store_true")

    # --- Sizing ---
    parser.add_argument("--dynamic-sizing", action="store_true")
    parser.add_argument("--position-size-pct", type=float, default=0.05)
    parser.add_argument("--alpaca", action="store_true", help="Use Alpaca Broker instead of PaperBroker")

    # --- Optimizer ---
    parser.add_argument("--use-optimizer", action="store_true")

    return parser.parse_args()


# ==========================================================
# MAIN
# ==========================================================
def main():
    args = parse_args()

    # -------------------------
    # CONFIG
    # -------------------------
    strategy_config = StrategyConfig()
    run_config = RunConfig()

    # Apply CLI → StrategyConfig
    strategy_config.underlying = args.underlying
    # quantity belongs to RunConfig; keep in strategy_config for compatibility if present
    try:
        strategy_config.quantity = args.quantity
    except Exception:
        pass

    strategy_config.dte_min = args.dte_min
    strategy_config.dte_max = args.dte_max

    strategy_config.target_short_delta_low = args.delta_low
    strategy_config.target_short_delta_high = args.delta_high

    strategy_config.wing_width_min = args.wing_min
    strategy_config.wing_width_max = args.wing_max
    strategy_config.min_credit_to_width = args.min_credit_ratio

    strategy_config.iv_rank_min = args.ivr_min
    strategy_config.vix_threshold = args.vix_max

    strategy_config.profit_take_pct = args.profit_pct
    strategy_config.loss_close_multiple = args.loss_multiple
    strategy_config.max_hold_days = args.max_hold_days

    strategy_config.max_positions = args.max_positions
    strategy_config.max_portfolio_alloc = args.max_alloc

    strategy_config.regime_iv_rank_widen = args.regime_ivr_widen
    strategy_config.regime_vix_widen = args.regime_vix_widen
    strategy_config.width_widen_increment = args.width_increment

    # Dynamic delta hedge flags
    strategy_config.dynamic_delta_hedge_enabled = args.enable_hedge
    strategy_config.dynamic_delta_hedge_threshold = args.hedge_threshold
    strategy_config.dynamic_delta_hedge_unit = args.hedge_unit

    # -------------------------
    # MTF
    # -------------------------
    strategy_config.use_mtf_filter = args.use_mtf
    if args.no_mtf_filter:
        strategy_config.use_mtf_filter = False

    strategy_config.mtf_consensus_min = args.mtf_consensus_min
    strategy_config.mtf_consensus_max = args.mtf_consensus_max
    strategy_config.mtf_timeframes = args.mtf_timeframes.split(",")

    # -------------------------
    # LIQUIDITY GATE (HARD SAFETY)
    # -------------------------
    strategy_config.use_liquidity_gate = True

    if args.mode == "backtest":
        strategy_config.use_liquidity_gate = False

    if args.no_liquidity_gate:
        strategy_config.use_liquidity_gate = False

    # -------------------------
    # RUN
    # -------------------------
    if args.mode == "backtest":
        # Populate RunConfig with CLI overrides
        try:
            run_config.backtest_cash = float(args.bt_cash)
        except Exception:
            pass

        # Parse dates (keep RunConfig fields as date objects)
        try:
            run_config.backtest_start = dt.datetime.fromisoformat(args.bt_start).date()
        except Exception:
            pass

        try:
            run_config.backtest_end = dt.datetime.fromisoformat(args.bt_end).date()
        except Exception:
            pass

        try:
            run_config.backtest_samples = int(args.bt_samples)
        except Exception:
            pass

        # Plotting: CLI uses --no-plot to disable plotting; default is True
        run_config.backtest_plot = not args.no_plot

        # Dynamic sizing flag
        run_config.dynamic_sizing = args.dynamic_sizing
        run_config.position_size_pct = args.position_size_pct

        # MTF flag mapping
        run_config.use_mtf = args.use_mtf
        if args.mtf_timeframes:
            run_config.mtf_timeframes = [t.strip().upper() for t in args.mtf_timeframes.split(",")]

        # API keys if provided
        if args.polygon_key:
            run_config.polygon_key = args.polygon_key
        if args.alpaca_key:
            run_config.alpaca_key = args.alpaca_key
        if args.alpaca_secret:
            run_config.alpaca_secret = args.alpaca_secret

        # Check for Optimizer Mode
        if args.use_optimizer:
            from core.optimizer import run_optimization
            run_optimization(strategy_config, run_config)
            return

        # Now call the backtest engine with the two config objects it expects
        run_backtest_and_report(strategy_config, run_config)
        return

    # -------------------------
    # LIVE / PAPER
    # -------------------------
    # For live/paper mode, ensure polygon key is provided either via CLI or RunConfig defaults
    polygon_key = args.polygon_key or run_config.polygon_key
    if not polygon_key:
        raise ValueError("Polygon API key required for live mode. Pass --polygon-key YOUR_KEY")

    poly_client = PolygonClient(api_key=polygon_key)
    
    if args.alpaca:
        from core.broker import AlpacaBroker
        broker = AlpacaBroker(run_config, poly_client)
        print("[Mode] ALPACA LIVE/PAPER TRADING ENABLED")
    else:
        from core.broker import PaperBroker
        broker = PaperBroker(polygon_client=poly_client, starting_equity=args.bt_cash)
        print("[Mode] LOCAL PAPER TRADING ENABLED")

    # Apply run config quantity if present
    quantity = args.quantity if args.quantity else run_config.quantity

    # Run live strategy
    run_strategy(broker, strategy_config)

    if not args.no_plot:
        # plot_metrics expects a list of trades; PaperBroker exposes collect_trade_log or trade_log
        try:
            trades = broker.collect_trade_log()
        except Exception:
            trades = getattr(broker, "trade_log", [])
        plot_metrics(trades)


# ==========================================================
if __name__ == "__main__":
    main()
