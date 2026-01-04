# core/backtest_engine.py - Iron Condor with MTF + Professional Reporting
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import StrategyConfig, RunConfig
import backtrader as bt
import matplotlib.pyplot as plt
import math
import pandas as pd
import numpy as np
from tabulate import tabulate
import mplfinance as mpf
import os
import csv
from matplotlib.backends.backend_pdf import PdfPages
from data_factory.sync_engine import MTFSyncEngine
from strategies.options_strategy import (
    OptionQuote, build_condor, calc_condor_credit, 
    PositionState, regime_wing_width
)

def run_backtest_headless(s_cfg: StrategyConfig, r_cfg: RunConfig, preloaded_df=None, preloaded_options=None, preloaded_sync=None, verbose=True):
    """
    Run backtest without generating reports, returning raw strategy object
    Used for optimization loops.
    """
    class PandasData(bt.feeds.PandasData):
        params = (
            ('datetime', None),
            ('open', 'open'),
            ('high', 'high'),
            ('low', 'low'),
            ('close', 'close'),
            ('volume', 'volume'),
            ('openinterest', None),
        )

    # === Load Data ===
    if preloaded_df is not None:
        df = preloaded_df
    else:
        # Fallback to loading from disk (standard backtest behavior)
        csv_path = os.path.join("reports", s_cfg.underlying, f"{s_cfg.underlying}_5.csv")
        if not os.path.exists(csv_path):
            print(f"[ERROR] Data not found: {csv_path}")
            return None
        df = pd.read_csv(csv_path, parse_dates=["timestamp"])
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
        df.set_index("timestamp", inplace=True)
        df.sort_index(inplace=True)
        
        # Optimization: Slice data if backtest_samples is set
        if r_cfg.backtest_samples and r_cfg.backtest_samples > 0:
            if len(df) > r_cfg.backtest_samples:
                df = df.iloc[-r_cfg.backtest_samples:]
    
    # === Load Synthetic Options Data ===
    if preloaded_options is not None:
        options_by_date = preloaded_options
    else:
        options_path = os.path.join("data", "synthetic_options", f"{s_cfg.underlying}_5min.csv")
        if not os.path.exists(options_path):
            print(f"[ERROR] Synthetic options data not found: {options_path}")
            return None
        
        # Load and optimize for fast lookup
        if verbose: print(f"[Data] Loading synthetic options: {options_path}...")
        options_df = pd.read_csv(options_path, parse_dates=["date", "expiration"])
        options_by_date = {}
        for date, group in options_df.groupby('date'):
            options_by_date[date.date()] = group.to_dict('records')
    
    # === Initialize MTF Sync Engine ===
    if preloaded_sync is not None:
        sync_engine = preloaded_sync
    else:
        sync_engine = MTFSyncEngine(s_cfg.underlying, r_cfg.mtf_timeframes) if r_cfg.use_mtf else None

    # === Strategy Definition ===
    class IronCondorStrategy(bt.Strategy):
        params = dict(s_cfg=None, r_cfg=None, sync_engine=None, options_data=None, verbose=True)

        def __init__(self):
            self.s_cfg = self.params.s_cfg
            self.r_cfg = self.params.r_cfg
            self.sync = self.params.sync_engine
            self.options_data = self.params.options_data
            self.verbose = self.params.verbose
            
            # Performance Tracking
            self.pnl = 0.0
            self.wins = 0
            self.losses = 0
            self.trades = 0
            self.peak = r_cfg.backtest_cash
            self.drawdowns = []
            self.equity_series = []
            self.trade_log = []
            
            # High-Fidelity Tracking
            self.active_position = None # Handle 1 position at a time for simplicity
            self.bars_since_trade = 100 
            self.price_cache = {} # Persistent cache for leg prices: {symbol: last_price}

        def next(self):
            dt_now = self.datas[0].datetime.datetime(0)
            date_now = dt_now.date()
            
            # 1. Update Equity tracking for every bar
            current_equity = self.r_cfg.backtest_cash + self.pnl
            
            # If we provide an option chain for this date, calculate floating P&L
            unrealized_pnl = 0.0
            exit_triggered = False
            exit_reason = ""
            
            if self.active_position:
                chain_records = self.options_data.get(date_now, [])
                if chain_records:
                    leg_symbols = [
                        self.active_position.legs.short_call.symbol,
                        self.active_position.legs.long_call.symbol,
                        self.active_position.legs.short_put.symbol,
                        self.active_position.legs.long_put.symbol
                    ]
                    
                    # Update cache with current bar's prices
                    for r in chain_records:
                        if r['option_symbol'] in leg_symbols:
                            self.price_cache[r['option_symbol']] = r['last_price']
                    
                    # Try to get all prices (current or cached)
                    current_prices = {s: self.price_cache.get(s) for s in leg_symbols if s in self.price_cache}
                    
                    if len(current_prices) == 4:
                        # Current replacement cost (to close)
                        current_cost = (
                            current_prices[self.active_position.legs.short_call.symbol] - 
                            current_prices[self.active_position.legs.long_call.symbol] +
                            current_prices[self.active_position.legs.short_put.symbol] -
                            current_prices[self.active_position.legs.long_put.symbol]
                        )
                        
                        unrealized_pnl = (self.active_position.credit_received - current_cost) * self.active_position.quantity * 100
                        
                        # Exit Rule: Profit Take / Stop Loss
                        target_profit = self.active_position.credit_received * self.s_cfg.profit_take_pct
                        stop_loss = self.active_position.credit_received * (1 + self.s_cfg.loss_close_multiple)
                        
                        if current_cost <= target_profit:
                            exit_triggered = True
                            exit_reason = "Profit Take"
                        elif current_cost >= stop_loss:
                            exit_triggered = True
                            exit_reason = "Stop Loss"
                    
                    # Exit Rule: Expiration (Date based, always checked)
                    dte = (self.active_position.legs.short_call.expiration - date_now).days
                    if dte <= 0:
                        exit_triggered = True
                        exit_reason = "Expiration"
                    
                    # Periodic Status
                    if self.verbose and len(self.equity_series) % 100 == 0:
                        cost_str = f"${current_cost:.2f}" if 'current_cost' in locals() else "N/A"
                        print(f"  [Position] PnL: ${unrealized_pnl:.2f} | Cost: {cost_str} | DTE: {dte}")
                            
                        if exit_triggered:
                            realized_pnl = unrealized_pnl
                            self.pnl += realized_pnl
                            self.trades += 1
                            if realized_pnl > 0: self.wins += 1
                            else: self.losses += 1
                            
                            print(f"[Trade] Closed: {exit_reason} | PnL: ${realized_pnl:.2f}")

                            # Market context at exit
                            exit_ohlcv = {
                                "open": self.data.open[0],
                                "high": self.data.high[0],
                                "low": self.data.low[0],
                                "close": self.data.close[0],
                                "volume": self.data.volume[0]
                            }
                            
                            self.trade_log.append({
                                "start": self.active_position.open_time,
                                "end": dt_now,
                                "result": "win" if realized_pnl > 0 else "loss",
                                "amount": realized_pnl,
                                "contracts": self.active_position.quantity,
                                "exit_reason": exit_reason,
                                "credit": self.active_position.credit_received,
                                "exit_cost": current_cost if 'current_cost' in locals() else 0.0,
                                "symbols": leg_symbols,
                                "ivr": self.active_position.metadata.get("ivr", 0),
                                "vix": self.active_position.metadata.get("vix", 0),
                                "mtf_consensus": self.active_position.mtf_consensus,
                                "wing_width": self.active_position.metadata.get("width", 0),
                                "strikes": self.active_position.metadata.get("strikes", {}),
                                "entry_ohlcv": self.active_position.metadata.get("entry_ohlcv", {}),
                                "exit_ohlcv": exit_ohlcv
                            })
                            self.active_position = None
                            unrealized_pnl = 0.0
                            self.bars_since_trade = 0 # Cooling Period

            # Track total equity including floating gain/loss
            total_equity = current_equity + unrealized_pnl
            self.peak = max(self.peak, total_equity)
            self.drawdowns.append(self.peak - total_equity)
            self.equity_series.append(total_equity)

            # 2. Skip if already in position or cooling down
            if self.active_position:
                return
                
            self.bars_since_trade += 1
            if self.bars_since_trade < 100:
                return

            # 3. Entry Logic
            chain_records = self.options_data.get(date_now, [])
            if not chain_records:
                return
                
            quote_chain = [OptionQuote(
                symbol=r['option_symbol'],
                expiration=r['expiration'].date(),
                strike=r['strike'],
                is_call=(r['contract_type'] == 'call'),
                bid=r['bid'],
                ask=r['ask'],
                mid=r['last_price'],
                delta=r['delta'],
                iv=r['implied_volatility']
            ) for r in chain_records]

            # Market context (Simulated until we have real indices)
            ivr = 35.0 + np.random.uniform(-5, 5)
            vix = 18.0 + np.random.uniform(-3, 3) 
            
            if ivr < self.s_cfg.iv_rank_min or vix > self.s_cfg.vix_threshold:
                return

            mtf_snapshot = self.sync.get_snapshot(dt_now) if self.sync else None
            from intelligence.fuzzy_engine import calculate_mtf_membership
            mu_mtf = calculate_mtf_membership(mtf_snapshot) if self.s_cfg.use_mtf_filter else 0.5
            
            if self.s_cfg.use_mtf_filter:
                if mu_mtf < self.s_cfg.mtf_consensus_min or mu_mtf > self.s_cfg.mtf_consensus_max:
                    return

            width = regime_wing_width(self.s_cfg, ivr, vix)
            condor = build_condor(quote_chain, self.s_cfg, date_now, self.data.close[0], width)
            
            if not condor:
                return

            credit = calc_condor_credit(condor)
            
            if credit < (width * self.s_cfg.min_credit_to_width):
                return

            # Fuzzy sizing
            from intelligence.fuzzy_engine import compute_position_size, calculate_iv_membership, calculate_regime_membership
            mu_iv = calculate_iv_membership(ivr)
            mu_regime = calculate_regime_membership(vix, self.s_cfg.vix_threshold)
            
            memberships = {'mtf': mu_mtf, 'iv': mu_iv, 'regime': mu_regime}
            quantity = compute_position_size(
                total_equity, width * 100.0, memberships, 
                {'mtf': 0.4, 'iv': 0.3, 'regime': 0.3},
                vix, 12.0, 35.0, self.r_cfg.position_size_pct
            )
            
            if quantity > 0:
                print(f"[Trade] Opened | Qty: {quantity} | Credit: ${credit:.2f} | Exp: {condor.short_call.expiration}")
                entry_ohlcv = {
                    "open": self.data.open[0],
                    "high": self.data.high[0],
                    "low": self.data.low[0],
                    "close": self.data.close[0],
                    "volume": self.data.volume[0]
                }
                self.active_position = PositionState(
                    id=f"BT-{len(self.trade_log)+1}",
                    legs=condor,
                    open_time=dt_now,
                    credit_received=credit,
                    quantity=quantity,
                    adjustments_done={"call":0, "put":0},
                    mtf_consensus=mu_mtf
                )
                self.active_position.metadata = {
                    "ivr": ivr,
                    "vix": vix,
                    "width": width,
                    "entry_ohlcv": entry_ohlcv,
                    "strikes": {
                        "short_call": condor.short_call.strike,
                        "long_call": condor.long_call.strike,
                        "short_put": condor.short_put.strike,
                        "long_put": condor.long_put.strike
                    }
                }

    # === Run Backtest ===
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(r_cfg.backtest_cash)
    cerebro.adddata(PandasData(dataname=df))
    cerebro.addstrategy(IronCondorStrategy, s_cfg=s_cfg, r_cfg=r_cfg, sync_engine=sync_engine, options_data=options_by_date)
    
    # Add Analyzers
    # Run quietly
    results = cerebro.run(maxcpus=1)
    if not results:
        return None
        
    return results[0]  # Return strategy instance

def run_backtest_and_report(s_cfg: StrategyConfig, r_cfg: RunConfig):
    """Execute backtest with comprehensive reporting"""
    
    # Run Headless
    strat = run_backtest_headless(s_cfg, r_cfg)
    
    if strat is None:
        print("[ERROR] Backtest failed to run")
        return

    # Calculate Metrics from Strategy Object
    final_equity = r_cfg.backtest_cash + strat.pnl
    net_profit = final_equity - r_cfg.backtest_cash
    pct_return = (net_profit / r_cfg.backtest_cash) * 100.0
    
    # Date calculations
    if strat.trade_log:
        start_date = strat.trade_log[0]["start"].date()
        end_date = strat.trade_log[-1]["end"].date()
    else:
        # Fallback to feed dates if no trades closed
        start_date = strat.data.datetime.date(-len(strat.data)+1)
        end_date = strat.data.datetime.date(0)
    
    days = (end_date - start_date).days
    years = days / 365.0 if days > 0 else 0.0
    cagr = ((final_equity / r_cfg.backtest_cash) ** (1 / years) - 1) * 100.0 if years > 0 and final_equity > 0 else 0.0

    wins = [t for t in strat.trade_log if t["result"] == "win"]
    losses = [t for t in strat.trade_log if t["result"] == "loss"]
    
    avg_win = sum(t["amount"] for t in wins) / len(wins) if wins else 0.0
    avg_loss = abs(sum(t["amount"] for t in losses)) / len(losses) if losses else 0.0
    win_rate = (len(wins) / len(strat.trade_log)) * 100.0 if strat.trade_log else 0.0
    expectancy = (avg_win * (win_rate/100)) - (avg_loss * (1 - win_rate/100))
    
    # --- Manual Metrics Calculation (Since Simulation mode bypasses Broker) ---
    
    # 1. Drawdown
    # strat.drawdowns tracks [peak - equity], so max(drawdowns) is max_dd money
    max_dd_money = max(strat.drawdowns) if strat.drawdowns else 0.0
    max_dd_pct = (max_dd_money / strat.peak * 100.0) if hasattr(strat, 'peak') and strat.peak > 0 else 0.0
    
    # 2. Sharpe Ratio (Manual)
    # Estimate from Equity Series (High frequency data)
    sharpe = 0.0
    if len(strat.equity_series) > 1:
        equity_curve = np.array(strat.equity_series)
        returns = np.diff(equity_curve) / equity_curve[:-1]
        
        if np.std(returns) > 0:
            # Annualize (Assume 5min bars -> 78 bars/day * 252 days = 19656 bars/year)
            # Adjust N based on actual data frequency if known, but 5min is dominant
            bars_per_year = 252 * 78 
            sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(bars_per_year)

    gross_win = sum(t["amount"] for t in wins)
    gross_loss = abs(sum(t["amount"] for t in losses))
    profit_factor = gross_win / gross_loss if gross_loss > 0 else 0.0
    
    largest_win = max([t["amount"] for t in wins]) if wins else 0.0
    largest_loss = min([t["amount"] for t in losses]) if losses else 0.0
    
    net_pnl_dd_ratio = net_profit / max_dd_money if max_dd_money > 0 else 0.0

    # 3. Total Time in Trade
    total_seconds = sum((t["end"] - t["start"]).total_seconds() for t in strat.trade_log)
    m, s = divmod(total_seconds, 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    time_in_trade_str = f"{int(d)}d {int(h):02d}:{int(m):02d}:{int(s):02d}"

    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")

    print("\n" + "="*60)
    print(f"BACKTEST RESULTS: {s_cfg.underlying}")
    print("="*60)
    print(f"Period:           {start_date_str} to {end_date_str} ({days} days)")
    print(f"Final Equity:     ${final_equity:,.2f}")
    print(f"Net Profit:       ${net_profit:,.2f} ({pct_return:.2f}%)")
    print(f"CAGR:             {cagr:.2f}%")
    print(f"Max Drawdown:     ${max_dd_money:,.2f} ({max_dd_pct:.2f}%)")
    print(f"Sharpe Ratio:     {sharpe:.2f}")
    print(f"Profit Factor:    {profit_factor:.2f}")
    print(f"Net PnL / DD:     {net_pnl_dd_ratio:.2f}")
    print(f"Time in Trade:    {time_in_trade_str}")
    print(f"Total Trades:     {len(strat.trade_log)}")
    print(f"Win Rate:         {win_rate:.2f}%")
    print(f"Avg Win:          ${avg_win:,.2f}")
    print(f"Avg Loss:         ${avg_loss:,.2f}")
    print(f"Expectancy:       ${expectancy:.2f}")
    print(f"Largest Win:      ${largest_win:,.2f}")
    print(f"Largest Loss:     ${largest_loss:,.2f}")
    print(f"MTF Filter:       {'ENABLED' if s_cfg.use_mtf_filter else 'DISABLED'}")
    print("="*60)

    # Export Trades
    if strat.trade_log:
        trades_df = pd.DataFrame(strat.trade_log)
        out_csv = os.path.join("reports", "trades.csv")
        trades_df.to_csv(out_csv, index=False)
        print(f"[Export] Trade log saved: {out_csv}")
    
    # Save PDF Report
    if r_cfg.backtest_plot:
        # Re-load DF just for plotting (inefficient but cleaner for now)
        csv_path = os.path.join("reports", s_cfg.underlying, f"{s_cfg.underlying}_5.csv")
        df = pd.read_csv(csv_path, parse_dates=["timestamp"])
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
        df.set_index("timestamp", inplace=True)
        df.sort_index(inplace=True)
        if r_cfg.backtest_samples and r_cfg.backtest_samples > 0:
            if len(df) > r_cfg.backtest_samples:
                df = df.iloc[-r_cfg.backtest_samples:]
    
        out_pdf = os.path.join("reports", "backtest_report.pdf")
        with PdfPages(out_pdf) as pdf:
            # Page 1: Metrics
            fig, ax = plt.subplots(figsize=(8.5, 11))
            ax.axis('off')
            ax.text(0.5, 0.95, f"Iron Condor Backtest Report: {s_cfg.underlying}", ha='center', fontsize=20, weight='bold')
            
            metrics = [
                ["Metric", "Value"],
                ["Period", f"{start_date_str} to {end_date_str}"],
                ["Initial Capital", f"${r_cfg.backtest_cash:,.2f}"],
                ["Final Equity", f"${final_equity:,.2f}"],
                ["Net Profit", f"${net_profit:,.2f} ({pct_return:.2f}%)"],
                ["Max Drawdown", f"${max_dd_money:,.2f} ({max_dd_pct:.2f}%)"],
                ["Sharpe Ratio", f"{sharpe:.2f}"],
                ["Profit Factor", f"{profit_factor:.2f}"],
                ["Net PnL / DD", f"{net_pnl_dd_ratio:.2f}"],
                ["Time in Trade", f"{time_in_trade_str}"],
                ["Total Trades", f"{len(strat.trade_log)}"],
                ["Win Rate", f"{win_rate:.2f}%"],
                ["Avg Win", f"${avg_win:,.2f}"],
                ["Avg Loss", f"${avg_loss:,.2f}"],
                ["Largest Win", f"${largest_win:,.2f}"],
                ["Largest Loss", f"${largest_loss:,.2f}"],
                ["MTF Filter", "ENABLED" if s_cfg.use_mtf_filter else "DISABLED"],
                ["Liquidity Gate", "ENABLED" if s_cfg.use_liquidity_gate else "DISABLED"],
            ]

            
            table = tabulate(metrics, headers="firstrow", tablefmt="grid")
            ax.text(0.1, 0.5, table, family='monospace', fontsize=12)
            pdf.savefig(fig)
            plt.close(fig)
            
            # Page 2: Equity Curve
            fig, axes = plt.subplots(2, 1, figsize=(11, 8.5), sharex=True)
            
            axes[0].plot(strat.equity_series, label="Equity", color="blue")
            axes[0].set_title("Equity Curve")
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

            axes[1].plot(strat.drawdowns, label="Drawdown", color="red", alpha=0.6)
            axes[1].fill_between(range(len(strat.drawdowns)), strat.drawdowns, 0, color='red', alpha=0.1)
            axes[1].set_title("Drawdown")
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

            # Page 3: Candlestick Chart with Strike Overlay
            window_size = 500
            start_idx = max(0, len(df) - window_size)
            df_window = df.iloc[start_idx:]
            
            # Create trade markers and strike segments
            buy_markers = pd.Series(np.nan, index=df_window.index)
            sell_markers = pd.Series(np.nan, index=df_window.index)
            alines = [] # [( (date1, price1), (date2, price1) ), ...]
            
            for t in strat.trade_log:
                if t["start"] in df_window.index:
                    buy_markers.loc[t["start"]] = t["entry_ohlcv"]["close"]
                if t["end"] in df_window.index:
                    sell_markers.loc[t["end"]] = t["exit_ohlcv"]["close"]
                
                # Add strike lines if either start or end is in window
                if (t["start"] in df_window.index) or (t["end"] in df_window.index):
                    plot_start = max(t["start"], df_window.index[0])
                    plot_end = min(t["end"], df_window.index[-1])
                    if plot_end >= plot_start:
                        s = t.get("strikes", {})
                        for val in s.values():
                            alines.append([ (plot_start, val), (plot_end, val) ])

            apds = []
            if not buy_markers.isna().all():
                apds.append(mpf.make_addplot(buy_markers, type="scatter", markersize=100, marker="^", color="green", label="Entry"))
            if not sell_markers.isna().all():
                apds.append(mpf.make_addplot(sell_markers, type="scatter", markersize=100, marker="v", color="red", label="Exit"))
            
            plot_kwargs = {
                'type': "candle",
                'style': "charles",
                'title': f"{s_cfg.underlying} Price Action (Last {len(df_window)} Bars)",
                'ylabel': "Price ($)",
                'volume': True,
                'returnfig': True,
                'figsize': (11, 8.5)
            }
            if apds:
                plot_kwargs['addplot'] = apds
            if alines:
                plot_kwargs['alines'] = dict(alines=alines, colors='gray', alpha=0.3, linewidths=0.5)
            
            try:
                fig, axlist = mpf.plot(df_window, **plot_kwargs)
                pdf.savefig(fig)
                plt.close(fig)
            except Exception as e:
                print(f"[Warning] Failed to generate candlestick chart: {e}")

            # Page 4: Performance Distributions & Heatmaps
            if strat.trade_log:
                fig, axes = plt.subplots(2, 1, figsize=(11, 8.5))
                
                # 1. P&L Distribution Histogram
                amounts = [t["amount"] for t in strat.trade_log]
                axes[0].hist(amounts, bins=min(len(amounts), 20), color="skyblue", edgecolor="black", alpha=0.7)
                axes[0].axvline(0, color="red", linestyle="--", alpha=0.5)
                axes[0].set_title("Realized P&L Distribution")
                axes[0].set_xlabel("Profit/Loss ($)")
                axes[0].set_ylabel("Frequency")
                
                # 2. Monthly Performance Table
                trades_df = pd.DataFrame(strat.trade_log)
                trades_df['month'] = trades_df['end'].dt.to_period('M')
                monthly_pnl = trades_df.groupby('month')['amount'].sum()
                
                if not monthly_pnl.empty:
                    m_data = [["Month", "P&L", "Trades"]]
                    for month, pnl in monthly_pnl.items():
                        m_count = len(trades_df[trades_df['month'] == month])
                        m_data.append([str(month), f"${pnl:,.2f}", m_count])
                    
                    axes[1].axis('off')
                    m_table = axes[1].table(cellText=m_data, loc='center', cellLoc='center', colWidths=[0.2, 0.2, 0.2])
                    m_table.auto_set_font_size(False)
                    m_table.set_fontsize(12)
                    m_table.scale(1.2, 1.2)
                    axes[1].set_title("Monthly Performance Summary", pad=20)
                
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

                # Page 5: Detailed Recent Trades
                fig, ax = plt.subplots(figsize=(11, 8.5))
                ax.axis('off')
                ax.text(0.5, 0.95, "Recent Trade History (Last 20)", ha='center', fontsize=16, weight='bold')
                
                recent_trades = strat.trade_log[-20:]
                t_data = [["Start Date", "End Date", "Result", "PnL", "Exit Reason"]]
                for t in recent_trades:
                    t_data.append([
                        t["start"].strftime("%Y-%m-%d %H:%M"),
                        t["end"].strftime("%Y-%m-%d %H:%M"),
                        t["result"].upper(),
                        f"${t['amount']:.2f}",
                        t["exit_reason"]
                    ])
                
                table = ax.table(cellText=t_data, loc='center', cellLoc='center')
                table.auto_set_font_size(False)
                table.set_fontsize(10)
                table.scale(1.0, 1.5)
                
                pdf.savefig(fig)
                plt.close(fig)

        print(f"[Export] PDF report saved: {out_pdf}")

    print("\n[Backtest] Complete!")