# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

**Quantor-MTFuzz** is an algorithmic options trading system for SPY Iron Condor strategies. It combines Multi-Timeframe (MTF) technical intelligence with Fuzzy Logic position sizing to execute backtests and live paper trades.

## Common Commands

### Backtesting

```bash
# Standard backtest with MTF and fuzzy sizing
python core/main.py --mode backtest --use-mtf --dynamic-sizing --bt-samples 0

# Quick sample backtest (500 bars)
python core/main.py --mode backtest --bt-samples 500

# Optimization (grid search over parameters)
python core/main.py --mode backtest --use-mtf --dynamic-sizing --bt-samples 0 --use-optimizer
```

### Live Trading

```bash
# Local paper trading
python core/main.py --mode live --polygon-key YOUR_KEY

# Alpaca paper trading
python core/main.py --mode live --alpaca --alpaca-key KEY --alpaca-secret SECRET --polygon-key KEY
```

## Architecture

### Data Flow

```
Market Data (CSV/Polygon) → MTF Sync → Fuzzy Engine → Entry Decision → Backtrader Strategy → Exit Rules
```

### Two-Layer Data Model

1. **Strategy Clock** (`reports/SPY/`): 5-min SPY price data drives timing, MA, RSI calculations
2. **Options Pricing** (`data/synthetic_options/`): Black-Scholes generated option chains for mark-to-market P&L

### Key Components

| Component | File | Purpose |
|-----------|------|---------|
| CLI Entry | `core/main.py` | Argument parsing, mode dispatch |
| Backtest Engine | `core/backtest_engine.py` | High-fidelity simulation with Backtrader |
| Optimizer | `core/optimizer.py` | Grid search, ranked by Net Profit / Max Drawdown |
| Strategy Logic | `strategies/options_strategy.py` | Iron Condor leg selection, entry/exit rules |
| Fuzzy Sizing | `intelligence/fuzzy_engine.py` | Position size scaling based on confidence |
| MTF Sync | `data_factory/sync_engine.py` | Multi-timeframe data alignment |
| Regime Filter | `intelligence/regime_filter.py` | Liquidity gate, volatility checks |
| Brokers | `core/broker.py` | `PaperBroker` (local) and `AlpacaBroker` (live) |

### Configuration

- `core/config.template.py` → Copy to `core/config.py` (gitignored)
- `StrategyConfig`: Delta targets, wing widths, IVR/VIX thresholds, exit rules
- `RunConfig`: API keys, backtest dates, position sizing

### Iron Condor Trade Rules

**Entry filters:**
- IV Rank >= threshold (default 20)
- VIX <= threshold (default 25)
- MTF consensus within neutral range (0.40-0.60)
- Liquidity gate passes (volume, spread checks)

**Exit rules:**
- Profit take: Close when cost <= credit × profit_take_pct
- Stop loss: Close when cost >= credit × loss_close_multiple
- Expiration: Close at DTE <= 0
- Max hold: Close at DTE <= max_hold_days

### Fuzzy Position Sizing Pipeline

1. **Base Quantity**: `q0 = floor((risk_fraction × equity) / max_loss_per_contract)`
2. **Fuzzy Confidence**: Weighted sum of MTF, IV, and regime memberships
3. **Volatility Penalty**: Normalized VIX scales down position
4. **Final Quantity**: `q = q0 × confidence × (1 - volatility_penalty)`

## Optimization

The optimizer (`--use-optimizer`) runs a grid search over parameters defined in `OPTIMIZATION_MATRIX` (core/optimizer.py):

- Pre-loads data once for efficiency
- Benchmarks hardware, estimates runtime
- Ranks results by Net Profit / Max Drawdown ratio
- Saves top 100 to `reports/top100_YYYYMMDD_HHMMSS.csv`
- Optionally applies best config to `core/config.py`

## Data Files

| Path | Content |
|------|---------|
| `reports/SPY/SPY_5.csv` | 5-min OHLCV data |
| `reports/SPY/SPY_15.csv` | 15-min OHLCV data |
| `reports/SPY/SPY_60.csv` | Hourly OHLCV data |
| `data/synthetic_options/SPY_5min.csv` | Generated option chains with Greeks |
| `reports/trades.csv` | Trade log from last backtest |
| `reports/backtest_report.pdf` | Multi-page PDF report |
