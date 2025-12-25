# Dashboard Usage Guide

## Overview

The trading bot has two dashboards that can run simultaneously:

1. **Live Dashboard** (port 5000) - For monitoring live/paper trading
2. **Training Dashboard** (port 5001) - For monitoring training runs

## Starting the Dashboards

### Live Dashboard (while bot is running)
```bash
# Start the live trading bot first
PYTHONIOENCODING=utf-8 python go_live_only.py models/long_run_20k &

# Then start the live dashboard
python dashboard_server.py
```
Visit: http://localhost:5000

### Training Dashboard (while training)
```bash
# Start training first
PAPER_TRADING=True python scripts/train_time_travel.py &

# Then start the training dashboard
python training_dashboard_server.py
```
Visit: http://localhost:5001

## Configuration

### config.json Settings

Add these sections to your `config.json`:

```json
{
    "dashboard": {
        "_note": "Live dashboard settings - port 5000",
        "port": 5000,
        "host": "0.0.0.0",
        "paper_trading_db": "data/paper_trading.db",
        "historical_db": "data/db/historical.db",
        "log_pattern": "logs/real_bot_simulation*.log",
        "refresh_interval_ms": 2000
    },
    "training_dashboard": {
        "_note": "Training dashboard settings - port 5001",
        "port": 5001,
        "host": "0.0.0.0",
        "paper_trading_db": "data/paper_trading.db",
        "historical_db": "data/db/historical.db",
        "log_pattern": "logs/real_bot_simulation*.log",
        "refresh_interval_ms": 2000
    }
}
```

## Key Metrics

### Live Dashboard Shows:
- Current balance and P&L
- Open positions with unrealized P&L
- Recent trades history
- Real-time SPY price chart
- HMM regime state
- Signal confidence

### Training Dashboard Shows:
- Training cycles progress
- Model predictions (LSTM/TCN)
- Trade history during training
- Win rate and loss tracking
- VIX with Bollinger Bands
- Prediction accuracy metrics

## Profitability Indicators

Based on our testing, **low trade rate is the key to profitability**:

| Trade Rate | Typical P&L | Notes |
|------------|-------------|-------|
| ~2% | +400% to +1000% | Highly selective, pre-trained model |
| ~10-15% | -50% to 0% | Fresh model, learning |
| ~18%+ | -90% | Too aggressive, untrained |

The trade rate is calculated as: `trades / cycles * 100`

A profitable model typically shows:
- Trade rate: 1-3%
- Win rate: 55-65%
- Per-trade P&L: $100+ average

## API Endpoints

Both dashboards provide these endpoints:

- `/api/data` - Main dashboard data
- `/api/chart` - Price chart data
- `/api/health` - Server health check
- `/api/debug` - Debug information

## Troubleshooting

### Port Conflicts
If you get a port conflict, check which processes are using the ports:
```bash
netstat -ano | findstr ":5000 :5001"
```

### No Data Showing
1. Check log file pattern matches your logs
2. Verify the database files exist
3. Check the `/api/debug` endpoint for diagnostics

### Charts Not Loading
1. Ensure historical database has data
2. Check browser console for JavaScript errors
3. Try refreshing the page (F5)
