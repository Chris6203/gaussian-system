# Dashboard Usage Guide

## Overview

The trading system includes a **unified dashboard** that serves three dashboards from a single entry point:

| Dashboard | Port | Purpose |
|-----------|------|---------|
| Live | 5000 | Monitor live/paper trading |
| Training | 5001 | Monitor training runs |
| History | 5002 | Browse historical models |

## Quick Start

```bash
# Start all dashboards
python dashboard.py

# Start specific dashboard
python dashboard.py --live              # Live trading only
python dashboard.py --training          # Training only
python dashboard.py --history           # History browser only

# Check status
python dashboard.py --status

# Custom port
python dashboard.py --live --port 8080
```

## Usage Scenarios

### 1. Live Trading (with bot running)

```bash
# Terminal 1: Start the bot
python bot.py models/run_20260105_123456

# Terminal 2: Start the live dashboard
python dashboard.py --live
```
Visit: http://localhost:5000

### 2. Training Monitoring

```bash
# Terminal 1: Start training
python scripts/train_time_travel.py

# Terminal 2: Start the training dashboard
python dashboard.py --training
```
Visit: http://localhost:5001

### 3. All Dashboards (full monitoring)

```bash
# Start all dashboards at once
python dashboard.py

# Or start specific combination
python dashboard.py --live --training
```

Live: http://localhost:5000
Training: http://localhost:5001
History: http://localhost:5002

## Server Configuration

### server_config.json

Server IPs are configured in `server_config.json` with **automatic localhost fallback**:

```json
{
  "primary": {
    "ip": "192.168.20.235",
    "description": "Primary server hosting dashboard and data manager"
  },
  "fallback": {
    "ip": "localhost",
    "description": "Localhost fallback for standalone operation"
  },
  "dashboard": { "port": 5000 },
  "training_dashboard": { "port": 5001 },
  "data_manager": { "port": 5050, "timeout_seconds": 3 }
}
```

**Behavior:**
- On startup, system checks if primary server (192.168.20.235:5050) is reachable
- If unreachable, automatically falls back to localhost for standalone operation
- No code changes needed when switching between network and standalone modes

**To migrate to a new server**, update `primary.ip` in this file.

### config.json Dashboard Settings

```json
{
    "dashboard": {
        "port": 5000,
        "host": "0.0.0.0",
        "paper_trading_db": "data/paper_trading.db",
        "historical_db": "data/db/historical.db",
        "log_pattern": "logs/real_bot_simulation*.log",
        "refresh_interval_ms": 2000
    },
    "training_dashboard": {
        "port": 5001,
        "host": "0.0.0.0",
        "log_pattern": "logs/real_bot_simulation*.log",
        "refresh_interval_ms": 2000
    }
}
```

## Dashboard Features

### Live Dashboard (port 5000)

- Current balance and P&L
- Open positions with unrealized P&L
- Recent trades history
- Real-time SPY price chart with predictions
- VIX with Bollinger Bands (toggleable)
- HMM regime state
- Signal confidence
- All times in Eastern Time (market time)

### Training Dashboard (port 5001)

- Training cycles progress
- Model predictions (TCN/LSTM/Transformer/Mamba2)
- Trade history during training
- Win rate and loss tracking
- VIX with Bollinger Bands
- Prediction accuracy metrics

### History Dashboard (port 5002)

- Browse all experiment runs
- View SUMMARY.txt and ANALYSIS.md
- Compare configurations
- Sort by P&L, win rate, trades

## Chart Features

### Timezone Display
All charts display in **Eastern Time (ET)** for consistent market hours (9:30 AM - 4:00 PM ET).

### VIX Bollinger Bands
- **VIX line** (green): Current VIX value
- **BB Upper/Lower** (dashed): 2 standard deviation bands
- **BB Middle** (dotted): 20-period SMA

VIX color coding:
- Green: VIX < 15 (low volatility)
- Yellow: VIX 15-25 (moderate)
- Red: VIX > 25 (high volatility)

## API Endpoints

All dashboards provide these endpoints:

| Endpoint | Description |
|----------|-------------|
| `/api/data` | Main dashboard data |
| `/api/chart` | Price chart data with VIX |
| `/api/health` | Server health check |
| `/api/debug` | Debug information |

## Remote Access

To access dashboards from another machine:

1. Ensure `host` is set to `0.0.0.0` in config
2. Check firewall allows the ports (5000-5002)
3. Access via: `http://<server-ip>:<port>`

Example:
```
http://192.168.20.235:5000  # Live dashboard
http://192.168.20.235:5001  # Training dashboard
```

## Troubleshooting

### Port Conflicts
```bash
# Linux
lsof -i :5000
lsof -i :5001

# Windows
netstat -ano | findstr ":5000 :5001"
```

### No Data Showing
1. Check log file pattern matches your logs
2. Verify database files exist in `data/`
3. Check `/api/debug` endpoint

### Charts Not Loading
1. Ensure historical database has data
2. Check browser console for JavaScript errors
3. Hard refresh (Ctrl+F5) to clear cache

### Dashboard Status
```bash
python dashboard.py --status
```

Shows which dashboards are running and their URLs.

## File Locations

| File | Purpose |
|------|---------|
| `dashboard.py` | Entry point (root) |
| `core/dashboards/dashboard_server.py` | Live dashboard implementation |
| `core/dashboards/training_dashboard_server.py` | Training dashboard implementation |
| `core/dashboards/history_dashboard_server.py` | History dashboard implementation |
| `server_config.json` | Server IP configuration |
