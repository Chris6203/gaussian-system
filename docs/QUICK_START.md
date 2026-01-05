# Quick Start Guide

## Overview

The Gaussian Options Trading System has **3 main entry points**:

| Script | Purpose |
|--------|---------|
| `bot.py` | Run the trading bot (paper or live) |
| `dashboard.py` | Start web dashboards for monitoring |
| `experiments.py` | Run automated experiment optimization |

---

## 1. Installation

```bash
# Clone the repository
git clone https://github.com/Chris6203/gaussian-system.git
cd gaussian-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Copy configuration
cp config.example.json config.json
# Edit config.json with your API keys
```

---

## 2. Configuration

### config.json
Edit with your API credentials:
```json
{
  "credentials": {
    "tradier_api_key": "YOUR_KEY",
    "tradier_account_id": "YOUR_ACCOUNT"
  }
}
```

### server_config.json
Server IPs with automatic localhost fallback:
```json
{
  "primary": { "ip": "192.168.20.235" },
  "fallback": { "ip": "localhost" },
  "data_manager": { "port": 5050, "timeout_seconds": 3 }
}
```
If primary server is unreachable, system falls back to localhost automatically.

---

## 3. Training

Train a model before trading:

```bash
# Basic training (5000 cycles)
python scripts/train_time_travel.py

# Custom training
MODEL_RUN_DIR=models/my_test TT_MAX_CYCLES=10000 python scripts/train_time_travel.py

# With specific architecture
TEMPORAL_ENCODER=mamba2 python scripts/train_time_travel.py
```

Training creates a model directory in `models/run_YYYYMMDD_HHMMSS/`

---

## 4. Trading Bot

### List Available Models
```bash
python bot.py --list
```

### Use Best Model (Default)
```bash
# Run with the best model (from best_model.json)
python bot.py

# Set a model as the best
python bot.py --set-best models/run_20260105_123456
```

### Paper Trading (Default)
```bash
python bot.py models/run_20260105_123456
```

### Live Trading
```bash
python bot.py models/run_20260105_123456 --live
```

### Check Status
```bash
python bot.py --status
```

---

## 5. Dashboards

### Start All Dashboards
```bash
python dashboard.py
```
- Live: http://localhost:5000
- Training: http://localhost:5001
- History: http://localhost:5002

### Start Specific Dashboard
```bash
python dashboard.py --live      # Live trading dashboard
python dashboard.py --training  # Training dashboard
python dashboard.py --history   # History browser
```

### Check Dashboard Status
```bash
python dashboard.py --status
```

---

## 6. Experiments

### Run Layer 1 Optimizer
Continuously tests configurations from the idea queue:
```bash
python experiments.py
```

### Run Layer 2 Meta Optimizer
Analyzes results and suggests new configurations:
```bash
python experiments.py --meta
```

### Run Both Layers
```bash
python experiments.py --both
```

### Check Status
```bash
python experiments.py --status
python experiments.py --queue
```

### Add Experiment Idea
```bash
python experiments.py --add "Test wider stop loss"
```

---

## 7. Common Workflows

### Development Workflow
```bash
# Terminal 1: Train
python scripts/train_time_travel.py

# Terminal 2: Monitor training
python dashboard.py --training
```

### Paper Trading Workflow
```bash
# Terminal 1: Run bot
python bot.py models/run_xxx

# Terminal 2: Monitor
python dashboard.py --live
```

### Experiment Workflow
```bash
# Terminal 1: Run experiments
python experiments.py --both

# Terminal 2: Monitor
python experiments.py --status
```

---

## 8. Directory Structure

```
gaussian-system/
├── bot.py              # Trading bot entry point
├── dashboard.py        # Dashboard entry point
├── experiments.py      # Experiment entry point
├── config.json         # Main configuration (gitignored)
├── config_loader.py    # Server config with localhost fallback
├── server_config.json  # Server IPs (primary + fallback)
├── best_model.json     # Default model for bot.py
│
├── core/               # Core implementations
├── backend/            # Trading logic
├── bot_modules/        # Neural networks
├── scripts/            # Training scripts
├── models/             # Trained models
├── data/               # Databases
└── logs/               # Log files
```

---

## 9. Getting Help

```bash
python bot.py --help
python dashboard.py --help
python experiments.py --help
```

See also:
- `CLAUDE.md` - AI assistant instructions
- `docs/SYSTEM_ARCHITECTURE.md` - Full architecture
- `docs/DASHBOARD_USAGE.md` - Dashboard details
- `RESULTS_TRACKER.md` - Experiment results

---

## 10. Server Migration

To move the system to a new server:

1. Update `server_config.json` with new IP
2. Ensure firewall allows ports 5000-5002, 5050
3. No changes needed for standalone mode (auto-fallback to localhost)

```json
// server_config.json
{
  "primary": { "ip": "NEW_SERVER_IP" },
  "fallback": { "ip": "localhost" }
}
```

**Standalone Mode:** If primary server is unreachable, system automatically uses localhost. No configuration changes required.
