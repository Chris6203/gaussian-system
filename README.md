# Gaussian Options Trading System

A sophisticated algorithmic options trading platform combining Bayesian neural networks, Hidden Markov Models (HMM), and reinforcement learning for automated SPY options trading.

**Author:** Chris Peters
**License:** MIT

## Overview

This system implements a multi-component trading architecture:

- **Bayesian Neural Networks** with Gaussian kernel processors for price/volatility prediction
- **Multi-dimensional HMM** for market regime detection (trend/volatility/liquidity)
- **Reinforcement Learning (PPO)** for entry/exit decisions
- **Consensus Entry Controller** with 8 signal agreement system
- **Central Data Manager** for multi-bot competition and leaderboard tracking
- Paper trading and live execution via Tradier API

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       DATA PIPELINE                              │
│  Market Data → Features → HMM Regime → Predictor → RL State     │
└─────────────────────────────────────────────────────────────────┘
                                │
                ┌───────────────┴───────────────┐
                ▼                               ▼
    ┌───────────────────────┐       ┌───────────────────────┐
    │  PREDICTOR (FROZEN)   │       │   HMM REGIME          │
    │  UnifiedOptionsPredictor       │   MultiDimensionalHMM │
    │  • TCN/LSTM temporal  │       │   • Trend states      │
    │  • Bayesian heads     │       │   • Volatility states │
    │  • RBF kernels        │       │   • Liquidity states  │
    └───────────────────────┘       └───────────────────────┘
                │
                ▼
    ┌─────────────────────────────────────────────────────────┐
    │              ENTRY CONTROLLER                            │
    │  • Consensus (8 signals must agree) - Recommended       │
    │  • Bandit (rule-based with exploration)                 │
    │  • RL Policy (neural network)                           │
    │  • Q-Scorer (offline Q-regression)                      │
    └─────────────────────────────────────────────────────────┘
                │
                ▼
    ┌─────────────────────────────────────────────────────────┐
    │              EXIT POLICY (Layered)                       │
    │  1. Hard safety rules (stop loss, trailing stop, etc.)  │
    │  2. Model-based exit (XGBoost or neural)                │
    └─────────────────────────────────────────────────────────┘
```

## Components

### Trading Bot (`unified_options_trading_bot.py`)
The main orchestrator that combines all components for live/paper trading.

### Data Manager (`data-manager/`)
Centralized data collection and bot competition platform:
- Collects historical market data (equities and options)
- Web dashboard for configuration and monitoring
- Bot leaderboard for tracking performance
- Full configuration capture for strategy replication
- REST API for bot registration and data access

### Backend Modules (`backend/`)
- `consensus_entry_controller.py` - Multi-signal agreement entry system
- `unified_rl_policy.py` - PPO-based RL entry policy
- `multi_dimensional_hmm.py` - 3×3×3 regime detection
- `bot_reporter.py` - Reports trades/metrics to Data Manager
- `config_schema.py` - Configuration extraction and hashing

### Neural Networks (`bot_modules/`)
- `neural_networks.py` - TCN, LSTM, Bayesian layers, RBF kernels
- `gaussian_processor.py` - Gaussian kernel processing
- `signals.py` - Signal combination logic

### Features (`features/`)
Modular feature pipeline:
- `breadth.py` - Market breadth indicators
- `macro.py` - Macro economic features (ETFs, sectors)
- `options_surface.py` - Options surface features
- `crypto.py` - Crypto correlation features

## Quick Start

### 1. Clone and Setup

```bash
git clone https://github.com/YOUR_USERNAME/gaussian-trading-system.git
cd gaussian-trading-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure

Copy the example configuration and add your API keys:

```bash
cp config.example.json config.json
```

Edit `config.json` with your:
- Tradier API credentials
- Data Manager connection (optional)
- Trading parameters

### 3. Run Time-Travel Training

Train the model on historical data:

```bash
python scripts/train_time_travel.py
```

### 4. Paper Trading

```bash
# Set PAPER_ONLY_MODE = True in go_live_only.py
python go_live_only.py models/run_YYYYMMDD_HHMMSS
```

### 5. Live Trading

```bash
# Set PAPER_ONLY_MODE = False in go_live_only.py
python go_live_only.py models/run_YYYYMMDD_HHMMSS
```

## Data Manager Setup

The Data Manager is a separate Flask application for centralized data collection and bot competition.

### Installation

```bash
cd data-manager

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install flask flask-cors requests python-dotenv bcrypt

# Copy environment file
cp .env.example .env
# Edit .env with your API keys
```

### Running

```bash
# Start data collector
python run.py run

# Start web dashboard (separate terminal)
python run.py web --host 0.0.0.0 --port 5050
```

### Bot Competition

Multiple bots can connect to the Data Manager to:
- Pull historical market data
- Report trades and metrics
- Compete on the leaderboard
- Share configurations for replication

```python
from backend.bot_reporter import BotReporter

reporter = BotReporter(config)
reporter.register()  # Registers with full config capture

# Report trades (non-blocking)
reporter.report_trade_async(
    symbol="SPY",
    action="BUY_CALL",
    quantity=1,
    price=2.50,
    pnl=25.00
)
```

## Configuration

All configuration is centralized in `config.json`:

```json
{
  "entry_controller": {
    "type": "consensus",
    "consensus": {
      "min_signals_to_trade": 5,
      "min_weighted_confidence": 0.55
    }
  },
  "trading": {
    "symbol": "SPY",
    "initial_balance": 5000.00,
    "max_positions": 3
  },
  "risk_management": {
    "stop_loss_pct": 8.0,
    "take_profit_pct": 15.0,
    "max_hold_minutes": 45
  },
  "data_manager": {
    "enabled": true,
    "base_url": "http://your-server:5050",
    "api_key": "dm_your_api_key_here"
  }
}
```

## Entry Controllers

| Type | Description |
|------|-------------|
| `consensus` | **Recommended** - 8 signals must agree to trade |
| `bandit` | Rule-based with exploration (for initial training) |
| `rl` | Neural network RL policy |
| `q_scorer` | Offline Q-regression model |

## Project Structure

```
gaussian-system/
├── backend/                 # Core trading infrastructure
│   ├── consensus_entry_controller.py
│   ├── unified_rl_policy.py
│   ├── multi_dimensional_hmm.py
│   ├── bot_reporter.py
│   └── config_schema.py
├── bot_modules/            # Neural network components
│   ├── neural_networks.py
│   └── gaussian_processor.py
├── data-manager/           # Centralized data & competition
│   ├── app/
│   │   ├── web.py
│   │   ├── storage.py
│   │   └── dashboard_template.py
│   └── run.py
├── execution/              # Live trading execution
├── features/               # Feature pipeline
├── scripts/                # Training and utilities
├── config.json             # Main configuration
└── unified_options_trading_bot.py  # Main orchestrator
```

## API Documentation

### Data Manager REST API

**Authentication:** API key in `X-API-Key` header

#### Bot Registration
```http
POST /api/v1/bots/register
{
  "name": "GaussianBot-PC1",
  "owner": "chris",
  "config": { ... },
  "config_hash": "abc123..."
}
```

#### Report Trade
```http
POST /api/v1/bots/{bot_id}/trade
{
  "symbol": "SPY",
  "action": "BUY_CALL",
  "quantity": 1,
  "price": 2.50,
  "pnl": 25.00
}
```

#### Get Leaderboard
```http
GET /api/v1/leaderboard?metric=total_pnl&hours=24
```

#### Export Bot Config
```http
GET /api/v1/bots/{bot_id}/config/export
```

## Testing

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_features.py -v
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Disclaimer

This software is for educational and research purposes only. Trading options involves significant risk of loss. Past performance does not guarantee future results. Always paper trade first and never risk more than you can afford to lose.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Author

**Chris Peters**

---

*Built with Python, PyTorch, and a lot of coffee.*
