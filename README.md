# Gaussian Options Trading System

A sophisticated algorithmic options trading platform combining Bayesian neural networks, Hidden Markov Models (HMM), and reinforcement learning for automated SPY options trading.

**Author:** Chris Peters
**License:** MIT

## Overview

This system implements a multi-component trading architecture:

- **Bayesian Neural Networks** with Gaussian kernel processors for price/volatility prediction
- **Swappable Temporal Encoders** - TCN, Transformer, **Mamba2 (State Space Model)**
- **Multi-dimensional HMM** (3×3×3 = 27 states) for market regime detection
- **Two-Layer Automated Optimization** - continuous experimentation with AI-driven improvement
- **Post-Experiment Analysis** - automatic trade pattern analysis and recommendations
- Paper trading and live execution via Tradier API

### Latest Results (Phase 51)

| Configuration | P&L | Win Rate |
|---------------|-----|----------|
| Mamba2 + Signal Filtering | **+34.85%** | 39.8% |
| Transformer + Skew Exits | +32.65% | 38.2% |
| TCN Baseline | +4.21% | 38.8% |

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
git clone https://github.com/Chris6203/gaussian-system.git
cd gaussian-system

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

### 4. Start Trading Bot

```bash
# Paper trading (default)
python bot.py models/run_YYYYMMDD_HHMMSS

# Live trading
python bot.py models/run_YYYYMMDD_HHMMSS --live

# List available models
python bot.py --list
```

### 5. Start Dashboard

```bash
# Start all dashboards
python dashboard.py

# Or start specific dashboard
python dashboard.py --live      # Live trading dashboard
python dashboard.py --training  # Training dashboard
```

### 6. Run Experiments

```bash
# Run experiment optimizer
python experiments.py

# Show experiment status
python experiments.py --status
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

### Server Configuration (server_config.json)

Server IPs are centralized with **automatic localhost fallback**:

```json
{
  "primary": {
    "ip": "192.168.20.235",
    "description": "Primary server"
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

**Behavior:** System checks if primary server is reachable on startup. If not, automatically falls back to localhost for standalone operation.

### Main Configuration (config.json)

All trading configuration is centralized in `config.json`:

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
    "api_key": "dm_your_api_key_here"
  }
}
```

**Note:** `data_manager.base_url` is optional - if not set, uses `server_config.json` with fallback.

### Best Model (best_model.json)

The bot uses `best_model.json` to determine the default model:

```bash
# Set the best model
python bot.py --set-best models/run_20260105_123456

# Run with best model (no model path needed)
python bot.py
```

## Temporal Encoders

The predictor supports multiple temporal encoding architectures:

| Encoder | Description | Best For |
|---------|-------------|----------|
| `tcn` | Temporal Convolutional Network (default) | Fast training, baseline |
| `transformer` | 2-layer causal transformer with RoPE | Complex patterns |
| `mamba2` | State Space Model (SSD formulation) | **Best P&L (+34.85%)** |
| `lstm` | Bidirectional LSTM | Sequential patterns |

```bash
# Use different encoders
TEMPORAL_ENCODER=mamba2 python scripts/train_time_travel.py
TEMPORAL_ENCODER=transformer python scripts/train_time_travel.py
```

### Mamba2 Architecture

Pure PyTorch implementation of Mamba2 (Structured State-space Duality):
- Selective state spaces with data-dependent parameters
- 1D convolution for local context
- Learned gating for input/output projections
- No CUDA kernels required

## Two-Layer Automated Optimization

The system includes a sophisticated two-layer optimization architecture:

```
┌─────────────────────────────────────────────────────────────────┐
│                    LAYER 2: META OPTIMIZER                       │
│  scripts/meta_optimizer.py (runs every 30 min)                  │
│  • Reads all ANALYSIS.md files from experiments                 │
│  • Aggregates patterns (confidence inversions, losing signals)  │
│  • Generates new experiment suggestions                         │
│  • Feeds ideas to Layer 1 queue                                 │
└──────────────────────────────┬──────────────────────────────────┘
                               │ idea_queue.json
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LAYER 1: CONTINUOUS OPTIMIZER                 │
│  scripts/continuous_optimizer.py (runs continuously)            │
│  • Pulls ideas from queue                                       │
│  • Runs experiments with 5K cycles                              │
│  • Saves results to models/<run>/                               │
│  • Triggers post-experiment analysis                            │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    POST-EXPERIMENT ANALYSIS                      │
│  tools/post_experiment_analysis.py                              │
│  • Analyzes all trades from experiment                          │
│  • Compares winners vs losers                                   │
│  • Identifies losing signal strategies                          │
│  • Generates ANALYSIS.md with recommendations                   │
└─────────────────────────────────────────────────────────────────┘
```

### Running the Optimization System

```bash
# Start Layer 1 (continuous experiment runner)
nohup python scripts/continuous_optimizer.py --loop > /tmp/optimizer.log 2>&1 &

# Start Layer 2 (meta analyzer - every 30 min)
nohup python scripts/meta_optimizer.py --loop --interval 1800 > /tmp/meta_optimizer.log 2>&1 &

# Run analysis on a single experiment
python tools/post_experiment_analysis.py models/run_20260105_123456

# Backfill analysis for all past experiments
python tools/backfill_analysis.py
```

### Key Files

| File | Purpose |
|------|---------|
| `scripts/continuous_optimizer.py` | Layer 1 - runs experiments from queue |
| `scripts/meta_optimizer.py` | Layer 2 - analyzes patterns, suggests new configs |
| `tools/post_experiment_analysis.py` | Generates ANALYSIS.md with trade insights |
| `tools/backfill_analysis.py` | Batch analysis for past experiments |
| `.claude/collab/idea_queue.json` | Experiment idea queue |

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
├── bot.py                  # Entry point: Trading bot
├── dashboard.py            # Entry point: Unified dashboard
├── experiments.py          # Entry point: Experiment system
├── config.json             # Main configuration (gitignored)
├── config.example.json     # Configuration template
├── config_loader.py        # Server config with localhost fallback
├── server_config.json      # Server IPs (primary + fallback)
├── best_model.json         # Default model for bot.py
│
├── core/                   # Core implementations
│   ├── go_live_only.py
│   ├── unified_options_trading_bot.py
│   └── dashboards/
│       ├── dashboard_server.py
│       ├── training_dashboard_server.py
│       └── history_dashboard_server.py
│
├── backend/                # Trading infrastructure
│   ├── unified_rl_policy.py
│   ├── multi_dimensional_hmm.py
│   └── paper_trading_system.py
│
├── bot_modules/            # Neural network components
│   └── neural_networks.py  # TCN, Transformer, Mamba2, LSTM
│
├── scripts/                # Training and optimization
│   ├── train_time_travel.py
│   ├── continuous_optimizer.py
│   └── meta_optimizer.py
│
├── tools/                  # Analysis utilities
├── features/               # Feature pipeline
├── execution/              # Live trading execution
├── models/                 # Experiment results
├── data/                   # Databases
├── logs/                   # Log files
├── archive/                # Old/deprecated files
└── data-manager/           # Data collection service
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

## Environment Variables

Key environment variables for configuration:

| Variable | Default | Description |
|----------|---------|-------------|
| `TEMPORAL_ENCODER` | `tcn` | Temporal encoder: `tcn`, `transformer`, `mamba2`, `lstm` |
| `PREDICTOR_ARCH` | `v2_slim_bayesian` | Predictor architecture |
| `MODEL_RUN_DIR` | auto | Output directory for experiment |
| `TT_MAX_CYCLES` | 5000 | Number of training cycles |
| `PAPER_TRADING` | True | Enable paper trading mode |
| `BLOCK_SIGNAL_STRATEGIES` | - | Comma-separated signals to block |
| `HARD_STOP_LOSS_PCT` | 8 | Stop loss percentage |
| `HARD_TAKE_PROFIT_PCT` | 15 | Take profit percentage |
| `MAX_HOLD_MINUTES` | 45 | Maximum position hold time |

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
