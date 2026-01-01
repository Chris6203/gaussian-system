# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

**Gaussian Options Trading Bot** - an algorithmic SPY options trading system combining:
- Bayesian neural networks with Gaussian kernel processors for price/volatility prediction
- **V3 Multi-Horizon Predictor** (5m, 15m, 30m, 45m predictions) - **NEW BEST: +1327% P&L**
- Multi-dimensional HMM (3×3×3 = 27 states) for market regime detection
- Multiple entry controllers: bandit (default), RL (PPO), Q-Scorer, consensus
- Paper trading and live execution via Tradier API

The system uses a modular architecture with swappable predictors and temporal encoders.

## Common Commands

### Training & Simulation

```bash
# BEST CONFIGURATION - V3 Multi-Horizon Predictor (+1327% P&L in 10K test)
PREDICTOR_ARCH=v3_multi_horizon python scripts/train_time_travel.py

# Alternative - Transformer encoder (+801% P&L in 10K test)
TEMPORAL_ENCODER=transformer python scripts/train_time_travel.py

# Combined (experimental)
PREDICTOR_ARCH=v3_multi_horizon TEMPORAL_ENCODER=transformer python scripts/train_time_travel.py

# Standard time-travel training (default V2 predictor)
python scripts/train_time_travel.py

# With environment overrides
MODEL_RUN_DIR=models/my_test TT_MAX_CYCLES=5000 PAPER_TRADING=True python scripts/train_time_travel.py

# Train predictor (Phase 1: supervised)
python scripts/train_predictor.py --output models/predictor_v2.pt --arch v2_slim_bayesian

# Train RL policy (Phase 2: frozen predictor)
python scripts/train_rl.py --predictor models/predictor_v2.pt --freeze-predictor
```

### Live Trading

```bash
# Paper trading (default - PAPER_ONLY_MODE=True in go_live_only.py)
python go_live_only.py models/run_YYYYMMDD_HHMMSS

# Live trading (set PAPER_ONLY_MODE=False in go_live_only.py first!)
python go_live_only.py models/run_YYYYMMDD_HHMMSS
```

### Q-Scorer System

```bash
# Full pipeline: dataset → train → eval
python training/run_q_pipeline.py --horizon 15 --cycles 2500

# Deploy Q-scorer
set ENTRY_CONTROLLER=q_scorer && python go_live_only.py models/run_YYYYMMDD_HHMMSS
```

### Testing & Diagnostics

```bash
python -m pytest tests/                    # Run all tests
python -m pytest tests/test_features.py -v # Single test file
python scripts/diagnose_win_rate.py        # Debug poor performance
python training_dashboard_server.py        # Web dashboard (port 5001)
python dashboard_hub_server.py             # Unified hub with Agent API (port 5003)
```

### Data Management

```bash
python scripts/fetch_historical_data.py    # Fetch from Polygon/Tradier
python verify_data.py                       # Verify data quality
python check_schema.py                      # Check DB schema
```

## Architecture Overview

### Data Flow

```
Market Data → Features (50-500 dims) → HMM Regime → Predictor → Entry Policy → Exit Policy → Execution
```

### Two-Phase Training

1. **Phase 1 (Supervised)**: Train `UnifiedOptionsPredictor` on historical data
2. **Phase 2 (RL)**: Freeze predictor, train entry/exit policies

**"Frozen predictor"** means: predictor still runs every cycle to generate predictions, but its weights are NOT updated during RL training. This prevents conflicting gradients.

### Key Components

| Component | File | Purpose |
|-----------|------|---------|
| Main Orchestrator | `unified_options_trading_bot.py` | Combines all components |
| Neural Predictor | `bot_modules/neural_networks.py` | TCN/LSTM + Bayesian heads + RBF kernels |
| HMM Regime | `backend/multi_dimensional_hmm.py` | 3×3×3 trend/vol/liquidity detection |
| Entry Policy | `backend/unified_rl_policy.py` | 18 state features → 4 actions (HOLD/CALL/PUT/EXIT) |
| Exit Manager | `backend/unified_exit_manager.py` | Hard rules first, then model-based |
| Paper Trading | `backend/paper_trading_system.py` | Simulated execution |
| Live Execution | `execution/tradier_adapter.py` | Tradier API integration |

### Directory Structure

| Directory | Purpose |
|-----------|---------|
| `backend/` | Core trading infrastructure (policies, HMM, execution) |
| `bot_modules/` | Neural networks, features, signals |
| `execution/` | Live trading execution layer |
| `features/` | Modular feature pipeline (macro, equity, options, etc.) |
| `scripts/` | Training scripts (`train_time_travel.py` is primary) |
| `training/` | Q-Scorer training pipeline |
| `experiments/` | Experiment runners (isolated from core) |
| `tools/` | Analysis utilities |
| `docs/` | Architecture documentation |
| `data-manager/` | Separate data collection service |

## Configuration

All configuration via `config.json` (single source of truth). **Note: config.json is gitignored because it contains API keys.**

### Key Sections

| Section | Purpose |
|---------|---------|
| `entry_controller.type` | `bandit` (default), `rl`, `q_scorer`, `consensus` |
| `architecture.exit_policy` | Hard stops + XGBoost/NN exit |
| `architecture.predictor` | Neural architecture (v2_slim_bayesian) |
| `time_travel_training` | Simulation parameters |
| `credentials` | API keys (Tradier, Polygon, FMP) |

### Entry Controllers

| Type | Description | Best For |
|------|-------------|----------|
| `bandit` | HMM-only with strict thresholds | **Default - proven profitable** |
| `rl` | Neural network PPO policy | After collecting training data |
| `q_scorer` | Offline Q-regression | Experimental |
| `consensus` | Multi-signal agreement | High-confidence trades only |

### Exit Rules (Always Active)

| Rule | Threshold | Description |
|------|-----------|-------------|
| Stop Loss | -8% | Hard exit on loss |
| Take Profit | +12% | Hard exit on gain |
| Trailing Stop | +4% activation, 2% trail | Lock in profits |
| Max Hold | 45 min | Prevent theta decay |
| Expiry | <30 min | Exit before expiration |

## Neural Network Details

### UnifiedOptionsPredictor

**Inputs:** Current features `[B, D]` + Sequence `[B, 60, D]` (60 timesteps = 1 hour)

**Architecture:** RBF Kernel → TCN/LSTM (64-dim) → Bayesian Residual Blocks → Multi-head outputs

**Outputs:**
- `return_mean`, `return_std`: Predicted return ± uncertainty
- `direction_probs`: [DOWN, NEUTRAL, UP]
- `confidence`: 0-1
- `risk_adjusted_return`: **Always use this, not raw return_mean**

### UnifiedRLPolicy State (18 features)

| Category | Features |
|----------|----------|
| Position | `is_in_trade`, `is_call`, `pnl_pct`, `drawdown` |
| Time | `minutes_held`, `minutes_to_expiry` |
| Prediction | `predicted_direction`, `confidence`, `momentum_5m` |
| Market | `vix_level`, `volume_spike` |
| HMM Regime | `hmm_trend`, `hmm_vol`, `hmm_liq`, `hmm_conf` |
| Greeks | `theta_decay`, `delta` |

**Actions:** 0=HOLD, 1=BUY_CALL, 2=BUY_PUT, 3=EXIT

### HMM Regime Values

All normalized 0-1: `hmm_trend` (0=Bearish, 1=Bullish), `hmm_volatility`, `hmm_liquidity`, `hmm_confidence`

**Trade blocking:** HMM-neural conflicts are blocked (e.g., bullish HMM + PUT prediction)

## Data Sources

| Source | Priority | Use | Notes |
|--------|----------|-----|-------|
| **Data Manager** | 0 (highest) | Centralized data | Remote server at 31.97.215.206:5050 |
| Tradier | 1 | Real-time + options | Live trading API |
| Polygon | 2 | Historical 1-min bars | Starter plan: 5 years |
| FMP | 3 | Alternative historical | 300 req/min limit |
| Yahoo | 4 | Backup | Free |

### Data Manager Integration

The bot uses a remote Data Manager server for centralized data collection. Configure in `config.json`:

```json
"data_manager": {
    "enabled": true,
    "base_url": "http://31.97.215.206:5050",
    "api_key": "dm_your_api_key_here"
}
```

**Commands:**
```bash
# Test connection
python scripts/test_datamanager.py

# Sync data for training (fetches from Data Manager → local SQLite)
python scripts/sync_from_datamanager.py --days 30

# Live trading automatically uses Data Manager as primary source
```

## Databases

SQLite in `data/`:
- `paper_trading.db`: Paper trading records (trades linked by `run_id`)
- `historical.db`: Historical market data
- `experiments.db`: Experiment tracking (240+ runs, scoreboard data)
- `unified_options_bot.db`: Main trading database

## Common Gotchas

1. **Frozen predictor**: Predictor runs but weights don't update during RL training
2. **Use `risk_adjusted_return`**: Not raw `return_mean`
3. **HMM alignment**: Conflicting signals are blocked by design
4. **Hard exits fire first**: Before model-based exit decisions
5. **Sequence length = 60**: Models expect exactly 60 timesteps
6. **config.json has API keys**: It's gitignored, don't commit it
7. **Settlement T+1**: Options settle next business day
8. **Paper vs live**: Check `PAPER_ONLY_MODE` in `go_live_only.py`
9. **Q-Scorer Q_INVERT_FIX=1**: Required to fix anti-selection bug
10. **Walk-forward split**: Q-scorer uses time-ordered split, not random

## Documentation Reference

| Document | Content |
|----------|---------|
| `docs/ARCH_FLOW_V2.md` | V2 architecture, frozen predictor, data flow |
| `docs/SYSTEM_ARCHITECTURE.md` | Complete system overview |
| `docs/Q_SCORER_SYSTEM_MAP.md` | Q-Scorer architecture |
| `docs/NEURAL_NETWORK_REFERENCE.md` | NN architecture details |
| `RESULTS_TRACKER.md` | Experiment results and findings |
| `backend/dashboard/agent_api.py` | Agent API for AI collaboration |
| `backend/dashboard/scoreboard_api.py` | Scoreboard REST endpoints |
| `backend/dashboard/trades_api.py` | Trade browser REST endpoints |

## Debugging Win Rate

See `docs/SYSTEM_ARCHITECTURE.md` "Debugging Win Rate" section. Quick reference:

| Symptom | Fix |
|---------|-----|
| All losses | Widen `stop_loss_pct` |
| Small wins, big losses | Tighten `stop_loss_pct` |
| Many small losses | Increase `min_confidence` |
| Missing good trades | Lower thresholds |
| Theta eating profits | Reduce `max_hold_minutes` |

## Dashboards

The system has multiple dashboards for monitoring and analysis:

### Dashboard Hub (NEW - Port 5003)

Unified dashboard with scoreboard, trade browser, and agent API:

```bash
python dashboard_hub_server.py  # Hub on port 5003
```

**Features:**
- Experiment Scoreboard - Sort/filter all 240+ experiments (click column headers to sort)
- Run Detail Modal - Click any run name to see P&L chart and trade history
- Trade Browser - Browse/filter trades with aggregations
- Agent API - REST endpoints for AI collaboration
- Links to existing Training and Live dashboards

**Access URLs:**
- Local: `http://localhost:5003/`
- Network: `http://192.168.20.235/gaussian/` (via nginx)
- External: `http://50.127.71.5/gaussian/` (if gateway configured)

### Existing Dashboards

| Dashboard | Port | Command | Purpose |
|-----------|------|---------|---------|
| Training | 5001 | `python training_dashboard_server.py` | Real-time training monitoring |
| Live | 5000 | `python dashboard_server.py` | Live trading with Tradier |
| History | 5002 | `python history_dashboard_server.py` | Browse past models |

## Agent API for AI Collaboration

REST API for AI agents (Claude, Codex, Gemini) to query experiments and suggest improvements.

### Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /api/agent/summary` | AI-friendly system overview |
| `GET /api/agent/experiments` | All experiments with filtering |
| `GET /api/agent/experiments/best` | Top N performers |
| `GET /api/agent/experiments/compare?runs=a,b` | Compare multiple runs |
| `GET /api/agent/suggest` | Context for generating suggestions |
| `POST /api/agent/ideas` | Submit new experiment ideas |
| `GET /api/agent/status` | Currently running experiments |

### Example Usage

```bash
# Get AI-friendly summary
curl http://localhost:5003/api/agent/summary

# Get top 10 experiments
curl "http://localhost:5003/api/agent/experiments/best?limit=10"

# Submit an experiment idea
curl -X POST http://localhost:5003/api/agent/ideas \
  -H "Content-Type: application/json" \
  -d '{"title": "Test wider stops", "hypothesis": "May reduce premature exits", "env_vars": {"HARD_STOP_LOSS_PCT": "12"}}'
```

### For External AI Integration

```python
import requests

# Fetch all experiments for analysis
experiments = requests.get("http://server:5003/api/agent/experiments").json()

# Get suggestion context
context = requests.get("http://server:5003/api/agent/suggest").json()
print(context['patterns_observed'])  # What's working
print(context['worst_runs'])          # What to avoid
```

## Trade Tracking

All trades made during training are automatically linked to their run via `run_id`. This enables:
- Viewing all trades for a specific experiment run
- P&L curve visualization per run
- Comparing trade patterns across different configurations

**How it works:**
1. `train_time_travel.py` sets `run_id` on the paper trader at startup
2. All trades saved to `paper_trading.db` include the `run_id` column
3. Dashboard Hub displays trades grouped by run with P&L charts

**Viewing Trades for a Run:**
- Click any run name in the Dashboard Hub scoreboard
- Modal shows: P&L chart, stats (drawdown, win rate), and trade table
- Or use API: `GET /api/trades?run_id=run_20251220_143000`

**P&L Curve API:**
```bash
# Get cumulative P&L data for charting
curl "http://localhost:5003/api/trades/pnl-curve?run_id=run_20251220_143000"
```

## Data Manager Subsystem

`data-manager/` is a separate Flask app for collecting historical market data:

```bash
cd data-manager
python run.py run --once     # Collect once
python run.py web            # Dashboard on port 5050
python run.py backfill 30    # Backfill 30 days
```

See `data-manager/CLAUDE.md` for details.
