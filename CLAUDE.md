# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

**Gaussian Options Trading Bot** - an algorithmic SPY options trading system combining:
- Bayesian neural networks with Gaussian kernel processors for price/volatility prediction
- **65% win rate achieved** with combo_dow configuration (Phase 42)
- Multi-dimensional HMM (3×3×3 = 27 states) for market regime detection
- Multiple entry controllers: bandit (default), RL (PPO), Q-Scorer, consensus
- Paper trading and live execution via Tradier API

The system uses a modular architecture with swappable predictors and temporal encoders.

### Critical Bug Fix (2026-01-01)
A P&L calculation bug was discovered and fixed. **All results before this date showing massive gains (+1327%, +284,618%, etc.) are INVALID.** The bug caused trades to be credited ~165x their actual value due to a missing SQL placeholder. See RESULTS_TRACKER.md "CRITICAL BUG FOUND" section for details.

## Common Commands

### Training & Simulation

```bash
# BEST WIN RATE (65%) - Phase 42 combo_dow configuration
HARD_STOP_LOSS_PCT=50 HARD_TAKE_PROFIT_PCT=10 TRAIN_MAX_CONF=0.25 \
DAY_OF_WEEK_FILTER=1 SKIP_MONDAY=1 SKIP_FRIDAY=1 \
python scripts/train_time_travel.py

# BEST P&L (+54.8%) - Wide stops + small TP
HARD_STOP_LOSS_PCT=50 HARD_TAKE_PROFIT_PCT=10 python scripts/train_time_travel.py

# TCN + Skew Exits (+431% in 20K validation, $36.77/trade)
SKEW_EXIT_ENABLED=1 SKEW_EXIT_MODE=partial python scripts/train_time_travel.py

# Transformer + Skew Exits (+88% 5K, 2.5x improvement over transformer alone)
TEMPORAL_ENCODER=transformer SKEW_EXIT_ENABLED=1 SKEW_EXIT_MODE=partial python scripts/train_time_travel.py

# Transformer only (+32.65% OOS profit, validated post-bug-fix)
TEMPORAL_ENCODER=transformer python scripts/train_time_travel.py

# V3 Multi-Horizon Predictor (experimental, needs validation)
PREDICTOR_ARCH=v3_multi_horizon python scripts/train_time_travel.py

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

### Architecture V4 Improvements (2026-01-02)

**Key Finding:** Edge comes from skew (fat-tail winners), not win rate. One +584% trade = 1293% of P&L.

| Component | File | Purpose |
|-----------|------|---------|
| EV Gate | `backend/ev_gate.py` | Gate on positive EV with Bayesian prior |
| Regime Calibration | `backend/regime_calibration.py` | Per-regime confidence calibrators |
| Regime Attribution | `backend/regime_attribution.py` | Track & auto-disable bad regimes |
| Greeks-Aware Exits | `backend/greeks_aware_exits.py` | VIX/delta-adjusted stops |
| Skew Exit Manager | `backend/skew_exit_manager.py` | Partial TP + trailing runner |

**Skew-Optimized Exits** (capture big winners):
```bash
SKEW_EXIT_ENABLED=1 SKEW_EXIT_MODE=partial python scripts/train_time_travel.py
```

**Winning Presets** (use pre-configured environments):
```bash
source configs/winning_presets.sh
use_skew_optimized  # Partial TP + trailing runner
run_experiment my_test
```

See `docs/ARCHITECTURE_IMPROVEMENTS_V4.md` for full documentation.

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

## Confidence Calibration (Phase 14)

**CRITICAL FINDING**: Raw model confidence does NOT predict trade outcomes!

Analysis of 1,596 trades showed:
- 0.2 confidence → 38.8% win rate
- 0.3 confidence → 37.5% win rate
- 0.4 confidence → 36.6% win rate

Win rate *decreases* as confidence increases - the confidence head was not calibrated.

### Solution: Online Platt/Isotonic Calibration

The `CalibrationTracker` learns the true mapping between raw confidence and P(profit):

```python
# Enabled by default (PNL_CAL_GATE=1)
# Blocks trades where calibrated P(profit) < 42%
PNL_CAL_MIN_PROB=0.42
PNL_CAL_MIN_SAMPLES=30  # Learn from 30 trades before gating
```

### How It Works

1. **Record Entry**: `pnl_calibration_tracker.record_trade_entry(confidence, direction, price)`
2. **Record Exit**: `pnl_calibration_tracker.record_pnl_outcome(trade_id, pnl)`
3. **Calibrate**: After 30 samples, `calibrate_pnl(raw_conf)` returns true P(profit)
4. **Gate**: Block trades where calibrated P(profit) < threshold

### Tracking for Analysis

Both values are saved to trades table:
- `ml_confidence`: Raw model output (may be miscalibrated)
- `calibrated_confidence`: P(profit) from calibration tracker

Query to check calibration quality:
```sql
SELECT
    ROUND(ml_confidence, 1) as raw_conf,
    ROUND(calibrated_confidence, 2) as cal_conf,
    COUNT(*) as trades,
    ROUND(100.0 * SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END) / COUNT(*), 1) as actual_win_pct
FROM trades
WHERE calibrated_confidence IS NOT NULL
GROUP BY ROUND(ml_confidence, 1)
ORDER BY raw_conf;
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PNL_CAL_GATE` | `1` | Enable calibration gating |
| `PNL_CAL_MIN_PROB` | `0.42` | Minimum P(profit) to allow trade |
| `PNL_CAL_MIN_SAMPLES` | `30` | Trades needed before gating |

## Debugging Win Rate

See `docs/SYSTEM_ARCHITECTURE.md` "Debugging Win Rate" section. Quick reference:

| Symptom | Fix |
|---------|-----|
| All losses | Widen `stop_loss_pct` |
| Small wins, big losses | Tighten `stop_loss_pct` |
| Many small losses | Increase `min_confidence` |
| Missing good trades | Lower thresholds |
| Theta eating profits | Reduce `max_hold_minutes` |
| Confidence not predictive | Check calibration stats |

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

## Trade Tuning Data

Every trade now captures detailed context for strategy optimization. This data helps analyze what conditions lead to winning vs losing trades.

### Entry Context Fields

Captured when a trade is opened (`paper_trading.db` → `trades` table):

| Field | Type | Description |
|-------|------|-------------|
| `spy_price` | REAL | SPY price at entry |
| `vix_level` | REAL | VIX at entry |
| `hmm_trend` | REAL | HMM trend state (0=bearish, 0.5=neutral, 1=bullish) |
| `hmm_volatility` | REAL | HMM volatility state (0=low, 1=high) |
| `hmm_liquidity` | REAL | HMM liquidity state |
| `hmm_confidence` | REAL | HMM confidence (0-1) |
| `predicted_return` | REAL | Model's predicted return at entry |
| `prediction_timeframe` | TEXT | Timeframe used (5min, 15min, 30min, etc.) |
| `entry_controller` | TEXT | Controller type: bandit, rl, consensus, q_scorer |
| `signal_strategy` | TEXT | Strategy: NEURAL_BULLISH, HMM_TREND, etc. |
| `signal_reasoning` | TEXT | Full reasoning chain (semicolon-separated) |
| `momentum_5m` | REAL | 5-minute momentum at entry |
| `momentum_15m` | REAL | 15-minute momentum at entry |
| `volume_spike` | REAL | Volume spike indicator |
| `direction_probs` | TEXT | JSON array: [down_prob, neutral_prob, up_prob] |

### Exit Context Fields

Captured when a trade is closed:

| Field | Type | Description |
|-------|------|-------------|
| `exit_spy_price` | REAL | SPY price at exit |
| `exit_vix_level` | REAL | VIX at exit |
| `hold_minutes` | REAL | Actual hold duration in minutes |
| `max_drawdown_pct` | REAL | Maximum drawdown during trade (tracked continuously) |
| `max_gain_pct` | REAL | Maximum gain during trade (tracked continuously) |
| `exit_hmm_trend` | REAL | HMM trend at exit (detect regime changes) |

### Querying Tuning Data

```sql
-- Find trades where HMM regime changed during trade
SELECT * FROM trades
WHERE abs(hmm_trend - exit_hmm_trend) > 0.3
AND profit_loss < 0;

-- Analyze win rate by entry controller
SELECT entry_controller,
       COUNT(*) as trades,
       SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END) as wins,
       AVG(profit_loss) as avg_pnl
FROM trades
WHERE entry_controller IS NOT NULL
GROUP BY entry_controller;

-- Find optimal confidence threshold
SELECT
    CASE
        WHEN ml_confidence < 0.6 THEN 'low (<60%)'
        WHEN ml_confidence < 0.8 THEN 'medium (60-80%)'
        ELSE 'high (>80%)'
    END as conf_bucket,
    COUNT(*) as trades,
    AVG(profit_loss) as avg_pnl,
    AVG(CASE WHEN profit_loss > 0 THEN 1.0 ELSE 0.0 END) as win_rate
FROM trades
GROUP BY conf_bucket;
```

### Migration

If upgrading from an older version, run the migration to add tuning columns:

```bash
python scripts/migrate_add_tuning_fields.py
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
