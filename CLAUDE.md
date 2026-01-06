# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

**Gaussian Options Trading Bot** - an algorithmic SPY options trading system combining:
- Bayesian neural networks with Gaussian kernel processors for price/volatility prediction
- **NEW BEST (Phase 52): SKIP_MONDAY strategy - +1630% P&L, P&L/DD ratio 35.03**
- Multi-dimensional HMM (3×3×3 = 27 states) for market regime detection
- Multiple entry controllers: bandit (default), RL (PPO), Q-Scorer, consensus
- Paper trading and live execution via Tradier API

The system uses a modular architecture with swappable predictors and temporal encoders.

### Critical Bug Fix (2026-01-01)
A P&L calculation bug was discovered and fixed. **All results before this date showing massive gains (+1327%, +284,618%, etc.) are INVALID.** The bug caused trades to be credited ~165x their actual value due to a missing SQL placeholder. See RESULTS_TRACKER.md "CRITICAL BUG FOUND" section for details.

### CRITICAL: Confidence Head Is Broken (2026-01-06)
**The neural network's confidence head outputs INVERTED values** because it has NO loss function training it by default.

| Confidence | Actual Win Rate | Problem |
|------------|-----------------|---------|
| 40%+ | **0%** | Completely wrong |
| 15-20% | **7.2%** | Best performance at LOW confidence |

**Why**: The confidence head (`nn.Linear(64, 1)`) exists but `TRAIN_CONFIDENCE_BCE=0` by default, so no loss trains it. It learns backwards correlations through gradient leakage.

**Workarounds**:
```bash
# Option 1: Filter out broken high-confidence signals (RECOMMENDED)
TRAIN_MAX_CONF=0.25 python scripts/train_time_travel.py

# Option 2: Use direction entropy instead of confidence head
USE_ENTROPY_CONFIDENCE=1 python scripts/train_time_travel.py

# Option 3: Pretrain confidence head with BCE loss first
python scripts/pretrain_confidence.py --epochs 100 --output models/pretrained_bce.pt
LOAD_PRETRAINED=1 PRETRAINED_MODEL_PATH=models/pretrained_bce.pt python scripts/train_time_travel.py
```

See Phase 36 in RESULTS_TRACKER.md for full analysis.

## Common Commands

### Training & Simulation

```bash
# =========================================================
# BEST OVERALL (Phase 52): SKIP_MONDAY - +1630% P&L, P&L/DD 35.03
# =========================================================
USE_TRAILING_STOP=1 TRAILING_ACTIVATION_PCT=10 TRAILING_STOP_PCT=5 \
ENABLE_TDA=1 TDA_REGIME_FILTER=1 TRAIN_MAX_CONF=0.25 \
DAY_OF_WEEK_FILTER=1 SKIP_MONDAY=1 SKIP_FRIDAY=0 \
python scripts/train_time_travel.py

# Or use the launcher script:
./run_live_skip_monday.sh

# ALTERNATIVE: Phase 42 combo_dow configuration (65% win rate)
HARD_STOP_LOSS_PCT=50 HARD_TAKE_PROFIT_PCT=10 TRAIN_MAX_CONF=0.25 \
DAY_OF_WEEK_FILTER=1 SKIP_MONDAY=1 SKIP_FRIDAY=1 \
python scripts/train_time_travel.py

# ALTERNATIVE: Wide stops + small TP (+54.8%)
HARD_STOP_LOSS_PCT=50 HARD_TAKE_PROFIT_PCT=10 python scripts/train_time_travel.py

# TCN + Skew Exits (+431% in 20K validation, $36.77/trade)
SKEW_EXIT_ENABLED=1 SKEW_EXIT_MODE=partial python scripts/train_time_travel.py

# Transformer + Skew Exits (+88% 5K, 2.5x improvement over transformer alone)
TEMPORAL_ENCODER=transformer SKEW_EXIT_ENABLED=1 SKEW_EXIT_MODE=partial python scripts/train_time_travel.py

# Transformer only (+32.65% OOS profit, validated post-bug-fix)
TEMPORAL_ENCODER=transformer python scripts/train_time_travel.py

# Mamba2 State Space Model encoder (Phase 51 - linear complexity SSM)
TEMPORAL_ENCODER=mamba2 python scripts/train_time_travel.py

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
# RECOMMENDED: Use SKIP_MONDAY launcher (best config)
./run_live_skip_monday.sh                    # Paper trading
touch go_live.flag && ./run_live_skip_monday.sh  # LIVE trading

# Alternative: Manual launch with model directory
python core/go_live_only.py models/COMBO_SKIP_MONDAY_20K

# Paper trading (default - no go_live.flag)
python core/go_live_only.py models/run_YYYYMMDD_HHMMSS

# Live trading (create go_live.flag first!)
touch go_live.flag
python core/go_live_only.py models/run_YYYYMMDD_HHMMSS
```

### Q-Scorer System

```bash
# Full pipeline: dataset → train → eval
python training/run_q_pipeline.py --horizon 15 --cycles 2500

# Deploy Q-scorer
ENTRY_CONTROLLER=q_scorer python bot.py models/run_YYYYMMDD_HHMMSS
```

### Dashboards

```bash
# Start all dashboards (live:5000, training:5001, history:5002)
python dashboard.py

# Start specific dashboard
python dashboard.py --live              # Live trading dashboard only
python dashboard.py --training          # Training dashboard only
python dashboard.py --status            # Show dashboard status
```

### Experiments

```bash
# Run Layer 1 optimizer (continuous experiments)
python experiments.py

# Run Layer 2 meta optimizer (analyzes patterns)
python experiments.py --meta

# Run both layers
python experiments.py --both

# Show experiment status
python experiments.py --status

# Add experiment idea
python experiments.py --add "Test wider stops"
```

### Testing & Diagnostics

```bash
python -m pytest tests/                    # Run all tests
python -m pytest tests/test_features.py -v # Single test file
python scripts/diagnose_win_rate.py        # Debug poor performance
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

### Entry Points (Root Directory)

| Script | Purpose |
|--------|---------|
| `bot.py` | Live/paper trading bot entry point |
| `dashboard.py` | Unified dashboard server |
| `experiments.py` | Experiment system (Layer 1 + Layer 2) |

### Key Components

| Component | File | Purpose |
|-----------|------|---------|
| Main Orchestrator | `core/unified_options_trading_bot.py` | Combines all components |
| Neural Predictor | `bot_modules/neural_networks.py` | TCN/LSTM + Bayesian heads + RBF kernels |
| HMM Regime | `backend/multi_dimensional_hmm.py` | 3×3×3 trend/vol/liquidity detection |
| Entry Policy | `backend/unified_rl_policy.py` | 18-22 state features → 4 actions (HOLD/CALL/PUT/EXIT) |
| Sentiment | `features/sentiment.py` | Fear & Greed, PCR, VIX, News sentiment |
| Exit Manager | `backend/unified_exit_manager.py` | Hard rules first, then model-based |
| Paper Trading | `backend/paper_trading_system.py` | Simulated execution |
| Live Execution | `execution/tradier_adapter.py` | Tradier API integration |

### Directory Structure

| Directory | Purpose |
|-----------|---------|
| `core/` | Core bot and dashboard implementations |
| `backend/` | Trading infrastructure (policies, HMM, execution) |
| `bot_modules/` | Neural networks, features, signals |
| `execution/` | Live trading execution layer |
| `features/` | Modular feature pipeline (macro, equity, options, etc.) |
| `scripts/` | Training scripts (`train_time_travel.py` is primary) |
| `training/` | Q-Scorer training pipeline |
| `experiments/` | Experiment runners (isolated from core) |
| `tools/` | Analysis utilities |
| `docs/` | Architecture documentation |
| `archive/` | Old/deprecated files |
| `data-manager/` | Separate data collection service |

## Configuration

### Server Configuration (server_config.json)

Server IPs are centralized in `server_config.json` with **automatic localhost fallback**:

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
- On startup, the system checks if primary server (192.168.20.235:5050) is reachable
- If unreachable, automatically falls back to localhost for standalone operation
- Use `config_loader.get_data_manager_url()` to get the correct URL

To migrate the system to a new server, update the IP address in `primary.ip`.

### Main Configuration (config.json)

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

### UnifiedRLPolicy State (18-22 features)

| Category | Features |
|----------|----------|
| Position | `is_in_trade`, `is_call`, `pnl_pct`, `drawdown` |
| Time | `minutes_held`, `minutes_to_expiry` |
| Prediction | `predicted_direction`, `confidence`, `momentum_5m` |
| Market | `vix_level`, `volume_spike` |
| HMM Regime | `hmm_trend`, `hmm_vol`, `hmm_liq`, `hmm_conf` |
| Greeks | `theta_decay`, `delta` |
| Sentiment* | `fear_greed`, `pcr`, `contrarian`, `news` |

*Sentiment features enabled by default (SENTIMENT_FEATURES_ENABLED=1)

**Actions:** 0=HOLD, 1=BUY_CALL, 2=BUY_PUT, 3=EXIT

### HMM Regime Values

All normalized 0-1: `hmm_trend` (0=Bearish, 1=Bullish), `hmm_volatility`, `hmm_liquidity`, `hmm_confidence`

**Trade blocking:** HMM-neural conflicts are blocked (e.g., bullish HMM + PUT prediction)

## Data Sources

| Source | Priority | Use | Notes |
|--------|----------|-----|-------|
| **Data Manager** | 0 (highest) | Centralized data | Uses server_config.json (primary: 192.168.20.235, fallback: localhost) |
| Tradier | 1 | Real-time + options | Live trading API |
| Polygon | 2 | Historical 1-min bars | Starter plan: 5 years |
| FMP | 3 | Alternative historical | 300 req/min limit |
| Yahoo | 4 | Backup | Free |

### Sentiment Data Sources

| Source | Data | API | Notes |
|--------|------|-----|-------|
| **Fear & Greed Index** | Market sentiment (0-100) | `api.alternative.me/fng/` | Free, cached 5 min |
| **Polygon News** | News sentiment | Polygon API | Uses existing API key |
| **Put/Call Ratio** | Options flow | Computed | From options chain data |
| **VIX** | Volatility sentiment | Market data | Already in features |

**Sentiment features** (`features/sentiment.py`):
- `fear_greed`: 0=extreme fear, 1=extreme greed
- `pcr_contrarian`: High PCR = bullish (contrarian), Low = bearish
- `composite_contrarian`: Combined contrarian signal
- `news_sentiment`: Polygon news sentiment (-1 to +1)

**Enable/disable:** `SENTIMENT_FEATURES_ENABLED=0` to disable (default: enabled)

### TDA (Topological Data Analysis) Features

Uses persistent homology to detect structural patterns in price data:
- Regime transitions invisible to traditional indicators
- Loop detection (cyclical behavior)
- Crash-like geometry patterns

| Feature | Description |
|---------|-------------|
| `tda_entropy_h0/h1` | Topological complexity (H0=trends, H1=loops) |
| `tda_amplitude_h0/h1` | Persistence strength |
| `tda_loop_trend_ratio` | Loops vs trends ratio (high=choppy) |
| `tda_complexity` | Overall topological complexity |

**Requires:** `pip install giotto-tda`
**Enable/disable:** `ENABLE_TDA=0` to disable (default: enabled if giotto-tda installed)

### Data Manager Integration

The bot uses a remote Data Manager server for centralized data collection.

**Option 1: Explicit URL in config.json**
```json
"data_manager": {
    "enabled": true,
    "base_url": "http://192.168.20.235:5050",
    "api_key": "dm_your_api_key_here"
}
```

**Option 2: Use server_config.json (recommended)**
- Leave `base_url` empty in config.json
- System uses `server_config.json` with automatic localhost fallback
- If primary server unreachable, falls back to localhost:5050

**Commands:**
```bash
# Test connection (uses server_config.json fallback)
python scripts/test_datamanager.py

# Sync data for training (fetches from Data Manager → local SQLite)
python scripts/sync_from_datamanager.py --days 30

# Check which server is active
python -c "from config_loader import get_data_manager_url; print(get_data_manager_url())"
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
8. **Paper vs live**: Use `python bot.py model --live` for live, default is paper
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

### Dashboard System

The unified `dashboard.py` manages all dashboards:

```bash
python dashboard.py                # Start all dashboards
python dashboard.py --live         # Live trading only (port 5000)
python dashboard.py --training     # Training monitor only (port 5001)
python dashboard.py --history      # Model history only (port 5002)
python dashboard.py --status       # Check what's running
```

| Dashboard | Port | Purpose |
|-----------|------|---------|
| Live | 5000 | Live/paper trading monitor |
| Training | 5001 | Training experiment monitor |
| History | 5002 | Browse past model runs |

**Multi-Machine Setup:**
```
┌─────────────────┐
│ Training Box 1  │────────────────────┐
└─────────────────┘                    │
                                       ▼
┌─────────────────┐     ┌──────────────────────────┐
│ Training Box 2  │────▶│   Dashboard Server       │
└─────────────────┘     │   (192.168.20.235)       │
                        └──────────────────────────┘
┌─────────────────┐              ▲
│ Live Trading    │──────────────┘
│ Machine         │
└─────────────────┘
```

Server IP is configured in `server_config.json` for easy migration.

### Dashboard Front-End Structure

Shared CSS/JS files for consistent styling:

```
static/
  css/
    variables.css   # Shared colors, spacing
    base.css        # Reset, typography, layout
    components.css  # Cards, tables, buttons, modals
    chart.css       # Chart.js styles
  js/
    utils.js        # fmt.currency(), fmt.pct(), etc.
    api.js          # DashboardAPI fetch wrapper
    chart-config.js # Chart.js factory functions
    components.js   # Render helpers
templates/
    unified.html    # Unified tabbed dashboard
    hub.html        # Hub-only template
    training.html   # Training-only template
    dashboard.html  # Live-only template
    history.html    # History-only template
```

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

## Automated Optimization System (Phase 51+)

Two-layer optimization architecture for continuous improvement:

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                  Layer 2: Meta Optimizer                     │
│            (scripts/meta_optimizer.py - every 30 min)        │
│                                                              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │  Read All   │───▶│  Aggregate  │───▶│  Generate   │     │
│  │ ANALYSIS.md │    │  Patterns   │    │  New Ideas  │     │
│  └─────────────┘    └─────────────┘    └──────┬──────┘     │
└────────────────────────────────────────────────┼────────────┘
                                                 │
                                                 ▼
┌─────────────────────────────────────────────────────────────┐
│              .claude/collab/idea_queue.json                  │
└────────────────────────────────────────────────┬────────────┘
                                                 │
                                                 ▼
┌─────────────────────────────────────────────────────────────┐
│                Layer 1: Continuous Optimizer                 │
│          (scripts/continuous_optimizer.py - 24/7)           │
│                                                              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │ Pick Ideas  │───▶│  Run 5K     │───▶│  Auto 10K   │     │
│  │ from Queue  │    │  Test       │    │  Validation │     │
│  └─────────────┘    └─────────────┘    └──────┬──────┘     │
└────────────────────────────────────────────────┼────────────┘
                                                 │
                                                 ▼
┌─────────────────────────────────────────────────────────────┐
│            models/<run>/ANALYSIS.md + SUMMARY.txt            │
│               (auto-generated post-experiment)               │
└─────────────────────────────────────────────────────────────┘
```

### Layer 1: Continuous Optimizer

Tests specific configurations 24/7:

```bash
# Start Layer 1 optimizer (runs indefinitely)
python scripts/continuous_optimizer.py

# Single batch then exit
python scripts/continuous_optimizer.py --single

# Dry run (no actual experiments)
python scripts/continuous_optimizer.py --dry-run
```

**Features:**
- Pulls ideas from `.claude/collab/idea_queue.json`
- Runs 5K cycle quick tests
- Auto-promotes winners (per-trade P&L > $0) to 10K validation
- Generates ANALYSIS.md for each experiment
- Git commits results automatically

### Layer 2: Meta Optimizer

Analyzes all experiments and suggests improvements:

```bash
# Run once
python scripts/meta_optimizer.py

# Run every 30 minutes (recommended)
python scripts/meta_optimizer.py --loop --interval 1800

# Preview suggestions without adding to queue
python scripts/meta_optimizer.py --dry-run
```

**What it analyzes:**
- Confidence calibration (inverted in 61% of experiments)
- Losing signal strategies (NEURAL_BEARISH loses in 221/247 experiments)
- Winning configuration patterns
- Optimal env var values

**Output:**
- Generates new experiment ideas
- Adds them to idea_queue.json for Layer 1 to test

### Post-Experiment Analysis

Automatically generates ANALYSIS.md with insights:

```bash
# Analyze single experiment
python tools/post_experiment_analysis.py models/EXP-0172_IDEA-271

# Backfill analysis for all past experiments
python tools/backfill_analysis.py

# Re-analyze all experiments
python tools/backfill_analysis.py --force
```

**ANALYSIS.md contains:**
- Win rate and P&L breakdown
- Confidence calibration check
- Exit reason performance
- Signal strategy performance
- Specific recommendations (e.g., `BLOCK_SIGNAL_STRATEGIES=X,Y`)

### Temporal Encoders

Swappable sequence processing architectures:

| Encoder | Env Var | Complexity | Notes |
|---------|---------|------------|-------|
| TCN (default) | `TEMPORAL_ENCODER=tcn` | O(L) | Proven performer |
| Transformer | `TEMPORAL_ENCODER=transformer` | O(L²) | +32% validated |
| **Mamba2** | `TEMPORAL_ENCODER=mamba2` | O(L) | **+35% with filters** |
| LSTM | `TEMPORAL_ENCODER=lstm` | O(L) | Legacy fallback |

**Mamba2 Configuration:**
```bash
TEMPORAL_ENCODER=mamba2 \
MAMBA2_N_LAYERS=4 \
MAMBA2_D_STATE=64 \
BLOCK_SIGNAL_STRATEGIES=MOMENTUM,VOLATILITY_EXPANSION \
python scripts/train_time_travel.py
```

### Key Findings from Meta Analysis (247 experiments)

| Finding | Recommendation |
|---------|----------------|
| 61% have inverted confidence | Use `USE_ENTROPY_CONFIDENCE_V2=1` |
| NEURAL_BEARISH loses in 90% | Use `BLOCK_SIGNAL_STRATEGIES=NEURAL_BEARISH` |
| NEURAL_BULLISH loses in 64% | Block this signal type too |
| Mamba2 + filters = +35% P&L | Best new architecture combo |

### Idea Queue Format

`.claude/collab/idea_queue.json`:
```json
{
  "ideas": [
    {
      "id": "IDEA-273",
      "source": "meta_optimizer",
      "hypothesis": "Block losing strategies",
      "env_vars": {"BLOCK_SIGNAL_STRATEGIES": "NEURAL_BEARISH,NEURAL_BULLISH"},
      "status": "pending",
      "priority": 1
    }
  ]
}
```

**Status values:** `pending` → `running` → `passed_quick`/`rejected` → `validated`/`failed_validation`

## Quantor-MTFuzz Integration

**Credits:** Adapted from Jerry Mahabub & John Draper's [spy-iron-condor-trading](https://github.com/trextrader/spy-iron-condor-trading)

Integration of key components from the Quantor-MTFuzz deterministic trading framework developed by Jerry Mahabub and John Draper.

### Components

| Component | File | Purpose |
|-----------|------|---------|
| Fuzzy Position Sizer | `integrations/quantor/fuzzy_sizer.py` | 9-factor membership functions for position sizing |
| Regime Filter | `integrations/quantor/regime_filter.py` | 5-regime classification with trading gates |
| Volatility Analyzer | `integrations/quantor/volatility.py` | Realized vol, IV skew, VRP calculations |
| Data Alignment | `integrations/quantor/data_alignment.py` | Backtest quality tracking with confidence decay |
| Facade | `integrations/quantor/facade.py` | Unified interface for all components |

### Data Alignment (Backtest Quality)

Ensures backtest reliability by tracking data freshness:

```python
from integrations.quantor import DataAligner, AlignmentDiagnosticsTracker

# Alignment modes: EXACT (1.0), PRIOR (decays), STALE (<0.5), NONE (0.0)
# iv_conf decays: 0.5^(lag_sec / half_life_sec)
```

**Wired into train_time_travel.py:**
- Tracks alignment each cycle
- Applies `iv_conf` as confidence multiplier to all entry controllers
- Fail-fast stops backtests when data quality degrades (>30% stale)
- Reports alignment stats in final summary

**Environment Variables:**
| Variable | Default | Description |
|----------|---------|-------------|
| `ALIGNMENT_ENABLED` | `1` | Enable alignment tracking |
| `ALIGNMENT_FAIL_FAST` | `1` | Stop on poor data quality |
| `ALIGNMENT_MAX_LAG_SEC` | `600` | Max acceptable lag (10 min) |
| `ALIGNMENT_IV_DECAY_HALF_LIFE` | `300` | Confidence decay half-life (5 min) |

### Regime Filter

5-regime classification with automatic trading gates:

| Regime | VIX | Trend | Allowed |
|--------|-----|-------|---------|
| CRASH | >35 | Any | NO |
| BULL_TREND | <20 | Up | CALLS only |
| BEAR_TREND | <25 | Down | PUTS only |
| HIGH_VOL_RANGE | 25-35 | Sideways | Both (reduced) |
| LOW_VOL_RANGE | <18 | Sideways | Both |

### Fuzzy Position Sizing

9 factors weighted into final position size:
- RSI, ADX, Bollinger Bands position
- ATR, Volume ratio, MACD
- Stochastic, OBV trend, Momentum

**Usage:**
```python
from integrations.quantor import QuantorFacade

quantor = QuantorFacade()
result = quantor.analyze(equity=10000, max_loss=500, direction="CALL", ...)
if result.should_trade:
    size = result.position_size
```

### Testing

```bash
python -m pytest tests/test_quantor_integration.py -v  # 41 tests
```

See `integrations/quantor/README.md` for full documentation
