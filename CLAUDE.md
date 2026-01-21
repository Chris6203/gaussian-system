# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

**Gaussian Options Trading Bot** - an algorithmic SPY options trading system combining:
- Bayesian neural networks with Gaussian kernel processors
- Multi-dimensional HMM (3×3×3 = 27 states) for regime detection
- Multiple entry controllers: bandit (default), RL (PPO), Q-Scorer, consensus
- Paper trading and live execution via Tradier API

**BEST CONFIG (Phase 84):** `SMART_ENTRY_GATE=1 SMART_COOLDOWN_MINUTES=30`
- **64.3% WR at 20K** (above 52.2% breakeven)
- **+$48.33 P&L** at 20K (+0.97%)
- **$0.86/trade** average profit (56 trades)
- Key insight: 30-min cooldown (vs 60-min) finds 40% more trades with 16.8% higher win rate

## Known Issues & Fixes

### Confidence Head Is INVERTED (CRITICAL)
The confidence head outputs inverted values - **LOW confidence = better trades!**

| Confidence | Win Rate | Problem |
|------------|----------|---------|
| 40%+ | **0%** | Completely wrong |
| 15-20% | **7.2%** | Best at LOW confidence |

**Fixes:**
```bash
# Option 1: SmartEntryGate (RECOMMENDED - uses inverted confidence)
SMART_ENTRY_GATE=1 python scripts/train_time_travel.py

# Option 2: Filter high confidence signals
TRAIN_MAX_CONF=0.25 python scripts/train_time_travel.py

# Option 3: Use proper P(win) calculation
USE_PROPER_CONFIDENCE=1 python scripts/train_time_travel.py
```

### Other Fixed Bugs
- **Confidence Inversion (2026-01-12):** Confidence head outputs INVERTED. All fixes FAILED 20K:
  - CALIBRATION: 60% WR at 500 → **34.9%** at 20K ❌
  - BCE Training: 44.9% WR at 5K → **38.7%** at 20K ❌
- **P&L Bug (2026-01-01):** Results before this date showing massive gains are INVALID
- **TEMPORAL_ENCODER Bug (2026-01-06):** V2/V3 predictors ignored env var, always used LSTM
- **HMM Gate Bug (2026-01-07):** Was blocking good signals. Fix: `SKIP_HMM_ALIGNMENT=1`
- **Direction Threshold Bug:** `MIN_DIRECTION_THRESHOLD=0.5` is too high (use 0.05)

### Institutional Fixes (Phase 61c)
Three fixes based on professional trading feedback:

| Fix | Env Var | Purpose |
|-----|---------|---------|
| Disable confidence gating | `DISABLE_CONFIDENCE_GATE=1` | Confidence is inverted, gating makes it WORSE |
| Realistic paper fills | `REALISTIC_FILLS=1` | Mid-price fills inflate training results |
| Train on option P&L | `TRAIN_ON_OPTION_PNL=1` | Already default |

**5K Results:** 47.1% WR (-0.72% P&L) - improved from 37.6% baseline
**Status:** Needs 20K validation

### CRITICAL: 5K Tests Are Unreliable
| Test | 5K Win Rate | 20K Win Rate | Inflation |
|------|-------------|--------------|-----------|
| CALIBRATION | 60% | 34.9% | +25% |
| BCE Training | 44.9% | 38.7% | +6% |

**Rule:** Never trust 5K results. Always validate at 20K+.

## Quick Start

### Training
```bash
# BEST CONFIG (Phase 84) - VALIDATED AT 20K: 64.3% WR, +$48.33 P&L
SMART_ENTRY_GATE=1 SMART_COOLDOWN_MINUTES=30 python scripts/train_time_travel.py

# Key insight: 30-min cooldown finds more opportunities than 60-min default
# Results: 64.3% WR, 56 trades, +$0.86/trade (vs 47.5% WR, 40 trades with 60-min)

# With options
MODEL_RUN_DIR=models/my_test TT_MAX_CYCLES=20000 python scripts/train_time_travel.py
```

### Live Trading
```bash
./run_live_skip_monday.sh                    # Paper trading
touch go_live.flag && ./run_live_skip_monday.sh  # LIVE trading
```

### Dashboards
```bash
python dashboard.py              # All dashboards (5000, 5001, 5002)
python dashboard.py --status     # Check status
```

## Key Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MUSICAL_TRADING` | `0` | **Phase 61b FAILED**: 54.5% at 5K → 36.0% at 20K (no edge) |
| `MUSICAL_EAR` | `0` | Feedback learning from outcomes |
| `MUSICAL_KEY` | `0` | Regime-aware trading (bull/bear/sideways) |
| `MUSICAL_DYNAMICS` | `0` | Kelly position sizing |
| `MUSICAL_RESOLUTION` | `0` | Musical phrase exits |
| `DISABLE_CONFIDENCE_GATE` | `0` | **Phase 61c**: Bypass broken confidence checks |
| `REALISTIC_FILLS` | `0` | **Phase 61c**: Harsher paper fills (2x spread/slippage) |
| `CALL_ONLY` | `0` | **Phase 62**: CALLs only (PUTs lose 4x more) |
| `FAST_LOSS_CUT_ENABLED` | `1` | Set to 0 to disable RL fast cuts (0% WR) |
| `TT_MAX_HOLD_MINUTES` | `30` | Set to 60 for better outcomes (58.8% WR) |
| `SMART_ENTRY_GATE` | `0` | Use inverted confidence filtering |
| `SMART_COOLDOWN_MINUTES` | `60` | **Phase 84**: Use 30 for best results |
| `TRAIN_MAX_CONF` | `1.0` | Max confidence to trade (0.25 = only low conf) |
| `SKIP_MONDAY` | `0` | Skip Monday trades |
| `TEMPORAL_ENCODER` | `tcn` | Options: tcn, transformer, mamba2, lstm |
| `SKIP_HMM_ALIGNMENT` | `0` | Skip HMM alignment gate |
| `TT_MAX_CYCLES` | `5000` | Backtest cycles (use 20000 for validation) |
| `MODEL_RUN_DIR` | `models/run_*` | Output directory |

## Architecture

### Data Flow
```
Market Data → Features (50-500 dims) → HMM Regime → Predictor → Entry Policy → Exit Policy → Execution
```

### Key Files
| Component | File |
|-----------|------|
| Main Orchestrator | `core/unified_options_trading_bot.py` |
| Neural Predictor | `bot_modules/neural_networks.py` |
| HMM Regime | `backend/multi_dimensional_hmm.py` |
| Entry Policy | `backend/unified_rl_policy.py` |
| Exit Manager | `backend/unified_exit_manager.py` |
| Paper Trading | `backend/paper_trading_system.py` |
| **Musical Trading** | `backend/musical_trading.py` |
| Smart Entry Gate | `backend/smart_entry_gate.py` |
| Multi-Strategy | `backend/multi_strategy_options.py` |
| **Data Engine** | `backend/data_engine.py` |

### Directory Structure
| Directory | Purpose |
|-----------|---------|
| `core/` | Core bot and dashboard |
| `backend/` | Trading infrastructure |
| `bot_modules/` | Neural networks, features |
| `scripts/` | Training scripts |
| `features/` | Feature pipeline |
| `integrations/quantor/` | Jerry's bot components |

## Entry Controllers

| Type | Description | Config |
|------|-------------|--------|
| `bandit` | HMM-only with strict thresholds | **Default** |
| `rl` | Neural network PPO policy | After collecting data |
| `q_scorer` | Offline Q-regression | Experimental |
| `consensus` | Multi-signal agreement | High-confidence only |

## Exit Rules (Always Active)

| Rule | Default | Description |
|------|---------|-------------|
| Stop Loss | -8% | Hard exit on loss |
| Take Profit | +12% | Hard exit on gain |
| Trailing Stop | +4% activation, 2% trail | Lock profits |
| Max Hold | 45 min | Prevent theta decay |

## Neural Network

**UnifiedOptionsPredictor:**
- Inputs: Features [B, D] + Sequence [B, 60, D]
- Architecture: RBF Kernel → TCN/LSTM → Bayesian Heads
- Outputs: return_mean, return_std, direction_probs, confidence
- **Always use `risk_adjusted_return`, not raw return_mean**

**Actions:** 0=HOLD, 1=BUY_CALL, 2=BUY_PUT, 3=EXIT

## Multi-Strategy Options (NEW)

Supports multiple options strategies beyond single-leg calls/puts:

```python
from backend.multi_strategy_options import create_multi_strategy_system, StrategyType

executor, adapter = create_multi_strategy_system()

# Available strategies
StrategyType.BUY_CALL, StrategyType.BUY_PUT
StrategyType.IRON_CONDOR, StrategyType.STRADDLE, StrategyType.STRANGLE
StrategyType.BULL_PUT_SPREAD, StrategyType.BEAR_CALL_SPREAD
```

## Data Sources

| Source | Priority | Notes |
|--------|----------|-------|
| Data Manager | 0 | Uses `server_config.json` |
| Tradier | 1 | Live trading API |
| Polygon | 2 | Historical 1-min bars |

## Data Engine (NEW - Jerry Integration)

Inspired by Jerry's spy-iron-condor-trading data architecture:

```python
from backend.data_engine import create_data_engine, MTFSyncEngine

# Create engine with IV confidence decay
engine = create_data_engine(preload=True)

# Get snapshot with alignment metadata
snapshot = engine.get_snapshot('SPY', pd.Timestamp.now())
print(f"IV Confidence: {snapshot.alignment.iv_conf:.1%}")
print(f"Lag: {snapshot.alignment.lag_sec:.0f}s")
print(f"Mode: {snapshot.alignment.mode.value}")

# Multi-timeframe consensus
mtf = MTFSyncEngine(engine)
consensus = mtf.get_mtf_consensus('SPY', timestamp, 'rsi_14')
```

**IV Confidence Decay Formula:**
```
iv_conf = 0.5^(lag_sec / half_life)
```
- 0 sec lag → 100% confidence
- 5 min lag → 50% confidence (default half_life=300)
- 10 min lag → 25% confidence

**Alignment Modes:**
| Mode | Description |
|------|-------------|
| EXACT | Timestamp match within 1 min |
| PRIOR | Used recent prior data (<10 min) |
| STALE | Data is old (>10 min) |
| NONE | No data available |

**New Features Added to Pipeline:**
- `dq_agg_confidence` - Aggregate data quality [0-1]
- `dq_spy_conf`, `dq_vix_conf` - Per-symbol confidence
- `mtf_rsi_alignment` - Multi-timeframe RSI consensus [0-1]
- `mtf_macd_signal` - MTF MACD signal (-1, 0, 1)

## Databases

SQLite in `data/`:
- `paper_trading.db` - Trades linked by `run_id`
- `historical.db` - Historical market data
- `experiments.db` - Experiment tracking

## Common Gotchas

1. **Confidence is inverted** - Use `CONFIDENCE_CALIBRATION=1` (Phase 60, 60% WR) or SmartEntryGate
2. **Frozen predictor** - Weights don't update during RL training
3. **Use risk_adjusted_return** - Not raw return_mean
4. **HMM alignment** - Skip with SKIP_HMM_ALIGNMENT=1
5. **config.json is gitignored** - Contains API keys
6. **Drift gate too aggressive** - Use `SIMONS_DRIFT_GATE_THRESHOLD=100` to disable

## Documentation

| Document | Content |
|----------|---------|
| `docs/SYSTEM_ARCHITECTURE.md` | Complete system overview |
| `docs/CONFIDENCE_HEAD_ANALYSIS.md` | Confidence calibration details |
| `docs/ARCHITECTURE_IMPROVEMENTS_V4.md` | V4 architecture docs |
| `RESULTS_TRACKER.md` | Experiment results |

## Quantor-MTFuzz Integration

From Jerry Mahabub & John Draper's [spy-iron-condor-trading](https://github.com/trextrader/spy-iron-condor-trading):

| Component | File |
|-----------|------|
| Fuzzy Sizer | `integrations/quantor/fuzzy_sizer.py` |
| Regime Filter | `integrations/quantor/regime_filter.py` |
| Data Alignment | `integrations/quantor/data_alignment.py` |

## SmartEntryGate (Phase 84 - BEST)

Uses **inverted** confidence since confidence head is broken:

```python
# Key insight: ml_confidence is INVERTED - lower = better!
SMART_ENTRY_GATE=1  # Enable
SMART_COOLDOWN_MINUTES=30  # (Phase 84) Shorter cooldown = more opportunities
SMART_MIN_INV_CONF=0.70  # Min inverted conf (original < 30%)
SMART_MIN_PRED_RET=0.0002  # Min predicted return
SMART_MIN_VOLUME=0.9  # Min volume spike
SMART_MAX_DAILY=4  # Max trades per day
SMART_OPTION_TYPE=all  # 'all', 'puts_only', 'calls_only'
```

**Phase 84 Finding:** 30-min cooldown beats 60-min (64.3% WR vs 47.5% WR, +4.8x P&L)

## Time-of-Day Patterns (Phase 72 Discovery)

**Critical Finding:** Morning trades are 7x more profitable than afternoon trades!

| Time | Win Rate | Avg Win | Avg Loss | Net/Trade |
|------|----------|---------|----------|-----------|
| 9-10am | 58-67% | $2.48 | -$2.12 | +$0.91 |
| 10-12pm | 61-67% | $2.30 | -$1.95 | +$0.53 |
| 1-3pm | 25-50% | $0.35 | -$2.80 | -$1.50 |

**Afternoon Volume Pattern:**
- Low volume (< 1.5): **25% WR** (terrible)
- Medium volume (1.5-2.5): **100% WR** (excellent)
- High volume (> 2.5): **50% WR** (mediocre)

**SmartEntryGate auto-filters afternoon low-volume trades:**
```python
# In SmartEntryGate - already implemented
if current_time.hour in (13, 14, 15):
    if vol < 1.5 or vol > 2.5:
        return False, "Afternoon needs medium vol"
```

## Confidence Calibrator (Phase 60 - DID NOT HOLD UP)

**WARNING:** The 60% WR at 500 cycles was SAMPLING VARIANCE. At 5K cycles: **46.9% WR** (same as baseline).

```bash
# Does NOT improve win rate at scale
CONFIDENCE_CALIBRATION=1 python scripts/train_time_travel.py
```

**How it works:**
```
Raw Confidence  → Calibrated Confidence
19%             → 81%  (inverted: 1.0 - 0.19)
25%             → 75%
27%             → 73%
```

**Environment Variables:**
| Variable | Default | Description |
|----------|---------|-------------|
| `CONFIDENCE_CALIBRATION` | `0` | Enable calibration (inverts confidence) |
| `CONFIDENCE_FEEDBACK` | `0` | Learn from P&L outcomes (no effect) |
| `CONFIDENCE_CONSISTENCY` | `0` | Check direction agreement (no effect) |
| `CONFIDENCE_MEMORY` | `0` | Recent accuracy scaling (no effect) |
| `CONF_CAL_WINDOW` | `50` | Outcome window size |
| `CONF_CAL_MIN_SAMPLES` | `20` | Min samples before using actual WR |

**Key Finding:** Only CALIBRATION matters. Other verses have no effect.

## Jerry Integration (spy-iron-condor-trading)

Key features from Jerry's Quantor-MTFuzz system to integrate:

### Priority 1: Quick Wins
1. **VRP Gate** - Only trade when IV > Realized Vol by >2%
2. **Skew-Aware Strikes** - Penalize high put-skew trades
3. **Regime Filter** - Already in `integrations/quantor/regime_filter.py`

### Priority 2: Position Sizing
4. **Fuzzy 10-Factor Sizing** - Adaptive sizing based on multiple signals
5. **Portfolio Greeks Limits** - Hard delta/vega limits

### Priority 3: Advanced
6. **MTF Consensus** - Cross-timeframe alignment
7. **Mamba 2 Neural** - State-space model instead of LSTM

### Jerry's Mamba 2 Architecture (Unique)
Unlike standard Mamba, Jerry's implementation:
- **32 layers deep** (vs typical 4-8)
- **5 simple features**: `[log_ret, RSI, ATR%, vol_ratio, norm_time]`
- **Pads to 1024 dims** - Large embedding space
- **Batch precompute** - Runs entire dataset at once for GPU efficiency
- **CPU fallback** - MockMambaKernel for no-GPU environments
- **Return → Probs** - Converts raw output to bull/bear/neutral probabilities

**Files to copy from Jerry's repo:**
| Source | Destination | Purpose |
|--------|-------------|---------|
| `intelligence/fuzzy_engine.py` | `integrations/quantor/` | Fuzzy sizing |
| `core/risk_manager.py` | `backend/` | Greeks limits |
| `analytics/realized_vol.py` | `features/` | VRP calculation |

## Feature Dimension Configuration

Time-of-day features (50-53) are available but disabled by default:

```bash
# Enable time features for neural network
INCLUDE_TIME_FEATURES=1 python scripts/train_time_travel.py

# Enable gaussian pattern features (54-58)
INCLUDE_GAUSSIAN_FEATURES=1 python scripts/train_time_travel.py
```

**Note:** Adding time features without pretraining hurts performance.
Use SmartEntryGate's hard-coded afternoon filter instead.
