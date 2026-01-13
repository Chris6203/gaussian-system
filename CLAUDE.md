# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

**Gaussian Options Trading Bot** - an algorithmic SPY options trading system combining:
- Bayesian neural networks with Gaussian kernel processors
- Multi-dimensional HMM (3×3×3 = 27 states) for regime detection
- Multiple entry controllers: bandit (default), RL (PPO), Q-Scorer, consensus
- Paper trading and live execution via Tradier API

**Best Config (Phase 57):** `SMART_ENTRY_GATE=1` - Uses inverted confidence filtering (+34.38% P&L)

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
- **Confidence Inversion (2026-01-12):** Confidence head outputs INVERTED. Fix: `CONFIDENCE_CALIBRATION=1 SIMONS_DRIFT_GATE_THRESHOLD=100` → **60% WR, +$9.93 P&L**
- **P&L Bug (2026-01-01):** Results before this date showing massive gains are INVALID
- **TEMPORAL_ENCODER Bug (2026-01-06):** V2/V3 predictors ignored env var, always used LSTM
- **HMM Gate Bug (2026-01-07):** Was blocking good signals. Fix: `SKIP_HMM_ALIGNMENT=1`
- **Direction Threshold Bug:** `MIN_DIRECTION_THRESHOLD=0.5` is too high (use 0.05)

## Quick Start

### Training
```bash
# NEW BEST (Phase 60): Confidence Calibration - 60% WR, +$9.93 P&L
CONFIDENCE_CALIBRATION=1 SIMONS_DRIFT_GATE_THRESHOLD=100 python scripts/train_time_travel.py

# Best entry filter (Phase 57)
SMART_ENTRY_GATE=1 python scripts/train_time_travel.py

# Best overall (Phase 52: SKIP_MONDAY)
./run_live_skip_monday.sh

# With options
MODEL_RUN_DIR=models/my_test TT_MAX_CYCLES=5000 python scripts/train_time_travel.py
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
| `CONFIDENCE_CALIBRATION` | `0` | **Phase 60 FIX**: Invert broken confidence head (60% WR) |
| `SIMONS_DRIFT_GATE_THRESHOLD` | `50` | Drift gate % (100=disabled, recommended) |
| `SMART_ENTRY_GATE` | `0` | Use inverted confidence filtering |
| `TRAIN_MAX_CONF` | `1.0` | Max confidence to trade (0.25 = only low conf) |
| `SKIP_MONDAY` | `0` | Skip Monday trades |
| `TEMPORAL_ENCODER` | `tcn` | Options: tcn, transformer, mamba2, lstm |
| `SKIP_HMM_ALIGNMENT` | `0` | Skip HMM alignment gate |
| `SKEW_EXIT_ENABLED` | `0` | Enable skew-optimized exits |
| `TT_MAX_CYCLES` | `5000` | Backtest cycles |
| `MODEL_RUN_DIR` | `models/run_*` | Output directory |
| `MEAN_REVERSION_GATE` | `0` | Phase 58: Filter trades at RSI extremes only |
| `MR_ENTRY_THRESHOLD` | `0.5` | Min signal strength for mean reversion |
| `MR_AVOID_LUNCH` | `1` | Skip 11:00-14:00 (choppy hours) |

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
| Smart Entry Gate | `backend/smart_entry_gate.py` |
| Multi-Strategy | `backend/multi_strategy_options.py` |

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

## SmartEntryGate (Phase 57 - BEST)

Uses **inverted** confidence since confidence head is broken:

```python
# Key insight: ml_confidence is INVERTED - lower = better!
SMART_ENTRY_GATE=1  # Enable
SMART_MIN_INV_CONF=0.70  # Min inverted conf (original < 30%)
SMART_MIN_PRED_RET=0.0002  # Min predicted return
SMART_MIN_VOLUME=0.9  # Min volume spike
SMART_MAX_DAILY=4  # Max trades per day
SMART_OPTION_TYPE=all  # 'all', 'puts_only', 'calls_only'
```

**Key Finding:** PUTS outperform CALLS by 4x (-$0.15/trade vs -$0.62/trade)

## Confidence Calibrator (Phase 60 - BEST)

Fixes the broken confidence head by inverting raw confidence to calibrated confidence:

```bash
# Best config: 60% WR, +$9.93 P&L
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

## Potential Improvements (TODO)

Based on Phase 60 findings, these could further improve results:

1. **Longer Validation** - Validate 60% WR holds at 5K/20K cycles
2. **Combine with Phase 58** - Test CALIBRATION + SPY-VIX gate + Kelly sizing
3. **Direction-Specific Calibration** - Separate calibration for CALLS vs PUTS
4. **Regime-Specific Calibration** - Different inversion per market regime
5. **Retrain Confidence Head** - Fix the NN instead of inverting output
6. **Adaptive Calibration** - Learn actual WR per bucket over time
7. **Combine with SmartEntryGate** - Both use inverted confidence
