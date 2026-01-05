# Best Trading Configurations

**Last Updated**: 2026-01-05

This document tracks the best validated configurations for the Gaussian Options Trading Bot.

---

## Current Best Results (Post Bug-Fix)

| Rank | Run Name | P&L% | WR% | Trades | $/Trade |
|------|----------|------|-----|--------|---------|
| 1 | EXP-0067_IDEA-117 | +490.81% | 33.3% | 965 | $25.43 |
| 2 | EXP-NEW_236 | +468.49% | 44.2% | 79 | $296.51 |
| 3 | v3_jerry_features | +450.63% | 27.8% | 269 | $83.76 |
| 4 | v4_skew_partial_20k | +430.98% | 32.3% | 586 | $36.77 |
| 5 | EXP-NEW_245 | +429.86% | 46.9% | 98 | $219.32 |
| 6 | test_jerry_improvements | +429.49% | 33.1% | 76 | $282.56 |
| 7 | dec_validation_v2 | +413.41% | 59.8% | 61 | $338.86 |
| 8 | EXP-0168_IDEA-267 | +137.28% | 56.7% | 47 | $146.04 |
| 9 | transformer_baseline_jan2 | +35.44% | 47.0% | 105 | $16.88 |
| 10 | EXP-0172_IDEA-271 (Mamba2) | +34.85% | 39.8% | 79 | $22.06 |

**Note:** Results filtered to P&L < 500% to exclude pre-bugfix runs.

---

## IMPORTANT: Bug Fix Notice

A P&L calculation bug was fixed on 2026-01-01. **All results before this date showing massive gains (+1327%, +284,618%, etc.) are INVALID.** Only trust results from runs after the fix.

---

## Validated Best Configuration

### Transformer Encoder (+35.44% P&L)

**Status**: VALIDATED - Best performing post-bug-fix

```bash
TEMPORAL_ENCODER=transformer \
MODEL_RUN_DIR=models/transformer_test \
TT_MAX_CYCLES=5000 \
TT_PRINT_EVERY=500 \
PAPER_TRADING=True \
python scripts/train_time_travel.py
```

**Key Settings**:
| Setting | Value | Notes |
|---------|-------|-------|
| Temporal Encoder | `transformer` | 2-layer causal transformer with RoPE |
| Entry Controller | `bandit` | HMM-only with strict thresholds |
| Stop Loss | 8% | Hard exit on loss |
| Take Profit | 12% | Hard exit on gain |
| Max Hold | 45 min | Prevent theta decay |
| PnL Calibration | Enabled | Gates on P(profit) >= 42% |

**Results**:
- OOS P&L: +35.44%
- Win Rate: 47%
- Per-Trade P&L: $16.88
- Validated on 5K cycles

---

## Experimental Configurations

### TCN + Skew Exits (+431% in 20K validation)

**Status**: EXPERIMENTAL - Observed +431% but "guesses wildly"

```bash
TEMPORAL_ENCODER=tcn \
SKEW_EXIT_ENABLED=1 \
SKEW_EXIT_MODE=partial \
MODEL_RUN_DIR=models/tcn_skew_test \
TT_MAX_CYCLES=20000 \
TT_PRINT_EVERY=2000 \
PAPER_TRADING=True \
python scripts/train_time_travel.py
```

**Skew Exit Settings**:
| Setting | Value | Description |
|---------|-------|-------------|
| SKEW_EXIT_ENABLED | 1 | Enable skew-optimized exits |
| SKEW_EXIT_MODE | partial | Take partial profit, let runner ride |

**Observation**: High P&L but entry decisions appear random (~47% win rate). Good candidates for improvement with filtering.

---

### Combo DOW (Day of Week Filter)

**Status**: EXPERIMENTAL - Server showed 65% win rate, local showed 44%

```bash
HARD_STOP_LOSS_PCT=50 \
HARD_TAKE_PROFIT_PCT=10 \
TRAIN_MAX_CONF=0.25 \
DAY_OF_WEEK_FILTER=1 \
SKIP_MONDAY=1 \
SKIP_FRIDAY=1 \
MODEL_RUN_DIR=models/combo_dow_test \
TT_MAX_CYCLES=5000 \
PAPER_TRADING=True \
python scripts/train_time_travel.py
```

**Observations**:
- Very wide stops (50%) with quick TP (10%)
- Skips Monday and Friday (earnings/weekend risk)
- Results vary by time period

---

## Architecture Components

### Neural Network State Features (21 total)

The unified RL policy now uses 21 state features:

| # | Feature | Range | Description |
|---|---------|-------|-------------|
| 1-4 | Position State | - | in_trade, is_call, pnl%, drawdown |
| 5-6 | Time | - | minutes_held, minutes_to_expiry |
| 7-9 | Prediction | - | direction, confidence, momentum_5m |
| 10-11 | Market | - | vix, volume_spike |
| 12-15 | HMM Regime | 0-1 | trend, volatility, liquidity, hmm_conf |
| 16-17 | Greeks | - | theta, delta |
| 18 | Position Size | 0-1 | Normalized position size |
| **19-21** | **Condor Regime** | **0-1** | **NEW: condor_suitability, mtf_consensus, trending_count** |

### Condor Regime Features (NEW)

Uses Iron Condor entry logic **inversely** as weights:

| Feature | Range | Description |
|---------|-------|-------------|
| condor_suitability | 0-1 | 0 = trending (good for directional), 1 = neutral (condor territory) |
| mtf_consensus | 0-1 | Multi-timeframe consensus (0 = bearish, 0.5 = neutral, 1 = bullish) |
| trending_signal_count | 0-1 | Normalized count of trending indicators |

**Concept**: When Iron Condor would enter (neutral market), directional trades are less likely to succeed.

---

## Entry Controllers

| Type | Command | Best For |
|------|---------|----------|
| bandit | `ENTRY_CONTROLLER=bandit` | Default - HMM-only, proven |
| rl | `ENTRY_CONTROLLER=rl` | After collecting training data |
| q_scorer | `ENTRY_CONTROLLER=q_scorer` | Experimental offline Q-learning |
| consensus | `ENTRY_CONTROLLER=consensus` | Multi-signal agreement |

---

## Exit Rules (Always Active)

| Rule | Threshold | Env Override |
|------|-----------|--------------|
| Stop Loss | 8% | `HARD_STOP_LOSS_PCT=8` |
| Take Profit | 12% | `HARD_TAKE_PROFIT_PCT=12` |
| Trailing Stop | +4% activation, 2% trail | Config-based |
| Max Hold | 45 min | `TT_MAX_HOLD_MINUTES=45` |
| Expiry | <30 min | Hard-coded |

---

## Run Commands

### Quick Test (5K cycles)
```bash
python scripts/train_time_travel.py
```

### Production Validation (20K cycles)
```bash
TT_MAX_CYCLES=20000 TT_PRINT_EVERY=2000 python scripts/train_time_travel.py
```

### Fresh Model (reset checkpoints)
```bash
rm -rf models/my_test
MODEL_RUN_DIR=models/my_test python scripts/train_time_travel.py
```

---

## Key Findings

1. **Win rate is a red herring** - 40% win rate can make money with proper position sizing
2. **Edge comes from skew** - Fat-tail winners matter more than high win rate
3. **Transformer > TCN** - Validated +35.44% vs TCN's period-sensitive results
4. **Exit management > Entry selection** - Cutting losses fast is more important than picking winners
5. **Condor regime = inverse filter** - Iron Condor logic can filter out choppy markets

---

## Files Reference

| File | Purpose |
|------|---------|
| `backend/unified_rl_policy.py` | Entry/exit policy with 21 state features |
| `backend/condor_regime_filter.py` | Iron Condor logic as regime filter |
| `backend/skew_exit_manager.py` | Partial TP + trailing runner |
| `scripts/train_time_travel.py` | Main training script |
| `run_condor_features_test.bat` | Test TCN + Skew + Condor features |
