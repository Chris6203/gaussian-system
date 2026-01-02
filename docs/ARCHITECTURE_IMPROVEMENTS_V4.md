# Architecture Improvements V4 - Skew & EV Optimization

**Date:** 2026-01-02
**Key Finding:** Edge comes from skew (fat-tail winners), not win rate.

## Critical Data Analysis Results

### Confidence Root Cause (VERIFIED)
```
ML confidence -> is_win:       -0.0657 (ANTI-predictive!)
Consensus strength -> is_win:  -0.0043 (not predictive)
ECE (calibration error):        0.089 (>0.05 = badly miscalibrated)
```
**Conclusion:** Neither ML confidence nor consensus strength predicts winning.

### Skew Analysis (v3_calibrated, 1006 trades)
```
Win Rate:        38.3% (below 40% breakeven)
Avg Win:         +2.7%
Avg Loss:        -1.7%
95th %ile win:   +4.4%
Max win:         +584.7% (ONE TRADE)

Top 10 trades:   $1,567 (1387% of total P&L!)
Total P&L:       $113

Without top outlier: -$1,300
```
**Conclusion:** One +584% trade drove entire profitability. Fixed 12% TP would have capped it at ~$290.

### Best Experiment (EXP-0069)
```
Win Rate:  42.3% (above 40% breakeven)
P&L:       +813.8%
Trades:    291
```
**Key:** Conditional WR > 40% when traded.

## New Components Created

### 1. EV Gate (`backend/ev_gate.py`)

Gates trades on positive expected value after costs.

**Bayesian Prior Blend (not hard floor):**
```python
P(win)_posterior = w * P(win)_prior + (1-w) * P(win)_model
```

**Environment Variables:**
| Variable | Default | Description |
|----------|---------|-------------|
| `EV_GATE_ENABLED` | `0` | Enable EV gate |
| `EV_MIN_RATIO` | `0.0` | Minimum EV/risk ratio |
| `EV_SPREAD_COST` | `0.002` | Spread cost (0.2%) |
| `EV_THETA_DECAY` | `0.003` | Theta decay/hour (0.3%) |
| `EV_WIN_PROB_PRIOR` | `0.40` | Prior P(win) base rate |
| `EV_PRIOR_WEIGHT` | `0.3` | Prior weight (0=model, 1=prior) |

**Usage:**
```python
from backend.ev_gate import EVGate
gate = EVGate()
result = gate.evaluate(signal, config)
if not result.passed:
    print(f"Blocked: {result.rejection_reason}")
```

### 2. Regime Calibration (`backend/regime_calibration.py`)

Per-regime calibrators for HMM regime + time-of-day.

**Regime Buckets:**
- Volatility: low (<0.4), normal (0.4-0.6), high (>0.6)
- Time: open (9:30-10:30), midday (10:30-14:30), close (14:30-16:00)

**Environment Variables:**
| Variable | Default | Description |
|----------|---------|-------------|
| `REGIME_CALIBRATION` | `1` | Enable regime calibration |

### 3. Regime Attribution (`backend/regime_attribution.py`)

Tracks performance per regime, auto-disables underperformers.

**Environment Variables:**
| Variable | Default | Description |
|----------|---------|-------------|
| `REGIME_AUTO_DISABLE` | `1` | Auto-disable bad regimes |
| `REGIME_DISABLE_PNL` | `-100` | Disable if PnL below this |
| `REGIME_DISABLE_WR` | `0.30` | Disable if WR below this |
| `REGIME_MIN_TRADES` | `20` | Min trades before decision |
| `REGIME_CONF_BOOST` | `0.10` | Raise threshold for marginal |

### 4. Greeks-Aware Exits (`backend/greeks_aware_exits.py`)

Dynamic stop/TP based on VIX, delta, theta, uncertainty.

**Adjustments:**
- High delta (>0.7): Tighter stops (0.8x)
- High VIX: Wider stops (up to 1.5x)
- Near expiry (<1hr): Tighter max hold (15min)
- High uncertainty: Wider stops, smaller position

**Environment Variables:**
| Variable | Default | Description |
|----------|---------|-------------|
| `GREEKS_AWARE_EXITS` | `1` | Enable Greeks-aware exits |
| `GREEKS_IV_FACTOR` | `0.5` | IV adjustment factor |
| `GREEKS_DELTA_FACTOR` | `0.3` | Delta adjustment factor |

### 5. Skew Exit Manager (`backend/skew_exit_manager.py`)

Captures fat-tail winners with partial TP + trailing runners.

**Exit Modes:**
| Mode | Behavior |
|------|----------|
| `fixed` | Standard 12%/-8% stops |
| `partial` | Take 50% at 10%, runner with trail |
| `trailing` | No fixed TP, trailing stop only |
| `trend_adaptive` | Trailing in trends, partial in chop |

**Environment Variables:**
| Variable | Default | Description |
|----------|---------|-------------|
| `SKEW_EXIT_ENABLED` | `0` | Enable skew exits |
| `SKEW_EXIT_MODE` | `partial` | Exit mode |
| `PARTIAL_TP_PCT` | `0.10` | Partial TP threshold (10%) |
| `PARTIAL_TAKE_FRACTION` | `0.50` | Take 50% at partial |
| `RUNNER_TRAIL_ACTIVATION` | `0.15` | Activate trail at 15% |
| `RUNNER_TRAIL_DISTANCE` | `0.05` | Trail 5% behind peak |

**Usage:**
```python
from backend.skew_exit_manager import get_skew_exit_manager, ExitMode
mgr = get_skew_exit_manager()
decision = mgr.evaluate_exit(
    trade_id="abc123",
    current_pnl_pct=0.12,
    peak_pnl_pct=0.15,
    hmm_trend=0.75,
    minutes_held=20,
)
if decision.should_exit:
    exit_fraction = decision.exit_fraction  # 0.5 for partial, 1.0 for full
```

## Recommended Configurations

### Conservative (Proven Baseline)
```bash
# v3_calibrated baseline - 274% P&L
PREDICTOR_ARCH=v3_multi_horizon
# No new gates (stick to what works)
```

### Skew-Optimized (Capture Fat Tails)
```bash
SKEW_EXIT_ENABLED=1
SKEW_EXIT_MODE=partial
PARTIAL_TP_PCT=0.10
PARTIAL_TAKE_FRACTION=0.50
RUNNER_TRAIL_ACTIVATION=0.15
RUNNER_TRAIL_DISTANCE=0.05
```

### Full Architecture V4
```bash
# EV Gate with Bayesian prior
EV_GATE_ENABLED=1
EV_WIN_PROB_PRIOR=0.40
EV_PRIOR_WEIGHT=0.3

# Regime tracking
REGIME_CALIBRATION=1
REGIME_AUTO_DISABLE=1

# Greeks-aware exits
GREEKS_AWARE_EXITS=1

# Skew exits
SKEW_EXIT_ENABLED=1
SKEW_EXIT_MODE=trend_adaptive
```

## Key Insights

1. **Don't trust confidence for entry selection** - it's anti-predictive
2. **Edge is in skew, not WR** - let winners run, don't cap at 12%
3. **Conditional WR matters** - only trade when regime/timing align
4. **One big winner can make the whole run** - capture it with trailing stops

## Files Created/Modified

| File | Purpose |
|------|---------|
| `backend/ev_gate.py` | EV gate with Bayesian prior |
| `backend/regime_calibration.py` | Per-regime calibrators |
| `backend/regime_attribution.py` | Regime tracking + auto-disable |
| `backend/greeks_aware_exits.py` | Greeks-aware dynamic exits |
| `backend/skew_exit_manager.py` | Partial TP + trailing runner |
| `backend/paper_trading_system.py` | Integrated Greeks-aware exits |

## Testing

Run component tests:
```bash
python tests/test_architecture_v4.py
```

Run skew exit experiment:
```bash
SKEW_EXIT_ENABLED=1 SKEW_EXIT_MODE=partial \
MODEL_RUN_DIR=models/skew_test \
TT_MAX_CYCLES=5000 \
python scripts/train_time_travel.py
```
