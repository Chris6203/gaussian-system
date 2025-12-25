# Critical Fixes Applied to Gaussian Options Bot

This document summarizes the 7 critical issues identified and the fixes applied.

---

## Issue 1: Objective / Horizon Misalignment ✅

### Problem
- Neural net trained on direction over 15-minute horizon
- But trading 0-1 DTE options held 45-120 minutes
- Direction prediction can be correct but option still loses money (IV crush, theta decay)
- Training/calibration target ≠ reward/execution target

### Fix: Dual Calibration System

**File: `backend/calibration_tracker.py`**

Added two calibrators:
1. **direction_calibrator** - Original behavior, calibrates P(direction correct at 15min)
2. **pnl_calibrator** - NEW: Calibrates P(option PnL > 0 at actual exit)

Key methods added:
- `record_trade_entry()` - Record trade for PnL tracking
- `record_pnl_outcome()` - Record actual option P&L when trade exits
- `calibrate_pnl()` - **PRIMARY GATE** for trade decisions

```python
# Usage in gating layer
pnl_calibrated = tracker.calibrate_pnl(raw_confidence=0.65)  # USE THIS
direction_calibrated = tracker.calibrate(raw_confidence=0.65)  # Supplementary
```

---

## Issue 2: Confidence / Threshold RL Feedback Loop ✅

### Problem
- Multiple overlapping thresholds: config, regime strategies, RL learner
- RL could adjust threshold ±0.2, causing wild swings
- Feedback loop: RL raises threshold → fewer trades → calibration noisy → worse gating

### Fix: Centralized Threshold Management

**File: `config.json`**

Added `threshold_management` section:
```json
"threshold_management": {
    "rl_threshold_delta_max": 0.03,
    "effective_threshold_floor": 0.50,
    "effective_threshold_ceiling": 0.70,
    "regime_multipliers": {
        "ultra_low_vol": 0.82,
        "normal_vol": 1.00,
        "extreme_vol": 1.36
    }
}
```

**File: `backend/rl_threshold_learner.py`**

Formula: `effective_threshold = clamp(base * regime_mult + rl_delta, floor, ceiling)`
- RL delta limited to ±3% (was ±20%)
- Hard floor/ceiling at 50%/70%

Key changes:
- `get_effective_threshold(regime_multiplier)` - Single canonical threshold derivation
- All threshold adjustments now modify `rl_delta`, not raw threshold
- Warnings logged when hitting bounds

---

## Issue 3: HMM Regime Mapping vs Strategy Table ✅

### Problem
- HMM can discover 2-7 states per dimension via BIC
- regime_strategies.py assumes fixed VIX buckets
- States can "rotate" when HMM retrains, silently breaking mapping

### Fix: Regime Mapper Module

**New file: `backend/regime_mapper.py`**

Creates canonical regime mapping layer:
- Input: HMM states + VIX level
- Output: Canonical regime (ULTRA_LOW_VOL, NORMAL_VOL, etc.)

Key features:
- VIX takes precedence for structural risk controls
- HMM provides fine-tuning within VIX bucket
- Sanity checks: HMM says "low vol" but VIX > 35 → override to EXTREME_VOL
- Structure drift detection: warns when HMM state count changes

```python
from backend.regime_mapper import RegimeMapper, get_canonical_regime

mapper = RegimeMapper()
regime = mapper.map_regime(
    vix_level=18.5,
    hmm_trend_state=0,
    hmm_vol_state=1
)
print(regime.name)  # "NORMAL_VOL"
print(regime.confidence_multiplier)  # 1.0
```

---

## Issue 4: Cold Start & Data Sparsity ✅

### Problem
- Multiple gates require "enough data" (calibration ≥50, HMM training, RL buffers)
- At startup: bot may not trade at all
- Strict requirements → 0 trades → never learn

### Fix: Cold Start Mode

**File: `config.json`**

Added `cold_start` section:
```json
"cold_start": {
    "enabled": true,
    "min_calibration_samples": 50,
    "min_calibration_samples_for_full_size": 200,
    "cold_start_confidence_threshold": 0.60,
    "cold_start_position_scale": 0.25
}
```

**File: `backend/risk_manager.py`**

Cold start logic:
- samples < 50: Conservative threshold (0.60), 25% position size
- samples 50-200: Gradual ramp to full size
- samples ≥ 200: Full size enabled

---

## Issue 5: 0-1 DTE Options vs Hold Time / Close Constraints ✅

### Problem
- Trading 0-1 DTE with max_hold = 120 minutes
- Midday 0DTE can hit closing bell before natural exit
- RL exit policy evaluated on truncated trajectories

### Fix: Time-to-Close Constraints

**File: `config.json`**

Added time constraints:
```json
"options": {
    "time_to_close_constraints": {
        "safety_margin_minutes": 20,
        "min_time_to_close_for_0dte_entry": 90,
        "effective_max_hold_formula": "min(config_max_hold, minutes_to_close - safety_margin)"
    }
}
```

**File: `backend/risk_manager.py`**

```python
def get_effective_max_hold(self, config_max_hold, minutes_to_close, safety_margin=20):
    return min(config_max_hold, max(0, minutes_to_close - safety_margin))
```

Blocks 0DTE entries with < 90 minutes to close.

---

## Issue 6: Risk Limits: Inconsistent Percentages ✅

### Problem
- Risk limits scattered across config, order execution, portfolio management
- Easy to update one place and forget others
- Per-trade: 2% risk, 5% position size (multiple implementations)

### Fix: Centralized Risk Manager

**File: `config.json`**

Single source of truth:
```json
"risk": {
    "max_risk_per_trade_pct": 0.02,
    "max_position_size_pct": 0.05,
    "max_daily_loss_pct_stop": 0.10,
    "max_weekly_loss_pct_stop": 0.15,
    "max_concurrent_positions": 5,
    "position_size_confidence_clamp": {"min": 0.5, "max": 0.7}
}
```

**New file: `backend/risk_manager.py`**

Single module for all risk calculations:
- `check_trade_allowed()` - Pre-trade risk checks
- `calculate_position_size()` - Position sizing with confidence clamping
- `get_effective_max_hold()` - Time-constrained max hold

Confidence clamping prevents over-concentration:
```python
# Clamps confidence contribution to 0.83x - 1.17x
# Instead of raw 0.5x - 2.0x from high/low confidence
clamped_conf = clamp(confidence, 0.5, 0.7)
confidence_adj = clamped_conf / 0.6
```

---

## Issue 7: Simulation vs Live: Exact Parity ✅

### Problem
- Separate code paths for sim vs live
- Sim might bypass gating, liquidity checks, risk limits
- Different P&L formulas, missing spread/commission
- Rosier backtest than live performance

### Fix: Unified Trading Environment

**New file: `backend/trading_environment.py`**

RL Gym-like interface:
```python
env = TradingEnvironment(broker=SimulatedBroker(...), mode='simulation')

obs = env.reset()
while not done:
    action = model.predict(obs)
    obs, reward, done, truncated, info = env.step(action)
```

Same code path for sim and live:
- Same gating checks
- Same risk limits
- Same P&L formulas

**ExecutionSimulator** ensures realistic fills:
- Fill at worse side of spread (buy at ask, sell at bid)
- Additional slippage for low volume
- Commission simulation

---

## Files Modified

| File | Changes |
|------|---------|
| `config.json` | Added `risk`, `cold_start`, `threshold_management`, updated `options` |
| `backend/calibration_tracker.py` | Added PnL calibrator |
| `backend/rl_threshold_learner.py` | Centralized threshold derivation |
| `backend/regime_mapper.py` | **NEW** - Canonical regime mapping |
| `backend/risk_manager.py` | **NEW** - Centralized risk management |
| `backend/trading_environment.py` | **NEW** - Unified sim/live environment |
| `backend/integration.py` | Added integration for new components |
| `backend/live_trading_engine.py` | Added PnL entry/outcome recording for calibration |
| `backend/trading_environment.py` | Added training mode with negative balance support |

## Training Mode Enhancement

**Problem**: Training would stop when balance hit zero, preventing the model from learning from losing streaks and drawdowns.

**Solution**: Added `allow_negative_balance` training mode:

```json
"training": {
    "allow_negative_balance": true,
    "auto_replenish_on_negative": true,
    "replenish_threshold": 500.0,
    "track_true_pnl": true
}
```

**Key features**:
- Account auto-replenishes when balance gets too low during training
- **True P&L is tracked separately** so performance metrics remain accurate
- Model learns from entire market conditions including drawdowns
- `get_training_stats()` returns both display balance and true P&L

---

## Integration Notes

### To use the new calibration:
```python
from backend.calibration_tracker import CalibrationTracker

tracker = CalibrationTracker(direction_horizon=15, pnl_horizon=60)

# Record entries
tracker.record_trade_entry(trade_id, confidence, direction, entry_price)

# When trade closes
tracker.record_pnl_outcome(trade_id, actual_pnl)

# For gating (use PnL-calibrated!)
should_trade = tracker.calibrate_pnl(raw_confidence) >= threshold
```

### To use centralized risk:
```python
from backend.risk_manager import RiskManager

rm = RiskManager(config)

# Pre-trade check
allowed, result, reason = rm.check_trade_allowed(balance, positions, daily_pnl)

# Position sizing
assessment = rm.calculate_position_size(balance, option_price, confidence, calibration_samples)
contracts = assessment.contracts
```

### To use regime mapper:
```python
from backend.regime_mapper import get_canonical_regime

regime = get_canonical_regime(vix_level=18.5, hmm_regime_info=hmm_output)
threshold_mult = regime.confidence_multiplier
position_scale = regime.position_scale
```

### To use unified environment:
```python
from backend.trading_environment import create_trading_environment

env = create_trading_environment(mode='simulation', initial_balance=5000)
obs = env.reset()

# Same interface for simulation AND live
result = env.step({'action': 'BUY_CALLS', 'confidence': 0.65, ...})
```

