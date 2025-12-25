# XGBoost Exit Policy Analysis

**Date:** 2025-12-15
**Issue:** XGBoost exiting trades at 0.1%-1% gains with 99%+ confidence

---

## Executive Summary

**Root Cause Identified:** The XGBoost exit policy's labeling strategy creates a systematic bias toward premature exits at tiny profits.

**Labeling Logic (lines 369-384 in `xgboost_exit_policy.py`):**
```python
# Should we have exited at this point?
# Exit was right if:
# 1. P&L was positive and final P&L was worse (missed profit)
if exp.pnl_pct > 0 and exit_pnl_pct < exp.pnl_pct:
    # Was profitable, ended up worse -> should have exited
    exp.should_have_exited = 1
```

**Problem:** This labels **ANY** positive P&L that later decreases as "should have exited", even if the decrease is from +0.5% to +0.4%.

**Result:**
- Model learns that small profits rarely grow into large profits
- XGBoost exits at 99.5% confidence when P&L = +0.1% to +1.0%
- Average hold time: 1-10 minutes (far below Q-scorer's 15-minute training horizon)

---

## Observed Behavior from Optimization Logs

From the optimization run output (process 3bcf92):

```
[EXIT #1] Trade: 0520f533, Reason: XGB EXIT: 99.5% prob (P&L: +0.6%), Time Held: 2.0 min
[EXIT #2] Trade: 16fab8f8, Reason: XGB EXIT: 99.5% prob (P&L: +1.0%), Time Held: 1.0 min
[EXIT #3] Trade: f08add39, Reason: XGB EXIT: 99.5% prob (P&L: +0.0%), Time Held: 2.0 min
[EXIT #4] Trade: e2ff2162, Reason: XGB EXIT: 99.5% prob (P&L: +0.1%), Time Held: 5.0 min
[EXIT #5] Trade: b81ce2ae, Reason: XGB EXIT: 99.5% prob (P&L: +0.8%), Time Held: 2.0 min
```

**Pattern:** XGBoost is **extremely confident** (99.5%) that trades should exit at minuscule profits.

---

## Why This Happens: Label Leakage

### Current Labeling Strategy

The code labels exit decisions **retrospectively** after a trade closes:

1. **During Trade:** Store snapshots of state at each check (pnl_pct, time_held, etc.)
2. **After Trade Closes:** Look back at all snapshots and label them:
   - **Label = 1 (exit)** if: `snapshot_pnl > final_pnl` (profit got worse)
   - **Label = 0 (hold)** otherwise

### The Feedback Loop

In a **losing environment** (which we have: 0-1.6% win rates):

1. Most trades eventually lose money or close with small profits
2. **Any** intermediate positive P&L becomes labeled as "should have exited"
3. Model learns: "If profitable → exit immediately"
4. This creates a **self-fulfilling prophecy**:
   - Exits at +0.5% before it can reach +5%
   - Never allows winners to develop
   - Confirms the pattern that small profits don't grow

### Training Data from Existing Model

The loaded XGBoost model (`models/xgboost_exit.pkl`, 86MB) was trained on **193,871 experiences** with **99.4% accuracy**.

**High Accuracy Paradox:** The 99.4% accuracy means the model is very good at predicting the training labels—but the labels themselves encode premature exit behavior.

---

## Impact on Q-Scorer Training

### The Mismatch

**Q-Scorer Training:**
- **Horizon:** 15 minutes
- **Labels:** Ghost rewards assume trade held for 15 minutes
- **Objective:** Predict expected P&L at 15-minute mark

**XGBoost Exits:**
- **Actual Hold Time:** 1-10 minutes (median ~2 minutes)
- **Exit Trigger:** +0.1% to +1.0% gains
- **Objective:** Exit as soon as any profit appears

**Result:** Q-scorer is trained to find 15-minute opportunities, but XGBoost closes them after 2 minutes at +0.5%.

---

## Solution: Multi-Part Fix

### Part 1: Improve XGBoost Labeling Strategy

**Current (Flawed):**
```python
if exp.pnl_pct > 0 and exit_pnl_pct < exp.pnl_pct:
    exp.should_have_exited = 1  # Label as exit
```

**Proposed (Better):**
```python
# Only label as "should exit" if the difference is significant
profit_threshold = 2.0  # Only call it a missed exit if lost 2%+ of profit
time_threshold = prediction_timeframe * 0.75  # Don't exit before 75% of prediction window

if exp.pnl_pct > 0 and time_held >= time_threshold:
    profit_loss = exp.pnl_pct - exit_pnl_pct
    if profit_loss > profit_threshold:
        # Lost significant profit AND held long enough
        exp.should_have_exited = 1
    elif exp.pnl_pct > 8.0 and profit_loss > 1.0:
        # Had large profit (+8%), lost any significant amount → exit
        exp.should_have_exited = 1
    else:
        # Small profit fluctuations are normal, don't exit
        exp.should_have_exited = 0
elif exp.pnl_pct < -5 and exit_pnl_pct < exp.pnl_pct - 2.0:
    # Big loss that got bigger → should have cut loss
    exp.should_have_exited = 1
else:
    exp.should_have_exited = 0
```

**Key Changes:**
1. **Time gate:** Don't exit before reaching 75% of prediction timeframe (11 minutes for 15-min predictions)
2. **Profit threshold:** Require significant profit loss (2%+) before labeling as "missed exit"
3. **Large profit exception:** Allow exits on large profits (+8%) with smaller loss threshold (1%)
4. **Stop-loss focus:** Only label stop-loss on large losses (-5%+) that grow significantly

### Part 2: Align Ghost Rewards with Simulator

**Current Ghost Reward Calculation** (`backend/decision_pipeline.py::GhostTradeEvaluator`):
- Uses lightweight option pricing proxy (Delta × underlying move × 100)
- Assumes trade held for full horizon (e.g., 15 minutes)
- Simple friction/theta estimates

**Problem:** Doesn't match actual simulator behavior with:
- Black-Scholes option pricing
- XGBoost early exits
- Realized friction/slippage

**Solution: Hybrid Approach**

Instead of pure ghost rewards, use **simulator-aligned ghost rewards**:

```python
def evaluate_with_simulator(self, state, action, horizon_minutes):
    """
    Evaluate action by running a mini-simulation with actual simulator logic.

    Returns reward that matches what the simulator would actually produce.
    """
    # 1. Get option price at entry (Black-Scholes)
    entry_price = self._get_bs_option_price(
        underlying=state.spy_price,
        strike=self._select_strike(state.spy_price, action),
        dte=state.days_to_expiration,
        volatility=state.implied_vol,
        is_call=(action == 'BUY_CALLS')
    )

    # 2. Simulate price movement
    future_prices = self._get_future_prices(state.timestamp, horizon_minutes)

    # 3. Run XGBoost exit check at each minute
    for minute, future_price in enumerate(future_prices, 1):
        exit_price = self._get_bs_option_price(...)
        pnl_pct = (exit_price - entry_price) / entry_price * 100

        # Check if XGBoost would exit here
        should_exit = self._check_xgboost_exit(
            pnl_pct=pnl_pct,
            time_held=minute,
            prediction_timeframe=horizon_minutes
        )

        if should_exit:
            # Calculate net reward with actual friction
            gross_pnl = (exit_price - entry_price) * 100  # Per contract
            net_pnl = gross_pnl - self.friction_cost - (self.theta_per_minute * minute)
            return net_pnl

    # If didn't exit early, use final price at horizon
    final_price = self._get_bs_option_price(...)
    gross_pnl = (final_price - entry_price) * 100
    net_pnl = gross_pnl - self.friction_cost - (self.theta_per_minute * horizon_minutes)
    return net_pnl
```

**Benefits:**
- Labels match what simulator will actually produce
- Accounts for XGBoost early exits
- Uses same option pricing (Black-Scholes)
- Includes realistic friction

---

## Implementation Plan

### Step 1: Fix XGBoost Labeling (Immediate)

**File:** `backend/xgboost_exit_policy.py`
**Lines:** 369-384 (in `store_exit_experience` method)

**Changes:**
1. Add time threshold check (don't exit before 75% of prediction window)
2. Add profit loss threshold (require 2%+ loss before labeling as missed exit)
3. Special case for large profits (allow exits at +8% with 1% threshold)

**Testing:**
- Delete `models/xgboost_exit.pkl` to force retraining
- Run 500-cycle time-travel with fixed labeling
- Verify: Trades held longer (target: 10+ minutes average)
- Verify: Exits at larger P&L (target: +3% to +8% instead of +0.5%)

### Step 2: Align Ghost Rewards (Medium Priority)

**File:** `backend/decision_pipeline.py`
**Method:** `GhostTradeEvaluator.evaluate`

**Changes:**
1. Implement `_get_bs_option_price()` using Black-Scholes formula
2. Implement `_check_xgboost_exit()` to simulate exit decisions
3. Replace lightweight proxy with full simulation
4. Add configuration flag: `USE_SIMULATOR_ALIGNED_LABELS=True`

**Testing:**
- Generate new dataset (1500 cycles) with aligned labels
- Compare label distribution: old vs new
- Train Q-scorer on new labels
- Run optimization sweep

### Step 3: End-to-End Validation

**Baseline Comparison:**
- Run time-travel with old entry stack (no Q-scorer): 1000 cycles
- Run time-travel with Q-scorer + fixed XGBoost: 1000 cycles
- Compare: P&L, win rate, avg hold time, avg P&L per trade

**Success Criteria:**
- Q-scorer win rate > 10% (up from 0-1.6%)
- Average hold time > 10 minutes (up from 2 minutes)
- P&L per trade > -$2 (up from -$10 to -$15)

---

## Configuration Overrides for Testing

### Disable XGBoost Early Exits (Quick Test)

Set high exit threshold to essentially disable XGBoost:

```bash
export TT_XGB_EXIT_THRESHOLD=0.95  # Exit only if 95%+ confident (very rare)
```

Or use simple fallback rules:

```bash
# Force XGBoost to use simple rules by preventing training
export XGB_MIN_TRADES_FOR_ML=999999
```

### Test with Fixed Horizon

Override XGBoost to respect prediction timeframe:

```python
# In paper_trading_system.py, modify XGBoost exit check:
if time_held_minutes < prediction_timeframe * 0.75:
    # Don't allow XGBoost to exit before reaching 75% of horizon
    return False, 0.0, {'reason': 'Respecting prediction horizon'}
```

---

## Expected Outcomes

### After XGBoost Labeling Fix

**Before Fix:**
- Exit at: +0.1% to +1.0%
- Hold time: 1-10 minutes
- Confidence: 99.5%
- Pattern: Exit ASAP at any profit

**After Fix:**
- Exit at: +3% to +10%
- Hold time: 10-20 minutes
- Confidence: 60-80% (less extreme)
- Pattern: Allow winners to develop, exit on significant profit loss

### After Ghost Reward Alignment

**Before Alignment:**
- Training: Optimize for 15-min horizon
- Evaluation: Exits at 2-min with +0.5%
- Mismatch: Model optimizes for world that doesn't exist

**After Alignment:**
- Training: Optimize for XGBoost-adjusted returns
- Evaluation: Matches training assumptions
- Alignment: Model optimizes for actual execution environment

---

## Next Steps

1. **Implement XGBoost labeling fix** (backend/xgboost_exit_policy.py:369-384)
2. **Test XGBoost fix in isolation** (500 cycles, no Q-scorer, just verify better exits)
3. **Implement ghost reward alignment** (backend/decision_pipeline.py)
4. **Regenerate Q-scorer dataset** (1500+ cycles with aligned labels)
5. **Retrain Q-scorer** (same hyperparameters, new labels)
6. **Re-run optimization** (84 configs with fixed system)
7. **Compare results** (old vs new P&L, win rate, hold times)

---

## Files to Modify

1. **backend/xgboost_exit_policy.py** (lines 369-384)
   - Fix labeling strategy

2. **backend/decision_pipeline.py** (GhostTradeEvaluator class)
   - Implement simulator-aligned ghost rewards
   - Add Black-Scholes option pricing
   - Add XGBoost exit simulation

3. **backend/paper_trading_system.py** (optional - for testing)
   - Add horizon respect logic to XGBoost exit check
   - Add configuration flags for testing

4. **scripts/train_time_travel.py** (optional - for env vars)
   - Add `USE_SIMULATOR_ALIGNED_LABELS` env var support
   - Add `XGB_MIN_HOLD_MINUTES` env var for testing
