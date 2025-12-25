## Q-Scorer System Map (Offline “Q Regression” Entry Controller)

This document maps how the **new Q-scorer entry system** works end-to-end in `output3`, and how we’re iterating it safely.

### What we are trying to do
- **Replace fragile online RL for entries** with a conservative offline model that estimates the **expected net reward** of:
  - `HOLD`
  - `BUY_CALLS`
  - `BUY_PUTS`
- Train it **offline** from logged state + counterfactual labels, then deploy it as an **entry-only controller**.
- Keep exits unchanged (existing exit stack stays in place).

---

## Core Data Flow

### 1) Live/Sim Bot Loop → Decision logging (always-on dataset)
**Where:** `output3/unified_options_trading_bot.py`

- Each cycle generates a `signal` (includes multi-timeframe prediction, confidence, regime, and execution heads).
- The bot writes a `DecisionRecord` every cycle (even when `signal=None` → synthetic HOLD record).
- If executed action is `HOLD`, it can also write a missed-opportunity record (depending on mode).

**Key idea:** The dataset is “per-cycle”, not just executed trades.

---

### 2) DecisionPipeline: Full state visibility + JSONL writers
**Where:** `output3/backend/decision_pipeline.py`

Produces two append-only JSONL streams:

#### A) `DecisionRecord` JSONL
- Contains:
  - `trade_state`: rich snapshot (including `predictor_embedding` + scalars)
  - `feature_vector` / `feature_names` / `schema_version` (stable encoding for analysis)
  - `proposed_action`, `executed_action`, `rejection_reasons`, etc.

#### B) `MissedOpportunityRecord` JSONL
- Contains per-action counterfactual labels:
  - `ghost_reward_call`
  - `ghost_reward_put`
  - plus `ghost_details` (friction, theta, per-horizon info)

Two label modes:
- **expected**: derived from predicted returns (quick, but can be biased)
- **realized**: derived from future prices in time-travel (preferred for training labels)

---

### 3) GhostTradeEvaluator (label generator)
**Where:** `output3/backend/decision_pipeline.py`

We use a lightweight option proxy to compute **net** reward:
- gross option PnL approx (delta-$):
  - \( pnl_{call} \approx \Delta \cdot (S \cdot r) \cdot 100 \)
  - \( pnl_{put} \approx -\Delta \cdot (S \cdot r) \cdot 100 \)
- subtract:
  - friction (spread/slippage round-trip + fees)
  - theta decay proxy

These rewards become the training targets for the Q model.

---

## Model + Deployment Flow

### 4) Dataset generation by time-travel
**Where:** `output3/scripts/train_time_travel.py`

Responsibilities:
- Replays historical bars (“time travel”) and runs the bot loop offline.
- Writes:
  - `state/decision_records.jsonl`
  - `state/missed_opportunities_full.jsonl`
  - `SUMMARY.txt` (simulation performance summary)
- Supports switching entry controller:
  - baseline (“old”): bandit/RL entry stack inside the script
  - new (“q_scorer”): `ENTRY_CONTROLLER=q_scorer`

Important knobs:
- `MODEL_RUN_DIR`: where outputs go
- `TT_MAX_CYCLES`: how long to run
- `TT_TRAIN_MAX_POSITIONS`: max concurrent trades (controls aggressiveness)
- `TRAINING_START_DATE`: align baseline vs q_scorer runs on same window

---

### 5) Train offline Q-scorer (supervised regression)
**Where:** `output3/training/train_q_scorer.py`

Trains:
- A small MLP `QScorerNet` predicting:
  - `Q_hold`, `Q_call`, `Q_put`

Inputs:
- `predictor_embedding` (64-d)
- a curated set of scalar features (regime, VIX, execution heads, etc.)

Labels:
- `Q_hold = 0`
- `Q_call = ghost_reward_call` at one configured horizon (e.g. 30m)
- `Q_put  = ghost_reward_put` at same horizon

Training details:
- walk-forward split (time-ordered)
- weighted MSE (upweights positive tails)
- label clipping (stability)
- saves artifacts atomically (safe for hot reload):
  - `output3/models/q_scorer.pt`
  - `output3/models/q_scorer_metadata.json`

---

### 6) Inference controller + hot reload
**Where:** `output3/backend/q_entry_controller.py`

Loads `q_scorer.pt` + metadata and returns an entry decision:
- computes Qs
- picks argmax among call/put
- trades only if `best_trade_q > ENTRY_Q_THRESHOLD`, else HOLD

Safety controls:
- **Hot reload** (checks mtimes every `Q_SCORER_RELOAD_SEC`)
- Optional conservative pre-gates:
  - `Q_ENTRY_MIN_CONF`
  - `Q_ENTRY_MIN_ABS_RET`

---

### 7) Wiring into the bot
**Where:** `output3/unified_options_trading_bot.py`

If `ENTRY_CONTROLLER=q_scorer`:
- the bot uses `QEntryController` for entry selection
- does not let RL override entry in that mode
- logs `q_values` + reasons into the signal for observability

---

## Optimization Loop (what we’re doing now)

### 8) One-command pipeline runner (dataset → train → eval)
**Where:** `output3/training/run_q_pipeline.py`

Runs:
- dataset generation (time-travel)
- Q-scorer training
- Q-scorer evaluation (`eval_q_scorer.py`)
- deploy check

---

### 9) Entry parameter optimization (grid search)
**Where:** `output3/training/optimize_q_entry.py`

Purpose:
- Retrain Q once (optional), then run many time-travel backtests with different:
  - `ENTRY_Q_THRESHOLD`
  - `Q_ENTRY_MIN_CONF`
  - `Q_ENTRY_MIN_ABS_RET`

Key improvement:
- `--min-trades` prevents “best = do nothing”.

Outputs:
- `leaderboard.csv`
- `best_params.json`
- `best/SUMMARY.txt`

---

### 10) Baseline vs new controller comparison (apples-to-apples)
**Where:** `output3/training/compare_time_travel_entry.py`

Runs the same time window twice:
- baseline (old entry stack)
- q_scorer (new entry stack)

Outputs:
- `baseline/SUMMARY.txt`
- `q_scorer/SUMMARY.txt`
- `COMPARE.json` (includes delta P&L, trades, etc.)

---

## What “good” looks like (near-term)
- Q-scorer should **trade less than baseline**, but not “0 trades”.
- In comparisons, aim for:
  - less negative P&L than baseline
  - stable trade counts
  - fewer catastrophic runs (risk control)

## Known current limitation
If baseline win-rate is ~0% across the board, the simulator's option pricing/fill/exit assumptions may be too punitive or inconsistent with label generation. In that case, we must align:
- (A) the "ghost reward" objective used for training
with
- (B) the realized P&L in the paper-trader simulation
so the optimizer isn't optimizing one world and being judged in another.

---

## Critical Bug Fix: Q-Value Anti-Selection (2025-12-17)

### Problem Discovered
The Q-scorer was systematically selecting **losing trades** while rejecting winners - classic anti-selection bug:
- **Trades TAKEN**: 22 total, 13.6% win rate, mean return -0.000903
- **Trades REJECTED**: 1,511 total, 51.3% win rate, mean return +0.000014
- High-confidence rejected trades (conf>0.3): 54.8% win rate, mean return +0.000042

Analysis file: `models/filtering_analysis.txt`

### Root Cause
**Inverted Q-values at inference time**:
- The model was trained with reward labels that had the correct sign
- But at inference, Q_call and Q_put were being selected with inverted preferences
- This caused the model to systematically choose trades with NEGATIVE expected values

Evidence:
```
Without fix: 13.6% win rate (anti-selection)
Expected after fix: ~51% win rate (proper selection)
```

### Solution Implemented
Added Q-value inversion at inference time in `backend/q_entry_controller.py:309-314`:

```python
# CRITICAL FIX: Invert Q-values to fix anti-selection bug
# The model was trained with inverted labels, so we flip them at inference
invert_q = bool(os.environ.get("Q_INVERT_FIX", "1"))  # Default ON
if invert_q:
    q_call = -q_call
    q_put = -q_put
```

**Environment variable**:
- `Q_INVERT_FIX=1`: Enable fix (DEFAULT, recommended)
- `Q_INVERT_FIX=0`: Disable fix (for testing/comparison)

### Additional Robustness Fixes
1. **Zero-embedding fallback** (`q_entry_controller.py:203-206`):
   - When `predictor_embedding` is missing, use zero vector instead of failing
   - Enables time-travel mode testing without predictor embeddings

2. **Q_DEBUG logging** (`train_time_travel.py:1973-1977`):
   - Added debug logging to trace Q-scorer activation
   - Helps diagnose entry controller selection issues

### Testing the Fix
To test Q-scorer with inversion fix:
```bash
set ENTRY_CONTROLLER=q_scorer
set Q_INVERT_FIX=1
set Q_SCORER_MODEL_PATH=models/q_scorer_bs_full.pt
set Q_SCORER_METADATA_PATH=models/q_scorer_bs_full_metadata.json
set ENTRY_Q_THRESHOLD=0
python scripts/train_time_travel.py
```

Expected outcome: Win rate should improve from 13.6% to ~51% (matching rejected trades baseline).

### Files Modified
- `backend/q_entry_controller.py`: Q-value inversion fix
- `scripts/train_time_travel.py`: Q_DEBUG logging
- `tools/analyze_filtering.py`: Analysis tool for identifying the bug

### Next Steps
1. Run full backtest with Q_INVERT_FIX=1
2. Verify win rate improves to ~51%
3. Compare P&L vs baseline entry controller
4. If validated, retrain model with correct label signs (future improvement)





