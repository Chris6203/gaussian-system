# Chat Summary: Q-Scorer System Development

## What We've Built

### Goal
Replace fragile online Reinforcement Learning (RL) for **entry decisions** with a conservative **offline Q-regression model** that estimates expected net reward for:
- `HOLD` (Q_hold = 0)
- `BUY_CALLS` (Q_call)
- `BUY_PUTS` (Q_put)

The model learns from historical data + counterfactual "ghost" rewards, then deploys as an entry-only controller. **Exits remain unchanged** (existing exit stack stays in place).

---

## System Components Created

### 1. **Data Pipeline** (`backend/decision_pipeline.py`)
- **DecisionRecord JSONL**: Logs every cycle with full state (`predictor_embedding`, scalars, VIX, HMM, etc.)
- **MissedOpportunityRecord JSONL**: Counterfactual labels for HOLD actions, computing "ghost rewards" for CALL/PUT at multiple horizons (10m, 15m, 20m, 30m, 60m)
- **GhostTradeEvaluator**: Realistic reward calculation (delta-based P&L minus friction: spread, slippage, fees, theta)

### 2. **Training** (`training/train_q_scorer.py`)
- Reads DecisionRecord + MissedOpportunityRecord JSONL
- Builds dataset: `(predictor_embedding + scalars) → [Q_hold, Q_call, Q_put]`
- Walk-forward time split (not random)
- Weighted MSE loss (upweight positive returns)
- Output calibration (`best_offset` to align predictions with realized rewards)
- Saves: `q_scorer.pt` + `q_scorer_metadata.json` (atomic writes for hot-reload safety)

### 3. **Inference Controller** (`backend/q_entry_controller.py`)
- Loads trained model + metadata
- Hot-reloading: checks file mtime, reloads model without restart
- Conservative pre-gates: `Q_ENTRY_MIN_CONF`, `Q_ENTRY_MIN_ABS_RET`
- Decision: `argmax(Q_call, Q_put)` only if `max > ENTRY_Q_THRESHOLD`, else HOLD
- Maintains existing gating (market hours, PDT, max positions)

### 4. **Dataset Generation** (`scripts/train_time_travel.py`)
- Historical simulation mode
- Generates DecisionRecord + MissedOpportunityRecord for offline training
- Configurable via env vars: `TT_Q_LABELS`, `Q_HORIZON_MINUTES`, `TT_MAX_CYCLES`, etc.
- Can run with Q-scorer controller (`ENTRY_CONTROLLER=q_scorer`) or baseline (bandit/RL)

### 5. **Evaluation Tools**
- **`tools/eval_q_scorer.py`**: Evaluates model on logged decisions (trade rate, predicted Q vs. realized/ghost reward, missed positive-EV count)
- **`training/compare_time_travel_entry.py`**: Apples-to-apples comparison (baseline vs. Q-scorer on same historical window)
- **`training/optimize_q_entry.py`**: Grid search over `ENTRY_Q_THRESHOLD`, `Q_ENTRY_MIN_CONF`, `Q_ENTRY_MIN_ABS_RET` with 2-stage optimization (coarse sweep + fine re-evaluation)

### 6. **Pipeline Runner** (`training/run_q_pipeline.py`)
- One-command workflow: dataset generation → training → evaluation → optional deploy
- Enforces `PAPER_TRADING` mode during data generation

### 7. **Documentation**
- **`docs/Q_SCORER_SYSTEM_MAP.md`**: End-to-end architecture map
- **`docs/NEURAL_ARCHITECTURE_CURRENT.md`**: Full neural network architecture

---

## Key Configuration Variables

### Training/Dataset
- `Q_HORIZON_MINUTES`: Target horizon for Q-labels (default: 30)
- `TT_Q_LABELS`: Label mode (`all`, `flat_only`, `hold_only`)
- `TT_MAX_CYCLES`: Limit cycles for dataset generation
- `GHOST_PREMIUM_PCT`: Contract cost assumption (default: 0.003 = 0.3%)
- `GHOST_SPREAD_SLIPPAGE_RT_PCT`: Spread/slippage assumption
- `GHOST_FEE_PER_CONTRACT`: Fee per contract
- `GHOST_THETA_DAILY_PCT`: Theta decay assumption

### Inference/Deployment
- `ENTRY_CONTROLLER`: Set to `q_scorer` to enable Q-scorer mode
- `ENTRY_Q_THRESHOLD`: Minimum Q-value to trade (default: 0.0)
- `Q_ENTRY_MIN_CONF`: Pre-gate: minimum confidence (default: 0.0 = disabled)
- `Q_ENTRY_MIN_ABS_RET`: Pre-gate: minimum absolute predicted return (default: 0.0 = disabled)
- `Q_SCORER_RELOAD_SEC`: Hot-reload check interval (default: 30s)
- `MODEL_RUN_DIR`: Directory containing `q_scorer.pt` + `q_scorer_metadata.json`

---

## Current Issue & Next Steps

### **Bug Fix Needed** (Blocking Training)
**Location**: `output3/scripts/train_time_travel.py` (line ~2570)

**Problem**: When `TT_Q_LABELS=all`, the code only writes `MissedOpportunityRecord` if the **exact** `Q_HORIZON_MINUTES` is found in `horizon_returns`. If that specific horizon isn't available for a cycle (e.g., missing 15m price), no record is written, leading to `matched=0` in `train_q_scorer.py`.

**Fix Required**: Remove the `if int(q_h) in horizon_returns:` check and **always write** `MissedOpportunityRecord` when `do_label=True`, storing whatever `horizon_returns` are available. `train_q_scorer.py` can then skip rows missing the configured horizon.

### **Immediate Next Steps**
1. ✅ Fix the `MissedOpportunityRecord` writing logic in `train_time_travel.py`
2. ✅ Regenerate dataset with 15-minute horizon (2500 cycles)
3. ✅ Train Q-scorer with `--horizon 15`
4. ✅ Evaluate model performance
5. ✅ Run comparison: baseline vs. Q-scorer on same historical window
6. ✅ Optimize entry parameters (`optimize_q_entry.py`) if needed
7. ✅ Deploy to live bot (set `ENTRY_CONTROLLER=q_scorer`)

---

## Recent Performance

**Last Training Run** (`run_qpipeline_entry_train`):
- Cycles: 900
- Trades: 24
- Win Rate: 0.0%
- P&L: -$592.85 (-11.86%)
- **Issue**: Model was too conservative (likely due to label quality or calibration)

**Last Dataset Generation** (`run_qdataset_entry`):
- Cycles: 11,831
- Trades: 7,266
- Final Balance: -$31,528.72
- **Note**: This was baseline (bandit/RL) mode, not Q-scorer mode

---

## Architecture Flow

```
Live/Sim Bot Loop
    ↓
DecisionPipeline (always-on logging)
    ├─→ DecisionRecord JSONL (every cycle)
    └─→ MissedOpportunityRecord JSONL (HOLD actions)
         ↓
train_time_travel.py (historical dataset generation)
    ↓
train_q_scorer.py (offline Q-regression)
    ├─→ q_scorer.pt (model weights)
    └─→ q_scorer_metadata.json (calibration, features)
         ↓
q_entry_controller.py (inference + hot-reload)
    ↓
unified_options_trading_bot.py (wiring: ENTRY_CONTROLLER=q_scorer)
```

---

## Files Modified/Created

### Core System
- `backend/decision_pipeline.py` (enhanced with `predictor_embedding`, ghost rewards)
- `backend/q_entry_controller.py` (new: inference controller)
- `unified_options_trading_bot.py` (wiring for Q-scorer mode)

### Training/Evaluation
- `training/train_q_scorer.py` (new: offline Q-regression training)
- `scripts/train_time_travel.py` (enhanced: Q-label logging, Q-scorer mode)
- `tools/eval_q_scorer.py` (new: evaluation metrics)
- `training/run_q_pipeline.py` (new: one-command pipeline)
- `training/optimize_q_entry.py` (new: parameter optimization)
- `training/compare_time_travel_entry.py` (new: baseline comparison)

### Documentation
- `docs/Q_SCORER_SYSTEM_MAP.md` (new: system architecture)
- `docs/NEURAL_ARCHITECTURE_CURRENT.md` (updated: full neural architecture)

---

## Design Principles

1. **Offline Learning**: No fragile online RL; train once, deploy
2. **Conservative**: Pre-gates prevent low-confidence/low-return trades
3. **Realistic Labels**: Ghost rewards include all friction (spread, slippage, fees, theta)
4. **Hot-Reloadable**: Model updates without bot restart (atomic writes)
5. **Non-Invasive**: Exits unchanged; only entry logic replaced when enabled
6. **Observable**: Full logging of decisions, veto reasons, Q-values

---

## Status: **Ready to Fix Bug & Continue Training**

The system is functionally complete but blocked by the `MissedOpportunityRecord` writing bug. Once fixed, we can regenerate a robust 15-minute dataset and train a properly calibrated Q-scorer.




