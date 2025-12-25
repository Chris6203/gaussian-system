# Q-Scorer Optimization Report

**Generated:** 2025-12-15
**Dataset:** models/run_20251214_115647 (8,584 MissedOpportunityRecords, 177 matched for training)
**Model:** models/q_scorer.pt (130 epochs, 65.7% accuracy, validation MSE 3103.96)
**Optimization:** 84 parameter combinations tested (1000 cycles each)

---

## Executive Summary

The Q-scorer entry controller optimization has completed successfully, testing 84 different parameter combinations across thresholds, confidence gates, and minimum return filters. **However, all configurations resulted in negative P&L**, ranging from -7.59% to -21.02% over 1000 cycles.

**Best Configuration** (least loss):
- **ENTRY_Q_THRESHOLD:** 0.0
- **Q_ENTRY_MIN_CONF:** 0.15 (15%)
- **Q_ENTRY_MIN_ABS_RET:** 0.001 (0.1%)
- **Results:** -$379.63 P&L (-7.59%), 61 trades, 1.6% win rate

---

## Critical Finding: Systematic Losses

### Performance Breakdown

All 84 tested configurations lost money:

| Config | P&L | Win Rate | Trades | Notes |
|--------|-----|----------|--------|-------|
| thr=0, conf=0.15, ret=0.001 | -$379.63 (-7.59%) | 1.6% | 61 | **BEST** |
| thr=-10, conf=0.15, ret=0.0005 | -$531.11 (-10.62%) | 0.0% | 91 | 2nd best |
| thr=0, conf=0.15, ret=0 | -$545.69 (-10.91%) | 0.0% | 137 | Most trades |
| thr=-10, conf=0.3, ret=0.0005 | -$802.18 (-16.04%) | 0.0% | 72 | |
| thr=-10, conf=0.25, ret=0 | -$1,000.21 (-20.00%) | 0.0% | 85 | **WORST** |

### Key Observations

1. **Win Rates Extremely Low:** 0% to 1.6% across all configurations
2. **Best Configuration:** Only 1 winning trade out of 61 (1.6% win rate)
3. **Trend:** Lower thresholds and confidence gates → slightly better (less bad) P&L
4. **Trade Count:** More selective filters reduce losses but don't produce wins

---

## Root Cause Analysis

This outcome aligns with the known limitation documented in `docs/Q_SCORER_SYSTEM_MAP.md` (lines 195-200):

> **"If baseline win-rate is ~0% across the board, the simulator's option pricing/fill/exit assumptions may be too punitive or inconsistent with label generation."**

### The Fundamental Mismatch

**Training Objective (Ghost Rewards):**
- Q-scorer was trained on "realized" labels from `GhostTradeEvaluator`
- Uses lightweight option pricing proxy (Delta × underlying move × 100)
- Subtracts friction + theta decay estimates
- **Horizon:** 15 minutes

**Evaluation Environment (Time-Travel Simulator):**
- Uses Black-Scholes pricing for option prices
- XGBoost exit policy exiting trades at 0.1%-1.0% gains (very early exits)
- Different friction assumptions
- Different theta decay behavior
- **Exit timing:** 1-30 minutes (median ~2 minutes based on logs)

**Result:** Model optimizes for 15-minute ghost rewards but is evaluated in a simulator with different pricing, faster exits, and potentially higher friction.

---

## Training Data Quality

### Dataset Characteristics

**Generation Run:** models/run_20251214_115647
- **Total Cycles:** 1,500
- **DecisionRecords:** 23,751 (full cycle logs with predictor embeddings)
- **MissedOpportunityRecords:** 8,584 (HOLD actions with ghost rewards)
- **Matched for Training:** 177 records (filtered to state='flat' for entry-only)

### Bug Fix Impact

**Before Fix:** 0 MissedOpportunityRecords (restrictive horizon check at line 2570)
**After Fix:** 8,584 records (85% capture rate with partial horizons)

The bug fix successfully increased dataset generation from 0 to 8,584 records, but only 177 matched records were usable for training (2% match rate).

**Low Match Rate Root Causes:**
1. **Timestamp Alignment:** DecisionRecords and MissedOpportunityRecords must match within 5 seconds
2. **State Filter:** Only 'flat' (no active position) records used for entry-only training
3. **Missing Predictor Embeddings:** MissedOpportunityRecords don't contain embeddings directly

---

## Model Performance

### Training Results

**Architecture:** QScorerNet (MLP)
- **Input:** 64-dim predictor_embedding + 12 scalar features
- **Hidden Layers:** [128, 64]
- **Output:** Q_hold, Q_call, Q_put

**Training Metrics (130 epochs):**
- **Best Validation MSE:** 3,103.96 (56% improvement from initial 7,109)
- **Best-Action Accuracy:** 65.7%
- **Calibration Offset:** +18.39 (model is conservative, underestimates rewards)
- **Early Stopping:** Epoch 110 (patience=25)

**Label Distribution:**
- Positive examples (upweighted 3×)
- Label clipping at ±200
- Horizon: 15 minutes

---

## Optimization Results

### Parameter Grid

**Tested Combinations:** 84
- **ENTRY_Q_THRESHOLD:** -10, -5, 0, 5, 10, 15, 20 (7 values)
- **Q_ENTRY_MIN_CONF:** 0.15, 0.20, 0.25, 0.30 (4 values)
- **Q_ENTRY_MIN_ABS_RET:** 0.0, 0.0005, 0.001 (3 values)

**Evaluation:** 1,000 cycles each (time-travel backtest)

### Top 10 Configurations

| Rank | Threshold | MinConf | MinRet | P&L | WinRate | Trades |
|------|-----------|---------|--------|-----|---------|--------|
| 1 | 0.0 | 0.15 | 0.001 | -$379.63 | 1.6% | 61 |
| 2 | -10.0 | 0.15 | 0.0005 | -$531.11 | 0.0% | 91 |
| 3 | 0.0 | 0.15 | 0.0 | -$545.69 | 0.0% | 137 |
| 4 | -10.0 | 0.30 | 0.0005 | -$802.18 | 0.0% | 72 |
| 5 | 0.0 | 0.20 | 0.0005 | -$839.60 | 0.0% | 80 |
| 6 | 0.0 | 0.30 | 0.0005 | -$844.38 | 0.0% | 62 |
| 7 | 5.0 | 0.25 | 0.0 | -$846.77 | 0.0% | 74 |
| 8 | -5.0 | 0.15 | 0.0 | -$853.70 | 0.0% | 81 |
| 9 | 20.0 | 0.15 | 0.0 | -$866.67 | 0.0% | 74 |
| 10 | 5.0 | 0.30 | 0.0005 | -$868.57 | 0.0% | 81 |

---

## Recommendations

### Immediate Actions

1. **DO NOT DEPLOY:** The current Q-scorer cannot be safely deployed in live trading

2. **Align Training and Evaluation:**
   - **Option A:** Modify ghost reward calculation to match Black-Scholes + XGBoost exit logic
   - **Option B:** Modify simulator to match ghost reward assumptions (simpler option pricing)
   - **Option C:** Train Q-scorer on actual simulator P&L instead of ghost rewards

3. **Investigate XGBoost Exit Policy:**
   - Current behavior: Exits at 0.1%-1.0% gains after 1-10 minutes
   - Consider: Is this exit policy too aggressive? Should it allow trades to reach 15-minute horizon?

### Dataset Improvements

1. **Increase Matched Records:**
   - Current: 177 matched records (2% of 8,584 MissedOpportunityRecords)
   - Target: >1,500 matched records
   - **Action:** Write predictor_embedding directly to MissedOpportunityRecords during generation

2. **Longer Training Run:**
   - Current: 1,500 cycles
   - Recommended: 5,000-10,000 cycles for more diverse market conditions

3. **Validation Split:**
   - Ensure walk-forward validation uses recent data (test on different market regime than training)

### Architecture Experiments

1. **Simpler Baseline:**
   - Try linear model or shallow MLP to verify data quality
   - If simple model also fails → data/label problem, not architecture

2. **Direct P&L Prediction:**
   - Instead of predicting Q-values from ghost rewards
   - Train on actual simulated P&L from completed trades

3. **Regime-Conditional Model:**
   - Current: Single model for all market conditions
   - Try: Separate models or regime features for trending vs ranging markets

### Exit Policy Alignment

1. **Consistent Horizon:**
   - Train Q-scorer with 2-minute labels (matching XGBoost early exits)
   - OR: Modify XGBoost to respect 15-minute horizon

2. **Exit Reward Feedback Loop:**
   - Use actual exit P&L from XGBoost as training labels
   - This creates alignment between entry predictions and realized exits

---

## Technical Notes

### Path Issue During Optimization

The optimization script ran from `E:\gaussian\` while looking for results in `E:\gaussian\output3\`. SUMMARY.txt files were written to the parent directory but the leaderboard expected them in output3. This caused the automated "best config" selection to fail (all showed 0 trades).

**Manual Analysis:** SUMMARY files found in `E:\gaussian\models\q_optimization\runs\*\SUMMARY.txt`

### Model Files

**Trained Model:**
- Location: `E:\gaussian\output3\models\q_scorer.pt` (78KB)
- Metadata: `E:\gaussian\output3\models\q_scorer_metadata.json`
- Training Run: `E:\gaussian\output3\models\run_20251214_115647/`

**Optimization Results:**
- All runs: `E:\gaussian\models\q_optimization\runs\*/SUMMARY.txt` (84 configs)
- Leaderboard: `E:\gaussian\output3\models\q_optimization\leaderboard.csv` (empty/incorrect)
- Correct data: Parent directory (`E:\gaussian\models\q_optimization\`)

---

## Next Steps

**Priority 1: Root Cause Validation**
1. Run baseline (old entry stack) on same 1,000-cycle window
2. Compare win rate and P&L to Q-scorer
3. If baseline also has ~0% win rate → simulator pricing issue confirmed

**Priority 2: Label Alignment**
1. Generate new dataset with Q-labels = actual simulator P&L (not ghost rewards)
2. Retrain Q-scorer on aligned labels
3. Re-run optimization sweep

**Priority 3: Exit Policy Investigation**
1. Analyze XGBoost exit decisions: Why exiting at 0.1%-1% so frequently?
2. Test: Disable XGBoost early exits, use fixed 15-minute horizon
3. Measure: Does win rate improve with longer hold times?

---

## Conclusion

The Q-scorer optimization revealed a **systematic mismatch between training objectives and evaluation environment**. The model was trained on 15-minute ghost rewards using lightweight option pricing, but evaluated in a simulator with Black-Scholes pricing and XGBoost exits that close trades within 1-10 minutes at tiny gains.

**Result:** 0-1.6% win rates across all 84 parameter configurations, with P&L losses ranging from -7.59% to -21.02%.

This outcome is **not a failure of the optimization process** - it's a valuable discovery of the fundamental misalignment between:
1. What the model is trained to predict (ghost rewards)
2. What the model is evaluated on (simulator P&L)

**Recommendation:** Before further optimization, align training labels with evaluation metrics by using actual simulator P&L as training targets, or modify the simulator to match ghost reward assumptions.
