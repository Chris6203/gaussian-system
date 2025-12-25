# Architecture Comparison: output vs output3

This document compares the architecture of `E:\gaussian\output` (original bot) vs `E:\gaussian\output3` (current development) to identify components that could be adapted.

## Executive Summary

| Component | output | output3 | Impact |
|-----------|--------|---------|--------|
| Stop Loss | -35% | -8% | output gives more room to recover |
| Take Profit | +40% | +12% | output captures larger wins |
| Exit Ratio | 1.14:1 (40/35) | 1.5:1 (12/8) | Similar math, different scale |
| Max Hold | 24 hours | 45 minutes | **MAJOR DIFFERENCE** |
| Prediction Timeframes | 7 (15m to 48h) | 3 (15m, 1h, 4h) | output uses more context |
| Adaptive Weights | Yes (RLThresholdLearner) | No | output learns optimal factors |
| Timeframe Learning | Yes (AdaptiveTimeframeWeights) | No | output adapts to accuracy |

**Key Insight**: output is designed for **longer-term options trading** (24h hold, 40% profit targets), while output3 is designed for **intraday scalping** (45m hold, 12% targets). These are fundamentally different strategies.

---

## 1. Exit Strategy Comparison

### output (Longer-Term)
```python
# backend/paper_trading_system.py lines 410-411
self.stop_loss_pct = 0.35   # 35% stop loss
self.take_profit_pct = 0.40 # 40% take profit

# Max hold: 24 hours
max_hold_hours: float = 24.0
```

**Math**: With 40% WR, need R:R of 1.5:1 to break even.
- Ratio: 40/35 = 1.14:1 (slightly below break-even)
- BUT: 24h hold allows time for positions to recover

### output3 (Scalping)
```python
# config.json -> architecture.exit_policy
"hard_stop_loss_pct": -8.0,
"hard_take_profit_pct": 12.0,
"hard_max_hold_minutes": 45
```

**Math**: With 40% WR:
- Ratio: 12/8 = 1.5:1 (exactly break-even)
- Problem: 45m hold often triggers theta decay before profit target

### Key Difference
output's 24h hold allows time for larger directional moves to materialize. output3's 45m hold often exits too early due to:
1. Theta decay eating profits
2. Intraday noise triggering stops
3. Not enough time for prediction to prove correct

---

## 2. Prediction Timeframes

### output (7 Timeframes with Adaptive Weights)
```python
# config.json -> prediction_timeframes
"15min":  { "weight": 0.25, "enabled": true },
"30min":  { "weight": 0.20, "enabled": true },
"1hour":  { "weight": 0.15, "enabled": true },
"4hour":  { "weight": 0.15, "enabled": true },
"12hour": { "weight": 0.10, "enabled": true },
"24hour": { "weight": 0.10, "enabled": true },
"48hour": { "weight": 0.05, "enabled": true }
```

Plus **AdaptiveTimeframeWeights** that learns which timeframes are most accurate and adjusts weights dynamically.

### output3 (3 Fixed Timeframes)
```python
# config.json -> prediction_timeframes
"15min": { "weight": 0.40, "enabled": true },
"1hour": { "weight": 0.35, "enabled": true },
"4hour": { "weight": 0.25, "enabled": true }
# Others disabled
```

No adaptive learning - weights are fixed.

---

## 3. Adaptive Learning Components (output only)

### RLThresholdLearner (`backend/rl_threshold_learner.py`)

**Purpose**: Instead of hardcoded thresholds like:
```python
# Hard-coded (bad):
if confidence >= 0.40 AND return >= 0.002 AND momentum >= 0.001:
    trade()
```

Uses a neural network to learn optimal WEIGHTS:
```python
# Learned (good):
score = w1*confidence + w2*return + w3*momentum + w4*volume
if score >= learned_threshold:
    trade()
```

**Key Features**:
- Neural network learns which factors matter most from trade outcomes
- Tracks rejected signals to learn from missed opportunities
- Self-adapting threshold based on performance:
  - If winners have much higher scores than losers: lower threshold (trade more)
  - If winners and losers have similar scores: raise threshold (be selective)
- Experience replay buffer (10,000 experiences)
- Hot-loadable/saveable checkpoints

**Architecture**:
```python
class ThresholdLearner(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=16):
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # 4 inputs: conf, ret, mom, vol
            nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),          # 1 output: composite score
            nn.Sigmoid()                        # 0-1 range
        )
```

**Training Loop**:
1. Store trade outcomes (inputs + P&L)
2. Sample batch of experiences
3. Binary classification: predict 1 for winners, 0 for losers
4. BCE loss + Adam optimizer
5. Adapt threshold based on score distributions

### AdaptiveTimeframeWeights (`backend/adaptive_timeframe_weights.py`)

**Purpose**: Learns which prediction timeframes are most accurate and adjusts their weights.

**Key Insight**: If 30-minute predictions are 80% accurate but 15-minute predictions are only 50% accurate, weight the 30-minute predictions higher!

**Algorithm**:
1. Track predictions made at each timeframe
2. Validate predictions against actual outcomes
3. Calculate accuracy per timeframe
4. Adjust weights based on relative performance:
   - Better than average accuracy: increase weight
   - Worse than average: decrease weight
5. Normalize weights to sum to 1.0

**Configuration**:
```python
self.min_predictions_for_update = 20  # Need data before adjusting
self.weight_learning_rate = 0.05      # How fast to adapt
self.min_weight = 0.05                # Floor (5%)
self.max_weight = 0.60                # Ceiling (60%)
```

---

## 4. What output3 is Missing

1. **Adaptive Threshold Learning**
   - output3 uses fixed thresholds from config.json
   - No learning which factors (confidence, momentum, return) matter most
   - No missed opportunity tracking

2. **Adaptive Timeframe Weights**
   - output3 has static weights (0.40, 0.35, 0.25)
   - No learning which timeframes are most accurate
   - No validation of predictions against outcomes

3. **Longer Hold Times**
   - output3's 45m max hold is fundamentally limiting
   - Prediction horizon (15m) misaligned with hold time (45m)
   - Theta decay eats profits before target hit

4. **Wider Exit Bands**
   - output3's 8% stop / 12% TP is too tight for options
   - Normal intraday volatility triggers stops
   - Need wider bands or volatility-adjusted exits

---

## 5. Recommendations for output3

### Priority 1: Fix Exit Strategy Mismatch
The core problem is structural, not feature-quality:
- With 40% win rate and 8%/12% exits, we barely break even
- But transaction costs (spread, slippage, theta) push us negative

**Options**:
A. Widen stops to -20% / +30% (like output's proportions)
B. Extend hold time to 2-4 hours
C. Add volatility-adjusted exits (wider stops in high VIX)

### Priority 2: Add Adaptive Threshold Learning
Port `RLThresholdLearner` from output:
- Learns optimal weights for trading factors
- Adapts threshold based on performance
- Tracks missed opportunities

### Priority 3: Add Adaptive Timeframe Weights
Port `AdaptiveTimeframeWeights` from output:
- Learns which prediction horizons are accurate
- Weights accurate timeframes higher
- Could dramatically improve direction prediction

### Priority 4: Align Prediction Horizon with Hold Time
Current mismatch:
- Predicting 15 minutes ahead
- Holding for 45 minutes
- Position often reverses after prediction horizon

**Fix**: Either:
A. Reduce hold time to 15-20 minutes
B. Increase prediction horizon to 45-60 minutes
C. Use multi-horizon predictions with weighted exits

---

## 6. Implementation Roadmap

### Phase 1: Quick Wins (Config Changes)
1. Widen exit bands: -15% stop / +25% take profit
2. Reduce max hold to 30 minutes (closer to prediction horizon)
3. Test with 5K cycles

### Phase 2: Port Adaptive Learning
1. Copy `rl_threshold_learner.py` from output
2. Integrate into entry decision pipeline
3. Train on 10K cycles to learn factor weights

### Phase 3: Port Timeframe Adaptation
1. Copy `adaptive_timeframe_weights.py` from output
2. Enable all 7 prediction timeframes
3. Let system learn which are accurate

### Phase 4: Validate
1. Run 20K validation test
2. Compare per-trade P&L vs baseline
3. Document improvements

---

## Summary Table: Components to Port

| Component | File | Complexity | Expected Impact |
|-----------|------|------------|-----------------|
| RLThresholdLearner | `backend/rl_threshold_learner.py` | Medium | High - learns optimal entry factors |
| AdaptiveTimeframeWeights | `backend/adaptive_timeframe_weights.py` | Low | Medium - optimizes prediction weights |
| Wider Exit Bands | Config change | Low | High - structural fix |
| Longer Hold Time | Config change | Low | High - aligns with predictions |

---

*Document created: 2025-12-21*
*Purpose: Identify architectural improvements from output that could benefit output3*
