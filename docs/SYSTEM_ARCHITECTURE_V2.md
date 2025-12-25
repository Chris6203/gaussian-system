# System Architecture V2 - Trading Bot Decision Flow

## Overview

This document maps the complete decision flow from market data to trade execution, identifying architectural issues and improvement opportunities.

**Last Updated:** 2025-12-22
**Current Best Config:** Phase 16 (Baseline + PnL Calibration Gate)
**Performance:** -4.34% over 20K cycles (vs -93% for other configs)

---

## 1. HIGH-LEVEL ARCHITECTURE

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MARKET DATA LAYER                                  │
│  SPY (1-min OHLCV) + QQQ + VIX + BTC + Sector ETFs + Options Chain          │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         FEATURE ENGINEERING                                  │
│  50 base features: momentum, volatility, technicals, cross-asset            │
│  + 9 enhanced (time-of-day cyclical, gaussian pattern similarity)           │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
┌───────────────────────────┐       ┌───────────────────────────┐
│  NEURAL NETWORK PREDICTOR │       │   HMM REGIME DETECTOR     │
│  UnifiedOptionsPredictor  │       │   MultiDimensionalHMM     │
│  ─────────────────────────│       │   ─────────────────────── │
│  • TCN temporal encoder   │       │   • 3×3×3 state space     │
│  • Bayesian uncertainty   │       │   • Trend/Vol/Liquidity   │
│  • Multi-timeframe (15m,  │       │   • 27 possible regimes   │
│    30m, 1h, 4h)           │       │   • Retrain every 24h     │
│  ─────────────────────────│       │   ─────────────────────── │
│  OUTPUT:                  │       │   OUTPUT:                 │
│  • predicted_return       │       │   • hmm_trend (0-1)       │
│  • direction (UP/DN/NEUT) │       │   • hmm_volatility (0-1)  │
│  • confidence (0.20-0.35) │       │   • hmm_liquidity (0-1)   │
│  • volatility_pred        │       │   • hmm_confidence        │
└───────────────────────────┘       └───────────────────────────┘
                    │                               │
                    └───────────────┬───────────────┘
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          ENTRY CONTROLLER                                    │
│  Configurable: "bandit" (active), "rl", "consensus", "q_scorer", "v3"       │
│  ─────────────────────────────────────────────────────────────────────────  │
│  BANDIT MODE (Current Best):                                                │
│  • HMM-only entry: trend > 0.65 (bullish) or < 0.35 (bearish)              │
│  • Confidence > 0.60                                                        │
│  • Volatility filter: hmm_volatility < 0.70                                 │
│  • Neural confirmation: predicted_direction agrees                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          ENTRY GATE STACK                                    │
│  ─────────────────────────────────────────────────────────────────────────  │
│  GATE 1: RLThresholdLearner (16-feature neural filter)                      │
│          • Blocks ~96% of signals when working well                         │
│          • Uses learned weights for: confidence, predicted_return,          │
│            momentum, volume_spike, vix, hmm_trend, etc.                     │
│                                                                             │
│  GATE 2: PnL Calibration Gate (Phase 13)                                    │
│          • P(profit|confidence) must be >= 40%                              │
│          • Uses Platt+Isotonic calibration from trade outcomes              │
│          • Learns from first 30 trades, then gates                          │
│                                                                             │
│  GATE 3: Safety Filter (optional)                                           │
│          • Min confidence: 0.40                                             │
│          • Max VIX: 35.0                                                    │
│          • Regime conflict check                                            │
│                                                                             │
│  GATE 4: Execute Gate                                                       │
│          • Confidence >= 0.55                                               │
│          • Active positions < 3                                             │
│          • Min time between trades: 5 min                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          TRADE EXECUTION                                     │
│  PaperTradingSystem or LiveTradingEngine (Tradier API)                      │
│  • Position size: 1 contract (scaled by confidence & regime)                │
│  • Order type: Market (paper) or Limit (live)                               │
│  • Records to database: unified_signals, paper_trades tables                │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          EXIT DECISION LOOP                                  │
│  (Every 1-minute cycle while position open)                                 │
│  ─────────────────────────────────────────────────────────────────────────  │
│  HARD RULES (Priority Order - First Match Wins):                            │
│  1. Stop Loss:     P&L <= -8.0%           → EXIT                           │
│  2. Take Profit:   P&L >= +12.0%          → EXIT                           │
│  3. Max Hold:      held >= 45 minutes     → EXIT (FORCE_CLOSE)             │
│  4. Near Expiry:   < 30 min to expiry     → EXIT                           │
│  5. Trailing Stop: P&L < (high - 2.0%)    → EXIT (if was +4%+)             │
│  ─────────────────────────────────────────────────────────────────────────  │
│  MODEL-BASED EXIT (If hard rules didn't trigger):                           │
│  • XGBoostExitPolicy: P(exit) > 0.55                                        │
│  • Anti-premature: min_profit 3%, min_hold 10min                            │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. CRITICAL ARCHITECTURAL ISSUES

### Issue 1: Confidence Calibration Misalignment (MAJOR)

**Problem:**
- Neural network outputs confidence in range **0.20-0.35** (typical)
- Entry thresholds set at **0.55+**
- Result: Most signals filtered before evaluation

**Evidence:**
```python
# From config.json
"base_confidence_threshold": 0.55,  # Production gate
"training_min_confidence": 0.20,     # Training relaxed (acknowledges problem)
"min_weighted_confidence": 0.15,     # Consensus (super relaxed)
```

**Impact:** Good trades blocked, only "lucky" high-confidence signals pass.

**Solution Options:**
1. Add Platt scaling layer to recalibrate confidence
2. Lower thresholds to match actual distribution
3. Train predictor with calibration loss term

---

### Issue 2: Horizon Misalignment (MAJOR)

**Problem:**
- Predictor trained on **15-minute** horizon
- Max hold time is **45 minutes** (3x longer)
- RL reward calculated at exit, not at prediction horizon

**Evidence:**
```python
# From config.json
"horizons": {
    "prediction_minutes": 15,   # What we predict
    "rl_reward_minutes": 15,    # What RL is trained on
    "exit_reference_minutes": 15
}
# BUT:
"hard_max_hold_minutes": 45     # Actual hold time!
```

**Impact:**
- Position drifts randomly after 15 minutes
- Win rate drops as prediction "expires"
- Exit often by FORCE_CLOSE (time), not by prediction

**Solution Options:**
1. Reduce max_hold to 15-20 minutes (align with prediction)
2. Train predictor on 45-minute horizon
3. Use adaptive hold time based on prediction confidence

---

### Issue 3: Exit Ratio Asymmetry (MAJOR)

**Problem:**
- Stop Loss: **-8%** (max loss per trade)
- Take Profit: **+12%** (max gain per trade)
- Ratio: 1.5:1 (need 40% win rate to break even)

**Evidence:**
```
Phase 16 20K results: 29.8% win rate
Required for break-even: 40% win rate
Gap: -10% win rate deficit
```

**Impact:** Math doesn't work at current win rates.

**Solution Options:**
1. Tighten stop loss to -5% (need only 29% win rate)
2. Widen take profit to +20% (better risk/reward)
3. Use asymmetric sizing (larger on high-confidence)

---

### Issue 4: Gate Proliferation (MODERATE)

**Problem:** Too many independent gates filtering trades:
1. Entry Controller (bandit/RL/consensus)
2. RLThresholdLearner
3. PnL Calibration Gate
4. Safety Filter
5. Execute Gate
6. Regime Filter (optional)

**Impact:**
- Each gate is "dumb" about what others blocked
- Good trades can be killed by any single gate
- Hard to debug why trades are blocked

**Solution Options:**
1. Consolidate into single configurable entry filter
2. Use weighted ensemble of gates (not AND logic)
3. Track gate vetoes to identify over-filtering

---

### Issue 5: HMM-Neural Conflict Not Enforced (MODERATE)

**Problem:**
- HMM says "Bearish" but neural predicts "UP"
- Currently: Trade may execute anyway
- config.json: `"require_trend_alignment": true` but NOT consistently enforced

**Impact:** Conflicting signals lead to random-walk trades.

**Solution Options:**
1. Hard veto on HMM-Neural disagreement
2. Reduce confidence when disagreement detected
3. Require 2/3 agreement (HMM + Neural + Momentum)

---

### Issue 6: Regime Filter Too Relaxed (MINOR)

**Problem:**
```json
"regime_filter": {
    "trend_strength_min": 0.05,    // Only 5% trend clarity
    "avoid_choppy": false,          // Disabled!
    "require_liquidity": false,     // Disabled!
    "confidence_min": 0.2           // Only 20% HMM confidence
}
```

**Impact:** Trades in choppy/ranging markets → losses.

**Solution Options:**
1. Set `avoid_choppy: true`
2. Raise `trend_strength_min` to 0.20
3. Enable liquidity requirement

---

## 3. IMPROVEMENT OPTIONS RANKED

| Priority | Issue | Solution | Expected Impact | Effort |
|----------|-------|----------|-----------------|--------|
| **1** | Horizon Misalignment | Reduce max_hold to 20min | High (+10-15% P&L) | Low |
| **2** | Exit Ratio | Tighten stop to -5% | High (break-even possible) | Low |
| **3** | Confidence Calibration | Add Platt scaling | Medium (+5% win rate) | Medium |
| **4** | Gate Proliferation | Consolidate gates | Medium (fewer blocked) | High |
| **5** | HMM-Neural Conflict | Hard veto on disagree | Medium (+3% win rate) | Low |
| **6** | Regime Filter | Enable choppy filter | Low (+1-2% P&L) | Low |

---

## 4. RECOMMENDED NEXT EXPERIMENTS

### Experiment A: Aligned Horizons
```bash
# Reduce max hold to match prediction horizon
# Edit config.json:
"hard_max_hold_minutes": 20  # Was 45
"time_travel_training.max_hold_minutes": 20
```

### Experiment B: Tighter Stop Loss
```bash
# Improve risk/reward ratio for lower win rates
# Edit config.json:
"hard_stop_loss_pct": -5.0  # Was -8.0
"hard_take_profit_pct": 15.0  # Was 12.0 (ratio now 3:1)
```

### Experiment C: Hard HMM Alignment
```bash
# Only trade when HMM and Neural agree
# Requires code change in unified_rl_policy.py
# Add check: if hmm_trend > 0.55 and predicted_direction == "UP" -> OK
#           if hmm_trend < 0.45 and predicted_direction == "DOWN" -> OK
#           else -> VETO
```

### Experiment D: Confidence Recalibration
```bash
# Add Platt scaling to neural output
# Requires code change in neural_networks.py
# Train sigmoid(a*conf + b) to map confidence to P(correct)
```

---

## 5. DATA FLOW DIAGRAM

```
                    ┌──────────────────┐
                    │   Historical DB  │
                    │  (market_data)   │
                    └────────┬─────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────────────┐
│                    TIME-TRAVEL TRAINING LOOP                    │
│  scripts/train_time_travel.py                                  │
│  ───────────────────────────────────────────────────────────── │
│  for timestamp in historical_data:                             │
│      features = generate_features(ohlcv)         # 59 dims     │
│      hmm_regime = hmm.predict(features)          # 27 states   │
│      neural_pred = predictor(features, sequence) # returns, dir│
│      signal = entry_controller(neural_pred, hmm_regime)        │
│                                                                │
│      if signal != HOLD:                                        │
│          for gate in [rl_learner, pnl_cal, safety, execute]:   │
│              if gate.veto(signal):                             │
│                  signal = HOLD                                 │
│                  break                                         │
│                                                                │
│      if signal != HOLD:                                        │
│          trade = execute_trade(signal)                         │
│          active_positions.append(trade)                        │
│                                                                │
│      for position in active_positions:                         │
│          exit_decision = check_exit(position)                  │
│          if exit_decision.should_exit:                         │
│              close_position(position)                          │
│              rl_policy.learn(position.outcome)                 │
└────────────────────────────────────────────────────────────────┘
```

---

## 6. KEY FILES REFERENCE

| Component | File | Key Functions |
|-----------|------|---------------|
| Main Bot | `unified_options_trading_bot.py` | `generate_time_travel_signal()`, `execute_trade()` |
| Training Loop | `scripts/train_time_travel.py` | Main training script |
| Neural Predictor | `bot_modules/neural_networks.py` | `UnifiedOptionsPredictor.forward()` |
| HMM Regime | `backend/multi_dimensional_hmm.py` | `MultiDimensionalHMM.predict()` |
| Entry Controller | `backend/unified_rl_policy.py` | `decide_from_signal()` (bandit mode) |
| Consensus Entry | `backend/consensus_entry_controller.py` | `ConsensusEntryController.decide()` |
| Exit Manager | `backend/unified_exit_manager.py` | `should_exit()` |
| XGBoost Exit | `backend/xgboost_exit_policy.py` | `predict_exit_probability()` |
| RL Threshold | `backend/rl_threshold_learner.py` | `evaluate_signal()` |
| PnL Calibration | `bot_modules/calibration_utils.py` | `calibrate_pnl()` |

---

## 7. CONFIGURATION QUICK REFERENCE

```json
// config.json - Key settings for trading behavior
{
  "entry_controller": {
    "type": "bandit"  // Options: bandit, rl, consensus, q_scorer, v3
  },

  "exit_policy": {
    "hard_stop_loss_pct": -8.0,      // Max loss per trade
    "hard_take_profit_pct": 12.0,    // Max gain per trade
    "hard_max_hold_minutes": 45,     // ISSUE: 3x prediction horizon
    "trailing_stop_activation": 4.0, // Activate at +4%
    "trailing_stop_distance": 2.0    // Trail by 2%
  },

  "architecture": {
    "horizons": {
      "prediction_minutes": 15,      // Neural prediction horizon
      "rl_reward_minutes": 15,       // RL training horizon
      "exit_reference_minutes": 15   // Exit calculation horizon
    }
  },

  "trading": {
    "base_confidence_threshold": 0.55, // ISSUE: Neural outputs 0.20-0.35
    "max_positions": 3
  }
}
```

---

## 8. SUMMARY

**What's Working:**
- PnL Calibration Gate dramatically reduces losses (20x improvement)
- Bandit mode (HMM-only) is most stable entry strategy
- Multi-gate filtering prevents catastrophic trades

**What's Broken:**
- Horizon misalignment (15m prediction, 45m hold)
- Exit ratio math doesn't work at current win rates
- Confidence calibration mismatch

**Immediate Actions:**
1. Reduce max_hold to 20 minutes
2. Tighten stop loss to -5%
3. Test combined effect

**Expected Outcome:** Break-even to profitable with better alignment.
