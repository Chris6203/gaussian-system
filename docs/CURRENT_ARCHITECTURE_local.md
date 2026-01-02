# Current Architecture - Gaussian Options Trading Bot

**Generated:** January 2, 2026
**Model:** dec_validation_v2 (59.8% Win Rate)

---

## High-Level Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA PIPELINE                                      │
│  Market Data → Features (50-60 dim) → HMM Regime → Predictor → RL Policy    │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
        ┌───────────────────────────────┼───────────────────────────────┐
        ▼                               ▼                               ▼
┌───────────────────┐        ┌───────────────────┐        ┌───────────────────┐
│   DATA SOURCES    │        │  PREDICTION ENGINE │        │  DECISION ENGINE  │
│                   │        │                   │        │                   │
│ • Tradier (live)  │        │ • Neural Network  │        │ • Unified RL      │
│ • Polygon (hist)  │        │ • HMM Regime      │        │ • XGBoost Exit    │
│ • FMP, Yahoo      │        │ • RBF Kernels     │        │ • Hard Safety     │
└───────────────────┘        └───────────────────┘        └───────────────────┘
```

---

## Neural Network: UnifiedOptionsPredictor

**Location:** `bot_modules/neural_networks.py`

```
INPUT FEATURES (per timestep):
├── Price: OHLCV, returns, momentum
├── Technical: RSI, MACD, Bollinger Bands
├── VIX: level, percentile, rate of change
├── Cross-asset: QQQ, TLT, UUP correlations
└── Volume: spike detection, relative volume

                ┌─────────────────────────────────┐
                │  CURRENT FEATURES [1 × D]       │
                │  (D = ~50-60 features)          │
                └─────────────────────────────────┘
                                │
                ┌───────────────┴───────────────┐
                ▼                               ▼
┌───────────────────────────┐   ┌───────────────────────────┐
│   RBF KERNEL LAYER        │   │   TCN TEMPORAL ENCODER    │
│   (Gaussian features)     │   │   (5 layers, dilation)    │
│                           │   │                           │
│   Input: [1, D]           │   │   Input: [60, D] sequence │
│   Output: [1, 125]        │   │   (60 timesteps = 1 hour) │
│   (25 centers × 5 scales) │   │   Output: [1, 64]         │
└───────────────────────────┘   └───────────────────────────┘
                │                               │
                └───────────────┬───────────────┘
                                ▼
                ┌─────────────────────────────────┐
                │  RESIDUAL BLOCKS (Bayesian)     │
                │  256 → 256 → 128 → 64           │
                │  (dropout, layer norm)          │
                └─────────────────────────────────┘
                                │
                ┌───────────────┼───────────────┐
                ▼               ▼               ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│  RETURN HEAD    │ │  DIRECTION HEAD │ │  CONFIDENCE     │
│  (Bayesian)     │ │  (Bayesian)     │ │  (Bayesian)     │
│                 │ │                 │ │                 │
│  predicted_     │ │  [DOWN, NEUTRAL │ │  0-1 score      │
│  return (%)     │ │   UP] probs     │ │  (sigmoid)      │
└─────────────────┘ └─────────────────┘ └─────────────────┘
```

### Neural Network Outputs

| Output | Type | Description |
|--------|------|-------------|
| `predicted_return` | float | Expected % return in prediction horizon |
| `predicted_volatility` | float | Expected volatility |
| `direction_probs` | [3] array | [DOWN, NEUTRAL, UP] probabilities |
| `confidence` | float 0-1 | Model's confidence in prediction |
| `risk_adjusted_return` | float | **Always use this, not raw return** |

### TCN Architecture (Temporal Convolutional Network)

```
Input: [Batch, 60 timesteps, D features]
       │
       ▼
┌──────────────────────────────────────────────────────┐
│  Layer 1: Conv1D (dilation=1)  → receptive field: 3  │
│  Layer 2: Conv1D (dilation=2)  → receptive field: 7  │
│  Layer 3: Conv1D (dilation=4)  → receptive field: 15 │
│  Layer 4: Conv1D (dilation=8)  → receptive field: 31 │
│  Layer 5: Conv1D (dilation=16) → receptive field: 63 │
└──────────────────────────────────────────────────────┘
       │
       ▼ (Attention pooling)
Output: [Batch, 64] context vector
```

---

## HMM Regime Detection

**Location:** `backend/multi_dimensional_hmm.py`

### Multi-Dimensional HMM (3×3×3 = 27 States)

```
              ┌──────────────────────────────────────────┐
              │           MARKET DATA                     │
              │  (returns, volatility, volume)            │
              └──────────────────────────────────────────┘
                                │
          ┌─────────────────────┼─────────────────────┐
          ▼                     ▼                     ▼
┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
│  TREND HMM      │   │  VOLATILITY HMM │   │  LIQUIDITY HMM  │
│  (2-7 states)   │   │  (2-7 states)   │   │  (2-7 states)   │
│                 │   │                 │   │                 │
│  Features:      │   │  Features:      │   │  Features:      │
│  • SMA slopes   │   │  • Realized vol │   │  • Volume ratio │
│  • Momentum     │   │  • VIX level    │   │  • Spread proxy │
│  • ROC          │   │  • ATR          │   │  • Tick volume  │
└─────────────────┘   └─────────────────┘   └─────────────────┘
          │                     │                     │
          ▼                     ▼                     ▼
    ┌───────────┐         ┌───────────┐         ┌───────────┐
    │ "Bullish" │         │ "Low"     │         │ "High"    │
    │ "Neutral" │         │ "Normal"  │         │ "Normal"  │
    │ "Bearish" │         │ "High"    │         │ "Low"     │
    └───────────┘         └───────────┘         └───────────┘
          │                     │                     │
          └─────────────────────┼─────────────────────┘
                                ▼
              ┌──────────────────────────────────────────┐
              │         COMBINED REGIME OUTPUT           │
              │         (27 possible states)             │
              └──────────────────────────────────────────┘
```

### HMM Outputs

| Output | Range | Description |
|--------|-------|-------------|
| `hmm_trend` | 0.0 - 1.0 | 0=Bearish, 0.5=Neutral, 1=Bullish |
| `hmm_volatility` | 0.0 - 1.0 | 0=Low, 0.5=Normal, 1=High |
| `hmm_liquidity` | 0.0 - 1.0 | 0=Low, 0.5=Normal, 1=High |
| `hmm_confidence` | 0.0 - 1.0 | Confidence in regime detection |

### HMM-Neural Alignment Rules

| HMM Trend | Neural Prediction | Result |
|-----------|------------------|--------|
| Bullish (>0.6) | CALL | ✅ ALIGNED - Trade with 1.15x confidence boost |
| Bullish (>0.6) | PUT | ❌ CONFLICT - Block trade |
| Bearish (<0.4) | PUT | ✅ ALIGNED - Trade with 1.15x confidence boost |
| Bearish (<0.4) | CALL | ❌ CONFLICT - Block trade |
| Neutral + High Vol | Any | ❌ CHOPPY - Block trade |

---

## Unified RL Policy

**Location:** `backend/unified_rl_policy.py`

### State Vector (18 Features)

```
┌─────────────────────────────────────────────────────────────────┐
│                 TradeState (18 features)                         │
├─────────────────────────────────────────────────────────────────┤
│ Position (4):                                                   │
│   is_in_trade, is_call, pnl%, max_drawdown                      │
│                                                                 │
│ Time (2):                                                       │
│   minutes_held, minutes_to_expiry                               │
│                                                                 │
│ Prediction (3):                                                 │
│   predicted_direction, confidence, momentum_5m                  │
│                                                                 │
│ Market (2):                                                     │
│   vix_level, volume_spike                                       │
│                                                                 │
│ HMM Regime (4):                                                 │
│   hmm_trend, hmm_volatility, hmm_liquidity, hmm_confidence      │
│                                                                 │
│ Greeks (2):                                                     │
│   theta_decay, delta                                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │     MODE CHECK                │
              │  trades < 50 → BANDIT MODE    │
              │  trades ≥ 50 → FULL RL MODE   │
              └───────────────────────────────┘
```

### Actions

| Action | Code | Description |
|--------|------|-------------|
| HOLD | 0 | Do nothing |
| BUY_CALL | 1 | Buy call options (bullish) |
| BUY_PUT | 2 | Buy put options (bearish) |
| EXIT | 3 | Close current position |

### Two Operating Modes

#### BANDIT MODE (First 50 trades)
- Uses rule-based heuristics
- 5% exploration rate for data gathering
- Quality gates still apply
- Purpose: Gather diverse training data before RL kicks in

#### FULL RL MODE (After 50 trades)
- Neural network policy decides actions
- Temperature-based action sampling
- Learns from experience buffer
- PPO-style training with advantage estimation

---

## Entry Quality Gates

**ALL must pass to enter a trade:**

| Gate | Threshold | Description |
|------|-----------|-------------|
| Confidence | ≥ 55% (65% after 2 losses) | Model must be confident |
| VIX Range | 13-30 | Avoid extreme volatility |
| Direction | ≥ 0.2% predicted move | Meaningful edge required |
| HMM Alignment | Trend matches prediction | Don't fight the trend |
| Not Choppy | !(neutral + high vol) | Avoid whipsaw markets |
| Volume | spike ≥ 0.5 | Market must be active |

---

## Exit Rules

### Priority Order (Hard Rules First)

```
1. HARD SAFETY RULES (Always First - Cannot be Overridden)
   ├── Stop Loss: -8%
   ├── Take Profit: +12%
   ├── Trailing Stop: +4% activation, 2% trail distance
   ├── Max Hold: 45 minutes
   └── Near Expiry: < 30 min to expiration

2. MODEL-BASED EXIT (If no safety rule triggered)
   └── XGBoost Exit Policy
       ├── 2.6M training experiences
       ├── 99.4% accuracy
       └── Exit threshold: 50%
```

### Exit Parameters

| Parameter | Value | Effect |
|-----------|-------|--------|
| `stop_loss_pct` | 8% | Cut losses at -8% |
| `take_profit_pct` | 12% | Take profits at +12% |
| `trailing_stop_activation` | 4% | Start trailing after +4% |
| `trailing_stop_distance` | 2% | Trail by 2% from high |
| `max_hold_minutes` | 45 | Force exit after 45 min |

---

## Current Production Model: dec_validation_v2

### Performance Metrics

| Metric | Value |
|--------|-------|
| **Win Rate** | **59.8%** (49 wins, 33 losses) |
| **P&L** | +$20,670.38 (+413.41%) |
| **Total Trades** | 61 |
| **Trade Rate** | 2.0% (rejects 98% of signals) |
| **Per-Trade P&L** | ~$339 |
| **Training Time** | 1,211 seconds (~20 minutes) |

### The Secret: Pre-trained Conservative Outputs

```
long_run_20k (20K cycles)  →  Learned conservative predictions
                           →  Loaded into dec_validation_v2
                           →  Conservative outputs reject 98% of signals
                           →  Only highest quality 2% trade
                           →  59.8% win rate
```

| Factor | Fresh Model | Pre-trained Model |
|--------|-------------|-------------------|
| Confidence outputs | High (50-80%) | Low (15-30%) |
| Signals passing gate | 14.5% | 2.0% |
| Trade quality | Random | High-quality only |
| Win rate | ~40% | ~60% |
| Per-trade P&L | -$100 to +$50 | **+$339** |

---

## Complete Decision Flow (Every 1-Minute Cycle)

```
Step 1: FETCH DATA
─────────────────────
Market Data (Tradier/Polygon)
    ↓
Build Features (50-60 features)
    ↓
Feature Buffer (last 60 timesteps)

Step 2: NEURAL NETWORK PREDICTION
─────────────────────────────────
Feature Buffer → UnifiedOptionsPredictor
    ↓
Outputs:
  • predicted_return: +0.3% (expects 0.3% up in 15min)
  • direction_probs: [0.15, 0.20, 0.65] (65% bullish)
  • confidence: 0.58 (58% confident)

Step 3: HMM REGIME CHECK
────────────────────────
Market Data → MultiDimensionalHMM
    ↓
Outputs:
  • hmm_trend: 0.72 (bullish regime)
  • hmm_volatility: 0.45 (normal vol)
  • hmm_confidence: 0.68

Step 4: SIGNAL COMBINATION
──────────────────────────
Neural Output + HMM Regime → SignalCombiner
    ↓
Combined Signal:
  • action: BUY_CALLS
  • confidence: 0.62 (boosted because aligned with HMM)

Step 5: UNIFIED RL DECISION
───────────────────────────
TradeState → UnifiedRLPolicy._bandit_decision()
    ↓
QUALITY GATES:
  ✓ Confidence 62% > 55% threshold
  ✓ VIX 18 is in range [13, 30]
  ✓ Direction +0.3% > 0.2% threshold
  ✓ HMM bullish (0.72) aligns with CALL
  ✓ Not in choppy market
    ↓
ACTION: BUY_CALL with 62% confidence

Step 6: POSITION MANAGEMENT (if in trade)
─────────────────────────────────────────
Current Position → check every minute
    ↓
Checks:
  • Current PnL: +5%
  • Max PnL seen: +7%
  • Minutes held: 25
  • HMM still bullish? Yes
    ↓
Trailing stop at 7% - 2% = 5%
PnL 5% = 5%, HOLD (at threshold)

Step 7: LEARNING (when trade closes)
────────────────────────────────────
Trade Result → record_step_reward()
    ↓
  • Store experience in buffer
  • Update win/loss stats
  • If buffer > 32 samples: train_step()
```

---

## Key Files Reference

| Component | File | Purpose |
|-----------|------|---------|
| Main Bot | `unified_options_trading_bot.py` | Orchestrates everything |
| Neural Network | `bot_modules/neural_networks.py` | Price/direction prediction |
| HMM Regime | `backend/multi_dimensional_hmm.py` | Market regime detection |
| RL Policy | `backend/unified_rl_policy.py` | Entry/exit decisions |
| XGBoost Exit | `backend/xgboost_exit_policy.py` | ML-based exit timing |
| Signals | `bot_modules/signals.py` | Signal combination |
| Paper Trading | `backend/paper_trading_system.py` | Trade execution simulation |
| Live Execution | `execution/tradier_adapter.py` | Tradier API integration |
| Feature Pipeline | `features/pipeline.py` | Feature computation |
| Data Sources | `backend/enhanced_data_sources.py` | Multi-source data fetching |

---

## Configuration Summary

### Entry Controller
- **Type:** Bandit (HMM-based)
- **Confidence Threshold:** 20% (pre-trained model outputs are conservative)
- **Edge Threshold:** 0.08%

### Exit Configuration
- **Stop Loss:** -8%
- **Take Profit:** +12%
- **Trailing Stop:** +4% activation, 2% trail
- **Max Hold:** 45 minutes

### Architecture Settings
- **Predictor:** v2_slim_bayesian
- **Temporal Encoder:** TCN (5 layers)
- **Sequence Length:** 60 timesteps (1 hour)
- **Hidden Dim:** 128
- **Dropout:** 0.15-0.20

---

## Live Trading Commands

### Start in Paper Mode
```bash
python go_live_only.py models/dec_validation_v2
```

### Switch to Live (Without Restart)
```bash
# Create flag file
echo LIVE > go_live.flag

# Or use helper script
go_live.bat
```

### Switch Back to Paper
```bash
# Delete flag file
del go_live.flag

# Or use helper script
go_paper.bat
```

### Start Dashboard
```bash
python training_dashboard_server.py
# Access at http://localhost:5001
```

---

## Architecture Diagram (Full)

```
                                 ┌─────────────────┐
                                 │   MARKET DATA   │
                                 │  (Tradier/etc)  │
                                 └────────┬────────┘
                                          │
                    ┌─────────────────────┼─────────────────────┐
                    ▼                     ▼                     ▼
           ┌────────────────┐    ┌────────────────┐    ┌────────────────┐
           │ FEATURE BUILDER│    │   VIX DATA     │    │  CROSS-ASSET   │
           │  (OHLCV, TA)   │    │ (Level, ROC)   │    │ (QQQ,TLT,UUP)  │
           └───────┬────────┘    └───────┬────────┘    └───────┬────────┘
                   │                     │                     │
                   └─────────────────────┼─────────────────────┘
                                         │
                                         ▼
                              ┌─────────────────────┐
                              │   FEATURE BUFFER    │
                              │  (60 timesteps)     │
                              └──────────┬──────────┘
                                         │
                   ┌─────────────────────┼─────────────────────┐
                   ▼                                           ▼
        ┌─────────────────────┐                    ┌─────────────────────┐
        │   NEURAL NETWORK    │                    │    HMM REGIME       │
        │ (UnifiedPredictor)  │                    │ (MultiDimHMM)       │
        │                     │                    │                     │
        │  Outputs:           │                    │  Outputs:           │
        │  • predicted_return │                    │  • hmm_trend        │
        │  • direction_probs  │                    │  • hmm_volatility   │
        │  • confidence       │                    │  • hmm_confidence   │
        └──────────┬──────────┘                    └──────────┬──────────┘
                   │                                          │
                   └────────────────────┬─────────────────────┘
                                        │
                                        ▼
                             ┌─────────────────────┐
                             │   SIGNAL COMBINER   │
                             │ (Confidence adjust) │
                             └──────────┬──────────┘
                                        │
                                        ▼
                             ┌─────────────────────┐
                             │  UNIFIED RL POLICY  │
                             │                     │
                             │  Quality Gates:     │
                             │  • Confidence > 55% │
                             │  • VIX in range     │
                             │  • HMM alignment    │
                             │  • Not choppy       │
                             └──────────┬──────────┘
                                        │
                          ┌─────────────┼─────────────┐
                          ▼             ▼             ▼
                    ┌─────────┐   ┌─────────┐   ┌─────────┐
                    │  HOLD   │   │  ENTER  │   │  EXIT   │
                    │         │   │ (CALL/  │   │         │
                    │         │   │  PUT)   │   │         │
                    └─────────┘   └────┬────┘   └────┬────┘
                                       │             │
                                       ▼             ▼
                             ┌─────────────────────────────┐
                             │    PAPER TRADING SYSTEM     │
                             │         (or Tradier)        │
                             │                             │
                             │  • Execute order            │
                             │  • Track position           │
                             │  • Monitor P&L              │
                             │  • Apply stops              │
                             └──────────────┬──────────────┘
                                            │
                                            ▼
                             ┌─────────────────────────────┐
                             │      LEARNING LOOP          │
                             │                             │
                             │  • Record experience        │
                             │  • Calculate rewards        │
                             │  • Train RL policy          │
                             │  • Update win/loss stats    │
                             └─────────────────────────────┘
```

---

*Document generated: January 2, 2026*
*Architecture version: V2 with Frozen Predictor*
*Production model: dec_validation_v2*
