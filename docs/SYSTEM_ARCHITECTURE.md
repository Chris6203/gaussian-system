# Unified Options Trading Bot - System Architecture

## Table of Contents
1. [Overview](#overview)
2. [Neural Network Architecture](#neural-network-architecture)
3. [HMM Regime Detection](#hmm-regime-detection)
4. [Unified RL Policy](#unified-rl-policy)
5. [Complete Decision Flow](#complete-decision-flow)
6. [Key Parameters to Tune](#key-parameters-to-tune)
7. [Debugging Win Rate](#debugging-win-rate)

---

## Overview

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                           UNIFIED OPTIONS TRADING BOT                                     │
│                              (unified_options_trading_bot.py)                             │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                           │
                    ┌──────────────────────┼──────────────────────┐
                    ▼                      ▼                      ▼
    ┌───────────────────────┐  ┌───────────────────────┐  ┌───────────────────────┐
    │   DATA PIPELINE       │  │   PREDICTION ENGINE   │  │   DECISION ENGINE     │
    │                       │  │                       │  │                       │
    │  • Tradier/Polygon    │  │  • Neural Network     │  │  • Signal Combiner    │
    │  • Feature Builder    │  │  • HMM Regime         │  │  • Unified RL Policy  │
    │  • VIX Data           │  │  • Multi-Timeframe    │  │  • Risk Manager       │
    └───────────────────────┘  └───────────────────────┘  └───────────────────────┘
```

### Key Files

| Component | File | Purpose |
|-----------|------|---------|
| Main Bot | `unified_options_trading_bot.py` | Orchestrates everything |
| Neural Network | `bot_modules/neural_networks.py` | Price/direction prediction |
| HMM Regime | `backend/multi_dimensional_hmm.py` | Market regime detection |
| RL Policy | `backend/unified_rl_policy.py` | Entry/exit decisions |
| Signals | `bot_modules/signals.py` | Signal combination |
| Paper Trading | `backend/paper_trading_system.py` | Trade execution simulation |

---

## Neural Network Architecture

### UnifiedOptionsPredictor

Location: `bot_modules/neural_networks.py`

```
INPUT FEATURES (per timestep):
├── Price data: OHLCV, returns, momentum
├── Technical: RSI, MACD, Bollinger Bands
├── VIX: level, percentile, rate of change
├── Cross-asset: QQQ, TLT, UUP correlations
└── Volume: spike detection, relative volume

                    ┌─────────────────────────────────────┐
                    │  CURRENT FEATURES [1 x D]           │
                    │  (D = ~50-100 features)             │
                    └─────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
    ┌───────────────────────────┐   ┌───────────────────────────┐
    │   RBF KERNEL LAYER        │   │   TEMPORAL ENCODER        │
    │   (Gaussian features)     │   │   (TCN or LSTM)           │
    │                           │   │                           │
    │   Input: [1, D]           │   │   Input: [60, D] sequence │
    │   Output: [1, 125]        │   │   (60 timesteps = 1 hour) │
    │   (25 centers × 5 scales) │   │   Output: [1, 64]         │
    └───────────────────────────┘   └───────────────────────────┘
                    │                               │
                    └───────────────┬───────────────┘
                                    ▼
                    ┌─────────────────────────────────────┐
                    │  CONCATENATED FEATURES              │
                    │  [1, D + 125 + 64]                  │
                    └─────────────────────────────────────┘
                                    │
                                    ▼
                    ┌─────────────────────────────────────┐
                    │  RESIDUAL BLOCKS (Bayesian)         │
                    │  256 → 256 → 128 → 64               │
                    │  (with dropout, layer norm)         │
                    └─────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
    ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
    │  RETURN HEAD    │ │  DIRECTION HEAD │ │  CONFIDENCE HEAD│
    │  (Bayesian)     │ │  (Bayesian)     │ │  (Bayesian)     │
    │                 │ │                 │ │                 │
    │  Output:        │ │  Output:        │ │  Output:        │
    │  predicted_     │ │  [DOWN, NEUTRAL │ │  0-1 score      │
    │  return (%)     │ │   UP] probs     │ │  (sigmoid)      │
    └─────────────────┘ └─────────────────┘ └─────────────────┘
```

### TCN (Temporal Convolutional Network)

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

### Neural Network Outputs

| Output | Type | Description |
|--------|------|-------------|
| `predicted_return` | float | Expected % return in prediction horizon |
| `predicted_volatility` | float | Expected volatility |
| `direction_probs` | [3] array | [DOWN, NEUTRAL, UP] probabilities |
| `confidence` | float 0-1 | Model's confidence in prediction |
| `fillability` | float 0-1 | Probability of getting filled |
| `exp_slippage` | float | Expected slippage in $ |

---

## HMM Regime Detection

### MultiDimensionalHMM

Location: `backend/multi_dimensional_hmm.py`

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
              └──────────────────────────────────────────┘
```

### HMM Outputs

| Output | Range | Description |
|--------|-------|-------------|
| `hmm_trend` | 0.0 - 1.0 | 0=Bearish, 0.5=Neutral, 1=Bullish |
| `hmm_volatility` | 0.0 - 1.0 | 0=Low, 0.5=Normal, 1=High |
| `hmm_liquidity` | 0.0 - 1.0 | 0=Low, 0.5=Normal, 1=High |
| `hmm_confidence` | 0.0 - 1.0 | Confidence in regime detection |

### How HMM Affects Trading

| HMM Trend | Neural Prediction | Result |
|-----------|------------------|--------|
| Bullish (>0.6) | CALL | ✅ ALIGNED - Trade with 1.15x confidence boost |
| Bullish (>0.6) | PUT | ❌ CONFLICT - Block trade |
| Bearish (<0.4) | PUT | ✅ ALIGNED - Trade with 1.15x confidence boost |
| Bearish (<0.4) | CALL | ❌ CONFLICT - Block trade |
| Neutral + High Vol | Any | ❌ CHOPPY - Block trade |

---

## Unified RL Policy

### Architecture

Location: `backend/unified_rl_policy.py`

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    UnifiedRLPolicy                                           │
└─────────────────────────────────────────────────────────────────────────────┘

                    ┌─────────────────────────────────────┐
                    │         TradeState (18 features)    │
                    ├─────────────────────────────────────┤
                    │ Position (4):                       │
                    │   is_in_trade, is_call, pnl%,      │
                    │   max_drawdown                      │
                    │ Time (2):                           │
                    │   minutes_held, minutes_to_expiry   │
                    │ Prediction (3):                     │
                    │   predicted_direction, confidence,  │
                    │   momentum_5m                       │
                    │ Market (2):                         │
                    │   vix_level, volume_spike           │
                    │ HMM Regime (4):                     │
                    │   hmm_trend, hmm_volatility,       │
                    │   hmm_liquidity, hmm_confidence     │
                    │ Greeks (2):                         │
                    │   theta_decay, delta                │
                    └─────────────────────────────────────┘
                                    │
                                    ▼
                    ┌─────────────────────────────────────┐
                    │     MODE CHECK                      │
                    │  total_trades < 50?                 │
                    │     YES → BANDIT MODE               │
                    │     NO  → FULL RL MODE              │
                    └─────────────────────────────────────┘
```

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

### Actions

| Action | Code | Description |
|--------|------|-------------|
| HOLD | 0 | Do nothing |
| BUY_CALL | 1 | Buy call options (bullish) |
| BUY_PUT | 2 | Buy put options (bearish) |
| EXIT | 3 | Close current position |

### Safety Checks (Always Applied)

```python
# These ALWAYS trigger, regardless of RL decision:

1. STOP LOSS: Exit if PnL <= -8%
2. TRAILING STOP: If max_profit >= 8%, exit if PnL drops 4% from max
3. PROFIT TARGET: Exit if PnL >= 15% (only if trailing not active)
4. EXPIRY: Exit if < 30 minutes to expiration
5. TIME DECAY: Exit if held > 90 min with < 3% profit
```

### Entry Quality Gates

```python
# ALL of these must pass to enter a trade:

1. Confidence >= 55% (or 65% after 2 consecutive losses)
2. VIX between 13 and 30
3. |predicted_direction| >= 0.2% (20 basis points predicted move)
4. HMM trend aligns with prediction
5. Not in choppy market (neutral trend + high volatility)
6. Volume spike >= 0.5 (not dead market)
```

---

## Complete Decision Flow

### Every 1-Minute Cycle

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         EVERY 1-MINUTE CYCLE                                 │
└─────────────────────────────────────────────────────────────────────────────┘

Step 1: FETCH DATA
─────────────────────
Market Data (Tradier/Polygon)
    ↓
Build Features (50-100 features)
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
  ✓ Direction +0.3% > 0.2% threshold (20 basis points)
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
  • Momentum still positive? Yes
    ↓
Trailing stop at 7% - 4% = 3%
PnL 5% > 3%, so HOLD (don't exit yet)

Step 7: LEARNING (when trade closes)
────────────────────────────────────
Trade Result → record_step_reward()
    ↓
  • Store experience in buffer
  • Update win/loss stats
  • If buffer > 32 samples: train_step()
  • Adjust consecutive_losses counter
```

---

## Key Parameters to Tune

### Entry Filters

| Parameter | Current Value | Effect |
|-----------|---------------|--------|
| `min_confidence_to_trade` | 0.55 | Higher = fewer but better trades |
| `min_direction_threshold` | 0.02 | Higher = require stronger signals |
| `vix_range` | [13, 30] | Avoid extreme volatility environments |
| `require_hmm_alignment` | True | Only trade WITH the trend |

### Exit Parameters

| Parameter | Current Value | Effect |
|-----------|---------------|--------|
| `stop_loss_pct` | 0.08 (8%) | Tighter = cut losses faster |
| `trailing_stop_activation` | 0.08 (8%) | When to start trailing |
| `trailing_stop_distance` | 0.04 (4%) | How much to trail by |
| `profit_target_pct` | 0.15 (15%) | Fixed profit target |
| `max_hold_minutes` | 90 | Force exit after this time |

### Neural Network

| Parameter | Current Value | Effect |
|-----------|---------------|--------|
| `sequence_length` | 60 | How many timesteps to look at |
| `hidden_dim` | 128 | Network capacity |
| `learning_rate` | 0.0001 | Lower = more stable training |
| `dropout` | 0.2-0.35 | Higher = more regularization |

### RL Policy

| Parameter | Current Value | Effect |
|-----------|---------------|--------|
| `bandit_mode_trades` | 50 | Trades before RL takes over |
| `exploration_rate` | 0.05 | Random action probability |
| `gamma` | 0.99 | Future reward discount |
| `entropy_coef` | 0.05 | Exploration encouragement |

---

## Debugging Win Rate

### 1. Check Neural Network Quality

**What to look for:**
- `predicted_return` should vary (not always ~0)
- `confidence` should vary (not stuck at 0.5)
- Direction probabilities should show conviction (not always 0.33/0.33/0.33)

**How to check:**
```python
# Add this logging to see predictions:
logger.info(f"NEURAL: return={pred_return:.3%}, conf={confidence:.1%}, "
            f"dir=[{dirs[0]:.1%},{dirs[1]:.1%},{dirs[2]:.1%}]")
```

### 2. Check HMM Regime Quality

**What to look for:**
- `hmm_trend` should vary (not always 0.5)
- `hmm_confidence` should be high (>0.6) when trend is clear
- Regime should match actual price action

**How to check:**
```python
logger.info(f"HMM: trend={hmm_trend:.2f}, vol={hmm_vol:.2f}, "
            f"conf={hmm_conf:.1%}")
```

### 3. Check Decision Quality

**What to look for:**
- Trades only taken when prediction ALIGNS with HMM trend
- Weak signals being filtered out
- Not taking random exploration trades too often

**How to check:**
```python
logger.info(f"DECISION: action={action}, reason={details['reason']}, "
            f"gates_passed={details.get('setup_quality', 'N/A')}")
```

### 4. Check Exit Timing

**What to look for:**
- Are stops being hit too quickly? (Might need wider stops)
- Are profits being taken too early? (Let winners run longer)
- Are positions held too long? (Theta eating profits)

**Key metrics to track:**
- Average win size vs average loss size
- Average hold time for winners vs losers
- % of trades hitting stop loss vs profit target vs time exit

### 5. Common Issues and Fixes

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| All losses | Stops too tight | Increase `stop_loss_pct` |
| Small wins, big losses | Not cutting losses | Decrease `stop_loss_pct` |
| Many small losses | Taking bad trades | Increase `min_confidence_to_trade` |
| Missing good trades | Filters too strict | Decrease thresholds slightly |
| Theta eating profits | Holding too long | Decrease `max_hold_minutes` |
| Whipsawed in choppy | Trading against trend | Ensure HMM alignment is enforced |

---

## Reward Shaping

The RL policy learns from rewards. Here's how rewards are calculated:

### Step Rewards (every minute while in trade)

```python
reward = 0.0

# 1. P&L Change (main signal)
pnl_change = current_pnl - previous_pnl
reward += pnl_change * 15.0  # 1% gain = +0.15

# 2. Theta Decay Cost
theta_per_minute = abs(theta_decay) / 390
reward -= theta_per_minute * 3.0  # Time costs money

# 3. Drawdown Penalty
if drawdown_increased:
    reward -= new_drawdown * 5.0  # Don't let losses run

# 4. Momentum Alignment Bonus
if position_aligned_with_momentum:
    reward += 0.01  # Small bonus for being with the trend
```

### Exit Rewards

```python
# Winning exits
if pnl > 10%: reward += 0.5   # Excellent
if pnl > 5%:  reward += 0.3   # Good
if pnl > 0%:  reward += 0.1   # Small win

# Losing exits
if pnl > -5%: reward += 0.1   # Good loss management
if pnl < -8%: reward -= 0.2   # Let it run too far

# Time efficiency
if pnl > 0 and held < 30min: reward += 0.15  # Quick win
if held < 5min and |pnl| < 2%: reward -= 0.25  # Churning
```

---

## Architecture Diagram (ASCII)

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

## Quick Reference

### Winning Trade Flow
```
1. Neural predicts +0.5% return with 65% confidence
2. HMM shows bullish trend (0.75) with 70% confidence
3. Signal combined → BUY_CALL with 72% effective confidence
4. RL policy checks gates → All pass ✓
5. Enter CALL position
6. Price moves up → PnL hits +10%
7. Trailing stop activates at 8%
8. Price pulls back to +7%
9. Still above trailing stop (10% - 4% = 6%)
10. Price recovers to +12%
11. Price drops to +9%
12. Below trailing stop (12% - 4% = 8%)
13. EXIT → +9% profit recorded
14. RL learns: "This pattern = good entry"
```

### Losing Trade Flow
```
1. Neural predicts -0.3% return with 58% confidence
2. HMM shows bearish trend (0.28) with 65% confidence
3. Signal combined → BUY_PUT with 63% effective confidence
4. RL policy checks gates → All pass ✓
5. Enter PUT position
6. Price doesn't drop as expected → PnL at -3%
7. HMM trend flips to neutral (0.52)
8. RL detects trend reversal → EXIT signal
9. EXIT → -3% loss recorded (cut early!)
10. RL learns: "Exit when HMM flips"
```

---

*Last updated: December 2024*

