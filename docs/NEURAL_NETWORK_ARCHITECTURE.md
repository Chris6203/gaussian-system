# Neural Network Architecture - Output3 Trading Bot

This document describes all neural network architectures used in the trading bot for debugging purposes.

---

## Table of Contents
1. [UnifiedOptionsPredictor](#1-unifiedoptionspredictor) - Main prediction model
2. [OptionsTCN](#2-optionstcn) - Temporal Convolutional Network
3. [OptionsLSTM](#3-optionslstm) - Legacy LSTM encoder
4. [UnifiedPolicyNetwork (RL)](#4-unifiedpolicynetwork) - Unified RL policy
5. [RLTradingPolicy (PPO)](#5-rltradingpolicy-ppo) - Entry decisions
6. [ExitPolicyNetwork (RL Exit)](#6-exitpolicynetwork) - Exit timing
7. [XGBoostExitPolicy](#7-xgboostexitpolicy) - XGBoost-based exit (non-neural)
8. [Supporting Layers](#8-supporting-layers)

---

## 1. UnifiedOptionsPredictor

**File:** `bot_modules/neural_networks.py`  
**Purpose:** Main prediction model combining temporal encoding with Bayesian prediction heads.

### Architecture Diagram

```
                                    ┌─────────────────────────────────────────────────┐
                                    │           UnifiedOptionsPredictor               │
                                    └─────────────────────────────────────────────────┘
                                                          │
                    ┌─────────────────────────────────────┼─────────────────────────────────────┐
                    │                                     │                                     │
                    ▼                                     ▼                                     ▼
        ┌───────────────────┐               ┌───────────────────┐               ┌───────────────────┐
        │   Current Features │               │  Sequence Features │               │   RBF Features     │
        │    cur: [B, D]     │               │   seq: [B, T, D]   │               │   (if enabled)     │
        └─────────┬─────────┘               └─────────┬─────────┘               └─────────┬─────────┘
                  │                                   │                                   │
                  │                                   ▼                                   │
                  │                       ┌───────────────────┐                           │
                  │                       │  Temporal Encoder  │                           │
                  │                       │  (TCN or LSTM)     │                           │
                  │                       │   → [B, 64]        │                           │
                  │                       └─────────┬─────────┘                           │
                  │                                 │                                     │
                  │                                 │                    ┌────────────────┘
                  │                                 │                    │
                  │                                 │                    ▼
                  │                                 │        ┌───────────────────┐
                  │                                 │        │  RBFKernelLayer   │
                  │                                 │        │  n_centers=25     │
                  │                                 │        │  scales=[5]       │
                  │                                 │        │   → [B, 125]      │
                  │                                 │        └─────────┬─────────┘
                  │                                 │                  │
                  └─────────────────────────────────┼──────────────────┘
                                                    │
                                                    ▼
                                        ┌───────────────────┐
                                        │     Concatenate    │
                                        │ [D + 64 + 125]     │
                                        └─────────┬─────────┘
                                                  │
                                                  ▼
                                    ┌─────────────────────────┐
                                    │      Input Projection    │
                                    │  Linear(combined, 256)   │
                                    │  LayerNorm(256)          │
                                    │  GELU()                  │
                                    └───────────┬─────────────┘
                                                │
                                                ▼
                                    ┌─────────────────────────┐
                                    │     ResidualBlock 1      │
                                    │  256 → 256, dropout=0.35 │
                                    └───────────┬─────────────┘
                                                │
                                                ▼
                                    ┌─────────────────────────┐
                                    │     ResidualBlock 2      │
                                    │  256 → 128, dropout=0.30 │
                                    └───────────┬─────────────┘
                                                │
                                                ▼
                                    ┌─────────────────────────┐
                                    │     ResidualBlock 3      │
                                    │  128 → 64, dropout=0.20  │
                                    └───────────┬─────────────┘
                                                │
                                                ▼
                                    ┌─────────────────────────┐
                                    │      Head Common         │
                                    │  Linear(64, 64)          │
                                    │  LayerNorm(64)           │
                                    │  GELU()                  │
                                    └───────────┬─────────────┘
                                                │
                ┌───────────┬───────────┬───────┴───────┬───────────┬───────────┬───────────┐
                ▼           ▼           ▼               ▼           ▼           ▼           ▼
        ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐
        │return_head│ │ vol_head  │ │ dir_head  │ │ conf_head │ │fill_head  │ │slip_head  │ │ ttf_head  │
        │Bayesian(1)│ │Bayesian(1)│ │Bayesian(3)│ │Bayesian(1)│ │Bayesian(1)│ │Bayesian(1)│ │Bayesian(1)│
        │  (return) │ │(volatility│ │(direction)│ │(confidence│ │(fillability│ │(slippage) │ │(time-to-  │
        │           │ │           │ │ DOWN/     │ │   0-1)    │ │           │ │           │ │   fill)   │
        │           │ │           │ │ NEUTRAL/  │ │ sigmoid   │ │  sigmoid  │ │           │ │  ReLU     │
        │           │ │           │ │   UP)     │ │           │ │           │ │           │ │           │
        └───────────┘ └───────────┘ └───────────┘ └───────────┘ └───────────┘ └───────────┘ └───────────┘
```

### Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `feature_dim` | Required | Number of input features |
| `sequence_length` | 60 | Temporal sequence length |
| `use_gaussian_kernels` | True | Enable RBF kernel features |
| `use_mamba` | False | Use TCN (True) or LSTM (False) |

### Output Dictionary

```python
{
    "return": torch.Tensor,      # Predicted return [B, 1]
    "volatility": torch.Tensor,  # Predicted volatility [B, 1]
    "direction": torch.Tensor,   # Direction probs [B, 3] (DOWN, NEUTRAL, UP)
    "confidence": torch.Tensor,  # Confidence score [B, 1] (0-1, sigmoid)
    "fillability": torch.Tensor, # P(fill at mid-peg) [B, 1] (0-1, sigmoid)
    "exp_slippage": torch.Tensor,# Expected slippage $ [B, 1]
    "exp_ttf": torch.Tensor,     # Expected time-to-fill seconds [B, 1] (ReLU)
}
```

---

## 2. OptionsTCN

**File:** `bot_modules/neural_networks.py`  
**Purpose:** Temporal Convolutional Network for time series encoding. Faster than LSTM (parallelizable).

### Architecture Diagram

```
                    ┌─────────────────────────────────────────────────┐
                    │                  OptionsTCN                      │
                    │   input_dim → hidden_dim=128 → output_dim=64    │
                    └─────────────────────────────────────────────────┘
                                            │
                                            ▼
                            ┌───────────────────────────────┐
                            │       Input: [B, T, D]         │
                            └───────────────┬───────────────┘
                                            │
                                            ▼
                            ┌───────────────────────────────┐
                            │      Input Projection          │
                            │   Linear(input_dim, 128)       │
                            │   LayerNorm(128)               │
                            │        → [B, T, 128]           │
                            └───────────────┬───────────────┘
                                            │
                                            ▼
                            ┌───────────────────────────────┐
                            │     Transpose: [B, 128, T]     │
                            └───────────────┬───────────────┘
                                            │
            ┌───────────────────────────────┼───────────────────────────────┐
            │                               │                               │
            ▼                               ▼                               ▼
    ┌───────────────┐               ┌───────────────┐               ┌───────────────┐
    │  TCN Block 0  │               │  TCN Block 1  │               │  TCN Block 2  │
    │  dilation=1   │      →        │  dilation=2   │      →        │  dilation=4   │
    │  (kernel=3)   │               │  (kernel=3)   │               │  (kernel=3)   │
    └───────────────┘               └───────────────┘               └───────────────┘
            │                               │                               │
            └───────────────────────────────┼───────────────────────────────┘
                                            │
            ┌───────────────────────────────┼───────────────────────────────┐
            │                               │                               │
            ▼                               ▼                               ▼
    ┌───────────────┐               ┌───────────────┐
    │  TCN Block 3  │               │  TCN Block 4  │
    │  dilation=8   │      →        │  dilation=16  │
    │  (kernel=3)   │               │  (kernel=3)   │
    └───────────────┘               └───────────────┘
                                            │
                                            ▼
                            ┌───────────────────────────────┐
                            │     Transpose: [B, T, 128]     │
                            │        LayerNorm(128)          │
                            └───────────────┬───────────────┘
                                            │
                                            ▼
                            ┌───────────────────────────────┐
                            │       Attention Pooling        │
                            │   Linear(128, 64) → Tanh       │
                            │   Linear(64, 1) → Softmax      │
                            │   Weighted sum → [B, 128]      │
                            └───────────────┬───────────────┘
                                            │
                                            ▼
                            ┌───────────────────────────────┐
                            │      Context Projection        │
                            │   Linear(128, 128)             │
                            │   LayerNorm(128) → GELU        │
                            │   Dropout(0.2)                 │
                            │   Linear(128, 64)              │
                            │        → [B, 64]               │
                            └───────────────────────────────┘
```

### TCN Block Detail

```
        ┌─────────────────────────────────────────┐
        │              TCN Block                   │
        │   (dilated causal convolution)          │
        └─────────────────────────────────────────┘
                          │
            Input: [B, C_in, T]
                          │
        ┌─────────────────┴─────────────────┐
        │                                   │
        ▼                                   ▼
┌───────────────────┐               ┌───────────────────┐
│   Conv1d Layer 1   │               │   Residual Path   │
│   kernel=3         │               │   (Conv1d 1x1 if  │
│   dilation=d       │               │    C_in ≠ C_out)  │
│   causal padding   │               └─────────┬─────────┘
└─────────┬─────────┘                         │
          │                                   │
          ▼                                   │
┌───────────────────┐                         │
│   BatchNorm1d      │                         │
│   GELU             │                         │
│   Dropout(0.2)     │                         │
└─────────┬─────────┘                         │
          │                                   │
          ▼                                   │
┌───────────────────┐                         │
│   Conv1d Layer 2   │                         │
│   kernel=3         │                         │
│   dilation=d       │                         │
│   causal padding   │                         │
└─────────┬─────────┘                         │
          │                                   │
          ▼                                   │
┌───────────────────┐                         │
│   BatchNorm1d      │                         │
│   GELU             │                         │
│   Dropout(0.2)     │                         │
└─────────┬─────────┘                         │
          │                                   │
          └───────────────┬───────────────────┘
                          │
                          ▼
                    ┌───────────┐
                    │   Add     │
                    │  (x + res)│
                    └───────────┘
```

### Receptive Field Calculation

With 5 layers and dilations [1, 2, 4, 8, 16]:
- Receptive field ≈ 2^5 × kernel_size ≈ **90+ timesteps**

---

## 3. OptionsLSTM

**File:** `bot_modules/neural_networks.py`  
**Purpose:** Legacy LSTM encoder (fallback, slower than TCN).

### Architecture Diagram

```
                    ┌─────────────────────────────────────────────────┐
                    │                  OptionsLSTM                     │
                    │   Bidirectional LSTM with Attention Pooling     │
                    └─────────────────────────────────────────────────┘
                                            │
                                            ▼
                            ┌───────────────────────────────┐
                            │       Input: [B, T, D]         │
                            └───────────────┬───────────────┘
                                            │
                                            ▼
                            ┌───────────────────────────────┐
                            │     Bidirectional LSTM         │
                            │   hidden_dim=128               │
                            │   num_layers=3                 │
                            │   dropout=0.35                 │
                            │        → [B, T, 256]           │
                            │   (256 = 128 × 2 directions)   │
                            └───────────────┬───────────────┘
                                            │
                                            ▼
                            ┌───────────────────────────────┐
                            │       LayerNorm(256)           │
                            └───────────────┬───────────────┘
                                            │
                                            ▼
                            ┌───────────────────────────────┐
                            │       Attention Pooling        │
                            │   Linear(256, 128)             │
                            │   LayerNorm(128) → Tanh        │
                            │   Dropout(0.2)                 │
                            │   Linear(128, 1) → Softmax     │
                            │   Weighted sum → [B, 256]      │
                            └───────────────┬───────────────┘
                                            │
                                            ▼
                            ┌───────────────────────────────┐
                            │      Context Projection        │
                            │   Linear(256, 128)             │
                            │   LayerNorm(128) → GELU        │
                            │   Dropout(0.3)                 │
                            │   Linear(128, 64)              │
                            │        → [B, 64]               │
                            └───────────────────────────────┘
```

---

## 4. UnifiedPolicyNetwork

**File:** `backend/unified_rl_policy.py`  
**Purpose:** Single policy for ENTRY, HOLD, and EXIT decisions.

### Architecture Diagram

```
                    ┌─────────────────────────────────────────────────┐
                    │            UnifiedPolicyNetwork                  │
                    │     state_dim=18 → hidden_dim=64 → 4 actions    │
                    └─────────────────────────────────────────────────┘
                                            │
                                            ▼
                            ┌───────────────────────────────┐
                            │     State Vector (18 features) │
                            │  • Position (4): in_trade,     │
                            │    is_call, pnl%, drawdown     │
                            │  • Time (2): mins_held,        │
                            │    mins_to_expiry              │
                            │  • Prediction (3): direction,  │
                            │    confidence, momentum        │
                            │  • Market (2): vix, vol_spike  │
                            │  • HMM (4): trend, vol, liq,   │
                            │    hmm_conf                    │
                            │  • Greeks (2): theta, delta    │
                            └───────────────┬───────────────┘
                                            │
                                            ▼
                            ┌───────────────────────────────┐
                            │     Shared Feature Extractor   │
                            │   Linear(18, 64)               │
                            │   LayerNorm(64) → ReLU         │
                            │   Dropout(0.1)                 │
                            │   Linear(64, 64)               │
                            │   LayerNorm(64) → ReLU         │
                            └───────────────┬───────────────┘
                                            │
                ┌───────────────────────────┼───────────────────────────┐
                │                           │                           │
                ▼                           ▼                           ▼
    ┌───────────────────┐       ┌───────────────────┐       ┌───────────────────┐
    │    Action Head     │       │    Value Head      │       │  Exit Urgency Head │
    │   Linear(64, 32)   │       │   Linear(64, 32)   │       │   Linear(64, 16)   │
    │   ReLU             │       │   ReLU             │       │   ReLU             │
    │   Linear(32, 4)    │       │   Linear(32, 1)    │       │   Linear(16, 1)    │
    │                    │       │                    │       │   Sigmoid          │
    │ [HOLD, CALL, PUT,  │       │  State value V(s)  │       │   (0-1 urgency)    │
    │       EXIT]        │       │                    │       │                    │
    └───────────────────┘       └───────────────────┘       └───────────────────┘
```

### State Features (18 total)

| Category | Features | Count |
|----------|----------|-------|
| Position | `is_in_trade`, `is_call`, `pnl%`, `drawdown` | 4 |
| Time | `minutes_held`, `minutes_to_expiry` | 2 |
| Prediction | `direction`, `confidence`, `momentum_5m` | 3 |
| Market | `vix_level`, `volume_spike` | 2 |
| HMM Regime | `trend`, `volatility`, `liquidity`, `confidence` | 4 |
| Greeks | `theta_decay`, `delta` | 2 |

### Actions

| Index | Action | Description |
|-------|--------|-------------|
| 0 | HOLD | Do nothing |
| 1 | BUY_CALL | Buy call option |
| 2 | BUY_PUT | Buy put option |
| 3 | EXIT | Close position |

---

## 5. RLTradingPolicy (PPO)

**File:** `backend/rl_trading_policy.py`  
**Purpose:** Proximal Policy Optimization for entry decisions.

### Architecture Diagram

```
                    ┌─────────────────────────────────────────────────┐
                    │            RLTradingPolicy (PPO)                 │
                    │     state_dim=32 → hidden_dim=256 → 5 actions   │
                    └─────────────────────────────────────────────────┘
                                            │
                                            ▼
                            ┌───────────────────────────────┐
                            │     State Vector (32 features) │
                            │  • predicted_return × 100      │
                            │  • predicted_volatility × 100  │
                            │  • confidence                  │
                            │  • direction_probs[3]          │
                            │  • balance_log                 │
                            │  • position_ratio              │
                            │  • pnl_ratio                   │
                            │  • vix_normalized              │
                            │  • momentum_5min × 100         │
                            │  • momentum_15min × 100        │
                            │  • volume_spike / 2            │
                            │  • rsi_normalized              │
                            │  • win_rate_last_10            │
                            │  • avg_return_last_10          │
                            │  • recent_drawdown             │
                            │  • trades_today / 10           │
                            │  • market_open_progress        │
                            │  • days_until_expiry / 7       │
                            │  • is_last_hour                │
                            │  • hmm_trend, hmm_vol, hmm_liq │
                            │  • vix_bb_pos, vix_roc, vix_% │
                            │  • price_jerk                  │
                            │  • context_trend_alignment     │
                            │  • tech/crypto/sector momentum │
                            └───────────────┬───────────────┘
                                            │
                                            ▼
                            ┌───────────────────────────────┐
                            │    SharedFeatureExtractor      │
                            │   Linear(32, 256)              │
                            │   LayerNorm(256) → ReLU        │
                            │   Dropout(0.2)                 │
                            │   Linear(256, 128)             │
                            │   LayerNorm(128) → ReLU        │
                            │   Dropout(0.1)                 │
                            │        → [B, 128]              │
                            └───────────────┬───────────────┘
                                            │
                ┌───────────────────────────┴───────────────────────────┐
                │                                                       │
                ▼                                                       ▼
    ┌───────────────────────────────┐               ┌───────────────────────────────┐
    │       ActorNetwork             │               │       CriticNetwork            │
    │   (uses shared features)       │               │   (uses shared features)       │
    │                                │               │                                │
    │   Linear(128, 64)              │               │   Linear(128, 64)              │
    │   LayerNorm(64) → ReLU         │               │   LayerNorm(64) → ReLU         │
    │   Dropout(0.1)                 │               │   Dropout(0.1)                 │
    │   Linear(64, 32) → ReLU        │               │   Linear(64, 32) → ReLU        │
    │   Linear(32, 5)                │               │   Linear(32, 1)                │
    │                                │               │                                │
    │   Output: action logits        │               │   Output: state value V(s)     │
    └───────────────────────────────┘               └───────────────────────────────┘
```

### Actions (5 total)

| Index | Action | Position Size |
|-------|--------|---------------|
| 0 | HOLD | 0 |
| 1 | BUY_CALL_1X | 1 |
| 2 | BUY_CALL_2X | 2 |
| 3 | BUY_PUT_1X | 1 |
| 4 | BUY_PUT_2X | 2 |

### PPO Hyperparameters

| Parameter | Value |
|-----------|-------|
| `learning_rate` | 0.0003 |
| `gamma` | 0.99 |
| `entropy_coef` | 0.05 |
| `ppo_clip_range` | 0.2 |
| `ppo_epochs` | 4 |
| `max_grad_norm` | 0.5 |

---

## 6. ExitPolicyNetwork

**File:** `backend/rl_exit_policy.py`  
**Purpose:** RL-based exit timing for options trades.

### Architecture Diagram

```
                    ┌─────────────────────────────────────────────────┐
                    │            ExitPolicyNetwork                     │
                    │   input_dim=20 → hidden=64 → exit_score (0-1)   │
                    └─────────────────────────────────────────────────┘
                                            │
                                            ▼
                            ┌───────────────────────────────┐
                            │     Input Features (20 total)  │
                            │                                │
                            │ BASE (8):                      │
                            │  • time_ratio (held/predicted) │
                            │  • pnl_normalized (-1 to 1)    │
                            │  • days_to_expiry_norm         │
                            │  • move_completion             │
                            │  • entry_confidence            │
                            │  • is_winning (0/1)            │
                            │  • past_predicted_time (0/1)   │
                            │  • time_remaining_ratio        │
                            │                                │
                            │ EXPANDED (12):                 │
                            │  • live_pred_change            │
                            │  • conf_ratio                  │
                            │  • vix_normalized              │
                            │  • spread_normalized           │
                            │  • signal_alignment            │
                            │  • signal_conflict             │
                            │  • hmm_trend (0-1)             │
                            │  • hmm_vol (0-1)               │
                            │  • hmm_liq (0-1)               │
                            │  • vix_bb_pos                  │
                            │  • vix_roc                     │
                            │  • vix_percentile              │
                            └───────────────┬───────────────┘
                                            │
                                            ▼
                            ┌───────────────────────────────┐
                            │       Input Projection         │
                            │   Linear(20, 64)               │
                            │   LayerNorm(64) → ReLU         │
                            └───────────────┬───────────────┘
                                            │
                                            ▼
                            ┌───────────────────────────────┐
                            │         Hidden Block 1         │
                            │   Linear(64, 64)               │
                            │   LayerNorm(64) → ReLU         │
                            │   Dropout(0.25)                │
                            │   + Scaled Residual (×0.1)     │
                            └───────────────┬───────────────┘
                                            │
                                            ▼
                            ┌───────────────────────────────┐
                            │         Hidden Block 2         │
                            │   Linear(64, 32)               │
                            │   LayerNorm(32) → ReLU         │
                            │   Dropout(0.2)                 │
                            └───────────────┬───────────────┘
                                            │
                ┌───────────────────────────┴───────────────────────────┐
                │                                                       │
                ▼                                                       ▼
    ┌───────────────────────────────┐               ┌───────────────────────────────┐
    │         Exit Head              │               │      Hold Time Head           │
    │   Linear(32, 16) → ReLU        │               │   Linear(32, 16) → ReLU       │
    │   Linear(16, 1) → Sigmoid      │               │   Linear(16, 1) → Softplus    │
    │                                │               │                                │
    │   Output: exit_score (0-1)     │               │   Output: optimal_hold_mins   │
    │   0 = HOLD, 1 = EXIT NOW       │               │   (non-negative)              │
    └───────────────────────────────┘               └───────────────────────────────┘
```

### Exit Decision Thresholds

| Condition | Threshold | Action |
|-----------|-----------|--------|
| Big Profit | ≥ 40% | Force EXIT |
| Critical Loss | ≤ -20% | Force EXIT |
| Near Expiry | < 1 day | Force EXIT |
| RL Score | ≥ 0.6 (dynamic) | EXIT |
| RL Score | < threshold | HOLD |

---

## 7. XGBoostExitPolicy

**File:** `backend/xgboost_exit_policy.py`  
**Purpose:** XGBoost classifier for exit decisions (non-neural, interpretable).

### Feature Vector (12 features)

```
┌─────────────────────────────────────────────────────────────────┐
│                    XGBoost Features                              │
├─────────────────────────────────────────────────────────────────┤
│  1. pnl_pct           - Current P&L percentage                  │
│  2. time_held_minutes - How long position held                  │
│  3. days_to_expiry    - Days until option expires               │
│  4. vix_level         - VIX volatility index                    │
│  5. predicted_move_pct- Original predicted move                 │
│  6. actual_move_pct   - Actual move so far                      │
│  7. entry_confidence  - Confidence at entry                     │
│  8. high_water_mark   - Best P&L seen this trade                │
│  9. drawdown_from_high- Current drawdown from peak              │
│ 10. time_ratio        - time_held / prediction_timeframe        │
│ 11. is_call           - 1 for CALL, 0 for PUT                   │
│ 12. signal_alignment  - Current signal alignment (-1 to 1)      │
└─────────────────────────────────────────────────────────────────┘
```

### XGBoost Configuration

| Parameter | Value |
|-----------|-------|
| `n_estimators` | 50 |
| `max_depth` | 4 |
| `learning_rate` | 0.1 |
| `min_child_weight` | 3 |
| `subsample` | 0.8 |
| `exit_probability_threshold` | 0.55 |

---

## 8. Supporting Layers

### BayesianLinear

**Purpose:** Provides uncertainty estimates via weight sampling.

```
┌─────────────────────────────────────────────────────────────────┐
│                     BayesianLinear Layer                         │
├─────────────────────────────────────────────────────────────────┤
│  Parameters:                                                     │
│    • weight_mu      [out, in] - Mean of weight distribution     │
│    • weight_logvar  [out, in] - Log-variance of weights         │
│    • bias_mu        [out]     - Mean of bias distribution       │
│    • bias_logvar    [out]     - Log-variance of biases          │
│                                                                  │
│  Forward (Reparameterization Trick):                            │
│    weight = weight_mu + exp(0.5 * weight_logvar) * ε_w          │
│    bias   = bias_mu   + exp(0.5 * bias_logvar)   * ε_b          │
│    output = x @ weight.T + bias                                  │
│                                                                  │
│  Benefits:                                                       │
│    • Epistemic uncertainty estimation                            │
│    • Natural regularization                                      │
│    • Prevents overconfident predictions                          │
└─────────────────────────────────────────────────────────────────┘
```

### RBFKernelLayer

**Purpose:** Non-linear feature expansion using Radial Basis Functions.

```
┌─────────────────────────────────────────────────────────────────┐
│                     RBFKernelLayer                               │
├─────────────────────────────────────────────────────────────────┤
│  Parameters:                                                     │
│    • centers   [n_centers, input_dim] - Learnable RBF centers   │
│    • log_scales[n_scales]             - Learnable scale params  │
│                                                                  │
│  Default Configuration:                                          │
│    • n_centers = 25                                              │
│    • scales = [0.1, 0.5, 1.0, 2.0, 5.0]                         │
│    • output_dim = 25 × 5 = 125                                   │
│                                                                  │
│  Forward:                                                        │
│    for each scale s:                                             │
│      dist² = ||x - centers||²                                    │
│      rbf = exp(-dist² / (2 * s²))                               │
│    output = concat(all rbf outputs)                              │
└─────────────────────────────────────────────────────────────────┘
```

### ResidualBlock

**Purpose:** Stable deep training with skip connections.

```
┌─────────────────────────────────────────────────────────────────┐
│                     ResidualBlock                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Input ───────────────────────────────────┐                     │
│    │                                      │                     │
│    ▼                                      │                     │
│  BayesianLinear(in_dim, out_dim)         │                     │
│    │                                      │                     │
│    ▼                                      │                     │
│  LayerNorm(out_dim)                       │                     │
│    │                                      │ (skip connection    │
│    ▼                                      │  with optional      │
│  GELU()                                   │  projection)        │
│    │                                      │                     │
│    ▼                                      │                     │
│  Dropout(dropout)                         │                     │
│    │                                      ▼                     │
│    │                            Linear(in, out) if needed       │
│    │                                      │                     │
│    └──────────────┬───────────────────────┘                     │
│                   │                                              │
│                   ▼                                              │
│            output + residual × 0.1                               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Summary: Model Sizes

| Model | Parameters (approx) | Purpose |
|-------|--------------------:|---------|
| UnifiedOptionsPredictor | ~200K | Main prediction |
| OptionsTCN | ~150K | Time series encoding |
| OptionsLSTM | ~400K | Legacy time series |
| UnifiedPolicyNetwork | ~15K | RL entry/exit |
| RLTradingPolicy | ~100K | PPO entry |
| ExitPolicyNetwork | ~8K | RL exit timing |
| XGBoostExitPolicy | ~2K trees | XGB exit timing |

---

## Data Flow Summary

```
Market Data
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Feature Computation                           │
│  (FeatureComputer: technical, volatility, volume, cross-asset)  │
└─────────────────────────────────────────────────────────────────┘
    │
    ├───────────────────────────────────────────────┐
    │                                               │
    ▼                                               ▼
┌───────────────────────┐               ┌───────────────────────┐
│ UnifiedOptionsPredictor│               │ HMM Regime Models      │
│  (TCN/LSTM + Bayesian) │               │  (trend/vol/liquidity) │
└───────────────────────┘               └───────────────────────┘
    │                                               │
    │   predictions                                 │   regime info
    └───────────────────────────┬───────────────────┘
                                │
                                ▼
                ┌───────────────────────────────────┐
                │         Policy Decision            │
                │  (UnifiedRLPolicy or PPO Policy)  │
                └───────────────────────────────────┘
                                │
            ┌───────────────────┴───────────────────┐
            │                                       │
            ▼                                       ▼
    ┌───────────────────┐               ┌───────────────────┐
    │   ENTRY Decision   │               │   EXIT Decision    │
    │   BUY_CALL/PUT/HOLD│               │  (ExitPolicy or    │
    │                    │               │   XGBoost)         │
    └───────────────────┘               └───────────────────┘
            │                                       │
            └───────────────────┬───────────────────┘
                                │
                                ▼
                        Trade Execution
```

---

## Debug Checklist

When debugging the neural networks, check:

1. **Input Shapes**: Verify feature dimensions match expected input_dim
2. **NaN/Inf Values**: Check for numerical instability in features
3. **Gradient Flow**: Monitor gradients for vanishing/exploding
4. **Confidence Calibration**: Are confidence scores meaningful?
5. **Action Distribution**: Is policy collapsing to single action?
6. **Loss Curves**: Track training/validation loss
7. **Feature Importance**: Use XGBoost feature importance for insights
8. **Regime Detection**: Verify HMM states make sense

### Logging Debug Info

```python
# Enable debug logging for neural networks
import logging
logging.getLogger('bot_modules.neural_networks').setLevel(logging.DEBUG)
logging.getLogger('backend.unified_rl_policy').setLevel(logging.DEBUG)
logging.getLogger('backend.rl_exit_policy').setLevel(logging.DEBUG)
```



