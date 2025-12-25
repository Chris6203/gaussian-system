# Neural Network Architecture Reference

**Purpose**: This document provides a complete reference of the neural network architecture used in the Gaussian Options Trading Bot. Use this to verify implementations, debug issues, or cross-check with other AI systems.

**Last Updated**: 2025-12-22

---

## System Overview

The trading bot uses a two-phase architecture:

1. **Phase 1: Predictor** - Supervised learning for price/direction prediction
2. **Phase 2: RL Policy** - Reinforcement learning for entry/exit decisions

Key design principle: The predictor is **frozen during RL training** to prevent conflicting gradients.

---

## 1. Predictor Architecture (`UnifiedOptionsPredictorV2`)

**File**: `bot_modules/neural_networks.py`

### Input Specification

| Input | Shape | Description |
|-------|-------|-------------|
| `cur` | `[B, D]` | Current feature vector (D ≈ 50-100 features) |
| `seq` | `[B, T, D]` | Historical sequence (T=60 timesteps @ 1min) |

### Architecture Flow

```
Current Features [B, D]    Sequence Features [B, T, D]
       │                           │
       │                    ┌──────┴──────┐
       │                    │ TCN Encoder │
       │                    │ (5 layers)  │
       │                    └──────┬──────┘
       │                           │
       │                    Temporal Context [B, 64]
       │                           │
       ├───────────────────────────┤
       │                           │
       │    ┌─────────────────┐    │
       │    │ RBF Kernels     │    │ (optional, disabled by default)
       │    │ 25 centers × 5  │    │
       │    │ scales = 125    │    │
       │    └────────┬────────┘    │
       │             │             │
       └─────────────┼─────────────┘
                     │
            Concatenate → [B, D + 64 + (125 or 0)]
                     │
            ┌────────┴────────┐
            │ Input Projection│
            │ Linear(→256)    │
            │ LayerNorm, GELU │
            └────────┬────────┘
                     │
            ┌────────┴────────┐
            │ SlimResidualBlock │ 256→256, dropout=0.20
            └────────┬────────┘
                     │
            ┌────────┴────────┐
            │ SlimResidualBlock │ 256→128, dropout=0.15
            └────────┬────────┘
                     │
            ┌────────┴────────┐
            │ SlimResidualBlock │ 128→64, dropout=0.10
            └────────┬────────┘
                     │
            ┌────────┴────────┐
            │   Head Common   │ Linear(64→64), LayerNorm, GELU
            └────────┬────────┘
                     │
        ┌────────────┼────────────┬──────────────┐
        │            │            │              │
   ┌────┴────┐ ┌─────┴─────┐ ┌────┴────┐  ┌─────┴─────┐
   │ Return  │ │ Direction │ │Confidence│  │Execution │
   │ Head    │ │ Head (3)  │ │ Head    │  │ Heads    │
   └────┬────┘ └─────┬─────┘ └────┬────┘  └─────┬────┘
        │            │            │              │
    [B, 1]      [B, 3]       [B, 1]         [B, 3]
```

### Component Details

#### 1.1 TCN Encoder (`OptionsTCN`)

```python
class OptionsTCN:
    """
    Temporal Convolutional Network for sequence encoding.

    Parameters:
        input_dim: Feature dimension (D)
        hidden_dim: 128 (internal channels)
        num_layers: 5 (TCN blocks)
        kernel_size: 3
        dropout: 0.15 (V2) or 0.20 (V1)

    Architecture:
        1. Input projection: [B, T, D] → [B, T, 128]
        2. Transpose for conv: [B, T, 128] → [B, 128, T]
        3. TCN blocks with dilations [1, 2, 4, 8, 16]
        4. Transpose back: [B, 128, T] → [B, T, 128]
        5. Attention pooling: [B, T, 128] → [B, 128]
        6. Context projection: [B, 128] → [B, 64]

    Receptive field: 1 + 2*(k-1)*sum(dilations) = 1 + 4*(1+2+4+8+16) = 125 timesteps
    Note: With 2 convs per block and k=3, RF covers full T=60 sequence
    """
```

#### 1.2 TCN Block

```python
class TCNBlock:
    """
    Single TCN block with dilated causal convolutions.

    Key features:
    - Causal padding (no future information leakage)
    - Dilated convolutions for exponential receptive field growth
    - Residual connection with optional projection

    Forward:
        1. Conv1d with dilation → remove right padding (causal)
        2. BatchNorm → GELU → Dropout
        3. Conv1d with dilation → remove right padding (causal)
        4. BatchNorm → GELU → Dropout
        5. Add residual (with projection if channel mismatch)
    """
```

#### 1.3 Bayesian Linear Layer

```python
class BayesianLinear:
    """
    Linear layer with learnable weight uncertainty.

    Parameters:
        weight_mu: Mean of weights [out, in]
        weight_logvar: Log-variance of weights [out, in]
        bias_mu: Mean of bias [out]
        bias_logvar: Log-variance of bias [out]

    Forward (reparameterization trick):
        weight_std = exp(0.5 * weight_logvar)
        weight = weight_mu + weight_std * epsilon  # epsilon ~ N(0,1)
        bias_std = exp(0.5 * bias_logvar)
        bias = bias_mu + bias_std * epsilon
        return linear(x, weight, bias)

    Purpose: Provides epistemic uncertainty estimates
    """
```

#### 1.4 RBF Kernel Layer (Optional, Disabled by Default)

```python
class RBFKernelLayer:
    """
    Radial Basis Function feature expansion.

    Parameters:
        n_centers: 25 (learnable cluster centers)
        scales: [0.1, 0.5, 1.0, 2.0, 5.0] (bandwidth parameters)

    Output dimension: 25 * 5 = 125 features

    Forward:
        For each scale s and center c:
            dist2 = ||x - c||^2
            output = exp(-dist2 / (2 * s^2))

    Note: Currently DISABLED in config to reduce overfitting
    """
```

### Output Specification

| Output | Shape | Activation | Description |
|--------|-------|------------|-------------|
| `embedding` | `[B, 64]` | None | Latent representation for downstream models |
| `return` | `[B, 1]` | None | Predicted return (raw value) |
| `volatility` | `[B, 1]` | None | Predicted volatility |
| `direction` | `[B, 3]` | Softmax | [DOWN, NEUTRAL, UP] probabilities |
| `confidence` | `[B, 1]` | Sigmoid | Model confidence (0-1) |
| `fillability` | `[B, 1]` | Sigmoid | P(fill at mid-peg) |
| `exp_slippage` | `[B, 1]` | None | Expected slippage in dollars |
| `exp_ttf` | `[B, 1]` | ReLU | Expected time-to-fill (seconds) |

---

## 2. RL Policy Architecture (`UnifiedRLPolicy`)

**File**: `backend/unified_rl_policy.py`

### State Vector (18 Features)

| Category | Features | Description |
|----------|----------|-------------|
| Position (4) | `is_in_trade`, `is_call`, `pnl_pct`, `drawdown` | Current position state |
| Time (2) | `minutes_held`, `minutes_to_expiry` | Temporal context |
| Prediction (3) | `predicted_direction`, `confidence`, `momentum_5m` | From predictor |
| Market (2) | `vix_level`, `volume_spike` | Market conditions |
| HMM Regime (4) | `hmm_trend`, `hmm_volatility`, `hmm_liquidity`, `hmm_confidence` | Regime detection |
| Greeks (2) | `theta_decay`, `delta` | Options Greeks |

### Network Architecture

```
State Vector [B, 18]
       │
┌──────┴──────┐
│   Features  │
│ Linear(18→64)│
│ LayerNorm   │
│ ReLU        │
│ Dropout(0.1)│
│ Linear(64→64)│
│ LayerNorm   │
│ ReLU        │
└──────┬──────┘
       │
   Features [B, 64]
       │
┌──────┴──────┬────────────┬─────────────┐
│             │            │             │
│  ┌──────────┴─────────┐  │  ┌──────────┴─────────┐
│  │   Action Head      │  │  │   Value Head       │
│  │ Linear(64→32)→ReLU │  │  │ Linear(64→32)→ReLU │
│  │ Linear(32→4)       │  │  │ Linear(32→1)       │
│  └──────────┬─────────┘  │  └──────────┬─────────┘
│             │            │             │
│         [B, 4]           │          [B, 1]
│    (action logits)       │    (state value)
│                          │
│        ┌─────────────────┴───────────┐
│        │      Exit Urgency Head      │
│        │ Linear(64→16) → ReLU        │
│        │ Linear(16→1) → Sigmoid      │
│        └──────────────┬──────────────┘
│                       │
│                    [B, 1]
│               (exit urgency 0-1)
```

### Action Space

| Action | Value | Description |
|--------|-------|-------------|
| HOLD | 0 | Do nothing / maintain position |
| BUY_CALL | 1 | Enter long call position |
| BUY_PUT | 2 | Enter long put position |
| EXIT | 3 | Close current position |

### Operating Modes

#### Bandit Mode (First 50 Trades)

```python
# Rule-based heuristics with 5% exploration
if random() < 0.05:  # Explore
    return random_action()
else:  # Exploit based on rules
    if in_trade and (stop_loss_hit or take_profit_hit):
        return EXIT
    if not in_trade and high_confidence_signal:
        return BUY_CALL/BUY_PUT based on direction
    return HOLD
```

#### Full RL Mode (After 50 Trades)

```python
# Neural network policy with temperature sampling
action_logits, value, exit_urgency = network(state)
action_probs = softmax(action_logits / temperature)
action = sample(action_probs)
```

---

## 3. HMM Regime Detection

**File**: `backend/multi_dimensional_hmm.py`

### Architecture

```
Market Features
      │
┌─────┴─────┐
│  3×3×3    │
│    HMM    │
│  (27 states)│
└─────┬─────┘
      │
┌─────┴─────────────────────────────┐
│                                    │
│  trend: 0=bearish, 0.5=neutral, 1=bullish
│  volatility: 0=low, 0.5=normal, 1=high
│  liquidity: 0=low, 0.5=normal, 1=high
│  confidence: 0-1 (regime detection confidence)
│
└────────────────────────────────────┘
```

### Regime States

| Dimension | States | Description |
|-----------|--------|-------------|
| Trend | 3 | Bearish (0), Neutral (1), Bullish (2) |
| Volatility | 3 | Low (0), Normal (1), High (2) |
| Liquidity | 3 | Low (0), Normal (1), High (2) |

---

## 4. Exit Policy Architecture

**File**: `backend/unified_exit_manager.py`

### Exit Decision Flow

```
Position Update
      │
      ▼
┌─────────────────────────────────────┐
│         HARD SAFETY RULES           │  ← ALWAYS RUN FIRST
│  (config: architecture.exit_policy) │
├─────────────────────────────────────┤
│  1. Stop Loss: -8% → EXIT           │
│  2. Take Profit: +12% → EXIT        │
│  3. Max Hold: 45 min → FORCE CLOSE  │
│  4. Trailing Stop: +8% → trail 4%   │
│  5. Expiry: <30min to expire → EXIT │
└─────────────────┬───────────────────┘
                  │ (if no hard exit)
                  ▼
┌─────────────────────────────────────┐
│      MODEL-BASED EXIT (Optional)    │
│         XGBoost / Neural            │
├─────────────────────────────────────┤
│  • Confidence decay detection       │
│  • Momentum reversal signals        │
│  • Theta decay vs expected gain     │
└─────────────────────────────────────┘
```

---

## 5. Data Flow Summary

```
Historical Data (1-min bars)
         │
         ▼
┌───────────────────┐
│ Feature Pipeline  │ → 50-100 features
│ (features/*.py)   │
└────────┬──────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
Current    Sequence
[B, D]    [B, 60, D]
    │         │
    └────┬────┘
         │
         ▼
┌───────────────────┐
│    Predictor      │
│ (UnifiedOptions   │
│  PredictorV2)     │
└────────┬──────────┘
         │
         ├─→ embedding [B, 64]
         ├─→ direction [B, 3]
         ├─→ confidence [B, 1]
         └─→ return, volatility, etc.
                    │
                    ▼
         ┌───────────────────┐
         │  Build RL State   │ → 18 features
         │  + HMM Regime     │
         │  + Greeks         │
         └────────┬──────────┘
                  │
                  ▼
         ┌───────────────────┐
         │   RL Policy       │
         │ (UnifiedRLPolicy) │
         └────────┬──────────┘
                  │
         Action: HOLD/BUY_CALL/BUY_PUT/EXIT
```

---

## 6. Key Configuration Parameters

**File**: `config.json`

| Parameter | Location | Current Value | Description |
|-----------|----------|---------------|-------------|
| `interval` | `data_sources.interval` | `1m` | Data bar interval |
| `sequence_length` | `neural_network.sequence_length` | `60` | Timesteps in sequence |
| `use_gaussian_kernels` | `neural_network.use_gaussian_kernels` | `false` | RBF layer enabled |
| `hard_stop_loss_pct` | `architecture.exit_policy` | `-8.0` | Stop loss threshold |
| `hard_take_profit_pct` | `architecture.exit_policy` | `12.0` | Take profit threshold |
| `hard_max_hold_minutes` | `architecture.exit_policy` | `45` | Max hold time |
| `prediction_minutes` | `architecture.horizons` | `15` | Prediction horizon |

---

## 7. V2 Architecture Improvements (NEW)

**File**: `backend/unified_rl_policy_v2.py`

### Key Improvements

| Improvement | V1 | V2 |
|-------------|----|----|
| State features | 18 scalars | 40 (18 scalars + 16 embedding + 6 probs/heads) |
| Predictor info | 3 scalars (direction, confidence, momentum) | Full 64-dim embedding compressed to 16 |
| Temporal awareness | None (MLP) | GRU over last 10 states |
| Entry gating | Raw confidence threshold | EV-based (friction-aware) |
| Exit actions | Single EXIT | EXIT_FAST (market) + EXIT_PATIENT (limit) |
| Reward shaping | Sparse | Time penalty + drawdown penalty + theta cost |

### V2 State Vector (40 features)

```
Scalars (24):
├── Position (4): in_trade, is_call, pnl%, drawdown
├── Time (2): minutes_held, minutes_to_expiry
├── Market (2): vix, volume_spike
├── HMM (4): trend, volatility, liquidity, confidence
├── Greeks (2): theta, delta
├── Direction probs (3): P(DOWN), P(NEUTRAL), P(UP)
├── Execution (3): fillability, slippage, ttf
├── Return/Vol (2): predicted_return, predicted_volatility
└── Momentum/Conf (2): momentum_5m, raw_confidence

Embedding (16):
└── Compressed from predictor's 64-dim embedding via learned adapter
```

### V2 Action Space (5 actions)

| Action | Value | Description |
|--------|-------|-------------|
| HOLD | 0 | Do nothing |
| BUY_CALL | 1 | Enter call position |
| BUY_PUT | 2 | Enter put position |
| EXIT_FAST | 3 | Close immediately (market order) |
| EXIT_PATIENT | 4 | Close patiently (limit order, wait for fill) |

### EV-Based Entry Gating

```python
# Instead of: if confidence > 0.55: trade
# V2 uses:
def compute_EV(state, action):
    p_up = direction_probs[2]
    p_down = direction_probs[0]
    expected_move = abs(predicted_return)
    friction = spread + slippage + theta_cost

    if action == BUY_CALL:
        return p_up * expected_move - p_down * expected_move - friction
    else:  # BUY_PUT
        return p_down * expected_move - p_up * expected_move - friction

# Only trade if EV > 0
```

### Multi-Horizon Predictor (V3)

**File**: `bot_modules/neural_networks.py` → `UnifiedOptionsPredictorV3`

Predicts returns and directions for multiple horizons: {5m, 15m, 30m, 45m}

```python
# Per-horizon outputs:
result['return_5m'], result['direction_5m'], result['confidence_5m']
result['return_15m'], result['direction_15m'], result['confidence_15m']
result['return_30m'], result['direction_30m'], result['confidence_30m']
result['return_45m'], result['direction_45m'], result['confidence_45m']

# RL can choose which horizon has best edge
best_horizon, edge = predictor.get_best_horizon(predictions)
```

---

## 8. Known Architectural Issues

### Issue 1: Horizon Misalignment (ADDRESSED in V3)

**Problem**: Prediction horizon (15 min) << Max hold time (45 min)
**Impact**: Positions drift after prediction "expires"
**Location**: `config.json` → `architecture.horizons` vs `exit_policy`

### Issue 2: Confidence Miscalibration

**Problem**: Neural outputs 20-35% confidence, thresholds at 55%+
**Impact**: Very few trades pass the gate
**Location**: `trading.base_confidence_threshold` vs actual neural outputs

### Issue 3: Exit Ratio Asymmetry

**Problem**: -8%/+12% = 1.5:1 ratio needs 40% win rate to break even
**Impact**: Current ~35% win rate loses money even with "good" entries
**Location**: `architecture.exit_policy`

---

## 8. Verification Checklist

Use this checklist to verify the architecture is working correctly:

- [ ] Predictor outputs `direction` as 3-class softmax (not binary)
- [ ] RL state has exactly 18 features
- [ ] TCN has 5 layers with dilations [1, 2, 4, 8, 16]
- [ ] Hard safety rules run BEFORE model-based exit
- [ ] Predictor is frozen (`requires_grad=False`) during RL training
- [ ] HMM regime values are normalized to [0, 1] range
- [ ] Confidence head uses Sigmoid activation

---

## 9. Code Locations Quick Reference

| Component | File | Class/Function |
|-----------|------|----------------|
| Predictor V1 | `bot_modules/neural_networks.py` | `UnifiedOptionsPredictor` |
| Predictor V2 | `bot_modules/neural_networks.py` | `UnifiedOptionsPredictorV2` |
| TCN Encoder | `bot_modules/neural_networks.py` | `OptionsTCN` |
| TCN Block | `bot_modules/neural_networks.py` | `TCNBlock` |
| Bayesian Layer | `bot_modules/neural_networks.py` | `BayesianLinear` |
| RBF Kernels | `bot_modules/neural_networks.py` | `RBFKernelLayer` |
| RL Policy | `backend/unified_rl_policy.py` | `UnifiedRLPolicy` |
| Policy Network | `backend/unified_rl_policy.py` | `UnifiedPolicyNetwork` |
| HMM | `backend/multi_dimensional_hmm.py` | `MultiDimensionalHMM` |
| Exit Manager | `backend/unified_exit_manager.py` | `UnifiedExitManager` |
| Feature Pipeline | `features/pipeline.py` | `FeaturePipeline` |
