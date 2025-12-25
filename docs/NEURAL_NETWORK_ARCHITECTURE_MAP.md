# Neural Network Architecture Map - Output3 Trading Bot

**Complete mapping of all neural network architectures in the output3 folder**

**Last Updated:** 2024  
**Purpose:** Comprehensive reference for debugging, understanding, and extending the trading bot's neural components

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Core Prediction Models](#core-prediction-models)
3. [Reinforcement Learning Policies](#reinforcement-learning-policies)
4. [Supporting Components](#supporting-components)
5. [Regime Detection Models](#regime-detection-models)
6. [Entry Controllers](#entry-controllers)
7. [Data Flow Summary](#data-flow-summary)
8. [Model Sizes & Parameters](#model-sizes--parameters)

---

## Architecture Overview

The trading bot uses **two main neural subsystems**:

### 1. **Predictor System** (Supervised Learning)
- **Purpose:** Predict market movements, volatility, direction, and execution quality
- **Models:** `UnifiedOptionsPredictor`, `UnifiedOptionsPredictorV2`, `DirectionPredictor`, `DirectionPredictorV3`
- **File:** `bot_modules/neural_networks.py`

### 2. **Policy System** (Reinforcement Learning)
- **Purpose:** Make trading decisions (entry/exit/hold)
- **Models:** `RLTradingPolicy` (PPO), `UnifiedRLPolicy`, `ExitPolicyNetwork`
- **Files:** `backend/rl_trading_policy.py`, `backend/unified_rl_policy.py`, `backend/rl_exit_policy.py`

---

## Core Prediction Models

### 1. UnifiedOptionsPredictor

**File:** `bot_modules/neural_networks.py`  
**Purpose:** Main prediction model combining temporal encoding with Bayesian prediction heads

#### Architecture Flow

```
Input: Current Features [B, D] + Sequence [B, T, D]
    │
    ├─→ Temporal Encoder (TCN or LSTM) → [B, 64]
    │
    ├─→ Current Features [B, D]
    │
    └─→ RBF Kernel Layer (optional) → [B, 125]
    
    ↓ Concatenate
    [B, D + 64 + 125]
    
    ↓ Input Projection
    Linear(D+64+125, 256) → LayerNorm → GELU
    
    ↓ Residual Block 1
    ResidualBlock(256 → 256, dropout=0.35)
    
    ↓ Residual Block 2
    ResidualBlock(256 → 128, dropout=0.30)
    
    ↓ Residual Block 3
    ResidualBlock(128 → 64, dropout=0.20)
    
    ↓ Head Common
    Linear(64, 64) → LayerNorm → GELU
    
    ↓ Multi-Head Outputs
    ├─→ return_head: BayesianLinear(64, 1)
    ├─→ vol_head: BayesianLinear(64, 1)
    ├─→ dir_head: BayesianLinear(64, 3)  [DOWN, NEUTRAL, UP]
    ├─→ conf_head: BayesianLinear(64, 1) → Sigmoid
    ├─→ fillability_head: BayesianLinear(64, 1) → Sigmoid
    ├─→ slippage_head: BayesianLinear(64, 1)
    └─→ ttf_head: BayesianLinear(64, 1) → ReLU
```

#### Key Components

**Temporal Encoder Options:**

1. **OptionsTCN** (Preferred)
   - 5 TCN blocks with dilations [1, 2, 4, 8, 16]
   - Receptive field: ~90 timesteps
   - Attention pooling
   - Output: [B, 64]

2. **OptionsLSTM** (Legacy)
   - Bidirectional LSTM (3 layers, hidden_dim=128)
   - Attention pooling
   - Output: [B, 64]

**RBF Kernel Layer:**
- 25 centers × 5 scales = 125 features
- Non-linear feature expansion

**Residual Blocks:**
- Use `BayesianLinear` for uncertainty estimation
- Scaled residual connections (×0.1)

#### Output Dictionary

```python
{
    "embedding": [B, 64],      # Shared latent representation
    "return": [B, 1],          # Predicted return
    "volatility": [B, 1],      # Predicted volatility
    "direction": [B, 3],       # [P(DOWN), P(NEUTRAL), P(UP)]
    "confidence": [B, 1],      # 0-1 confidence score
    "fillability": [B, 1],    # P(fill at mid-peg)
    "exp_slippage": [B, 1],   # Expected slippage ($)
    "exp_ttf": [B, 1]         # Expected time-to-fill (seconds)
}
```

---

### 2. UnifiedOptionsPredictorV2 (Slim Bayesian)

**File:** `bot_modules/neural_networks.py`  
**Purpose:** Reduced regularization variant for better pattern learning

#### Key Differences from V1

- **Deterministic backbone:** No `BayesianLinear` in residual blocks (uses standard `nn.Linear`)
- **Lower dropout:** 0.20 → 0.15 → 0.10 (vs 0.35 → 0.30 → 0.20)
- **Bayesian only in heads:** Uncertainty estimation only in final prediction heads
- **More capacity:** Better for capturing complex patterns

#### Architecture

Same structure as V1, but:
- `SlimResidualBlock` instead of `ResidualBlock`
- Lower dropout rates
- Deterministic backbone with Bayesian heads

---

### 3. DirectionPredictor

**File:** `bot_modules/neural_networks.py`  
**Purpose:** Dedicated direction-only model with multi-scale temporal analysis

#### Architecture

```
Input: Current [B, D] + Sequence [B, T, D]
    │
    ├─→ TCN_1m (last 15 bars) → [B, 64]
    ├─→ TCN_5m (subsampled) → [B, 64]
    ├─→ TCN_15m (subsampled) → [B, 64]
    └─→ RBF Layer → [B, 125]
    
    ↓ Concatenate
    [B, D + 125 + 64×3] = [B, D + 317]
    
    ↓ Direction Network
    Linear(317, 512) → LayerNorm → GELU → Dropout(0.15)
    Linear(512, 256) → LayerNorm → GELU → Dropout(0.10)
    Linear(256, 128) → LayerNorm → GELU → Dropout(0.05)
    Linear(128, 64) → LayerNorm → GELU
    
    ↓ Temporal Attention
    MultiheadAttention(64, num_heads=4)
    
    ↓ Output Heads
    ├─→ direction_logits: Linear(64, 2) → Softmax [DOWN, UP]
    ├─→ confidence_head: Linear(64, 1) → Sigmoid
    └─→ magnitude_head: Linear(64, 1) → Abs
```

**Features:**
- Multi-scale temporal analysis (1m, 5m, 15m)
- Temporal attention mechanism
- Binary output (UP vs DOWN, no NEUTRAL)
- Target: 70%+ direction accuracy

---

### 4. DirectionPredictorV3

**File:** `bot_modules/neural_networks.py`  
**Purpose:** Simplified direction-only model for better generalization

#### Architecture

```
Input: Current [B, D] + Sequence [B, T, D] (T=30)
    │
    └─→ Lightweight TCN (3 blocks, dilations 1,2,4)
        TCNBlock(D, 64, dilation=1)
        TCNBlock(64, 64, dilation=2)
        TCNBlock(64, 64, dilation=4)
        → [B, T, 64]
    
    ↓ Attention Pooling
    Linear(64, 32) → Tanh → Linear(32, 1) → Softmax
    → Weighted sum → [B, 64]
    
    ↓ Concatenate with Current
    [B, D + 64]
    
    ↓ Backbone (Deterministic)
    Linear(D+64, 128) → LayerNorm → GELU → Dropout(0.15)
    Linear(128, 64) → LayerNorm → GELU → Dropout(0.10)
    
    ↓ Output Heads
    ├─→ direction_head: Linear(64, 2) → Softmax [DOWN, UP]
    └─→ confidence_head: Linear(64, 1) → Sigmoid
```

**Key Design Principles:**
- **NO RBF kernels** (removed overfitting source)
- Single lightweight TCN (3 blocks vs 5)
- Reduced sequence length (30 vs 60)
- Binary output only (UP vs DOWN)
- Deterministic backbone, Bayesian only on output
- Target: 55-60% direction accuracy

---

## Reinforcement Learning Policies

### 1. RLTradingPolicy (PPO)

**File:** `backend/rl_trading_policy.py`  
**Purpose:** Proximal Policy Optimization for entry decisions

#### Architecture

```
State Vector [32 features]
    │
    └─→ SharedFeatureExtractor
        Linear(32, 256) → LayerNorm → ReLU → Dropout(0.2)
        Linear(256, 128) → LayerNorm → ReLU → Dropout(0.1)
        → [B, 128]
    
    ├─→ ActorNetwork
    │   Linear(128, 64) → LayerNorm → ReLU → Dropout(0.1)
    │   Linear(64, 32) → ReLU
    │   Linear(32, 5) → Action logits
    │   [HOLD, BUY_CALL_1X, BUY_CALL_2X, BUY_PUT_1X, BUY_PUT_2X]
    │
    └─→ CriticNetwork
        Linear(128, 64) → LayerNorm → ReLU → Dropout(0.1)
        Linear(64, 32) → ReLU
        Linear(32, 1) → State value V(s)
```

#### State Features (32 total)

1. **Prediction Features (4):**
   - predicted_return × 100
   - predicted_volatility × 100
   - confidence
   - direction_probs[3]

2. **Account Features (3):**
   - balance_log
   - position_ratio
   - pnl_ratio

3. **Market Features (6):**
   - vix_normalized
   - momentum_5min × 100
   - momentum_15min × 100
   - volume_spike / 2
   - rsi_normalized
   - price_jerk

4. **Performance Features (3):**
   - win_rate_last_10
   - avg_return_last_10
   - recent_drawdown

5. **Time Features (3):**
   - trades_today / 10
   - market_open_progress
   - days_until_expiry / 7
   - is_last_hour

6. **HMM Features (4):**
   - hmm_trend
   - hmm_vol
   - hmm_liq
   - hmm_confidence

7. **VIX Extended (3):**
   - vix_bb_pos
   - vix_roc
   - vix_percentile

8. **Context Features (4):**
   - context_trend_alignment
   - tech_momentum × 100
   - crypto_momentum × 100
   - sector_rotation × 100

#### PPO Hyperparameters

- Learning rate: 0.0003
- Gamma: 0.99
- Entropy coefficient: 0.05
- PPO clip range: 0.2
- PPO epochs: 4
- Max grad norm: 0.5
- Warmup episodes: 50

---

### 2. UnifiedRLPolicy

**File:** `backend/unified_rl_policy.py`  
**Purpose:** Single unified policy for ENTRY, HOLD, and EXIT decisions

#### Architecture

```
State Vector [18 features]
    │
    └─→ Shared Feature Extractor
        Linear(18, 64) → LayerNorm → ReLU → Dropout(0.1)
        Linear(64, 64) → LayerNorm → ReLU
        → [B, 64]
    
    ├─→ Action Head
    │   Linear(64, 32) → ReLU
    │   Linear(32, 4) → Action logits
    │   [HOLD, BUY_CALL, BUY_PUT, EXIT]
    │
    ├─→ Value Head
    │   Linear(64, 32) → ReLU
    │   Linear(32, 1) → State value V(s)
    │
    └─→ Exit Urgency Head
        Linear(64, 16) → ReLU
        Linear(16, 1) → Sigmoid → Exit urgency (0-1)
```

#### State Features (18 total)

1. **Position (4):**
   - is_in_trade
   - is_call
   - pnl%
   - drawdown

2. **Time (2):**
   - minutes_held
   - minutes_to_expiry

3. **Prediction (3):**
   - direction
   - confidence
   - momentum_5m

4. **Market (2):**
   - vix_level
   - volume_spike

5. **HMM Regime (4):**
   - trend
   - volatility
   - liquidity
   - hmm_confidence

6. **Greeks (2):**
   - theta_decay
   - delta

#### Modes

1. **Bandit Mode** (first 50 trades): Contextual bandit for fast initial learning
2. **Full RL Mode** (after 50 trades): Full PPO with temporal credit assignment

---

### 3. ExitPolicyNetwork

**File:** `backend/rl_exit_policy.py`  
**Purpose:** RL-based exit timing for options trades

#### Architecture

```
Input Features [20 features]
    │
    └─→ Input Projection
        Linear(20, 64) → LayerNorm → ReLU
    
    ↓ Hidden Block 1 (with residual)
    Linear(64, 64) → LayerNorm → ReLU → Dropout(0.25)
    + Scaled Residual (×0.1)
    
    ↓ Hidden Block 2
    Linear(64, 32) → LayerNorm → ReLU → Dropout(0.2)
    
    ├─→ Exit Head
    │   Linear(32, 16) → ReLU
    │   Linear(16, 1) → Sigmoid → exit_score (0-1)
    │
    └─→ Hold Time Head
        Linear(32, 16) → ReLU
        Linear(16, 1) → Softplus → optimal_hold_mins
```

#### Input Features (20 total)

**Base Features (8):**
1. time_ratio (held/predicted)
2. pnl_normalized (-1 to 1)
3. days_to_expiry_norm
4. move_completion
5. entry_confidence
6. is_winning (0/1)
7. past_predicted_time (0/1)
8. time_remaining_ratio

**Expanded Features (12):**
9. live_pred_change
10. conf_ratio
11. vix_normalized
12. spread_normalized
13. signal_alignment
14. signal_conflict
15. hmm_trend (0-1)
16. hmm_vol (0-1)
17. hmm_liq (0-1)
18. vix_bb_pos
19. vix_roc
20. vix_percentile

#### Exit Decision Thresholds

- Big Profit: ≥ 40% → Force EXIT
- Critical Loss: ≤ -20% → Force EXIT
- Near Expiry: < 1 day → Force EXIT
- RL Score: ≥ 0.6 (dynamic) → EXIT
- RL Score: < threshold → HOLD

---

## Supporting Components

### 1. BayesianLinear

**File:** `bot_modules/neural_networks.py`  
**Purpose:** Bayesian linear layer with learnable weight uncertainty

#### Implementation

```python
Parameters:
- weight_mu: [out, in] - Mean of weight distribution
- weight_logvar: [out, in] - Log-variance of weights
- bias_mu: [out] - Mean of bias distribution
- bias_logvar: [out] - Log-variance of biases

Forward (Reparameterization Trick):
weight = weight_mu + exp(0.5 * weight_logvar) * ε_w
bias = bias_mu + exp(0.5 * bias_logvar) * ε_b
output = x @ weight.T + bias
```

**Benefits:**
- Epistemic uncertainty estimation
- Natural regularization
- Prevents overconfident predictions

---

### 2. RBFKernelLayer

**File:** `bot_modules/neural_networks.py`  
**Purpose:** Non-linear feature expansion using Radial Basis Functions

#### Implementation

```python
Parameters:
- centers: [n_centers, input_dim] - Learnable RBF centers
- log_scales: [n_scales] - Learnable scale parameters

Default Configuration:
- n_centers = 25
- scales = [0.1, 0.5, 1.0, 2.0, 5.0]
- output_dim = 25 × 5 = 125

Forward:
for each scale s:
    dist² = ||x - centers||²
    rbf = exp(-dist² / (2 * s²))
output = concat(all rbf outputs)
```

---

### 3. TCNBlock

**File:** `bot_modules/neural_networks.py`  
**Purpose:** Temporal Convolutional Network block with dilated causal convolutions

#### Architecture

```
Input: [B, C_in, T]
    │
    ├─→ Conv1d(kernel=3, dilation=d, causal padding)
    │   → BatchNorm1d → GELU → Dropout(0.2)
    │
    ├─→ Conv1d(kernel=3, dilation=d, causal padding)
    │   → BatchNorm1d → GELU → Dropout(0.2)
    │
    └─→ Residual Path
        Conv1d(1x1) if C_in ≠ C_out, else Identity
    
    ↓ Add
    output + residual
```

**Key Features:**
- Dilated convolutions for exponentially growing receptive field
- Causal padding (no future information leakage)
- Residual connections for deep networks
- Fully parallelizable (faster than LSTM)

---

### 4. ResidualBlock

**File:** `bot_modules/neural_networks.py`  
**Purpose:** Residual block with Bayesian layers for stable deep training

#### Architecture

```
Input
    │
    ├─→ BayesianLinear(in_dim, out_dim)
    │   → LayerNorm → GELU → Dropout
    │
    └─→ Skip Connection
        Linear(in, out) if dims differ, else Identity
    
    ↓ Add (scaled residual)
    output + residual × 0.1
```

---

## Regime Detection Models

### 1. MultiDimensionalHMM

**File:** `backend/multi_dimensional_hmm.py`  
**Purpose:** Statistical HMM for market regime detection

#### Architecture

Three independent HMMs:

1. **Trend HMM**
   - Auto-discovers optimal states (2-7) using BIC
   - Features: Returns (5, 10, 20 bar), momentum, price vs SMA

2. **Volatility HMM**
   - Auto-discovers optimal states (2-7) using BIC
   - Features: Realized volatility (5, 10, 20 bar), vol of vol, True Range

3. **Liquidity HMM**
   - Auto-discovers optimal states (2-7) using BIC
   - Features: Volume z-scores, volume trend, spread proxy, price impact

**Total Regimes:** trend_states × vol_states × liq_states (up to 343)

**Training:** Baum-Welch EM algorithm (max_iter=200, tol=1e-4)

---

### 2. NeuralRegimeClassifier

**File:** `backend/multi_dimensional_hmm.py`  
**Purpose:** Neural network complement to HMM for regime detection

#### Architecture

```
Input Features [15 features]
    │
    └─→ Encoder
        Linear(15, 64) → LayerNorm → ReLU → Dropout(0.3)
        Linear(64, 64) → LayerNorm → ReLU → Dropout(0.2)
    
    ├─→ Regime Head
    │   Linear(64, 32) → ReLU
    │   Linear(32, n_regimes) → Combined regime probabilities
    │
    ├─→ Trend Head
    │   Linear(64, 3) → Trend state probabilities
    │
    ├─→ Vol Head
    │   Linear(64, 3) → Volatility state probabilities
    │
    ├─→ Liq Head
    │   Linear(64, 3) → Liquidity state probabilities
    │
    └─→ Transition Head
        Linear(64 + n_regimes, 32) → ReLU
        Linear(32, n_regimes) → Next regime probabilities
```

**Advantages over HMM:**
- Can learn non-linear regime patterns
- Faster inference (no Viterbi)
- Better at capturing regime transitions

---

### 3. RegimeModelManager

**File:** `backend/regime_models.py`  
**Purpose:** Manages regime-specific neural models

#### Architecture

Per-regime models (simpler than main predictor):

```
Input Features [D]
    │
    └─→ Regime-Specific Model
        Linear(D, 128) → ReLU → Dropout(0.3)
        Linear(128, 64) → ReLU → Dropout(0.2)
        Linear(64, 32) → ReLU
        Linear(32, 1) → Tanh → Direction (-1 to 1)
```

**Regimes Supported:**
- low_vol, normal_vol, high_vol
- trending_up, trending_down, sideways
- low_vol_trending, high_vol_crisis

**Transfer Learning:** Initialized from general model weights

---

## Entry Controllers

### 1. V3EntryController

**File:** `backend/v3_entry_controller.py`  
**Purpose:** Entry decisions using DirectionPredictorV3

**Uses:** `DirectionPredictorV3` model (see above)

**Features:**
- HMM alignment filter
- Confidence threshold (default: 0.55)
- Computes features from price history

---

### 2. DirectionEntryController

**File:** `backend/direction_entry_controller.py`  
**Purpose:** Entry decisions using DirectionPredictor

**Uses:** `DirectionPredictor` model (see above)

**Features:**
- Online learning from trade outcomes
- Exploration mode (first 100 trades)
- HMM alignment checking
- Confidence-based position sizing

---

## Data Flow Summary

```
Market Data
    │
    ▼
Feature Computation
(FeatureComputer: technical, volatility, volume, cross-asset)
    │
    ├───────────────────────────────────────────────┐
    │                                               │
    ▼                                               ▼
UnifiedOptionsPredictor                    MultiDimensionalHMM
(TCN/LSTM + Bayesian)                      (trend/vol/liquidity)
    │                                               │
    │   predictions                                 │   regime info
    └───────────────────────────┬───────────────────┘
                                │
                                ▼
                    Policy Decision
            (UnifiedRLPolicy or RLTradingPolicy)
                                │
            ┌───────────────────┴───────────────────┐
            │                                       │
            ▼                                       ▼
    ENTRY Decision                        EXIT Decision
    (Entry Controllers)                   (ExitPolicyNetwork)
            │                                       │
            └───────────────────┬───────────────────┘
                                │
                                ▼
                        Trade Execution
```

---

## Model Sizes & Parameters

| Model | Parameters (approx) | Purpose | File |
|-------|--------------------:|---------|------|
| UnifiedOptionsPredictor | ~200K | Main prediction | `bot_modules/neural_networks.py` |
| UnifiedOptionsPredictorV2 | ~200K | Slim Bayesian variant | `bot_modules/neural_networks.py` |
| OptionsTCN | ~150K | Time series encoding | `bot_modules/neural_networks.py` |
| OptionsLSTM | ~400K | Legacy time series | `bot_modules/neural_networks.py` |
| DirectionPredictor | ~300K | Multi-scale direction | `bot_modules/neural_networks.py` |
| DirectionPredictorV3 | ~50K | Simplified direction | `bot_modules/neural_networks.py` |
| UnifiedPolicyNetwork | ~15K | RL entry/exit | `backend/unified_rl_policy.py` |
| RLTradingPolicy | ~100K | PPO entry | `backend/rl_trading_policy.py` |
| ExitPolicyNetwork | ~8K | RL exit timing | `backend/rl_exit_policy.py` |
| NeuralRegimeClassifier | ~20K | Regime detection | `backend/multi_dimensional_hmm.py` |
| RegimeModelManager | ~10K per regime | Regime-specific | `backend/regime_models.py` |

---

## Key Design Patterns

### 1. **Shared Feature Extractors**
- Actor-Critic architectures share feature extraction
- Reduces parameters and improves learning efficiency

### 2. **Bayesian Uncertainty**
- BayesianLinear layers provide epistemic uncertainty
- Confidence scores reflect model uncertainty

### 3. **Residual Connections**
- Scaled residuals (×0.1) for stable deep training
- Prevents gradient vanishing/exploding

### 4. **Multi-Head Outputs**
- Single backbone with multiple specialized heads
- Efficient parameter sharing

### 5. **Temporal Encoding**
- TCN (preferred) or LSTM for sequence processing
- Attention pooling for sequence aggregation

### 6. **Progressive Complexity**
- Simple models first (DirectionPredictorV3)
- More complex models for specialized tasks (DirectionPredictor)

---

## Debugging Checklist

When debugging neural networks, check:

1. **Input Shapes:** Verify feature dimensions match expected input_dim
2. **NaN/Inf Values:** Check for numerical instability in features
3. **Gradient Flow:** Monitor gradients for vanishing/exploding
4. **Confidence Calibration:** Are confidence scores meaningful?
5. **Action Distribution:** Is policy collapsing to single action?
6. **Loss Curves:** Track training/validation loss
7. **Feature Importance:** Use XGBoost feature importance for insights
8. **Regime Detection:** Verify HMM states make sense

### Logging Debug Info

```python
# Enable debug logging for neural networks
import logging
logging.getLogger('bot_modules.neural_networks').setLevel(logging.DEBUG)
logging.getLogger('backend.unified_rl_policy').setLevel(logging.DEBUG)
logging.getLogger('backend.rl_exit_policy').setLevel(logging.DEBUG)
logging.getLogger('backend.rl_trading_policy').setLevel(logging.DEBUG)
```

---

## File Locations

| Component | File Path |
|-----------|-----------|
| Core Predictors | `bot_modules/neural_networks.py` |
| PPO Entry Policy | `backend/rl_trading_policy.py` |
| Unified RL Policy | `backend/unified_rl_policy.py` |
| Exit Policy | `backend/rl_exit_policy.py` |
| HMM Regime Detection | `backend/multi_dimensional_hmm.py` |
| Regime Models | `backend/regime_models.py` |
| V3 Entry Controller | `backend/v3_entry_controller.py` |
| Direction Entry Controller | `backend/direction_entry_controller.py` |

---

## Version History

- **V1:** Original UnifiedOptionsPredictor with Bayesian backbone
- **V2:** Slim Bayesian (deterministic backbone, Bayesian heads)
- **V3:** Simplified direction-only predictor (no RBF, lightweight TCN)

---

**End of Architecture Map**


