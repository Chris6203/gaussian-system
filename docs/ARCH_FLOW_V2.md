# Architecture Flow V2 - Gaussian Options Trading Bot

## Overview

This document describes the refactored architecture with clear separation of concerns:

1. **Single Entry Policy** - One decision maker for entries
2. **Single Exit Policy** - One model-based exit brain + hard safety rules
3. **Frozen Predictor** - Forecasting separated from RL/control
4. **Uncertainty Integration** - Bayesian uncertainty flows to all decision points
5. **Deep Regime Integration** - HMM regime as embeddings, not just scalars

---

## Component Map

### 1. Forecasting Layer (Oracle)

**`UnifiedOptionsPredictor`** (`bot_modules/neural_networks.py`)
- TCN or LSTM temporal encoder
- BayesianLinear heads for uncertainty estimates
- RBF kernel features for non-linear patterns
- **Outputs:**
  - `return_mean`, `return_std` (predicted return + uncertainty)
  - `direction` logits → `dir_probs`, `dir_entropy`
  - `volatility` prediction
  - `confidence` score
  - Execution quality predictions (fillability, slippage, TTF)

**Training:** Offline on historical data, then frozen during RL training.

### 2. Entry Policies

| Policy | Type | Config Key | Description |
|--------|------|------------|-------------|
| `UnifiedRLPolicy` | PPO-style | `entry_policy.type = "unified_ppo"` | Single policy for entry/exit with 4 actions |
| `RLTradingPolicy` | PPO | `entry_policy.type = "ppo"` | Legacy 5-action PPO (HOLD, CALL 1x/2x, PUT 1x/2x) |
| `AdaptiveRulesPolicy` | Rule-based | `entry_policy.type = "rules"` | Interpretable rule-based with learned thresholds |

**Live Trading:** Only ONE entry policy is active, controlled by `entry_policy.type` in config.

### 3. Exit Policies

| Policy | Type | Config Key | Description |
|--------|------|------------|-------------|
| `ExitPolicyNetwork` | NN | `exit_policy.type = "nn_exit"` | Neural network exit timing |
| `XGBoostExitPolicy` | XGBoost | `exit_policy.type = "xgboost_exit"` | Tree-based exit classifier |
| Hard Rules | Rule-based | Always active | Safety backstops (stop loss, max hold, expiry) |

**Exit Flow:**
```
1. Hard Safety Rules (ALWAYS FIRST)
   ├── Max loss threshold (e.g., -15%)
   ├── Max hold time exceeded
   ├── Near expiry (< 30 min)
   └── Portfolio max drawdown

2. If no safety rule triggered:
   └── Model-based exit (configured type)
       ├── nn_exit: ExitPolicyNetwork
       └── xgboost_exit: XGBoostExitPolicy
```

### 4. HMM / Regime Detection

**`MultiDimensionalHMM`** (`backend/multi_dimensional_hmm.py`)
- 3x3x3 = 27 regime states (Trend × Volatility × Liquidity)
- Outputs: trend_state, vol_state, liq_state, confidence

**Integration:**
- Regime embedding (8-16 dim vector) fed to predictor and policies
- Not just scalar features anymore

### 5. Safety Filter (Optional)

**`EntrySafetyFilter`** (`backend/safety_filter.py`)
- Can APPROVE, DOWNGRADE, or VETO entry proposals
- Uses same state features as RL policy
- Lightweight check before execution

---

## Understanding "Frozen Predictor"

The `predictor.frozen_during_rl = true` setting is often misunderstood. Here's what it means:

### What "Frozen" DOES Mean:

1. **Predictor weights are NOT updated during RL training**
   - The predictor's parameters are excluded from RL optimizers
   - `requires_grad = False` is set on all predictor parameters
   - No gradient flows back through the predictor during RL backward passes

2. **Forward passes use `torch.no_grad()`**
   - Efficient inference without building computation graphs
   - Reduces memory usage during RL training

### What "Frozen" Does NOT Mean:

1. **The predictor STILL RUNS every step/minute!**
   - During RL training, the predictor generates predictions every step
   - These predictions feed into the RL state vector
   - Uncertainty estimates are still computed via MC sampling

2. **Predictions are still essential for RL**
   - RL policies receive: `return_mean`, `return_std`, `dir_probs`, `dir_entropy`
   - The predictor is a "frozen oracle" that informs RL decisions

### Why Freeze During RL?

1. **Prevents conflicting gradients**
   - RL optimizes for rewards (trading outcomes)
   - Predictor should optimize for prediction accuracy
   - Training both together can create conflicting objectives

2. **Enables phased training**
   - Phase 1: Train predictor on labeled historical data (supervised)
   - Phase 2: Freeze predictor, train RL policy on simulated trading
   - This separation leads to more stable, better-performing systems

### Code Pattern

```python
# RL Training Setup
from backend.arch_v2 import init_arch_v2

arch = init_arch_v2("config.json")
arch.prepare_for_rl_training()  # Freezes predictor if config says so

# RL training loop
for episode in range(num_episodes):
    # Predictor STILL RUNS to generate predictions
    pred = arch.predictor_manager.predict_with_uncertainty(cur, seq)
    
    # Predictions feed into RL state
    state = build_rl_state(pred)
    
    # RL policy makes decisions
    action = arch.entry_policy.select_action(state)
    
    # Only RL policy is trained
    arch.entry_policy.train_step()
    
    # Predictor weights unchanged!
```

### Verification

Use `arch.verify_predictor_frozen()` to confirm predictor is frozen:

```python
assert arch.verify_predictor_frozen(), "Predictor should be frozen during RL!"
```

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           DATA PIPELINE                                  │
├─────────────────────────────────────────────────────────────────────────┤
│  Market Data → Features → HMM Regime → Predictor → RL State + Exit State│
└─────────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      PREDICTOR (FROZEN)                                  │
├─────────────────────────────────────────────────────────────────────────┤
│  UnifiedOptionsPredictor                                                 │
│  ├── TCN/LSTM temporal encoding                                         │
│  ├── Bayesian heads → return_mean, return_std, dir_probs, dir_entropy  │
│  └── Execution quality predictions                                       │
│                                                                          │
│  Mode: arch.predictor_frozen = true (default for RL training)           │
└─────────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     STATE BUILDER                                        │
├─────────────────────────────────────────────────────────────────────────┤
│  Combines:                                                               │
│  ├── Predictor outputs (return_mean, return_std, dir_probs, dir_entropy)│
│  ├── Regime embedding (from HMM)                                         │
│  ├── Account state (balance, positions, P&L)                            │
│  ├── Time features (minutes held, to expiry)                            │
│  └── Market features (VIX, momentum, volume)                            │
└─────────────────────────────────────────────────────────────────────────┘
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
┌─────────────────────────────┐ ┌─────────────────────────────┐
│     ENTRY POLICY            │ │      EXIT POLICY            │
├─────────────────────────────┤ ├─────────────────────────────┤
│  Configured by:             │ │  Priority Order:            │
│  entry_policy.type          │ │  1. Hard safety rules       │
│                             │ │  2. exit_policy.type        │
│  Options:                   │ │                             │
│  ├── unified_ppo (default)  │ │  Options:                   │
│  ├── ppo (legacy)           │ │  ├── xgboost_exit (default) │
│  └── rules                  │ │  └── nn_exit                │
└─────────────────────────────┘ └─────────────────────────────┘
                    │                       │
                    ▼                       ▼
┌─────────────────────────────┐ ┌─────────────────────────────┐
│  SAFETY FILTER (Optional)   │ │   FINAL EXIT DECISION       │
├─────────────────────────────┤ │                             │
│  Can: APPROVE/DOWNGRADE/VETO│ │  EXIT or HOLD               │
│  Enabled by:                │ │                             │
│  safety_filter.enabled=true │ │                             │
└─────────────────────────────┘ └─────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         EXECUTION                                        │
├─────────────────────────────────────────────────────────────────────────┤
│  LiquidityExecutor → TradierAdapter → Tradier API                       │
│  (or PaperTradingSystem for simulation)                                  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Configuration Reference

All architecture options are in `config.json` under the `architecture` key:

```json
{
  "architecture": {
    "entry_policy": {
      "type": "unified_ppo",  // "unified_ppo" | "ppo" | "rules"
      "description": "Single entry policy for live trading"
    },
    "exit_policy": {
      "type": "xgboost_exit",  // "xgboost_exit" | "nn_exit"
      "description": "Model-based exit (hard rules always apply first)"
    },
    "predictor": {
      "arch": "v2_slim_bayesian",  // "v1_original" | "v2_slim_bayesian"
      "frozen_during_rl": true,
      "description": "Predictor architecture and training mode"
    },
    "safety_filter": {
      "enabled": true,
      "can_veto": true,
      "can_downgrade_size": true,
      "description": "Optional safety filter for entry decisions"
    },
    "regime": {
      "use_embedding": true,
      "embedding_dim": 8,
      "description": "HMM regime as learned embedding vs scalar features"
    },
    "uncertainty": {
      "mc_samples": 5,
      "inject_into_state": true,
      "description": "Bayesian uncertainty sampling and injection"
    },
    "horizons": {
      "prediction_minutes": 15,
      "rl_reward_minutes": 15,
      "exit_reference_minutes": 15,
      "description": "Aligned time horizons across all components"
    }
  }
}
```

---

## Code Examples

### Initializing V2 Architecture

```python
from backend.arch_v2 import init_arch_v2, get_arch_v2

# Initialize from config
arch = init_arch_v2("config.json", device="cuda")

# Load saved checkpoints
arch.load_checkpoints("models")

# Access components
predictor = arch.predictor_manager
entry_policy = arch.entry_policy
exit_manager = arch.exit_manager
safety_filter = arch.safety_filter  # None if disabled
```

### Getting Predictions with Uncertainty

```python
from backend.predictor_manager import PredictorManager

# Manager handles MC sampling automatically
manager = arch.predictor_manager

# Get prediction with uncertainty estimates
pred = manager.predict_with_uncertainty(cur_features, seq_features)

print(f"Return: {pred.return_mean:.4f} ± {pred.return_std:.4f}")
print(f"Direction probs: {pred.dir_probs}")
print(f"Direction entropy: {pred.dir_entropy:.4f}")
print(f"Risk-adjusted return: {pred.risk_adjusted_return:.4f}")

# Use risk-adjusted return instead of raw prediction
if pred.risk_adjusted_return > 0.5 and not pred.is_high_uncertainty:
    # Good signal
    pass
```

### Making Entry Decisions

```python
from backend.arch_v2 import get_entry_policy, get_safety_filter

entry_policy = get_entry_policy()
safety_filter = get_safety_filter()

# Build state for policy
from backend.unified_rl_policy import TradeState

state = TradeState(
    is_in_trade=False,
    predicted_direction=pred.risk_adjusted_return,
    prediction_confidence=pred.raw_confidence,
    vix_level=18.0,
    hmm_trend=0.6,  # from HMM regime
    hmm_volatility=0.5,
    hmm_liquidity=0.5,
    hmm_confidence=0.8,
)

# Get action from policy
action, confidence, details = entry_policy.select_action(state)

# Apply safety filter (if enabled)
if safety_filter and action != 0:  # Not HOLD
    from backend.safety_filter import FilterVerdict
    
    decision = safety_filter.evaluate(
        proposed_action="BUY_CALLS" if action == 1 else "BUY_PUTS",
        proposed_size=1,
        confidence=confidence,
        vix_level=18.0,
        hmm_trend=0.6,
        hmm_vol=0.5,
        volume_spike=1.2,
        minutes_to_close=120,
    )
    
    if decision.verdict == FilterVerdict.VETO:
        action = 0  # Convert to HOLD
    elif decision.verdict == FilterVerdict.DOWNGRADE:
        # Use reduced size
        pass
```

### Making Exit Decisions

```python
from backend.unified_exit_manager import UnifiedExitManager, PositionInfo, MarketState

exit_manager = arch.exit_manager

# Create position info
position = PositionInfo(
    trade_id="trade_123",
    entry_price=1.50,
    current_price=1.65,
    entry_time=entry_time,
    option_type="CALL",
    days_to_expiry=7,
    entry_confidence=0.65,
    predicted_move_pct=0.5,
    high_water_mark_pct=12.0,  # Best P&L seen
)

market_state = MarketState(
    vix_level=18.0,
    hmm_trend=0.6,
    hmm_vol=0.5,
)

# Get exit decision
decision = exit_manager.should_exit(position, market_state)

if decision.should_exit:
    print(f"EXIT: {decision.reason}")
    print(f"Rule type: {decision.rule_type}")
else:
    print(f"HOLD: score={decision.exit_score:.2f}")
```

### Using Regime Embedding

```python
from backend.regime_embedding import RegimeEmbedding

# Create embedding layer
embedder = arch.regime_embedding

# Get embedding from HMM regime dict
hmm_regime = {
    'trend_state': 2,  # Bullish
    'volatility_state': 1,  # Normal
    'liquidity_state': 1,  # Normal
}
regime_vec = embedder.get_embedding_from_hmm_dict(hmm_regime)

# Concatenate to features
features_with_regime = torch.cat([features, regime_vec], dim=-1)
```

---

## Training Scripts

### Predictor Training (Phase 1)
```bash
# Train predictor on historical data
python scripts/train_predictor.py --output models/predictor_v2.pt --arch v2_slim_bayesian
```

### RL Training (Phase 2 - Predictor Frozen)
```bash
# Train RL with frozen predictor
python scripts/train_rl.py --predictor models/predictor_v2.pt --freeze-predictor
```

### End-to-End Training (Time-Travel)
```bash
python scripts/train_time_travel.py --config config.json
```

---

## Live Trading

```bash
# Start with V2 architecture (reads from config.json)
python go_live_only.py

# Or with specific architecture overrides
python go_live_only.py --entry-policy ppo --exit-policy nn_exit
```

---

## New Files Created

| File | Description |
|------|-------------|
| `backend/arch_config.py` | Central architecture configuration |
| `backend/arch_v2.py` | V2 architecture integration |
| `backend/safety_filter.py` | Entry safety filter |
| `backend/unified_exit_manager.py` | Unified exit management |
| `backend/predictor_manager.py` | Predictor with uncertainty |
| `backend/regime_embedding.py` | HMM regime embeddings |
| `backend/horizon_alignment.py` | Time horizon alignment |
| `bot_modules/neural_networks.py` | Added V2 predictor variant |
| `docs/ARCH_FLOW_V2.md` | This documentation |

---

## Migration Notes

### From Legacy to V2:
1. Set `architecture.entry_policy.type = "unified_ppo"`
2. Set `architecture.exit_policy.type = "xgboost_exit"`
3. Set `architecture.predictor.frozen_during_rl = true`
4. Legacy policies are still available as fallback

### Backward Compatibility:
- All existing saved models work (paths unchanged)
- Legacy flags like `use_rl_policy`, `use_unified_rl` still respected
- New architecture config takes precedence when present

### Key Behavior Changes:
1. **Predictor is frozen by default during RL training** - prevents conflicting gradients
2. **Single exit policy** - hard rules always apply first, then one model-based exit
3. **Safety filter can veto entries** - enable with `safety_filter.enabled = true`
4. **Uncertainty injected into state** - RL sees `return_std`, `dir_entropy`
5. **Regime embedding** - HMM regime as learned vector, not just scalars
