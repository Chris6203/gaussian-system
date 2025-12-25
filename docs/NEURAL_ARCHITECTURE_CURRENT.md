# Neural Architecture (Current) — output3

This document is a **current, code-aligned** map of the neural architecture used by the `output3` bot.

If you want the big, diagram-heavy deep dive that already exists, see:
- `output3/NEURAL_NETWORK_ARCHITECTURE.md`

---

## 1) High-level: what neural components exist?

There are **two distinct neural subsystems**:

1) **Predictor (supervised / self-supervised style)**: `UnifiedOptionsPredictor` (plus `UnifiedOptionsPredictorV2`)
- Produces **predicted return**, **volatility**, **direction probabilities**, **confidence**, and execution-related heads (**fillability**, **slippage**, **time-to-fill**).
- Implemented in `output3/bot_modules/neural_networks.py`.

2) **Policy (RL-style decision model)**: `UnifiedRLPolicy` / `UnifiedPolicyNetwork`
- Produces **action logits** for **HOLD / BUY_CALL / BUY_PUT / EXIT**, plus **value estimate** and **exit urgency**.
- Implemented in `output3/backend/unified_rl_policy.py`.

The main bot orchestrator is:
- `output3/unified_options_trading_bot.py` (imports both predictor components and unified RL policy)

---

## 2) Predictor: UnifiedOptionsPredictor (main neural model)

**Code:** `output3/bot_modules/neural_networks.py`

### 2.1 Inputs

The predictor consumes two conceptual inputs:

- **Current feature vector**: `cur: [B, D]`
- **Sequence features** (lookback window): `seq: [B, T, D]`

Where:
- `B` = batch
- `T` = sequence length (commonly 60)
- `D` = feature dimension (your engineered feature count)

### 2.2 Temporal encoder (TCN or LSTM)

The sequence is encoded into a fixed-size context vector **`[B, 64]`** via one of:

- **TCN path (preferred):** `OptionsTCN`
  - Built from stacked dilated causal `TCNBlock`s (dilations 1,2,4,8,16).
  - Attention pooling over time.
  - Output: `[B, 64]`

- **LSTM path (legacy fallback):** `OptionsLSTM`
  - LSTM encoder + pooling.
  - Output: `[B, 64]`

### 2.3 Optional kernel expansion

- **RBFKernelLayer** can expand current features into non-linear similarity features.
- Default behavior in the existing architecture docs: `n_centers=25`, `scales=[0.1, 0.5, 1.0, 2.0, 5.0]` → output `[B, 125]`.

### 2.4 Fusion + deep residual MLP

The model concatenates:

- current features `[B, D]`
- temporal context `[B, 64]`
- optional RBF features `[B, 125]`

Then applies:

- an input projection MLP
- several `ResidualBlock`s (with Bayesian layers) to reach a shared hidden representation

### 2.5 Multi-head outputs

From a common head trunk, it produces multiple heads:

- **return**: predicted return (regression)
- **volatility**: predicted volatility
- **direction**: probabilities over (DOWN / NEUTRAL / UP)
- **confidence**: scalar 0..1
- **fillability**: scalar 0..1
- **exp_slippage**: expected slippage
- **exp_ttf**: expected time-to-fill

These are all part of the architecture described in `output3/NEURAL_NETWORK_ARCHITECTURE.md` and implemented in `bot_modules/neural_networks.py`.

---

## 3) Core predictor building blocks (implemented)

**Code:** `output3/bot_modules/neural_networks.py`

### 3.1 BayesianLinear

- Bayesian linear layer using reparameterization:
  - `w = mu + std * epsilon`
- Purpose: epistemic uncertainty in the final heads / residual blocks.

### 3.2 RBFKernelLayer

- Learns `n_centers` and evaluates Gaussian RBF similarities at multiple scales.
- Purpose: non-linear feature expansion (kernelized similarity to learned prototypes).

### 3.3 TCNBlock

- Two `Conv1d` layers per block
- Causal padding (no future leakage)
- Dilations grow exponentially to expand receptive field
- Residual connection (projection if channel mismatch)

### 3.4 OptionsTCN

- `input_proj: Linear(D → hidden_dim)`
- stack of `TCNBlock(hidden_dim → hidden_dim)`
- attention pooling over time
- `ctx` projection to `[B, 64]`

### 3.5 OptionsLSTM

- Legacy sequence encoder (kept for compatibility)

### 3.6 ResidualBlock

- Residual MLP-style block that uses Bayesian layers (uncertainty-aware deep MLP).

---

## 4) RL policy: UnifiedRLPolicy / UnifiedPolicyNetwork

**Code:** `output3/backend/unified_rl_policy.py`

### 4.1 State features (18)

`UnifiedPolicyNetwork` uses **18 scalar features** (see module docstring):

- **Position (4):** in_trade, is_call, pnl%, drawdown
- **Time (2):** minutes_held, minutes_to_expiry
- **Prediction (3):** direction, confidence, momentum_5m
- **Market (2):** vix, volume_spike
- **HMM (4):** trend, volatility, liquidity, hmm_confidence
- **Greeks (2):** theta, delta

### 4.2 MLP architecture

- Shared trunk:
  - `Linear(18 → 64) → LayerNorm → ReLU → Dropout(0.1)`
  - `Linear(64 → 64) → LayerNorm → ReLU`

- Heads:
  - **Action head:** `Linear(64 → 32) → ReLU → Linear(32 → 4)`
  - **Value head:** `Linear(64 → 32) → ReLU → Linear(32 → 1)`
  - **Exit urgency:** `Linear(64 → 16) → ReLU → Linear(16 → 1) → Sigmoid`

### 4.3 Actions

- `0`: HOLD
- `1`: BUY_CALL
- `2`: BUY_PUT
- `3`: EXIT

---

## 5) How these connect in the bot

**Orchestrator:** `output3/unified_options_trading_bot.py`

- Builds features / multi-timeframe predictions.
- Produces a `signal` dict (contains `predicted_return`, confidence fields, momentum, volume spike, etc.).
- (Optionally) uses `UnifiedRLPolicy.select_action(...)` to choose entry / exit actions.
- Execution/position updates are handled in `backend/paper_trading_system.py` and `backend/live_trading_engine.py`.

---

## 6) “Where things live” map (organized view of output3)

### 6.1 Neural code

- **Predictor architectures**: `output3/bot_modules/neural_networks.py`
  - `BayesianLinear`, `RBFKernelLayer`, `TCNBlock`, `OptionsTCN`, `OptionsLSTM`, `ResidualBlock`, `UnifiedOptionsPredictor`

- **Unified RL policy**: `output3/backend/unified_rl_policy.py`
  - `TradeState`, `UnifiedPolicyNetwork`, `UnifiedRLPolicy`

### 6.2 Feature creation

- **Feature pipeline**: `output3/features/pipeline.py`
- **Feature integration**: `output3/features/integration.py`
- **Bot-side feature utilities**: `output3/bot_modules/features.py`

### 6.3 Execution + risk

- **Paper/live execution simulation**: `output3/backend/paper_trading_system.py`
- **Live cycle loop + gating**: `output3/backend/live_trading_engine.py`
- **Liquidity execution layer**: `output3/execution/liquidity_exec.py`, `output3/execution/tradier_adapter.py`

### 6.4 Docs

- **Full neural architecture diagrams**: `output3/NEURAL_NETWORK_ARCHITECTURE.md`
- **System architecture**: `output3/SYSTEM_ARCHITECTURE.md`
- **Arch flow (v2)**: `output3/docs/ARCH_FLOW_V2.md`

### 6.5 Observability (additive)

- **Decision visibility + missed opportunity logging**: `output3/backend/decision_pipeline.py`
- **Missed summary tool**: `output3/tools/summarize_missed.py`

---

## 7) If you want the architecture “printed” verbatim

The most complete, already-formatted, diagram-rich version is:
- `output3/NEURAL_NETWORK_ARCHITECTURE.md`

This file (`NEURAL_ARCHITECTURE_CURRENT.md`) is the **navigation + mapping** layer so you can quickly find the code that corresponds to each block.




