You are helping me refactor and improve the architecture of my Gaussian options trading bot.

First, IMPORTANT CONTEXT:
- There is a design document in this repo called `NEURAL_NETWORK_ARCHITECTURE.md` that describes:
  - `UnifiedOptionsPredictor` (TCN/LSTM with BayesianLinear heads, RBF features, etc.).
  - `UnifiedPolicyNetwork` (small feedforward network with 18-dim state, 64-dim hidden, 4 actions).
  - `RLTradingPolicy` (PPO-style RL policy with 5 actions: HOLD, CALL x1/x2, PUT x1/x2).
  - `ExitPolicyNetwork` (exit_score + optimal_hold_mins).
  - `XGBoostExitPolicy` (tree-based exit classifier).
  - HMM / regime classifier and regime features (trend, volatility, liquidity, hmm_conf, etc.).
  - Various hard exit rules (profit %, loss %, near expiry, etc.).

Your job is to:
1. Analyze the current implementation in code (NOT just the markdown).
2. Cleanly simplify and strengthen the architecture according to the following goals.
3. Implement concrete changes (configs, refactors, new modules) WITHOUT breaking the system’s ability to train and run.

-------------------------------------------------------------------------------
HIGH-LEVEL ARCHITECTURE GOALS
-------------------------------------------------------------------------------

A. **Single main entry policy + optional safety filter**
B. **Single main exit policy + clear priority with hard safety rules**
C. **Forecasting (“oracle”) clearly separated from RL/control**
D. **Uncertainty from Bayesian heads fed into RL and exits**
E. **HMM/regime integrated more deeply (not just a couple of scalar features)**
F. **Predictor regularization & capacity tuned (esp. dropout and Bayesian usage)**
G. **Horizon alignment between predictor labels, RL rewards, and exit logic**

-------------------------------------------------------------------------------
STEP 1: DISCOVER AND MAP EXISTING COMPONENTS
-------------------------------------------------------------------------------

1. Find the actual code for these components (class names may vary slightly, so search by name and keywords):
   - `UnifiedOptionsPredictor`
   - `UnifiedPolicyNetwork`
   - PPO / RL policy for trading (e.g., `RLTradingPolicy`, `PPOPolicy`, or similar)
   - `ExitPolicyNetwork`
   - `XGBoostExitPolicy` (or the module that wraps XGBoost for exits)
   - HMM / regime code (HMM training, regime features, etc.)
   - Any code that:
     - Builds RL states (`state_builder`, `build_state`, etc.)
     - Wires predictor outputs into RL or exit logic
     - Applies hard exit rules (big profit, big loss, expiry, etc.)

2. Create (or update) a short internal doc or comment that maps:
   - Which modules/classes are used for ENTRY decisions in live trading.
   - Which are used for EXIT decisions in live trading.
   - Where the predictor is invoked and whether its weights are frozen or updated during RL.
   - How HMM/regimes are computed and injected into state.

Keep that mapping in comments and/or a small markdown file (e.g., `docs/ARCH_FLOW_V2.md`) so it’s easy to reason about.

-------------------------------------------------------------------------------
STEP 2: ENFORCE A SINGLE MAIN ENTRY POLICY
-------------------------------------------------------------------------------

Goal: In live trading, there should be **one and only one** primary entry decision-maker.

1. Determine whether the current live system uses:
   - `UnifiedPolicyNetwork`, or
   - PPO-based `RLTradingPolicy` (or similar), or both.

2. Add a configuration flag in a central config file, something like:
   - `entry_policy.type` ∈ { "ppo", "unified_policy" }

3. Refactor the code so that:
   - Only the configured entry policy actually emits live `BUY_CALL`, `BUY_PUT`, `HOLD`, etc.
   - The other policy (if present) is:
     - Either disabled entirely for live trading, OR
     - Used only as a “safety filter” or diagnostic (see Step 4), but NOT as a competing decision-maker.

4. Ensure that training scripts reflect this:
   - If PPO is the main entry policy for v2, make that clear in config and trainer wiring.
   - `UnifiedPolicyNetwork` should NOT be silently used as an alternative entry policy in the same live run.

-------------------------------------------------------------------------------
STEP 3: ENFORCE A SINGLE MAIN EXIT POLICY WITH CLEAR PRIORITY
-------------------------------------------------------------------------------

Goal: In live trading, there should be **one and only one** model-based exit brain, with hard safety rules clearly layered on top.

1. Find where exits are currently decided:
   - Hard rules (e.g., large profit %, large loss %, near expiry).
   - `ExitPolicyNetwork` usage.
   - `XGBoostExitPolicy` usage.
   - Any other heuristics.

2. Introduce a central, explicit exit flow like:

   - Apply **hard safety rules** first:
     - Max allowed loss (e.g., -X%).
     - Max allowed holding time or near expiry.
     - Portfolio-wide max drawdown, etc.
   - If no safety rule is triggered:
     - Use a single configured model-based exit:
       - `exit_policy.type` ∈ { "nn_exit", "xgboost_exit" }.

3. Refactor exit code so that:
   - Only the chosen model-based exit is used for non-safety exit decisions.
   - The other exit model (if present) is used only for **offline evaluation** or as a teacher (see Step 6).

4. Make sure all tests / evaluation scripts use the same exit pipeline for consistency.

-------------------------------------------------------------------------------
STEP 4: OPTIONAL SAFETY FILTER FOR ENTRY
-------------------------------------------------------------------------------

Goal: Keep RL powerful but constrained by a lightweight safety filter.

1. Choose a simple, small network (for example `UnifiedPolicyNetwork` repurposed) or a simple rule-based system as a **safety shield**.

2. Implement an entry flow like:

   - RL entry policy proposes an action: `a_rl ∈ {HOLD, BUY_CALL, BUY_PUT, ...}`.
   - Safety filter receives:
     - The same state features that RL sees (or a subset).
     - The proposed action.
   - Safety filter can:
     - APPROVE the action.
     - DOWNGRADE it (e.g., reduce size or change 2x → 1x).
     - VETO it to HOLD if risk is too high or liquidity is too low.

3. Add a config to enable/disable this safety filter and keep the logic simple and transparent.

-------------------------------------------------------------------------------
STEP 5: SEPARATE FORECASTING FROM RL / CONTROL
-------------------------------------------------------------------------------

Goal: Treat `UnifiedOptionsPredictor` as a “frozen oracle” during RL training (unless we explicitly choose to update it in phases).

1. Identify all training scripts where:
   - Predictor and RL are trained in the same loop or same experiment.

2. Refactor training so that:
   - There is a clearly defined step to train `UnifiedOptionsPredictor` **offline** on historical data.
   - After training, we save a checkpoint (e.g., `predictor_vX.pt`).

3. In RL training:
   - Load the predictor checkpoint and **freeze its weights** (no gradients).
   - Remove any optimizer parameters for the predictor from the RL trainer.
   - Treat predictor outputs as environment features, not learnable inside RL.

4. OPTIONAL (if needed): support “phased training”:
   - Phase 1: train predictor on updated data.
   - Phase 2: freeze predictor and train RL.
   - This should not be fully simultaneous; RL should see a mostly stationary predictor.

-------------------------------------------------------------------------------
STEP 6: WIRE UNCERTAINTY INTO RL STATE AND EXIT LOGIC
-------------------------------------------------------------------------------

Goal: Use BayesianLinear to expose uncertainty (not just a single “confidence” scalar).

1. In `UnifiedOptionsPredictor`, implement (if not already):

   - K forward passes per state (with BayesianLinear sampling) to estimate:
     - `return_mean`, `return_std` for the target horizon.
     - Direction logits → `dir_probs` and `dir_entropy`.

   OR, if Monte Carlo sampling already exists, make sure these statistics are returned in a consistent structured output.

2. Update the state-building code for:
   - PPO / RL trading policy.
   - Exit policy.

   So that RL states now include:
   - `return_mean`
   - `return_std`
   - `dir_probs` (or at least `dir_long_prob`, `dir_short_prob`, etc.)
   - `dir_entropy` as a measure of uncertainty / ambiguity.

3. Use these in decision logic:
   - Entry: prefer trades with good risk-adjusted return: e.g., `return_mean / (return_std + epsilon)` rather than raw predicted return.
   - Exit: if uncertainty spikes (e.g., `return_std` increases or `dir_entropy` rises beyond a threshold), allow the exit model to be more aggressive.

4. Make sure any “confidence” scalar head is either:
   - Re-defined as a function of these uncertainty metrics, or
   - Removed if redundant.

-------------------------------------------------------------------------------
STEP 7: DEEPER HMM / REGIME INTEGRATION
-------------------------------------------------------------------------------

Goal: The predictor and policies should meaningfully incorporate market regime beyond just passing a scalar.

1. Add regime embedding logic:
   - For each HMM regime (e.g., trending/bullish, trending/bearish, choppy, etc.), create a small embedding vector (e.g., size 8–16).
   - This can be a learned embedding layer or a small MLP mapping regime one-hot to a vector.

2. Integrate regime embedding into:
   - `UnifiedOptionsPredictor`:
     - Concatenate regime embedding to the “current features” and/or sequence latent before the shared head.
   - RL and exit policies:
     - Include the regime embedding in the state vector.

3. OPTIONAL advanced regime-aware policy:
   - Implement a simple mixture-of-experts:
     - Shared trunk → K small “expert” heads.
     - HMM regime embedding → gating network to weight experts.
   - Keep K small (e.g., 2–4) to avoid complexity explosion.

4. Ensure that all data pipelines that produce RL states always include the current regime and its embedding.

-------------------------------------------------------------------------------
STEP 8: SIMPLIFY AND TUNE PREDICTOR REGULARIZATION
-------------------------------------------------------------------------------

Goal: Make sure `UnifiedOptionsPredictor` is not underfitting due to too much dropout / too many Bayesian layers.

1. Inspect the actual predictor implementation and:
   - Note the exact dropout values in each layer.
   - Note where BayesianLinear is used (only heads or also deeper in the backbone).

2. Create an alternative config (e.g., `predictor_v2`) that:
   - Reduces dropout somewhat (example: 0.2 → 0.15 → 0.1 per block).
   - Keeps BayesianLinear primarily in the **final shared head and output heads**.
   - Uses deterministic Linear layers in the backbone for stability.

3. Make this new config selectable via a config option:
   - `predictor.arch` ∈ { "v1_original", "v2_slim_bayesian" }

4. Add a simple evaluation script or notebook to compare:
   - Predictive accuracy.
   - Calibration of probabilities (Brier score, reliability plots) for direction and return bins.

-------------------------------------------------------------------------------
STEP 9: ALIGN TIME HORIZONS END-TO-END
-------------------------------------------------------------------------------

Goal: Ensure that the **prediction horizon**, **RL reward horizon**, and **exit logic timings** are consistent.

1. Locate where labels for:
   - `return` and `volatility` are created (e.g., next 15m, next 30m, until expiry, etc.).
2. Locate where exit policy inputs include:
   - `time_ratio (held/predicted)`,
   - `time_remaining_ratio`,
   - `past_predicted_time`, any other timing-related features.

3. Make the following explicit:
   - Define a clear prediction horizon in config: `prediction_horizon_minutes` (or similar).
   - Ensure label-generation for `return` and `volatility` uses that horizon.
   - Ensure RL reward windows (e.g., when you measure P&L and assign credit) are compatible with this horizon.
   - Ensure exit policy features referencing “predicted time” or “hold time ratio” are derived from the same horizon config.

4. Add assertions or logging that print:
   - Horizon used for label generation.
   - Horizon used in exit logic.
   - Horizon assumed by RL (if any).

-------------------------------------------------------------------------------
STEP 10: CLEANUP, CONFIGS, AND DOCUMENTATION
-------------------------------------------------------------------------------

1. Add config flags that control:
   - `entry_policy.type` ("ppo" vs "unified_policy").
   - `exit_policy.type` ("nn_exit" vs "xgboost_exit").
   - `predictor.arch` ("v1_original" vs "v2_slim_bayesian").
   - `safety_filter.enabled` (true/false).

2. Make sure the main training and live-trading entrypoints read these configs and:
   - Print the chosen architecture options at startup.
   - Fail fast if mismatched combinations are used (e.g., exit policy requested but not implemented).

3. Update or create a new doc `docs/ARCH_FLOW_V2.md` that:
   - Describes the final data flow:
     - Data → Features → `UnifiedOptionsPredictor` → RL state + exit state → Safety filter → Entry/Exit actions.
   - Explicitly calls out:
     - Single entry/exit policies.
     - Where uncertainty is injected.
     - How regime is handled.
     - How time horizon is defined.

4. At the end, show me:
   - A summary of all code changes (files touched, classes added/modified).
   - How to run:
     - Predictor training.
     - RL training.
     - Paper/live trading with the new architecture.

Please implement these changes incrementally with clear commits or at least clearly separated diffs so I can review them. Do not remove existing functionality unless it’s dead/unreachable; instead, deprecate or gate it behind configs where reasonable.
