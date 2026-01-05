#!/bin/bash
# Winning Configuration Presets
# Source this file before running experiments
# Usage: source configs/winning_presets.sh && use_skew_optimized

# =============================================================================
# PRESET: BASELINE (Proven +274% P&L)
# =============================================================================
use_baseline() {
    echo "=== BASELINE PRESET (v3_calibrated: +274% P&L) ==="
    export PREDICTOR_ARCH=v3_multi_horizon
    export TT_MAX_CYCLES=5000
    export PAPER_TRADING=True
    # No extra gates - stick to what works
    unset EV_GATE_ENABLED
    unset REGIME_CALIBRATION
    unset REGIME_AUTO_DISABLE
    unset GREEKS_AWARE_EXITS
    unset SKEW_EXIT_ENABLED
    echo "Environment set for baseline run"
}

# =============================================================================
# PRESET: SKEW-OPTIMIZED (Capture fat-tail winners)
# =============================================================================
use_skew_optimized() {
    echo "=== SKEW-OPTIMIZED PRESET (partial TP + trailing runner) ==="
    export PREDICTOR_ARCH=v3_multi_horizon
    export TT_MAX_CYCLES=5000
    export PAPER_TRADING=True

    # Skew exit manager
    export SKEW_EXIT_ENABLED=1
    export SKEW_EXIT_MODE=partial
    export PARTIAL_TP_PCT=0.10
    export PARTIAL_TAKE_FRACTION=0.50
    export RUNNER_TRAIL_ACTIVATION=0.15
    export RUNNER_TRAIL_DISTANCE=0.05

    echo "Environment set for skew-optimized run"
}

# =============================================================================
# PRESET: TREND-ADAPTIVE (Different behavior in trends vs chop)
# =============================================================================
use_trend_adaptive() {
    echo "=== TREND-ADAPTIVE PRESET (trailing in trends, partial in chop) ==="
    export PREDICTOR_ARCH=v3_multi_horizon
    export TT_MAX_CYCLES=5000
    export PAPER_TRADING=True

    # Skew exit with trend adaptation
    export SKEW_EXIT_ENABLED=1
    export SKEW_EXIT_MODE=trend_adaptive
    export PARTIAL_TP_PCT=0.10
    export PARTIAL_TAKE_FRACTION=0.50
    export RUNNER_TRAIL_ACTIVATION=0.15
    export RUNNER_TRAIL_DISTANCE=0.05

    echo "Environment set for trend-adaptive run"
}

# =============================================================================
# PRESET: FULL ARCHITECTURE V4 (All improvements)
# =============================================================================
use_full_v4() {
    echo "=== FULL V4 PRESET (all architecture improvements) ==="
    export PREDICTOR_ARCH=v3_multi_horizon
    export TT_MAX_CYCLES=5000
    export PAPER_TRADING=True

    # EV Gate with Bayesian prior
    export EV_GATE_ENABLED=1
    export EV_WIN_PROB_PRIOR=0.40
    export EV_PRIOR_WEIGHT=0.3

    # Regime tracking
    export REGIME_CALIBRATION=1
    export REGIME_AUTO_DISABLE=1

    # Greeks-aware exits
    export GREEKS_AWARE_EXITS=1

    # Skew exits
    export SKEW_EXIT_ENABLED=1
    export SKEW_EXIT_MODE=trend_adaptive

    echo "Environment set for full V4 run"
}

# =============================================================================
# PRESET: CONSERVATIVE (Only proven improvements)
# =============================================================================
use_conservative() {
    echo "=== CONSERVATIVE PRESET (proven improvements only) ==="
    export PREDICTOR_ARCH=v3_multi_horizon
    export TT_MAX_CYCLES=5000
    export PAPER_TRADING=True

    # Just Greeks-aware exits (blended with regime stops)
    export GREEKS_AWARE_EXITS=1

    echo "Environment set for conservative run"
}

# =============================================================================
# BEST EXPERIMENT REPLICA (EXP-0069: +813% P&L)
# =============================================================================
use_exp0069_replica() {
    echo "=== EXP-0069 REPLICA (+813% P&L) ==="
    export PREDICTOR_ARCH=v3_multi_horizon
    export TT_MAX_CYCLES=5000
    export PAPER_TRADING=True

    # This was run with default settings
    # Key: 42.3% conditional WR in that time period
    echo "Note: This experiment's success was timing-dependent"
    echo "Environment set for EXP-0069 replica"
}

# =============================================================================
# HELPER: Show current config
# =============================================================================
show_config() {
    echo "=== Current Configuration ==="
    echo "PREDICTOR_ARCH: ${PREDICTOR_ARCH:-default}"
    echo "TT_MAX_CYCLES: ${TT_MAX_CYCLES:-5000}"
    echo "PAPER_TRADING: ${PAPER_TRADING:-True}"
    echo ""
    echo "EV_GATE_ENABLED: ${EV_GATE_ENABLED:-0}"
    echo "EV_WIN_PROB_PRIOR: ${EV_WIN_PROB_PRIOR:-0.40}"
    echo "EV_PRIOR_WEIGHT: ${EV_PRIOR_WEIGHT:-0.3}"
    echo ""
    echo "REGIME_CALIBRATION: ${REGIME_CALIBRATION:-0}"
    echo "REGIME_AUTO_DISABLE: ${REGIME_AUTO_DISABLE:-0}"
    echo ""
    echo "GREEKS_AWARE_EXITS: ${GREEKS_AWARE_EXITS:-0}"
    echo ""
    echo "SKEW_EXIT_ENABLED: ${SKEW_EXIT_ENABLED:-0}"
    echo "SKEW_EXIT_MODE: ${SKEW_EXIT_MODE:-partial}"
}

# =============================================================================
# HELPER: Run experiment with current config
# =============================================================================
run_experiment() {
    local name="${1:-experiment}"
    export MODEL_RUN_DIR="models/${name}_$(date +%Y%m%d_%H%M%S)"
    echo "Running experiment: $MODEL_RUN_DIR"
    show_config
    python scripts/train_time_travel.py
}

echo "Winning presets loaded! Available commands:"
echo "  use_baseline        - Proven +274% baseline"
echo "  use_skew_optimized  - Partial TP + trailing runner"
echo "  use_trend_adaptive  - Trailing in trends, partial in chop"
echo "  use_full_v4         - All V4 improvements"
echo "  use_conservative    - Only proven improvements"
echo "  use_exp0069_replica - EXP-0069 replica (+813%)"
echo "  show_config         - Show current environment"
echo "  run_experiment NAME - Run with current config"
