#!/bin/bash
# =============================================================================
# LIVE TRADING LAUNCHER - SKIP_MONDAY Strategy
# =============================================================================
# Best Model: COMBO_SKIP_MONDAY_20K
# P&L: +1630.24% | P&L/DD: 35.03 | Win Rate: 43.0%
# =============================================================================

# Configuration (from winning 20K validation)
export USE_TRAILING_STOP=1
export TRAILING_ACTIVATION_PCT=10
export TRAILING_STOP_PCT=5
export ENABLE_TDA=1
export TDA_REGIME_FILTER=1
export TRAIN_MAX_CONF=0.25
export DAY_OF_WEEK_FILTER=1
export SKIP_MONDAY=1
export SKIP_FRIDAY=0

# Model directory
MODEL_DIR="models/COMBO_SKIP_MONDAY_20K"

echo "============================================================"
echo "SKIP_MONDAY Strategy - LIVE TRADING"
echo "============================================================"
echo "Model:     $MODEL_DIR"
echo "P&L:       +1630.24% (20K validation)"
echo "P&L/DD:    35.03"
echo "Win Rate:  43.0%"
echo ""
echo "Configuration:"
echo "  - SKIP_MONDAY=1 (no Monday trades)"
echo "  - USE_TRAILING_STOP=1 (10% activation, 5% trail)"
echo "  - TRAIN_MAX_CONF=0.25 (low confidence = better)"
echo "  - TDA_REGIME_FILTER=1 (regime filtering)"
echo "============================================================"
echo ""

# Check for go_live.flag
if [ -f "go_live.flag" ]; then
    echo "WARNING: go_live.flag EXISTS - LIVE TRADING ENABLED!"
else
    echo "Paper trading mode (create 'go_live.flag' to enable live)"
fi
echo ""

# Run the bot
cd /var/ai/simon/gaussian-system
python core/go_live_only.py "$MODEL_DIR"
