#!/bin/bash
#
# Gaussian Trading Bot - Distributed Optimizer Startup
#
# This script starts the continuous optimizer with distributed sync support.
# Multiple machines can run this simultaneously - they coordinate via GitHub.
#
# Usage:
#   ./start_optimizer.sh                    # Start with defaults
#   ./start_optimizer.sh --dry-run          # Test without running experiments
#   MACHINE_ID=gpu-server-1 ./start_optimizer.sh   # Set custom machine ID
#
# Environment Variables:
#   MACHINE_ID           - Unique machine identifier (default: hostname)
#   PARALLEL_EXPERIMENTS - Number of parallel experiments (default: 4)
#   QUICK_TEST_CYCLES    - Cycles for quick test (default: 5000)
#   VALIDATION_CYCLES    - Cycles for validation (default: 20000)
#   GIT_SYNC_ENABLED     - Enable GitHub sync (default: 1)
#

set -e

# Change to script directory
cd "$(dirname "$0")"

# Set default machine ID if not provided
export MACHINE_ID="${MACHINE_ID:-$(hostname | cut -c1-20)}"

# Create logs directory
mkdir -p logs

echo "=============================================="
echo "Gaussian Distributed Optimizer"
echo "=============================================="
echo "Machine ID: $MACHINE_ID"
echo "Parallel:   ${PARALLEL_EXPERIMENTS:-4} experiments"
echo "Quick test: ${QUICK_TEST_CYCLES:-5000} cycles"
echo "Validation: ${VALIDATION_CYCLES:-20000} cycles"
echo "Git Sync:   ${GIT_SYNC_ENABLED:-1}"
echo "=============================================="
echo ""

# Check if already running
if pgrep -f "continuous_optimizer.py" > /dev/null 2>&1; then
    echo "WARNING: Optimizer may already be running!"
    echo "PIDs: $(pgrep -f continuous_optimizer.py | tr '\n' ' ')"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Pull latest from GitHub
echo "Pulling latest from GitHub..."
git pull --rebase origin main || echo "Warning: Git pull failed, continuing anyway..."

# Start optimizer
if [[ "$1" == "--foreground" ]] || [[ "$1" == "-f" ]]; then
    echo "Starting optimizer in foreground..."
    python scripts/continuous_optimizer.py "$@"
else
    echo "Starting optimizer in background..."
    nohup python scripts/continuous_optimizer.py "$@" > logs/optimizer_output.log 2>&1 &
    PID=$!
    echo "Started with PID: $PID"
    echo ""
    echo "Monitor with:  tail -f logs/optimizer_output.log"
    echo "Stop with:     kill $PID"
    echo ""

    # Wait a moment and show initial output
    sleep 3
    echo "Initial output:"
    echo "---------------"
    tail -20 logs/optimizer_output.log
fi
