#!/bin/bash
# Pretrain all temporal encoders and run 1-month (20K cycle) tests
#
# This script:
# 1. Pretrains each encoder (TCN, LSTM, Transformer, Mamba2) on all available data
# 2. Runs a 20K cycle test with each pretrained model
# 3. Compares results

set -e

cd /var/ai/simon/gaussian-system

# Create output directory
mkdir -p models/pretrained

echo "============================================================"
echo "PHASE 1: PRETRAINING ALL ENCODERS"
echo "============================================================"

# Pretrain TCN (default)
echo ""
echo ">>> Pretraining TCN encoder..."
TEMPORAL_ENCODER=tcn python scripts/pretrain_predictor.py \
    --epochs 30 \
    --batch-size 128 \
    --output models/pretrained/predictor_tcn.pt \
    --max-samples 500000

# Pretrain LSTM
echo ""
echo ">>> Pretraining LSTM encoder..."
TEMPORAL_ENCODER=lstm python scripts/pretrain_predictor.py \
    --epochs 30 \
    --batch-size 128 \
    --output models/pretrained/predictor_lstm.pt \
    --max-samples 500000

# Pretrain Transformer
echo ""
echo ">>> Pretraining Transformer encoder..."
TEMPORAL_ENCODER=transformer python scripts/pretrain_predictor.py \
    --epochs 30 \
    --batch-size 128 \
    --output models/pretrained/predictor_transformer.pt \
    --max-samples 500000

# Pretrain Mamba2
echo ""
echo ">>> Pretraining Mamba2 encoder..."
TEMPORAL_ENCODER=mamba2 python scripts/pretrain_predictor.py \
    --epochs 30 \
    --batch-size 128 \
    --output models/pretrained/predictor_mamba2.pt \
    --max-samples 500000

echo ""
echo "============================================================"
echo "PHASE 2: 20K CYCLE TESTS WITH PRETRAINED MODELS"
echo "============================================================"

# Test TCN pretrained
echo ""
echo ">>> Testing pretrained TCN (20K cycles)..."
TEMPORAL_ENCODER=tcn \
LOAD_PRETRAINED=1 \
PRETRAINED_MODEL_PATH=models/pretrained/predictor_tcn.pt \
MODEL_RUN_DIR=models/pretrained_test_tcn \
TT_MAX_CYCLES=20000 \
TT_PRINT_EVERY=2000 \
PAPER_TRADING=True \
python scripts/train_time_travel.py

# Test LSTM pretrained
echo ""
echo ">>> Testing pretrained LSTM (20K cycles)..."
TEMPORAL_ENCODER=lstm \
LOAD_PRETRAINED=1 \
PRETRAINED_MODEL_PATH=models/pretrained/predictor_lstm.pt \
MODEL_RUN_DIR=models/pretrained_test_lstm \
TT_MAX_CYCLES=20000 \
TT_PRINT_EVERY=2000 \
PAPER_TRADING=True \
python scripts/train_time_travel.py

# Test Transformer pretrained
echo ""
echo ">>> Testing pretrained Transformer (20K cycles)..."
TEMPORAL_ENCODER=transformer \
LOAD_PRETRAINED=1 \
PRETRAINED_MODEL_PATH=models/pretrained/predictor_transformer.pt \
MODEL_RUN_DIR=models/pretrained_test_transformer \
TT_MAX_CYCLES=20000 \
TT_PRINT_EVERY=2000 \
PAPER_TRADING=True \
python scripts/train_time_travel.py

# Test Mamba2 pretrained
echo ""
echo ">>> Testing pretrained Mamba2 (20K cycles)..."
TEMPORAL_ENCODER=mamba2 \
LOAD_PRETRAINED=1 \
PRETRAINED_MODEL_PATH=models/pretrained/predictor_mamba2.pt \
MODEL_RUN_DIR=models/pretrained_test_mamba2 \
TT_MAX_CYCLES=20000 \
TT_PRINT_EVERY=2000 \
PAPER_TRADING=True \
python scripts/train_time_travel.py

echo ""
echo "============================================================"
echo "RESULTS COMPARISON"
echo "============================================================"

# Show results
for encoder in tcn lstm transformer mamba2; do
    echo ""
    echo "=== $encoder (pretrained) ==="
    if [ -f "models/pretrained_test_$encoder/SUMMARY.txt" ]; then
        grep -E "P&L:|Win Rate:|Total Trades:" "models/pretrained_test_$encoder/SUMMARY.txt" || true
    else
        echo "No results found"
    fi
done

echo ""
echo "Done!"
