@echo off
echo Running dec_validation_v2 validation test...
set LOAD_PRETRAINED=1
set PRETRAINED_MODEL_PATH=models/dec_validation_v2/state/trained_model.pth
set MODEL_RUN_DIR=models/dec_v2_validation_test
set TT_MAX_CYCLES=5000
set TT_PRINT_EVERY=500
set PAPER_TRADING=True
python scripts/train_time_travel.py
