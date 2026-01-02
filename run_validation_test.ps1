# Run dec_validation_v2 validation on December 2025 (original validation period)
Set-Location "E:\gaussian-system"
$env:LOAD_PRETRAINED = "1"
$env:PRETRAINED_MODEL_PATH = "models/dec_validation_v2/state/trained_model.pth"
$env:MODEL_RUN_DIR = "models/dec_v2_dec2025_retest"
$env:TT_MAX_CYCLES = "3000"
$env:TT_PRINT_EVERY = "500"
$env:PAPER_TRADING = "True"
# Use December 2025 date range (original validation period)
$env:TRAINING_START_DATE = "2025-12-01"
$env:TRAINING_END_DATE = "2025-12-24"
Write-Host "Starting validation test on December 2025 (original validation period)..."
python scripts/train_time_travel.py
