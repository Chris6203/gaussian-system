# Run with Transformer encoder (best OOS generalization from Phase 35)
Set-Location "E:\gaussian-system"
$env:TEMPORAL_ENCODER = "transformer"
$env:MODEL_RUN_DIR = "models/transformer_jan2026_test"
$env:TT_MAX_CYCLES = "5000"
$env:TT_PRINT_EVERY = "500"
$env:PAPER_TRADING = "True"
# Use recent data
$env:TRAINING_START_DATE = "2025-12-01"
$env:TRAINING_END_DATE = "2025-12-31"
Write-Host "Starting transformer encoder test (best OOS generalization)..."
Write-Host "Config: TEMPORAL_ENCODER=transformer, Dec 2025 data, 5000 cycles"
python scripts/train_time_travel.py
