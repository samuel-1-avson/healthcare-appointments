# scripts/train-model.ps1
# Train or retrain the ML model

param(
    [string]$DataPath = "data/processed/appointments_features.csv",
    [string]$OutputDir = "models/production"
)

$ProjectRoot = Split-Path $PSScriptRoot -Parent
Set-Location $ProjectRoot

# Activate venv
$venvActivate = Join-Path $ProjectRoot "venv\Scripts\Activate.ps1"
if (Test-Path $venvActivate) { & $venvActivate }

Write-Host "Training ML Model..." -ForegroundColor Cyan

python -c @"
import sys
sys.path.insert(0, '.')

from src.ml.train import train_model
train_model('$DataPath', '$OutputDir')
"@

Write-Host "Model training complete!" -ForegroundColor Green