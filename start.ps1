# start.ps1
# Quick start script - just runs the development server
# Usage: .\start.ps1

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot

Write-Host "Starting Healthcare No-Show Prediction System..." -ForegroundColor Cyan

# Activate venv if exists
$venvActivate = Join-Path $ProjectRoot "venv\Scripts\Activate.ps1"
if (Test-Path $venvActivate) {
    & $venvActivate
}

# Load .env
$envFile = Join-Path $ProjectRoot ".env"
if (Test-Path $envFile) {
    Get-Content $envFile | ForEach-Object {
        if ($_ -match "^\s*([^#][^=]+)=(.*)$") {
            [Environment]::SetEnvironmentVariable($Matches[1].Trim(), $Matches[2].Trim(), "Process")
        }
    }
}

Write-Host ""
Write-Host "API starting at: http://localhost:8000" -ForegroundColor Green
Write-Host "Docs available at: http://localhost:8000/docs" -ForegroundColor Green
Write-Host ""

python -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000