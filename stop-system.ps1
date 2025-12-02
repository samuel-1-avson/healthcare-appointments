# stop-system.ps1
# Stop all running services

param(
    [switch]$Docker,
    [switch]$All
)

Write-Host "Stopping Healthcare No-Show Prediction System..." -ForegroundColor Yellow

# Stop Docker containers
if ($Docker -or $All) {
    Write-Host "Stopping Docker containers..." -ForegroundColor Gray
    docker compose down 2>&1 | Out-Null
    Write-Host "Docker containers stopped" -ForegroundColor Green
}

# Find and stop Python processes running uvicorn
if (-not $Docker -or $All) {
    Write-Host "Stopping Python processes..." -ForegroundColor Gray
    
    $processes = Get-Process -Name "python" -ErrorAction SilentlyContinue | 
    Where-Object { $_.CommandLine -like "*uvicorn*" -or $_.CommandLine -like "*src.api.main*" }
    
    if ($processes) {
        $processes | Stop-Process -Force
        Write-Host "Stopped $($processes.Count) Python process(es)" -ForegroundColor Green
    }
    else {
        Write-Host "No running API processes found" -ForegroundColor Gray
    }
}

Write-Host ""
Write-Host "System stopped!" -ForegroundColor Green