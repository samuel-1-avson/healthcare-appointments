# test-system.ps1
# ============================================================
# Run tests and health checks
# ============================================================

param(
    [switch]$Unit,
    [switch]$Integration,
    [switch]$Scripts,
    [switch]$All,
    [switch]$HealthCheck,
    [string]$ApiUrl = "http://localhost:8000"
)

$ErrorActionPreference = "Continue"
$ProjectRoot = $PSScriptRoot

function Write-TestHeader {
    param([string]$Title)
    Write-Host ""
    Write-Host "Testing: $Title" -ForegroundColor Cyan
    Write-Host ("-" * 40) -ForegroundColor DarkGray
}

function Test-Endpoint {
    param(
        [string]$Name,
        [string]$Url,
        [string]$Method = "GET",
        [string]$Body = $null
    )
    
    try {
        $params = @{
            Uri         = $Url
            Method      = $Method
            ContentType = "application/json"
            TimeoutSec  = 10
        }
        
        if ($Body) {
            $params.Body = $Body
        }
        
        $response = Invoke-RestMethod @params
        Write-Host "[✓] $Name" -ForegroundColor Green
        return $true
    }
    catch {
        Write-Host "[✗] $Name - $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
}

function Run-TestScripts {
    Write-TestHeader "Standalone Test Scripts"
    
    # Activate venv if not already active
    $venvActivate = Join-Path $ProjectRoot "venv\Scripts\Activate.ps1"
    if (Test-Path $venvActivate) {
        & $venvActivate
    }

    $scripts = @(
        @{ Name = "Day 1: Data Loading"; Path = "test_day1.py" },
        @{ Name = "Day 2: Cleaning & Features"; Path = "test_day2.py" },
        @{ Name = "Day 3: Risk Scoring"; Path = "test_day3.py" }
    )

    foreach ($script in $scripts) {
        Write-Host "Running $($script.Name)..." -ForegroundColor Yellow
        $scriptPath = Join-Path $ProjectRoot $script.Path
        
        if (Test-Path $scriptPath) {
            python $scriptPath
            if ($LASTEXITCODE -eq 0) {
                Write-Host "[✓] $($script.Name) passed" -ForegroundColor Green
            }
            else {
                Write-Host "[✗] $($script.Name) failed" -ForegroundColor Red
            }
        }
        else {
            Write-Host "[!] Script not found: $($script.Path)" -ForegroundColor Red
        }
        Write-Host ""
    }
}

# Activate venv
$venvActivate = Join-Path $ProjectRoot "venv\Scripts\Activate.ps1"
if (Test-Path $venvActivate) {
    & $venvActivate
}

# Health Check
if ($HealthCheck -or $All) {
    Write-TestHeader "API Health Checks"
    
    $passed = 0
    $total = 0
    
    # Health endpoint
    $total++
    if (Test-Endpoint "Health Check" "$ApiUrl/health") { $passed++ }
    
    # Ready endpoint
    $total++
    if (Test-Endpoint "Readiness Check" "$ApiUrl/ready") { $passed++ }
    
    # Live endpoint
    $total++
    if (Test-Endpoint "Liveness Check" "$ApiUrl/live") { $passed++ }
    
    # Docs endpoint
    $total++
    if (Test-Endpoint "API Docs" "$ApiUrl/docs") { $passed++ }
    
    # Prediction endpoint
    $total++
    $predictionBody = '{"age": 35, "gender": "F", "lead_days": 7, "sms_received": 1}'
    if (Test-Endpoint "Prediction Endpoint" "$ApiUrl/api/v1/predict" "POST" $predictionBody) { $passed++ }
    
    # LLM Health
    $total++
    if (Test-Endpoint "LLM Health" "$ApiUrl/api/v1/llm/health") { $passed++ }
    
    # RAG Status
    $total++
    if (Test-Endpoint "RAG Index Status" "$ApiUrl/api/v1/rag/index/status") { $passed++ }
    
    Write-Host ""
    Write-Host "Results: $passed/$total endpoints passed" -ForegroundColor $(if ($passed -eq $total) { "Green" } else { "Yellow" })
}

# Unit Tests
if ($Unit -or $All) {
    Write-TestHeader "Unit Tests"
    python -m pytest tests/ -v --ignore=tests/test_api --ignore=tests/test_async_predictions.py --ignore=tests/test_observability.py --ignore=tests/test_e2e_staging.py --ignore=tests/test_redis.py --ignore=tests/chaos/ --ignore=tests/e2e/
}

# Integration Tests
if ($Integration -or $All) {
    Write-TestHeader "Integration Tests"
    python -m pytest tests/test_api/ -v
}

# Standalone Scripts
if ($Scripts -or $All) {
    Run-TestScripts
}

# If no specific test selected, run health check
if (-not ($Unit -or $Integration -or $Scripts -or $All -or $HealthCheck)) {
    Write-Host "Usage: .\test-system.ps1 [-Unit] [-Integration] [-Scripts] [-All] [-HealthCheck]" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Examples:" -ForegroundColor Gray
    Write-Host "  .\test-system.ps1 -HealthCheck    # Test API endpoints"
    Write-Host "  .\test-system.ps1 -Unit           # Run unit tests"
    Write-Host "  .\test-system.ps1 -Scripts        # Run standalone test scripts"
    Write-Host "  .\test-system.ps1 -All            # Run all tests"
}