# scripts/docker-test.ps1
# ============================================================
# Test Docker deployment
# ============================================================

param(
    [string]$ApiUrl = "http://localhost:8000"
)

Write-Host "Testing Healthcare API..." -ForegroundColor Cyan
Write-Host ""

$tests = @(
    @{Name = "Health Check"; Url = "/health"; Method = "GET" },
    @{Name = "API Docs"; Url = "/docs"; Method = "GET" },
    @{Name = "Readiness"; Url = "/ready"; Method = "GET" },
    @{Name = "LLM Health"; Url = "/api/v1/llm/health"; Method = "GET" },
    @{Name = "RAG Status"; Url = "/api/v1/rag/index/status"; Method = "GET" }
)

$passed = 0
$failed = 0

foreach ($test in $tests) {
    try {
        $response = Invoke-RestMethod -Uri "$ApiUrl$($test.Url)" -Method $test.Method -TimeoutSec 10
        Write-Host "[PASS] $($test.Name)" -ForegroundColor Green
        $passed++
    }
    catch {
        Write-Host "[FAIL] $($test.Name): $($_.Exception.Message)" -ForegroundColor Red
        $failed++
    }
}

# Test prediction endpoint
Write-Host ""
Write-Host "Testing Prediction API..." -ForegroundColor Cyan

try {
    $body = @{
        age          = 35
        gender       = "F"
        lead_days    = 7
        sms_received = 1
    } | ConvertTo-Json
    
    $response = Invoke-RestMethod -Uri "$ApiUrl/api/v1/predict" -Method POST -Body $body -ContentType "application/json" -TimeoutSec 10
    Write-Host "[PASS] Prediction endpoint" -ForegroundColor Green
    Write-Host "  -> Risk Tier: $($response.risk.tier)" -ForegroundColor Gray
    Write-Host "  -> Probability: $($response.probability)" -ForegroundColor Gray
    $passed++
}
catch {
    Write-Host "[FAIL] Prediction endpoint: $($_.Exception.Message)" -ForegroundColor Red
    $failed++
}

Write-Host ""
Write-Host "Results: $passed passed, $failed failed" -ForegroundColor $(if ($failed -eq 0) { "Green" } else { "Yellow" })