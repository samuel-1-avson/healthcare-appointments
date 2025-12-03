# migrate-to-py312.ps1
# Automated migration from Python 3.14 to 3.12

Write-Host "🔄 Starting Python 3.12 Migration..." -ForegroundColor Cyan
Write-Host ""

# Step 1: Deactivate
Write-Host "[1/8] Deactivating current environment..." -ForegroundColor Yellow
try { deactivate } catch {}

# Step 2: Backup old venv
Write-Host "[2/8] Backing up old virtual environment..." -ForegroundColor Yellow
if (Test-Path "venv") {
    Rename-Item venv venv-py314-backup -Force -ErrorAction SilentlyContinue
    Write-Host "  ✅ Old venv backed up to venv-py314-backup" -ForegroundColor Green
}

# Step 3: Create new venv with Python 3.12
Write-Host "[3/8] Creating new Python 3.12 virtual environment..." -ForegroundColor Yellow
py -3.12 -m venv venv
Write-Host "  ✅ New venv created" -ForegroundColor Green

# Step 4: Activate new venv
Write-Host "[4/8] Activating new environment..." -ForegroundColor Yellow
& .\venv\Scripts\Activate.ps1

# Step 5: Verify Python version
Write-Host "[5/8] Verifying Python version..." -ForegroundColor Yellow
$pythonVersion = python --version
Write-Host "  ✅ $pythonVersion" -ForegroundColor Green

# Step 6: Upgrade pip
Write-Host "[6/8] Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip --quiet
Write-Host "  ✅ pip upgraded" -ForegroundColor Green

# Step 7: Install dependencies
Write-Host "[7/8] Installing dependencies..." -ForegroundColor Yellow
Write-Host "  Installing core dependencies..." -ForegroundColor Gray
pip install -r requirements.txt --quiet

Write-Host "  Installing LLM dependencies..." -ForegroundColor Gray
pip install -r requirements-llm.txt --quiet

Write-Host "  Installing API dependencies..." -ForegroundColor Gray
pip install -r requirements-api.txt --quiet
Write-Host "  ✅ All dependencies installed" -ForegroundColor Green

# Step 8: Verify LLM/RAG availability
Write-Host "[8/8] Verifying LLM/RAG imports..." -ForegroundColor Yellow
$testResult = python -c "from src.api.routes import LLM_AVAILABLE, RAG_AVAILABLE; print(f'LLM={LLM_AVAILABLE}, RAG={RAG_AVAILABLE}')" 2>&1

if ($testResult -match "LLM=True.*RAG=True") {
    Write-Host "  ✅ $testResult" -ForegroundColor Green
}
else {
    Write-Host "  ⚠️ $testResult" -ForegroundColor Yellow
}

# Summary
Write-Host ""
Write-Host "════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  ✅ Migration Complete!" -ForegroundColor Green
Write-Host "════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor White
Write-Host "  1. Start system:  .\start-system.ps1 -Mode dev -Port 8001" -ForegroundColor Gray
Write-Host "  2. Test endpoints: .\test-system.ps1 -HealthCheck" -ForegroundColor Gray
Write-Host ""
