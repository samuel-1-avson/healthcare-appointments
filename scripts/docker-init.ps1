# scripts/docker-init.ps1
# ============================================================
# First-time Docker setup for Healthcare System
# ============================================================

param(
    [switch]$SkipBuild,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path $PSScriptRoot -Parent

Write-Host @"

╔═══════════════════════════════════════════════════════════╗
║       Healthcare System - Docker Initialization           ║
╚═══════════════════════════════════════════════════════════╝

"@ -ForegroundColor Cyan

Set-Location $ProjectRoot

# Step 1: Create directory structure
Write-Host "[1/6] Creating directory structure..." -ForegroundColor Cyan

$directories = @(
    "requirements",
    "scripts",
    "data/documents",
    "data/raw",
    "data/processed",
    "data/vector_store",
    "models/production",
    "logs",
    "evals/results",
    "prompts",
    "nginx"
)

foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "  Created: $dir" -ForegroundColor Gray
    }
}
Write-Host "  [OK] Directories ready" -ForegroundColor Green

# Step 2: Create requirements files if missing
Write-Host "`n[2/6] Checking requirements files..." -ForegroundColor Cyan

$reqFiles = @{
    "requirements/base.txt" = @"
pandas>=2.0.0,<3.0.0
numpy>=1.24.0,<2.0.0
scipy>=1.10.0,<2.0.0
scikit-learn>=1.3.0,<2.0.0
joblib>=1.3.0,<2.0.0
pyyaml>=6.0,<7.0
python-dotenv>=1.0.0,<2.0.0
tqdm>=4.65.0,<5.0.0
"@
    "requirements/api.txt"  = @"
fastapi>=0.100.0,<1.0.0
uvicorn[standard]>=0.23.0,<1.0.0
gunicorn>=21.0.0,<22.0.0
pydantic>=2.0.0,<3.0.0
pydantic-settings>=2.0.0,<3.0.0
httpx>=0.24.0,<1.0.0
aiohttp>=3.9.0,<4.0.0
python-multipart>=0.0.6,<1.0.0
"@
    "requirements/llm.txt"  = @"
openai>=1.12.0,<2.0.0
tiktoken>=0.5.0,<1.0.0
langchain>=0.2.0,<0.3.0
langchain-core>=0.2.0,<0.3.0
langchain-openai>=0.1.0,<0.2.0
langchain-community>=0.2.0,<0.3.0
faiss-cpu>=1.7.4,<1.8.0
tenacity>=8.2.0,<9.0.0
"@
    "requirements/dev.txt"  = @"
pytest>=7.3.0,<8.0.0
pytest-cov>=4.0.0,<5.0.0
black>=23.3.0,<24.0.0
flake8>=6.0.0,<7.0.0
"@
}

foreach ($file in $reqFiles.Keys) {
    if (-not (Test-Path $file) -or $Force) {
        $reqFiles[$file] | Out-File -FilePath $file -Encoding utf8
        Write-Host "  Created: $file" -ForegroundColor Gray
    }
}
Write-Host "  [OK] Requirements files ready" -ForegroundColor Green

# Step 3: Create .env file
Write-Host "`n[3/6] Setting up environment..." -ForegroundColor Cyan

if (-not (Test-Path ".env") -or $Force) {
    @"
# Healthcare No-Show Prediction System - Docker Environment
# ============================================================

# API Configuration
API_PORT=8000
DEBUG=false
LOG_LEVEL=INFO
WORKERS=2

# LLM Configuration (Required for LLM features)
OPENAI_API_KEY=your-openai-key-here
# ANTHROPIC_API_KEY=your-anthropic-key-here

# LLM Settings
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
LLM_CACHE=true

# LangChain Tracing (Optional)
LANGCHAIN_TRACING=false
# LANGCHAIN_API_KEY=your-langsmith-key
LANGCHAIN_PROJECT=healthcare-assistant

# Optional Services
# REDIS_PORT=6379
# NGINX_PORT=80
"@ | Out-File ".env" -Encoding utf8
    Write-Host "  Created: .env" -ForegroundColor Gray
    Write-Host "  [IMPORTANT] Edit .env and add your OPENAI_API_KEY" -ForegroundColor Yellow
}
Write-Host "  [OK] Environment configured" -ForegroundColor Green

# Step 4: Create sample policy document
Write-Host "`n[4/6] Creating sample documents..." -ForegroundColor Cyan

$sampleDoc = "data/documents/appointment_policy.md"
if (-not (Test-Path $sampleDoc) -or $Force) {
    @"
# Healthcare Clinic Appointment Policy

## 1. Scheduling Appointments

Patients may schedule appointments through:
- **Online Portal:** Available 24/7 at patient.clinic.com
- **Phone:** Call (555) 123-4567, Monday-Friday 8am-6pm
- **In Person:** Visit our front desk during business hours

## 2. Cancellation Policy

- **24+ hours notice:** No penalty, full flexibility
- **12-24 hours notice:** Recorded as late cancellation
- **Less than 12 hours:** May be recorded as no-show

## 3. No-Show Policy

A "no-show" occurs when a patient:
- Fails to arrive for their scheduled appointment
- Arrives more than 15 minutes late without notice

### Consequences
| No-Shows (12 months) | Action |
|---------------------|--------|
| 1st | Verbal reminder |
| 2nd | Written warning |
| 3rd | `$75 fee, pre-payment required |
| 4th | Review for possible discharge |

## 4. Reminder System

Patients receive automatic reminders:
- 7 days before: Email
- 48 hours before: SMS
- 24 hours before: SMS with confirmation request
"@ | Out-File -FilePath $sampleDoc -Encoding utf8
    Write-Host "  Created: $sampleDoc" -ForegroundColor Gray
}
Write-Host "  [OK] Sample documents ready" -ForegroundColor Green

# Step 5: Create entrypoint script
Write-Host "`n[5/6] Creating Docker scripts..." -ForegroundColor Cyan

$entrypoint = "scripts/docker-entrypoint.sh"
if (-not (Test-Path $entrypoint) -or $Force) {
    @"
#!/bin/bash
set -e

echo "Starting Healthcare No-Show Prediction API"
echo "Python version: `$(python --version)"

case "`$1" in
    api)
        exec gunicorn src.api.main:app \
            --bind `${NOSHOW_HOST:-0.0.0.0}:`${NOSHOW_PORT:-8000} \
            --workers `${NOSHOW_WORKERS:-2} \
            --worker-class uvicorn.workers.UvicornWorker \
            --access-logfile - \
            --error-logfile -
        ;;
    dev)
        exec uvicorn src.api.main:app \
            --host `${NOSHOW_HOST:-0.0.0.0} \
            --port `${NOSHOW_PORT:-8000} \
            --reload
        ;;
    *)
        exec "`$@"
        ;;
esac
"@ | Out-File -FilePath $entrypoint -Encoding utf8 -NoNewline
    Write-Host "  Created: $entrypoint" -ForegroundColor Gray
}
Write-Host "  [OK] Docker scripts ready" -ForegroundColor Green

# Step 6: Build Docker images
if (-not $SkipBuild) {
    Write-Host "`n[6/6] Building Docker images..." -ForegroundColor Cyan
    Write-Host "  This may take several minutes on first run..." -ForegroundColor Gray
    
    docker compose build
    
    Write-Host "  [OK] Docker images built" -ForegroundColor Green
}
else {
    Write-Host "`n[6/6] Skipping Docker build (-SkipBuild)" -ForegroundColor Yellow
}

# Summary
Write-Host @"

╔═══════════════════════════════════════════════════════════╗
║                 Initialization Complete!                  ║
╚═══════════════════════════════════════════════════════════╝

Next Steps:
-----------
1. Edit .env and add your OPENAI_API_KEY
2. Start the system:
   
   .\scripts\docker-start.ps1              # Development mode
   .\scripts\docker-start.ps1 -Mode prod   # Production mode
   .\scripts\docker-start.ps1 -Help        # See all options

3. Access the API:
   - API Docs:  http://localhost:8000/docs
   - Health:    http://localhost:8000/health

"@ -ForegroundColor Cyan