# setup.ps1
# ============================================================
# Healthcare No-Show Prediction System - Setup Script
# Run this once to set up the development environment
# ============================================================

param(
    [switch]$SkipVenv,
    [switch]$SkipDeps,
    [switch]$SkipModel,
    [switch]$SkipRAG
)

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot

function Write-Step {
    param([int]$Number, [string]$Message)
    Write-Host ""
    Write-Host "[$Number] $Message" -ForegroundColor Cyan
    Write-Host ("-" * 50) -ForegroundColor DarkGray
}

Write-Host @"

╔═══════════════════════════════════════════════════════════╗
║       Healthcare No-Show Prediction System Setup          ║
╚═══════════════════════════════════════════════════════════╝

"@ -ForegroundColor Cyan

# Step 1: Create virtual environment
if (-not $SkipVenv) {
    Write-Step 1 "Creating Virtual Environment"
    
    $venvPath = Join-Path $ProjectRoot "venv"
    
    if (Test-Path $venvPath) {
        Write-Host "Virtual environment already exists. Skipping..." -ForegroundColor Yellow
    }
    else {
        python -m venv $venvPath
        Write-Host "Virtual environment created!" -ForegroundColor Green
    }
    
    # Activate
    $activateScript = Join-Path $venvPath "Scripts\Activate.ps1"
    & $activateScript
    Write-Host "Virtual environment activated!" -ForegroundColor Green
}

# Step 2: Install dependencies
if (-not $SkipDeps) {
    Write-Step 2 "Installing Dependencies"
    
    Write-Host "Upgrading pip..." -ForegroundColor Gray
    python -m pip install --upgrade pip -q
    
    $requirements = Join-Path $ProjectRoot "requirements.txt"
    if (Test-Path $requirements) {
        Write-Host "Installing core dependencies..." -ForegroundColor Gray
        pip install -r $requirements -q
        Write-Host "Core dependencies installed!" -ForegroundColor Green
    }
    
    $requirementsLLM = Join-Path $ProjectRoot "requirements-llm.txt"
    if (Test-Path $requirementsLLM) {
        Write-Host "Installing LLM dependencies..." -ForegroundColor Gray
        pip install -r $requirementsLLM -q
        Write-Host "LLM dependencies installed!" -ForegroundColor Green
    }
}

# Step 3: Create directories
Write-Step 3 "Creating Directory Structure"

$directories = @(
    "data\raw",
    "data\processed",
    "data\documents",
    "data\vector_store",
    "models\production",
    "outputs\figures",
    "outputs\reports",
    "evals\results",
    "logs"
)

foreach ($dir in $directories) {
    $fullPath = Join-Path $ProjectRoot $dir
    if (-not (Test-Path $fullPath)) {
        New-Item -ItemType Directory -Path $fullPath -Force | Out-Null
        Write-Host "Created: $dir" -ForegroundColor Gray
    }
}
Write-Host "Directory structure ready!" -ForegroundColor Green

# Step 4: Create .env file
Write-Step 4 "Configuring Environment"

$envFile = Join-Path $ProjectRoot ".env"
$envExample = Join-Path $ProjectRoot ".env.example"

if (-not (Test-Path $envFile)) {
    if (Test-Path $envExample) {
        Copy-Item $envExample $envFile
        Write-Host "Created .env from .env.example" -ForegroundColor Green
    }
    else {
        @"
# Healthcare No-Show Prediction System Configuration
# ==================================================

# API Keys (Required for LLM features)
OPENAI_API_KEY=your-openai-key-here
# ANTHROPIC_API_KEY=your-anthropic-key-here

# API Configuration
NOSHOW_HOST=0.0.0.0
NOSHOW_PORT=8000
NOSHOW_DEBUG=true
NOSHOW_LOG_LEVEL=INFO

# Model Paths
NOSHOW_MODEL_PATH=models/production/model.joblib
NOSHOW_PREPROCESSOR_PATH=models/production/preprocessor.joblib

# LLM Configuration
NOSHOW_LLM_DEFAULT_PROVIDER=openai
NOSHOW_LLM_DEFAULT_MODEL=gpt-4o-mini
NOSHOW_LLM_CACHE_ENABLED=true

# LangChain (Optional - for tracing)
# LANGCHAIN_API_KEY=your-langsmith-key
# LANGCHAIN_PROJECT=healthcare-assistant
# LANGCHAIN_TRACING_V2=true
"@ | Out-File -FilePath $envFile -Encoding utf8
        Write-Host "Created default .env file" -ForegroundColor Green
    }
    
    Write-Host ""
    Write-Host "IMPORTANT: Edit .env and add your OpenAI API key!" -ForegroundColor Yellow
}
else {
    Write-Host ".env file already exists" -ForegroundColor Gray
}

# Step 5: Create sample documents for RAG
Write-Step 5 "Creating Sample Documents"

$docsPath = Join-Path $ProjectRoot "data\documents"
$sampleDoc = Join-Path $docsPath "appointment_policy.md"

if (-not (Test-Path $sampleDoc)) {
    @"
# Healthcare Clinic Appointment Policy

## 1. Scheduling Appointments

Patients may schedule appointments through:
- Online Portal: Available 24/7
- Phone: Call (555) 123-4567, Monday-Friday 8am-6pm
- In Person: Visit our front desk during business hours

## 2. Cancellation Policy

- **24+ hours notice:** No penalty
- **Less than 24 hours:** May be recorded as late cancellation
- **No notice:** Recorded as no-show

## 3. No-Show Policy

A "no-show" occurs when a patient:
- Fails to arrive for their scheduled appointment
- Arrives more than 15 minutes late without prior notice

### Consequences
| No-Show Count | Action |
|---------------|--------|
| 1st | Verbal reminder |
| 2nd | Written warning |
| 3rd | \$75 fee, pre-payment required |

## 4. Reminder System

Patients receive automatic reminders:
- 7 days before: Email
- 48 hours before: SMS
- 24 hours before: SMS with confirmation request
"@ | Out-File -FilePath $sampleDoc -Encoding utf8
    Write-Host "Created sample policy document" -ForegroundColor Green
}
else {
    Write-Host "Sample documents already exist" -ForegroundColor Gray
}

# Step 6: Verify installation
Write-Step 6 "Verifying Installation"

try {
    python -c "import fastapi; print('FastAPI:', fastapi.__version__)"
    python -c "import sklearn; print('scikit-learn:', sklearn.__version__)"
    python -c "import langchain; print('LangChain:', langchain.__version__)"
    python -c "import langchain_openai; print('LangChain-OpenAI: OK')"
    Write-Host ""
    Write-Host "All core packages installed successfully!" -ForegroundColor Green
}
catch {
    Write-Host "Some packages may be missing. Run: pip install -r requirements.txt -r requirements-llm.txt" -ForegroundColor Yellow
}

# Summary
Write-Host @"

╔═══════════════════════════════════════════════════════════╗
║                    Setup Complete!                        ║
╚═══════════════════════════════════════════════════════════╝

Next Steps:
-----------
1. Edit .env file and add your OpenAI API key
2. (Optional) Train ML model: python -m src.ml.train
3. (Optional) Initialize RAG: .\start-system.ps1 -WithRAG
4. Start the system: .\start-system.ps1

Quick Commands:
---------------
  .\start.ps1                     # Quick start (development)
  .\start-system.ps1 -Mode prod   # Production mode
  .\start-system.ps1 -Mode docker # Docker mode

Documentation:
--------------
  API Docs:  http://localhost:8000/docs
  README:    $ProjectRoot\README.md

"@ -ForegroundColor Cyan