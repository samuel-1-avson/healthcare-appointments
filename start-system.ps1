# start-system.ps1
# ============================================================
# Healthcare No-Show Prediction System - Startup Script
# ============================================================
# Usage:
#   .\start-system.ps1                    # Start in development mode
#   .\start-system.ps1 -Mode prod         # Start in production mode
#   .\start-system.ps1 -Mode docker       # Start with Docker
#   .\start-system.ps1 -WithRAG           # Initialize RAG index on startup
#   .\start-system.ps1 -Help              # Show help
# ============================================================

param(
    [ValidateSet("dev", "prod", "docker")]
    [string]$Mode = "dev",
    
    [switch]$WithRAG,
    [switch]$SkipEnvCheck,
    [switch]$Rebuild,
    [switch]$Help,
    
    [string]$ApiHost = "0.0.0.0",
    [int]$Port = 8000,
    [int]$Workers = 4
)

# ============================================================
# Configuration
# ============================================================

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
$VenvPath = Join-Path $ProjectRoot "venv"
$EnvFile = Join-Path $ProjectRoot ".env"
$RequirementsFile = Join-Path $ProjectRoot "requirements.txt"
$RequirementsLLMFile = Join-Path $ProjectRoot "requirements-llm.txt"
$RequirementsApiFile = Join-Path $ProjectRoot "requirements-api.txt"

# Colors for output
function Write-ColorOutput {
    param(
        [string]$Message,
        [string]$Color = "White"
    )
    Write-Host $Message -ForegroundColor $Color
}

function Write-Header {
    param([string]$Title)
    Write-Host ""
    Write-Host "=" * 60 -ForegroundColor Cyan
    Write-Host "  $Title" -ForegroundColor Cyan
    Write-Host "=" * 60 -ForegroundColor Cyan
    Write-Host ""
}

function Write-Success { param([string]$Message) Write-Host "[✓] $Message" -ForegroundColor Green }
function Write-Error { param([string]$Message) Write-Host "[✗] $Message" -ForegroundColor Red }
function Write-Warning { param([string]$Message) Write-Host "[!] $Message" -ForegroundColor Yellow }
function Write-Info { param([string]$Message) Write-Host "[i] $Message" -ForegroundColor Cyan }

# ============================================================
# Help
# ============================================================

if ($Help) {
    Write-Header "Healthcare No-Show Prediction System"
    Write-Host @"
USAGE:
    .\start-system.ps1 [OPTIONS]

OPTIONS:
    -Mode <dev|prod|docker>    Startup mode (default: dev)
    -WithRAG                   Initialize RAG vector store on startup
    -SkipEnvCheck              Skip environment validation
    -Rebuild                   Rebuild Docker containers (docker mode only)
    -Host <address>            API host (default: 0.0.0.0)
    -Port <number>             API port (default: 8000)
    -Workers <number>          Number of workers for production (default: 4)
    -Help                      Show this help message

EXAMPLES:
    .\start-system.ps1                      # Development server with auto-reload
    .\start-system.ps1 -Mode prod           # Production server with Gunicorn
    .\start-system.ps1 -Mode docker         # Run with Docker Compose
    .\start-system.ps1 -WithRAG -Port 8080  # Dev server with RAG on port 8080

ENDPOINTS:
    API Docs:        http://localhost:$Port/docs
    Health Check:    http://localhost:$Port/health
    Predictions:     http://localhost:$Port/api/v1/predict
    LLM Chat:        http://localhost:$Port/api/v1/llm/chat
    RAG Q&A:         http://localhost:$Port/api/v1/rag/ask
"@
    exit 0
}

# ============================================================
# Banner
# ============================================================

Write-Host ""
Write-Host @"
╔═══════════════════════════════════════════════════════════════╗
║     _   _            _ _   _                                  ║
║    | | | | ___  __ _| | |_| |__   ___ __ _ _ __ ___          ║
║    | |_| |/ _ \/ _` | | __| '_ \ / __/ _` | '__/ _ \         ║
║    |  _  |  __/ (_| | | |_| | | | (_| (_| | | |  __/         ║
║    | | | | | (_) |__| | | | | | | |_) | | | | | | | | |      ║
║    \_| |_/\___\__,__|_| |_| |_| |_| .__/|_| |_| |_| |_|      ║
║                                   |_|                        ║
╚═══════════════════════════════════════════════════════════════╝
"@ -ForegroundColor Cyan
Write-Host ""

# ============================================================
# Prerequisites
# ============================================================

function Test-Prerequisites {
    Write-Header "Checking Prerequisites"
    
    $allPassed = $true
    
    # Check Python
    try {
        $pythonVersion = python --version 2>&1
        if ($pythonVersion -match "Python (\d+)\.(\d+)") {
            $major = [int]$Matches[1]
            $minor = [int]$Matches[2]
            if ($major -ge 3 -and $minor -ge 9) {
                Write-Success "Python $major.$minor detected"
            }
            else {
                Write-Warning "Python 3.9+ recommended (found $major.$minor)"
            }
        }
    }
    catch {
        Write-Error "Python not found. Please install Python 3.9+"
        $allPassed = $false
    }
    
    # Check pip
    try {
        $pipVersion = pip --version 2>&1
        Write-Success "pip available"
    }
    catch {
        Write-Error "pip not found"
        $allPassed = $false
    }
    
    # Check Docker (for docker mode)
    if ($Mode -eq "docker") {
        try {
            $dockerVersion = docker --version 2>&1
            Write-Success "Docker available: $dockerVersion"
            
            $composeVersion = docker compose version 2>&1
            Write-Success "Docker Compose available"
        }
        catch {
            Write-Error "Docker not found. Required for docker mode."
            $allPassed = $false
        }
    }
    
    # Check .env file
    if (Test-Path $EnvFile) {
        Write-Success ".env file found"
        
        # Check for required API keys
        $envContent = Get-Content $EnvFile -Raw
        if ($envContent -match "OPENAI_API_KEY=sk-") {
            Write-Success "OpenAI API key configured"
        }
        else {
            Write-Warning "OpenAI API key not set (LLM features may not work)"
        }
    }
    else {
        Write-Warning ".env file not found"
        Write-Info "Creating .env from .env.example..."
        
        $envExample = Join-Path $ProjectRoot ".env.example"
        if (Test-Path $envExample) {
            Copy-Item $envExample $EnvFile
            Write-Success "Created .env file - please configure your API keys"
        }
        else {
            Write-Warning "No .env.example found. Creating minimal .env..."
            @"
# Healthcare No-Show Prediction System Configuration

# API Keys (Required for LLM features)
OPENAI_API_KEY=your-openai-key-here
# ANTHROPIC_API_KEY=your-anthropic-key-here

# API Configuration
NOSHOW_HOST=0.0.0.0
NOSHOW_PORT=8000
NOSHOW_DEBUG=true
NOSHOW_LOG_LEVEL=INFO

# LLM Configuration
NOSHOW_LLM_DEFAULT_MODEL=gpt-4o-mini
"@ | Out-File -FilePath $EnvFile -Encoding utf8
            Write-Warning "Please edit .env and add your API keys"
        }
    }
    
    # Check model files
    $modelPath = Join-Path $ProjectRoot "models\production\model.joblib"
    if (Test-Path $modelPath) {
        Write-Success "ML model found"
    }
    else {
        Write-Warning "ML model not found at $modelPath"
        Write-Info "Predictions may fail until model is trained"
    }
    
    # Check documents for RAG
    $docsPath = Join-Path $ProjectRoot "data\documents"
    if (Test-Path $docsPath) {
        $docCount = (Get-ChildItem $docsPath -Filter "*.md").Count
        Write-Success "Documents directory found ($docCount markdown files)"
    }
    else {
        Write-Warning "Documents directory not found - RAG features may not work"
    }
    
    return $allPassed
}

# ============================================================
# Virtual Environment Setup
# ============================================================

function Initialize-VirtualEnv {
    Write-Header "Setting Up Virtual Environment"
    
    if (-not (Test-Path $VenvPath)) {
        Write-Info "Creating virtual environment..."
        python -m venv $VenvPath
        Write-Success "Virtual environment created"
    }
    else {
        Write-Success "Virtual environment exists"
    }
    
    # Activate virtual environment
    $activateScript = Join-Path $VenvPath "Scripts\Activate.ps1"
    if (Test-Path $activateScript) {
        Write-Info "Activating virtual environment..."
        & $activateScript
        Write-Success "Virtual environment activated"
    }
    else {
        Write-Error "Could not find activation script"
        exit 1
    }
    
    # Install/update dependencies
    Write-Info "Checking dependencies..."
    
    if (Test-Path $RequirementsFile) {
        pip install -q -r $RequirementsFile
        Write-Success "Core dependencies installed"
    }
    
    if (Test-Path $RequirementsLLMFile) {
        pip install -q -r $RequirementsLLMFile
        Write-Success "LLM dependencies installed"
    }

    if (Test-Path $RequirementsApiFile) {
        pip install -q -r $RequirementsApiFile
        Write-Success "API dependencies installed"
    }
}

# ============================================================
# Load Environment Variables
# ============================================================

function Import-EnvFile {
    Write-Info "Loading environment variables..."
    
    if (Test-Path $EnvFile) {
        Get-Content $EnvFile | ForEach-Object {
            if ($_ -match "^\s*([^#][^=]+)=(.*)$") {
                $name = $Matches[1].Trim()
                $value = $Matches[2].Trim()
                # Remove quotes if present
                $value = $value -replace '^["'']|["'']$', ''
                [Environment]::SetEnvironmentVariable($name, $value, "Process")
            }
        }
        Write-Success "Environment variables loaded"
    }
}

# ============================================================
# Initialize RAG
# ============================================================

function Initialize-RAG {
    Write-Header "Initializing RAG Vector Store"
    
    $docsPath = Join-Path $ProjectRoot "data\documents"
    $vectorStorePath = Join-Path $ProjectRoot "data\vector_store"
    
    if (-not (Test-Path $docsPath)) {
        Write-Warning "No documents directory found. Skipping RAG initialization."
        return
    }
    
    Write-Info "Loading and indexing documents..."
    
    # Run Python script to initialize RAG
    $initScript = @"
import sys
sys.path.insert(0, '$($ProjectRoot -replace '\\', '/')')

from src.llm.rag import load_policy_documents, VectorStoreManager

print("Loading documents...")
docs = load_policy_documents("$($docsPath -replace '\\', '/')")
print(f"Loaded {len(docs)} documents")

print("Creating vector store...")
manager = VectorStoreManager()
manager.create_from_documents(docs, chunk_size=1000, chunk_overlap=200)
manager.save("default")
print("Vector store created and saved!")
"@
    
    $initScript | python
    
    if ($LASTEXITCODE -eq 0) {
        Write-Success "RAG vector store initialized"
    }
    else {
        Write-Error "RAG initialization failed"
    }
}

# ============================================================
# Start Servers
# ============================================================

function Start-DevServer {
    Write-Header "Starting Development Server"
    
    Write-Info "Server will start with auto-reload enabled"
    Write-Host ""
    Write-Host "  API Docs:     http://localhost:$Port/docs" -ForegroundColor Green
    Write-Host "  Health:       http://localhost:$Port/health" -ForegroundColor Green
    Write-Host "  Predictions:  http://localhost:$Port/api/v1/predict" -ForegroundColor Green
    Write-Host "  LLM Chat:     http://localhost:$Port/api/v1/llm/chat" -ForegroundColor Green
    Write-Host "  RAG Q&A:      http://localhost:$Port/api/v1/rag/ask" -ForegroundColor Green
    Write-Host ""
    Write-Info "Press Ctrl+C to stop the server"
    Write-Host ""
    
    # Set environment variables
    $env:NOSHOW_HOST = $ApiHost
    $env:NOSHOW_PORT = $Port
    $env:NOSHOW_DEBUG = "true"
    
    # Start uvicorn with reload
    python -m uvicorn src.api.main:app --host $ApiHost --port $Port --reload
}

function Start-ProdServer {
    Write-Header "Starting Production Server"
    
    Write-Info "Starting with $Workers workers..."
    Write-Host ""
    Write-Host "  API:     http://${ApiHost}:$Port" -ForegroundColor Green
    Write-Host "  Docs:    http://${ApiHost}:$Port/docs" -ForegroundColor Green
    Write-Host "  Health:  http://${ApiHost}:$Port/health" -ForegroundColor Green
    Write-Host ""
    Write-Info "Press Ctrl+C to stop the server"
    Write-Host ""
    
    # Set environment variables
    $env:NOSHOW_HOST = $ApiHost
    $env:NOSHOW_PORT = $Port
    $env:NOSHOW_DEBUG = "false"
    
    # Check if gunicorn is available (Linux/Mac) or use uvicorn with workers
    try {
        gunicorn --version 2>&1 | Out-Null
        gunicorn src.api.main:app `
            --bind ${ApiHost}:$Port `
            --workers $Workers `
            --worker-class uvicorn.workers.UvicornWorker `
            --access-logfile - `
            --error-logfile -
    }
    catch {
        Write-Warning "Gunicorn not available (Windows). Using uvicorn with workers..."
        python -m uvicorn src.api.main:app `
            --host $ApiHost `
            --port $Port `
            --workers $Workers
    }
}

function Start-DockerServer {
    Write-Header "Starting Docker Containers"
    
    $composeFiles = @("-f", "docker-compose.yaml")
    $devOverride = Join-Path $ProjectRoot "docker-compose.dev.yaml"
    
    if (Test-Path $devOverride) {
        $composeFiles += @("-f", "docker-compose.dev.yaml")
        Write-Info "Using development override"
    }
    
    if (-not (Test-Path (Join-Path $ProjectRoot "docker-compose.yaml"))) {
        Write-Error "docker-compose.yaml not found"
        exit 1
    }
    
    # Stop existing containers
    Write-Info "Stopping existing containers..."
    docker compose down 2>&1 | Out-Null
    
    # Build if requested
    if ($Rebuild) {
        Write-Info "Rebuilding containers..."
        docker compose build --no-cache
    }
    
    Write-Info "Starting containers..."
    Write-Host ""
    Write-Host "  API:     http://localhost:$Port" -ForegroundColor Green
    Write-Host "  Docs:    http://localhost:$Port/docs" -ForegroundColor Green
    Write-Host ""
    Write-Info "Press Ctrl+C to stop containers"
    Write-Host ""
    
    # Set port via environment
    $env:API_PORT = $Port
    
    # Start with docker compose
    $cmdArgs = @("compose") + $composeFiles + @("up")
    & docker $cmdArgs
}

# ============================================================
# Main Execution
# ============================================================

try {
    # Change to project root
    Set-Location $ProjectRoot
    
    # Run prerequisite checks
    if (-not $SkipEnvCheck) {
        $prereqsPassed = Test-Prerequisites
        if (-not $prereqsPassed -and $Mode -ne "docker") {
            Write-Host ""
            Write-Error "Prerequisites check failed. Use -SkipEnvCheck to bypass."
            exit 1
        }
    }
    
    # Mode-specific setup and start
    switch ($Mode) {
        "dev" {
            Initialize-VirtualEnv
            Import-EnvFile
            
            if ($WithRAG) {
                Initialize-RAG
            }
            
            Start-DevServer
        }
        
        "prod" {
            Initialize-VirtualEnv
            Import-EnvFile
            
            if ($WithRAG) {
                Initialize-RAG
            }
            
            Start-ProdServer
        }
        
        "docker" {
            Import-EnvFile
            Start-DockerServer
        }
    }
    
}
catch {
    Write-Host ""
    Write-Error "An error occurred: $_"
    Write-Host $_.ScriptStackTrace -ForegroundColor Red
    exit 1
}