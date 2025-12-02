# scripts/docker-start.ps1
# ============================================================
# Start Healthcare System with Docker
# ============================================================

param(
    [ValidateSet("dev", "prod", "build")]
    [string]$Mode = "dev",
    
    [switch]$Build,
    [switch]$Detach,
    [switch]$WithRedis,
    [switch]$WithNginx,
    [switch]$InitRAG,
    [switch]$Logs,
    [switch]$Stop,
    [switch]$Clean,
    [switch]$Help
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path $PSScriptRoot -Parent

# Colors
function Write-Color {
    param([string]$Text, [string]$Color = "White")
    Write-Host $Text -ForegroundColor $Color
}

# Banner
Write-Host ""
Write-Host @"
╔═══════════════════════════════════════════════════════════╗
║     Healthcare No-Show Prediction System (Docker)         ║
╚═══════════════════════════════════════════════════════════╝
"@ -ForegroundColor Cyan

# Help
if ($Help) {
    Write-Host @"
USAGE:
    .\docker-start.ps1 [OPTIONS]

OPTIONS:
    -Mode <dev|prod>    Start in development or production mode
    -Build              Force rebuild containers
    -Detach             Run in background (detached mode)
    -WithRedis          Include Redis cache service
    -WithNginx          Include Nginx reverse proxy
    -InitRAG            Initialize RAG vector store after start
    -Logs               Follow container logs
    -Stop               Stop all containers
    -Clean              Stop and remove all containers/volumes
    -Help               Show this help

EXAMPLES:
    .\docker-start.ps1                      # Dev mode, attached
    .\docker-start.ps1 -Mode prod -Detach   # Production, background
    .\docker-start.ps1 -Build               # Rebuild and start
    .\docker-start.ps1 -Stop                # Stop containers
    .\docker-start.ps1 -Clean               # Full cleanup
"@
    exit 0
}

# Change to project root
Set-Location $ProjectRoot

# Check Docker
try {
    docker --version | Out-Null
    docker compose version | Out-Null
    Write-Color "[OK] Docker is available" "Green"
}
catch {
    Write-Color "[ERROR] Docker not found. Please install Docker Desktop." "Red"
    exit 1
}

# Check .env file
if (-not (Test-Path ".env")) {
    if (Test-Path ".env.example") {
        Write-Color "[INFO] Creating .env from .env.example" "Yellow"
        Copy-Item ".env.example" ".env"
        Write-Color "[WARN] Please edit .env and add your OPENAI_API_KEY" "Yellow"
    }
    else {
        Write-Color "[WARN] No .env file found. Creating minimal .env" "Yellow"
        @"
# API Configuration
API_PORT=8000
DEBUG=false
LOG_LEVEL=INFO
WORKERS=2

# LLM Configuration (Required for LLM features)
OPENAI_API_KEY=your-openai-key-here

# Optional
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
"@ | Out-File ".env" -Encoding utf8
    }
}

# Stop containers
if ($Stop) {
    Write-Color "Stopping containers..." "Yellow"
    docker compose down
    Write-Color "[OK] Containers stopped" "Green"
    exit 0
}

# Clean everything
if ($Clean) {
    Write-Color "Cleaning up containers, volumes, and images..." "Yellow"
    docker compose down -v --remove-orphans
    docker image prune -f
    Write-Color "[OK] Cleanup complete" "Green"
    exit 0
}

# Show logs
if ($Logs) {
    Write-Color "Following container logs (Ctrl+C to stop)..." "Cyan"
    docker compose logs -f
    exit 0
}

# Build compose command
$composeFiles = @("-f", "docker-compose.yaml")

if ($Mode -eq "dev") {
    if (Test-Path "docker-compose.dev.yaml") {
        $composeFiles += @("-f", "docker-compose.dev.yaml")
    }
    Write-Color "[INFO] Development mode" "Cyan"
}
else {
    Write-Color "[INFO] Production mode" "Cyan"
}

# Profiles
$profiles = @()
if ($WithRedis) {
    $profiles += "with-redis"
    Write-Color "[INFO] Including Redis" "Cyan"
}
if ($WithNginx) {
    $profiles += "with-nginx"
    Write-Color "[INFO] Including Nginx" "Cyan"
}

# Build command
$upCommand = @("up")

if ($Build) {
    $upCommand += "--build"
    Write-Color "[INFO] Building images..." "Yellow"
}

if ($Detach) {
    $upCommand += "-d"
}

# Add profiles
foreach ($profile in $profiles) {
    $upCommand += "--profile"
    $upCommand += $profile
}

# Execute docker compose
Write-Host ""
Write-Color "Starting containers..." "Cyan"
$fullCommand = $composeFiles + $upCommand

& docker compose @fullCommand

# Wait for API to be ready if detached
if ($Detach) {
    Write-Color "Waiting for API to be ready..." "Yellow"
    
    $maxAttempts = 30
    $attempt = 0
    $ready = $false
    
    while (-not $ready -and $attempt -lt $maxAttempts) {
        Start-Sleep -Seconds 2
        $attempt++
        
        try {
            $response = Invoke-RestMethod -Uri "http://localhost:8000/health" -Method Get -TimeoutSec 5
            if ($response.status -eq "healthy") {
                $ready = $true
            }
        }
        catch {
            Write-Host "." -NoNewline
        }
    }
    
    Write-Host ""
    
    if ($ready) {
        Write-Color "[OK] API is ready!" "Green"
        Write-Host ""
        Write-Color "  API:     http://localhost:8000" "Green"
        Write-Color "  Docs:    http://localhost:8000/docs" "Green"
        Write-Color "  Health:  http://localhost:8000/health" "Green"
    }
    else {
        Write-Color "[WARN] API may not be ready yet. Check logs with: docker compose logs -f" "Yellow"
    }
    
    # Initialize RAG if requested
    if ($InitRAG) {
        Write-Host ""
        Write-Color "Initializing RAG vector store..." "Cyan"
        docker compose exec api python -c @"
from src.llm.rag import load_policy_documents, VectorStoreManager
docs = load_policy_documents('/app/data/documents')
print(f'Loaded {len(docs)} documents')
if docs:
    manager = VectorStoreManager()
    manager.create_from_documents(docs)
    manager.save('default')
    print('Vector store initialized!')
else:
    print('No documents found. Add .md files to data/documents/')
"@
    }
}

Write-Host ""
Write-Color "Done!" "Green"