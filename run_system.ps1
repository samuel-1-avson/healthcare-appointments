# Run System Script
# =================
# This script runs the Healthcare Appointments No-Show Prediction System.
# It checks for Docker, builds the image, and starts the service.

$ErrorActionPreference = "Stop"

function Write-Header {
    param([string]$Message)
    Write-Host "==================================================" -ForegroundColor Cyan
    Write-Host $Message -ForegroundColor Cyan
    Write-Host "==================================================" -ForegroundColor Cyan
}

function Write-Success {
    param([string]$Message)
    Write-Host "[SUCCESS] $Message" -ForegroundColor Green
}

function Write-ErrorMsg {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

function Check-Docker {
    Write-Host "Checking for Docker..."
    try {
        docker --version | Out-Null
        docker-compose --version | Out-Null
        Write-Success "Docker and Docker Compose are installed."
    }
    catch {
        Write-ErrorMsg "Docker or Docker Compose is not installed or not in PATH."
        Write-Host "Please install Docker Desktop to run this system."
        exit 1
    }
}

function Build-System {
    Write-Header "Building Docker Image"
    Write-Host "Building 'noshow-api' image..."
    
    # Build with docker-compose
    docker-compose build
    
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Build complete."
    }
    else {
        Write-ErrorMsg "Build failed."
        exit 1
    }
}

function Start-System {
    Write-Header "Starting System"
    Write-Host "Starting services..."
    
    # Run in detached mode
    docker-compose up -d
    
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Services started."
    }
    else {
        Write-ErrorMsg "Failed to start services."
        exit 1
    }
    
    # Wait for health check
    Write-Host "Waiting for API to be healthy..."
    $retries = 30
    $healthy = $false
    
    for ($i = 0; $i -lt $retries; $i++) {
        $status = docker inspect --format='{{json .State.Health.Status}}' noshow-api 2>$null
        if ($status -eq '"healthy"') {
            $healthy = $true
            break
        }
        Start-Sleep -Seconds 2
        Write-Host "." -NoNewline
    }
    Write-Host ""
    
    if ($healthy) {
        Write-Success "API is healthy and running!"
        Write-Header "System Information"
        Write-Host "API URL:      http://localhost:8000"
        Write-Host "Docs URL:     http://localhost:8000/docs"
        Write-Host "Health Check: http://localhost:8000/health"
        Write-Host ""
        Write-Host "To stop the system, run: docker-compose down"
    }
    else {
        Write-ErrorMsg "API failed to become healthy. Checking logs..."
        docker-compose logs api
        exit 1
    }
}

# Main Execution Flow
try {
    Write-Header "Healthcare No-Show Prediction System"
    
    Check-Docker
    Build-System
    Start-System
    
}
catch {
    Write-ErrorMsg "An unexpected error occurred: $_"
    exit 1
}
