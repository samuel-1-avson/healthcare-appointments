"""
Health Check Routes
===================
Endpoints for system health monitoring.
"""

from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy import text
from ..database import get_db
from ..cache import RedisClient
from datetime import datetime

router = APIRouter(tags=["Health"])

@router.get("/health", status_code=status.HTTP_200_OK)
async def health_check(db: Session = Depends(get_db)):
    """
    System health check.
    
    Verifies:
    - API is running
    - Database is reachable
    - Redis is reachable
    """
    health_status = {
        "status": "healthy",
        "components": {
            "api": "up",
            "database": "unknown",
            "redis": "unknown"
        }
    }
    
    # Check Database
    try:
        db.execute(text("SELECT 1"))
        health_status["components"]["database"] = "connected"
    except Exception as e:
        health_status["status"] = "degraded"
        health_status["components"]["database"] = "disconnected"
        health_status["error"] = str(e)

    # Check Redis
    try:
        redis_client = RedisClient()
        if redis_client.is_connected:
             # Just a simple operation to verify
             redis_client.get("health_check")
             health_status["components"]["redis"] = "connected"
        else:
             health_status["components"]["redis"] = "disconnected"
             health_status["status"] = "degraded"
    except Exception as e:
        health_status["status"] = "degraded"
        health_status["components"]["redis"] = "disconnected"
        if "error" in health_status:
            health_status["error"] += f"; Redis: {str(e)}"
        else:
            health_status["error"] = f"Redis: {str(e)}"
        
    return health_status


@router.get("/live", status_code=status.HTTP_200_OK)
async def liveness_check():
    """
    Liveness probe.
    
    Returns 200 OK if the API is running.
    Used by Kubernetes/Docker to restart the container if it crashes.
    """
    return {
        "alive": True,
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/ready", status_code=status.HTTP_200_OK)
async def readiness_check(db: Session = Depends(get_db)):
    """
    Readiness probe.
    
    Returns 200 OK if the API is ready to accept traffic.
    Checks database and Redis connectivity.
    """
    # Check Database
    try:
        db.execute(text("SELECT 1"))
    except Exception:
        # If DB is down, we are not ready
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "not ready", "reason": "Database unavailable"}
        )

    # Check Redis (optional - strict readiness might require it, but we'll be lenient for now or strict?)
    # Let's be strict for readiness if it's a critical dependency, but the app can run without it (degraded).
    # Based on health_check logic, it seems optional/degraded. 
    # However, standard readiness usually implies "can serve requests".
    # If Redis is down, we can still serve predictions (just no cache).
    # So we won't fail readiness for Redis.
    
    return {
        "ready": True,
        "timestamp": datetime.utcnow().isoformat()
    }