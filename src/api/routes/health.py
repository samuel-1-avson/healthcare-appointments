"""
Health Check Routes
===================
Endpoints for system health monitoring.
"""

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session
from sqlalchemy import text
from ..database import get_db
from ..cache import RedisClient

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