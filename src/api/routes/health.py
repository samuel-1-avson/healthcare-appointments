"""
Health Check Routes
===================
Endpoints for monitoring API health and status.
"""

import platform
import sys
from datetime import datetime
from typing import Dict, Any

from fastapi import APIRouter, Depends

from ..config import get_settings, Settings
from ..schemas import HealthResponse, HealthStatus, ModelInfo
from ..predict import get_predictor, NoShowPredictor


router = APIRouter()


def get_system_info() -> Dict[str, Any]:
    """Get system information."""
    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or "Unknown"
    }


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Check API health status and model availability"
)
async def health_check(
    settings: Settings = Depends(get_settings),
    predictor: NoShowPredictor = Depends(get_predictor)
) -> HealthResponse:
    """
    Perform health check.
    
    Returns:
    - API status (healthy/degraded/unhealthy)
    - Model loading status
    - System information
    """
    # Determine health status
    if predictor.is_loaded:
        status = HealthStatus.HEALTHY
    else:
        status = HealthStatus.DEGRADED
    
    # Get model info if loaded
    model_info = None
    if predictor.is_loaded:
        info = predictor.get_model_info()
        model_info = ModelInfo(
            name=info.get('name', 'Unknown'),
            version=info.get('version', '1.0.0'),
            type=info.get('type', 'Unknown'),
            features=info.get('features', 0),
            trained_at=info.get('trained_at'),
            metrics=info.get('metrics')
        )
    
    return HealthResponse(
        status=status,
        timestamp=datetime.utcnow(),
        version=settings.api_version,
        model_loaded=predictor.is_loaded,
        model_info=model_info,
        system=get_system_info() if settings.debug else None
    )


@router.get(
    "/ready",
    summary="Readiness Check",
    description="Check if API is ready to serve requests"
)
async def readiness_check(
    predictor: NoShowPredictor = Depends(get_predictor)
) -> Dict[str, Any]:
    """
    Readiness probe for Kubernetes/container orchestration.
    
    Returns 200 if model is loaded and ready.
    """
    if predictor.is_loaded:
        return {
            "ready": True,
            "timestamp": datetime.utcnow().isoformat()
        }
    else:
        return {
            "ready": False,
            "reason": "Model not loaded",
            "timestamp": datetime.utcnow().isoformat()
        }


@router.get(
    "/live",
    summary="Liveness Check", 
    description="Check if API process is alive"
)
async def liveness_check() -> Dict[str, Any]:
    """
    Liveness probe for Kubernetes/container orchestration.
    
    Always returns 200 if the process is running.
    """
    return {
        "alive": True,
        "timestamp": datetime.utcnow().isoformat()
    }