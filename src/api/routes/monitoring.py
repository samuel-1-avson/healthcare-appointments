# src/api/routes/monitoring.py
"""
Monitoring API Routes
====================
Endpoints for model monitoring and drift detection.
"""

import logging
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import PlainTextResponse
import pandas as pd

from ..monitoring import get_drift_monitor
from ..schemas import AppointmentFeatures
from prometheus_client import Gauge, generate_latest, CONTENT_TYPE_LATEST

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/monitoring", tags=["monitoring"])

# Prometheus metrics
drift_score_gauge = Gauge('model_drift_score', 'Dataset drift score from Evidently')
drift_detected_gauge = Gauge('model_drift_detected', 'Whether drift is detected (0 or 1)')
drifted_features_gauge = Gauge('model_drifted_features_count', 'Number of drifted features')


@router.get("/drift",
response_model=Dict[str, Any],
summary="Get data drift metrics")
async def get_drift_metrics() -> Dict[str, Any]:
    """
    Get current data drift metrics.
    
    Returns cached drift metrics if available, otherwise returns
    a message indicating monitoring is pending.
    """
    monitor = get_drift_monitor()
    
    # Try to get cached metrics
    metrics = monitor.get_cached_metrics()
    
    if metrics is None:
        return {
            "status": "no_data",
            "message": "No drift analysis available yet. Send prediction requests to trigger monitoring.",
            "drift_detected": False,
        }
    
    # Update Prometheus metrics
    if metrics.get("dataset_drift_score") is not None:
        drift_score_gauge.set(metrics["dataset_drift_score"])
        drift_detected_gauge.set(1 if metrics.get("drift_detected") else 0)
        drifted_features_gauge.set(metrics.get("num_drifted_features", 0))
    
    return metrics


@router.post("/reference",
summary="Update reference dataset")
async def update_reference_dataset(
    data: list[AppointmentFeatures] = Body(..., description="Reference dataset")
) -> Dict[str, str]:
    """
    Update the reference dataset for drift detection.
    
    The reference dataset should represent "normal" or "expected" data
    that incoming predictions will be compared against.
    """
    try:
        monitor = get_drift_monitor()
        
        # Convert Pydantic models to DataFrame
        records = [item.model_dump() for item in data]
        df = pd.DataFrame(records)
        
        # Set reference data
        monitor.set_reference_data(df)
        
        logger.info(f"Reference dataset updated with {len(df)} records")
        
        return {
            "status": "success",
            "message": f"Reference dataset updated with {len(df)} records",
            "num_records": len(df),
        }
        
    except Exception as e:
        logger.error(f"Failed to update reference dataset: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics",
response_class=PlainTextResponse,
summary="Prometheus metrics endpoint")
async def prometheus_metrics():
    """
    Expose Prometheus metrics for drift monitoring.
    
    This endpoint can be scraped by Prometheus to collect drift metrics.
    """
    # Update metrics from cache
    monitor = get_drift_monitor()
    metrics = monitor.get_cached_metrics()
    
    if metrics and metrics.get("dataset_drift_score") is not None:
        drift_score_gauge.set(metrics["dataset_drift_score"])
        drift_detected_gauge.set(1 if metrics.get("drift_detected") else 0)
        drifted_features_gauge.set(metrics.get("num_drifted_features", 0))
    
    # Generate Prometheus format
    return generate_latest()


@router.post("/analyze",
summary="Trigger drift analysis")
async def analyze_drift(
    data: list[AppointmentFeatures] = Body(..., description="Current data to analyze")
) -> Dict[str, Any]:
    """
    Manually trigger drift analysis on provided data.
    
    Compares the provided data against the reference dataset.
    """
    try:
        monitor = get_drift_monitor()
        
        # Convert to DataFrame
        records = [item.model_dump() for item in data]
        current_df = pd.DataFrame(records)
        
        # Calculate drift
        result = monitor.calculate_drift(current_df, cache_result=True)
        
        return result
        
    except Exception as e:
        logger.error(f"Drift analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
