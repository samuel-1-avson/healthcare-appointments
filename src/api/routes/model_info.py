"""
Model Information Routes
========================
Endpoints for model information and metadata.
"""

from typing import Dict, Any, List
from datetime import datetime
import logging
import pandas as pd
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException

from ..config import get_settings, Settings
from ..schemas import ModelInfo
from ..predict import get_predictor, NoShowPredictor


router = APIRouter()
logger = logging.getLogger(__name__)


@router.get(
    "",
    response_model=Dict[str, Any],
    summary="Get Model Information",
    description="Get information about the loaded ML model"
)
async def get_model_info(
    predictor: NoShowPredictor = Depends(get_predictor)
) -> Dict[str, Any]:
    """
    Get detailed model information.
    
    Returns:
    - Model name and version
    - Model type (algorithm)
    - Number of features
    - Training metrics (if available)
    """
    if not predictor.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )
    
    return predictor.get_model_info()


@router.get(
    "/features",
    summary="Get Feature Information",
    description="Get list of features used by the model"
)
async def get_features(
    predictor: NoShowPredictor = Depends(get_predictor)
) -> Dict[str, Any]:
    """
    Get feature information.
    
    Returns list of features expected by the model
    and their descriptions.
    """
    if not predictor.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Feature descriptions
    feature_descriptions = {
        "age": "Patient age in years (0-120)",
        "gender": "Patient gender (M/F/O)",
        "scholarship": "Welfare program enrollment (0=No, 1=Yes)",
        "hypertension": "Has hypertension (0=No, 1=Yes)",
        "diabetes": "Has diabetes (0=No, 1=Yes)",
        "alcoholism": "Has alcoholism (0=No, 1=Yes)",
        "handicap": "Disability level (0-4)",
        "sms_received": "SMS reminder sent (0=No, 1=Yes)",
        "lead_days": "Days between scheduling and appointment",
        "neighbourhood": "Patient neighbourhood",
        "appointment_weekday": "Day of week for appointment",
        "patient_historical_noshow_rate": "Patient's historical no-show rate",
        "is_first_appointment": "First appointment for this patient (0/1)"
    }
    
    return {
        "feature_count": len(predictor.feature_names) if predictor.feature_names else "Unknown",
        "features": predictor.feature_names,
        "descriptions": feature_descriptions,
        "required": ["age", "gender", "lead_days"],
        "optional": list(set(feature_descriptions.keys()) - {"age", "gender", "lead_days"})
    }


@router.get(
    "/metrics",
    summary="Get Model Metrics",
    description="Get performance metrics from model training"
)
async def get_metrics(
    predictor: NoShowPredictor = Depends(get_predictor)
) -> Dict[str, Any]:
    """
    Get model performance metrics.
    
    Returns training/validation metrics if available.
    """
    if not predictor.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    metrics = predictor.metadata.get('metrics', {})
    
    if not metrics:
        return {
            "message": "No metrics available",
            "note": "Metrics are stored during model training"
        }
    
    return {
        "metrics": metrics,
        "description": {
            "roc_auc": "Area under ROC curve (higher is better)",
            "f1": "F1 score (balance of precision and recall)",
            "precision": "Precision (correct positive predictions / all positive predictions)",
            "recall": "Recall (correct positive predictions / all actual positives)",
            "accuracy": "Overall accuracy"
        }
    }


@router.post(
    "/reload",
    summary="Reload Model",
    description="Reload the model from disk (admin only)"
)
async def reload_model(
    predictor: NoShowPredictor = Depends(get_predictor)
) -> Dict[str, str]:
    """
    Reload the model from disk.
    
    Useful after deploying a new model version.
    
    **Note:** In production, this should be protected by authentication.
    """
    try:
        from ..predict import reload_predictor
        reload_predictor()
        
        return {
            "status": "success",
            "message": "Model reloaded successfully",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to reload model: {str(e)}"
        )


@router.get(
    "/history",
    summary="Get Historical Data",
    description="Get aggregated historical data for dashboard charts"
)
async def get_history(
    settings: Settings = Depends(get_settings)
) -> Dict[str, Any]:
    """
    Get historical aggregation data.
    
    Reads from generated summary CSV files if available.
    """
    
    # Define paths (assuming standard structure)
    # In Docker, these should be mounted or available
    # We try multiple locations to be robust
    base_paths = [
        Path("data/dashboard"),
        Path("/app/data/dashboard"),
        Path("../data/dashboard"),
        Path("c:/Users/samue/Desktop/NSP/healthcare-appointments/data/dashboard") # Local fallback
    ]
    
    dashboard_dir = None
    for p in base_paths:
        if p.exists():
            dashboard_dir = p
            break
            
    if not dashboard_dir:
        return {
            "status": "no_data",
            "message": "Historical data not found. Please run data generation script."
        }
        
    try:
        # Load Daily Summary
        daily_path = dashboard_dir / "summary_daily.csv"
        daily_data = []
        if daily_path.exists():
            df_daily = pd.read_csv(daily_path)
            # Sort by date
            df_daily = df_daily.sort_values('date')
            # Take last 30 days for chart
            df_daily = df_daily.tail(30)
            daily_data = df_daily.to_dict(orient='records')
            
        # Load Risk Tier Summary
        tier_path = dashboard_dir / "summary_risk_tier.csv"
        tier_data = []
        if tier_path.exists():
            df_tier = pd.read_csv(tier_path)
            tier_data = df_tier.to_dict(orient='records')

        # Load Recent Alerts (Priority Interventions)
        # Robustly find project root from this file's location
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent.parent
        
        alerts_path = project_root / "outputs" / "sql" / "sql_priority_interventions.csv"
        
        alerts_data = []
        if alerts_path.exists():
            df_alerts = pd.read_csv(alerts_path)
            # Take top 5
            df_alerts = df_alerts.head(5)
            alerts_data = df_alerts.to_dict(orient='records')
        else:
            # Fallback
            alt_path = dashboard_dir.parent.parent / "outputs" / "sql" / "sql_priority_interventions.csv"
            if alt_path.exists():
                df_alerts = pd.read_csv(alt_path)
                alerts_data = df_alerts.head(5).to_dict(orient='records')

        # Load Patient Segments
        segments_path = dashboard_dir / "summary_patient_segments.csv"
        segments_data = []
        if segments_path.exists():
            df_segments = pd.read_csv(segments_path)
            segments_data = df_segments.to_dict(orient='records')

        # Load Behavior Evolution
        behavior_path = dashboard_dir / "summary_behavior.csv"
        behavior_data = []
        if behavior_path.exists():
            df_behavior = pd.read_csv(behavior_path)
            behavior_data = df_behavior.to_dict(orient='records')

        return {
            "status": "success",
            "daily_trends": daily_data,
            "risk_distribution": tier_data,
            "recent_alerts": alerts_data,
            "patient_segments": segments_data,
            "behavior_evolution": behavior_data
        }
        
    except Exception as e:
        logger.error(f"Failed to load history: {e}")
        raise HTTPException(status_code=500, detail=str(e))
