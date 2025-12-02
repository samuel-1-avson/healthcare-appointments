"""
Model Information Routes
========================
Endpoints for model information and metadata.
"""

from typing import Dict, Any, List
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException

from ..config import get_settings, Settings
from ..schemas import ModelInfo
from ..predict import get_predictor, NoShowPredictor


router = APIRouter()


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