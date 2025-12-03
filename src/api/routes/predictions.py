# src/api/routes/predictions.py
"""
Prediction Routes
=================
Endpoints for making no-show predictions.
"""

import logging
from typing import Optional, Union
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, Body
from fastapi.responses import JSONResponse

from ..config import get_settings, Settings
from ..schemas import (
    AppointmentFeatures,
    PredictionResponse,
    AsyncPredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    ErrorResponse
)
from ..predict import (
    get_predictor, 
    NoShowPredictor,
    PredictionError
)
from ..tasks import predict_noshow_task, batch_predict_task
from celery.result import AsyncResult


router = APIRouter()
logger = logging.getLogger(__name__)


@router.post(
    "",
    response_model=Union[PredictionResponse, AsyncPredictionResponse],
    summary="Predict No-Show",
    description="Predict whether a patient will miss their appointment",
    responses={
        200: {"description": "Successful prediction"},
        400: {"model": ErrorResponse, "description": "Invalid input"},
        500: {"model": ErrorResponse, "description": "Prediction error"},
        503: {"model": ErrorResponse, "description": "Model not available"}
    }
)
async def predict_single(
    appointment: AppointmentFeatures = Body(..., description="Appointment features"),
    threshold: Optional[float] = Query(
        None,
        ge=0.0,
        le=1.0,
        description="Custom classification threshold (default: 0.5)"
    ),
    include_explanation: bool = Query(
        False,
        description="Include feature importance explanation"
    ),
    predictor: NoShowPredictor = Depends(get_predictor),
    settings: Settings = Depends(get_settings),
    async_mode: bool = Query(False, description="Run prediction asynchronously")
) -> Union[PredictionResponse, AsyncPredictionResponse]:
    """
    Make a no-show prediction for a single appointment.
    
    ## Input Features
    
    **Required:**
    - `age`: Patient age (0-120)
    - `gender`: M, F, or O
    - `lead_days`: Days between scheduling and appointment
    
    **Optional (improve accuracy):**
    - `scholarship`: Welfare program enrollment (0/1)
    - `hypertension`, `diabetes`, `alcoholism`: Medical conditions (0/1)
    - `handicap`: Disability level (0-4)
    - `sms_received`: SMS reminder sent (0/1)
    - `patient_historical_noshow_rate`: Previous no-show rate (0-1)
    
    ## Response
    
    Returns prediction with:
    - Binary prediction (0=will show, 1=will no-show)
    - Probability of no-show
    - Risk tier (MINIMAL to CRITICAL)
    - Recommended intervention
    - Optional explanation of key factors
    
    If `async_mode=True`, returns `{"task_id": "...", "status": "processing"}`.
    """
    # Check if model is loaded (only for sync mode)
    if not async_mode and not predictor.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not available. Please try again later."
        )
    
    try:
        if async_mode:
            # Run asynchronously
            task = predict_noshow_task.delay(appointment.model_dump())
            return AsyncPredictionResponse(task_id=task.id, status="processing")
        else:
            # Run synchronously
            result = predictor.predict(
                appointment,
                threshold=threshold,
                include_explanation=include_explanation
            )
            return result
        
    except PredictionError as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post(
    "/batch",
    response_model=Union[BatchPredictionResponse, AsyncPredictionResponse],
    summary="Batch Predict No-Shows",
    description="Predict no-shows for multiple appointments at once",
    responses={
        200: {"description": "Successful batch prediction"},
        400: {"model": ErrorResponse, "description": "Invalid input"},
        500: {"model": ErrorResponse, "description": "Prediction error"}
    }
)
async def predict_batch(
    request: BatchPredictionRequest,
    predictor: NoShowPredictor = Depends(get_predictor),
    async_mode: bool = Query(False, description="Run prediction asynchronously")
) -> Union[BatchPredictionResponse, AsyncPredictionResponse]:
    """
    Make predictions for multiple appointments.
    
    Accepts up to 1000 appointments per request.
    If `async_mode=True`, returns `{"task_id": "...", "status": "processing"}`.
    """
    if not async_mode and not predictor.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not available"
        )
    
    try:
        if async_mode:
            # Run asynchronously
            task = batch_predict_task.delay([a.model_dump() for a in request.appointments])
            return AsyncPredictionResponse(task_id=task.id, status="processing")
        else:
            # Run synchronously
            result = predictor.predict_batch(
                request.appointments,
                threshold=request.threshold,
                include_explanations=request.include_explanations
            )
            
            logger.info(
                f"Batch prediction: {len(request.appointments)} appointments, "
                f"{result.summary['predicted_noshows']} predicted no-shows"
            )
            
            return result
        
    except PredictionError as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/quick",
    summary="Quick Prediction",
    description="Minimal input prediction with defaults"
)
async def predict_quick(
    age: int = Query(..., ge=0, le=120, description="Patient age"),
    gender: str = Query(..., description="Gender (M/F)"),
    lead_days: int = Query(..., ge=0, description="Days until appointment"),
    sms_received: int = Query(1, ge=0, le=1, description="SMS reminder sent"),
    predictor: NoShowPredictor = Depends(get_predictor)
) -> dict:
    """
    Quick prediction with minimal required inputs.
    
    Uses default values for optional features.
    Useful for simple integrations or testing.
    """
    if not predictor.is_loaded:
        raise HTTPException(status_code=503, detail="Model not available")
    
    # Create appointment with defaults
    appointment = AppointmentFeatures(
        age=age,
        gender=gender,
        lead_days=lead_days,
        sms_received=sms_received
    )
    
    result = predictor.predict(appointment)
    
    return {
        "probability": result.probability,
        "prediction": "No-Show" if result.prediction == 1 else "Will Attend",
        "risk_tier": result.risk.tier,
        "intervention": result.intervention.action
    }


@router.get(
    "/thresholds",
    summary="Get Threshold Information",
    description="Get information about classification thresholds and risk tiers"
)
async def get_thresholds(
    settings: Settings = Depends(get_settings)
) -> dict:
    """
    Get threshold and risk tier information.
    
    Useful for understanding how predictions are classified.
    """
    from ..config import RiskTierConfig
    
    return {
        "default_threshold": settings.default_threshold,
        "risk_tiers": RiskTierConfig.TIERS,
        "explanation": {
            "threshold": "Probability threshold for binary classification. "
                        "Predictions above this are classified as no-show.",
            "risk_tiers": "Risk categories based on probability. "
                         "Each tier has recommended interventions."
        }
    }


@router.get(
    "/tasks/{task_id}",
    summary="Get Task Status",
    description="Get the status and result of a background task"
)
async def get_task_status(task_id: str) -> dict:
    """
    Get task status.
    """
    task_result = AsyncResult(task_id)
    
    response = {
        "task_id": task_id,
        "status": task_result.status,
    }
    
    if task_result.successful():
        response["result"] = task_result.result
    elif task_result.failed():
        response["error"] = str(task_result.result)
        
    return response