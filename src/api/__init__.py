"""
Healthcare No-Show Prediction API
=================================

This package provides a FastAPI-based REST API for predicting
appointment no-shows using trained ML models.

Modules:
--------
- config: API configuration and settings
- schemas: Pydantic models for request/response validation
- predict: Prediction logic and model interface
- routes: API endpoint definitions
- main: FastAPI application factory

Usage:
------
    # Start the server
    python serve_model.py
    
    # Or with uvicorn directly
    uvicorn src.api.main:app --reload
    
    # Make predictions
    curl -X POST "http://localhost:8000/api/v1/predict" \
         -H "Content-Type: application/json" \
         -d '{"age": 35, "gender": "F", "lead_days": 7, ...}'
"""

from .config import get_settings, Settings
from .main import create_app, app

__all__ = [
    'create_app',
    'app',
    'get_settings',
    'Settings',
]

__version__ = '1.0.0'