"""
Test Configuration and Fixtures
================================
Shared fixtures and configuration for API tests.
"""

import sys
import json
from pathlib import Path
from typing import Generator, Dict, Any
from unittest.mock import MagicMock, patch
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fastapi.testclient import TestClient
import numpy as np
from src.api.main import app
from src.api.schemas import (
    PredictionResponse, 
    BatchPredictionResponse, 
    RiskAssessment, 
    InterventionRecommendation,
    RiskTier
)
from src.api.predict import get_predictor


# ==================== Fixtures ====================

@pytest.fixture(scope="session")
def client():
    """Create a test client."""
    with TestClient(app) as c:
        yield c

@pytest.fixture(scope="session")
def mock_model():
    """Create a mock ML model for testing."""
    model = MagicMock()
    
    # Mock predict_proba to return reasonable probabilities
    def mock_predict_proba(X):
        n_samples = X.shape[0] if hasattr(X, 'shape') else len(X)
        # Return random but reproducible probabilities
        np.random.seed(42)
        probs = np.random.rand(n_samples)
        return np.column_stack([1 - probs, probs])
    
    model.predict_proba = mock_predict_proba
    model.predict = lambda X: (mock_predict_proba(X)[:, 1] >= 0.5).astype(int)
    model.feature_importances_ = np.random.rand(20)
    
    return model


@pytest.fixture(scope="session")
def mock_preprocessor():
    """Create a mock preprocessor for testing."""
    preprocessor = MagicMock()
    
    # Mock transform to return numpy array
    def mock_transform(X):
        if hasattr(X, 'shape'):
            return np.random.rand(X.shape[0], 20)
        return np.random.rand(1, 20)
    
    return preprocessor


@pytest.fixture(autouse=True)
def app_overrides(mock_model, mock_preprocessor):
    """Override app dependencies."""
    # Mock the predictor
    mock_predictor = MagicMock()
    mock_predictor.model = mock_model
    mock_predictor.preprocessor = mock_preprocessor
    mock_predictor.is_loaded = True
    
    # Mock predict method to return valid Pydantic model
    def mock_predict(data, **kwargs):
        return PredictionResponse(
            prediction=0,
            probability=0.25,
            risk=RiskAssessment(
                tier=RiskTier.LOW,
                probability=0.25,
                confidence="High",
                color="#00FF00",
                emoji="🟢"
            ),
            intervention=InterventionRecommendation(
                action="Standard Reminder",
                sms_reminders=1,
                phone_call=False,
                priority="Normal",
                notes=None
            ),
            explanation=None,
            model_version="1.0.0",
            timestamp="2023-01-01T00:00:00"
        )
    
    mock_predictor.predict = mock_predict
    
    # Mock predict_batch method
    def mock_predict_batch(data, **kwargs):
        predictions = []
        for i in range(len(data)):
            predictions.append(mock_predict(data[i]))
        
        return BatchPredictionResponse(
            predictions=predictions,
            summary={
                "high_risk_count": 0,
                "average_risk_score": 2.5,
                "predicted_noshows": 0,
                "predicted_shows": len(data),
                "total": len(data),
                "avg_probability": 0.25,
                "risk_distribution": {"LOW": len(data)}
            },
            processing_time_ms=100.0
        )

    mock_predictor.predict_batch = mock_predict_batch
    
    # Mock get_model_info
    mock_predictor.get_model_info.return_value = {
        "name": "XGBoost",
        "version": "1.0.0",
        "type": "Classifier",
        "features": 3,
        "metrics": {"accuracy": 0.85}
    }

    # Use dependency overrides
    app.dependency_overrides[get_predictor] = lambda: mock_predictor
    
    yield
    
    # Clean up overrides
    app.dependency_overrides.clear()


@pytest.fixture
def sample_appointment() -> Dict[str, Any]:
    """Sample appointment data for testing."""
    return {
        "age": 35,
        "gender": "F",
        "scholarship": 0,
        "hypertension": 0,
        "diabetes": 0,
        "alcoholism": 0,
        "handicap": 0,
        "sms_received": 1,
        "lead_days": 7
    }


@pytest.fixture
def sample_appointment_minimal() -> Dict[str, Any]:
    """Minimal appointment data (only required fields)."""
    return {
        "age": 25,
        "gender": "M",
        "lead_days": 3
    }


@pytest.fixture
def sample_appointment_full() -> Dict[str, Any]:
    """Full appointment data with all fields."""
    return {
        "age": 45,
        "gender": "F",
        "scholarship": 1,
        "hypertension": 1,
        "diabetes": 0,
        "alcoholism": 0,
        "handicap": 0,
        "sms_received": 1,
        "lead_days": 14,
        "neighbourhood": "JARDIM CAMBURI",
        "appointment_weekday": "Monday",
        "appointment_month": "May",
        "patient_historical_noshow_rate": 0.15,
        "patient_total_appointments": 5,
        "is_first_appointment": 0
    }


@pytest.fixture
def sample_batch_request(sample_appointment) -> Dict[str, Any]:
    """Sample batch prediction request."""
    appointments = [
        sample_appointment,
        {**sample_appointment, "age": 25, "lead_days": 1},
        {**sample_appointment, "age": 65, "lead_days": 30},
    ]
    return {
        "appointments": appointments,
        "include_explanations": False,
        "threshold": 0.5
    }


@pytest.fixture
def invalid_appointment() -> Dict[str, Any]:
    """Invalid appointment data for testing validation."""
    return {
        "age": -5,  # Invalid: negative age
        "gender": "X",  # Invalid: not M/F/O
        "lead_days": 500  # Invalid: too high
    }