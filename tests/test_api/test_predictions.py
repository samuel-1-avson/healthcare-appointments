"""
Prediction Endpoint Tests
=========================
Tests for prediction API endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from typing import Dict, Any

from tests.test_api.utils import assert_valid_prediction_response


class TestSinglePrediction:
    """Tests for single prediction endpoint."""
    
    def test_predict_with_valid_data(
        self, 
        client: TestClient, 
        sample_appointment: Dict[str, Any]
    ):
        """Test prediction with valid appointment data."""
        response = client.post(
            "/api/v1/predict",
            json=sample_appointment
        )
        
        assert response.status_code == 200
        data = response.json()
        assert_valid_prediction_response(data)
    
    def test_predict_with_minimal_data(
        self, 
        client: TestClient, 
        sample_appointment_minimal: Dict[str, Any]
    ):
        """Test prediction with minimal required fields."""
        response = client.post(
            "/api/v1/predict",
            json=sample_appointment_minimal
        )
        
        assert response.status_code == 200
        data = response.json()
        assert_valid_prediction_response(data)
    
    def test_predict_with_full_data(
        self, 
        client: TestClient, 
        sample_appointment_full: Dict[str, Any]
    ):
        """Test prediction with all fields populated."""
        response = client.post(
            "/api/v1/predict",
            json=sample_appointment_full
        )
        
        assert response.status_code == 200
        data = response.json()
        assert_valid_prediction_response(data)
    
    def test_predict_with_custom_threshold(
        self, 
        client: TestClient, 
        sample_appointment: Dict[str, Any]
    ):
        """Test prediction with custom threshold parameter."""
        response = client.post(
            "/api/v1/predict?threshold=0.3",
            json=sample_appointment
        )
        
        assert response.status_code == 200
        data = response.json()
        assert_valid_prediction_response(data)
    
    def test_predict_with_explanation(
        self, 
        client: TestClient, 
        sample_appointment: Dict[str, Any]
    ):
        """Test prediction with explanation requested."""
        response = client.post(
            "/api/v1/predict?include_explanation=true",
            json=sample_appointment
        )
        
        assert response.status_code == 200
        data = response.json()
        assert_valid_prediction_response(data)
        # Note: explanation may be None depending on implementation
    
    def test_predict_returns_risk_tier(
        self, 
        client: TestClient, 
        sample_appointment: Dict[str, Any]
    ):
        """Test that prediction includes valid risk tier."""
        response = client.post(
            "/api/v1/predict",
            json=sample_appointment
        )
        
        assert response.status_code == 200
        data = response.json()
        
        valid_tiers = ["CRITICAL", "HIGH", "MEDIUM", "LOW", "MINIMAL"]
        assert data["risk"]["tier"] in valid_tiers
    
    def test_predict_returns_intervention(
        self, 
        client: TestClient, 
        sample_appointment: Dict[str, Any]
    ):
        """Test that prediction includes intervention recommendation."""
        response = client.post(
            "/api/v1/predict",
            json=sample_appointment
        )
        
        assert response.status_code == 200
        data = response.json()
        
        intervention = data["intervention"]
        assert "action" in intervention
        assert "sms_reminders" in intervention
        assert isinstance(intervention["sms_reminders"], int)
        assert "phone_call" in intervention
        assert isinstance(intervention["phone_call"], bool)
    
    def test_predict_response_has_timestamp(
        self, 
        client: TestClient, 
        sample_appointment: Dict[str, Any]
    ):
        """Test that prediction includes timestamp."""
        response = client.post(
            "/api/v1/predict",
            json=sample_appointment
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "timestamp" in data


class TestPredictionValidation:
    """Tests for input validation on prediction endpoint."""
    
    def test_predict_rejects_missing_required_fields(self, client: TestClient):
        """Test that missing required fields are rejected."""
        # Missing age
        response = client.post(
            "/api/v1/predict",
            json={"gender": "F", "lead_days": 7}
        )
        assert response.status_code == 422
    
    def test_predict_rejects_invalid_age(self, client: TestClient):
        """Test that invalid age values are rejected."""
        # Negative age
        response = client.post(
            "/api/v1/predict",
            json={"age": -5, "gender": "F", "lead_days": 7}
        )
        assert response.status_code == 422
        
        # Age too high
        response = client.post(
            "/api/v1/predict",
            json={"age": 150, "gender": "F", "lead_days": 7}
        )
        assert response.status_code == 422
    
    def test_predict_rejects_invalid_gender(self, client: TestClient):
        """Test that invalid gender values are rejected."""
        response = client.post(
            "/api/v1/predict",
            json={"age": 35, "gender": "X", "lead_days": 7}
        )
        assert response.status_code == 422
    
    def test_predict_rejects_invalid_lead_days(self, client: TestClient):
        """Test that invalid lead_days values are rejected."""
        # Negative lead days
        response = client.post(
            "/api/v1/predict",
            json={"age": 35, "gender": "F", "lead_days": -1}
        )
        assert response.status_code == 422
    
    def test_predict_rejects_invalid_binary_fields(self, client: TestClient):
        """Test that binary fields only accept 0 or 1."""
        response = client.post(
            "/api/v1/predict",
            json={
                "age": 35, 
                "gender": "F", 
                "lead_days": 7,
                "sms_received": 5  # Invalid: should be 0 or 1
            }
        )
        assert response.status_code == 422
    
    def test_predict_rejects_invalid_threshold(self, client: TestClient):
        """Test that invalid threshold values are rejected."""
        response = client.post(
            "/api/v1/predict?threshold=1.5",
            json={"age": 35, "gender": "F", "lead_days": 7}
        )
        assert response.status_code == 422
    
    def test_predict_accepts_gender_variations(self, client: TestClient):
        """Test that various gender formats are accepted."""
        valid_genders = ["M", "F", "m", "f", "Male", "Female"]
        
        for gender in valid_genders:
            response = client.post(
                "/api/v1/predict",
                json={"age": 35, "gender": gender, "lead_days": 7}
            )
            assert response.status_code == 200, f"Failed for gender: {gender}"


class TestBatchPrediction:
    """Tests for batch prediction endpoint."""
    
    def test_batch_predict_success(
        self, 
        client: TestClient, 
        sample_batch_request: Dict[str, Any]
    ):
        """Test batch prediction with valid data."""
        response = client.post(
            "/api/v1/predict/batch",
            json=sample_batch_request
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "predictions" in data
        assert "summary" in data
        assert "processing_time_ms" in data
        
        # Check predictions list
        assert len(data["predictions"]) == len(sample_batch_request["appointments"])
        
        for pred in data["predictions"]:
            assert_valid_prediction_response(pred)
    
    def test_batch_predict_includes_summary(
        self, 
        client: TestClient, 
        sample_batch_request: Dict[str, Any]
    ):
        """Test that batch prediction includes summary statistics."""
        response = client.post(
            "/api/v1/predict/batch",
            json=sample_batch_request
        )
        
        assert response.status_code == 200
        data = response.json()
        
        summary = data["summary"]
        assert "total" in summary
        assert "predicted_noshows" in summary
        assert "predicted_shows" in summary
        assert "avg_probability" in summary
        assert "risk_distribution" in summary
    
    def test_batch_predict_empty_list_rejected(self, client: TestClient):
        """Test that empty appointment list is rejected."""
        response = client.post(
            "/api/v1/predict/batch",
            json={"appointments": []}
        )
        assert response.status_code == 422
    
    def test_batch_predict_large_batch(self, client: TestClient):
        """Test batch prediction with many appointments."""
        # Create 100 appointments
        appointments = [
            {"age": 20 + i % 60, "gender": "F" if i % 2 else "M", "lead_days": i % 30}
            for i in range(100)
        ]
        
        response = client.post(
            "/api/v1/predict/batch",
            json={"appointments": appointments}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["predictions"]) == 100


class TestQuickPrediction:
    """Tests for quick prediction endpoint."""
    
    def test_quick_predict_success(self, client: TestClient):
        """Test quick prediction with query parameters."""
        response = client.post(
            "/api/v1/predict/quick?age=35&gender=F&lead_days=7&sms_received=1"
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "probability" in data
        assert "prediction" in data
        assert "risk_tier" in data
        assert "intervention" in data
    
    def test_quick_predict_missing_params(self, client: TestClient):
        """Test quick prediction with missing required parameters."""
        response = client.post(
            "/api/v1/predict/quick?age=35&gender=F"  # Missing lead_days
        )
        assert response.status_code == 422


class TestThresholdInfo:
    """Tests for threshold information endpoint."""
    
    def test_get_thresholds(self, client: TestClient):
        """Test getting threshold information."""
        response = client.get("/api/v1/predict/thresholds")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "default_threshold" in data
        assert "risk_tiers" in data
        assert "explanation" in data