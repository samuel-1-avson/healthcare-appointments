"""
Integration Tests
=================
End-to-end integration tests for the API.
"""

import pytest
from fastapi.testclient import TestClient
import time


class TestAPIIntegration:
    """End-to-end integration tests."""
    
    def test_full_prediction_workflow(self, client: TestClient):
        """Test complete prediction workflow."""
        # 1. Check health
        health_response = client.get("/health")
        assert health_response.status_code == 200
        assert health_response.json()["status"] == "healthy"
        
        # 2. Get model info
        model_response = client.get("/api/v1/model")
        assert model_response.status_code == 200
        
        # 3. Make prediction
        predict_response = client.post(
            "/api/v1/predict",
            json={
                "age": 35,
                "gender": "F",
                "lead_days": 7,
                "sms_received": 1
            }
        )
        assert predict_response.status_code == 200
        prediction = predict_response.json()
        
        # 4. Verify prediction structure
        assert "prediction" in prediction
        assert "probability" in prediction
        assert "risk" in prediction
        assert "intervention" in prediction
    
    def test_batch_prediction_workflow(self, client: TestClient):
        """Test batch prediction workflow."""
        # Create batch of varied appointments
        appointments = [
            {"age": 25, "gender": "M", "lead_days": 1, "sms_received": 1},
            {"age": 35, "gender": "F", "lead_days": 7, "sms_received": 1},
            {"age": 55, "gender": "M", "lead_days": 21, "sms_received": 0},
            {"age": 70, "gender": "F", "lead_days": 30, "sms_received": 1},
        ]
        
        response = client.post(
            "/api/v1/predict/batch",
            json={"appointments": appointments}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify all predictions returned
        assert len(data["predictions"]) == 4
        
        # Verify summary
        summary = data["summary"]
        assert summary["total"] == 4
        assert summary["predicted_noshows"] + summary["predicted_shows"] == 4
    
    def test_error_handling_workflow(self, client: TestClient):
        """Test error handling across endpoints."""
        # Invalid prediction request
        response = client.post(
            "/api/v1/predict",
            json={"age": "invalid"}  # Invalid type
        )
        assert response.status_code == 422
        error = response.json()
        assert "error" in error or "detail" in error
        
        # Invalid endpoint
        response = client.get("/api/v1/nonexistent")
        assert response.status_code == 404
    
    def test_concurrent_predictions(self, client: TestClient):
        """Test handling concurrent prediction requests."""
        import concurrent.futures
        
        def make_prediction(age):
            return client.post(
                "/api/v1/predict",
                json={"age": age, "gender": "F", "lead_days": 7}
            )
        
        # Make 10 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_prediction, 20 + i) for i in range(10)]
            results = [f.result() for f in futures]
        
        # All should succeed
        for result in results:
            assert result.status_code == 200
    
    def test_response_times(self, client: TestClient):
        """Test API response times are acceptable."""
        endpoints = [
            ("GET", "/health"),
            ("GET", "/ready"),
            ("GET", "/api/v1/model"),
            ("GET", "/api/v1/predict/thresholds"),
        ]
        
        for method, path in endpoints:
            start = time.time()
            if method == "GET":
                response = client.get(path)
            elapsed = time.time() - start
            
            assert response.status_code == 200
            assert elapsed < 1.0, f"{path} took {elapsed:.2f}s"
        
        # Prediction should be fast too
        start = time.time()
        response = client.post(
            "/api/v1/predict",
            json={"age": 35, "gender": "F", "lead_days": 7}
        )
        elapsed = time.time() - start
        
        assert response.status_code == 200
        assert elapsed < 2.0, f"Prediction took {elapsed:.2f}s"