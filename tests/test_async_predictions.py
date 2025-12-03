# tests/test_async_predictions.py
"""
Async Prediction Tests
======================
Integration tests for asynchronous predictions using Celery.
"""

import pytest
import httpx
import time

BASE_URL = "http://localhost:8001"

def test_async_prediction_flow():
    """Test the full async prediction flow."""
    payload = {
        "age": 30,
        "gender": "M",
        "lead_days": 5,
        "scholarship": 0,
        "hypertension": 0,
        "diabetes": 0,
        "alcoholism": 0,
        "handicap": 0,
        "sms_received": 1,
        "patient_historical_noshow_rate": 0.1
    }
    
    with httpx.Client(base_url=BASE_URL, timeout=30.0) as client:
        # 1. Submit async prediction
        response = client.post("/api/v1/predict?async_mode=true", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "task_id" in data
        assert data["status"] == "processing"
        task_id = data["task_id"]
        
        # 2. Poll for result
        max_retries = 30
        for _ in range(max_retries):
            status_response = client.get(f"/api/v1/predict/tasks/{task_id}")
            assert status_response.status_code == 200
            status_data = status_response.json()
            
            if status_data["status"] == "SUCCESS":
                result = status_data["result"]
                assert "probability" in result
                assert "risk" in result
                assert "tier" in result["risk"]
                break
            elif status_data["status"] == "FAILURE":
                pytest.fail(f"Task failed: {status_data.get('error')}")
            
            time.sleep(1)
        else:
            pytest.fail("Task timed out")

def test_async_batch_prediction_flow():
    """Test the full async batch prediction flow."""
    payload = {
        "appointments": [
            {"age": 30, "gender": "M", "lead_days": 5},
            {"age": 50, "gender": "F", "lead_days": 15}
        ]
    }
    
    with httpx.Client(base_url=BASE_URL, timeout=30.0) as client:
        # 1. Submit async batch prediction
        response = client.post("/api/v1/predict/batch?async_mode=true", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "task_id" in data
        task_id = data["task_id"]
        
        # 2. Poll for result
        max_retries = 30
        for _ in range(max_retries):
            status_response = client.get(f"/api/v1/predict/tasks/{task_id}")
            assert status_response.status_code == 200
            status_data = status_response.json()
            
            if status_data["status"] == "SUCCESS":
                results = status_data["result"]
                assert isinstance(results, list)
                assert len(results) == 2
                break
            elif status_data["status"] == "FAILURE":
                pytest.fail(f"Task failed: {status_data.get('error')}")
            
            time.sleep(1)
        else:
            pytest.fail("Task timed out")
