"""
Health Endpoint Tests
=====================
Tests for API health check and monitoring endpoints.
"""

import pytest
from fastapi.testclient import TestClient


class TestHealthEndpoints:
    """Tests for health check endpoints."""
    
    def test_root_redirects_to_docs(self, client: TestClient):
        """Test that root path redirects to documentation."""
        response = client.get("/", follow_redirects=False)
        assert response.status_code in [301, 302, 307]
        assert "/docs" in response.headers.get("location", "")
    
    def test_health_check_success(self, client: TestClient):
        """Test health check endpoint returns healthy status."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] in ["healthy", "degraded"]
        assert "timestamp" in data
        assert "version" in data
        assert "model_loaded" in data
    
    def test_health_check_includes_model_info(self, client: TestClient):
        """Test health check includes model information when loaded."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        if data["model_loaded"]:
            assert "model_info" in data
            model_info = data["model_info"]
            assert "name" in model_info
            assert "version" in model_info
            assert "type" in model_info
    
    def test_readiness_check(self, client: TestClient):
        """Test readiness probe endpoint."""
        response = client.get("/ready")
        assert response.status_code == 200
        
        data = response.json()
        assert "ready" in data
        assert "timestamp" in data
    
    def test_liveness_check(self, client: TestClient):
        """Test liveness probe endpoint."""
        response = client.get("/live")
        assert response.status_code == 200
        
        data = response.json()
        assert data["alive"] is True
        assert "timestamp" in data
    
    def test_health_response_time(self, client: TestClient):
        """Test that health check responds quickly."""
        import time
        
        start = time.time()
        response = client.get("/health")
        elapsed = time.time() - start
        
        assert response.status_code == 200
        assert elapsed < 1.0  # Should respond in under 1 second


class TestAPIInfo:
    """Tests for API information endpoints."""
    
    def test_openapi_schema_available(self, client: TestClient):
        """Test that OpenAPI schema is available."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        
        data = response.json()
        assert "openapi" in data
        assert "info" in data
        assert "paths" in data
    
    def test_docs_available(self, client: TestClient):
        """Test that Swagger docs are available."""
        response = client.get("/docs")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")
    
    def test_redoc_available(self, client: TestClient):
        """Test that ReDoc is available."""
        response = client.get("/redoc")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")