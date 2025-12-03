"""
Observability Tests
===================
Tests for metrics, logging, and health checks.
"""

import pytest
import httpx

BASE_URL = "http://localhost:8001"
PROMETHEUS_URL = "http://localhost:9090"

def test_health_check():
    """Test health check endpoint."""
    with httpx.Client(base_url=BASE_URL) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["components"]["api"] == "up"
        assert data["components"]["database"] == "connected"
        assert data["components"]["redis"] == "connected"

def test_metrics_endpoint():
    """Test Prometheus metrics endpoint."""
    with httpx.Client(base_url=BASE_URL) as client:
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "http_requests_total" in response.text
        assert "http_request_duration_seconds" in response.text

def test_prometheus_availability():
    """Test Prometheus is running and accessible."""
    with httpx.Client(base_url=PROMETHEUS_URL) as client:
        response = client.get("/-/healthy")
        assert response.status_code == 200
        assert "Prometheus Server is Healthy" in response.text
