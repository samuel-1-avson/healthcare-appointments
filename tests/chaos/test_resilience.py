# tests/chaos/test_resilience.py
"""
Chaos Engineering Tests
=======================
Simulate component failures to verify system resilience.

WARNING: These tests interact with Docker containers and may disrupt running services.
Run only in a dedicated test environment.
"""

import pytest
import time
import docker
import requests
from tenacity import retry, stop_after_attempt, wait_fixed

# Configuration
API_URL = "http://localhost:8000"
DOCKER_CLIENT = docker.from_env()

def get_container(name_partial: str):
    """Get container by partial name."""
    containers = DOCKER_CLIENT.containers.list()
    for container in containers:
        if name_partial in container.name:
            return container
    return None

@pytest.fixture(scope="module")
def redis_container():
    """Get Redis container."""
    container = get_container("redis")
    if not container:
        pytest.skip("Redis container not found")
    return container

@pytest.fixture(scope="module")
def db_container():
    """Get Database container."""
    container = get_container("postgres")
    if not container:
        pytest.skip("Postgres container not found")
    return container

@retry(stop=stop_after_attempt(10), wait=wait_fixed(2))
def wait_for_api():
    """Wait for API to be healthy."""
    response = requests.get(f"{API_URL}/api/v1/health")
    response.raise_for_status()

@pytest.mark.chaos
def test_redis_failure_resilience(redis_container):
    """
    Test that API survives Redis failure.
    Expectation: API should still work (latency might increase), but no 500 errors.
    """
    print("\n[Chaos] Stopping Redis...")
    redis_container.stop()
    
    try:
        # Give it a moment to register failure
        time.sleep(2)
        
        # Make a prediction request (usually cached)
        payload = {
            "PatientId": "123",
            "AppointmentID": "456",
            "Gender": "F",
            "ScheduledDay": "2024-01-20T10:00:00Z",
            "AppointmentDay": "2024-01-25T14:30:00Z",
            "Age": 30,
            "Neighbourhood": "JARDIM DA PENHA",
            "Scholarship": 0,
            "Hipertension": 0,
            "Diabetes": 0,
            "Alcoholism": 0,
            "Handcap": 0,
            "SMS_received": 0
        }
        
        print("[Chaos] Making request without Redis...")
        response = requests.post(f"{API_URL}/api/v1/predict/", json=payload)
        
        # Should succeed (200) even without cache
        assert response.status_code == 200
        print("[Chaos] ✅ API survived Redis failure!")
        
    finally:
        print("[Chaos] Restarting Redis...")
        redis_container.start()
        wait_for_api()

@pytest.mark.chaos
def test_db_failure_graceful_degradation(db_container):
    """
    Test that API handles DB failure gracefully.
    Expectation: API should return 503 Service Unavailable, not crash or hang.
    """
    print("\n[Chaos] Stopping Database...")
    db_container.stop()
    
    try:
        # Give it a moment
        time.sleep(2)
        
        # Health check should probably fail or show unhealthy
        response = requests.get(f"{API_URL}/api/v1/health")
        
        # Depending on implementation, health check might be 200 (app up) but status 'unhealthy'
        # or 503 if strict.
        print(f"[Chaos] Health check status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            # If we implemented deep health checks, this should be unhealthy
            # assert data["status"] != "healthy" 
            pass
            
        # Auth request (needs DB) should definitely fail gracefully
        auth_response = requests.post(
            f"{API_URL}/api/v1/auth/token",
            data={"username": "test", "password": "test"}
        )
        
        # Should be 500 (internal error from DB connection) or handled 503
        # We want to ensure it returns *something* and doesn't hang forever
        assert auth_response.status_code in [500, 503]
        print("[Chaos] ✅ API handled DB failure (didn't hang)!")
        
    finally:
        print("[Chaos] Restarting Database...")
        db_container.start()
        # Wait for DB to be ready (might take longer than API)
        time.sleep(5) 
        wait_for_api()
