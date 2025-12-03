import pytest
import httpx
import time

# Use localhost for external testing
API_URL = "http://localhost:8001/health"

def test_redis_via_health_check():
    """Test Redis connection via API health endpoint."""
    try:
        # Retry a few times to allow container to stabilize
        for _ in range(5):
            try:
                response = httpx.get(API_URL)
                if response.status_code == 200:
                    break
            except httpx.ConnectError:
                time.sleep(1)
                continue
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["components"]["database"] == "connected"
        assert data["components"]["redis"] == "connected"
        
    except Exception as e:
        pytest.fail(f"Health check failed: {e}")


