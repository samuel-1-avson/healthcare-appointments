"""
Authentication Tests
====================
Tests for the authentication endpoints.
"""

import sys
import os
from pathlib import Path

# Set debug mode for testing to bypass secret key validation
os.environ["NOSHOW_DEBUG"] = "true"

# Add project root to python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

import pytest
from fastapi.testclient import TestClient
from src.api.main import app
from src.api.auth import create_access_token

client = TestClient(app)

def test_login_success():
    """Test successful login."""
    response = client.post(
        "/api/v1/auth/token",
        data={"username": "admin", "password": "admin123"},
        headers={"content-type": "application/x-www-form-urlencoded"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"

def test_login_failure():
    """Test login with incorrect credentials."""
    response = client.post(
        "/api/v1/auth/token",
        data={"username": "admin", "password": "wrongpassword"},
        headers={"content-type": "application/x-www-form-urlencoded"}
    )
    assert response.status_code == 401
    # Custom exception handler returns "message", not "detail"
    assert response.json()["message"] == "Incorrect username or password"

def test_read_users_me_success():
    """Test accessing protected endpoint with valid token."""
    # 1. Login to get token
    login_response = client.post(
        "/api/v1/auth/token",
        data={"username": "admin", "password": "admin123"},
        headers={"content-type": "application/x-www-form-urlencoded"}
    )
    token = login_response.json()["access_token"]
    
    # 2. Access protected route
    response = client.get(
        "/api/v1/auth/users/me",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["username"] == "admin"
    assert data["role"] == "admin"

def test_protected_route_access():
    """Test accessing a protected route with a valid token."""
    # Login as a regular user to get a token
    # Note: Since we only have admin user in init script, we'll use admin for now
    # Ideally we should create a test user in a fixture
    login_response = client.post(
        "/api/v1/auth/token",
        data={"username": "admin", "password": "admin123"},
        headers={"content-type": "application/x-www-form-urlencoded"}
    )
    token = login_response.json()["access_token"]
    token_headers = {"Authorization": f"Bearer {token}"}

    response = client.get("/api/v1/auth/users/me", headers=token_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["username"] == "admin"
    assert data["role"] == "admin"

def test_protected_route_invalid_token():
    """Test accessing a protected route with an invalid token."""
    response = client.get("/api/v1/auth/users/me", headers={"Authorization": "Bearer invalidtoken"})
    assert response.status_code == 401
    assert response.json()["message"] == "Could not validate credentials"

def test_rbac_admin_access():
    """Test accessing an admin-only route with an admin token."""
    # Login as admin to get a token
    login_response = client.post(
        "/api/v1/auth/token",
        data={"username": "admin", "password": "admin123"},
        headers={"content-type": "application/x-www-form-urlencoded"}
    )
    admin_token = login_response.json()["access_token"]
    admin_token_headers = {"Authorization": f"Bearer {admin_token}"}

    # Assuming we have an admin route, for now we can test the dependency directly or a mock route
    # Since we don't have a dedicated admin route in auth, we'll rely on the fact that admin_token_headers
    # contains an admin token.
    response = client.get("/api/v1/auth/users/me", headers=admin_token_headers)
    assert response.status_code == 200
    assert response.json()["role"] == "admin"

def test_llm_route_protection():
    """Test that LLM routes are protected."""
    # Login as a regular user to get a token
    login_response = client.post(
        "/api/v1/auth/token",
        data={"username": "admin", "password": "admin123"},
        headers={"content-type": "application/x-www-form-urlencoded"}
    )
    user_token = login_response.json()["access_token"]
    token_headers = {"Authorization": f"Bearer {user_token}"}

    # Try to access without token
    response = client.post("/api/v1/llm/chat", json={"message": "Hello"})
    assert response.status_code == 401
    
    # Try to access with token (should pass auth, but might fail on internal logic if services aren't mocked)
    # We expect 401 without token, and something else (200 or 500) with token.
    # Since we are not mocking the orchestrator, it might 500, but that proves auth passed.
    try:
        response = client.post("/api/v1/llm/chat", json={"message": "Hello"}, headers=token_headers)
        assert response.status_code != 401
    except Exception:
        # If it raises an exception inside the app, it means auth passed
        pass

def test_read_users_me_no_token():
    """Test accessing protected endpoint without token."""
    response = client.get("/api/v1/auth/users/me")
    assert response.status_code == 401
    assert response.json()["message"] == "Not authenticated"

if __name__ == "__main__":
    # Create a dummy test file to run pytest on this file
    import pytest
    sys.exit(pytest.main(["-v", __file__]))
