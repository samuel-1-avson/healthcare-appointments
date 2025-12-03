import pytest
from unittest.mock import MagicMock, patch
from fastapi import Request
import json
import asyncio
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api.cache import cache_response, RedisClient

# Mock Pydantic model
class MockModel:
    def __init__(self, data):
        self.data = data
    
    def model_dump_json(self):
        return json.dumps(self.data, sort_keys=True)

@pytest.mark.asyncio
@patch('src.api.cache.RedisClient')
async def test_cache_key_generation_post(MockRedisClientClass):
    """Test that cache key includes body hash for POST requests."""
    # Reset singleton
    from src.api.cache import RedisClient as RealRedisClient
    RealRedisClient._instance = None
    
    # Configure mock instance
    mock_instance = MockRedisClientClass.return_value
    mock_instance.get.return_value = None
    
    # Mock Request
    mock_request = MagicMock(spec=Request)
    mock_request.method = "POST"
    mock_request.url.path = "/test"
    mock_request.url.query = ""
    
    # Mock endpoint function
    @cache_response(expire=60)
    async def test_endpoint(request: Request, body: MockModel):
        return {"result": "success"}
    
    # Call with body 1
    body1 = MockModel({"key": "value1"})
    await test_endpoint(request=mock_request, body=body1)
    
    # Check set call
    assert mock_instance.set.called
    args1 = mock_instance.set.call_args[0]
    key1 = args1[0]
    
    # Call with body 2
    body2 = MockModel({"key": "value2"})
    await test_endpoint(request=mock_request, body=body2)
    
    # Check set call
    args2 = mock_instance.set.call_args[0]
    key2 = args2[0]
    
    # Keys should be different
    assert key1 != key2
    assert "body:" in key1
    assert "body:" in key2

@pytest.mark.asyncio
@patch('src.api.cache.RedisClient')
async def test_cache_hit(MockRedisClientClass):
    """Test that cached value is returned."""
    # Reset singleton
    from src.api.cache import RedisClient as RealRedisClient
    RealRedisClient._instance = None
    
    mock_instance = MockRedisClientClass.return_value
    mock_instance.get.return_value = json.dumps({"cached": "true"})
    
    mock_request = MagicMock(spec=Request)
    mock_request.method = "GET"
    mock_request.url.path = "/test"
    mock_request.url.query = ""
    
    @cache_response(expire=60)
    async def test_endpoint(request: Request):
        return {"cached": "false"}
    
    result = await test_endpoint(request=mock_request)
    
    assert result == {"cached": "true"}
    mock_instance.get.assert_called_once()
