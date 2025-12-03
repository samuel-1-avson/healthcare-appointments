"""
Redis Cache Module
==================
Handles Redis connection and caching logic.
"""

import json
import logging
from functools import wraps
from typing import Optional, Any, Callable
import redis
from fastapi import Request, Response

from .config import get_settings

logger = logging.getLogger(__name__)

class RedisClient:
    """Singleton Redis client wrapper."""
    
    _instance = None
    _client: Optional[redis.Redis] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RedisClient, cls).__new__(cls)
        return cls._instance
    
    def connect(self):
        """Initialize Redis connection."""
        settings = get_settings()
        if not settings.redis_enabled:
            logger.info("Redis disabled by configuration")
            return

        try:
            self._client = redis.Redis(
                host=settings.redis_host,
                port=settings.redis_port,
                db=settings.redis_db,
                decode_responses=True,
                socket_timeout=5
            )
            self._client.ping()
            logger.info(f"✅ Connected to Redis at {settings.redis_host}:{settings.redis_port}")
        except redis.ConnectionError as e:
            logger.warning(f"⚠️ Failed to connect to Redis: {e}")
            self._client = None
            
    def close(self):
        """Close Redis connection."""
        if self._client:
            self._client.close()
            logger.info("Redis connection closed")
            
    def get(self, key: str) -> Optional[str]:
        """Get value from cache."""
        if not self._client:
            return None
        try:
            return self._client.get(key)
        except Exception as e:
            logger.error(f"Redis GET error: {e}")
            return None
            
    def set(self, key: str, value: str, expire: int = 300) -> bool:
        """Set value in cache with expiration."""
        if not self._client:
            return False
        try:
            return self._client.set(key, value, ex=expire)
        except Exception as e:
            logger.error(f"Redis SET error: {e}")
            return False

    @property
    def is_connected(self) -> bool:
        return self._client is not None


def cache_response(expire: int = 60):
    """
    Decorator to cache FastAPI response.
    
    Parameters
    ----------
    expire : int
        Expiration time in seconds
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract request object
            request: Optional[Request] = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break
            if not request:
                for key, value in kwargs.items():
                    if isinstance(value, Request) or key == "request":
                        request = value
                        break
            
            # If no request object, we can't generate a key based on URL/Body
            if request is None:
                logger.warning("Cache decorator: No request object found in args/kwargs")
                return await func(*args, **kwargs)
                
            # Generate cache key base
            key_parts = [f"cache:{request.url.path}:{request.url.query}"]
            logger.info(f"Cache key base: {key_parts[0]}")
            
            # For POST/PUT/PATCH, include body hash
            if request.method in ["POST", "PUT", "PATCH"]:
                try:
                    # Check for Pydantic models in kwargs
                    body_content = ""
                    for name, value in kwargs.items():
                        if hasattr(value, "model_dump_json"):
                            body_content += value.model_dump_json()
                        elif isinstance(value, dict):
                             body_content += json.dumps(value, sort_keys=True)
                    
                    if body_content:
                        import hashlib
                        body_hash = hashlib.md5(body_content.encode()).hexdigest()
                        key_parts.append(f"body:{body_hash}")
                        
                except Exception as e:
                    logger.warning(f"Failed to hash request body for cache key: {e}")
            
            cache_key = ":".join(key_parts)
            
            redis_client = RedisClient()
            
            # Try to get from cache
            cached_data = redis_client.get(cache_key)
            if cached_data:
                logger.info(f"Cache HIT: {cache_key}")
                try:
                    data = json.loads(cached_data)
                    return data
                except json.JSONDecodeError:
                    pass
            
            # Execute function
            response = await func(*args, **kwargs)
            
            # Cache response
            try:
                if hasattr(response, "model_dump_json"):
                    data_str = response.model_dump_json()
                elif hasattr(response, "json"):
                     data_str = response.json()
                elif isinstance(response, dict) or isinstance(response, list):
                    data_str = json.dumps(response)
                else:
                    return response
                
                redis_client.set(cache_key, data_str, expire=expire)
                logger.info(f"Cache SET: {cache_key}")
            except Exception as e:
                logger.warning(f"Failed to cache response: {e}")
                
            return response
        return wrapper
    return decorator


def get_redis_client() -> Optional[redis.Redis]:
    """
    Get the underlying Redis client instance.
    
    Returns
    -------
    redis.Redis or None
        Redis client if connected
    """
    client = RedisClient()
    if not client.is_connected:
        client.connect()
    return client._client
