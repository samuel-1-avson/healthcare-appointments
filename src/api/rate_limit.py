# src/api/rate_limit.py
"""
Rate Limiting for API Endpoints
================================
Redis-based rate limiting using token bucket algorithm.
"""

import logging
import time
from typing import Optional
from functools import wraps

from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
import redis

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Token bucket rate limiter using Redis.
    
    Allows burst traffic while maintaining average rate limit.
    """
    
    def __init__(
        self,
        redis_client: redis.Redis,
        requests_per_minute: int = 10,
        requests_per_hour: int = 100
    ):
        """
        Initialize rate limiter.
        
        Parameters
        ----------
        redis_client : redis.Redis
            Redis client instance
        requests_per_minute : int
            Requests allowed per minute
        requests_per_hour : int
            Requests allowed per hour
        """
        self.redis = redis_client
        self.rpm_limit = requests_per_minute
        self.rph_limit = requests_per_hour
    
    def is_rate_limited(
        self,
        key: str,
        limit: int,
        window: int
    ) -> tuple[bool, dict]:
        """
        Check if key has exceeded rate limit.
        
        Parameters
        ----------
        key : str
            Redis key for tracking (e.g., "ratelimit:user:123:minute")
        limit : int
            Maximum requests allowed
        window : int
            Time window in seconds
            
        Returns
        -------
        tuple[bool, dict]
            (is_limited, rate_limit_info)
        """
        try:
            # Get current count
            current = self.redis.get(key)
            
            if current is None:
                # First request in window
                self.redis.setex(key, window, 1)
                remaining = limit - 1
                reset_time = int(time.time()) + window
                
                return False, {
                    "limit": limit,
                    "remaining": remaining,
                    "reset": reset_time
                }
            
            current_count = int(current)
            
            if current_count >= limit:
                # Rate limit exceeded
                ttl = self.redis.ttl(key)
                reset_time = int(time.time()) + ttl
                
                return True, {
                    "limit": limit,
                    "remaining": 0,
                    "reset": reset_time,
                    "retry_after": ttl
                }
            
            # Increment counter
            self.redis.incr(key)
            remaining = limit - current_count - 1
            ttl = self.redis.ttl(key)
            reset_time = int(time.time()) + ttl
            
            return False, {
                "limit": limit,
                "remaining": remaining,
                "reset": reset_time
            }
            
        except Exception as e:
            logger.error(f"Rate limit check failed: {e}")
            # Fail open - allow request if Redis is down
            return False, {
                "limit": limit,
                "remaining": limit,
                "reset": int(time.time()) + window
            }
    
    def check_rate_limit(
        self,
        user_id: str,
        endpoint: Optional[str] = None
    ) -> tuple[bool, dict, dict]:
        """
        Check both per-minute and per-hour rate limits.
        
        Parameters
        ----------
        user_id : str
            User identifier
        endpoint : str, optional
            Specific endpoint (for per-endpoint limits)
            
        Returns
        -------
        tuple[bool, dict, dict]
            (is_limited, minute_info, hour_info)
        """
        # Construct keys
        base_key = f"ratelimit:user:{user_id}"
        if endpoint:
            base_key += f":{endpoint}"
        
        minute_key = f"{base_key}:minute"
        hour_key = f"{base_key}:hour"
        
        # Check minute limit
        minute_limited, minute_info = self.is_rate_limited(
            minute_key,
            self.rpm_limit,
            60
        )
        
        # Check hour limit
        hour_limited, hour_info = self.is_rate_limited(
            hour_key,
            self.rph_limit,
            3600
        )
        
        is_limited = minute_limited or hour_limited
        
        return is_limited, minute_info, hour_info


def rate_limit(
    requests_per_minute: int = 10,
    requests_per_hour: int = 100,
    identifier_func: callable = None
):
    """
    Decorator for rate limiting endpoints.
    
    Parameters
    ----------
    requests_per_minute : int
        Requests allowed per minute
    requests_per_hour : int
        Requests allowed per hour
    identifier_func : callable
        Function to extract user identifier from request
        
    Returns
    -------
    Callable
        Decorated function
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get request object
            request: Optional[Request] = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break
            
            if not request:
                # No request object, skip rate limiting
                return await func(*args, **kwargs)
            
            # Get user identifier
            if identifier_func:
                user_id = identifier_func(request)
            else:
                # Default: use IP address or user ID from auth
                user_id = getattr(request.state, "user_id", None)
                if not user_id:
                    user_id = request.client.host if request.client else "unknown"
            
            # Get Redis client
            from .cache import get_redis_client
            redis_client = get_redis_client()
            
            # Check rate limit
            limiter = RateLimiter(
                redis_client,
                requests_per_minute,
                requests_per_hour
            )
            
            endpoint = request.url.path
            is_limited, minute_info, hour_info = limiter.check_rate_limit(
                user_id,
                endpoint
            )
            
            # Add rate limit headers
            headers = {
                "X-RateLimit-Limit-Minute": str(minute_info["limit"]),
                "X-RateLimit-Remaining-Minute": str(minute_info["remaining"]),
                "X-RateLimit-Reset-Minute": str(minute_info["reset"]),
                "X-RateLimit-Limit-Hour": str(hour_info["limit"]),
                "X-RateLimit-Remaining-Hour": str(hour_info["remaining"]),
                "X-RateLimit-Reset-Hour": str(hour_info["reset"]),
            }
            
            if is_limited:
                # Determine which limit was hit
                retry_after = minute_info.get("retry_after") or hour_info.get("retry_after")
                headers["Retry-After"] = str(retry_after)
                
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded. Please try again later.",
                    headers=headers
                )
            
            # Call original function
            response = await func(*args, **kwargs)
            
            # Add headers to response if possible
            if hasattr(response, "headers"):
                for key, value in headers.items():
                    response.headers[key] = value
            
            return response
        
        return wrapper
    return decorator
