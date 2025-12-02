# src/llm/production/error_handling.py
"""
Error Handling
==============
Robust error handling for production LLM applications.
"""

import logging
from typing import Optional, Callable, Any, TypeVar, List
from functools import wraps
from datetime import datetime
import time
import asyncio

logger = logging.getLogger(__name__)

T = TypeVar('T')


# ==================== Custom Exceptions ====================

class LLMError(Exception):
    """Base exception for LLM errors."""
    
    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message)
        self.details = details or {}
        self.timestamp = datetime.utcnow()


class RateLimitError(LLMError):
    """Rate limit exceeded."""
    
    def __init__(self, message: str, retry_after: Optional[float] = None):
        super().__init__(message)
        self.retry_after = retry_after


class ContentFilterError(LLMError):
    """Content was filtered by safety systems."""
    
    def __init__(self, message: str, filter_type: str = "unknown"):
        super().__init__(message)
        self.filter_type = filter_type


class TimeoutError(LLMError):
    """Request timed out."""
    pass


class ModelUnavailableError(LLMError):
    """Model is unavailable."""
    pass


class InvalidInputError(LLMError):
    """Invalid input provided."""
    pass


# ==================== Error Handler ====================

class ErrorHandler:
    """
    Central error handling for LLM operations.
    
    Features:
    - Error classification
    - Automatic retry with backoff
    - Fallback responses
    - Error logging and alerting
    """
    
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        fallback_response: Optional[str] = None
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.fallback_response = fallback_response or (
            "I apologize, but I'm experiencing technical difficulties. "
            "Please try again in a moment or contact support."
        )
        
        # Error tracking
        self._error_counts: dict = {}
        self._last_errors: List[dict] = []
    
    def handle(
        self,
        error: Exception,
        context: Optional[dict] = None
    ) -> dict:
        """
        Handle an error and return appropriate response.
        
        Parameters
        ----------
        error : Exception
            The error to handle
        context : dict, optional
            Additional context
        
        Returns
        -------
        dict
            Error response with details
        """
        error_type = type(error).__name__
        
        # Track error
        self._error_counts[error_type] = self._error_counts.get(error_type, 0) + 1
        self._last_errors.append({
            "type": error_type,
            "message": str(error),
            "timestamp": datetime.utcnow().isoformat(),
            "context": context
        })
        
        # Keep only last 100 errors
        if len(self._last_errors) > 100:
            self._last_errors = self._last_errors[-100:]
        
        # Log error
        logger.error(f"{error_type}: {error}", exc_info=True)
        
        # Classify and respond
        if isinstance(error, RateLimitError):
            return {
                "error": True,
                "error_type": "rate_limit",
                "message": "Too many requests. Please wait and try again.",
                "retry_after": error.retry_after,
                "user_message": self.fallback_response
            }
        
        elif isinstance(error, ContentFilterError):
            return {
                "error": True,
                "error_type": "content_filter",
                "message": "Request was filtered for safety reasons.",
                "user_message": "I cannot process this request. Please rephrase your question."
            }
        
        elif isinstance(error, TimeoutError):
            return {
                "error": True,
                "error_type": "timeout",
                "message": "Request timed out.",
                "user_message": "The request took too long. Please try again."
            }
        
        else:
            return {
                "error": True,
                "error_type": "unknown",
                "message": str(error),
                "user_message": self.fallback_response
            }
    
    def get_stats(self) -> dict:
        """Get error statistics."""
        return {
            "error_counts": self._error_counts,
            "total_errors": sum(self._error_counts.values()),
            "recent_errors": self._last_errors[-10:]
        }


# ==================== Decorators ====================

def safe_llm_call(
    max_retries: int = 3,
    base_delay: float = 1.0,
    fallback: Optional[Any] = None,
    on_error: Optional[Callable] = None
):
    """
    Decorator for safe LLM calls with retry logic.
    
    Example
    -------
    >>> @safe_llm_call(max_retries=3, fallback="Default response")
    ... def generate_response(prompt):
    ...     return llm.invoke(prompt)
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_error = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                    
                except Exception as e:
                    last_error = e
                    
                    # Check if retryable
                    if isinstance(e, (RateLimitError, TimeoutError)):
                        delay = min(base_delay * (2 ** attempt), 60)
                        
                        if isinstance(e, RateLimitError) and e.retry_after:
                            delay = e.retry_after
                        
                        logger.warning(f"Retry {attempt + 1}/{max_retries} after {delay}s: {e}")
                        time.sleep(delay)
                    
                    elif isinstance(e, ContentFilterError):
                        # Don't retry content filter errors
                        break
                    
                    else:
                        # Unknown error - maybe retry
                        if attempt < max_retries - 1:
                            delay = base_delay * (2 ** attempt)
                            logger.warning(f"Retry {attempt + 1}/{max_retries}: {e}")
                            time.sleep(delay)
            
            # All retries exhausted
            if on_error:
                on_error(last_error)
            
            if fallback is not None:
                logger.warning(f"Using fallback after {max_retries} attempts")
                return fallback
            
            raise last_error
        
        return wrapper
    return decorator


def with_fallback(fallback_func: Callable):
    """
    Decorator to use fallback function on error.
    
    Example
    -------
    >>> @with_fallback(lambda x: "Fallback response")
    ... def generate(prompt):
    ...     return llm.invoke(prompt)
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Using fallback due to: {e}")
                return fallback_func(*args, **kwargs)
        
        return wrapper
    return decorator


def with_timeout(seconds: float):
    """
    Decorator to add timeout to a function.
    
    Example
    -------
    >>> @with_timeout(30)
    ... def slow_function():
    ...     pass
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            import signal
            
            def handler(signum, frame):
                raise TimeoutError(f"Function timed out after {seconds}s")
            
            # Set alarm (Unix only)
            try:
                old_handler = signal.signal(signal.SIGALRM, handler)
                signal.alarm(int(seconds))
                
                try:
                    result = func(*args, **kwargs)
                finally:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)
                
                return result
                
            except AttributeError:
                # Windows doesn't have SIGALRM
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


# ==================== Circuit Breaker ====================

class CircuitBreaker:
    """
    Circuit breaker pattern for LLM calls.
    
    Prevents cascading failures by stopping calls
    when error rate is too high.
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        success_threshold: int = 3
    ):
        """
        Initialize circuit breaker.
        
        Parameters
        ----------
        failure_threshold : int
            Failures before opening circuit
        recovery_timeout : float
            Seconds before trying again
        success_threshold : int
            Successes needed to close circuit
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        
        self._state = "closed"  # closed, open, half-open
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = None
    
    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """Use as decorator."""
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            # Check circuit state
            if self._state == "open":
                if self._should_attempt_reset():
                    self._state = "half-open"
                else:
                    raise LLMError("Circuit breaker is open")
            
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
                
            except Exception as e:
                self._on_failure()
                raise
        
        return wrapper
    
    def _should_attempt_reset(self) -> bool:
        """Check if we should try to reset."""
        if self._last_failure_time is None:
            return True
        
        elapsed = (datetime.utcnow() - self._last_failure_time).total_seconds()
        return elapsed >= self.recovery_timeout
    
    def _on_success(self):
        """Handle successful call."""
        if self._state == "half-open":
            self._success_count += 1
            if self._success_count >= self.success_threshold:
                self._state = "closed"
                self._failure_count = 0
                self._success_count = 0
                logger.info("Circuit breaker closed")
        else:
            self._failure_count = 0
    
    def _on_failure(self):
        """Handle failed call."""
        self._failure_count += 1
        self._last_failure_time = datetime.utcnow()
        
        if self._state == "half-open":
            self._state = "open"
            self._success_count = 0
            logger.warning("Circuit breaker reopened")
        
        elif self._failure_count >= self.failure_threshold:
            self._state = "open"
            logger.warning("Circuit breaker opened")
    
    @property
    def state(self) -> str:
        """Get current state."""
        return self._state