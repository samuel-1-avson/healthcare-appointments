# src/llm/resilience.py
"""
LLM Resilience & Reliability
=============================
Utilities for building resilient LLM integrations with retry logic,
circuit breakers, and fallback strategies.
"""

import logging
import time
from typing import Callable, Any, Optional, List, Type
from functools import wraps
from enum import Enum

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
from openai import RateLimitError, APIError, Timeout

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Blocked due to failures
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreaker:
    """
    Circuit breaker pattern for protecting against cascading failures.
    
    When too many failures occur, the circuit "opens" and blocks requests
    for a cooldown period before trying again.
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: Type[Exception] = Exception
    ):
        """
        Initialize circuit breaker.
        
        Parameters
        ----------
        failure_threshold : int
            Number of failures before opening circuit
        recovery_timeout : int
            Seconds to wait before attempting recovery
        expected_exception : Type[Exception]
            Exception type to track for failures
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = CircuitState.CLOSED
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection.
        
        Parameters
        ----------
        func : Callable
            Function to execute
        *args, **kwargs
            Function arguments
            
        Returns
        -------
        Any
            Function result
            
        Raises
        ------
        Exception
            If circuit is open or function fails
        """
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                logger.info("Circuit breaker entering HALF_OPEN state")
            else:
                raise Exception(
                    f"Circuit breaker is OPEN. "
                    f"Try again in {self._time_until_reset():.0f}s"
                )
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return True
        return time.time() - self.last_failure_time >= self.recovery_timeout
    
    def _time_until_reset(self) -> float:
        """Calculate seconds until reset attempt."""
        if self.last_failure_time is None:
            return 0
        elapsed = time.time() - self.last_failure_time
        return max(0, self.recovery_timeout - elapsed)
    
    def _on_success(self):
        """Handle successful call."""
        self.failure_count = 0
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED
            logger.info("Circuit breaker CLOSED - service recovered")
    
    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.error(
                f"Circuit breaker OPEN after {self.failure_count} failures. "
                f"Cooldown: {self.recovery_timeout}s"
            )


def retry_with_exponential_backoff(
    max_attempts: int = 3,
    min_wait: int = 1,
    max_wait: int = 10
):
    """
    Decorator for retrying functions with exponential backoff.
    
    Parameters
    ----------
    max_attempts : int
        Maximum number of retry attempts
    min_wait : int
        Minimum wait time in seconds
    max_wait : int
        Maximum wait time in seconds
        
    Returns
    -------
    Callable
        Decorated function
    """
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
        retry=retry_if_exception_type((RateLimitError, APIError, Timeout)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True
    )


class ModelFallbackChain:
    """
    Implement fallback strategy across multiple LLM models.
    
    Try primary model first, fall back to cheaper/faster models on failure.
    """
    
    def __init__(self, models: List[str]):
        """
        Initialize fallback chain.
        
        Parameters
        ----------
        models : List[str]
            List of model names in order of preference
            Example: ["gpt-4", "gpt-3.5-turbo", "gpt-3.5-turbo-16k"]
        """
        self.models = models
        self.current_index = 0
    
    def get_current_model(self) -> str:
        """Get current model in the chain."""
        return self.models[self.current_index]
    
    def fallback(self) -> Optional[str]:
        """
        Move to next model in chain.
        
        Returns
        -------
        str or None
            Next model name, or None if no more fallbacks
        """
        self.current_index += 1
        if self.current_index >= len(self.models):
            logger.error("All fallback models exhausted")
            return None
        
        next_model = self.models[self.current_index]
        logger.warning(f"Falling back to model: {next_model}")
        return next_model
    
    def reset(self):
        """Reset to primary model."""
        self.current_index = 0
    
    def execute_with_fallback(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute function with automatic fallback on failure.
        
        Parameters
        ----------
        func : Callable
            Function that accepts 'model' parameter
        *args, **kwargs
            Function arguments
            
        Returns
        -------
        Any
            Function result
            
        Raises
        ------
        Exception
            If all models in chain fail
        """
        self.reset()
        last_exception = None
        
        for model in self.models:
            try:
                logger.info(f"Attempting with model: {model}")
                kwargs['model'] = model
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                logger.warning(f"Model {model} failed: {e}")
                last_exception = e
                continue
        
        # All models failed
        logger.error("All models in fallback chain failed")
        raise last_exception or Exception("All fallback models failed")


# Global circuit breaker instances
_openai_circuit_breaker = CircuitBreaker(
    failure_threshold=5,
    recovery_timeout=60,
    expected_exception=(RateLimitError, APIError, Timeout)
)


def get_openai_circuit_breaker() -> CircuitBreaker:
    """Get the global OpenAI circuit breaker instance."""
    return _openai_circuit_breaker
