# src/llm/tracing.py
"""
LLM Tracing & Observability
===========================
Tracing, logging, and observability for LLM operations.

Integrates with:
- LangSmith (LangChain's tracing platform)
- Custom logging
- Metrics collection
"""

import os
import logging
import time
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime
from dataclasses import dataclass, field
from functools import wraps
import json

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult


logger = logging.getLogger(__name__)


# ==================== Metrics Collection ====================

@dataclass
class LLMMetrics:
    """Collected LLM metrics."""
    
    total_requests: int = 0
    total_tokens: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_cost: float = 0.0
    total_latency_ms: float = 0.0
    
    errors: int = 0
    cache_hits: int = 0
    
    requests_by_model: Dict[str, int] = field(default_factory=dict)
    requests_by_chain: Dict[str, int] = field(default_factory=dict)
    
    def record_request(
        self,
        model: str,
        chain: str,
        tokens: int,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float,
        cost: float = 0.0
    ):
        """Record a request."""
        self.total_requests += 1
        self.total_tokens += tokens
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_latency_ms += latency_ms
        self.total_cost += cost
        
        self.requests_by_model[model] = self.requests_by_model.get(model, 0) + 1
        self.requests_by_chain[chain] = self.requests_by_chain.get(chain, 0) + 1
    
    def record_error(self):
        """Record an error."""
        self.errors += 1
    
    def record_cache_hit(self):
        """Record a cache hit."""
        self.cache_hits += 1
    
    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        avg_latency = (
            self.total_latency_ms / self.total_requests 
            if self.total_requests > 0 else 0
        )
        
        return {
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "estimated_cost_usd": round(self.total_cost, 4),
            "avg_latency_ms": round(avg_latency, 2),
            "errors": self.errors,
            "error_rate": self.errors / max(self.total_requests, 1),
            "cache_hits": self.cache_hits,
            "cache_hit_rate": self.cache_hits / max(self.total_requests + self.cache_hits, 1),
            "by_model": self.requests_by_model,
            "by_chain": self.requests_by_chain
        }


# Global metrics instance
_metrics = LLMMetrics()


def get_metrics() -> LLMMetrics:
    """Get global metrics instance."""
    return _metrics


def reset_metrics():
    """Reset all metrics."""
    global _metrics
    _metrics = LLMMetrics()


# ==================== Callback Handler ====================

class HealthcareTracingHandler(BaseCallbackHandler):
    """
    Custom callback handler for tracing LLM operations.
    
    Captures:
    - Request/response timing
    - Token usage
    - Errors
    - Chain execution flow
    """
    
    def __init__(
        self,
        log_prompts: bool = False,
        log_responses: bool = False,
        metrics: Optional[LLMMetrics] = None
    ):
        self.log_prompts = log_prompts
        self.log_responses = log_responses
        self.metrics = metrics or get_metrics()
        
        self._request_start: Dict[str, float] = {}
        self._chain_stack: List[str] = []
    
    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        **kwargs
    ):
        """Called when LLM starts."""
        run_id = str(kwargs.get("run_id", ""))
        self._request_start[run_id] = time.time()
        
        if self.log_prompts:
            for i, prompt in enumerate(prompts):
                logger.debug(f"Prompt {i}: {prompt[:200]}...")
    
    def on_llm_end(
        self,
        response: LLMResult,
        **kwargs
    ):
        """Called when LLM completes."""
        run_id = str(kwargs.get("run_id", ""))
        start_time = self._request_start.pop(run_id, time.time())
        latency_ms = (time.time() - start_time) * 1000
        
        # Extract token usage
        token_usage = {}
        if response.llm_output:
            token_usage = response.llm_output.get("token_usage", {})
        
        # Record metrics
        self.metrics.record_request(
            model=kwargs.get("model", "unknown"),
            chain=self._chain_stack[-1] if self._chain_stack else "direct",
            tokens=token_usage.get("total_tokens", 0),
            prompt_tokens=token_usage.get("prompt_tokens", 0),
            completion_tokens=token_usage.get("completion_tokens", 0),
            latency_ms=latency_ms
        )
        
        if self.log_responses:
            for gen in response.generations:
                for g in gen:
                    logger.debug(f"Response: {g.text[:200]}...")
    
    def on_llm_error(self, error: Exception, **kwargs):
        """Called on LLM error."""
        self.metrics.record_error()
        logger.error(f"LLM Error: {error}")
    
    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        **kwargs
    ):
        """Called when chain starts."""
        chain_name = serialized.get("name", "unknown")
        self._chain_stack.append(chain_name)
        logger.debug(f"Chain started: {chain_name}")
    
    def on_chain_end(self, outputs: Dict[str, Any], **kwargs):
        """Called when chain completes."""
        if self._chain_stack:
            chain_name = self._chain_stack.pop()
            logger.debug(f"Chain completed: {chain_name}")
    
    def on_chain_error(self, error: Exception, **kwargs):
        """Called on chain error."""
        if self._chain_stack:
            self._chain_stack.pop()
        logger.error(f"Chain Error: {error}")
    
    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        **kwargs
    ):
        """Called when tool starts."""
        tool_name = serialized.get("name", "unknown")
        logger.info(f"Tool called: {tool_name}")
    
    def on_tool_end(self, output: str, **kwargs):
        """Called when tool completes."""
        logger.debug(f"Tool output: {output[:100]}...")
    
    def on_tool_error(self, error: Exception, **kwargs):
        """Called on tool error."""
        logger.error(f"Tool Error: {error}")


# ==================== LangSmith Integration ====================

def setup_langsmith(
    project_name: str = "healthcare-assistant",
    api_key: Optional[str] = None
):
    """
    Set up LangSmith tracing.
    
    Parameters
    ----------
    project_name : str
        LangSmith project name
    api_key : str, optional
        LangSmith API key (or from env)
    """
    api_key = api_key or os.getenv("LANGCHAIN_API_KEY")
    
    if not api_key:
        logger.warning("LangSmith API key not found. Tracing disabled.")
        return False
    
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = project_name
    os.environ["LANGCHAIN_API_KEY"] = api_key
    
    logger.info(f"LangSmith tracing enabled for project: {project_name}")
    return True


def disable_langsmith():
    """Disable LangSmith tracing."""
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    logger.info("LangSmith tracing disabled")


# ==================== Decorators ====================

def trace_llm_call(chain_name: str = "unknown"):
    """
    Decorator to trace LLM calls.
    
    Example
    -------
    >>> @trace_llm_call("explanation")
    ... def explain_prediction(data):
    ...     # LLM call here
    ...     pass
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            metrics = get_metrics()
            
            try:
                result = func(*args, **kwargs)
                
                latency_ms = (time.time() - start_time) * 1000
                logger.info(f"{chain_name} completed in {latency_ms:.0f}ms")
                
                return result
                
            except Exception as e:
                metrics.record_error()
                logger.error(f"{chain_name} failed: {e}")
                raise
        
        return wrapper
    return decorator


def trace_async_llm_call(chain_name: str = "unknown"):
    """Async version of trace_llm_call."""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            metrics = get_metrics()
            
            try:
                result = await func(*args, **kwargs)
                
                latency_ms = (time.time() - start_time) * 1000
                logger.info(f"{chain_name} completed in {latency_ms:.0f}ms")
                
                return result
                
            except Exception as e:
                metrics.record_error()
                logger.error(f"{chain_name} failed: {e}")
                raise
        
        return wrapper
    return decorator


# ==================== Request Logger ====================

class RequestLogger:
    """
    Log LLM requests for debugging and analysis.
    
    Saves requests to file for later analysis.
    """
    
    def __init__(self, log_dir: str = "logs/llm"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self._log_file = os.path.join(
            log_dir, 
            f"requests_{datetime.now().strftime('%Y%m%d')}.jsonl"
        )
    
    def log_request(
        self,
        chain: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log a request."""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "chain": chain,
            "inputs": self._sanitize(inputs),
            "outputs": self._sanitize(outputs),
            "metadata": metadata or {}
        }
        
        with open(self._log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")
    
    def _sanitize(self, data: Any) -> Any:
        """Sanitize data for logging (remove sensitive info)."""
        if isinstance(data, dict):
            return {
                k: self._sanitize(v) 
                for k, v in data.items()
                if k.lower() not in ["api_key", "password", "token"]
            }
        elif isinstance(data, list):
            return [self._sanitize(item) for item in data]
        elif isinstance(data, str) and len(data) > 1000:
            return data[:1000] + "..."
        return data