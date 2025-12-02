# src/llm/client.py
"""
LLM Client Wrapper
==================
Unified interface for interacting with LLM providers.

Features:
- Multi-provider support (OpenAI, Anthropic)
- Automatic retry with exponential backoff
- Token counting and cost tracking
- Response caching
- Structured output parsing
"""

import logging
import time
import json
import hashlib
from typing import Optional, Dict, Any, List, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from abc import ABC, abstractmethod

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)

from .config import LLMConfig, get_llm_config, LLMModelConfig


logger = logging.getLogger(__name__)


# ==================== Data Classes ====================

@dataclass
class TokenUsage:
    """Token usage statistics."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    
    @property
    def cost(self) -> float:
        """Calculate cost (requires model config)."""
        return 0.0  # Override in subclass with pricing


@dataclass
class LLMResponse:
    """Standardized LLM response."""
    content: str
    model: str
    provider: str
    usage: TokenUsage
    finish_reason: Optional[str] = None
    latency_ms: float = 0.0
    cached: bool = False
    timestamp: datetime = field(default_factory=datetime.utcnow)
    raw_response: Optional[Dict] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "content": self.content,
            "model": self.model,
            "provider": self.provider,
            "usage": {
                "prompt_tokens": self.usage.prompt_tokens,
                "completion_tokens": self.usage.completion_tokens,
                "total_tokens": self.usage.total_tokens
            },
            "finish_reason": self.finish_reason,
            "latency_ms": self.latency_ms,
            "cached": self.cached,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class Message:
    """Chat message."""
    role: str  # "system", "user", "assistant"
    content: str
    
    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}


# ==================== Provider Clients ====================

class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def complete(
        self,
        messages: List[Message],
        model_config: LLMModelConfig,
        **kwargs
    ) -> LLMResponse:
        """Generate a completion."""
        pass
    
    @abstractmethod
    def count_tokens(self, text: str, model: str) -> int:
        """Count tokens in text."""
        pass


class OpenAIProvider(BaseLLMProvider):
    """OpenAI API provider."""
    
    def __init__(self, api_key: str):
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
            self._available = True
        except ImportError:
            logger.warning("OpenAI package not installed")
            self._available = False
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI: {e}")
            self._available = False
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((Exception,))
    )
    def complete(
        self,
        messages: List[Message],
        model_config: LLMModelConfig,
        **kwargs
    ) -> LLMResponse:
        """Generate completion using OpenAI."""
        if not self._available:
            raise RuntimeError("OpenAI client not available")
        
        start_time = time.time()
        
        response = self.client.chat.completions.create(
            model=model_config.name,
            messages=[m.to_dict() for m in messages],
            temperature=kwargs.get("temperature", model_config.temperature),
            max_tokens=kwargs.get("max_tokens", model_config.max_tokens),
            top_p=kwargs.get("top_p", model_config.top_p)
        )
        
        latency_ms = (time.time() - start_time) * 1000
        
        usage = TokenUsage(
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens
        )
        
        return LLMResponse(
            content=response.choices[0].message.content,
            model=model_config.name,
            provider="openai",
            usage=usage,
            finish_reason=response.choices[0].finish_reason,
            latency_ms=latency_ms,
            raw_response=response.model_dump()
        )
    
    def count_tokens(self, text: str, model: str) -> int:
        """Count tokens using tiktoken."""
        try:
            import tiktoken
            encoding = tiktoken.encoding_for_model(model)
            return len(encoding.encode(text))
        except Exception:
            # Rough estimate: 4 chars per token
            return len(text) // 4


class AnthropicProvider(BaseLLMProvider):
    """Anthropic API provider."""
    
    def __init__(self, api_key: str):
        try:
            from anthropic import Anthropic
            self.client = Anthropic(api_key=api_key)
            self._available = True
        except ImportError:
            logger.warning("Anthropic package not installed")
            self._available = False
        except Exception as e:
            logger.error(f"Failed to initialize Anthropic: {e}")
            self._available = False
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    def complete(
        self,
        messages: List[Message],
        model_config: LLMModelConfig,
        **kwargs
    ) -> LLMResponse:
        """Generate completion using Anthropic."""
        if not self._available:
            raise RuntimeError("Anthropic client not available")
        
        start_time = time.time()
        
        # Separate system message
        system_msg = ""
        chat_messages = []
        for msg in messages:
            if msg.role == "system":
                system_msg = msg.content
            else:
                chat_messages.append(msg.to_dict())
        
        response = self.client.messages.create(
            model=model_config.name,
            system=system_msg,
            messages=chat_messages,
            temperature=kwargs.get("temperature", model_config.temperature),
            max_tokens=kwargs.get("max_tokens", model_config.max_tokens)
        )
        
        latency_ms = (time.time() - start_time) * 1000
        
        usage = TokenUsage(
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            total_tokens=response.usage.input_tokens + response.usage.output_tokens
        )
        
        return LLMResponse(
            content=response.content[0].text,
            model=model_config.name,
            provider="anthropic",
            usage=usage,
            finish_reason=response.stop_reason,
            latency_ms=latency_ms,
            raw_response={"id": response.id, "type": response.type}
        )
    
    def count_tokens(self, text: str, model: str) -> int:
        """Estimate tokens (Anthropic doesn't have public tokenizer)."""
        # Rough estimate: 4 chars per token
        return len(text) // 4


# ==================== Main Client ====================

class LLMClient:
    """
    Unified LLM Client.
    
    Provides a single interface for multiple LLM providers with:
    - Automatic provider selection
    - Response caching
    - Usage tracking
    - Error handling
    
    Example
    -------
    >>> client = LLMClient()
    >>> response = client.complete(
    ...     prompt="Explain this prediction...",
    ...     system_prompt="You are a healthcare assistant."
    ... )
    >>> print(response.content)
    """
    
    def __init__(self, config: Optional[LLMConfig] = None):
        """
        Initialize the LLM client.
        
        Parameters
        ----------
        config : LLMConfig, optional
            Configuration object (default: load from environment)
        """
        self.config = config or get_llm_config()
        self.logger = logging.getLogger(__name__)
        
        # Initialize providers
        self._providers: Dict[str, BaseLLMProvider] = {}
        self._init_providers()
        
        # Simple cache (in production, use Redis)
        self._cache: Dict[str, LLMResponse] = {}
        
        # Usage tracking
        self._total_tokens = 0
        self._total_cost = 0.0
        self._request_count = 0
        
        self.logger.info(f"LLMClient initialized with providers: {list(self._providers.keys())}")
    
    def _init_providers(self):
        """Initialize available providers."""
        if self.config.has_openai():
            self._providers["openai"] = OpenAIProvider(self.config.openai_api_key)
            self.logger.info("OpenAI provider initialized")
        
        if self.config.has_anthropic():
            self._providers["anthropic"] = AnthropicProvider(self.config.anthropic_api_key)
            self.logger.info("Anthropic provider initialized")
        
        if not self._providers:
            self.logger.warning("No LLM providers configured!")
    
    def _get_cache_key(self, messages: List[Message], model: str) -> str:
        """Generate cache key from messages."""
        content = json.dumps([m.to_dict() for m in messages]) + model
        return hashlib.md5(content.encode()).hexdigest()
    
    def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        use_cache: bool = True,
        **kwargs
    ) -> LLMResponse:
        """
        Generate a completion.
        
        Parameters
        ----------
        prompt : str
            User prompt
        system_prompt : str, optional
            System prompt for context
        model : str, optional
            Model to use (default: config default)
        temperature : float, optional
            Sampling temperature
        max_tokens : int, optional
            Maximum response tokens
        use_cache : bool
            Whether to use caching
        **kwargs
            Additional model parameters
        
        Returns
        -------
        LLMResponse
            Generated response with metadata
        """
        # Build messages
        messages = []
        if system_prompt:
            messages.append(Message(role="system", content=system_prompt))
        messages.append(Message(role="user", content=prompt))
        
        return self.chat(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            use_cache=use_cache,
            **kwargs
        )
    
    def chat(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        use_cache: bool = True,
        **kwargs
    ) -> LLMResponse:
        """
        Generate a chat completion.
        
        Parameters
        ----------
        messages : List[Message]
            Chat messages
        model : str, optional
            Model to use
        temperature : float, optional
            Sampling temperature
        max_tokens : int, optional
            Maximum response tokens
        use_cache : bool
            Whether to use caching
        
        Returns
        -------
        LLMResponse
            Generated response
        """
        # Get model config
        model_name = model or self.config.default_model
        model_config = self.config.get_model_config(model_name)
        
        # Override with provided values
        if temperature is not None:
            model_config.temperature = temperature
        if max_tokens is not None:
            model_config.max_tokens = max_tokens
        
        # Check cache
        if use_cache and self.config.cache_enabled:
            cache_key = self._get_cache_key(messages, model_name)
            if cache_key in self._cache:
                self.logger.debug("Cache hit")
                cached = self._cache[cache_key]
                cached.cached = True
                return cached
        
        # Get provider
        provider = self._providers.get(model_config.provider)
        if not provider:
            raise RuntimeError(f"Provider {model_config.provider} not available")
        
        # Log prompt if enabled
        if self.config.log_prompts:
            self.logger.info(f"Prompt: {messages[-1].content[:200]}...")
        
        # Generate completion
        try:
            response = provider.complete(messages, model_config, **kwargs)
            
            # Update tracking
            self._total_tokens += response.usage.total_tokens
            self._request_count += 1
            
            # Calculate cost
            cost = (
                (response.usage.prompt_tokens / 1000) * model_config.input_cost_per_1k +
                (response.usage.completion_tokens / 1000) * model_config.output_cost_per_1k
            )
            self._total_cost += cost
            
            # Log response if enabled
            if self.config.log_responses:
                self.logger.info(f"Response ({response.latency_ms:.0f}ms): {response.content[:200]}...")
            
            # Cache result
            if use_cache and self.config.cache_enabled:
                self._cache[cache_key] = response
            
            return response
            
        except Exception as e:
            self.logger.error(f"LLM completion failed: {e}")
            raise
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            "total_requests": self._request_count,
            "total_tokens": self._total_tokens,
            "estimated_cost_usd": round(self._total_cost, 4),
            "cache_size": len(self._cache),
            "available_providers": list(self._providers.keys())
        }
    
    def clear_cache(self):
        """Clear the response cache."""
        self._cache.clear()
        self.logger.info("Cache cleared")
    
    @property
    def is_available(self) -> bool:
        """Check if any provider is available."""
        return len(self._providers) > 0


# Singleton instance
_llm_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    """Get or create the LLM client singleton."""
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client


def reset_llm_client():
    """Reset the LLM client singleton."""
    global _llm_client
    _llm_client = None