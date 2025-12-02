# src/llm/langchain_config.py
"""
LangChain Configuration
=======================
Configuration and initialization for LangChain components.
"""

import os
import logging
from typing import Optional, Dict, Any, Literal
from functools import lru_cache
from dataclasses import dataclass

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


class LangChainSettings(BaseSettings):
    """
    LangChain configuration settings.
    
    Loads from environment with LANGCHAIN_ prefix.
    """
    
    # API Keys
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, alias="ANTHROPIC_API_KEY")
    
    # LangSmith Tracing
    langsmith_api_key: Optional[str] = Field(default=None, alias="LANGCHAIN_API_KEY")
    langsmith_project: str = Field(default="healthcare-assistant", alias="LANGCHAIN_PROJECT")
    tracing_enabled: bool = Field(default=False, alias="LANGCHAIN_TRACING_V2")
    
    # Model defaults
    default_model: str = Field(default="gpt-4o-mini")
    default_temperature: float = Field(default=0.3)
    default_max_tokens: int = Field(default=1024)
    
    # Timeouts
    request_timeout: int = Field(default=60)
    max_retries: int = Field(default=3)
    
    # Memory settings
    memory_type: Literal["buffer", "summary", "buffer_window"] = Field(default="buffer_window")
    memory_max_messages: int = Field(default=10)
    
    # Caching
    cache_enabled: bool = Field(default=True)
    
    # Prediction API (your existing API)
    prediction_api_url: str = Field(default="http://localhost:8000")
    
    class Config:
        env_file = ".env"
        extra = "ignore"
    
    def setup_environment(self):
        """Set up environment variables for LangChain."""
        if self.openai_api_key:
            os.environ["OPENAI_API_KEY"] = self.openai_api_key
        
        if self.anthropic_api_key:
            os.environ["ANTHROPIC_API_KEY"] = self.anthropic_api_key
        
        if self.langsmith_api_key:
            os.environ["LANGCHAIN_API_KEY"] = self.langsmith_api_key
            os.environ["LANGCHAIN_PROJECT"] = self.langsmith_project
        
        if self.tracing_enabled:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
        
        logger.info("LangChain environment configured")


@lru_cache()
def get_langchain_settings() -> LangChainSettings:
    """Get cached LangChain settings."""
    settings = LangChainSettings()
    settings.setup_environment()
    return settings


# ==================== Model Factories ====================

def get_chat_model(
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
    **kwargs
):
    """
    Get a LangChain chat model.
    
    Automatically selects provider based on model name.
    
    Parameters
    ----------
    model_name : str, optional
        Model name (default: from settings)
    temperature : float, optional
        Sampling temperature
    **kwargs
        Additional model parameters
    
    Returns
    -------
    BaseChatModel
        LangChain chat model instance
    """
    settings = get_langchain_settings()
    
    model_name = model_name or settings.default_model
    temperature = temperature if temperature is not None else settings.default_temperature
    
    # Determine provider from model name
    if model_name.startswith("gpt") or model_name.startswith("o1"):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            max_tokens=kwargs.get("max_tokens", settings.default_max_tokens),
            timeout=settings.request_timeout,
            max_retries=settings.max_retries,
            **kwargs
        )
    
    elif model_name.startswith("claude"):
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=model_name,
            temperature=temperature,
            max_tokens=kwargs.get("max_tokens", settings.default_max_tokens),
            timeout=settings.request_timeout,
            max_retries=settings.max_retries,
            **kwargs
        )
    
    else:
        # Default to OpenAI
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            **kwargs
        )


def get_embeddings_model(model_name: str = "text-embedding-3-small"):
    """
    Get a LangChain embeddings model.
    
    Parameters
    ----------
    model_name : str
        Embeddings model name
    
    Returns
    -------
    Embeddings
        LangChain embeddings instance
    """
    from langchain_openai import OpenAIEmbeddings
    
    return OpenAIEmbeddings(model=model_name)