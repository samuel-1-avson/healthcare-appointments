# src/llm/config.py
"""
LLM Configuration - Updated for LangChain
==========================================
"""

import os
from typing import Optional, Dict, Any, Literal, List
from pathlib import Path
from functools import lru_cache

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings


class LLMModelConfig(BaseModel):
    """Configuration for a specific LLM model."""
    
    name: str = Field(description="Model name/identifier")
    provider: Literal["openai", "anthropic", "local"] = Field(default="openai")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, ge=1)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    input_cost_per_1k: float = Field(default=0.0)
    output_cost_per_1k: float = Field(default=0.0)


class LangChainConfig(BaseModel):
    """LangChain-specific configuration."""
    
    # Tracing
    tracing_enabled: bool = Field(default=False)
    langsmith_api_key: Optional[str] = Field(default=None)
    langsmith_project: str = Field(default="healthcare-assistant")
    
    # Memory
    memory_type: Literal["buffer", "summary", "buffer_window"] = Field(default="buffer_window")
    memory_max_tokens: int = Field(default=2000)
    memory_window_size: int = Field(default=10)  # For buffer_window
    
    # Chains
    chain_verbose: bool = Field(default=False)
    chain_max_iterations: int = Field(default=5)
    
    # Callbacks
    enable_logging_callback: bool = Field(default=True)
    enable_metrics_callback: bool = Field(default=True)
    
    # Caching
    cache_type: Literal["memory", "redis", "none"] = Field(default="memory")
    redis_url: Optional[str] = Field(default=None)


class LLMConfig(BaseSettings):
    """Complete LLM Configuration Settings."""
    
    # API Keys
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, alias="ANTHROPIC_API_KEY")
    
    # Default settings
    default_provider: Literal["openai", "anthropic", "local"] = Field(default="local")
    default_model: str = Field(default="llama3")
    
    # Model configurations
    models: Dict[str, LLMModelConfig] = Field(default_factory=lambda: {
        "gpt-4o-mini": LLMModelConfig(
            name="gpt-4o-mini",
            provider="openai",
            temperature=0.3,
            max_tokens=1024,
            input_cost_per_1k=0.00015,
            output_cost_per_1k=0.0006
        ),
        "gpt-4o": LLMModelConfig(
            name="gpt-4o",
            provider="openai",
            temperature=0.3,
            max_tokens=2048,
            input_cost_per_1k=0.005,
            output_cost_per_1k=0.015
        ),
        "claude-3-haiku": LLMModelConfig(
            name="claude-3-haiku-20240307",
            provider="anthropic",
            temperature=0.3,
            max_tokens=1024,
            input_cost_per_1k=0.00025,
            output_cost_per_1k=0.00125
        ),
        "claude-3-sonnet": LLMModelConfig(
            name="claude-3-5-sonnet-20241022",
            provider="anthropic",
            temperature=0.3,
            max_tokens=2048,
            input_cost_per_1k=0.003,
            output_cost_per_1k=0.015
        ),
        "llama3": LLMModelConfig(
            name="llama3",
            provider="local",
            temperature=0.7,
            max_tokens=2048,
            input_cost_per_1k=0.0,
            output_cost_per_1k=0.0
        )
    })
    
    # LangChain settings
    langchain: LangChainConfig = Field(default_factory=LangChainConfig)
    
    # Rate limiting
    max_retries: int = Field(default=3)
    retry_delay: float = Field(default=1.0)
    timeout: float = Field(default=30.0)
    
    # Logging
    log_prompts: bool = Field(default=False)
    log_responses: bool = Field(default=False)
    
    # Safety
    content_filter_enabled: bool = Field(default=True)
    max_prompt_length: int = Field(default=10000)
    
    # Caching
    cache_enabled: bool = Field(default=True)
    cache_ttl: int = Field(default=3600)
    
    # ML API Integration
    ml_api_base_url: str = Field(default="http://localhost:8000")
    ml_api_timeout: float = Field(default=10.0)
    
    class Config:
        env_prefix = "NOSHOW_LLM_"
        env_file = ".env"
        extra = "ignore"
    
    @field_validator("openai_api_key", "anthropic_api_key", mode="before")
    @classmethod
    def get_from_env(cls, v, info):
        if v is None:
            env_key = info.field_name.upper()
            return os.getenv(env_key)
        return v
    
    def get_model_config(self, model_name: Optional[str] = None) -> LLMModelConfig:
        name = model_name or self.default_model
        if name not in self.models:
            # FORCE LOCAL provider to bypass environment variable corruption
            return LLMModelConfig(name=name, provider="local")
        return self.models[name]
    
    def has_openai(self) -> bool:
        return self.openai_api_key is not None
    
    def has_anthropic(self) -> bool:
        return self.anthropic_api_key is not None


@lru_cache()
def get_llm_config() -> LLMConfig:
    """Get cached LLM configuration."""
    return LLMConfig()


# Risk tier configuration
RISK_TIER_CONFIG = {
    "CRITICAL": {"threshold": 0.8, "color": "#e74c3c", "emoji": "🔴", "priority": 1},
    "HIGH": {"threshold": 0.6, "color": "#e67e22", "emoji": "🟠", "priority": 2},
    "MEDIUM": {"threshold": 0.4, "color": "#f39c12", "emoji": "🟡", "priority": 3},
    "LOW": {"threshold": 0.2, "color": "#2ecc71", "emoji": "🟢", "priority": 4},
    "MINIMAL": {"threshold": 0.0, "color": "#27ae60", "emoji": "✅", "priority": 5}
}