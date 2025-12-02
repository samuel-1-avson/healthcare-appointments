"""
LangChain Configuration Compatibility Layer
===========================================
Provides compatibility with existing code expecting langchain_config.
"""

import logging
from typing import Optional, Any
from functools import lru_cache

from .config import get_llm_config, LLMConfig

logger = logging.getLogger(__name__)

def get_langchain_settings() -> LLMConfig:
    """
    Get LLM configuration (compatibility alias).
    """
    return get_llm_config()

def get_chat_model(
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
    streaming: bool = False
) -> Any:
    """
    Get a LangChain chat model instance.
    
    Parameters
    ----------
    temperature : float, optional
        Sampling temperature
    model_name : str, optional
        Model name to use
    streaming : bool
        Whether to enable streaming
        
    Returns
    -------
    BaseChatModel
        LangChain chat model
    """
    config = get_llm_config()
    model = model_name or config.default_model
    model_config = config.get_model_config(model)
    temp = temperature if temperature is not None else model_config.temperature
    
    try:
        if model_config.provider == "openai":
            from langchain_openai import ChatOpenAI
            
            if not config.openai_api_key:
                logger.warning("OpenAI API key not found")
                return None
                
            return ChatOpenAI(
                model=model_config.name,
                temperature=temp,
                openai_api_key=config.openai_api_key,
                streaming=streaming
            )
            
        elif model_config.provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            
            if not config.anthropic_api_key:
                logger.warning("Anthropic API key not found")
                return None
                
            return ChatAnthropic(
                model=model_config.name,
                temperature=temp,
                anthropic_api_key=config.anthropic_api_key,
                streaming=streaming
            )
            
        elif model_config.provider == "local":
            from langchain_community.chat_models import ChatOllama
            
            # Use host.docker.internal for Docker -> Host communication
            base_url = "http://host.docker.internal:11434"
            
            return ChatOllama(
                model=model_config.name,
                temperature=temp,
                base_url=base_url
            )
            
        else:
            logger.warning(f"Unsupported provider: {model_config.provider}")
            return None
            
    except ImportError as e:
        logger.error(f"Failed to import LangChain provider: {e}")
        return None
    except Exception as e:
        logger.error(f"Failed to create chat model: {e}")
        return None
