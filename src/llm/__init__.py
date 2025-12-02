# src/llm/__init__.py
"""
LLM Module for Healthcare Appointment Assistant
===============================================

This module provides:
- LLM client wrappers (OpenAI, Anthropic)
- Prompt templates for healthcare domain
- Prediction explanation generation
- Policy Q&A capabilities

Week 8-9: Prompt Engineering
Week 10: LangChain Integration
Week 11: RAG Pipeline
Week 12: Evaluation & Safety
"""

from .config import LLMConfig, get_llm_config
from .client import LLMClient, get_llm_client
from .prompts import (
    RiskExplainerPrompt,
    InterventionPrompt,
    PolicyQAPrompt,
    PromptTemplate
)

__all__ = [
    "LLMConfig",
    "get_llm_config",
    "LLMClient", 
    "get_llm_client",
    "RiskExplainerPrompt",
    "InterventionPrompt",
    "PolicyQAPrompt",
    "PromptTemplate"
]