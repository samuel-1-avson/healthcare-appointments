# src/llm/agents/__init__.py
"""
LangChain Agents Module
=======================

Agents for autonomous healthcare assistant tasks:
- HealthcareAgent: Main conversational agent with tools
- TriageAgent: Batch appointment prioritization
"""

from .healthcare_agent import (
    HealthcareAgent,
    create_healthcare_agent,
    generate_smart_fill
)

__all__ = [
    "HealthcareAgent",
    "create_healthcare_agent",
    "generate_smart_fill"
]