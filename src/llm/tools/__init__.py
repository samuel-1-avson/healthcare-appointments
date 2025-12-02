# src/llm/tools/__init__.py
"""
LangChain Tools Module
======================

Custom tools for the Healthcare Assistant:
- Prediction Tool: Call the ML prediction API
- Patient Lookup Tool: Get patient history
- Schedule Tool: Check appointment availability
"""

from .prediction_tool import (
    PredictionTool,
    create_prediction_tool,
    create_batch_prediction_tool
)

__all__ = [
    "PredictionTool",
    "create_prediction_tool",
    "create_batch_prediction_tool"
]