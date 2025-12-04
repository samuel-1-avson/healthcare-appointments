# src/llm/evaluation/__init__.py
"""
LLM Evaluation Module
=====================

Comprehensive evaluation for LLM applications:
- RAG quality metrics (Ragas)
- Hallucination detection
- Safety testing
- Regression testing
- Custom metrics
"""

from .metrics import (
    EvaluationMetrics,
    MetricResult,
    calculate_metrics
)
from .ragas_eval import (
    RagasEvaluator
)
from .hallucination import (
    HallucinationDetector,
    detect_hallucinations
)
from .safety import (
    SafetyEvaluator,
    RedTeamTest,
    run_safety_tests
)
from .regression import (
    RegressionTestSuite,
    RegressionTest,
    run_regression_tests
)
from .framework import (
    EvaluationFramework,
    EvaluationConfig,
    EvaluationReport
)

__all__ = [
    "EvaluationMetrics",
    "MetricResult",
    "calculate_metrics",
    "RagasEvaluator",
    "HallucinationDetector",
    "detect_hallucinations",
    "SafetyEvaluator",
    "RedTeamTest",
    "run_safety_tests",
    "RegressionTestSuite",
    "RegressionTest",
    "run_regression_tests",
    "EvaluationFramework",
    "EvaluationConfig",
    "EvaluationReport"
]