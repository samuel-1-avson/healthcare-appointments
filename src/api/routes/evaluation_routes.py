# src/api/routes/evaluation_routes.py
"""
Evaluation API Routes
=====================
FastAPI endpoints for LLM evaluation.
"""

import logging
from typing import Optional, List, Dict, Any
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, Body, BackgroundTasks
from pydantic import BaseModel, Field

from ...llm.evaluation import (
    EvaluationFramework,
    EvaluationConfig,
    RagasEvaluator,
    HallucinationDetector,
    SafetyEvaluator,
    RegressionTestSuite,
    EvaluationMetrics
)
from ...llm.rag.chains import RAGChain
from ...llm.rag.vector_store import get_vector_store
from ..auth import get_current_admin_user, User


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/evaluate", tags=["Evaluation"])


# ==================== Schemas ====================

class EvaluationRequest(BaseModel):
    """Request for running evaluation."""
    evaluation_types: List[str] = Field(
        default=["rag_quality", "hallucination", "safety"],
        description="Types of evaluation to run"
    )
    use_golden_set: bool = Field(default=True)
    custom_questions: Optional[List[str]] = Field(default=None)


class HallucinationCheckRequest(BaseModel):
    """Request for hallucination check."""
    question: str
    answer: str
    contexts: List[str]
    
    
class SafetyTestRequest(BaseModel):
    """Request for safety testing."""
    categories: Optional[List[str]] = Field(default=None)
    custom_tests: Optional[List[Dict]] = Field(default=None)


class MetricsRequest(BaseModel):
    """Request for metrics calculation."""
    question: str
    answer: str
    ground_truth: Optional[str] = None
    contexts: Optional[List[str]] = None
    metrics: Optional[List[str]] = None


# ==================== State ====================

_rag_chain: Optional[RAGChain] = None


def get_rag_chain() -> RAGChain:
    """Get RAG chain for evaluation."""
    global _rag_chain
    
    if _rag_chain is None:
        vector_store = get_vector_store()
        if vector_store.vector_store is None:
            raise HTTPException(
                status_code=503,
                detail="Vector store not initialized"
            )
        _rag_chain = RAGChain(vector_store)
    
    return _rag_chain


# ==================== Endpoints ====================

@router.post(
    "/full",
    summary="Run Full Evaluation",
    description="Run comprehensive evaluation suite"
)
async def run_full_evaluation(
    request: EvaluationRequest,
    background_tasks: BackgroundTasks,
    user: User = Depends(get_current_admin_user)
) -> Dict[str, Any]:
    """
    Run full evaluation suite on the RAG system.
    
    Includes:
    - RAG quality metrics (faithfulness, relevancy, etc.)
    - Hallucination detection
    - Safety testing
    """
    from ...llm.evaluation.framework import EvaluationType
    
    try:
        rag_chain = get_rag_chain()
        
        # Create config
        config = EvaluationConfig(
            evaluation_types=[EvaluationType(t) for t in request.evaluation_types]
        )
        
        # Create framework
        framework = EvaluationFramework(config)
        framework.register_rag_chain(rag_chain)
        
        # Load golden set
        if request.use_golden_set:
            from ...llm.rag.evaluation import create_healthcare_golden_set
            golden = create_healthcare_golden_set()
            framework._golden_set = golden
        
        # Run evaluation
        report = framework.run_full_evaluation()
        
        return {
            "status": "completed",
            "overall_passed": report.overall_passed,
            "summary": report.summary,
            "recommendations": report.recommendations,
            "report_id": report.run_id
        }
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/rag-quality",
    summary="Evaluate RAG Quality",
    description="Run RAG quality metrics using Ragas"
)
async def evaluate_rag_quality(
    questions: Optional[List[str]] = Body(default=None),
    ground_truths: Optional[List[str]] = Body(default=None),
    use_golden_set: bool = Query(default=True),
    user: User = Depends(get_current_admin_user)
) -> Dict[str, Any]:
    """
    Evaluate RAG quality metrics.
    
    Metrics:
    - Faithfulness
    - Answer Relevancy
    - Context Precision
    - Context Recall (if ground truths provided)
    """
    try:
        rag_chain = get_rag_chain()
        evaluator = RagasEvaluator()
        
        if use_golden_set and not questions:
            from ...llm.rag.evaluation import create_healthcare_golden_set
            golden = create_healthcare_golden_set()
            questions, ground_truths = golden.to_eval_format()
        
        if not questions:
            raise HTTPException(status_code=400, detail="No questions provided")
        
        # Limit for demo
        questions = questions[:10]
        if ground_truths:
            ground_truths = ground_truths[:10]
        
        evaluator.add_samples_from_chain(rag_chain, questions, ground_truths)
        results = evaluator.evaluate()
        
        return results
        
    except Exception as e:
        logger.error(f"RAG quality evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/hallucination",
    summary="Check for Hallucinations",
    description="Detect hallucinations in a response"
)
async def check_hallucination(
    request: HallucinationCheckRequest,
    user: User = Depends(get_current_admin_user)
) -> Dict[str, Any]:
    """
    Check if a response contains hallucinations.
    
    Extracts claims from the answer and verifies
    each against the provided context.
    """
    try:
        detector = HallucinationDetector()
        result = detector.detect(
            question=request.question,
            answer=request.answer,
            contexts=request.contexts
        )
        
        return {
            "has_hallucination": result.has_hallucination,
            "confidence": result.confidence,
            "claims_checked": result.claims_checked,
            "claims_verified": result.claims_verified,
            "claims_unsupported": result.claims_unsupported,
            "issues": result.issues
        }
        
    except Exception as e:
        logger.error(f"Hallucination check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/safety",
    summary="Run Safety Tests",
    description="Run red team safety tests"
)
async def run_safety_tests(
    request: SafetyTestRequest,
    user: User = Depends(get_current_admin_user)
) -> Dict[str, Any]:
    """
    Run safety and red team tests.
    
    Categories:
    - prompt_injection
    - jailbreak
    - medical_advice
    - data_leakage
    - policy_violation
    """
    try:
        rag_chain = get_rag_chain()
        
        evaluator = SafetyEvaluator()
        evaluator.load_default_tests()
        
        if request.custom_tests:
            for test in request.custom_tests:
                evaluator.add_test(**test)
        
        categories = None
        if request.categories:
            from ...llm.evaluation.safety import SafetyCategory
            categories = [SafetyCategory(c) for c in request.categories]
        
        results = evaluator.run_tests(rag_chain, categories=categories)
        
        return results
        
    except Exception as e:
        logger.error(f"Safety testing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/metrics",
    summary="Calculate Metrics",
    description="Calculate custom evaluation metrics"
)
async def calculate_metrics(
    request: MetricsRequest,
    user: User = Depends(get_current_admin_user)
) -> Dict[str, Any]:
    """
    Calculate evaluation metrics for a response.
    
    Available metrics:
    - keyword_coverage
    - answer_completeness
    - no_medical_advice
    - no_pii_disclosure
    - readability
    - conciseness
    - actionability
    """
    try:
        metrics = EvaluationMetrics()
        
        if request.metrics:
            results = {}
            for name in request.metrics:
                results[name] = metrics.calculate(
                    name,
                    request.question,
                    request.answer,
                    request.ground_truth,
                    request.contexts
                )
        else:
            results = metrics.calculate_all(
                request.question,
                request.answer,
                request.ground_truth,
                request.contexts
            )
        
        return {
            "results": {name: r.to_dict() for name, r in results.items()},
            "summary": metrics.get_summary(results)
        }
        
    except Exception as e:
        logger.error(f"Metrics calculation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/regression/baseline",
    summary="Create Regression Baseline",
    description="Create a baseline for regression testing"
)
async def create_regression_baseline(
    questions: List[str] = Body(..., description="Questions for baseline"),
    user: User = Depends(get_current_admin_user)
) -> Dict[str, Any]:
    """
    Create a regression testing baseline.
    
    Captures current responses to serve as reference
    for future regression testing.
    """
    try:
        rag_chain = get_rag_chain()
        
        suite = RegressionTestSuite()
        baseline = suite.create_baseline(rag_chain, questions)
        
        # Save baseline
        from pathlib import Path
        baseline_path = Path("evals/baselines") / f"baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        suite.save_baseline(baseline, str(baseline_path))
        
        return {
            "status": "created",
            "path": str(baseline_path),
            "test_count": len(baseline["tests"]),
            "metadata": baseline["metadata"]
        }
        
    except Exception as e:
        logger.error(f"Baseline creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/regression/run",
    summary="Run Regression Tests",
    description="Run regression tests against baseline"
)
async def run_regression_tests(
    baseline_path: str = Query(..., description="Path to baseline file"),
    user: User = Depends(get_current_admin_user)
) -> Dict[str, Any]:
    """
    Run regression tests against a baseline.
    
    Compares current responses to baseline and
    reports any degradation.
    """
    try:
        rag_chain = get_rag_chain()
        
        suite = RegressionTestSuite()
        suite.load_baseline(baseline_path)
        
        results = suite.run(rag_chain)
        
        return results
        
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Baseline not found")
    except Exception as e:
        logger.error(f"Regression testing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/health",
    summary="Evaluation Health",
    description="Get evaluation system health"
)
async def evaluation_health() -> Dict[str, Any]:
    """Get health status of evaluation system."""
    try:
        rag_chain = get_rag_chain()
        initialized = True
    except:
        initialized = False
    
    return {
        "status": "healthy" if initialized else "not_initialized",
        "rag_chain_ready": initialized,
        "timestamp": datetime.utcnow().isoformat()
    }