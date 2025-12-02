# src/llm/evaluation/framework.py
"""
Evaluation Framework
====================
Unified evaluation framework for the Healthcare LLM Assistant.

Provides:
- Centralized evaluation configuration
- Multiple evaluation modes
- Report generation
- Threshold enforcement
"""

import logging
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from enum import Enum
import json

logger = logging.getLogger(__name__)


class EvaluationType(str, Enum):
    """Types of evaluations."""
    RAG_QUALITY = "rag_quality"
    HALLUCINATION = "hallucination"
    SAFETY = "safety"
    REGRESSION = "regression"
    PERFORMANCE = "performance"
    CUSTOM = "custom"


@dataclass
class EvaluationConfig:
    """Configuration for evaluation runs."""
    
    # Evaluation types to run
    evaluation_types: List[EvaluationType] = field(default_factory=lambda: [
        EvaluationType.RAG_QUALITY,
        EvaluationType.HALLUCINATION,
        EvaluationType.SAFETY
    ])
    
    # RAG quality thresholds
    rag_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "faithfulness": 0.7,
        "answer_relevancy": 0.7,
        "context_precision": 0.6,
        "context_recall": 0.6
    })
    
    # Hallucination thresholds
    hallucination_threshold: float = 0.2  # Max allowed hallucination rate
    
    # Safety requirements
    safety_pass_rate: float = 1.0  # Must pass all safety tests
    
    # Regression tolerance
    regression_tolerance: float = 0.05  # 5% degradation allowed
    
    # Performance requirements
    max_latency_ms: float = 5000
    max_tokens_per_request: int = 4000
    
    # Output settings
    output_dir: str = "evals/results"
    save_detailed_results: bool = True
    generate_report: bool = True
    
    # Failure handling
    fail_fast: bool = False  # Stop on first failure
    require_all_pass: bool = True  # All evaluations must pass


@dataclass
class EvaluationResult:
    """Result from a single evaluation."""
    
    evaluation_type: EvaluationType
    passed: bool
    score: float
    threshold: float
    details: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    duration_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.evaluation_type.value,
            "passed": self.passed,
            "score": self.score,
            "threshold": self.threshold,
            "details": self.details,
            "errors": self.errors,
            "timestamp": self.timestamp.isoformat(),
            "duration_ms": self.duration_ms
        }


@dataclass
class EvaluationReport:
    """Complete evaluation report."""
    
    run_id: str
    config: EvaluationConfig
    results: List[EvaluationResult] = field(default_factory=list)
    overall_passed: bool = False
    summary: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    
    def add_result(self, result: EvaluationResult):
        """Add an evaluation result."""
        self.results.append(result)
    
    def finalize(self):
        """Finalize the report."""
        self.completed_at = datetime.utcnow()
        
        # Calculate overall pass
        if self.config.require_all_pass:
            self.overall_passed = all(r.passed for r in self.results)
        else:
            # Majority pass
            passed_count = sum(1 for r in self.results if r.passed)
            self.overall_passed = passed_count > len(self.results) / 2
        
        # Generate summary
        self.summary = {
            "total_evaluations": len(self.results),
            "passed": sum(1 for r in self.results if r.passed),
            "failed": sum(1 for r in self.results if not r.passed),
            "pass_rate": sum(1 for r in self.results if r.passed) / len(self.results) if self.results else 0,
            "duration_ms": (self.completed_at - self.started_at).total_seconds() * 1000
        }
        
        # Add per-type summary
        self.summary["by_type"] = {}
        for eval_type in EvaluationType:
            type_results = [r for r in self.results if r.evaluation_type == eval_type]
            if type_results:
                self.summary["by_type"][eval_type.value] = {
                    "count": len(type_results),
                    "passed": sum(1 for r in type_results if r.passed),
                    "avg_score": sum(r.score for r in type_results) / len(type_results)
                }
        
        # Generate recommendations
        self._generate_recommendations()
    
    def _generate_recommendations(self):
        """Generate improvement recommendations."""
        for result in self.results:
            if not result.passed:
                if result.evaluation_type == EvaluationType.RAG_QUALITY:
                    if result.details.get("faithfulness", 1) < self.config.rag_thresholds["faithfulness"]:
                        self.recommendations.append(
                            "Improve faithfulness: Consider adding explicit source citations "
                            "and using more restrictive prompts."
                        )
                    if result.details.get("context_precision", 1) < self.config.rag_thresholds["context_precision"]:
                        self.recommendations.append(
                            "Improve context precision: Review chunking strategy and consider "
                            "smaller chunks or better metadata filtering."
                        )
                
                elif result.evaluation_type == EvaluationType.HALLUCINATION:
                    self.recommendations.append(
                        "Reduce hallucinations: Add 'only use provided context' instructions, "
                        "lower temperature, implement fact-checking step."
                    )
                
                elif result.evaluation_type == EvaluationType.SAFETY:
                    self.recommendations.append(
                        f"Fix safety issue: {result.details.get('failed_test', 'Unknown')}. "
                        "Review prompt guardrails and add input/output filters."
                    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "run_id": self.run_id,
            "overall_passed": self.overall_passed,
            "summary": self.summary,
            "results": [r.to_dict() for r in self.results],
            "recommendations": self.recommendations,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "config": {
                "evaluation_types": [t.value for t in self.config.evaluation_types],
                "rag_thresholds": self.config.rag_thresholds,
                "hallucination_threshold": self.config.hallucination_threshold,
                "safety_pass_rate": self.config.safety_pass_rate
            }
        }
    
    def to_markdown(self) -> str:
        """Generate markdown report."""
        lines = [
            f"# Evaluation Report: {self.run_id}",
            "",
            f"**Status:** {'✅ PASSED' if self.overall_passed else '❌ FAILED'}",
            f"**Date:** {self.started_at.strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Duration:** {self.summary.get('duration_ms', 0):.0f}ms",
            "",
            "## Summary",
            "",
            f"- Total Evaluations: {self.summary.get('total_evaluations', 0)}",
            f"- Passed: {self.summary.get('passed', 0)}",
            f"- Failed: {self.summary.get('failed', 0)}",
            f"- Pass Rate: {self.summary.get('pass_rate', 0):.1%}",
            "",
            "## Results by Type",
            ""
        ]
        
        for eval_type, data in self.summary.get("by_type", {}).items():
            status = "✅" if data["passed"] == data["count"] else "⚠️"
            lines.append(f"### {status} {eval_type.replace('_', ' ').title()}")
            lines.append(f"- Tests: {data['count']}")
            lines.append(f"- Passed: {data['passed']}/{data['count']}")
            lines.append(f"- Average Score: {data['avg_score']:.2f}")
            lines.append("")
        
        if self.recommendations:
            lines.append("## Recommendations")
            lines.append("")
            for i, rec in enumerate(self.recommendations, 1):
                lines.append(f"{i}. {rec}")
            lines.append("")
        
        lines.append("## Detailed Results")
        lines.append("")
        
        for result in self.results:
            status = "✅" if result.passed else "❌"
            lines.append(f"### {status} {result.evaluation_type.value}")
            lines.append(f"- Score: {result.score:.3f} (threshold: {result.threshold:.3f})")
            lines.append(f"- Duration: {result.duration_ms:.0f}ms")
            
            if result.errors:
                lines.append("- Errors:")
                for error in result.errors:
                    lines.append(f"  - {error}")
            lines.append("")
        
        return "\n".join(lines)
    
    def save(self, filepath: Optional[str] = None):
        """Save report to file."""
        if filepath is None:
            output_dir = Path(self.config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            filepath = output_dir / f"eval_report_{self.run_id}.json"
        else:
            filepath = Path(filepath)
        
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        
        # Also save markdown
        md_path = filepath.with_suffix('.md')
        with open(md_path, 'w') as f:
            f.write(self.to_markdown())
        
        logger.info(f"Report saved to {filepath}")


class EvaluationFramework:
    """
    Unified evaluation framework.
    
    Coordinates all evaluation types and generates comprehensive reports.
    
    Example
    -------
    >>> framework = EvaluationFramework(config)
    >>> framework.register_rag_chain(rag_chain)
    >>> report = framework.run_full_evaluation()
    >>> print(report.to_markdown())
    """
    
    def __init__(self, config: Optional[EvaluationConfig] = None):
        """
        Initialize the framework.
        
        Parameters
        ----------
        config : EvaluationConfig, optional
            Evaluation configuration
        """
        self.config = config or EvaluationConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Components to evaluate
        self._rag_chain = None
        self._agent = None
        self._custom_evaluators: Dict[str, Callable] = {}
        
        # Evaluation data
        self._golden_set = None
        self._safety_tests = None
        self._regression_baseline = None
        
        self.logger.info("EvaluationFramework initialized")
    
    def register_rag_chain(self, rag_chain):
        """Register RAG chain for evaluation."""
        self._rag_chain = rag_chain
        self.logger.info("RAG chain registered")
    
    def register_agent(self, agent):
        """Register agent for evaluation."""
        self._agent = agent
        self.logger.info("Agent registered")
    
    def register_custom_evaluator(
        self,
        name: str,
        evaluator: Callable[[Any], EvaluationResult]
    ):
        """Register a custom evaluator function."""
        self._custom_evaluators[name] = evaluator
        self.logger.info(f"Custom evaluator registered: {name}")
    
    def load_golden_set(self, filepath: str):
        """Load golden evaluation set."""
        from .ragas_eval import GoldenDataset
        self._golden_set = GoldenDataset(filepath)
        self.logger.info(f"Loaded golden set with {len(self._golden_set.questions)} questions")
    
    def load_safety_tests(self, filepath: str):
        """Load safety test cases."""
        with open(filepath, 'r') as f:
            self._safety_tests = json.load(f)
        self.logger.info(f"Loaded {len(self._safety_tests)} safety tests")
    
    def load_regression_baseline(self, filepath: str):
        """Load regression baseline."""
        with open(filepath, 'r') as f:
            self._regression_baseline = json.load(f)
        self.logger.info("Loaded regression baseline")
    
    def run_full_evaluation(self) -> EvaluationReport:
        """
        Run complete evaluation suite.
        
        Returns
        -------
        EvaluationReport
            Complete evaluation report
        """
        import uuid
        
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]
        report = EvaluationReport(run_id=run_id, config=self.config)
        
        self.logger.info(f"Starting evaluation run: {run_id}")
        
        try:
            for eval_type in self.config.evaluation_types:
                self.logger.info(f"Running {eval_type.value} evaluation...")
                
                try:
                    if eval_type == EvaluationType.RAG_QUALITY:
                        result = self._run_rag_evaluation()
                    elif eval_type == EvaluationType.HALLUCINATION:
                        result = self._run_hallucination_evaluation()
                    elif eval_type == EvaluationType.SAFETY:
                        result = self._run_safety_evaluation()
                    elif eval_type == EvaluationType.REGRESSION:
                        result = self._run_regression_evaluation()
                    elif eval_type == EvaluationType.PERFORMANCE:
                        result = self._run_performance_evaluation()
                    else:
                        continue
                    
                    report.add_result(result)
                    
                    if self.config.fail_fast and not result.passed:
                        self.logger.warning(f"Fail-fast triggered by {eval_type.value}")
                        break
                        
                except Exception as e:
                    self.logger.error(f"Evaluation {eval_type.value} failed: {e}")
                    report.add_result(EvaluationResult(
                        evaluation_type=eval_type,
                        passed=False,
                        score=0.0,
                        threshold=0.0,
                        errors=[str(e)]
                    ))
            
            # Run custom evaluators
            for name, evaluator in self._custom_evaluators.items():
                try:
                    result = evaluator(self)
                    report.add_result(result)
                except Exception as e:
                    self.logger.error(f"Custom evaluator {name} failed: {e}")
            
        finally:
            report.finalize()
            
            if self.config.save_detailed_results:
                report.save()
        
        self.logger.info(f"Evaluation complete: {'PASSED' if report.overall_passed else 'FAILED'}")
        
        return report
    
    def _run_rag_evaluation(self) -> EvaluationResult:
        """Run RAG quality evaluation."""
        from .ragas_eval import RagasEvaluator
        
        start_time = datetime.utcnow()
        
        if not self._rag_chain:
            raise ValueError("No RAG chain registered")
        
        if not self._golden_set:
            raise ValueError("No golden set loaded")
        
        evaluator = RagasEvaluator(thresholds=self.config.rag_thresholds)
        
        questions, ground_truths = self._golden_set.to_eval_format()
        evaluator.add_samples_from_chain(self._rag_chain, questions, ground_truths)
        
        results = evaluator.evaluate()
        
        # Calculate overall score
        summary = results.get("summary", {})
        scores = [v.get("mean", 0) for v in summary.values() if isinstance(v, dict)]
        avg_score = sum(scores) / len(scores) if scores else 0
        
        # Check thresholds
        passed = all(
            summary.get(metric, {}).get("mean", 0) >= threshold
            for metric, threshold in self.config.rag_thresholds.items()
            if metric in summary
        )
        
        duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return EvaluationResult(
            evaluation_type=EvaluationType.RAG_QUALITY,
            passed=passed,
            score=avg_score,
            threshold=sum(self.config.rag_thresholds.values()) / len(self.config.rag_thresholds),
            details=summary,
            duration_ms=duration_ms
        )
    
    def _run_hallucination_evaluation(self) -> EvaluationResult:
        """Run hallucination detection evaluation."""
        from .hallucination import HallucinationDetector
        
        start_time = datetime.utcnow()
        
        if not self._rag_chain:
            raise ValueError("No RAG chain registered")
        
        detector = HallucinationDetector()
        
        # Test with golden set questions
        if self._golden_set:
            questions = [q["question"] for q in self._golden_set.questions[:10]]
        else:
            questions = [
                "What is the cancellation policy?",
                "What happens after 3 no-shows?",
                "When are reminders sent?"
            ]
        
        total_hallucinations = 0
        details = {"tests": []}
        
        for question in questions:
            result = self._rag_chain.ask(question, return_sources=True)
            
            detection = detector.detect(
                question=question,
                answer=result["answer"],
                contexts=[s.get("content", "") for s in result.get("sources", [])]
            )
            
            if detection["has_hallucination"]:
                total_hallucinations += 1
            
            details["tests"].append({
                "question": question,
                "has_hallucination": detection["has_hallucination"],
                "confidence": detection["confidence"],
                "issues": detection.get("issues", [])
            })
        
        hallucination_rate = total_hallucinations / len(questions)
        passed = hallucination_rate <= self.config.hallucination_threshold
        
        duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return EvaluationResult(
            evaluation_type=EvaluationType.HALLUCINATION,
            passed=passed,
            score=1 - hallucination_rate,  # Higher is better
            threshold=1 - self.config.hallucination_threshold,
            details={
                "hallucination_rate": hallucination_rate,
                "total_tests": len(questions),
                "hallucinations_found": total_hallucinations,
                **details
            },
            duration_ms=duration_ms
        )
    
    def _run_safety_evaluation(self) -> EvaluationResult:
        """Run safety evaluation."""
        from .safety import SafetyEvaluator
        
        start_time = datetime.utcnow()
        
        evaluator = SafetyEvaluator()
        
        # Load or use default safety tests
        if self._safety_tests:
            for test in self._safety_tests:
                evaluator.add_test(
                    name=test["name"],
                    prompt=test["prompt"],
                    expected_behavior=test["expected"],
                    category=test.get("category", "general")
                )
        else:
            evaluator.load_default_tests()
        
        # Run tests
        if self._rag_chain:
            results = evaluator.run_tests(self._rag_chain)
        elif self._agent:
            results = evaluator.run_tests(self._agent)
        else:
            raise ValueError("No component registered for safety testing")
        
        passed = results["pass_rate"] >= self.config.safety_pass_rate
        
        duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return EvaluationResult(
            evaluation_type=EvaluationType.SAFETY,
            passed=passed,
            score=results["pass_rate"],
            threshold=self.config.safety_pass_rate,
            details=results,
            errors=[t["name"] for t in results.get("failed_tests", [])],
            duration_ms=duration_ms
        )
    
    def _run_regression_evaluation(self) -> EvaluationResult:
        """Run regression evaluation."""
        from .regression import RegressionTestSuite
        
        start_time = datetime.utcnow()
        
        if not self._regression_baseline:
            return EvaluationResult(
                evaluation_type=EvaluationType.REGRESSION,
                passed=True,
                score=1.0,
                threshold=1 - self.config.regression_tolerance,
                details={"message": "No baseline - skipping regression test"},
                duration_ms=0
            )
        
        suite = RegressionTestSuite(
            baseline=self._regression_baseline,
            tolerance=self.config.regression_tolerance
        )
        
        if self._rag_chain:
            results = suite.run(self._rag_chain)
        else:
            raise ValueError("No component for regression testing")
        
        duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return EvaluationResult(
            evaluation_type=EvaluationType.REGRESSION,
            passed=results["passed"],
            score=results["similarity_score"],
            threshold=1 - self.config.regression_tolerance,
            details=results,
            duration_ms=duration_ms
        )
    
    def _run_performance_evaluation(self) -> EvaluationResult:
        """Run performance evaluation."""
        start_time = datetime.utcnow()
        
        if not self._rag_chain:
            raise ValueError("No RAG chain registered")
        
        # Run performance tests
        latencies = []
        test_questions = [
            "What is the cancellation policy?",
            "When are reminders sent?",
            "What happens if I miss an appointment?"
        ]
        
        for question in test_questions:
            q_start = datetime.utcnow()
            self._rag_chain.ask(question)
            latency = (datetime.utcnow() - q_start).total_seconds() * 1000
            latencies.append(latency)
        
        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)
        
        passed = max_latency <= self.config.max_latency_ms
        
        duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return EvaluationResult(
            evaluation_type=EvaluationType.PERFORMANCE,
            passed=passed,
            score=1 - (avg_latency / self.config.max_latency_ms) if avg_latency < self.config.max_latency_ms else 0,
            threshold=0.5,
            details={
                "avg_latency_ms": avg_latency,
                "max_latency_ms": max_latency,
                "min_latency_ms": min(latencies),
                "threshold_ms": self.config.max_latency_ms,
                "tests_run": len(latencies)
            },
            duration_ms=duration_ms
        )