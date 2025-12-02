# src/llm/evaluation/regression.py
"""
Regression Testing
==================
Prevent quality degradation across model/prompt updates.

Features:
- Baseline comparison
- Output similarity checking
- Metric tracking over time
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json
import hashlib

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from ..langchain_config import get_chat_model


logger = logging.getLogger(__name__)


@dataclass
class RegressionTest:
    """A single regression test case."""
    
    id: str
    question: str
    baseline_answer: str
    required_elements: List[str]  # Must be in answer
    forbidden_elements: List[str]  # Must NOT be in answer
    min_similarity: float = 0.7
    category: str = "general"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            "baseline_answer": self.baseline_answer,
            "required_elements": self.required_elements,
            "forbidden_elements": self.forbidden_elements,
            "min_similarity": self.min_similarity,
            "category": self.category
        }


@dataclass
class RegressionResult:
    """Result from a regression test."""
    
    test: RegressionTest
    passed: bool
    current_answer: str
    similarity_score: float
    missing_elements: List[str]
    forbidden_found: List[str]
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_id": self.test.id,
            "passed": self.passed,
            "similarity_score": self.similarity_score,
            "missing_elements": self.missing_elements,
            "forbidden_found": self.forbidden_found,
            "baseline_preview": self.test.baseline_answer[:100],
            "current_preview": self.current_answer[:100]
        }


class RegressionTestSuite:
    """
    Regression test suite for LLM applications.
    
    Compares current outputs against baseline to detect degradation.
    
    Example
    -------
    >>> suite = RegressionTestSuite()
    >>> suite.load_baseline("baselines/v1.json")
    >>> results = suite.run(rag_chain)
    >>> if not results["passed"]:
    ...     print("Regression detected!")
    """
    
    def __init__(
        self,
        baseline: Optional[Dict[str, Any]] = None,
        tolerance: float = 0.05
    ):
        """
        Initialize regression suite.
        
        Parameters
        ----------
        baseline : dict, optional
            Baseline data
        tolerance : float
            Allowed score degradation (0.05 = 5%)
        """
        self.tests: List[RegressionTest] = []
        self.results: List[RegressionResult] = []
        self.tolerance = tolerance
        self.baseline_metadata = {}
        
        # For semantic similarity
        self.model = get_chat_model(temperature=0)
        
        if baseline:
            self._load_from_dict(baseline)
        
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def _load_from_dict(self, data: Dict[str, Any]):
        """Load tests from dictionary."""
        self.baseline_metadata = data.get("metadata", {})
        
        for test_data in data.get("tests", []):
            self.tests.append(RegressionTest(
                id=test_data["id"],
                question=test_data["question"],
                baseline_answer=test_data["baseline_answer"],
                required_elements=test_data.get("required_elements", []),
                forbidden_elements=test_data.get("forbidden_elements", []),
                min_similarity=test_data.get("min_similarity", 0.7),
                category=test_data.get("category", "general")
            ))
    
    def load_baseline(self, filepath: str):
        """Load baseline from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self._load_from_dict(data)
        self.logger.info(f"Loaded {len(self.tests)} regression tests from {filepath}")
    
    def add_test(
        self,
        question: str,
        baseline_answer: str,
        required_elements: Optional[List[str]] = None,
        forbidden_elements: Optional[List[str]] = None,
        min_similarity: float = 0.7,
        category: str = "general"
    ):
        """Add a regression test."""
        test_id = hashlib.md5(question.encode()).hexdigest()[:8]
        
        self.tests.append(RegressionTest(
            id=test_id,
            question=question,
            baseline_answer=baseline_answer,
            required_elements=required_elements or [],
            forbidden_elements=forbidden_elements or [],
            min_similarity=min_similarity,
            category=category
        ))
    
    def create_baseline(self, target, questions: List[str]) -> Dict[str, Any]:
        """
        Create a baseline from current system output.
        
        Parameters
        ----------
        target : RAGChain
            System to baseline
        questions : List[str]
            Questions to include
        
        Returns
        -------
        dict
            Baseline data
        """
        tests = []
        
        for question in questions:
            try:
                result = target.ask(question)
                answer = result.get("answer", str(result))
                
                test_id = hashlib.md5(question.encode()).hexdigest()[:8]
                
                tests.append({
                    "id": test_id,
                    "question": question,
                    "baseline_answer": answer,
                    "required_elements": [],
                    "forbidden_elements": [],
                    "min_similarity": 0.7,
                    "category": "general"
                })
                
            except Exception as e:
                self.logger.warning(f"Failed to get baseline for: {question[:50]}... Error: {e}")
        
        baseline = {
            "metadata": {
                "created_at": datetime.utcnow().isoformat(),
                "test_count": len(tests),
                "version": "1.0.0"
            },
            "tests": tests
        }
        
        return baseline
    
    def save_baseline(self, baseline: Dict[str, Any], filepath: str):
        """Save baseline to file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(baseline, f, indent=2)
        
        self.logger.info(f"Baseline saved to {filepath}")
    
    def run(self, target) -> Dict[str, Any]:
        """
        Run regression tests.
        
        Parameters
        ----------
        target : RAGChain
            System to test
        
        Returns
        -------
        dict
            Test results
        """
        self.results = []
        
        self.logger.info(f"Running {len(self.tests)} regression tests...")
        
        for test in self.tests:
            try:
                result = self._run_single_test(target, test)
                self.results.append(result)
                
                status = "✅" if result.passed else "❌"
                self.logger.debug(f"{status} {test.id}: similarity={result.similarity_score:.2f}")
                
            except Exception as e:
                self.logger.error(f"Test {test.id} failed: {e}")
                self.results.append(RegressionResult(
                    test=test,
                    passed=False,
                    current_answer=f"Error: {str(e)}",
                    similarity_score=0.0,
                    missing_elements=[],
                    forbidden_found=[],
                    details={"error": str(e)}
                ))
        
        return self._generate_summary()
    
    def _run_single_test(self, target, test: RegressionTest) -> RegressionResult:
        """Run a single regression test."""
        # Get current answer
        result = target.ask(test.question)
        current_answer = result.get("answer", str(result))
        
        # Calculate similarity
        similarity = self._calculate_similarity(test.baseline_answer, current_answer)
        
        # Check required elements
        missing = []
        current_lower = current_answer.lower()
        for element in test.required_elements:
            if element.lower() not in current_lower:
                missing.append(element)
        
        # Check forbidden elements
        forbidden_found = []
        for element in test.forbidden_elements:
            if element.lower() in current_lower:
                forbidden_found.append(element)
        
        # Determine pass/fail
        passed = (
            similarity >= test.min_similarity - self.tolerance and
            len(missing) == 0 and
            len(forbidden_found) == 0
        )
        
        return RegressionResult(
            test=test,
            passed=passed,
            current_answer=current_answer,
            similarity_score=similarity,
            missing_elements=missing,
            forbidden_found=forbidden_found
        )
    
    def _calculate_similarity(self, baseline: str, current: str) -> float:
        """Calculate semantic similarity between baseline and current."""
        # Simple word overlap similarity
        baseline_words = set(baseline.lower().split())
        current_words = set(current.lower().split())
        
        if not baseline_words or not current_words:
            return 0.0
        
        intersection = len(baseline_words & current_words)
        union = len(baseline_words | current_words)
        
        jaccard = intersection / union if union else 0
        
        # Also consider coverage of baseline
        coverage = intersection / len(baseline_words) if baseline_words else 0
        
        # Weighted combination
        return 0.4 * jaccard + 0.6 * coverage
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate test results summary."""
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        
        avg_similarity = sum(r.similarity_score for r in self.results) / total if total else 0
        
        # Group by category
        by_category = {}
        for result in self.results:
            cat = result.test.category
            if cat not in by_category:
                by_category[cat] = {"passed": 0, "failed": 0, "avg_similarity": 0}
            
            if result.passed:
                by_category[cat]["passed"] += 1
            else:
                by_category[cat]["failed"] += 1
        
        # Failed tests
        failed_tests = [
            {
                "id": r.test.id,
                "question": r.test.question[:50] + "...",
                "similarity": r.similarity_score,
                "missing": r.missing_elements,
                "forbidden_found": r.forbidden_found
            }
            for r in self.results if not r.passed
        ]
        
        return {
            "passed": passed == total,
            "total_tests": total,
            "tests_passed": passed,
            "tests_failed": total - passed,
            "pass_rate": passed / total if total else 0,
            "similarity_score": avg_similarity,
            "by_category": by_category,
            "failed_tests": failed_tests,
            "baseline_version": self.baseline_metadata.get("version", "unknown"),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def save_results(self, filepath: str):
        """Save test results."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "summary": self._generate_summary(),
            "detailed_results": [r.to_dict() for r in self.results]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        self.logger.info(f"Regression results saved to {filepath}")


# ==================== Convenience Functions ====================

def run_regression_tests(
    target,
    baseline_path: str,
    tolerance: float = 0.05
) -> Dict[str, Any]:
    """
    Run regression tests.
    
    Parameters
    ----------
    target : RAGChain
        System to test
    baseline_path : str
        Path to baseline file
    tolerance : float
        Allowed degradation
    
    Returns
    -------
    dict
        Test results
    """
    suite = RegressionTestSuite(tolerance=tolerance)
    suite.load_baseline(baseline_path)
    return suite.run(target)


def create_baseline(
    target,
    questions: List[str],
    output_path: str
) -> Dict[str, Any]:
    """
    Create a baseline from current system.
    
    Parameters
    ----------
    target : RAGChain
        System to baseline
    questions : List[str]
        Test questions
    output_path : str
        Where to save baseline
    
    Returns
    -------
    dict
        Created baseline
    """
    suite = RegressionTestSuite()
    baseline = suite.create_baseline(target, questions)
    suite.save_baseline(baseline, output_path)
    return baseline