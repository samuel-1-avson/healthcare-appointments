# src/llm/rag/evaluation.py
"""
RAG Evaluation
==============
Evaluate RAG pipeline quality using Ragas and custom metrics.

Metrics:
- Faithfulness: Is the answer grounded in the context?
- Answer Relevancy: Does the answer address the question?
- Context Precision: Are retrieved docs relevant?
- Context Recall: Did we retrieve all needed info?
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


# Try importing ragas
try:
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        answer_correctness
    )
    from datasets import Dataset
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    logger.warning("Ragas not installed. Install with: pip install ragas")


@dataclass
class EvalSample:
    """A single evaluation sample."""
    
    question: str
    answer: str  # Generated answer
    contexts: List[str]  # Retrieved contexts
    ground_truth: Optional[str] = None  # Expected answer
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalResult:
    """Evaluation result for a sample."""
    
    sample_id: int
    question: str
    scores: Dict[str, float]
    passed: bool
    details: Dict[str, Any] = field(default_factory=dict)


class RAGEvaluator:
    """
    Evaluate RAG pipeline quality.
    
    Uses Ragas metrics and custom evaluations.
    
    Example
    -------
    >>> evaluator = RAGEvaluator()
    >>> evaluator.add_sample(question="...", answer="...", contexts=["..."])
    >>> results = evaluator.evaluate()
    >>> print(results["summary"])
    """
    
    def __init__(
        self,
        thresholds: Optional[Dict[str, float]] = None
    ):
        """
        Initialize evaluator.
        
        Parameters
        ----------
        thresholds : dict, optional
            Minimum scores for passing
        """
        self.thresholds = thresholds or {
            "faithfulness": 0.7,
            "answer_relevancy": 0.7,
            "context_precision": 0.6,
            "context_recall": 0.6
        }
        
        self.samples: List[EvalSample] = []
        self.results: List[EvalResult] = []
        
        logger.info("RAGEvaluator initialized")
    
    def add_sample(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
        **metadata
    ):
        """
        Add an evaluation sample.
        
        Parameters
        ----------
        question : str
            The question asked
        answer : str
            Generated answer
        contexts : List[str]
            Retrieved context chunks
        ground_truth : str, optional
            Expected/correct answer
        **metadata
            Additional metadata
        """
        sample = EvalSample(
            question=question,
            answer=answer,
            contexts=contexts,
            ground_truth=ground_truth,
            metadata=metadata
        )
        self.samples.append(sample)
        logger.debug(f"Added sample {len(self.samples)}: {question[:50]}...")
    
    def add_samples_from_chain(
        self,
        rag_chain,
        questions: List[str],
        ground_truths: Optional[List[str]] = None
    ):
        """
        Add samples by running questions through RAG chain.
        
        Parameters
        ----------
        rag_chain : RAGChain
            RAG chain to evaluate
        questions : List[str]
            Questions to ask
        ground_truths : List[str], optional
            Expected answers
        """
        for i, question in enumerate(questions):
            result = rag_chain.ask(question, return_sources=True)
            
            contexts = [
                source.get("content", "")
                for source in result.get("sources", [])
            ]
            
            ground_truth = ground_truths[i] if ground_truths else None
            
            self.add_sample(
                question=question,
                answer=result["answer"],
                contexts=contexts,
                ground_truth=ground_truth,
                latency_ms=result.get("metadata", {}).get("latency_ms")
            )
    
    def evaluate(self) -> Dict[str, Any]:
        """
        Run evaluation on all samples.
        
        Returns
        -------
        dict
            Evaluation results with summary
        """
        if not self.samples:
            raise ValueError("No samples to evaluate")
        
        logger.info(f"Evaluating {len(self.samples)} samples...")
        
        if RAGAS_AVAILABLE:
            return self._evaluate_with_ragas()
        else:
            return self._evaluate_basic()
    
    def _evaluate_with_ragas(self) -> Dict[str, Any]:
        """Evaluate using Ragas library."""
        # Prepare data for Ragas
        data = {
            "question": [s.question for s in self.samples],
            "answer": [s.answer for s in self.samples],
            "contexts": [s.contexts for s in self.samples],
        }
        
        # Add ground truth if available
        if all(s.ground_truth for s in self.samples):
            data["ground_truth"] = [s.ground_truth for s in self.samples]
            metrics = [
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
                answer_correctness
            ]
        else:
            metrics = [
                faithfulness,
                answer_relevancy,
                context_precision
            ]
        
        dataset = Dataset.from_dict(data)
        
        # Run evaluation
        try:
            results = evaluate(
                dataset,
                metrics=metrics
            )
            
            # Convert to dict
            scores = results.to_pandas().to_dict('records')
            
            # Calculate summary
            summary = {}
            for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
                if metric in results:
                    values = [r.get(metric, 0) for r in scores if r.get(metric) is not None]
                    if values:
                        summary[metric] = {
                            "mean": sum(values) / len(values),
                            "min": min(values),
                            "max": max(values)
                        }
            
            # Check thresholds
            passed_count = 0
            for i, score in enumerate(scores):
                passed = all(
                    score.get(m, 0) >= self.thresholds.get(m, 0)
                    for m in self.thresholds
                    if m in score
                )
                if passed:
                    passed_count += 1
                
                self.results.append(EvalResult(
                    sample_id=i,
                    question=self.samples[i].question,
                    scores={k: v for k, v in score.items() if isinstance(v, (int, float))},
                    passed=passed
                ))
            
            return {
                "status": "success",
                "evaluator": "ragas",
                "sample_count": len(self.samples),
                "passed_count": passed_count,
                "pass_rate": passed_count / len(self.samples),
                "summary": summary,
                "detailed_results": scores,
                "thresholds": self.thresholds,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Ragas evaluation failed: {e}")
            return self._evaluate_basic()
    
    def _evaluate_basic(self) -> Dict[str, Any]:
        """Basic evaluation without Ragas."""
        logger.info("Running basic evaluation (Ragas not available)")
        
        results = []
        
        for i, sample in enumerate(self.samples):
            scores = self._calculate_basic_scores(sample)
            
            passed = all(
                scores.get(m, 0) >= self.thresholds.get(m, 0)
                for m in self.thresholds
            )
            
            result = EvalResult(
                sample_id=i,
                question=sample.question,
                scores=scores,
                passed=passed
            )
            results.append(result)
            self.results.append(result)
        
        # Calculate summary
        summary = {}
        for metric in ["answer_length", "context_used", "keyword_overlap"]:
            values = [r.scores.get(metric, 0) for r in results]
            if values:
                summary[metric] = {
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values)
                }
        
        passed_count = sum(1 for r in results if r.passed)
        
        return {
            "status": "success",
            "evaluator": "basic",
            "sample_count": len(self.samples),
            "passed_count": passed_count,
            "pass_rate": passed_count / len(self.samples),
            "summary": summary,
            "detailed_results": [
                {"sample_id": r.sample_id, "scores": r.scores, "passed": r.passed}
                for r in results
            ],
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def _calculate_basic_scores(self, sample: EvalSample) -> Dict[str, float]:
        """Calculate basic scores for a sample."""
        scores = {}
        
        # Answer length (normalized)
        answer_words = len(sample.answer.split())
        scores["answer_length"] = min(answer_words / 100, 1.0)
        
        # Context usage (check if answer words appear in context)
        context_text = " ".join(sample.contexts).lower()
        answer_words_set = set(sample.answer.lower().split())
        context_words_set = set(context_text.split())
        
        if answer_words_set:
            overlap = len(answer_words_set & context_words_set) / len(answer_words_set)
            scores["context_used"] = overlap
        else:
            scores["context_used"] = 0
        
        # Keyword overlap with question
        question_words = set(sample.question.lower().split())
        if question_words:
            q_overlap = len(question_words & answer_words_set) / len(question_words)
            scores["keyword_overlap"] = q_overlap
        else:
            scores["keyword_overlap"] = 0
        
        # Ground truth comparison
        if sample.ground_truth:
            gt_words = set(sample.ground_truth.lower().split())
            if gt_words:
                gt_overlap = len(gt_words & answer_words_set) / len(gt_words)
                scores["ground_truth_overlap"] = gt_overlap
        
        return scores
    
    def save_results(self, filepath: str):
        """Save evaluation results to file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "samples": [
                {
                    "question": s.question,
                    "answer": s.answer,
                    "contexts": s.contexts,
                    "ground_truth": s.ground_truth,
                    "metadata": s.metadata
                }
                for s in self.samples
            ],
            "results": [
                {
                    "sample_id": r.sample_id,
                    "question": r.question,
                    "scores": r.scores,
                    "passed": r.passed
                }
                for r in self.results
            ],
            "thresholds": self.thresholds,
            "saved_at": datetime.utcnow().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved results to {filepath}")
    
    def load_samples(self, filepath: str):
        """Load samples from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        for sample_data in data.get("samples", []):
            self.add_sample(
                question=sample_data["question"],
                answer=sample_data.get("answer", ""),
                contexts=sample_data.get("contexts", []),
                ground_truth=sample_data.get("ground_truth")
            )
        
        logger.info(f"Loaded {len(self.samples)} samples from {filepath}")


# ==================== Golden Dataset ====================

class GoldenDataset:
    """
    Manage a golden evaluation dataset for RAG.
    
    Provides standard questions and expected answers
    for consistent evaluation.
    """
    
    def __init__(self, filepath: Optional[str] = None):
        self.filepath = Path(filepath) if filepath else Path("evals/rag_golden_set.json")
        self.questions: List[Dict[str, Any]] = []
        
        if self.filepath.exists():
            self.load()
    
    def add_question(
        self,
        question: str,
        expected_answer: str,
        category: str = "general",
        difficulty: str = "medium",
        required_sources: Optional[List[str]] = None
    ):
        """Add a question to the golden set."""
        self.questions.append({
            "question": question,
            "expected_answer": expected_answer,
            "category": category,
            "difficulty": difficulty,
            "required_sources": required_sources or []
        })
    
    def save(self):
        """Save golden dataset."""
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.filepath, 'w') as f:
            json.dump({
                "questions": self.questions,
                "metadata": {
                    "count": len(self.questions),
                    "updated_at": datetime.utcnow().isoformat()
                }
            }, f, indent=2)
    
    def load(self):
        """Load golden dataset."""
        with open(self.filepath, 'r') as f:
            data = json.load(f)
        self.questions = data.get("questions", [])
    
    def get_questions(self, category: Optional[str] = None) -> List[Dict]:
        """Get questions, optionally filtered by category."""
        if category:
            return [q for q in self.questions if q["category"] == category]
        return self.questions
    
    def to_eval_format(self) -> Tuple[List[str], List[str]]:
        """Convert to evaluation format (questions, ground_truths)."""
        questions = [q["question"] for q in self.questions]
        ground_truths = [q["expected_answer"] for q in self.questions]
        return questions, ground_truths


def create_healthcare_golden_set() -> GoldenDataset:
    """Create a golden dataset for healthcare policies."""
    dataset = GoldenDataset()
    
    # Cancellation policy questions
    dataset.add_question(
        question="What is the cancellation policy?",
        expected_answer="Patients must provide at least 24 hours notice to cancel without penalty. Cancellations with less than 24 hours notice are considered late cancellations.",
        category="cancellation",
        difficulty="easy"
    )
    
    dataset.add_question(
        question="What happens if I cancel less than 12 hours before my appointment?",
        expected_answer="Cancellations made less than 12 hours before the appointment may be recorded as a no-show equivalent.",
        category="cancellation",
        difficulty="medium"
    )
    
    # No-show policy questions
    dataset.add_question(
        question="What is considered a no-show?",
        expected_answer="A no-show occurs when a patient fails to arrive for their scheduled appointment, arrives more than 15 minutes late without prior notice, or does not cancel or reschedule before the appointment time.",
        category="noshow",
        difficulty="easy"
    )
    
    dataset.add_question(
        question="What are the consequences of multiple no-shows?",
        expected_answer="First no-show: verbal reminder. Second no-show: written warning letter. Third no-show: $75 fee and pre-payment required. Fourth no-show: review by patient relations, possible discharge.",
        category="noshow",
        difficulty="medium"
    )
    
    dataset.add_question(
        question="Can no-show fees be waived?",
        expected_answer="Yes, exceptions may be made for medical emergencies, hospitalization, death in immediate family, severe weather emergencies, or transportation emergencies with documentation. Patients may appeal within 14 days.",
        category="noshow",
        difficulty="hard"
    )
    
    # Reminder questions
    dataset.add_question(
        question="When are appointment reminders sent?",
        expected_answer="Reminders are sent at: 7 days before (email), 48 hours before (SMS), 24 hours before (SMS with confirmation request), and 2 hours before (final SMS).",
        category="reminders",
        difficulty="medium"
    )
    
    # Intervention questions
    dataset.add_question(
        question="What interventions are used for high-risk patients?",
        expected_answer="High-risk patients receive phone calls at 72 hours before, enhanced SMS reminders, transportation assistance offers, barrier assessment, and waitlist backup scheduling.",
        category="intervention",
        difficulty="hard"
    )
    
    dataset.save()
    
    return dataset