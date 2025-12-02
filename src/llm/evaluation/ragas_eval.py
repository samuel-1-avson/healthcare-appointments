# src/llm/evaluation/ragas_eval.py
"""
Ragas Evaluation
================
Integration with Ragas library for RAG evaluation.

Metrics:
- Faithfulness: Is the answer grounded in context?
- Answer Relevancy: Does the answer address the question?
- Context Precision: Are retrieved docs relevant?
- Context Recall: Did we get all needed info?
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path

logger = logging.getLogger(__name__)

# Try importing ragas
try:
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        answer_correctness,
        answer_similarity
    )
    from datasets import Dataset
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    logger.warning("Ragas not installed. Install with: pip install ragas datasets")


@dataclass
class EvalSample:
    """A single evaluation sample."""
    
    question: str
    answer: str
    contexts: List[str]
    ground_truth: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class RagasEvaluator:
    """
    Evaluate RAG pipelines using Ragas metrics.
    
    Metrics explained:
    - **Faithfulness**: Measures if claims in answer are supported by context
    - **Answer Relevancy**: Measures if answer addresses the question
    - **Context Precision**: Measures if retrieved chunks are relevant
    - **Context Recall**: Measures if all relevant info was retrieved
    
    Example
    -------
    >>> evaluator = RagasEvaluator()
    >>> evaluator.add_samples_from_chain(rag_chain, questions, ground_truths)
    >>> results = evaluator.evaluate()
    >>> print(results["summary"])
    """
    
    def __init__(
        self,
        thresholds: Optional[Dict[str, float]] = None,
        metrics: Optional[List[str]] = None
    ):
        """
        Initialize the evaluator.
        
        Parameters
        ----------
        thresholds : dict, optional
            Minimum scores for passing each metric
        metrics : list, optional
            Which metrics to compute
        """
        self.thresholds = thresholds or {
            "faithfulness": 0.7,
            "answer_relevancy": 0.7,
            "context_precision": 0.6,
            "context_recall": 0.6
        }
        
        self.metric_names = metrics or list(self.thresholds.keys())
        self.samples: List[EvalSample] = []
        self.results: Optional[Dict] = None
        
        if not RAGAS_AVAILABLE:
            logger.warning("Ragas not available. Using fallback evaluation.")
        
        logger.info(f"RagasEvaluator initialized with metrics: {self.metric_names}")
    
    def add_sample(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
        **metadata
    ):
        """Add an evaluation sample."""
        self.samples.append(EvalSample(
            question=question,
            answer=answer,
            contexts=contexts,
            ground_truth=ground_truth,
            metadata=metadata
        ))
    
    def add_samples_from_chain(
        self,
        rag_chain,
        questions: List[str],
        ground_truths: Optional[List[str]] = None
    ):
        """
        Add samples by running questions through a RAG chain.
        
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
            try:
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
                
            except Exception as e:
                logger.warning(f"Failed to process question: {question[:50]}... Error: {e}")
    
    def evaluate(self) -> Dict[str, Any]:
        """
        Run evaluation on all samples.
        
        Returns
        -------
        dict
            Evaluation results with scores and summary
        """
        if not self.samples:
            raise ValueError("No samples to evaluate")
        
        logger.info(f"Evaluating {len(self.samples)} samples...")
        
        if RAGAS_AVAILABLE:
            return self._evaluate_with_ragas()
        else:
            return self._evaluate_fallback()
    
    def _evaluate_with_ragas(self) -> Dict[str, Any]:
        """Evaluate using Ragas library."""
        # Prepare dataset
        data = {
            "question": [s.question for s in self.samples],
            "answer": [s.answer for s in self.samples],
            "contexts": [s.contexts for s in self.samples],
        }
        
        # Add ground truth if available
        has_ground_truth = all(s.ground_truth for s in self.samples)
        if has_ground_truth:
            data["ground_truth"] = [s.ground_truth for s in self.samples]
        
        dataset = Dataset.from_dict(data)
        
        # Select metrics
        metrics_to_use = []
        
        if "faithfulness" in self.metric_names:
            metrics_to_use.append(faithfulness)
        if "answer_relevancy" in self.metric_names:
            metrics_to_use.append(answer_relevancy)
        if "context_precision" in self.metric_names:
            metrics_to_use.append(context_precision)
        if has_ground_truth:
            if "context_recall" in self.metric_names:
                metrics_to_use.append(context_recall)
            if "answer_correctness" in self.metric_names:
                metrics_to_use.append(answer_correctness)
        
        # Run evaluation
        try:
            results = evaluate(dataset, metrics=metrics_to_use)
            
            # Convert to pandas and then dict
            df = results.to_pandas()
            scores = df.to_dict('records')
            
            # Calculate summary statistics
            summary = {}
            for metric in self.metric_names:
                if metric in df.columns:
                    values = df[metric].dropna().tolist()
                    if values:
                        summary[metric] = {
                            "mean": sum(values) / len(values),
                            "min": min(values),
                            "max": max(values),
                            "std": self._std(values)
                        }
            
            # Check thresholds
            passed_count = 0
            for score in scores:
                passed = all(
                    score.get(m, 0) >= self.thresholds.get(m, 0)
                    for m in self.thresholds
                    if m in score
                )
                if passed:
                    passed_count += 1
            
            self.results = {
                "status": "success",
                "evaluator": "ragas",
                "sample_count": len(self.samples),
                "passed_count": passed_count,
                "pass_rate": passed_count / len(self.samples),
                "summary": summary,
                "detailed_scores": scores,
                "thresholds": self.thresholds,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            return self.results
            
        except Exception as e:
            logger.error(f"Ragas evaluation failed: {e}")
            return self._evaluate_fallback()
    
    def _evaluate_fallback(self) -> Dict[str, Any]:
        """Fallback evaluation without Ragas."""
        logger.info("Running fallback evaluation")
        
        scores = []
        
        for sample in self.samples:
            score = self._calculate_basic_scores(sample)
            scores.append(score)
        
        # Calculate summary
        summary = {}
        for metric in ["answer_overlap", "context_coverage", "length_score"]:
            values = [s.get(metric, 0) for s in scores]
            if values:
                summary[metric] = {
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values)
                }
        
        passed_count = sum(
            1 for s in scores
            if s.get("answer_overlap", 0) > 0.3 and s.get("context_coverage", 0) > 0.3
        )
        
        return {
            "status": "success",
            "evaluator": "fallback",
            "sample_count": len(self.samples),
            "passed_count": passed_count,
            "pass_rate": passed_count / len(self.samples),
            "summary": summary,
            "detailed_scores": scores,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def _calculate_basic_scores(self, sample: EvalSample) -> Dict[str, float]:
        """Calculate basic scores for a sample."""
        scores = {}
        
        answer_words = set(sample.answer.lower().split())
        context_text = " ".join(sample.contexts).lower()
        context_words = set(context_text.split())
        
        # Context coverage
        if answer_words:
            overlap = len(answer_words & context_words)
            scores["context_coverage"] = overlap / len(answer_words)
        else:
            scores["context_coverage"] = 0
        
        # Answer overlap with ground truth
        if sample.ground_truth:
            gt_words = set(sample.ground_truth.lower().split())
            if gt_words:
                overlap = len(answer_words & gt_words)
                scores["answer_overlap"] = overlap / len(gt_words)
            else:
                scores["answer_overlap"] = 0
        else:
            scores["answer_overlap"] = 0.5  # Neutral if no ground truth
        
        # Length score
        word_count = len(sample.answer.split())
        scores["length_score"] = min(word_count / 50, 1.0)  # Normalize to 50 words
        
        return scores
    
    def _std(self, values: List[float]) -> float:
        """Calculate standard deviation."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5
    
    def save_results(self, filepath: str):
        """Save evaluation results to file."""
        if not self.results:
            raise ValueError("No results to save. Run evaluate() first.")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Results saved to {filepath}")
    
    def get_failed_samples(self) -> List[Dict]:
        """Get samples that failed evaluation."""
        if not self.results:
            return []
        
        failed = []
        scores = self.results.get("detailed_scores", [])
        
        for i, score in enumerate(scores):
            passed = all(
                score.get(m, 0) >= self.thresholds.get(m, 0)
                for m in self.thresholds
                if m in score
            )
            
            if not passed:
                failed.append({
                    "index": i,
                    "question": self.samples[i].question,
                    "scores": score
                })
        
        return failed


class GoldenDataset:
    """
    Manage golden evaluation datasets.
    
    Provides standard questions and expected answers
    for consistent RAG evaluation.
    """
    
    def __init__(self, filepath: Optional[str] = None):
        self.filepath = Path(filepath) if filepath else Path("evals/golden_set.json")
        self.questions: List[Dict[str, Any]] = []
        
        if self.filepath.exists():
            self.load()
    
    def add_question(
        self,
        question: str,
        expected_answer: str,
        category: str = "general",
        difficulty: str = "medium",
        required_keywords: Optional[List[str]] = None,
        forbidden_keywords: Optional[List[str]] = None
    ):
        """Add a question to the golden set."""
        self.questions.append({
            "question": question,
            "expected_answer": expected_answer,
            "category": category,
            "difficulty": difficulty,
            "required_keywords": required_keywords or [],
            "forbidden_keywords": forbidden_keywords or []
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
        
        logger.info(f"Golden set saved with {len(self.questions)} questions")
    
    def load(self):
        """Load golden dataset."""
        with open(self.filepath, 'r') as f:
            data = json.load(f)
        self.questions = data.get("questions", [])
        logger.info(f"Loaded {len(self.questions)} golden questions")
    
    def to_eval_format(self) -> Tuple[List[str], List[str]]:
        """Convert to evaluation format."""
        questions = [q["question"] for q in self.questions]
        ground_truths = [q["expected_answer"] for q in self.questions]
        return questions, ground_truths
    
    def get_by_category(self, category: str) -> List[Dict]:
        """Get questions by category."""
        return [q for q in self.questions if q["category"] == category]
    
    def get_by_difficulty(self, difficulty: str) -> List[Dict]:
        """Get questions by difficulty."""
        return [q for q in self.questions if q["difficulty"] == difficulty]