# src/llm/ragas_evaluation.py
"""
RAGAS Evaluation for RAG Systems
=================================
Evaluate RAG system quality using RAGAS metrics.

RAGAS (RAG Assessment) measures:
- Faithfulness: Is the answer faithful to the retrieved context?
- Answer Relevance: Is the answer relevant to the question?
- Context Precision: Are the retrieved contexts relevant?
- Context Recall: Did we retrieve all necessary contexts?
"""

import logging
from typing import List, Dict, Any, Optional
from datasets import Dataset

try:
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    logging.warning("RAGAS not available. Install with: pip install ragas")

logger = logging.getLogger(__name__)


class RAGASEvaluator:
    """
    Evaluate RAG system quality using RAGAS metrics.
    """
    
    def __init__(self, llm=None, embeddings=None):
        """
        Initialize RAGAS evaluator.
        
        Parameters
        ----------
        llm : Optional
            LangChain LLM for evaluation
        embeddings : Optional
            LangChain embeddings for evaluation
        """
        if not RAGAS_AVAILABLE:
            logger.error("RAGAS is not installed. Evaluation will not work.")
            return
        
        self.llm = llm
        self.embeddings = embeddings
        self.metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ]
    
    def create_evaluation_dataset(
        self,
        test_cases: List[Dict[str, Any]]
    ) -> Dataset:
        """
        Create a RAGAS-compatible evaluation dataset.
        
        Parameters
        ----------
        test_cases : List[Dict]
            List of test cases with:
            - question: str
            - answer: str
            - contexts: List[str]
            - ground_truth: str (optional)
            
        Returns
        -------
        Dataset
            HuggingFace dataset for RAGAS evaluation
        """
        data = {
            "question": [],
            "answer": [],
            "contexts": [],
            "ground_truth": []
        }
        
        for case in test_cases:
            data["question"].append(case["question"])
            data["answer"].append(case["answer"])
            data["contexts"].append(case["contexts"])
            data["ground_truth"].append(case.get("ground_truth", ""))
        
        return Dataset.from_dict(data)
    
    def evaluate(
        self,
        test_cases: List[Dict[str, Any]],
        metrics: Optional[List] = None
    ) -> Dict[str, float]:
        """
        Evaluate RAG system on test cases.
        
        Parameters
        ----------
        test_cases : List[Dict]
            Test cases to evaluate
        metrics : List, optional
            Specific metrics to use (default: all metrics)
            
        Returns
        -------
        dict
            Evaluation results with metric scores
        """
        if not RAGAS_AVAILABLE:
            logger.error("RAGAS not available")
            return {}
        
        # Create dataset
        dataset = self.create_evaluation_dataset(test_cases)
        
        # Use specified metrics or default
        eval_metrics = metrics or self.metrics
        
        try:
            # Run evaluation
            result = evaluate(
                dataset,
                metrics=eval_metrics,
                llm=self.llm,
                embeddings=self.embeddings
            )
            
            logger.info(f"RAGAS Evaluation Results: {result}")
            
            return result
            
        except Exception as e:
            logger.error(f"RAGAS evaluation failed: {e}", exc_info=True)
            return {}
    
    def evaluate_single(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Evaluate a single RAG response.
        
        Parameters
        ----------
        question : str
            User question
        answer : str
            RAG system answer
        contexts : List[str]
            Retrieved contexts
        ground_truth : str, optional
            Expected answer
            
        Returns
        -------
        dict
            Metric scores
        """
        test_case = [{
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": ground_truth or ""
        }]
        
        return self.evaluate(test_case)


def create_healthcare_test_cases() -> List[Dict[str, Any]]:
    """
    Create sample test cases for healthcare no-show domain.
    
    Returns
    -------
    List[Dict]
        Test cases
    """
    return [
        {
            "question": "What factors contribute to patient no-shows?",
            "answer": "Based on the data, key factors include: longer wait times (lead days), lack of SMS reminders, and certain health conditions like diabetes and hypertension.",
            "contexts": [
                "Patients with lead times greater than 7 days have a 30% higher no-show rate.",
                "SMS reminders reduce no-show rates by 20%.",
                "Patients with chronic conditions (diabetes, hypertension) show 15% higher no-show rates."
            ],
            "ground_truth": "Lead time, SMS reminders, and chronic health conditions are the main factors affecting no-show rates."
        },
        {
            "question": "How effective are SMS reminders?",
            "answer": "SMS reminders are highly effective, reducing no-show rates by approximately 20% according to our data analysis.",
            "contexts": [
                "Analysis shows SMS reminders reduce no-shows by 20%.",
                "Patients who receive SMS reminders are more likely to attend appointments.",
                "Cost of SMS reminders is offset by reduced no-show losses."
            ],
            "ground_truth": "SMS reminders reduce no-show rates by about 20%."
        },
        {
            "question": "What is the impact of appointment lead time?",
            "answer": "Longer lead times significantly increase no-show probability. Appointments scheduled more than a week in advance have 30% higher no-show rates.",
            "contexts": [
                "Lead time is the most significant predictor of no-shows.",
                "Appointments with >7 days lead time show 30% increase in no-shows.",
                "Optimal lead time is 2-5 days for best attendance."
            ],
            "ground_truth": "Longer lead times (>7 days) increase no-show rates by 30%."
        }
    ]


def run_evaluation_report() -> Dict[str, Any]:
    """
    Run comprehensive RAGAS evaluation and generate report.
    
    Returns
    -------
    dict
        Evaluation report with scores and analysis
    """
    if not RAGAS_AVAILABLE:
        return {
            "status": "error",
            "message": "RAGAS not available"
        }
    
    logger.info("Running RAGAS evaluation...")
    
    # Create evaluator
    evaluator = RAGASEvaluator()
    
    # Get test cases
    test_cases = create_healthcare_test_cases()
    
    # Run evaluation
    results = evaluator.evaluate(test_cases)
    
    # Create report
    report = {
        "status": "success",
        "test_cases_count": len(test_cases),
        "metrics": results,
        "analysis": _analyze_results(results)
    }
    
    logger.info(f"Evaluation complete: {report}")
    
    return report


def _analyze_results(results: Dict[str, float]) -> Dict[str, str]:
    """
    Analyze RAGAS results and provide recommendations.
    
    Parameters
    ----------
    results : dict
        RAGAS metric scores
        
    Returns
    -------
    dict
        Analysis and recommendations
    """
    analysis = {}
    
    # Faithfulness analysis
    if "faithfulness" in results:
        score = results["faithfulness"]
        if score < 0.7:
            analysis["faithfulness"] = "LOW - Answers may not be faithful to retrieved contexts. Review context retrieval."
        elif score < 0.85:
            analysis["faithfulness"] = "MEDIUM - Some improvement needed in answer generation."
        else:
            analysis["faithfulness"] = "GOOD - Answers are faithful to contexts."
    
    # Answer relevancy analysis
    if "answer_relevancy" in results:
        score = results["answer_relevancy"]
        if score < 0.7:
            analysis["answer_relevancy"] = "LOW - Answers may not directly address questions. Improve prompts."
        elif score < 0.85:
            analysis["answer_relevancy"] = "MEDIUM - Answers could be more focused."
        else:
            analysis["answer_relevancy"] = "GOOD - Answers are highly relevant."
    
    # Context precision analysis
    if "context_precision" in results:
        score = results["context_precision"]
        if score < 0.7:
            analysis["context_precision"] = "LOW - Retrieved contexts contain noise. Improve retrieval."
        elif score < 0.85:
            analysis["context_precision"] = "MEDIUM - Some irrelevant contexts retrieved."
        else:
            analysis["context_precision"] = "GOOD - Retrieved contexts are precise."
    
    # Context recall analysis
    if "context_recall" in results:
        score = results["context_recall"]
        if score < 0.7:
            analysis["context_recall"] = "LOW - Missing important contexts. Increase retrieval k or improve embeddings."
        elif score < 0.85:
            analysis["context_recall"] = "MEDIUM - Some relevant contexts may be missed."
        else:
            analysis["context_recall"] = "GOOD - All relevant contexts retrieved."
    
    return analysis


# Global evaluator instance
_evaluator: Optional['RAGASEvaluator'] = None


def get_ragas_evaluator() -> 'RAGASEvaluator':
    """Get global RAGAS evaluator instance."""
    global _evaluator
    if _evaluator is None:
        _evaluator = RAGASEvaluator()
    return _evaluator
