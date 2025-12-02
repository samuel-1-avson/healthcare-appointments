# src/llm/evaluation/hallucination.py
"""
Hallucination Detection
=======================
Detect when LLM generates information not grounded in context.

Methods:
- Claim extraction and verification
- Semantic entailment checking
- Source attribution verification
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import re

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from ..langchain_config import get_chat_model


logger = logging.getLogger(__name__)


@dataclass
class HallucinationResult:
    """Result of hallucination detection."""
    
    has_hallucination: bool
    confidence: float
    issues: List[Dict[str, Any]]
    claims_checked: int
    claims_verified: int
    claims_unsupported: int
    details: Dict[str, Any]


class HallucinationDetector:
    """
    Detect hallucinations in LLM responses.
    
    Uses multiple methods:
    1. Claim extraction: Break answer into verifiable claims
    2. Context verification: Check if each claim is supported
    3. Semantic entailment: Use LLM to verify support
    
    Example
    -------
    >>> detector = HallucinationDetector()
    >>> result = detector.detect(
    ...     question="What is the cancellation policy?",
    ...     answer="Patients must cancel 24 hours ahead. Fee is $100.",
    ...     contexts=["Cancel 24 hours in advance. First cancellation is free."]
    ... )
    >>> print(result.has_hallucination)  # True (fee amount is wrong)
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        confidence_threshold: float = 0.7
    ):
        """
        Initialize the detector.
        
        Parameters
        ----------
        model_name : str, optional
            LLM model for verification
        confidence_threshold : float
            Minimum confidence for claim support
        """
        self.model = get_chat_model(model_name, temperature=0)
        self.confidence_threshold = confidence_threshold
        
        self._claim_extractor = self._build_claim_extractor()
        self._claim_verifier = self._build_claim_verifier()
        
        logger.info("HallucinationDetector initialized")
    
    def _build_claim_extractor(self):
        """Build chain to extract claims from answer."""
        prompt = PromptTemplate.from_template("""
Extract factual claims from this answer. Each claim should be a single, verifiable statement.

Answer: {answer}

List each claim on a new line, numbered:
1. [first claim]
2. [second claim]
...

Only include factual claims that can be verified. Skip opinions or hedged statements.

Claims:
""")
        return prompt | self.model | StrOutputParser()
    
    def _build_claim_verifier(self):
        """Build chain to verify claims against context."""
        prompt = PromptTemplate.from_template("""
Determine if the following claim is supported by the provided context.

Claim: {claim}

Context:
{context}

Analyze step by step:
1. Does the context contain information related to this claim?
2. Does the context explicitly or implicitly support this claim?
3. Is there any contradiction?

Respond with:
- SUPPORTED: if the context clearly supports the claim
- PARTIALLY_SUPPORTED: if the context partially supports but misses details
- NOT_SUPPORTED: if the context does not contain supporting information
- CONTRADICTED: if the context contradicts the claim

Your response (SUPPORTED/PARTIALLY_SUPPORTED/NOT_SUPPORTED/CONTRADICTED) followed by a brief explanation:
""")
        return prompt | self.model | StrOutputParser()
    
    def detect(
        self,
        question: str,
        answer: str,
        contexts: List[str]
    ) -> HallucinationResult:
        """
        Detect hallucinations in an answer.
        
        Parameters
        ----------
        question : str
            Original question
        answer : str
            Generated answer
        contexts : List[str]
            Retrieved context chunks
        
        Returns
        -------
        HallucinationResult
            Detection results
        """
        # Extract claims
        claims = self._extract_claims(answer)
        
        if not claims:
            return HallucinationResult(
                has_hallucination=False,
                confidence=1.0,
                issues=[],
                claims_checked=0,
                claims_verified=0,
                claims_unsupported=0,
                details={"message": "No verifiable claims found"}
            )
        
        # Combine contexts
        combined_context = "\n\n".join(contexts)
        
        # Verify each claim
        issues = []
        verified = 0
        unsupported = 0
        
        for claim in claims:
            verification = self._verify_claim(claim, combined_context)
            
            if verification["status"] in ["SUPPORTED", "PARTIALLY_SUPPORTED"]:
                verified += 1
            else:
                unsupported += 1
                issues.append({
                    "claim": claim,
                    "status": verification["status"],
                    "explanation": verification["explanation"]
                })
        
        # Calculate overall result
        has_hallucination = unsupported > 0
        confidence = verified / len(claims) if claims else 1.0
        
        return HallucinationResult(
            has_hallucination=has_hallucination,
            confidence=confidence,
            issues=issues,
            claims_checked=len(claims),
            claims_verified=verified,
            claims_unsupported=unsupported,
            details={
                "claims": claims,
                "contexts_used": len(contexts)
            }
        )
    
    def _extract_claims(self, answer: str) -> List[str]:
        """Extract claims from answer."""
        try:
            result = self._claim_extractor.invoke({"answer": answer})
            
            # Parse numbered claims
            claims = []
            for line in result.strip().split("\n"):
                # Remove numbering and clean
                clean = re.sub(r'^\d+[\.\)]\s*', '', line).strip()
                if clean and len(clean) > 10:  # Skip very short claims
                    claims.append(clean)
            
            return claims
            
        except Exception as e:
            logger.warning(f"Claim extraction failed: {e}")
            # Fallback: split by sentences
            sentences = re.split(r'[.!?]+', answer)
            return [s.strip() for s in sentences if len(s.strip()) > 10]
    
    def _verify_claim(self, claim: str, context: str) -> Dict[str, str]:
        """Verify a single claim against context."""
        try:
            result = self._claim_verifier.invoke({
                "claim": claim,
                "context": context[:3000]  # Limit context length
            })
            
            # Parse response
            result = result.strip()
            
            for status in ["SUPPORTED", "PARTIALLY_SUPPORTED", "NOT_SUPPORTED", "CONTRADICTED"]:
                if status in result.upper():
                    # Extract explanation
                    parts = result.split(":", 1)
                    explanation = parts[1].strip() if len(parts) > 1 else ""
                    return {"status": status, "explanation": explanation}
            
            # Default to unsupported if can't parse
            return {"status": "NOT_SUPPORTED", "explanation": result}
            
        except Exception as e:
            logger.warning(f"Claim verification failed: {e}")
            return {"status": "UNKNOWN", "explanation": str(e)}
    
    def batch_detect(
        self,
        samples: List[Dict[str, Any]]
    ) -> List[HallucinationResult]:
        """
        Detect hallucinations in multiple samples.
        
        Parameters
        ----------
        samples : List[Dict]
            List of {question, answer, contexts} dicts
        
        Returns
        -------
        List[HallucinationResult]
            Results for each sample
        """
        results = []
        
        for sample in samples:
            result = self.detect(
                question=sample["question"],
                answer=sample["answer"],
                contexts=sample.get("contexts", [])
            )
            results.append(result)
        
        return results
    
    def get_summary(self, results: List[HallucinationResult]) -> Dict[str, Any]:
        """Get summary statistics from multiple results."""
        total = len(results)
        with_hallucination = sum(1 for r in results if r.has_hallucination)
        
        total_claims = sum(r.claims_checked for r in results)
        unsupported_claims = sum(r.claims_unsupported for r in results)
        
        return {
            "total_samples": total,
            "samples_with_hallucination": with_hallucination,
            "hallucination_rate": with_hallucination / total if total else 0,
            "total_claims_checked": total_claims,
            "unsupported_claims": unsupported_claims,
            "claim_support_rate": 1 - (unsupported_claims / total_claims) if total_claims else 1,
            "avg_confidence": sum(r.confidence for r in results) / total if total else 0
        }


# ==================== Convenience Functions ====================

def detect_hallucinations(
    question: str,
    answer: str,
    contexts: List[str]
) -> Dict[str, Any]:
    """
    Convenience function for hallucination detection.
    
    Returns dict with detection results.
    """
    detector = HallucinationDetector()
    result = detector.detect(question, answer, contexts)
    
    return {
        "has_hallucination": result.has_hallucination,
        "confidence": result.confidence,
        "issues": result.issues,
        "claims_checked": result.claims_checked,
        "claims_verified": result.claims_verified,
        "claims_unsupported": result.claims_unsupported
    }