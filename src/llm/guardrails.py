# src/llm/guardrails.py
"""
LLM Guardrails & Safety
=======================
Output validation, content safety, and prompt injection prevention.
"""

import logging
import json
import re
from typing import Any, Dict, Optional, List

logger = logging.getLogger(__name__)


class OutputValidator:
    """
    Validate LLM outputs for safety, structure, and relevance.
    """
    
    # Patterns that might indicate prompt injection
    INJECTION_PATTERNS = [
        r"ignore\s+(previous|above|all)\s+instructions?",
        r"disregard\s+(previous|above|all)",
        r"forget\s+(everything|all|previous)",
        r"new\s+instructions?:",
        r"system\s*:\s*you\s+are",
        r"<\s*\|im_start\|",
        r"<\s*\|im_end\|",
    ]
    
    # Healthcare-related keywords for topic validation
    HEALTHCARE_KEYWORDS = [
        "appointment", "patient", "doctor", "medical",
        "healthcare", "hospital", "clinic", "health",
        "diagnosis", "treatment", "medication", "symptom",
        "disease", "prescription", "no-show", "visit"
    ]
    
    def __init__(self):
        """Initialize output validator."""
        self.injection_regex = re.compile(
            "|".join(self.INJECTION_PATTERNS),
            re.IGNORECASE
        )
    
    def validate_json(self, text: str) -> tuple[bool, Optional[Dict]]:
        """
        Validate that text is valid JSON.
        
        Parameters
        ----------
        text : str
            Text to validate
            
        Returns
        -------
        tuple[bool, Optional[dict]]
            (is_valid, parsed_json or None)
        """
        try:
            parsed = json.loads(text)
            return True, parsed
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON output: {e}")
            return False, None
    
    def check_prompt_injection(self, text: str) -> tuple[bool, List[str]]:
        """
        Check for potential prompt injection attempts.
        
        Parameters
        ----------
        text : str
            Text to check
            
        Returns
        -------
        tuple[bool, List[str]]
            (injection_detected, matched_patterns)
        """
        matches = self.injection_regex.findall(text.lower())
        
        if matches:
            logger.warning(f"Potential prompt injection detected: {matches}")
            return True, matches
        
        return False, []
    
    def check_topic_relevance(
        self,
        text: str,
        min_keywords: int = 1
    ) -> tuple[bool, int]:
        """
        Check if text is relevant to healthcare domain.
        
        Parameters
        ----------
        text : str
            Text to check
        min_keywords : int
            Minimum healthcare keywords required
            
        Returns
        -------
        tuple[bool, int]
            (is_relevant, keyword_count)
        """
        text_lower = text.lower()
        keyword_count = sum(
            1 for keyword in self.HEALTHCARE_KEYWORDS
            if keyword in text_lower
        )
        
        is_relevant = keyword_count >= min_keywords
        
        if not is_relevant:
            logger.warning(
                f"Low topic relevance: only {keyword_count} healthcare keywords found"
            )
        
        return is_relevant, keyword_count
    
    def check_content_safety(self, text: str) -> tuple[bool, List[str]]:
        """
        Check for potentially harmful content.
        
        Parameters
        ----------
        text : str
            Text to check
            
        Returns
        -------
        tuple[bool, List[str]]
            (is_safe, concerns)
        """
        concerns = []
        text_lower = text.lower()
        
        # Check for medical advice disclaimer
        advice_keywords = ["diagnose", "prescribe", "treat", "cure"]
        if any(keyword in text_lower for keyword in advice_keywords):
            if "not a substitute" not in text_lower and "consult" not in text_lower:
                concerns.append("Medical advice without disclaimer")
        
        # Check for personal data requests
        pii_requests = ["social security", "credit card", "password", "ssn"]
        if any(request in text_lower for request in pii_requests):
            concerns.append("Requests sensitive personal information")
        
        is_safe = len(concerns) == 0
        
        if not is_safe:
            logger.warning(f"Content safety concerns: {concerns}")
        
        return is_safe, concerns
    
    def validate_output(
        self,
        output: str,
        expected_format: str = "text",
        check_safety: bool = True,
        check_relevance: bool = True,
        check_injection: bool = True
    ) -> Dict[str, Any]:
        """
        Comprehensive output validation.
        
        Parameters
        ----------
        output : str
            LLM output to validate
        expected_format : str
            Expected format ("text" or "json")
        check_safety : bool
            Whether to check content safety
        check_relevance : bool
            Whether to check topic relevance
        check_injection : bool
            Whether to check for prompt injection
            
        Returns
        -------
        dict
            Validation results
        """
        results = {
            "valid": True,
            "output": output,
            "issues": []
        }
        
        # Format validation
        if expected_format == "json":
            is_valid_json, parsed = self.validate_json(output)
            results["json_valid"] = is_valid_json
            results["parsed"] = parsed
            
            if not is_valid_json:
                results["valid"] = False
                results["issues"].append("Invalid JSON format")
        
        # Prompt injection check
        if check_injection:
            injection_detected, patterns = self.check_prompt_injection(output)
            results["injection_detected"] = injection_detected
            
            if injection_detected:
                results["valid"] = False
                results["issues"].append(f"Prompt injection detected: {patterns}")
        
        # Topic relevance check
        if check_relevance:
            is_relevant, keyword_count = self.check_topic_relevance(output)
            results["topic_relevant"] = is_relevant
            results["healthcare_keyword_count"] = keyword_count
            
            if not is_relevant:
                results["issues"].append("Low topic relevance to healthcare")
                # Don't mark as invalid, just a warning
        
        # Content safety check
        if check_safety:
            is_safe, concerns = self.check_content_safety(output)
            results["content_safe"] = is_safe
            results["safety_concerns"] = concerns
            
            if not is_safe:
                results["valid"] = False
                results["issues"].extend(concerns)
        
        return results


# Global validator instance
_validator: Optional[OutputValidator] = None


def get_output_validator() -> OutputValidator:
    """Get global output validator instance."""
    global _validator
    if _validator is None:
        _validator = OutputValidator()
    return _validator
