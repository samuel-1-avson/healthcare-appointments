# src/llm/tools/explanation_tool.py
"""
Explanation Tool
================
Tool for generating natural language explanations of predictions.
"""

import logging
from typing import Optional, Type, Dict, Any

from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun

from ..client import get_llm_client
from ..prompts import RiskExplainerPrompt

logger = logging.getLogger(__name__)


class ExplanationInput(BaseModel):
    """Input for explanation tool."""
    
    probability: float = Field(description="No-show probability (0-1)")
    risk_tier: str = Field(description="Risk tier: MINIMAL, LOW, MEDIUM, HIGH, CRITICAL")
    age: int = Field(description="Patient age")
    gender: str = Field(description="Patient gender")
    lead_days: int = Field(description="Days until appointment")
    sms_received: int = Field(default=0, description="SMS sent (0/1)")
    previous_noshows: Optional[int] = Field(default=None, description="Previous no-shows")
    explanation_style: str = Field(
        default="detailed",
        description="Style: 'detailed', 'concise', or 'patient-friendly'"
    )


class ExplanationTool(BaseTool):
    """
    Tool for generating human-readable prediction explanations.
    
    Uses LLM to translate ML predictions into actionable insights
    for healthcare staff.
    """
    
    name: str = "explain_prediction"
    description: str = """
    Generates a human-readable explanation of a no-show prediction.
    
    Use this tool when you need to:
    - Explain why a patient is high/low risk
    - Provide context for intervention decisions
    - Communicate risk factors to staff
    
    Input: prediction details (probability, risk_tier, patient info)
    Output: Natural language explanation with actionable recommendations
    """
    args_schema: Type[BaseModel] = ExplanationInput
    
    def _run(
        self,
        probability: float,
        risk_tier: str,
        age: int,
        gender: str,
        lead_days: int,
        sms_received: int = 0,
        previous_noshows: Optional[int] = None,
        explanation_style: str = "detailed",
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Generate explanation."""
        try:
            # Build prompt
            explainer = RiskExplainerPrompt(
                include_examples=True,
                explanation_style=explanation_style
            )
            
            patient_data = {
                "age": age,
                "gender": gender,
                "lead_days": lead_days,
                "sms_received": sms_received
            }
            
            patient_history = {}
            if previous_noshows is not None:
                patient_history["noshow_count"] = previous_noshows
            
            system_prompt, user_prompt = explainer.build(
                probability=probability,
                risk_tier=risk_tier,
                patient_data=patient_data,
                patient_history=patient_history
            )
            
            # Get LLM response
            client = get_llm_client()
            response = client.complete(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.3
            )
            
            return response.content
            
        except Exception as e:
            logger.error(f"Explanation generation failed: {e}")
            return self._fallback_explanation(probability, risk_tier, lead_days)
    
    async def _arun(self, **kwargs) -> str:
        """Async version."""
        # For now, use sync version
        return self._run(**kwargs)
    
    def _fallback_explanation(
        self,
        probability: float,
        risk_tier: str,
        lead_days: int
    ) -> str:
        """Simple fallback explanation."""
        explanations = {
            "CRITICAL": f"This appointment has CRITICAL risk ({probability:.0%} no-show probability). Immediate intervention recommended.",
            "HIGH": f"This appointment has HIGH risk ({probability:.0%} no-show probability). Proactive outreach advised.",
            "MEDIUM": f"This appointment has MEDIUM risk ({probability:.0%} no-show probability). Enhanced reminders suggested.",
            "LOW": f"This appointment has LOW risk ({probability:.0%} no-show probability). Standard reminders sufficient.",
            "MINIMAL": f"This appointment has MINIMAL risk ({probability:.0%} no-show probability). Patient is reliable."
        }
        
        base = explanations.get(risk_tier, f"Risk: {risk_tier}, Probability: {probability:.0%}")
        
        if lead_days > 14:
            base += f" Note: {lead_days}-day lead time increases risk."
        
        return base


_explanation_tool: Optional[ExplanationTool] = None


def get_explanation_tool() -> ExplanationTool:
    """Get or create explanation tool singleton."""
    global _explanation_tool
    if _explanation_tool is None:
        _explanation_tool = ExplanationTool()
    return _explanation_tool