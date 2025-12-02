# src/llm/chains/explanation_chain.py
"""
Risk Explanation Chain
======================
Chain for generating human-readable explanations of no-show predictions.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from pydantic import BaseModel, Field

from .base import BaseHealthcareChain


# ==================== Output Schema ====================

class RiskExplanationOutput(BaseModel):
    """Structured output for risk explanation."""
    
    summary: str = Field(description="2-3 sentence summary of the risk")
    risk_factors: List[str] = Field(description="Top contributing risk factors")
    protective_factors: List[str] = Field(description="Factors reducing risk")
    recommendations: List[str] = Field(description="Recommended actions")
    confidence_note: str = Field(description="Note about prediction confidence")


# ==================== Prompts ====================

EXPLANATION_SYSTEM_PROMPT = """You are a Healthcare Appointment Risk Analyst. Your job is to explain ML model predictions about patient no-shows in clear, actionable language for healthcare staff.

Guidelines:
- Use plain language, avoid jargon
- Be specific about risk factors
- Focus on actionable recommendations
- Acknowledge uncertainty appropriately
- Never blame patients; focus on circumstances

Risk Tier Interpretation:
- CRITICAL (>80%): Immediate intervention required
- HIGH (60-80%): Proactive outreach recommended
- MEDIUM (40-60%): Enhanced monitoring
- LOW (20-40%): Standard process
- MINIMAL (<20%): Reliable patient"""


EXPLANATION_USER_PROMPT = """Explain this no-show prediction for healthcare staff.

## Prediction Results
- **Risk Tier**: {risk_tier}
- **No-Show Probability**: {probability:.1%}
- **Confidence Level**: {confidence}

## Patient Information
- Age: {age} years
- Gender: {gender}
- Appointment Lead Time: {lead_days} days
- SMS Reminder: {sms_status}
- Day of Week: {appointment_day}

## Health Conditions
- Hypertension: {has_hypertension}
- Diabetes: {has_diabetes}
- Scholarship/Welfare: {has_scholarship}

## History
{history_summary}

## Model's Top Factors
{factors_summary}

---

Provide:
1. A clear 2-3 sentence summary of the risk level
2. The top 3 factors increasing no-show risk
3. Any protective factors reducing risk
4. 2-3 specific, actionable recommendations
5. A brief note on prediction confidence"""


# ==================== Chain ====================

class RiskExplanationChain(BaseHealthcareChain):
    """
    Chain for generating risk explanations.
    
    Takes prediction results and patient data, produces
    human-readable explanations for healthcare staff.
    
    Example
    -------
    >>> chain = RiskExplanationChain()
    >>> result = chain.invoke({
    ...     "probability": 0.72,
    ...     "risk_tier": "HIGH",
    ...     "patient_data": {"age": 35, "lead_days": 14, ...}
    ... })
    >>> print(result["output"])
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        temperature: float = 0.3,
        include_structured_output: bool = False,
        **kwargs
    ):
        self.include_structured_output = include_structured_output
        super().__init__(model_name, temperature, **kwargs)
    
    def _get_system_prompt(self) -> str:
        return EXPLANATION_SYSTEM_PROMPT
    
    def _build_chain(self):
        """Build the explanation chain using LCEL."""
        
        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", EXPLANATION_SYSTEM_PROMPT),
            ("human", EXPLANATION_USER_PROMPT)
        ])
        
        # Build chain with LCEL (LangChain Expression Language)
        chain = (
            # Preprocess inputs
            RunnableLambda(self._preprocess_inputs)
            # Apply prompt
            | prompt
            # Call model
            | self.model
            # Parse output
            | StrOutputParser()
        )
        
        return chain
    
    def _preprocess_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess inputs for the prompt."""
        patient_data = inputs.get("patient_data", {})
        model_factors = inputs.get("model_factors", {})
        history = inputs.get("patient_history", {})
        
        # Format SMS status
        sms_status = "Enabled ✓" if patient_data.get("sms_received", 0) else "Not enabled ✗"
        
        # Format history summary
        if history:
            history_lines = []
            if "total_appointments" in history:
                history_lines.append(f"- Previous appointments: {history['total_appointments']}")
            if "noshow_rate" in history:
                history_lines.append(f"- Historical no-show rate: {history['noshow_rate']:.1%}")
            if history.get("is_first"):
                history_lines.append("- First-time patient")
            history_summary = "\n".join(history_lines) if history_lines else "No history available"
        else:
            history_summary = "No history available"
        
        # Format factors summary
        if model_factors:
            factor_lines = []
            for factor in model_factors.get("risk_factors", [])[:3]:
                factor_lines.append(f"- {factor.get('name', 'Unknown')}: {factor.get('impact', 'N/A')}")
            factors_summary = "\n".join(factor_lines) if factor_lines else "Factor details not available"
        else:
            factors_summary = "Factor details not available"
        
        return {
            "risk_tier": inputs.get("risk_tier", "UNKNOWN"),
            "probability": inputs.get("probability", 0),
            "confidence": inputs.get("confidence", "Unknown"),
            "age": patient_data.get("age", "Unknown"),
            "gender": patient_data.get("gender", "Unknown"),
            "lead_days": patient_data.get("lead_days", "Unknown"),
            "sms_status": sms_status,
            "appointment_day": patient_data.get("appointment_weekday", "Unknown"),
            "has_hypertension": "Yes" if patient_data.get("hypertension") else "No",
            "has_diabetes": "Yes" if patient_data.get("diabetes") else "No",
            "has_scholarship": "Yes" if patient_data.get("scholarship") else "No",
            "history_summary": history_summary,
            "factors_summary": factors_summary
        }
    
    def explain(
        self,
        probability: float,
        risk_tier: str,
        patient_data: Dict[str, Any],
        model_factors: Optional[Dict[str, Any]] = None,
        patient_history: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate an explanation for a prediction.
        
        Convenience method with explicit parameters.
        
        Parameters
        ----------
        probability : float
            No-show probability (0-1)
        risk_tier : str
            Risk tier (CRITICAL, HIGH, MEDIUM, LOW, MINIMAL)
        patient_data : dict
            Patient and appointment information
        model_factors : dict, optional
            Feature importance information
        patient_history : dict, optional
            Patient's appointment history
        
        Returns
        -------
        str
            Human-readable explanation
        """
        result = self.invoke({
            "probability": probability,
            "risk_tier": risk_tier,
            "confidence": self._get_confidence(probability),
            "patient_data": patient_data,
            "model_factors": model_factors or {},
            "patient_history": patient_history or {}
        })
        
        return result["output"]
    
    def _get_confidence(self, probability: float) -> str:
        """Determine confidence level from probability."""
        if probability < 0.2 or probability > 0.8:
            return "High"
        elif probability < 0.35 or probability > 0.65:
            return "Moderate"
        else:
            return "Low"