# src/llm/prompts/risk_explainer.py
"""
Risk Explanation Prompts
========================
Prompts for explaining no-show predictions to healthcare staff.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass

from .templates import PromptTemplate, HEALTHCARE_SYSTEM_PROMPT, RISK_EXPLANATION_EXAMPLES


# ==================== Templates ====================

RISK_EXPLANATION_TEMPLATE = PromptTemplate(
    name="risk_explanation",
    version="1.0.0",
    description="Explain a no-show prediction with contributing factors",
    template="""Analyze this no-show prediction and explain it clearly for healthcare staff.

## Prediction Details
- **No-Show Probability**: {probability:.1%}
- **Risk Tier**: {risk_tier}
- **Confidence**: {confidence}

## Patient & Appointment Information
- Age: {age} years old
- Gender: {gender}
- Lead Time: {lead_days} days until appointment
- SMS Reminder: {sms_status}
- Appointment Day: {appointment_day}

## Health Factors
- Hypertension: {hypertension}
- Diabetes: {diabetes}
- Enrolled in Welfare Program: {scholarship}

## Historical Data
{history_section}

## Top Risk/Protective Factors from Model
{factors_section}

---

Please provide:
1. A clear explanation of the risk level (2-3 sentences)
2. The top 3 factors contributing to this prediction
3. Specific, actionable recommendations for this case
4. Any important caveats or considerations

Keep your response focused, practical, and appropriate for healthcare staff."""
)


QUICK_EXPLANATION_TEMPLATE = PromptTemplate(
    name="quick_risk_explanation",
    version="1.0.0",
    description="Brief risk explanation for dashboard display",
    template="""Provide a 2-3 sentence explanation for this no-show prediction:

Risk: {risk_tier} ({probability:.0%} probability)
Key factors: {key_factors}

Be concise and actionable."""
)


PATIENT_COMMUNICATION_TEMPLATE = PromptTemplate(
    name="patient_communication",
    version="1.0.0",
    description="Generate patient-friendly reminder message",
    template="""Create a friendly appointment reminder message for this patient.

Appointment Details:
- Date: {appointment_date}
- Time: {appointment_time}
- Provider: {provider_name}
- Type: {appointment_type}

Patient Context:
- Risk Level: {risk_tier} (don't mention this to patient)
- Preferred Name: {patient_name}

Tone: Warm, supportive, professional
Length: 2-3 short paragraphs
Include: Date/time confirmation, what to bring, how to reschedule if needed

Do NOT mention the prediction or risk assessment to the patient."""
)


# ==================== Prompt Builder ====================

@dataclass
class RiskExplainerPrompt:
    """
    Builder for risk explanation prompts.
    
    Constructs appropriate prompts based on prediction data
    and desired explanation style.
    
    Example
    -------
    >>> explainer = RiskExplainerPrompt()
    >>> prompt = explainer.build(
    ...     probability=0.65,
    ...     risk_tier="MEDIUM",
    ...     patient_data={"age": 35, "lead_days": 14}
    ... )
    """
    
    include_examples: bool = True
    explanation_style: str = "detailed"  # detailed, concise, technical
    
    def build(
        self,
        probability: float,
        risk_tier: str,
        confidence: str = "Moderate",
        patient_data: Optional[Dict[str, Any]] = None,
        model_factors: Optional[Dict[str, Any]] = None,
        patient_history: Optional[Dict[str, Any]] = None
    ) -> tuple[str, str]:
        """
        Build the complete prompt.
        
        Returns
        -------
        tuple[str, str]
            (system_prompt, user_prompt)
        """
        patient_data = patient_data or {}
        model_factors = model_factors or {}
        patient_history = patient_history or {}
        
        # Build system prompt
        system_prompt = HEALTHCARE_SYSTEM_PROMPT
        
        if self.include_examples:
            system_prompt += "\n\n## Example Explanations\n"
            for example in RISK_EXPLANATION_EXAMPLES[:2]:
                system_prompt += f"\nInput: {example['input']}\nOutput: {example['output']}\n"
        
        # Build history section
        if patient_history:
            history_section = self._format_history(patient_history)
        else:
            history_section = "- No previous appointment history available"
        
        # Build factors section
        if model_factors:
            factors_section = self._format_factors(model_factors)
        else:
            factors_section = "- Factor details not available"
        
        # Format user prompt
        user_prompt = RISK_EXPLANATION_TEMPLATE.format(
            probability=probability,
            risk_tier=risk_tier,
            confidence=confidence,
            age=patient_data.get("age", "Unknown"),
            gender=patient_data.get("gender", "Unknown"),
            lead_days=patient_data.get("lead_days", "Unknown"),
            sms_status="Enabled" if patient_data.get("sms_received", 0) else "Not enabled",
            appointment_day=patient_data.get("appointment_weekday", "Unknown"),
            hypertension="Yes" if patient_data.get("hypertension", 0) else "No",
            diabetes="Yes" if patient_data.get("diabetes", 0) else "No",
            scholarship="Yes" if patient_data.get("scholarship", 0) else "No",
            history_section=history_section,
            factors_section=factors_section
        )
        
        return system_prompt, user_prompt
    
    def _format_history(self, history: Dict[str, Any]) -> str:
        """Format patient history section."""
        lines = []
        
        if "total_appointments" in history:
            lines.append(f"- Total previous appointments: {history['total_appointments']}")
        
        if "noshow_rate" in history:
            lines.append(f"- Historical no-show rate: {history['noshow_rate']:.1%}")
        
        if "last_noshow" in history:
            lines.append(f"- Last no-show: {history['last_noshow']}")
        
        if "is_first" in history and history["is_first"]:
            lines.append("- This is the patient's first appointment")
        
        return "\n".join(lines) if lines else "- No history available"
    
    def _format_factors(self, factors: Dict[str, Any]) -> str:
        """Format model factors section."""
        lines = []
        
        risk_factors = factors.get("risk_factors", [])
        protective_factors = factors.get("protective_factors", [])
        
        if risk_factors:
            lines.append("**Increasing Risk:**")
            for factor in risk_factors[:3]:
                lines.append(f"  - {factor['name']}: {factor['impact']}")
        
        if protective_factors:
            lines.append("\n**Decreasing Risk:**")
            for factor in protective_factors[:3]:
                lines.append(f"  - {factor['name']}: {factor['impact']}")
        
        return "\n".join(lines) if lines else "- No factor details available"
    
    def build_quick(
        self,
        probability: float,
        risk_tier: str,
        key_factors: list[str]
    ) -> tuple[str, str]:
        """Build a quick/concise explanation prompt."""
        system_prompt = "You are a healthcare assistant. Provide brief, actionable explanations."
        
        user_prompt = QUICK_EXPLANATION_TEMPLATE.format(
            probability=probability,
            risk_tier=risk_tier,
            key_factors=", ".join(key_factors[:3])
        )
        
        return system_prompt, user_prompt


# ==================== Convenience Functions ====================

def explain_prediction(
    prediction_result: Dict[str, Any],
    patient_data: Dict[str, Any],
    style: str = "detailed"
) -> tuple[str, str]:
    """
    Convenience function to build explanation prompt from prediction result.
    
    Parameters
    ----------
    prediction_result : dict
        Result from NoShowPredictor.predict()
    patient_data : dict
        Patient and appointment data
    style : str
        Explanation style (detailed, concise)
    
    Returns
    -------
    tuple[str, str]
        (system_prompt, user_prompt)
    """
    explainer = RiskExplainerPrompt(explanation_style=style)
    
    return explainer.build(
        probability=prediction_result.get("probability", 0),
        risk_tier=prediction_result.get("risk", {}).get("tier", "UNKNOWN"),
        confidence=prediction_result.get("risk", {}).get("confidence", "Unknown"),
        patient_data=patient_data,
        model_factors=prediction_result.get("explanation", {})
    )