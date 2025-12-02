# src/llm/chains/intervention_chain.py
"""
Intervention Recommendation Chain
==================================
Chain for generating intervention strategies based on risk assessment.
"""

from typing import Dict, Any, Optional, List
from enum import Enum

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel
from pydantic import BaseModel, Field

from .base import BaseHealthcareChain


# ==================== Output Schemas ====================

class InterventionPriority(str, Enum):
    IMMEDIATE = "immediate"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


class InterventionPlan(BaseModel):
    """Structured intervention plan."""
    
    primary_action: str = Field(description="Main action to take")
    priority: InterventionPriority = Field(description="Action priority")
    timeline: str = Field(description="When to take action")
    phone_script: Optional[str] = Field(default=None, description="Script for phone calls")
    sms_count: int = Field(default=1, description="Number of SMS reminders")
    backup_plan: str = Field(description="What to do if primary action fails")
    estimated_effort: str = Field(description="Staff time estimate")


# ==================== Prompts ====================

INTERVENTION_SYSTEM_PROMPT = """You are a Healthcare Intervention Specialist. You create targeted intervention plans to reduce patient no-shows.

Core Principles:
1. MATCH intervention intensity to risk level
2. PRIORITIZE high-impact, low-effort actions
3. FOCUS on patient support, not punishment
4. CONSIDER resource constraints
5. PROVIDE specific, actionable steps

Risk-Based Intervention Framework:
- CRITICAL (>80%): Personal call + confirmation required + waitlist backup
- HIGH (60-80%): Phone call + double SMS + day-before confirmation
- MEDIUM (40-60%): Double SMS reminder + optional call
- LOW (20-40%): Standard SMS reminder
- MINIMAL (<20%): Standard process, no extra intervention"""


SINGLE_INTERVENTION_PROMPT = """Create an intervention plan for this appointment.

## Risk Assessment
- **Risk Level**: {risk_tier} ({probability:.0%} no-show probability)
- **Days Until Appointment**: {lead_days}

## Patient Profile
- Age: {age}
- First-time patient: {is_first}
- Previous no-shows: {previous_noshows}
- SMS enabled: {sms_enabled}
- Neighborhood: {neighbourhood}

## Current Status
- Scheduled reminders: {current_reminders}
- Last contact: {last_contact}

## Available Resources
- Staff availability: {staff_availability}
- Phone slots today: {phone_slots}

---

Provide a complete intervention plan:

1. **PRIMARY ACTION**: The single most important thing to do
2. **TIMELINE**: Specific timing (e.g., "Call today before 2pm")
3. **PHONE SCRIPT**: If calling, exactly what to say
4. **SMS MESSAGE**: If texting, the exact message
5. **BACKUP PLAN**: What to do if patient doesn't respond
6. **SUCCESS INDICATOR**: How to know the intervention worked"""


BATCH_TRIAGE_PROMPT = """Triage these {count} appointments for intervention.

## Today's Risk Distribution
{risk_summary}

## Available Resources
- Staff hours: {staff_hours}
- Phone call slots: {phone_slots}
- Automated SMS: Unlimited

## High-Risk Appointments (Intervention Needed)
{high_risk_list}

---

Create a prioritized action plan:

1. **IMMEDIATE** (Do in next hour): List specific appointments
2. **TODAY** (Complete by end of day): List appointments  
3. **AUTOMATED** (SMS sufficient): Count and approach
4. **NO ACTION** (Low risk): Count

For each priority group, specify:
- Which appointments
- Who should handle
- Estimated time needed"""


# ==================== Chain ====================

class InterventionChain(BaseHealthcareChain):
    """
    Chain for generating intervention recommendations.
    
    Supports both single-appointment and batch triage modes.
    
    Example
    -------
    >>> chain = InterventionChain()
    >>> plan = chain.recommend(
    ...     probability=0.75,
    ...     risk_tier="HIGH",
    ...     patient_data={"age": 35, "lead_days": 7}
    ... )
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        temperature: float = 0.4,
        **kwargs
    ):
        super().__init__(model_name, temperature, **kwargs)
        
        # Build batch chain separately
        self._batch_chain = self._build_batch_chain()
    
    def _get_system_prompt(self) -> str:
        return INTERVENTION_SYSTEM_PROMPT
    
    def _build_chain(self):
        """Build single intervention chain."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", INTERVENTION_SYSTEM_PROMPT),
            ("human", SINGLE_INTERVENTION_PROMPT)
        ])
        
        chain = (
            RunnableLambda(self._preprocess_single)
            | prompt
            | self.model
            | StrOutputParser()
        )
        
        return chain
    
    def _build_batch_chain(self):
        """Build batch triage chain."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", INTERVENTION_SYSTEM_PROMPT),
            ("human", BATCH_TRIAGE_PROMPT)
        ])
        
        chain = (
            RunnableLambda(self._preprocess_batch)
            | prompt
            | self.model
            | StrOutputParser()
        )
        
        return chain
    
    def _preprocess_single(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess single appointment inputs."""
        patient_data = inputs.get("patient_data", {})
        constraints = inputs.get("constraints", {})
        
        return {
            "risk_tier": inputs.get("risk_tier", "MEDIUM"),
            "probability": inputs.get("probability", 0.5),
            "lead_days": patient_data.get("lead_days", 0),
            "age": patient_data.get("age", "Unknown"),
            "is_first": "Yes" if patient_data.get("is_first_appointment") else "No",
            "previous_noshows": patient_data.get("previous_noshows", 0),
            "sms_enabled": "Yes" if patient_data.get("sms_received") else "No",
            "neighbourhood": patient_data.get("neighbourhood", "Unknown"),
            "current_reminders": patient_data.get("scheduled_reminders", "Standard (1 SMS)"),
            "last_contact": patient_data.get("last_contact", "At scheduling"),
            "staff_availability": constraints.get("staff_availability", "Normal"),
            "phone_slots": constraints.get("phone_slots", "Available")
        }
    
    def _preprocess_batch(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess batch inputs."""
        appointments = inputs.get("appointments", [])
        resources = inputs.get("resources", {})
        
        # Count by risk tier
        tier_counts = {}
        for apt in appointments:
            tier = apt.get("risk_tier", "MEDIUM")
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
        
        # Build risk summary
        risk_lines = []
        for tier in ["CRITICAL", "HIGH", "MEDIUM", "LOW", "MINIMAL"]:
            count = tier_counts.get(tier, 0)
            if count > 0:
                risk_lines.append(f"- {tier}: {count} appointments")
        
        # Build high-risk list (CRITICAL and HIGH)
        high_risk = [
            apt for apt in appointments 
            if apt.get("risk_tier") in ["CRITICAL", "HIGH"]
        ]
        high_risk_lines = []
        for i, apt in enumerate(high_risk[:10]):  # Top 10
            high_risk_lines.append(
                f"{i+1}. {apt.get('risk_tier')} ({apt.get('probability', 0):.0%}) - "
                f"{apt.get('lead_days', '?')} days out, "
                f"history: {apt.get('previous_noshows', 0)} no-shows"
            )
        
        return {
            "count": len(appointments),
            "risk_summary": "\n".join(risk_lines),
            "staff_hours": resources.get("staff_hours", 4),
            "phone_slots": resources.get("phone_slots", 10),
            "high_risk_list": "\n".join(high_risk_lines) or "None"
        }
    
    def recommend(
        self,
        probability: float,
        risk_tier: str,
        patient_data: Dict[str, Any],
        constraints: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate intervention recommendation for single appointment.
        
        Parameters
        ----------
        probability : float
            No-show probability
        risk_tier : str
            Risk tier
        patient_data : dict
            Patient and appointment data
        constraints : dict, optional
            Resource constraints
        
        Returns
        -------
        str
            Intervention recommendation
        """
        result = self.invoke({
            "probability": probability,
            "risk_tier": risk_tier,
            "patient_data": patient_data,
            "constraints": constraints or {}
        })
        
        return result["output"]
    
    def triage_batch(
        self,
        appointments: List[Dict[str, Any]],
        resources: Dict[str, Any]
    ) -> str:
        """
        Generate triage plan for batch of appointments.
        
        Parameters
        ----------
        appointments : list
            List of appointment predictions
        resources : dict
            Available resources
        
        Returns
        -------
        str
            Triage and intervention plan
        """
        result = self._batch_chain.invoke({
            "appointments": appointments,
            "resources": resources
        })
        
        return result