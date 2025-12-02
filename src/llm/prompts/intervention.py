# src/llm/prompts/intervention.py
"""
Intervention Recommendation Prompts
====================================
Prompts for generating intervention strategies based on risk levels.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from .templates import PromptTemplate, HEALTHCARE_SYSTEM_PROMPT, SAFETY_GUIDELINES


INTERVENTION_SYSTEM_PROMPT = HEALTHCARE_SYSTEM_PROMPT + """

When recommending interventions:
1. Match intervention intensity to risk level
2. Consider resource constraints (staff time, costs)
3. Focus on patient support, not penalties
4. Include specific, actionable steps
5. Consider the patient's individual circumstances

""" + SAFETY_GUIDELINES


INTERVENTION_TEMPLATE = PromptTemplate(
    name="intervention_recommendation",
    version="1.0.0",
    description="Generate intervention recommendations based on risk",
    template="""Generate intervention recommendations for this appointment.

## Risk Assessment
- **Risk Tier**: {risk_tier}
- **No-Show Probability**: {probability:.1%}
- **Lead Time**: {lead_days} days until appointment

## Patient Profile
- Age: {age}
- First-time patient: {is_first_appointment}
- Previous no-shows: {previous_noshows}
- SMS enabled: {sms_enabled}
- Neighborhood: {neighbourhood}

## Current Intervention
- Scheduled SMS reminders: {current_sms_count}
- Phone call scheduled: {phone_call_scheduled}

## Constraints
- Staff availability: {staff_availability}
- Budget tier: {budget_tier}

---

Please provide a structured intervention plan with:

1. **Primary Action** (Most important step)
2. **Timeline** (When to take each action)
3. **Communication Script** (What to say if calling)
4. **Backup Plan** (If primary intervention fails)
5. **Success Metrics** (How to know if it worked)

Tailor recommendations to the specific risk level and patient profile."""
)


BATCH_PRIORITIZATION_TEMPLATE = PromptTemplate(
    name="batch_prioritization",
    version="1.0.0",
    description="Prioritize interventions for a list of appointments",
    template="""You have {total_appointments} appointments to review for intervention prioritization.

## Today's Appointments Summary
{appointments_summary}

## Available Resources
- Phone call slots available: {phone_slots}
- Staff hours for outreach: {staff_hours}

## Priority Distribution
- Critical risk: {critical_count}
- High risk: {high_count}  
- Medium risk: {medium_count}
- Low/Minimal risk: {low_count}

---

Create a prioritized action plan:

1. **Immediate Priority** (Do today)
2. **High Priority** (Do within 24 hours)
3. **Standard Priority** (Automated reminders sufficient)
4. **Resource Allocation** (How to distribute staff time)

Focus on maximizing impact with available resources."""
)


@dataclass
class InterventionPrompt:
    """
    Builder for intervention recommendation prompts.
    
    Example
    -------
    >>> intervention = InterventionPrompt()
    >>> system, user = intervention.build_single(
    ...     probability=0.72,
    ...     risk_tier="HIGH",
    ...     patient_data={"age": 35, "lead_days": 10}
    ... )
    """
    
    include_scripts: bool = True
    include_cost_analysis: bool = False
    
    def build_single(
        self,
        probability: float,
        risk_tier: str,
        patient_data: Dict[str, Any],
        constraints: Optional[Dict[str, Any]] = None
    ) -> tuple[str, str]:
        """
        Build intervention prompt for a single appointment.
        
        Returns
        -------
        tuple[str, str]
            (system_prompt, user_prompt)
        """
        constraints = constraints or {}
        
        user_prompt = INTERVENTION_TEMPLATE.format(
            risk_tier=risk_tier,
            probability=probability,
            lead_days=patient_data.get("lead_days", 0),
            age=patient_data.get("age", "Unknown"),
            is_first_appointment="Yes" if patient_data.get("is_first_appointment") else "No",
            previous_noshows=patient_data.get("previous_noshows", "Unknown"),
            sms_enabled="Yes" if patient_data.get("sms_received") else "No",
            neighbourhood=patient_data.get("neighbourhood", "Unknown"),
            current_sms_count=patient_data.get("current_sms_count", 1),
            phone_call_scheduled="Yes" if patient_data.get("phone_call_scheduled") else "No",
            staff_availability=constraints.get("staff_availability", "Normal"),
            budget_tier=constraints.get("budget_tier", "Standard")
        )
        
        return INTERVENTION_SYSTEM_PROMPT, user_prompt
    
    def build_batch(
        self,
        appointments: List[Dict[str, Any]],
        resources: Dict[str, Any]
    ) -> tuple[str, str]:
        """
        Build prioritization prompt for batch of appointments.
        
        Parameters
        ----------
        appointments : list
            List of appointment predictions
        resources : dict
            Available resources (staff, phone slots, etc.)
        
        Returns
        -------
        tuple[str, str]
            (system_prompt, user_prompt)
        """
        # Count by risk tier
        tier_counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0, "MINIMAL": 0}
        for apt in appointments:
            tier = apt.get("risk_tier", "MEDIUM")
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
        
        # Build summary
        summary_lines = []
        for i, apt in enumerate(appointments[:10]):  # Show top 10
            summary_lines.append(
                f"  {i+1}. {apt.get('risk_tier', '?')} risk ({apt.get('probability', 0):.0%}) - "
                f"Lead time: {apt.get('lead_days', '?')} days"
            )
        
        if len(appointments) > 10:
            summary_lines.append(f"  ... and {len(appointments) - 10} more")
        
        user_prompt = BATCH_PRIORITIZATION_TEMPLATE.format(
            total_appointments=len(appointments),
            appointments_summary="\n".join(summary_lines),
            phone_slots=resources.get("phone_slots", 10),
            staff_hours=resources.get("staff_hours", 4),
            critical_count=tier_counts["CRITICAL"],
            high_count=tier_counts["HIGH"],
            medium_count=tier_counts["MEDIUM"],
            low_count=tier_counts["LOW"] + tier_counts["MINIMAL"]
        )
        
        return INTERVENTION_SYSTEM_PROMPT, user_prompt