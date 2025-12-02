# src/llm/prompts/templates.py
"""
Base Prompt Template System
===========================
Flexible prompt template system with variable substitution,
validation, and versioning.
"""

import re
import json
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
from abc import ABC, abstractmethod


@dataclass
class PromptTemplate:
    """
    A reusable prompt template with variable substitution.
    
    Attributes
    ----------
    name : str
        Template name for identification
    template : str
        The prompt template with {variable} placeholders
    description : str
        Human-readable description
    version : str
        Template version for tracking changes
    variables : Set[str]
        Required variables (auto-extracted from template)
    
    Example
    -------
    >>> template = PromptTemplate(
    ...     name="greeting",
    ...     template="Hello {name}, you have {risk_level} risk.",
    ...     description="Simple greeting with risk"
    ... )
    >>> prompt = template.format(name="John", risk_level="low")
    """
    
    name: str
    template: str
    description: str = ""
    version: str = "1.0.0"
    variables: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Extract variables from template."""
        self.variables = set(re.findall(r'\{(\w+)\}', self.template))
    
    def format(self, **kwargs) -> str:
        """
        Format the template with provided variables.
        
        Parameters
        ----------
        **kwargs
            Variable values to substitute
        
        Returns
        -------
        str
            Formatted prompt
        
        Raises
        ------
        ValueError
            If required variables are missing
        """
        # Check for missing variables
        missing = self.variables - set(kwargs.keys())
        if missing:
            raise ValueError(f"Missing required variables: {missing}")
        
        # Format and return
        return self.template.format(**kwargs)
    
    def validate(self, **kwargs) -> List[str]:
        """
        Validate variables without formatting.
        
        Returns list of validation errors (empty if valid).
        """
        errors = []
        
        # Check missing
        missing = self.variables - set(kwargs.keys())
        if missing:
            errors.append(f"Missing variables: {missing}")
        
        # Check extra
        extra = set(kwargs.keys()) - self.variables
        if extra:
            errors.append(f"Unexpected variables: {extra}")
        
        return errors
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "template": self.template,
            "description": self.description,
            "version": self.version,
            "variables": list(self.variables),
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PromptTemplate":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            template=data["template"],
            description=data.get("description", ""),
            version=data.get("version", "1.0.0"),
            metadata=data.get("metadata", {})
        )


class PromptLibrary:
    """
    Collection of prompt templates with management utilities.
    
    Provides:
    - Template registration and retrieval
    - Version tracking
    - Export/import functionality
    """
    
    def __init__(self):
        self._templates: Dict[str, PromptTemplate] = {}
        self._history: List[Dict] = []
    
    def register(self, template: PromptTemplate) -> None:
        """Register a new template."""
        self._templates[template.name] = template
        self._history.append({
            "action": "register",
            "template": template.name,
            "version": template.version,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    def get(self, name: str) -> Optional[PromptTemplate]:
        """Get a template by name."""
        return self._templates.get(name)
    
    def list_templates(self) -> List[str]:
        """List all template names."""
        return list(self._templates.keys())
    
    def export_all(self) -> Dict[str, Any]:
        """Export all templates to dictionary."""
        return {
            "templates": {
                name: t.to_dict() 
                for name, t in self._templates.items()
            },
            "exported_at": datetime.utcnow().isoformat()
        }
    
    def import_templates(self, data: Dict[str, Any]) -> int:
        """Import templates from dictionary. Returns count imported."""
        count = 0
        for name, template_data in data.get("templates", {}).items():
            template = PromptTemplate.from_dict(template_data)
            self.register(template)
            count += 1
        return count


# ==================== System Prompts ====================

HEALTHCARE_SYSTEM_PROMPT = """You are a Healthcare Appointment Assistant, an AI designed to help healthcare providers understand and reduce patient no-shows.

Your role is to:
1. Explain ML model predictions in clear, non-technical language
2. Provide actionable intervention recommendations
3. Answer questions about appointment policies
4. Help staff communicate effectively with patients

Guidelines:
- Be empathetic and patient-focused
- Use plain language, avoid medical jargon when possible
- Always emphasize that predictions are probabilistic, not certain
- Focus on supportive interventions, never punitive measures
- Respect patient privacy and confidentiality
- When uncertain, acknowledge limitations

You have access to the no-show prediction model which considers factors like:
- Appointment lead time
- Patient history
- Demographics
- Day/time of appointment
- SMS reminder status"""


SAFETY_GUIDELINES = """
IMPORTANT SAFETY GUIDELINES:
1. Never provide medical diagnosis or treatment advice
2. Never reveal specific patient information in examples
3. Never suggest discriminatory practices
4. Always recommend human review for high-stakes decisions
5. Acknowledge uncertainty when appropriate
6. Redirect medical questions to healthcare providers
"""


# ==================== Few-Shot Examples ====================

RISK_EXPLANATION_EXAMPLES = [
    {
        "input": {
            "probability": 0.75,
            "risk_tier": "HIGH",
            "age": 28,
            "lead_days": 21,
            "sms_received": 0,
            "previous_noshows": 2
        },
        "output": """Based on our analysis, this appointment has a **high risk** of no-show (75% probability).

**Key Risk Factors:**
1. **Long wait time**: The appointment is scheduled 21 days out. Longer lead times often lead to forgotten appointments.
2. **No SMS reminder**: The patient hasn't been set up for SMS reminders, which are effective at reducing no-shows.
3. **Previous no-shows**: This patient has missed 2 previous appointments, suggesting a pattern.

**Recommended Actions:**
- Enable SMS reminders immediately
- Consider a phone call reminder 48-72 hours before
- If possible, offer a sooner appointment slot

This is a prediction based on patterns, not a certainty. Personal circumstances can always change."""
    },
    {
        "input": {
            "probability": 0.12,
            "risk_tier": "LOW",
            "age": 55,
            "lead_days": 3,
            "sms_received": 1,
            "previous_noshows": 0
        },
        "output": """Great news! This appointment has a **low risk** of no-show (12% probability).

**Positive Indicators:**
1. **Short lead time**: The appointment is in just 3 days, making it easy to remember.
2. **SMS reminder enabled**: The patient will receive a reminder.
3. **Reliable history**: This patient has attended all previous appointments.

**Recommendation:**
- Standard reminder process is sufficient
- This patient demonstrates high engagement - consider for priority scheduling
- No additional intervention needed"""
    }
]


INTERVENTION_EXAMPLES = [
    {
        "risk_tier": "CRITICAL",
        "probability": 0.85,
        "context": "Patient has missed 3 of last 4 appointments",
        "intervention": """**Immediate Action Required**

For this critical-risk appointment (85% no-show probability), implement the full intervention protocol:

1. **Personal Outreach** (Today)
   - Phone call from staff member
   - Understand any barriers to attendance
   - Offer flexible scheduling if needed

2. **Confirmation Required**
   - Request verbal confirmation of attendance
   - If unable to reach, send certified letter

3. **Barrier Assessment**
   - Ask about transportation needs
   - Check if appointment time works with their schedule
   - Inquire about any concerns about the visit

4. **Alternative Options**
   - Offer telehealth if appropriate
   - Consider home visit for mobility issues
   - Provide transportation assistance information

5. **Waitlist Backup**
   - Have a waitlist patient ready for this slot
   - Confirm backup 24 hours before

*Note: Focus on support, not penalties. Understanding why patients miss appointments leads to better solutions.*"""
    }
]