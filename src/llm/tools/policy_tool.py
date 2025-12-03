# src/llm/tools/policy_tool.py
"""
Policy Search Tool
==================
Tool for searching and retrieving policy information.
(Foundation for RAG in Week 11)
"""

import logging
from typing import Optional, Type, List, Dict, Any

from pydantic import BaseModel, Field
from pydantic.v1 import PrivateAttr  # Added for Pydantic v1 compatibility
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun

logger = logging.getLogger(__name__)


# ==================== Mock Policy Database ====================
# In Week 11, this will be replaced with actual vector search

POLICY_DATABASE = {
    "cancellation": {
        "title": "Appointment Cancellation Policy",
        "content": """
CANCELLATION POLICY (Updated January 2024)

Section 1: Cancellation Timeline
- Patients must provide at least 24 hours notice to cancel appointments
- Cancellations made less than 24 hours are considered "late cancellations"
- Late cancellations may be counted toward no-show limits

Section 2: How to Cancel
- Call the scheduling line: 555-0123
- Use the patient portal online
- Reply to SMS reminders with "CANCEL"

Section 3: Rescheduling
- Patients can reschedule up to 2 hours before appointment time
- Same-day rescheduling is accommodated when possible
- Priority given to patients with good attendance history
"""
    },
    "noshow": {
        "title": "No-Show Policy",
        "content": """
NO-SHOW POLICY

Definition:
A no-show occurs when a patient fails to attend their appointment without 
prior notification. Arriving more than 15 minutes late may count as a no-show.

Consequences:
- First no-show: Verbal reminder of policy
- Second no-show: Written warning sent
- Third no-show (within 12 months): May require pre-payment
- Continued no-shows: May be discharged from practice

Exceptions:
- Medical emergencies (with documentation)
- Family emergencies
- Severe weather events
- Transportation emergencies

Appeals:
Patients may appeal no-show status by contacting Patient Relations
within 7 days of the missed appointment.
"""
    },
    "reminders": {
        "title": "Appointment Reminder System",
        "content": """
REMINDER SYSTEM

Standard Reminders:
- 7 days before: Email reminder (if email on file)
- 3 days before: SMS reminder (if mobile on file)
- 1 day before: SMS or phone call

High-Risk Appointments:
- Additional reminder at 48 hours
- Phone call confirmation at 24 hours
- Waitlist patient on standby

Opting Out:
Patients may opt out of reminders but assume responsibility 
for remembering appointments.
"""
    },
    "scheduling": {
        "title": "Appointment Scheduling Guidelines",
        "content": """
SCHEDULING GUIDELINES

Booking:
- Appointments can be booked up to 3 months in advance
- Same-day appointments available for urgent needs
- Follow-up appointments scheduled at checkout

Wait Times:
- Target: Less than 15 minutes past appointment time
- Patients running late should call ahead
- Double-booking allowed only for quick visits

Priority Scheduling:
- Urgent medical needs
- Patients with excellent attendance
- Follow-up appointments within treatment windows
"""
    }
}


class PolicySearchInput(BaseModel):
    """Input for policy search."""
    
    query: str = Field(description="The policy question or topic to search for")
    topic: Optional[str] = Field(
        default=None,
        description="Specific topic: cancellation, noshow, reminders, scheduling"
    )


class PolicySearchTool(BaseTool):
    """
    Tool for searching policy documents.
    
    In Week 11, this will use vector similarity search (RAG).
    Currently uses keyword matching as a placeholder.
    """
    
    # Fix for Pydantic v1 "unable to infer type for attribute artifact" on Python 3.14
    _artifact: Any = PrivateAttr(default=None)
    
    name: str = "search_policies"
    description: str = """
    Searches the healthcare appointment policy database.
    
    Use this tool when you need to:
    - Answer questions about cancellation policies
    - Explain no-show consequences
    - Describe the reminder system
    - Clarify scheduling rules
    
    Input: A question or topic to search for
    Output: Relevant policy excerpts
    """
    args_schema: Type[BaseModel] = PolicySearchInput
    
    def _run(
        self,
        query: str,
        topic: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Search policies."""
        try:
            # Determine topic from query if not provided
            if topic is None:
                topic = self._detect_topic(query)
            
            # Get relevant policy
            if topic and topic in POLICY_DATABASE:
                policy = POLICY_DATABASE[topic]
                return f"""
POLICY DOCUMENT: {policy['title']}
{'=' * 50}

{policy['content']}

Source: Internal Policy Database
Last Updated: January 2024
"""
            else:
                # Return all potentially relevant policies
                return self._search_all_policies(query)
                
        except Exception as e:
            logger.error(f"Policy search failed: {e}")
            return "Unable to search policies. Please consult with administration."
    
    async def _arun(self, query: str, topic: Optional[str] = None) -> str:
        """Async version."""
        return self._run(query, topic)
    
    def _detect_topic(self, query: str) -> Optional[str]:
        """Detect policy topic from query."""
        query_lower = query.lower()
        
        topic_keywords = {
            "cancellation": ["cancel", "reschedul", "change appointment"],
            "noshow": ["no-show", "noshow", "miss", "missed", "didn't come", "absent"],
            "reminders": ["remind", "sms", "notification", "alert", "text"],
            "scheduling": ["schedul", "book", "appointment time", "wait", "priority"]
        }
        
        for topic, keywords in topic_keywords.items():
            if any(kw in query_lower for kw in keywords):
                return topic
        
        return None
    
    def _search_all_policies(self, query: str) -> str:
        """Search across all policies (simple keyword matching)."""
        query_words = set(query.lower().split())
        results = []
        
        for topic, policy in POLICY_DATABASE.items():
            content_words = set(policy['content'].lower().split())
            overlap = len(query_words & content_words)
            if overlap > 0:
                results.append((overlap, topic, policy))
        
        if not results:
            return "No matching policies found. Please contact administration for assistance."
        
        # Sort by relevance (overlap count)
        results.sort(reverse=True, key=lambda x: x[0])
        
        # Return top 2 results
        output = "RELEVANT POLICIES FOUND:\n" + "=" * 50 + "\n\n"
        
        for _, topic, policy in results[:2]:
            output += f"📋 {policy['title']}\n"
            output += "-" * 40 + "\n"
            # Return first 500 chars of content
            output += policy['content'][:500] + "...\n\n"
        
        return output


_policy_tool: Optional[PolicySearchTool] = None


def get_policy_tool() -> PolicySearchTool:
    """Get or create policy tool singleton."""
    global _policy_tool
    if _policy_tool is None:
        _policy_tool = PolicySearchTool()
    return _policy_tool