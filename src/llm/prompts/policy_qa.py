# src/llm/prompts/policy_qa.py
"""
Policy Q&A Prompts
==================
Prompts for answering questions about appointment policies.
(Foundation for RAG in Week 11)
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from .templates import PromptTemplate, SAFETY_GUIDELINES


POLICY_QA_SYSTEM_PROMPT = """You are a Healthcare Appointment Policy Assistant. Your role is to answer questions about appointment scheduling, cancellations, no-shows, and related policies.

Guidelines:
1. Answer based ONLY on the provided policy context
2. If the answer isn't in the context, say "I don't have information about that in the current policies"
3. Be precise and cite specific policy sections when possible
4. If a question requires human judgment, recommend speaking with a supervisor
5. Never make up policies or procedures

""" + SAFETY_GUIDELINES


POLICY_QA_TEMPLATE = PromptTemplate(
    name="policy_qa",
    version="1.0.0",
    description="Answer policy questions with context",
    template="""Answer the following question about appointment policies.

## Policy Context
{context}

## Question
{question}

## Instructions
- Answer based ONLY on the provided context
- If unsure, acknowledge uncertainty
- Quote relevant policy sections when helpful
- Suggest who to contact for questions beyond policy scope

Answer:"""
)


POLICY_QA_NO_CONTEXT_TEMPLATE = PromptTemplate(
    name="policy_qa_no_context",
    version="1.0.0",
    description="Answer common policy questions without RAG",
    template="""Answer this question about healthcare appointment policies.

Question: {question}

Provide a helpful, general answer based on common healthcare practices.
Note that specific policies may vary by organization.
If the question requires organization-specific information, recommend checking with administration."""
)


POLICY_CLARIFICATION_TEMPLATE = PromptTemplate(
    name="policy_clarification",
    version="1.0.0",
    description="Clarify ambiguous policy questions",
    template="""The user's question needs clarification before I can provide an accurate answer.

Original question: {question}

Please ask 1-2 clarifying questions to better understand what the user needs.
Focus on:
- Specific policy area (cancellation, rescheduling, no-show consequences)
- Role (patient, staff, provider)
- Specific scenario they're asking about"""
)


# Common Q&A pairs for few-shot learning
POLICY_QA_EXAMPLES = [
    {
        "question": "What happens if a patient misses their appointment?",
        "context": "No-Show Policy: Patients who miss appointments without 24-hour notice will be marked as no-shows. After 3 no-shows within 12 months, patients will receive a warning letter. Continued no-shows may result in requiring appointments to be pre-paid.",
        "answer": "According to our No-Show Policy, when a patient misses an appointment without providing at least 24 hours notice, it's recorded as a no-show. The consequences escalate: after 3 no-shows within a 12-month period, patients receive a warning letter. If no-shows continue after the warning, the patient may be required to pre-pay for future appointments. The goal is to encourage attendance while maintaining access to care."
    },
    {
        "question": "Can patients reschedule same-day?",
        "context": "Rescheduling Policy: Patients may reschedule appointments up to 2 hours before the scheduled time without penalty. Same-day cancellations (less than 24 hours notice) may be counted as late cancellations.",
        "answer": "Yes, patients can reschedule on the same day, but there are important conditions. According to the Rescheduling Policy, patients can reschedule without penalty if they do so at least 2 hours before their appointment time. However, any same-day cancellation (made with less than 24 hours notice) may be recorded as a late cancellation. I'd recommend patients call as early as possible when they know they need to reschedule."
    }
]


@dataclass
class PolicyQAPrompt:
    """
    Builder for policy Q&A prompts.
    
    This is a foundational class that will be extended in Week 11
    with RAG capabilities for retrieving relevant policy context.
    
    Example
    -------
    >>> qa = PolicyQAPrompt()
    >>> system, user = qa.build(
    ...     question="What is the cancellation policy?",
    ...     context="Cancellation policy text..."
    ... )
    """
    
    include_examples: bool = True
    require_context: bool = False
    
    def build(
        self,
        question: str,
        context: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> tuple[str, str]:
        """
        Build Q&A prompt.
        
        Parameters
        ----------
        question : str
            User's question
        context : str, optional
            Retrieved policy context (from RAG)
        conversation_history : list, optional
            Previous conversation turns
        
        Returns
        -------
        tuple[str, str]
            (system_prompt, user_prompt)
        """
        # Build system prompt with examples
        system_prompt = POLICY_QA_SYSTEM_PROMPT
        
        if self.include_examples:
            system_prompt += "\n\n## Example Q&A\n"
            for ex in POLICY_QA_EXAMPLES:
                system_prompt += f"\nQ: {ex['question']}\nContext: {ex['context'][:100]}...\nA: {ex['answer'][:200]}...\n"
        
        # Build user prompt
        if context:
            user_prompt = POLICY_QA_TEMPLATE.format(
                context=context,
                question=question
            )
        elif self.require_context:
            # Return clarification request if context is required but missing
            user_prompt = f"I need to search our policy database to answer: '{question}'. Please wait while I retrieve relevant policies."
        else:
            user_prompt = POLICY_QA_NO_CONTEXT_TEMPLATE.format(
                question=question
            )
        
        return system_prompt, user_prompt
    
    def build_with_history(
        self,
        question: str,
        history: List[Dict[str, str]],
        context: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Build messages list with conversation history.
        
        Returns list of message dicts for chat completion.
        """
        messages = [{"role": "system", "content": POLICY_QA_SYSTEM_PROMPT}]
        
        # Add history
        for turn in history[-5:]:  # Last 5 turns
            messages.append(turn)
        
        # Add current question with context
        if context:
            current = f"Policy Context:\n{context}\n\nQuestion: {question}"
        else:
            current = question
        
        messages.append({"role": "user", "content": current})
        
        return messages


# ==================== Utility Functions ====================

def detect_policy_topic(question: str) -> str:
    """
    Detect the policy topic from a question.
    
    Used for routing to appropriate policy documents in RAG.
    """
    question_lower = question.lower()
    
    if any(word in question_lower for word in ["cancel", "reschedul"]):
        return "cancellation"
    elif any(word in question_lower for word in ["no-show", "noshow", "miss", "missed"]):
        return "noshow"
    elif any(word in question_lower for word in ["remind", "sms", "notification"]):
        return "reminders"
    elif any(word in question_lower for word in ["pay", "cost", "fee", "charge"]):
        return "billing"
    elif any(word in question_lower for word in ["wait", "queue", "priority"]):
        return "scheduling"
    else:
        return "general"