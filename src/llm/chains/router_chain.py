# src/llm/chains/router_chain.py
"""
Intent Router Chain
===================
Routes user queries to the appropriate specialized chain.
"""

import logging
import time
from typing import Dict, Any, Optional, Literal
from enum import Enum

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field

from .base import BaseHealthcareChain, ChainResult, get_llm

logger = logging.getLogger(__name__)


class IntentType(str, Enum):
    """Types of user intents."""
    PREDICTION = "prediction"
    EXPLANATION = "explanation"
    INTERVENTION = "intervention"
    POLICY_QUESTION = "policy_question"
    GENERAL_QUESTION = "general_question"
    GREETING = "greeting"
    UNKNOWN = "unknown"


class RoutingDecision(BaseModel):
    """Structured routing decision."""
    intent: IntentType
    confidence: float = Field(ge=0.0, le=1.0)
    entities: Dict[str, Any] = Field(default_factory=dict)
    reasoning: str


ROUTER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an intent classifier for a Healthcare Appointment Assistant.

Classify user messages into one of these categories:

1. **prediction** - User wants to check no-show risk for a patient
   Examples: "What's the risk for this patient?", "Will they show up?", "Check no-show probability"
   
2. **explanation** - User wants to understand a prediction
   Examples: "Why is this high risk?", "Explain the factors", "What's causing this?"
   
3. **intervention** - User wants recommendations on what to do
   Examples: "What should we do?", "How do we prevent this?", "Give me an action plan"
   
4. **policy_question** - User asking about policies/procedures
   Examples: "What's the cancellation policy?", "How many no-shows before discharge?"
   
5. **general_question** - General questions about the system
   Examples: "How does this work?", "What can you help with?"
   
6. **greeting** - Greetings or social interaction
   Examples: "Hello", "Thanks", "Hi there"

Also extract any entities like age, gender, dates mentioned.

Respond with JSON:
{{"intent": "category", "confidence": 0.0-1.0, "entities": {{}}, "reasoning": "brief explanation"}}
"""),
    ("human", "{message}")
])


class IntentRouterChain(BaseHealthcareChain):
    """
    Routes user messages to appropriate handlers.
    
    This chain classifies user intent and extracts relevant entities,
    enabling the main conversation chain to delegate to specialized chains.
    
    Example
    -------
    >>> router = IntentRouterChain()
    >>> result = router.run(message="What's the no-show risk for a 35 year old?")
    >>> print(result.metadata["intent"])  # "prediction"
    """
    
    def __init__(self, **kwargs):
        super().__init__(temperature=0.1, **kwargs)  # Low temperature for consistency
    
    def _build_chain(self):
        """Build the router chain with structured output."""
        return (
            ROUTER_PROMPT
            | self._llm.with_structured_output(RoutingDecision)
        )
    
    def run(self, message: str) -> ChainResult:
        """
        Classify user intent.
        
        Parameters
        ----------
        message : str
            User message to classify
        
        Returns
        -------
        ChainResult
            Classification result with intent and entities
        """
        start_time = time.time()
        
        try:
            decision = self._chain.invoke({"message": message})
            
            latency = (time.time() - start_time) * 1000
            
            return ChainResult(
                success=True,
                content=decision.reasoning,
                chain_name="IntentRouterChain",
                metadata={
                    "intent": decision.intent.value,
                    "confidence": decision.confidence,
                    "entities": decision.entities,
                    "reasoning": decision.reasoning
                },
                latency_ms=latency
            )
            
        except Exception as e:
            # Fallback to simple keyword matching
            return self._fallback_routing(message, start_time)
    
    def _fallback_routing(self, message: str, start_time: float) -> ChainResult:
        """Simple keyword-based fallback routing."""
        message_lower = message.lower()
        
        if any(w in message_lower for w in ["risk", "predict", "probability", "chance"]):
            intent = IntentType.PREDICTION
        elif any(w in message_lower for w in ["why", "explain", "factor", "cause"]):
            intent = IntentType.EXPLANATION
        elif any(w in message_lower for w in ["do", "action", "recommend", "intervention"]):
            intent = IntentType.INTERVENTION
        elif any(w in message_lower for w in ["policy", "rule", "cancel", "no-show policy"]):
            intent = IntentType.POLICY_QUESTION
        elif any(w in message_lower for w in ["hello", "hi", "thanks", "thank"]):
            intent = IntentType.GREETING
        else:
            intent = IntentType.GENERAL_QUESTION
        
        latency = (time.time() - start_time) * 1000
        
        return ChainResult(
            success=True,
            content=f"Classified as {intent.value} (fallback)",
            chain_name="IntentRouterChain",
            metadata={
                "intent": intent.value,
                "confidence": 0.6,
                "entities": {},
                "reasoning": "Fallback keyword matching",
                "fallback": True
            },
            latency_ms=latency
        )
    
    def get_intent(self, message: str) -> IntentType:
        """Quick method to get just the intent."""
        result = self.run(message)
        return IntentType(result.metadata.get("intent", "unknown"))