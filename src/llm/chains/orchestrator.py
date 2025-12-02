# src/llm/chains/orchestrator.py
"""
Healthcare Chain Orchestrator
=============================
Orchestrates multiple chains for complex healthcare workflows.
"""

import logging
from typing import Dict, Any, Optional, List, Literal
from datetime import datetime
from enum import Enum

from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableBranch
from langchain_core.output_parsers import StrOutputParser

from .explanation_chain import RiskExplanationChain
from .intervention_chain import InterventionChain
from .conversation_chain import ConversationChain, EnhancedConversationChain
from ..langchain_config import get_chat_model
from ..tools.prediction_tool import PredictionTool


logger = logging.getLogger(__name__)


class IntentType(str, Enum):
    """User intent types."""
    PREDICT = "predict"
    EXPLAIN = "explain"
    INTERVENE = "intervene"
    POLICY = "policy"
    GENERAL = "general"


class HealthcareOrchestrator:
    """
    Orchestrates healthcare chains based on user intent.
    
    Routes requests to appropriate chains:
    - Prediction requests -> PredictionTool + ExplanationChain
    - Intervention requests -> InterventionChain
    - Policy questions -> PolicyQA (RAG in Week 11)
    - General chat -> ConversationChain
    
    Example
    -------
    >>> orchestrator = HealthcareOrchestrator()
    >>> result = orchestrator.process(
    ...     "What's the risk for a 35-year-old patient scheduled in 2 weeks?",
    ...     session_id="user-123"
    ... )
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        temperature: float = 0.3
    ):
        """Initialize the orchestrator with all chains."""
        self.model = get_chat_model(model_name, temperature)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize chains
        self.explanation_chain = RiskExplanationChain(model_name, temperature)
        self.intervention_chain = InterventionChain(model_name, temperature)
        self.conversation_chain = EnhancedConversationChain(model_name, temperature + 0.1)
        
        # Initialize tools
        self.prediction_tool = PredictionTool()
        
        # Build intent classifier
        self._intent_classifier = self._build_intent_classifier()
        
        # Tracking
        self._request_count = 0
        self._intent_distribution: Dict[str, int] = {}
        
        self.logger.info("HealthcareOrchestrator initialized")
    
    def _build_intent_classifier(self):
        """Build a chain for classifying user intent."""
        
        intent_prompt = """Classify the user's intent into one of these categories:

Categories:
- PREDICT: User wants a no-show prediction (provides patient data like age, appointment date)
- EXPLAIN: User wants explanation of a prediction or risk factors
- INTERVENE: User wants intervention recommendations
- POLICY: User asks about policies, procedures, or rules
- GENERAL: General conversation, greetings, or other

User message: {message}

Respond with ONLY the category name (PREDICT, EXPLAIN, INTERVENE, POLICY, or GENERAL)."""
        
        from langchain_core.prompts import PromptTemplate
        prompt = PromptTemplate.from_template(intent_prompt)
        
        chain = prompt | self.model | StrOutputParser() | self._parse_intent
        
        return chain
    
    def _parse_intent(self, response: str) -> IntentType:
        """Parse intent from classifier response."""
        response = response.strip().upper()
        
        intent_map = {
            "PREDICT": IntentType.PREDICT,
            "EXPLAIN": IntentType.EXPLAIN,
            "INTERVENE": IntentType.INTERVENE,
            "POLICY": IntentType.POLICY,
            "GENERAL": IntentType.GENERAL
        }
        
        return intent_map.get(response, IntentType.GENERAL)
    
    def classify_intent(self, message: str) -> IntentType:
        """Classify user intent."""
        return self._intent_classifier.invoke({"message": message})
    
    def process(
        self,
        message: str,
        session_id: str,
        context: Optional[Dict[str, Any]] = None,
        force_intent: Optional[IntentType] = None
    ) -> Dict[str, Any]:
        """
        Process a user message through the appropriate chain.
        
        Parameters
        ----------
        message : str
            User message
        session_id : str
            Session ID for conversation tracking
        context : dict, optional
            Additional context (prediction data, patient info)
        force_intent : IntentType, optional
            Force a specific intent (skip classification)
        
        Returns
        -------
        dict
            Response with output, intent, and metadata
        """
        start_time = datetime.utcnow()
        self._request_count += 1
        
        try:
            # Classify intent
            if force_intent:
                intent = force_intent
            else:
                intent = self.classify_intent(message)
            
            # Track intent distribution
            self._intent_distribution[intent.value] = \
                self._intent_distribution.get(intent.value, 0) + 1
            
            self.logger.info(f"Processing message with intent: {intent.value}")
            
            if intent == IntentType.PREDICT:
                output = self._handle_prediction(message, context)
            elif intent == IntentType.EXPLAIN:
                output = self._handle_explanation(message, context)
            elif intent == IntentType.INTERVENE:
                output = self._handle_intervention(message, context)
            elif intent == IntentType.POLICY:
                output = self._handle_policy(message, session_id)
            else:
                output = self._handle_general(message, session_id)
            
            return {
                "output": output,
                "intent": intent.value,
                "session_id": session_id,
                "metadata": {
                    "latency_ms": (datetime.utcnow() - start_time).total_seconds() * 1000,
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Processing error: {e}")
            # Default to GENERAL intent if classification failed
            intent_val = intent.value if 'intent' in locals() else IntentType.GENERAL.value
            
            return {
                "output": f"I encountered an error processing your request: {str(e)}. Please check if the LLM provider is running and accessible.",
                "intent": intent_val,
                "error": str(e),
                "session_id": session_id
            }
    
    def _handle_prediction(
        self,
        message: str,
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Handle prediction requests."""
        
        # If context has prediction data, use it
        if context and "prediction" in context:
            return self._explain_existing_prediction(context)
        
        # Otherwise, guide user to provide data
        return """To make a prediction, I need some patient information:

**Required:**
- Patient age
- Days until appointment (lead time)
- Gender (M/F)

**Optional (improves accuracy):**
- SMS reminder enabled?
- Previous no-show history
- Medical conditions

Please provide these details, or use our prediction API directly at `/api/v1/predict`."""
    
    def _handle_explanation(
        self,
        message: str,
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Handle explanation requests."""
        
        if context and "prediction" in context and "patient_data" in context:
            result = self.explanation_chain.explain(
                probability=context["prediction"].get("probability", 0),
                risk_tier=context["prediction"].get("risk", {}).get("tier", "UNKNOWN"),
                patient_data=context["patient_data"],
                model_factors=context.get("model_factors"),
                patient_history=context.get("patient_history")
            )
            return result
        
        # General explanation
        return self.conversation_chain.chat(
            session_id="system",
            message=message
        )
    
    def _handle_intervention(
        self,
        message: str,
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Handle intervention requests."""
        
        if context and "prediction" in context:
            return self.intervention_chain.recommend(
                probability=context["prediction"].get("probability", 0.5),
                risk_tier=context["prediction"].get("risk", {}).get("tier", "MEDIUM"),
                patient_data=context.get("patient_data", {}),
                constraints=context.get("constraints")
            )
        
        # Generic intervention guidance
        return """I can provide intervention recommendations when you share:

1. **Risk Level**: HIGH, MEDIUM, or LOW
2. **Days until appointment**
3. **Patient history** (optional)

For batch prioritization, share multiple appointments and I'll help you triage them."""
    
    def _handle_policy(
        self,
        message: str,
        session_id: str
    ) -> str:
        """Handle policy questions."""
        # In Week 11, this will use RAG
        return self.conversation_chain.chat(session_id, message)
    
    def _handle_general(
        self,
        message: str,
        session_id: str
    ) -> str:
        """Handle general conversation."""
        return self.conversation_chain.chat(session_id, message)
    
    def _explain_existing_prediction(self, context: Dict[str, Any]) -> str:
        """Explain an existing prediction from context."""
        prediction = context["prediction"]
        patient_data = context.get("patient_data", {})
        
        return self.explanation_chain.explain(
            probability=prediction.get("probability", 0),
            risk_tier=prediction.get("risk", {}).get("tier", "UNKNOWN"),
            patient_data=patient_data
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get orchestrator statistics."""
        return {
            "total_requests": self._request_count,
            "intent_distribution": self._intent_distribution,
            "chains": {
                "explanation": self.explanation_chain.get_stats(),
                "intervention": self.intervention_chain.get_stats(),
                "conversation": self.conversation_chain.get_stats()
            }
        }