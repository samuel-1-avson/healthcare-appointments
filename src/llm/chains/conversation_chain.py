# src/llm/chains/conversation_chain.py
"""
Conversation Chain
==================
Main conversational chain with memory for the Healthcare Assistant.
"""

from typing import Dict, Any, Optional, List

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableWithMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from .base import BaseHealthcareChain
from ..memory.conversation_memory import (
    get_memory_manager,
    ConversationMemoryManager,
    InMemoryHistory
)
from ..langchain_config import get_chat_model


# ==================== System Prompt ====================

CONVERSATION_SYSTEM_PROMPT = """You are a Healthcare Appointment Assistant helping healthcare staff manage patient appointments and reduce no-shows.

Your capabilities:
1. EXPLAIN no-show predictions in plain language
2. RECOMMEND interventions based on risk levels
3. ANSWER questions about appointment policies
4. ANALYZE patterns in appointment data
5. PROVIDE communication templates for patient outreach

Guidelines:
- Be helpful, professional, and empathetic
- Focus on practical, actionable advice
- Acknowledge uncertainty when appropriate
- Never provide medical diagnosis or treatment advice
- Protect patient privacy in all responses

When discussing predictions:
- Explain that they're probabilistic, not certain
- Focus on modifiable risk factors
- Suggest supportive interventions, not punitive measures

Current context: You're helping healthcare staff at a clinic. They may ask about specific appointments, policies, or best practices for reducing no-shows."""


# ==================== Chain ====================

class ConversationChain(BaseHealthcareChain):
    """
    Conversational chain with memory for multi-turn interactions.
    
    Supports:
    - Conversation history tracking
    - Context-aware responses
    - Session management
    
    Example
    -------
    >>> chain = ConversationChain()
    >>> session_id = chain.create_session()
    >>> response = chain.chat(session_id, "What is a high-risk patient?")
    >>> response = chain.chat(session_id, "How should I contact them?")
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        temperature: float = 0.4,
        memory_manager: Optional[ConversationMemoryManager] = None,
        **kwargs
    ):
        self.memory_manager = memory_manager or get_memory_manager()
        super().__init__(model_name, temperature, **kwargs)
    
    def _get_system_prompt(self) -> str:
        return CONVERSATION_SYSTEM_PROMPT
    
    def _build_chain(self):
        """Build conversation chain with message history."""
        
        # Prompt with history placeholder
        prompt = ChatPromptTemplate.from_messages([
            ("system", CONVERSATION_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])
        
        # Base chain
        chain = prompt | self.model | StrOutputParser()
        
        # Wrap with message history
        chain_with_history = RunnableWithMessageHistory(
            chain,
            self._get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history"
        )
        
        return chain_with_history
    
    def _get_session_history(self, session_id: str) -> InMemoryHistory:
        """Get history for a session."""
        self.memory_manager.get_or_create_session(session_id)
        return self.memory_manager.get_history(session_id)
    
    def create_session(self, session_id: Optional[str] = None) -> str:
        """Create a new conversation session."""
        return self.memory_manager.create_session(session_id)
    
    def chat(
        self,
        session_id: str,
        message: str,
        **kwargs
    ) -> str:
        """
        Send a message and get a response.
        
        Parameters
        ----------
        session_id : str
            Conversation session ID
        message : str
            User message
        **kwargs
            Additional parameters
        
        Returns
        -------
        str
            Assistant response
        """
        # Ensure session exists
        self.memory_manager.get_or_create_session(session_id)
        
        # Invoke chain with session config
        response = self._chain.invoke(
            {"input": message},
            config={"configurable": {"session_id": session_id}}
        )
        
        return response
    
    async def achat(
        self,
        session_id: str,
        message: str
    ) -> str:
        """Async chat."""
        self.memory_manager.get_or_create_session(session_id)
        
        response = await self._chain.ainvoke(
            {"input": message},
            config={"configurable": {"session_id": session_id}}
        )
        
        return response
    
    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        """
        Get conversation history.
        
        Returns list of {role, content} dicts.
        """
        messages = self.memory_manager.get_conversation_messages(session_id)
        
        history = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                history.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                history.append({"role": "assistant", "content": msg.content})
        
        return history
    
    def clear_history(self, session_id: str) -> None:
        """Clear conversation history."""
        self.memory_manager.clear_session(session_id)
    
    def end_session(self, session_id: str) -> None:
        """End and delete a session."""
        self.memory_manager.delete_session(session_id)


# ==================== Enhanced Conversation Chain ====================

class EnhancedConversationChain(ConversationChain):
    """
    Enhanced conversation chain with:
    - Context injection (current prediction, patient data)
    - Tool awareness
    - Structured responses
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        temperature: float = 0.4,
        **kwargs
    ):
        super().__init__(model_name, temperature, **kwargs)
        
        # Context storage per session
        self._session_context: Dict[str, Dict[str, Any]] = {}
    
    def set_context(
        self,
        session_id: str,
        prediction: Optional[Dict[str, Any]] = None,
        patient_data: Optional[Dict[str, Any]] = None,
        **extra_context
    ) -> None:
        """
        Set context for a session.
        
        This context will be available to the LLM for more relevant responses.
        
        Parameters
        ----------
        session_id : str
            Session ID
        prediction : dict, optional
            Current prediction result
        patient_data : dict, optional
            Patient/appointment data
        **extra_context
            Additional context
        """
        self._session_context[session_id] = {
            "prediction": prediction,
            "patient_data": patient_data,
            **extra_context
        }
    
    def chat_with_context(
        self,
        session_id: str,
        message: str
    ) -> str:
        """Chat with session context included."""
        context = self._session_context.get(session_id, {})
        
        # Build context string
        context_parts = []
        
        if context.get("prediction"):
            pred = context["prediction"]
            context_parts.append(
                f"Current Prediction: {pred.get('risk', {}).get('tier', 'Unknown')} risk "
                f"({pred.get('probability', 0):.0%} probability)"
            )
        
        if context.get("patient_data"):
            patient = context["patient_data"]
            context_parts.append(
                f"Patient: {patient.get('age', '?')} years old, "
                f"appointment in {patient.get('lead_days', '?')} days"
            )
        
        # Inject context into message
        if context_parts:
            enhanced_message = f"[Context: {'; '.join(context_parts)}]\n\n{message}"
        else:
            enhanced_message = message
        
        return self.chat(session_id, enhanced_message)
    
    def get_context(self, session_id: str) -> Dict[str, Any]:
        """Get current context for a session."""
        return self._session_context.get(session_id, {})
    
    def clear_context(self, session_id: str) -> None:
        """Clear context for a session."""
        if session_id in self._session_context:
            del self._session_context[session_id]