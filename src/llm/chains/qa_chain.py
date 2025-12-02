# src/llm/chains/qa_chain.py
"""
Policy Q&A Chain
================
Chain for answering questions about appointment policies.
(Foundation for RAG in Week 11)
"""

import logging
import time
from typing import Dict, Any, Optional, List

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

from .base import BaseHealthcareChain, ChainResult
from ..tools import get_policy_tool

logger = logging.getLogger(__name__)


QA_SYSTEM_PROMPT = """You are a Healthcare Policy Assistant. You answer questions about 
appointment scheduling, cancellations, no-shows, and related policies.

Guidelines:
1. Answer based ONLY on the provided policy context
2. If the answer isn't in the context, say "I don't have information about that"
3. Be precise and cite policy sections when possible
4. For complex situations, recommend speaking with a supervisor
5. Never make up policies

Current date context: Policies are current as of January 2024."""


QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", QA_SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", """Policy Context:
{context}

Question: {question}

Please provide a clear, accurate answer based on the policy context above.""")
])


class PolicyQAChain(BaseHealthcareChain):
    """
    Chain for answering policy questions.
    
    Currently uses simple policy lookup.
    In Week 11, this will be enhanced with RAG.
    
    Example
    -------
    >>> chain = PolicyQAChain()
    >>> result = chain.run(question="What happens after 3 no-shows?")
    >>> print(result.content)
    """
    
    def __init__(self, use_memory: bool = True, **kwargs):
        self.use_memory = use_memory
        self.policy_tool = get_policy_tool()
        self.chat_history: List = []
        super().__init__(**kwargs)
    
    def _build_chain(self):
        """Build the Q&A chain."""
        return QA_PROMPT | self._llm | StrOutputParser()
    
    def run(
        self,
        question: str,
        context: Optional[str] = None,
        clear_history: bool = False
    ) -> ChainResult:
        """
        Answer a policy question.
        
        Parameters
        ----------
        question : str
            The question to answer
        context : str, optional
            Policy context (if not provided, will search)
        clear_history : bool
            Whether to clear conversation history
        
        Returns
        -------
        ChainResult
            Answer with metadata
        """
        start_time = time.time()
        
        if clear_history:
            self.chat_history = []
        
        try:
            # Get context if not provided
            if context is None:
                context = self.policy_tool.invoke({"query": question})
            
            # Prepare chat history
            history = self.chat_history if self.use_memory else []
            
            # Generate answer
            answer = self._chain.invoke({
                "context": context,
                "question": question,
                "chat_history": history
            })
            
            # Update history
            if self.use_memory:
                self.chat_history.append(HumanMessage(content=question))
                self.chat_history.append(AIMessage(content=answer))
                
                # Keep history manageable
                if len(self.chat_history) > 20:
                    self.chat_history = self.chat_history[-20:]
            
            latency = (time.time() - start_time) * 1000
            
            return ChainResult(
                success=True,
                content=answer,
                chain_name="PolicyQAChain",
                metadata={
                    "question": question,
                    "context_length": len(context),
                    "history_length": len(self.chat_history)
                },
                latency_ms=latency
            )
            
        except Exception as e:
            return self._handle_error(e, "PolicyQAChain")
    
    def clear_history(self):
        """Clear conversation history."""
        self.chat_history = []
    
    def get_history(self) -> List[Dict[str, str]]:
        """Get formatted conversation history."""
        return [
            {"role": "user" if isinstance(m, HumanMessage) else "assistant", 
             "content": m.content}
            for m in self.chat_history
        ]