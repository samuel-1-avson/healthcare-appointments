# src/llm/chains/base.py
"""
Base Chain Classes
==================
Foundation classes for healthcare chains.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from datetime import datetime

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from ..langchain_config import get_chat_model


logger = logging.getLogger(__name__)


class BaseHealthcareChain(ABC):
    """
    Abstract base class for healthcare chains.
    
    Provides common functionality:
    - Model initialization
    - Logging and tracing
    - Error handling
    - Metadata tracking
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        temperature: float = 0.3,
        verbose: bool = False
    ):
        """
        Initialize the chain.
        
        Parameters
        ----------
        model_name : str, optional
            LLM model to use
        temperature : float
            Sampling temperature
        verbose : bool
            Enable verbose logging
        """
        self.model = get_chat_model(model_name, temperature)
        self.verbose = verbose
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Tracking
        self._call_count = 0
        self._total_tokens = 0
        self._created_at = datetime.utcnow()
        
        # Build the chain
        self._chain = self._build_chain()
        
        self.logger.info(f"{self.__class__.__name__} initialized with {model_name or 'default model'}")
    
    @abstractmethod
    def _build_chain(self):
        """Build the LangChain chain. Override in subclasses."""
        pass
    
    @abstractmethod
    def _get_system_prompt(self) -> str:
        """Get the system prompt. Override in subclasses."""
        pass
    
    def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Invoke the chain.
        
        Parameters
        ----------
        inputs : dict
            Input variables for the chain
        
        Returns
        -------
        dict
            Chain output with metadata
        """
        start_time = datetime.utcnow()
        
        try:
            result = self._chain.invoke(inputs)
            
            # Track usage
            self._call_count += 1
            
            return {
                "output": result,
                "metadata": {
                    "chain": self.__class__.__name__,
                    "latency_ms": (datetime.utcnow() - start_time).total_seconds() * 1000,
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
        
        except Exception as e:
            self.logger.error(f"Chain invocation failed: {e}")
            raise
    
    async def ainvoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Async invoke."""
        start_time = datetime.utcnow()
        
        result = await self._chain.ainvoke(inputs)
        self._call_count += 1
        
        return {
            "output": result,
            "metadata": {
                "chain": self.__class__.__name__,
                "latency_ms": (datetime.utcnow() - start_time).total_seconds() * 1000,
                "timestamp": datetime.utcnow().isoformat()
            }
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get chain statistics."""
        return {
            "chain_name": self.__class__.__name__,
            "call_count": self._call_count,
            "created_at": self._created_at.isoformat(),
            "uptime_seconds": (datetime.utcnow() - self._created_at).total_seconds()
        }