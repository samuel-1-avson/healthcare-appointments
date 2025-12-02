# src/llm/memory/conversation_memory.py
"""
Conversation Memory
===================
Memory management for healthcare assistant conversations.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import uuid
import json

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage
)
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.memory import (
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
    ConversationSummaryMemory,
    ConversationSummaryBufferMemory
)

from ..langchain_config import get_langchain_settings, get_chat_model


logger = logging.getLogger(__name__)


# ==================== Custom Message History ====================

@dataclass
class ConversationTurn:
    """A single conversation turn."""
    role: str
    content: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


class InMemoryHistory(BaseChatMessageHistory):
    """
    In-memory chat history with metadata tracking.
    
    For production, replace with Redis or database-backed storage.
    """
    
    def __init__(self, session_id: str, max_messages: int = 50):
        self.session_id = session_id
        self.max_messages = max_messages
        self._messages: List[BaseMessage] = []
        self._metadata: List[Dict] = []
        self.created_at = datetime.utcnow()
        self.last_activity = datetime.utcnow()
    
    @property
    def messages(self) -> List[BaseMessage]:
        """Get all messages."""
        return self._messages
    
    def add_message(self, message: BaseMessage) -> None:
        """Add a message."""
        self._messages.append(message)
        self._metadata.append({
            "timestamp": datetime.utcnow().isoformat(),
            "type": type(message).__name__
        })
        self.last_activity = datetime.utcnow()
        
        # Trim if exceeds max
        if len(self._messages) > self.max_messages:
            self._messages = self._messages[-self.max_messages:]
            self._metadata = self._metadata[-self.max_messages:]
    
    def add_user_message(self, message: str) -> None:
        """Add a user message."""
        self.add_message(HumanMessage(content=message))
    
    def add_ai_message(self, message: str) -> None:
        """Add an AI message."""
        self.add_message(AIMessage(content=message))
    
    def clear(self) -> None:
        """Clear history."""
        self._messages = []
        self._metadata = []
    
    def get_summary(self) -> Dict[str, Any]:
        """Get history summary."""
        return {
            "session_id": self.session_id,
            "message_count": len(self._messages),
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "human_messages": sum(1 for m in self._messages if isinstance(m, HumanMessage)),
            "ai_messages": sum(1 for m in self._messages if isinstance(m, AIMessage))
        }


# ==================== Memory Manager ====================

class ConversationMemoryManager:
    """
    Manager for conversation memories across sessions.
    
    Handles:
    - Session creation and retrieval
    - Memory cleanup
    - Session persistence (in production, use Redis/DB)
    
    Example
    -------
    >>> manager = ConversationMemoryManager()
    >>> session_id = manager.create_session()
    >>> memory = manager.get_memory(session_id)
    """
    
    def __init__(
        self,
        memory_type: str = "buffer_window",
        max_messages: int = 10,
        session_timeout_minutes: int = 60
    ):
        """
        Initialize memory manager.
        
        Parameters
        ----------
        memory_type : str
            Type of memory: buffer, buffer_window, summary
        max_messages : int
            Maximum messages to retain (for buffer_window)
        session_timeout_minutes : int
            Session timeout for cleanup
        """
        self.memory_type = memory_type
        self.max_messages = max_messages
        self.session_timeout = timedelta(minutes=session_timeout_minutes)
        
        # Session storage (in production: use Redis)
        self._sessions: Dict[str, InMemoryHistory] = {}
        self._memories: Dict[str, Any] = {}
        
        logger.info(f"ConversationMemoryManager initialized with {memory_type} memory")
    
    def create_session(self, session_id: Optional[str] = None) -> str:
        """
        Create a new conversation session.
        
        Parameters
        ----------
        session_id : str, optional
            Custom session ID (auto-generated if not provided)
        
        Returns
        -------
        str
            Session ID
        """
        session_id = session_id or str(uuid.uuid4())
        
        # Create history
        history = InMemoryHistory(
            session_id=session_id,
            max_messages=self.max_messages * 2  # Store more than window size
        )
        self._sessions[session_id] = history
        
        # Create memory
        memory = self._create_memory(history)
        self._memories[session_id] = memory
        
        logger.info(f"Created session: {session_id}")
        return session_id
    
    def _create_memory(self, history: InMemoryHistory):
        """Create appropriate memory type."""
        settings = get_langchain_settings()
        
        if self.memory_type == "buffer":
            return ConversationBufferMemory(
                chat_memory=history,
                return_messages=True,
                memory_key="chat_history"
            )
        
        elif self.memory_type == "buffer_window":
            return ConversationBufferWindowMemory(
                chat_memory=history,
                k=self.max_messages,
                return_messages=True,
                memory_key="chat_history"
            )
        
        elif self.memory_type == "summary":
            llm = get_chat_model(temperature=0)
            return ConversationSummaryMemory(
                llm=llm,
                chat_memory=history,
                return_messages=True,
                memory_key="chat_history"
            )
        
        elif self.memory_type == "summary_buffer":
            llm = get_chat_model(temperature=0)
            return ConversationSummaryBufferMemory(
                llm=llm,
                chat_memory=history,
                max_token_limit=1000,
                return_messages=True,
                memory_key="chat_history"
            )
        
        else:
            raise ValueError(f"Unknown memory type: {self.memory_type}")
    
    def get_memory(self, session_id: str):
        """
        Get memory for a session.
        
        Parameters
        ----------
        session_id : str
            Session ID
        
        Returns
        -------
        Memory
            LangChain memory instance
        
        Raises
        ------
        KeyError
            If session doesn't exist
        """
        if session_id not in self._sessions:
            raise KeyError(f"Session not found: {session_id}")
        
        return self._memories[session_id]
    
    def get_history(self, session_id: str) -> InMemoryHistory:
        """Get raw history for a session."""
        if session_id not in self._sessions:
            raise KeyError(f"Session not found: {session_id}")
        
        return self._sessions[session_id]
    
    def get_or_create_session(self, session_id: str) -> str:
        """Get existing session or create new one."""
        if session_id not in self._sessions:
            return self.create_session(session_id)
        return session_id
    
    def add_exchange(
        self,
        session_id: str,
        user_message: str,
        ai_message: str
    ) -> None:
        """
        Add a user-AI exchange to memory.
        
        Parameters
        ----------
        session_id : str
            Session ID
        user_message : str
            User's message
        ai_message : str
            AI's response
        """
        if session_id not in self._sessions:
            raise KeyError(f"Session not found: {session_id}")
        
        memory = self._memories[session_id]
        memory.save_context(
            {"input": user_message},
            {"output": ai_message}
        )
    
    def get_conversation_messages(
        self,
        session_id: str
    ) -> List[BaseMessage]:
        """Get all messages for a session."""
        if session_id not in self._sessions:
            return []
        
        return self._sessions[session_id].messages
    
    def clear_session(self, session_id: str) -> None:
        """Clear a session's memory."""
        if session_id in self._sessions:
            self._sessions[session_id].clear()
            self._memories[session_id].clear()
            logger.info(f"Cleared session: {session_id}")
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session entirely."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            del self._memories[session_id]
            logger.info(f"Deleted session: {session_id}")
            return True
        return False
    
    def cleanup_expired(self) -> int:
        """
        Remove expired sessions.
        
        Returns
        -------
        int
            Number of sessions removed
        """
        now = datetime.utcnow()
        expired = []
        
        for session_id, history in self._sessions.items():
            if now - history.last_activity > self.session_timeout:
                expired.append(session_id)
        
        for session_id in expired:
            self.delete_session(session_id)
        
        if expired:
            logger.info(f"Cleaned up {len(expired)} expired sessions")
        
        return len(expired)
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all active sessions."""
        return [
            history.get_summary()
            for history in self._sessions.values()
        ]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory manager statistics."""
        return {
            "active_sessions": len(self._sessions),
            "memory_type": self.memory_type,
            "max_messages_per_session": self.max_messages,
            "session_timeout_minutes": self.session_timeout.total_seconds() / 60
        }


# ==================== Factory Functions ====================

# Global manager instance
_memory_manager: Optional[ConversationMemoryManager] = None


def get_memory_manager() -> ConversationMemoryManager:
    """Get or create the global memory manager."""
    global _memory_manager
    
    if _memory_manager is None:
        settings = get_langchain_settings()
        _memory_manager = ConversationMemoryManager(
            memory_type=settings.langchain.memory_type,
            max_messages=settings.langchain.memory_window_size
        )
    
    return _memory_manager


def create_memory(
    memory_type: str = "buffer_window",
    max_messages: int = 10
):
    """
    Create a standalone memory instance.
    
    Parameters
    ----------
    memory_type : str
        Type: buffer, buffer_window, summary
    max_messages : int
        Max messages to keep
    
    Returns
    -------
    Memory
        LangChain memory instance
    """
    history = InMemoryHistory(
        session_id=str(uuid.uuid4()),
        max_messages=max_messages * 2
    )
    
    if memory_type == "buffer":
        return ConversationBufferMemory(
            chat_memory=history,
            return_messages=True
        )
    elif memory_type == "buffer_window":
        return ConversationBufferWindowMemory(
            chat_memory=history,
            k=max_messages,
            return_messages=True
        )
    else:
        return ConversationBufferMemory(
            chat_memory=history,
            return_messages=True
        )


def get_memory_for_session(session_id: str):
    """Get or create memory for a session."""
    manager = get_memory_manager()
    manager.get_or_create_session(session_id)
    return manager.get_memory(session_id)