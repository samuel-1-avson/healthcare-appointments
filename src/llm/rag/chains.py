# src/llm/rag/chains.py
"""
RAG Chains
==========
Complete RAG chains for policy Q&A.

Includes:
- Basic RAG chain
- Conversational RAG with history
- Multi-step RAG
- Citation-aware RAG
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage

from .retriever import PolicyRetriever, create_retriever
from .vector_store import VectorStoreManager, get_vector_store
from ..langchain_config import get_chat_model
from ..memory.conversation_memory import get_memory_manager


logger = logging.getLogger(__name__)


# ==================== Prompts ====================

RAG_SYSTEM_PROMPT = """You are a Healthcare Policy Assistant. Your role is to answer questions about appointment policies, procedures, and guidelines using ONLY the provided context.

Guidelines:
1. Answer based ONLY on the provided policy context
2. If the answer isn't in the context, say "I don't have information about that in the available policies"
3. Be precise and cite specific policy sections when possible
4. Use bullet points for clarity when listing multiple items
5. If a question requires human judgment or is outside policy scope, recommend speaking with a supervisor

Important:
- Do NOT make up policies or procedures
- Do NOT provide medical advice
- Always be helpful and professional"""


RAG_QA_PROMPT = """Answer the question based on the following policy context.

## Policy Context
{context}

## Question
{question}

## Instructions
- Use ONLY information from the context above
- If the answer isn't in the context, say so clearly
- Quote relevant policy sections when helpful
- Be concise but complete

Answer:"""


RAG_CONVERSATIONAL_PROMPT = """You are a Healthcare Policy Assistant engaged in a conversation. 
Use the chat history and the provided policy context to answer the question.

## Policy Context
{context}

## Current Question
{question}

## Instructions
- Consider the conversation history for context
- Answer based on the policy documents provided
- If referring to something from earlier in the conversation, be clear
- If you don't have enough information, ask for clarification

Answer:"""


QUERY_REWRITE_PROMPT = """Given the conversation history and the latest question, 
rewrite the question to be a standalone question that captures all necessary context.

Chat History:
{chat_history}

Latest Question: {question}

Standalone Question:"""


CITATION_PROMPT = """Answer the question using the provided policy documents.
Include citations in your answer using [1], [2], etc. format.

## Documents
{numbered_context}

## Question
{question}

Provide your answer with citations:"""


# ==================== RAG Chain ====================

class RAGChain:
    """
    Basic RAG chain for policy Q&A.
    
    Flow:
    1. Receive question
    2. Retrieve relevant documents
    3. Generate answer using context
    
    Example
    -------
    >>> rag = RAGChain(vector_store_manager)
    >>> answer = rag.ask("What is the cancellation policy?")
    """
    
    def __init__(
        self,
        vector_store: Optional[VectorStoreManager] = None,
        model_name: Optional[str] = None,
        temperature: float = 0.2,
        retriever_k: int = 4
    ):
        """
        Initialize RAG chain.
        
        Parameters
        ----------
        vector_store : VectorStoreManager, optional
            Vector store for retrieval
        model_name : str, optional
            LLM model name
        temperature : float
            Generation temperature
        retriever_k : int
            Number of documents to retrieve
        """
        self.vector_store = vector_store or get_vector_store()
        self.model = get_chat_model(model_name, temperature)
        self.retriever = create_retriever(
            vector_store=self.vector_store,
            top_k=retriever_k
        )
        
        self._chain = self._build_chain()
        
        # Statistics
        self._stats = {
            "queries": 0,
            "avg_context_length": 0,
            "avg_response_time_ms": 0
        }
        
        logger.info("RAGChain initialized")
    
    def _build_chain(self):
        """Build the RAG chain using LCEL."""
        
        # Prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", RAG_SYSTEM_PROMPT),
            ("human", RAG_QA_PROMPT)
        ])
        
        # Chain with retrieval
        chain = (
            RunnableParallel(
                context=self.retriever | self._format_docs,
                question=RunnablePassthrough()
            )
            | prompt
            | self.model
            | StrOutputParser()
        )
        
        return chain
    
    def _format_docs(self, docs: List[Document]) -> str:
        """Format documents into context string."""
        if not docs:
            return "No relevant policy documents found."
        
        formatted = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get("filename", "Unknown")
            section = doc.metadata.get("section", "")
            header = doc.metadata.get("header", "")
            
            header_info = f" - {header}" if header else ""
            section_info = f"\nSection: {section}" if section else ""
            
            formatted.append(
                f"[Document {i+1}: {source}{header_info}]{section_info}\n{doc.page_content}"
            )
        
        return "\n\n---\n\n".join(formatted)
    
    def ask(
        self,
        question: str,
        return_sources: bool = False
    ) -> Dict[str, Any]:
        """
        Ask a question.
        
        Parameters
        ----------
        question : str
            User question
        return_sources : bool
            Whether to return source documents
        
        Returns
        -------
        dict
            Answer with optional sources
        """
        start_time = datetime.utcnow()
        
        # Get retrieved documents for sources
        docs = self.retriever.invoke(question)
        
        # Get answer
        answer = self._chain.invoke(question)
        
        # Calculate stats
        latency_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        self._stats["queries"] += 1
        
        result = {
            "answer": answer,
            "question": question,
            "metadata": {
                "latency_ms": latency_ms,
                "documents_retrieved": len(docs),
                "timestamp": datetime.utcnow().isoformat()
            }
        }
        
        if return_sources:
            result["sources"] = [
                {
                    "content": doc.page_content[:200] + "...",
                    "source": doc.metadata.get("source", "Unknown"),
                    "section": doc.metadata.get("section", "")
                }
                for doc in docs
            ]
        
        return result
    
    def invoke(self, question: str) -> str:
        """Simple invoke returning just the answer."""
        return self.ask(question)["answer"]
    
    async def ainvoke(self, question: str) -> str:
        """Async invoke."""
        return await self._chain.ainvoke(question)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get chain statistics."""
        return self._stats


# ==================== Conversational RAG ====================

class ConversationalRAGChain:
    """
    RAG chain with conversation history.
    
    Maintains context across multiple questions and
    rewrites follow-up questions to be standalone.
    
    Example
    -------
    >>> rag = ConversationalRAGChain(vector_store)
    >>> session = rag.create_session()
    >>> answer1 = rag.ask(session, "What's the cancellation policy?")
    >>> answer2 = rag.ask(session, "What if I cancel same-day?")  # Uses context
    """
    
    def __init__(
        self,
        vector_store: Optional[VectorStoreManager] = None,
        model_name: Optional[str] = None,
        temperature: float = 0.2,
        retriever_k: int = 4,
        max_history: int = 5
    ):
        """
        Initialize conversational RAG.
        
        Parameters
        ----------
        vector_store : VectorStoreManager
            Vector store for retrieval
        model_name : str, optional
            LLM model name
        temperature : float
            Generation temperature
        retriever_k : int
            Documents to retrieve
        max_history : int
            Max conversation turns to keep
        """
        self.vector_store = vector_store or get_vector_store()
        self.model = get_chat_model(model_name, temperature)
        self.retriever = create_retriever(
            vector_store=self.vector_store,
            top_k=retriever_k
        )
        self.max_history = max_history
        
        # Memory manager
        self.memory_manager = get_memory_manager()
        
        # Build chains
        self._query_rewriter = self._build_query_rewriter()
        self._rag_chain = self._build_rag_chain()
        
        logger.info("ConversationalRAGChain initialized")
    
    def _build_query_rewriter(self):
        """Build chain to rewrite questions with context."""
        prompt = PromptTemplate.from_template(QUERY_REWRITE_PROMPT)
        
        return prompt | self.model | StrOutputParser()
    
    def _build_rag_chain(self):
        """Build the main RAG chain."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", RAG_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", RAG_CONVERSATIONAL_PROMPT)
        ])
        
        return prompt | self.model | StrOutputParser()
    
    def _format_history(self, messages: List) -> str:
        """Format message history as string."""
        formatted = []
        for msg in messages[-self.max_history * 2:]:  # Last N exchanges
            role = "Human" if isinstance(msg, HumanMessage) else "Assistant"
            formatted.append(f"{role}: {msg.content}")
        return "\n".join(formatted)
    
    def _format_docs(self, docs: List[Document]) -> str:
        """Format documents for context."""
        if not docs:
            return "No relevant policy documents found."
        
        parts = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get("filename", "Unknown")
            parts.append(f"[{i+1}] {source}:\n{doc.page_content}")
        
        return "\n\n".join(parts)
    
    def create_session(self, session_id: Optional[str] = None) -> str:
        """Create a new conversation session."""
        return self.memory_manager.create_session(session_id)
    
    def ask(
        self,
        session_id: str,
        question: str,
        return_sources: bool = False
    ) -> Dict[str, Any]:
        """
        Ask a question in a conversation.
        
        Parameters
        ----------
        session_id : str
            Conversation session ID
        question : str
            User question
        return_sources : bool
            Whether to return sources
        
        Returns
        -------
        dict
            Answer with metadata
        """
        start_time = datetime.utcnow()
        
        # Get conversation history
        self.memory_manager.get_or_create_session(session_id)
        history = self.memory_manager.get_conversation_messages(session_id)
        
        # Rewrite question if there's history
        if history:
            history_str = self._format_history(history)
            standalone_question = self._query_rewriter.invoke({
                "chat_history": history_str,
                "question": question
            })
            logger.debug(f"Rewritten question: {standalone_question}")
        else:
            standalone_question = question
        
        # Retrieve documents
        docs = self.retriever.invoke(standalone_question)
        context = self._format_docs(docs)
        
        # Generate answer
        answer = self._rag_chain.invoke({
            "context": context,
            "question": question,
            "chat_history": history[-self.max_history * 2:]
        })
        
        # Save to history
        self.memory_manager.add_exchange(session_id, question, answer)
        
        # Build result
        latency_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        result = {
            "answer": answer,
            "question": question,
            "standalone_question": standalone_question if history else None,
            "session_id": session_id,
            "metadata": {
                "latency_ms": latency_ms,
                "documents_retrieved": len(docs),
                "conversation_turns": len(history) // 2 + 1,
                "timestamp": datetime.utcnow().isoformat()
            }
        }
        
        if return_sources:
            result["sources"] = [
                {
                    "content": doc.page_content[:200] + "...",
                    "source": doc.metadata.get("source", "Unknown")
                }
                for doc in docs
            ]
        
        return result
    
    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        """Get conversation history."""
        messages = self.memory_manager.get_conversation_messages(session_id)
        return [
            {
                "role": "user" if isinstance(m, HumanMessage) else "assistant",
                "content": m.content
            }
            for m in messages
        ]
    
    def clear_history(self, session_id: str):
        """Clear conversation history."""
        self.memory_manager.clear_session(session_id)


# ==================== Citation RAG ====================

class CitationRAGChain:
    """
    RAG chain that includes citations in responses.
    
    Provides numbered citations linking to source documents.
    """
    
    def __init__(
        self,
        vector_store: Optional[VectorStoreManager] = None,
        model_name: Optional[str] = None,
        retriever_k: int = 5
    ):
        self.vector_store = vector_store or get_vector_store()
        self.model = get_chat_model(model_name, temperature=0.1)
        self.retriever = create_retriever(
            vector_store=self.vector_store,
            top_k=retriever_k
        )
        
        self._chain = self._build_chain()
    
    def _build_chain(self):
        """Build citation-aware chain."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", RAG_SYSTEM_PROMPT),
            ("human", CITATION_PROMPT)
        ])
        
        return prompt | self.model | StrOutputParser()
    
    def _format_numbered_context(self, docs: List[Document]) -> Tuple[str, List[Dict]]:
        """Format documents with numbers for citation."""
        formatted = []
        sources = []
        
        for i, doc in enumerate(docs):
            source_info = {
                "number": i + 1,
                "source": doc.metadata.get("source", "Unknown"),
                "filename": doc.metadata.get("filename", "Unknown"),
                "section": doc.metadata.get("section", ""),
                "content_preview": doc.page_content[:100]
            }
            sources.append(source_info)
            
            formatted.append(
                f"[{i+1}] Source: {source_info['filename']}\n"
                f"Section: {source_info['section']}\n"
                f"Content: {doc.page_content}"
            )
        
        return "\n\n".join(formatted), sources
    
    def ask(self, question: str) -> Dict[str, Any]:
        """Ask with citations."""
        # Retrieve
        docs = self.retriever.invoke(question)
        numbered_context, sources = self._format_numbered_context(docs)
        
        # Generate
        answer = self._chain.invoke({
            "numbered_context": numbered_context,
            "question": question
        })
        
        return {
            "answer": answer,
            "citations": sources,
            "question": question
        }


# ==================== Factory Functions ====================

def create_rag_chain(
    vector_store: Optional[VectorStoreManager] = None,
    conversational: bool = False,
    with_citations: bool = False,
    **kwargs
):
    """
    Create a RAG chain.
    
    Parameters
    ----------
    vector_store : VectorStoreManager, optional
        Vector store to use
    conversational : bool
        Use conversational RAG with history
    with_citations : bool
        Include citations in responses
    **kwargs
        Additional chain parameters
    
    Returns
    -------
    RAG chain instance
    """
    if with_citations:
        return CitationRAGChain(vector_store, **kwargs)
    elif conversational:
        return ConversationalRAGChain(vector_store, **kwargs)
    else:
        return RAGChain(vector_store, **kwargs)