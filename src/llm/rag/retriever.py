# src/llm/rag/retriever.py
"""
Policy Retriever
================
Advanced retrieval for policy documents.

Features:
- Multi-query retrieval
- Contextual compression
- Hybrid search
- Query transformation
"""

import logging
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from .vector_store import VectorStoreManager, get_vector_store
from ..langchain_config import get_chat_model


logger = logging.getLogger(__name__)


@dataclass
class RetrievalConfig:
    """Configuration for retrieval."""
    
    # Basic search
    top_k: int = 4
    search_type: str = "similarity"  # similarity, mmr
    
    # MMR settings
    mmr_diversity: float = 0.5
    mmr_fetch_k: int = 20
    
    # Score threshold
    score_threshold: Optional[float] = None
    
    # Query expansion
    use_query_expansion: bool = False
    expansion_count: int = 3
    
    # Reranking
    use_reranking: bool = False
    rerank_top_k: int = 10
    
    # Context
    include_metadata: bool = True


class PolicyRetriever(BaseRetriever):
    """
    Advanced retriever for policy documents.
    
    Supports:
    - Basic similarity search
    - MMR (diverse results)
    - Query expansion
    - Contextual compression
    
    Example
    -------
    >>> retriever = PolicyRetriever(vector_store_manager)
    >>> docs = retriever.invoke("What's the cancellation policy?")
    """
    
    vector_store: VectorStoreManager
    config: RetrievalConfig = None
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(
        self,
        vector_store: VectorStoreManager,
        config: Optional[RetrievalConfig] = None,
        **kwargs
    ):
        super().__init__(
            vector_store=vector_store,
            config=config or RetrievalConfig(),
            **kwargs
        )
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """
        Retrieve relevant documents.
        
        Parameters
        ----------
        query : str
            Search query
        run_manager : CallbackManagerForRetrieverRun
            Callback manager
        
        Returns
        -------
        List[Document]
            Retrieved documents
        """
        # Query expansion
        if self.config.use_query_expansion:
            queries = self._expand_query(query)
            queries.insert(0, query)  # Include original
        else:
            queries = [query]
        
        # Retrieve for each query
        all_docs = []
        seen_contents = set()
        
        for q in queries:
            if self.config.search_type == "mmr":
                docs = self.vector_store.mmr_search(
                    q,
                    k=self.config.top_k,
                    fetch_k=self.config.mmr_fetch_k,
                    lambda_mult=self.config.mmr_diversity
                )
            elif self.config.score_threshold:
                docs = self.vector_store.search(
                    q,
                    k=self.config.top_k,
                    score_threshold=self.config.score_threshold
                )
            else:
                docs = self.vector_store.search(
                    q,
                    k=self.config.top_k
                )
            
            # Deduplicate
            for doc in docs:
                content_hash = hash(doc.page_content)
                if content_hash not in seen_contents:
                    seen_contents.add(content_hash)
                    all_docs.append(doc)
        
        # Limit total results
        all_docs = all_docs[:self.config.top_k * 2]
        
        # Rerank if enabled
        if self.config.use_reranking:
            all_docs = self._rerank(query, all_docs)
        
        # Final limit
        all_docs = all_docs[:self.config.top_k]
        
        self.logger.debug(f"Retrieved {len(all_docs)} documents for: {query[:50]}...")
        
        return all_docs
    
    def _expand_query(self, query: str) -> List[str]:
        """
        Expand query into multiple related queries.
        
        Uses LLM to generate alternative phrasings.
        """
        model = get_chat_model(temperature=0.7)
        
        prompt = PromptTemplate.from_template("""
Generate {count} alternative ways to ask this question about healthcare appointment policies.
Keep them concise and focused on the same topic.

Original question: {query}

Alternative questions (one per line):
""")
        
        chain = prompt | model | StrOutputParser()
        
        try:
            result = chain.invoke({
                "query": query,
                "count": self.config.expansion_count
            })
            
            # Parse alternatives
            alternatives = [
                line.strip().lstrip("0123456789.-) ")
                for line in result.split("\n")
                if line.strip() and not line.strip().startswith("Alternative")
            ]
            
            return alternatives[:self.config.expansion_count]
            
        except Exception as e:
            self.logger.warning(f"Query expansion failed: {e}")
            return []
    
    def _rerank(
        self,
        query: str,
        documents: List[Document]
    ) -> List[Document]:
        """
        Rerank documents using LLM.
        
        Uses the LLM to score relevance of each document.
        """
        if not documents:
            return documents
        
        model = get_chat_model(temperature=0)
        
        prompt = PromptTemplate.from_template("""
Rate the relevance of this document chunk to the question.
Score from 0-10, where 10 is perfectly relevant.

Question: {query}

Document:
{document}

Respond with just the number (0-10):
""")
        
        chain = prompt | model | StrOutputParser()
        
        scored_docs = []
        
        for doc in documents[:self.config.rerank_top_k]:
            try:
                score_str = chain.invoke({
                    "query": query,
                    "document": doc.page_content[:500]
                })
                
                score = float(score_str.strip())
                scored_docs.append((doc, score))
                
            except Exception:
                scored_docs.append((doc, 5.0))  # Default middle score
        
        # Sort by score
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        return [doc for doc, score in scored_docs]
    
    def search(
        self,
        query: str,
        k: Optional[int] = None
    ) -> List[Document]:
        """
        Search for documents.
        
        Convenience method with optional k override.
        """
        original_k = self.config.top_k
        
        if k:
            self.config.top_k = k
        
        try:
            return self.invoke(query)
        finally:
            self.config.top_k = original_k
    
    def search_with_context(
        self,
        query: str,
        k: int = 4
    ) -> Dict[str, Any]:
        """
        Search and return formatted context.
        
        Returns documents with formatted context string.
        """
        docs = self.search(query, k=k)
        
        # Format context
        context_parts = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", "Unknown")
            section = doc.metadata.get("section", "")
            
            context_parts.append(
                f"[Document {i+1}] (Source: {source})\n"
                f"{f'Section: {section}' if section else ''}\n"
                f"{doc.page_content}\n"
            )
        
        return {
            "documents": docs,
            "context": "\n---\n".join(context_parts),
            "sources": list(set(d.metadata.get("source", "Unknown") for d in docs))
        }


# ==================== Factory Functions ====================

def create_retriever(
    vector_store: Optional[VectorStoreManager] = None,
    top_k: int = 4,
    search_type: str = "similarity",
    use_query_expansion: bool = False
) -> PolicyRetriever:
    """
    Create a policy retriever.
    
    Parameters
    ----------
    vector_store : VectorStoreManager, optional
        Vector store to use
    top_k : int
        Number of documents to retrieve
    search_type : str
        Search type: similarity or mmr
    use_query_expansion : bool
        Whether to use query expansion
    
    Returns
    -------
    PolicyRetriever
        Configured retriever
    """
    store = vector_store or get_vector_store()
    
    config = RetrievalConfig(
        top_k=top_k,
        search_type=search_type,
        use_query_expansion=use_query_expansion
    )
    
    return PolicyRetriever(
        vector_store=store,
        config=config
    )