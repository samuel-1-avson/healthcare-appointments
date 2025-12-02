# src/llm/rag/__init__.py
"""
RAG (Retrieval-Augmented Generation) Module
============================================

Components for building the policy Q&A system:
- Document loading and processing
- Text chunking
- Embeddings
- Vector store (FAISS)
- Retrieval chains
"""

from .document_loader import (
    DocumentLoader,
    load_policy_documents,
    load_document
)
from .chunking import (
    TextChunker,
    ChunkingStrategy,
    create_chunks
)
from .embeddings import (
    EmbeddingsManager,
    get_embeddings
)
from .vector_store import (
    VectorStoreManager,
    get_vector_store,
    create_vector_store
)
from .retriever import (
    PolicyRetriever,
    create_retriever
)

__all__ = [
    "DocumentLoader",
    "load_policy_documents",
    "load_document",
    "TextChunker",
    "ChunkingStrategy",
    "create_chunks",
    "EmbeddingsManager",
    "get_embeddings",
    "VectorStoreManager",
    "get_vector_store",
    "create_vector_store",
    "PolicyRetriever",
    "create_retriever"
]