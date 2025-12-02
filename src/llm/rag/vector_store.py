# src/llm/rag/vector_store.py
"""
Vector Store Management
=======================
Store and retrieve document embeddings.

Supports:
- FAISS (local, fast)
- Chroma (local, persistent)
- Pinecone (cloud, scalable)
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Literal
from pathlib import Path
import json
from datetime import datetime

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores import FAISS

from .embeddings import EmbeddingsManager, get_embeddings
from .chunking import TextChunker, create_chunks


logger = logging.getLogger(__name__)


class VectorStoreManager:
    """
    Manage vector stores for RAG.
    
    Features:
    - Create and load vector stores
    - Add/remove documents
    - Similarity search
    - Persistence
    
    Example
    -------
    >>> manager = VectorStoreManager()
    >>> manager.create_from_documents(documents)
    >>> results = manager.search("What is the cancellation policy?")
    """
    
    def __init__(
        self,
        store_type: Literal["faiss", "chroma"] = "faiss",
        embeddings_manager: Optional[EmbeddingsManager] = None,
        persist_directory: Optional[str] = None
    ):
        """
        Initialize vector store manager.
        
        Parameters
        ----------
        store_type : str
            Type of vector store
        embeddings_manager : EmbeddingsManager, optional
            Custom embeddings manager
        persist_directory : str, optional
            Directory for persistence
        """
        self.store_type = store_type
        self.embeddings_manager = embeddings_manager or get_embeddings()
        self.persist_directory = Path(persist_directory) if persist_directory else Path("data/vector_store")
        
        self._vector_store: Optional[VectorStore] = None
        self._metadata: Dict[str, Any] = {}
        
        # Ensure directory exists
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"VectorStoreManager initialized with {store_type}")
    
    def create_from_documents(
        self,
        documents: List[Document],
        chunk: bool = True,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ) -> VectorStore:
        """
        Create vector store from documents.
        
        Parameters
        ----------
        documents : List[Document]
            Documents to index
        chunk : bool
            Whether to chunk documents
        chunk_size : int
            Size of chunks
        chunk_overlap : int
            Overlap between chunks
        
        Returns
        -------
        VectorStore
            Created vector store
        """
        # Chunk if needed
        if chunk:
            logger.info(f"Chunking {len(documents)} documents...")
            chunks = create_chunks(
                documents,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            logger.info(f"Created {len(chunks)} chunks")
        else:
            chunks = documents
        
        # Create vector store
        logger.info("Creating vector store...")
        
        if self.store_type == "faiss":
            self._vector_store = FAISS.from_documents(
                chunks,
                self.embeddings_manager.embeddings
            )
        else:
            raise ValueError(f"Unsupported store type: {self.store_type}")
        
        # Store metadata
        self._metadata = {
            "created_at": datetime.utcnow().isoformat(),
            "document_count": len(documents),
            "chunk_count": len(chunks),
            "store_type": self.store_type,
            "embeddings_model": self.embeddings_manager.model_name
        }
        
        logger.info(f"Vector store created with {len(chunks)} chunks")
        
        return self._vector_store
    
    def add_documents(
        self,
        documents: List[Document],
        chunk: bool = True
    ) -> List[str]:
        """
        Add documents to existing vector store.
        
        Parameters
        ----------
        documents : List[Document]
            Documents to add
        chunk : bool
            Whether to chunk documents
        
        Returns
        -------
        List[str]
            IDs of added documents
        """
        if not self._vector_store:
            raise RuntimeError("No vector store initialized. Call create_from_documents first.")
        
        if chunk:
            chunks = create_chunks(documents)
        else:
            chunks = documents
        
        ids = self._vector_store.add_documents(chunks)
        
        # Update metadata
        self._metadata["chunk_count"] = self._metadata.get("chunk_count", 0) + len(chunks)
        self._metadata["last_updated"] = datetime.utcnow().isoformat()
        
        logger.info(f"Added {len(chunks)} chunks to vector store")
        
        return ids
    
    def search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None
    ) -> List[Document]:
        """
        Search for similar documents.
        
        Parameters
        ----------
        query : str
            Search query
        k : int
            Number of results
        filter : dict, optional
            Metadata filter
        score_threshold : float, optional
            Minimum similarity score
        
        Returns
        -------
        List[Document]
            Similar documents
        """
        if not self._vector_store:
            raise RuntimeError("No vector store initialized.")
        
        if score_threshold:
            # Use similarity search with score
            results = self._vector_store.similarity_search_with_score(
                query,
                k=k,
                filter=filter
            )
            # Filter by threshold
            results = [
                doc for doc, score in results 
                if score >= score_threshold
            ]
        else:
            results = self._vector_store.similarity_search(
                query,
                k=k,
                filter=filter
            )
        
        return results
    
    def search_with_scores(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """
        Search with similarity scores.
        
        Returns documents with their similarity scores.
        """
        if not self._vector_store:
            raise RuntimeError("No vector store initialized.")
        
        return self._vector_store.similarity_search_with_score(
            query,
            k=k,
            filter=filter
        )
    
    def mmr_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5
    ) -> List[Document]:
        """
        Maximum Marginal Relevance search.
        
        Returns diverse results, not just most similar.
        
        Parameters
        ----------
        query : str
            Search query
        k : int
            Number of results
        fetch_k : int
            Number to fetch before reranking
        lambda_mult : float
            Diversity parameter (0=max diversity, 1=max relevance)
        """
        if not self._vector_store:
            raise RuntimeError("No vector store initialized.")
        
        return self._vector_store.max_marginal_relevance_search(
            query,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult
        )
    
    def save(self, name: str = "default"):
        """
        Save vector store to disk.
        
        Parameters
        ----------
        name : str
            Name for the saved store
        """
        if not self._vector_store:
            raise RuntimeError("No vector store to save.")
        
        save_path = self.persist_directory / name
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        if self.store_type == "faiss":
            self._vector_store.save_local(str(save_path))
        
        # Save metadata
        metadata_path = save_path / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self._metadata, f, indent=2)
        
        logger.info(f"Vector store saved to {save_path}")
    
    def load(self, name: str = "default") -> VectorStore:
        """
        Load vector store from disk.
        
        Parameters
        ----------
        name : str
            Name of the saved store
        
        Returns
        -------
        VectorStore
            Loaded vector store
        """
        load_path = self.persist_directory / name
        
        if not load_path.exists():
            raise FileNotFoundError(f"No vector store found at {load_path}")
        
        # Load FAISS index
        if self.store_type == "faiss":
            self._vector_store = FAISS.load_local(
                str(load_path),
                self.embeddings_manager.embeddings,
                allow_dangerous_deserialization=True
            )
        
        # Load metadata
        metadata_path = load_path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self._metadata = json.load(f)
        
        logger.info(f"Vector store loaded from {load_path}")
        
        return self._vector_store
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        return {
            "initialized": self._vector_store is not None,
            "store_type": self.store_type,
            "metadata": self._metadata,
            "embeddings": self.embeddings_manager.get_stats()
        }
    
    def as_retriever(
        self,
        search_type: str = "similarity",
        search_kwargs: Optional[Dict[str, Any]] = None
    ):
        """
        Get as a LangChain retriever.
        
        Parameters
        ----------
        search_type : str
            Search type: similarity, mmr, similarity_score_threshold
        search_kwargs : dict
            Search arguments (k, score_threshold, etc.)
        
        Returns
        -------
        VectorStoreRetriever
            LangChain retriever
        """
        if not self._vector_store:
            raise RuntimeError("No vector store initialized.")
        
        search_kwargs = search_kwargs or {"k": 4}
        
        return self._vector_store.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )
    
    @property
    def vector_store(self) -> Optional[VectorStore]:
        """Get the underlying vector store."""
        return self._vector_store


# ==================== Convenience Functions ====================

_vector_store_manager: Optional[VectorStoreManager] = None


def get_vector_store(
    store_type: str = "faiss",
    persist_directory: Optional[str] = None
) -> VectorStoreManager:
    """Get or create the vector store manager singleton."""
    global _vector_store_manager
    
    if _vector_store_manager is None:
        _vector_store_manager = VectorStoreManager(
            store_type=store_type,
            persist_directory=persist_directory
        )
    
    return _vector_store_manager


def create_vector_store(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    save_name: Optional[str] = None
) -> VectorStoreManager:
    """
    Create a vector store from documents.
    
    Convenience function for quick setup.
    """
    manager = get_vector_store()
    manager.create_from_documents(
        documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    if save_name:
        manager.save(save_name)
    
    return manager