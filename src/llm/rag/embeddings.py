# src/llm/rag/embeddings.py
"""
Embeddings Management
=====================
Generate and manage text embeddings for RAG.

Supports:
- OpenAI embeddings (primary)
- Azure OpenAI embeddings
- Local embeddings (sentence-transformers) - optional
- Caching for efficiency
- Fallback handling for missing dependencies

Features:
- Automatic provider selection based on availability
- Embedding caching to reduce API costs
- Batch processing with progress tracking
- Cost estimation and tracking
- Dimension validation
"""

import logging
import hashlib
import os
from typing import List, Dict, Any, Optional, Literal, Union, Tuple
from pathlib import Path
from datetime import datetime
from functools import lru_cache
import json

logger = logging.getLogger(__name__)


# ============================================================
# Dependency Checks
# ============================================================

# Check for langchain_core
try:
    from langchain_core.embeddings import Embeddings
    LANGCHAIN_CORE_AVAILABLE = True
except ImportError:
    LANGCHAIN_CORE_AVAILABLE = False
    logger.warning("langchain-core not available")
    
    # Create a dummy base class
    class Embeddings:
        """Dummy Embeddings class when langchain not available."""
        pass

# Check for OpenAI embeddings
try:
    from langchain_openai import OpenAIEmbeddings
    OPENAI_EMBEDDINGS_AVAILABLE = True
except ImportError:
    OPENAI_EMBEDDINGS_AVAILABLE = False
    logger.info("langchain-openai not available, OpenAI embeddings disabled")
    OpenAIEmbeddings = None

# Check for Azure OpenAI embeddings
try:
    from langchain_openai import AzureOpenAIEmbeddings
    AZURE_OPENAI_AVAILABLE = True
except ImportError:
    AZURE_OPENAI_AVAILABLE = False
    AzureOpenAIEmbeddings = None

# Check for HuggingFace/sentence-transformers
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        HUGGINGFACE_AVAILABLE = True
    except ImportError:
        HUGGINGFACE_AVAILABLE = False
        logger.info("sentence-transformers not available, local embeddings disabled")
        HuggingFaceEmbeddings = None

# Check for direct OpenAI client (fallback)
try:
    import openai
    OPENAI_CLIENT_AVAILABLE = True
except ImportError:
    OPENAI_CLIENT_AVAILABLE = False
    openai = None

# Check for numpy
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None


# ============================================================
# Configuration
# ============================================================

class EmbeddingModelConfig:
    """Configuration for embedding models."""
    
    # OpenAI Models
    OPENAI_MODELS = {
        "text-embedding-3-small": {
            "dimension": 1536,
            "max_tokens": 8191,
            "cost_per_1k_tokens": 0.00002,
            "description": "Efficient, good for most use cases"
        },
        "text-embedding-3-large": {
            "dimension": 3072,
            "max_tokens": 8191,
            "cost_per_1k_tokens": 0.00013,
            "description": "Highest quality, larger dimension"
        },
        "text-embedding-ada-002": {
            "dimension": 1536,
            "max_tokens": 8191,
            "cost_per_1k_tokens": 0.0001,
            "description": "Legacy model, still widely used"
        }
    }
    
    # Local Models (sentence-transformers)
    LOCAL_MODELS = {
        "all-MiniLM-L6-v2": {
            "dimension": 384,
            "max_tokens": 256,
            "cost_per_1k_tokens": 0,
            "description": "Fast, small, good quality"
        },
        "all-mpnet-base-v2": {
            "dimension": 768,
            "max_tokens": 384,
            "cost_per_1k_tokens": 0,
            "description": "Higher quality, medium size"
        },
        "multi-qa-mpnet-base-dot-v1": {
            "dimension": 768,
            "max_tokens": 512,
            "cost_per_1k_tokens": 0,
            "description": "Optimized for Q&A retrieval"
        },
        "paraphrase-MiniLM-L6-v2": {
            "dimension": 384,
            "max_tokens": 256,
            "cost_per_1k_tokens": 0,
            "description": "Good for semantic similarity"
        }
    }
    
    @classmethod
    def get_model_config(
        cls, 
        provider: str, 
        model_name: str
    ) -> Dict[str, Any]:
        """Get configuration for a specific model."""
        if provider == "openai":
            return cls.OPENAI_MODELS.get(model_name, {
                "dimension": 1536,
                "max_tokens": 8191,
                "cost_per_1k_tokens": 0.0001,
                "description": "Unknown model"
            })
        elif provider == "local":
            return cls.LOCAL_MODELS.get(model_name, {
                "dimension": 384,
                "max_tokens": 256,
                "cost_per_1k_tokens": 0,
                "description": "Unknown local model"
            })
        else:
            return {}
    
    @classmethod
    def list_available_models(cls) -> Dict[str, List[str]]:
        """List all available models by provider."""
        available = {}
        
        if OPENAI_EMBEDDINGS_AVAILABLE or OPENAI_CLIENT_AVAILABLE:
            available["openai"] = list(cls.OPENAI_MODELS.keys())
        
        if HUGGINGFACE_AVAILABLE:
            available["local"] = list(cls.LOCAL_MODELS.keys())
        
        return available


# ============================================================
# Fallback Embeddings (Direct OpenAI Client)
# ============================================================

class DirectOpenAIEmbeddings:
    """
    Direct OpenAI embeddings using the openai package.
    
    Fallback when langchain-openai is not available.
    """
    
    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None
    ):
        if not OPENAI_CLIENT_AVAILABLE:
            raise RuntimeError("openai package not installed")
        
        self.model = model
        self.client = openai.OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self._dimension = EmbeddingModelConfig.get_model_config("openai", model).get("dimension", 1536)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        if not texts:
            return []
        
        # OpenAI API has a limit, batch if needed
        batch_size = 100
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            response = self.client.embeddings.create(
                model=self.model,
                input=batch
            )
            
            # Sort by index to maintain order
            sorted_embeddings = sorted(response.data, key=lambda x: x.index)
            batch_embeddings = [e.embedding for e in sorted_embeddings]
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        response = self.client.embeddings.create(
            model=self.model,
            input=[text]
        )
        return response.data[0].embedding


# ============================================================
# Simple Embeddings (No Dependencies)
# ============================================================

class SimpleHashEmbeddings(Embeddings):
    """
    Simple hash-based embeddings for testing when no providers available.
    
    WARNING: These are NOT semantic embeddings and should only be used
    for testing or when no other option is available.
    """
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        logger.warning(
            "Using SimpleHashEmbeddings - NOT suitable for production! "
            "Install openai or sentence-transformers for real embeddings."
        )
    
    def _hash_to_vector(self, text: str) -> List[float]:
        """Convert text to a deterministic pseudo-random vector."""
        # Use hash to seed random generator for reproducibility
        hash_value = int(hashlib.sha256(text.encode()).hexdigest(), 16)
        
        if NUMPY_AVAILABLE:
            rng = np.random.RandomState(hash_value % (2**32))
            vector = rng.randn(self.dimension)
            # Normalize
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm
            return vector.tolist()
        else:
            # Pure Python fallback
            import random
            random.seed(hash_value % (2**32))
            vector = [random.gauss(0, 1) for _ in range(self.dimension)]
            norm = sum(x*x for x in vector) ** 0.5
            if norm > 0:
                vector = [x / norm for x in vector]
            return vector
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents using hash-based method."""
        return [self._hash_to_vector(text) for text in texts]
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a query using hash-based method."""
        return self._hash_to_vector(text)

    def __call__(self, text: str) -> List[float]:
        """Allow object to be called like a function (LangChain compatibility)."""
        return self.embed_query(text)


# ============================================================
# Main Embeddings Manager
# ============================================================

class EmbeddingsManager:
    """
    Manage embeddings for RAG with multiple provider support.
    
    Features:
    - Multiple embedding providers (OpenAI, local, fallback)
    - Automatic provider selection
    - Caching to reduce API costs
    - Batch processing
    - Cost tracking
    - Graceful degradation when dependencies missing
    
    Example
    -------
    >>> manager = EmbeddingsManager(provider="openai")
    >>> embeddings = manager.embed_texts(["Hello world", "Goodbye"])
    >>> print(f"Dimension: {len(embeddings[0])}")
    
    >>> # With automatic provider selection
    >>> manager = EmbeddingsManager()  # Picks best available
    >>> embeddings = manager.embed_texts(["Test"])
    """
    
    def __init__(
        self,
        provider: Optional[Literal["openai", "azure", "local", "auto", "simple"]] = "auto",
        model_name: Optional[str] = None,
        cache_dir: Optional[str] = None,
        use_cache: bool = True,
        api_key: Optional[str] = None,
        show_progress: bool = True
    ):
        """
        Initialize embeddings manager.
        
        Parameters
        ----------
        provider : str, optional
            Embedding provider:
            - "openai": OpenAI API embeddings
            - "azure": Azure OpenAI embeddings
            - "local": Local sentence-transformers
            - "auto": Automatically select best available
            - "simple": Hash-based (testing only)
        model_name : str, optional
            Specific model name (uses provider default if not specified)
        cache_dir : str, optional
            Directory for caching embeddings
        use_cache : bool
            Whether to cache embeddings
        api_key : str, optional
            API key for OpenAI/Azure (or use environment variable)
        show_progress : bool
            Show progress bar for batch operations
        """
        self.use_cache = use_cache
        self.show_progress = show_progress
        self._api_key = api_key
        
        # Auto-select provider if needed
        if provider == "auto" or provider is None:
            provider = self._auto_select_provider()
        
        self.provider = provider
        
        # Set model name
        self.model_name = model_name or self._get_default_model(provider)
        
        # Initialize cache
        self.cache_dir = Path(cache_dir) if cache_dir else Path("data/embeddings_cache")
        if use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._cache = self._load_cache()
        else:
            self._cache = {}
        
        # Initialize embeddings model
        self._embeddings = self._create_embeddings()
        
        # Get model config
        self._model_config = EmbeddingModelConfig.get_model_config(
            self.provider, 
            self.model_name
        )
        
        # Statistics
        self._stats = {
            "total_embeddings": 0,
            "cache_hits": 0,
            "api_calls": 0,
            "total_tokens_estimated": 0,
            "estimated_cost_usd": 0.0,
            "errors": 0
        }
        
        self._initialized_at = datetime.utcnow()
        
        logger.info(
            f"EmbeddingsManager initialized: provider={self.provider}, "
            f"model={self.model_name}, dimension={self._model_config.get('dimension', 'unknown')}"
        )
    
    def _auto_select_provider(self) -> str:
        """Automatically select the best available provider."""
        # Prefer Local (HuggingFace) if available to avoid OpenAI dependency
        if HUGGINGFACE_AVAILABLE:
            logger.info("Auto-selected provider: local (sentence-transformers)")
            return "local"

        # Fallback to OpenAI if available
        if OPENAI_EMBEDDINGS_AVAILABLE:
            logger.info("Auto-selected provider: openai (langchain-openai)")
            return "openai"
        
        # Fall back to direct OpenAI client
        if OPENAI_CLIENT_AVAILABLE and os.getenv("OPENAI_API_KEY"):
            logger.info("Auto-selected provider: openai (direct client)")
            return "openai"
        
        # Last resort: simple hash embeddings
        logger.warning("No embedding providers available, using simple hash embeddings")
        return "simple"
    
    def _get_default_model(self, provider: str) -> str:
        """Get default model for a provider."""
        defaults = {
            "openai": "text-embedding-3-small",
            "azure": "text-embedding-3-small",
            "local": "all-MiniLM-L6-v2",
            "simple": "hash-384"
        }
        return defaults.get(provider, "text-embedding-3-small")
    
    def _create_embeddings(self):
        """Create the embeddings model based on provider."""
        try:
            if self.provider == "openai":
                return self._create_openai_embeddings()
            
            elif self.provider == "azure":
                return self._create_azure_embeddings()
            
            elif self.provider == "local":
                return self._create_local_embeddings()
            
            elif self.provider == "simple":
                return SimpleHashEmbeddings(dimension=384)
            
            else:
                raise ValueError(f"Unknown provider: {self.provider}")
                
        except Exception as e:
            logger.error(f"Failed to create embeddings for {self.provider}: {e}")
            logger.warning("Falling back to simple hash embeddings")
            self.provider = "simple"
            return SimpleHashEmbeddings(dimension=384)
    
    def _create_openai_embeddings(self):
        """Create OpenAI embeddings."""
        # Try langchain-openai first
        if OPENAI_EMBEDDINGS_AVAILABLE:
            kwargs = {"model": self.model_name}
            if self._api_key:
                kwargs["api_key"] = self._api_key
            if self.show_progress:
                kwargs["show_progress_bar"] = True
            
            return OpenAIEmbeddings(**kwargs)
        
        # Fall back to direct client
        if OPENAI_CLIENT_AVAILABLE:
            return DirectOpenAIEmbeddings(
                model=self.model_name,
                api_key=self._api_key
            )
        
        raise RuntimeError(
            "OpenAI embeddings not available. Install with: "
            "pip install langchain-openai  OR  pip install openai"
        )
    
    def _create_azure_embeddings(self):
        """Create Azure OpenAI embeddings."""
        if not AZURE_OPENAI_AVAILABLE:
            raise RuntimeError(
                "Azure OpenAI not available. Install with: pip install langchain-openai"
            )
        
        return AzureOpenAIEmbeddings(
            model=self.model_name,
            api_key=self._api_key or os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
        )
    
    def _create_local_embeddings(self):
        """Create local HuggingFace embeddings."""
        if not HUGGINGFACE_AVAILABLE:
            raise RuntimeError(
                "Local embeddings not available. Install with: "
                "pip install sentence-transformers langchain-community"
            )
        
        return HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
    
    # ============================================================
    # Core Embedding Methods
    # ============================================================
    
    def embed_text(self, text: str) -> List[float]:
        """
        Embed a single text.
        
        Parameters
        ----------
        text : str
            Text to embed
        
        Returns
        -------
        List[float]
            Embedding vector
        """
        return self.embed_texts([text])[0]
    
    def embed_texts(
        self, 
        texts: List[str],
        batch_size: int = 100
    ) -> List[List[float]]:
        """
        Embed multiple texts with caching.
        
        Parameters
        ----------
        texts : List[str]
            Texts to embed
        batch_size : int
            Batch size for API calls
        
        Returns
        -------
        List[List[float]]
            List of embedding vectors
        """
        if not texts:
            return []
        
        results: List[Optional[List[float]]] = [None] * len(texts)
        texts_to_embed: List[Tuple[int, str]] = []  # (original_index, text)
        
        # Check cache first
        for i, text in enumerate(texts):
            if not text or not text.strip():
                # Handle empty texts
                results[i] = [0.0] * self._model_config.get("dimension", 384)
                continue
            
            cache_key = self._get_cache_key(text)
            
            if self.use_cache and cache_key in self._cache:
                results[i] = self._cache[cache_key]
                self._stats["cache_hits"] += 1
            else:
                texts_to_embed.append((i, text))
        
        # Embed uncached texts
        if texts_to_embed:
            try:
                # Extract just the texts for embedding
                texts_only = [text for _, text in texts_to_embed]
                
                self._stats["api_calls"] += 1
                embeddings = self._embeddings.embed_documents(texts_only)
                
                # Store results and update cache
                for (original_idx, text), embedding in zip(texts_to_embed, embeddings):
                    results[original_idx] = embedding
                    
                    if self.use_cache:
                        cache_key = self._get_cache_key(text)
                        self._cache[cache_key] = embedding
                
                # Update stats
                self._update_stats(texts_only)
                
            except Exception as e:
                logger.error(f"Embedding failed: {e}")
                self._stats["errors"] += 1
                
                # Fill failed embeddings with zeros
                dimension = self._model_config.get("dimension", 384)
                for original_idx, _ in texts_to_embed:
                    if results[original_idx] is None:
                        results[original_idx] = [0.0] * dimension
        
        self._stats["total_embeddings"] += len(texts)
        
        return results
    
    def embed_query(self, query: str) -> List[float]:
        """
        Embed a query.
        
        Some models optimize differently for queries vs documents.
        
        Parameters
        ----------
        query : str
            Query text
        
        Returns
        -------
        List[float]
            Query embedding
        """
        if not query or not query.strip():
            return [0.0] * self._model_config.get("dimension", 384)
        
        # Check cache
        cache_key = self._get_cache_key(f"__query__:{query}")
        
        if self.use_cache and cache_key in self._cache:
            self._stats["cache_hits"] += 1
            return self._cache[cache_key]
        
        try:
            embedding = self._embeddings.embed_query(query)
            
            if self.use_cache:
                self._cache[cache_key] = embedding
            
            self._update_stats([query])
            return embedding
            
        except Exception as e:
            logger.error(f"Query embedding failed: {e}")
            self._stats["errors"] += 1
            return [0.0] * self._model_config.get("dimension", 384)
    
    # ============================================================
    # Cache Management
    # ============================================================
    
    def _get_cache_key(self, text: str) -> str:
        """Generate a unique cache key for text."""
        content = f"{self.provider}:{self.model_name}:{text}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _load_cache(self) -> Dict[str, List[float]]:
        """Load cache from disk."""
        cache_file = self.cache_dir / f"embeddings_cache_{self.provider}_{self.model_name.replace('/', '_')}.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                logger.info(f"Loaded {len(data)} cached embeddings from {cache_file}")
                return data
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        
        return {}
    
    def save_cache(self) -> bool:
        """
        Save cache to disk.
        
        Returns
        -------
        bool
            True if saved successfully
        """
        if not self.use_cache or not self._cache:
            return True
        
        cache_file = self.cache_dir / f"embeddings_cache_{self.provider}_{self.model_name.replace('/', '_')}.json"
        
        try:
            with open(cache_file, 'w') as f:
                json.dump(self._cache, f)
            logger.info(f"Saved {len(self._cache)} embeddings to cache")
            return True
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
            return False
    
    def clear_cache(self):
        """Clear the embeddings cache."""
        self._cache = {}
        
        # Remove cache files
        if self.cache_dir.exists():
            for cache_file in self.cache_dir.glob("embeddings_cache_*.json"):
                try:
                    cache_file.unlink()
                except Exception as e:
                    logger.warning(f"Failed to delete {cache_file}: {e}")
        
        logger.info("Embeddings cache cleared")
    
    # ============================================================
    # Statistics and Info
    # ============================================================
    
    def _update_stats(self, texts: List[str]):
        """Update statistics after embedding."""
        # Estimate tokens (rough: 4 chars per token)
        total_chars = sum(len(t) for t in texts)
        estimated_tokens = total_chars // 4
        
        self._stats["total_tokens_estimated"] += estimated_tokens
        
        # Estimate cost
        cost_per_1k = self._model_config.get("cost_per_1k_tokens", 0)
        cost = (estimated_tokens / 1000) * cost_per_1k
        self._stats["estimated_cost_usd"] += cost
    
    def get_dimension(self) -> int:
        """Get the embedding dimension."""
        return self._model_config.get("dimension", 384)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        return {
            "provider": self.provider,
            "model": self.model_name,
            "dimension": self._model_config.get("dimension", "unknown"),
            "max_tokens": self._model_config.get("max_tokens", "unknown"),
            "cost_per_1k_tokens": self._model_config.get("cost_per_1k_tokens", 0),
            "description": self._model_config.get("description", "")
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get embedding statistics."""
        uptime = (datetime.utcnow() - self._initialized_at).total_seconds()
        
        return {
            **self._stats,
            "cache_size": len(self._cache),
            "cache_hit_rate": (
                self._stats["cache_hits"] / self._stats["total_embeddings"]
                if self._stats["total_embeddings"] > 0 else 0
            ),
            "uptime_seconds": uptime,
            "model_info": self.get_model_info(),
            "providers_available": {
                "openai": OPENAI_EMBEDDINGS_AVAILABLE or OPENAI_CLIENT_AVAILABLE,
                "azure": AZURE_OPENAI_AVAILABLE,
                "local": HUGGINGFACE_AVAILABLE
            }
        }
    
    @property
    def embeddings(self) -> Any:
        """Get the underlying embeddings model."""
        return self._embeddings
    
    @property
    def is_available(self) -> bool:
        """Check if embeddings are available."""
        return self._embeddings is not None
    
    def __repr__(self) -> str:
        return (
            f"EmbeddingsManager(provider='{self.provider}', "
            f"model='{self.model_name}', "
            f"dimension={self.get_dimension()})"
        )


# ============================================================
# Singleton Management
# ============================================================

_embeddings_manager: Optional[EmbeddingsManager] = None


def get_embeddings(
    provider: str = "auto",
    model_name: Optional[str] = None,
    force_new: bool = False
) -> EmbeddingsManager:
    """
    Get or create the embeddings manager singleton.
    
    Parameters
    ----------
    provider : str
        Embedding provider ("openai", "local", "auto")
    model_name : str, optional
        Model name
    force_new : bool
        Force creation of new instance
    
    Returns
    -------
    EmbeddingsManager
        Singleton instance
    """
    global _embeddings_manager
    
    if _embeddings_manager is None or force_new:
        _embeddings_manager = EmbeddingsManager(
            provider=provider,
            model_name=model_name
        )
    
    return _embeddings_manager