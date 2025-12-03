# src/api/cache/semantic.py
"""
Semantic Caching Module
=======================
Cache LLM/RAG responses based on semantic similarity of queries.
"""
import json
import logging
import numpy as np
from typing import Optional, List, Dict, Any
from openai import OpenAI
from ..config import get_settings
from ..cache import RedisClient

logger = logging.getLogger(__name__)

class SemanticCache:
    """
    Cache for semantic similarity matching.
    
    Stores:
    - Key: "semantic:cache:{hash}" -> JSON {query, embedding, response}
    - Or uses a list/scan approach for small scale.
    
    For production, use a Vector DB (Pinecone/Weaviate) or RediSearch.
    Here we implement a simplified version using Redis keys and manual cosine check
    (suitable for <1000 cached items).
    """
    
    def __init__(self, threshold: float = 0.95):
        self.client = RedisClient().client
        self.settings = get_settings()
        self.openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.threshold = threshold
        self.prefix = "semantic:cache:"

    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text."""
        try:
            response = self.openai.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            return []

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity."""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def get(self, query: str) -> Optional[str]:
        """
        Retrieve cached response if a semantically similar query exists.
        """
        if not self.client:
            return None
            
        try:
            query_embedding = self._get_embedding(query)
            if not query_embedding:
                return None
                
            # Scan all cached items (inefficient for large scale, fine for demo)
            # In prod, use RediSearch KNN
            keys = self.client.keys(f"{self.prefix}*")
            
            best_sim = -1.0
            best_response = None
            
            for key in keys:
                data = json.loads(self.client.get(key))
                cached_embedding = data.get("embedding")
                
                if not cached_embedding:
                    continue
                    
                sim = self._cosine_similarity(query_embedding, cached_embedding)
                
                if sim > best_sim:
                    best_sim = sim
                    best_response = data.get("response")
            
            if best_sim >= self.threshold:
                logger.info(f"✅ Semantic Cache Hit! Similarity: {best_sim:.4f}")
                return best_response
                
            return None
            
        except Exception as e:
            logger.error(f"Semantic cache get failed: {e}")
            return None

    def set(self, query: str, response: str):
        """Cache a query and its response."""
        if not self.client:
            return
            
        try:
            embedding = self._get_embedding(query)
            if not embedding:
                return
                
            # Use hash of query as key suffix
            import hashlib
            key_hash = hashlib.md5(query.encode()).hexdigest()
            key = f"{self.prefix}{key_hash}"
            
            data = {
                "query": query,
                "embedding": embedding,
                "response": response
            }
            
            # Expire in 24 hours
            self.client.setex(key, 86400, json.dumps(data))
            
        except Exception as e:
            logger.error(f"Semantic cache set failed: {e}")
