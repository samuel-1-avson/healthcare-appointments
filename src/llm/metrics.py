"""
LLM Metrics Tracking
====================
Track LLM usage and performance metrics.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import threading

logger = logging.getLogger(__name__)


class MetricsTracker:
    """
    Track LLM metrics in-memory.
    
    For production, consider using Prometheus or similar.
    """
    
    def __init__(self):
        self._lock = threading.Lock()
        self._requests: Dict[str, int] = defaultdict(int)
        self._latencies: Dict[str, list] = defaultdict(list)
        self._tokens: Dict[str, int] = defaultdict(int)
        self._errors: Dict[str, int] = defaultdict(int)
        self._feedback_ratings: Dict[str, list] = defaultdict(list)
        self._start_time = datetime.utcnow()
    
    def track_request(
        self,
        endpoint: str,
        latency_ms: float,
        tokens: int = 0,
        success: bool = True
    ) -> None:
        """Track an LLM request."""
        with self._lock:
            self._requests[endpoint] += 1
            self._latencies[endpoint].append(latency_ms)
            self._tokens[endpoint] += tokens
            
            if not success:
                self._errors[endpoint] += 1
            
            # Keep only last 1000 latencies per endpoint
            if len(self._latencies[endpoint]) > 1000:
                self._latencies[endpoint] = self._latencies[endpoint][-1000:]
    
    def track_feedback(self, endpoint: str, rating: int) -> None:
        """Track user feedback rating."""
        with self._lock:
            self._feedback_ratings[endpoint].append(rating)
            
            # Keep only last 1000 ratings
            if len(self._feedback_ratings[endpoint]) > 1000:
                self._feedback_ratings[endpoint] = self._feedback_ratings[endpoint][-1000:]
    
    def get_metrics(self, endpoint: Optional[str] = None) -> Dict[str, Any]:
        """Get metrics for an endpoint or all endpoints."""
        with self._lock:
            if endpoint:
                return self._get_endpoint_metrics(endpoint)
            
            # All endpoints
            result = {
                "uptime_seconds": (datetime.utcnow() - self._start_time).total_seconds(),
                "endpoints": {}
            }
            
            for ep in self._requests.keys():
                result["endpoints"][ep] = self._get_endpoint_metrics(ep)
            
            return result
    
    def _get_endpoint_metrics(self, endpoint: str) -> Dict[str, Any]:
        """Get metrics for a specific endpoint."""
        latencies = self._latencies.get(endpoint, [])
        ratings = self._feedback_ratings.get(endpoint, [])
        
        metrics = {
            "total_requests": self._requests.get(endpoint, 0),
            "total_errors": self._errors.get(endpoint, 0),
            "total_tokens": self._tokens.get(endpoint, 0),
            "error_rate": 0.0,
            "latency": {},
            "feedback": {}
        }
        
        # Error rate
        if metrics["total_requests"] > 0:
            metrics["error_rate"] = metrics["total_errors"] / metrics["total_requests"]
        
        # Latency stats
        if latencies:
            import numpy as np
            metrics["latency"] = {
                "mean_ms": float(np.mean(latencies)),
                "p50_ms": float(np.percentile(latencies, 50)),
                "p95_ms": float(np.percentile(latencies, 95)),
                "p99_ms": float(np.percentile(latencies, 99)),
            }
        
        # Feedback stats
        if ratings:
            import numpy as np
            metrics["feedback"] = {
                "count": len(ratings),
                "average_rating": float(np.mean(ratings)),
                "distribution": {i: ratings.count(i) for i in range(1, 6)}
            }
        
        return metrics
    
    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._requests.clear()
            self._latencies.clear()
            self._tokens.clear()
            self._errors.clear()
            self._feedback_ratings.clear()
            self._start_time = datetime.utcnow()


# Singleton
_metrics_tracker: Optional[MetricsTracker] = None


def get_metrics_tracker() -> MetricsTracker:
    """Get metrics tracker singleton."""
    global _metrics_tracker
    if _metrics_tracker is None:
        _metrics_tracker = MetricsTracker()
    return _metrics_tracker