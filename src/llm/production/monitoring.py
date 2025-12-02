# src/llm/production/monitoring.py
"""
Monitoring & Observability
==========================
Production monitoring for LLM applications.
"""

import logging
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import json
import threading

logger = logging.getLogger(__name__)


@dataclass
class LLMMetric:
    """A single metric data point."""
    
    name: str
    value: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    labels: Dict[str, str] = field(default_factory=dict)


class MetricsCollector:
    """
    Collect and aggregate metrics for LLM operations.
    
    Tracks:
    - Request counts
    - Latency (p50, p95, p99)
    - Token usage
    - Error rates
    - Cache hit rates
    """
    
    def __init__(self, retention_minutes: int = 60):
        """
        Initialize metrics collector.
        
        Parameters
        ----------
        retention_minutes : int
            How long to keep metrics
        """
        self.retention = timedelta(minutes=retention_minutes)
        
        # Metrics storage
        self._metrics: Dict[str, deque] = {
            "request_count": deque(maxlen=10000),
            "latency_ms": deque(maxlen=10000),
            "tokens_used": deque(maxlen=10000),
            "errors": deque(maxlen=1000),
            "cache_hits": deque(maxlen=1000),
            "cache_misses": deque(maxlen=1000)
        }
        
        self._lock = threading.Lock()
    
    def record(self, name: str, value: float, **labels):
        """Record a metric."""
        metric = LLMMetric(name=name, value=value, labels=labels)
        
        with self._lock:
            if name not in self._metrics:
                self._metrics[name] = deque(maxlen=10000)
            self._metrics[name].append(metric)
    
    def record_request(
        self,
        latency_ms: float,
        tokens: int,
        model: str,
        success: bool = True
    ):
        """Record a request with all metrics."""
        self.record("request_count", 1, model=model)
        self.record("latency_ms", latency_ms, model=model)
        self.record("tokens_used", tokens, model=model)
        
        if not success:
            self.record("errors", 1, model=model)
    
    def record_cache(self, hit: bool):
        """Record cache hit/miss."""
        if hit:
            self.record("cache_hits", 1)
        else:
            self.record("cache_misses", 1)
    
    def get_stats(self, window_minutes: int = 5) -> Dict[str, Any]:
        """
        Get aggregated stats for time window.
        
        Parameters
        ----------
        window_minutes : int
            Time window in minutes
        
        Returns
        -------
        dict
            Aggregated statistics
        """
        cutoff = datetime.utcnow() - timedelta(minutes=window_minutes)
        
        with self._lock:
            # Filter by time window
            recent_latencies = [
                m.value for m in self._metrics.get("latency_ms", [])
                if m.timestamp > cutoff
            ]
            
            recent_requests = len([
                m for m in self._metrics.get("request_count", [])
                if m.timestamp > cutoff
            ])
            
            recent_errors = len([
                m for m in self._metrics.get("errors", [])
                if m.timestamp > cutoff
            ])
            
            recent_tokens = sum([
                m.value for m in self._metrics.get("tokens_used", [])
                if m.timestamp > cutoff
            ])
            
            cache_hits = len([
                m for m in self._metrics.get("cache_hits", [])
                if m.timestamp > cutoff
            ])
            
            cache_misses = len([
                m for m in self._metrics.get("cache_misses", [])
                if m.timestamp > cutoff
            ])
        
        # Calculate percentiles
        def percentile(data: List[float], p: float) -> float:
            if not data:
                return 0
            sorted_data = sorted(data)
            idx = int(len(sorted_data) * p / 100)
            return sorted_data[min(idx, len(sorted_data) - 1)]
        
        total_cache = cache_hits + cache_misses
        
        return {
            "window_minutes": window_minutes,
            "requests": {
                "total": recent_requests,
                "per_minute": recent_requests / window_minutes if window_minutes else 0
            },
            "latency_ms": {
                "p50": percentile(recent_latencies, 50),
                "p95": percentile(recent_latencies, 95),
                "p99": percentile(recent_latencies, 99),
                "avg": sum(recent_latencies) / len(recent_latencies) if recent_latencies else 0
            },
            "tokens": {
                "total": int(recent_tokens),
                "per_request": recent_tokens / recent_requests if recent_requests else 0
            },
            "errors": {
                "total": recent_errors,
                "rate": recent_errors / recent_requests if recent_requests else 0
            },
            "cache": {
                "hits": cache_hits,
                "misses": cache_misses,
                "hit_rate": cache_hits / total_cache if total_cache else 0
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        stats = self.get_stats()
        
        lines = [
            f'llm_requests_total {stats["requests"]["total"]}',
            f'llm_latency_p50_ms {stats["latency_ms"]["p50"]}',
            f'llm_latency_p95_ms {stats["latency_ms"]["p95"]}',
            f'llm_latency_p99_ms {stats["latency_ms"]["p99"]}',
            f'llm_tokens_total {stats["tokens"]["total"]}',
            f'llm_errors_total {stats["errors"]["total"]}',
            f'llm_cache_hit_rate {stats["cache"]["hit_rate"]}'
        ]
        
        return "\n".join(lines)


class AlertManager:
    """
    Alert management for LLM monitoring.
    
    Triggers alerts based on thresholds.
    """
    
    def __init__(self):
        self.alerts: List[Dict[str, Any]] = []
        self.alert_handlers: List[Callable] = []
        
        # Default thresholds
        self.thresholds = {
            "error_rate": 0.1,  # 10%
            "latency_p95_ms": 5000,
            "latency_p99_ms": 10000
        }
    
    def set_threshold(self, name: str, value: float):
        """Set alert threshold."""
        self.thresholds[name] = value
    
    def add_handler(self, handler: Callable[[Dict], None]):
        """Add alert handler function."""
        self.alert_handlers.append(handler)
    
    def check(self, stats: Dict[str, Any]):
        """Check stats against thresholds and trigger alerts."""
        # Check error rate
        if stats["errors"]["rate"] > self.thresholds["error_rate"]:
            self._trigger_alert(
                "high_error_rate",
                f"Error rate {stats['errors']['rate']:.1%} exceeds threshold",
                severity="high",
                value=stats["errors"]["rate"]
            )
        
        # Check latency
        if stats["latency_ms"]["p95"] > self.thresholds["latency_p95_ms"]:
            self._trigger_alert(
                "high_latency",
                f"P95 latency {stats['latency_ms']['p95']:.0f}ms exceeds threshold",
                severity="medium",
                value=stats["latency_ms"]["p95"]
            )
    
    def _trigger_alert(
        self,
        alert_type: str,
        message: str,
        severity: str = "medium",
        **details
    ):
        """Trigger an alert."""
        alert = {
            "type": alert_type,
            "message": message,
            "severity": severity,
            "timestamp": datetime.utcnow().isoformat(),
            **details
        }
        
        self.alerts.append(alert)
        logger.warning(f"ALERT [{severity.upper()}]: {message}")
        
        # Call handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")
    
    def get_active_alerts(self, hours: int = 24) -> List[Dict]:
        """Get recent alerts."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        
        return [
            a for a in self.alerts
            if datetime.fromisoformat(a["timestamp"]) > cutoff
        ]


class LLMMonitor:
    """
    Complete monitoring solution for LLM applications.
    
    Combines metrics collection, alerting, and health checks.
    
    Example
    -------
    >>> monitor = LLMMonitor()
    >>> with monitor.track_request("gpt-4"):
    ...     response = llm.invoke(prompt)
    >>> print(monitor.get_health())
    """
    
    def __init__(self):
        self.metrics = MetricsCollector()
        self.alerts = AlertManager()
        
        self._start_time = datetime.utcnow()
    
    class RequestTracker:
        """Context manager for tracking requests."""
        
        def __init__(self, monitor: 'LLMMonitor', model: str):
            self.monitor = monitor
            self.model = model
            self.start_time = None
            self.tokens = 0
        
        def __enter__(self):
            self.start_time = datetime.utcnow()
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            latency_ms = (datetime.utcnow() - self.start_time).total_seconds() * 1000
            success = exc_type is None
            
            self.monitor.metrics.record_request(
                latency_ms=latency_ms,
                tokens=self.tokens,
                model=self.model,
                success=success
            )
            
            return False
        
        def set_tokens(self, tokens: int):
            self.tokens = tokens
    
    def track_request(self, model: str = "unknown") -> RequestTracker:
        """
        Context manager for tracking requests.
        
        Example
        -------
        >>> with monitor.track_request("gpt-4") as tracker:
        ...     response = llm.invoke(prompt)
        ...     tracker.set_tokens(response.usage.total_tokens)
        """
        return self.RequestTracker(self, model)
    
    def record_cache(self, hit: bool):
        """Record cache hit/miss."""
        self.metrics.record_cache(hit)
    
    def get_stats(self, window_minutes: int = 5) -> Dict[str, Any]:
        """Get current statistics."""
        stats = self.metrics.get_stats(window_minutes)
        
        # Check for alerts
        self.alerts.check(stats)
        
        return stats
    
    def get_health(self) -> Dict[str, Any]:
        """Get health status."""
        stats = self.get_stats()
        
        # Determine health status
        if stats["errors"]["rate"] > 0.5:
            status = "unhealthy"
        elif stats["errors"]["rate"] > 0.1 or stats["latency_ms"]["p95"] > 5000:
            status = "degraded"
        else:
            status = "healthy"
        
        return {
            "status": status,
            "uptime_seconds": (datetime.utcnow() - self._start_time).total_seconds(),
            "stats": stats,
            "active_alerts": len(self.alerts.get_active_alerts(hours=1)),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def export_metrics(self) -> str:
        """Export metrics for monitoring systems."""
        return self.metrics.export_prometheus()