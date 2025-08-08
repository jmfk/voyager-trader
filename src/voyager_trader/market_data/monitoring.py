"""Monitoring and metrics collection for market data operations."""

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from threading import Lock
from typing import Any, Dict, List, Optional

from .types import Symbol


@dataclass
class RequestMetrics:
    """Metrics for individual requests."""

    timestamp: float
    provider: str
    method: str
    symbol: Optional[Symbol] = None
    duration_ms: float = 0.0
    success: bool = True
    error_type: Optional[str] = None
    cache_hit: bool = False
    data_size_bytes: int = 0


@dataclass
class ProviderMetrics:
    """Aggregated metrics for a data provider."""

    name: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    cache_hits: int = 0
    avg_response_time_ms: float = 0.0
    error_rate: float = 0.0
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None
    current_failure_streak: int = 0
    total_data_bytes: int = 0
    uptime_percentage: float = 100.0
    rate_limit_hits: int = 0

    # Recent activity tracking
    recent_requests: deque = field(default_factory=lambda: deque(maxlen=1000))
    recent_errors: deque = field(default_factory=lambda: deque(maxlen=100))


class MetricsCollector:
    """Collects and aggregates metrics for market data operations."""

    def __init__(self, retention_hours: int = 24):
        self.retention_hours = retention_hours
        self.start_time = time.time()

        # Thread-safe collections
        self._lock = Lock()
        self._provider_metrics: Dict[str, ProviderMetrics] = {}
        self._recent_requests: deque = deque(maxlen=10000)  # Keep last 10k requests

        # Counters for quick access
        self._request_counts: Dict[str, int] = defaultdict(int)
        self._error_counts: Dict[str, int] = defaultdict(int)
        self._cache_hit_counts: Dict[str, int] = defaultdict(int)

        # Performance tracking
        self._response_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))

    def record_request(
        self,
        provider: str,
        method: str,
        symbol: Optional[Symbol] = None,
        duration_ms: float = 0.0,
        success: bool = True,
        error_type: Optional[str] = None,
        cache_hit: bool = False,
        data_size_bytes: int = 0,
    ) -> None:
        """Record metrics for a request."""
        with self._lock:
            metrics = RequestMetrics(
                timestamp=time.time(),
                provider=provider,
                method=method,
                symbol=symbol,
                duration_ms=duration_ms,
                success=success,
                error_type=error_type,
                cache_hit=cache_hit,
                data_size_bytes=data_size_bytes,
            )

            self._recent_requests.append(metrics)
            self._update_provider_metrics(metrics)

    def _update_provider_metrics(self, metrics: RequestMetrics) -> None:
        """Update aggregated provider metrics."""
        provider = metrics.provider

        if provider not in self._provider_metrics:
            self._provider_metrics[provider] = ProviderMetrics(name=provider)

        pm = self._provider_metrics[provider]

        # Update counters
        pm.total_requests += 1
        self._request_counts[provider] += 1

        if metrics.success:
            pm.successful_requests += 1
            pm.last_success = datetime.fromtimestamp(metrics.timestamp)
            pm.current_failure_streak = 0
        else:
            pm.failed_requests += 1
            pm.last_failure = datetime.fromtimestamp(metrics.timestamp)
            pm.current_failure_streak += 1
            self._error_counts[provider] += 1
            if metrics.error_type:
                pm.recent_errors.append(
                    {
                        "timestamp": metrics.timestamp,
                        "type": metrics.error_type,
                        "method": metrics.method,
                        "symbol": metrics.symbol,
                    }
                )

        if metrics.cache_hit:
            pm.cache_hits += 1
            self._cache_hit_counts[provider] += 1

        pm.total_data_bytes += metrics.data_size_bytes

        # Update response time tracking
        if metrics.duration_ms > 0:
            self._response_times[provider].append(metrics.duration_ms)
            # Calculate rolling average
            recent_times = list(self._response_times[provider])
            pm.avg_response_time_ms = sum(recent_times) / len(recent_times)

        # Update error rate
        pm.error_rate = (
            (pm.failed_requests / pm.total_requests) * 100
            if pm.total_requests > 0
            else 0
        )

        # Update uptime (simplified calculation based on recent success rate)
        recent_success_rate = (
            (pm.successful_requests / pm.total_requests) * 100
            if pm.total_requests > 0
            else 100
        )
        pm.uptime_percentage = recent_success_rate

        # Store in provider's recent requests
        pm.recent_requests.append(metrics)

    def record_rate_limit(self, provider: str) -> None:
        """Record a rate limit hit."""
        with self._lock:
            if provider in self._provider_metrics:
                self._provider_metrics[provider].rate_limit_hits += 1

    def get_provider_metrics(self, provider: str) -> Optional[ProviderMetrics]:
        """Get metrics for a specific provider."""
        with self._lock:
            return self._provider_metrics.get(provider)

    def get_all_provider_metrics(self) -> Dict[str, ProviderMetrics]:
        """Get metrics for all providers."""
        with self._lock:
            return self._provider_metrics.copy()

    def get_system_metrics(self) -> Dict[str, Any]:
        """Get overall system metrics."""
        with self._lock:
            current_time = time.time()
            uptime_seconds = current_time - self.start_time

            # Calculate totals across all providers
            total_requests = sum(
                pm.total_requests for pm in self._provider_metrics.values()
            )
            total_successful = sum(
                pm.successful_requests for pm in self._provider_metrics.values()
            )
            total_failed = sum(
                pm.failed_requests for pm in self._provider_metrics.values()
            )
            total_cache_hits = sum(
                pm.cache_hits for pm in self._provider_metrics.values()
            )
            total_data_bytes = sum(
                pm.total_data_bytes for pm in self._provider_metrics.values()
            )

            # Calculate recent activity (last hour)
            one_hour_ago = current_time - 3600
            recent_requests = [
                r for r in self._recent_requests if r.timestamp > one_hour_ago
            ]

            avg_response_time = 0.0
            if self._response_times:
                all_times = []
                for times in self._response_times.values():
                    all_times.extend(times)
                if all_times:
                    avg_response_time = sum(all_times) / len(all_times)

            return {
                "uptime_seconds": uptime_seconds,
                "uptime_formatted": str(timedelta(seconds=int(uptime_seconds))),
                "total_requests": total_requests,
                "successful_requests": total_successful,
                "failed_requests": total_failed,
                "cache_hits": total_cache_hits,
                "cache_hit_rate_percent": (total_cache_hits / total_requests * 100)
                if total_requests > 0
                else 0,
                "error_rate_percent": (total_failed / total_requests * 100)
                if total_requests > 0
                else 0,
                "avg_response_time_ms": avg_response_time,
                "total_data_bytes": total_data_bytes,
                "total_data_mb": total_data_bytes / (1024 * 1024),
                "requests_per_hour": len(recent_requests),
                "active_providers": len(self._provider_metrics),
                "providers": list(self._provider_metrics.keys()),
            }

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of all providers."""
        with self._lock:
            health_status = {"overall_healthy": True, "providers": {}, "alerts": []}

            current_time = time.time()

            for provider, metrics in self._provider_metrics.items():
                provider_health = {
                    "healthy": True,
                    "uptime_percentage": metrics.uptime_percentage,
                    "error_rate": metrics.error_rate,
                    "current_failure_streak": metrics.current_failure_streak,
                    "avg_response_time_ms": metrics.avg_response_time_ms,
                    "last_success": metrics.last_success.isoformat()
                    if metrics.last_success
                    else None,
                    "last_failure": metrics.last_failure.isoformat()
                    if metrics.last_failure
                    else None,
                }

                # Determine if provider is healthy
                is_healthy = (
                    metrics.error_rate < 10
                    and metrics.current_failure_streak < 5  # Less than 10% error rate
                    and metrics.avg_response_time_ms  # Less than 5 consecutive failures
                    < 5000
                    and (  # Less than 5 second average response
                        not metrics.last_success
                        or current_time - metrics.last_success.timestamp() < 300
                    )  # Success within 5 minutes
                )

                provider_health["healthy"] = is_healthy

                if not is_healthy:
                    health_status["overall_healthy"] = False

                    # Add specific alerts
                    if metrics.error_rate >= 10:
                        health_status["alerts"].append(
                            f"{provider}: High error rate ({metrics.error_rate:.1f}%)"
                        )
                    if metrics.current_failure_streak >= 5:
                        health_status["alerts"].append(
                            f"{provider}: {metrics.current_failure_streak} consecutive failures"
                        )
                    if metrics.avg_response_time_ms >= 5000:
                        health_status["alerts"].append(
                            f"{provider}: Slow response time ({metrics.avg_response_time_ms:.0f}ms)"
                        )

                health_status["providers"][provider] = provider_health

            return health_status

    def reset_metrics(self, provider: Optional[str] = None) -> None:
        """Reset metrics for a provider or all providers."""
        with self._lock:
            if provider:
                if provider in self._provider_metrics:
                    self._provider_metrics[provider] = ProviderMetrics(name=provider)
            else:
                self._provider_metrics.clear()
                self._recent_requests.clear()
                self._request_counts.clear()
                self._error_counts.clear()
                self._cache_hit_counts.clear()
                self._response_times.clear()

    def cleanup_old_metrics(self) -> None:
        """Clean up old metrics beyond retention period."""
        with self._lock:
            cutoff_time = time.time() - (self.retention_hours * 3600)

            # Clean up recent requests
            while (
                self._recent_requests
                and self._recent_requests[0].timestamp < cutoff_time
            ):
                self._recent_requests.popleft()

            # Clean up provider-specific recent data
            for metrics in self._provider_metrics.values():
                # Clean recent requests
                while (
                    metrics.recent_requests
                    and metrics.recent_requests[0].timestamp < cutoff_time
                ):
                    metrics.recent_requests.popleft()

                # Clean recent errors
                while (
                    metrics.recent_errors
                    and metrics.recent_errors[0]["timestamp"] < cutoff_time
                ):
                    metrics.recent_errors.popleft()


# Global metrics collector instance
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


def record_request_metrics(
    provider: str,
    method: str,
    symbol: Optional[Symbol] = None,
    duration_ms: float = 0.0,
    success: bool = True,
    error_type: Optional[str] = None,
    cache_hit: bool = False,
    data_size_bytes: int = 0,
) -> None:
    """Convenience function to record request metrics."""
    collector = get_metrics_collector()
    collector.record_request(
        provider=provider,
        method=method,
        symbol=symbol,
        duration_ms=duration_ms,
        success=success,
        error_type=error_type,
        cache_hit=cache_hit,
        data_size_bytes=data_size_bytes,
    )


def get_health_status() -> Dict[str, Any]:
    """Convenience function to get system health status."""
    collector = get_metrics_collector()
    return collector.get_health_status()


def get_system_metrics() -> Dict[str, Any]:
    """Convenience function to get system metrics."""
    collector = get_metrics_collector()
    return collector.get_system_metrics()
