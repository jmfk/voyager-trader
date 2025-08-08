"""Async data fetcher with rate limiting and error handling."""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from .interfaces import DataFetcher, DataSource
from .rate_limiter import RateLimiter

logger = logging.getLogger(__name__)


class RetryConfig:
    """Configuration for retry behavior."""

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        backoff_factor: float = 2.0,
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor


class AsyncDataFetcher(DataFetcher):
    """Async data fetcher with rate limiting, retries, and error handling."""

    def __init__(
        self,
        rate_limiter: Optional[RateLimiter] = None,
        retry_config: Optional[RetryConfig] = None,
        max_concurrent_requests: int = 10,
    ):
        self.rate_limiter = rate_limiter or RateLimiter()
        self.retry_config = retry_config or RetryConfig()
        self.max_concurrent_requests = max_concurrent_requests

        # Semaphore to limit concurrent requests
        self._semaphore = asyncio.Semaphore(max_concurrent_requests)

        # Statistics tracking
        self._stats = {
            "requests_made": 0,
            "requests_successful": 0,
            "requests_failed": 0,
            "total_retry_attempts": 0,
            "rate_limit_waits": 0,
        }

    async def fetch_data(
        self,
        source: DataSource,
        method: str,
        *args,
        **kwargs,
    ) -> Any:
        """
        Fetch data from source with rate limiting and retry logic.

        Args:
            source: Data source instance
            method: Method name to call on the source
            *args: Positional arguments for the method
            **kwargs: Keyword arguments for the method

        Returns:
            Result from the data source method
        """
        if not source.is_enabled:
            raise Exception(f"Data source {source.name} is disabled")

        async with self._semaphore:
            return await self._fetch_with_retry(source, method, *args, **kwargs)

    async def _fetch_with_retry(
        self,
        source: DataSource,
        method: str,
        *args,
        **kwargs,
    ) -> Any:
        """Execute fetch with retry logic."""
        last_exception = None

        for attempt in range(self.retry_config.max_retries + 1):
            try:
                # Apply rate limiting
                await self._apply_rate_limiting(source.name)

                # Get the method from the source
                source_method = getattr(source, method)
                if not callable(source_method):
                    raise AttributeError(f"Method {method} not found on {source.name}")

                # Track request
                self._stats["requests_made"] += 1
                start_time = time.time()

                # Execute the method
                result = await source_method(*args, **kwargs)

                # Track success
                self._stats["requests_successful"] += 1
                duration = time.time() - start_time

                logger.debug(
                    f"Successfully fetched data from {source.name}.{method} "
                    f"in {duration:.2f}s (attempt {attempt + 1})"
                )

                return result

            except Exception as e:
                last_exception = e
                self._stats["requests_failed"] += 1

                logger.warning(
                    f"Request failed for {source.name}.{method} "
                    f"(attempt {attempt + 1}): {e}"
                )

                # If this was the last attempt, don't wait
                if attempt == self.retry_config.max_retries:
                    break

                # Calculate delay for next attempt
                delay = min(
                    self.retry_config.base_delay
                    * (self.retry_config.backoff_factor**attempt),
                    self.retry_config.max_delay,
                )

                logger.debug(f"Waiting {delay:.1f}s before retry...")
                await asyncio.sleep(delay)
                self._stats["total_retry_attempts"] += 1

        # All retries failed
        logger.error(
            f"All {self.retry_config.max_retries + 1} attempts failed for "
            f"{source.name}.{method}: {last_exception}"
        )
        raise last_exception

    async def _apply_rate_limiting(self, provider: str) -> None:
        """Apply rate limiting for a provider."""
        try:
            await self.rate_limiter.acquire(provider)
        except Exception as e:
            self._stats["rate_limit_waits"] += 1
            logger.debug(f"Rate limiting applied for {provider}: {e}")

    async def batch_fetch(
        self,
        requests: List[Dict],
    ) -> List[Any]:
        """
        Fetch multiple data requests concurrently.

        Args:
            requests: List of request dictionaries with keys:
                - source: DataSource instance
                - method: Method name to call
                - args: Positional arguments (optional)
                - kwargs: Keyword arguments (optional)

        Returns:
            List of results corresponding to each request
        """
        if not requests:
            return []

        logger.info(f"Starting batch fetch of {len(requests)} requests")
        start_time = time.time()

        # Create tasks for all requests
        tasks = []
        for i, request in enumerate(requests):
            source = request["source"]
            method = request["method"]
            args = request.get("args", [])
            kwargs = request.get("kwargs", {})

            task = asyncio.create_task(
                self.fetch_data(source, method, *args, **kwargs),
                name=f"fetch_{i}_{source.name}_{method}",
            )
            tasks.append(task)

        # Execute all tasks concurrently
        results = []
        completed_tasks = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(completed_tasks):
            if isinstance(result, Exception):
                logger.error(f"Batch request {i} failed: {result}")
                results.append(None)  # or raise exception based on requirements
            else:
                results.append(result)

        duration = time.time() - start_time
        successful_count = sum(1 for r in results if r is not None)

        logger.info(
            f"Batch fetch completed in {duration:.2f}s: "
            f"{successful_count}/{len(requests)} successful"
        )

        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get fetcher statistics."""
        success_rate = 0
        if self._stats["requests_made"] > 0:
            success_rate = (
                self._stats["requests_successful"] / self._stats["requests_made"]
            )

        return {
            **self._stats,
            "success_rate": success_rate,
            "max_concurrent_requests": self.max_concurrent_requests,
            "retry_config": {
                "max_retries": self.retry_config.max_retries,
                "base_delay": self.retry_config.base_delay,
                "max_delay": self.retry_config.max_delay,
                "backoff_factor": self.retry_config.backoff_factor,
            },
        }

    def reset_stats(self) -> None:
        """Reset statistics counters."""
        for key in self._stats:
            self._stats[key] = 0


class HealthMonitor:
    """Monitors health of data sources and manages failover."""

    def __init__(self, check_interval: int = 300):  # 5 minutes
        self.check_interval = check_interval
        self._health_status: Dict[str, Dict] = {}
        self._monitoring = False
        self._monitor_task: Optional[asyncio.Task] = None

    async def start_monitoring(self, sources: List[DataSource]) -> None:
        """Start health monitoring for data sources."""
        if self._monitoring:
            return

        self._monitoring = True
        self._monitor_task = asyncio.create_task(self._monitor_loop(sources))
        logger.info("Started health monitoring for data sources")

    async def stop_monitoring(self) -> None:
        """Stop health monitoring."""
        self._monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped health monitoring")

    async def _monitor_loop(self, sources: List[DataSource]) -> None:
        """Main monitoring loop."""
        while self._monitoring:
            try:
                for source in sources:
                    await self._check_source_health(source)

                await asyncio.sleep(self.check_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health monitor loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying

    async def _check_source_health(self, source: DataSource) -> None:
        """Check health of a single data source."""
        try:
            start_time = time.time()
            is_healthy = await source.health_check()
            response_time = time.time() - start_time

            self._health_status[source.name] = {
                "healthy": is_healthy,
                "last_check": datetime.utcnow(),
                "response_time": response_time,
                "consecutive_failures": (
                    0
                    if is_healthy
                    else self._health_status.get(source.name, {}).get(
                        "consecutive_failures", 0
                    )
                    + 1
                ),
            }

            # Disable source if too many consecutive failures
            if not is_healthy:
                failures = self._health_status[source.name]["consecutive_failures"]
                if failures >= 3:  # Disable after 3 consecutive failures
                    source.disable()
                    logger.warning(
                        f"Disabled {source.name} due to {failures} consecutive health check failures"
                    )
            else:
                # Re-enable if healthy
                if not source.is_enabled:
                    source.enable()
                    logger.info(
                        f"Re-enabled {source.name} after successful health check"
                    )

        except Exception as e:
            logger.error(f"Health check failed for {source.name}: {e}")

    def get_health_status(self) -> Dict[str, Dict]:
        """Get current health status of all monitored sources."""
        return self._health_status.copy()
