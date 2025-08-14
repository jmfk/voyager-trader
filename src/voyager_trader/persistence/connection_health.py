"""
Database connection health checking and lifecycle management.

This module provides connection health validation, lifecycle tracking,
and automatic recovery for the database connection pool.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional

import aiosqlite

logger = logging.getLogger(__name__)


class ConnectionStatus(Enum):
    """Connection health status."""

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    STALE = "stale"
    EXPIRED = "expired"


@dataclass
class ConnectionMetrics:
    """Connection usage and health metrics."""

    created_at: datetime = field(default_factory=datetime.utcnow)
    last_used: datetime = field(default_factory=datetime.utcnow)
    last_health_check: datetime = field(default_factory=datetime.utcnow)
    usage_count: int = 0
    health_check_count: int = 0
    health_check_failures: int = 0
    total_query_time: float = 0.0

    @property
    def age(self) -> timedelta:
        """Get connection age."""
        return datetime.utcnow() - self.created_at

    @property
    def idle_time(self) -> timedelta:
        """Get time since last use."""
        return datetime.utcnow() - self.last_used

    @property
    def average_query_time(self) -> float:
        """Get average query execution time."""
        return self.total_query_time / max(self.usage_count, 1)

    @property
    def health_check_success_rate(self) -> float:
        """Get health check success rate."""
        if self.health_check_count == 0:
            return 1.0
        return (
            self.health_check_count - self.health_check_failures
        ) / self.health_check_count


@dataclass
class HealthCheckConfig:
    """Configuration for connection health checks."""

    # Health check settings
    enabled: bool = True
    timeout: float = 1.0  # seconds
    query: str = "SELECT 1"
    interval: int = 30  # seconds for periodic checks

    # Connection lifecycle limits
    max_age: int = 3600  # seconds (1 hour)
    max_usage: int = 1000  # operations per connection
    max_idle_time: int = 300  # seconds (5 minutes)

    # Health check thresholds
    max_consecutive_failures: int = 3
    health_check_cache_duration: float = 5.0  # seconds

    # Pool management
    min_healthy_connections: int = 2
    health_check_batch_size: int = 5


class HealthyConnection:
    """
    Wrapper for database connections with health checking and lifecycle tracking.

    Provides connection validation, usage tracking, and automatic health monitoring
    to ensure connection pool reliability.
    """

    def __init__(
        self,
        connection: aiosqlite.Connection,
        config: HealthCheckConfig,
        connection_id: Optional[str] = None,
    ):
        """
        Initialize healthy connection wrapper.

        Args:
            connection: The underlying aiosqlite connection
            config: Health check configuration
            connection_id: Optional unique identifier for the connection
        """
        self.connection = connection
        self.config = config
        self.connection_id = connection_id or f"conn_{id(connection)}"

        # Connection state
        self._status = ConnectionStatus.HEALTHY
        self._metrics = ConnectionMetrics()
        self._lock = asyncio.Lock()

        # Health check caching
        self._last_health_result: Optional[bool] = None
        self._last_health_check_time: float = 0

        logger.debug(f"Created healthy connection: {self.connection_id}")

    @property
    def status(self) -> ConnectionStatus:
        """Get current connection status."""
        return self._status

    @property
    def metrics(self) -> ConnectionMetrics:
        """Get connection metrics."""
        return self._metrics

    @property
    def is_healthy(self) -> bool:
        """Check if connection is currently healthy."""
        return self._status == ConnectionStatus.HEALTHY

    @property
    def is_expired(self) -> bool:
        """Check if connection has exceeded age or usage limits."""
        return (
            self._metrics.age.total_seconds() > self.config.max_age
            or self._metrics.usage_count > self.config.max_usage
        )

    @property
    def is_stale(self) -> bool:
        """Check if connection has been idle too long."""
        return self._metrics.idle_time.total_seconds() > self.config.max_idle_time

    async def validate_health(self, force: bool = False) -> bool:
        """
        Validate connection health with optional caching.

        Args:
            force: Force health check even if cached result is available

        Returns:
            True if connection is healthy, False otherwise
        """
        if not self.config.enabled:
            return True

        # Check cache first (unless forced)
        current_time = time.time()
        if (
            not force
            and self._last_health_result is not None
            and (current_time - self._last_health_check_time)
            < self.config.health_check_cache_duration
        ):
            return self._last_health_result

        async with self._lock:
            # Double-check after acquiring lock
            if (
                not force
                and self._last_health_result is not None
                and (current_time - self._last_health_check_time)
                < self.config.health_check_cache_duration
            ):
                return self._last_health_result

            # Perform actual health check
            is_healthy = await self._perform_health_check()

            # Update cache and metrics
            self._last_health_result = is_healthy
            self._last_health_check_time = current_time
            self._metrics.last_health_check = datetime.utcnow()
            self._metrics.health_check_count += 1

            if not is_healthy:
                self._metrics.health_check_failures += 1
                self._status = ConnectionStatus.UNHEALTHY
                logger.warning(
                    f"Connection {self.connection_id} failed health check "
                    f"({self._metrics.health_check_failures} failures)"
                )
            else:
                # Reset status if health check passes
                if self._status == ConnectionStatus.UNHEALTHY:
                    self._status = ConnectionStatus.HEALTHY
                    logger.info(f"Connection {self.connection_id} recovered")

            return is_healthy

    async def _perform_health_check(self) -> bool:
        """
        Perform the actual health check query.

        Returns:
            True if health check succeeds, False otherwise
        """
        try:
            # Use asyncio.wait_for to enforce timeout
            start_time = time.time()

            async def health_query():
                cursor = await self.connection.execute(self.config.query)
                result = await cursor.fetchone()
                await cursor.close()
                return result

            result = await asyncio.wait_for(health_query(), timeout=self.config.timeout)

            # Update query timing metrics
            query_time = time.time() - start_time
            self._metrics.total_query_time += query_time

            # Validate result based on query type
            if self.config.query == "SELECT 1":
                return result == (1,)
            else:
                return result is not None

        except (asyncio.TimeoutError, aiosqlite.Error, Exception) as e:
            logger.debug(
                f"Health check failed for {self.connection_id}: {type(e).__name__}: {e}"
            )
            return False

    async def record_usage(self, query_time: Optional[float] = None) -> None:
        """
        Record connection usage for metrics tracking.

        Args:
            query_time: Optional query execution time in seconds
        """
        self._metrics.usage_count += 1
        self._metrics.last_used = datetime.utcnow()

        if query_time is not None:
            self._metrics.total_query_time += query_time

        # Check for expiration or staleness
        if self.is_expired:
            self._status = ConnectionStatus.EXPIRED
            logger.debug(f"Connection {self.connection_id} expired")
        elif self.is_stale:
            self._status = ConnectionStatus.STALE
            logger.debug(f"Connection {self.connection_id} is stale")

    async def close(self) -> None:
        """Close the underlying connection."""
        try:
            await self.connection.close()
            logger.debug(f"Closed connection: {self.connection_id}")
        except Exception as e:
            logger.warning(f"Error closing connection {self.connection_id}: {e}")

    def get_stats(self) -> Dict[str, any]:
        """
        Get connection statistics.

        Returns:
            Dictionary with connection statistics
        """
        return {
            "connection_id": self.connection_id,
            "status": self._status.value,
            "age_seconds": self._metrics.age.total_seconds(),
            "idle_seconds": self._metrics.idle_time.total_seconds(),
            "usage_count": self._metrics.usage_count,
            "health_check_count": self._metrics.health_check_count,
            "health_check_failures": self._metrics.health_check_failures,
            "health_check_success_rate": self._metrics.health_check_success_rate,
            "average_query_time": self._metrics.average_query_time,
            "is_healthy": self.is_healthy,
            "is_expired": self.is_expired,
            "is_stale": self.is_stale,
        }


class ConnectionHealthManager:
    """
    Manager for connection pool health monitoring and maintenance.

    Provides periodic health checks, connection lifecycle management,
    and pool statistics for the database connection pool.
    """

    def __init__(self, config: HealthCheckConfig):
        """
        Initialize connection health manager.

        Args:
            config: Health check configuration
        """
        self.config = config
        self._monitoring_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()

        logger.info("Initialized ConnectionHealthManager")

    async def start_monitoring(self, pool: List[HealthyConnection]) -> None:
        """
        Start periodic health monitoring for connection pool.

        Args:
            pool: List of healthy connections to monitor
        """
        if self._monitoring_task is not None:
            return

        self._monitoring_task = asyncio.create_task(self._monitoring_loop(pool))
        logger.info("Started connection pool health monitoring")

    async def stop_monitoring(self) -> None:
        """Stop periodic health monitoring."""
        if self._monitoring_task is None:
            return

        self._shutdown_event.set()

        try:
            await asyncio.wait_for(self._monitoring_task, timeout=5.0)
        except asyncio.TimeoutError:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass

        self._monitoring_task = None
        self._shutdown_event.clear()
        logger.info("Stopped connection pool health monitoring")

    async def _monitoring_loop(self, pool: List[HealthyConnection]) -> None:
        """
        Main monitoring loop for periodic health checks.

        Args:
            pool: List of healthy connections to monitor
        """
        try:
            while not self._shutdown_event.is_set():
                try:
                    await self._check_pool_health(pool)
                except Exception as e:
                    logger.error(f"Error in health monitoring loop: {e}")

                # Wait for next interval or shutdown
                try:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(), timeout=self.config.interval
                    )
                    break  # Shutdown requested
                except asyncio.TimeoutError:
                    continue  # Continue monitoring

        except asyncio.CancelledError:
            logger.debug("Health monitoring loop cancelled")
            raise

    async def _check_pool_health(self, pool: List[HealthyConnection]) -> None:
        """
        Check health of all connections in the pool.

        Args:
            pool: List of healthy connections to check
        """
        if not pool:
            return

        # Process connections in batches to avoid overwhelming
        batch_size = self.config.health_check_batch_size

        for i in range(0, len(pool), batch_size):
            batch = pool[i : i + batch_size]

            # Check batch concurrently
            health_tasks = [
                conn.validate_health(force=True)
                for conn in batch
                if conn.is_healthy  # Only check healthy connections
            ]

            if health_tasks:
                await asyncio.gather(*health_tasks, return_exceptions=True)

        # Log pool statistics
        healthy_count = sum(1 for conn in pool if conn.is_healthy)
        total_count = len(pool)

        logger.debug(
            f"Pool health check complete: {healthy_count}/{total_count} "
            f"healthy connections"
        )

    def get_pool_statistics(self, pool: List[HealthyConnection]) -> Dict[str, any]:
        """
        Get comprehensive pool statistics.

        Args:
            pool: List of healthy connections

        Returns:
            Dictionary with pool statistics
        """
        if not pool:
            return {
                "total_connections": 0,
                "healthy_connections": 0,
                "unhealthy_connections": 0,
                "stale_connections": 0,
                "expired_connections": 0,
                "health_check_success_rate": 0.0,
                "average_age": 0.0,
                "average_usage": 0.0,
                "total_usage": 0,
            }

        # Count by status
        status_counts = {
            ConnectionStatus.HEALTHY: 0,
            ConnectionStatus.UNHEALTHY: 0,
            ConnectionStatus.STALE: 0,
            ConnectionStatus.EXPIRED: 0,
        }

        # Aggregate metrics
        total_usage = 0
        total_age = 0.0
        total_health_checks = 0
        total_health_failures = 0

        for conn in pool:
            status_counts[conn.status] += 1
            total_usage += conn.metrics.usage_count
            total_age += conn.metrics.age.total_seconds()
            total_health_checks += conn.metrics.health_check_count
            total_health_failures += conn.metrics.health_check_failures

        return {
            "total_connections": len(pool),
            "healthy_connections": status_counts[ConnectionStatus.HEALTHY],
            "unhealthy_connections": status_counts[ConnectionStatus.UNHEALTHY],
            "stale_connections": status_counts[ConnectionStatus.STALE],
            "expired_connections": status_counts[ConnectionStatus.EXPIRED],
            "health_check_success_rate": (
                (total_health_checks - total_health_failures)
                / max(total_health_checks, 1)
            ),
            "average_age": total_age / len(pool),
            "average_usage": total_usage / len(pool),
            "total_usage": total_usage,
        }
