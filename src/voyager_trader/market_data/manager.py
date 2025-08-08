"""Data source manager for registering and managing multiple data providers."""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Type

from .fetcher import AsyncDataFetcher, HealthMonitor
from .interfaces import DataSource
from .rate_limiter import RateLimiter
from .sources import AlphaVantageDataSource, MockDataSource, YahooFinanceDataSource

logger = logging.getLogger(__name__)


class DataSourceRegistry:
    """Registry for available data source implementations."""

    def __init__(self):
        self._source_classes: Dict[str, Type[DataSource]] = {}

        # Register built-in sources
        self.register("alpha_vantage", AlphaVantageDataSource)
        self.register("yahoo_finance", YahooFinanceDataSource)
        self.register("mock", MockDataSource)

    def register(self, name: str, source_class: Type[DataSource]) -> None:
        """Register a data source class."""
        self._source_classes[name] = source_class
        logger.info(f"Registered data source: {name}")

    def create_source(self, name: str, config: Optional[Dict] = None) -> DataSource:
        """Create a data source instance."""
        if name not in self._source_classes:
            raise ValueError(
                f"Unknown data source: {name}. Available: {list(self._source_classes.keys())}"
            )

        source_class = self._source_classes[name]
        return source_class(config)

    def get_available_sources(self) -> List[str]:
        """Get list of available data source names."""
        return list(self._source_classes.keys())


class DataSourceManager:
    """Manages multiple data sources with failover and load balancing."""

    def __init__(
        self,
        sources_config: Optional[Dict[str, Dict]] = None,
        rate_limiter: Optional[RateLimiter] = None,
        enable_health_monitoring: bool = True,
    ):
        self.registry = DataSourceRegistry()
        self.sources: Dict[str, DataSource] = {}
        self.rate_limiter = rate_limiter or RateLimiter()
        self.fetcher = AsyncDataFetcher(self.rate_limiter)

        # Priority order for failover
        self._priority_order: List[str] = []

        # Health monitoring
        self.health_monitor = HealthMonitor() if enable_health_monitoring else None
        self._health_monitoring_started = False

        # Initialize sources from config
        if sources_config:
            self._initialize_sources(sources_config)

    def _initialize_sources(self, sources_config: Dict[str, Dict]) -> None:
        """Initialize data sources from configuration."""
        for source_name, config in sources_config.items():
            try:
                source = self.registry.create_source(source_name, config)
                self.add_source(source, config.get("priority", 100))

                # Configure rate limits if specified
                if "rate_limit" in config:
                    rate_config = config["rate_limit"]
                    self.rate_limiter.set_limit(
                        source_name,
                        rate_config.get("requests_per_minute", 60),
                        rate_config.get("requests_per_second"),
                    )

                logger.info(f"Initialized data source: {source_name}")

            except Exception as e:
                logger.error(f"Failed to initialize data source {source_name}: {e}")

    def add_source(self, source: DataSource, priority: int = 100) -> None:
        """
        Add a data source to the manager.

        Args:
            source: Data source instance
            priority: Priority for failover (lower = higher priority)
        """
        self.sources[source.name] = source

        # Insert into priority order
        inserted = False
        for i, existing_name in enumerate(self._priority_order):
            existing_source = self.sources[existing_name]
            existing_priority = getattr(existing_source, "_priority", 100)

            if priority < existing_priority:
                self._priority_order.insert(i, source.name)
                inserted = True
                break

        if not inserted:
            self._priority_order.append(source.name)

        # Store priority on source for reference
        source._priority = priority

        logger.info(f"Added data source {source.name} with priority {priority}")

    def remove_source(self, name: str) -> bool:
        """Remove a data source from the manager."""
        if name not in self.sources:
            return False

        del self.sources[name]
        self._priority_order.remove(name)

        logger.info(f"Removed data source: {name}")
        return True

    def get_source(self, name: str) -> Optional[DataSource]:
        """Get a specific data source by name."""
        return self.sources.get(name)

    def get_available_sources(self, enabled_only: bool = True) -> List[DataSource]:
        """Get list of available data sources."""
        sources = list(self.sources.values())
        if enabled_only:
            sources = [s for s in sources if s.is_enabled]
        return sources

    def get_primary_source(self) -> Optional[DataSource]:
        """Get the primary (highest priority) enabled data source."""
        for source_name in self._priority_order:
            source = self.sources[source_name]
            if source.is_enabled:
                return source
        return None

    async def fetch_with_failover(
        self,
        method: str,
        *args,
        sources: Optional[List[str]] = None,
        **kwargs,
    ) -> Any:
        """
        Fetch data with automatic failover between sources.

        Args:
            method: Method name to call on data sources
            *args: Positional arguments for the method
            sources: List of source names to try (defaults to all enabled)
            **kwargs: Keyword arguments for the method

        Returns:
            Result from the first successful data source

        Raises:
            Exception: If all data sources fail
        """
        # Determine which sources to try
        if sources:
            source_names = [name for name in sources if name in self.sources]
        else:
            source_names = self._priority_order

        # Filter to enabled sources only
        enabled_sources = [
            name for name in source_names if self.sources[name].is_enabled
        ]

        if not enabled_sources:
            raise Exception("No enabled data sources available")

        last_exception = None

        for source_name in enabled_sources:
            source = self.sources[source_name]

            try:
                result = await self.fetcher.fetch_data(source, method, *args, **kwargs)

                logger.debug(f"Successfully fetched data using {source_name}")
                return result

            except Exception as e:
                logger.warning(f"Data source {source_name} failed: {e}")
                last_exception = e
                continue

        # All sources failed
        raise Exception(f"All data sources failed. Last error: {last_exception}")

    async def batch_fetch_with_failover(
        self,
        requests: List[Dict],
        max_concurrent: int = 10,
    ) -> List[Any]:
        """
        Batch fetch with failover for each request.

        Args:
            requests: List of request dictionaries with method, args, kwargs
            max_concurrent: Maximum concurrent requests

        Returns:
            List of results, None for failed requests
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def fetch_single(request: Dict) -> Any:
            async with semaphore:
                try:
                    return await self.fetch_with_failover(
                        request["method"],
                        *request.get("args", []),
                        sources=request.get("sources"),
                        **request.get("kwargs", {}),
                    )
                except Exception as e:
                    logger.error(f"Batch request failed: {e}")
                    return None

        tasks = [fetch_single(req) for req in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to None
        return [None if isinstance(r, Exception) else r for r in results]

    async def health_check_all(self) -> Dict[str, bool]:
        """Run health checks on all data sources."""
        results = {}

        tasks = []
        for source in self.sources.values():
            task = asyncio.create_task(
                source.health_check(), name=f"health_check_{source.name}"
            )
            tasks.append((source.name, task))

        for source_name, task in tasks:
            try:
                is_healthy = await task
                results[source_name] = is_healthy

                if not is_healthy:
                    logger.warning(f"Data source {source_name} failed health check")
            except Exception as e:
                logger.error(f"Health check error for {source_name}: {e}")
                results[source_name] = False

        return results

    async def start_health_monitoring(self) -> None:
        """Start background health monitoring."""
        if not self.health_monitor or self._health_monitoring_started:
            return

        sources = list(self.sources.values())
        await self.health_monitor.start_monitoring(sources)
        self._health_monitoring_started = True

    async def stop_health_monitoring(self) -> None:
        """Stop background health monitoring."""
        if self.health_monitor and self._health_monitoring_started:
            await self.health_monitor.stop_monitoring()
            self._health_monitoring_started = False

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics for all components."""
        stats = {
            "sources": {},
            "fetcher": self.fetcher.get_stats(),
            "rate_limiter": {},
            "health_monitor": None,
        }

        # Source statistics
        for name, source in self.sources.items():
            stats["sources"][name] = {
                "enabled": source.is_enabled,
                "priority": getattr(source, "_priority", 100),
            }

        # Rate limiter statistics
        for source_name in self.sources:
            stats["rate_limiter"][source_name] = self.rate_limiter.get_stats(
                source_name
            )

        # Health monitor statistics
        if self.health_monitor:
            stats["health_monitor"] = self.health_monitor.get_health_status()

        return stats

    async def cleanup(self) -> None:
        """Clean up resources."""
        await self.stop_health_monitoring()

        # Close any sources that have cleanup methods
        for source in self.sources.values():
            if hasattr(source, "close"):
                try:
                    await source.close()
                except Exception as e:
                    logger.error(f"Error closing source {source.name}: {e}")

    def __repr__(self) -> str:
        enabled_count = len([s for s in self.sources.values() if s.is_enabled])
        return f"DataSourceManager({enabled_count}/{len(self.sources)} sources enabled)"
