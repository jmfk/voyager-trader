"""Main Market Data Service orchestrator."""

import logging
import time
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional

from ..models.market import OHLCV, OrderBook, TickData
from ..models.types import TimeFrame
from .cache import DataCache
from .manager import DataSourceManager
from .monitoring import record_request_metrics
from .normalizer import DataNormalizer
from .rate_limiter import RateLimiter
from .types import Symbol
from .validator import DataValidator

logger = logging.getLogger(__name__)


class MarketDataService:
    """
    Central orchestrator for all market data operations.

    Provides a unified interface for fetching, validating, and caching
    market data from multiple sources with automatic failover.
    """

    def __init__(
        self,
        sources_config: Optional[Dict[str, Dict]] = None,
        cache_config: Optional[Dict[str, Any]] = None,
        validator_config: Optional[Dict[str, Any]] = None,
        enable_caching: bool = True,
        enable_validation: bool = True,
        enable_health_monitoring: bool = True,
    ):
        # Initialize components
        self.rate_limiter = RateLimiter()
        self.source_manager = DataSourceManager(
            sources_config, self.rate_limiter, enable_health_monitoring
        )

        # Optional components
        self.cache = DataCache(**(cache_config or {})) if enable_caching else None
        self.validator = DataValidator(validator_config) if enable_validation else None
        self.normalizer = DataNormalizer()

        # Configuration flags
        self.enable_caching = enable_caching
        self.enable_validation = enable_validation

        # Service state
        self._started = False

    async def start(self) -> None:
        """Start the market data service."""
        if self._started:
            return

        await self.source_manager.start_health_monitoring()
        self._started = True

        logger.info("Market Data Service started")

    async def stop(self) -> None:
        """Stop the market data service."""
        if not self._started:
            return

        await self.source_manager.cleanup()
        self._started = False

        logger.info("Market Data Service stopped")

    async def get_historical_ohlcv(
        self,
        symbol: Symbol,
        timeframe: TimeFrame,
        start_date: datetime,
        end_date: datetime,
        limit: Optional[int] = None,
        sources: Optional[List[str]] = None,
        force_refresh: bool = False,
    ) -> List[OHLCV]:
        """
        Get historical OHLCV data with caching and validation.

        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            start_date: Start date for data
            end_date: End date for data
            limit: Maximum number of bars to return
            sources: Preferred data sources (optional)
            force_refresh: Skip cache and fetch fresh data

        Returns:
            List of validated OHLCV bars
        """
        start_time = time.time()
        cache_hit = False

        # Check cache first (unless force refresh)
        if self.enable_caching and not force_refresh:
            cached_data = await self.cache.get(
                "historical_ohlcv",
                "get_historical_ohlcv",
                args=(symbol, timeframe, start_date, end_date, limit),
            )
            if cached_data:
                cache_hit = True
                duration_ms = (time.time() - start_time) * 1000
                record_request_metrics(
                    provider="cache",
                    method="get_historical_ohlcv",
                    symbol=symbol,
                    duration_ms=duration_ms,
                    success=True,
                    cache_hit=True,
                    data_size_bytes=len(str(cached_data).encode()),
                )
                logger.debug(
                    f"Retrieved {len(cached_data)} OHLCV bars from cache for {symbol}"
                )
                return cached_data

        # Fetch from data sources with failover
        try:
            data = await self.source_manager.fetch_with_failover(
                "get_historical_ohlcv",
                symbol,
                timeframe,
                start_date,
                end_date,
                limit,
                sources=sources,
            )

            # Validate data if validation is enabled
            if self.enable_validation and data:
                data = self.validator.validate_batch_ohlcv(data)

            # Cache the result
            if self.enable_caching and data:
                await self.cache.set(
                    "historical_ohlcv",
                    "get_historical_ohlcv",
                    data,
                    args=(symbol, timeframe, start_date, end_date, limit),
                    memory_ttl=300,  # 5 minutes in memory
                    disk_ttl=3600,  # 1 hour on disk
                )

            logger.info(f"Fetched {len(data) if data else 0} OHLCV bars for {symbol}")
            return data or []

        except Exception as e:
            logger.error(f"Failed to fetch historical OHLCV for {symbol}: {e}")
            raise

    async def get_latest_ohlcv(
        self,
        symbol: Symbol,
        timeframe: TimeFrame,
        sources: Optional[List[str]] = None,
        force_refresh: bool = False,
    ) -> Optional[OHLCV]:
        """
        Get the latest OHLCV bar for a symbol.

        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            sources: Preferred data sources (optional)
            force_refresh: Skip cache and fetch fresh data

        Returns:
            Latest OHLCV bar or None
        """
        # Check cache first (with shorter TTL for latest data)
        if self.enable_caching and not force_refresh:
            cached_data = await self.cache.get(
                "latest_ohlcv",
                "get_latest_ohlcv",
                args=(symbol, timeframe),
            )
            if cached_data:
                logger.debug(f"Retrieved latest OHLCV from cache for {symbol}")
                return cached_data

        try:
            data = await self.source_manager.fetch_with_failover(
                "get_latest_ohlcv",
                symbol,
                timeframe,
                sources=sources,
            )

            # Validate data
            if self.enable_validation and data:
                self.validator.validate_ohlcv(data)

            # Cache with short TTL
            if self.enable_caching and data:
                await self.cache.set(
                    "latest_ohlcv",
                    "get_latest_ohlcv",
                    data,
                    args=(symbol, timeframe),
                    memory_ttl=60,  # 1 minute in memory
                    disk_ttl=300,  # 5 minutes on disk
                )

            return data

        except Exception as e:
            logger.error(f"Failed to fetch latest OHLCV for {symbol}: {e}")
            raise

    async def stream_tick_data(
        self,
        symbol: Symbol,
        sources: Optional[List[str]] = None,
    ) -> AsyncGenerator[TickData, None]:
        """
        Stream real-time tick data for a symbol.

        Args:
            symbol: Trading symbol
            sources: Preferred data sources (optional)

        Yields:
            Validated tick data
        """
        # Get the primary data source for streaming
        if sources:
            source_name = sources[0]
            source = self.source_manager.get_source(source_name)
        else:
            source = self.source_manager.get_primary_source()

        if not source:
            raise Exception("No available data source for streaming")

        logger.info(f"Starting tick stream for {symbol} using {source.name}")

        previous_tick = None

        async for tick in source.stream_tick_data(symbol):
            try:
                # Validate tick data
                if self.enable_validation:
                    self.validator.validate_tick_data(tick, previous_tick)
                    previous_tick = tick

                yield tick

            except Exception as e:
                logger.warning(f"Invalid tick data for {symbol}: {e}")
                continue

    async def get_order_book(
        self,
        symbol: Symbol,
        depth: int = 10,
        sources: Optional[List[str]] = None,
        force_refresh: bool = False,
    ) -> Optional[OrderBook]:
        """
        Get order book snapshot for a symbol.

        Args:
            symbol: Trading symbol
            depth: Number of levels to include
            sources: Preferred data sources (optional)
            force_refresh: Skip cache and fetch fresh data

        Returns:
            Order book snapshot or None
        """
        # Check cache (very short TTL for order book data)
        if self.enable_caching and not force_refresh:
            cached_data = await self.cache.get(
                "order_book",
                "get_order_book",
                args=(symbol, depth),
            )
            if cached_data:
                return cached_data

        try:
            data = await self.source_manager.fetch_with_failover(
                "get_order_book",
                symbol,
                depth,
                sources=sources,
            )

            # Validate order book
            if self.enable_validation and data:
                self.validator.validate_order_book(data)

            # Cache with very short TTL
            if self.enable_caching and data:
                await self.cache.set(
                    "order_book",
                    "get_order_book",
                    data,
                    args=(symbol, depth),
                    memory_ttl=5,  # 5 seconds in memory
                    disk_ttl=30,  # 30 seconds on disk
                )

            return data

        except Exception as e:
            logger.error(f"Failed to fetch order book for {symbol}: {e}")
            raise

    async def get_supported_symbols(
        self,
        sources: Optional[List[str]] = None,
    ) -> List[Symbol]:
        """
        Get list of symbols supported across data sources.

        Args:
            sources: Specific sources to check (optional)

        Returns:
            List of supported symbols
        """
        if sources:
            # Get symbols from specific sources
            all_symbols = set()
            for source_name in sources:
                source = self.source_manager.get_source(source_name)
                if source and source.is_enabled:
                    try:
                        symbols = await source.get_supported_symbols()
                        all_symbols.update(symbols)
                    except Exception as e:
                        logger.warning(f"Failed to get symbols from {source_name}: {e}")

            return sorted(list(all_symbols))
        else:
            # Get symbols from all enabled sources
            all_symbols = set()
            for source in self.source_manager.get_available_sources():
                try:
                    symbols = await source.get_supported_symbols()
                    all_symbols.update(symbols)
                except Exception as e:
                    logger.warning(f"Failed to get symbols from {source.name}: {e}")

            return sorted(list(all_symbols))

    async def validate_symbol(
        self,
        symbol: Symbol,
        sources: Optional[List[str]] = None,
    ) -> bool:
        """
        Validate that a symbol is supported by at least one data source.

        Args:
            symbol: Symbol to validate
            sources: Specific sources to check (optional)

        Returns:
            True if symbol is supported
        """
        if sources:
            source_list = [self.source_manager.get_source(name) for name in sources]
            source_list = [s for s in source_list if s and s.is_enabled]
        else:
            source_list = self.source_manager.get_available_sources()

        # Check if any source supports the symbol
        for source in source_list:
            try:
                if await source.validate_symbol(symbol):
                    return True
            except Exception as e:
                logger.debug(f"Symbol validation failed for {source.name}: {e}")
                continue

        return False

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform comprehensive health check of the service.

        Returns:
            Health status dictionary
        """
        health_status = {
            "service_started": self._started,
            "timestamp": datetime.utcnow().isoformat(),
            "components": {},
        }

        # Check data sources
        source_health = await self.source_manager.health_check_all()
        health_status["components"]["data_sources"] = source_health

        # Check cache if enabled
        if self.cache:
            cache_stats = await self.cache.get_stats()
            health_status["components"]["cache"] = cache_stats

        # Overall health
        healthy_sources = sum(1 for status in source_health.values() if status)
        health_status["overall_health"] = {
            "healthy": healthy_sources > 0 and self._started,
            "healthy_sources": healthy_sources,
            "total_sources": len(source_health),
        }

        return health_status

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive service statistics."""
        stats = {
            "service_started": self._started,
            "configuration": {
                "caching_enabled": self.enable_caching,
                "validation_enabled": self.enable_validation,
            },
            "components": self.source_manager.get_stats(),
        }

        # Add cache stats placeholder (would need async context to call get_stats)
        if self.cache:
            stats["components"]["cache"] = {"async_stats_available": True}

        # Add validator stats
        if self.validator:
            stats["components"]["validator"] = self.validator.get_validation_stats()

        return stats

    async def clear_cache(self) -> None:
        """Clear all cached data."""
        if self.cache:
            await self.cache.clear()
            logger.info("Market data cache cleared")

    def add_data_source(
        self, source_name: str, config: Dict[str, Any], priority: int = 100
    ) -> None:
        """
        Add a new data source at runtime.

        Args:
            source_name: Name of the data source type
            config: Configuration for the data source
            priority: Priority for failover (lower = higher priority)
        """
        source = self.source_manager.registry.create_source(source_name, config)
        self.source_manager.add_source(source, priority)

    def remove_data_source(self, source_name: str) -> bool:
        """
        Remove a data source at runtime.

        Args:
            source_name: Name of the data source to remove

        Returns:
            True if source was removed
        """
        return self.source_manager.remove_source(source_name)

    def __repr__(self) -> str:
        enabled_sources = len(self.source_manager.get_available_sources())
        return f"MarketDataService(started={self._started}, sources={enabled_sources})"
