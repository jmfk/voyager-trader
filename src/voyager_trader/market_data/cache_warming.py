"""Cache warming strategies for market data."""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from ..models.types import TimeFrame
from .types import Symbol

logger = logging.getLogger(__name__)


class CacheWarmingStrategy:
    """Base class for cache warming strategies."""

    def __init__(self, service, enabled: bool = True):
        self.service = service
        self.enabled = enabled

    async def warm_cache(self) -> None:
        """Warm the cache according to this strategy."""
        raise NotImplementedError


class PopularSymbolsWarming(CacheWarmingStrategy):
    """Warm cache for popular symbols that are frequently accessed."""

    def __init__(
        self,
        service,
        symbols: List[Symbol],
        timeframes: Optional[List[TimeFrame]] = None,
        enabled: bool = True,
    ):
        super().__init__(service, enabled)
        self.symbols = symbols
        self.timeframes = timeframes or [
            TimeFrame.MINUTE_1,
            TimeFrame.MINUTE_5,
            TimeFrame.DAY_1,
        ]

    async def warm_cache(self) -> None:
        """Warm cache for popular symbols with recent data."""
        if not self.enabled:
            return

        logger.info(f"Warming cache for {len(self.symbols)} popular symbols")
        end_time = datetime.utcnow()

        for symbol in self.symbols:
            try:
                for timeframe in self.timeframes:
                    # Warm with last 24 hours for intraday, 30 days for daily
                    if timeframe in [
                        TimeFrame.MINUTE_1,
                        TimeFrame.MINUTE_5,
                        TimeFrame.MINUTE_15,
                    ]:
                        start_time = end_time - timedelta(hours=24)
                    else:
                        start_time = end_time - timedelta(days=30)

                    # Pre-fetch data to warm the cache
                    await self.service.get_historical_ohlcv(
                        symbol=symbol,
                        timeframe=timeframe,
                        start_date=start_time,
                        end_date=end_time,
                        force_refresh=False,  # Use cache if available
                    )

                    # Small delay to avoid overwhelming providers
                    await asyncio.sleep(0.1)

            except Exception as e:
                logger.warning(f"Failed to warm cache for {symbol}: {e}")

        logger.info("Cache warming completed for popular symbols")


class ScheduledWarming(CacheWarmingStrategy):
    """Warm cache on a schedule (e.g., before market open)."""

    def __init__(
        self,
        service,
        symbols: List[Symbol],
        schedule_times: List[str],
        timeframes: Optional[List[TimeFrame]] = None,
        enabled: bool = True,
    ):
        super().__init__(service, enabled)
        self.symbols = symbols
        self.schedule_times = schedule_times  # Format: "HH:MM"
        self.timeframes = timeframes or [TimeFrame.DAY_1, TimeFrame.HOUR_1]
        self._last_run: Optional[datetime] = None

    async def warm_cache(self) -> None:
        """Check if it's time to warm cache and do so if needed."""
        if not self.enabled:
            return

        current_time = datetime.utcnow()
        current_time_str = current_time.strftime("%H:%M")

        # Check if current time matches any scheduled time
        should_run = False
        for schedule_time in self.schedule_times:
            if current_time_str == schedule_time:
                # Only run once per scheduled time
                if (
                    not self._last_run
                    or (current_time - self._last_run).total_seconds() > 3600
                ):
                    should_run = True
                    break

        if not should_run:
            return

        logger.info(f"Starting scheduled cache warming at {current_time_str}")
        self._last_run = current_time

        end_time = current_time

        for symbol in self.symbols:
            try:
                for timeframe in self.timeframes:
                    # Get appropriate historical range
                    if timeframe == TimeFrame.DAY_1:
                        start_time = end_time - timedelta(days=90)  # 3 months
                    elif timeframe == TimeFrame.HOUR_1:
                        start_time = end_time - timedelta(days=7)  # 1 week
                    else:
                        start_time = end_time - timedelta(days=1)  # 1 day

                    await self.service.get_historical_ohlcv(
                        symbol=symbol,
                        timeframe=timeframe,
                        start_date=start_time,
                        end_date=end_time,
                        force_refresh=True,  # Refresh to get latest data
                    )

                    await asyncio.sleep(0.2)  # Longer delay for scheduled warming

            except Exception as e:
                logger.warning(
                    f"Failed to warm cache for {symbol} in scheduled run: {e}"
                )

        logger.info("Scheduled cache warming completed")


class AdaptiveWarming(CacheWarmingStrategy):
    """Adaptively warm cache based on access patterns."""

    def __init__(self, service, min_access_count: int = 3, enabled: bool = True):
        super().__init__(service, enabled)
        self.min_access_count = min_access_count
        self.access_counts: Dict[str, int] = {}
        self.last_warming: Dict[str, datetime] = {}

    def record_access(self, symbol: Symbol, timeframe: TimeFrame) -> None:
        """Record access to symbol/timeframe combination."""
        key = f"{symbol}:{timeframe}"
        self.access_counts[key] = self.access_counts.get(key, 0) + 1

    async def warm_cache(self) -> None:
        """Warm cache for frequently accessed symbol/timeframe combinations."""
        if not self.enabled:
            return

        current_time = datetime.utcnow()
        candidates = []

        # Find candidates for warming
        for key, count in self.access_counts.items():
            if count >= self.min_access_count:
                # Check if we've warmed this recently
                if (
                    key not in self.last_warming
                    or (current_time - self.last_warming[key]).total_seconds() > 3600
                ):  # 1 hour
                    candidates.append(key)

        if not candidates:
            return

        logger.info(
            f"Adaptive warming for {len(candidates)} frequently accessed combinations"
        )

        for key in candidates[:20]:  # Limit to top 20 to avoid overload
            try:
                symbol, timeframe_str = key.split(":", 1)
                timeframe = TimeFrame(timeframe_str)

                # Determine appropriate time range
                if "MINUTE" in timeframe_str:
                    start_time = current_time - timedelta(hours=6)
                elif "HOUR" in timeframe_str:
                    start_time = current_time - timedelta(days=3)
                else:
                    start_time = current_time - timedelta(days=30)

                await self.service.get_historical_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=start_time,
                    end_date=current_time,
                    force_refresh=False,
                )

                self.last_warming[key] = current_time
                await asyncio.sleep(0.1)

            except Exception as e:
                logger.warning(f"Failed to adaptively warm cache for {key}: {e}")


class CacheWarmer:
    """Coordinates multiple cache warming strategies."""

    def __init__(self, service):
        self.service = service
        self.strategies: List[CacheWarmingStrategy] = []
        self._running = False
        self._task: Optional[asyncio.Task] = None

    def add_strategy(self, strategy: CacheWarmingStrategy) -> None:
        """Add a warming strategy."""
        self.strategies.append(strategy)

    def remove_strategy(self, strategy: CacheWarmingStrategy) -> None:
        """Remove a warming strategy."""
        if strategy in self.strategies:
            self.strategies.remove(strategy)

    async def start(self, interval_minutes: int = 60) -> None:
        """Start the cache warming background task."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._warming_loop(interval_minutes))
        logger.info(f"Cache warmer started with {len(self.strategies)} strategies")

    async def stop(self) -> None:
        """Stop the cache warming background task."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Cache warmer stopped")

    async def warm_now(self) -> None:
        """Immediately run all warming strategies."""
        logger.info("Running immediate cache warming")
        for strategy in self.strategies:
            try:
                await strategy.warm_cache()
            except Exception as e:
                logger.error(
                    f"Error in warming strategy {strategy.__class__.__name__}: {e}"
                )

    async def _warming_loop(self, interval_minutes: int) -> None:
        """Background loop that runs warming strategies."""
        while self._running:
            try:
                await self.warm_now()
                await asyncio.sleep(interval_minutes * 60)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in warming loop: {e}")
                await asyncio.sleep(60)  # Wait a minute before retrying
