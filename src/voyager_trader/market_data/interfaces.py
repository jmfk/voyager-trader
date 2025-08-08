"""Abstract interfaces for market data integration."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import AsyncGenerator, Dict, List, Optional

from ..models.market import OHLCV, OrderBook, TickData
from ..models.types import TimeFrame

# Import centralized Symbol type
from .types import Symbol


class DataSource(ABC):
    """Abstract base class for market data sources."""

    def __init__(self, name: str, config: Optional[Dict] = None):
        self.name = name
        self.config = config or {}
        self._enabled = True

    @abstractmethod
    async def get_historical_ohlcv(
        self,
        symbol: Symbol,
        timeframe: TimeFrame,
        start_date: datetime,
        end_date: datetime,
        limit: Optional[int] = None,
    ) -> List[OHLCV]:
        """Fetch historical OHLCV data for a symbol."""

    @abstractmethod
    async def get_latest_ohlcv(
        self,
        symbol: Symbol,
        timeframe: TimeFrame,
    ) -> Optional[OHLCV]:
        """Fetch the latest OHLCV bar for a symbol."""

    @abstractmethod
    async def stream_tick_data(
        self,
        symbol: Symbol,
    ) -> AsyncGenerator[TickData, None]:
        """Stream real-time tick data for a symbol."""

    @abstractmethod
    async def get_order_book(
        self,
        symbol: Symbol,
        depth: int = 10,
    ) -> Optional[OrderBook]:
        """Fetch current order book snapshot."""

    @abstractmethod
    async def get_supported_symbols(self) -> List[Symbol]:
        """Get list of symbols supported by this data source."""

    @abstractmethod
    async def validate_symbol(self, symbol: Symbol) -> bool:
        """Check if a symbol is supported by this data source."""

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the data source is healthy and accessible."""

    @property
    def is_enabled(self) -> bool:
        """Check if data source is enabled."""
        return self._enabled

    def enable(self) -> None:
        """Enable the data source."""
        self._enabled = True

    def disable(self) -> None:
        """Disable the data source."""
        self._enabled = False

    def __str__(self) -> str:
        return f"DataSource({self.name})"

    def __repr__(self) -> str:
        return f"DataSource(name={self.name}, enabled={self._enabled})"


class DataFetcher(ABC):
    """Abstract interface for data fetching with rate limiting and retries."""

    @abstractmethod
    async def fetch_data(
        self,
        source: DataSource,
        method: str,
        *args,
        **kwargs,
    ) -> any:
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

    @abstractmethod
    async def batch_fetch(
        self,
        requests: List[Dict],
    ) -> List[any]:
        """
        Fetch multiple data requests concurrently.

        Args:
            requests: List of request dictionaries with source, method, args, kwargs

        Returns:
            List of results corresponding to each request
        """
