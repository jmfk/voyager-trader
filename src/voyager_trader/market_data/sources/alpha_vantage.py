"""Alpha Vantage market data source implementation."""

import asyncio
import logging
from datetime import datetime
from typing import AsyncGenerator, Dict, List, Optional

import aiohttp

from ...models.market import OHLCV, OrderBook, TickData
from ...models.types import AssetClass
from ...models.types import Symbol as SymbolModel
from ...models.types import TimeFrame


def create_symbol(code: str) -> SymbolModel:
    """Create a Symbol object from a string code."""
    return SymbolModel(code=code, asset_class=AssetClass.EQUITY)


Symbol = str  # Use string symbols for market data
from ..interfaces import DataSource
from ..normalizer import DataNormalizer

logger = logging.getLogger(__name__)


class AlphaVantageDataSource(DataSource):
    """Alpha Vantage API data source implementation."""

    BASE_URL = "https://www.alphavantage.co/query"

    TIMEFRAME_MAPPING = {
        TimeFrame.MINUTE_1: "1min",
        TimeFrame.MINUTE_5: "5min",
        TimeFrame.MINUTE_15: "15min",
        TimeFrame.MINUTE_30: "30min",
        TimeFrame.HOUR_1: "60min",
        TimeFrame.DAY_1: "Daily",
        TimeFrame.WEEK_1: "Weekly",
        TimeFrame.MONTH_1: "Monthly",
    }

    def __init__(self, config: Optional[Dict] = None):
        super().__init__("alpha_vantage", config)
        self.api_key = self.config.get("api_key")
        if not self.api_key:
            raise ValueError("Alpha Vantage API key is required")

        self.session: Optional[aiohttp.ClientSession] = None
        self.normalizer = DataNormalizer()
        self._supported_symbols: Optional[List[Symbol]] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session

    async def _make_request(self, params: Dict[str, str]) -> Dict:
        """Make API request with error handling."""
        params["apikey"] = self.api_key

        session = await self._get_session()

        try:
            async with session.get(self.BASE_URL, params=params) as response:
                if response.status != 200:
                    raise Exception(f"API request failed with status {response.status}")

                data = await response.json()

                # Check for API errors
                if "Error Message" in data:
                    raise Exception(f"Alpha Vantage error: {data['Error Message']}")
                if "Note" in data:
                    # Rate limit or other notice
                    logger.warning(f"Alpha Vantage notice: {data['Note']}")

                return data

        except aiohttp.ClientError as e:
            raise Exception(f"HTTP client error: {e}")
        except asyncio.TimeoutError:
            raise Exception("Request timeout")

    async def get_historical_ohlcv(
        self,
        symbol: Symbol,
        timeframe: TimeFrame,
        start_date: datetime,
        end_date: datetime,
        limit: Optional[int] = None,
    ) -> List[OHLCV]:
        """Fetch historical OHLCV data from Alpha Vantage."""
        if not self.is_enabled:
            return []

        if timeframe not in self.TIMEFRAME_MAPPING:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

        av_timeframe = self.TIMEFRAME_MAPPING[timeframe]

        # Determine function based on timeframe
        if av_timeframe in ["1min", "5min", "15min", "30min", "60min"]:
            function = "TIME_SERIES_INTRADAY"
            params = {
                "function": function,
                "symbol": symbol,
                "interval": av_timeframe,
                "outputsize": "full",  # Get maximum data
            }
        else:
            function = f"TIME_SERIES_{av_timeframe.upper()}"
            params = {
                "function": function,
                "symbol": symbol,
            }

        try:
            raw_data = await self._make_request(params)
            ohlcv_data = self.normalizer.normalize_ohlcv_from_alpha_vantage(
                symbol, timeframe, raw_data
            )

            # Filter by date range
            filtered_data = [
                bar for bar in ohlcv_data if start_date <= bar.timestamp <= end_date
            ]

            # Apply limit if specified
            if limit and limit < len(filtered_data):
                filtered_data = filtered_data[-limit:]  # Get most recent

            logger.info(
                f"Fetched {len(filtered_data)} OHLCV bars for {symbol} "
                f"({timeframe}) from Alpha Vantage"
            )

            return filtered_data

        except Exception as e:
            logger.error(f"Failed to fetch OHLCV data from Alpha Vantage: {e}")
            raise

    async def get_latest_ohlcv(
        self,
        symbol: Symbol,
        timeframe: TimeFrame,
    ) -> Optional[OHLCV]:
        """Fetch the latest OHLCV bar."""
        try:
            # Get recent data and return the latest
            end_date = datetime.utcnow()
            start_date = datetime.utcnow().replace(day=1)  # From beginning of month

            data = await self.get_historical_ohlcv(
                symbol, timeframe, start_date, end_date, limit=1
            )

            return data[0] if data else None

        except Exception as e:
            logger.error(f"Failed to fetch latest OHLCV from Alpha Vantage: {e}")
            return None

    async def stream_tick_data(
        self,
        symbol: Symbol,
    ) -> AsyncGenerator[TickData, None]:
        """
        Stream real-time tick data (Alpha Vantage doesn't support real-time streaming).

        This implementation polls for the latest data at regular intervals.
        """
        logger.warning(
            "Alpha Vantage doesn't support real-time streaming. "
            "Using polling with 1-minute intervals."
        )

        last_timestamp = None

        while True:
            try:
                # Get latest 1-minute bar
                latest_ohlcv = await self.get_latest_ohlcv(symbol, TimeFrame.MINUTE_1)

                if latest_ohlcv and (
                    not last_timestamp or latest_ohlcv.timestamp > last_timestamp
                ):
                    # Convert OHLCV to tick data (using close price)
                    tick = TickData(
                        symbol=create_symbol(symbol),
                        timestamp=latest_ohlcv.timestamp,
                        price=latest_ohlcv.close,
                        size=latest_ohlcv.volume,
                        tick_type="trade",
                        exchange="alpha_vantage",
                    )

                    last_timestamp = latest_ohlcv.timestamp
                    yield tick

                # Wait before next poll (Alpha Vantage has rate limits)
                await asyncio.sleep(60)  # 1 minute

            except Exception as e:
                logger.error(f"Error in tick data stream: {e}")
                await asyncio.sleep(10)  # Wait before retrying

    async def get_order_book(
        self,
        symbol: Symbol,
        depth: int = 10,
    ) -> Optional[OrderBook]:
        """Alpha Vantage doesn't provide order book data."""
        logger.warning("Alpha Vantage doesn't provide order book data")
        return None

    async def get_supported_symbols(self) -> List[Symbol]:
        """Get list of supported symbols."""
        if self._supported_symbols is None:
            # Alpha Vantage supports most US stocks and major cryptocurrencies
            # For now, return a subset. In production, this could be fetched from their API
            self._supported_symbols = [
                "AAPL",
                "GOOGL",
                "MSFT",
                "AMZN",
                "TSLA",
                "NVDA",
                "META",
                "NFLX",
                "AMD",
                "INTC",
                "BTC-USD",
                "ETH-USD",
                "SPY",
                "QQQ",
                "IWM",
            ]

        return self._supported_symbols

    async def validate_symbol(self, symbol: Symbol) -> bool:
        """Check if symbol is supported."""
        try:
            # Try to fetch latest data for the symbol
            result = await self.get_latest_ohlcv(symbol, TimeFrame.DAY)
            return result is not None
        except Exception:
            return False

    async def health_check(self) -> bool:
        """Check if Alpha Vantage API is accessible."""
        try:
            params = {
                "function": "GLOBAL_QUOTE",
                "symbol": "AAPL",
            }
            await self._make_request(params)
            return True
        except Exception as e:
            logger.error(f"Alpha Vantage health check failed: {e}")
            return False

    async def close(self) -> None:
        """Close HTTP session."""
        if self.session:
            await self.session.close()
            self.session = None

    def __del__(self):
        """Cleanup on deletion."""
        if self.session and not self.session.closed:
            # Schedule session cleanup
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self.close())
