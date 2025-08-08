"""Yahoo Finance market data source implementation."""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import AsyncGenerator, Dict, List, Optional

import yfinance as yf

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


class YahooFinanceDataSource(DataSource):
    """Yahoo Finance data source implementation using yfinance library."""

    TIMEFRAME_MAPPING = {
        TimeFrame.MINUTE_1: "1m",
        TimeFrame.MINUTE_5: "5m",
        TimeFrame.MINUTE_15: "15m",
        TimeFrame.MINUTE_30: "30m",
        TimeFrame.HOUR_1: "1h",
        TimeFrame.HOUR_4: "4h",
        TimeFrame.DAY_1: "1d",
        TimeFrame.WEEK_1: "1wk",
        TimeFrame.MONTH_1: "1mo",
    }

    def __init__(self, config: Optional[Dict] = None):
        super().__init__("yahoo_finance", config)
        self.normalizer = DataNormalizer()
        self._supported_symbols: Optional[List[Symbol]] = None

        # Yahoo Finance specific settings
        self.max_intraday_period = self.config.get("max_intraday_period", "60d")
        self.max_daily_period = self.config.get("max_daily_period", "max")

    async def get_historical_ohlcv(
        self,
        symbol: Symbol,
        timeframe: TimeFrame,
        start_date: datetime,
        end_date: datetime,
        limit: Optional[int] = None,
    ) -> List[OHLCV]:
        """Fetch historical OHLCV data from Yahoo Finance."""
        if not self.is_enabled:
            return []

        if timeframe not in self.TIMEFRAME_MAPPING:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

        yf_interval = self.TIMEFRAME_MAPPING[timeframe]

        try:
            # Run yfinance in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            raw_data = await loop.run_in_executor(
                None,
                self._fetch_yfinance_data,
                symbol,
                yf_interval,
                start_date,
                end_date,
            )

            if raw_data.empty:
                logger.warning(f"No data returned from Yahoo Finance for {symbol}")
                return []

            # Normalize data
            ohlcv_data = self.normalizer.normalize_ohlcv_from_yahoo(
                symbol, timeframe, raw_data
            )

            # Apply limit if specified
            if limit and limit < len(ohlcv_data):
                ohlcv_data = ohlcv_data[-limit:]  # Get most recent

            logger.info(
                f"Fetched {len(ohlcv_data)} OHLCV bars for {symbol} "
                f"({timeframe}) from Yahoo Finance"
            )

            return ohlcv_data

        except Exception as e:
            logger.error(f"Failed to fetch OHLCV data from Yahoo Finance: {e}")
            raise

    def _fetch_yfinance_data(
        self, symbol: Symbol, interval: str, start_date: datetime, end_date: datetime
    ):
        """Fetch data using yfinance (synchronous)."""
        ticker = yf.Ticker(symbol)

        # Convert datetime to string format expected by yfinance
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")

        return ticker.history(
            start=start_str,
            end=end_str,
            interval=interval,
            auto_adjust=True,
            prepost=True,
            actions=False,
        )

    async def get_latest_ohlcv(
        self,
        symbol: Symbol,
        timeframe: TimeFrame,
    ) -> Optional[OHLCV]:
        """Fetch the latest OHLCV bar."""
        try:
            # Get recent data
            end_date = datetime.utcnow()

            # Determine start date based on timeframe
            if timeframe in [
                TimeFrame.MINUTE_1,
                TimeFrame.MINUTE_5,
                TimeFrame.MINUTE_15,
                TimeFrame.MINUTE_30,
                TimeFrame.HOUR_1,
            ]:
                start_date = end_date - timedelta(days=5)  # Last 5 days for intraday
            else:
                start_date = end_date - timedelta(days=30)  # Last month for daily+

            data = await self.get_historical_ohlcv(
                symbol, timeframe, start_date, end_date, limit=1
            )

            return data[0] if data else None

        except Exception as e:
            logger.error(f"Failed to fetch latest OHLCV from Yahoo Finance: {e}")
            return None

    async def stream_tick_data(
        self,
        symbol: Symbol,
    ) -> AsyncGenerator[TickData, None]:
        """
        Stream real-time tick data using Yahoo Finance.

        Yahoo Finance doesn't provide true real-time streaming,
        so we poll for the latest 1-minute data.
        """
        logger.info(f"Starting Yahoo Finance tick stream for {symbol}")

        last_timestamp = None

        while True:
            try:
                # Get latest 1-minute bar
                latest_ohlcv = await self.get_latest_ohlcv(symbol, TimeFrame.MINUTE_1)

                if latest_ohlcv and (
                    not last_timestamp or latest_ohlcv.timestamp > last_timestamp
                ):
                    # Convert OHLCV to tick data
                    tick = TickData(
                        symbol=create_symbol(symbol),
                        timestamp=latest_ohlcv.timestamp,
                        price=latest_ohlcv.close,
                        size=latest_ohlcv.volume,
                        tick_type="trade",
                        exchange="yahoo_finance",
                    )

                    last_timestamp = latest_ohlcv.timestamp
                    yield tick

                # Wait before next poll
                await asyncio.sleep(30)  # 30 seconds

            except Exception as e:
                logger.error(f"Error in Yahoo Finance tick stream: {e}")
                await asyncio.sleep(10)  # Wait before retrying

    async def get_order_book(
        self,
        symbol: Symbol,
        depth: int = 10,
    ) -> Optional[OrderBook]:
        """Yahoo Finance doesn't provide order book data."""
        logger.warning("Yahoo Finance doesn't provide order book data")
        return None

    async def get_supported_symbols(self) -> List[Symbol]:
        """Get list of commonly supported symbols."""
        if self._supported_symbols is None:
            # Yahoo Finance supports a vast number of symbols
            # Return a subset of commonly traded ones
            self._supported_symbols = [
                # US Stocks
                "AAPL",
                "GOOGL",
                "GOOG",
                "MSFT",
                "AMZN",
                "TSLA",
                "META",
                "NFLX",
                "NVDA",
                "AMD",
                "INTC",
                "CRM",
                "ORCL",
                "ADBE",
                "PYPL",
                "UBER",
                "LYFT",
                # ETFs
                "SPY",
                "QQQ",
                "IWM",
                "VTI",
                "VOO",
                "IVV",
                "VEA",
                "VWO",
                "AGG",
                "BND",
                # Indices (with Yahoo suffix)
                "^GSPC",
                "^DJI",
                "^IXIC",
                "^RUT",
                "^VIX",
                # Cryptocurrencies
                "BTC-USD",
                "ETH-USD",
                "ADA-USD",
                "DOT-USD",
                "LINK-USD",
                # Forex
                "EURUSD=X",
                "GBPUSD=X",
                "USDJPY=X",
                "USDCAD=X",
                "AUDUSD=X",
                # Commodities
                "GC=F",
                "SI=F",
                "CL=F",
                "NG=F",
                "ZC=F",
                "ZS=F",
            ]

        return self._supported_symbols

    async def validate_symbol(self, symbol: Symbol) -> bool:
        """Check if symbol is supported by trying to fetch data."""
        try:
            # Try to get basic info about the symbol
            loop = asyncio.get_event_loop()
            ticker_info = await loop.run_in_executor(
                None, self._get_ticker_info, symbol
            )

            # Check if we got valid info
            return bool(ticker_info and ticker_info.get("regularMarketPrice"))

        except Exception as e:
            logger.debug(f"Symbol validation failed for {symbol}: {e}")
            return False

    def _get_ticker_info(self, symbol: Symbol) -> Dict:
        """Get ticker info synchronously."""
        try:
            ticker = yf.Ticker(symbol)
            return ticker.info
        except Exception:
            return {}

    async def health_check(self) -> bool:
        """Check if Yahoo Finance is accessible."""
        try:
            # Try to fetch data for a known symbol
            test_result = await self.validate_symbol("AAPL")
            return test_result
        except Exception as e:
            logger.error(f"Yahoo Finance health check failed: {e}")
            return False

    def get_symbol_info(self, symbol: Symbol) -> Optional[Dict]:
        """
        Get detailed information about a symbol.

        This is a Yahoo Finance specific method that provides
        additional symbol metadata.
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            return {
                "name": info.get("longName"),
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "market_cap": info.get("marketCap"),
                "exchange": info.get("exchange"),
                "currency": info.get("currency"),
                "country": info.get("country"),
                "website": info.get("website"),
                "description": info.get("longBusinessSummary"),
            }
        except Exception as e:
            logger.error(f"Failed to get symbol info for {symbol}: {e}")
            return None
