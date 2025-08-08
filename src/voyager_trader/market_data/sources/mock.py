"""Mock market data source for testing and development."""

import asyncio
import logging
import random
from datetime import datetime, timedelta
from decimal import Decimal
from typing import AsyncGenerator, Dict, List, Optional

from ...models.market import OHLCV, OrderBook, OrderBookLevel, TickData
from ...models.types import AssetClass
from ...models.types import Symbol as SymbolModel
from ...models.types import TimeFrame


def create_symbol(code: str) -> SymbolModel:
    """Create a Symbol object from a string code."""
    return SymbolModel(code=code, asset_class=AssetClass.EQUITY)


Symbol = str  # Use string symbols for market data
from ..interfaces import DataSource

logger = logging.getLogger(__name__)


class MockDataSource(DataSource):
    """Mock data source that generates realistic market data for testing."""

    def __init__(self, config: Optional[Dict] = None):
        super().__init__("mock", config)

        # Configuration
        self.base_price = Decimal(str(self.config.get("base_price", 100.0)))
        self.volatility = self.config.get("volatility", 0.02)  # 2% daily volatility
        self.trend = self.config.get("trend", 0.0)  # No trend by default
        self.volume_base = self.config.get("volume_base", 1000000)

        # State for consistent data generation
        self._price_states: Dict[Symbol, Decimal] = {}
        self._last_timestamps: Dict[str, datetime] = {}

        # Predefined symbol list
        self._supported_symbols = [
            "MOCK_AAPL",
            "MOCK_GOOGL",
            "MOCK_MSFT",
            "MOCK_AMZN",
            "MOCK_TSLA",
            "MOCK_NVDA",
            "MOCK_META",
            "MOCK_NFLX",
            "MOCK_SPY",
            "MOCK_QQQ",
        ]

    def _initialize_price_state(self, symbol: Symbol) -> Decimal:
        """Initialize price state for a symbol."""
        if symbol not in self._price_states:
            # Use symbol hash to create consistent base price
            symbol_hash = hash(symbol) % 1000
            base_offset = Decimal(str(symbol_hash / 10.0))  # 0-99.9 offset
            self._price_states[symbol] = self.base_price + base_offset

        return self._price_states[symbol]

    def _generate_price_move(
        self, current_price: Decimal, time_delta_minutes: float = 1.0
    ) -> Decimal:
        """Generate realistic price movement."""
        # Scale volatility by time
        scaled_volatility = self.volatility * (time_delta_minutes / (24 * 60)) ** 0.5

        # Random walk with trend
        random_change = random.gauss(0, scaled_volatility)
        trend_change = self.trend * (time_delta_minutes / (24 * 60))

        total_change = random_change + trend_change
        new_price = current_price * (1 + Decimal(str(total_change)))

        # Ensure price stays positive
        return max(new_price, Decimal("0.01"))

    def _generate_volume(self, price_change_percent: float) -> Decimal:
        """Generate volume based on price movement."""
        # Higher volume with bigger price moves
        volume_multiplier = 1 + abs(price_change_percent) * 5
        base_volume = random.uniform(0.5, 1.5) * self.volume_base
        return Decimal(str(int(base_volume * volume_multiplier)))

    async def get_historical_ohlcv(
        self,
        symbol: Symbol,
        timeframe: TimeFrame,
        start_date: datetime,
        end_date: datetime,
        limit: Optional[int] = None,
    ) -> List[OHLCV]:
        """Generate historical OHLCV data."""
        if not self.is_enabled:
            return []

        # Initialize price state
        current_price = self._initialize_price_state(symbol)

        # Generate time series
        timeframe_minutes = self._timeframe_to_minutes(timeframe)
        current_time = start_date
        ohlcv_data = []

        while current_time <= end_date:
            # Generate OHLC for this bar
            open_price = current_price

            # Generate intrabar movements
            intrabar_prices = [open_price]
            for _ in range(4):  # Generate a few intrabar points
                next_price = self._generate_price_move(
                    intrabar_prices[-1], timeframe_minutes / 4
                )
                intrabar_prices.append(next_price)

            high_price = max(intrabar_prices)
            low_price = min(intrabar_prices)
            close_price = intrabar_prices[-1]

            # Generate volume
            price_change_percent = float((close_price - open_price) / open_price * 100)
            volume = self._generate_volume(price_change_percent)

            # Create OHLCV bar
            ohlcv = OHLCV(
                symbol=create_symbol(symbol),
                timestamp=current_time,
                timeframe=timeframe,
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=volume,
                trades_count=random.randint(100, 10000),
                vwap=(high_price + low_price + close_price) / 3,
            )

            ohlcv_data.append(ohlcv)

            # Update state
            current_price = close_price
            current_time += timedelta(minutes=timeframe_minutes)

        # Update price state
        self._price_states[symbol] = current_price

        # Apply limit
        if limit and len(ohlcv_data) > limit:
            ohlcv_data = ohlcv_data[-limit:]

        logger.info(f"Generated {len(ohlcv_data)} mock OHLCV bars for {symbol}")
        return ohlcv_data

    async def get_latest_ohlcv(
        self,
        symbol: Symbol,
        timeframe: TimeFrame,
    ) -> Optional[OHLCV]:
        """Generate latest OHLCV bar."""
        end_time = datetime.utcnow()
        timeframe_minutes = self._timeframe_to_minutes(timeframe)
        start_time = end_time - timedelta(minutes=timeframe_minutes)

        data = await self.get_historical_ohlcv(
            symbol, timeframe, start_time, end_time, limit=1
        )

        return data[0] if data else None

    async def stream_tick_data(
        self,
        symbol: Symbol,
    ) -> AsyncGenerator[TickData, None]:
        """Generate real-time tick data stream."""
        logger.info(f"Starting mock tick stream for {symbol}")

        current_price = self._initialize_price_state(symbol)

        while True:
            try:
                # Generate tick
                new_price = self._generate_price_move(current_price, 0.1)  # 0.1 minute
                size = Decimal(str(random.randint(100, 10000)))

                tick = TickData(
                    symbol=create_symbol(symbol),
                    timestamp=datetime.utcnow(),
                    price=new_price,
                    size=size,
                    tick_type="trade",
                    exchange="mock",
                    conditions=["regular"],
                )

                current_price = new_price
                self._price_states[symbol] = current_price

                yield tick

                # Random delay between ticks
                await asyncio.sleep(random.uniform(0.1, 2.0))

            except Exception as e:
                logger.error(f"Error generating mock tick: {e}")
                await asyncio.sleep(1)

    async def get_order_book(
        self,
        symbol: Symbol,
        depth: int = 10,
    ) -> Optional[OrderBook]:
        """Generate mock order book."""
        current_price = self._initialize_price_state(symbol)

        # Generate bid levels
        bids = []
        bid_price = current_price - Decimal("0.01")
        for i in range(depth):
            size = Decimal(str(random.randint(1000, 50000)))
            bids.append(
                OrderBookLevel(
                    price=bid_price, size=size, orders_count=random.randint(1, 10)
                )
            )
            bid_price -= Decimal("0.01")

        # Generate ask levels
        asks = []
        ask_price = current_price + Decimal("0.01")
        for i in range(depth):
            size = Decimal(str(random.randint(1000, 50000)))
            asks.append(
                OrderBookLevel(
                    price=ask_price, size=size, orders_count=random.randint(1, 10)
                )
            )
            ask_price += Decimal("0.01")

        return OrderBook(
            symbol=create_symbol(symbol),
            timestamp=datetime.utcnow(),
            bids=bids,
            asks=asks,
            sequence=random.randint(1000000, 9999999),
        )

    async def get_supported_symbols(self) -> List[Symbol]:
        """Get list of supported mock symbols."""
        return self._supported_symbols

    async def validate_symbol(self, symbol: Symbol) -> bool:
        """Check if symbol is supported."""
        return symbol in self._supported_symbols

    async def health_check(self) -> bool:
        """Mock health check always returns True."""
        return True

    def _timeframe_to_minutes(self, timeframe: TimeFrame) -> int:
        """Convert timeframe to minutes."""
        mapping = {
            TimeFrame.MINUTE_1: 1,
            TimeFrame.MINUTE_5: 5,
            TimeFrame.MINUTE_15: 15,
            TimeFrame.MINUTE_30: 30,
            TimeFrame.HOUR_1: 60,
            TimeFrame.HOUR_4: 240,
            TimeFrame.DAY_1: 1440,
            TimeFrame.WEEK_1: 10080,
            TimeFrame.MONTH_1: 43200,
        }
        return mapping.get(timeframe, 1)

    def set_price(self, symbol: Symbol, price: Decimal) -> None:
        """Set current price for a symbol (for testing)."""
        self._price_states[symbol] = price

    def set_volatility(self, volatility: float) -> None:
        """Set volatility parameter (for testing)."""
        self.volatility = volatility

    def set_trend(self, trend: float) -> None:
        """Set trend parameter (for testing)."""
        self.trend = trend

    def reset_state(self) -> None:
        """Reset all internal state (for testing)."""
        self._price_states.clear()
        self._last_timestamps.clear()
