"""Tests for MockDataSource."""

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from src.voyager_trader.market_data.sources.mock import MockDataSource
from src.voyager_trader.models.types import TimeFrame


@pytest.fixture
def mock_source():
    """Create a mock data source for testing."""
    config = {
        "base_price": 100.0,
        "volatility": 0.02,
        "trend": 0.0,
        "volume_base": 1000000,
    }
    return MockDataSource(config)


@pytest.mark.asyncio
async def test_mock_source_initialization(mock_source):
    """Test mock source initialization."""
    assert mock_source.name == "mock"
    assert mock_source.is_enabled
    assert mock_source.base_price == Decimal("100.0")
    assert mock_source.volatility == 0.02


@pytest.mark.asyncio
async def test_get_supported_symbols(mock_source):
    """Test getting supported symbols."""
    symbols = await mock_source.get_supported_symbols()

    assert isinstance(symbols, list)
    assert len(symbols) > 0
    assert "MOCK_AAPL" in symbols
    assert "MOCK_GOOGL" in symbols


@pytest.mark.asyncio
async def test_validate_symbol(mock_source):
    """Test symbol validation."""
    # Valid symbol
    assert await mock_source.validate_symbol("MOCK_AAPL") is True

    # Invalid symbol
    assert await mock_source.validate_symbol("INVALID_SYMBOL") is False


@pytest.mark.asyncio
async def test_health_check(mock_source):
    """Test health check."""
    assert await mock_source.health_check() is True


@pytest.mark.asyncio
async def test_get_historical_ohlcv(mock_source):
    """Test fetching historical OHLCV data."""
    symbol = "MOCK_AAPL"
    timeframe = TimeFrame.DAY_1
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=30)

    data = await mock_source.get_historical_ohlcv(
        symbol, timeframe, start_date, end_date
    )

    assert isinstance(data, list)
    assert len(data) > 0

    # Check first bar
    bar = data[0]
    assert bar.symbol.code == symbol
    assert bar.timeframe == timeframe
    assert bar.timestamp >= start_date
    assert bar.timestamp <= end_date

    # Validate OHLC relationships
    assert bar.low <= bar.open <= bar.high
    assert bar.low <= bar.close <= bar.high
    assert bar.volume >= 0


@pytest.mark.asyncio
async def test_get_historical_ohlcv_with_limit(mock_source):
    """Test fetching historical data with limit."""
    symbol = "MOCK_AAPL"
    timeframe = TimeFrame.DAY_1
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=100)
    limit = 10

    data = await mock_source.get_historical_ohlcv(
        symbol, timeframe, start_date, end_date, limit=limit
    )

    assert len(data) <= limit


@pytest.mark.asyncio
async def test_get_latest_ohlcv(mock_source):
    """Test fetching latest OHLCV bar."""
    symbol = "MOCK_AAPL"
    timeframe = TimeFrame.MINUTE_15

    bar = await mock_source.get_latest_ohlcv(symbol, timeframe)

    assert bar is not None
    assert bar.symbol.code == symbol
    assert bar.timeframe == timeframe
    assert bar.low <= bar.open <= bar.high
    assert bar.low <= bar.close <= bar.high


@pytest.mark.asyncio
async def test_get_order_book(mock_source):
    """Test fetching order book."""
    symbol = "MOCK_AAPL"
    depth = 5

    order_book = await mock_source.get_order_book(symbol, depth)

    assert order_book is not None
    assert order_book.symbol.code == symbol
    assert len(order_book.bids) == depth
    assert len(order_book.asks) == depth

    # Check bid/ask ordering
    for i in range(1, len(order_book.bids)):
        assert order_book.bids[i].price < order_book.bids[i - 1].price

    for i in range(1, len(order_book.asks)):
        assert order_book.asks[i].price > order_book.asks[i - 1].price

    # Check spread
    assert order_book.spread is not None
    assert order_book.spread > 0


@pytest.mark.asyncio
async def test_stream_tick_data(mock_source):
    """Test streaming tick data."""
    symbol = "MOCK_AAPL"

    tick_count = 0
    async for tick in mock_source.stream_tick_data(symbol):
        assert tick.symbol.code == symbol
        assert tick.price > 0
        assert tick.size > 0
        assert tick.tick_type == "trade"
        assert tick.exchange == "mock"

        tick_count += 1
        if tick_count >= 3:  # Test a few ticks
            break

    assert tick_count == 3


@pytest.mark.asyncio
async def test_price_state_consistency(mock_source):
    """Test that price states are consistent across calls."""
    symbol = "MOCK_AAPL"

    # Initialize price state
    bar1 = await mock_source.get_latest_ohlcv(symbol, TimeFrame.MINUTE_1)
    price1 = bar1.close

    # Get another bar - should continue from previous price
    bar2 = await mock_source.get_latest_ohlcv(symbol, TimeFrame.MINUTE_1)
    price2 = bar2.open

    # Prices should be close (within reasonable variation)
    price_diff_percent = abs(float((price2 - price1) / price1 * 100))
    assert price_diff_percent < 10  # Should be within 10% variation


@pytest.mark.asyncio
async def test_disabled_source():
    """Test behavior when source is disabled."""
    mock_source = MockDataSource()
    mock_source.disable()

    assert not mock_source.is_enabled

    # Should return empty data when disabled
    symbol = "MOCK_AAPL"
    timeframe = TimeFrame.DAY_1
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=10)

    data = await mock_source.get_historical_ohlcv(
        symbol, timeframe, start_date, end_date
    )

    assert data == []


def test_set_price(mock_source):
    """Test manually setting price for a symbol."""
    symbol = "MOCK_AAPL"
    new_price = Decimal("150.00")

    mock_source.set_price(symbol, new_price)
    assert mock_source._price_states[symbol] == new_price


def test_set_volatility(mock_source):
    """Test setting volatility parameter."""
    new_volatility = 0.05
    mock_source.set_volatility(new_volatility)
    assert mock_source.volatility == new_volatility


def test_set_trend(mock_source):
    """Test setting trend parameter."""
    new_trend = 0.01
    mock_source.set_trend(new_trend)
    assert mock_source.trend == new_trend


def test_reset_state(mock_source):
    """Test resetting internal state."""
    # Set some state
    mock_source.set_price("MOCK_AAPL", Decimal("150.00"))
    assert len(mock_source._price_states) > 0

    # Reset state
    mock_source.reset_state()
    assert len(mock_source._price_states) == 0


@pytest.mark.asyncio
async def test_intraday_timeframes(mock_source):
    """Test different intraday timeframes."""
    symbol = "MOCK_AAPL"
    timeframes = [
        TimeFrame.MINUTE_1,
        TimeFrame.MINUTE_5,
        TimeFrame.MINUTE_15,
        TimeFrame.HOUR_1,
    ]

    for timeframe in timeframes:
        bar = await mock_source.get_latest_ohlcv(symbol, timeframe)
        assert bar is not None
        assert bar.timeframe == timeframe


@pytest.mark.asyncio
async def test_multiple_symbols(mock_source):
    """Test handling multiple symbols."""
    symbols = ["MOCK_AAPL", "MOCK_GOOGL", "MOCK_MSFT"]

    for symbol in symbols:
        bar = await mock_source.get_latest_ohlcv(symbol, TimeFrame.DAY_1)
        assert bar is not None
        assert bar.symbol.code == symbol

        # Each symbol should have different base prices
        if symbol != symbols[0]:
            # Prices should be different due to symbol hash
            assert bar.open != mock_source.base_price
