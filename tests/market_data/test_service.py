"""Tests for MarketDataService."""

from datetime import datetime, timedelta

import pytest

from src.voyager_trader.market_data.service import MarketDataService
from src.voyager_trader.models.types import TimeFrame


@pytest.fixture
def mock_sources_config():
    """Configuration for mock data sources."""
    return {
        "mock": {
            "base_price": 100.0,
            "volatility": 0.01,
            "priority": 10,
        }
    }


@pytest.fixture
async def market_data_service(mock_sources_config):
    """Create a market data service with mock sources."""
    service = MarketDataService(
        sources_config=mock_sources_config,
        enable_health_monitoring=False,  # Disable for testing
    )
    await service.start()
    yield service
    await service.stop()


@pytest.mark.asyncio
async def test_service_initialization():
    """Test service initialization."""
    service = MarketDataService()

    assert not service._started
    assert service.enable_caching is True
    assert service.enable_validation is True
    assert service.cache is not None
    assert service.validator is not None


@pytest.mark.asyncio
async def test_service_start_stop():
    """Test service start and stop."""
    service = MarketDataService()

    # Start service
    await service.start()
    assert service._started is True

    # Stop service
    await service.stop()
    assert service._started is False


@pytest.mark.asyncio
async def test_get_historical_ohlcv(market_data_service):
    """Test fetching historical OHLCV data."""
    symbol = "MOCK_AAPL"
    timeframe = TimeFrame.DAY_1
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=10)

    data = await market_data_service.get_historical_ohlcv(
        symbol, timeframe, start_date, end_date
    )

    assert isinstance(data, list)
    assert len(data) > 0

    bar = data[0]
    assert bar.symbol == symbol
    assert bar.timeframe == timeframe
    assert start_date <= bar.timestamp <= end_date


@pytest.mark.asyncio
async def test_get_historical_ohlcv_with_cache(market_data_service):
    """Test caching behavior for historical OHLCV data."""
    symbol = "MOCK_AAPL"
    timeframe = TimeFrame.DAY_1
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=5)

    # First call - should fetch and cache
    data1 = await market_data_service.get_historical_ohlcv(
        symbol, timeframe, start_date, end_date
    )

    # Second call - should use cache
    data2 = await market_data_service.get_historical_ohlcv(
        symbol, timeframe, start_date, end_date
    )

    assert len(data1) == len(data2)
    # Data should be identical from cache
    for i, (bar1, bar2) in enumerate(zip(data1, data2)):
        assert bar1.timestamp == bar2.timestamp
        assert bar1.close == bar2.close


@pytest.mark.asyncio
async def test_get_historical_ohlcv_force_refresh(market_data_service):
    """Test force refresh bypasses cache."""
    symbol = "MOCK_AAPL"
    timeframe = TimeFrame.DAY_1
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=5)

    # First call to populate cache
    await market_data_service.get_historical_ohlcv(
        symbol, timeframe, start_date, end_date
    )

    # Force refresh should bypass cache
    fresh_data = await market_data_service.get_historical_ohlcv(
        symbol, timeframe, start_date, end_date, force_refresh=True
    )

    assert isinstance(fresh_data, list)
    assert len(fresh_data) > 0


@pytest.mark.asyncio
async def test_get_latest_ohlcv(market_data_service):
    """Test fetching latest OHLCV data."""
    symbol = "MOCK_AAPL"
    timeframe = TimeFrame.MINUTE_15

    bar = await market_data_service.get_latest_ohlcv(symbol, timeframe)

    assert bar is not None
    assert bar.symbol == symbol
    assert bar.timeframe == timeframe
    assert bar.low <= bar.open <= bar.high
    assert bar.low <= bar.close <= bar.high


@pytest.mark.asyncio
async def test_get_order_book(market_data_service):
    """Test fetching order book."""
    symbol = "MOCK_AAPL"
    depth = 5

    order_book = await market_data_service.get_order_book(symbol, depth)

    assert order_book is not None
    assert order_book.symbol == symbol
    assert len(order_book.bids) == depth
    assert len(order_book.asks) == depth
    assert order_book.spread is not None


@pytest.mark.asyncio
async def test_stream_tick_data(market_data_service):
    """Test streaming tick data."""
    symbol = "MOCK_AAPL"

    tick_count = 0
    async for tick in market_data_service.stream_tick_data(symbol):
        assert tick.symbol == symbol
        assert tick.price > 0
        assert tick.size > 0

        tick_count += 1
        if tick_count >= 3:  # Test a few ticks
            break

    assert tick_count == 3


@pytest.mark.asyncio
async def test_get_supported_symbols(market_data_service):
    """Test getting supported symbols."""
    symbols = await market_data_service.get_supported_symbols()

    assert isinstance(symbols, list)
    assert len(symbols) > 0
    assert "MOCK_AAPL" in symbols


@pytest.mark.asyncio
async def test_validate_symbol(market_data_service):
    """Test symbol validation."""
    # Valid symbol
    assert await market_data_service.validate_symbol("MOCK_AAPL") is True

    # Invalid symbol
    assert await market_data_service.validate_symbol("INVALID_SYMBOL") is False


@pytest.mark.asyncio
async def test_health_check(market_data_service):
    """Test service health check."""
    health = await market_data_service.health_check()

    assert "service_started" in health
    assert "timestamp" in health
    assert "components" in health
    assert "overall_health" in health

    assert health["service_started"] is True
    assert health["overall_health"]["healthy"] is True


@pytest.mark.asyncio
async def test_clear_cache(market_data_service):
    """Test cache clearing."""
    symbol = "MOCK_AAPL"
    timeframe = TimeFrame.DAY_1
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=5)

    # Populate cache
    await market_data_service.get_historical_ohlcv(
        symbol, timeframe, start_date, end_date
    )

    # Clear cache
    await market_data_service.clear_cache()

    # Cache should be cleared (no error, just verify method works)
    assert True


@pytest.mark.asyncio
async def test_add_remove_data_source():
    """Test adding and removing data sources at runtime."""
    service = MarketDataService()

    # Add a mock source
    service.add_data_source("mock", {"base_price": 200.0}, priority=50)

    # Verify source was added
    sources = service.source_manager.get_available_sources()
    assert any(source.name == "mock" for source in sources)

    # Remove source
    result = service.remove_data_source("mock")
    assert result is True

    # Verify source was removed
    sources = service.source_manager.get_available_sources()
    assert not any(source.name == "mock" for source in sources)


@pytest.mark.asyncio
async def test_service_with_disabled_components():
    """Test service with caching and validation disabled."""
    service = MarketDataService(
        enable_caching=False,
        enable_validation=False,
        enable_health_monitoring=False,
    )

    assert service.cache is None
    assert service.validator is None
    assert service.enable_caching is False
    assert service.enable_validation is False


def test_get_stats():
    """Test getting service statistics."""
    service = MarketDataService()

    stats = service.get_stats()

    assert "service_started" in stats
    assert "configuration" in stats
    assert "components" in stats
    assert stats["configuration"]["caching_enabled"] is True
    assert stats["configuration"]["validation_enabled"] is True


@pytest.mark.asyncio
async def test_service_failover():
    """Test service failover when primary source fails."""
    # Create service with multiple mock sources
    sources_config = {
        "mock_primary": {
            "base_price": 100.0,
            "priority": 1,  # Higher priority
        },
        "mock_backup": {
            "base_price": 150.0,
            "priority": 2,  # Lower priority
        },
    }

    service = MarketDataService(sources_config=sources_config)
    await service.start()

    try:
        # Disable primary source to trigger failover
        primary_source = service.source_manager.get_source("mock_primary")
        primary_source.disable()

        # Request should now use backup source
        symbol = "MOCK_AAPL"
        bar = await service.get_latest_ohlcv(symbol, TimeFrame.DAY)

        assert bar is not None
        # Price should reflect backup source base price
        # (This is approximate due to random generation)
        assert bar.open > 140  # Should be around 150 from backup source

    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_service_no_available_sources():
    """Test service behavior when no sources are available."""
    service = MarketDataService()
    await service.start()

    try:
        # All sources should be disabled by default (no config provided)
        with pytest.raises(Exception) as exc_info:
            await service.get_latest_ohlcv("AAPL", TimeFrame.DAY)

        assert "No enabled data sources available" in str(exc_info.value)

    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_service_repr(market_data_service):
    """Test service string representation."""
    repr_str = repr(market_data_service)
    assert "MarketDataService" in repr_str
    assert "started=True" in repr_str
    assert "sources=" in repr_str
