"""Tests for DataValidator."""

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from src.voyager_trader.market_data.types import create_symbol
from src.voyager_trader.market_data.validator import DataValidationError, DataValidator
from src.voyager_trader.models.market import OHLCV, OrderBook, OrderBookLevel, TickData
from src.voyager_trader.models.types import TimeFrame


@pytest.fixture
def validator():
    """Create a data validator for testing."""
    return DataValidator()


@pytest.fixture
def valid_ohlcv():
    """Create a valid OHLCV bar for testing."""
    return OHLCV(
        symbol=create_symbol("AAPL"),
        timestamp=datetime.now(timezone.utc),
        timeframe=TimeFrame.DAY_1,
        open=Decimal("100.00"),
        high=Decimal("105.00"),
        low=Decimal("99.00"),
        close=Decimal("103.00"),
        volume=Decimal("1000000"),
    )


@pytest.fixture
def valid_tick():
    """Create a valid tick data for testing."""
    return TickData(
        symbol=create_symbol("AAPL"),
        timestamp=datetime.now(timezone.utc),
        price=Decimal("100.00"),
        size=Decimal("1000"),
        tick_type="trade",
        exchange="NYSE",
    )


@pytest.fixture
def valid_order_book():
    """Create a valid order book for testing."""
    bids = [
        OrderBookLevel(price=Decimal("99.99"), size=Decimal("1000")),
        OrderBookLevel(price=Decimal("99.98"), size=Decimal("2000")),
    ]
    asks = [
        OrderBookLevel(price=Decimal("100.01"), size=Decimal("1500")),
        OrderBookLevel(price=Decimal("100.02"), size=Decimal("1000")),
    ]

    return OrderBook(
        symbol=create_symbol("AAPL"),
        timestamp=datetime.now(timezone.utc),
        bids=bids,
        asks=asks,
    )


def test_validator_initialization():
    """Test validator initialization with custom config."""
    config = {
        "max_price_change_percent": 25.0,
        "max_volume_multiplier": 50.0,
        "min_price": 0.001,
        "max_price": 500000.0,
    }
    validator = DataValidator(config)

    assert validator.max_price_change_percent == 25.0
    assert validator.max_volume_multiplier == 50.0
    assert validator.min_price == 0.001
    assert validator.max_price == 500000.0


def test_validate_valid_ohlcv(validator, valid_ohlcv):
    """Test validating a valid OHLCV bar."""
    assert validator.validate_ohlcv(valid_ohlcv) is True


def test_validate_ohlcv_price_range_errors(validator):
    """Test OHLCV validation with price range errors."""
    # Price too low
    with pytest.raises(DataValidationError):
        ohlcv = OHLCV(
            symbol=create_symbol("TEST"),
            timestamp=datetime.now(timezone.utc),
            timeframe=TimeFrame.DAY_1,
            open=Decimal("0.001"),  # Too low
            high=Decimal("0.002"),
            low=Decimal("0.0009"),
            close=Decimal("0.0015"),
            volume=Decimal("1000"),
        )
        validator.validate_ohlcv(ohlcv)

    # Price too high
    with pytest.raises(DataValidationError):
        ohlcv = OHLCV(
            symbol=create_symbol("TEST"),
            timestamp=datetime.now(timezone.utc),
            timeframe=TimeFrame.DAY_1,
            open=Decimal("2000000"),  # Too high
            high=Decimal("2100000"),
            low=Decimal("1900000"),
            close=Decimal("2050000"),
            volume=Decimal("1000"),
        )
        validator.validate_ohlcv(ohlcv)


def test_validate_ohlcv_negative_volume(validator):
    """Test OHLCV validation with negative volume."""
    # Pydantic validates negative volume at model creation
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ohlcv = OHLCV(
            symbol=create_symbol("TEST"),
            timestamp=datetime.now(timezone.utc),
            timeframe=TimeFrame.DAY_1,
            open=Decimal("100.00"),
            high=Decimal("105.00"),
            low=Decimal("99.00"),
            close=Decimal("103.00"),
            volume=Decimal("-1000"),  # Negative volume
        )
        validator.validate_ohlcv(ohlcv)


def test_validate_ohlcv_price_continuity(validator, valid_ohlcv):
    """Test OHLCV validation with price continuity checks."""
    # Create a previous bar
    previous = OHLCV(
        symbol=create_symbol("AAPL"),
        timestamp=datetime.now(timezone.utc) - timedelta(days=1),
        timeframe=TimeFrame.DAY_1,
        open=Decimal("100.00"),
        high=Decimal("105.00"),
        low=Decimal("99.00"),
        close=Decimal("102.00"),
        volume=Decimal("500000"),
    )

    # Valid continuation
    assert validator.validate_ohlcv(valid_ohlcv, previous) is True

    # Invalid - huge price jump
    with pytest.raises(DataValidationError):
        huge_jump_ohlcv = OHLCV(
            symbol=create_symbol("AAPL"),
            timestamp=datetime.now(timezone.utc),
            timeframe=TimeFrame.DAY_1,
            open=Decimal("300.00"),  # 200% jump from previous close
            high=Decimal("310.00"),
            low=Decimal("295.00"),
            close=Decimal("305.00"),
            volume=Decimal("1000000"),
        )
        validator.validate_ohlcv(huge_jump_ohlcv, previous)


def test_validate_ohlcv_timestamp(validator):
    """Test OHLCV validation with invalid timestamps."""
    # Timestamp too far in past
    with pytest.raises(DataValidationError):
        old_ohlcv = OHLCV(
            symbol=create_symbol("TEST"),
            timestamp=datetime(1900, 1, 1, tzinfo=timezone.utc),  # Too old
            timeframe=TimeFrame.DAY_1,
            open=Decimal("100.00"),
            high=Decimal("105.00"),
            low=Decimal("99.00"),
            close=Decimal("103.00"),
            volume=Decimal("1000"),
        )
        validator.validate_ohlcv(old_ohlcv)

    # Timestamp too far in future
    with pytest.raises(DataValidationError):
        future_ohlcv = OHLCV(
            symbol=create_symbol("TEST"),
            timestamp=datetime.now(timezone.utc) + timedelta(days=10),  # Too far future
            timeframe=TimeFrame.DAY_1,
            open=Decimal("100.00"),
            high=Decimal("105.00"),
            low=Decimal("99.00"),
            close=Decimal("103.00"),
            volume=Decimal("1000"),
        )
        validator.validate_ohlcv(future_ohlcv)


def test_validate_valid_tick(validator, valid_tick):
    """Test validating a valid tick."""
    assert validator.validate_tick_data(valid_tick) is True


def test_validate_tick_negative_values(validator):
    """Test tick validation with negative values."""
    # Negative price - Pydantic validates at model creation
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        tick = TickData(
            symbol=create_symbol("TEST"),
            timestamp=datetime.now(timezone.utc),
            price=Decimal("-100.00"),  # Negative price
            size=Decimal("1000"),
            tick_type="trade",
        )
        validator.validate_tick_data(tick)

    # Negative size
    with pytest.raises(ValidationError):
        tick = TickData(
            symbol=create_symbol("TEST"),
            timestamp=datetime.now(timezone.utc),
            price=Decimal("100.00"),
            size=Decimal("-1000"),  # Negative size
            tick_type="trade",
        )
        validator.validate_tick_data(tick)


def test_validate_tick_sequence(validator, valid_tick):
    """Test tick validation with sequence checks."""
    # Create previous tick
    previous = TickData(
        symbol=create_symbol("AAPL"),
        timestamp=datetime.now(timezone.utc) - timedelta(seconds=1),
        price=Decimal("100.00"),
        size=Decimal("1000"),
        tick_type="trade",
    )

    # Valid sequence
    assert validator.validate_tick_data(valid_tick, previous) is True

    # Invalid - older timestamp
    with pytest.raises(DataValidationError):
        old_tick = TickData(
            symbol=create_symbol("AAPL"),
            timestamp=datetime.now(timezone.utc)
            - timedelta(hours=1),  # Older than previous
            price=Decimal("101.00"),
            size=Decimal("1000"),
            tick_type="trade",
        )
        validator.validate_tick_data(old_tick, previous)


def test_validate_valid_order_book(validator, valid_order_book):
    """Test validating a valid order book."""
    assert validator.validate_order_book(valid_order_book) is True


def test_validate_order_book_empty(validator):
    """Test order book validation with empty levels."""
    with pytest.raises(DataValidationError):
        empty_book = OrderBook(
            symbol=create_symbol("TEST"),
            timestamp=datetime.now(timezone.utc),
            bids=[],  # Empty
            asks=[],  # Empty
        )
        validator.validate_order_book(empty_book)


def test_validate_order_book_wide_spread(validator):
    """Test order book validation with excessively wide spread."""
    # Create order book with very wide spread (>10% default threshold)
    bids = [OrderBookLevel(price=Decimal("90.00"), size=Decimal("1000"))]
    asks = [OrderBookLevel(price=Decimal("110.00"), size=Decimal("1000"))]

    with pytest.raises(DataValidationError):
        wide_spread_book = OrderBook(
            symbol=create_symbol("TEST"),
            timestamp=datetime.now(timezone.utc),
            bids=bids,
            asks=asks,
        )
        validator.validate_order_book(wide_spread_book)


def test_validate_order_book_invalid_sizes(validator):
    """Test order book validation with invalid sizes."""
    # Zero size bid
    with pytest.raises(DataValidationError):
        bids = [OrderBookLevel(price=Decimal("99.99"), size=Decimal("0"))]  # Zero size
        asks = [OrderBookLevel(price=Decimal("100.01"), size=Decimal("1000"))]

        book = OrderBook(
            symbol=create_symbol("TEST"),
            timestamp=datetime.now(timezone.utc),
            bids=bids,
            asks=asks,
        )
        validator.validate_order_book(book)


def test_validate_batch_ohlcv(validator):
    """Test batch OHLCV validation."""
    # Create a mix of valid data - one with price issues that validator catches
    valid_bar = OHLCV(
        symbol=create_symbol("AAPL"),
        timestamp=datetime.now(timezone.utc),
        timeframe=TimeFrame.DAY_1,
        open=Decimal("100.00"),
        high=Decimal("105.00"),
        low=Decimal("99.00"),
        close=Decimal("103.00"),
        volume=Decimal("1000000"),
    )

    # Create bar with extreme price that validator should reject
    extreme_price_bar = OHLCV(
        symbol=create_symbol("AAPL"),
        timestamp=datetime.now(timezone.utc) + timedelta(days=10),  # Too far in future
        timeframe=TimeFrame.DAY_1,
        open=Decimal("100.00"),
        high=Decimal("105.00"),
        low=Decimal("99.00"),
        close=Decimal("103.00"),
        volume=Decimal("1000000"),
    )

    data_list = [valid_bar, extreme_price_bar, valid_bar]

    # Should return only valid entries
    valid_data = validator.validate_batch_ohlcv(data_list)
    assert len(valid_data) == 2  # Only the valid bars
    assert all(bar.volume >= 0 for bar in valid_data)


def test_get_validation_stats(validator):
    """Test getting validation statistics."""
    stats = validator.get_validation_stats()

    assert "config" in stats
    assert "max_price_change_percent" in stats["config"]
    assert "max_volume_multiplier" in stats["config"]
    assert "min_price" in stats["config"]
    assert "max_price" in stats["config"]
    assert "max_spread_percent" in stats["config"]


def test_validator_with_custom_thresholds():
    """Test validator with custom validation thresholds."""
    config = {
        "max_price_change_percent": 100.0,  # Very permissive
        "max_volume_multiplier": 1000.0,
        "min_price": 0.0001,
        "max_price": 10000000.0,
        "max_spread_percent": 50.0,
    }

    validator = DataValidator(config)

    # This should pass with custom thresholds but fail with defaults
    previous = OHLCV(
        symbol=create_symbol("CRYPTO"),
        timestamp=datetime.now(timezone.utc) - timedelta(minutes=1),
        timeframe=TimeFrame.MINUTE_1,
        open=Decimal("1.00"),
        high=Decimal("1.05"),
        low=Decimal("0.95"),
        close=Decimal("1.00"),
        volume=Decimal("1000"),
    )

    current = OHLCV(
        symbol=create_symbol("CRYPTO"),
        timestamp=datetime.now(timezone.utc),
        timeframe=TimeFrame.MINUTE_1,
        open=Decimal("1.80"),  # 80% jump - would fail default but pass custom
        high=Decimal("1.85"),
        low=Decimal("1.75"),
        close=Decimal("1.82"),
        volume=Decimal("10000"),  # 10x volume - would fail default but pass custom
    )

    assert validator.validate_ohlcv(current, previous) is True
