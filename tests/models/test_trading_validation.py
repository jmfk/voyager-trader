"""Validation tests for trading models to improve coverage."""

from datetime import datetime
from decimal import Decimal

import pytest

from src.voyager_trader.models.trading import Order, Portfolio, Position, Trade
from src.voyager_trader.models.types import (
    AssetClass,
    Currency,
    Money,
    OrderSide,
    OrderStatus,
    OrderType,
    PositionType,
    Quantity,
    Symbol,
)


def test_order_price_validation_int_conversion():
    """Test price validation with int conversion."""
    # Test int to Decimal conversion
    order = Order(
        symbol=Symbol(code="AAPL", asset_class=AssetClass.EQUITY),
        order_type=OrderType.LIMIT,
        side=OrderSide.BUY,
        quantity=Quantity(amount=Decimal("100")),
        price=150,  # int input
    )
    assert order.price == Decimal("150.00000000")


def test_order_price_validation_float_conversion():
    """Test price validation with float conversion."""
    # Test float to Decimal conversion
    order = Order(
        symbol=Symbol(code="AAPL", asset_class=AssetClass.EQUITY),
        order_type=OrderType.LIMIT,
        side=OrderSide.BUY,
        quantity=Quantity(amount=Decimal("100")),
        price=150.75,  # float input
    )
    assert order.price == Decimal("150.75000000")


def test_order_price_validation_negative_price():
    """Test price validation with negative price."""
    with pytest.raises(ValueError, match="Prices must be positive"):
        Order(
            symbol=Symbol(code="AAPL", asset_class=AssetClass.EQUITY),
            order_type=OrderType.LIMIT,
            side=OrderSide.BUY,
            quantity=Quantity(amount=Decimal("100")),
            price=-150.00,
        )


def test_order_price_validation_zero_price():
    """Test price validation with zero price."""
    with pytest.raises(ValueError, match="Prices must be positive"):
        Order(
            symbol=Symbol(code="AAPL", asset_class=AssetClass.EQUITY),
            order_type=OrderType.LIMIT,
            side=OrderSide.BUY,
            quantity=Quantity(amount=Decimal("100")),
            price=0,
        )


def test_order_stop_price_validation():
    """Test stop price validation."""
    # Test valid stop price
    order = Order(
        symbol=Symbol(code="AAPL", asset_class=AssetClass.EQUITY),
        order_type=OrderType.STOP,
        side=OrderSide.SELL,
        quantity=Quantity(amount=Decimal("100")),
        stop_price=145.50,
    )
    assert order.stop_price == Decimal("145.50000000")

    # Test negative stop price
    with pytest.raises(ValueError, match="Prices must be positive"):
        Order(
            symbol=Symbol(code="AAPL", asset_class=AssetClass.EQUITY),
            order_type=OrderType.STOP,
            side=OrderSide.SELL,
            quantity=Quantity(amount=Decimal("100")),
            stop_price=-145.50,
        )


def test_order_average_fill_price_validation():
    """Test average fill price validation."""
    order = Order(
        symbol=Symbol(code="AAPL", asset_class=AssetClass.EQUITY),
        order_type=OrderType.MARKET,
        side=OrderSide.BUY,
        quantity=Quantity(amount=Decimal("100")),
        average_fill_price=150.123456789,  # float with many decimals
    )
    assert order.average_fill_price == Decimal(
        "150.12345679"
    )  # quantized to 8 decimals


def test_order_time_in_force_validation_valid():
    """Test time in force validation with valid values."""
    valid_tif_values = ["DAY", "GTC", "IOC", "FOK", "GTD", "ATO", "ATC"]

    for tif in valid_tif_values:
        # Test uppercase
        order = Order(
            symbol=Symbol(code="AAPL", asset_class=AssetClass.EQUITY),
            order_type=OrderType.LIMIT,
            side=OrderSide.BUY,
            quantity=Quantity(amount=Decimal("100")),
            price=Decimal("150.00"),
            time_in_force=tif,
        )
        assert order.time_in_force == tif

        # Test lowercase conversion
        order_lower = Order(
            symbol=Symbol(code="AAPL", asset_class=AssetClass.EQUITY),
            order_type=OrderType.LIMIT,
            side=OrderSide.BUY,
            quantity=Quantity(amount=Decimal("100")),
            price=Decimal("150.00"),
            time_in_force=tif.lower(),
        )
        assert order_lower.time_in_force == tif


def test_order_time_in_force_validation_invalid():
    """Test time in force validation with invalid values."""
    with pytest.raises(ValueError, match="Time in force must be one of"):
        Order(
            symbol=Symbol(code="AAPL", asset_class=AssetClass.EQUITY),
            order_type=OrderType.LIMIT,
            side=OrderSide.BUY,
            quantity=Quantity(amount=Decimal("100")),
            price=Decimal("150.00"),
            time_in_force="INVALID",
        )


def test_order_remaining_quantity():
    """Test remaining quantity computed field."""
    order = Order(
        symbol=Symbol(code="AAPL", asset_class=AssetClass.EQUITY),
        order_type=OrderType.LIMIT,
        side=OrderSide.BUY,
        quantity=Quantity(amount=Decimal("100")),
        price=Decimal("150.00"),
        filled_quantity=Quantity(amount=Decimal("30")),
    )
    assert order.remaining_quantity.amount == Decimal("70")


def test_order_fill_percentage_normal():
    """Test fill percentage computation."""
    order = Order(
        symbol=Symbol(code="AAPL", asset_class=AssetClass.EQUITY),
        order_type=OrderType.LIMIT,
        side=OrderSide.BUY,
        quantity=Quantity(amount=Decimal("100")),
        price=Decimal("150.00"),
        filled_quantity=Quantity(amount=Decimal("30")),
    )
    assert order.fill_percentage == Decimal("30")


def test_order_fill_percentage_zero_quantity():
    """Test fill percentage with zero total quantity."""
    order = Order(
        symbol=Symbol(code="AAPL", asset_class=AssetClass.EQUITY),
        order_type=OrderType.LIMIT,
        side=OrderSide.BUY,
        quantity=Quantity(amount=Decimal("0")),
        price=Decimal("150.00"),
        filled_quantity=Quantity(amount=Decimal("0")),
    )
    assert order.fill_percentage == Decimal("0")


def test_order_type_properties():
    """Test order type checking properties."""
    # Market order
    market_order = Order(
        symbol=Symbol(code="AAPL", asset_class=AssetClass.EQUITY),
        order_type=OrderType.MARKET,
        side=OrderSide.BUY,
        quantity=Quantity(amount=Decimal("100")),
    )
    assert market_order.is_market_order is True
    assert market_order.is_limit_order is False
    assert market_order.is_stop_order is False

    # Limit order
    limit_order = Order(
        symbol=Symbol(code="AAPL", asset_class=AssetClass.EQUITY),
        order_type=OrderType.LIMIT,
        side=OrderSide.BUY,
        quantity=Quantity(amount=Decimal("100")),
        price=Decimal("150.00"),
    )
    assert limit_order.is_market_order is False
    assert limit_order.is_limit_order is True
    assert limit_order.is_stop_order is False

    # Stop order
    stop_order = Order(
        symbol=Symbol(code="AAPL", asset_class=AssetClass.EQUITY),
        order_type=OrderType.STOP,
        side=OrderSide.SELL,
        quantity=Quantity(amount=Decimal("100")),
        stop_price=Decimal("145.00"),
    )
    assert stop_order.is_market_order is False
    assert stop_order.is_limit_order is False
    assert stop_order.is_stop_order is True


def test_order_status_properties():
    """Test order status checking properties."""
    order = Order(
        symbol=Symbol(code="AAPL", asset_class=AssetClass.EQUITY),
        order_type=OrderType.LIMIT,
        side=OrderSide.BUY,
        quantity=Quantity(amount=Decimal("100")),
        price=Decimal("150.00"),
        status=OrderStatus.PENDING,
    )

    # Test active order
    assert order.is_active is True
    assert order.is_filled is False
    assert order.is_cancelled is False

    # Test filled order
    filled_order = order.update(status=OrderStatus.FILLED)
    assert filled_order.is_filled is True
    assert filled_order.is_active is False

    # Test cancelled order
    cancelled_order = order.update(status=OrderStatus.CANCELLED)
    assert cancelled_order.is_cancelled is True
    assert cancelled_order.is_active is False


def test_order_side_properties():
    """Test order side checking properties."""
    buy_order = Order(
        symbol=Symbol(code="AAPL", asset_class=AssetClass.EQUITY),
        order_type=OrderType.LIMIT,
        side=OrderSide.BUY,
        quantity=Quantity(amount=Decimal("100")),
        price=Decimal("150.00"),
    )

    assert buy_order.is_buy is True
    assert buy_order.is_sell is False

    sell_order = Order(
        symbol=Symbol(code="AAPL", asset_class=AssetClass.EQUITY),
        order_type=OrderType.LIMIT,
        side=OrderSide.SELL,
        quantity=Quantity(amount=Decimal("100")),
        price=Decimal("150.00"),
    )

    assert sell_order.is_buy is False
    assert sell_order.is_sell is True


def test_order_partial_execution():
    """Test order partial execution method."""
    order = Order(
        symbol=Symbol(code="AAPL", asset_class=AssetClass.EQUITY),
        order_type=OrderType.LIMIT,
        side=OrderSide.BUY,
        quantity=Quantity(amount=Decimal("100")),
        price=Decimal("150.00"),
    )

    # Test partial execution
    partial_order = order.execute_partial(
        quantity=Quantity(amount=Decimal("50")),
        price=Decimal("149.95"),
    )
    assert partial_order.filled_quantity.amount == Decimal("50")
    assert partial_order.status == OrderStatus.PARTIALLY_FILLED


def test_position_pnl_calculations():
    """Test position PnL calculations."""
    # Long position with profit
    long_position = Position(
        symbol=Symbol(code="AAPL", asset_class=AssetClass.EQUITY),
        position_type=PositionType.LONG,
        quantity=Quantity(amount=Decimal("100")),
        entry_price=Decimal("150.00"),
        current_price=Decimal("160.00"),
        entry_timestamp=datetime.utcnow(),
    )
    # Test unrealized pnl
    assert long_position.unrealized_pnl == Decimal("1000.00")  # (160-150) * 100

    # Test market value
    assert long_position.market_value == Decimal("16000.00")  # 160 * 100

    # Test cost basis
    assert long_position.cost_basis == Decimal("15000.00")  # 150 * 100


def test_position_status_properties():
    """Test position status properties."""
    open_position = Position(
        symbol=Symbol(code="AAPL", asset_class=AssetClass.EQUITY),
        position_type=PositionType.LONG,
        quantity=Quantity(amount=Decimal("100")),
        entry_price=Decimal("150.00"),
        current_price=Decimal("160.00"),
        entry_timestamp=datetime.utcnow(),
    )

    # Test position type properties
    assert open_position.is_long is True
    assert open_position.is_short is False
    assert open_position.is_open is True
    assert open_position.is_closed is False

    # Test profitable property
    assert open_position.is_profitable is True

    # Test short position
    short_position = Position(
        symbol=Symbol(code="AAPL", asset_class=AssetClass.EQUITY),
        position_type=PositionType.SHORT,
        quantity=Quantity(amount=Decimal("100")),
        entry_price=Decimal("150.00"),
        current_price=Decimal("140.00"),
        entry_timestamp=datetime.utcnow(),
    )

    assert short_position.is_long is False
    assert short_position.is_short is True


def test_position_update_methods():
    """Test position update methods."""
    position = Position(
        symbol=Symbol(code="AAPL", asset_class=AssetClass.EQUITY),
        position_type=PositionType.LONG,
        quantity=Quantity(amount=Decimal("100")),
        entry_price=Decimal("150.00"),
        current_price=Decimal("150.00"),
        entry_timestamp=datetime.utcnow(),
    )

    # Test price update
    updated_position = position.update_price(Decimal("160.00"))
    assert updated_position.current_price == Decimal("160.00")

    # Test add to position
    added_position = position.add_to_position(
        quantity=Quantity(amount=Decimal("50")), price=Decimal("155.00")
    )
    assert added_position.quantity.amount == Decimal("150")


def test_position_reduce_method():
    """Test position reduce method."""
    position = Position(
        symbol=Symbol(code="AAPL", asset_class=AssetClass.EQUITY),
        position_type=PositionType.LONG,
        quantity=Quantity(amount=Decimal("100")),
        entry_price=Decimal("150.00"),
        current_price=Decimal("160.00"),
        entry_timestamp=datetime.utcnow(),
    )

    # Test reduce position
    reduced_position = position.reduce_position(
        quantity=Quantity(amount=Decimal("25")),
    )
    assert reduced_position.quantity.amount == Decimal("75")


def test_portfolio_metrics():
    """Test portfolio metrics."""
    portfolio = Portfolio(
        name="Test Portfolio",
        account_id="account-123",
        base_currency=Currency.USD,
        cash_balance=Money(amount=Decimal("25000"), currency=Currency.USD),
        total_value=Money(amount=Decimal("100000"), currency=Currency.USD),
        realized_pnl=Money(amount=Decimal("5000"), currency=Currency.USD),
        unrealized_pnl=Money(amount=Decimal("2000"), currency=Currency.USD),
        positions={"AAPL": "pos-1", "GOOGL": "pos-2"},
    )

    # Test position count
    assert portfolio.position_count == 2

    # Test total PnL
    assert portfolio.total_pnl.amount == Decimal("7000")

    # Test cash percentage
    assert portfolio.cash_percentage == Decimal("25.00")


def test_portfolio_position_operations():
    """Test portfolio position operations."""
    portfolio = Portfolio(
        name="Test Portfolio",
        account_id="account-123",
        base_currency=Currency.USD,
        cash_balance=Money(amount=Decimal("50000"), currency=Currency.USD),
        total_value=Money(amount=Decimal("100000"), currency=Currency.USD),
        positions={"AAPL": "pos-1"},
    )

    # Test add position
    updated_portfolio = portfolio.add_position("pos-2", "GOOGL")
    assert len(updated_portfolio.positions) == 2
    assert updated_portfolio.positions["GOOGL"] == "pos-2"


def test_trade_properties():
    """Test trade properties."""
    # Test buy trade
    buy_trade = Trade(
        symbol=Symbol(code="AAPL", asset_class=AssetClass.EQUITY),
        side=OrderSide.BUY,
        quantity=Quantity(amount=Decimal("100")),
        price=Decimal("150.25"),
        timestamp=datetime.utcnow(),
        order_id="order-123",
    )

    # Test side properties
    assert buy_trade.is_buy is True
    assert buy_trade.is_sell is False

    # Test notional value
    assert buy_trade.notional_value == Decimal("15025.00")

    # Test total cost (should equal notional for trade without commission)
    assert buy_trade.total_cost == Decimal("15025.00")

    # Test sell trade
    sell_trade = Trade(
        symbol=Symbol(code="AAPL", asset_class=AssetClass.EQUITY),
        side=OrderSide.SELL,
        quantity=Quantity(amount=Decimal("50")),
        price=Decimal("155.00"),
        timestamp=datetime.utcnow(),
        order_id="order-456",
    )

    assert sell_trade.is_buy is False
    assert sell_trade.is_sell is True
