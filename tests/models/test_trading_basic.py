"""Basic tests for trading models to improve coverage."""

from datetime import datetime
from decimal import Decimal

from src.voyager_trader.models.trading import Account, Order, Portfolio, Position, Trade
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


def test_order_basic():
    """Basic Order creation and methods."""
    order = Order(
        symbol=Symbol(code="AAPL", asset_class=AssetClass.EQUITY),
        order_type=OrderType.LIMIT,
        side=OrderSide.BUY,
        quantity=Quantity(amount=Decimal("100")),
        price=Decimal("150.00"),
    )

    # Test basic properties
    assert order.symbol.code == "AAPL"
    assert order.order_type == OrderType.LIMIT
    assert order.side == OrderSide.BUY

    # Test update method
    updated = order.update(status=OrderStatus.FILLED)
    assert updated.status == OrderStatus.FILLED


def test_trade_basic():
    """Basic Trade creation."""
    trade = Trade(
        symbol=Symbol(code="AAPL", asset_class=AssetClass.EQUITY),
        side=OrderSide.BUY,
        quantity=Quantity(amount=Decimal("100")),
        price=Decimal("151.25"),
        timestamp=datetime.utcnow(),
        order_id="order-123",
    )

    assert trade.symbol.code == "AAPL"
    assert trade.side == OrderSide.BUY
    assert trade.order_id == "order-123"


def test_position_basic():
    """Basic Position creation and methods."""
    position = Position(
        symbol=Symbol(code="AAPL", asset_class=AssetClass.EQUITY),
        position_type=PositionType.LONG,
        quantity=Quantity(amount=Decimal("100")),
        entry_price=Decimal("150.00"),
        entry_timestamp=datetime.utcnow(),
    )

    assert position.symbol.code == "AAPL"
    assert position.position_type == PositionType.LONG

    # Test update method
    updated = position.update(current_price=Decimal("155.00"))
    assert updated.current_price == Decimal("155.00")


def test_account_basic():
    """Basic Account creation."""
    account = Account(
        name="Test Account",
        account_type="margin",
        base_currency=Currency.USD,
        cash_balance=Money(amount=Decimal("100000"), currency=Currency.USD),
        total_equity=Money(amount=Decimal("100000"), currency=Currency.USD),
        buying_power=Money(amount=Decimal("200000"), currency=Currency.USD),
        maintenance_margin=Money(amount=Decimal("0"), currency=Currency.USD),
        max_position_size=Decimal("10"),
        max_portfolio_risk=Decimal("5"),
    )

    assert account.name == "Test Account"
    assert account.base_currency == Currency.USD
    assert account.cash_balance.amount == Decimal("100000")


def test_portfolio_basic():
    """Basic Portfolio creation."""
    portfolio = Portfolio(
        name="Test Portfolio",
        account_id="account-123",
        base_currency=Currency.USD,
        cash_balance=Money(amount=Decimal("50000"), currency=Currency.USD),
        total_value=Money(amount=Decimal("100000"), currency=Currency.USD),
    )

    assert portfolio.name == "Test Portfolio"
    assert portfolio.account_id == "account-123"
    assert portfolio.base_currency == Currency.USD
