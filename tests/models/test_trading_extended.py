"""Extended tests for trading models to improve coverage."""

from datetime import datetime
from decimal import Decimal

from src.voyager_trader.models.trading import (
    Account,
    Order,
    OrderExecuted,
    Portfolio,
    Position,
    PositionClosed,
    PositionOpened,
    Trade,
)
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


def test_order_extended():
    """Extended Order tests to improve coverage."""
    # Test market order (no price)
    market_order = Order(
        symbol=Symbol(code="AAPL", asset_class=AssetClass.EQUITY),
        order_type=OrderType.MARKET,
        side=OrderSide.SELL,
        quantity=Quantity(amount=Decimal("50")),
    )
    assert market_order.price is None

    # Test order with commission
    order_with_commission = Order(
        symbol=Symbol(code="GOOGL", asset_class=AssetClass.EQUITY),
        order_type=OrderType.LIMIT,
        side=OrderSide.BUY,
        quantity=Quantity(amount=Decimal("10")),
        price=Decimal("2500.00"),
        commission=Money(amount=Decimal("5.00"), currency=Currency.USD),
    )
    assert order_with_commission.commission.amount == Decimal("5.00")

    # Test order status transitions
    filled_order = order_with_commission.update(
        status=OrderStatus.FILLED,
        filled_quantity=Quantity(amount=Decimal("10")),
        average_fill_price=Decimal("2505.00"),
    )
    assert filled_order.status == OrderStatus.FILLED
    assert filled_order.filled_quantity.amount == Decimal("10")
    assert filled_order.average_fill_price == Decimal("2505.00")


def test_position_extended():
    """Extended Position tests to improve coverage."""
    # Test position with stop loss
    position = Position(
        symbol=Symbol(code="MSFT", asset_class=AssetClass.EQUITY),
        position_type=PositionType.LONG,
        quantity=Quantity(amount=Decimal("200")),
        entry_price=Decimal("300.00"),
        entry_timestamp=datetime.utcnow(),
        stop_loss=Decimal("280.00"),
        take_profit=Decimal("350.00"),
        current_price=Decimal("320.00"),
    )

    assert position.stop_loss == Decimal("280.00")
    assert position.take_profit == Decimal("350.00")
    assert position.current_price == Decimal("320.00")

    # Test position with notes
    position_with_notes = position.update(notes="Strong momentum play")
    assert position_with_notes.notes == "Strong momentum play"

    # Test short position
    short_position = Position(
        symbol=Symbol(code="AAPL", asset_class=AssetClass.EQUITY),
        position_type=PositionType.SHORT,
        quantity=Quantity(amount=Decimal("100")),
        entry_price=Decimal("150.00"),
        entry_timestamp=datetime.utcnow(),
        current_price=Decimal("140.00"),
    )
    assert short_position.position_type == PositionType.SHORT


def test_account_extended():
    """Extended Account tests to improve coverage."""
    # Test account with different types
    cash_account = Account(
        name="Cash Account",
        account_type="cash",
        base_currency=Currency.USD,
        cash_balance=Money(amount=Decimal("50000"), currency=Currency.USD),
        total_equity=Money(amount=Decimal("50000"), currency=Currency.USD),
        buying_power=Money(amount=Decimal("50000"), currency=Currency.USD),
        maintenance_margin=Money(amount=Decimal("0"), currency=Currency.USD),
        max_position_size=Decimal("5"),
        max_portfolio_risk=Decimal("3"),
    )

    assert cash_account.account_type == "cash"
    assert cash_account.max_position_size == Decimal("5")

    # Test account with different currencies
    eur_account = Account(
        name="EUR Account",
        account_type="margin",
        base_currency=Currency.EUR,
        cash_balance=Money(amount=Decimal("100000"), currency=Currency.EUR),
        total_equity=Money(amount=Decimal("120000"), currency=Currency.EUR),
        buying_power=Money(amount=Decimal("240000"), currency=Currency.EUR),
        maintenance_margin=Money(amount=Decimal("20000"), currency=Currency.EUR),
        max_position_size=Decimal("15"),
        max_portfolio_risk=Decimal("8"),
    )

    assert eur_account.base_currency == Currency.EUR
    assert eur_account.cash_balance.currency == Currency.EUR


def test_portfolio_extended():
    """Extended Portfolio tests to improve coverage."""
    # Test portfolio with different currencies
    eur_portfolio = Portfolio(
        name="EUR Portfolio",
        account_id="eur-account-123",
        base_currency=Currency.EUR,
        cash_balance=Money(amount=Decimal("25000"), currency=Currency.EUR),
        total_value=Money(amount=Decimal("75000"), currency=Currency.EUR),
    )

    assert eur_portfolio.base_currency == Currency.EUR
    assert eur_portfolio.cash_balance.currency == Currency.EUR

    # Test portfolio with positions
    portfolio_with_positions = eur_portfolio.update(
        positions={"ASML": "pos-456", "SAP": "pos-789"},
        realized_pnl=Money(amount=Decimal("2500"), currency=Currency.EUR),
        unrealized_pnl=Money(amount=Decimal("1200"), currency=Currency.EUR),
    )

    assert len(portfolio_with_positions.positions) == 2
    assert "ASML" in portfolio_with_positions.positions
    assert portfolio_with_positions.realized_pnl.amount == Decimal("2500")


def test_trade_extended():
    """Extended Trade tests to improve coverage."""
    # Test trade with commission
    trade = Trade(
        symbol=Symbol(code="NVDA", asset_class=AssetClass.EQUITY),
        side=OrderSide.BUY,
        quantity=Quantity(amount=Decimal("50")),
        price=Decimal("800.50"),
        timestamp=datetime.utcnow(),
        order_id="order-nvda-123",
        commission=Money(amount=Decimal("2.50"), currency=Currency.USD),
    )

    assert trade.commission.amount == Decimal("2.50")

    # Test sell trade
    sell_trade = Trade(
        symbol=Symbol(code="TSLA", asset_class=AssetClass.EQUITY),
        side=OrderSide.SELL,
        quantity=Quantity(amount=Decimal("25")),
        price=Decimal("250.75"),
        timestamp=datetime.utcnow(),
        order_id="order-tsla-456",
    )

    assert sell_trade.side == OrderSide.SELL


def test_domain_events():
    """Test domain events."""
    # Test OrderExecuted event
    order_event = OrderExecuted(
        aggregate_id="portfolio-123",
        order_id="order-456",
        symbol=Symbol(code="AAPL", asset_class=AssetClass.EQUITY),
        side=OrderSide.BUY,
        quantity=Quantity(amount=Decimal("100")),
        price=Decimal("150.00"),
    )
    assert order_event.order_id == "order-456"
    assert order_event.price == Decimal("150.00")

    # Test PositionOpened event
    position_event = PositionOpened(
        aggregate_id="portfolio-123",
        position_id="pos-789",
        symbol=Symbol(code="GOOGL", asset_class=AssetClass.EQUITY),
        position_type=PositionType.LONG,
        quantity=Quantity(amount=Decimal("10")),
    )
    assert position_event.position_id == "pos-789"

    # Test PositionClosed event
    close_event = PositionClosed(
        aggregate_id="portfolio-123",
        position_id="pos-789",
        symbol=Symbol(code="GOOGL", asset_class=AssetClass.EQUITY),
        pnl=Money(amount=Decimal("500"), currency=Currency.USD),
    )
    assert close_event.pnl.amount == Decimal("500")
