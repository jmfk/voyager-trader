"""Tests for persistent trading service."""

import pytest
import tempfile
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from src.voyager_trader.models.trading import Account, Order, Portfolio, Position, Trade
from src.voyager_trader.models.types import (
    Currency, Money, OrderSide, OrderStatus, OrderType, 
    PositionType, Quantity, Symbol
)
from src.voyager_trader.persistence.database import DatabaseManager
from src.voyager_trader.persistence.trading_service import PersistentTradingService


@pytest.fixture
async def trading_service():
    """Create a trading service with temporary database."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    
    db_manager = DatabaseManager(
        database_url=f"sqlite:///{db_path}",
        pool_size=2,
        max_overflow=2,
        echo=False
    )
    
    await db_manager.initialize()
    service = PersistentTradingService(db_manager)
    
    yield service
    
    await db_manager.close()
    Path(db_path).unlink(missing_ok=True)


@pytest.fixture
def sample_account():
    """Create a sample account."""
    return Account(
        name="Test Trading Account",
        account_type="margin",
        base_currency=Currency.USD,
        cash_balance=Money(amount=Decimal("50000"), currency=Currency.USD),
        total_equity=Money(amount=Decimal("50000"), currency=Currency.USD),
        buying_power=Money(amount=Decimal("100000"), currency=Currency.USD),
        maintenance_margin=Money(amount=Decimal("0"), currency=Currency.USD),
        max_position_size=Decimal("5.0"),
        max_portfolio_risk=Decimal("2.0"),
    )


@pytest.fixture
def sample_portfolio(sample_account):
    """Create a sample portfolio."""
    return Portfolio(
        name="Test Trading Portfolio",
        account_id=sample_account.id,
        base_currency=Currency.USD,
        cash_balance=Money(amount=Decimal("25000"), currency=Currency.USD),
        total_value=Money(amount=Decimal("50000"), currency=Currency.USD),
    )


@pytest.mark.asyncio
async def test_create_account_with_audit(trading_service, sample_account):
    """Test account creation with audit logging."""
    # Create account
    created_account = await trading_service.create_account(
        sample_account, 
        user_id="test-user-123"
    )
    
    assert created_account.id == sample_account.id
    assert created_account.name == sample_account.name
    
    # Verify account was saved
    found_account = await trading_service.get_account(sample_account.id)
    assert found_account is not None
    assert found_account.name == sample_account.name
    
    # Verify audit log was created
    audit_logs = await trading_service.get_audit_logs_for_entity(
        "account", sample_account.id
    )
    assert len(audit_logs) == 1
    assert audit_logs[0]["action"] == "account_created"
    assert audit_logs[0]["user_id"] == "test-user-123"


@pytest.mark.asyncio
async def test_update_account_balances_with_audit(trading_service, sample_account):
    """Test account balance update with audit logging."""
    # Create account first
    await trading_service.create_account(sample_account)
    
    # Update balances
    new_cash = Money(amount=Decimal("45000"), currency=Currency.USD)
    new_equity = Money(amount=Decimal("55000"), currency=Currency.USD)
    new_buying_power = Money(amount=Decimal("110000"), currency=Currency.USD)
    new_margin = Money(amount=Decimal("5000"), currency=Currency.USD)
    
    updated_account = await trading_service.update_account_balances(
        account_id=sample_account.id,
        cash_balance=new_cash,
        total_equity=new_equity,
        buying_power=new_buying_power,
        maintenance_margin=new_margin,
        user_id="test-user-123"
    )
    
    assert updated_account is not None
    assert updated_account.cash_balance.amount == Decimal("45000")
    assert updated_account.total_equity.amount == Decimal("55000")
    
    # Verify audit log
    audit_logs = await trading_service.get_audit_logs_for_entity(
        "account", sample_account.id
    )
    assert len(audit_logs) == 2  # Creation + update
    
    update_log = next(log for log in audit_logs if log["action"] == "account_balances_updated")
    assert update_log["user_id"] == "test-user-123"
    assert update_log["old_values"] is not None
    assert update_log["new_values"] is not None


@pytest.mark.asyncio
async def test_create_portfolio_with_audit(trading_service, sample_account, sample_portfolio):
    """Test portfolio creation with audit logging."""
    # Create account dependency
    await trading_service.create_account(sample_account)
    
    # Create portfolio
    created_portfolio = await trading_service.create_portfolio(
        sample_portfolio,
        user_id="test-user-123"
    )
    
    assert created_portfolio.id == sample_portfolio.id
    assert created_portfolio.account_id == sample_account.id
    
    # Verify portfolio was saved
    found_portfolio = await trading_service.get_portfolio(sample_portfolio.id)
    assert found_portfolio is not None
    
    # Verify audit log
    audit_logs = await trading_service.get_audit_logs_for_entity(
        "portfolio", sample_portfolio.id
    )
    assert len(audit_logs) == 1
    assert audit_logs[0]["action"] == "portfolio_created"


@pytest.mark.asyncio
async def test_create_order_with_audit(trading_service):
    """Test order creation with audit logging."""
    order = Order(
        symbol=Symbol(code="TSLA", asset_class="equity"),
        order_type=OrderType.LIMIT,
        side=OrderSide.BUY,
        quantity=Quantity(amount=Decimal("50")),
        price=Decimal("250.00"),
    )
    
    created_order = await trading_service.create_order(
        order,
        strategy_id="momentum-strategy-1",
        user_id="test-user-123"
    )
    
    assert created_order.strategy_id == "momentum-strategy-1"
    assert created_order.symbol.code == "TSLA"
    
    # Verify audit log
    audit_logs = await trading_service.get_audit_logs_for_entity(
        "order", order.id
    )
    assert len(audit_logs) == 1
    
    log = audit_logs[0]
    assert log["action"] == "order_created"
    assert log["strategy_id"] == "momentum-strategy-1"
    assert log["metadata"]["symbol"] == "TSLA"
    assert log["metadata"]["side"] == "buy"


@pytest.mark.asyncio
async def test_update_order_with_audit(trading_service):
    """Test order update with audit logging."""
    order = Order(
        symbol=Symbol(code="NVDA", asset_class="equity"),
        order_type=OrderType.LIMIT,
        side=OrderSide.SELL,
        quantity=Quantity(amount=Decimal("25")),
        price=Decimal("900.00"),
    )
    
    # Create initial order
    created_order = await trading_service.create_order(order)
    
    # Execute partial fill
    updated_order = created_order.execute_partial(
        quantity=Quantity(amount=Decimal("10")),
        price=Decimal("905.00")
    )
    
    # Update order
    saved_order = await trading_service.update_order(
        updated_order,
        strategy_id="test-strategy"
    )
    
    assert saved_order.status == OrderStatus.PARTIALLY_FILLED
    assert saved_order.filled_quantity.amount == Decimal("10")
    
    # Verify audit logs
    audit_logs = await trading_service.get_audit_logs_for_entity(
        "order", order.id
    )
    assert len(audit_logs) == 2  # Create + update
    
    update_log = next(log for log in audit_logs if log["action"] == "order_updated")
    assert "status_change" in update_log["metadata"]


@pytest.mark.asyncio
async def test_create_trade_with_audit(trading_service):
    """Test trade creation with audit logging."""
    # Create order first
    order = Order(
        symbol=Symbol(code="AMZN", asset_class="equity"),
        order_type=OrderType.MARKET,
        side=OrderSide.BUY,
        quantity=Quantity(amount=Decimal("10")),
    )
    created_order = await trading_service.create_order(order)
    
    # Create trade
    trade = Trade(
        symbol=Symbol(code="AMZN", asset_class="equity"),
        side=OrderSide.BUY,
        quantity=Quantity(amount=Decimal("10")),
        price=Decimal("3450.00"),
        timestamp=datetime.now(timezone.utc),
        order_id=created_order.id,
    )
    
    created_trade = await trading_service.create_trade(
        trade,
        strategy_id="buy-the-dip-v2"
    )
    
    assert created_trade.strategy_id == "buy-the-dip-v2"
    assert created_trade.notional_value == Decimal("34500.00")
    
    # Verify audit log
    audit_logs = await trading_service.get_audit_logs_for_entity(
        "trade", trade.id
    )
    assert len(audit_logs) == 1
    
    log = audit_logs[0]
    assert log["action"] == "trade_executed"
    assert log["metadata"]["notional_value"] == 34500.0


@pytest.mark.asyncio
async def test_create_position_with_audit(trading_service, sample_account, sample_portfolio):
    """Test position creation with audit logging."""
    # Create dependencies
    await trading_service.create_account(sample_account)
    await trading_service.create_portfolio(sample_portfolio)
    
    position = Position(
        symbol=Symbol(code="MSFT", asset_class="equity"),
        position_type=PositionType.LONG,
        quantity=Quantity(amount=Decimal("75")),
        entry_price=Decimal("420.00"),
        entry_timestamp=datetime.now(timezone.utc),
        portfolio_id=sample_portfolio.id,
    )
    
    created_position = await trading_service.create_position(
        position,
        strategy_id="tech-growth-momentum"
    )
    
    assert created_position.strategy_id == "tech-growth-momentum"
    assert created_position.cost_basis == Decimal("31500.00")
    
    # Verify audit log
    audit_logs = await trading_service.get_audit_logs_for_entity(
        "position", position.id
    )
    assert len(audit_logs) == 1
    
    log = audit_logs[0]
    assert log["action"] == "position_opened"
    assert log["metadata"]["cost_basis"] == 31500.0


@pytest.mark.asyncio
async def test_query_operations(trading_service, sample_account):
    """Test various query operations."""
    # Create test data
    await trading_service.create_account(sample_account)
    
    # Test get active accounts
    active_accounts = await trading_service.get_active_accounts()
    assert len(active_accounts) == 1
    assert active_accounts[0].id == sample_account.id
    
    # Create multiple orders
    orders = []
    for i in range(3):
        order = Order(
            symbol=Symbol(code=f"STOCK{i}", asset_class="equity"),
            order_type=OrderType.LIMIT,
            side=OrderSide.BUY,
            quantity=Quantity(amount=Decimal("100")),
            price=Decimal(f"{100 + i * 10}.00"),
        )
        orders.append(order)
        await trading_service.create_order(order)
    
    # Test get active orders
    active_orders = await trading_service.get_active_orders()
    assert len(active_orders) == 3
    
    # Test get orders by symbol
    stock0_orders = await trading_service.get_orders_by_symbol("STOCK0")
    assert len(stock0_orders) == 1
    assert stock0_orders[0].symbol.code == "STOCK0"


@pytest.mark.asyncio
async def test_portfolio_summary(trading_service, sample_account, sample_portfolio):
    """Test portfolio summary generation."""
    # Create account and portfolio
    await trading_service.create_account(sample_account)
    await trading_service.create_portfolio(sample_portfolio)
    
    # Create some positions
    positions = []
    for i in range(2):
        position = Position(
            symbol=Symbol(code=f"POS{i}", asset_class="equity"),
            position_type=PositionType.LONG,
            quantity=Quantity(amount=Decimal("50")),
            entry_price=Decimal(f"{200 + i * 50}.00"),
            entry_timestamp=datetime.now(timezone.utc),
            portfolio_id=sample_portfolio.id,
        )
        positions.append(position)
        await trading_service.create_position(position)
    
    # Get portfolio summary
    summary = await trading_service.get_portfolio_summary(sample_portfolio.id)
    
    assert summary is not None
    assert "portfolio" in summary
    assert "positions" in summary
    assert summary["total_positions"] == 2
    assert summary["open_position_count"] == 2
    assert summary["closed_position_count"] == 0


@pytest.mark.asyncio
async def test_trading_statistics(trading_service):
    """Test trading statistics calculation."""
    # Create some trades
    trades_data = [
        ("AAPL", OrderSide.BUY, "100", "150.00"),
        ("AAPL", OrderSide.SELL, "50", "155.00"),
        ("GOOGL", OrderSide.BUY, "10", "2800.00"),
        ("MSFT", OrderSide.BUY, "25", "420.00"),
    ]
    
    start_date = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    
    for symbol, side, qty, price in trades_data:
        # Create order first
        order = Order(
            symbol=Symbol(code=symbol, asset_class="equity"),
            order_type=OrderType.MARKET,
            side=side,
            quantity=Quantity(amount=Decimal(qty)),
        )
        created_order = await trading_service.create_order(order)
        
        # Create trade
        trade = Trade(
            symbol=Symbol(code=symbol, asset_class="equity"),
            side=side,
            quantity=Quantity(amount=Decimal(qty)),
            price=Decimal(price),
            timestamp=datetime.now(timezone.utc),
            order_id=created_order.id,
        )
        await trading_service.create_trade(trade)
    
    # Get statistics
    end_date = datetime.now(timezone.utc).replace(hour=23, minute=59, second=59, microsecond=999999)
    stats = await trading_service.get_trading_statistics(start_date, end_date)
    
    assert stats["total_trades"] == 4
    assert stats["buy_trades"] == 3
    assert stats["sell_trades"] == 1
    assert stats["total_volume"] == 185.0  # 100 + 50 + 10 + 25
    
    # Expected notional: 15000 + 7750 + 28000 + 10500 = 61250
    assert stats["total_notional"] == 61250.0


@pytest.mark.asyncio
async def test_audit_queries(trading_service, sample_account):
    """Test audit log queries."""
    # Create account (generates audit log)
    await trading_service.create_account(sample_account, user_id="test-user")
    
    # Update account (generates another audit log)
    await trading_service.update_account_balances(
        account_id=sample_account.id,
        cash_balance=Money(amount=Decimal("40000"), currency=Currency.USD),
        total_equity=Money(amount=Decimal("40000"), currency=Currency.USD),
        buying_power=Money(amount=Decimal("80000"), currency=Currency.USD),
        maintenance_margin=Money(amount=Decimal("0"), currency=Currency.USD),
        user_id="test-user",
    )
    
    # Test get audit logs for entity
    entity_logs = await trading_service.get_audit_logs_for_entity(
        "account", sample_account.id
    )
    assert len(entity_logs) == 2
    
    # Test get audit logs by action
    creation_logs = await trading_service.get_audit_logs_by_action("account_created")
    assert len(creation_logs) == 1
    assert creation_logs[0]["entity_id"] == sample_account.id