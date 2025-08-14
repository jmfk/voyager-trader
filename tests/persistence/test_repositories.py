"""Tests for repository implementations."""

import tempfile
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest
import pytest_asyncio

from src.voyager_trader.models.trading import Account, Order, Portfolio, Position, Trade
from src.voyager_trader.models.types import (
    Currency,
    Money,
    OrderSide,
    OrderStatus,
    OrderType,
    PositionType,
    Quantity,
    Symbol,
)
from src.voyager_trader.persistence.database import DatabaseManager
from src.voyager_trader.persistence.repositories import (
    AccountRepository,
    AuditLogRepository,
    OrderRepository,
    PortfolioRepository,
    PositionRepository,
    TradeRepository,
)


@pytest_asyncio.fixture
async def db_manager():
    """Create a temporary database manager for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    manager = DatabaseManager(
        database_url=f"sqlite:///{db_path}", pool_size=2, max_overflow=2, echo=False
    )

    await manager.initialize()

    yield manager

    await manager.close()
    Path(db_path).unlink(missing_ok=True)


@pytest.fixture
def sample_account():
    """Create a sample account for testing."""
    return Account(
        name="Test Account",
        account_type="cash",
        base_currency=Currency.USD,
        cash_balance=Money(amount=Decimal("10000"), currency=Currency.USD),
        total_equity=Money(amount=Decimal("10000"), currency=Currency.USD),
        buying_power=Money(amount=Decimal("10000"), currency=Currency.USD),
        maintenance_margin=Money(amount=Decimal("0"), currency=Currency.USD),
        max_position_size=Decimal("10.0"),
        max_portfolio_risk=Decimal("2.0"),
    )


@pytest.fixture
def sample_portfolio(sample_account):
    """Create a sample portfolio for testing."""
    return Portfolio(
        name="Test Portfolio",
        account_id=sample_account.id,
        base_currency=Currency.USD,
        cash_balance=Money(amount=Decimal("5000"), currency=Currency.USD),
        total_value=Money(amount=Decimal("10000"), currency=Currency.USD),
    )


@pytest.fixture
def sample_order():
    """Create a sample order for testing."""
    return Order(
        symbol=Symbol(code="AAPL", asset_class="equity"),
        order_type=OrderType.LIMIT,
        side=OrderSide.BUY,
        quantity=Quantity(amount=Decimal("100")),
        price=Decimal("150.00"),
    )


@pytest.fixture
def sample_trade(sample_order):
    """Create a sample trade for testing."""
    return Trade(
        symbol=Symbol(code="AAPL", asset_class="equity"),
        side=OrderSide.BUY,
        quantity=Quantity(amount=Decimal("100")),
        price=Decimal("150.00"),
        timestamp=datetime.now(timezone.utc),
        order_id=sample_order.id,
    )


@pytest.fixture
def sample_position(sample_portfolio):
    """Create a sample position for testing."""
    return Position(
        symbol=Symbol(code="AAPL", asset_class="equity"),
        position_type=PositionType.LONG,
        quantity=Quantity(amount=Decimal("100")),
        entry_price=Decimal("150.00"),
        entry_timestamp=datetime.now(timezone.utc),
    )


@pytest.mark.asyncio
async def test_account_repository_crud(db_manager, sample_account):
    """Test Account repository CRUD operations."""
    repo = AccountRepository(db_manager)

    # Test create
    saved_account = await repo.save(sample_account)
    assert saved_account.id == sample_account.id
    assert saved_account.name == sample_account.name

    # Test find by id
    found_account = await repo.find_by_id(sample_account.id)
    assert found_account is not None
    assert found_account.name == sample_account.name
    assert found_account.cash_balance.amount == sample_account.cash_balance.amount

    # Test find by name
    found_by_name = await repo.find_by_name(sample_account.name)
    assert found_by_name is not None
    assert found_by_name.id == sample_account.id

    # Test update
    updated_account = found_account.update(name="Updated Test Account")
    saved_updated = await repo.save(updated_account)
    assert saved_updated.name == "Updated Test Account"
    assert saved_updated.version == found_account.version + 1

    # Test find active accounts
    active_accounts = await repo.find_active_accounts()
    assert len(active_accounts) == 1
    assert active_accounts[0].id == sample_account.id

    # Test delete
    deleted = await repo.delete(sample_account.id)
    assert deleted is True

    # Verify deletion
    not_found = await repo.find_by_id(sample_account.id)
    assert not_found is None


@pytest.mark.asyncio
async def test_portfolio_repository_crud(db_manager, sample_account, sample_portfolio):
    """Test Portfolio repository CRUD operations."""
    # First save the account (foreign key dependency)
    account_repo = AccountRepository(db_manager)
    await account_repo.save(sample_account)

    repo = PortfolioRepository(db_manager)

    # Test create
    saved_portfolio = await repo.save(sample_portfolio)
    assert saved_portfolio.id == sample_portfolio.id
    assert saved_portfolio.name == sample_portfolio.name

    # Test find by id
    found_portfolio = await repo.find_by_id(sample_portfolio.id)
    assert found_portfolio is not None
    assert found_portfolio.name == sample_portfolio.name
    assert found_portfolio.account_id == sample_account.id

    # Test find by account id
    portfolios_by_account = await repo.find_by_account_id(sample_account.id)
    assert len(portfolios_by_account) == 1
    assert portfolios_by_account[0].id == sample_portfolio.id

    # Test update
    updated_portfolio = found_portfolio.update_metrics(
        unrealized_pnl=Money(amount=Decimal("500"), currency=Currency.USD),
        total_value=Money(amount=Decimal("10500"), currency=Currency.USD),
    )
    saved_updated = await repo.save(updated_portfolio)
    assert saved_updated.unrealized_pnl.amount == Decimal("500")


@pytest.mark.asyncio
async def test_order_repository_crud(db_manager, sample_order):
    """Test Order repository CRUD operations."""
    repo = OrderRepository(db_manager)

    # Test create
    saved_order = await repo.save(sample_order)
    assert saved_order.id == sample_order.id
    assert saved_order.symbol.code == "AAPL"

    # Test find by id
    found_order = await repo.find_by_id(sample_order.id)
    assert found_order is not None
    assert found_order.symbol.code == sample_order.symbol.code
    assert found_order.quantity.amount == sample_order.quantity.amount

    # Test find by symbol
    orders_by_symbol = await repo.find_by_symbol("AAPL")
    assert len(orders_by_symbol) == 1
    assert orders_by_symbol[0].id == sample_order.id

    # Test find by status
    orders_by_status = await repo.find_by_status(OrderStatus.PENDING)
    assert len(orders_by_status) == 1

    # Test find active orders
    active_orders = await repo.find_active_orders()
    assert len(active_orders) == 1

    # Test order execution
    executed_order = found_order.execute_partial(
        quantity=Quantity(amount=Decimal("50")), price=Decimal("151.00")
    )
    saved_executed = await repo.save(executed_order)
    assert saved_executed.status == OrderStatus.PARTIALLY_FILLED
    assert saved_executed.filled_quantity.amount == Decimal("50")


@pytest.mark.asyncio
async def test_trade_repository_crud(db_manager, sample_order, sample_trade):
    """Test Trade repository CRUD operations."""
    # First save the order (foreign key dependency)
    order_repo = OrderRepository(db_manager)
    await order_repo.save(sample_order)

    repo = TradeRepository(db_manager)

    # Test create
    saved_trade = await repo.save(sample_trade)
    assert saved_trade.id == sample_trade.id
    assert saved_trade.symbol.code == "AAPL"

    # Test find by id
    found_trade = await repo.find_by_id(sample_trade.id)
    assert found_trade is not None
    assert found_trade.order_id == sample_order.id
    assert found_trade.notional_value == Decimal("15000.00")

    # Test find by symbol
    trades_by_symbol = await repo.find_by_symbol("AAPL")
    assert len(trades_by_symbol) == 1

    # Test find by order id
    trades_by_order = await repo.find_by_order_id(sample_order.id)
    assert len(trades_by_order) == 1

    # Test find by date range
    start_date = datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    end_date = datetime.now(timezone.utc).replace(
        hour=23, minute=59, second=59, microsecond=999999
    )
    trades_by_date = await repo.find_by_date_range(start_date, end_date)
    assert len(trades_by_date) == 1


@pytest.mark.asyncio
async def test_position_repository_crud(
    db_manager, sample_account, sample_portfolio, sample_position
):
    """Test Position repository CRUD operations."""
    # Save dependencies
    account_repo = AccountRepository(db_manager)
    await account_repo.save(sample_account)

    portfolio_repo = PortfolioRepository(db_manager)
    await portfolio_repo.save(sample_portfolio)

    repo = PositionRepository(db_manager)

    # Test create
    saved_position = await repo.save(sample_position)
    assert saved_position.id == sample_position.id
    assert saved_position.symbol.code == "AAPL"

    # Test find by id
    found_position = await repo.find_by_id(sample_position.id)
    assert found_position is not None
    assert found_position.portfolio_id == sample_portfolio.id
    assert found_position.cost_basis == Decimal("15000.00")

    # Test find open positions
    open_positions = await repo.find_open_positions()
    assert len(open_positions) == 1

    # Test find by portfolio
    positions_by_portfolio = await repo.find_by_portfolio_id(sample_portfolio.id)
    assert len(positions_by_portfolio) == 1

    # Test find by symbol
    positions_by_symbol = await repo.find_by_symbol("AAPL")
    assert len(positions_by_symbol) == 1

    # Test position update
    updated_position = found_position.update_price(Decimal("155.00"))
    saved_updated = await repo.save(updated_position)
    assert saved_updated.current_price == Decimal("155.00")
    assert saved_updated.unrealized_pnl == Decimal("500.00")  # (155-150) * 100

    # Test position closing
    closed_position = updated_position.close_position(
        exit_price=Decimal("160.00"), exit_timestamp=datetime.now(timezone.utc)
    )
    saved_closed = await repo.save(closed_position)
    assert saved_closed.is_closed is True
    assert saved_closed.current_price == Decimal("160.00")


@pytest.mark.asyncio
async def test_audit_log_repository(db_manager):
    """Test AuditLog repository operations."""
    repo = AuditLogRepository(db_manager)

    # Test create log
    test_values = {"field1": "value1", "field2": 42}
    log_id = await repo.create_log(
        action="test_action",
        entity_type="test_entity",
        entity_id="test-123",
        new_values=test_values,
        user_id="user-123",
        metadata={"test": True},
    )

    assert log_id is not None

    # Test find by entity
    logs_by_entity = await repo.find_by_entity("test_entity", "test-123")
    assert len(logs_by_entity) == 1
    assert logs_by_entity[0]["action"] == "test_action"
    assert logs_by_entity[0]["new_values"] == test_values

    # Test find by action
    logs_by_action = await repo.find_by_action("test_action")
    assert len(logs_by_action) == 1

    # Test with old values
    old_values = {"field1": "old_value", "field2": 24}
    new_values = {"field1": "new_value", "field2": 48}

    await repo.create_log(
        action="test_update",
        entity_type="test_entity",
        entity_id="test-123",
        old_values=old_values,
        new_values=new_values,
    )

    update_logs = await repo.find_by_action("test_update")
    assert len(update_logs) == 1
    assert update_logs[0]["old_values"] == old_values
    assert update_logs[0]["new_values"] == new_values


@pytest.mark.asyncio
async def test_repository_find_all(db_manager, sample_account):
    """Test find_all with pagination."""
    repo = AccountRepository(db_manager)

    # Create multiple accounts
    accounts = []
    for i in range(5):
        account = Account(
            name=f"Account {i}",
            account_type="cash",
            base_currency=Currency.USD,
            cash_balance=Money(amount=Decimal("1000"), currency=Currency.USD),
            total_equity=Money(amount=Decimal("1000"), currency=Currency.USD),
            buying_power=Money(amount=Decimal("1000"), currency=Currency.USD),
            maintenance_margin=Money(amount=Decimal("0"), currency=Currency.USD),
            max_position_size=Decimal("10.0"),
            max_portfolio_risk=Decimal("2.0"),
        )
        accounts.append(account)
        await repo.save(account)

    # Test find all without pagination
    all_accounts = await repo.find_all()
    assert len(all_accounts) == 5

    # Test find all with pagination
    page1 = await repo.find_all(limit=2, offset=0)
    assert len(page1) == 2

    page2 = await repo.find_all(limit=2, offset=2)
    assert len(page2) == 2

    page3 = await repo.find_all(limit=2, offset=4)
    assert len(page3) == 1
