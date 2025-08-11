"""Tests for audit service."""

import tempfile
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from src.voyager_trader.models.trading import Account
from src.voyager_trader.models.types import Currency, Money
from src.voyager_trader.persistence.audit_service import AuditService
from src.voyager_trader.persistence.database import DatabaseManager


@pytest.fixture
async def audit_service():
    """Create audit service with temporary database."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    db_manager = DatabaseManager(
        database_url=f"sqlite:///{db_path}", pool_size=2, max_overflow=2, echo=False
    )

    await db_manager.initialize()
    service = AuditService(db_manager)

    yield service

    await db_manager.close()
    Path(db_path).unlink(missing_ok=True)


@pytest.fixture
def sample_account():
    """Create a sample account for testing."""
    return Account(
        name="Audit Test Account",
        account_type="cash",
        base_currency=Currency.USD,
        cash_balance=Money(amount=Decimal("10000"), currency=Currency.USD),
        total_equity=Money(amount=Decimal("10000"), currency=Currency.USD),
        buying_power=Money(amount=Decimal("10000"), currency=Currency.USD),
        maintenance_margin=Money(amount=Decimal("0"), currency=Currency.USD),
        max_position_size=Decimal("10.0"),
        max_portfolio_risk=Decimal("2.0"),
    )


@pytest.mark.asyncio
async def test_log_entity_created(audit_service, sample_account):
    """Test logging entity creation."""
    log_id = await audit_service.log_entity_created(
        entity=sample_account,
        user_id="test-user-123",
        strategy_id="test-strategy",
        metadata={"source": "test"},
    )

    assert log_id is not None

    # Verify log was created
    logs = await audit_service.get_entity_history("account", sample_account.id)
    assert len(logs) == 1

    log = logs[0]
    assert log["action"] == "account_created"
    assert log["entity_type"] == "account"
    assert log["entity_id"] == sample_account.id
    assert log["user_id"] == "test-user-123"
    assert log["strategy_id"] == "test-strategy"
    assert log["metadata"]["source"] == "test"


@pytest.mark.asyncio
async def test_log_entity_updated(audit_service, sample_account):
    """Test logging entity updates."""
    # Create initial entity
    await audit_service.log_entity_created(sample_account, user_id="test-user")

    # Update entity
    updated_account = sample_account.update(name="Updated Account Name")

    log_id = await audit_service.log_entity_updated(
        old_entity=sample_account,
        new_entity=updated_account,
        user_id="test-user-123",
        metadata={"change_reason": "name_update"},
    )

    assert log_id is not None

    # Verify logs
    logs = await audit_service.get_entity_history("account", sample_account.id)
    assert len(logs) == 2

    update_log = next(log for log in logs if log["action"] == "account_updated")
    assert update_log["old_values"]["name"] == "Audit Test Account"
    assert update_log["new_values"]["name"] == "Updated Account Name"
    assert update_log["metadata"]["change_reason"] == "name_update"


@pytest.mark.asyncio
async def test_log_entity_deleted(audit_service, sample_account):
    """Test logging entity deletion."""
    # Create and then delete entity
    await audit_service.log_entity_created(sample_account)

    log_id = await audit_service.log_entity_deleted(
        entity=sample_account,
        user_id="admin-user",
        metadata={"deletion_reason": "cleanup"},
    )

    assert log_id is not None

    # Verify logs
    logs = await audit_service.get_entity_history("account", sample_account.id)
    assert len(logs) == 2

    delete_log = next(log for log in logs if log["action"] == "account_deleted")
    assert delete_log["user_id"] == "admin-user"
    assert delete_log["old_values"]["name"] == "Audit Test Account"
    assert delete_log["new_values"] == {}


@pytest.mark.asyncio
async def test_log_order_submitted(audit_service):
    """Test logging order submission."""
    log_id = await audit_service.log_order_submitted(
        order_id="order-123",
        symbol="AAPL",
        side="buy",
        quantity=100.0,
        order_type="limit",
        price=150.00,
        strategy_id="momentum-v1",
        user_id="trader-1",
    )

    assert log_id is not None

    logs = await audit_service.get_entity_history("order", "order-123")
    assert len(logs) == 1

    log = logs[0]
    assert log["action"] == "order_submitted_to_broker"
    assert log["metadata"]["symbol"] == "AAPL"
    assert log["metadata"]["price"] == 150.00


@pytest.mark.asyncio
async def test_log_order_filled(audit_service):
    """Test logging order fill."""
    log_id = await audit_service.log_order_filled(
        order_id="order-456",
        symbol="TSLA",
        quantity=50.0,
        fill_price=800.00,
        commission=5.00,
        strategy_id="breakout-v2",
    )

    assert log_id is not None

    logs = await audit_service.get_entity_history("order", "order-456")
    assert len(logs) == 1

    log = logs[0]
    assert log["action"] == "order_filled"
    assert log["metadata"]["notional_value"] == 40000.0  # 50 * 800
    assert log["metadata"]["commission"] == 5.00


@pytest.mark.asyncio
async def test_log_position_lifecycle(audit_service):
    """Test logging complete position lifecycle."""
    position_id = "position-789"

    # Log position opening
    open_log_id = await audit_service.log_position_opened(
        position_id=position_id,
        symbol="NVDA",
        position_type="long",
        quantity=25.0,
        entry_price=900.00,
        cost_basis=22500.00,
        strategy_id="ai-growth",
    )

    # Log position closing
    close_log_id = await audit_service.log_position_closed(
        position_id=position_id,
        symbol="NVDA",
        exit_price=950.00,
        realized_pnl=1250.00,  # (950-900) * 25
        holding_period_days=15.5,
        strategy_id="ai-growth",
    )

    assert open_log_id is not None
    assert close_log_id is not None

    # Verify logs
    logs = await audit_service.get_entity_history("position", position_id)
    assert len(logs) == 2

    open_log = next(log for log in logs if log["action"] == "position_opened")
    close_log = next(log for log in logs if log["action"] == "position_closed")

    assert open_log["metadata"]["cost_basis"] == 22500.0
    assert close_log["metadata"]["realized_pnl"] == 1250.0
    assert close_log["metadata"]["holding_period_days"] == 15.5


@pytest.mark.asyncio
async def test_log_system_events(audit_service):
    """Test logging system events."""
    # Log system startup
    startup_log_id = await audit_service.log_system_startup(
        version="1.0.0",
        config={"debug": False, "max_positions": 10},
        user_id="system-admin",
    )

    # Log system shutdown
    shutdown_log_id = await audit_service.log_system_shutdown(
        reason="planned_maintenance", user_id="system-admin"
    )

    assert startup_log_id is not None
    assert shutdown_log_id is not None

    # Verify logs
    system_logs = await audit_service.get_entity_history("system", "voyager_trader")
    assert len(system_logs) == 2

    startup_log = next(log for log in system_logs if log["action"] == "system_startup")
    shutdown_log = next(
        log for log in system_logs if log["action"] == "system_shutdown"
    )

    assert startup_log["metadata"]["system_version"] == "1.0.0"
    assert shutdown_log["metadata"]["shutdown_reason"] == "planned_maintenance"


@pytest.mark.asyncio
async def test_log_strategy_lifecycle(audit_service):
    """Test logging strategy lifecycle events."""
    strategy_id = "mean-reversion-v3"

    # Log strategy start
    start_log_id = await audit_service.log_strategy_started(
        strategy_id=strategy_id,
        strategy_name="Mean Reversion V3",
        parameters={"lookback": 20, "threshold": 2.0, "max_positions": 5},
        user_id="trader-1",
    )

    # Log strategy stop
    stop_log_id = await audit_service.log_strategy_stopped(
        strategy_id=strategy_id,
        reason="user_requested",
        performance_metrics={"total_pnl": 5000.0, "win_rate": 0.65, "trades": 50},
        user_id="trader-1",
    )

    assert start_log_id is not None
    assert stop_log_id is not None

    # Verify logs
    strategy_logs = await audit_service.get_entity_history("strategy", strategy_id)
    assert len(strategy_logs) == 2

    start_log = next(
        log for log in strategy_logs if log["action"] == "strategy_started"
    )
    stop_log = next(log for log in strategy_logs if log["action"] == "strategy_stopped")

    assert start_log["metadata"]["parameters"]["lookback"] == 20
    assert stop_log["metadata"]["performance_metrics"]["total_pnl"] == 5000.0


@pytest.mark.asyncio
async def test_log_error(audit_service):
    """Test logging errors."""
    log_id = await audit_service.log_error(
        error_type="ConnectionError",
        error_message="Failed to connect to broker API",
        entity_type="broker",
        entity_id="alpaca-api",
        stack_trace="Traceback (most recent call last):\n...",
        strategy_id="momentum-v1",
    )

    assert log_id is not None

    # Verify error log
    logs = await audit_service.get_entity_history("broker", "alpaca-api")
    assert len(logs) == 1

    log = logs[0]
    assert log["action"] == "error_occurred"
    assert log["metadata"]["error_type"] == "ConnectionError"
    assert log["strategy_id"] == "momentum-v1"


@pytest.mark.asyncio
async def test_log_risk_events(audit_service):
    """Test logging risk management events."""
    # Log risk limit breach
    breach_log_id = await audit_service.log_risk_limit_breached(
        limit_type="max_drawdown",
        limit_value=5.0,
        current_value=7.2,
        entity_type="portfolio",
        entity_id="portfolio-123",
        action_taken="suspend_trading",
        strategy_id="high-freq-v1",
    )

    # Log circuit breaker
    breaker_log_id = await audit_service.log_circuit_breaker_triggered(
        breaker_type="loss_limit",
        trigger_condition="daily_loss > $10000",
        entity_type="account",
        entity_id="account-456",
        strategy_id="scalping-v2",
    )

    assert breach_log_id is not None
    assert breaker_log_id is not None

    # Verify risk logs
    portfolio_logs = await audit_service.get_entity_history(
        "portfolio", "portfolio-123"
    )
    account_logs = await audit_service.get_entity_history("account", "account-456")

    assert len(portfolio_logs) == 1
    assert len(account_logs) == 1

    breach_log = portfolio_logs[0]
    breaker_log = account_logs[0]

    assert breach_log["action"] == "risk_limit_breached"
    assert breach_log["metadata"]["action_taken"] == "suspend_trading"

    assert breaker_log["action"] == "circuit_breaker_triggered"
    assert breaker_log["metadata"]["breaker_type"] == "loss_limit"


@pytest.mark.asyncio
async def test_log_user_sessions(audit_service):
    """Test logging user session events."""
    user_id = "trader-123"
    session_id = "session-abc-789"

    # Log user login
    login_log_id = await audit_service.log_user_login(
        user_id=user_id,
        ip_address="192.168.1.100",
        user_agent="Mozilla/5.0...",
        session_id=session_id,
    )

    # Log user logout
    logout_log_id = await audit_service.log_user_logout(
        user_id=user_id, session_duration_minutes=125.5, session_id=session_id
    )

    assert login_log_id is not None
    assert logout_log_id is not None

    # Verify session logs
    user_logs = await audit_service.get_entity_history("user", user_id)
    assert len(user_logs) == 2

    login_log = next(log for log in user_logs if log["action"] == "user_login")
    logout_log = next(log for log in user_logs if log["action"] == "user_logout")

    assert login_log["metadata"]["ip_address"] == "192.168.1.100"
    assert logout_log["metadata"]["session_duration_minutes"] == 125.5


@pytest.mark.asyncio
async def test_log_configuration_changes(audit_service):
    """Test logging configuration changes."""
    log_id = await audit_service.log_configuration_changed(
        config_key="max_position_size",
        old_value="5.0",
        new_value="7.5",
        user_id="admin-user",
    )

    assert log_id is not None

    # Verify configuration log
    config_logs = await audit_service.get_entity_history(
        "configuration", "max_position_size"
    )
    assert len(config_logs) == 1

    log = config_logs[0]
    assert log["action"] == "configuration_changed"
    assert log["old_values"]["value"] == "5.0"
    assert log["new_values"]["value"] == "7.5"
    assert log["user_id"] == "admin-user"


@pytest.mark.asyncio
async def test_get_audit_trail_filtering(audit_service):
    """Test audit trail filtering and querying."""
    # Create various audit logs
    await audit_service.log_system_startup("1.0.0", {}, user_id="admin")
    await audit_service.log_user_login("user-1", ip_address="10.0.0.1")
    await audit_service.log_user_login("user-2", ip_address="10.0.0.2")
    await audit_service.log_order_submitted("order-1", "AAPL", "buy", 100, "market")
    await audit_service.log_order_submitted(
        "order-2", "GOOGL", "sell", 50, "limit", price=2800.0
    )

    # Test filtering by action
    login_logs = await audit_service.get_audit_trail(action="user_login", limit=10)
    assert len(login_logs) == 2

    # Test filtering by entity
    system_logs = await audit_service.get_audit_trail(
        entity_type="system", entity_id="voyager_trader"
    )
    assert len(system_logs) == 1
    assert system_logs[0]["action"] == "system_startup"

    # Test with date range (would need more sophisticated implementation)
    start_date = datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    end_date = datetime.now(timezone.utc).replace(
        hour=23, minute=59, second=59, microsecond=999999
    )

    recent_logs = await audit_service.get_audit_trail(
        start_date=start_date, end_date=end_date, limit=100
    )
    # Should return recent logs (implementation dependent)
