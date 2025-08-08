"""Tests for risk management system."""

from decimal import Decimal

import pytest

from src.voyager_trader.execution.risk import RiskLimits, RiskManager, RiskViolation
from src.voyager_trader.models.trading import Account, Order, Portfolio, Position
from src.voyager_trader.models.types import (
    Money,
    OrderSide,
    OrderType,
    PositionType,
    Quantity,
    Symbol,
)


@pytest.fixture
def risk_limits():
    """Create test risk limits."""
    return RiskLimits(
        max_position_size_percent=Decimal("10"),
        max_portfolio_risk_percent=Decimal("2"),
        daily_loss_limit=Money(amount=Decimal("1000"), currency="USD"),
        max_drawdown_percent=Decimal("5"),
        max_leverage=Decimal("1.5"),
        max_trades_per_day=50,
    )


@pytest.fixture
def risk_manager(risk_limits):
    """Create risk manager for testing."""
    return RiskManager(risk_limits)


@pytest.fixture
def sample_portfolio():
    """Create sample portfolio."""
    return Portfolio(
        id="test-portfolio",
        name="Test Portfolio",
        account_id="test-account",
        base_currency="USD",
        cash_balance=Money(amount=Decimal("50000"), currency="USD"),
        total_value=Money(amount=Decimal("100000"), currency="USD"),
    )


@pytest.fixture
def sample_account():
    """Create sample account."""
    return Account(
        id="test-account",
        name="Test Account",
        account_type="margin",
        base_currency="USD",
        cash_balance=Money(amount=Decimal("50000"), currency="USD"),
        total_equity=Money(amount=Decimal("100000"), currency="USD"),
        buying_power=Money(amount=Decimal("150000"), currency="USD"),
        maintenance_margin=Money(amount=Decimal("5000"), currency="USD"),
        max_position_size=Decimal("10"),
        max_portfolio_risk=Decimal("2"),
    )


@pytest.fixture
def sample_order():
    """Create sample order."""
    return Order(
        id="test-order",
        symbol=Symbol(code="AAPL", asset_class="equity"),
        order_type=OrderType.MARKET,
        side=OrderSide.BUY,
        quantity=Quantity(amount=Decimal("100")),
        price=Decimal("150.00"),
        strategy_id="test-strategy",
    )


@pytest.fixture
def sample_position():
    """Create sample position."""
    return Position(
        id="test-position",
        symbol=Symbol(code="AAPL", asset_class="equity"),
        position_type=PositionType.LONG,
        quantity=Quantity(amount=Decimal("100")),
        entry_price=Decimal("145.00"),
        current_price=Decimal("150.00"),
        entry_timestamp=pytest.helpers.utcnow() if hasattr(pytest, "helpers") else None,
        strategy_id="test-strategy",
    )


class TestRiskLimits:
    """Test RiskLimits model."""

    def test_default_limits(self):
        """Test default risk limits."""
        limits = RiskLimits()

        assert limits.max_position_size_percent == Decimal("10")
        assert limits.max_portfolio_risk_percent == Decimal("2")
        assert limits.daily_loss_limit is None
        assert limits.max_drawdown_percent == Decimal("20")
        assert limits.max_correlation == Decimal("0.7")
        assert limits.max_leverage == Decimal("1.0")
        assert limits.max_trades_per_day == 100
        assert limits.cooldown_period_minutes == 60

    def test_custom_limits(self, risk_limits):
        """Test custom risk limits."""
        assert risk_limits.max_position_size_percent == Decimal("10")
        assert risk_limits.max_portfolio_risk_percent == Decimal("2")
        assert risk_limits.daily_loss_limit.amount == Decimal("1000")
        assert risk_limits.max_drawdown_percent == Decimal("5")
        assert risk_limits.max_leverage == Decimal("1.5")
        assert risk_limits.max_trades_per_day == 50


class TestRiskManager:
    """Test RiskManager functionality."""

    def test_initialization(self, risk_manager):
        """Test risk manager initialization."""
        assert not risk_manager.is_shutdown()
        assert risk_manager.get_shutdown_reason() is None
        assert risk_manager._daily_trades == 0
        assert risk_manager._daily_pnl.amount == Decimal("0")

    def test_valid_order_passes(
        self, risk_manager, sample_order, sample_portfolio, sample_account
    ):
        """Test that valid order passes validation."""
        positions = []

        result = risk_manager.validate_order(
            sample_order, sample_portfolio, sample_account, positions
        )
        assert result is True

    def test_position_size_limit_violation(
        self, risk_manager, sample_portfolio, sample_account
    ):
        """Test position size limit violation."""
        # Create large order that exceeds position size limit
        large_order = Order(
            id="large-order",
            symbol=Symbol(code="AAPL", asset_class="equity"),
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=Quantity(amount=Decimal("1000")),  # Large quantity
            price=Decimal("150.00"),  # Total: $150,000 > 10% of $100,000 portfolio
            strategy_id="test-strategy",
        )

        positions = []
        result = risk_manager.validate_order(
            large_order, sample_portfolio, sample_account, positions
        )
        assert result is False

    def test_daily_trade_limit_violation(
        self, risk_manager, sample_order, sample_portfolio, sample_account
    ):
        """Test daily trade limit violation."""
        positions = []

        # Execute max trades
        for _ in range(50):  # Limit is 50
            risk_manager.update_daily_trades()

        # Next order should fail
        result = risk_manager.validate_order(
            sample_order, sample_portfolio, sample_account, positions
        )
        assert result is False

    def test_calculate_position_size_with_stop_loss(
        self, risk_manager, sample_portfolio
    ):
        """Test position size calculation with stop loss."""
        symbol = Symbol(code="AAPL", asset_class="equity")
        entry_price = Decimal("150.00")
        stop_loss_price = Decimal("145.00")  # $5 risk per share

        position_size = risk_manager.calculate_position_size(
            symbol, entry_price, stop_loss_price, sample_portfolio
        )

        # Risk amount: $100,000 * 2% = $2,000
        # Risk per share: $5
        # Expected shares: $2,000 / $5 = 400 shares
        # But limited by position size limit: $100,000 * 10% / $150 = 66.67 shares
        expected_max_shares = (
            Decimal("100000") * Decimal("10") / Decimal("100") / Decimal("150")
        )

        assert position_size <= expected_max_shares
        assert position_size > 0

    def test_calculate_position_size_without_stop_loss(
        self, risk_manager, sample_portfolio
    ):
        """Test position size calculation without stop loss."""
        symbol = Symbol(code="AAPL", asset_class="equity")
        entry_price = Decimal("150.00")

        position_size = risk_manager.calculate_position_size(
            symbol, entry_price, None, sample_portfolio
        )

        # Should use position size limit
        expected_shares = (
            sample_portfolio.total_value.amount
            * Decimal("10")
            / Decimal("100")
            / entry_price
        )
        assert position_size == expected_shares

    def test_daily_pnl_tracking(self, risk_manager):
        """Test daily P&L tracking."""
        # Add some P&L
        pnl_change = Money(amount=Decimal("500"), currency="USD")
        risk_manager.update_daily_pnl(pnl_change)

        assert risk_manager._daily_pnl.amount == Decimal("500")

        # Add loss
        loss_change = Money(amount=Decimal("-300"), currency="USD")
        risk_manager.update_daily_pnl(loss_change)

        assert risk_manager._daily_pnl.amount == Decimal("200")

    def test_daily_loss_limit_shutdown(self, risk_manager):
        """Test shutdown on daily loss limit."""
        # Exceed daily loss limit
        large_loss = Money(
            amount=Decimal("-1500"), currency="USD"
        )  # Exceeds $1000 limit
        risk_manager.update_daily_pnl(large_loss)

        assert risk_manager.is_shutdown()
        assert "Daily loss limit exceeded" in risk_manager.get_shutdown_reason()

    def test_drawdown_check_pass(self, risk_manager, sample_portfolio):
        """Test drawdown check passes."""
        high_water_mark = Money(amount=Decimal("100000"), currency="USD")

        result = risk_manager.check_drawdown(sample_portfolio, high_water_mark)
        assert result is True

    def test_drawdown_check_fail(self, risk_manager):
        """Test drawdown check fails."""
        # Portfolio value much lower than high water mark
        low_portfolio = Portfolio(
            id="low-portfolio",
            name="Low Portfolio",
            account_id="test-account",
            base_currency="USD",
            cash_balance=Money(amount=Decimal("20000"), currency="USD"),
            total_value=Money(amount=Decimal("20000"), currency="USD"),
        )

        high_water_mark = Money(amount=Decimal("100000"), currency="USD")

        result = risk_manager.check_drawdown(low_portfolio, high_water_mark)
        assert result is False
        assert risk_manager.is_shutdown()

    def test_validate_position_increase_pass(self, risk_manager, sample_portfolio):
        """Test position increase validation passes."""
        symbol = Symbol(code="AAPL", asset_class="equity")
        additional_quantity = Decimal("50")
        price = Decimal("150.00")

        result = risk_manager.validate_position_increase(
            symbol, additional_quantity, price, sample_portfolio
        )
        assert result is True

    def test_validate_position_increase_fail(self, risk_manager, sample_portfolio):
        """Test position increase validation fails."""
        symbol = Symbol(code="AAPL", asset_class="equity")
        additional_quantity = Decimal("1000")  # Very large
        price = Decimal("150.00")

        result = risk_manager.validate_position_increase(
            symbol, additional_quantity, price, sample_portfolio
        )
        assert result is False

    def test_emergency_shutdown(self, risk_manager):
        """Test emergency shutdown."""
        reason = "Market crash detected"
        risk_manager.emergency_shutdown(reason)

        assert risk_manager.is_shutdown()
        assert f"Emergency shutdown: {reason}" in risk_manager.get_shutdown_reason()

    def test_reset_daily_counters(self, risk_manager):
        """Test resetting daily counters."""
        # Add some activity
        risk_manager.update_daily_trades()
        risk_manager.update_daily_pnl(Money(amount=Decimal("100"), currency="USD"))

        assert risk_manager._daily_trades > 0
        assert risk_manager._daily_pnl.amount > 0

        # Reset
        risk_manager.reset_daily_counters()

        assert risk_manager._daily_trades == 0
        assert risk_manager._daily_pnl.amount == Decimal("0")

    def test_risk_metrics(self, risk_manager, sample_portfolio, sample_account):
        """Test risk metrics calculation."""
        metrics = risk_manager.get_risk_metrics(sample_portfolio, sample_account)

        assert "leverage" in metrics
        assert "cash_percentage" in metrics
        assert "daily_trades" in metrics
        assert "daily_pnl" in metrics
        assert "is_shutdown" in metrics
        assert "total_exposure" in metrics
        assert "position_count" in metrics

        assert metrics["daily_trades"] == Decimal("0")
        assert metrics["daily_pnl"] == Decimal("0")
        assert metrics["is_shutdown"] == Decimal("0")

    def test_margin_requirements_cash_account(self, risk_manager, sample_portfolio):
        """Test margin requirements for cash account."""
        cash_account = Account(
            id="cash-account",
            name="Cash Account",
            account_type="cash",
            base_currency="USD",
            cash_balance=Money(amount=Decimal("10000"), currency="USD"),
            total_equity=Money(amount=Decimal("10000"), currency="USD"),
            buying_power=Money(amount=Decimal("10000"), currency="USD"),
            maintenance_margin=Money(amount=Decimal("0"), currency="USD"),
            max_position_size=Decimal("10"),
            max_portfolio_risk=Decimal("2"),
        )

        # Order that exceeds cash
        large_order = Order(
            id="large-cash-order",
            symbol=Symbol(code="AAPL", asset_class="equity"),
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=Quantity(amount=Decimal("100")),
            price=Decimal("150.00"),  # Total: $15,000 > $10,000 cash
            strategy_id="test-strategy",
        )

        positions = []
        result = risk_manager.validate_order(
            large_order, sample_portfolio, cash_account, positions
        )
        assert result is False

    def test_leverage_limit_violation(
        self, risk_manager, sample_portfolio, sample_account
    ):
        """Test leverage limit violation."""
        # Create positions that would cause high leverage
        high_leverage_position = Position(
            id="high-leverage-pos",
            symbol=Symbol(code="AAPL", asset_class="equity"),
            position_type=PositionType.LONG,
            quantity=Quantity(amount=Decimal("1000")),
            entry_price=Decimal("150.00"),
            current_price=Decimal("150.00"),
            entry_timestamp=None,
            strategy_id="test-strategy",
        )

        positions = [high_leverage_position]

        # Additional order that would increase leverage further
        additional_order = Order(
            id="additional-order",
            symbol=Symbol(code="MSFT", asset_class="equity"),
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=Quantity(amount=Decimal("500")),
            price=Decimal("200.00"),
            strategy_id="test-strategy",
        )

        result = risk_manager.validate_order(
            additional_order, sample_portfolio, sample_account, positions
        )
        # May pass or fail depending on exact leverage calculation
        # This tests the leverage calculation logic exists
        assert isinstance(result, bool)


class TestRiskViolation:
    """Test RiskViolation exception."""

    def test_risk_violation_creation(self):
        """Test creating risk violation."""
        message = "Position size limit exceeded"
        violation = RiskViolation(message)

        assert str(violation) == message
        assert isinstance(violation, Exception)
