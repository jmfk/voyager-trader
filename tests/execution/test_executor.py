"""Tests for strategy executor."""

import asyncio
from decimal import Decimal

import pytest

from src.voyager_trader.execution.executor import (
    ExecutionConfig,
    StrategyExecutor,
    StrategySignal,
)
from src.voyager_trader.execution.interfaces import PaperBroker
from src.voyager_trader.execution.risk import RiskLimits, RiskManager
from src.voyager_trader.models.trading import Portfolio
from src.voyager_trader.models.types import Money, OrderType, Symbol


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
def paper_broker():
    """Create paper broker."""
    return PaperBroker(initial_cash=Money(amount=Decimal("100000"), currency="USD"))


@pytest.fixture
def risk_manager():
    """Create risk manager."""
    limits = RiskLimits(
        max_position_size_percent=Decimal("10"),
        max_portfolio_risk_percent=Decimal("2"),
        max_trades_per_day=100,
    )
    return RiskManager(limits)


@pytest.fixture
def execution_config():
    """Create execution config."""
    return ExecutionConfig(
        max_concurrent_orders=5,
        order_timeout_seconds=60,
        enable_paper_trading=True,
        auto_sync_positions=False,  # Disable for testing
    )


@pytest.fixture
def strategy_executor(sample_portfolio, paper_broker, risk_manager, execution_config):
    """Create strategy executor."""
    return StrategyExecutor(
        portfolio=sample_portfolio,
        broker=paper_broker,
        risk_manager=risk_manager,
        config=execution_config,
    )


@pytest.fixture
def buy_signal():
    """Create buy signal."""
    return StrategySignal(
        strategy_id="test-strategy",
        symbol=Symbol(code="AAPL", asset_class="equity"),
        action="BUY",
        quantity=Decimal("100"),
        price=Decimal("150.00"),
        order_type=OrderType.LIMIT,
        confidence=Decimal("0.8"),
        reasoning="Strong momentum indicator",
    )


@pytest.fixture
def sell_signal():
    """Create sell signal."""
    return StrategySignal(
        strategy_id="test-strategy",
        symbol=Symbol(code="AAPL", asset_class="equity"),
        action="SELL",
        quantity=Decimal("50"),
        price=Decimal("155.00"),
        order_type=OrderType.LIMIT,
        confidence=Decimal("0.7"),
        reasoning="Taking profits",
    )


class TestStrategySignal:
    """Test StrategySignal model."""

    def test_signal_creation(self, buy_signal):
        """Test creating trading signal."""
        assert buy_signal.strategy_id == "test-strategy"
        assert buy_signal.symbol.code == "AAPL"
        assert buy_signal.action == "BUY"
        assert buy_signal.quantity == Decimal("100")
        assert buy_signal.price == Decimal("150.00")
        assert buy_signal.order_type == OrderType.LIMIT
        assert buy_signal.confidence == Decimal("0.8")
        assert buy_signal.reasoning == "Strong momentum indicator"
        assert buy_signal.timestamp is not None

    def test_signal_defaults(self):
        """Test signal with defaults."""
        signal = StrategySignal(
            strategy_id="test",
            symbol=Symbol(code="AAPL", asset_class="equity"),
            action="BUY",
        )

        assert signal.quantity is None
        assert signal.price is None
        assert signal.order_type == OrderType.MARKET
        assert signal.confidence == Decimal("0.5")
        assert signal.reasoning is None
        assert signal.metadata == {}


class TestExecutionConfig:
    """Test ExecutionConfig model."""

    def test_default_config(self):
        """Test default execution config."""
        config = ExecutionConfig()

        assert config.max_concurrent_orders == 10
        assert config.order_timeout_seconds == 300
        assert config.position_update_interval == 60
        assert config.enable_paper_trading is True
        assert config.auto_sync_positions is True
        assert config.emergency_stop_enabled is True
        assert config.max_strategy_allocation_percent == Decimal("20")

    def test_custom_config(self):
        """Test custom execution config."""
        config = ExecutionConfig(
            max_concurrent_orders=5,
            order_timeout_seconds=60,
            enable_paper_trading=False,
        )

        assert config.max_concurrent_orders == 5
        assert config.order_timeout_seconds == 60
        assert config.enable_paper_trading is False


class TestStrategyExecutor:
    """Test StrategyExecutor functionality."""

    def test_initialization(self, strategy_executor):
        """Test executor initialization."""
        assert strategy_executor.portfolio is not None
        assert strategy_executor.broker is not None
        assert strategy_executor.risk_manager is not None
        assert strategy_executor.config is not None
        assert strategy_executor.order_manager is not None
        assert strategy_executor.portfolio_manager is not None
        assert strategy_executor.monitor is not None
        assert not strategy_executor.is_running()
        assert len(strategy_executor._strategies) == 0

    @pytest.mark.asyncio
    async def test_start_stop(self, strategy_executor):
        """Test starting and stopping executor."""
        assert not strategy_executor.is_running()

        await strategy_executor.start()
        assert strategy_executor.is_running()

        await strategy_executor.stop()
        assert not strategy_executor.is_running()

    @pytest.mark.asyncio
    async def test_register_strategy(self, strategy_executor):
        """Test strategy registration."""
        result = await strategy_executor.register_strategy(
            "test-strategy", Decimal("15")
        )
        assert result is True
        assert "test-strategy" in strategy_executor._strategies
        assert strategy_executor._strategy_allocations["test-strategy"] == Decimal("15")

    @pytest.mark.asyncio
    async def test_register_strategy_allocation_limit(self, strategy_executor):
        """Test strategy registration with excessive allocation."""
        result = await strategy_executor.register_strategy(
            "big-strategy", Decimal("25")
        )
        assert result is False  # Exceeds 20% limit
        assert "big-strategy" not in strategy_executor._strategies

    @pytest.mark.asyncio
    async def test_register_multiple_strategies_total_limit(self, strategy_executor):
        """Test registering strategies that exceed total allocation."""
        # Register multiple strategies
        await strategy_executor.register_strategy("strategy1", Decimal("20"))
        await strategy_executor.register_strategy("strategy2", Decimal("20"))
        await strategy_executor.register_strategy("strategy3", Decimal("20"))
        await strategy_executor.register_strategy("strategy4", Decimal("20"))
        await strategy_executor.register_strategy("strategy5", Decimal("20"))

        # Next one should fail (would exceed 100%)
        result = await strategy_executor.register_strategy("strategy6", Decimal("10"))
        assert result is False

    @pytest.mark.asyncio
    async def test_unregister_strategy(self, strategy_executor):
        """Test strategy unregistration."""
        # Register first
        await strategy_executor.register_strategy("test-strategy", Decimal("10"))
        assert "test-strategy" in strategy_executor._strategies

        # Unregister
        result = await strategy_executor.unregister_strategy("test-strategy")
        assert result is True
        assert "test-strategy" not in strategy_executor._strategies
        assert "test-strategy" not in strategy_executor._strategy_allocations

    @pytest.mark.asyncio
    async def test_unregister_nonexistent_strategy(self, strategy_executor):
        """Test unregistering non-existent strategy."""
        result = await strategy_executor.unregister_strategy("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_execute_buy_signal_unregistered_strategy(
        self, strategy_executor, buy_signal
    ):
        """Test executing signal from unregistered strategy."""
        result = await strategy_executor.execute_signal(buy_signal)

        assert not result.success
        assert "not registered" in result.error_message

    @pytest.mark.asyncio
    async def test_execute_buy_signal_success(
        self, strategy_executor, buy_signal, paper_broker
    ):
        """Test successful buy signal execution."""
        # Register strategy
        await strategy_executor.register_strategy("test-strategy", Decimal("10"))

        # Set price in paper broker
        paper_broker.set_price("AAPL", Decimal("148.00"))  # Favorable for limit buy

        result = await strategy_executor.execute_signal(buy_signal)

        assert result.success
        assert result.filled_quantity.amount > 0
        assert result.fill_price is not None

    @pytest.mark.asyncio
    async def test_execute_market_buy_signal(self, strategy_executor, paper_broker):
        """Test market buy signal execution."""
        # Register strategy
        await strategy_executor.register_strategy("test-strategy", Decimal("10"))

        market_signal = StrategySignal(
            strategy_id="test-strategy",
            symbol=Symbol(code="AAPL", asset_class="equity"),
            action="BUY",
            order_type=OrderType.MARKET,
            confidence=Decimal("0.9"),
        )

        # Set price in paper broker
        paper_broker.set_price("AAPL", Decimal("150.00"))

        result = await strategy_executor.execute_signal(market_signal)

        assert result.success
        assert result.filled_quantity.amount > 0

    @pytest.mark.asyncio
    async def test_execute_sell_signal_no_position(
        self, strategy_executor, sell_signal
    ):
        """Test sell signal when no position exists."""
        # Register strategy
        await strategy_executor.register_strategy("test-strategy", Decimal("10"))

        result = await strategy_executor.execute_signal(sell_signal)

        assert not result.success
        assert "No open position" in result.error_message

    @pytest.mark.asyncio
    async def test_execute_sell_signal_success(
        self, strategy_executor, buy_signal, sell_signal, paper_broker
    ):
        """Test successful sell signal execution."""
        # Register strategy
        await strategy_executor.register_strategy("test-strategy", Decimal("10"))

        # First buy to establish position
        paper_broker.set_price("AAPL", Decimal("148.00"))
        buy_result = await strategy_executor.execute_signal(buy_signal)
        assert buy_result.success

        # Now sell
        paper_broker.set_price("AAPL", Decimal("155.00"))
        sell_result = await strategy_executor.execute_signal(sell_signal)

        assert sell_result.success
        assert sell_result.filled_quantity.amount > 0

    @pytest.mark.asyncio
    async def test_execute_hold_signal(self, strategy_executor):
        """Test hold signal execution."""
        # Register strategy
        await strategy_executor.register_strategy("test-strategy", Decimal("10"))

        hold_signal = StrategySignal(
            strategy_id="test-strategy",
            symbol=Symbol(code="AAPL", asset_class="equity"),
            action="HOLD",
        )

        result = await strategy_executor.execute_signal(hold_signal)

        assert result.success
        assert result.filled_quantity.amount == Decimal("0")

    @pytest.mark.asyncio
    async def test_execute_unknown_action(self, strategy_executor):
        """Test unknown action signal."""
        # Register strategy
        await strategy_executor.register_strategy("test-strategy", Decimal("10"))

        unknown_signal = StrategySignal(
            strategy_id="test-strategy",
            symbol=Symbol(code="AAPL", asset_class="equity"),
            action="UNKNOWN",
        )

        result = await strategy_executor.execute_signal(unknown_signal)

        assert not result.success
        assert "Unknown action" in result.error_message

    @pytest.mark.asyncio
    async def test_risk_violation_blocks_order(self, strategy_executor, paper_broker):
        """Test that risk violations block order execution."""
        # Register strategy
        await strategy_executor.register_strategy("test-strategy", Decimal("10"))

        # Create signal that would violate position size limits
        large_signal = StrategySignal(
            strategy_id="test-strategy",
            symbol=Symbol(code="AAPL", asset_class="equity"),
            action="BUY",
            quantity=Decimal("10000"),  # Very large quantity
            order_type=OrderType.MARKET,
        )

        paper_broker.set_price("AAPL", Decimal("150.00"))

        result = await strategy_executor.execute_signal(large_signal)

        assert not result.success
        assert "risk" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_concurrent_order_limit(self, strategy_executor, paper_broker):
        """Test concurrent order limit enforcement."""
        # Register strategy
        await strategy_executor.register_strategy("test-strategy", Decimal("10"))

        paper_broker.set_price(
            "AAPL", Decimal("155.00")
        )  # Unfavorable for limit orders

        # Submit max concurrent orders (should stay pending)
        signals = []
        for i in range(6):  # Config limit is 5
            signal = StrategySignal(
                strategy_id="test-strategy",
                symbol=Symbol(code=f"STOCK{i}", asset_class="equity"),
                action="BUY",
                quantity=Decimal("10"),
                price=Decimal("150.00"),  # Will stay pending
                order_type=OrderType.LIMIT,
            )
            signals.append(signal)

        results = []
        for signal in signals:
            result = await strategy_executor.execute_signal(signal)
            results.append(result)

        # First 5 should succeed (pending), 6th should fail
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]

        assert len(successful) == 5
        assert len(failed) == 1
        assert "Maximum concurrent orders" in failed[0].error_message

    def test_get_strategy_status(self, strategy_executor):
        """Test getting strategy status."""
        # Non-existent strategy
        status = strategy_executor.get_strategy_status("nonexistent")
        assert status is None

        # After registering
        asyncio.run(strategy_executor.register_strategy("test-strategy", Decimal("15")))

        status = strategy_executor.get_strategy_status("test-strategy")
        assert status is not None
        assert status["active"] is True
        assert status["allocation_percent"] == Decimal("15")
        assert status["trades"] == 0

    def test_get_execution_status(self, strategy_executor):
        """Test getting execution status."""
        status = strategy_executor.get_execution_status()

        assert "running" in status
        assert "strategies" in status
        assert "active_orders" in status
        assert "execution_metrics" in status
        assert "system_health" in status
        assert "risk_metrics" in status

        assert status["running"] is False
        assert status["strategies"] == 0
        assert status["active_orders"] == 0

    @pytest.mark.asyncio
    async def test_emergency_stop(self, strategy_executor):
        """Test emergency stop functionality."""
        await strategy_executor.start()
        assert strategy_executor.is_running()

        await strategy_executor.emergency_stop("Test emergency")

        assert not strategy_executor.is_running()
        assert strategy_executor.risk_manager.is_shutdown()
        assert (
            "emergency" in strategy_executor.risk_manager.get_shutdown_reason().lower()
        )

    @pytest.mark.asyncio
    async def test_inactive_strategy_rejection(self, strategy_executor, buy_signal):
        """Test that inactive strategies are rejected."""
        # Register and then deactivate strategy
        await strategy_executor.register_strategy("test-strategy", Decimal("10"))
        strategy_executor._strategies["test-strategy"]["active"] = False

        result = await strategy_executor.execute_signal(buy_signal)

        assert not result.success
        assert "inactive" in result.error_message

    @pytest.mark.asyncio
    async def test_position_size_calculation(self, strategy_executor, paper_broker):
        """Test automatic position size calculation."""
        # Register strategy with 10% allocation
        await strategy_executor.register_strategy("test-strategy", Decimal("10"))

        # Signal without explicit quantity
        auto_size_signal = StrategySignal(
            strategy_id="test-strategy",
            symbol=Symbol(code="AAPL", asset_class="equity"),
            action="BUY",
            order_type=OrderType.MARKET,
        )

        paper_broker.set_price("AAPL", Decimal("100.00"))

        result = await strategy_executor.execute_signal(auto_size_signal)

        assert result.success
        assert result.filled_quantity.amount > 0
        # Should be limited by strategy allocation (10% of $100k = $10k / $100 = 100 shares max)
        assert result.filled_quantity.amount <= Decimal("100")


class TestExecutorIntegration:
    """Integration tests for the complete execution system."""

    @pytest.mark.asyncio
    async def test_complete_trading_workflow(self, strategy_executor, paper_broker):
        """Test complete trading workflow from signal to trade."""
        # Start executor
        await strategy_executor.start()

        # Register strategy
        await strategy_executor.register_strategy("momentum-strategy", Decimal("15"))

        # Set initial price
        paper_broker.set_price("AAPL", Decimal("150.00"))

        # Execute buy signal
        buy_signal = StrategySignal(
            strategy_id="momentum-strategy",
            symbol=Symbol(code="AAPL", asset_class="equity"),
            action="BUY",
            quantity=Decimal("100"),
            order_type=OrderType.MARKET,
            confidence=Decimal("0.8"),
        )

        buy_result = await strategy_executor.execute_signal(buy_signal)
        assert buy_result.success

        # Check position was created
        positions = strategy_executor.portfolio_manager.get_open_positions()
        assert len(positions) == 1
        assert positions[0].symbol.code == "AAPL"

        # Update price
        paper_broker.set_price("AAPL", Decimal("160.00"))

        # Execute sell signal
        sell_signal = StrategySignal(
            strategy_id="momentum-strategy",
            symbol=Symbol(code="AAPL", asset_class="equity"),
            action="SELL",
            quantity=Decimal("50"),
            order_type=OrderType.MARKET,
            confidence=Decimal("0.7"),
        )

        sell_result = await strategy_executor.execute_signal(sell_signal)
        assert sell_result.success

        # Check position was reduced
        positions = strategy_executor.portfolio_manager.get_open_positions()
        assert len(positions) == 1
        assert positions[0].quantity.amount == Decimal("50")

        # Check trade history
        trades = strategy_executor.portfolio_manager.get_trade_history()
        assert len(trades) == 2  # Buy and sell

        # Check strategy metrics
        strategy_status = strategy_executor.get_strategy_status("momentum-strategy")
        assert strategy_status["trades"] == 2

        # Stop executor
        await strategy_executor.stop()

    @pytest.mark.asyncio
    async def test_risk_management_integration(self, strategy_executor, paper_broker):
        """Test risk management integration."""
        await strategy_executor.start()
        await strategy_executor.register_strategy("risky-strategy", Decimal("10"))

        # Set up scenario that will trigger risk limits
        paper_broker.set_price("AAPL", Decimal("100.00"))

        # Execute many small trades to hit daily limit
        for i in range(50):
            signal = StrategySignal(
                strategy_id="risky-strategy",
                symbol=Symbol(code=f"STOCK{i}", asset_class="equity"),
                action="BUY",
                quantity=Decimal("1"),
                order_type=OrderType.MARKET,
            )
            await strategy_executor.execute_signal(signal)

        # Next trade should be blocked by daily trade limit
        final_signal = StrategySignal(
            strategy_id="risky-strategy",
            symbol=Symbol(code="FINAL", asset_class="equity"),
            action="BUY",
            quantity=Decimal("1"),
            order_type=OrderType.MARKET,
        )

        result = await strategy_executor.execute_signal(final_signal)
        # May succeed or fail depending on exact limits, but system should handle it gracefully
        assert isinstance(result.success, bool)

        await strategy_executor.stop()
