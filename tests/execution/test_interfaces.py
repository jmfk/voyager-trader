"""Tests for brokerage interfaces."""

from decimal import Decimal

import pytest

from src.voyager_trader.execution.interfaces import ExecutionResult, PaperBroker
from src.voyager_trader.models.trading import Order
from src.voyager_trader.models.types import (
    Money,
    OrderSide,
    OrderStatus,
    OrderType,
    Quantity,
    Symbol,
)

# Mock imports available for future use


@pytest.fixture
def paper_broker():
    """Create paper broker for testing."""
    initial_cash = Money(amount=Decimal("100000"), currency="USD")
    return PaperBroker(initial_cash=initial_cash)


@pytest.fixture
def sample_order():
    """Create sample order for testing."""
    return Order(
        id="test-order-1",
        symbol=Symbol(code="AAPL", asset_class="equity"),
        order_type=OrderType.MARKET,
        side=OrderSide.BUY,
        quantity=Quantity(amount=Decimal("100")),
        strategy_id="test-strategy",
    )


@pytest.fixture
def sample_limit_order():
    """Create sample limit order for testing."""
    return Order(
        id="test-limit-1",
        symbol=Symbol(code="AAPL", asset_class="equity"),
        order_type=OrderType.LIMIT,
        side=OrderSide.BUY,
        quantity=Quantity(amount=Decimal("100")),
        price=Decimal("150.00"),
        strategy_id="test-strategy",
    )


class TestPaperBroker:
    """Test paper trading broker."""

    def test_initialization(self):
        """Test paper broker initialization."""
        initial_cash = Money(amount=Decimal("50000"), currency="USD")
        broker = PaperBroker(initial_cash=initial_cash)

        assert broker.get_cash_balance() == initial_cash
        assert len(broker.get_trade_history()) == 0

    def test_default_initialization(self):
        """Test default initialization."""
        broker = PaperBroker()
        assert broker.get_cash_balance().amount == Decimal("100000")
        assert broker.get_cash_balance().currency == "USD"

    @pytest.mark.asyncio
    async def test_market_order_execution(self, paper_broker, sample_order):
        """Test market order execution."""
        # Set a mock price
        paper_broker.set_price("AAPL", Decimal("150.00"))

        result = await paper_broker.submit_order(sample_order)

        assert result.success
        assert result.order_id == sample_order.id
        assert result.filled_quantity.amount == Decimal("100")
        assert result.fill_price == Decimal("150.00")
        assert result.commission is not None
        assert result.commission.amount > 0

    @pytest.mark.asyncio
    async def test_limit_order_execution_favorable_price(
        self, paper_broker, sample_limit_order
    ):
        """Test limit order execution with favorable price."""
        # Set price below limit price for buy order
        paper_broker.set_price("AAPL", Decimal("145.00"))

        result = await paper_broker.submit_order(sample_limit_order)

        assert result.success
        assert result.filled_quantity.amount == Decimal("100")
        assert result.fill_price == Decimal("150.00")  # Should fill at limit price

    @pytest.mark.asyncio
    async def test_limit_order_no_execution_unfavorable_price(
        self, paper_broker, sample_limit_order
    ):
        """Test limit order with unfavorable price remains pending."""
        # Set price above limit price for buy order
        paper_broker.set_price("AAPL", Decimal("155.00"))

        result = await paper_broker.submit_order(sample_limit_order)

        assert result.success
        assert result.filled_quantity.amount == Decimal("0")
        assert result.fill_price is None

        # Order should be pending
        order_status = await paper_broker.get_order_status(sample_limit_order.id)
        assert order_status is not None
        assert order_status.status == OrderStatus.SUBMITTED

    @pytest.mark.asyncio
    async def test_insufficient_funds(self, paper_broker):
        """Test insufficient funds handling."""
        # Create large order that exceeds available cash
        large_order = Order(
            id="large-order",
            symbol=Symbol(code="AAPL", asset_class="equity"),
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=Quantity(amount=Decimal("10000")),  # Very large quantity
            strategy_id="test-strategy",
        )

        paper_broker.set_price("AAPL", Decimal("150.00"))  # Total would be $1.5M

        result = await paper_broker.submit_order(large_order)

        assert not result.success
        assert "Insufficient funds" in result.error_message

    @pytest.mark.asyncio
    async def test_sell_order(self, paper_broker):
        """Test sell order execution."""
        # First buy to establish position
        buy_order = Order(
            id="buy-1",
            symbol=Symbol(code="AAPL", asset_class="equity"),
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=Quantity(amount=Decimal("100")),
            strategy_id="test-strategy",
        )

        paper_broker.set_price("AAPL", Decimal("150.00"))
        buy_result = await paper_broker.submit_order(buy_order)
        assert buy_result.success

        # Now sell
        sell_order = Order(
            id="sell-1",
            symbol=Symbol(code="AAPL", asset_class="equity"),
            order_type=OrderType.MARKET,
            side=OrderSide.SELL,
            quantity=Quantity(amount=Decimal("50")),
            strategy_id="test-strategy",
        )

        paper_broker.set_price("AAPL", Decimal("155.00"))
        sell_result = await paper_broker.submit_order(sell_order)

        assert sell_result.success
        assert sell_result.filled_quantity.amount == Decimal("50")
        assert sell_result.fill_price == Decimal("155.00")

    @pytest.mark.asyncio
    async def test_order_cancellation(self, paper_broker, sample_limit_order):
        """Test order cancellation."""
        # Set unfavorable price so order stays pending
        paper_broker.set_price("AAPL", Decimal("155.00"))

        result = await paper_broker.submit_order(sample_limit_order)
        assert result.success
        assert result.filled_quantity.amount == Decimal("0")

        # Cancel the order
        cancel_result = await paper_broker.cancel_order(sample_limit_order.id)
        assert cancel_result

        # Check order status
        order_status = await paper_broker.get_order_status(sample_limit_order.id)
        assert order_status is not None
        assert order_status.status == OrderStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_order_modification(self, paper_broker, sample_limit_order):
        """Test order modification."""
        # Set unfavorable price so order stays pending
        paper_broker.set_price("AAPL", Decimal("155.00"))

        result = await paper_broker.submit_order(sample_limit_order)
        assert result.success

        # Modify the order
        modify_result = await paper_broker.modify_order(
            sample_limit_order.id, price=Decimal("160.00")
        )
        assert modify_result

        # Check modified order
        order_status = await paper_broker.get_order_status(sample_limit_order.id)
        assert order_status is not None
        assert order_status.price == Decimal("160.00")

    @pytest.mark.asyncio
    async def test_account_info(self, paper_broker):
        """Test account information retrieval."""
        account = await paper_broker.get_account_info()

        assert account is not None
        assert account.name == "Paper Trading Account"
        assert account.account_type == "paper"
        assert account.cash_balance.amount == Decimal("100000")
        assert account.base_currency == "USD"

    @pytest.mark.asyncio
    async def test_position_tracking(self, paper_broker):
        """Test position tracking."""
        # Buy some shares
        buy_order = Order(
            id="buy-position",
            symbol=Symbol(code="AAPL", asset_class="equity"),
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=Quantity(amount=Decimal("100")),
            strategy_id="test-strategy",
        )

        paper_broker.set_price("AAPL", Decimal("150.00"))
        await paper_broker.submit_order(buy_order)

        # Check positions
        positions = await paper_broker.get_positions()
        assert len(positions) == 1

        position = positions[0]
        assert position.symbol.code == "AAPL"
        assert position.quantity.amount == Decimal("100")
        assert position.entry_price == Decimal("150.00")
        assert position.is_long
        assert position.is_open

    @pytest.mark.asyncio
    async def test_cash_balance_updates(self, paper_broker):
        """Test cash balance updates after trades."""
        initial_cash = paper_broker.get_cash_balance().amount

        # Buy shares
        buy_order = Order(
            id="cash-test",
            symbol=Symbol(code="AAPL", asset_class="equity"),
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=Quantity(amount=Decimal("100")),
            strategy_id="test-strategy",
        )

        paper_broker.set_price("AAPL", Decimal("150.00"))
        result = await paper_broker.submit_order(buy_order)

        # Check cash was deducted
        new_cash = paper_broker.get_cash_balance().amount
        expected_cost = Decimal("150.00") * Decimal("100") + result.commission.amount

        assert new_cash == initial_cash - expected_cost
        assert new_cash < initial_cash

    @pytest.mark.asyncio
    async def test_trade_history(self, paper_broker):
        """Test trade history tracking."""
        initial_history_length = len(paper_broker.get_trade_history())

        # Execute a trade
        buy_order = Order(
            id="history-test",
            symbol=Symbol(code="AAPL", asset_class="equity"),
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=Quantity(amount=Decimal("100")),
            strategy_id="test-strategy",
        )

        paper_broker.set_price("AAPL", Decimal("150.00"))
        await paper_broker.submit_order(buy_order)

        # Check trade was recorded
        history = paper_broker.get_trade_history()
        assert len(history) == initial_history_length + 1

        trade = history[-1]
        assert trade.symbol.code == "AAPL"
        assert trade.quantity.amount == Decimal("100")
        assert trade.price == Decimal("150.00")
        assert trade.is_buy

    @pytest.mark.asyncio
    async def test_no_price_data_error(self, paper_broker, sample_order):
        """Test handling when no price data is available."""
        # Don't set any price data
        result = await paper_broker.submit_order(sample_order)

        assert not result.success
        assert "No price data" in result.error_message

    @pytest.mark.asyncio
    async def test_current_price_generation(self, paper_broker):
        """Test mock price generation."""
        # Get price for new symbol (should generate mock price)
        price = await paper_broker.get_current_price(
            Symbol(code="TSLA", asset_class="equity")
        )

        assert price is not None
        assert isinstance(price, Decimal)
        assert price > 0

        # Getting price again should return same price
        price2 = await paper_broker.get_current_price(
            Symbol(code="TSLA", asset_class="equity")
        )
        assert price2 == price


class TestExecutionResult:
    """Test ExecutionResult model."""

    def test_successful_result_creation(self):
        """Test creating successful execution result."""
        result = ExecutionResult(
            success=True,
            order_id="test-123",
            trade_id="trade-456",
            filled_quantity=Quantity(amount=Decimal("100")),
            fill_price=Decimal("150.00"),
            commission=Money(amount=Decimal("1.50"), currency="USD"),
        )

        assert result.success
        assert result.order_id == "test-123"
        assert result.trade_id == "trade-456"
        assert result.filled_quantity.amount == Decimal("100")
        assert result.fill_price == Decimal("150.00")
        assert result.commission.amount == Decimal("1.50")
        assert result.error_message is None

    def test_failed_result_creation(self):
        """Test creating failed execution result."""
        result = ExecutionResult(
            success=False,
            order_id="test-123",
            filled_quantity=Quantity(amount=Decimal("0")),
            error_message="Insufficient funds",
        )

        assert not result.success
        assert result.order_id == "test-123"
        assert result.trade_id is None
        assert result.filled_quantity.amount == Decimal("0")
        assert result.fill_price is None
        assert result.error_message == "Insufficient funds"
