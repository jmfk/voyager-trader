"""
Strategy execution orchestrator.

Main component that safely executes trading strategies with comprehensive
risk management, monitoring, and error handling.
"""

import asyncio
import logging
import uuid
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from ..models.trading import Order, Portfolio
from ..models.types import OrderSide, OrderType, Quantity, Symbol
from .interfaces import BrokerageInterface, ExecutionResult
from .manager import OrderManager, PortfolioManager
from .monitor import ExecutionMonitor
from .risk import RiskManager, RiskViolation

logger = logging.getLogger(__name__)


class StrategySignal(BaseModel):
    """Trading signal from strategy."""

    strategy_id: str = Field(description="Strategy identifier")
    symbol: Symbol = Field(description="Trading symbol")
    action: str = Field(description="Action: BUY, SELL, HOLD")
    quantity: Optional[Decimal] = Field(default=None, description="Suggested quantity")
    price: Optional[Decimal] = Field(default=None, description="Suggested price")
    order_type: OrderType = Field(default=OrderType.MARKET, description="Order type")
    confidence: Decimal = Field(
        default=Decimal("0.5"), description="Signal confidence 0-1"
    )
    reasoning: Optional[str] = Field(default=None, description="Strategy reasoning")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ExecutionConfig(BaseModel):
    """Execution engine configuration."""

    max_concurrent_orders: int = Field(
        default=10, description="Maximum concurrent orders"
    )
    order_timeout_seconds: int = Field(
        default=300, description="Order timeout in seconds"
    )
    position_update_interval: int = Field(
        default=60, description="Position update interval in seconds"
    )
    enable_paper_trading: bool = Field(
        default=True, description="Enable paper trading mode"
    )
    auto_sync_positions: bool = Field(
        default=True, description="Auto-sync positions with broker"
    )
    emergency_stop_enabled: bool = Field(
        default=True, description="Enable emergency stop"
    )
    max_strategy_allocation_percent: Decimal = Field(
        default=Decimal("20"), description="Max allocation per strategy"
    )


class StrategyExecutor:
    """Main strategy execution orchestrator."""

    def __init__(
        self,
        portfolio: Portfolio,
        broker: BrokerageInterface,
        risk_manager: RiskManager,
        config: ExecutionConfig = None,
    ):
        """Initialize strategy executor."""
        self.portfolio = portfolio
        self.broker = broker
        self.risk_manager = risk_manager
        self.config = config or ExecutionConfig()

        # Initialize components
        self.order_manager = OrderManager(broker)
        self.portfolio_manager = PortfolioManager(portfolio, broker)
        self.monitor = ExecutionMonitor()

        # State tracking
        self._running = False
        self._strategies: Dict[str, Dict] = {}
        self._strategy_allocations: Dict[str, Decimal] = {}
        self._active_orders: Dict[str, Order] = {}
        self._lock = asyncio.Lock()

        # Background tasks
        self._tasks: List[asyncio.Task] = []

        logger.info(f"StrategyExecutor initialized for portfolio {portfolio.id}")

    async def start(self) -> None:
        """Start the execution engine."""
        if self._running:
            logger.warning("StrategyExecutor is already running")
            return

        self._running = True

        # Start background tasks
        if self.config.auto_sync_positions:
            task = asyncio.create_task(self._position_sync_loop())
            self._tasks.append(task)

        task = asyncio.create_task(self._order_monitoring_loop())
        self._tasks.append(task)

        task = asyncio.create_task(self._portfolio_monitoring_loop())
        self._tasks.append(task)

        logger.info("StrategyExecutor started")

    async def stop(self) -> None:
        """Stop the execution engine."""
        self._running = False

        # Cancel all background tasks
        for task in self._tasks:
            task.cancel()

        # Wait for tasks to complete
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

        self._tasks.clear()

        # Cancel all pending orders
        active_orders = self.order_manager.get_active_orders()
        for order in active_orders:
            await self.order_manager.cancel_order(order.id)

        logger.info("StrategyExecutor stopped")

    async def register_strategy(
        self, strategy_id: str, allocation_percent: Decimal = Decimal("10")
    ) -> bool:
        """Register a strategy with the executor."""
        if allocation_percent > self.config.max_strategy_allocation_percent:
            logger.error(
                f"Strategy allocation {allocation_percent}% exceeds maximum {self.config.max_strategy_allocation_percent}%"
            )
            return False

        # Check total allocation doesn't exceed 100%
        total_allocation = sum(self._strategy_allocations.values()) + allocation_percent
        if total_allocation > 100:
            logger.error(
                f"Total strategy allocation would exceed 100%: {total_allocation}%"
            )
            return False

        async with self._lock:
            self._strategies[strategy_id] = {
                "active": True,
                "registered_at": datetime.utcnow(),
                "trades": 0,
                "pnl": Decimal("0"),
            }
            self._strategy_allocations[strategy_id] = allocation_percent

        logger.info(
            f"Strategy {strategy_id} registered with {allocation_percent}% allocation"
        )
        return True

    async def unregister_strategy(self, strategy_id: str) -> bool:
        """Unregister a strategy."""
        if strategy_id not in self._strategies:
            logger.warning(f"Strategy {strategy_id} not found")
            return False

        async with self._lock:
            # Cancel any pending orders for this strategy
            strategy_orders = self.order_manager.get_orders_by_strategy(strategy_id)
            for order in strategy_orders:
                if order.is_active:
                    await self.order_manager.cancel_order(order.id)

            # Close any open positions for this strategy
            strategy_positions = self.portfolio_manager.get_positions_by_strategy(
                strategy_id
            )
            for position in strategy_positions:
                if position.is_open:
                    current_price = await self.broker.get_current_price(position.symbol)
                    if current_price:
                        await self.portfolio_manager.close_position(
                            position.symbol.code, current_price
                        )

            # Remove strategy
            self._strategies.pop(strategy_id, None)
            self._strategy_allocations.pop(strategy_id, None)

        logger.info(f"Strategy {strategy_id} unregistered")
        return True

    async def execute_signal(self, signal: StrategySignal) -> ExecutionResult:
        """Execute trading signal from strategy."""
        try:
            # Check if strategy is registered
            if signal.strategy_id not in self._strategies:
                return ExecutionResult(
                    success=False,
                    order_id="",
                    filled_quantity=Quantity(amount=Decimal("0")),
                    error_message=f"Strategy {signal.strategy_id} not registered",
                )

            # Check if strategy is active
            if not self._strategies[signal.strategy_id]["active"]:
                return ExecutionResult(
                    success=False,
                    order_id="",
                    filled_quantity=Quantity(amount=Decimal("0")),
                    error_message=f"Strategy {signal.strategy_id} is inactive",
                )

            logger.info(
                f"Processing signal from {signal.strategy_id}: {signal.action} {signal.symbol.code}"
            )

            # Process signal based on action
            if signal.action.upper() == "BUY":
                return await self._execute_buy_signal(signal)
            elif signal.action.upper() == "SELL":
                return await self._execute_sell_signal(signal)
            elif signal.action.upper() == "HOLD":
                return ExecutionResult(
                    success=True,
                    order_id="",
                    filled_quantity=Quantity(amount=Decimal("0")),
                )
            else:
                return ExecutionResult(
                    success=False,
                    order_id="",
                    filled_quantity=Quantity(amount=Decimal("0")),
                    error_message=f"Unknown action: {signal.action}",
                )

        except Exception as e:
            logger.error(f"Error executing signal: {e}")
            self.monitor.record_error(f"Signal execution error: {e}")

            return ExecutionResult(
                success=False,
                order_id="",
                filled_quantity=Quantity(amount=Decimal("0")),
                error_message=str(e),
            )

    async def _execute_buy_signal(self, signal: StrategySignal) -> ExecutionResult:
        """Execute buy signal."""
        try:
            # Get current account and portfolio state
            account = await self.broker.get_account_info()
            current_portfolio = self.portfolio_manager.get_current_portfolio()
            positions = self.portfolio_manager.get_all_positions()

            # Calculate position size
            if signal.quantity:
                quantity = signal.quantity
            else:
                # Calculate based on risk and allocation
                allocation = self._strategy_allocations.get(
                    signal.strategy_id, Decimal("10")
                )
                allocation_amount = current_portfolio.total_value.amount * (
                    allocation / 100
                )

                current_price = await self.broker.get_current_price(signal.symbol)
                if not current_price:
                    return ExecutionResult(
                        success=False,
                        order_id="",
                        filled_quantity=Quantity(amount=Decimal("0")),
                        error_message=f"No price data for {signal.symbol.code}",
                    )

                quantity = self.risk_manager.calculate_position_size(
                    signal.symbol,
                    current_price,
                    None,  # No stop loss for now
                    current_portfolio,
                )

                # Limit by strategy allocation
                max_shares_by_allocation = allocation_amount / current_price
                quantity = min(quantity, max_shares_by_allocation)

            if quantity <= 0:
                return ExecutionResult(
                    success=False,
                    order_id="",
                    filled_quantity=Quantity(amount=Decimal("0")),
                    error_message="Calculated quantity is zero or negative",
                )

            # Create order
            order = Order(
                id=str(uuid.uuid4()),
                symbol=signal.symbol,
                order_type=signal.order_type,
                side=OrderSide.BUY,
                quantity=Quantity(amount=quantity),
                price=signal.price,
                strategy_id=signal.strategy_id,
                tags=[f"signal_confidence:{signal.confidence}"],
            )

            # Validate order with risk manager
            if not self.risk_manager.validate_order(
                order, current_portfolio, account, positions
            ):
                return ExecutionResult(
                    success=False,
                    order_id=order.id,
                    filled_quantity=Quantity(amount=Decimal("0")),
                    error_message="Order failed risk validation",
                )

            # Check concurrent order limit
            active_orders = self.order_manager.get_active_orders()
            if len(active_orders) >= self.config.max_concurrent_orders:
                return ExecutionResult(
                    success=False,
                    order_id=order.id,
                    filled_quantity=Quantity(amount=Decimal("0")),
                    error_message="Maximum concurrent orders reached",
                )

            # Submit order
            self.monitor.record_order_submitted(order)
            result = await self.order_manager.submit_order(order)

            if result.success:
                async with self._lock:
                    self._active_orders[order.id] = order

                # Update risk manager
                self.risk_manager.update_daily_trades()

                # Record trade if filled
                if result.trade_id:
                    # Create trade record (simplified)
                    from ..models.trading import Trade

                    trade = Trade(
                        id=result.trade_id,
                        symbol=signal.symbol,
                        side=OrderSide.BUY,
                        quantity=result.filled_quantity,
                        price=result.fill_price or signal.price or Decimal("0"),
                        timestamp=datetime.utcnow(),
                        order_id=order.id,
                        commission=result.commission,
                        strategy_id=signal.strategy_id,
                    )

                    await self.portfolio_manager.process_trade(trade)
                    self.monitor.record_trade(trade)

                    # Update strategy stats
                    self._strategies[signal.strategy_id]["trades"] += 1

            return result

        except RiskViolation as e:
            logger.warning(f"Risk violation on buy signal: {e}")
            return ExecutionResult(
                success=False,
                order_id="",
                filled_quantity=Quantity(amount=Decimal("0")),
                error_message=f"Risk violation: {e}",
            )
        except Exception as e:
            logger.error(f"Error executing buy signal: {e}")
            self.monitor.record_error(f"Buy signal error: {e}")
            raise

    async def _execute_sell_signal(self, signal: StrategySignal) -> ExecutionResult:
        """Execute sell signal."""
        try:
            # Check if we have a position to sell
            position = self.portfolio_manager.get_position(signal.symbol.code)
            if not position or not position.is_open:
                return ExecutionResult(
                    success=False,
                    order_id="",
                    filled_quantity=Quantity(amount=Decimal("0")),
                    error_message=f"No open position for {signal.symbol.code}",
                )

            # Determine quantity to sell
            if signal.quantity:
                quantity = min(signal.quantity, position.quantity.amount)
            else:
                quantity = position.quantity.amount  # Close entire position

            # Create sell order
            order = Order(
                id=str(uuid.uuid4()),
                symbol=signal.symbol,
                order_type=signal.order_type,
                side=OrderSide.SELL,
                quantity=Quantity(amount=quantity),
                price=signal.price,
                strategy_id=signal.strategy_id,
                tags=[f"signal_confidence:{signal.confidence}", "position_close"],
            )

            # Submit order (sells generally have fewer restrictions)
            self.monitor.record_order_submitted(order)
            result = await self.order_manager.submit_order(order)

            if result.success:
                async with self._lock:
                    self._active_orders[order.id] = order

                # Update risk manager
                self.risk_manager.update_daily_trades()

                # Record trade if filled
                if result.trade_id:
                    from ..models.trading import Trade

                    trade = Trade(
                        id=result.trade_id,
                        symbol=signal.symbol,
                        side=OrderSide.SELL,
                        quantity=result.filled_quantity,
                        price=result.fill_price or signal.price or Decimal("0"),
                        timestamp=datetime.utcnow(),
                        order_id=order.id,
                        commission=result.commission,
                        strategy_id=signal.strategy_id,
                    )

                    await self.portfolio_manager.process_trade(trade)
                    self.monitor.record_trade(trade)

                    # Update strategy stats
                    self._strategies[signal.strategy_id]["trades"] += 1

            return result

        except Exception as e:
            logger.error(f"Error executing sell signal: {e}")
            self.monitor.record_error(f"Sell signal error: {e}")
            raise

    async def _position_sync_loop(self) -> None:
        """Background task to sync positions with broker."""
        while self._running:
            try:
                await self.portfolio_manager.sync_with_broker()
                await asyncio.sleep(self.config.position_update_interval)
            except Exception as e:
                logger.error(f"Error in position sync loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying

    async def _order_monitoring_loop(self) -> None:
        """Background task to monitor order status."""
        while self._running:
            try:
                await self.order_manager.sync_pending_orders()

                # Update active orders list
                async with self._lock:
                    active_orders = self.order_manager.get_active_orders()
                    self._active_orders = {order.id: order for order in active_orders}

                await asyncio.sleep(30)  # Check every 30 seconds
            except Exception as e:
                logger.error(f"Error in order monitoring loop: {e}")
                await asyncio.sleep(60)

    async def _portfolio_monitoring_loop(self) -> None:
        """Background task to monitor portfolio and risk."""
        while self._running:
            try:
                current_portfolio = self.portfolio_manager.get_current_portfolio()
                high_water_mark = self.portfolio_manager.get_high_water_mark()

                # Record portfolio snapshot
                self.monitor.record_portfolio_snapshot(current_portfolio)

                # Check drawdown limits
                if not self.risk_manager.check_drawdown(
                    current_portfolio, high_water_mark
                ):
                    logger.critical("Maximum drawdown exceeded - stopping execution")
                    await self.stop()
                    return

                # Check if risk manager has initiated shutdown
                if self.risk_manager.is_shutdown():
                    logger.critical(
                        f"Risk manager shutdown - stopping execution: {self.risk_manager.get_shutdown_reason()}"
                    )
                    await self.stop()
                    return

                await asyncio.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Error in portfolio monitoring loop: {e}")
                await asyncio.sleep(60)

    def get_strategy_status(self, strategy_id: str) -> Optional[Dict]:
        """Get strategy status and performance."""
        if strategy_id not in self._strategies:
            return None

        status = self._strategies[strategy_id].copy()
        status["allocation_percent"] = self._strategy_allocations.get(
            strategy_id, Decimal("0")
        )
        status["metrics"] = self.monitor.get_strategy_metrics(strategy_id)

        return status

    def get_execution_status(self) -> Dict:
        """Get overall execution status."""
        return {
            "running": self._running,
            "strategies": len(self._strategies),
            "active_orders": len(self._active_orders),
            "execution_metrics": self.monitor.get_execution_metrics(),
            "system_health": self.monitor.get_system_health(),
            "risk_metrics": self.risk_manager.get_risk_metrics(
                self.portfolio_manager.get_current_portfolio(),
                (
                    asyncio.run(self.broker.get_account_info())
                    if asyncio.get_event_loop().is_running()
                    else None
                ),
            ),
        }

    async def emergency_stop(self, reason: str = "Manual emergency stop") -> None:
        """Emergency stop all trading activity."""
        logger.critical(f"EMERGENCY STOP: {reason}")

        # Initiate risk manager shutdown
        self.risk_manager.emergency_shutdown(reason)

        # Stop the executor
        await self.stop()

    def is_running(self) -> bool:
        """Check if executor is running."""
        return self._running
