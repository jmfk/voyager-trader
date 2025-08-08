"""
Order and Portfolio management for trading execution.

Provides order lifecycle management and portfolio state management
with thread-safe operations and comprehensive tracking.
"""

import asyncio
import logging
import uuid
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional

from pydantic import BaseModel

from ..models.trading import Order, Portfolio, Position, Trade
from ..models.types import Money, OrderSide, OrderStatus, Symbol
from .interfaces import BrokerageInterface, ExecutionResult

logger = logging.getLogger(__name__)


class OrderUpdate(BaseModel):
    """Order status update."""

    order_id: str
    status: OrderStatus
    filled_quantity: Optional[Decimal] = None
    fill_price: Optional[Decimal] = None
    commission: Optional[Money] = None
    timestamp: datetime = datetime.utcnow()


class OrderManager:
    """Manages order lifecycle and execution."""

    def __init__(self, broker: BrokerageInterface):
        """Initialize order manager."""
        self.broker = broker
        self._orders: Dict[str, Order] = {}
        self._pending_orders: Dict[str, Order] = {}
        self._lock = asyncio.Lock()

        logger.info("OrderManager initialized")

    async def submit_order(self, order: Order) -> ExecutionResult:
        """Submit order for execution."""
        async with self._lock:
            try:
                # Store order as pending
                self._pending_orders[order.id] = order
                self._orders[order.id] = order

                logger.info(
                    f"Submitting order {order.id}: {order.side} {order.quantity.amount} {order.symbol.code}"
                )

                # Submit to broker
                result = await self.broker.submit_order(order)

                if result.success:
                    # Update order with execution details
                    if result.filled_quantity.amount > 0:
                        updated_order = order.execute_partial(
                            result.filled_quantity,
                            result.fill_price or order.price or Decimal("0"),
                        )

                        if result.commission:
                            updated_order = updated_order.update(
                                commission=result.commission
                            )

                        self._orders[order.id] = updated_order

                        # Remove from pending if fully filled
                        if updated_order.is_filled:
                            self._pending_orders.pop(order.id, None)

                    logger.info(f"Order {order.id} submitted successfully")
                else:
                    # Update order status to rejected
                    rejected_order = order.update(
                        status=OrderStatus.REJECTED,
                        tags=order.tags + [f"rejected:{result.error_message}"],
                    )
                    self._orders[order.id] = rejected_order
                    self._pending_orders.pop(order.id, None)

                    logger.warning(f"Order {order.id} rejected: {result.error_message}")

                return result

            except Exception as e:
                logger.error(f"Error submitting order {order.id}: {e}")

                # Mark order as failed
                failed_order = order.update(
                    status=OrderStatus.REJECTED, tags=order.tags + [f"error:{str(e)}"]
                )
                self._orders[order.id] = failed_order
                self._pending_orders.pop(order.id, None)

                return ExecutionResult(
                    success=False,
                    order_id=order.id,
                    filled_quantity=order.filled_quantity,
                    error_message=str(e),
                )

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order."""
        async with self._lock:
            if order_id not in self._orders:
                logger.warning(f"Attempt to cancel unknown order {order_id}")
                return False

            order = self._orders[order_id]
            if not order.can_cancel():
                logger.warning(
                    f"Order {order_id} cannot be cancelled (status: {order.status})"
                )
                return False

            try:
                # Cancel with broker
                success = await self.broker.cancel_order(order_id)

                if success:
                    # Update local order
                    cancelled_order = order.cancel("User requested")
                    self._orders[order_id] = cancelled_order
                    self._pending_orders.pop(order_id, None)

                    logger.info(f"Order {order_id} cancelled successfully")
                else:
                    logger.warning(f"Failed to cancel order {order_id} with broker")

                return success

            except Exception as e:
                logger.error(f"Error cancelling order {order_id}: {e}")
                return False

    async def modify_order(self, order_id: str, **modifications) -> bool:
        """Modify order."""
        async with self._lock:
            if order_id not in self._orders:
                logger.warning(f"Attempt to modify unknown order {order_id}")
                return False

            order = self._orders[order_id]
            if not order.can_modify():
                logger.warning(
                    f"Order {order_id} cannot be modified (status: {order.status})"
                )
                return False

            try:
                # Modify with broker
                success = await self.broker.modify_order(order_id, **modifications)

                if success:
                    # Update local order
                    modified_order = order.update(**modifications)
                    self._orders[order_id] = modified_order

                    logger.info(f"Order {order_id} modified successfully")
                else:
                    logger.warning(f"Failed to modify order {order_id} with broker")

                return success

            except Exception as e:
                logger.error(f"Error modifying order {order_id}: {e}")
                return False

    async def update_order_status(self, order_id: str) -> Optional[Order]:
        """Update order status from broker."""
        try:
            broker_order = await self.broker.get_order_status(order_id)
            if broker_order:
                async with self._lock:
                    self._orders[order_id] = broker_order

                    # Remove from pending if no longer active
                    if not broker_order.is_active:
                        self._pending_orders.pop(order_id, None)

                return broker_order
            return None

        except Exception as e:
            logger.error(f"Error updating order status for {order_id}: {e}")
            return None

    async def sync_pending_orders(self) -> None:
        """Sync all pending orders with broker."""
        async with self._lock:
            pending_ids = list(self._pending_orders.keys())

        for order_id in pending_ids:
            await self.update_order_status(order_id)

    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID."""
        return self._orders.get(order_id)

    def get_orders_by_symbol(self, symbol: Symbol) -> List[Order]:
        """Get all orders for symbol."""
        return [order for order in self._orders.values() if order.symbol == symbol]

    def get_orders_by_strategy(self, strategy_id: str) -> List[Order]:
        """Get all orders for strategy."""
        return [
            order for order in self._orders.values() if order.strategy_id == strategy_id
        ]

    def get_active_orders(self) -> List[Order]:
        """Get all active orders."""
        return list(self._pending_orders.values())

    def get_all_orders(self) -> List[Order]:
        """Get all orders."""
        return list(self._orders.values())

    def get_order_stats(self) -> Dict[str, int]:
        """Get order statistics."""
        stats = {}
        for status in OrderStatus:
            stats[status.value] = sum(
                1 for order in self._orders.values() if order.status == status
            )
        return stats


class PortfolioManager:
    """Manages portfolio state and positions."""

    def __init__(self, portfolio: Portfolio, broker: BrokerageInterface):
        """Initialize portfolio manager."""
        self.portfolio = portfolio
        self.broker = broker
        self._positions: Dict[str, Position] = {}
        self._trades: List[Trade] = []
        self._lock = asyncio.Lock()
        self._high_water_mark = portfolio.total_value

        logger.info(f"PortfolioManager initialized for portfolio {portfolio.id}")

    async def process_trade(self, trade: Trade) -> None:
        """Process executed trade and update positions."""
        async with self._lock:
            try:
                logger.info(
                    f"Processing trade {trade.id}: {trade.side} {trade.quantity.amount} {trade.symbol.code}"
                )

                # Store trade
                self._trades.append(trade)

                # Update or create position
                symbol_code = trade.symbol.code

                if symbol_code in self._positions:
                    position = self._positions[symbol_code]

                    if trade.is_buy:
                        # Add to position
                        updated_position = position.add_to_position(
                            trade.quantity, trade.price
                        )
                        updated_position = updated_position.update(
                            current_price=trade.price
                        )
                    else:
                        # Reduce position
                        updated_position = position.reduce_position(trade.quantity)
                        updated_position = updated_position.update(
                            current_price=trade.price
                        )

                        # If position closed, remove from active positions
                        if updated_position.is_closed:
                            self.portfolio = self.portfolio.remove_position(
                                symbol_code,
                                updated_position.unrealized_pnl
                                or Money(amount=Decimal("0"), currency="USD"),
                            )

                    self._positions[symbol_code] = updated_position

                else:
                    # Create new position for buy orders
                    if trade.is_buy:
                        from ..models.types import PositionType

                        position = Position(
                            id=str(uuid.uuid4()),
                            symbol=trade.symbol,
                            position_type=PositionType.LONG,
                            quantity=trade.quantity,
                            entry_price=trade.price,
                            current_price=trade.price,
                            entry_timestamp=trade.timestamp,
                            entry_trades=[trade.id],
                            strategy_id=trade.strategy_id,
                        )

                        self._positions[symbol_code] = position
                        self.portfolio = self.portfolio.add_position(
                            position.id, symbol_code
                        )

                # Update portfolio metrics
                await self._update_portfolio_metrics()

                logger.info(f"Trade {trade.id} processed successfully")

            except Exception as e:
                logger.error(f"Error processing trade {trade.id}: {e}")
                raise

    async def update_position_prices(self, price_updates: Dict[str, Decimal]) -> None:
        """Update position prices with current market data."""
        async with self._lock:
            updated = False

            for symbol_code, price in price_updates.items():
                if symbol_code in self._positions:
                    position = self._positions[symbol_code]
                    updated_position = position.update_price(price)
                    self._positions[symbol_code] = updated_position
                    updated = True

            if updated:
                await self._update_portfolio_metrics()

    async def close_position(
        self, symbol: str, close_price: Decimal
    ) -> Optional[Position]:
        """Close position."""
        async with self._lock:
            if symbol not in self._positions:
                logger.warning(f"Attempt to close non-existent position {symbol}")
                return None

            position = self._positions[symbol]
            if position.is_closed:
                logger.warning(f"Position {symbol} is already closed")
                return position

            # Close position
            closed_position = position.close_position(close_price)
            self._positions[symbol] = closed_position

            # Update portfolio
            pnl = closed_position.unrealized_pnl or Money(
                amount=Decimal("0"), currency="USD"
            )
            self.portfolio = self.portfolio.remove_position(symbol, pnl)

            await self._update_portfolio_metrics()

            logger.info(f"Position {symbol} closed with P&L: {pnl.amount}")
            return closed_position

    async def sync_with_broker(self) -> None:
        """Sync positions with broker."""
        try:
            broker_positions = await self.broker.get_positions()
            broker_account = await self.broker.get_account_info()

            async with self._lock:
                # Update positions from broker data
                for broker_position in broker_positions:
                    symbol_code = broker_position.symbol.code
                    self._positions[symbol_code] = broker_position

                # Update portfolio cash from account
                self.portfolio = self.portfolio.update(
                    cash_balance=broker_account.cash_balance
                )

                await self._update_portfolio_metrics()

            logger.info("Portfolio synced with broker")

        except Exception as e:
            logger.error(f"Error syncing with broker: {e}")

    async def _update_portfolio_metrics(self) -> None:
        """Update portfolio metrics atomically.

        Note: This method should only be called from within an async lock context
        to prevent race conditions with concurrent portfolio updates.
        """
        # Capture current portfolio state for atomic calculation
        current_portfolio = self.portfolio
        current_positions = dict(self._positions)

        # Calculate all metrics in one pass to ensure consistency
        total_value = current_portfolio.cash_balance.amount
        unrealized_pnl = Decimal("0")

        # Sum position values and unrealized P&L
        for position in current_positions.values():
            if position.is_open:
                if position.market_value:
                    total_value += position.market_value
                if position.unrealized_pnl:
                    unrealized_pnl += position.unrealized_pnl

        # Update high water mark atomically
        new_high_water_mark = self._high_water_mark
        if total_value > self._high_water_mark.amount:
            new_high_water_mark = Money(
                amount=total_value, currency=current_portfolio.base_currency
            )

        # Calculate drawdown using the potentially updated high water mark
        if new_high_water_mark.amount > 0:
            drawdown_percent = (
                (new_high_water_mark.amount - total_value)
                / new_high_water_mark.amount
                * 100
            )
        else:
            drawdown_percent = Decimal("0")

        # Determine final max drawdown
        final_max_drawdown = max(drawdown_percent, current_portfolio.max_drawdown)

        # Apply all updates atomically
        self._high_water_mark = new_high_water_mark
        self.portfolio = current_portfolio.update_metrics(
            unrealized_pnl=Money(
                amount=unrealized_pnl, currency=current_portfolio.base_currency
            ),
            total_value=Money(
                amount=total_value, currency=current_portfolio.base_currency
            ),
        )

        # Update max drawdown if it changed
        if final_max_drawdown > current_portfolio.max_drawdown:
            self.portfolio = self.portfolio.update(max_drawdown=final_max_drawdown)

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position by symbol."""
        return self._positions.get(symbol)

    def get_all_positions(self) -> List[Position]:
        """Get all positions."""
        return list(self._positions.values())

    def get_open_positions(self) -> List[Position]:
        """Get open positions."""
        return [pos for pos in self._positions.values() if pos.is_open]

    def get_positions_by_strategy(self, strategy_id: str) -> List[Position]:
        """Get positions by strategy."""
        return [
            pos for pos in self._positions.values() if pos.strategy_id == strategy_id
        ]

    def get_trade_history(self) -> List[Trade]:
        """Get trade history."""
        return self._trades.copy()

    def get_current_portfolio(self) -> Portfolio:
        """Get current portfolio state."""
        return self.portfolio

    def get_high_water_mark(self) -> Money:
        """Get high water mark."""
        return self._high_water_mark

    def get_portfolio_metrics(self) -> Dict[str, Decimal]:
        """Get portfolio performance metrics."""
        total_trades = len(self._trades)
        winning_trades = len(
            [t for t in self._trades if t.side == OrderSide.SELL]
        )  # Simplified

        return {
            "total_value": self.portfolio.total_value.amount,
            "cash_percentage": self.portfolio.cash_percentage,
            "unrealized_pnl": self.portfolio.unrealized_pnl.amount,
            "realized_pnl": self.portfolio.realized_pnl.amount,
            "total_pnl": self.portfolio.total_pnl.amount,
            "max_drawdown": self.portfolio.max_drawdown,
            "position_count": Decimal(str(self.portfolio.position_count)),
            "total_trades": Decimal(str(total_trades)),
            "high_water_mark": self._high_water_mark.amount,
        }
