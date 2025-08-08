"""
Brokerage interfaces for trading execution.

Provides abstract interfaces for brokerage integration and a paper trading implementation.
"""

import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from ..models.trading import Account, Order, Position, Trade
from ..models.types import Money, OrderStatus, Quantity, Symbol


class ExecutionResult(BaseModel):
    """Result of order execution."""

    success: bool = Field(description="Whether execution was successful")
    order_id: str = Field(description="Order ID")
    trade_id: Optional[str] = Field(default=None, description="Trade ID if executed")
    filled_quantity: Quantity = Field(description="Quantity filled")
    fill_price: Optional[Decimal] = Field(default=None, description="Fill price")
    commission: Optional[Money] = Field(default=None, description="Commission charged")
    error_message: Optional[str] = Field(
        default=None, description="Error message if failed"
    )


class BrokerageInterface(ABC):
    """Abstract interface for brokerage integration."""

    @abstractmethod
    async def submit_order(self, order: Order) -> ExecutionResult:
        """Submit order to brokerage."""

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order."""

    @abstractmethod
    async def modify_order(self, order_id: str, **modifications) -> bool:
        """Modify existing order."""

    @abstractmethod
    async def get_order_status(self, order_id: str) -> Optional[Order]:
        """Get current order status."""

    @abstractmethod
    async def get_account_info(self) -> Account:
        """Get account information."""

    @abstractmethod
    async def get_positions(self) -> List[Position]:
        """Get current positions."""

    @abstractmethod
    async def get_current_price(self, symbol: Symbol) -> Optional[Decimal]:
        """Get current market price for symbol."""


class PaperBroker(BrokerageInterface):
    """Paper trading broker implementation."""

    def __init__(self, initial_cash: Money = None):
        """Initialize paper broker."""
        if initial_cash is None:
            initial_cash = Money(amount=Decimal("100000"), currency="USD")

        self._cash = initial_cash
        self._orders: Dict[str, Order] = {}
        self._positions: Dict[str, Position] = {}
        self._trades: List[Trade] = []
        self._prices: Dict[str, Decimal] = {}
        self._commission_rate = Decimal("0.001")  # 0.1% commission

    async def submit_order(self, order: Order) -> ExecutionResult:
        """Submit order for paper trading execution."""
        try:
            # Store order
            self._orders[order.id] = order

            # Get current price
            price = await self.get_current_price(order.symbol)
            if price is None:
                return ExecutionResult(
                    success=False,
                    order_id=order.id,
                    filled_quantity=Quantity(amount=Decimal("0")),
                    error_message=f"No price data for {order.symbol.code}",
                )

            # Determine execution price
            if order.is_market_order:
                exec_price = price
            elif order.is_limit_order and order.price:
                # For paper trading, assume limit orders fill immediately if price is favorable
                if (order.is_buy and price <= order.price) or (
                    order.is_sell and price >= order.price
                ):
                    exec_price = order.price
                else:
                    # Order remains pending
                    updated_order = order.update(status=OrderStatus.SUBMITTED)
                    self._orders[order.id] = updated_order
                    return ExecutionResult(
                        success=True,
                        order_id=order.id,
                        filled_quantity=Quantity(amount=Decimal("0")),
                    )
            else:
                exec_price = price

            # Calculate commission
            notional = order.quantity.amount * exec_price
            commission = Money(amount=notional * self._commission_rate, currency="USD")

            # Check if we have sufficient funds
            total_cost = notional + commission.amount
            if order.is_buy and total_cost > self._cash.amount:
                return ExecutionResult(
                    success=False,
                    order_id=order.id,
                    filled_quantity=Quantity(amount=Decimal("0")),
                    error_message="Insufficient funds",
                )

            # Execute the order
            trade = Trade(
                id=str(uuid.uuid4()),
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                price=exec_price,
                timestamp=datetime.utcnow(),
                order_id=order.id,
                commission=commission,
                strategy_id=order.strategy_id,
            )

            # Update order
            updated_order = order.execute_partial(order.quantity, exec_price)
            updated_order = updated_order.update(commission=commission)
            self._orders[order.id] = updated_order

            # Update cash
            if order.is_buy:
                self._cash = Money(
                    amount=self._cash.amount - total_cost, currency="USD"
                )
            else:
                self._cash = Money(
                    amount=self._cash.amount + notional - commission.amount,
                    currency="USD",
                )

            # Update positions
            await self._update_position(trade)

            # Store trade
            self._trades.append(trade)

            return ExecutionResult(
                success=True,
                order_id=order.id,
                trade_id=trade.id,
                filled_quantity=order.quantity,
                fill_price=exec_price,
                commission=commission,
            )

        except Exception as e:
            return ExecutionResult(
                success=False,
                order_id=order.id,
                filled_quantity=Quantity(amount=Decimal("0")),
                error_message=str(e),
            )

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order."""
        if order_id not in self._orders:
            return False

        order = self._orders[order_id]
        if not order.can_cancel():
            return False

        cancelled_order = order.cancel("User requested")
        self._orders[order_id] = cancelled_order
        return True

    async def modify_order(self, order_id: str, **modifications) -> bool:
        """Modify order (simplified for paper trading)."""
        if order_id not in self._orders:
            return False

        order = self._orders[order_id]
        if not order.can_modify():
            return False

        # For paper trading, just update the order
        updated_order = order.update(**modifications)
        self._orders[order_id] = updated_order
        return True

    async def get_order_status(self, order_id: str) -> Optional[Order]:
        """Get order status."""
        return self._orders.get(order_id)

    async def get_account_info(self) -> Account:
        """Get paper account information."""
        total_equity = self._cash.amount

        # Add position values
        for position in self._positions.values():
            if position.market_value:
                total_equity += position.market_value

        return Account(
            id=str(uuid.uuid4()),
            name="Paper Trading Account",
            account_type="paper",
            base_currency="USD",
            cash_balance=self._cash,
            total_equity=Money(amount=total_equity, currency="USD"),
            buying_power=self._cash,  # Simplified
            maintenance_margin=Money(amount=Decimal("0"), currency="USD"),
            max_position_size=Decimal("10"),  # 10% max per position
            max_portfolio_risk=Decimal("2"),  # 2% max portfolio risk
        )

    async def get_positions(self) -> List[Position]:
        """Get current positions."""
        return list(self._positions.values())

    async def get_current_price(self, symbol: Symbol) -> Optional[Decimal]:
        """Get current price (mock implementation)."""
        # In a real implementation, this would fetch from market data service
        # For paper trading, we'll use stored prices or generate mock prices
        if symbol.code in self._prices:
            return self._prices[symbol.code]

        # Generate a mock price for demo purposes
        base_price = Decimal("100.00")
        import random

        variation = Decimal(str(random.uniform(-5, 5)))
        mock_price = base_price + variation
        self._prices[symbol.code] = mock_price
        return mock_price

    async def _update_position(self, trade: Trade) -> None:
        """Update position based on trade."""
        symbol_code = trade.symbol.code

        if symbol_code in self._positions:
            position = self._positions[symbol_code]

            if trade.is_buy:
                # Add to position
                new_quantity = position.quantity + trade.quantity
                total_cost = (
                    position.quantity.amount * position.entry_price + trade.total_cost
                )
                new_avg_price = total_cost / new_quantity.amount

                updated_position = position.update(
                    quantity=new_quantity,
                    entry_price=new_avg_price,
                    current_price=trade.price,
                )
            else:
                # Reduce position
                if trade.quantity.amount >= position.quantity.amount:
                    # Close position
                    updated_position = position.close_position(trade.price)
                    self._positions[symbol_code] = updated_position
                    return
                else:
                    updated_position = position.reduce_position(trade.quantity)
                    updated_position = updated_position.update(
                        current_price=trade.price
                    )

            self._positions[symbol_code] = updated_position
        else:
            # Create new position
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

    def set_price(self, symbol: str, price: Decimal) -> None:
        """Set price for testing purposes."""
        self._prices[symbol] = price

    def get_cash_balance(self) -> Money:
        """Get current cash balance."""
        return self._cash

    def get_trade_history(self) -> List[Trade]:
        """Get trade history."""
        return self._trades.copy()
