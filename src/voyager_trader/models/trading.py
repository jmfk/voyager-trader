"""
Trading entities for VOYAGER-Trader.

This module defines core trading domain entities including Portfolio,
Position, Order, Trade, and Account models.
"""

from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional

from pydantic import Field, computed_field, field_validator

from .base import AggregateRoot, BaseEntity, DomainEvent
from .types import (
    Currency,
    Money,
    OrderSide,
    OrderStatus,
    OrderType,
    PositionType,
    Quantity,
    Symbol,
)


class OrderExecuted(DomainEvent):
    """Domain event for order execution."""

    order_id: str = Field(description="ID of the executed order")
    symbol: Symbol = Field(description="Symbol of the executed order")
    side: OrderSide = Field(description="Order side")
    quantity: Quantity = Field(description="Executed quantity")
    price: Decimal = Field(description="Execution price")

    def __init__(self, **data):
        data["event_type"] = "OrderExecuted"
        super().__init__(**data)


class PositionOpened(DomainEvent):
    """Domain event for position opening."""

    position_id: str = Field(description="ID of the opened position")
    symbol: Symbol = Field(description="Position symbol")
    position_type: PositionType = Field(description="Position type")
    quantity: Quantity = Field(description="Position quantity")

    def __init__(self, **data):
        data["event_type"] = "PositionOpened"
        super().__init__(**data)


class PositionClosed(DomainEvent):
    """Domain event for position closing."""

    position_id: str = Field(description="ID of the closed position")
    symbol: Symbol = Field(description="Position symbol")
    pnl: Money = Field(description="Position P&L")

    def __init__(self, **data):
        data["event_type"] = "PositionClosed"
        super().__init__(**data)


class Order(BaseEntity):
    """
    Trading order entity.

    Represents a trading order with its lifecycle management,
    execution tracking, and state transitions.
    """

    symbol: Symbol = Field(description="Trading symbol")
    order_type: OrderType = Field(description="Order type")
    side: OrderSide = Field(description="Order side (buy/sell)")
    quantity: Quantity = Field(description="Order quantity")
    price: Optional[Decimal] = Field(
        default=None, description="Order price (None for market orders)"
    )
    stop_price: Optional[Decimal] = Field(default=None, description="Stop price")
    time_in_force: str = Field(default="DAY", description="Time in force")
    status: OrderStatus = Field(default=OrderStatus.PENDING, description="Order status")
    filled_quantity: Quantity = Field(
        default_factory=lambda: Quantity(amount=Decimal("0")),
        description="Filled quantity",
    )
    average_fill_price: Optional[Decimal] = Field(
        default=None, description="Average fill price"
    )
    commission: Optional[Money] = Field(default=None, description="Commission paid")
    tags: List[str] = Field(default_factory=list, description="Order tags")
    strategy_id: Optional[str] = Field(
        default=None, description="Associated strategy ID"
    )
    parent_order_id: Optional[str] = Field(
        default=None, description="Parent order ID for brackets"
    )
    child_order_ids: List[str] = Field(
        default_factory=list, description="Child order IDs"
    )

    @field_validator("price", "stop_price", "average_fill_price")
    @classmethod
    def validate_prices(cls, v: Optional[Decimal]) -> Optional[Decimal]:
        """Validate price precision."""
        if v is not None:
            if isinstance(v, (int, float)):
                v = Decimal(str(v))
            if v <= 0:
                raise ValueError("Prices must be positive")
            return v.quantize(Decimal("0.00000001"))
        return v

    @field_validator("time_in_force")
    @classmethod
    def validate_time_in_force(cls, v: str) -> str:
        """Validate time in force values."""
        valid_tif = {"DAY", "GTC", "IOC", "FOK", "GTD", "ATO", "ATC"}
        if v.upper() not in valid_tif:
            raise ValueError(f"Time in force must be one of: {valid_tif}")
        return v.upper()

    @computed_field
    @property
    def remaining_quantity(self) -> Quantity:
        """Calculate remaining quantity to be filled."""
        return self.quantity - self.filled_quantity

    @computed_field
    @property
    def fill_percentage(self) -> Decimal:
        """Calculate fill percentage."""
        if self.quantity.amount == 0:
            return Decimal("0")
        return (self.filled_quantity.amount / self.quantity.amount) * 100

    @property
    def is_market_order(self) -> bool:
        """Check if this is a market order."""
        return self.order_type == OrderType.MARKET

    @property
    def is_limit_order(self) -> bool:
        """Check if this is a limit order."""
        return self.order_type == OrderType.LIMIT

    @property
    def is_stop_order(self) -> bool:
        """Check if this is a stop order."""
        return self.order_type in {OrderType.STOP, OrderType.STOP_LIMIT}

    @property
    def is_buy(self) -> bool:
        """Check if this is a buy order."""
        return self.side == OrderSide.BUY

    @property
    def is_sell(self) -> bool:
        """Check if this is a sell order."""
        return self.side == OrderSide.SELL

    @property
    def is_active(self) -> bool:
        """Check if order is active (can be filled)."""
        return self.status in {
            OrderStatus.PENDING,
            OrderStatus.SUBMITTED,
            OrderStatus.ACCEPTED,
            OrderStatus.PARTIALLY_FILLED,
        }

    @property
    def is_filled(self) -> bool:
        """Check if order is completely filled."""
        return self.status == OrderStatus.FILLED

    @property
    def is_cancelled(self) -> bool:
        """Check if order is cancelled."""
        return self.status == OrderStatus.CANCELLED

    def can_cancel(self) -> bool:
        """Check if order can be cancelled."""
        return self.is_active and self.status != OrderStatus.PENDING

    def can_modify(self) -> bool:
        """Check if order can be modified."""
        return self.is_active and self.filled_quantity.amount == 0

    def execute_partial(self, quantity: Quantity, price: Decimal) -> "Order":
        """Execute partial fill on the order."""
        if not self.is_active:
            raise ValueError("Cannot execute inactive order")
        if quantity.amount > self.remaining_quantity.amount:
            raise ValueError("Execution quantity exceeds remaining quantity")

        new_filled = self.filled_quantity + quantity
        new_avg_price = None

        if self.average_fill_price is not None:
            # Calculate new average fill price
            total_value = (
                self.filled_quantity.amount * self.average_fill_price
                + quantity.amount * price
            )
            new_avg_price = total_value / new_filled.amount
        else:
            new_avg_price = price

        # Determine new status
        if new_filled.amount >= self.quantity.amount:
            new_status = OrderStatus.FILLED
        else:
            new_status = OrderStatus.PARTIALLY_FILLED

        return self.update(
            filled_quantity=new_filled,
            average_fill_price=new_avg_price,
            status=new_status,
        )

    def cancel(self, reason: str = "") -> "Order":
        """Cancel the order."""
        if not self.can_cancel():
            raise ValueError("Order cannot be cancelled")

        tags = self.tags.copy()
        if reason:
            tags.append(f"cancelled:{reason}")

        return self.update(status=OrderStatus.CANCELLED, tags=tags)


class Trade(BaseEntity):
    """
    Executed trade entity.

    Represents a completed trade execution with all relevant details
    for performance tracking and analysis.
    """

    symbol: Symbol = Field(description="Trading symbol")
    side: OrderSide = Field(description="Trade side")
    quantity: Quantity = Field(description="Trade quantity")
    price: Decimal = Field(description="Execution price")
    timestamp: datetime = Field(description="Execution timestamp")
    order_id: str = Field(description="Originating order ID")
    position_id: Optional[str] = Field(
        default=None, description="Associated position ID"
    )
    commission: Optional[Money] = Field(default=None, description="Commission paid")
    fees: Optional[Money] = Field(default=None, description="Additional fees")
    exchange: Optional[str] = Field(default=None, description="Execution exchange")
    strategy_id: Optional[str] = Field(
        default=None, description="Associated strategy ID"
    )
    tags: List[str] = Field(default_factory=list, description="Trade tags")

    @field_validator("price")
    @classmethod
    def validate_price(cls, v: Decimal) -> Decimal:
        """Validate price precision."""
        if isinstance(v, (int, float)):
            v = Decimal(str(v))
        if v <= 0:
            raise ValueError("Price must be positive")
        return v.quantize(Decimal("0.00000001"))

    @computed_field
    @property
    def notional_value(self) -> Decimal:
        """Calculate notional value of the trade."""
        return self.quantity.amount * self.price

    @computed_field
    @property
    def total_cost(self) -> Decimal:
        """Calculate total cost including commissions and fees."""
        cost = self.notional_value
        if self.commission:
            cost += self.commission.amount
        if self.fees:
            cost += self.fees.amount
        return cost

    @property
    def is_buy(self) -> bool:
        """Check if this is a buy trade."""
        return self.side == OrderSide.BUY

    @property
    def is_sell(self) -> bool:
        """Check if this is a sell trade."""
        return self.side == OrderSide.SELL


class Position(BaseEntity):
    """
    Trading position entity.

    Represents an open position in a financial instrument with
    entry/exit tracking and P&L calculation.
    """

    symbol: Symbol = Field(description="Position symbol")
    position_type: PositionType = Field(description="Position type (long/short)")
    quantity: Quantity = Field(description="Position quantity")
    entry_price: Decimal = Field(description="Average entry price")
    current_price: Optional[Decimal] = Field(
        default=None, description="Current market price"
    )
    entry_timestamp: datetime = Field(description="Position entry timestamp")
    exit_timestamp: Optional[datetime] = Field(
        default=None, description="Position exit timestamp"
    )
    entry_trades: List[str] = Field(default_factory=list, description="Entry trade IDs")
    exit_trades: List[str] = Field(default_factory=list, description="Exit trade IDs")
    strategy_id: Optional[str] = Field(
        default=None, description="Associated strategy ID"
    )
    stop_loss: Optional[Decimal] = Field(default=None, description="Stop loss price")
    take_profit: Optional[Decimal] = Field(
        default=None, description="Take profit price"
    )
    tags: List[str] = Field(default_factory=list, description="Position tags")

    @field_validator("entry_price", "current_price", "stop_loss", "take_profit")
    @classmethod
    def validate_prices(cls, v: Optional[Decimal]) -> Optional[Decimal]:
        """Validate price precision."""
        if v is not None:
            if isinstance(v, (int, float)):
                v = Decimal(str(v))
            if v <= 0:
                raise ValueError("Prices must be positive")
            return v.quantize(Decimal("0.00000001"))
        return v

    @computed_field
    @property
    def unrealized_pnl(self) -> Optional[Decimal]:
        """Calculate unrealized P&L."""
        if self.current_price is None:
            return None

        price_diff = self.current_price - self.entry_price
        if self.position_type == PositionType.SHORT:
            price_diff = -price_diff

        return price_diff * self.quantity.amount

    @computed_field
    @property
    def unrealized_pnl_percent(self) -> Optional[Decimal]:
        """Calculate unrealized P&L percentage."""
        pnl = self.unrealized_pnl
        if pnl is None:
            return None

        entry_value = self.entry_price * self.quantity.amount
        if entry_value == 0:
            return Decimal("0")

        return (pnl / entry_value) * 100

    @computed_field
    @property
    def market_value(self) -> Optional[Decimal]:
        """Calculate current market value."""
        if self.current_price is None:
            return None
        return self.current_price * self.quantity.amount

    @computed_field
    @property
    def cost_basis(self) -> Decimal:
        """Calculate cost basis."""
        return self.entry_price * self.quantity.amount

    @property
    def is_long(self) -> bool:
        """Check if this is a long position."""
        return self.position_type == PositionType.LONG

    @property
    def is_short(self) -> bool:
        """Check if this is a short position."""
        return self.position_type == PositionType.SHORT

    @property
    def is_open(self) -> bool:
        """Check if position is still open."""
        return self.exit_timestamp is None

    @property
    def is_closed(self) -> bool:
        """Check if position is closed."""
        return self.exit_timestamp is not None

    @property
    def is_profitable(self) -> Optional[bool]:
        """Check if position is currently profitable."""
        pnl = self.unrealized_pnl
        if pnl is None:
            return None
        return pnl > 0

    def update_price(self, new_price: Decimal) -> "Position":
        """Update current price of the position."""
        return self.update(current_price=new_price)

    def close_position(
        self, exit_price: Decimal, exit_timestamp: datetime = None
    ) -> "Position":
        """Close the position."""
        if self.is_closed:
            raise ValueError("Position is already closed")

        if exit_timestamp is None:
            exit_timestamp = datetime.utcnow()

        return self.update(current_price=exit_price, exit_timestamp=exit_timestamp)

    def add_to_position(self, quantity: Quantity, price: Decimal) -> "Position":
        """Add to existing position (average up/down)."""
        if self.is_closed:
            raise ValueError("Cannot add to closed position")

        total_quantity = self.quantity + quantity
        total_cost = self.quantity.amount * self.entry_price + quantity.amount * price
        new_avg_price = total_cost / total_quantity.amount

        return self.update(quantity=total_quantity, entry_price=new_avg_price)

    def reduce_position(self, quantity: Quantity) -> "Position":
        """Reduce position size."""
        if self.is_closed:
            raise ValueError("Cannot reduce closed position")
        if quantity.amount > self.quantity.amount:
            raise ValueError("Reduction quantity exceeds position size")

        new_quantity = self.quantity - quantity

        # If fully reduced, close the position
        if new_quantity.amount == 0:
            return self.close_position(
                exit_price=self.current_price or self.entry_price,
                exit_timestamp=datetime.utcnow(),
            )

        return self.update(quantity=new_quantity)


class Account(BaseEntity):
    """
    Trading account entity.

    Represents a trading account with cash balance, buying power,
    and risk management parameters.
    """

    name: str = Field(description="Account name")
    account_type: str = Field(description="Account type (cash, margin, etc.)")
    base_currency: Currency = Field(description="Account base currency")
    cash_balance: Money = Field(description="Cash balance")
    total_equity: Money = Field(description="Total account equity")
    buying_power: Money = Field(description="Available buying power")
    maintenance_margin: Money = Field(description="Maintenance margin requirement")
    day_trading_buying_power: Optional[Money] = Field(
        default=None, description="Day trading buying power"
    )
    max_position_size: Decimal = Field(
        description="Maximum position size as percentage"
    )
    max_portfolio_risk: Decimal = Field(
        description="Maximum portfolio risk as percentage"
    )
    daily_loss_limit: Optional[Money] = Field(
        default=None, description="Daily loss limit"
    )
    is_active: bool = Field(default=True, description="Whether account is active")
    risk_parameters: Dict[str, Decimal] = Field(
        default_factory=dict, description="Risk parameters"
    )

    @field_validator("max_position_size", "max_portfolio_risk")
    @classmethod
    def validate_percentages(cls, v: Decimal) -> Decimal:
        """Validate percentage values."""
        if isinstance(v, (int, float)):
            v = Decimal(str(v))
        if v < 0 or v > 100:
            raise ValueError("Percentage must be between 0 and 100")
        return v

    @computed_field
    @property
    def unrestricted_buying_power(self) -> Money:
        """Calculate unrestricted buying power."""
        return Money(
            amount=min(self.buying_power.amount, self.cash_balance.amount),
            currency=self.base_currency,
        )

    @computed_field
    @property
    def margin_used(self) -> Money:
        """Calculate margin currently used."""
        return Money(
            amount=self.total_equity.amount - self.cash_balance.amount,
            currency=self.base_currency,
        )

    @property
    def is_margin_account(self) -> bool:
        """Check if this is a margin account."""
        return self.account_type.lower() == "margin"

    @property
    def is_cash_account(self) -> bool:
        """Check if this is a cash account."""
        return self.account_type.lower() == "cash"

    @property
    def has_day_trading_power(self) -> bool:
        """Check if account has day trading buying power."""
        return self.day_trading_buying_power is not None

    def can_trade(self, notional_value: Money) -> bool:
        """Check if account can execute trade of given notional value."""
        if not self.is_active:
            return False
        return notional_value.amount <= self.buying_power.amount

    def update_balances(
        self,
        cash_balance: Money,
        total_equity: Money,
        buying_power: Money,
        maintenance_margin: Money,
    ) -> "Account":
        """Update account balances."""
        return self.update(
            cash_balance=cash_balance,
            total_equity=total_equity,
            buying_power=buying_power,
            maintenance_margin=maintenance_margin,
        )


class Portfolio(AggregateRoot):
    """
    Portfolio aggregate root.

    Manages all positions and provides portfolio-level metrics,
    risk management, and performance tracking.
    """

    name: str = Field(description="Portfolio name")
    account_id: str = Field(description="Associated account ID")
    base_currency: Currency = Field(description="Portfolio base currency")
    positions: Dict[str, str] = Field(
        default_factory=dict, description="Position IDs by symbol"
    )
    cash_balance: Money = Field(description="Cash balance")
    total_value: Money = Field(description="Total portfolio value")
    unrealized_pnl: Money = Field(
        default_factory=lambda: Money(amount=Decimal("0"), currency=Currency.USD)
    )
    realized_pnl: Money = Field(
        default_factory=lambda: Money(amount=Decimal("0"), currency=Currency.USD)
    )
    daily_pnl: Money = Field(
        default_factory=lambda: Money(amount=Decimal("0"), currency=Currency.USD)
    )
    max_drawdown: Decimal = Field(
        default=Decimal("0"), description="Maximum drawdown percentage"
    )
    risk_metrics: Dict[str, Decimal] = Field(
        default_factory=dict, description="Risk metrics"
    )
    performance_metrics: Dict[str, Decimal] = Field(
        default_factory=dict, description="Performance metrics"
    )

    _domain_events: List[DomainEvent] = []

    @computed_field
    @property
    def position_count(self) -> int:
        """Get number of open positions."""
        return len(self.positions)

    @computed_field
    @property
    def total_pnl(self) -> Money:
        """Calculate total P&L."""
        return Money(
            amount=self.realized_pnl.amount + self.unrealized_pnl.amount,
            currency=self.base_currency,
        )

    @computed_field
    @property
    def cash_percentage(self) -> Decimal:
        """Calculate cash as percentage of total value."""
        if self.total_value.amount == 0:
            return Decimal("100")
        return (self.cash_balance.amount / self.total_value.amount) * 100

    def is_valid(self) -> bool:
        """Check if portfolio is in a valid state."""
        # Portfolio is valid if total value equals cash + position values
        # This is a simplified check - real implementation would sum position values
        return self.total_value.amount >= 0 and self.cash_balance.amount >= 0

    def get_domain_events(self) -> List[DomainEvent]:
        """Get domain events generated by this portfolio."""
        events = self._domain_events.copy()
        self._domain_events.clear()
        return events

    def add_position(self, position_id: str, symbol: str) -> "Portfolio":
        """Add a position to the portfolio."""
        new_positions = self.positions.copy()
        new_positions[symbol] = position_id

        event = PositionOpened(
            aggregate_id=self.id,
            position_id=position_id,
            symbol=Symbol(code=symbol, asset_class="equity"),  # Simplified
            position_type=PositionType.LONG,  # Default
            quantity=Quantity(amount=Decimal("0")),  # Default
        )

        updated_portfolio = self.update(positions=new_positions)
        updated_portfolio._domain_events.append(event)
        return updated_portfolio

    def remove_position(self, symbol: str, pnl: Money) -> "Portfolio":
        """Remove a position from the portfolio."""
        if symbol not in self.positions:
            raise ValueError(f"Position {symbol} not found in portfolio")

        new_positions = self.positions.copy()
        position_id = new_positions.pop(symbol)

        # Update realized P&L
        new_realized_pnl = Money(
            amount=self.realized_pnl.amount + pnl.amount, currency=self.base_currency
        )

        event = PositionClosed(
            aggregate_id=self.id,
            position_id=position_id,
            symbol=Symbol(code=symbol, asset_class="equity"),  # Simplified
            pnl=pnl,
        )

        updated_portfolio = self.update(
            positions=new_positions, realized_pnl=new_realized_pnl
        )
        updated_portfolio._domain_events.append(event)
        return updated_portfolio

    def update_metrics(self, unrealized_pnl: Money, total_value: Money) -> "Portfolio":
        """Update portfolio metrics."""
        return self.update(unrealized_pnl=unrealized_pnl, total_value=total_value)

    def calculate_position_size(self, symbol: str, risk_percent: Decimal) -> Money:
        """Calculate position size based on portfolio risk."""
        risk_amount = self.total_value.amount * (risk_percent / 100)
        return Money(amount=risk_amount, currency=self.base_currency)

    def get_exposure_by_asset_class(self) -> Dict[str, Decimal]:
        """Get portfolio exposure by asset class."""
        # Simplified implementation - would need position details
        return {"equity": Decimal("80"), "cash": self.cash_percentage}

    def get_risk_metrics(self) -> Dict[str, Decimal]:
        """Get current risk metrics."""
        return {
            "total_exposure": Decimal("100") - self.cash_percentage,
            "max_drawdown": self.max_drawdown,
            "unrealized_pnl_percent": (
                (self.unrealized_pnl.amount / self.total_value.amount) * 100
                if self.total_value.amount > 0
                else Decimal("0")
            ),
            **self.risk_metrics,
        }
