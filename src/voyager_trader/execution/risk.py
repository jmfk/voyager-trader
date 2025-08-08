"""
Risk management for trading execution.

Provides comprehensive risk controls including position limits, loss limits,
and portfolio risk management.
"""

import logging
from decimal import Decimal
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from ..models.trading import Account, Order, Portfolio, Position
from ..models.types import Money, OrderSide, Symbol

logger = logging.getLogger(__name__)


class RiskLimits(BaseModel):
    """Risk limit configuration."""

    max_position_size_percent: Decimal = Field(
        default=Decimal("10"), description="Max position size as % of portfolio"
    )
    max_portfolio_risk_percent: Decimal = Field(
        default=Decimal("2"), description="Max portfolio risk per trade"
    )
    daily_loss_limit: Optional[Money] = Field(
        default=None, description="Daily loss limit"
    )
    max_drawdown_percent: Decimal = Field(
        default=Decimal("20"), description="Maximum drawdown before stop"
    )
    max_correlation: Decimal = Field(
        default=Decimal("0.7"), description="Maximum correlation between positions"
    )
    max_leverage: Decimal = Field(
        default=Decimal("1.0"), description="Maximum leverage ratio"
    )
    max_trades_per_day: int = Field(default=100, description="Maximum trades per day")
    cooldown_period_minutes: int = Field(
        default=60, description="Cooldown period after loss limit hit"
    )


class RiskViolation(Exception):
    """Exception raised when risk limits are violated."""



class RiskManager:
    """Comprehensive risk management system."""

    def __init__(self, limits: RiskLimits = None):
        """Initialize risk manager."""
        self.limits = limits or RiskLimits()
        self._daily_trades = 0
        self._daily_pnl = Money(amount=Decimal("0"), currency="USD")
        self._is_shutdown = False
        self._shutdown_reason: Optional[str] = None

        logger.info(f"RiskManager initialized with limits: {self.limits}")

    def validate_order(
        self,
        order: Order,
        portfolio: Portfolio,
        account: Account,
        positions: List[Position],
    ) -> bool:
        """Validate order against risk limits."""
        try:
            self._check_shutdown_status()
            self._check_daily_trade_limit()
            self._check_position_size_limit(order, portfolio)
            self._check_portfolio_risk_limit(order, portfolio, account)
            self._check_leverage_limit(order, account, positions)
            self._check_margin_requirements(order, account)

            logger.debug(f"Order {order.id} passed all risk checks")
            return True

        except RiskViolation as e:
            logger.warning(f"Risk violation for order {order.id}: {e}")
            return False

    def validate_position_increase(
        self,
        symbol: Symbol,
        additional_quantity: Decimal,
        price: Decimal,
        portfolio: Portfolio,
    ) -> bool:
        """Validate increasing position size."""
        try:
            additional_value = additional_quantity * price
            max_position_value = portfolio.total_value.amount * (
                self.limits.max_position_size_percent / 100
            )

            # Get current position value
            current_value = Decimal("0")
            if symbol.code in portfolio.positions:
                # Would need position details to calculate exact value
                # For now, approximate based on total portfolio exposure
                current_exposure = (
                    portfolio.total_value.amount - portfolio.cash_balance.amount
                )
                estimated_position_value = current_exposure / max(
                    len(portfolio.positions), 1
                )
                current_value = estimated_position_value

            total_position_value = current_value + additional_value

            if total_position_value > max_position_value:
                raise RiskViolation(
                    f"Position size would exceed limit: {total_position_value} > {max_position_value}"
                )

            return True

        except RiskViolation as e:
            logger.warning(
                f"Position increase validation failed for {symbol.code}: {e}"
            )
            return False

    def calculate_position_size(
        self,
        symbol: Symbol,
        entry_price: Decimal,
        stop_loss_price: Optional[Decimal],
        portfolio: Portfolio,
        risk_percent: Optional[Decimal] = None,
    ) -> Decimal:
        """Calculate appropriate position size based on risk."""
        if risk_percent is None:
            risk_percent = self.limits.max_portfolio_risk_percent

        # Risk amount in dollars
        risk_amount = portfolio.total_value.amount * (risk_percent / 100)

        if stop_loss_price is not None:
            # Calculate position size based on stop loss
            risk_per_share = abs(entry_price - stop_loss_price)
            if risk_per_share > 0:
                shares = risk_amount / risk_per_share
            else:
                shares = Decimal("0")
        else:
            # Use position size limit if no stop loss
            max_position_value = portfolio.total_value.amount * (
                self.limits.max_position_size_percent / 100
            )
            shares = max_position_value / entry_price

        # Ensure we don't exceed position size limits
        max_position_value = portfolio.total_value.amount * (
            self.limits.max_position_size_percent / 100
        )
        max_shares = max_position_value / entry_price

        return min(shares, max_shares)

    def update_daily_pnl(self, pnl_change: Money) -> None:
        """Update daily P&L tracking."""
        self._daily_pnl = Money(
            amount=self._daily_pnl.amount + pnl_change.amount,
            currency=self._daily_pnl.currency,
        )

        # Check daily loss limit
        if (
            self.limits.daily_loss_limit
            and self._daily_pnl.amount < -self.limits.daily_loss_limit.amount
        ):
            self._initiate_shutdown("Daily loss limit exceeded")

    def update_daily_trades(self) -> None:
        """Update daily trade count."""
        self._daily_trades += 1

        if self._daily_trades > self.limits.max_trades_per_day:
            self._initiate_shutdown("Daily trade limit exceeded")

    def check_drawdown(self, portfolio: Portfolio, high_water_mark: Money) -> bool:
        """Check if portfolio drawdown exceeds limits."""
        if high_water_mark.amount == 0:
            return True

        current_drawdown = (
            (high_water_mark.amount - portfolio.total_value.amount)
            / high_water_mark.amount
            * 100
        )

        if current_drawdown > self.limits.max_drawdown_percent:
            self._initiate_shutdown(
                f"Maximum drawdown exceeded: {current_drawdown:.2f}%"
            )
            return False

        return True

    def is_shutdown(self) -> bool:
        """Check if risk manager has initiated shutdown."""
        return self._is_shutdown

    def get_shutdown_reason(self) -> Optional[str]:
        """Get shutdown reason if applicable."""
        return self._shutdown_reason

    def reset_daily_counters(self) -> None:
        """Reset daily counters (call at start of new trading day)."""
        self._daily_trades = 0
        self._daily_pnl = Money(amount=Decimal("0"), currency="USD")
        logger.info("Daily risk counters reset")

    def emergency_shutdown(self, reason: str) -> None:
        """Emergency shutdown of trading."""
        self._initiate_shutdown(f"Emergency shutdown: {reason}")

    def _check_shutdown_status(self) -> None:
        """Check if system is shutdown."""
        if self._is_shutdown:
            raise RiskViolation(f"System shutdown: {self._shutdown_reason}")

    def _check_daily_trade_limit(self) -> None:
        """Check daily trade limit."""
        if self._daily_trades >= self.limits.max_trades_per_day:
            raise RiskViolation(f"Daily trade limit exceeded: {self._daily_trades}")

    def _check_position_size_limit(self, order: Order, portfolio: Portfolio) -> None:
        """Check position size limits."""
        order_value = order.quantity.amount * (
            order.price or Decimal("100")
        )  # Fallback price for market orders
        max_position_value = portfolio.total_value.amount * (
            self.limits.max_position_size_percent / 100
        )

        if order_value > max_position_value:
            raise RiskViolation(
                f"Order value {order_value} exceeds position limit {max_position_value}"
            )

    def _check_portfolio_risk_limit(
        self, order: Order, portfolio: Portfolio, account: Account
    ) -> None:
        """Check portfolio risk limits."""
        # Simplified risk check - in practice would use more sophisticated models
        order_value = order.quantity.amount * (order.price or Decimal("100"))
        max_risk_value = portfolio.total_value.amount * (
            self.limits.max_portfolio_risk_percent / 100
        )

        # Assume worst case scenario for risk calculation
        if (
            order_value > max_risk_value * 10
        ):  # 10x multiplier for risk vs position size
            raise RiskViolation(
                f"Order risk {order_value} exceeds portfolio risk limit {max_risk_value}"
            )

    def _check_leverage_limit(
        self, order: Order, account: Account, positions: List[Position]
    ) -> None:
        """Check leverage limits."""
        # Calculate current leverage
        total_position_value = sum(
            pos.market_value or pos.cost_basis for pos in positions if pos.is_open
        )
        current_leverage = (
            total_position_value / account.total_equity.amount
            if account.total_equity.amount > 0
            else Decimal("0")
        )

        # Add order value
        order_value = order.quantity.amount * (order.price or Decimal("100"))
        if order.side == OrderSide.BUY:
            new_leverage = (
                total_position_value + order_value
            ) / account.total_equity.amount
        else:
            new_leverage = current_leverage  # Selling reduces leverage

        if new_leverage > self.limits.max_leverage:
            raise RiskViolation(
                f"Order would exceed leverage limit: {new_leverage:.2f} > {self.limits.max_leverage}"
            )

    def _check_margin_requirements(self, order: Order, account: Account) -> None:
        """Check margin requirements."""
        if not account.is_margin_account:
            # Cash account - need full cash for purchases
            if order.side == OrderSide.BUY:
                order_value = order.quantity.amount * (order.price or Decimal("100"))
                if order_value > account.cash_balance.amount:
                    raise RiskViolation("Insufficient cash for purchase")
        else:
            # Margin account - check buying power
            order_value = order.quantity.amount * (order.price or Decimal("100"))
            if order_value > account.buying_power.amount:
                raise RiskViolation("Insufficient buying power")

    def _initiate_shutdown(self, reason: str) -> None:
        """Initiate system shutdown."""
        self._is_shutdown = True
        self._shutdown_reason = reason
        logger.critical(f"TRADING SHUTDOWN INITIATED: {reason}")

    def get_risk_metrics(
        self, portfolio: Portfolio, account: Account
    ) -> Dict[str, Decimal]:
        """Get current risk metrics."""
        total_exposure = portfolio.total_value.amount - portfolio.cash_balance.amount
        leverage = (
            total_exposure / account.total_equity.amount
            if account.total_equity.amount > 0
            else Decimal("0")
        )

        return {
            "leverage": leverage,
            "cash_percentage": portfolio.cash_percentage,
            "daily_trades": Decimal(str(self._daily_trades)),
            "daily_pnl": self._daily_pnl.amount,
            "is_shutdown": Decimal("1") if self._is_shutdown else Decimal("0"),
            "total_exposure": total_exposure,
            "position_count": Decimal(str(portfolio.position_count)),
        }
