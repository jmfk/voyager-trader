"""
Common value objects and enums for VOYAGER-Trader domain models.

This module defines shared types, enums, and value objects used
across multiple domain models.
"""

from decimal import Decimal
from enum import Enum
from typing import Optional

from pydantic import Field, field_validator

from .base import ValueObject


class Currency(str, Enum):
    """Supported currencies for trading operations."""

    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    JPY = "JPY"
    CHF = "CHF"
    CAD = "CAD"
    AUD = "AUD"
    NZD = "NZD"
    BTC = "BTC"
    ETH = "ETH"


class AssetClass(str, Enum):
    """Asset classes for trading instruments."""

    EQUITY = "equity"
    FOREX = "forex"
    COMMODITY = "commodity"
    BOND = "bond"
    CRYPTO = "crypto"
    DERIVATIVE = "derivative"
    INDEX = "index"
    ETF = "etf"


class OrderType(str, Enum):
    """Types of trading orders."""

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"
    BRACKET = "bracket"
    OCO = "oco"  # One-Cancels-Other


class OrderSide(str, Enum):
    """Side of trading orders."""

    BUY = "buy"
    SELL = "sell"


class OrderStatus(str, Enum):
    """Status of trading orders."""

    PENDING = "pending"
    SUBMITTED = "submitted"
    ACCEPTED = "accepted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class PositionType(str, Enum):
    """Types of trading positions."""

    LONG = "long"
    SHORT = "short"


class TimeFrame(str, Enum):
    """Time frames for market data and analysis."""

    TICK = "tick"
    SECOND_1 = "1s"
    SECOND_5 = "5s"
    SECOND_15 = "15s"
    SECOND_30 = "30s"
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAY_1 = "1d"
    WEEK_1 = "1w"
    MONTH_1 = "1M"


class SignalType(str, Enum):
    """Types of trading signals."""

    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    EXIT_LONG = "exit_long"
    EXIT_SHORT = "exit_short"
    REDUCE_POSITION = "reduce_position"
    INCREASE_POSITION = "increase_position"


class SignalStrength(str, Enum):
    """Strength levels for trading signals."""

    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"


class StrategyStatus(str, Enum):
    """Status of trading strategies."""

    DRAFT = "draft"
    BACKTESTING = "backtesting"
    PAPER_TRADING = "paper_trading"
    LIVE_TRADING = "live_trading"
    PAUSED = "paused"
    STOPPED = "stopped"
    ARCHIVED = "archived"


class SkillCategory(str, Enum):
    """Categories for VOYAGER skills."""

    MARKET_ANALYSIS = "market_analysis"
    TECHNICAL_ANALYSIS = "technical_analysis"
    FUNDAMENTAL_ANALYSIS = "fundamental_analysis"
    RISK_MANAGEMENT = "risk_management"
    POSITION_SIZING = "position_sizing"
    ENTRY_TIMING = "entry_timing"
    EXIT_TIMING = "exit_timing"
    PORTFOLIO_MANAGEMENT = "portfolio_management"
    DATA_PROCESSING = "data_processing"
    SIGNAL_GENERATION = "signal_generation"


class SkillComplexity(str, Enum):
    """Complexity levels for VOYAGER skills."""

    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class TaskStatus(str, Enum):
    """Status of VOYAGER curriculum tasks."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


class Money(ValueObject):
    """Value object representing monetary amounts with currency."""

    amount: Decimal = Field(description="The monetary amount")
    currency: Currency = Field(description="The currency of the amount")

    @field_validator("amount")
    @classmethod
    def validate_amount(cls, v: Decimal) -> Decimal:
        """Validate that amount has appropriate precision."""
        if isinstance(v, (int, float)):
            v = Decimal(str(v))
        # Round to 8 decimal places for crypto compatibility
        return v.quantize(Decimal("0.00000001"))

    def __str__(self) -> str:
        """String representation of the monetary amount."""
        return f"{self.amount} {self.currency}"

    def __add__(self, other: "Money") -> "Money":
        """Add two Money objects of the same currency."""
        if self.currency != other.currency:
            raise ValueError(f"Cannot add {self.currency} and {other.currency}")
        return Money(amount=self.amount + other.amount, currency=self.currency)

    def __sub__(self, other: "Money") -> "Money":
        """Subtract two Money objects of the same currency."""
        if self.currency != other.currency:
            raise ValueError(f"Cannot subtract {other.currency} from {self.currency}")
        return Money(amount=self.amount - other.amount, currency=self.currency)

    def __mul__(self, factor: Decimal) -> "Money":
        """Multiply money by a factor."""
        if isinstance(factor, (int, float)):
            factor = Decimal(str(factor))
        return Money(amount=self.amount * factor, currency=self.currency)

    def __truediv__(self, divisor: Decimal) -> "Money":
        """Divide money by a divisor."""
        if isinstance(divisor, (int, float)):
            divisor = Decimal(str(divisor))
        return Money(amount=self.amount / divisor, currency=self.currency)

    def is_positive(self) -> bool:
        """Check if the amount is positive."""
        return self.amount > 0

    def is_negative(self) -> bool:
        """Check if the amount is negative."""
        return self.amount < 0

    def is_zero(self) -> bool:
        """Check if the amount is zero."""
        return self.amount == 0

    def abs(self) -> "Money":
        """Return absolute value of the money."""
        return Money(amount=abs(self.amount), currency=self.currency)


class Price(ValueObject):
    """Value object representing a price with bid/ask spread."""

    bid: Decimal = Field(description="Bid price")
    ask: Decimal = Field(description="Ask price")
    currency: Currency = Field(description="Price currency")

    @field_validator("bid", "ask")
    @classmethod
    def validate_prices(cls, v: Decimal) -> Decimal:
        """Validate that prices are positive and have appropriate precision."""
        if isinstance(v, (int, float)):
            v = Decimal(str(v))
        if v <= 0:
            raise ValueError("Prices must be positive")
        return v.quantize(Decimal("0.00000001"))

    @property
    def mid(self) -> Decimal:
        """Calculate mid price between bid and ask."""
        return (self.bid + self.ask) / 2

    @property
    def spread(self) -> Decimal:
        """Calculate spread between bid and ask."""
        return self.ask - self.bid

    @property
    def spread_bps(self) -> Decimal:
        """Calculate spread in basis points."""
        return (self.spread / self.mid) * 10000

    def __str__(self) -> str:
        """String representation of the price."""
        return f"{self.bid}/{self.ask} {self.currency}"


class Quantity(ValueObject):
    """Value object representing a quantity of an asset."""

    amount: Decimal = Field(description="The quantity amount")

    @field_validator("amount")
    @classmethod
    def validate_amount(cls, v: Decimal) -> Decimal:
        """Validate quantity precision."""
        if isinstance(v, (int, float)):
            v = Decimal(str(v))
        return v.quantize(Decimal("0.00000001"))

    def __str__(self) -> str:
        """String representation of the quantity."""
        return str(self.amount)

    def __add__(self, other: "Quantity") -> "Quantity":
        """Add two quantities."""
        return Quantity(amount=self.amount + other.amount)

    def __sub__(self, other: "Quantity") -> "Quantity":
        """Subtract two quantities."""
        return Quantity(amount=self.amount - other.amount)

    def __mul__(self, factor: Decimal) -> "Quantity":
        """Multiply quantity by a factor."""
        if isinstance(factor, (int, float)):
            factor = Decimal(str(factor))
        return Quantity(amount=self.amount * factor)

    def __truediv__(self, divisor: Decimal) -> "Quantity":
        """Divide quantity by a divisor."""
        if isinstance(divisor, (int, float)):
            divisor = Decimal(str(divisor))
        return Quantity(amount=self.amount / divisor)

    def is_positive(self) -> bool:
        """Check if quantity is positive."""
        return self.amount > 0

    def is_negative(self) -> bool:
        """Check if quantity is negative."""
        return self.amount < 0

    def is_zero(self) -> bool:
        """Check if quantity is zero."""
        return self.amount == 0

    def abs(self) -> "Quantity":
        """Return absolute value of the quantity."""
        return Quantity(amount=abs(self.amount))


class Symbol(ValueObject):
    """Value object representing a trading symbol."""

    code: str = Field(description="Symbol code (e.g., AAPL, EURUSD)")
    exchange: Optional[str] = Field(default=None, description="Exchange code")
    asset_class: AssetClass = Field(description="Asset class")
    base_currency: Optional[Currency] = Field(
        default=None, description="Base currency for forex pairs"
    )
    quote_currency: Optional[Currency] = Field(
        default=None, description="Quote currency for forex pairs"
    )

    @field_validator("code")
    @classmethod
    def validate_code(cls, v: str) -> str:
        """Validate symbol code format."""
        if not v or not v.strip():
            raise ValueError("Symbol code cannot be empty")
        return v.strip().upper()

    def __str__(self) -> str:
        """String representation of the symbol."""
        if self.exchange:
            return f"{self.code}@{self.exchange}"
        return self.code

    @property
    def is_forex(self) -> bool:
        """Check if this is a forex symbol."""
        return self.asset_class == AssetClass.FOREX

    @property
    def is_crypto(self) -> bool:
        """Check if this is a crypto symbol."""
        return self.asset_class == AssetClass.CRYPTO
