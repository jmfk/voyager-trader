"""
Market data models for VOYAGER-Trader.

This module defines models for market data including OHLCV bars,
tick data, order book snapshots, and market events.
"""

from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional

from pydantic import Field, field_validator

from .base import ValueObject, VoyagerBaseModel
from .types import Symbol, TimeFrame


class OHLCV(ValueObject):
    """
    Open, High, Low, Close, Volume market data bar.

    Represents aggregated market data for a specific time period.
    Optimized for time-series storage in InfluxDB.
    """

    symbol: Symbol = Field(description="Trading symbol")
    timestamp: datetime = Field(description="Bar timestamp (start of period)")
    timeframe: TimeFrame = Field(description="Time frame of the bar")
    open: Decimal = Field(description="Opening price")
    high: Decimal = Field(description="Highest price")
    low: Decimal = Field(description="Lowest price")
    close: Decimal = Field(description="Closing price")
    volume: Decimal = Field(description="Trading volume")
    trades_count: Optional[int] = Field(default=None, description="Number of trades")
    vwap: Optional[Decimal] = Field(
        default=None, description="Volume weighted average price"
    )

    @field_validator("open", "high", "low", "close", "volume")
    @classmethod
    def validate_prices(cls, v: Decimal) -> Decimal:
        """Validate price and volume precision."""
        if isinstance(v, (int, float)):
            v = Decimal(str(v))
        if v < 0:
            raise ValueError("Prices and volume must be non-negative")
        return v.quantize(Decimal("0.00000001"))

    @field_validator("trades_count")
    @classmethod
    def validate_trades_count(cls, v: Optional[int]) -> Optional[int]:
        """Validate trades count is positive."""
        if v is not None and v < 0:
            raise ValueError("Trades count must be non-negative")
        return v

    def model_post_init(self, __context) -> None:
        """Validate OHLC relationships after initialization."""
        # First check that low <= high
        if not (self.low <= self.high):
            raise ValueError("Low must be less than or equal to high")
        # Then check that open and close are within the range
        if not (self.low <= self.open <= self.high):
            raise ValueError("Open price must be between low and high")
        if not (self.low <= self.close <= self.high):
            raise ValueError("Close price must be between low and high")

    @property
    def typical_price(self) -> Decimal:
        """Calculate typical price (HLC/3)."""
        return (self.high + self.low + self.close) / 3

    @property
    def price_change(self) -> Decimal:
        """Calculate absolute price change."""
        return self.close - self.open

    @property
    def price_change_percent(self) -> Decimal:
        """Calculate percentage price change."""
        if self.open == 0:
            return Decimal("0")
        return (self.price_change / self.open) * 100

    @property
    def true_range(self) -> Decimal:
        """Calculate true range for volatility analysis."""
        # For single bar, true range is simply high - low
        return self.high - self.low

    @property
    def body_size(self) -> Decimal:
        """Calculate candlestick body size."""
        return abs(self.close - self.open)

    @property
    def upper_wick(self) -> Decimal:
        """Calculate upper wick size."""
        return self.high - max(self.open, self.close)

    @property
    def lower_wick(self) -> Decimal:
        """Calculate lower wick size."""
        return min(self.open, self.close) - self.low

    @property
    def is_bullish(self) -> bool:
        """Check if the bar is bullish (close > open)."""
        return self.close > self.open

    @property
    def is_bearish(self) -> bool:
        """Check if the bar is bearish (close < open)."""
        return self.close < self.open

    @property
    def is_doji(self) -> bool:
        """Check if the bar is a doji (open ≈ close)."""
        # Consider doji if body is less than 10% of the range
        range_size = self.high - self.low
        if range_size == 0:
            return True
        return (self.body_size / range_size) < Decimal("0.1")


class TickData(ValueObject):
    """
    Individual trade tick data.

    Represents the finest granularity of market data with individual
    trades or quotes. Optimized for high-frequency data storage.
    """

    symbol: Symbol = Field(description="Trading symbol")
    timestamp: datetime = Field(description="Tick timestamp with microsecond precision")
    price: Decimal = Field(description="Trade or quote price")
    size: Decimal = Field(description="Trade size or quote size")
    tick_type: str = Field(description="Type of tick (trade, bid, ask)")
    exchange: Optional[str] = Field(default=None, description="Exchange identifier")
    conditions: Optional[List[str]] = Field(
        default=None, description="Trade conditions"
    )

    @field_validator("price", "size")
    @classmethod
    def validate_tick_data(cls, v: Decimal) -> Decimal:
        """Validate tick data precision."""
        if isinstance(v, (int, float)):
            v = Decimal(str(v))
        if v < 0:
            raise ValueError("Price and size must be non-negative")
        return v.quantize(Decimal("0.00000001"))

    @property
    def is_trade(self) -> bool:
        """Check if this is a trade tick."""
        return self.tick_type.lower() == "trade"

    @property
    def is_bid(self) -> bool:
        """Check if this is a bid quote."""
        return self.tick_type.lower() == "bid"

    @property
    def is_ask(self) -> bool:
        """Check if this is an ask quote."""
        return self.tick_type.lower() == "ask"


class OrderBookLevel(ValueObject):
    """Single level in an order book."""

    price: Decimal = Field(description="Price level")
    size: Decimal = Field(description="Total size at this level")
    orders_count: Optional[int] = Field(default=None, description="Number of orders")

    @field_validator("price", "size")
    @classmethod
    def validate_level_data(cls, v: Decimal) -> Decimal:
        """Validate order book level data."""
        if isinstance(v, (int, float)):
            v = Decimal(str(v))
        if v < 0:
            raise ValueError("Price and size must be non-negative")
        return v.quantize(Decimal("0.00000001"))


class OrderBook(ValueObject):
    """
    Order book snapshot with bid/ask levels.

    Represents market depth at a specific point in time with
    multiple price levels on both sides.
    """

    symbol: Symbol = Field(description="Trading symbol")
    timestamp: datetime = Field(description="Snapshot timestamp")
    bids: List[OrderBookLevel] = Field(description="Bid levels (highest first)")
    asks: List[OrderBookLevel] = Field(description="Ask levels (lowest first)")
    sequence: Optional[int] = Field(default=None, description="Sequence number")

    @field_validator("bids")
    @classmethod
    def validate_bids(cls, v: List[OrderBookLevel]) -> List[OrderBookLevel]:
        """Validate bid levels are sorted highest first."""
        if len(v) > 1:
            for i in range(1, len(v)):
                if v[i].price > v[i - 1].price:
                    raise ValueError("Bid levels must be sorted highest first")
        return v

    @field_validator("asks")
    @classmethod
    def validate_asks(cls, v: List[OrderBookLevel]) -> List[OrderBookLevel]:
        """Validate ask levels are sorted lowest first."""
        if len(v) > 1:
            for i in range(1, len(v)):
                if v[i].price < v[i - 1].price:
                    raise ValueError("Ask levels must be sorted lowest first")
        return v

    def model_post_init(self, __context) -> None:
        """Validate order book consistency after initialization."""
        if self.bids and self.asks:
            best_bid = self.bids[0].price
            best_ask = self.asks[0].price
            if best_bid >= best_ask:
                raise ValueError("Best bid must be less than best ask")

    @property
    def best_bid(self) -> Optional[OrderBookLevel]:
        """Get the best (highest) bid level."""
        return self.bids[0] if self.bids else None

    @property
    def best_ask(self) -> Optional[OrderBookLevel]:
        """Get the best (lowest) ask level."""
        return self.asks[0] if self.asks else None

    @property
    def spread(self) -> Optional[Decimal]:
        """Calculate bid-ask spread."""
        best_bid = self.best_bid
        best_ask = self.best_ask
        if best_bid and best_ask:
            return best_ask.price - best_bid.price
        return None

    @property
    def mid_price(self) -> Optional[Decimal]:
        """Calculate mid price between best bid and ask."""
        best_bid = self.best_bid
        best_ask = self.best_ask
        if best_bid and best_ask:
            return (best_bid.price + best_ask.price) / 2
        return None

    @property
    def spread_bps(self) -> Optional[Decimal]:
        """Calculate spread in basis points."""
        spread = self.spread
        mid = self.mid_price
        if spread and mid and mid > 0:
            return (spread / mid) * 10000
        return None

    def total_bid_size(self, levels: int = None) -> Decimal:
        """Calculate total bid size for given number of levels."""
        levels = levels or len(self.bids)
        return sum(level.size for level in self.bids[:levels])

    def total_ask_size(self, levels: int = None) -> Decimal:
        """Calculate total ask size for given number of levels."""
        levels = levels or len(self.asks)
        return sum(level.size for level in self.asks[:levels])

    def imbalance_ratio(self, levels: int = 5) -> Optional[Decimal]:
        """
        Calculate order book imbalance ratio.

        Ratio > 1 means more bid volume (bullish pressure)
        Ratio < 1 means more ask volume (bearish pressure)
        """
        bid_volume = self.total_bid_size(levels)
        ask_volume = self.total_ask_size(levels)
        if ask_volume > 0:
            return bid_volume / ask_volume
        return None


class MarketEvent(VoyagerBaseModel):
    """
    General market event (news, announcements, etc.).

    Represents market-moving events that may impact trading decisions.
    """

    symbol: Optional[Symbol] = Field(
        default=None, description="Related symbol (if applicable)"
    )
    timestamp: datetime = Field(description="Event timestamp")
    event_type: str = Field(description="Type of market event")
    title: str = Field(description="Event title")
    description: Optional[str] = Field(default=None, description="Event description")
    source: str = Field(description="Event source")
    importance: str = Field(description="Event importance (low, medium, high)")
    tags: List[str] = Field(default_factory=list, description="Event tags")
    metadata: Dict[str, str] = Field(
        default_factory=dict, description="Additional metadata"
    )

    @field_validator("importance")
    @classmethod
    def validate_importance(cls, v: str) -> str:
        """Validate importance level."""
        valid_levels = {"low", "medium", "high"}
        v_lower = v.lower()
        if v_lower not in valid_levels:
            raise ValueError(f"Importance must be one of: {valid_levels}")
        return v_lower

    @property
    def is_high_importance(self) -> bool:
        """Check if this is a high importance event."""
        return self.importance == "high"


class MarketSession(VoyagerBaseModel):
    """
    Market session information.

    Represents trading session details including open/close times
    and session characteristics.
    """

    exchange: str = Field(description="Exchange identifier")
    session_type: str = Field(description="Session type (pre, regular, post)")
    start_time: datetime = Field(description="Session start time")
    end_time: datetime = Field(description="Session end time")
    timezone: str = Field(description="Session timezone")
    is_active: bool = Field(description="Whether session is currently active")
    volume: Optional[Decimal] = Field(default=None, description="Session volume")

    @property
    def duration_minutes(self) -> int:
        """Calculate session duration in minutes."""
        delta = self.end_time - self.start_time
        return int(delta.total_seconds() / 60)

    @property
    def is_regular_session(self) -> bool:
        """Check if this is a regular trading session."""
        return self.session_type.lower() == "regular"

    @property
    def is_extended_hours(self) -> bool:
        """Check if this is extended hours trading."""
        return self.session_type.lower() in {"pre", "post"}


class MarketSnapshot(VoyagerBaseModel):
    """
    Complete market snapshot at a point in time.

    Aggregates multiple types of market data for comprehensive
    market state representation.
    """

    timestamp: datetime = Field(description="Snapshot timestamp")
    symbols: List[Symbol] = Field(description="Symbols included in snapshot")
    ohlcv_data: Dict[str, OHLCV] = Field(description="OHLCV data by symbol")
    order_books: Dict[str, OrderBook] = Field(description="Order books by symbol")
    last_trades: Dict[str, TickData] = Field(description="Last trade by symbol")
    market_events: List[MarketEvent] = Field(description="Recent market events")
    session_info: Dict[str, MarketSession] = Field(
        description="Session info by exchange"
    )

    @property
    def symbol_count(self) -> int:
        """Get number of symbols in snapshot."""
        return len(self.symbols)

    @property
    def active_sessions(self) -> List[MarketSession]:
        """Get list of active trading sessions."""
        return [session for session in self.session_info.values() if session.is_active]

    @property
    def high_importance_events(self) -> List[MarketEvent]:
        """Get high importance market events."""
        return [event for event in self.market_events if event.is_high_importance]

    def get_symbol_data(self, symbol: str) -> Dict[str, any]:
        """Get all data for a specific symbol."""
        return {
            "ohlcv": self.ohlcv_data.get(symbol),
            "order_book": self.order_books.get(symbol),
            "last_trade": self.last_trades.get(symbol),
        }

    def get_market_status(self) -> str:
        """Get overall market status based on active sessions."""
        active_count = len(self.active_sessions)
        total_count = len(self.session_info)

        if active_count == 0:
            return "closed"
        elif active_count == total_count:
            return "open"
        else:
            return "partial"
