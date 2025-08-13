"""Data validation and quality assurance for market data."""

import logging
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional

from ..models.market import OHLCV, OrderBook, TickData

logger = logging.getLogger(__name__)


class DataValidationError(Exception):
    """Exception raised for data validation failures."""


class DataValidator:
    """Validates market data quality and detects anomalies."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

        # Default validation thresholds
        self.max_price_change_percent = self.config.get(
            "max_price_change_percent", 50.0
        )
        self.max_volume_multiplier = self.config.get("max_volume_multiplier", 100.0)
        self.min_price = self.config.get("min_price", 0.01)
        self.max_price = self.config.get("max_price", 1000000.0)
        self.max_spread_percent = self.config.get("max_spread_percent", 10.0)

    def validate_ohlcv(self, data: OHLCV, previous: Optional[OHLCV] = None) -> bool:
        """
        Validate OHLCV data for quality and consistency.

        Args:
            data: OHLCV data to validate
            previous: Previous OHLCV bar for comparison

        Returns:
            True if data is valid

        Raises:
            DataValidationError: If validation fails
        """
        try:
            # Basic price validation
            self._validate_price_range(data.open, "open")
            self._validate_price_range(data.high, "high")
            self._validate_price_range(data.low, "low")
            self._validate_price_range(data.close, "close")

            # Volume validation
            if data.volume < 0:
                raise DataValidationError("Volume cannot be negative")

            # OHLC relationship validation (already in model)
            # This is redundant but adds extra safety
            if not (data.low <= data.open <= data.high):
                raise DataValidationError("Open price not within high-low range")
            if not (data.low <= data.close <= data.high):
                raise DataValidationError("Close price not within high-low range")

            # Comparison with previous bar if available
            if previous:
                self._validate_price_continuity(data, previous)
                self._validate_volume_continuity(data, previous)

            # Timestamp validation
            self._validate_timestamp(data.timestamp)

            return True

        except DataValidationError:
            logger.warning(
                f"OHLCV validation failed for {data.symbol} at {data.timestamp}"
            )
            raise

    def validate_tick_data(
        self, data: TickData, previous: Optional[TickData] = None
    ) -> bool:
        """
        Validate tick data for quality and consistency.

        Args:
            data: Tick data to validate
            previous: Previous tick for comparison

        Returns:
            True if data is valid

        Raises:
            DataValidationError: If validation fails
        """
        try:
            # Basic price validation
            self._validate_price_range(data.price, "price")

            # Size validation
            if data.size < 0:
                raise DataValidationError("Tick size cannot be negative")

            # Timestamp validation
            self._validate_timestamp(data.timestamp)

            # Sequence validation with previous tick
            if previous:
                if data.timestamp < previous.timestamp:
                    raise DataValidationError("Tick timestamp is older than previous")

                # Check for unrealistic price jumps
                price_change_percent = abs(
                    (data.price - previous.price) / previous.price * 100
                )
                if price_change_percent > self.max_price_change_percent:
                    raise DataValidationError(
                        f"Price change {price_change_percent:.2f}% exceeds threshold"
                    )

            return True

        except DataValidationError:
            logger.warning(
                f"Tick data validation failed for {data.symbol} at {data.timestamp}"
            )
            raise

    def validate_order_book(self, data: OrderBook) -> bool:
        """
        Validate order book data for quality and consistency.

        Args:
            data: Order book to validate

        Returns:
            True if data is valid

        Raises:
            DataValidationError: If validation fails
        """
        try:
            # Timestamp validation
            self._validate_timestamp(data.timestamp)

            # Bid/ask level validation
            if not data.bids and not data.asks:
                raise DataValidationError("Order book has no bid or ask levels")

            # Spread validation
            spread = data.spread
            if spread and data.mid_price:
                spread_percent = (spread / data.mid_price) * 100
                if spread_percent > self.max_spread_percent:
                    raise DataValidationError(
                        f"Spread {spread_percent:.2f}% exceeds threshold"
                    )

            # Validate individual levels
            for i, bid in enumerate(data.bids):
                self._validate_price_range(bid.price, f"bid[{i}].price")
                if bid.size <= 0:
                    raise DataValidationError(f"Bid level {i} has invalid size")

            for i, ask in enumerate(data.asks):
                self._validate_price_range(ask.price, f"ask[{i}].price")
                if ask.size <= 0:
                    raise DataValidationError(f"Ask level {i} has invalid size")

            return True

        except DataValidationError:
            logger.warning(
                f"Order book validation failed for {data.symbol} at {data.timestamp}"
            )
            raise

    def validate_batch_ohlcv(self, data_list: List[OHLCV]) -> List[OHLCV]:
        """
        Validate a batch of OHLCV data and return valid entries.

        Args:
            data_list: List of OHLCV data to validate

        Returns:
            List of valid OHLCV entries
        """
        valid_data = []
        previous = None

        for i, data in enumerate(data_list):
            try:
                self.validate_ohlcv(data, previous)
                valid_data.append(data)
                previous = data
            except DataValidationError as e:
                logger.warning(f"Skipping invalid OHLCV entry {i}: {e}")
                continue

        return valid_data

    def _validate_price_range(self, price: Decimal, field_name: str) -> None:
        """Validate price is within reasonable range."""
        if price < self.min_price:
            raise DataValidationError(
                f"{field_name} {price} below minimum {self.min_price}"
            )
        if price > self.max_price:
            raise DataValidationError(
                f"{field_name} {price} above maximum {self.max_price}"
            )

    def _validate_price_continuity(self, current: OHLCV, previous: OHLCV) -> None:
        """Validate price continuity between bars."""
        # Check for unrealistic price jumps
        price_fields = ["open", "high", "low", "close"]

        for field in price_fields:
            current_price = getattr(current, field)
            previous_close = previous.close

            if previous_close > 0:
                change_percent = abs(
                    (current_price - previous_close) / previous_close * 100
                )
                if change_percent > self.max_price_change_percent:
                    raise DataValidationError(
                        f"{field} price change {change_percent:.2f}% exceeds threshold"
                    )

    def _validate_volume_continuity(self, current: OHLCV, previous: OHLCV) -> None:
        """Validate volume continuity between bars."""
        if previous.volume > 0:
            volume_ratio = current.volume / previous.volume
            if volume_ratio > self.max_volume_multiplier:
                raise DataValidationError(
                    f"Volume ratio {volume_ratio:.2f} exceeds threshold"
                )

    def _validate_timestamp(self, timestamp: datetime) -> None:
        """Validate timestamp is reasonable."""
        now = datetime.now(timezone.utc)

        # Check if timestamp is too far in the past (more than 10 years)
        if timestamp < now - timedelta(days=3650):
            raise DataValidationError(f"Timestamp {timestamp} too far in past")

        # Check if timestamp is in the future (more than 1 day)
        if timestamp > now + timedelta(days=1):
            raise DataValidationError(f"Timestamp {timestamp} too far in future")

    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics and configuration."""
        return {
            "config": {
                "max_price_change_percent": self.max_price_change_percent,
                "max_volume_multiplier": self.max_volume_multiplier,
                "min_price": float(self.min_price),
                "max_price": float(self.max_price),
                "max_spread_percent": self.max_spread_percent,
            }
        }
