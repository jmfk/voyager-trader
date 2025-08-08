"""Data normalization for converting provider-specific formats to standard models."""

import logging
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

from ..models.market import OHLCV, OrderBook, OrderBookLevel, TickData
from ..models.types import AssetClass
from ..models.types import Symbol
from ..models.types import Symbol as SymbolModel
from ..models.types import TimeFrame


def create_symbol(code: str) -> SymbolModel:
    """Create a Symbol object from a string code."""
    return SymbolModel(code=code, asset_class=AssetClass.EQUITY)


logger = logging.getLogger(__name__)


class DataNormalizer:
    """Normalizes data from different providers into standard models."""

    def __init__(self):
        self._symbol_mappings: Dict[str, Dict[str, str]] = {}

    def normalize_ohlcv_from_alpha_vantage(
        self,
        symbol: Symbol,
        timeframe: TimeFrame,
        raw_data: Dict[str, Any],
    ) -> List[OHLCV]:
        """
        Normalize Alpha Vantage OHLCV data.

        Alpha Vantage format:
        {
            "Time Series (Daily)": {
                "2023-01-01": {
                    "1. open": "100.00",
                    "2. high": "105.00",
                    "3. low": "99.00",
                    "4. close": "103.00",
                    "5. volume": "1000000"
                }
            }
        }
        """
        normalized_data = []

        # Find the time series key (varies by timeframe)
        time_series_key = self._find_time_series_key(raw_data)
        if not time_series_key:
            logger.warning("No time series data found in Alpha Vantage response")
            return normalized_data

        time_series = raw_data[time_series_key]

        for timestamp_str, ohlcv_data in time_series.items():
            try:
                # Parse timestamp
                timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))

                # Extract OHLCV values
                ohlcv = OHLCV(
                    symbol=create_symbol(symbol),
                    timestamp=timestamp,
                    timeframe=timeframe,
                    open=Decimal(ohlcv_data["1. open"]),
                    high=Decimal(ohlcv_data["2. high"]),
                    low=Decimal(ohlcv_data["3. low"]),
                    close=Decimal(ohlcv_data["4. close"]),
                    volume=Decimal(ohlcv_data["5. volume"]),
                )

                normalized_data.append(ohlcv)

            except (KeyError, ValueError) as e:
                logger.warning(
                    f"Failed to normalize Alpha Vantage data for {timestamp_str}: {e}"
                )
                continue

        # Sort by timestamp (oldest first)
        normalized_data.sort(key=lambda x: x.timestamp)
        return normalized_data

    def normalize_ohlcv_from_yahoo(
        self,
        symbol: Symbol,
        timeframe: TimeFrame,
        raw_data: Any,
    ) -> List[OHLCV]:
        """
        Normalize Yahoo Finance OHLCV data.

        Yahoo Finance typically returns pandas DataFrame or similar structure
        with columns: Open, High, Low, Close, Volume
        """
        normalized_data = []

        try:
            # Assuming raw_data is a pandas DataFrame-like object
            if hasattr(raw_data, "iterrows"):
                for timestamp, row in raw_data.iterrows():
                    try:
                        ohlcv = OHLCV(
                            symbol=create_symbol(symbol),
                            timestamp=(
                                timestamp.to_pydatetime()
                                if hasattr(timestamp, "to_pydatetime")
                                else timestamp
                            ),
                            timeframe=timeframe,
                            open=Decimal(str(row["Open"])),
                            high=Decimal(str(row["High"])),
                            low=Decimal(str(row["Low"])),
                            close=Decimal(str(row["Close"])),
                            volume=Decimal(str(row["Volume"])),
                        )
                        normalized_data.append(ohlcv)
                    except (KeyError, ValueError) as e:
                        logger.warning(
                            f"Failed to normalize Yahoo data for {timestamp}: {e}"
                        )
                        continue
            else:
                logger.warning("Yahoo Finance data is not in expected format")

        except Exception as e:
            logger.error(f"Error normalizing Yahoo Finance data: {e}")

        return normalized_data

    def normalize_tick_from_generic(
        self,
        symbol: Symbol,
        raw_data: Dict[str, Any],
    ) -> Optional[TickData]:
        """
        Normalize generic tick data format.

        Expected format:
        {
            "timestamp": "2023-01-01T12:00:00Z",
            "price": "100.00",
            "size": "1000",
            "type": "trade",
            "exchange": "NYSE"
        }
        """
        try:
            timestamp_str = raw_data["timestamp"]
            timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))

            tick_data = TickData(
                symbol=create_symbol(symbol),
                timestamp=timestamp,
                price=Decimal(str(raw_data["price"])),
                size=Decimal(str(raw_data["size"])),
                tick_type=raw_data.get("type", "trade"),
                exchange=raw_data.get("exchange"),
                conditions=raw_data.get("conditions"),
            )

            return tick_data

        except (KeyError, ValueError) as e:
            logger.warning(f"Failed to normalize tick data: {e}")
            return None

    def normalize_order_book_from_generic(
        self,
        symbol: Symbol,
        raw_data: Dict[str, Any],
    ) -> Optional[OrderBook]:
        """
        Normalize generic order book data.

        Expected format:
        {
            "timestamp": "2023-01-01T12:00:00Z",
            "bids": [
                {"price": "99.99", "size": "1000"},
                {"price": "99.98", "size": "2000"}
            ],
            "asks": [
                {"price": "100.01", "size": "1500"},
                {"price": "100.02", "size": "1000"}
            ]
        }
        """
        try:
            timestamp_str = raw_data["timestamp"]
            timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))

            # Normalize bid levels
            bids = []
            for bid_data in raw_data.get("bids", []):
                bid = OrderBookLevel(
                    price=Decimal(str(bid_data["price"])),
                    size=Decimal(str(bid_data["size"])),
                    orders_count=bid_data.get("orders_count"),
                )
                bids.append(bid)

            # Normalize ask levels
            asks = []
            for ask_data in raw_data.get("asks", []):
                ask = OrderBookLevel(
                    price=Decimal(str(ask_data["price"])),
                    size=Decimal(str(ask_data["size"])),
                    orders_count=ask_data.get("orders_count"),
                )
                asks.append(ask)

            order_book = OrderBook(
                symbol=create_symbol(symbol),
                timestamp=timestamp,
                bids=bids,
                asks=asks,
                sequence=raw_data.get("sequence"),
            )

            return order_book

        except (KeyError, ValueError) as e:
            logger.warning(f"Failed to normalize order book data: {e}")
            return None

    def set_symbol_mapping(self, provider: str, mappings: Dict[str, str]) -> None:
        """
        Set symbol mappings for a provider.

        Args:
            provider: Provider name
            mappings: Dict mapping standard symbols to provider-specific symbols
        """
        self._symbol_mappings[provider] = mappings

    def map_symbol_to_provider(self, provider: str, symbol: Symbol) -> str:
        """
        Map standard symbol to provider-specific format.

        Args:
            provider: Provider name
            symbol: Standard symbol

        Returns:
            Provider-specific symbol
        """
        if provider in self._symbol_mappings:
            return self._symbol_mappings[provider].get(symbol, symbol)
        return symbol

    def map_symbol_from_provider(self, provider: str, provider_symbol: str) -> Symbol:
        """
        Map provider-specific symbol to standard format.

        Args:
            provider: Provider name
            provider_symbol: Provider-specific symbol

        Returns:
            Standard symbol
        """
        if provider in self._symbol_mappings:
            # Reverse lookup
            for standard, provider_specific in self._symbol_mappings[provider].items():
                if provider_specific == provider_symbol:
                    return standard
        return provider_symbol

    def _find_time_series_key(self, data: Dict[str, Any]) -> Optional[str]:
        """Find the time series key in Alpha Vantage response."""
        possible_keys = [
            "Time Series (Daily)",
            "Time Series (1min)",
            "Time Series (5min)",
            "Time Series (15min)",
            "Time Series (30min)",
            "Time Series (60min)",
            "Time Series (Intraday)",
            "Weekly Time Series",
            "Monthly Time Series",
        ]

        for key in possible_keys:
            if key in data:
                return key

        # Fallback: find any key that contains "Time Series"
        for key in data.keys():
            if "Time Series" in key:
                return key

        return None

    def get_supported_providers(self) -> List[str]:
        """Get list of supported provider formats."""
        return [
            "alpha_vantage",
            "yahoo_finance",
            "generic",
        ]
