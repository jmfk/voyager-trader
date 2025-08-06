"""Tests for market data models."""

from datetime import datetime
from decimal import Decimal

import pytest

from voyager_trader.models.market import (
    OHLCV,
    MarketEvent,
    MarketSession,
    OrderBook,
    OrderBookLevel,
    TickData,
)
from voyager_trader.models.types import AssetClass, Symbol, TimeFrame


class TestOHLCV:
    """Test OHLCV market data model."""

    def create_test_symbol(self) -> Symbol:
        """Create a test symbol."""
        return Symbol(code="AAPL", asset_class=AssetClass.EQUITY)

    def create_valid_ohlcv(self) -> OHLCV:
        """Create a valid OHLCV instance."""
        return OHLCV(
            symbol=self.create_test_symbol(),
            timestamp=datetime(2025, 1, 1, 9, 30),
            timeframe=TimeFrame.MINUTE_1,
            open=Decimal("150.00"),
            high=Decimal("152.50"),
            low=Decimal("149.50"),
            close=Decimal("151.00"),
            volume=Decimal("100000"),
            trades_count=250,
            vwap=Decimal("150.75"),
        )

    def test_ohlcv_creation(self):
        """Test OHLCV creation with valid data."""
        ohlcv = self.create_valid_ohlcv()

        assert ohlcv.symbol.code == "AAPL"
        assert ohlcv.open == Decimal("150.00000000")
        assert ohlcv.high == Decimal("152.50000000")
        assert ohlcv.low == Decimal("149.50000000")
        assert ohlcv.close == Decimal("151.00000000")
        assert ohlcv.volume == Decimal("100000.00000000")

    def test_ohlcv_validation_negative_values(self):
        """Test OHLCV validation rejects negative values."""
        with pytest.raises(ValueError, match="Prices and volume must be non-negative"):
            OHLCV(
                symbol=self.create_test_symbol(),
                timestamp=datetime.now(),
                timeframe=TimeFrame.MINUTE_1,
                open=Decimal("-150"),
                high=Decimal("152"),
                low=Decimal("149"),
                close=Decimal("151"),
                volume=Decimal("100000"),
            )

    def test_ohlcv_validation_ohlc_relationships(self):
        """Test OHLCV validation of OHLC relationships."""
        # Open not between high and low
        with pytest.raises(ValueError, match="Open price must be between low and high"):
            OHLCV(
                symbol=self.create_test_symbol(),
                timestamp=datetime.now(),
                timeframe=TimeFrame.MINUTE_1,
                open=Decimal("155"),  # Above high
                high=Decimal("152"),
                low=Decimal("149"),
                close=Decimal("151"),
                volume=Decimal("100000"),
            )

        # Close not between high and low
        with pytest.raises(
            ValueError, match="Close price must be between low and high"
        ):
            OHLCV(
                symbol=self.create_test_symbol(),
                timestamp=datetime.now(),
                timeframe=TimeFrame.MINUTE_1,
                open=Decimal("150"),
                high=Decimal("152"),
                low=Decimal("149"),
                close=Decimal("148"),  # Below low
                volume=Decimal("100000"),
            )

        # Low greater than high
        with pytest.raises(ValueError, match="Low must be less than or equal to high"):
            OHLCV(
                symbol=self.create_test_symbol(),
                timestamp=datetime.now(),
                timeframe=TimeFrame.MINUTE_1,
                open=Decimal("150"),
                high=Decimal("149"),  # Lower than low
                low=Decimal("152"),
                close=Decimal("151"),
                volume=Decimal("100000"),
            )

    def test_ohlcv_calculated_properties(self):
        """Test OHLCV calculated properties."""
        ohlcv = self.create_valid_ohlcv()

        # Typical price = (H + L + C) / 3
        expected_typical = (
            Decimal("152.50") + Decimal("149.50") + Decimal("151.00")
        ) / 3
        assert ohlcv.typical_price == expected_typical

        # Price change = Close - Open
        assert ohlcv.price_change == Decimal("1.00")

        # Price change percent = ((Close - Open) / Open) * 100
        expected_pct = (Decimal("1.00") / Decimal("150.00")) * 100
        assert abs(ohlcv.price_change_percent - expected_pct) < Decimal("0.01")

        # True range = High - Low (for single bar)
        assert ohlcv.true_range == Decimal("3.00")

        # Body size = |Close - Open|
        assert ohlcv.body_size == Decimal("1.00")

        # Upper wick = High - max(Open, Close)
        assert ohlcv.upper_wick == Decimal("1.50")  # 152.50 - 151.00

        # Lower wick = min(Open, Close) - Low
        assert ohlcv.lower_wick == Decimal("0.50")  # 150.00 - 149.50

    def test_ohlcv_candlestick_patterns(self):
        """Test OHLCV candlestick pattern detection."""
        # Bullish candle (close > open)
        bullish = OHLCV(
            symbol=self.create_test_symbol(),
            timestamp=datetime.now(),
            timeframe=TimeFrame.MINUTE_1,
            open=Decimal("100"),
            high=Decimal("105"),
            low=Decimal("99"),
            close=Decimal("103"),
            volume=Decimal("1000"),
        )

        assert bullish.is_bullish
        assert not bullish.is_bearish
        assert not bullish.is_doji

        # Bearish candle (close < open)
        bearish = OHLCV(
            symbol=self.create_test_symbol(),
            timestamp=datetime.now(),
            timeframe=TimeFrame.MINUTE_1,
            open=Decimal("103"),
            high=Decimal("105"),
            low=Decimal("99"),
            close=Decimal("100"),
            volume=Decimal("1000"),
        )

        assert not bearish.is_bullish
        assert bearish.is_bearish
        assert not bearish.is_doji

        # Doji candle (open ≈ close)
        doji = OHLCV(
            symbol=self.create_test_symbol(),
            timestamp=datetime.now(),
            timeframe=TimeFrame.MINUTE_1,
            open=Decimal("100.00"),
            high=Decimal("100.10"),
            low=Decimal("99.90"),
            close=Decimal("100.01"),  # Very small body
            volume=Decimal("1000"),
        )

        assert doji.is_doji
        assert not doji.is_bullish
        assert not doji.is_bearish


class TestTickData:
    """Test TickData model."""

    def test_tick_data_creation(self):
        """Test TickData creation."""
        tick = TickData(
            symbol=Symbol(code="AAPL", asset_class=AssetClass.EQUITY),
            timestamp=datetime.now(),
            price=Decimal("150.25"),
            size=Decimal("100"),
            tick_type="trade",
            exchange="NASDAQ",
            conditions=["regular"],
        )

        assert tick.price == Decimal("150.25000000")
        assert tick.size == Decimal("100.00000000")
        assert tick.tick_type == "trade"

    def test_tick_data_validation(self):
        """Test TickData validation."""
        with pytest.raises(ValueError, match="Price and size must be non-negative"):
            TickData(
                symbol=Symbol(code="AAPL", asset_class=AssetClass.EQUITY),
                timestamp=datetime.now(),
                price=Decimal("-150"),
                size=Decimal("100"),
                tick_type="trade",
            )

    def test_tick_data_type_checks(self):
        """Test TickData type checking methods."""
        trade_tick = TickData(
            symbol=Symbol(code="AAPL", asset_class=AssetClass.EQUITY),
            timestamp=datetime.now(),
            price=Decimal("150"),
            size=Decimal("100"),
            tick_type="trade",
        )

        bid_tick = TickData(
            symbol=Symbol(code="AAPL", asset_class=AssetClass.EQUITY),
            timestamp=datetime.now(),
            price=Decimal("149.95"),
            size=Decimal("500"),
            tick_type="bid",
        )

        ask_tick = TickData(
            symbol=Symbol(code="AAPL", asset_class=AssetClass.EQUITY),
            timestamp=datetime.now(),
            price=Decimal("150.05"),
            size=Decimal("300"),
            tick_type="ask",
        )

        assert trade_tick.is_trade
        assert not trade_tick.is_bid
        assert not trade_tick.is_ask

        assert not bid_tick.is_trade
        assert bid_tick.is_bid
        assert not bid_tick.is_ask

        assert not ask_tick.is_trade
        assert not ask_tick.is_bid
        assert ask_tick.is_ask


class TestOrderBook:
    """Test OrderBook and OrderBookLevel models."""

    def create_test_order_book(self) -> OrderBook:
        """Create a test order book."""
        bids = [
            OrderBookLevel(
                price=Decimal("99.95"), size=Decimal("1000"), orders_count=5
            ),
            OrderBookLevel(
                price=Decimal("99.90"), size=Decimal("1500"), orders_count=8
            ),
            OrderBookLevel(
                price=Decimal("99.85"), size=Decimal("2000"), orders_count=10
            ),
        ]

        asks = [
            OrderBookLevel(
                price=Decimal("100.05"), size=Decimal("800"), orders_count=3
            ),
            OrderBookLevel(
                price=Decimal("100.10"), size=Decimal("1200"), orders_count=6
            ),
            OrderBookLevel(
                price=Decimal("100.15"), size=Decimal("1800"), orders_count=9
            ),
        ]

        return OrderBook(
            symbol=Symbol(code="AAPL", asset_class=AssetClass.EQUITY),
            timestamp=datetime.now(),
            bids=bids,
            asks=asks,
            sequence=12345,
        )

    def test_order_book_level_validation(self):
        """Test OrderBookLevel validation."""
        with pytest.raises(ValueError, match="Price and size must be non-negative"):
            OrderBookLevel(price=Decimal("-10"), size=Decimal("100"))

        with pytest.raises(ValueError, match="Price and size must be non-negative"):
            OrderBookLevel(price=Decimal("10"), size=Decimal("-100"))

    def test_order_book_creation(self):
        """Test OrderBook creation."""
        order_book = self.create_test_order_book()

        assert len(order_book.bids) == 3
        assert len(order_book.asks) == 3
        assert order_book.sequence == 12345

    def test_order_book_bid_sorting_validation(self):
        """Test OrderBook validates bid levels are sorted highest first."""
        # Invalid bid ordering (should be highest first)
        invalid_bids = [
            OrderBookLevel(price=Decimal("99.85"), size=Decimal("1000")),
            OrderBookLevel(
                price=Decimal("99.95"), size=Decimal("1500")
            ),  # Higher price after lower
        ]

        asks = [OrderBookLevel(price=Decimal("100.05"), size=Decimal("800"))]

        with pytest.raises(ValueError, match="Bid levels must be sorted highest first"):
            OrderBook(
                symbol=Symbol(code="AAPL", asset_class=AssetClass.EQUITY),
                timestamp=datetime.now(),
                bids=invalid_bids,
                asks=asks,
            )

    def test_order_book_ask_sorting_validation(self):
        """Test OrderBook validates ask levels are sorted lowest first."""
        bids = [OrderBookLevel(price=Decimal("99.95"), size=Decimal("1000"))]

        # Invalid ask ordering (should be lowest first)
        invalid_asks = [
            OrderBookLevel(price=Decimal("100.15"), size=Decimal("800")),
            OrderBookLevel(
                price=Decimal("100.05"), size=Decimal("1200")
            ),  # Lower price after higher
        ]

        with pytest.raises(ValueError, match="Ask levels must be sorted lowest first"):
            OrderBook(
                symbol=Symbol(code="AAPL", asset_class=AssetClass.EQUITY),
                timestamp=datetime.now(),
                bids=bids,
                asks=invalid_asks,
            )

    def test_order_book_crossed_market_validation(self):
        """Test OrderBook validates best bid < best ask."""
        # Crossed market (best bid >= best ask)
        bids = [OrderBookLevel(price=Decimal("100.10"), size=Decimal("1000"))]
        asks = [OrderBookLevel(price=Decimal("100.05"), size=Decimal("800"))]

        with pytest.raises(ValueError, match="Best bid must be less than best ask"):
            OrderBook(
                symbol=Symbol(code="AAPL", asset_class=AssetClass.EQUITY),
                timestamp=datetime.now(),
                bids=bids,
                asks=asks,
            )

    def test_order_book_properties(self):
        """Test OrderBook calculated properties."""
        order_book = self.create_test_order_book()

        # Best levels
        assert order_book.best_bid.price == Decimal("99.95000000")
        assert order_book.best_ask.price == Decimal("100.05000000")

        # Spread
        expected_spread = Decimal("100.05") - Decimal("99.95")
        assert order_book.spread == expected_spread

        # Mid price
        expected_mid = (Decimal("99.95") + Decimal("100.05")) / 2
        assert order_book.mid_price == expected_mid

        # Spread in basis points
        spread_bps = (expected_spread / expected_mid) * 10000
        assert abs(order_book.spread_bps - spread_bps) < Decimal("0.01")

    def test_order_book_size_calculations(self):
        """Test OrderBook size calculation methods."""
        order_book = self.create_test_order_book()

        # Total bid size (all levels)
        total_bid_size = Decimal("1000") + Decimal("1500") + Decimal("2000")
        assert order_book.total_bid_size() == total_bid_size

        # Total ask size (all levels)
        total_ask_size = Decimal("800") + Decimal("1200") + Decimal("1800")
        assert order_book.total_ask_size() == total_ask_size

        # Limited levels
        assert order_book.total_bid_size(2) == Decimal("2500")  # First 2 levels
        assert order_book.total_ask_size(1) == Decimal("800")  # First level only

    def test_order_book_imbalance_ratio(self):
        """Test OrderBook imbalance ratio calculation."""
        order_book = self.create_test_order_book()

        # For top 3 levels (default)
        bid_volume = Decimal("4500")  # 1000 + 1500 + 2000
        ask_volume = Decimal("3800")  # 800 + 1200 + 1800
        expected_ratio = bid_volume / ask_volume

        assert abs(order_book.imbalance_ratio() - expected_ratio) < Decimal("0.01")

    def test_empty_order_book_properties(self):
        """Test OrderBook properties with empty sides."""
        empty_book = OrderBook(
            symbol=Symbol(code="AAPL", asset_class=AssetClass.EQUITY),
            timestamp=datetime.now(),
            bids=[],
            asks=[],
        )

        assert empty_book.best_bid is None
        assert empty_book.best_ask is None
        assert empty_book.spread is None
        assert empty_book.mid_price is None
        assert empty_book.spread_bps is None
        assert empty_book.imbalance_ratio() is None


class TestMarketEvent:
    """Test MarketEvent model."""

    def test_market_event_creation(self):
        """Test MarketEvent creation."""
        event = MarketEvent(
            symbol=Symbol(code="AAPL", asset_class=AssetClass.EQUITY),
            timestamp=datetime.now(),
            event_type="earnings",
            title="Apple Q4 Earnings",
            description="Apple reports Q4 earnings after market close",
            source="Bloomberg",
            importance="high",
            tags=["earnings", "apple", "q4"],
            metadata={"sector": "technology", "market_cap": "3T"},
        )

        assert event.title == "Apple Q4 Earnings"
        assert event.importance == "high"
        assert "earnings" in event.tags
        assert event.metadata["sector"] == "technology"

    def test_market_event_importance_validation(self):
        """Test MarketEvent importance validation."""
        # Valid importance levels
        for importance in ["low", "medium", "high"]:
            event = MarketEvent(
                timestamp=datetime.now(),
                event_type="test",
                title="Test Event",
                source="Test",
                importance=importance,
            )
            assert event.importance == importance

        # Invalid importance level
        with pytest.raises(ValueError, match="Importance must be one of"):
            MarketEvent(
                timestamp=datetime.now(),
                event_type="test",
                title="Test Event",
                source="Test",
                importance="invalid",
            )

        # Case insensitive
        event = MarketEvent(
            timestamp=datetime.now(),
            event_type="test",
            title="Test Event",
            source="Test",
            importance="HIGH",
        )
        assert event.importance == "high"

    def test_market_event_properties(self):
        """Test MarketEvent property methods."""
        high_event = MarketEvent(
            timestamp=datetime.now(),
            event_type="earnings",
            title="High Impact Event",
            source="Test",
            importance="high",
        )

        low_event = MarketEvent(
            timestamp=datetime.now(),
            event_type="news",
            title="Low Impact Event",
            source="Test",
            importance="low",
        )

        assert high_event.is_high_importance
        assert not low_event.is_high_importance


class TestMarketSession:
    """Test MarketSession model."""

    def test_market_session_creation(self):
        """Test MarketSession creation."""
        start_time = datetime(2025, 1, 1, 9, 30)
        end_time = datetime(2025, 1, 1, 16, 0)

        session = MarketSession(
            exchange="NYSE",
            session_type="regular",
            start_time=start_time,
            end_time=end_time,
            timezone="America/New_York",
            is_active=True,
            volume=Decimal("1000000"),
        )

        assert session.exchange == "NYSE"
        assert session.session_type == "regular"
        assert session.is_active

    def test_market_session_duration(self):
        """Test MarketSession duration calculation."""
        start_time = datetime(2025, 1, 1, 9, 30)
        end_time = datetime(2025, 1, 1, 16, 0)  # 6.5 hours = 390 minutes

        session = MarketSession(
            exchange="NYSE",
            session_type="regular",
            start_time=start_time,
            end_time=end_time,
            timezone="America/New_York",
            is_active=True,
        )

        assert session.duration_minutes == 390

    def test_market_session_type_properties(self):
        """Test MarketSession type checking properties."""
        regular_session = MarketSession(
            exchange="NYSE",
            session_type="regular",
            start_time=datetime.now(),
            end_time=datetime.now(),
            timezone="America/New_York",
            is_active=True,
        )

        pre_session = MarketSession(
            exchange="NYSE",
            session_type="pre",
            start_time=datetime.now(),
            end_time=datetime.now(),
            timezone="America/New_York",
            is_active=True,
        )

        assert regular_session.is_regular_session
        assert not regular_session.is_extended_hours

        assert not pre_session.is_regular_session
        assert pre_session.is_extended_hours


if __name__ == "__main__":
    pytest.main([__file__])
