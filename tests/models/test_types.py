"""Tests for common types and value objects."""

from decimal import Decimal

import pytest

from voyager_trader.models.types import (
    AssetClass,
    Currency,
    Money,
    OrderSide,
    OrderStatus,
    OrderType,
    Price,
    Quantity,
    Symbol,
    TimeFrame,
)


class TestEnums:
    """Test enum definitions."""

    def test_currency_enum(self):
        """Test Currency enum values."""
        assert Currency.USD == "USD"
        assert Currency.EUR == "EUR"
        assert Currency.BTC == "BTC"
        assert len(Currency) >= 10  # Should have multiple currencies

    def test_asset_class_enum(self):
        """Test AssetClass enum values."""
        assert AssetClass.EQUITY == "equity"
        assert AssetClass.FOREX == "forex"
        assert AssetClass.CRYPTO == "crypto"

    def test_order_enums(self):
        """Test order-related enums."""
        assert OrderType.MARKET == "market"
        assert OrderType.LIMIT == "limit"

        assert OrderSide.BUY == "buy"
        assert OrderSide.SELL == "sell"

        assert OrderStatus.PENDING == "pending"
        assert OrderStatus.FILLED == "filled"

    def test_timeframe_enum(self):
        """Test TimeFrame enum values."""
        assert TimeFrame.MINUTE_1 == "1m"
        assert TimeFrame.HOUR_1 == "1h"
        assert TimeFrame.DAY_1 == "1d"


class TestMoney:
    """Test Money value object."""

    def test_money_creation(self):
        """Test Money creation and validation."""
        money = Money(amount=Decimal("100.50"), currency=Currency.USD)
        assert money.amount == Decimal("100.50")
        assert money.currency == Currency.USD

    def test_money_precision(self):
        """Test Money decimal precision handling."""
        money = Money(amount=100.123456789, currency=Currency.USD)
        assert money.amount == Decimal("100.12345679")  # Rounded to 8 decimal places

    def test_money_string_representation(self):
        """Test Money string representation."""
        money = Money(amount=Decimal("100.50"), currency=Currency.USD)
        assert str(money) == "100.50000000 USD"

    def test_money_addition(self):
        """Test Money addition."""
        money1 = Money(amount=Decimal("100"), currency=Currency.USD)
        money2 = Money(amount=Decimal("50.25"), currency=Currency.USD)
        result = money1 + money2

        assert result.amount == Decimal("150.25")
        assert result.currency == Currency.USD

    def test_money_addition_different_currencies(self):
        """Test Money addition with different currencies fails."""
        money1 = Money(amount=Decimal("100"), currency=Currency.USD)
        money2 = Money(amount=Decimal("50"), currency=Currency.EUR)

        with pytest.raises(ValueError, match="Cannot add USD and EUR"):
            money1 + money2

    def test_money_subtraction(self):
        """Test Money subtraction."""
        money1 = Money(amount=Decimal("100"), currency=Currency.USD)
        money2 = Money(amount=Decimal("25.50"), currency=Currency.USD)
        result = money1 - money2

        assert result.amount == Decimal("74.50")
        assert result.currency == Currency.USD

    def test_money_multiplication(self):
        """Test Money multiplication by factor."""
        money = Money(amount=Decimal("100"), currency=Currency.USD)
        result = money * Decimal("2.5")

        assert result.amount == Decimal("250.00000000")
        assert result.currency == Currency.USD

    def test_money_division(self):
        """Test Money division by divisor."""
        money = Money(amount=Decimal("100"), currency=Currency.USD)
        result = money / Decimal("4")

        assert result.amount == Decimal("25.00000000")
        assert result.currency == Currency.USD

    def test_money_boolean_checks(self):
        """Test Money boolean check methods."""
        positive = Money(amount=Decimal("100"), currency=Currency.USD)
        negative = Money(amount=Decimal("-50"), currency=Currency.USD)
        zero = Money(amount=Decimal("0"), currency=Currency.USD)

        assert positive.is_positive()
        assert not positive.is_negative()
        assert not positive.is_zero()

        assert not negative.is_positive()
        assert negative.is_negative()
        assert not negative.is_zero()

        assert not zero.is_positive()
        assert not zero.is_negative()
        assert zero.is_zero()

    def test_money_abs(self):
        """Test Money absolute value."""
        negative = Money(amount=Decimal("-100.50"), currency=Currency.USD)
        result = negative.abs()

        assert result.amount == Decimal("100.50000000")
        assert result.currency == Currency.USD


class TestPrice:
    """Test Price value object."""

    def test_price_creation(self):
        """Test Price creation and validation."""
        price = Price(bid=Decimal("99.50"), ask=Decimal("99.55"), currency=Currency.USD)
        assert price.bid == Decimal("99.50000000")
        assert price.ask == Decimal("99.55000000")
        assert price.currency == Currency.USD

    def test_price_validation_positive(self):
        """Test Price validation requires positive values."""
        with pytest.raises(ValueError, match="Prices must be positive"):
            Price(bid=Decimal("-10"), ask=Decimal("10"), currency=Currency.USD)

        with pytest.raises(ValueError, match="Prices must be positive"):
            Price(bid=Decimal("10"), ask=Decimal("-10"), currency=Currency.USD)

    def test_price_mid(self):
        """Test Price mid calculation."""
        price = Price(bid=Decimal("99.50"), ask=Decimal("99.60"), currency=Currency.USD)
        assert price.mid == Decimal("99.55")

    def test_price_spread(self):
        """Test Price spread calculation."""
        price = Price(bid=Decimal("99.50"), ask=Decimal("99.60"), currency=Currency.USD)
        assert price.spread == Decimal("0.10000000")

    def test_price_spread_bps(self):
        """Test Price spread in basis points."""
        price = Price(
            bid=Decimal("100.00"), ask=Decimal("100.10"), currency=Currency.USD
        )
        expected_bps = (Decimal("0.10") / Decimal("100.05")) * 10000
        assert abs(price.spread_bps - expected_bps) < Decimal("0.01")

    def test_price_string_representation(self):
        """Test Price string representation."""
        price = Price(bid=Decimal("99.50"), ask=Decimal("99.55"), currency=Currency.USD)
        assert str(price) == "99.50000000/99.55000000 USD"


class TestQuantity:
    """Test Quantity value object."""

    def test_quantity_creation(self):
        """Test Quantity creation."""
        qty = Quantity(amount=Decimal("100.5"))
        assert qty.amount == Decimal("100.50000000")

    def test_quantity_string_representation(self):
        """Test Quantity string representation."""
        qty = Quantity(amount=Decimal("100.5"))
        assert str(qty) == "100.50000000"

    def test_quantity_arithmetic(self):
        """Test Quantity arithmetic operations."""
        qty1 = Quantity(amount=Decimal("100"))
        qty2 = Quantity(amount=Decimal("50"))

        # Addition
        result = qty1 + qty2
        assert result.amount == Decimal("150.00000000")

        # Subtraction
        result = qty1 - qty2
        assert result.amount == Decimal("50.00000000")

        # Multiplication
        result = qty1 * Decimal("2")
        assert result.amount == Decimal("200.00000000")

        # Division
        result = qty1 / Decimal("4")
        assert result.amount == Decimal("25.00000000")

    def test_quantity_boolean_checks(self):
        """Test Quantity boolean methods."""
        positive = Quantity(amount=Decimal("100"))
        negative = Quantity(amount=Decimal("-50"))
        zero = Quantity(amount=Decimal("0"))

        assert positive.is_positive()
        assert not positive.is_negative()
        assert not positive.is_zero()

        assert not negative.is_positive()
        assert negative.is_negative()
        assert not negative.is_zero()

        assert not zero.is_positive()
        assert not zero.is_negative()
        assert zero.is_zero()

    def test_quantity_abs(self):
        """Test Quantity absolute value."""
        negative = Quantity(amount=Decimal("-100.5"))
        result = negative.abs()
        assert result.amount == Decimal("100.50000000")


class TestSymbol:
    """Test Symbol value object."""

    def test_symbol_creation(self):
        """Test Symbol creation."""
        symbol = Symbol(code="AAPL", exchange="NASDAQ", asset_class=AssetClass.EQUITY)
        assert symbol.code == "AAPL"
        assert symbol.exchange == "NASDAQ"
        assert symbol.asset_class == AssetClass.EQUITY

    def test_symbol_code_validation(self):
        """Test Symbol code validation and normalization."""
        symbol = Symbol(code="  aapl  ", asset_class=AssetClass.EQUITY)
        assert symbol.code == "AAPL"  # Trimmed and uppercased

        with pytest.raises(ValueError, match="Symbol code cannot be empty"):
            Symbol(code="", asset_class=AssetClass.EQUITY)

        with pytest.raises(ValueError, match="Symbol code cannot be empty"):
            Symbol(code="   ", asset_class=AssetClass.EQUITY)

    def test_symbol_string_representation(self):
        """Test Symbol string representation."""
        symbol_with_exchange = Symbol(
            code="AAPL", exchange="NASDAQ", asset_class=AssetClass.EQUITY
        )
        assert str(symbol_with_exchange) == "AAPL@NASDAQ"

        symbol_without_exchange = Symbol(code="AAPL", asset_class=AssetClass.EQUITY)
        assert str(symbol_without_exchange) == "AAPL"

    def test_symbol_forex_properties(self):
        """Test Symbol forex-specific properties."""
        forex_symbol = Symbol(
            code="EURUSD",
            asset_class=AssetClass.FOREX,
            base_currency=Currency.EUR,
            quote_currency=Currency.USD,
        )

        assert forex_symbol.is_forex
        assert not forex_symbol.is_crypto
        assert forex_symbol.base_currency == Currency.EUR
        assert forex_symbol.quote_currency == Currency.USD

    def test_symbol_crypto_properties(self):
        """Test Symbol crypto-specific properties."""
        crypto_symbol = Symbol(
            code="BTCUSD",
            asset_class=AssetClass.CRYPTO,
            base_currency=Currency.BTC,
            quote_currency=Currency.USD,
        )

        assert crypto_symbol.is_crypto
        assert not crypto_symbol.is_forex

    def test_symbol_equity_properties(self):
        """Test Symbol equity properties."""
        equity_symbol = Symbol(code="AAPL", asset_class=AssetClass.EQUITY)

        assert not equity_symbol.is_forex
        assert not equity_symbol.is_crypto


class TestValueObjectEquality:
    """Test value object equality and hashing."""

    def test_money_equality(self):
        """Test Money equality."""
        money1 = Money(amount=Decimal("100"), currency=Currency.USD)
        money2 = Money(amount=Decimal("100"), currency=Currency.USD)
        money3 = Money(amount=Decimal("100"), currency=Currency.EUR)

        assert money1 == money2
        assert money1 != money3
        assert hash(money1) == hash(money2)
        assert hash(money1) != hash(money3)

    def test_symbol_equality(self):
        """Test Symbol equality."""
        symbol1 = Symbol(code="AAPL", asset_class=AssetClass.EQUITY)
        symbol2 = Symbol(code="AAPL", asset_class=AssetClass.EQUITY)
        symbol3 = Symbol(code="MSFT", asset_class=AssetClass.EQUITY)

        assert symbol1 == symbol2
        assert symbol1 != symbol3
        assert hash(symbol1) == hash(symbol2)
        assert hash(symbol1) != hash(symbol3)

    def test_value_objects_in_collections(self):
        """Test value objects can be used in sets and as dict keys."""
        money1 = Money(amount=Decimal("100"), currency=Currency.USD)
        money2 = Money(amount=Decimal("100"), currency=Currency.USD)  # Same as money1
        money3 = Money(amount=Decimal("200"), currency=Currency.USD)

        # Set should deduplicate identical value objects
        money_set = {money1, money2, money3}
        assert len(money_set) == 2

        # Can be used as dict keys
        money_dict = {money1: "first", money2: "second", money3: "third"}
        assert len(money_dict) == 2
        assert money_dict[money1] == "second"  # money2 overwrote money1


if __name__ == "__main__":
    pytest.main([__file__])
