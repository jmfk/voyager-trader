"""Centralized type definitions for market data module."""

from typing import Union

from ..models.types import AssetClass
from ..models.types import Symbol as SymbolModel

# For market data operations, we use string symbols for simplicity and performance
# When we need full Symbol objects, we convert using create_symbol()
Symbol = str


def create_symbol(
    code: str, asset_class: AssetClass = AssetClass.EQUITY
) -> SymbolModel:
    """
    Create a Symbol object from a string code.

    Args:
        code: The symbol code (e.g., "AAPL", "BTC-USD")
        asset_class: The asset class, defaults to EQUITY

    Returns:
        A properly constructed Symbol object
    """
    return SymbolModel(code=code, asset_class=asset_class)


def normalize_symbol(symbol: Union[str, SymbolModel]) -> str:
    """
    Normalize a symbol to string format.

    Args:
        symbol: Either a string or Symbol object

    Returns:
        The symbol as a string
    """
    if isinstance(symbol, SymbolModel):
        return symbol.code
    return symbol


def ensure_symbol_object(symbol: Union[str, SymbolModel]) -> SymbolModel:
    """
    Ensure we have a Symbol object.

    Args:
        symbol: Either a string or Symbol object

    Returns:
        A Symbol object
    """
    if isinstance(symbol, str):
        return create_symbol(symbol)
    return symbol
