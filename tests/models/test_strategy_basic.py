"""Basic tests for strategy models to improve coverage."""

from datetime import datetime
from decimal import Decimal

from src.voyager_trader.models.strategy import (
    Backtest,
    IndicatorType,
    RuleOperator,
    Signal,
    Strategy,
    TradingRule,
)
from src.voyager_trader.models.types import (
    AssetClass,
    Currency,
    Money,
    SignalStrength,
    SignalType,
    StrategyStatus,
    Symbol,
    TimeFrame,
)


def test_signal_basic():
    """Basic Signal creation."""
    signal = Signal(
        symbol=Symbol(code="AAPL", asset_class=AssetClass.EQUITY),
        signal_type=SignalType.BUY,
        strength=SignalStrength.STRONG,
        confidence=Decimal("85.5"),
        timestamp=datetime.utcnow(),
        rationale="Strong momentum",
        strategy_id="strategy-123",
    )

    assert signal.symbol.code == "AAPL"
    assert signal.signal_type == SignalType.BUY
    assert signal.strength == SignalStrength.STRONG
    assert signal.confidence == Decimal("85.5")


def test_trading_rule_basic():
    """Basic TradingRule creation."""
    rule = TradingRule(
        name="RSI Oversold",
        description="Buy when RSI < 30",
        rule_type="entry",
        indicator_type=IndicatorType.RSI,
        operator=RuleOperator.AND,
        parameters={"period": 14, "threshold": 30},
        conditions=[{"indicator": "rsi", "operator": "<", "value": 30}],
    )

    assert rule.name == "RSI Oversold"
    assert rule.indicator_type == IndicatorType.RSI
    assert rule.operator == RuleOperator.AND


def test_strategy_basic():
    """Basic Strategy creation."""
    strategy = Strategy(
        name="RSI Mean Reversion",
        description="Buy oversold, sell overbought",
        entry_rules=["rule-123"],
        exit_rules=["rule-456"],
        risk_rules=[],
        symbols=[Symbol(code="AAPL", asset_class=AssetClass.EQUITY)],
        timeframes=[TimeFrame.DAY_1],
        max_position_size=Decimal("10.0"),
        risk_per_trade=Decimal("2.0"),
    )

    assert strategy.name == "RSI Mean Reversion"
    assert len(strategy.symbols) == 1
    assert len(strategy.entry_rules) == 1
    assert strategy.status == StrategyStatus.DRAFT


def test_backtest_basic():
    """Basic Backtest creation."""
    backtest = Backtest(
        strategy_id="strategy-123",
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 12, 31),
        initial_capital=Money(amount=Decimal("100000"), currency=Currency.USD),
        final_capital=Money(amount=Decimal("115000"), currency=Currency.USD),
        symbols=[Symbol(code="AAPL", asset_class=AssetClass.EQUITY)],
        timeframes=[TimeFrame.DAY_1],
        total_trades=150,
        winning_trades=95,
        losing_trades=55,
        total_commission=Money(amount=Decimal("50"), currency=Currency.USD),
        metrics={},
    )

    assert backtest.strategy_id == "strategy-123"
    assert backtest.total_trades == 150
    assert backtest.winning_trades == 95
    assert backtest.losing_trades == 55
