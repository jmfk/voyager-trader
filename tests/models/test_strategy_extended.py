"""Extended tests for strategy models to improve coverage."""

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


def test_signal_extended():
    """Extended Signal tests to improve coverage."""
    # Test signal with all optional fields
    signal = Signal(
        symbol=Symbol(code="AAPL", asset_class=AssetClass.EQUITY),
        signal_type=SignalType.BUY,
        strength=SignalStrength.STRONG,
        confidence=Decimal("90.0"),
        timestamp=datetime.utcnow(),
        rationale="Technical breakout above resistance",
        strategy_id="strategy-rsi-v2",
        expiry=datetime.utcnow().replace(hour=23, minute=59),
        target_price=Decimal("155.00"),
        stop_loss=Decimal("145.00"),
        take_profit=Decimal("165.00"),
        position_size=Decimal("5.0"),
        indicators={"rsi": 25.5, "volume": 2000000},
        metadata={"pattern": "bullish_breakout", "strength": "high"},
    )

    assert signal.expiry is not None
    assert signal.target_price == Decimal("155.00")
    assert signal.stop_loss == Decimal("145.00")
    assert signal.take_profit == Decimal("165.00")
    assert signal.position_size == Decimal("5.0")
    assert "rsi" in signal.indicators
    assert "pattern" in signal.metadata

    # Test different signal types
    sell_signal = Signal(
        symbol=Symbol(code="GOOGL", asset_class=AssetClass.EQUITY),
        signal_type=SignalType.SELL,
        strength=SignalStrength.MODERATE,
        confidence=Decimal("75.0"),
        timestamp=datetime.utcnow(),
        rationale="Overbought conditions",
        strategy_id="strategy-momentum",
    )
    assert sell_signal.signal_type == SignalType.SELL

    hold_signal = Signal(
        symbol=Symbol(code="MSFT", asset_class=AssetClass.EQUITY),
        signal_type=SignalType.HOLD,
        strength=SignalStrength.WEAK,
        confidence=Decimal("60.0"),
        timestamp=datetime.utcnow(),
        rationale="Mixed signals",
        strategy_id="strategy-mixed",
    )
    assert hold_signal.signal_type == SignalType.HOLD


def test_trading_rule_extended():
    """Extended TradingRule tests to improve coverage."""
    # Test rule with weight
    rule = TradingRule(
        name="Advanced RSI Strategy",
        description="RSI with volume confirmation",
        rule_type="entry",
        indicator_type=IndicatorType.RSI,
        operator=RuleOperator.AND,
        parameters={"rsi_period": 14, "rsi_threshold": 30, "volume_multiplier": 1.5},
        conditions=[
            {"indicator": "rsi", "operator": "<", "value": 30},
            {"indicator": "volume", "operator": ">", "value": "avg_volume * 1.5"},
        ],
        weight=Decimal("2.5"),
        is_active=True,
    )

    assert rule.weight == Decimal("2.5")
    assert rule.is_active is True

    # Test different operators
    or_rule = TradingRule(
        name="Multiple Entry Signals",
        description="RSI OR MACD signal",
        rule_type="entry",
        indicator_type=IndicatorType.RSI,
        operator=RuleOperator.OR,
        parameters={"threshold": 30},
        conditions=[
            {"indicator": "rsi", "operator": "<", "value": 30},
            {"indicator": "macd", "operator": ">", "value": 0},
        ],
    )
    assert or_rule.operator == RuleOperator.OR

    # Test different indicator types
    volume_rule = TradingRule(
        name="Volume Surge",
        description="High volume detection",
        rule_type="confirmation",
        indicator_type=IndicatorType.VOLUME,
        operator=RuleOperator.AND,
        parameters={"multiplier": 2.0},
        conditions=[
            {"indicator": "volume", "operator": ">", "value": "avg_volume * 2"}
        ],
    )
    assert volume_rule.indicator_type == IndicatorType.VOLUME


def test_strategy_extended():
    """Extended Strategy tests to improve coverage."""
    # Test strategy with version and status
    strategy = Strategy(
        name="Comprehensive RSI Strategy",
        description="Advanced RSI strategy with multiple confirmations",
        version="2.1",
        status=StrategyStatus.PAPER_TRADING,
        entry_rules=["rule-entry-1", "rule-entry-2"],
        exit_rules=["rule-exit-1", "rule-exit-2"],
        risk_rules=["rule-risk-1"],
        symbols=[
            Symbol(code="AAPL", asset_class=AssetClass.EQUITY),
            Symbol(code="GOOGL", asset_class=AssetClass.EQUITY),
            Symbol(code="MSFT", asset_class=AssetClass.EQUITY),
        ],
        timeframes=[TimeFrame.DAY_1, TimeFrame.HOUR_1],
        max_position_size=Decimal("8.0"),
        risk_per_trade=Decimal("2.0"),
    )

    assert strategy.version == "2.1"
    assert strategy.status == StrategyStatus.PAPER_TRADING
    assert len(strategy.symbols) == 3
    assert len(strategy.timeframes) == 2
    assert strategy.max_position_size == Decimal("8.0")
    assert strategy.risk_per_trade == Decimal("2.0")


def test_backtest_extended():
    """Extended Backtest tests to improve coverage."""
    # Test backtest with additional metrics
    backtest = Backtest(
        strategy_id="strategy-comprehensive",
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 12, 31),
        initial_capital=Money(amount=Decimal("100000"), currency=Currency.USD),
        final_capital=Money(amount=Decimal("125000"), currency=Currency.USD),
        symbols=[
            Symbol(code="AAPL", asset_class=AssetClass.EQUITY),
            Symbol(code="GOOGL", asset_class=AssetClass.EQUITY),
            Symbol(code="MSFT", asset_class=AssetClass.EQUITY),
        ],
        timeframes=[TimeFrame.DAY_1, TimeFrame.HOUR_4],
        total_trades=200,
        winning_trades=130,
        losing_trades=70,
        total_commission=Money(amount=Decimal("150"), currency=Currency.USD),
        metrics={
            "total_return": Decimal("25.0"),
            "annualized_return": Decimal("25.0"),
            "sharpe_ratio": Decimal("2.1"),
            "max_drawdown": Decimal("6.5"),
            "win_rate": Decimal("65.0"),
            "profit_factor": Decimal("1.8"),
            "calmar_ratio": Decimal("3.8"),
        },
    )

    assert backtest.strategy_id == "strategy-comprehensive"
    assert len(backtest.symbols) == 3
    assert len(backtest.timeframes) == 2
    assert backtest.total_trades == 200
    assert backtest.winning_trades == 130
    assert backtest.losing_trades == 70
    assert backtest.total_commission.amount == Decimal("150")
    assert len(backtest.metrics) == 7
