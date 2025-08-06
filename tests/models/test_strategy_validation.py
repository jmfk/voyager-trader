"""Validation tests for strategy models to improve coverage."""

from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from src.voyager_trader.models.strategy import (
    Backtest,
    BacktestMetric,
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


def test_signal_confidence_validation_valid():
    """Test signal confidence validation with valid values."""
    signal = Signal(
        symbol=Symbol(code="AAPL", asset_class=AssetClass.EQUITY),
        signal_type=SignalType.BUY,
        strength=SignalStrength.STRONG,
        confidence=Decimal("85.5"),
        timestamp=datetime.utcnow(),
        rationale="Strong RSI oversold signal",
        strategy_id="rsi-strategy-1",
    )

    assert signal.confidence == Decimal("85.50")


def test_signal_confidence_validation_negative():
    """Test signal confidence validation with negative value."""
    with pytest.raises(ValueError, match="Confidence must be between 0 and 100"):
        Signal(
            symbol=Symbol(code="AAPL", asset_class=AssetClass.EQUITY),
            signal_type=SignalType.BUY,
            strength=SignalStrength.WEAK,
            confidence=-10.0,  # Invalid negative
            timestamp=datetime.utcnow(),
            rationale="Test signal",
            strategy_id="test-strategy",
        )


def test_signal_confidence_validation_over_100():
    """Test signal confidence validation with value over 100."""
    with pytest.raises(ValueError, match="Confidence must be between 0 and 100"):
        Signal(
            symbol=Symbol(code="SPY", asset_class=AssetClass.EQUITY),
            signal_type=SignalType.SELL,
            strength=SignalStrength.STRONG,
            confidence=150.0,  # Invalid over 100
            timestamp=datetime.utcnow(),
            rationale="Test signal",
            strategy_id="test-strategy",
        )


def test_signal_position_size_validation_valid():
    """Test signal position size validation with valid values."""
    signal = Signal(
        symbol=Symbol(code="MSFT", asset_class=AssetClass.EQUITY),
        signal_type=SignalType.BUY,
        strength=SignalStrength.MODERATE,
        confidence=Decimal("75"),
        timestamp=datetime.utcnow(),
        position_size=Decimal("25.5"),  # Valid percentage
        rationale="Medium confidence signal",
        strategy_id="test-strategy",
    )

    assert signal.position_size == Decimal("25.50")


def test_signal_position_size_validation_negative():
    """Test signal position size validation with negative value."""
    with pytest.raises(ValueError, match="Position size must be between 0 and 100"):
        Signal(
            symbol=Symbol(code="GOOGL", asset_class=AssetClass.EQUITY),
            signal_type=SignalType.BUY,
            strength=SignalStrength.WEAK,
            confidence=Decimal("50"),
            timestamp=datetime.utcnow(),
            position_size=-5.0,  # Invalid negative
            rationale="Test signal",
            strategy_id="test-strategy",
        )


def test_signal_position_size_validation_over_100():
    """Test signal position size validation with value over 100."""
    with pytest.raises(ValueError, match="Position size must be between 0 and 100"):
        Signal(
            symbol=Symbol(code="TSLA", asset_class=AssetClass.EQUITY),
            signal_type=SignalType.SELL,
            strength=SignalStrength.STRONG,
            confidence=Decimal("90"),
            timestamp=datetime.utcnow(),
            position_size=150.0,  # Invalid over 100
            rationale="Test signal",
            strategy_id="test-strategy",
        )


def test_signal_price_validation_positive():
    """Test signal price validation with positive values."""
    signal = Signal(
        symbol=Symbol(code="AMZN", asset_class=AssetClass.EQUITY),
        signal_type=SignalType.BUY,
        strength=SignalStrength.STRONG,
        confidence=Decimal("80"),
        timestamp=datetime.utcnow(),
        target_price=Decimal("150.25"),
        stop_loss=Decimal("145.00"),
        take_profit=Decimal("160.50"),
        rationale="Price levels set",
        strategy_id="test-strategy",
    )

    assert signal.target_price == Decimal("150.25000000")
    assert signal.stop_loss == Decimal("145.00000000")
    assert signal.take_profit == Decimal("160.50000000")


def test_signal_price_validation_zero():
    """Test signal price validation with zero values."""
    with pytest.raises(ValueError, match="Prices must be positive"):
        Signal(
            symbol=Symbol(code="NFLX", asset_class=AssetClass.EQUITY),
            signal_type=SignalType.BUY,
            strength=SignalStrength.MODERATE,
            confidence=Decimal("70"),
            timestamp=datetime.utcnow(),
            target_price=0,  # Invalid zero
            rationale="Test signal",
            strategy_id="test-strategy",
        )


def test_signal_price_validation_negative():
    """Test signal price validation with negative values."""
    with pytest.raises(ValueError, match="Prices must be positive"):
        Signal(
            symbol=Symbol(code="META", asset_class=AssetClass.EQUITY),
            signal_type=SignalType.SELL,
            strength=SignalStrength.WEAK,
            confidence=Decimal("60"),
            timestamp=datetime.utcnow(),
            stop_loss=-10.0,  # Invalid negative
            rationale="Test signal",
            strategy_id="test-strategy",
        )


def test_signal_is_buy_signal():
    """Test signal buy signal detection."""
    buy_signal = Signal(
        symbol=Symbol(code="AAPL", asset_class=AssetClass.EQUITY),
        signal_type=SignalType.BUY,
        strength=SignalStrength.STRONG,
        confidence=Decimal("85"),
        timestamp=datetime.utcnow(),
        rationale="Buy signal test",
        strategy_id="test-strategy",
    )

    increase_signal = Signal(
        symbol=Symbol(code="AAPL", asset_class=AssetClass.EQUITY),
        signal_type=SignalType.INCREASE_POSITION,
        strength=SignalStrength.MODERATE,
        confidence=Decimal("75"),
        timestamp=datetime.utcnow(),
        rationale="Increase position test",
        strategy_id="test-strategy",
    )

    assert buy_signal.is_buy_signal is True
    assert increase_signal.is_buy_signal is True


def test_signal_is_sell_signal():
    """Test signal sell signal detection."""
    sell_signal = Signal(
        symbol=Symbol(code="GOOGL", asset_class=AssetClass.EQUITY),
        signal_type=SignalType.SELL,
        strength=SignalStrength.STRONG,
        confidence=Decimal("90"),
        timestamp=datetime.utcnow(),
        rationale="Sell signal test",
        strategy_id="test-strategy",
    )

    exit_long_signal = Signal(
        symbol=Symbol(code="MSFT", asset_class=AssetClass.EQUITY),
        signal_type=SignalType.EXIT_LONG,
        strength=SignalStrength.MODERATE,
        confidence=Decimal("70"),
        timestamp=datetime.utcnow(),
        rationale="Exit long test",
        strategy_id="test-strategy",
    )

    assert sell_signal.is_sell_signal is True
    assert exit_long_signal.is_sell_signal is True


def test_signal_expiry_handling():
    """Test signal expiry functionality."""
    # Not expired signal
    future_time = datetime.utcnow() + timedelta(hours=1)
    active_signal = Signal(
        symbol=Symbol(code="TSLA", asset_class=AssetClass.EQUITY),
        signal_type=SignalType.BUY,
        strength=SignalStrength.STRONG,
        confidence=Decimal("85"),
        timestamp=datetime.utcnow(),
        expiry=future_time,
        rationale="Active signal",
        strategy_id="test-strategy",
    )

    # Expired signal
    past_time = datetime.utcnow() - timedelta(hours=1)
    expired_signal = Signal(
        symbol=Symbol(code="AMZN", asset_class=AssetClass.EQUITY),
        signal_type=SignalType.SELL,
        strength=SignalStrength.MODERATE,
        confidence=Decimal("75"),
        timestamp=datetime.utcnow(),
        expiry=past_time,
        rationale="Expired signal",
        strategy_id="test-strategy",
    )

    # No expiry signal
    no_expiry_signal = Signal(
        symbol=Symbol(code="NFLX", asset_class=AssetClass.EQUITY),
        signal_type=SignalType.BUY,
        strength=SignalStrength.WEAK,
        confidence=Decimal("60"),
        timestamp=datetime.utcnow(),
        rationale="No expiry signal",
        strategy_id="test-strategy",
    )

    assert active_signal.is_expired is False
    assert expired_signal.is_expired is True
    assert no_expiry_signal.is_expired is False


def test_signal_high_confidence():
    """Test signal high confidence detection."""
    high_confidence = Signal(
        symbol=Symbol(code="META", asset_class=AssetClass.EQUITY),
        signal_type=SignalType.BUY,
        strength=SignalStrength.STRONG,
        confidence=Decimal("85"),  # > 75
        timestamp=datetime.utcnow(),
        rationale="High confidence signal",
        strategy_id="test-strategy",
    )

    low_confidence = Signal(
        symbol=Symbol(code="NVDA", asset_class=AssetClass.EQUITY),
        signal_type=SignalType.SELL,
        strength=SignalStrength.WEAK,
        confidence=Decimal("65"),  # <= 75
        timestamp=datetime.utcnow(),
        rationale="Low confidence signal",
        strategy_id="test-strategy",
    )

    assert high_confidence.is_high_confidence is True
    assert low_confidence.is_high_confidence is False


def test_signal_risk_reward_ratio():
    """Test signal risk/reward ratio calculation."""
    # Valid risk/reward calculation
    signal_with_levels = Signal(
        symbol=Symbol(code="SPY", asset_class=AssetClass.EQUITY),
        signal_type=SignalType.BUY,
        strength=SignalStrength.STRONG,
        confidence=Decimal("80"),
        timestamp=datetime.utcnow(),
        target_price=Decimal("400.00"),
        stop_loss=Decimal("390.00"),  # Risk: 10
        take_profit=Decimal("420.00"),  # Reward: 20
        rationale="Risk/reward test",
        strategy_id="test-strategy",
    )

    # Missing price levels
    signal_incomplete = Signal(
        symbol=Symbol(code="QQQ", asset_class=AssetClass.EQUITY),
        signal_type=SignalType.BUY,
        strength=SignalStrength.MODERATE,
        confidence=Decimal("70"),
        timestamp=datetime.utcnow(),
        target_price=Decimal("350.00"),
        # Missing stop_loss and take_profit
        rationale="Incomplete levels",
        strategy_id="test-strategy",
    )

    # Zero risk scenario
    signal_zero_risk = Signal(
        symbol=Symbol(code="IWM", asset_class=AssetClass.EQUITY),
        signal_type=SignalType.BUY,
        strength=SignalStrength.STRONG,
        confidence=Decimal("85"),
        timestamp=datetime.utcnow(),
        target_price=Decimal("200.00"),
        stop_loss=Decimal("200.00"),  # Same as target = zero risk
        take_profit=Decimal("210.00"),
        rationale="Zero risk test",
        strategy_id="test-strategy",
    )

    assert signal_with_levels.risk_reward_ratio == Decimal("2.0")  # 20/10
    assert signal_incomplete.risk_reward_ratio is None
    assert signal_zero_risk.risk_reward_ratio is None


def test_trading_rule_weight_validation_positive():
    """Test trading rule weight validation with positive values."""
    rule = TradingRule(
        name="RSI Oversold Rule",
        description="Buy when RSI < 30",
        rule_type="entry",
        indicator_type=IndicatorType.RSI,
        operator=RuleOperator.LESS_THAN,
        parameters={"threshold": 30},
        conditions=[{"indicator": "RSI", "operator": "<", "value": 30}],
        weight=Decimal("2.5"),
    )

    assert rule.weight == Decimal("2.50")


def test_trading_rule_weight_validation_negative():
    """Test trading rule weight validation with negative values."""
    with pytest.raises(ValueError, match="Rule weight must be non-negative"):
        TradingRule(
            name="Invalid Weight Rule",
            description="Test negative weight",
            rule_type="entry",
            indicator_type=IndicatorType.SMA,
            operator=RuleOperator.GREATER_THAN,
            parameters={"period": 20},
            conditions=[{"indicator": "SMA", "operator": ">", "value": 100}],
            weight=Decimal("-1.0"),  # Invalid negative
        )


def test_trading_rule_type_properties():
    """Test trading rule type detection properties."""
    entry_rule = TradingRule(
        name="Entry Rule",
        description="Entry condition",
        rule_type="entry",
        indicator_type=IndicatorType.MACD,
        operator=RuleOperator.GREATER_THAN,
        parameters={"fast": 12, "slow": 26},
        conditions=[{"indicator": "MACD", "operator": ">", "value": 0}],
    )

    exit_rule = TradingRule(
        name="Exit Rule",
        description="Exit condition",
        rule_type="exit",
        indicator_type=IndicatorType.RSI,
        operator=RuleOperator.GREATER_THAN,
        parameters={"period": 14},
        conditions=[{"indicator": "RSI", "operator": ">", "value": 70}],
    )

    risk_rule = TradingRule(
        name="Risk Rule",
        description="Risk management",
        rule_type="risk",
        indicator_type=IndicatorType.ATR,
        operator=RuleOperator.GREATER_THAN,
        parameters={"period": 14},
        conditions=[{"indicator": "ATR", "operator": ">", "value": 2.0}],
    )

    assert entry_rule.is_entry_rule is True
    assert entry_rule.is_exit_rule is False
    assert entry_rule.is_risk_rule is False

    assert exit_rule.is_entry_rule is False
    assert exit_rule.is_exit_rule is True
    assert exit_rule.is_risk_rule is False

    assert risk_rule.is_entry_rule is False
    assert risk_rule.is_exit_rule is False
    assert risk_rule.is_risk_rule is True


def test_trading_rule_signal_contribution():
    """Test trading rule signal contribution calculation."""
    active_rule = TradingRule(
        name="Active Rule",
        description="Active rule test",
        rule_type="entry",
        indicator_type=IndicatorType.RSI,
        operator=RuleOperator.LESS_THAN,
        parameters={"threshold": 30},
        conditions=[{"indicator": "RSI", "operator": "<", "value": 30}],
        weight=Decimal("1.5"),
        is_active=True,
    )

    inactive_rule = TradingRule(
        name="Inactive Rule",
        description="Inactive rule test",
        rule_type="entry",
        indicator_type=IndicatorType.SMA,
        operator=RuleOperator.GREATER_THAN,
        parameters={"period": 20},
        conditions=[{"indicator": "SMA", "operator": ">", "value": 100}],
        weight=Decimal("2.0"),
        is_active=False,
    )

    market_data = {"RSI": 25, "SMA": 105}

    # Active rule should return its weight when condition is met
    assert active_rule.get_signal_contribution(market_data) == Decimal("1.5")

    # Inactive rule should return 0 regardless of conditions
    assert inactive_rule.get_signal_contribution(market_data) == Decimal("0")


def test_strategy_percentage_validation_valid():
    """Test strategy percentage validation with valid values."""
    strategy = Strategy(
        name="Test Strategy",
        description="Strategy with valid percentages",
        entry_rules=["rule1"],
        exit_rules=["rule2"],
        risk_rules=["rule3"],
        symbols=[Symbol(code="AAPL", asset_class=AssetClass.EQUITY)],
        timeframes=[TimeFrame.DAY_1],
        max_position_size=Decimal("25.0"),
        stop_loss_percent=Decimal("5.0"),
        take_profit_percent=Decimal("10.0"),
        risk_per_trade=Decimal("2.0"),
    )

    assert strategy.max_position_size == Decimal("25.00")
    assert strategy.stop_loss_percent == Decimal("5.00")
    assert strategy.take_profit_percent == Decimal("10.00")
    assert strategy.risk_per_trade == Decimal("2.00")


def test_strategy_percentage_validation_negative():
    """Test strategy percentage validation with negative values."""
    with pytest.raises(ValueError, match="Percentage must be between 0 and 100"):
        Strategy(
            name="Invalid Strategy",
            description="Strategy with negative percentage",
            entry_rules=["rule1"],
            exit_rules=["rule2"],
            risk_rules=["rule3"],
            symbols=[Symbol(code="SPY", asset_class=AssetClass.EQUITY)],
            timeframes=[TimeFrame.DAY_1],
            max_position_size=Decimal("-10.0"),  # Invalid negative
            risk_per_trade=Decimal("2.0"),
        )


def test_strategy_percentage_validation_over_100():
    """Test strategy percentage validation with values over 100."""
    with pytest.raises(ValueError, match="Percentage must be between 0 and 100"):
        Strategy(
            name="Invalid Strategy 2",
            description="Strategy with over 100 percentage",
            entry_rules=["rule1"],
            exit_rules=["rule2"],
            risk_rules=["rule3"],
            symbols=[Symbol(code="QQQ", asset_class=AssetClass.EQUITY)],
            timeframes=[TimeFrame.HOUR_1],
            max_position_size=Decimal("25.0"),
            stop_loss_percent=Decimal("150.0"),  # Invalid over 100
            risk_per_trade=Decimal("2.0"),
        )


def test_strategy_total_rules_count():
    """Test strategy total rules count calculation."""
    strategy = Strategy(
        name="Multi-Rule Strategy",
        description="Strategy with multiple rules",
        entry_rules=["entry1", "entry2"],
        exit_rules=["exit1", "exit2", "exit3"],
        risk_rules=["risk1"],
        symbols=[Symbol(code="MSFT", asset_class=AssetClass.EQUITY)],
        timeframes=[TimeFrame.MINUTE_15],
        max_position_size=Decimal("30.0"),
        risk_per_trade=Decimal("1.5"),
    )

    assert strategy.total_rules_count == 6  # 2 + 3 + 1


def test_strategy_status_properties():
    """Test strategy status detection properties."""
    draft_strategy = Strategy(
        name="Draft Strategy",
        description="Strategy in draft",
        status=StrategyStatus.DRAFT,
        entry_rules=["rule1"],
        exit_rules=["rule2"],
        risk_rules=["rule3"],
        symbols=[Symbol(code="GOOGL", asset_class=AssetClass.EQUITY)],
        timeframes=[TimeFrame.DAY_1],
        max_position_size=Decimal("20.0"),
        risk_per_trade=Decimal("2.0"),
    )

    live_strategy = Strategy(
        name="Live Strategy",
        description="Strategy live trading",
        status=StrategyStatus.LIVE_TRADING,
        entry_rules=["rule1"],
        exit_rules=["rule2"],
        risk_rules=["rule3"],
        symbols=[Symbol(code="TSLA", asset_class=AssetClass.EQUITY)],
        timeframes=[TimeFrame.HOUR_4],
        max_position_size=Decimal("15.0"),
        risk_per_trade=Decimal("1.0"),
    )

    backtest_strategy = Strategy(
        name="Backtest Strategy",
        description="Strategy in backtesting",
        status=StrategyStatus.BACKTESTING,
        entry_rules=["rule1"],
        exit_rules=["rule2"],
        risk_rules=["rule3"],
        symbols=[Symbol(code="AMZN", asset_class=AssetClass.EQUITY)],
        timeframes=[TimeFrame.MINUTE_5],
        max_position_size=Decimal("10.0"),
        risk_per_trade=Decimal("3.0"),
    )

    assert draft_strategy.is_active is False
    assert draft_strategy.is_backtesting is False
    assert draft_strategy.is_live is False

    assert live_strategy.is_active is True
    assert live_strategy.is_backtesting is False
    assert live_strategy.is_live is True

    assert backtest_strategy.is_active is False
    assert backtest_strategy.is_backtesting is True
    assert backtest_strategy.is_live is False


def test_strategy_can_generate_signals():
    """Test strategy signal generation capability."""
    # Strategy that can generate signals
    capable_strategy = Strategy(
        name="Capable Strategy",
        description="Strategy that can generate signals",
        status=StrategyStatus.PAPER_TRADING,  # Active status
        entry_rules=["rule1"],  # Has entry rules
        exit_rules=["rule2"],
        risk_rules=["rule3"],
        symbols=[Symbol(code="NFLX", asset_class=AssetClass.EQUITY)],  # Has symbols
        timeframes=[TimeFrame.DAY_1],
        max_position_size=Decimal("25.0"),
        risk_per_trade=Decimal("2.0"),
    )

    # Strategy with no entry rules
    no_rules_strategy = Strategy(
        name="No Rules Strategy",
        description="Strategy without entry rules",
        status=StrategyStatus.LIVE_TRADING,
        entry_rules=[],  # No entry rules
        exit_rules=["rule2"],
        risk_rules=["rule3"],
        symbols=[Symbol(code="META", asset_class=AssetClass.EQUITY)],
        timeframes=[TimeFrame.DAY_1],
        max_position_size=Decimal("25.0"),
        risk_per_trade=Decimal("2.0"),
    )

    # Strategy with no symbols
    no_symbols_strategy = Strategy(
        name="No Symbols Strategy",
        description="Strategy without symbols",
        status=StrategyStatus.PAPER_TRADING,
        entry_rules=["rule1"],
        exit_rules=["rule2"],
        risk_rules=["rule3"],
        symbols=[],  # No symbols
        timeframes=[TimeFrame.DAY_1],
        max_position_size=Decimal("25.0"),
        risk_per_trade=Decimal("2.0"),
    )

    assert capable_strategy.can_generate_signals() is True
    assert no_rules_strategy.can_generate_signals() is False
    assert no_symbols_strategy.can_generate_signals() is False


def test_strategy_state_transitions():
    """Test strategy state transition methods."""
    draft_strategy = Strategy(
        name="Transition Strategy",
        description="Strategy for testing transitions",
        status=StrategyStatus.DRAFT,
        entry_rules=["rule1"],
        exit_rules=["rule2"],
        risk_rules=["rule3"],
        symbols=[Symbol(code="NVDA", asset_class=AssetClass.EQUITY)],
        timeframes=[TimeFrame.DAY_1],
        max_position_size=Decimal("20.0"),
        risk_per_trade=Decimal("2.5"),
    )

    # Test activation
    paper_strategy = draft_strategy.activate()
    assert paper_strategy.status == StrategyStatus.PAPER_TRADING

    # Test promotion to live
    live_strategy = paper_strategy.promote_to_live()
    assert live_strategy.status == StrategyStatus.LIVE_TRADING

    # Test pause
    paused_strategy = live_strategy.pause()
    assert paused_strategy.status == StrategyStatus.PAUSED

    # Test stop
    stopped_strategy = live_strategy.stop()
    assert stopped_strategy.status == StrategyStatus.STOPPED


def test_strategy_invalid_state_transitions():
    """Test invalid strategy state transitions."""
    live_strategy = Strategy(
        name="Live Strategy",
        description="Strategy already live",
        status=StrategyStatus.LIVE_TRADING,
        entry_rules=["rule1"],
        exit_rules=["rule2"],
        risk_rules=["rule3"],
        symbols=[Symbol(code="AAPL", asset_class=AssetClass.EQUITY)],
        timeframes=[TimeFrame.DAY_1],
        max_position_size=Decimal("15.0"),
        risk_per_trade=Decimal("2.0"),
    )

    # Cannot activate already live strategy
    with pytest.raises(
        ValueError, match="Cannot activate strategy in live_trading status"
    ):
        live_strategy.activate()

    # Cannot promote non-paper trading strategy
    with pytest.raises(
        ValueError, match="Cannot promote strategy in live_trading status"
    ):
        live_strategy.promote_to_live()


def test_strategy_performance_metrics_access():
    """Test strategy performance metrics access methods."""
    strategy = Strategy(
        name="Performance Strategy",
        description="Strategy with performance data",
        entry_rules=["rule1"],
        exit_rules=["rule2"],
        risk_rules=["rule3"],
        symbols=[Symbol(code="SPY", asset_class=AssetClass.EQUITY)],
        timeframes=[TimeFrame.DAY_1],
        max_position_size=Decimal("30.0"),
        risk_per_trade=Decimal("2.0"),
        performance_metrics={
            BacktestMetric.SHARPE_RATIO.value: Decimal("1.85"),
            BacktestMetric.MAX_DRAWDOWN.value: Decimal("8.5"),
            BacktestMetric.WIN_RATE.value: Decimal("67.3"),
        },
    )

    assert strategy.has_performance_data is True
    assert strategy.get_sharpe_ratio() == Decimal("1.85")
    assert strategy.get_max_drawdown() == Decimal("8.5")
    assert strategy.get_win_rate() == Decimal("67.3")


def test_backtest_trade_count_validation_negative():
    """Test backtest trade count validation with negative values."""
    with pytest.raises(ValueError, match="Trade counts must be non-negative"):
        Backtest(
            strategy_id="test-strategy",
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31),
            initial_capital=Money(amount=Decimal("100000"), currency=Currency.USD),
            final_capital=Money(amount=Decimal("110000"), currency=Currency.USD),
            symbols=[Symbol(code="AAPL", asset_class=AssetClass.EQUITY)],
            timeframes=[TimeFrame.DAY_1],
            total_trades=-10,  # Invalid negative
            winning_trades=5,
            losing_trades=3,
            total_commission=Money(amount=Decimal("100"), currency=Currency.USD),
            metrics={},
        )


def test_backtest_trade_count_consistency():
    """Test backtest trade count consistency validation."""
    with pytest.raises(
        ValueError, match="Winning \\+ losing trades cannot exceed total trades"
    ):
        Backtest(
            strategy_id="test-strategy",
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31),
            initial_capital=Money(amount=Decimal("100000"), currency=Currency.USD),
            final_capital=Money(amount=Decimal("105000"), currency=Currency.USD),
            symbols=[Symbol(code="SPY", asset_class=AssetClass.EQUITY)],
            timeframes=[TimeFrame.DAY_1],
            total_trades=10,
            winning_trades=8,
            losing_trades=5,  # 8 + 5 = 13 > 10 total
            total_commission=Money(amount=Decimal("50"), currency=Currency.USD),
            metrics={},
        )


def test_backtest_performance_calculations():
    """Test backtest performance calculation methods."""
    backtest = Backtest(
        strategy_id="performance-test",
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 12, 31),  # 365 days
        initial_capital=Money(amount=Decimal("100000"), currency=Currency.USD),
        final_capital=Money(
            amount=Decimal("125000"), currency=Currency.USD
        ),  # 25% return
        symbols=[Symbol(code="QQQ", asset_class=AssetClass.EQUITY)],
        timeframes=[TimeFrame.DAY_1],
        total_trades=100,
        winning_trades=65,
        losing_trades=35,
        total_commission=Money(amount=Decimal("500"), currency=Currency.USD),
        metrics={
            BacktestMetric.SHARPE_RATIO.value: Decimal("2.1"),
            BacktestMetric.MAX_DRAWDOWN.value: Decimal("6.8"),
            BacktestMetric.PROFIT_FACTOR.value: Decimal("1.8"),
        },
    )

    assert backtest.total_return == Decimal("25.0")  # 25% return
    assert backtest.win_rate == Decimal("65.0")  # 65% win rate
    assert backtest.sharpe_ratio == Decimal("2.1")
    assert backtest.max_drawdown == Decimal("6.8")
    assert backtest.profit_factor == Decimal("1.8")
    assert backtest.backtest_duration_days == 364  # 2023 is not leap year
    assert backtest.is_profitable is True


def test_backtest_zero_capital_handling():
    """Test backtest handling of zero initial capital."""
    backtest = Backtest(
        strategy_id="zero-capital-test",
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 6, 30),
        initial_capital=Money(
            amount=Decimal("0"), currency=Currency.USD
        ),  # Zero capital
        final_capital=Money(amount=Decimal("1000"), currency=Currency.USD),
        symbols=[Symbol(code="TSLA", asset_class=AssetClass.EQUITY)],
        timeframes=[TimeFrame.HOUR_1],
        total_trades=50,
        winning_trades=30,
        losing_trades=20,
        total_commission=Money(amount=Decimal("25"), currency=Currency.USD),
        metrics={},
    )

    assert backtest.total_return == Decimal("0")  # Should handle division by zero


def test_backtest_zero_trades_handling():
    """Test backtest handling of zero trades."""
    backtest = Backtest(
        strategy_id="no-trades-test",
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 3, 31),
        initial_capital=Money(amount=Decimal("50000"), currency=Currency.USD),
        final_capital=Money(amount=Decimal("50000"), currency=Currency.USD),
        symbols=[Symbol(code="IWM", asset_class=AssetClass.EQUITY)],
        timeframes=[TimeFrame.DAY_1],
        total_trades=0,  # No trades
        winning_trades=0,
        losing_trades=0,
        total_commission=Money(amount=Decimal("0"), currency=Currency.USD),
        metrics={},
    )

    assert backtest.win_rate == Decimal("0")  # Should handle division by zero


def test_backtest_has_good_metrics():
    """Test backtest good metrics evaluation."""
    good_backtest = Backtest(
        strategy_id="good-performance",
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 12, 31),
        initial_capital=Money(amount=Decimal("100000"), currency=Currency.USD),
        final_capital=Money(
            amount=Decimal("130000"), currency=Currency.USD
        ),  # Profitable
        symbols=[Symbol(code="AMZN", asset_class=AssetClass.EQUITY)],
        timeframes=[TimeFrame.DAY_1],
        total_trades=200,
        winning_trades=120,  # 60% win rate
        losing_trades=80,
        total_commission=Money(amount=Decimal("400"), currency=Currency.USD),
        metrics={
            BacktestMetric.SHARPE_RATIO.value: Decimal("1.5"),  # > 1.0
            BacktestMetric.MAX_DRAWDOWN.value: Decimal("8.0"),  # < 20
            BacktestMetric.WIN_RATE.value: Decimal("60.0"),  # > 50
        },
    )

    poor_backtest = Backtest(
        strategy_id="poor-performance",
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 12, 31),
        initial_capital=Money(amount=Decimal("100000"), currency=Currency.USD),
        final_capital=Money(
            amount=Decimal("85000"), currency=Currency.USD
        ),  # Not profitable
        symbols=[Symbol(code="META", asset_class=AssetClass.EQUITY)],
        timeframes=[TimeFrame.DAY_1],
        total_trades=150,
        winning_trades=60,  # 40% win rate
        losing_trades=90,
        total_commission=Money(amount=Decimal("300"), currency=Currency.USD),
        metrics={
            BacktestMetric.SHARPE_RATIO.value: Decimal("0.5"),  # <= 1.0
            BacktestMetric.MAX_DRAWDOWN.value: Decimal("25.0"),  # >= 20
            BacktestMetric.WIN_RATE.value: Decimal("40.0"),  # <= 50
        },
    )

    assert good_backtest.has_good_metrics is True
    assert poor_backtest.has_good_metrics is False


def test_backtest_get_metric():
    """Test backtest specific metric retrieval."""
    backtest = Backtest(
        strategy_id="metric-test",
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 6, 30),
        initial_capital=Money(amount=Decimal("75000"), currency=Currency.USD),
        final_capital=Money(amount=Decimal("82500"), currency=Currency.USD),
        symbols=[Symbol(code="NVDA", asset_class=AssetClass.EQUITY)],
        timeframes=[TimeFrame.HOUR_4],
        total_trades=75,
        winning_trades=50,
        losing_trades=25,
        total_commission=Money(amount=Decimal("150"), currency=Currency.USD),
        metrics={
            BacktestMetric.CALMAR_RATIO.value: Decimal("2.3"),
            BacktestMetric.SORTINO_RATIO.value: Decimal("3.1"),
            BacktestMetric.VAR_95.value: Decimal("2.8"),
        },
    )

    assert backtest.get_metric(BacktestMetric.CALMAR_RATIO) == Decimal("2.3")
    assert backtest.get_metric(BacktestMetric.SORTINO_RATIO) == Decimal("3.1")
    assert backtest.get_metric(BacktestMetric.VAR_95) == Decimal("2.8")
    assert backtest.get_metric(BacktestMetric.SHARPE_RATIO) is None  # Not in metrics


def test_backtest_comparison():
    """Test backtest comparison functionality."""
    better_backtest = Backtest(
        strategy_id="better-strategy",
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 12, 31),
        initial_capital=Money(amount=Decimal("100000"), currency=Currency.USD),
        final_capital=Money(
            amount=Decimal("125000"), currency=Currency.USD
        ),  # 25% return
        symbols=[Symbol(code="SPY", asset_class=AssetClass.EQUITY)],
        timeframes=[TimeFrame.DAY_1],
        total_trades=100,
        winning_trades=70,
        losing_trades=30,
        total_commission=Money(amount=Decimal("200"), currency=Currency.USD),
        metrics={
            BacktestMetric.SHARPE_RATIO.value: Decimal("2.0"),
            BacktestMetric.MAX_DRAWDOWN.value: Decimal("5.0"),
        },
    )

    worse_backtest = Backtest(
        strategy_id="worse-strategy",
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 12, 31),
        initial_capital=Money(amount=Decimal("100000"), currency=Currency.USD),
        final_capital=Money(
            amount=Decimal("115000"), currency=Currency.USD
        ),  # 15% return
        symbols=[Symbol(code="QQQ", asset_class=AssetClass.EQUITY)],
        timeframes=[TimeFrame.DAY_1],
        total_trades=120,
        winning_trades=60,
        losing_trades=60,
        total_commission=Money(amount=Decimal("240"), currency=Currency.USD),
        metrics={
            BacktestMetric.SHARPE_RATIO.value: Decimal("1.5"),
            BacktestMetric.MAX_DRAWDOWN.value: Decimal("8.0"),
        },
    )

    comparison = better_backtest.compare_to(worse_backtest)

    assert comparison["return"] == "better"  # 25% > 15%
    assert comparison["risk_adjusted"] == "better"  # 2.0 > 1.5 Sharpe
    assert comparison["drawdown"] == "better"  # 5.0 < 8.0 drawdown
