"""Basic tests for system models to improve coverage."""

from decimal import Decimal

from src.voyager_trader.models.system import (
    Agent,
    AgentState,
    Curriculum,
    CurriculumStrategy,
    DifficultyLevel,
    Environment,
    EnvironmentType,
    Task,
    TaskPriority,
    TaskType,
)
from src.voyager_trader.models.types import (
    AssetClass,
    Currency,
    Money,
    Symbol,
    TaskStatus,
    TimeFrame,
)


def test_task_basic():
    """Basic Task creation."""
    task = Task(
        title="Learn RSI Strategy",
        description="Implement basic RSI mean reversion strategy",
        task_type=TaskType.LEARNING,
        priority=TaskPriority.HIGH,
        difficulty=DifficultyLevel.BEGINNER,
        objectives=["Understand RSI indicator", "Implement buy/sell logic"],
        success_criteria=["Backtest shows positive returns", "Sharpe ratio > 1.0"],
    )

    assert task.title == "Learn RSI Strategy"
    assert task.task_type == TaskType.LEARNING
    assert task.priority == TaskPriority.HIGH
    assert task.difficulty == DifficultyLevel.BEGINNER
    assert task.status == TaskStatus.PENDING


def test_environment_basic():
    """Basic Environment creation."""
    env = Environment(
        name="Paper Trading Environment",
        environment_type=EnvironmentType.PAPER_TRADING,
        description="Safe environment for testing strategies",
        base_currency=Currency.USD,
        available_symbols=[
            Symbol(code="AAPL", asset_class=AssetClass.EQUITY),
            Symbol(code="GOOGL", asset_class=AssetClass.EQUITY),
        ],
        supported_timeframes=[TimeFrame.MINUTE_1, TimeFrame.DAY_1],
        market_hours={"NYSE": "09:30-16:00"},
        trading_constraints={"max_position_size": 0.1},
        risk_limits={"max_daily_loss": 1000},
        initial_capital=Money(amount=Decimal("100000"), currency=Currency.USD),
        current_capital=Money(amount=Decimal("100000"), currency=Currency.USD),
        commission_structure={"stock_trade": Decimal("0.005")},
        margin_requirements={"initial": Decimal("0.5")},
        data_providers=["Yahoo Finance"],
        execution_venues=["Paper Broker"],
        features=["Real-time data"],
        limitations=["No slippage simulation"],
        configuration={},
    )

    assert env.name == "Paper Trading Environment"
    assert env.environment_type == EnvironmentType.PAPER_TRADING
    assert env.base_currency == Currency.USD
    assert len(env.available_symbols) == 2


def test_agent_basic():
    """Basic Agent creation."""
    agent = Agent(
        name="VOYAGER Agent v1.0",
        description="Autonomous trading agent",
        capabilities=["Strategy development", "Risk management", "Market analysis"],
        current_environment_id="env-paper-trading",
    )

    assert agent.name == "VOYAGER Agent v1.0"
    assert agent.state == AgentState.INITIALIZING
    assert len(agent.capabilities) == 3
    assert agent.current_environment_id == "env-paper-trading"


def test_curriculum_basic():
    """Basic Curriculum creation."""
    curriculum = Curriculum(
        name="VOYAGER Trading Curriculum",
        description="Comprehensive trading education program",
        agent_id="agent-123",
        strategy=CurriculumStrategy.ADAPTIVE,
        target_skills=[
            "RSI",
            "Moving Averages",
            "Risk Management",
            "Portfolio Optimization",
        ],
    )

    assert curriculum.name == "VOYAGER Trading Curriculum"
    assert curriculum.strategy == CurriculumStrategy.ADAPTIVE
    assert curriculum.current_difficulty == DifficultyLevel.BEGINNER
    assert len(curriculum.target_skills) == 4
