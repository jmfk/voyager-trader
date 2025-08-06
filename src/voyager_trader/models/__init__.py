"""
VOYAGER-Trader Domain Models.

This package provides comprehensive domain models for the VOYAGER-Trader
system, including market data, trading entities, strategies, learning
components, and VOYAGER system models.

The models are organized into the following modules:
- base: Base classes and infrastructure
- types: Common value objects and enums
- market: Market data models (OHLCV, ticks, order books)
- trading: Trading entities (Portfolio, Position, Order, Trade)
- strategy: Strategy models (Signal, Rule, Strategy, Backtest)
- learning: Learning entities (Skill, Experience, Knowledge)
- system: VOYAGER system models (Curriculum, Environment, Agent, Task)
"""

# Base infrastructure
from .base import (
    AggregateRoot,
    BaseEntity,
    DomainEvent,
    Repository,
    Specification,
    ValueObject,
    VoyagerBaseModel,
)

# Learning models
from .learning import (  # Enums; Entities
    Experience,
    Knowledge,
    KnowledgeType,
    LearningOutcome,
    Performance,
    PerformanceMetric,
    Skill,
    SkillExecutionResult,
)

# Market data models
from .market import (
    OHLCV,
    MarketEvent,
    MarketSession,
    MarketSnapshot,
    OrderBook,
    OrderBookLevel,
    TickData,
)

# Strategy models
from .strategy import (  # Enums; Entities
    Backtest,
    BacktestMetric,
    IndicatorType,
    RuleOperator,
    Signal,
    Strategy,
    TradingRule,
)

# VOYAGER system models
from .system import (  # Enums; Entities
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

# Trading entities
from .trading import (  # Domain events; Entities
    Account,
    Order,
    OrderExecuted,
    Portfolio,
    Position,
    PositionClosed,
    PositionOpened,
    Trade,
)

# Common types and value objects
from .types import (  # Enums; Value objects
    AssetClass,
    Currency,
    Money,
    OrderSide,
    OrderStatus,
    OrderType,
    PositionType,
    Price,
    Quantity,
    SignalStrength,
    SignalType,
    SkillCategory,
    SkillComplexity,
    StrategyStatus,
    Symbol,
    TaskStatus,
    TimeFrame,
)

__all__ = [
    # Base infrastructure
    "VoyagerBaseModel",
    "BaseEntity",
    "ValueObject",
    "AggregateRoot",
    "DomainEvent",
    "Repository",
    "Specification",
    # Common types and enums
    "Currency",
    "AssetClass",
    "OrderType",
    "OrderSide",
    "OrderStatus",
    "PositionType",
    "TimeFrame",
    "SignalType",
    "SignalStrength",
    "StrategyStatus",
    "SkillCategory",
    "SkillComplexity",
    "TaskStatus",
    # Value objects
    "Money",
    "Price",
    "Quantity",
    "Symbol",
    # Market data models
    "OHLCV",
    "TickData",
    "OrderBookLevel",
    "OrderBook",
    "MarketEvent",
    "MarketSession",
    "MarketSnapshot",
    # Trading entities
    "OrderExecuted",
    "PositionOpened",
    "PositionClosed",
    "Order",
    "Trade",
    "Position",
    "Account",
    "Portfolio",
    # Strategy models
    "RuleOperator",
    "IndicatorType",
    "BacktestMetric",
    "Signal",
    "TradingRule",
    "Strategy",
    "Backtest",
    # Learning models
    "SkillExecutionResult",
    "LearningOutcome",
    "KnowledgeType",
    "PerformanceMetric",
    "Skill",
    "Experience",
    "Knowledge",
    "Performance",
    # VOYAGER system models
    "TaskPriority",
    "TaskType",
    "DifficultyLevel",
    "AgentState",
    "EnvironmentType",
    "CurriculumStrategy",
    "Task",
    "Environment",
    "Agent",
    "Curriculum",
]


def get_model_version() -> str:
    """Get the domain models version."""
    return "1.0.0"


def get_model_info() -> dict:
    """Get comprehensive model information."""
    return {
        "version": get_model_version(),
        "total_models": len(__all__),
        "categories": {
            "base_infrastructure": 6,
            "common_types": 17,
            "market_data": 7,
            "trading_entities": 8,
            "strategy_models": 7,
            "learning_models": 8,
            "system_models": 10,
        },
        "description": (
            "Comprehensive domain models for VOYAGER-Trader autonomous "
            "trading system"
        ),
    }
