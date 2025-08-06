"""Basic tests for learning models to improve coverage."""

from datetime import datetime
from decimal import Decimal

from src.voyager_trader.models.learning import (
    Experience,
    Knowledge,
    KnowledgeType,
    LearningOutcome,
    Performance,
    Skill,
)
from src.voyager_trader.models.types import (
    AssetClass,
    SkillCategory,
    SkillComplexity,
    Symbol,
    TimeFrame,
)


def test_skill_basic():
    """Basic Skill creation."""
    skill = Skill(
        name="RSI Mean Reversion",
        description="Buy when RSI < 30, sell when RSI > 70",
        category=SkillCategory.TECHNICAL_ANALYSIS,
        complexity=SkillComplexity.BASIC,
        code="def rsi_strategy(data): return data['rsi'] < 30",
        version="1.0",
        input_schema={"rsi": "float", "price": "float"},
        output_schema={"signal": "string", "confidence": "float"},
    )

    assert skill.name == "RSI Mean Reversion"
    assert skill.category == SkillCategory.TECHNICAL_ANALYSIS
    assert skill.complexity == SkillComplexity.BASIC
    assert skill.version == "1.0"
    assert skill.usage_count == 0


def test_experience_basic():
    """Basic Experience creation."""
    experience = Experience(
        title="RSI Strategy Backtest",
        description="Tested RSI mean reversion on AAPL",
        context={"market_condition": "trending", "volatility": "low"},
        actions_taken=["Implemented RSI strategy", "Ran backtest", "Analyzed results"],
        outcome=LearningOutcome.POSITIVE,
        outcome_details={"return": 15.2, "sharpe": 1.8, "max_drawdown": 5.1},
        lessons_learned=["RSI works well in ranging markets"],
        contributing_factors=["Low volatility period", "Mean reverting environment"],
        market_conditions={
            "trend": "sideways",
            "volatility": "low",
            "volume": "normal",
        },
        symbols_involved=[Symbol(code="AAPL", asset_class=AssetClass.EQUITY)],
        timeframe=TimeFrame.DAY_1,
        confidence_before=Decimal("60"),
        confidence_after=Decimal("80"),
        stress_level=Decimal("30"),
        complexity_score=Decimal("40"),
        novelty_score=Decimal("70"),
    )

    assert experience.title == "RSI Strategy Backtest"
    assert experience.outcome == LearningOutcome.POSITIVE
    assert len(experience.symbols_involved) == 1
    assert experience.timeframe == TimeFrame.DAY_1


def test_knowledge_basic():
    """Basic Knowledge creation."""
    knowledge = Knowledge(
        title="RSI Oversold Conditions",
        knowledge_type=KnowledgeType.MARKET_PATTERN,
        content="When RSI drops below 30, assets often bounce back within 2-3 days",
        confidence=Decimal("85"),
        supporting_evidence=["exp-1", "exp-2", "exp-3"],
        market_contexts=["ranging market", "low volatility"],
        conditions=["RSI < 30", "Volume > average", "No major news"],
        patterns=["mean_reversion", "oversold_bounce"],
        actionable_insights=["Buy signal when RSI < 30", "Set stop loss at recent low"],
    )

    assert knowledge.title == "RSI Oversold Conditions"
    assert knowledge.knowledge_type == KnowledgeType.MARKET_PATTERN
    assert knowledge.confidence == Decimal("85")
    assert len(knowledge.supporting_evidence) == 3


def test_performance_basic():
    """Basic Performance creation."""
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 3, 31)

    performance = Performance(
        entity_type="strategy",
        entity_id="rsi-strategy-v1",
        measurement_period_start=start_date,
        measurement_period_end=end_date,
        total_observations=100,
        successful_observations=75,
        metrics={
            "total_return": Decimal("12.5"),
            "sharpe_ratio": Decimal("1.8"),
            "max_drawdown": Decimal("5.2"),
            "volatility": Decimal("15.3"),
        },
    )

    assert performance.entity_type == "strategy"
    assert performance.entity_id == "rsi-strategy-v1"
    assert performance.total_observations == 100
    assert performance.successful_observations == 75
