"""Validation tests for learning models to improve coverage."""

from datetime import datetime
from decimal import Decimal

import pytest

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


def test_skill_version_validation():
    """Test skill version validation and updates."""
    skill = Skill(
        name="Moving Average Cross",
        description="Simple MA cross strategy",
        category=SkillCategory.TECHNICAL_ANALYSIS,
        complexity=SkillComplexity.BASIC,
        code="def ma_cross(data): return data['ma_fast'] > data['ma_slow']",
        version="1.0",
        input_schema={"ma_fast": "float", "ma_slow": "float"},
        output_schema={"signal": "string"},
    )

    assert skill.version == "1.0"


def test_skill_complexity_levels():
    """Test different skill complexity levels."""
    basic_skill = Skill(
        name="Basic RSI",
        description="Simple RSI strategy",
        category=SkillCategory.TECHNICAL_ANALYSIS,
        complexity=SkillComplexity.BASIC,
        code="def rsi_basic(data): return data['rsi'] < 30",
        version="1.0",
        input_schema={"rsi": "float"},
        output_schema={"signal": "string"},
    )

    intermediate_skill = Skill(
        name="Intermediate Strategy Model",
        description="Intermediate complexity model",
        category=SkillCategory.RISK_MANAGEMENT,
        complexity=SkillComplexity.INTERMEDIATE,
        code="def intermediate_strategy(data): return risk_model.assess(data)",
        version="1.5",
        input_schema={"risk_metrics": "array"},
        output_schema={"risk_score": "float", "recommendation": "string"},
    )

    assert basic_skill.complexity == SkillComplexity.BASIC
    assert intermediate_skill.complexity == SkillComplexity.INTERMEDIATE


def test_skill_category_types():
    """Test different skill category types."""
    technical_skill = Skill(
        name="Bollinger Bands",
        description="Bollinger band strategy",
        category=SkillCategory.TECHNICAL_ANALYSIS,
        complexity=SkillComplexity.INTERMEDIATE,
        code="def bollinger(data): return data['price'] < data['bb_lower']",
        version="1.0",
        input_schema={"price": "float", "bb_lower": "float"},
        output_schema={"signal": "string"},
    )

    risk_skill = Skill(
        name="Position Sizing",
        description="Kelly criterion position sizing",
        category=SkillCategory.RISK_MANAGEMENT,
        complexity=SkillComplexity.ADVANCED,
        code="def kelly_size(data): return data['edge'] / data['odds']",
        version="1.0",
        input_schema={"edge": "float", "odds": "float"},
        output_schema={"position_size": "float"},
    )

    assert technical_skill.category == SkillCategory.TECHNICAL_ANALYSIS
    assert risk_skill.category == SkillCategory.RISK_MANAGEMENT


def test_skill_usage_tracking():
    """Test skill usage count tracking."""
    skill = Skill(
        name="Test Skill",
        description="Test skill",
        category=SkillCategory.TECHNICAL_ANALYSIS,
        complexity=SkillComplexity.BASIC,
        code="def test(): return True",
        version="1.0",
        input_schema={},
        output_schema={},
        usage_count=0,
    )

    # Test incrementing usage
    updated_skill = skill.update(usage_count=skill.usage_count + 1)
    assert updated_skill.usage_count == 1


def test_skill_with_tags():
    """Test skill with tags."""
    skill = Skill(
        name="Tagged Skill",
        description="Skill with tags",
        category=SkillCategory.TECHNICAL_ANALYSIS,
        complexity=SkillComplexity.BASIC,
        code="def tagged(): return True",
        version="1.0",
        input_schema={},
        output_schema={},
        tags=["momentum", "reversal", "trending"],
    )

    assert len(skill.tags) == 3
    assert "momentum" in skill.tags


def test_experience_confidence_validation_valid():
    """Test experience confidence validation with valid values."""
    experience = Experience(
        title="Valid Confidence Test",
        description="Test valid confidence levels",
        context={"test": "valid"},
        actions_taken=["Action 1"],
        outcome=LearningOutcome.POSITIVE,
        outcome_details={"result": "good"},
        lessons_learned=["Lesson 1"],
        contributing_factors=["Factor 1"],
        market_conditions={"trend": "up"},
        symbols_involved=[Symbol(code="AAPL", asset_class=AssetClass.EQUITY)],
        timeframe=TimeFrame.DAY_1,
        confidence_before=Decimal("50.0"),
        confidence_after=Decimal("75.5"),
        stress_level=Decimal("25.0"),
        complexity_score=Decimal("60.0"),
        novelty_score=Decimal("80.0"),
    )

    assert experience.confidence_before == Decimal("50.0")
    assert experience.confidence_after == Decimal("75.5")


def test_experience_confidence_validation_negative():
    """Test experience confidence validation with negative values."""
    with pytest.raises(ValueError, match="must be between 0 and 100"):
        Experience(
            title="Invalid Confidence Test",
            description="Test invalid confidence levels",
            context={"test": "invalid"},
            actions_taken=["Action 1"],
            outcome=LearningOutcome.NEGATIVE,
            outcome_details={"result": "bad"},
            lessons_learned=["Lesson 1"],
            contributing_factors=["Factor 1"],
            market_conditions={"trend": "down"},
            symbols_involved=[Symbol(code="AAPL", asset_class=AssetClass.EQUITY)],
            timeframe=TimeFrame.DAY_1,
            confidence_before=Decimal("-10.0"),  # Invalid negative value
            confidence_after=Decimal("50.0"),
            stress_level=Decimal("40.0"),
            complexity_score=Decimal("30.0"),
            novelty_score=Decimal("20.0"),
        )


def test_experience_confidence_validation_over_100():
    """Test experience confidence validation with values over 100."""
    with pytest.raises(ValueError, match="must be between 0 and 100"):
        Experience(
            title="Over 100 Confidence Test",
            description="Test over 100 confidence levels",
            context={"test": "over100"},
            actions_taken=["Action 1"],
            outcome=LearningOutcome.MIXED,
            outcome_details={"result": "mixed"},
            lessons_learned=["Lesson 1"],
            contributing_factors=["Factor 1"],
            market_conditions={"trend": "sideways"},
            symbols_involved=[Symbol(code="SPY", asset_class=AssetClass.EQUITY)],
            timeframe=TimeFrame.HOUR_1,
            confidence_before=Decimal("50.0"),
            confidence_after=Decimal("150.0"),  # Invalid over 100 value
            stress_level=Decimal("20.0"),
            complexity_score=Decimal("40.0"),
            novelty_score=Decimal("60.0"),
        )


def test_experience_learning_outcomes():
    """Test different learning outcomes."""
    positive_exp = Experience(
        title="Positive Experience",
        description="Good outcome",
        context={"market": "bullish"},
        actions_taken=["Bought calls"],
        outcome=LearningOutcome.POSITIVE,
        outcome_details={"profit": 1000},
        lessons_learned=["Bull markets favor calls"],
        contributing_factors=["Strong earnings"],
        market_conditions={"trend": "up"},
        symbols_involved=[Symbol(code="AAPL", asset_class=AssetClass.EQUITY)],
        timeframe=TimeFrame.DAY_1,
        confidence_before=Decimal("60"),
        confidence_after=Decimal("80"),
        stress_level=Decimal("20"),
        complexity_score=Decimal("40"),
        novelty_score=Decimal("30"),
    )

    negative_exp = Experience(
        title="Negative Experience",
        description="Bad outcome",
        context={"market": "bearish"},
        actions_taken=["Bought calls"],
        outcome=LearningOutcome.NEGATIVE,
        outcome_details={"loss": -500},
        lessons_learned=["Don't fight the trend"],
        contributing_factors=["Market crash"],
        market_conditions={"trend": "down"},
        symbols_involved=[Symbol(code="SPY", asset_class=AssetClass.EQUITY)],
        timeframe=TimeFrame.DAY_1,
        confidence_before=Decimal("70"),
        confidence_after=Decimal("30"),
        stress_level=Decimal("80"),
        complexity_score=Decimal("60"),
        novelty_score=Decimal("90"),
    )

    assert positive_exp.outcome == LearningOutcome.POSITIVE
    assert negative_exp.outcome == LearningOutcome.NEGATIVE


def test_experience_with_multiple_symbols():
    """Test experience with multiple symbols."""
    symbols = [
        Symbol(code="AAPL", asset_class=AssetClass.EQUITY),
        Symbol(code="GOOGL", asset_class=AssetClass.EQUITY),
        Symbol(code="MSFT", asset_class=AssetClass.EQUITY),
    ]

    experience = Experience(
        title="Multi-Symbol Experience",
        description="Tested strategy on multiple symbols",
        context={"strategy": "momentum"},
        actions_taken=["Applied momentum strategy"],
        outcome=LearningOutcome.MIXED,
        outcome_details={"winners": 2, "losers": 1},
        lessons_learned=["Diversification helps"],
        contributing_factors=["Mixed market conditions"],
        market_conditions={"trend": "mixed"},
        symbols_involved=symbols,
        timeframe=TimeFrame.MINUTE_15,
        confidence_before=Decimal("50"),
        confidence_after=Decimal("65"),
        stress_level=Decimal("40"),
        complexity_score=Decimal("70"),
        novelty_score=Decimal("50"),
    )

    assert len(experience.symbols_involved) == 3
    assert experience.symbols_involved[0].code == "AAPL"


def test_knowledge_confidence_validation():
    """Test knowledge confidence validation."""
    knowledge = Knowledge(
        title="Market Pattern Knowledge",
        knowledge_type=KnowledgeType.MARKET_PATTERN,
        content="Pattern description",
        confidence=Decimal("85.5"),
        supporting_evidence=["exp1", "exp2"],
        market_contexts=["trending"],
        conditions=["RSI > 70"],
        patterns=["breakout"],
        actionable_insights=["Buy on breakout"],
    )

    assert knowledge.confidence == Decimal("85.5")


def test_knowledge_supporting_evidence():
    """Test knowledge with supporting evidence."""
    knowledge = Knowledge(
        title="Evidence-based Knowledge",
        knowledge_type=KnowledgeType.MARKET_PATTERN,
        content="Pattern supported by evidence",
        confidence=Decimal("80"),
        supporting_evidence=["exp1", "exp2", "exp3"],
        market_contexts=["trending", "volatile"],
        conditions=["volume > average"],
        patterns=["breakout", "momentum"],
        actionable_insights=["Buy on breakout", "Set tight stops"],
    )

    assert len(knowledge.supporting_evidence) == 3
    assert len(knowledge.market_contexts) == 2
    assert len(knowledge.patterns) == 2
    assert len(knowledge.actionable_insights) == 2


def test_performance_success_rate_calculation():
    """Test performance success rate calculation."""
    performance = Performance(
        entity_type="strategy",
        entity_id="test-strategy",
        measurement_period_start=datetime(2023, 1, 1),
        measurement_period_end=datetime(2023, 3, 31),
        total_observations=200,
        successful_observations=150,
        metrics={"return": Decimal("10.5")},
    )

    # Test success rate calculation
    assert performance.success_rate == Decimal("0.75")  # 150/200


def test_performance_success_rate_zero_observations():
    """Test performance success rate with zero observations."""
    performance = Performance(
        entity_type="strategy",
        entity_id="empty-strategy",
        measurement_period_start=datetime(2023, 1, 1),
        measurement_period_end=datetime(2023, 1, 31),
        total_observations=0,
        successful_observations=0,
        metrics={},
    )

    # Should handle division by zero
    assert performance.success_rate == Decimal("0")


def test_performance_with_detailed_metrics():
    """Test performance with detailed metrics."""
    detailed_metrics = {
        "total_return": Decimal("18.5"),
        "annual_return": Decimal("22.3"),
        "sharpe_ratio": Decimal("2.1"),
        "sortino_ratio": Decimal("3.2"),
        "max_drawdown": Decimal("4.8"),
        "calmar_ratio": Decimal("4.6"),
        "win_rate": Decimal("68.2"),
        "average_win": Decimal("2.4"),
        "average_loss": Decimal("-1.2"),
        "profit_factor": Decimal("2.0"),
    }

    performance = Performance(
        entity_type="strategy",
        entity_id="detailed-strategy",
        measurement_period_start=datetime(2023, 1, 1),
        measurement_period_end=datetime(2023, 12, 31),
        total_observations=365,
        successful_observations=249,
        metrics=detailed_metrics,
    )

    assert len(performance.metrics) == 10
    assert performance.metrics["sharpe_ratio"] == Decimal("2.1")
    assert performance.metrics["win_rate"] == Decimal("68.2")


def test_performance_metrics_access():
    """Test performance metrics access and updates."""
    metrics = {
        "total_return": Decimal("15.2"),
        "sharpe_ratio": Decimal("1.8"),
        "max_drawdown": Decimal("5.1"),
        "win_rate": Decimal("65.5"),
    }

    performance = Performance(
        entity_type="portfolio",
        entity_id="test-portfolio",
        measurement_period_start=datetime(2023, 1, 1),
        measurement_period_end=datetime(2023, 6, 30),
        total_observations=300,
        successful_observations=200,
        metrics=metrics,
    )

    assert performance.metrics["total_return"] == Decimal("15.2")
    assert performance.metrics["sharpe_ratio"] == Decimal("1.8")
    assert len(performance.metrics) == 4


def test_performance_validation_negative_observations():
    """Test performance validation with negative observations."""
    with pytest.raises(ValueError, match="must be non-negative"):
        Performance(
            entity_type="strategy",
            entity_id="invalid-strategy",
            measurement_period_start=datetime(2023, 1, 1),
            measurement_period_end=datetime(2023, 2, 1),
            total_observations=-10,  # Invalid negative
            successful_observations=5,
            metrics={},
        )


def test_performance_validation_successful_greater_than_total():
    """Test performance validation where successful > total observations."""
    with pytest.raises(ValueError, match="cannot exceed total observations"):
        Performance(
            entity_type="strategy",
            entity_id="invalid-strategy2",
            measurement_period_start=datetime(2023, 1, 1),
            measurement_period_end=datetime(2023, 2, 1),
            total_observations=50,
            successful_observations=75,  # Invalid: more successful than total
            metrics={},
        )
