"""Comprehensive tests for learning models to maximize coverage."""

from datetime import datetime
from decimal import Decimal

from src.voyager_trader.models.learning import (
    Experience,
    Knowledge,
    KnowledgeType,
    LearningOutcome,
    Performance,
    PerformanceMetric,
    Skill,
)
from src.voyager_trader.models.types import (
    AssetClass,
    Currency,
    Money,
    SkillCategory,
    SkillComplexity,
    Symbol,
    TimeFrame,
)


def test_skill_comprehensive():
    """Test comprehensive Skill functionality and methods."""
    skill = Skill(
        name="Advanced RSI Strategy",
        description="Sophisticated RSI trading with multiple confirmations",
        category=SkillCategory.TECHNICAL_ANALYSIS,
        complexity=SkillComplexity.ADVANCED,
        version="2.1",
        code="def advanced_rsi(data, period=14): return calculate_rsi(data, period)",
        language="python",
        dependencies=["numpy", "pandas", "talib"],
        required_skills=["basic-rsi", "volume-analysis"],
        input_schema={"data": "DataFrame", "period": "int", "threshold": "float"},
        output_schema={"signal": "str", "confidence": "float", "strength": "float"},
        parameters={"default_period": 14, "oversold": 30, "overbought": 70},
        examples=[
            {
                "input": {"data": "sample_data", "period": 14},
                "output": {"signal": "buy"},
            },
            {
                "input": {"data": "sample_data", "period": 21},
                "output": {"signal": "hold"},
            },
        ],
        performance_metrics={
            "accuracy": Decimal("87.5"),
            "precision": Decimal("82.3"),
            "recall": Decimal("91.2"),
        },
        usage_count=25,
        success_count=22,
        last_used=datetime(2024, 1, 20),
        created_at=datetime(2024, 1, 1),
        updated_at=datetime.utcnow(),
        is_active=True,
        tags=["rsi", "technical", "momentum"],
    )

    # Test comprehensive properties
    assert skill.name == "Advanced RSI Strategy"
    assert skill.category == SkillCategory.TECHNICAL_ANALYSIS
    assert skill.complexity == SkillComplexity.ADVANCED
    assert skill.version == "2.1"
    assert skill.code is not None
    assert skill.language == "python"
    assert len(skill.dependencies) == 3
    assert len(skill.required_skills) == 2
    assert len(skill.input_schema) == 3
    assert len(skill.output_schema) == 3
    assert len(skill.parameters) == 3
    assert len(skill.examples) == 2
    assert len(skill.performance_metrics) == 3
    assert skill.usage_count == 25
    assert skill.success_count == 22
    assert skill.last_used is not None
    assert skill.is_active
    assert len(skill.tags) == 3

    # Test computed properties
    expected_success_rate = (Decimal("22") / Decimal("25")) * 100
    assert abs(skill.success_rate - expected_success_rate) < Decimal("0.01")

    expected_failure_count = 25 - 22
    assert skill.failure_count == expected_failure_count

    # With usage, reliability should be higher than default 50
    assert skill.reliability_score > Decimal("50")
    assert skill.is_reliable
    assert not skill.is_experimental  # High usage count

    # Test skill usage incrementation (using update method directly)
    incremented_skill = skill.update(usage_count=26, success_count=23)
    assert incremented_skill.usage_count == 26
    assert incremented_skill.success_count == 23

    # Test code update
    new_code = (
        "def improved_rsi(data, period=14): return enhanced_calculation(data, period)"
    )
    updated_skill = skill.update_code(new_code, "2.2")
    assert updated_skill.code == new_code
    assert updated_skill.version == "2.2"

    # Test deactivation
    deactivated_skill = skill.deactivate("Replaced by newer version")
    assert not deactivated_skill.is_active

    # Test performance summary
    summary = skill.get_performance_summary()
    assert "usage_count" in summary
    assert "success_rate" in summary
    assert "reliability_score" in summary


def test_experience_comprehensive():
    """Test comprehensive Experience functionality."""
    experience = Experience(
        title="Complex Options Strategy Backtest",
        description="Tested iron condor strategy on high volatility stocks",
        context={
            "market_condition": "high_volatility",
            "volatility": "25.3%",
            "trend": "sideways",
            "sector": "technology",
        },
        actions_taken=[
            "Selected high IV stocks",
            "Set up iron condor spreads",
            "Managed positions daily",
            "Closed before expiration",
        ],
        outcome=LearningOutcome.POSITIVE,
        outcome_details={
            "total_return": 18.7,
            "win_rate": 73.2,
            "max_drawdown": 4.1,
            "sharpe_ratio": 2.1,
            "trades_count": 45,
        },
        lessons_learned=[
            "Iron condors work well in high IV environments",
            "Early closure often better than expiration",
            "Position sizing critical for risk management",
        ],
        contributing_factors=[
            "High implied volatility premium",
            "Sideways market movement",
            "Good position management",
        ],
        market_conditions={
            "trend": "sideways",
            "volatility": "high",
            "volume": "above_average",
            "sentiment": "neutral",
        },
        symbols_involved=[
            Symbol(code="AAPL", asset_class=AssetClass.EQUITY),
            Symbol(code="GOOGL", asset_class=AssetClass.EQUITY),
            Symbol(code="MSFT", asset_class=AssetClass.EQUITY),
        ],
        timeframe=TimeFrame.HOUR_1,
        duration_minutes=2100,  # 35 hours
        financial_impact=Money(amount=Decimal("3750"), currency=Currency.USD),
        confidence_before=Decimal("65"),
        confidence_after=Decimal("82"),
        stress_level=Decimal("45"),
        complexity_score=Decimal("75"),
        novelty_score=Decimal("85"),
        created_at=datetime(2024, 1, 10),
        tags=["options", "iron_condor", "volatility"],
    )

    # Test comprehensive properties
    assert experience.title == "Complex Options Strategy Backtest"
    assert experience.outcome == LearningOutcome.POSITIVE
    assert len(experience.context) == 4
    assert len(experience.actions_taken) == 4
    assert len(experience.outcome_details) == 5
    assert len(experience.lessons_learned) == 3
    assert len(experience.contributing_factors) == 3
    assert len(experience.market_conditions) == 4
    assert len(experience.symbols_involved) == 3
    assert experience.timeframe == TimeFrame.HOUR_1
    assert experience.duration_minutes == 2100
    assert experience.financial_impact.amount == Decimal("3750")
    assert experience.confidence_before == Decimal("65")
    assert experience.confidence_after == Decimal("82")
    assert experience.stress_level == Decimal("45")
    assert experience.complexity_score == Decimal("75")
    assert experience.novelty_score == Decimal("85")
    assert len(experience.tags) == 3

    # Test computed properties
    expected_confidence_change = Decimal("82") - Decimal("65")
    assert experience.confidence_change == expected_confidence_change

    assert experience.is_positive_outcome
    assert not experience.is_negative_outcome
    assert experience.is_novel  # novelty_score 85 > 50
    assert experience.is_complex  # complexity_score 75 > 70
    assert not experience.is_high_stress  # stress_level 45 < 70

    # Test learning value calculation
    learning_value = experience.learning_value
    assert learning_value > Decimal("0")
    assert learning_value <= Decimal("100")

    # Test pattern identification
    patterns = experience.identify_patterns()
    assert len(patterns) > 0
    assert any("positive_outcome" in pattern for pattern in patterns)

    # Test negative experience
    negative_exp = Experience(
        title="Failed Momentum Strategy",
        description="Momentum strategy failed in choppy market",
        context={"market_condition": "choppy"},
        actions_taken=["Applied momentum indicators"],
        outcome=LearningOutcome.NEGATIVE,
        outcome_details={"return": -12.3, "max_drawdown": 18.7},
        lessons_learned=["Momentum fails in choppy markets"],
        contributing_factors=["High volatility", "No clear trend"],
        market_conditions={"trend": "choppy", "volatility": "high"},
        symbols_involved=[Symbol(code="SPY", asset_class=AssetClass.EQUITY)],
        timeframe=TimeFrame.MINUTE_15,
        confidence_before=Decimal("75"),
        confidence_after=Decimal("45"),
        stress_level=Decimal("85"),
        complexity_score=Decimal("60"),
        novelty_score=Decimal("30"),
    )

    assert negative_exp.is_negative_outcome
    assert not negative_exp.is_positive_outcome
    assert negative_exp.is_high_stress
    assert not negative_exp.is_complex
    assert not negative_exp.is_novel
    assert negative_exp.confidence_change == Decimal("-30")

    # Test mixed outcome
    mixed_exp = Experience(
        title="Mixed Results Strategy",
        description="Strategy with mixed results",
        context={},
        actions_taken=["Applied strategy"],
        outcome=LearningOutcome.MIXED,
        outcome_details={},
        lessons_learned=["Mixed results require refinement"],
        contributing_factors=[],
        market_conditions={},
        symbols_involved=[],
        timeframe=TimeFrame.DAY_1,
        confidence_before=Decimal("60"),
        confidence_after=Decimal("60"),
    )

    assert not mixed_exp.is_positive_outcome
    assert not mixed_exp.is_negative_outcome


def test_knowledge_comprehensive():
    """Test comprehensive Knowledge functionality."""
    knowledge = Knowledge(
        title="Options Greeks Risk Management",
        knowledge_type=KnowledgeType.RISK_INSIGHT,
        content="Delta hedging is crucial for managing directional risk in options "
        "portfolios",
        confidence=Decimal("92"),
        supporting_evidence=["study-1", "backtest-2", "paper-3", "experience-4"],
        contradicting_evidence=["outlier-study"],
        symbols=[
            Symbol(code="SPY", asset_class=AssetClass.EQUITY),
            Symbol(code="QQQ", asset_class=AssetClass.EQUITY),
        ],
        timeframes=[TimeFrame.HOUR_1, TimeFrame.DAY_1],
        market_contexts=["high_volatility", "earnings_season", "options_expiration"],
        conditions=[
            "High gamma exposure",
            "Short time to expiration",
            "Volatile underlying",
        ],
        patterns=["gamma_squeeze", "pin_risk", "volatility_skew"],
        actionable_insights=[
            "Monitor delta exposure continuously",
            "Hedge when delta exceeds threshold",
            "Adjust position sizes for gamma risk",
        ],
        usage_count=15,
        success_count=13,
        last_applied=datetime(2024, 1, 18),
        created_at=datetime(2024, 1, 5),
        updated_at=datetime.utcnow(),
        validation_tests=[
            {"test": "backtest", "score": Decimal("88.5")},
            {"test": "live", "score": Decimal("91.2")},
        ],
        related_knowledge=["options-pricing", "volatility-modeling"],
        tags=["options", "greeks", "risk_management"],
    )

    # Test comprehensive properties
    assert knowledge.title == "Options Greeks Risk Management"
    assert knowledge.knowledge_type == KnowledgeType.RISK_INSIGHT
    assert knowledge.confidence == Decimal("92")
    assert len(knowledge.supporting_evidence) == 4
    assert len(knowledge.contradicting_evidence) == 1
    assert len(knowledge.symbols) == 2
    assert len(knowledge.timeframes) == 2
    assert len(knowledge.market_contexts) == 3
    assert len(knowledge.conditions) == 3
    assert len(knowledge.patterns) == 3
    assert len(knowledge.actionable_insights) == 3
    assert knowledge.usage_count == 15
    assert knowledge.success_count == 13
    assert knowledge.last_applied is not None
    assert len(knowledge.validation_tests) == 2
    assert len(knowledge.related_knowledge) == 2
    assert len(knowledge.tags) == 3

    # Test computed properties
    expected_success_rate = (Decimal("13") / Decimal("15")) * 100
    success_rate = expected_success_rate if knowledge.usage_count > 0 else Decimal("0")
    expected_reliability = (knowledge.confidence + success_rate) / 2
    assert abs(knowledge.reliability - expected_reliability) < Decimal("0.01")

    evidence_strength = len(knowledge.supporting_evidence) * 10
    evidence_strength -= len(knowledge.contradicting_evidence) * 5
    assert knowledge.evidence_strength >= Decimal(str(evidence_strength))

    scope = (
        len(knowledge.market_contexts)
        + len(knowledge.symbols)
        + len(knowledge.timeframes)
    )
    assert knowledge.applicability_scope == scope

    # Test evidence management
    updated_knowledge = knowledge.add_supporting_evidence(
        "new-study", increase_confidence=True
    )
    assert len(updated_knowledge.supporting_evidence) == 5
    assert updated_knowledge.confidence > knowledge.confidence

    contradicted = knowledge.add_contradicting_evidence(
        "counter-study", decrease_confidence=True
    )
    assert len(contradicted.contradicting_evidence) == 2
    assert contradicted.confidence < knowledge.confidence

    # Test application recording
    applied_knowledge = knowledge.record_application(successful=True)
    assert applied_knowledge.usage_count == 16
    assert applied_knowledge.success_count == 14

    failed_application = applied_knowledge.record_application(successful=False)
    assert failed_application.usage_count == 17
    assert failed_application.success_count == 14  # No increment

    # Test validation
    validated = knowledge.validate_with_outcome(successful=True)
    assert validated.confidence >= knowledge.confidence

    invalidated = knowledge.invalidate_with_evidence("major-contradiction")
    assert len(invalidated.contradicting_evidence) == 2

    # Test different knowledge types
    market_pattern = Knowledge(
        title="Morning Reversal Pattern",
        knowledge_type=KnowledgeType.MARKET_PATTERN,
        content="Markets often reverse direction in first hour of trading",
        confidence=Decimal("78"),
        supporting_evidence=["pattern-study"],
        market_contexts=["market_open"],
        conditions=["High overnight gap", "Strong volume"],
        patterns=["gap_reversal"],
        actionable_insights=["Wait for first hour confirmation"],
    )
    assert market_pattern.knowledge_type == KnowledgeType.MARKET_PATTERN

    strategy_improvement = Knowledge(
        title="RSI Parameter Optimization",
        knowledge_type=KnowledgeType.STRATEGY_IMPROVEMENT,
        content="RSI period of 21 works better than 14 for daily timeframe",
        confidence=Decimal("85"),
        supporting_evidence=["optimization-study"],
        market_contexts=["trending_market"],
        conditions=["Daily timeframe", "Trending conditions"],
        patterns=["trend_following"],
        actionable_insights=["Use RSI(21) for daily trends"],
    )
    assert strategy_improvement.knowledge_type == KnowledgeType.STRATEGY_IMPROVEMENT


def test_performance_comprehensive():
    """Test comprehensive Performance functionality."""
    performance = Performance(
        entity_type="strategy",
        entity_id="advanced-rsi-v2",
        measurement_period_start=datetime(2024, 1, 1),
        measurement_period_end=datetime(2024, 3, 31),
        total_observations=250,
        successful_observations=195,
        metrics={
            "total_return": Decimal("28.5"),
            "annualized_return": Decimal("114.0"),
            "sharpe_ratio": Decimal("2.3"),
            "max_drawdown": Decimal("6.8"),
            "volatility": Decimal("18.2"),
            "win_rate": Decimal("78.0"),
            "avg_win": Decimal("2.4"),
            "avg_loss": Decimal("1.1"),
            "profit_factor": Decimal("2.1"),
        },
        benchmark_comparisons={
            "SPY": Decimal("7.2"),
            "QQQ": Decimal("9.8"),
            "sector_avg": Decimal("5.1"),
        },
        time_series_data=[
            {"date": "2024-01-01", "value": Decimal("100000")},
            {"date": "2024-01-31", "value": Decimal("105200")},
            {"date": "2024-02-29", "value": Decimal("118300")},
            {"date": "2024-03-31", "value": Decimal("128500")},
        ],
        trend_analysis={
            "direction": "improving",
            "strength": 0.85,
            "consistency": 0.92,
            "momentum": "accelerating",
        },
        risk_measures={
            "var_95": Decimal("3.2"),
            "cvar_95": Decimal("4.8"),
            "max_consecutive_losses": 4,
            "downside_deviation": Decimal("12.1"),
        },
        statistical_measures={
            "strategy_alpha": Decimal("21.3"),
            "market_beta": Decimal("0.85"),
            "sector_exposure": Decimal("0.23"),
        },
    )

    # Test comprehensive properties
    assert performance.entity_type == "strategy"
    assert performance.entity_id == "advanced-rsi-v2"
    assert performance.total_observations == 250
    assert performance.successful_observations == 195
    assert len(performance.metrics) == 9
    assert len(performance.benchmark_comparisons) == 3
    assert len(performance.time_series_data) == 4
    assert len(performance.trend_analysis) == 4
    assert len(performance.risk_measures) == 4
    assert len(performance.statistical_measures) == 3

    # Test computed properties
    expected_success_rate = (Decimal("195") / Decimal("250")) * 100
    assert abs(performance.success_rate - expected_success_rate) < Decimal("0.01")

    # 90 days in Q1
    assert performance.measurement_duration_days == 90

    expected_obs_per_day = Decimal("250") / Decimal("90")
    assert abs(performance.observations_per_day - expected_obs_per_day) < Decimal(
        "0.01"
    )

    # Status checks
    assert performance.has_sufficient_data  # 250 > 30 and 90 > 7
    assert performance.is_improving  # trend direction = "improving"
    assert not performance.is_declining

    # Test metric access
    sharpe = performance.get_metric(PerformanceMetric.SHARPE_RATIO)
    assert sharpe == Decimal("2.3")

    win_rate = performance.get_metric(PerformanceMetric.WIN_RATE)
    assert win_rate == Decimal("78.0")

    # Test benchmark comparison
    vs_spy = performance.compare_to_benchmark("SPY")
    assert vs_spy == Decimal("7.2")

    vs_nonexistent = performance.compare_to_benchmark("NONEXISTENT")
    assert vs_nonexistent is None

    # Test observation addition
    new_observation = performance.add_observation(
        timestamp=datetime(2024, 4, 1),
        metrics={"daily_return": Decimal("1.2"), "volatility": Decimal("15.8")},
        successful=True,
    )
    assert new_observation.total_observations == 251
    assert new_observation.successful_observations == 196
    assert len(new_observation.time_series_data) == 5

    # Test failed observation
    failed_observation = performance.add_observation(
        timestamp=datetime(2024, 4, 2),
        metrics={"daily_return": Decimal("-0.8")},
        successful=False,
    )
    assert failed_observation.total_observations == 251
    assert failed_observation.successful_observations == 195  # No increment

    # Test metrics update
    updated_performance = performance.update_metrics(
        {"new_metric": Decimal("42.0"), "beta": Decimal("1.15")}
    )
    assert updated_performance.metrics["new_metric"] == Decimal("42.0")
    assert updated_performance.metrics["beta"] == Decimal("1.15")
    # Original metrics preserved
    assert updated_performance.metrics["sharpe_ratio"] == Decimal("2.3")

    # Test insufficient data performance
    insufficient_perf = Performance(
        entity_type="new_strategy",
        entity_id="test",
        measurement_period_start=datetime(2024, 3, 1),
        measurement_period_end=datetime(2024, 3, 3),  # Only 2 days
        total_observations=10,  # < 30
        successful_observations=8,
        metrics={},
    )
    assert not insufficient_perf.has_sufficient_data
    assert insufficient_perf.measurement_duration_days == 2

    # Test declining performance
    declining_perf = Performance(
        entity_type="failing_strategy",
        entity_id="test",
        measurement_period_start=datetime(2024, 1, 1),
        measurement_period_end=datetime(2024, 3, 31),
        total_observations=100,
        successful_observations=25,  # Low success rate
        metrics={"total_return": Decimal("-15.2")},
        trend_analysis={"direction": "declining", "strength": 0.9},
    )
    assert declining_perf.is_declining
    assert not declining_perf.is_improving
    assert declining_perf.success_rate == Decimal("25.0")
