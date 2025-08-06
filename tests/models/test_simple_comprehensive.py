"""Simple comprehensive tests to boost coverage without using non-existent methods."""

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
    SkillCategory,
    SkillComplexity,
    Symbol,
    TaskStatus,
    TimeFrame,
)


def test_comprehensive_task():
    """Comprehensive Task test with all fields and computed properties."""
    task = Task(
        title="Advanced Options Trading",
        description="Learn complex options strategies",
        task_type=TaskType.LEARNING,
        priority=TaskPriority.HIGH,
        difficulty=DifficultyLevel.ADVANCED,
        objectives=["Learn iron condors", "Master risk management"],
        success_criteria=["Profitable trades", "Low drawdown"],
        prerequisites=["basic-options"],
        required_skills=["options", "risk-management"],
        target_symbols=[Symbol(code="SPY", asset_class=AssetClass.EQUITY)],
        timeframe=TimeFrame.DAY_1,
        estimated_duration_minutes=240,
        deadline=datetime(2024, 12, 31),
        parameters={"risk_level": "medium", "capital": 10000},
        resources=["options-data", "strategies-book"],
        expected_outcomes=["profitable-strategy"],
        tags=["options", "advanced"],
    )

    # Test all properties
    assert task.title == "Advanced Options Trading"
    assert task.task_type == TaskType.LEARNING
    assert task.priority == TaskPriority.HIGH
    assert task.difficulty == DifficultyLevel.ADVANCED
    assert task.status == TaskStatus.PENDING
    assert len(task.objectives) == 2
    assert len(task.success_criteria) == 2
    assert len(task.prerequisites) == 1
    assert len(task.required_skills) == 2
    assert len(task.target_symbols) == 1
    assert task.timeframe == TimeFrame.DAY_1
    assert task.estimated_duration_minutes == 240
    assert task.deadline is not None
    assert len(task.parameters) == 2
    assert len(task.resources) == 2
    assert len(task.expected_outcomes) == 1
    assert len(task.tags) == 2

    # Test computed properties
    assert task.is_overdue is False  # Deadline in future
    assert task.is_pending()
    assert not task.is_in_progress()
    assert not task.is_completed()
    assert not task.is_failed()
    assert task.has_prerequisites()
    assert task.is_high_priority()
    assert task.completion_rate() == Decimal("0")

    # Test can_start logic
    assert not task.can_start([])  # No prerequisites met
    assert task.can_start(["basic-options"])  # Prerequisites met


def test_comprehensive_environment():
    """Comprehensive Environment test."""
    env = Environment(
        name="Live Trading Environment",
        environment_type=EnvironmentType.LIVE_TRADING,
        description="Production trading environment",
        base_currency=Currency.USD,
        available_symbols=[
            Symbol(code="AAPL", asset_class=AssetClass.EQUITY),
            Symbol(code="MSFT", asset_class=AssetClass.EQUITY),
        ],
        supported_timeframes=[TimeFrame.MINUTE_1, TimeFrame.DAY_1],
        market_hours={"NYSE": "09:30-16:00"},
        trading_constraints={"max_position": 0.1},
        risk_limits={"max_loss": 1000},
        initial_capital=Money(amount=Decimal("100000"), currency=Currency.USD),
        current_capital=Money(amount=Decimal("105000"), currency=Currency.USD),
        commission_structure={"stock": Decimal("0.005")},
        margin_requirements={"initial": Decimal("0.5")},
        data_providers=["Bloomberg"],
        execution_venues=["Interactive Brokers"],
        features=["Level 2 data"],
        limitations=["Market hours only"],
        configuration={"latency": "< 1ms"},
        metrics={"uptime": Decimal("99.9")},
        is_active=True,
    )

    # Test properties
    assert env.name == "Live Trading Environment"
    assert env.environment_type == EnvironmentType.LIVE_TRADING
    assert env.base_currency == Currency.USD
    assert len(env.available_symbols) == 2
    assert len(env.supported_timeframes) == 2
    assert len(env.market_hours) == 1
    assert len(env.trading_constraints) == 1
    assert len(env.risk_limits) == 1
    assert env.initial_capital.amount == Decimal("100000")
    assert env.current_capital.amount == Decimal("105000")
    assert len(env.commission_structure) == 1
    assert len(env.margin_requirements) == 1
    assert len(env.data_providers) == 1
    assert len(env.execution_venues) == 1
    assert len(env.features) == 1
    assert len(env.limitations) == 1
    assert len(env.configuration) == 1
    assert len(env.metrics) == 1
    assert env.is_active

    # Test computed properties
    assert env.symbol_count == 2
    expected_util = (Decimal("105000") - Decimal("100000")) / Decimal("100000") * 100
    assert abs(env.capital_utilization - expected_util) < Decimal("0.01")
    assert env.is_live_environment()
    assert not env.is_simulation()
    assert env.supports_real_money()


def test_comprehensive_agent():
    """Comprehensive Agent test."""
    agent = Agent(
        name="VOYAGER Trading Agent",
        description="Advanced trading agent",
        version="1.0",
        capabilities=["trading", "analysis"],
        current_environment_id="env-123",
        state=AgentState.LEARNING,
        learned_skills=["rsi", "macd"],
        active_tasks={"task-1": "learning-options"},
        completed_tasks=["basic-trading"],
        performance_metrics={"accuracy": Decimal("85.5")},
        total_trades=100,
        successful_trades=85,
        total_pnl=Money(amount=Decimal("5000"), currency=Currency.USD),
        error_count=2,
        last_activity=datetime.utcnow(),
        configuration={"risk_level": "medium"},
    )

    # Test properties
    assert agent.name == "VOYAGER Trading Agent"
    assert agent.version == "1.0"
    assert agent.state == AgentState.LEARNING
    assert agent.current_environment_id == "env-123"
    assert len(agent.capabilities) == 2
    assert len(agent.learned_skills) == 2
    assert len(agent.active_tasks) == 1
    assert len(agent.completed_tasks) == 1
    assert len(agent.performance_metrics) == 1
    assert agent.total_trades == 100
    assert agent.successful_trades == 85
    assert agent.error_count == 2
    assert len(agent.configuration) == 1

    # Test computed properties
    assert agent.skill_count == 2
    assert agent.active_task_count == 1
    expected_success = (Decimal("85") / Decimal("100")) * 100
    assert abs(agent.trade_success_rate - expected_success) < Decimal("0.01")
    assert agent.total_pnl.amount == Decimal("5000")
    assert agent.is_profitable()
    assert agent.has_errors()
    assert agent.is_learning()
    assert not agent.is_trading()


def test_comprehensive_curriculum():
    """Comprehensive Curriculum test."""
    curriculum = Curriculum(
        name="Trading Mastery Curriculum",
        description="Complete trading education",
        version="1.0",
        agent_id="agent-123",
        strategy=CurriculumStrategy.ADAPTIVE,
        current_difficulty=DifficultyLevel.INTERMEDIATE,
        target_skills=["options", "futures", "forex"],
        completed_skills=["stocks"],
        active_tasks={"task-1": TaskPriority.HIGH},
        completed_tasks=["intro"],
        failed_tasks=["advanced-derivatives"],
        performance_history=[{"score": 85, "timestamp": datetime.utcnow()}],
        progress_metrics={"learning_rate": Decimal("2.5")},
        adaptation_parameters={"threshold": 0.8},
        is_active=True,
    )

    # Test properties
    assert curriculum.name == "Trading Mastery Curriculum"
    assert curriculum.version == "1.0"
    assert curriculum.agent_id == "agent-123"
    assert curriculum.strategy == CurriculumStrategy.ADAPTIVE
    assert curriculum.current_difficulty == DifficultyLevel.INTERMEDIATE
    assert len(curriculum.target_skills) == 3
    assert len(curriculum.completed_skills) == 1
    assert len(curriculum.active_tasks) == 1
    assert len(curriculum.completed_tasks) == 1
    assert len(curriculum.failed_tasks) == 1
    assert len(curriculum.performance_history) == 1
    assert len(curriculum.progress_metrics) == 1
    assert len(curriculum.adaptation_parameters) == 1
    assert curriculum.is_active

    # Test computed properties
    assert curriculum.total_tasks == 2  # completed + failed
    assert curriculum.active_task_count == 1
    expected_completion = (
        Decimal("1") / Decimal("2")
    ) * 100  # 1 completed out of 2 total
    assert abs(curriculum.completion_rate - expected_completion) < Decimal("0.01")
    expected_success = (
        Decimal("1") / Decimal("2")
    ) * 100  # 1 completed out of 2 attempted
    assert abs(curriculum.success_rate - expected_success) < Decimal("0.01")
    expected_skill_progress = (
        Decimal("1") / Decimal("3")
    ) * 100  # 1 completed out of 3 target
    assert abs(
        curriculum.skill_development_progress - expected_skill_progress
    ) < Decimal("0.01")


def test_comprehensive_skill():
    """Comprehensive Skill test."""
    skill = Skill(
        name="Advanced RSI Trading",
        description="Sophisticated RSI-based trading strategy",
        category=SkillCategory.TECHNICAL_ANALYSIS,
        complexity=SkillComplexity.ADVANCED,
        version="2.0",
        code="def rsi_strategy(data): return analyze_rsi(data)",
        language="python",
        dependencies=["numpy", "pandas"],
        required_skills=["basic-rsi"],
        input_schema={"data": "DataFrame"},
        output_schema={"signal": "str"},
        parameters={"period": 14},
        examples=[{"input": "sample", "output": "buy"}],
        performance_metrics={"accuracy": Decimal("88.5")},
        usage_count=50,
        success_count=44,
        last_used=datetime.utcnow(),
        is_active=True,
        tags=["rsi", "momentum"],
    )

    # Test properties
    assert skill.name == "Advanced RSI Trading"
    assert skill.category == SkillCategory.TECHNICAL_ANALYSIS
    assert skill.complexity == SkillComplexity.ADVANCED
    assert skill.version == "2.0"
    assert skill.code is not None
    assert skill.language == "python"
    assert len(skill.dependencies) == 2
    assert len(skill.required_skills) == 1
    assert len(skill.input_schema) == 1
    assert len(skill.output_schema) == 1
    assert len(skill.parameters) == 1
    assert len(skill.examples) == 1
    assert len(skill.performance_metrics) == 1
    assert skill.usage_count == 50
    assert skill.success_count == 44
    assert skill.is_active
    assert len(skill.tags) == 2

    # Test computed properties
    expected_success_rate = (Decimal("44") / Decimal("50")) * 100
    assert abs(skill.success_rate - expected_success_rate) < Decimal("0.01")
    expected_failure_count = 50 - 44
    assert skill.failure_count == expected_failure_count
    assert skill.reliability_score > Decimal(
        "50"
    )  # Should be high with good success rate
    assert skill.is_reliable  # High success rate
    assert not skill.is_experimental  # High usage count


def test_comprehensive_experience():
    """Comprehensive Experience test."""
    experience = Experience(
        title="Options Strategy Testing",
        description="Testing iron condor in volatile market",
        context={"volatility": "high", "trend": "sideways"},
        actions_taken=["Setup iron condors", "Managed positions"],
        outcome=LearningOutcome.POSITIVE,
        outcome_details={"return": 15.2, "trades": 20},
        lessons_learned=["Iron condors work in high IV"],
        contributing_factors=["High volatility", "Good timing"],
        market_conditions={"trend": "sideways", "volatility": "high"},
        symbols_involved=[Symbol(code="SPY", asset_class=AssetClass.EQUITY)],
        timeframe=TimeFrame.DAY_1,
        duration_minutes=480,
        financial_impact=Money(amount=Decimal("1520"), currency=Currency.USD),
        confidence_before=Decimal("70"),
        confidence_after=Decimal("85"),
        stress_level=Decimal("60"),
        complexity_score=Decimal("80"),
        novelty_score=Decimal("75"),
        tags=["options", "iron_condor"],
    )

    # Test properties
    assert experience.title == "Options Strategy Testing"
    assert experience.outcome == LearningOutcome.POSITIVE
    assert len(experience.context) == 2
    assert len(experience.actions_taken) == 2
    assert len(experience.outcome_details) == 2
    assert len(experience.lessons_learned) == 1
    assert len(experience.contributing_factors) == 2
    assert len(experience.market_conditions) == 2
    assert len(experience.symbols_involved) == 1
    assert experience.timeframe == TimeFrame.DAY_1
    assert experience.duration_minutes == 480
    assert experience.financial_impact.amount == Decimal("1520")
    assert experience.confidence_before == Decimal("70")
    assert experience.confidence_after == Decimal("85")
    assert experience.stress_level == Decimal("60")
    assert experience.complexity_score == Decimal("80")
    assert experience.novelty_score == Decimal("75")
    assert len(experience.tags) == 2

    # Test computed properties
    expected_confidence_change = Decimal("85") - Decimal("70")
    assert experience.confidence_change == expected_confidence_change
    assert experience.is_positive_outcome
    assert not experience.is_negative_outcome
    assert experience.is_novel  # novelty_score 75 > 50
    assert experience.is_complex  # complexity_score 80 > 70
    assert not experience.is_high_stress  # stress_level 60 < 70
    assert experience.learning_value > Decimal("0")


def test_comprehensive_knowledge():
    """Comprehensive Knowledge test."""
    knowledge = Knowledge(
        title="RSI Divergence Patterns",
        knowledge_type=KnowledgeType.MARKET_PATTERN,
        content="RSI divergences often precede price reversals",
        confidence=Decimal("85"),
        supporting_evidence=["study-1", "backtest-2"],
        contradicting_evidence=["outlier-case"],
        symbols=[Symbol(code="AAPL", asset_class=AssetClass.EQUITY)],
        timeframes=[TimeFrame.HOUR_1, TimeFrame.DAY_1],
        market_contexts=["trending", "volatile"],
        conditions=["Strong trend", "High volume"],
        patterns=["bullish_divergence", "bearish_divergence"],
        actionable_insights=["Look for divergence confirmation"],
        usage_count=25,
        success_count=20,
        last_applied=datetime.utcnow(),
        validation_tests=[{"test": "backtest", "result": "positive"}],
        related_knowledge=["rsi-basics"],
        tags=["rsi", "divergence"],
    )

    # Test properties
    assert knowledge.title == "RSI Divergence Patterns"
    assert knowledge.knowledge_type == KnowledgeType.MARKET_PATTERN
    assert knowledge.confidence == Decimal("85")
    assert len(knowledge.supporting_evidence) == 2
    assert len(knowledge.contradicting_evidence) == 1
    assert len(knowledge.symbols) == 1
    assert len(knowledge.timeframes) == 2
    assert len(knowledge.market_contexts) == 2
    assert len(knowledge.conditions) == 2
    assert len(knowledge.patterns) == 2
    assert len(knowledge.actionable_insights) == 1
    assert knowledge.usage_count == 25
    assert knowledge.success_count == 20
    assert len(knowledge.validation_tests) == 1
    assert len(knowledge.related_knowledge) == 1
    assert len(knowledge.tags) == 2

    # Test computed properties
    expected_success_rate = (Decimal("20") / Decimal("25")) * 100
    expected_reliability = (knowledge.confidence + expected_success_rate) / 2
    assert abs(knowledge.reliability_score - expected_reliability) < Decimal("0.01")
    evidence_strength = (
        len(knowledge.supporting_evidence) * 10
        - len(knowledge.contradicting_evidence) * 15
    )
    assert knowledge.evidence_strength >= Decimal(str(evidence_strength))
    expected_scope = (
        len(knowledge.market_contexts) * 2
        + len(knowledge.symbols)
        + len(knowledge.timeframes) * 3
    )
    assert knowledge.applicability_scope == expected_scope
    assert knowledge.is_high_confidence  # confidence 85 > 80
    assert knowledge.is_controversial  # has contradicting evidence
    assert knowledge.is_actionable  # has actionable insights


def test_comprehensive_performance():
    """Comprehensive Performance test."""
    performance = Performance(
        entity_type="strategy",
        entity_id="rsi-strategy-v2",
        measurement_period_start=datetime(2024, 1, 1),
        measurement_period_end=datetime(2024, 3, 31),
        total_observations=100,
        successful_observations=80,
        metrics={
            "return": Decimal("25.5"),
            "sharpe": Decimal("2.1"),
            "max_drawdown": Decimal("5.2"),
        },
        benchmark_comparisons={"SPY": Decimal("8.2")},
        time_series_data=[
            {"date": "2024-01-01", "value": 100000},
            {"date": "2024-03-31", "value": 125500},
        ],
        statistical_measures={"volatility": Decimal("12.5")},
        risk_measures={"var": Decimal("2.1")},
        trend_analysis={"direction": "improving"},
        anomalies=[{"date": "2024-02-15", "type": "spike"}],
        improvement_suggestions=["Reduce position size"],
    )

    # Test properties
    assert performance.entity_type == "strategy"
    assert performance.entity_id == "rsi-strategy-v2"
    assert performance.total_observations == 100
    assert performance.successful_observations == 80
    assert len(performance.metrics) == 3
    assert len(performance.benchmark_comparisons) == 1
    assert len(performance.time_series_data) == 2
    assert len(performance.statistical_measures) == 1
    assert len(performance.risk_measures) == 1
    assert len(performance.trend_analysis) == 1
    assert len(performance.anomalies) == 1
    assert len(performance.improvement_suggestions) == 1

    # Test computed properties
    expected_success_rate = (Decimal("80") / Decimal("100")) * 100
    assert abs(performance.success_rate - expected_success_rate) < Decimal("0.01")
    assert performance.measurement_duration_days == 90  # Jan 1 to Mar 31
    expected_obs_per_day = Decimal("100") / Decimal("90")
    assert abs(performance.observations_per_day - expected_obs_per_day) < Decimal(
        "0.01"
    )
    assert performance.has_sufficient_data  # 100 > 30 observations and 90 > 7 days
    assert performance.is_improving  # trend direction is "improving"
    assert not performance.is_declining

    # Test metric access
    sharpe = performance.get_metric(PerformanceMetric.SHARPE_RATIO)
    assert sharpe == Decimal("2.1")

    # Test benchmark comparison
    spy_comparison = performance.compare_to_benchmark("SPY")
    assert spy_comparison == Decimal("8.2")
    nonexistent = performance.compare_to_benchmark("NONEXISTENT")
    assert nonexistent is None
