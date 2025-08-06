"""Comprehensive tests for system models to maximize coverage."""

from datetime import datetime
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


def test_task_lifecycle():
    """Test complete Task lifecycle and properties."""
    task = Task(
        title="Master RSI Strategy",
        description="Learn and implement RSI mean reversion",
        task_type=TaskType.LEARNING,
        priority=TaskPriority.HIGH,
        difficulty=DifficultyLevel.BEGINNER,
        objectives=["Learn RSI", "Implement strategy", "Backtest results"],
        success_criteria=["Positive returns", "Sharpe > 1.0"],
        prerequisites=["basic-indicators"],
        required_skills=["python", "trading"],
        target_symbols=[Symbol(code="AAPL", asset_class=AssetClass.EQUITY)],
        timeframe=TimeFrame.HOUR_1,
        estimated_duration_minutes=120,
        deadline=datetime(2024, 12, 31),
        parameters={"risk_level": "low"},
        resources=["documentation", "data"],
        expected_outcomes=["working_strategy"],
        tags=["rsi", "momentum"],
    )

    # Test initial state and properties
    assert task.title == "Master RSI Strategy"
    assert task.task_type == TaskType.LEARNING
    assert task.priority == TaskPriority.HIGH
    assert task.difficulty == DifficultyLevel.BEGINNER
    assert task.status == TaskStatus.PENDING
    assert len(task.objectives) == 3
    assert len(task.prerequisites) == 1
    assert len(task.required_skills) == 2
    assert len(task.target_symbols) == 1
    assert task.timeframe == TimeFrame.HOUR_1
    assert task.estimated_duration_minutes == 120
    assert task.deadline is not None
    assert "risk_level" in task.parameters
    assert len(task.resources) == 2
    assert len(task.expected_outcomes) == 1
    assert len(task.tags) == 2

    # Test readiness check
    assert not task.can_start([])  # No prerequisites completed
    assert task.can_start(["basic-indicators"])  # Prerequisites met

    # Test task start
    started_task = task.start_task()
    assert started_task.status == TaskStatus.IN_PROGRESS
    assert started_task.started_at is not None

    # Test progress update
    progress_task = started_task.update_progress(
        progress=Decimal("50"), feedback="Good progress on RSI implementation"
    )
    assert progress_task.progress_percentage == Decimal("50")
    assert len(progress_task.feedback) == 1

    # Test completion
    completed_task = progress_task.complete_task(
        results={"return": 12.5, "sharpe": 1.3},
        lessons_learned=["RSI works in ranging markets"],
    )
    assert completed_task.status == TaskStatus.COMPLETED
    assert completed_task.completed_at is not None
    assert "return" in completed_task.results
    assert len(completed_task.lessons_learned) == 1

    # Test duration calculation
    assert completed_task.duration_minutes is not None
    assert completed_task.duration_minutes >= 0


def test_task_failure():
    """Test Task failure scenarios."""
    task = Task(
        title="Difficult Task",
        description="A challenging task",
        task_type=TaskType.OPTIMIZATION,
        priority=TaskPriority.MEDIUM,
        difficulty=DifficultyLevel.ADVANCED,
        objectives=["Advanced optimization"],
        success_criteria=["Performance improvement"],
    )

    started_task = task.start_task()
    failed_task = started_task.fail_task("Too complex for current skill level")

    assert failed_task.status == TaskStatus.FAILED
    assert failed_task.completed_at is not None


def test_environment_comprehensive():
    """Test comprehensive Environment functionality."""
    env = Environment(
        name="Advanced Trading Environment",
        environment_type=EnvironmentType.LIVE_TRADING,
        description="Professional trading environment",
        base_currency=Currency.USD,
        available_symbols=[
            Symbol(code="AAPL", asset_class=AssetClass.EQUITY),
            Symbol(code="GOOGL", asset_class=AssetClass.EQUITY),
            Symbol(code="MSFT", asset_class=AssetClass.EQUITY),
        ],
        supported_timeframes=[TimeFrame.MINUTE_1, TimeFrame.HOUR_1, TimeFrame.DAY_1],
        market_hours={"NYSE": "09:30-16:00", "NASDAQ": "09:30-16:00"},
        trading_constraints={
            "max_position_size": 0.1,
            "max_daily_trades": 100,
            "min_order_size": 1,
        },
        risk_limits={
            "max_daily_loss": 5000,
            "max_portfolio_risk": 0.02,
            "max_single_position": 0.05,
        },
        initial_capital=Money(amount=Decimal("500000"), currency=Currency.USD),
        current_capital=Money(amount=Decimal("525000"), currency=Currency.USD),
        commission_structure={
            "stock_trade": Decimal("0.005"),
            "option_trade": Decimal("1.00"),
        },
        margin_requirements={"initial": Decimal("0.5"), "maintenance": Decimal("0.25")},
        data_providers=["Bloomberg", "Reuters", "Yahoo"],
        execution_venues=["Interactive Brokers", "TD Ameritrade"],
        features=["Level 2 data", "Real-time execution", "Risk controls"],
        limitations=["Market hours only", "US markets only"],
        configuration={
            "latency_target": "< 1ms",
            "backup_systems": True,
            "monitoring": "24/7",
        },
        metrics={"uptime": Decimal("99.9"), "avg_latency": Decimal("0.5")},
        is_active=True,
    )

    # Test properties and counts
    assert env.name == "Advanced Trading Environment"
    assert env.environment_type == EnvironmentType.LIVE_TRADING
    assert env.base_currency == Currency.USD
    assert env.symbol_count == 3
    assert len(env.supported_timeframes) == 3
    assert len(env.market_hours) == 2
    assert len(env.trading_constraints) == 3
    assert len(env.risk_limits) == 3
    assert len(env.commission_structure) == 2
    assert len(env.margin_requirements) == 2
    assert len(env.data_providers) == 3
    assert len(env.execution_venues) == 2
    assert len(env.features) == 3
    assert len(env.limitations) == 2
    assert len(env.configuration) == 3
    assert len(env.metrics) == 2
    assert env.is_active

    # Test capital utilization
    expected_utilization = (
        (Decimal("525000") - Decimal("500000")) / Decimal("500000") * 100
    )
    assert abs(env.capital_utilization - expected_utilization) < Decimal("0.01")

    # Test symbol management
    new_symbol = Symbol(code="TSLA", asset_class=AssetClass.EQUITY)
    updated_env = env.add_symbol(new_symbol)
    assert updated_env.symbol_count == 4
    assert new_symbol in updated_env.available_symbols

    removed_env = updated_env.remove_symbol(new_symbol)
    assert removed_env.symbol_count == 3
    assert new_symbol not in removed_env.available_symbols

    # Test capital updates
    new_capital = Money(amount=Decimal("530000"), currency=Currency.USD)
    capital_updated = env.update_capital(new_capital)
    assert capital_updated.current_capital == new_capital
    assert capital_updated.last_updated > env.last_updated

    # Test risk limit access
    daily_loss = env.get_risk_limit("max_daily_loss")
    assert daily_loss == 5000

    nonexistent = env.get_risk_limit("nonexistent")
    assert nonexistent is None

    # Test metrics updates
    new_metrics = {"trades_today": Decimal("45"), "profit_today": Decimal("2500")}
    metrics_updated = env.update_metrics(new_metrics)
    assert metrics_updated.metrics["trades_today"] == Decimal("45")
    assert metrics_updated.metrics["profit_today"] == Decimal("2500")
    # Original metrics should be preserved
    assert metrics_updated.metrics["uptime"] == Decimal("99.9")


def test_agent_comprehensive():
    """Test comprehensive Agent functionality."""
    agent = Agent(
        name="VOYAGER Pro Agent",
        description="Advanced autonomous trading agent",
        version="2.0",
        capabilities=[
            "Strategy development",
            "Risk management",
            "Portfolio optimization",
            "Market analysis",
        ],
        current_environment_id="live-trading-env",
        state=AgentState.LEARNING,
        learned_skills=["rsi-strategy", "momentum-trading"],
        active_tasks={"task-1": "learning-advanced-options"},
        completed_tasks=["basic-trading", "risk-management"],
        performance_metrics={
            "total_trades": Decimal("1000"),
            "win_rate": Decimal("65.5"),
            "avg_return": Decimal("2.1"),
        },
        total_trades=150,
        successful_trades=98,
        total_pnl=Money(amount=Decimal("12500"), currency=Currency.USD),
        error_count=5,
        last_activity=datetime.utcnow(),
        configuration={
            "risk_tolerance": "medium",
            "max_positions": 10,
            "auto_trade": False,
        },
    )

    # Test basic properties
    assert agent.name == "VOYAGER Pro Agent"
    assert agent.version == "2.0"
    assert agent.state == AgentState.LEARNING
    assert agent.current_environment_id == "live-trading-env"
    assert len(agent.capabilities) == 4
    assert len(agent.learned_skills) == 2
    assert len(agent.active_tasks) == 1
    assert len(agent.completed_tasks) == 2
    assert len(agent.performance_metrics) == 3
    assert agent.total_trades == 150
    assert agent.successful_trades == 98
    assert agent.error_count == 5
    assert agent.last_activity is not None
    assert len(agent.configuration) == 3

    # Test computed properties
    assert agent.skill_count == 2
    assert agent.active_task_count == 1
    expected_success_rate = (Decimal("98") / Decimal("150")) * 100
    assert abs(agent.trade_success_rate - expected_success_rate) < Decimal("0.01")
    assert agent.total_pnl.amount == Decimal("12500")
    assert agent.is_profitable
    assert agent.has_errors

    # Test state transitions
    active_agent = agent.transition_to(AgentState.ACTIVE)
    assert active_agent.state == AgentState.ACTIVE
    assert active_agent.is_active

    error_agent = agent.transition_to(AgentState.ERROR)
    assert error_agent.state == AgentState.ERROR
    assert not error_agent.is_active

    # Test skill management
    skilled_agent = agent.learn_skill("options-trading")
    assert skilled_agent.skill_count == 3
    assert "options-trading" in skilled_agent.learned_skills

    # Test task management
    tasked_agent = agent.start_task("advanced-portfolio")
    assert tasked_agent.active_task_count == 2
    assert "advanced-portfolio" in tasked_agent.active_tasks

    completed_agent = tasked_agent.complete_task("task-1")
    assert completed_agent.active_task_count == 1
    assert "learning-advanced-options" in completed_agent.completed_tasks

    # Test trading performance recording
    pnl = Money(amount=Decimal("500"), currency=Currency.USD)
    traded_agent = agent.record_trade(True, pnl)
    assert traded_agent.total_trades == 151
    assert traded_agent.successful_trades == 99
    assert traded_agent.total_pnl.amount == Decimal("13000")

    # Test error recording
    error_recorded = agent.record_error()
    assert error_recorded.error_count == 6

    # Test performance update
    new_metrics = {"new_metric": Decimal("42")}
    perf_updated = agent.update_performance(new_metrics)
    assert perf_updated.performance_metrics["new_metric"] == Decimal("42")
    # Original metrics preserved
    assert perf_updated.performance_metrics["total_trades"] == Decimal("1000")

    # Test agent summary
    summary = agent.get_agent_summary()
    assert summary["name"] == "VOYAGER Pro Agent"
    assert summary["state"] == agent.state.value
    assert "skill_count" in summary
    assert "trade_success_rate" in summary


def test_curriculum_comprehensive():
    """Test comprehensive Curriculum functionality."""
    curriculum = Curriculum(
        name="Professional Trading Curriculum",
        description="Advanced trading education and skill development",
        version="3.0",
        agent_id="agent-pro-123",
        strategy=CurriculumStrategy.ADAPTIVE,
        current_difficulty=DifficultyLevel.INTERMEDIATE,
        target_skills=[
            "Advanced RSI",
            "Options Trading",
            "Portfolio Management",
            "Risk Assessment",
        ],
        completed_skills=["Basic Trading", "Chart Reading"],
        active_tasks={
            "task-rsi": TaskPriority.HIGH,
            "task-options": TaskPriority.MEDIUM,
        },
        completed_tasks=["intro-trading", "basic-indicators"],
        failed_tasks=["complex-derivatives"],
        performance_history=[
            {"task": "intro-trading", "score": 95, "timestamp": datetime(2024, 1, 15)},
            {
                "task": "basic-indicators",
                "score": 88,
                "timestamp": datetime(2024, 1, 20),
            },
        ],
        progress_metrics={
            "learning_velocity": Decimal("2.5"),
            "retention_rate": Decimal("92.3"),
            "application_success": Decimal("87.1"),
        },
        adaptation_parameters={
            "difficulty_threshold": 0.8,
            "success_rate_target": 0.85,
            "failure_tolerance": 0.15,
        },
        created_at=datetime(2024, 1, 1),
        last_updated=datetime.utcnow(),
        is_active=True,
    )

    # Test basic properties
    assert curriculum.name == "Professional Trading Curriculum"
    assert curriculum.version == "3.0"
    assert curriculum.agent_id == "agent-pro-123"
    assert curriculum.strategy == CurriculumStrategy.ADAPTIVE
    assert curriculum.current_difficulty == DifficultyLevel.INTERMEDIATE
    assert len(curriculum.target_skills) == 4
    assert len(curriculum.completed_skills) == 2
    assert len(curriculum.active_tasks) == 2
    assert len(curriculum.completed_tasks) == 2
    assert len(curriculum.failed_tasks) == 1
    assert len(curriculum.performance_history) == 2
    assert len(curriculum.progress_metrics) == 3
    assert len(curriculum.adaptation_parameters) == 3
    assert curriculum.is_active

    # Test computed properties
    total_tasks = len(curriculum.completed_tasks) + len(curriculum.failed_tasks)
    assert curriculum.total_tasks == total_tasks
    assert curriculum.active_task_count == 2

    completion_rate = (len(curriculum.completed_tasks) / total_tasks) * 100
    assert abs(curriculum.completion_rate - completion_rate) < Decimal("0.01")

    success_rate = (len(curriculum.completed_tasks) / total_tasks) * 100
    assert abs(curriculum.success_rate - success_rate) < Decimal("0.01")

    skill_progress = (
        len(curriculum.completed_skills) / len(curriculum.target_skills)
    ) * 100
    assert abs(curriculum.skill_development_progress - skill_progress) < Decimal("0.01")

    # Test task management
    started_curriculum = curriculum.start_task(
        "advanced-portfolio", TaskPriority.CRITICAL
    )
    assert started_curriculum.active_task_count == 3
    assert "advanced-portfolio" in started_curriculum.active_tasks.values()

    completed_curriculum = started_curriculum.complete_task(
        "task-rsi", {"score": 92, "completion_time": 180, "quality": "excellent"}
    )
    assert completed_curriculum.active_task_count == 2
    assert "task-rsi" in completed_curriculum.completed_tasks
    assert len(completed_curriculum.performance_history) == 3

    failed_curriculum = completed_curriculum.fail_task(
        "task-options", "Requires more foundational knowledge"
    )
    assert "task-options" in failed_curriculum.failed_tasks
    assert len(failed_curriculum.performance_history) == 4

    # Test difficulty progression - simulate good performance
    good_performance = [
        {"success_rate": 90, "timestamp": datetime.utcnow()},
        {"success_rate": 92, "timestamp": datetime.utcnow()},
        {"success_rate": 88, "timestamp": datetime.utcnow()},
        {"success_rate": 94, "timestamp": datetime.utcnow()},
        {"success_rate": 89, "timestamp": datetime.utcnow()},
    ]

    high_perf_curriculum = curriculum.update(performance_history=good_performance)
    assert high_perf_curriculum.should_advance_difficulty()

    advanced_curriculum = high_perf_curriculum.advance_difficulty()
    assert advanced_curriculum.current_difficulty == DifficultyLevel.ADVANCED

    # Test difficulty reduction - simulate poor performance
    poor_performance = [
        {"success_rate": 45, "timestamp": datetime.utcnow()},
        {"success_rate": 38, "timestamp": datetime.utcnow()},
        {"success_rate": 42, "timestamp": datetime.utcnow()},
    ]

    poor_curriculum = curriculum.update(performance_history=poor_performance)
    assert poor_curriculum.should_reduce_difficulty()

    reduced_curriculum = poor_curriculum.reduce_difficulty()
    assert reduced_curriculum.current_difficulty == DifficultyLevel.BEGINNER

    # Test progress metrics update
    new_metrics = {"new_metric": Decimal("75.5")}
    metrics_updated = curriculum.update_progress_metrics(new_metrics)
    assert metrics_updated.progress_metrics["new_metric"] == Decimal("75.5")
    # Original metrics preserved
    assert metrics_updated.progress_metrics["learning_velocity"] == Decimal("2.5")

    # Test curriculum summary
    summary = curriculum.get_curriculum_summary()
    assert summary["name"] == "Professional Trading Curriculum"
    assert "completion_rate" in summary
    assert "current_difficulty" in summary
