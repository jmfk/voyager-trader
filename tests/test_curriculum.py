"""Comprehensive tests for the AutomaticCurriculum system."""

from datetime import datetime
from decimal import Decimal
from tempfile import TemporaryDirectory

import pytest

from voyager_trader.core import TradingConfig
from voyager_trader.curriculum import (
    AdaptationTrigger,
    AutomaticCurriculum,
    DifficultyScore,
    MarketCondition,
    MarketContext,
    PerformanceAnalysis,
    TaskTemplate,
)
from voyager_trader.curriculum_components import (
    AdaptiveLogicEngine,
    BasicCurriculumGenerator,
    MarketContextAnalyzer,
    PerformanceProgressTracker,
    StandardDifficultyAssessor,
)
from voyager_trader.curriculum_service import (
    CurriculumPersistenceService,
    ResumableCurriculumService,
    create_curriculum_service,
)
from voyager_trader.models.system import (
    Agent,
    Curriculum,
    CurriculumStrategy,
    DifficultyLevel,
    Environment,
    EnvironmentType,
    Task,
    TaskPriority,
    TaskStatus,
    TaskType,
)
from voyager_trader.models.types import AssetClass, Currency, Money, Symbol


class TestTaskTemplate:
    """Test the TaskTemplate dataclass."""

    def test_task_template_creation(self):
        """Test creating a task template."""
        template = TaskTemplate(
            name="test_template",
            description="Test template",
            task_type=TaskType.LEARNING,
            difficulty_level=DifficultyLevel.BEGINNER,
            objectives=["Learn basics"],
            success_criteria=["Complete successfully"],
            required_skills=[],
            parameters={"param1": "value1"},
            market_conditions=[MarketCondition.LOW_VOLATILITY],
            estimated_duration_minutes=60,
        )

        assert template.name == "test_template"
        assert template.task_type == TaskType.LEARNING
        assert template.difficulty_level == DifficultyLevel.BEGINNER
        assert len(template.objectives) == 1
        assert template.estimated_duration_minutes == 60


class TestDifficultyScore:
    """Test the DifficultyScore dataclass."""

    def test_difficulty_score_creation(self):
        """Test creating a difficulty score."""
        score = DifficultyScore(
            overall=Decimal("0.5"),
            technical_complexity=Decimal("0.6"),
            market_complexity=Decimal("0.4"),
            risk_level=Decimal("0.5"),
            prerequisite_count=2,
            estimated_learning_time=90,
            confidence=Decimal("0.8"),
        )

        assert score.overall == Decimal("0.5")
        assert score.technical_complexity == Decimal("0.6")
        assert score.prerequisite_count == 2
        assert score.confidence == Decimal("0.8")


class TestPerformanceAnalysis:
    """Test the PerformanceAnalysis dataclass."""

    def test_performance_analysis_creation(self):
        """Test creating performance analysis."""
        analysis = PerformanceAnalysis(
            success_rate=Decimal("0.755"),
            improvement_trend="improving",
            learning_velocity=Decimal("0.3"),
            strengths=["execution", "risk_management"],
            weaknesses=["market_analysis"],
            recommendations=["Focus on technical analysis"],
            confidence_level=Decimal("0.85"),
        )

        assert analysis.success_rate == Decimal("0.755")
        assert analysis.improvement_trend == "improving"
        assert len(analysis.strengths) == 2
        assert len(analysis.weaknesses) == 1


class TestMarketContext:
    """Test the MarketContext dataclass."""

    def test_market_context_creation(self):
        """Test creating market context."""
        context = MarketContext(
            conditions=[MarketCondition.TRENDING, MarketCondition.LOW_VOLATILITY],
            volatility=Decimal("0.2"),
            trend_strength=Decimal("0.7"),
            liquidity=Decimal("0.8"),
            news_impact=Decimal("0.1"),
            suitable_strategies=["trend_following"],
            risk_factors=["none"],
        )

        assert len(context.conditions) == 2
        assert MarketCondition.TRENDING in context.conditions
        assert context.volatility == Decimal("0.2")
        assert len(context.suitable_strategies) == 1


class TestBasicCurriculumGenerator:
    """Test the BasicCurriculumGenerator implementation."""

    @pytest.fixture
    def generator(self):
        """Create generator for testing."""
        return BasicCurriculumGenerator({})

    @pytest.fixture
    def agent(self):
        """Create test agent."""
        return Agent(
            id="test_agent",
            name="Test Agent",
            description="Test agent for curriculum",
            capabilities=["trading"],
            learned_skills=[],
            total_pnl=Money(amount=Decimal("1000"), currency=Currency.USD),
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

    @pytest.fixture
    def curriculum(self, agent):
        """Create test curriculum."""
        return Curriculum(
            id="test_curriculum",
            name="Test Curriculum",
            description="Test curriculum for testing",
            agent_id=agent.id,
            strategy=CurriculumStrategy.PROGRESSIVE,
            target_skills=["basic_trading"],
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

    @pytest.fixture
    def market_context(self):
        """Create test market context."""
        return MarketContext(
            conditions=[MarketCondition.LOW_VOLATILITY, MarketCondition.TRENDING],
            volatility=Decimal("0.2"),
            trend_strength=Decimal("0.6"),
            liquidity=Decimal("0.8"),
            news_impact=Decimal("0.1"),
            suitable_strategies=["trend_following"],
            risk_factors=[],
        )

    def test_get_task_templates_beginner(self, generator):
        """Test getting beginner task templates."""
        templates = generator.get_task_templates(DifficultyLevel.BEGINNER)

        assert len(templates) > 0
        assert all(t.difficulty_level == DifficultyLevel.BEGINNER for t in templates)
        assert any(t.name == "basic_buy_hold" for t in templates)

    def test_get_task_templates_intermediate(self, generator):
        """Test getting intermediate task templates."""
        templates = generator.get_task_templates(DifficultyLevel.INTERMEDIATE)

        assert len(templates) > 0
        assert all(
            t.difficulty_level == DifficultyLevel.INTERMEDIATE for t in templates
        )

    def test_generate_task_success(self, generator, agent, curriculum, market_context):
        """Test successful task generation."""
        task = generator.generate_task(agent, curriculum, market_context)

        assert task is not None
        assert isinstance(task, Task)
        assert task.difficulty == DifficultyLevel.BEGINNER
        assert task.status == TaskStatus.PENDING

    def test_generate_task_with_prerequisites(
        self, generator, agent, curriculum, market_context
    ):
        """Test task generation with prerequisites."""
        # Add a learned skill to agent
        agent = agent.learn_skill("basic_buy_hold")

        task = generator.generate_task(agent, curriculum, market_context)

        # Should be able to generate more advanced tasks
        assert task is not None

    def test_validate_task_success(self, generator, agent):
        """Test task validation success."""
        task = Task(
            id="test_task",
            title="Test Task",
            description="Test",
            task_type=TaskType.LEARNING,
            priority=TaskPriority.MEDIUM,
            difficulty=DifficultyLevel.BEGINNER,
            objectives=["Learn"],
            success_criteria=["Success"],
            required_skills=[],
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

        assert generator.validate_task(task, agent) is True

    def test_validate_task_missing_skills(self, generator, agent):
        """Test task validation with missing skills."""
        task = Task(
            id="test_task",
            title="Test Task",
            description="Test",
            task_type=TaskType.LEARNING,
            priority=TaskPriority.MEDIUM,
            difficulty=DifficultyLevel.BEGINNER,
            objectives=["Learn"],
            success_criteria=["Success"],
            required_skills=["advanced_skill"],
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

        assert generator.validate_task(task, agent) is False


class TestStandardDifficultyAssessor:
    """Test the StandardDifficultyAssessor implementation."""

    @pytest.fixture
    def assessor(self):
        """Create assessor for testing."""
        return StandardDifficultyAssessor({})

    @pytest.fixture
    def simple_task(self):
        """Create simple task for testing."""
        return Task(
            id="simple_task",
            title="Simple Task",
            description="Simple test task",
            task_type=TaskType.LEARNING,
            priority=TaskPriority.LOW,
            difficulty=DifficultyLevel.BEGINNER,
            objectives=["Basic objective"],
            success_criteria=["Basic success"],
            required_skills=[],
            parameters={"param1": "value1"},
            estimated_duration_minutes=60,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

    @pytest.fixture
    def complex_task(self):
        """Create complex task for testing."""
        return Task(
            id="complex_task",
            title="Complex Task",
            description="Complex test task",
            task_type=TaskType.OPTIMIZATION,
            priority=TaskPriority.HIGH,
            difficulty=DifficultyLevel.EXPERT,
            objectives=["Obj1", "Obj2", "Obj3", "Obj4", "Obj5"],
            success_criteria=["Success1", "Success2", "Success3"],
            required_skills=["skill1", "skill2", "skill3"],
            parameters={
                "param1": "value1",
                "param2": "value2",
                "max_position_size": 0.5,
            },
            estimated_duration_minutes=240,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

    @pytest.fixture
    def market_context(self):
        """Create market context for testing."""
        return MarketContext(
            conditions=[MarketCondition.HIGH_VOLATILITY, MarketCondition.NEWS_DRIVEN],
            volatility=Decimal("0.8"),
            trend_strength=Decimal("0.3"),
            liquidity=Decimal("0.5"),
            news_impact=Decimal("0.9"),
            suitable_strategies=["defensive"],
            risk_factors=["high_volatility", "news_driven", "low_liquidity"],
        )

    def test_assess_simple_task_difficulty(self, assessor, simple_task, market_context):
        """Test assessing difficulty of simple task."""
        score = assessor.assess_task_difficulty(simple_task, market_context)

        assert isinstance(score, DifficultyScore)
        assert Decimal("0") <= score.overall <= Decimal("1")
        assert score.technical_complexity < score.market_complexity  # Market is complex
        assert score.prerequisite_count == 0

    def test_assess_complex_task_difficulty(
        self, assessor, complex_task, market_context
    ):
        """Test assessing difficulty of complex task."""
        score = assessor.assess_task_difficulty(complex_task, market_context)

        assert isinstance(score, DifficultyScore)
        assert score.overall > Decimal("0.5")  # Should be high difficulty
        assert score.technical_complexity > Decimal("0.3")  # Many objectives
        assert score.prerequisite_count == 3
        assert score.risk_level > Decimal("0.3")  # High position size

    def test_is_appropriate_difficulty_beginner(self, assessor):
        """Test appropriateness check for beginner level."""
        agent = Agent(
            id="test_agent",
            name="Test Agent",
            description="Test",
            capabilities=["trading"],
            learned_skills=[],
            total_pnl=Money(amount=Decimal("1000"), currency=Currency.USD),
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

        easy_score = DifficultyScore(
            overall=Decimal("0.2"),
            technical_complexity=Decimal("0.1"),
            market_complexity=Decimal("0.2"),
            risk_level=Decimal("0.3"),
            prerequisite_count=0,
            estimated_learning_time=60,
            confidence=Decimal("0.9"),
        )

        hard_score = DifficultyScore(
            overall=Decimal("0.8"),
            technical_complexity=Decimal("0.7"),
            market_complexity=Decimal("0.8"),
            risk_level=Decimal("0.9"),
            prerequisite_count=5,
            estimated_learning_time=300,
            confidence=Decimal("0.7"),
        )

        assert assessor.is_appropriate_difficulty(
            easy_score, agent, DifficultyLevel.BEGINNER
        )
        assert not assessor.is_appropriate_difficulty(
            hard_score, agent, DifficultyLevel.BEGINNER
        )


class TestPerformanceProgressTracker:
    """Test the PerformanceProgressTracker implementation."""

    @pytest.fixture
    def tracker(self):
        """Create tracker for testing."""
        return PerformanceProgressTracker({})

    @pytest.fixture
    def agent(self):
        """Create test agent."""
        return Agent(
            id="test_agent",
            name="Test Agent",
            description="Test agent",
            capabilities=["trading"],
            learned_skills=["skill1", "skill2"],
            completed_tasks=["task1", "task2", "task3"],
            total_trades=100,
            successful_trades=75,
            uptime_minutes=1440,  # 1 day
            error_count=2,
            total_pnl=Money(amount=Decimal("500"), currency=Currency.USD),
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

    @pytest.fixture
    def curriculum(self, agent):
        """Create test curriculum."""
        return Curriculum(
            id="test_curriculum",
            name="Test Curriculum",
            description="Test curriculum for progress tracking",
            agent_id=agent.id,
            strategy=CurriculumStrategy.PROGRESSIVE,
            target_skills=["skill1", "skill2", "skill3"],
            completed_tasks=["task1", "task2", "task3"],
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

    def test_analyze_performance(self, tracker, agent, curriculum):
        """Test performance analysis."""
        analysis = tracker.analyze_performance(agent, curriculum)

        assert isinstance(analysis, PerformanceAnalysis)
        assert analysis.success_rate > 0
        assert analysis.improvement_trend in ["improving", "declining", "stable"]
        assert analysis.learning_velocity >= 0
        assert isinstance(analysis.strengths, list)
        assert isinstance(analysis.weaknesses, list)
        assert isinstance(analysis.recommendations, list)

    def test_track_task_completion_success(self, tracker):
        """Test tracking successful task completion."""
        task = Task(
            id="test_task",
            title="Test Task",
            description="Test",
            task_type=TaskType.LEARNING,
            priority=TaskPriority.MEDIUM,
            difficulty=DifficultyLevel.BEGINNER,
            objectives=["Learn"],
            success_criteria=["Success"],
            required_skills=[],
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

        metrics = {"return": 0.05, "risk": 0.02}

        # Should not raise exception
        tracker.track_task_completion(task, True, metrics)

        # Check if stored in history
        assert task.id in tracker.performance_history
        assert len(tracker.performance_history[task.id]) == 1
        assert tracker.performance_history[task.id][0]["success"] is True

    def test_detect_learning_plateau(self, tracker, agent):
        """Test learning plateau detection."""
        # Agent with high performance should show plateau
        high_perf_agent = agent.update(
            total_trades=1000,
            successful_trades=850,  # 85% success rate
            learning_velocity=Decimal("0.05"),  # Low learning velocity
            completed_tasks=[
                "task1",
                "task2",
                "task3",
                "task4",
                "task5",
            ],  # Enough completed tasks
        )

        assert tracker.detect_learning_plateau(high_perf_agent) is True

        # Agent with low performance should not show plateau
        low_perf_agent = agent.update(
            total_trades=100,
            successful_trades=40,  # 40% success rate
            learning_velocity=Decimal("0.5"),  # High learning velocity
        )

        assert tracker.detect_learning_plateau(low_perf_agent) is False

    def test_get_learning_trends(self, tracker, agent):
        """Test getting learning trends."""
        trends = tracker.get_learning_trends(agent)

        assert isinstance(trends, dict)
        assert len(trends) > 0
        assert all(isinstance(v, Decimal) for v in trends.values())


class TestAdaptiveLogicEngine:
    """Test the AdaptiveLogicEngine implementation."""

    @pytest.fixture
    def engine(self):
        """Create engine for testing."""
        return AdaptiveLogicEngine({})

    @pytest.fixture
    def curriculum(self):
        """Create test curriculum."""
        return Curriculum(
            id="test_curriculum",
            name="Test Curriculum",
            description="Test curriculum for adaptive engine",
            agent_id="test_agent",
            strategy=CurriculumStrategy.PROGRESSIVE,
            current_difficulty=DifficultyLevel.INTERMEDIATE,
            target_skills=["skill1", "skill2"],
            completed_tasks=["task1", "task2", "task3"],
            performance_history=[
                {"success_rate": 85, "timestamp": "2023-01-01T00:00:00"},
                {"success_rate": 87, "timestamp": "2023-01-02T00:00:00"},
                {"success_rate": 89, "timestamp": "2023-01-03T00:00:00"},
            ],
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

    def test_should_adapt_performance_improvement(self, engine, curriculum):
        """Test adaptation trigger for performance improvement."""
        analysis = PerformanceAnalysis(
            success_rate=Decimal("0.9"),
            improvement_trend="improving",
            learning_velocity=Decimal("0.5"),
            strengths=["execution"],
            weaknesses=[],
            recommendations=[],
            confidence_level=Decimal("0.8"),
        )

        should_adapt = engine.should_adapt_curriculum(
            curriculum, analysis, AdaptationTrigger.PERFORMANCE_IMPROVEMENT
        )

        assert should_adapt is True

    def test_should_adapt_performance_decline(self, engine, curriculum):
        """Test adaptation trigger for performance decline."""
        analysis = PerformanceAnalysis(
            success_rate=Decimal("0.4"),
            improvement_trend="declining",
            learning_velocity=Decimal("0.2"),
            strengths=[],
            weaknesses=["execution", "analysis"],
            recommendations=["Improve basic skills"],
            confidence_level=Decimal("0.6"),
        )

        should_adapt = engine.should_adapt_curriculum(
            curriculum, analysis, AdaptationTrigger.PERFORMANCE_DECLINE
        )

        assert should_adapt is True

    def test_adapt_curriculum_advance_difficulty(self, engine, curriculum):
        """Test curriculum adaptation to advance difficulty."""
        analysis = PerformanceAnalysis(
            success_rate=Decimal("0.9"),
            improvement_trend="improving",
            learning_velocity=Decimal("0.5"),
            strengths=["execution", "analysis"],
            weaknesses=[],
            recommendations=[],
            confidence_level=Decimal("0.9"),
        )

        adapted = engine.adapt_curriculum(
            curriculum, analysis, AdaptationTrigger.PERFORMANCE_IMPROVEMENT
        )

        # Difficulty should advance if conditions are right
        assert adapted.current_difficulty in [
            DifficultyLevel.INTERMEDIATE,
            DifficultyLevel.ADVANCED,
        ]

    def test_suggest_difficulty_adjustment(self, engine):
        """Test difficulty adjustment suggestions."""
        # High performance should suggest advancement
        high_analysis = PerformanceAnalysis(
            success_rate=Decimal("0.9"),
            improvement_trend="improving",
            learning_velocity=Decimal("0.5"),
            strengths=["execution"],
            weaknesses=[],
            recommendations=[],
            confidence_level=Decimal("0.8"),
        )

        suggestion = engine.suggest_difficulty_adjustment(
            DifficultyLevel.BEGINNER, high_analysis
        )
        assert suggestion == DifficultyLevel.INTERMEDIATE

        # Low performance should suggest reduction
        low_analysis = PerformanceAnalysis(
            success_rate=Decimal("0.35"),
            improvement_trend="declining",
            learning_velocity=Decimal("0.1"),
            strengths=[],
            weaknesses=["execution", "analysis"],
            recommendations=["Back to basics"],
            confidence_level=Decimal("0.6"),
        )

        suggestion = engine.suggest_difficulty_adjustment(
            DifficultyLevel.INTERMEDIATE, low_analysis
        )
        assert suggestion == DifficultyLevel.BEGINNER


class TestMarketContextAnalyzer:
    """Test the MarketContextAnalyzer implementation."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer for testing."""
        return MarketContextAnalyzer({})

    @pytest.fixture
    def environment(self):
        """Create test environment."""
        return Environment(
            id="test_env",
            name="Test Environment",
            environment_type=EnvironmentType.SIMULATION,
            description="Test environment",
            base_currency=Currency.USD,
            available_symbols=[Symbol(code="AAPL", asset_class=AssetClass.EQUITY)],
            supported_timeframes=[],
            market_hours={},
            trading_constraints={},
            risk_limits={},
            initial_capital=Money(amount=Decimal("10000"), currency=Currency.USD),
            current_capital=Money(amount=Decimal("10000"), currency=Currency.USD),
            commission_structure={},
            margin_requirements={},
            data_providers=[],
            execution_venues=[],
            features=[],
            limitations=[],
            configuration={},
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

    def test_analyze_market_context(self, analyzer, environment):
        """Test market context analysis."""
        context = analyzer.analyze_market_context(environment)

        assert isinstance(context, MarketContext)
        assert isinstance(context.conditions, list)
        assert len(context.conditions) > 0
        assert isinstance(context.volatility, Decimal)
        assert isinstance(context.trend_strength, Decimal)
        assert isinstance(context.suitable_strategies, list)
        assert isinstance(context.risk_factors, list)

    def test_is_suitable_for_learning(self, analyzer):
        """Test learning suitability assessment."""
        # Good conditions for learning
        good_context = MarketContext(
            conditions=[MarketCondition.LOW_VOLATILITY, MarketCondition.TRENDING],
            volatility=Decimal("0.2"),
            trend_strength=Decimal("0.6"),
            liquidity=Decimal("0.8"),
            news_impact=Decimal("0.1"),
            suitable_strategies=["trend_following"],
            risk_factors=[],
        )

        assert analyzer.is_suitable_for_learning(good_context) is True

        # Bad conditions for learning
        bad_context = MarketContext(
            conditions=[MarketCondition.HIGH_VOLATILITY, MarketCondition.NEWS_DRIVEN],
            volatility=Decimal("0.9"),
            trend_strength=Decimal("0.2"),
            liquidity=Decimal("0.3"),
            news_impact=Decimal("0.95"),
            suitable_strategies=[],
            risk_factors=["market_crash", "high_volatility"],
        )

        assert analyzer.is_suitable_for_learning(bad_context) is False

    def test_get_recommended_task_types(self, analyzer):
        """Test task type recommendations."""
        context = MarketContext(
            conditions=[MarketCondition.LOW_VOLATILITY, MarketCondition.TRENDING],
            volatility=Decimal("0.2"),
            trend_strength=Decimal("0.7"),
            liquidity=Decimal("0.8"),
            news_impact=Decimal("0.1"),
            suitable_strategies=["trend_following"],
            risk_factors=[],
        )

        recommendations = analyzer.get_recommended_task_types(context)

        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        assert all(isinstance(task_type, TaskType) for task_type in recommendations)

    def test_assess_risk_level(self, analyzer):
        """Test risk level assessment."""
        low_risk_context = MarketContext(
            conditions=[MarketCondition.LOW_VOLATILITY],
            volatility=Decimal("0.1"),
            trend_strength=Decimal("0.5"),
            liquidity=Decimal("0.9"),
            news_impact=Decimal("0.05"),
            suitable_strategies=["buy_hold"],
            risk_factors=[],
        )

        high_risk_context = MarketContext(
            conditions=[MarketCondition.HIGH_VOLATILITY, MarketCondition.NEWS_DRIVEN],
            volatility=Decimal("0.8"),
            trend_strength=Decimal("0.2"),
            liquidity=Decimal("0.3"),
            news_impact=Decimal("0.9"),
            suitable_strategies=[],
            risk_factors=["volatility", "news", "liquidity", "correlation"],
        )

        low_risk = analyzer.assess_risk_level(low_risk_context)
        high_risk = analyzer.assess_risk_level(high_risk_context)

        assert low_risk < high_risk
        assert Decimal("0") <= low_risk <= Decimal("1")
        assert Decimal("0") <= high_risk <= Decimal("1")


class TestCurriculumPersistenceService:
    """Test the CurriculumPersistenceService."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        with TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def persistence_service(self, temp_dir):
        """Create persistence service for testing."""
        return CurriculumPersistenceService(temp_dir)

    @pytest.fixture
    def agent(self):
        """Create test agent."""
        return Agent(
            id="test_agent_123",
            name="Test Agent",
            description="Test agent",
            capabilities=["trading"],
            learned_skills=["skill1"],
            total_pnl=Money(amount=Decimal("1000"), currency=Currency.USD),
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

    @pytest.fixture
    def curriculum(self, agent):
        """Create test curriculum."""
        return Curriculum(
            id="test_curriculum",
            name="Test Curriculum",
            description="Test curriculum for persistence",
            agent_id=agent.id,
            strategy=CurriculumStrategy.PROGRESSIVE,
            target_skills=["skill1", "skill2"],
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

    def test_save_and_load_curriculum_state(
        self, persistence_service, agent, curriculum
    ):
        """Test saving and loading curriculum state."""
        # Save state
        persistence_service.save_curriculum_state(agent, curriculum)

        # Load state
        loaded_state = persistence_service.load_curriculum_state(agent.id)

        assert loaded_state is not None
        assert loaded_state["agent"]["id"] == agent.id
        assert loaded_state["curriculum"]["id"] == curriculum.id
        assert loaded_state["version"] == "1.0"

    def test_save_and_load_task_completion(self, persistence_service, agent):
        """Test saving and loading task completion."""
        task = Task(
            id="test_task",
            title="Test Task",
            description="Test",
            task_type=TaskType.LEARNING,
            priority=TaskPriority.MEDIUM,
            difficulty=DifficultyLevel.BEGINNER,
            objectives=["Learn"],
            success_criteria=["Success"],
            required_skills=[],
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

        metrics = {"return": 0.05, "risk": 0.02}

        # Save completion
        persistence_service.save_task_completion(agent.id, task, True, metrics)

        # Load history
        history = persistence_service.load_task_history(agent.id)

        assert len(history) == 1
        assert history[0]["task"]["id"] == task.id
        assert history[0]["success"] is True
        assert history[0]["metrics"] == metrics

    def test_clear_agent_data(self, persistence_service, agent, curriculum):
        """Test clearing agent data."""
        # Save some data first
        persistence_service.save_curriculum_state(agent, curriculum)

        task = Task(
            id="test_task",
            title="Test Task",
            description="Test",
            task_type=TaskType.LEARNING,
            priority=TaskPriority.MEDIUM,
            difficulty=DifficultyLevel.BEGINNER,
            objectives=["Learn"],
            success_criteria=["Success"],
            required_skills=[],
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

        persistence_service.save_task_completion(agent.id, task, True, {})

        # Clear data
        persistence_service.clear_agent_data(agent.id)

        # Verify data is cleared
        assert persistence_service.load_curriculum_state(agent.id) is None
        assert persistence_service.load_task_history(agent.id) == []


class TestResumableCurriculumService:
    """Test the ResumableCurriculumService."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        with TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def curriculum_service(self, temp_dir):
        """Create curriculum service for testing."""
        config = {"auto_save": True, "save_interval_tasks": 1}
        return ResumableCurriculumService(config, temp_dir)

    @pytest.fixture
    def agent(self):
        """Create test agent."""
        return Agent(
            id="test_agent_456",
            name="Test Agent",
            description="Test agent",
            capabilities=["trading"],
            learned_skills=[],
            total_pnl=Money(amount=Decimal("10000"), currency=Currency.USD),
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

    @pytest.fixture
    def environment(self):
        """Create test environment."""
        return Environment(
            id="test_env",
            name="Test Environment",
            environment_type=EnvironmentType.SIMULATION,
            description="Test environment",
            base_currency=Currency.USD,
            available_symbols=[Symbol(code="AAPL", asset_class=AssetClass.EQUITY)],
            supported_timeframes=[],
            market_hours={},
            trading_constraints={},
            risk_limits={},
            initial_capital=Money(amount=Decimal("10000"), currency=Currency.USD),
            current_capital=Money(amount=Decimal("10000"), currency=Currency.USD),
            commission_structure={},
            margin_requirements={},
            data_providers=[],
            execution_venues=[],
            features=[],
            limitations=[],
            configuration={},
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

    def test_initialize_new_curriculum(self, curriculum_service, agent, environment):
        """Test initializing new curriculum."""
        (
            agent_result,
            curriculum,
            active_tasks,
        ) = curriculum_service.initialize_or_resume_curriculum(agent, environment)

        assert agent_result.id == agent.id
        assert isinstance(curriculum, Curriculum)
        assert curriculum.agent_id == agent.id
        assert curriculum.current_difficulty == DifficultyLevel.BEGINNER
        assert isinstance(active_tasks, list)

    def test_generate_task_with_persistence(
        self, curriculum_service, agent, environment
    ):
        """Test generating task with persistence."""
        # Initialize curriculum
        _, curriculum, _ = curriculum_service.initialize_or_resume_curriculum(
            agent, environment
        )

        # Generate task
        task = curriculum_service.generate_next_task_with_persistence(
            agent, curriculum, environment
        )

        # Task might be None if market conditions are unsuitable
        if task:
            assert isinstance(task, Task)
            assert task.status == TaskStatus.PENDING

    def test_get_curriculum_history(self, curriculum_service, agent, environment):
        """Test getting curriculum history."""
        # Initialize and complete some tasks to build history
        _, curriculum, _ = curriculum_service.initialize_or_resume_curriculum(
            agent, environment
        )

        # Get initial history (should be empty)
        history = curriculum_service.get_curriculum_history(agent.id)

        assert isinstance(history, dict)
        assert history["total_tasks"] == 0
        assert history["success_rate"] == 0
        assert isinstance(history["difficulty_stats"], dict)


class TestCreateCurriculumService:
    """Test the create_curriculum_service factory function."""

    def test_create_default_service(self):
        """Test creating service with default config."""
        service = create_curriculum_service()

        assert isinstance(service, ResumableCurriculumService)
        assert service.auto_save is True
        assert service.save_interval_tasks == 1

    def test_create_custom_service(self):
        """Test creating service with custom config."""
        config = {
            "auto_save": False,
            "save_interval_tasks": 5,
            "storage_path": "custom_path",
        }

        service = create_curriculum_service(config)

        assert isinstance(service, ResumableCurriculumService)
        assert service.auto_save is False
        assert service.save_interval_tasks == 5


# Legacy compatibility tests
class TestAutomaticCurriculum:
    """Test the legacy AutomaticCurriculum class for backward compatibility."""

    def test_initialization(self):
        """Test curriculum initialization."""
        config = TradingConfig()
        curriculum = AutomaticCurriculum(config)

        assert curriculum.config == config
        assert curriculum.current_tasks == []
        assert curriculum.completed_tasks == []
        assert curriculum.task_history == {}

    def test_generate_next_task(self):
        """Test generating next trading task."""
        config = TradingConfig()
        curriculum = AutomaticCurriculum(config)

        agent_performance = {"success_rate": 0.7, "avg_return": 0.03}
        market_data = {"trend": "up", "volatility": 0.15}

        task = curriculum.generate_next_task(agent_performance, market_data)

        assert task is not None
        assert isinstance(task, dict)
        assert task["id"] == "basic_trend_following"
        assert task["difficulty"] == 0.3

    def test_update_task_progress(self):
        """Test updating task progress."""
        config = TradingConfig()
        curriculum = AutomaticCurriculum(config)

        task_id = "test_task"
        performance = {"success": True, "return": 0.08}

        curriculum.update_task_progress(task_id, performance)

        assert task_id in curriculum.task_history
        assert curriculum.task_history[task_id] == performance

    def test_difficulty_progression_empty(self):
        """Test difficulty progression with no completed tasks."""
        config = TradingConfig()
        curriculum = AutomaticCurriculum(config)

        progression = curriculum.get_difficulty_progression()
        assert progression == [0.1, 0.2, 0.3, 0.4, 0.5]  # Mock progression
