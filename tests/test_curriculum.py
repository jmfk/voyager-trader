"""Tests for the AutomaticCurriculum functionality."""

# Curriculum tests - no external mocks needed currently

from voyager_trader.core import TradingConfig
from voyager_trader.curriculum import AutomaticCurriculum, TradingTask


class TestTradingTask:
    """Test the TradingTask dataclass."""

    def test_task_creation(self):
        """Test creating a trading task."""
        task = TradingTask(
            id="test_task",
            description="Test trading task",
            difficulty=0.5,
            market_conditions={"trend": "up"},
            success_criteria={"min_return": 0.05},
        )

        assert task.id == "test_task"
        assert task.description == "Test trading task"
        assert task.difficulty == 0.5
        assert task.market_conditions == {"trend": "up"}
        assert task.success_criteria == {"min_return": 0.05}
        assert task.prerequisites is None


class TestAutomaticCurriculum:
    """Test the AutomaticCurriculum class."""

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
        assert isinstance(task, TradingTask)
        assert task.id == "basic_trend_following"
        assert task.difficulty == 0.3

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
        assert progression == []
