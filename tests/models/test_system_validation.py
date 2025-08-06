"""Validation and advanced tests for system models to improve coverage."""

from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from src.voyager_trader.models.system import (
    Agent,
    Curriculum,
    CurriculumStrategy,
    DifficultyLevel,
    Task,
    TaskPriority,
    TaskType,
)
from src.voyager_trader.models.types import AssetClass, Symbol, TaskStatus, TimeFrame


def test_task_progress_validation_int_conversion():
    """Test task progress validation with int conversion."""
    task = Task(
        title="Test Task",
        description="Test description",
        task_type=TaskType.LEARNING,
        priority=TaskPriority.MEDIUM,
        difficulty=DifficultyLevel.BEGINNER,
        objectives=["Learn basics"],
        success_criteria=["Complete tutorial"],
        progress_percentage=75,  # int input
    )
    assert task.progress_percentage == Decimal("75.00")


def test_task_progress_validation_float_conversion():
    """Test task progress validation with float conversion."""
    task = Task(
        title="Test Task",
        description="Test description",
        task_type=TaskType.LEARNING,
        priority=TaskPriority.MEDIUM,
        difficulty=DifficultyLevel.BEGINNER,
        objectives=["Learn basics"],
        success_criteria=["Complete tutorial"],
        progress_percentage=33.75,  # float input
    )
    assert task.progress_percentage == Decimal("33.75")


def test_task_progress_validation_negative():
    """Test task progress validation with negative value."""
    with pytest.raises(ValueError, match="Progress must be between 0 and 100"):
        Task(
            title="Test Task",
            description="Test description",
            task_type=TaskType.LEARNING,
            priority=TaskPriority.MEDIUM,
            difficulty=DifficultyLevel.BEGINNER,
            objectives=["Learn basics"],
            success_criteria=["Complete tutorial"],
            progress_percentage=-10,
        )


def test_task_progress_validation_over_100():
    """Test task progress validation with value over 100."""
    with pytest.raises(ValueError, match="Progress must be between 0 and 100"):
        Task(
            title="Test Task",
            description="Test description",
            task_type=TaskType.LEARNING,
            priority=TaskPriority.MEDIUM,
            difficulty=DifficultyLevel.BEGINNER,
            objectives=["Learn basics"],
            success_criteria=["Complete tutorial"],
            progress_percentage=150,
        )


def test_task_estimated_duration_validation_positive():
    """Test task estimated duration validation with positive value."""
    task = Task(
        title="Test Task",
        description="Test description",
        task_type=TaskType.LEARNING,
        priority=TaskPriority.MEDIUM,
        difficulty=DifficultyLevel.BEGINNER,
        objectives=["Learn basics"],
        success_criteria=["Complete tutorial"],
        estimated_duration_minutes=120,
    )
    assert task.estimated_duration_minutes == 120


def test_task_estimated_duration_validation_negative():
    """Test task estimated duration validation with negative value."""
    with pytest.raises(ValueError, match="Duration must be positive"):
        Task(
            title="Test Task",
            description="Test description",
            task_type=TaskType.LEARNING,
            priority=TaskPriority.MEDIUM,
            difficulty=DifficultyLevel.BEGINNER,
            objectives=["Learn basics"],
            success_criteria=["Complete tutorial"],
            estimated_duration_minutes=-30,
        )


def test_task_estimated_duration_validation_zero():
    """Test task estimated duration validation with zero value."""
    with pytest.raises(ValueError, match="Duration must be positive"):
        Task(
            title="Test Task",
            description="Test description",
            task_type=TaskType.LEARNING,
            priority=TaskPriority.MEDIUM,
            difficulty=DifficultyLevel.BEGINNER,
            objectives=["Learn basics"],
            success_criteria=["Complete tutorial"],
            estimated_duration_minutes=0,
        )


def test_task_actual_duration_calculation():
    """Test task actual duration calculation."""
    # Create task with start and completion time
    start_time = datetime.utcnow()
    task = Task(
        title="Test Task",
        description="Test description",
        task_type=TaskType.LEARNING,
        priority=TaskPriority.MEDIUM,
        difficulty=DifficultyLevel.BEGINNER,
        objectives=["Learn basics"],
        success_criteria=["Complete tutorial"],
        started_at=start_time,
        completed_at=start_time + timedelta(minutes=45),
    )

    # Should calculate duration based on start and completion time
    assert task.actual_duration_minutes == 45


def test_task_actual_duration_no_completion():
    """Test task actual duration when not completed."""
    task = Task(
        title="Test Task",
        description="Test description",
        task_type=TaskType.LEARNING,
        priority=TaskPriority.MEDIUM,
        difficulty=DifficultyLevel.BEGINNER,
        objectives=["Learn basics"],
        success_criteria=["Complete tutorial"],
        started_at=datetime.utcnow(),
    )

    # Should return None when not completed
    assert task.actual_duration_minutes is None


def test_task_actual_duration_no_start():
    """Test task actual duration when not started."""
    task = Task(
        title="Test Task",
        description="Test description",
        task_type=TaskType.LEARNING,
        priority=TaskPriority.MEDIUM,
        difficulty=DifficultyLevel.BEGINNER,
        objectives=["Learn basics"],
        success_criteria=["Complete tutorial"],
    )

    # Should return None when not started
    assert task.actual_duration_minutes is None


def test_task_with_timeframe():
    """Test task with different timeframes."""
    task = Task(
        title="Timeframe Task",
        description="Task with specific timeframe",
        task_type=TaskType.RESEARCH,
        priority=TaskPriority.MEDIUM,
        difficulty=DifficultyLevel.INTERMEDIATE,
        objectives=["Research patterns"],
        success_criteria=["Find significant patterns"],
        timeframe=TimeFrame.DAY_1,
    )

    assert task.timeframe == TimeFrame.DAY_1


def test_task_with_target_symbols():
    """Test task with target symbols."""
    symbols = [
        Symbol(code="AAPL", asset_class=AssetClass.EQUITY),
        Symbol(code="GOOGL", asset_class=AssetClass.EQUITY),
    ]

    task = Task(
        title="Symbol Task",
        description="Task focused on specific symbols",
        task_type=TaskType.OPTIMIZATION,
        priority=TaskPriority.HIGH,
        difficulty=DifficultyLevel.ADVANCED,
        objectives=["Optimize for symbols"],
        success_criteria=["Improve performance"],
        target_symbols=symbols,
    )

    assert len(task.target_symbols) == 2
    assert task.target_symbols[0].code == "AAPL"


def test_task_with_parameters():
    """Test task with parameters."""
    task = Task(
        title="Parameterized Task",
        description="Task with custom parameters",
        task_type=TaskType.LEARNING,
        priority=TaskPriority.MEDIUM,
        difficulty=DifficultyLevel.BEGINNER,
        objectives=["Learn with parameters"],
        success_criteria=["Meet parameter goals"],
        parameters={"learning_rate": 0.01, "epochs": 100},
    )

    assert task.parameters["learning_rate"] == 0.01
    assert task.parameters["epochs"] == 100


def test_task_with_resources():
    """Test task with required resources."""
    task = Task(
        title="Resource Task",
        description="Task requiring resources",
        task_type=TaskType.PRACTICE,
        priority=TaskPriority.LOW,
        difficulty=DifficultyLevel.BEGINNER,
        objectives=["Practice with resources"],
        success_criteria=["Use resources effectively"],
        resources=["data_feed", "compute_cluster", "api_access"],
    )

    assert len(task.resources) == 3
    assert "data_feed" in task.resources


def test_agent_learn_skill():
    """Test agent learn_skill method."""
    agent = Agent(
        name="Test Agent",
        description="Test agent",
        capabilities=["Basic trading"],
        current_environment_id="env-1",
        learned_skills=["skill1"],
    )

    # Test learn skill method
    updated_agent = agent.learn_skill("skill2")
    assert "skill2" in updated_agent.learned_skills
    assert len(updated_agent.learned_skills) == 2


def test_task_with_deadline():
    """Test task with deadline."""
    future_deadline = datetime.utcnow() + timedelta(hours=24)
    task = Task(
        title="Deadline Task",
        description="Task with deadline",
        task_type=TaskType.ASSESSMENT,
        priority=TaskPriority.HIGH,
        difficulty=DifficultyLevel.INTERMEDIATE,
        objectives=["Complete assessment"],
        success_criteria=["Pass assessment"],
        deadline=future_deadline,
    )

    assert task.deadline == future_deadline


def test_task_with_tags():
    """Test task with tags."""
    task = Task(
        title="Tagged Task",
        description="Task with tags",
        task_type=TaskType.LEARNING,
        priority=TaskPriority.MEDIUM,
        difficulty=DifficultyLevel.BEGINNER,
        objectives=["Learn tagged concepts"],
        success_criteria=["Understand tags"],
        tags=["ml", "trading", "backtesting"],
    )

    assert len(task.tags) == 3
    assert "ml" in task.tags


def test_task_type_properties():
    """Test different task types."""
    learning_task = Task(
        title="Learning Task",
        description="Learn something new",
        task_type=TaskType.LEARNING,
        priority=TaskPriority.HIGH,
        difficulty=DifficultyLevel.BEGINNER,
        objectives=["Learn basics"],
        success_criteria=["Complete tutorial"],
    )

    exploration_task = Task(
        title="Exploration Task",
        description="Explore new strategies",
        task_type=TaskType.EXPLORATION,
        priority=TaskPriority.MEDIUM,
        difficulty=DifficultyLevel.INTERMEDIATE,
        objectives=["Explore strategies"],
        success_criteria=["Find viable strategy"],
    )

    assert learning_task.task_type == TaskType.LEARNING
    assert exploration_task.task_type == TaskType.EXPLORATION


def test_task_priority_levels():
    """Test different task priority levels."""
    low_task = Task(
        title="Low Priority Task",
        description="Not urgent",
        task_type=TaskType.PRACTICE,
        priority=TaskPriority.LOW,
        difficulty=DifficultyLevel.BEGINNER,
        objectives=["Practice skills"],
        success_criteria=["Complete practice"],
    )

    critical_task = Task(
        title="Critical Task",
        description="Very urgent",
        task_type=TaskType.VALIDATION,
        priority=TaskPriority.CRITICAL,
        difficulty=DifficultyLevel.EXPERT,
        objectives=["Validate system"],
        success_criteria=["Pass validation"],
    )

    assert low_task.priority == TaskPriority.LOW
    assert critical_task.priority == TaskPriority.CRITICAL


def test_task_difficulty_levels():
    """Test different task difficulty levels."""
    beginner_task = Task(
        title="Beginner Task",
        description="Easy task",
        task_type=TaskType.LEARNING,
        priority=TaskPriority.MEDIUM,
        difficulty=DifficultyLevel.BEGINNER,
        objectives=["Learn basics"],
        success_criteria=["Complete tutorial"],
    )

    expert_task = Task(
        title="Expert Task",
        description="Very difficult task",
        task_type=TaskType.OPTIMIZATION,
        priority=TaskPriority.HIGH,
        difficulty=DifficultyLevel.EXPERT,
        objectives=["Optimize algorithms"],
        success_criteria=["Achieve performance goals"],
    )

    assert beginner_task.difficulty == DifficultyLevel.BEGINNER
    assert expert_task.difficulty == DifficultyLevel.EXPERT


def test_task_status_enum_values():
    """Test task status enum values."""
    task = Task(
        title="Status Task",
        description="Task with status",
        task_type=TaskType.VALIDATION,
        priority=TaskPriority.MEDIUM,
        difficulty=DifficultyLevel.INTERMEDIATE,
        objectives=["Validate status"],
        success_criteria=["Status works"],
        status=TaskStatus.PENDING,
    )

    assert task.status == TaskStatus.PENDING

    # Test update to in progress
    updated_task = task.update(status=TaskStatus.IN_PROGRESS)
    assert updated_task.status == TaskStatus.IN_PROGRESS


def test_task_feedback_and_lessons():
    """Test task feedback and lessons learned."""
    task = Task(
        title="Learning Task",
        description="Task that teaches",
        task_type=TaskType.LEARNING,
        priority=TaskPriority.MEDIUM,
        difficulty=DifficultyLevel.BEGINNER,
        objectives=["Learn something"],
        success_criteria=["Knowledge gained"],
        feedback=["Good progress", "Need more practice"],
        lessons_learned=["Patience is key", "Practice makes perfect"],
    )

    assert len(task.feedback) == 2
    assert len(task.lessons_learned) == 2
    assert "Good progress" in task.feedback
    assert "Patience is key" in task.lessons_learned


def test_curriculum_strategies():
    """Test different curriculum strategies."""
    progressive = Curriculum(
        name="Progressive Curriculum",
        description="Progressive learning",
        agent_id="agent-1",
        strategy=CurriculumStrategy.PROGRESSIVE,
        target_skills=["skill1"],
    )

    adaptive = Curriculum(
        name="Adaptive Curriculum",
        description="Adaptive learning",
        agent_id="agent-2",
        strategy=CurriculumStrategy.ADAPTIVE,
        target_skills=["skill2"],
    )

    assert progressive.strategy == CurriculumStrategy.PROGRESSIVE
    assert adaptive.strategy == CurriculumStrategy.ADAPTIVE
