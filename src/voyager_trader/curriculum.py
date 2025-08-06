"""
Automatic Curriculum System implementation.

This module implements VOYAGER's automatic curriculum system with five core components:
- CurriculumGenerator: Creates new trading objectives and tasks
- DifficultyAssessor: Evaluates challenge complexity and appropriateness
- ProgressTracker: Monitors learning success rates and performance metrics
- AdaptiveEngine: Modifies curriculum based on performance feedback
- ContextAnalyzer: Considers market conditions in curriculum design

Follows the architecture defined in ADR-0010.
"""

import logging
import time
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol

from pydantic import BaseModel, Field, field_validator

from .models.system import (
    Agent,
    Curriculum,
    DifficultyLevel,
    Environment,
    Task,
    TaskType,
)


class MarketCondition(str, Enum):
    """Market condition types for context analysis."""

    BULL_MARKET = "bull_market"
    BEAR_MARKET = "bear_market"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    TRENDING = "trending"
    RANGING = "ranging"
    NEWS_DRIVEN = "news_driven"
    EARNINGS_SEASON = "earnings_season"
    MARKET_OPEN = "market_open"
    MARKET_CLOSE = "market_close"


class AdaptationTrigger(str, Enum):
    """Triggers for curriculum adaptation."""

    PERFORMANCE_DECLINE = "performance_decline"
    PERFORMANCE_IMPROVEMENT = "performance_improvement"
    TASK_COMPLETION = "task_completion"
    TASK_FAILURE = "task_failure"
    MARKET_CHANGE = "market_change"
    LEARNING_PLATEAU = "learning_plateau"
    SKILL_MASTERY = "skill_mastery"
    TIME_BASED = "time_based"


class TaskTemplate(BaseModel):
    """Template for generating curriculum tasks with validation."""

    name: str = Field(..., min_length=1, max_length=100)
    description: str = Field(..., min_length=1, max_length=500)
    task_type: TaskType
    difficulty_level: DifficultyLevel
    objectives: List[str] = Field(..., min_length=1, max_length=10)
    success_criteria: List[str] = Field(..., min_length=1, max_length=10)
    required_skills: List[str] = Field(default_factory=list, max_length=20)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    market_conditions: List[MarketCondition] = Field(
        default_factory=list, max_length=15
    )
    estimated_duration_minutes: int = Field(..., ge=1, le=1440)  # 1 min to 24 hours

    @field_validator("parameters")
    @classmethod
    def validate_parameters(cls, v):
        """Validate task parameters structure."""
        if not isinstance(v, dict):
            raise ValueError("Parameters must be a dictionary")

        # Validate common parameter types
        for key, value in v.items():
            if not isinstance(key, str):
                raise ValueError(f"Parameter key '{key}' must be a string")
            if key.startswith("_"):
                raise ValueError(f"Parameter key '{key}' cannot start with underscore")

        return v


class DifficultyScore(BaseModel):
    """Multi-dimensional difficulty assessment with validation."""

    overall: Decimal = Field(..., ge=0.0, le=1.0)
    technical_complexity: Decimal = Field(..., ge=0.0, le=1.0)
    market_complexity: Decimal = Field(..., ge=0.0, le=1.0)
    risk_level: Decimal = Field(..., ge=0.0, le=1.0)
    prerequisite_count: int = Field(..., ge=0, le=50)
    estimated_learning_time: int = Field(..., ge=1, le=10080)  # 1 min to 1 week
    confidence: Decimal = Field(..., ge=0.0, le=1.0)


class PerformanceAnalysis(BaseModel):
    """Analysis of agent performance trends with validation."""

    success_rate: Decimal = Field(..., ge=0.0, le=1.0)
    improvement_trend: str = Field(..., pattern=r"^(improving|declining|stable)$")
    learning_velocity: Decimal = Field(..., ge=0.0)
    strengths: List[str] = Field(default_factory=list, max_length=20)
    weaknesses: List[str] = Field(default_factory=list, max_length=20)
    recommendations: List[str] = Field(default_factory=list, max_length=20)
    confidence_level: Decimal = Field(..., ge=0.0, le=1.0)


class MarketContext(BaseModel):
    """Current market context for curriculum decisions with validation."""

    conditions: List[MarketCondition] = Field(default_factory=list, max_length=15)
    volatility: Decimal = Field(..., ge=0.0, le=2.0)  # 0-200% volatility
    trend_strength: Decimal = Field(..., ge=0.0, le=1.0)
    liquidity: Decimal = Field(..., ge=0.0, le=1.0)
    news_impact: Decimal = Field(..., ge=0.0, le=1.0)
    suitable_strategies: List[str] = Field(default_factory=list, max_length=30)
    risk_factors: List[str] = Field(default_factory=list, max_length=20)


# Core Curriculum Interfaces


class CurriculumGenerator(Protocol):
    """Interface for generating curriculum tasks."""

    def generate_task(
        self, agent: Agent, curriculum: Curriculum, context: MarketContext
    ) -> Optional[Task]:
        """Generate next appropriate task for the agent."""
        ...

    def get_task_templates(self, difficulty: DifficultyLevel) -> List[TaskTemplate]:
        """Get available task templates for given difficulty."""
        ...

    def validate_task(self, task: Task, agent: Agent) -> bool:
        """Validate if task is appropriate for agent."""
        ...


class DifficultyAssessor(Protocol):
    """Interface for assessing task difficulty."""

    def assess_task_difficulty(
        self, task: Task, context: MarketContext
    ) -> DifficultyScore:
        """Assess the difficulty of a given task."""
        ...

    def assess_template_difficulty(self, template: TaskTemplate) -> DifficultyScore:
        """Assess the difficulty of a task template."""
        ...

    def is_appropriate_difficulty(
        self, score: DifficultyScore, agent: Agent, target_level: DifficultyLevel
    ) -> bool:
        """Check if difficulty is appropriate for agent's current level."""
        ...


class ProgressTracker(Protocol):
    """Interface for tracking learning progress."""

    def analyze_performance(
        self, agent: Agent, curriculum: Curriculum
    ) -> PerformanceAnalysis:
        """Analyze agent's learning performance."""
        ...

    def track_task_completion(
        self, task: Task, success: bool, metrics: Dict[str, Any]
    ) -> None:
        """Track completion of a specific task."""
        ...

    def detect_learning_plateau(self, agent: Agent) -> bool:
        """Detect if agent has hit a learning plateau."""
        ...

    def get_learning_trends(self, agent: Agent) -> Dict[str, Decimal]:
        """Get learning trends by skill category."""
        ...


class AdaptiveEngine(Protocol):
    """Interface for curriculum adaptation."""

    def should_adapt_curriculum(
        self,
        curriculum: Curriculum,
        analysis: PerformanceAnalysis,
        trigger: AdaptationTrigger,
    ) -> bool:
        """Determine if curriculum needs adaptation."""
        ...

    def adapt_curriculum(
        self,
        curriculum: Curriculum,
        analysis: PerformanceAnalysis,
        trigger: AdaptationTrigger,
    ) -> Curriculum:
        """Adapt curriculum based on performance analysis."""
        ...

    def suggest_difficulty_adjustment(
        self, current_level: DifficultyLevel, analysis: PerformanceAnalysis
    ) -> Optional[DifficultyLevel]:
        """Suggest difficulty level adjustment."""
        ...


class ContextAnalyzer(Protocol):
    """Interface for analyzing market context."""

    def analyze_market_context(self, environment: Environment) -> MarketContext:
        """Analyze current market conditions."""
        ...

    def is_suitable_for_learning(self, context: MarketContext) -> bool:
        """Check if market conditions are suitable for learning."""
        ...

    def get_recommended_task_types(self, context: MarketContext) -> List[TaskType]:
        """Get task types recommended for current market conditions."""
        ...

    def assess_risk_level(self, context: MarketContext) -> Decimal:
        """Assess current market risk level."""
        ...


# Main Curriculum Service


class AutomaticCurriculumService:
    """
    Main service orchestrating the automatic curriculum system.

    Coordinates the five core components to provide adaptive learning
    curriculum for the VOYAGER trading agent.
    """

    def __init__(
        self,
        generator: CurriculumGenerator,
        difficulty_assessor: DifficultyAssessor,
        progress_tracker: ProgressTracker,
        adaptive_engine: AdaptiveEngine,
        context_analyzer: ContextAnalyzer,
        config: Dict[str, Any],
    ):
        """Initialize the curriculum service with all components."""
        self.generator = generator
        self.difficulty_assessor = difficulty_assessor
        self.progress_tracker = progress_tracker
        self.adaptive_engine = adaptive_engine
        self.context_analyzer = context_analyzer
        self.config = config
        self.logger = logging.getLogger(__name__)

    def generate_next_task(
        self, agent: Agent, curriculum: Curriculum, environment: Environment
    ) -> Optional[Task]:
        """
        Generate the next appropriate task for the agent.

        This is the main entry point that orchestrates all components.
        """
        self.logger.info(f"Generating next task for agent {agent.name}")
        self.logger.debug(
            f"Agent state: completed_tasks={len(agent.completed_tasks)}, "
            f"active_tasks={len(agent.active_tasks)}, "
            f"completion_rate={agent.task_completion_rate}"
        )

        # Analyze current market context
        start_time = time.time()
        market_context = self.context_analyzer.analyze_market_context(environment)
        context_analysis_time = time.time() - start_time

        self.logger.debug(
            f"Market context analysis completed in {context_analysis_time:.3f}s"
        )
        self.logger.debug(
            f"Market conditions: {market_context.conditions}, "
            f"volatility={market_context.volatility}, "
            f"trend_strength={market_context.trend_strength}"
        )

        # Check if conditions are suitable for learning
        if not self.context_analyzer.is_suitable_for_learning(market_context):
            self.logger.warning(
                f"Market conditions not suitable for learning: "
                f"conditions={market_context.conditions}, "
                f"volatility={market_context.volatility}"
            )
            return None

        # Analyze agent performance
        performance = self.progress_tracker.analyze_performance(agent, curriculum)

        # Check if curriculum needs adaptation
        if self.adaptive_engine.should_adapt_curriculum(
            curriculum, performance, AdaptationTrigger.PERFORMANCE_IMPROVEMENT
        ):
            self.logger.info("Adapting curriculum based on performance")
            curriculum = self.adaptive_engine.adapt_curriculum(
                curriculum, performance, AdaptationTrigger.PERFORMANCE_IMPROVEMENT
            )

        # Generate task using the generator
        task = self.generator.generate_task(agent, curriculum, market_context)

        if task:
            # Assess task difficulty
            difficulty_score = self.difficulty_assessor.assess_task_difficulty(
                task, market_context
            )

            # Validate appropriateness
            if not self.difficulty_assessor.is_appropriate_difficulty(
                difficulty_score, agent, curriculum.current_difficulty
            ):
                self.logger.info("Generated task difficulty not appropriate, retrying")
                return None

            self.logger.info(f"Generated task: {task.title}")

        return task

    def complete_task(
        self,
        task: Task,
        agent: Agent,
        curriculum: Curriculum,
        success: bool,
        metrics: Dict[str, Any],
    ) -> Curriculum:
        """Handle task completion and update curriculum."""
        self.logger.info(f"Completing task {task.title}, success: {success}")
        self.logger.debug(f"Task metrics: {metrics}")

        # Track the completion
        start_time = time.time()
        self.progress_tracker.track_task_completion(task, success, metrics)
        tracking_time = time.time() - start_time
        self.logger.debug(f"Task completion tracking took {tracking_time:.3f}s")

        # Update curriculum
        curriculum = (
            curriculum.complete_task(task.id, metrics)
            if success
            else curriculum.fail_task(task.id, metrics.get("failure_reason", "Unknown"))
        )

        self.logger.debug(
            f"Curriculum updated: difficulty={curriculum.current_difficulty}, "
            f"completed_tasks={len(curriculum.completed_tasks)}"
        )

        # Check for adaptation triggers
        performance = self.progress_tracker.analyze_performance(agent, curriculum)
        self.logger.debug(
            f"Performance analysis: success_rate={performance.success_rate}, "
            f"trend={performance.improvement_trend}, "
            f"velocity={performance.learning_velocity}"
        )

        trigger = (
            AdaptationTrigger.TASK_COMPLETION
            if success
            else AdaptationTrigger.TASK_FAILURE
        )

        if self.adaptive_engine.should_adapt_curriculum(
            curriculum, performance, trigger
        ):
            curriculum = self.adaptive_engine.adapt_curriculum(
                curriculum, performance, trigger
            )

        return curriculum

    def get_curriculum_status(
        self, agent: Agent, curriculum: Curriculum
    ) -> Dict[str, Any]:
        """Get comprehensive status of the curriculum system."""
        performance = self.progress_tracker.analyze_performance(agent, curriculum)
        trends = self.progress_tracker.get_learning_trends(agent)

        return {
            "curriculum_summary": curriculum.get_curriculum_summary(),
            "performance_analysis": {
                "success_rate": float(performance.success_rate),
                "trend": performance.improvement_trend,
                "learning_velocity": float(performance.learning_velocity),
                "confidence": float(performance.confidence_level),
            },
            "learning_trends": {k: float(v) for k, v in trends.items()},
            "needs_adaptation": self.adaptive_engine.should_adapt_curriculum(
                curriculum, performance, AdaptationTrigger.TIME_BASED
            ),
            "plateau_detected": self.progress_tracker.detect_learning_plateau(agent),
        }


# Legacy compatibility class
class AutomaticCurriculum(AutomaticCurriculumService):
    """Legacy class for backward compatibility."""

    def __init__(self, config):
        """Initialize with default implementations (to be provided)."""
        # TODO: Initialize with concrete implementations
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.current_tasks = []
        self.completed_tasks = []
        self.task_history = {}

    def generate_next_task(
        self, agent_performance: Dict[str, Any], market_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Legacy method for backward compatibility."""
        self.logger.info("Using legacy task generation...")

        # Simplified task structure for compatibility
        return {
            "id": "basic_trend_following",
            "description": "Implement basic trend following strategy",
            "difficulty": 0.3,
            "market_conditions": market_data,
            "success_criteria": {"min_return": 0.05, "max_drawdown": 0.1},
        }

    def update_task_progress(self, task_id: str, performance: Dict[str, Any]) -> None:
        """Legacy method for updating task progress."""
        self.task_history[task_id] = performance
        self.logger.info(f"Updated progress for task {task_id}")

    def get_difficulty_progression(self) -> List[float]:
        """Legacy method for getting difficulty progression."""
        return [0.1, 0.2, 0.3, 0.4, 0.5]  # Placeholder progression
