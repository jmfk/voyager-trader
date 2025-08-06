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
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol

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


@dataclass
class TaskTemplate:
    """Template for generating curriculum tasks."""

    name: str
    description: str
    task_type: TaskType
    difficulty_level: DifficultyLevel
    objectives: List[str]
    success_criteria: List[str]
    required_skills: List[str]
    parameters: Dict[str, Any]
    market_conditions: List[MarketCondition]
    estimated_duration_minutes: int


@dataclass
class DifficultyScore:
    """Multi-dimensional difficulty assessment."""

    overall: Decimal  # 0.0 to 1.0
    technical_complexity: Decimal
    market_complexity: Decimal
    risk_level: Decimal
    prerequisite_count: int
    estimated_learning_time: int
    confidence: Decimal  # How confident we are in this assessment


@dataclass
class PerformanceAnalysis:
    """Analysis of agent performance trends."""

    success_rate: Decimal
    improvement_trend: str  # "improving", "declining", "stable"
    learning_velocity: Decimal
    strengths: List[str]
    weaknesses: List[str]
    recommendations: List[str]
    confidence_level: Decimal


@dataclass
class MarketContext:
    """Current market context for curriculum decisions."""

    conditions: List[MarketCondition]
    volatility: Decimal
    trend_strength: Decimal
    liquidity: Decimal
    news_impact: Decimal
    suitable_strategies: List[str]
    risk_factors: List[str]


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

        # Analyze current market context
        market_context = self.context_analyzer.analyze_market_context(environment)

        # Check if conditions are suitable for learning
        if not self.context_analyzer.is_suitable_for_learning(market_context):
            self.logger.info("Market conditions not suitable for learning")
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

        # Track the completion
        self.progress_tracker.track_task_completion(task, success, metrics)

        # Update curriculum
        curriculum = (
            curriculum.complete_task(task.id, metrics)
            if success
            else curriculum.fail_task(task.id, metrics.get("failure_reason", "Unknown"))
        )

        # Check for adaptation triggers
        performance = self.progress_tracker.analyze_performance(agent, curriculum)

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
