"""
Learning entities for VOYAGER-Trader.

This module defines models for VOYAGER's learning capabilities including
Skills, Experience, Knowledge, and Performance tracking.
"""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import Field, computed_field, field_validator

from .base import BaseEntity
from .types import Money, SkillCategory, SkillComplexity, Symbol, TimeFrame


class SkillExecutionResult(str, Enum):
    """Results of skill execution."""

    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    ERROR = "error"
    TIMEOUT = "timeout"


class LearningOutcome(str, Enum):
    """Outcomes of learning experiences."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"


class KnowledgeType(str, Enum):
    """Types of trading knowledge."""

    MARKET_PATTERN = "market_pattern"
    RISK_INSIGHT = "risk_insight"
    STRATEGY_IMPROVEMENT = "strategy_improvement"
    EXECUTION_TECHNIQUE = "execution_technique"
    MARKET_CORRELATION = "market_correlation"
    TIMING_INSIGHT = "timing_insight"
    VOLATILITY_PATTERN = "volatility_pattern"
    SENTIMENT_INDICATOR = "sentiment_indicator"


class PerformanceMetric(str, Enum):
    """Performance metrics for skills and experiences."""

    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    PROFIT_FACTOR = "profit_factor"
    WIN_RATE = "win_rate"
    AVERAGE_WIN = "average_win"
    AVERAGE_LOSS = "average_loss"
    MAX_DRAWDOWN = "max_drawdown"
    SHARPE_RATIO = "sharpe_ratio"
    EXECUTION_TIME = "execution_time"
    RELIABILITY = "reliability"


class Skill(BaseEntity):
    """
    VOYAGER skill entity.

    Represents a learned trading skill with code, metadata,
    performance tracking, and usage statistics.
    """

    name: str = Field(description="Skill name")
    description: str = Field(description="Skill description")
    category: SkillCategory = Field(description="Skill category")
    complexity: SkillComplexity = Field(description="Skill complexity level")
    version: str = Field(default="1.0", description="Skill version")
    code: str = Field(description="Skill implementation code")
    language: str = Field(default="python", description="Programming language")
    dependencies: List[str] = Field(
        default_factory=list, description="Required dependencies/libraries"
    )
    required_skills: List[str] = Field(
        default_factory=list, description="Required prerequisite skills"
    )
    input_schema: Dict[str, Any] = Field(description="Input schema definition")
    output_schema: Dict[str, Any] = Field(description="Output schema definition")
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Default parameters"
    )
    examples: List[Dict[str, Any]] = Field(
        default_factory=list, description="Usage examples"
    )
    performance_metrics: Dict[str, Decimal] = Field(
        default_factory=dict, description="Performance metrics"
    )
    usage_count: int = Field(default=0, description="Number of times used")
    success_count: int = Field(default=0, description="Number of successful executions")
    last_used: Optional[datetime] = Field(
        default=None, description="Last usage timestamp"
    )
    is_active: bool = Field(default=True, description="Whether skill is active")
    tags: List[str] = Field(default_factory=list, description="Skill tags")
    learned_from: Optional[str] = Field(
        default=None, description="Source experience ID"
    )
    created_by: str = Field(default="voyager", description="Creator (voyager/human)")

    @field_validator("usage_count", "success_count")
    @classmethod
    def validate_counts(cls, v: int) -> int:
        """Validate counts are non-negative."""
        if v < 0:
            raise ValueError("Counts must be non-negative")
        return v

    def model_post_init(self, __context) -> None:
        """Validate success count doesn't exceed usage count."""
        if self.success_count > self.usage_count:
            raise ValueError("Success count cannot exceed usage count")

    @computed_field
    @property
    def success_rate(self) -> Decimal:
        """Calculate success rate percentage."""
        if self.usage_count == 0:
            return Decimal("0")
        return (Decimal(str(self.success_count)) / Decimal(str(self.usage_count))) * 100

    @computed_field
    @property
    def failure_count(self) -> int:
        """Calculate number of failed executions."""
        return self.usage_count - self.success_count

    @computed_field
    @property
    def reliability_score(self) -> Decimal:
        """Calculate reliability score based on success rate and usage."""
        if self.usage_count < 10:
            # Penalize skills with low usage
            return self.success_rate * (Decimal(str(self.usage_count)) / 10)
        return self.success_rate

    @property
    def is_reliable(self) -> bool:
        """Check if skill is reliable (>80% success rate with >10 uses)."""
        return self.reliability_score > 80 and self.usage_count > 10

    @property
    def is_experimental(self) -> bool:
        """Check if skill is experimental (low usage or success rate)."""
        return self.usage_count < 5 or self.success_rate < 60

    @property
    def has_dependencies(self) -> bool:
        """Check if skill has dependencies."""
        return len(self.dependencies) > 0 or len(self.required_skills) > 0

    def record_usage(
        self,
        result: SkillExecutionResult,
        performance_data: Optional[Dict[str, Decimal]] = None,
    ) -> "Skill":
        """Record skill usage and update metrics."""
        new_usage_count = self.usage_count + 1
        new_success_count = self.success_count

        if result == SkillExecutionResult.SUCCESS:
            new_success_count += 1

        updated_metrics = self.performance_metrics.copy()
        if performance_data:
            # Update performance metrics (simple averaging for now)
            for metric, value in performance_data.items():
                if metric in updated_metrics:
                    # Weighted average with previous values
                    old_value = updated_metrics[metric]
                    old_weight = Decimal(str(self.usage_count))
                    new_weight = Decimal("1")
                    total_weight = old_weight + new_weight
                    updated_metrics[metric] = (
                        old_value * old_weight + value * new_weight
                    ) / total_weight
                else:
                    updated_metrics[metric] = value

        return self.update(
            usage_count=new_usage_count,
            success_count=new_success_count,
            last_used=datetime.utcnow(),
            performance_metrics=updated_metrics,
        )

    def update_code(self, new_code: str, version: Optional[str] = None) -> "Skill":
        """Update skill code and increment version."""
        if version is None:
            current_version = float(self.version)
            version = str(current_version + 0.1)

        return self.update(code=new_code, version=version)

    def deactivate(self, reason: str = "") -> "Skill":
        """Deactivate skill."""
        tags = self.tags.copy()
        if reason:
            tags.append(f"deactivated:{reason}")

        return self.update(is_active=False, tags=tags)

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        return {
            "usage_count": self.usage_count,
            "success_rate": float(self.success_rate),
            "reliability_score": float(self.reliability_score),
            "is_reliable": self.is_reliable,
            "is_experimental": self.is_experimental,
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "performance_metrics": {
                k: float(v) for k, v in self.performance_metrics.items()
            },
        }


class Experience(BaseEntity):
    """
    Trading experience entity.

    Represents a trading experience with outcomes, lessons learned,
    and contributing factors for future skill development.
    """

    title: str = Field(description="Experience title")
    description: str = Field(description="Experience description")
    context: Dict[str, Any] = Field(
        description="Experience context (market conditions, etc.)"
    )
    actions_taken: List[str] = Field(description="Actions taken during experience")
    outcome: LearningOutcome = Field(description="Experience outcome")
    outcome_details: Dict[str, Any] = Field(description="Detailed outcome information")
    lessons_learned: List[str] = Field(description="Key lessons learned")
    contributing_factors: List[str] = Field(
        description="Factors that influenced outcome"
    )
    market_conditions: Dict[str, Any] = Field(
        description="Market conditions during experience"
    )
    symbols_involved: List[Symbol] = Field(
        default_factory=list, description="Symbols involved"
    )
    timeframe: Optional[TimeFrame] = Field(
        default=None, description="Experience timeframe"
    )
    duration_minutes: Optional[int] = Field(
        default=None, description="Experience duration"
    )
    financial_impact: Optional[Money] = Field(
        default=None, description="Financial impact"
    )
    skills_used: List[str] = Field(
        default_factory=list, description="Skills used during experience"
    )
    skills_discovered: List[str] = Field(
        default_factory=list, description="New skills discovered"
    )
    confidence_before: Decimal = Field(description="Confidence level before (0-100)")
    confidence_after: Decimal = Field(description="Confidence level after (0-100)")
    stress_level: Decimal = Field(description="Stress level during experience (0-100)")
    complexity_score: Decimal = Field(description="Experience complexity (0-100)")
    novelty_score: Decimal = Field(description="Experience novelty (0-100)")
    tags: List[str] = Field(default_factory=list, description="Experience tags")
    related_experiences: List[str] = Field(
        default_factory=list, description="Related experience IDs"
    )

    @field_validator(
        "confidence_before",
        "confidence_after",
        "stress_level",
        "complexity_score",
        "novelty_score",
    )
    @classmethod
    def validate_scores(cls, v: Decimal) -> Decimal:
        """Validate scores are between 0 and 100."""
        if isinstance(v, (int, float)):
            v = Decimal(str(v))
        if v < 0 or v > 100:
            raise ValueError("Scores must be between 0 and 100")
        return v.quantize(Decimal("0.01"))

    @field_validator("duration_minutes")
    @classmethod
    def validate_duration(cls, v: Optional[int]) -> Optional[int]:
        """Validate duration is positive."""
        if v is not None and v <= 0:
            raise ValueError("Duration must be positive")
        return v

    @computed_field
    @property
    def confidence_change(self) -> Decimal:
        """Calculate confidence change."""
        return self.confidence_after - self.confidence_before

    @computed_field
    @property
    def learning_value(self) -> Decimal:
        """Calculate learning value based on multiple factors."""
        # High learning value for:
        # - Novel experiences
        # - Complex experiences
        # - Experiences with significant outcomes
        # - Experiences that changed confidence

        novelty_factor = self.novelty_score / 100
        complexity_factor = self.complexity_score / 100
        outcome_factor = (
            Decimal("1.0")
            if self.outcome in {LearningOutcome.POSITIVE, LearningOutcome.NEGATIVE}
            else Decimal("0.5")
        )
        confidence_factor = abs(self.confidence_change) / 100

        return (
            (novelty_factor + complexity_factor + outcome_factor + confidence_factor)
            / 4
            * 100
        )

    @property
    def is_positive_outcome(self) -> bool:
        """Check if experience had positive outcome."""
        return self.outcome == LearningOutcome.POSITIVE

    @property
    def is_negative_outcome(self) -> bool:
        """Check if experience had negative outcome."""
        return self.outcome == LearningOutcome.NEGATIVE

    @property
    def is_high_stress(self) -> bool:
        """Check if experience was high stress."""
        return self.stress_level > 75

    @property
    def is_novel(self) -> bool:
        """Check if experience was novel."""
        return self.novelty_score > 75

    @property
    def is_complex(self) -> bool:
        """Check if experience was complex."""
        return self.complexity_score > 75

    @property
    def increased_confidence(self) -> bool:
        """Check if experience increased confidence."""
        return self.confidence_change > 0

    def extract_patterns(self) -> List[str]:
        """Extract patterns from experience for knowledge creation."""
        patterns = []

        # Pattern based on outcome and market conditions
        if self.outcome == LearningOutcome.POSITIVE:
            patterns.append(
                f"positive_outcome_with_{len(self.market_conditions)}_market_factors"
            )

        # Pattern based on skills used
        if len(self.skills_used) > 1:
            patterns.append(f"skill_combination_{len(self.skills_used)}_skills")

        # Pattern based on complexity and outcome
        if self.is_complex and self.is_positive_outcome:
            patterns.append("complex_positive_experience")

        # Pattern based on stress and performance
        if self.is_high_stress and self.is_positive_outcome:
            patterns.append("high_stress_success")

        return patterns

    def get_summary(self) -> Dict[str, Any]:
        """Get experience summary."""
        return {
            "title": self.title,
            "outcome": self.outcome.value,
            "learning_value": float(self.learning_value),
            "confidence_change": float(self.confidence_change),
            "financial_impact": (
                str(self.financial_impact) if self.financial_impact else None
            ),
            "lessons_count": len(self.lessons_learned),
            "skills_used": len(self.skills_used),
            "skills_discovered": len(self.skills_discovered),
            "complexity": float(self.complexity_score),
            "novelty": float(self.novelty_score),
        }


class Knowledge(BaseEntity):
    """
    Trading knowledge entity.

    Represents accumulated trading knowledge patterns, insights,
    and wisdom extracted from experiences and market observations.
    """

    title: str = Field(description="Knowledge title")
    knowledge_type: KnowledgeType = Field(description="Type of knowledge")
    content: str = Field(description="Knowledge content/description")
    confidence: Decimal = Field(description="Confidence in this knowledge (0-100)")
    supporting_evidence: List[str] = Field(
        description="Supporting evidence/experience IDs"
    )
    contradicting_evidence: List[str] = Field(
        default_factory=list, description="Contradicting evidence"
    )
    market_contexts: List[str] = Field(description="Market contexts where applicable")
    symbols: List[Symbol] = Field(
        default_factory=list, description="Applicable symbols"
    )
    timeframes: List[TimeFrame] = Field(
        default_factory=list, description="Applicable timeframes"
    )
    conditions: List[str] = Field(
        description="Conditions under which knowledge applies"
    )
    exceptions: List[str] = Field(default_factory=list, description="Known exceptions")
    patterns: List[str] = Field(description="Associated patterns")
    actionable_insights: List[str] = Field(description="Actionable insights derived")
    usage_count: int = Field(default=0, description="Times this knowledge was applied")
    success_count: int = Field(default=0, description="Successful applications")
    validation_tests: List[Dict[str, Any]] = Field(
        default_factory=list, description="Validation test results"
    )
    derived_skills: List[str] = Field(
        default_factory=list, description="Skills derived from this knowledge"
    )
    related_knowledge: List[str] = Field(
        default_factory=list, description="Related knowledge IDs"
    )
    tags: List[str] = Field(default_factory=list, description="Knowledge tags")

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: Decimal) -> Decimal:
        """Validate confidence is between 0 and 100."""
        if isinstance(v, (int, float)):
            v = Decimal(str(v))
        if v < 0 or v > 100:
            raise ValueError("Confidence must be between 0 and 100")
        return v.quantize(Decimal("0.01"))

    @field_validator("usage_count", "success_count")
    @classmethod
    def validate_counts(cls, v: int) -> int:
        """Validate counts are non-negative."""
        if v < 0:
            raise ValueError("Counts must be non-negative")
        return v

    def model_post_init(self, __context) -> None:
        """Validate success count doesn't exceed usage count."""
        if self.success_count > self.usage_count:
            raise ValueError("Success count cannot exceed usage count")

    @computed_field
    @property
    def reliability_score(self) -> Decimal:
        """Calculate reliability score based on validation and usage."""
        if self.usage_count == 0:
            # Base reliability on confidence and evidence
            evidence_factor = min(
                len(self.supporting_evidence) * 10, 50
            )  # Max 50 points
            contradiction_penalty = (
                len(self.contradicting_evidence) * 5
            )  # 5 points per contradiction
            return max(
                Decimal("0"),
                self.confidence * Decimal("0.5")
                + evidence_factor
                - contradiction_penalty,
            )

        # Calculate based on actual usage
        success_rate = (
            Decimal(str(self.success_count)) / Decimal(str(self.usage_count))
        ) * 100
        return (self.confidence + success_rate) / 2

    @computed_field
    @property
    def evidence_strength(self) -> Decimal:
        """Calculate strength of supporting evidence."""
        support_score = len(self.supporting_evidence) * 10
        contradiction_penalty = len(self.contradicting_evidence) * 15
        return max(Decimal("0"), Decimal(str(support_score - contradiction_penalty)))

    @computed_field
    @property
    def applicability_scope(self) -> int:
        """Calculate scope of applicability."""
        scope = 0
        scope += len(self.market_contexts) * 2
        scope += len(self.symbols)
        scope += len(self.timeframes) * 3
        return scope

    @property
    def is_high_confidence(self) -> bool:
        """Check if knowledge has high confidence."""
        return self.confidence > 80

    @property
    def is_well_validated(self) -> bool:
        """Check if knowledge is well validated."""
        return (
            len(self.supporting_evidence) >= 3
            and len(self.contradicting_evidence) == 0
            and self.usage_count > 5
        )

    @property
    def is_controversial(self) -> bool:
        """Check if knowledge is controversial (has contradicting evidence)."""
        return len(self.contradicting_evidence) > 0

    @property
    def is_actionable(self) -> bool:
        """Check if knowledge provides actionable insights."""
        return len(self.actionable_insights) > 0

    @property
    def has_been_applied(self) -> bool:
        """Check if knowledge has been practically applied."""
        return self.usage_count > 0

    def add_supporting_evidence(
        self, experience_id: str, increase_confidence: bool = True
    ) -> "Knowledge":
        """Add supporting evidence and optionally increase confidence."""
        new_evidence = self.supporting_evidence + [experience_id]
        new_confidence = self.confidence

        if increase_confidence and self.confidence < 95:
            # Increase confidence by small amount, max 95%
            increase = min(Decimal("2"), Decimal("95") - self.confidence)
            new_confidence = self.confidence + increase

        return self.update(supporting_evidence=new_evidence, confidence=new_confidence)

    def add_contradicting_evidence(
        self, experience_id: str, decrease_confidence: bool = True
    ) -> "Knowledge":
        """Add contradicting evidence and optionally decrease confidence."""
        new_evidence = self.contradicting_evidence + [experience_id]
        new_confidence = self.confidence

        if decrease_confidence and self.confidence > 10:
            # Decrease confidence, min 10%
            decrease = min(Decimal("5"), self.confidence - Decimal("10"))
            new_confidence = self.confidence - decrease

        return self.update(
            contradicting_evidence=new_evidence, confidence=new_confidence
        )

    def record_application(self, successful: bool) -> "Knowledge":
        """Record application of this knowledge."""
        new_usage = self.usage_count + 1
        new_success = self.success_count + (1 if successful else 0)

        return self.update(usage_count=new_usage, success_count=new_success)

    def derive_skill(self, skill_name: str) -> "Knowledge":
        """Record that a skill was derived from this knowledge."""
        new_derived = self.derived_skills + [skill_name]
        return self.update(derived_skills=new_derived)


class Performance(BaseEntity):
    """
    Performance tracking entity.

    Represents detailed performance analysis for skills, strategies,
    or overall trading performance with comprehensive metrics.
    """

    entity_type: str = Field(description="Type of entity (skill, strategy, portfolio)")
    entity_id: str = Field(description="ID of the entity being tracked")
    measurement_period_start: datetime = Field(
        description="Start of measurement period"
    )
    measurement_period_end: datetime = Field(description="End of measurement period")
    total_observations: int = Field(description="Total number of observations")
    successful_observations: int = Field(
        description="Number of successful observations"
    )
    metrics: Dict[str, Decimal] = Field(description="Performance metrics")
    benchmark_comparisons: Dict[str, Decimal] = Field(
        default_factory=dict, description="Comparisons to benchmarks"
    )
    time_series_data: List[Dict[str, Any]] = Field(
        default_factory=list, description="Time series performance data"
    )
    statistical_measures: Dict[str, Decimal] = Field(
        default_factory=dict, description="Statistical measures"
    )
    risk_measures: Dict[str, Decimal] = Field(
        default_factory=dict, description="Risk measures"
    )
    trend_analysis: Dict[str, Any] = Field(
        default_factory=dict, description="Trend analysis results"
    )
    anomalies: List[Dict[str, Any]] = Field(
        default_factory=list, description="Detected anomalies"
    )
    improvement_suggestions: List[str] = Field(
        default_factory=list, description="Improvement suggestions"
    )

    @field_validator("total_observations", "successful_observations")
    @classmethod
    def validate_observations(cls, v: int) -> int:
        """Validate observation counts."""
        if v < 0:
            raise ValueError("Observation counts must be non-negative")
        return v

    def model_post_init(self, __context) -> None:
        """Validate observation consistency."""
        if self.successful_observations > self.total_observations:
            raise ValueError("Successful observations cannot exceed total observations")

    @computed_field
    @property
    def success_rate(self) -> Decimal:
        """Calculate success rate percentage."""
        if self.total_observations == 0:
            return Decimal("0")
        return (
            Decimal(str(self.successful_observations))
            / Decimal(str(self.total_observations))
        ) * 100

    @computed_field
    @property
    def measurement_duration_days(self) -> int:
        """Calculate measurement period duration in days."""
        return (self.measurement_period_end - self.measurement_period_start).days

    @computed_field
    @property
    def observations_per_day(self) -> Decimal:
        """Calculate average observations per day."""
        if self.measurement_duration_days == 0:
            return Decimal(str(self.total_observations))
        return Decimal(str(self.total_observations)) / Decimal(
            str(self.measurement_duration_days)
        )

    @property
    def has_sufficient_data(self) -> bool:
        """Check if performance has sufficient data for analysis."""
        return self.total_observations >= 30 and self.measurement_duration_days >= 7

    @property
    def is_improving(self) -> bool:
        """Check if performance is improving based on trend analysis."""
        return self.trend_analysis.get("direction") == "improving"

    @property
    def is_declining(self) -> bool:
        """Check if performance is declining."""
        return self.trend_analysis.get("direction") == "declining"

    @property
    def has_anomalies(self) -> bool:
        """Check if performance has detected anomalies."""
        return len(self.anomalies) > 0

    def get_metric(self, metric: PerformanceMetric) -> Optional[Decimal]:
        """Get specific performance metric."""
        return self.metrics.get(metric.value)

    def compare_to_benchmark(self, benchmark_name: str) -> Optional[Decimal]:
        """Compare performance to a specific benchmark."""
        return self.benchmark_comparisons.get(benchmark_name)

    def add_observation(
        self, timestamp: datetime, metrics: Dict[str, Decimal], successful: bool = True
    ) -> "Performance":
        """Add a new performance observation."""
        new_total = self.total_observations + 1
        new_successful = self.successful_observations + (1 if successful else 0)

        # Add to time series
        new_time_series = self.time_series_data + [
            {
                "timestamp": timestamp.isoformat(),
                "metrics": {k: float(v) for k, v in metrics.items()},
                "successful": successful,
            }
        ]

        # Update aggregated metrics (simple averaging for now)
        updated_metrics = self.metrics.copy()
        for metric, value in metrics.items():
            if metric in updated_metrics:
                # Weighted average
                old_weight = Decimal(str(self.total_observations))
                new_weight = Decimal("1")
                total_weight = old_weight + new_weight
                updated_metrics[metric] = (
                    updated_metrics[metric] * old_weight + value * new_weight
                ) / total_weight
            else:
                updated_metrics[metric] = value

        return self.update(
            total_observations=new_total,
            successful_observations=new_successful,
            time_series_data=new_time_series,
            metrics=updated_metrics,
        )

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        return {
            "entity_type": self.entity_type,
            "entity_id": self.entity_id,
            "measurement_period_days": self.measurement_duration_days,
            "total_observations": self.total_observations,
            "success_rate": float(self.success_rate),
            "observations_per_day": float(self.observations_per_day),
            "has_sufficient_data": self.has_sufficient_data,
            "trend": self.trend_analysis.get("direction", "stable"),
            "key_metrics": {
                k: float(v) for k, v in list(self.metrics.items())[:5]
            },  # Top 5 metrics
            "anomalies_count": len(self.anomalies),
            "improvement_suggestions_count": len(self.improvement_suggestions),
        }
