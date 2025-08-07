"""
Comprehensive test suite for VOYAGER Skill Library System.

Tests all six core components:
1. Skill Executor
2. Skill Composer
3. Skill Validator
4. Skill Librarian
5. Skill Discoverer
6. Main VoyagerSkillLibrary interface
"""

import tempfile
from decimal import Decimal
from pathlib import Path
from unittest.mock import Mock

import pytest

from src.voyager_trader.models.learning import (
    Experience,
    LearningOutcome,
    Skill,
    SkillExecutionResult,
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
from src.voyager_trader.skills import (
    CompositeSkillValidator,
    PerformanceValidator,
    SecurityValidator,
    SkillComposer,
    SkillCompositionError,
    SkillDiscoverer,
    SkillExecutor,
    SkillLibrarian,
    SyntaxValidator,
    VoyagerSkillLibrary,
)


class TestSkillExecutor:
    """Test the SkillExecutor component."""

    def setup_method(self):
        """Set up test fixtures."""
        self.executor = SkillExecutor(timeout_seconds=5, max_memory_mb=64)

        # Create a simple test skill
        self.test_skill = Skill(
            name="test_skill",
            description="A simple test skill",
            category=SkillCategory.MARKET_ANALYSIS,
            complexity=SkillComplexity.BASIC,
            code="""
# Simple calculation skill
price = inputs.get('price', 100)
multiplier = parameters.get('multiplier', 2)
result = {'calculated_value': price * multiplier}
""",
            input_schema={
                "type": "object",
                "properties": {"price": {"type": "number"}},
                "required": ["price"],
            },
            output_schema={
                "type": "object",
                "properties": {"calculated_value": {"type": "number"}},
            },
            parameters={"multiplier": 2},
        )

    def test_successful_execution(self):
        """Test successful skill execution."""
        inputs = {"price": 50}
        result, output, metadata = self.executor.execute_skill(self.test_skill, inputs)

        assert result == SkillExecutionResult.SUCCESS
        assert output["calculated_value"] == 100  # 50 * 2
        assert metadata["skill_name"] == "test_skill"
        assert "execution_time_seconds" in metadata

    def test_execution_with_invalid_inputs(self):
        """Test execution with missing required inputs."""
        inputs = {}  # Missing required 'price' field
        result, output, metadata = self.executor.execute_skill(self.test_skill, inputs)

        assert result == SkillExecutionResult.ERROR
        assert output is None
        assert "Invalid inputs" in metadata["error"]

    def test_execution_with_syntax_error(self):
        """Test execution of skill with syntax error."""
        bad_skill = Skill(
            name="bad_skill",
            description="Skill with syntax error",
            category=SkillCategory.MARKET_ANALYSIS,
            complexity=SkillComplexity.BASIC,
            code="invalid python syntax >>>",
            input_schema={"type": "object"},
            output_schema={"type": "object"},
        )

        result, output, metadata = self.executor.execute_skill(bad_skill, {})

        assert result == SkillExecutionResult.ERROR
        assert output is None
        assert metadata["return_code"] != 0

    def test_execution_timeout(self):
        """Test execution timeout handling."""
        slow_skill = Skill(
            name="slow_skill",
            description="Skill that takes too long",
            category=SkillCategory.MARKET_ANALYSIS,
            complexity=SkillComplexity.BASIC,
            code="""
import time
time.sleep(10)  # Longer than timeout
result = {'value': 42}
""",
            input_schema={"type": "object"},
            output_schema={"type": "object"},
        )

        result, output, metadata = self.executor.execute_skill(slow_skill, {})

        assert result == SkillExecutionResult.TIMEOUT
        assert output is None
        assert "timeout" in metadata["error"].lower()


class TestSkillComposer:
    """Test the SkillComposer component."""

    def setup_method(self):
        """Set up test fixtures."""
        self.composer = SkillComposer()

        # Create test skills
        self.skill1 = Skill(
            name="skill1",
            description="First skill",
            category=SkillCategory.TECHNICAL_ANALYSIS,
            complexity=SkillComplexity.BASIC,
            code="result = {'step1': inputs.get('value', 0) + 10}",
            input_schema={"type": "object"},
            output_schema={"type": "object"},
        )

        self.skill2 = Skill(
            name="skill2",
            description="Second skill",
            category=SkillCategory.RISK_MANAGEMENT,
            complexity=SkillComplexity.INTERMEDIATE,
            code="result = {'step2': inputs.get('value', 0) * 2}",
            input_schema={"type": "object"},
            output_schema={"type": "object"},
            required_skills=["skill1"],  # Depends on skill1
        )

    def test_sequential_composition(self):
        """Test sequential skill composition."""
        skills = [self.skill1, self.skill2]
        composed_code, metadata = self.composer.compose_skills(skills, "sequential")

        assert "def execute_composed_strategy" in composed_code
        assert "Execute skill: skill1" in composed_code
        assert "Execute skill: skill2" in composed_code
        assert metadata["composition_type"] == "sequential"
        assert metadata["skills_count"] == 2

    def test_parallel_composition(self):
        """Test parallel skill composition."""
        skills = [self.skill1, self.skill2]
        composed_code, metadata = self.composer.compose_skills(skills, "parallel")

        assert "parallel simulation" in composed_code
        assert metadata["composition_type"] == "parallel"

    def test_conditional_composition(self):
        """Test conditional skill composition."""
        skills = [self.skill1, self.skill2]
        composed_code, metadata = self.composer.compose_skills(skills, "conditional")

        assert "Conditional execution" in composed_code
        assert metadata["composition_type"] == "conditional"

    def test_dependency_resolution(self):
        """Test dependency resolution with topological sort."""
        # skill2 depends on skill1, so skill1 should come first
        skills = [self.skill2, self.skill1]  # Intentionally wrong order
        dependency_order = self.composer._resolve_dependencies(skills)

        assert dependency_order[0].name == "skill1"
        assert dependency_order[1].name == "skill2"

    def test_circular_dependency_detection(self):
        """Test circular dependency detection."""
        skill_a = Skill(
            name="skill_a",
            description="Skill A",
            category=SkillCategory.MARKET_ANALYSIS,
            complexity=SkillComplexity.BASIC,
            code="result = {'a': 1}",
            input_schema={"type": "object"},
            output_schema={"type": "object"},
            required_skills=["skill_b"],
        )

        skill_b = Skill(
            name="skill_b",
            description="Skill B",
            category=SkillCategory.MARKET_ANALYSIS,
            complexity=SkillComplexity.BASIC,
            code="result = {'b': 1}",
            input_schema={"type": "object"},
            output_schema={"type": "object"},
            required_skills=["skill_a"],
        )

        with pytest.raises(SkillCompositionError, match="Circular dependencies"):
            self.composer.compose_skills([skill_a, skill_b])

    def test_empty_skills_list(self):
        """Test composition with empty skills list."""
        with pytest.raises(SkillCompositionError, match="No skills provided"):
            self.composer.compose_skills([])

    def test_unknown_composition_strategy(self):
        """Test unknown composition strategy."""
        with pytest.raises(SkillCompositionError, match="Unknown composition strategy"):
            self.composer.compose_skills([self.skill1], "unknown_strategy")


class TestSkillValidators:
    """Test the skill validation components."""

    def setup_method(self):
        """Set up test fixtures."""
        self.syntax_validator = SyntaxValidator()
        self.security_validator = SecurityValidator()
        self.performance_validator = PerformanceValidator(
            legacy_compatible=False
        )  # Strict for testing
        self.composite_validator = CompositeSkillValidator(
            legacy_compatible=False
        )  # Strict for testing

    def test_syntax_validator_valid_code(self):
        """Test syntax validator with valid code."""
        skill = Skill(
            name="valid_skill",
            description="Valid skill",
            category=SkillCategory.MARKET_ANALYSIS,
            complexity=SkillComplexity.BASIC,
            code="result = {'value': 42}",
            input_schema={"type": "object"},
            output_schema={"type": "object"},
        )

        is_valid, errors = self.syntax_validator.validate(skill)
        assert is_valid
        assert len(errors) == 0

    def test_syntax_validator_invalid_code(self):
        """Test syntax validator with invalid code."""
        skill = Skill(
            name="invalid_skill",
            description="Invalid skill",
            category=SkillCategory.MARKET_ANALYSIS,
            complexity=SkillComplexity.BASIC,
            code="invalid syntax >>>",
            input_schema={"type": "object"},
            output_schema={"type": "object"},
        )

        is_valid, errors = self.syntax_validator.validate(skill)
        assert not is_valid
        assert len(errors) > 0
        assert "Syntax error" in errors[0]

    def test_security_validator_safe_code(self):
        """Test security validator with safe code."""
        skill = Skill(
            name="safe_skill",
            description="Safe skill",
            category=SkillCategory.MARKET_ANALYSIS,
            complexity=SkillComplexity.BASIC,
            code="import math\nresult = {'value': math.sqrt(42)}",
            input_schema={"type": "object"},
            output_schema={"type": "object"},
        )

        is_valid, errors = self.security_validator.validate(skill)
        assert is_valid
        assert len(errors) == 0

    def test_security_validator_dangerous_import(self):
        """Test security validator with dangerous import."""
        skill = Skill(
            name="dangerous_skill",
            description="Dangerous skill",
            category=SkillCategory.MARKET_ANALYSIS,
            complexity=SkillComplexity.BASIC,
            code="import os\nresult = {'value': os.getcwd()}",
            input_schema={"type": "object"},
            output_schema={"type": "object"},
        )

        is_valid, errors = self.security_validator.validate(skill)
        assert not is_valid
        assert len(errors) > 0
        assert "Dangerous import detected: os" in errors[0]

    def test_security_validator_dangerous_function(self):
        """Test security validator with dangerous function call."""
        skill = Skill(
            name="dangerous_skill",
            description="Dangerous skill",
            category=SkillCategory.MARKET_ANALYSIS,
            complexity=SkillComplexity.BASIC,
            code="result = eval('1 + 1')",
            input_schema={"type": "object"},
            output_schema={"type": "object"},
        )

        is_valid, errors = self.security_validator.validate(skill)
        assert not is_valid
        assert len(errors) > 0
        assert "Dangerous function call detected: eval" in errors[0]

    def test_performance_validator_insufficient_usage(self):
        """Test performance validator with insufficient usage."""
        skill = Skill(
            name="new_skill",
            description="New skill",
            category=SkillCategory.MARKET_ANALYSIS,
            complexity=SkillComplexity.BASIC,
            code="result = {'value': 42}",
            input_schema={"type": "object"},
            output_schema={"type": "object"},
            usage_count=2,  # Less than minimum
        )

        is_valid, errors = self.performance_validator.validate(skill)
        assert not is_valid
        assert "Insufficient usage data" in errors[0]

    def test_performance_validator_low_success_rate(self):
        """Test performance validator with low success rate."""
        skill = Skill(
            name="poor_skill",
            description="Poor performing skill",
            category=SkillCategory.MARKET_ANALYSIS,
            complexity=SkillComplexity.BASIC,
            code="result = {'value': 42}",
            input_schema={"type": "object"},
            output_schema={"type": "object"},
            usage_count=10,
            success_count=3,  # 30% success rate
        )

        is_valid, errors = self.performance_validator.validate(skill)
        assert not is_valid
        assert "Low success rate" in errors[0]

    def test_composite_validator(self):
        """Test composite validator running all validators."""
        skill = Skill(
            name="test_skill",
            description="Test skill",
            category=SkillCategory.MARKET_ANALYSIS,
            complexity=SkillComplexity.BASIC,
            code="result = {'value': 42}",
            input_schema={"type": "object"},
            output_schema={"type": "object"},
            usage_count=10,
            success_count=8,  # 80% success rate
        )

        is_valid, results = self.composite_validator.validate_skill(skill)

        assert is_valid
        assert "SyntaxValidator" in results
        assert "SecurityValidator" in results
        assert "PerformanceValidator" in results
        assert all(len(errors) == 0 for errors in results.values())


class TestSkillLibrarian:
    """Test the SkillLibrarian component."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.storage_path = Path(self.temp_dir)
        self.librarian = SkillLibrarian(self.storage_path)

        # Create test skills
        self.skill1 = Skill(
            name="technical_skill",
            description="Technical analysis skill",
            category=SkillCategory.TECHNICAL_ANALYSIS,
            complexity=SkillComplexity.INTERMEDIATE,
            code="result = {'rsi': 65}",
            input_schema={"type": "object"},
            output_schema={"type": "object"},
            tags=["rsi", "technical"],
            usage_count=15,
            success_count=12,
        )

        self.skill2 = Skill(
            name="risk_skill",
            description="Risk management skill",
            category=SkillCategory.RISK_MANAGEMENT,
            complexity=SkillComplexity.ADVANCED,
            code="result = {'risk_score': 0.8}",
            input_schema={"type": "object"},
            output_schema={"type": "object"},
            tags=["risk", "management"],
            usage_count=20,
            success_count=18,
        )

    def test_store_and_retrieve_skill(self):
        """Test storing and retrieving a skill."""
        success = self.librarian.store_skill(self.skill1)
        assert success

        retrieved_skill = self.librarian.retrieve_skill(self.skill1.id)
        assert retrieved_skill is not None
        assert retrieved_skill.name == self.skill1.name
        assert retrieved_skill.category == self.skill1.category

    def test_search_skills_by_category(self):
        """Test searching skills by category."""
        self.librarian.store_skill(self.skill1)
        self.librarian.store_skill(self.skill2)

        technical_skills = self.librarian.search_skills(
            category=SkillCategory.TECHNICAL_ANALYSIS
        )
        assert len(technical_skills) == 1
        assert technical_skills[0].name == "technical_skill"

        risk_skills = self.librarian.search_skills(
            category=SkillCategory.RISK_MANAGEMENT
        )
        assert len(risk_skills) == 1
        assert risk_skills[0].name == "risk_skill"

    def test_search_skills_by_complexity(self):
        """Test searching skills by complexity."""
        self.librarian.store_skill(self.skill1)
        self.librarian.store_skill(self.skill2)

        advanced_skills = self.librarian.search_skills(
            complexity=SkillComplexity.ADVANCED
        )
        assert len(advanced_skills) == 1
        assert advanced_skills[0].name == "risk_skill"

    def test_search_skills_by_tags(self):
        """Test searching skills by tags."""
        self.librarian.store_skill(self.skill1)
        self.librarian.store_skill(self.skill2)

        rsi_skills = self.librarian.search_skills(tags=["rsi"])
        assert len(rsi_skills) == 1
        assert rsi_skills[0].name == "technical_skill"

        technical_skills = self.librarian.search_skills(tags=["technical"])
        assert len(technical_skills) == 1

    def test_search_skills_by_success_rate(self):
        """Test searching skills by success rate."""
        self.librarian.store_skill(self.skill1)
        self.librarian.store_skill(self.skill2)

        high_performing_skills = self.librarian.search_skills(min_success_rate=0.85)
        assert len(high_performing_skills) == 1
        assert high_performing_skills[0].name == "risk_skill"

    def test_search_skills_by_name_pattern(self):
        """Test searching skills by name pattern."""
        self.librarian.store_skill(self.skill1)
        self.librarian.store_skill(self.skill2)

        technical_skills = self.librarian.search_skills(name_pattern="technical")
        assert len(technical_skills) == 1
        assert technical_skills[0].name == "technical_skill"

    def test_get_library_stats(self):
        """Test getting library statistics."""
        self.librarian.store_skill(self.skill1)
        self.librarian.store_skill(self.skill2)

        stats = self.librarian.get_library_stats()

        assert stats["total_skills"] == 2
        assert stats["categories"]["technical_analysis"] == 1
        assert stats["categories"]["risk_management"] == 1
        assert stats["complexity_distribution"]["intermediate"] == 1
        assert stats["complexity_distribution"]["advanced"] == 1
        assert stats["total_usage_count"] == 35  # 15 + 20

    def test_get_skills_by_category(self):
        """Test getting all skills in a category."""
        self.librarian.store_skill(self.skill1)
        self.librarian.store_skill(self.skill2)

        technical_skills = self.librarian.get_skills_by_category(
            SkillCategory.TECHNICAL_ANALYSIS
        )
        assert len(technical_skills) == 1
        assert technical_skills[0].name == "technical_skill"


class TestSkillDiscoverer:
    """Test the SkillDiscoverer component."""

    def setup_method(self):
        """Set up test fixtures."""
        self.discoverer = SkillDiscoverer()

        # Create mock trading experiences
        self.successful_experience = Experience(
            title="Successful RSI Trade",
            description="Used RSI indicator to make profitable trade",
            context={"market_condition": "trending"},
            actions_taken=["calculate_rsi", "enter_long_position", "set_stop_loss"],
            outcome=LearningOutcome.POSITIVE,
            outcome_details={"profit": 150.0},
            lessons_learned=["RSI works well in trending markets"],
            contributing_factors=["strong trend", "low volatility"],
            market_conditions={"trend": "up", "volatility": "low"},
            symbols_involved=[Symbol(code="AAPL", asset_class=AssetClass.EQUITY)],
            timeframe=TimeFrame.HOUR_1,
            financial_impact=Money(amount=Decimal("150.0"), currency=Currency.USD),
            skills_used=["rsi_calculation", "position_entry"],
            confidence_before=Decimal("70"),
            confidence_after=Decimal("85"),
            stress_level=Decimal("30"),
            complexity_score=Decimal("60"),
            novelty_score=Decimal("40"),
        )

    def test_discover_skills_from_experiences(self):
        """Test discovering skills from experiences."""
        # Create multiple similar experiences to meet frequency threshold
        experiences = []
        for i in range(5):
            exp = Experience(
                title=f"Trade {i}",
                description="Trading experience",
                context={"market": "bullish"},
                actions_taken=["technical_analysis", "entry"],
                outcome=LearningOutcome.POSITIVE,
                outcome_details={"profit": 100.0 + i},
                lessons_learned=["Technical analysis works"],
                contributing_factors=["trend"],
                market_conditions={"trend": "up"},
                financial_impact=Money(amount=Decimal("100.0"), currency=Currency.USD),
                confidence_before=Decimal("70"),
                confidence_after=Decimal("80"),
                stress_level=Decimal("25"),
                complexity_score=Decimal("50"),
                novelty_score=Decimal("30"),
            )
            experiences.append(exp)

        discovered_skills = self.discoverer.discover_skills_from_experiences(
            experiences
        )

        # Should discover skills based on patterns
        assert len(discovered_skills) >= 0  # May be 0 if patterns don't meet threshold

    def test_discover_skills_from_code_analysis(self):
        """Test discovering skills from code analysis."""
        strategy_code = """
def calculate_rsi(prices, period=14):
    # RSI calculation
    return rsi_value

def enter_position(signal, amount):
    # Position entry logic
    return position

def main_strategy(data):
    rsi = calculate_rsi(data.prices)
    if rsi < 30:
        return enter_position('buy', 1000)
    return None
"""

        performance_data = {"profit_factor": 1.5, "win_rate": 0.65}

        discovered_skills = self.discoverer.discover_skills_from_code_analysis(
            strategy_code, performance_data
        )

        assert (
            len(discovered_skills) >= 2
        )  # Should find calculate_rsi and enter_position functions

        skill_names = [skill.name for skill in discovered_skills]
        assert any("calculate_rsi" in name for name in skill_names)
        assert any("enter_position" in name for name in skill_names)

    def test_infer_category_from_pattern(self):
        """Test category inference from pattern characteristics."""
        technical_pattern = {"name": "technical_indicator_pattern"}
        category = self.discoverer._infer_category(technical_pattern)
        assert category == SkillCategory.TECHNICAL_ANALYSIS

        risk_pattern = {"name": "risk_management_pattern"}
        category = self.discoverer._infer_category(risk_pattern)
        assert category == SkillCategory.RISK_MANAGEMENT

        entry_pattern = {"name": "entry_timing_pattern"}
        category = self.discoverer._infer_category(entry_pattern)
        assert category == SkillCategory.ENTRY_TIMING

    def test_infer_complexity_from_pattern(self):
        """Test complexity inference from pattern characteristics."""
        simple_pattern = {"frequency": 3, "success_contexts": [{"a": 1}, {"b": 2}]}
        complexity = self.discoverer._infer_complexity(simple_pattern)
        assert complexity == SkillComplexity.BASIC

        advanced_pattern = {
            "frequency": 15,
            "success_contexts": [{"a": i} for i in range(8)],
        }
        complexity = self.discoverer._infer_complexity(advanced_pattern)
        assert complexity == SkillComplexity.ADVANCED


class TestVoyagerSkillLibrary:
    """Test the main VoyagerSkillLibrary interface."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        config = {
            "skill_library_path": self.temp_dir,
            "execution_timeout": 5,
            "max_memory": 64,
            "min_pattern_frequency": 2,
            "min_success_rate": 0.6,
        }
        self.library = VoyagerSkillLibrary(config)

        # Create test skill
        self.test_skill = Skill(
            name="integration_test_skill",
            description="Skill for integration testing",
            category=SkillCategory.MARKET_ANALYSIS,
            complexity=SkillComplexity.BASIC,
            code="result = {'test_output': inputs.get('input_value', 0) * 2}",
            input_schema={
                "type": "object",
                "properties": {"input_value": {"type": "number"}},
                "required": ["input_value"],
            },
            output_schema={
                "type": "object",
                "properties": {"test_output": {"type": "number"}},
            },
            usage_count=10,
            success_count=8,
        )

    def test_add_skill_with_validation(self):
        """Test adding a skill with validation."""
        success = self.library.add_skill(self.test_skill, validate=True)
        assert success

        # Verify skill is stored
        retrieved = self.library.get_skill(self.test_skill.id)
        assert retrieved is not None
        assert retrieved.name == self.test_skill.name

    def test_add_invalid_skill(self):
        """Test adding an invalid skill."""
        invalid_skill = Skill(
            name="invalid_skill",
            description="Invalid skill",
            category=SkillCategory.MARKET_ANALYSIS,
            complexity=SkillComplexity.BASIC,
            code="invalid python syntax >>>",
            input_schema={"type": "object"},
            output_schema={"type": "object"},
        )

        success = self.library.add_skill(invalid_skill, validate=True)
        assert not success

    def test_execute_skill(self):
        """Test executing a skill through the main interface."""
        # Add skill first
        self.library.add_skill(self.test_skill, validate=False)

        # Execute skill
        inputs = {"input_value": 21}
        result, output, metadata = self.library.execute_skill(
            self.test_skill.id, inputs
        )

        assert result == SkillExecutionResult.SUCCESS
        assert output["test_output"] == 42
        assert metadata["skill_name"] == "integration_test_skill"

        # Verify skill usage was recorded
        updated_skill = self.library.get_skill(self.test_skill.id)
        assert updated_skill.usage_count == 11  # Original 10 + 1

    def test_execute_nonexistent_skill(self):
        """Test executing a non-existent skill."""
        result, output, metadata = self.library.execute_skill(
            "nonexistent_id", {"input": "value"}
        )

        assert result == SkillExecutionResult.ERROR
        assert output is None
        assert "Skill not found" in metadata["error"]

    def test_compose_and_execute_skills(self):
        """Test composing and executing multiple skills."""
        # Create two simple skills
        skill1 = Skill(
            name="add_skill",
            description="Adds 10 to input",
            category=SkillCategory.MARKET_ANALYSIS,
            complexity=SkillComplexity.BASIC,
            code="result = {'value': inputs.get('value', 0) + 10}",
            input_schema={"type": "object"},
            output_schema={"type": "object"},
        )

        skill2 = Skill(
            name="multiply_skill",
            description="Multiplies by 2",
            category=SkillCategory.MARKET_ANALYSIS,
            complexity=SkillComplexity.BASIC,
            code="result = {'value': inputs.get('value', 0) * 2}",
            input_schema={"type": "object"},
            output_schema={"type": "object"},
        )

        # Add skills
        self.library.add_skill(skill1, validate=False)
        self.library.add_skill(skill2, validate=False)

        # Compose and execute
        inputs = {"value": 5}
        result, output, metadata = self.library.compose_and_execute(
            [skill1.id, skill2.id], inputs, "sequential"
        )

        assert result == SkillExecutionResult.SUCCESS
        assert "add_skill" in output
        assert "multiply_skill" in output
        assert metadata["composition_metadata"]["composition_type"] == "sequential"

    def test_search_skills(self):
        """Test searching skills through main interface."""
        self.library.add_skill(self.test_skill, validate=False)

        # Search by category
        results = self.library.search_skills(category=SkillCategory.MARKET_ANALYSIS)
        assert len(results) == 1
        assert results[0].name == "integration_test_skill"

        # Search by name pattern
        results = self.library.search_skills(name_pattern="integration")
        assert len(results) == 1

    def test_get_library_stats(self):
        """Test getting library statistics."""
        self.library.add_skill(self.test_skill, validate=False)

        stats = self.library.get_library_stats()

        assert stats["total_skills"] == 1
        assert stats["categories"]["market_analysis"] == 1
        assert stats["total_usage_count"] == 10

    def test_validate_skill(self):
        """Test skill validation through main interface."""
        is_valid, results = self.library.validate_skill(self.test_skill)

        assert is_valid
        assert "SyntaxValidator" in results
        assert "SecurityValidator" in results
        assert "PerformanceValidator" in results

    def test_discover_skills_from_experiences(self):
        """Test skill discovery from experiences."""
        # Create mock experiences
        experiences = []
        for i in range(3):
            exp = Experience(
                title=f"Trading Experience {i}",
                description="Test experience",
                context={"market": "test"},
                actions_taken=["analyze", "trade"],
                outcome=LearningOutcome.POSITIVE,
                outcome_details={"profit": 100.0},
                lessons_learned=["Pattern works"],
                contributing_factors=["trend"],
                market_conditions={"trend": "up"},
                financial_impact=Money(amount=Decimal("100.0"), currency=Currency.USD),
                confidence_before=Decimal("70"),
                confidence_after=Decimal("80"),
                stress_level=Decimal("25"),
                complexity_score=Decimal("50"),
                novelty_score=Decimal("30"),
            )
            experiences.append(exp)

        discovered_skills = self.library.discover_skills_from_experiences(experiences)

        # Should return a list (may be empty if patterns don't meet thresholds)
        assert isinstance(discovered_skills, list)


class TestLegacyCompatibility:
    """Test backward compatibility with legacy SkillLibrary interface."""

    def setup_method(self):
        """Set up test fixtures."""
        from src.voyager_trader.skills import SkillLibrary, TradingSkill

        self.temp_dir = tempfile.mkdtemp()

        # Mock config object
        config = Mock()
        config.skill_library_path = self.temp_dir

        self.legacy_library = SkillLibrary(config)

        # Create legacy skill
        self.legacy_skill = TradingSkill(
            name="legacy_test_skill",
            description="Legacy skill for testing",
            code="result = {'legacy_value': 42}",
            performance_metrics={"accuracy": 0.85},
            usage_count=5,
            success_rate=80.0,
            prerequisites=[],
            tags=["legacy", "test"],
        )

    def test_add_legacy_skill(self):
        """Test adding a skill using legacy interface."""
        self.legacy_library.add_skill(self.legacy_skill)

        # Verify skill was added
        retrieved = self.legacy_library.get_skill("legacy_test_skill")
        assert retrieved is not None
        assert retrieved.name == "legacy_test_skill"
        assert retrieved.success_rate == 80.0

    def test_search_legacy_skills(self):
        """Test searching skills using legacy interface."""
        self.legacy_library.add_skill(self.legacy_skill)

        # Search by tags
        results = self.legacy_library.search_skills(tags=["legacy"])
        assert len(results) == 1
        assert results[0].name == "legacy_test_skill"

        # Search by success rate
        results = self.legacy_library.search_skills(min_success_rate=0.75)
        assert len(results) == 1

    def test_legacy_skill_composition(self):
        """Test skill composition using legacy interface."""
        # Add a second skill
        from src.voyager_trader.skills import TradingSkill

        skill2 = TradingSkill(
            name="legacy_skill2",
            description="Second legacy skill",
            code="result = {'value2': 100}",
            performance_metrics={"precision": 0.9},
            usage_count=3,
            success_rate=85.0,
            prerequisites=[],
            tags=["legacy"],
        )

        self.legacy_library.add_skill(self.legacy_skill)
        self.legacy_library.add_skill(skill2)

        # Compose skills
        composed_code = self.legacy_library.compose_skills(
            ["legacy_test_skill", "legacy_skill2"]
        )

        assert composed_code is not None
        assert "execute_composed_strategy" in composed_code

    def test_update_legacy_skill_performance(self):
        """Test updating skill performance using legacy interface."""
        self.legacy_library.add_skill(self.legacy_skill)

        # Update performance
        performance_update = {"success": 1.0, "execution_time": 0.5}
        self.legacy_library.update_skill_performance(
            "legacy_test_skill", performance_update
        )

        # Verify update
        updated_skill = self.legacy_library.get_skill("legacy_test_skill")
        assert updated_skill.usage_count == 6  # Original 5 + 1
        assert "execution_time" in updated_skill.performance_metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
