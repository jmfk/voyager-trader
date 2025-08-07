"""
Comprehensive tests for skills.py to increase coverage.
"""

import json
import tempfile
import time
from decimal import Decimal
from pathlib import Path

import pytest

from voyager_trader.models.learning import Experience, Skill, SkillExecutionResult
from voyager_trader.models.types import SkillCategory, SkillComplexity
from voyager_trader.skills import (
    CacheConfig,
    DatabaseConfig,
    PerformanceValidator,
    SecurityValidator,
    SkillComposer,
    SkillDiscoverer,
    SkillExecutor,
    SkillLibrarian,
    SyntaxValidator,
    VoyagerSkillLibrary,
)


class TestSkillValidators:
    """Test skill validation components."""

    def test_syntax_validator(self):
        """Test syntax validation."""
        validator = SyntaxValidator()

        # Valid syntax
        valid_skill = Skill(
            name="valid",
            description="Valid skill",
            category=SkillCategory.TECHNICAL_ANALYSIS,
            complexity=SkillComplexity.BASIC,
            code="result = {'value': 42}",
            input_schema={"type": "object"},
            output_schema={"type": "object"},
        )

        is_valid, errors = validator.validate(valid_skill)
        assert is_valid is True
        assert len(errors) == 0

        # Invalid syntax
        invalid_skill = Skill(
            name="invalid",
            description="Invalid skill",
            category=SkillCategory.TECHNICAL_ANALYSIS,
            complexity=SkillComplexity.BASIC,
            code="result = {'value': 42",  # Missing closing brace
            input_schema={"type": "object"},
            output_schema={"type": "object"},
        )

        is_valid, errors = validator.validate(invalid_skill)
        assert is_valid is False
        assert len(errors) > 0

    def test_security_validator(self):
        """Test security validation."""
        validator = SecurityValidator()

        # Safe skill
        safe_skill = Skill(
            name="safe",
            description="Safe skill",
            category=SkillCategory.TECHNICAL_ANALYSIS,
            complexity=SkillComplexity.BASIC,
            code="result = {'value': inputs.get('data', 0) * 2}",
            input_schema={"type": "object"},
            output_schema={"type": "object"},
        )

        is_valid, errors = validator.validate(safe_skill)
        assert is_valid is True
        assert len(errors) == 0

        # Unsafe skill with dangerous import
        unsafe_skill = Skill(
            name="unsafe",
            description="Unsafe skill",
            category=SkillCategory.TECHNICAL_ANALYSIS,
            complexity=SkillComplexity.BASIC,
            code="import os\nresult = {'files': os.listdir('.')}",
            input_schema={"type": "object"},
            output_schema={"type": "object"},
        )

        is_valid, errors = validator.validate(unsafe_skill)
        assert is_valid is False
        assert len(errors) > 0
        assert "Dangerous import" in errors[0]

    def test_performance_validator(self):
        """Test performance validation."""
        validator = PerformanceValidator(min_success_rate=0.8, min_usage_count=10)

        # High-performing skill
        good_skill = Skill(
            name="good",
            description="Good skill",
            category=SkillCategory.TECHNICAL_ANALYSIS,
            complexity=SkillComplexity.BASIC,
            code="result = {'value': 42}",
            input_schema={"type": "object"},
            output_schema={"type": "object"},
            usage_count=20,
            success_count=18,  # 90% success rate
        )

        is_valid, errors = validator.validate(good_skill)
        assert is_valid is True
        assert len(errors) == 0

        # Low-performing skill
        bad_skill = Skill(
            name="bad",
            description="Bad skill",
            category=SkillCategory.TECHNICAL_ANALYSIS,
            complexity=SkillComplexity.BASIC,
            code="result = {'value': 42}",
            input_schema={"type": "object"},
            output_schema={"type": "object"},
            usage_count=5,  # Too few uses
            success_count=2,  # 40% success rate
        )

        is_valid, errors = validator.validate(bad_skill)
        assert is_valid is False
        assert len(errors) > 0

        # Test legacy mode
        legacy_validator = PerformanceValidator(legacy_compatible=True)
        is_valid, errors = legacy_validator.validate(bad_skill)
        # Should be more lenient in legacy mode
        assert len(errors) <= 2


class TestSkillComposer:
    """Test skill composition functionality."""

    def setup_method(self):
        """Set up test composer."""
        self.composer = SkillComposer()

        self.skill1 = Skill(
            name="skill1",
            description="First skill",
            category=SkillCategory.TECHNICAL_ANALYSIS,
            complexity=SkillComplexity.BASIC,
            code="step1_result = inputs.get('value', 0) * 2",
            input_schema={"type": "object"},
            output_schema={"type": "object"},
        )

        self.skill2 = Skill(
            name="skill2",
            description="Second skill",
            category=SkillCategory.TECHNICAL_ANALYSIS,
            complexity=SkillComplexity.BASIC,
            code="step2_result = step1_result + 10 if 'step1_result' in locals() else 10",
            input_schema={"type": "object"},
            output_schema={"type": "object"},
            required_skills=["skill1"],  # Depends on skill1
        )

    def test_dependency_resolution(self):
        """Test dependency resolution."""
        skills = [self.skill2, self.skill1]  # Wrong order
        resolved = self.composer._resolve_dependencies(skills)

        assert resolved is not None
        assert len(resolved) == 2
        assert resolved[0].name == "skill1"  # Should be first due to dependency
        assert resolved[1].name == "skill2"

    def test_circular_dependency_detection(self):
        """Test circular dependency detection."""
        # Create circular dependency
        self.skill1.required_skills = ["skill2"]

        skills = [self.skill1, self.skill2]
        resolved = self.composer._resolve_dependencies(skills)

        assert resolved is None  # Should detect circular dependency

    def test_sequential_composition(self):
        """Test sequential skill composition."""
        skills = [self.skill1, self.skill2]
        code, metadata = self.composer.compose_skills(skills, "sequential")

        assert "execute_composed_strategy" in code
        assert metadata["composition_type"] == "sequential"
        assert metadata["skills_count"] == 2
        assert "skill1" in metadata["skill_names"]
        assert "skill2" in metadata["skill_names"]

    def test_parallel_composition(self):
        """Test parallel skill composition."""
        skills = [self.skill1, self.skill2]
        code, metadata = self.composer.compose_skills(skills, "parallel")

        assert "execute_composed_strategy" in code
        assert metadata["composition_type"] == "parallel"
        assert metadata["skills_count"] == 2

    def test_conditional_composition(self):
        """Test conditional skill composition."""
        skills = [self.skill1, self.skill2]
        code, metadata = self.composer.compose_skills(skills, "conditional")

        assert "execute_composed_strategy" in code
        assert metadata["composition_type"] == "conditional"
        assert metadata["skills_count"] == 2


class TestSkillDiscoverer:
    """Test skill discovery functionality."""

    def setup_method(self):
        """Set up test discoverer."""
        self.discoverer = SkillDiscoverer(min_pattern_frequency=2, min_success_rate=0.6)

    def test_discover_from_code_analysis(self):
        """Test discovering skills from code analysis."""
        strategy_code = """
def calculate_rsi(prices, period=14):
    # RSI calculation logic
    return rsi_value

def moving_average(prices, window=20):
    # Moving average calculation
    return ma_value

def main_strategy():
    rsi = calculate_rsi(prices)
    ma = moving_average(prices)
    return {'signal': 'buy' if rsi < 30 and price > ma else 'hold'}
"""

        performance_data = {"success_rate": 0.85, "profit": 1000}

        discovered_skills = self.discoverer.discover_skills_from_code_analysis(
            strategy_code, performance_data
        )

        # Should discover the function definitions
        assert len(discovered_skills) >= 3
        function_names = [skill.name for skill in discovered_skills]
        assert any("calculate_rsi" in name for name in function_names)
        assert any("moving_average" in name for name in function_names)
        assert any("main_strategy" in name for name in function_names)

    def test_infer_category_from_function_name(self):
        """Test category inference from function names."""
        # Test technical analysis functions
        func_node = type(
            "MockNode",
            (),
            {
                "name": "calculate_rsi",
                "body": ["mock"],
                "args": type("Args", (), {"args": []})(),
            },
        )()

        category = self.discoverer._analyze_function_category(func_node)
        assert category == SkillCategory.TECHNICAL_ANALYSIS

        # Test risk management functions
        func_node.name = "calculate_position_size"
        category = self.discoverer._analyze_function_category(func_node)
        assert category == SkillCategory.RISK_MANAGEMENT

    def test_complexity_analysis(self):
        """Test complexity analysis of functions."""
        # Simple function
        simple_func = type(
            "MockNode",
            (),
            {
                "body": ["line1", "line2"],  # 2 lines
                "args": type("Args", (), {"args": ["arg1"]})(),  # 1 arg
            },
        )()

        complexity = self.discoverer._analyze_function_complexity(simple_func)
        assert complexity == SkillComplexity.BASIC

        # Complex function
        complex_func = type(
            "MockNode",
            (),
            {
                "body": ["line" + str(i) for i in range(25)],  # 25 lines
                "args": type(
                    "Args", (), {"args": ["arg" + str(i) for i in range(6)]}
                )(),  # 6 args
            },
        )()

        complexity = self.discoverer._analyze_function_complexity(complex_func)
        assert complexity == SkillComplexity.ADVANCED


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
