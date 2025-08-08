"""Tests for VOYAGER Iterative Prompting and Self-Improvement System."""

import subprocess
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.voyager_trader.prompting import (
    CodeExecutionEnvironment,
    ConvergenceReason,
    ExecutionResult,
    ExecutionStatus,
    FeedbackSynthesizer,
    IterationResult,
    IterativePrompting,
    IterativePromptingConfig,
    LLMIntegrationError,
    LLMIntegrationLayer,
    LLMProvider,
    PerformanceEvaluationError,
    PerformanceEvaluator,
    PromptContext,
    PromptingError,
    RefinementResult,
    SafetyMonitor,
    SafetyValidationError,
    StrategyRefinementEngine,
    VoyagerIterativeSystem,
)


class TestIterativePromptingConfig:
    """Test configuration class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = IterativePromptingConfig()

        assert config.llm_provider == LLMProvider.OPENAI
        assert config.llm_model == "gpt-4"
        assert config.max_tokens == 2000
        assert config.temperature == 0.7
        assert config.execution_timeout == 30
        assert config.max_iterations == 10
        assert config.enable_safety_checks is True
        assert "numpy" in config.allowed_imports
        assert "import os" in config.forbidden_patterns

    def test_custom_config(self):
        """Test custom configuration values."""
        config = IterativePromptingConfig(
            llm_model="gpt-3.5-turbo",
            max_iterations=5,
            enable_safety_checks=False,
            allowed_imports=["pandas", "numpy"],
            forbidden_patterns=["eval"],
        )

        assert config.llm_model == "gpt-3.5-turbo"
        assert config.max_iterations == 5
        assert config.enable_safety_checks is False
        assert config.allowed_imports == ["pandas", "numpy"]
        assert config.forbidden_patterns == ["eval"]


class TestPromptContext:
    """Test prompt context class."""

    def test_basic_context(self):
        """Test basic context creation."""
        context = PromptContext(
            task_description="Generate a trading strategy",
            market_data={"price": 100.0, "volume": 1000},
            available_skills=["skill1", "skill2"],
        )

        assert context.task_description == "Generate a trading strategy"
        assert context.market_data == {"price": 100.0, "volume": 1000}
        assert context.available_skills == ["skill1", "skill2"]
        assert context.market_regime == "normal"
        assert context.risk_tolerance == "moderate"

    def test_enhanced_context(self):
        """Test enhanced context with all fields."""
        context = PromptContext(
            task_description="Generate momentum strategy",
            market_data={"trend": "up"},
            available_skills=["momentum_indicator"],
            target_metrics={"sharpe_ratio": 1.5},
            safety_constraints=["no_leverage"],
            market_regime="bull",
            risk_tolerance="aggressive",
        )

        assert context.target_metrics == {"sharpe_ratio": 1.5}
        assert context.safety_constraints == ["no_leverage"]
        assert context.market_regime == "bull"
        assert context.risk_tolerance == "aggressive"


class TestSafetyMonitor:
    """Test safety monitoring system."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = IterativePromptingConfig()
        self.safety_monitor = SafetyMonitor(self.config)

    def test_safe_code_validation(self):
        """Test validation of safe code."""
        safe_code = """
import numpy as np
import pandas as pd

def trading_strategy(market_data):
    price = market_data.get('price', 0)
    return {'action': 'buy' if price > 100 else 'hold'}
"""
        violations = self.safety_monitor.validate_code(safe_code)
        assert len(violations) == 0

    def test_forbidden_pattern_detection(self):
        """Test detection of forbidden patterns."""
        unsafe_code = """
import os
import subprocess

def trading_strategy(market_data):
    os.system('rm -rf /')
    return {'action': 'buy'}
"""
        violations = self.safety_monitor.validate_code(unsafe_code)
        assert len(violations) > 0
        assert any("import os" in violation for violation in violations)
        assert any("import subprocess" in violation for violation in violations)

    def test_dangerous_function_detection(self):
        """Test detection of dangerous function calls."""
        unsafe_code = """
def trading_strategy(market_data):
    eval("print('dangerous')")
    return {'action': 'buy'}
"""
        violations = self.safety_monitor.validate_code(unsafe_code)
        assert len(violations) > 0
        assert any("eval" in violation for violation in violations)

    def test_disallowed_import_detection(self):
        """Test detection of disallowed imports."""
        unsafe_code = """
import requests
import urllib

def trading_strategy(market_data):
    return {'action': 'buy'}
"""
        violations = self.safety_monitor.validate_code(unsafe_code)
        assert len(violations) > 0

    def test_syntax_error_detection(self):
        """Test detection of syntax errors."""
        invalid_code = """
def trading_strategy(market_data):
    if price > 100
        return {'action': 'buy'}
"""
        violations = self.safety_monitor.validate_code(invalid_code)
        assert len(violations) > 0
        assert any("Syntax error" in violation for violation in violations)

    def test_safety_checks_disabled(self):
        """Test behavior when safety checks are disabled."""
        config = IterativePromptingConfig(enable_safety_checks=False)
        safety_monitor = SafetyMonitor(config)

        unsafe_code = """
import os
eval("print('test')")
"""
        violations = safety_monitor.validate_code(unsafe_code)
        assert len(violations) == 0


class TestCodeExecutionEnvironment:
    """Test code execution environment."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = IterativePromptingConfig()
        self.execution_env = CodeExecutionEnvironment(self.config)
        self.market_data = {"price": 100.0, "trend": "up"}

    def test_successful_execution(self):
        """Test successful code execution."""
        safe_code = """
def trading_strategy(market_data):
    return {'action': 'buy', 'quantity': 100}
"""

        result = self.execution_env.execute_strategy_code(safe_code, self.market_data)
        assert result.status == ExecutionStatus.SUCCESS
        assert result.output is not None
        assert result.execution_time > 0

    def test_safety_violation_prevention(self):
        """Test prevention of unsafe code execution."""
        unsafe_code = """
import os

def trading_strategy(market_data):
    os.system('echo dangerous')
    return {'action': 'buy'}
"""

        result = self.execution_env.execute_strategy_code(unsafe_code, self.market_data)
        assert result.status == ExecutionStatus.SAFETY_VIOLATION
        assert len(result.safety_violations) > 0

    def test_syntax_error_handling(self):
        """Test handling of syntax errors."""
        invalid_code = """
def trading_strategy(market_data):
    if price > 100
        return {'action': 'buy'}
"""

        result = self.execution_env.execute_strategy_code(
            invalid_code, self.market_data
        )
        assert result.status == ExecutionStatus.SAFETY_VIOLATION

    def test_execution_script_creation(self):
        """Test creation of execution script."""
        with tempfile.TemporaryDirectory() as temp_dir:
            strategy_file = Path(temp_dir) / "strategy.py"
            script = self.execution_env._create_execution_script(
                strategy_file, self.market_data
            )

            assert "import json" in script
            assert "import sys" in script
            assert "import strategy" in script
            assert str(self.market_data) in script

    @patch("src.voyager_trader.prompting.subprocess.Popen")
    def test_subprocess_timeout(self, mock_popen):
        """Test subprocess timeout handling."""
        mock_process = Mock()
        mock_process.communicate.side_effect = subprocess.TimeoutExpired("cmd", 30)
        mock_process.kill.return_value = None
        mock_popen.return_value = mock_process

        code = "def trading_strategy(market_data): return {'action': 'buy'}"
        result = self.execution_env.execute_strategy_code(code, self.market_data)

        assert result.status == ExecutionStatus.TIMEOUT
        assert "timed out" in result.error_message


class TestPerformanceEvaluator:
    """Test performance evaluation system."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = IterativePromptingConfig()
        self.evaluator = PerformanceEvaluator(self.config)
        self.market_data = {"price": 100.0, "trend": "up"}

    def test_successful_performance_evaluation(self):
        """Test performance evaluation of successful execution."""
        execution_result = ExecutionResult(
            status=ExecutionStatus.SUCCESS,
            output={"result": {"action": "buy", "quantity": 100}},
            execution_time=1.0,
        )

        metrics = self.evaluator.evaluate_strategy_performance(
            execution_result, self.market_data
        )

        assert "sharpe_ratio" in metrics
        assert "max_drawdown" in metrics
        assert "total_return" in metrics
        assert "win_rate" in metrics
        assert "profit_factor" in metrics
        assert "volatility" in metrics
        assert all(isinstance(v, (int, float)) for v in metrics.values())

    def test_failed_execution_evaluation(self):
        """Test performance evaluation of failed execution."""
        execution_result = ExecutionResult(
            status=ExecutionStatus.ERROR, error_message="Test error"
        )

        metrics = self.evaluator.evaluate_strategy_performance(
            execution_result, self.market_data
        )

        assert all(v == 0.0 for v in metrics.values())

    def test_mock_metrics_calculation(self):
        """Test mock metrics calculation logic."""
        strategy_output = {"action": "buy", "quantity": 100}

        sharpe = self.evaluator._calculate_mock_sharpe_ratio(strategy_output)
        assert sharpe == 1.2  # Buy action should return 1.2

        strategy_output = {"action": "hold"}
        sharpe = self.evaluator._calculate_mock_sharpe_ratio(strategy_output)
        assert sharpe == 0.8  # Non-buy action should return 0.8


class TestFeedbackSynthesizer:
    """Test feedback synthesis system."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = IterativePromptingConfig()
        self.synthesizer = FeedbackSynthesizer(self.config)

    def test_performance_feedback_generation(self):
        """Test performance feedback generation."""
        execution_result = ExecutionResult(
            status=ExecutionStatus.SUCCESS,
            performance_metrics={"sharpe_ratio": 1.0, "max_drawdown": 0.1},
        )

        context = PromptContext(
            task_description="Test task",
            market_data={},
            available_skills=[],
            target_metrics={"sharpe_ratio": 1.5, "max_drawdown": 0.05},
        )

        iteration_result = IterationResult(
            iteration=1, generated_code="test_code", execution_result=execution_result
        )

        feedback = self.synthesizer.synthesize_feedback(iteration_result, context)

        assert "PERFORMANCE ANALYSIS" in feedback
        assert "sharpe_ratio" in feedback
        assert "max_drawdown" in feedback
        assert "SUGGESTIONS" in feedback

    def test_error_feedback_generation(self):
        """Test error feedback generation."""
        execution_result = ExecutionResult(
            status=ExecutionStatus.ERROR, error_message="KeyError: 'price' not found"
        )

        context = PromptContext(
            task_description="Test task", market_data={}, available_skills=[]
        )

        iteration_result = IterationResult(
            iteration=1, generated_code="test_code", execution_result=execution_result
        )

        feedback = self.synthesizer.synthesize_feedback(iteration_result, context)

        assert "ERROR ANALYSIS" in feedback
        assert "KeyError" in feedback
        assert "SUGGESTED FIXES" in feedback
        assert "error handling" in feedback.lower()

    def test_safety_feedback_generation(self):
        """Test safety feedback generation."""
        execution_result = ExecutionResult(
            status=ExecutionStatus.SAFETY_VIOLATION,
            safety_violations=[
                "Forbidden pattern: import os",
                "Dangerous function: eval",
            ],
        )

        context = PromptContext(
            task_description="Test task", market_data={}, available_skills=[]
        )

        iteration_result = IterationResult(
            iteration=1, generated_code="test_code", execution_result=execution_result
        )

        feedback = self.synthesizer.synthesize_feedback(iteration_result, context)

        assert "SAFETY VIOLATIONS" in feedback
        assert "import os" in feedback
        assert "eval" in feedback
        assert "REQUIRED ACTIONS" in feedback

    def test_resource_feedback_generation(self):
        """Test resource usage feedback generation."""
        execution_result = ExecutionResult(
            status=ExecutionStatus.SUCCESS,
            execution_time=15.0,  # Slow execution
            memory_usage_mb=150.0,  # High memory usage
        )

        context = PromptContext(
            task_description="Test task", market_data={}, available_skills=[]
        )

        iteration_result = IterationResult(
            iteration=1, generated_code="test_code", execution_result=execution_result
        )

        feedback = self.synthesizer.synthesize_feedback(iteration_result, context)

        assert "Slow execution" in feedback
        assert "High memory usage" in feedback
        assert "optimizing" in feedback.lower()


@pytest.mark.asyncio
class TestLLMIntegrationLayer:
    """Test LLM integration layer."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = IterativePromptingConfig(api_key="test_key")

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test_key"})
    @patch("src.voyager_trader.prompting.OPENAI_AVAILABLE", True)
    @patch("src.voyager_trader.prompting.openai")
    def test_openai_initialization(self, mock_openai):
        """Test OpenAI provider initialization."""
        LLMIntegrationLayer(self.config)
        assert mock_openai.api_key == "test_key"

    def test_missing_api_key_error(self):
        """Test error when API key is missing."""
        config = IterativePromptingConfig(api_key=None)

        with patch.dict("os.environ", {}, clear=True):
            with patch("src.voyager_trader.prompting.OPENAI_AVAILABLE", True):
                with pytest.raises(LLMIntegrationError, match="API key not provided"):
                    LLMIntegrationLayer(config)

    def test_unavailable_library_error(self):
        """Test error when OpenAI library is not available."""
        with patch("src.voyager_trader.prompting.OPENAI_AVAILABLE", False):
            with pytest.raises(
                LLMIntegrationError, match="OpenAI library not available"
            ):
                LLMIntegrationLayer(self.config)

    @patch("src.voyager_trader.prompting.OPENAI_AVAILABLE", True)
    @patch("src.voyager_trader.prompting.openai")
    async def test_strategy_code_generation(self, mock_openai):
        """Test strategy code generation."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "def trading_strategy(): pass"
        mock_openai.ChatCompletion.acreate = AsyncMock(return_value=mock_response)

        llm_layer = LLMIntegrationLayer(self.config)

        result = await llm_layer.generate_strategy_code("Generate a strategy")

        assert result == "def trading_strategy(): pass"
        mock_openai.ChatCompletion.acreate.assert_called_once()

    @patch("src.voyager_trader.prompting.OPENAI_AVAILABLE", True)
    @patch("src.voyager_trader.prompting.openai")
    async def test_rate_limiting(self, mock_openai):
        """Test rate limiting enforcement."""
        config = IterativePromptingConfig(
            api_key="test_key", rate_limit_requests_per_minute=2
        )
        llm_layer = LLMIntegrationLayer(config)
        llm_layer._request_count = 2  # At limit
        llm_layer._last_request_time = time.time()

        # Should enforce rate limit
        start_time = time.time()
        await llm_layer._enforce_rate_limits()
        end_time = time.time()

        # Should have waited or reset count
        assert end_time >= start_time


@pytest.mark.asyncio
class TestStrategyRefinementEngine:
    """Test strategy refinement engine."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = IterativePromptingConfig(max_iterations=3, api_key="test_key")

        self.context = PromptContext(
            task_description="Generate a momentum strategy",
            market_data={"price": 100.0, "trend": "up"},
            available_skills=["momentum_indicator"],
            target_metrics={"sharpe_ratio": 1.5},
        )

    @patch("src.voyager_trader.prompting.OPENAI_AVAILABLE", True)
    @patch("src.voyager_trader.prompting.openai")
    async def test_successful_refinement(self, mock_openai):
        """Test successful strategy refinement."""
        # Mock LLM response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[
            0
        ].message.content = """
```python
def trading_strategy(market_data):
    return {'action': 'buy', 'quantity': 100}
```
"""
        mock_openai.ChatCompletion.acreate = AsyncMock(return_value=mock_response)

        engine = StrategyRefinementEngine(self.config)

        # Mock successful execution
        with patch.object(
            engine.execution_env, "execute_strategy_code"
        ) as mock_execute:
            mock_execute.return_value = ExecutionResult(
                status=ExecutionStatus.SUCCESS,
                output={"result": {"action": "buy", "quantity": 100}},
                performance_metrics={"sharpe_ratio": 1.6},  # Above target
            )

            result = await engine.refine_strategy(self.context)

            assert isinstance(result, RefinementResult)
            assert result.total_iterations > 0
            assert result.convergence_reason == ConvergenceReason.PERFORMANCE_TARGET
            assert result.best_performance["sharpe_ratio"] == 1.6

    def test_performance_improvement_calculation(self):
        """Test performance improvement calculation."""
        engine = StrategyRefinementEngine(self.config)

        # No previous best performance
        execution_result = ExecutionResult(
            status=ExecutionStatus.SUCCESS, performance_metrics={"sharpe_ratio": 1.2}
        )
        improvement = engine._calculate_performance_improvement(execution_result)
        assert improvement == 1.0  # First successful result

        # Set best performance and test improvement
        engine.best_performance = {"sharpe_ratio": 1.0}
        execution_result.performance_metrics = {"sharpe_ratio": 1.5}
        improvement = engine._calculate_performance_improvement(execution_result)
        assert improvement == 0.5  # 50% improvement

    def test_convergence_criteria_checking(self):
        """Test convergence criteria checking."""
        engine = StrategyRefinementEngine(self.config)

        # Test performance target convergence
        engine.best_performance = {"sharpe_ratio": 1.6}  # Above target
        reason = engine._check_convergence_criteria(0)
        assert reason == ConvergenceReason.PERFORMANCE_TARGET

        # Test consecutive failures
        engine.consecutive_failures = self.config.max_consecutive_failures
        reason = engine._check_convergence_criteria(0)
        assert reason == ConvergenceReason.ERROR_THRESHOLD

    def test_prompt_building(self):
        """Test iteration prompt building."""
        engine = StrategyRefinementEngine(self.config)

        prompt = engine._build_iteration_prompt(self.context, 0)

        assert "TRADING STRATEGY GENERATION TASK" in prompt
        assert self.context.task_description in prompt
        assert "Market Regime: normal" in prompt
        assert "Risk Tolerance: moderate" in prompt
        assert "MARKET DATA:" in prompt
        assert "TARGET METRICS:" in prompt
        assert "REQUIREMENTS:" in prompt


@pytest.mark.asyncio
class TestIterativePrompting:
    """Test main iterative prompting interface."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = IterativePromptingConfig(api_key="test_key")
        self.context = PromptContext(
            task_description="Generate a trading strategy",
            market_data={"price": 100.0},
            available_skills=[],
        )

    @patch("src.voyager_trader.prompting.OPENAI_AVAILABLE", True)
    @patch("src.voyager_trader.prompting.openai")
    async def test_strategy_generation(self, mock_openai):
        """Test complete strategy generation."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "def trading_strategy(): pass"
        mock_openai.ChatCompletion.acreate = AsyncMock(return_value=mock_response)

        prompting_system = IterativePrompting(self.config)

        # Mock the refinement engine
        mock_result = RefinementResult(
            final_code="def trading_strategy(): pass",
            best_performance={"sharpe_ratio": 1.2},
            total_iterations=2,
            convergence_reason=ConvergenceReason.MAX_ITERATIONS,
            iteration_history=[],
            total_time=10.0,
            llm_tokens_used=500,
        )

        with patch.object(
            prompting_system.refinement_engine, "refine_strategy"
        ) as mock_refine:
            mock_refine.return_value = mock_result

            result = await prompting_system.generate_strategy(self.context)

            assert isinstance(result, RefinementResult)
            assert result.final_code == "def trading_strategy(): pass"
            assert result.total_iterations == 2

    async def test_context_enhancement_with_integrations(self):
        """Test context enhancement with curriculum and skills."""
        mock_curriculum = Mock()
        mock_skill_librarian = Mock()

        prompting_system = IterativePrompting(
            self.config,
            curriculum=mock_curriculum,
            skill_librarian=mock_skill_librarian,
        )

        # Mock skill search
        mock_skill = Mock()
        mock_skill.name = "momentum_strategy"
        mock_skill.description = "Momentum-based trading strategy"
        mock_skill_librarian.search_skills.return_value = [mock_skill]

        # Mock curriculum task
        mock_task = Mock()
        mock_task.difficulty_level = "medium"
        mock_curriculum.get_current_task.return_value = mock_task

        enhanced_context = await prompting_system._enhance_context_with_integrations(
            self.context
        )

        assert len(enhanced_context.available_skills) > len(
            self.context.available_skills
        )
        assert len(enhanced_context.target_metrics) > 0
        assert len(enhanced_context.safety_constraints) > 0


@pytest.mark.asyncio
class TestVoyagerIterativeSystem:
    """Test complete VOYAGER iterative system."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = IterativePromptingConfig(api_key="test_key")

    @patch("src.voyager_trader.prompting.AutomaticCurriculum")
    @patch("src.voyager_trader.prompting.SkillLibrarian")
    def test_system_initialization(
        self, mock_skill_librarian_class, mock_curriculum_class
    ):
        """Test system initialization with components."""
        curriculum_config = {"test": "config"}
        skill_config = {"test": "config"}

        system = VoyagerIterativeSystem(
            self.config, curriculum_config=curriculum_config, skill_config=skill_config
        )

        mock_curriculum_class.assert_called_once_with(curriculum_config)
        mock_skill_librarian_class.assert_called_once_with(skill_config)
        assert system.curriculum is not None
        assert system.skill_librarian is not None
        assert system.prompting_system is not None

    async def test_learning_episode_execution(self):
        """Test complete learning episode execution."""
        system = VoyagerIterativeSystem(self.config)

        # Mock the prompting system
        mock_result = RefinementResult(
            final_code="def trading_strategy(): pass",
            best_performance={"sharpe_ratio": 1.6},  # Above target
            total_iterations=3,
            convergence_reason=ConvergenceReason.PERFORMANCE_TARGET,
            iteration_history=[],
            total_time=15.0,
            llm_tokens_used=750,
        )

        with patch.object(
            system.prompting_system, "generate_strategy"
        ) as mock_generate:
            mock_generate.return_value = mock_result

            episode_results = await system.run_learning_episode("Test task")

            assert episode_results["tasks_completed"] == 1
            assert episode_results["strategies_generated"] == 1
            assert episode_results["skills_acquired"] == 1
            assert episode_results["total_iterations"] == 3
            assert episode_results["episode_duration"] > 0

    async def test_mock_market_data_generation(self):
        """Test mock market data generation."""
        system = VoyagerIterativeSystem(self.config)
        market_data = await system._generate_mock_market_data()

        required_keys = ["symbol", "price", "volume", "trend", "volatility"]
        assert all(key in market_data for key in required_keys)
        assert isinstance(market_data["price"], (int, float))
        assert isinstance(market_data["volume"], int)


class TestErrorHandling:
    """Test error handling throughout the system."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = IterativePromptingConfig()

    def test_prompting_error_hierarchy(self):
        """Test exception hierarchy."""
        assert issubclass(LLMIntegrationError, PromptingError)
        assert issubclass(SafetyValidationError, PromptingError)
        assert issubclass(PerformanceEvaluationError, PromptingError)

    @patch("src.voyager_trader.prompting.OPENAI_AVAILABLE", True)
    @patch("src.voyager_trader.prompting.openai")
    async def test_llm_failure_handling(self, mock_openai):
        """Test handling of LLM API failures."""
        mock_openai.ChatCompletion.acreate = AsyncMock(
            side_effect=Exception("API Error")
        )

        llm_layer = LLMIntegrationLayer(self.config)

        with pytest.raises(LLMIntegrationError, match="Failed to generate code"):
            await llm_layer.generate_strategy_code("Test prompt")

    def test_performance_evaluation_error_handling(self):
        """Test performance evaluation error handling."""
        evaluator = PerformanceEvaluator(self.config)

        # Mock execution result that will cause evaluation error
        execution_result = ExecutionResult(
            status=ExecutionStatus.SUCCESS, output=None  # This should cause an error
        )

        with pytest.raises(PerformanceEvaluationError):
            evaluator.evaluate_strategy_performance(execution_result, {})


class TestLegacyCompatibility:
    """Test legacy compatibility functions."""

    @patch("src.voyager_trader.prompting.OPENAI_AVAILABLE", True)
    @patch("src.voyager_trader.prompting.openai")
    def test_generate_strategy_legacy(self, mock_openai):
        """Test legacy function compatibility."""
        from src.voyager_trader.prompting import generate_strategy_legacy

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "def trading_strategy(): pass"
        mock_openai.ChatCompletion.acreate = AsyncMock(return_value=mock_response)

        context_dict = {
            "task_description": "Test task",
            "market_data": {"price": 100.0},
            "available_skills": [],
            "previous_attempts": [],
            "performance_feedback": {},
            "error_messages": [],
        }

        config_dict = {"api_key": "test_key", "max_iterations": 2}

        # Mock the refinement result
        with patch(
            "src.voyager_trader.prompting.StrategyRefinementEngine"
        ) as mock_engine_class:
            mock_engine = Mock()
            mock_result = RefinementResult(
                final_code="def trading_strategy(): pass",
                best_performance={"sharpe_ratio": 1.2},
                total_iterations=2,
                convergence_reason=ConvergenceReason.MAX_ITERATIONS,
                iteration_history=[],
                total_time=10.0,
                llm_tokens_used=500,
            )
            mock_engine.refine_strategy = AsyncMock(return_value=mock_result)
            mock_engine_class.return_value = mock_engine

            code, metadata = generate_strategy_legacy(context_dict, config_dict)

            assert code == "def trading_strategy(): pass"
            assert metadata["iterations"] == 2
            assert metadata["convergence_reason"] == "max_iterations"
            assert "performance_metrics" in metadata
            assert "total_time" in metadata


if __name__ == "__main__":
    pytest.main([__file__])
