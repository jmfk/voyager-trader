"""Tests for the IterativePrompting functionality."""

import os
from unittest.mock import patch

import pytest

from voyager_trader.core import TradingConfig
from voyager_trader.prompting import IterativePrompting, PromptContext


class TestPromptContext:
    """Test the PromptContext dataclass."""

    def test_context_creation(self):
        """Test creating a prompt context."""
        context = PromptContext(
            task_description="Implement momentum strategy",
            market_data={"trend": "up"},
            available_skills=["skill1", "skill2"],
            previous_attempts=[],
            performance_feedback={"success_rate": 0.7},
        )

        assert context.task_description == "Implement momentum strategy"
        assert context.market_data == {"trend": "up"}
        assert context.available_skills == ["skill1", "skill2"]
        assert context.previous_attempts == []
        assert context.performance_feedback == {"success_rate": 0.7}
        assert context.error_messages == []


class TestIterativePrompting:
    """Test the IterativePrompting class."""

    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"), reason="Requires OpenAI API key"
    )
    def test_initialization(self):
        """Test iterative prompting initialization."""
        config = TradingConfig()
        prompting = IterativePrompting(config)

        assert prompting.config == config
        assert prompting.conversation_history == []
        assert prompting.max_iterations == config.max_iterations

    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"), reason="Requires OpenAI API key"
    )
    @patch.object(IterativePrompting, "_query_llm")
    @patch.object(IterativePrompting, "_extract_code")
    @patch.object(IterativePrompting, "_is_code_improved")
    @patch.object(IterativePrompting, "_should_stop")
    def test_generate_strategy(
        self, mock_should_stop, mock_is_improved, mock_extract, mock_query
    ):
        """Test strategy generation process."""
        config = TradingConfig()
        prompting = IterativePrompting(config)

        context = PromptContext(
            task_description="Test task",
            market_data={},
            available_skills=[],
            previous_attempts=[],
            performance_feedback={},
        )

        # Mock the methods
        mock_query.return_value = "mock response"
        mock_extract.return_value = "def strategy(): return 'buy'"
        mock_is_improved.return_value = True
        mock_should_stop.return_value = True

        code, metadata = prompting.generate_strategy(context, max_iterations=1)

        assert code == "def strategy(): return 'buy'"
        assert metadata["iterations"] == 1
        assert "errors_encountered" in metadata
        assert "performance_history" in metadata

    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"), reason="Requires OpenAI API key"
    )
    def test_build_prompt_basic(self):
        """Test building a basic prompt."""
        config = TradingConfig()
        prompting = IterativePrompting(config)

        context = PromptContext(
            task_description="Test task",
            market_data={"trend": "up"},
            available_skills=["skill1"],
            previous_attempts=[],
            performance_feedback={},
        )

        prompt = prompting._build_prompt(context, "", 0)

        assert "Test task" in prompt
        assert "trend" in prompt
        assert "skill1" in prompt

    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"), reason="Requires OpenAI API key"
    )
    def test_build_prompt_with_code(self):
        """Test building prompt with existing code."""
        config = TradingConfig()
        prompting = IterativePrompting(config)

        context = PromptContext(
            task_description="Test task",
            market_data={},
            available_skills=[],
            previous_attempts=[],
            performance_feedback={},
        )

        current_code = "def old_strategy(): pass"
        prompt = prompting._build_prompt(context, current_code, 1)

        assert "Current Code:" in prompt
        assert current_code in prompt

    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"), reason="Requires OpenAI API key"
    )
    def test_build_prompt_with_feedback(self):
        """Test building prompt with performance feedback."""
        config = TradingConfig()
        prompting = IterativePrompting(config)

        context = PromptContext(
            task_description="Test task",
            market_data={},
            available_skills=[],
            previous_attempts=[],
            performance_feedback={"return": 0.05},
            error_messages=["Error 1", "Error 2"],
        )

        prompt = prompting._build_prompt(context, "", 0)

        assert "Performance Feedback:" in prompt
        assert "Previous Errors:" in prompt
        assert "Error 1" in prompt

    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"), reason="Requires OpenAI API key"
    )
    def test_extract_code(self):
        """Test extracting code from LLM response."""
        config = TradingConfig()
        prompting = IterativePrompting(config)

        response = """
        Here's the strategy:
        ```python
        def trading_strategy():
            return 'buy'
        ```
        That should work well.
        """

        code = prompting._extract_code(response)
        assert "def trading_strategy():" in code
        assert "return 'buy'" in code
        assert "Here's the strategy:" not in code

    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"), reason="Requires OpenAI API key"
    )
    def test_is_code_improved(self):
        """Test code improvement detection."""
        config = TradingConfig()
        prompting = IterativePrompting(config)

        context = PromptContext(
            task_description="Test",
            market_data={},
            available_skills=[],
            previous_attempts=[],
            performance_feedback={},
        )

        # Simple length-based improvement (placeholder logic)
        old_code = "def f(): pass"
        new_code = "def improved_function(): return 'better'"

        assert prompting._is_code_improved(old_code, new_code, context)

    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"), reason="Requires OpenAI API key"
    )
    def test_should_stop_high_performance(self):
        """Test stopping criteria with high performance."""
        config = TradingConfig()
        prompting = IterativePrompting(config)

        context = PromptContext(
            task_description="Test",
            market_data={},
            available_skills=[],
            previous_attempts=[],
            performance_feedback={"success_rate": 0.9},
        )

        should_stop = prompting._should_stop(context, "def strategy(): pass", 5)
        assert should_stop is True

    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"), reason="Requires OpenAI API key"
    )
    def test_should_stop_low_performance(self):
        """Test stopping criteria with low performance."""
        config = TradingConfig()
        prompting = IterativePrompting(config)

        context = PromptContext(
            task_description="Test",
            market_data={},
            available_skills=[],
            previous_attempts=[],
            performance_feedback={"success_rate": 0.3},
        )

        should_stop = prompting._should_stop(context, "def strategy(): pass", 5)
        assert should_stop is False
