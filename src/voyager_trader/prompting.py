"""
Iterative Prompting implementation.

Handles the iterative prompting mechanism for code generation
and strategy development, similar to VOYAGER's approach.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class PromptContext:
    """Context for iterative prompting."""

    task_description: str
    market_data: Dict[str, Any]
    available_skills: List[str]
    previous_attempts: List[Dict[str, Any]]
    performance_feedback: Dict[str, Any]
    error_messages: List[str] = None


class IterativePrompting:
    """
    Iterative prompting system for strategy generation.

    Manages the conversation flow with LLMs to generate and refine
    trading strategies through iterative feedback.
    """

    def __init__(self, config):
        """Initialize the iterative prompting system."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.conversation_history = []
        self.max_iterations = config.max_iterations

    def generate_strategy(
        self, context: PromptContext, max_iterations: Optional[int] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate a trading strategy through iterative prompting.

        Args:
            context: Prompting context with task and feedback
            max_iterations: Override default max iterations

        Returns:
            Tuple of (generated_code, generation_metadata)
        """
        max_iter = max_iterations or self.max_iterations
        self.logger.info(f"Starting strategy generation with max {max_iter} iterations")

        current_code = ""
        metadata = {
            "iterations": 0,
            "errors_encountered": [],
            "performance_history": [],
        }

        for iteration in range(max_iter):
            metadata["iterations"] = iteration + 1

            # Generate prompt based on current context
            prompt = self._build_prompt(context, current_code, iteration)

            # Get response from LLM (placeholder)
            response = self._query_llm(prompt)

            # Extract and validate code
            new_code = self._extract_code(response)

            if self._is_code_improved(current_code, new_code, context):
                current_code = new_code
                self.logger.info(f"Iteration {iteration + 1}: Code improved")
            else:
                self.logger.info(f"Iteration {iteration + 1}: No improvement")

            # Check stopping criteria
            if self._should_stop(context, current_code, iteration):
                break

        return current_code, metadata

    def _build_prompt(
        self, context: PromptContext, current_code: str, iteration: int
    ) -> str:
        """Build the prompt for the current iteration."""

        base_prompt = f"""
        Task: {context.task_description}

        Market Data: {context.market_data}

        Available Skills: {', '.join(context.available_skills)}

        """

        if current_code:
            base_prompt += f"""
            Current Code:
            ```python
            {current_code}
            ```

            """

        if context.performance_feedback:
            base_prompt += f"""
            Performance Feedback: {context.performance_feedback}

            """

        if context.error_messages:
            base_prompt += f"""
            Previous Errors: {'; '.join(context.error_messages)}

            """

        base_prompt += """
        Please generate an improved trading strategy implementation.
        Focus on addressing any errors and improving performance metrics.
        """

        return base_prompt

    def _query_llm(self, prompt: str) -> str:
        """Query the LLM with the given prompt."""
        # Placeholder implementation
        self.logger.info("Querying LLM for strategy generation...")

        # This would integrate with OpenAI API or similar
        return f"""
        def trading_strategy(market_data):
            # Generated trading strategy
            if market_data['trend'] == 'up':
                return {'action': 'buy', 'quantity': 100}
            else:
                return {'action': 'hold'}
        """

    def _extract_code(self, response: str) -> str:
        """Extract executable code from LLM response."""
        # Simple extraction - look for code blocks
        lines = response.split("\n")
        code_lines = []
        in_code_block = False

        for line in lines:
            if line.strip().startswith("```python"):
                in_code_block = True
                continue
            elif line.strip().startswith("```"):
                in_code_block = False
                continue
            elif in_code_block:
                code_lines.append(line)

        return "\n".join(code_lines)

    def _is_code_improved(
        self, old_code: str, new_code: str, context: PromptContext
    ) -> bool:
        """Determine if new code is an improvement over old code."""
        # Placeholder logic
        return len(new_code) > len(old_code)

    def _should_stop(
        self, context: PromptContext, current_code: str, iteration: int
    ) -> bool:
        """Determine if iteration should stop."""
        # Stop if we have reasonable code and performance is good
        if current_code and context.performance_feedback.get("success_rate", 0) > 0.8:
            return True

        return False
