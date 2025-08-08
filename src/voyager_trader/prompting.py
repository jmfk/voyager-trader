"""Iterative Prompting and Self-Improvement System.

Comprehensive implementation of VOYAGER's iterative prompting mechanism
with six core components:
1. LLM Integration Layer - Multi-provider LLM support with rate limiting
2. Code Execution Environment - Sandboxed execution with resource monitoring
3. Performance Evaluation System - Multi-metric strategy assessment
4. Feedback Synthesis - Converts results into actionable improvement prompts
5. Strategy Refinement Engine - Orchestrates iterative improvement
6. Safety Monitoring - Real-time safety validation and risk limiting

Follows architecture defined in ADR-0014.
"""

import ast
import json
import logging
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from .curriculum import AutomaticCurriculum
from .llm_service import (
    LLMError,
    LLMProvider,
    LLMRequest,
    ProviderError,
    UniversalLLMClient,
    get_global_llm_client,
)
from .models.learning import Skill
from .models.types import SkillCategory, SkillComplexity
from .skills import SkillLibrarian


class PromptingError(Exception):
    """Base exception for prompting system errors."""


class LLMIntegrationError(PromptingError):
    """Exception raised during LLM integration."""


class CodeExecutionError(PromptingError):
    """Exception raised during code execution."""


class SafetyValidationError(PromptingError):
    """Exception raised during safety validation."""


class PerformanceEvaluationError(PromptingError):
    """Exception raised during performance evaluation."""


class ExecutionStatus(str, Enum):
    """Code execution status."""

    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    MEMORY_LIMIT = "memory_limit"
    SAFETY_VIOLATION = "safety_violation"


class ConvergenceReason(str, Enum):
    """Reasons for iteration convergence."""

    PERFORMANCE_TARGET = "performance_target"
    MAX_ITERATIONS = "max_iterations"
    NO_IMPROVEMENT = "no_improvement"
    ERROR_THRESHOLD = "error_threshold"
    SAFETY_VIOLATION = "safety_violation"


@dataclass
class IterativePromptingConfig:
    """Configuration for iterative prompting system."""

    # LLM Configuration
    llm_provider: LLMProvider = LLMProvider.OPENAI
    llm_model: str = "gpt-4"
    max_tokens: int = 2000
    temperature: float = 0.7
    api_key: Optional[str] = None
    rate_limit_requests_per_minute: int = 60

    # Execution Configuration
    execution_timeout: int = 30
    memory_limit_mb: int = 512
    max_cpu_percent: float = 50.0
    python_path: str = sys.executable

    # Iteration Configuration
    max_iterations: int = 10
    convergence_threshold: float = 0.01
    min_performance_improvement: float = 0.05
    max_consecutive_failures: int = 3

    # Safety Configuration
    enable_safety_checks: bool = True
    allowed_imports: List[str] = field(
        default_factory=lambda: [
            "numpy",
            "pandas",
            "math",
            "statistics",
            "decimal",
            "datetime",
            "json",
            "typing",
            "dataclasses",
            "enum",
            "collections",
        ]
    )
    forbidden_patterns: List[str] = field(
        default_factory=lambda: [
            "import os",
            "import subprocess",
            "import sys",
            "__import__",
            "eval",
            "exec",
            "open",
            "file",
            "input",
            "raw_input",
            "subprocess",
            "os.system",
            "os.popen",
            "globals",
            "locals",
        ]
    )
    max_execution_memory_mb: int = 256

    # Performance Configuration
    performance_metrics: List[str] = field(
        default_factory=lambda: [
            "sharpe_ratio",
            "max_drawdown",
            "total_return",
            "win_rate",
            "profit_factor",
            "volatility",
            "execution_time",
        ]
    )
    target_sharpe_ratio: float = 1.5
    max_drawdown_threshold: float = 0.15


@dataclass
class PromptContext:
    """Enhanced context for iterative prompting."""

    task_description: str
    market_data: Dict[str, Any]
    available_skills: List[str]
    previous_attempts: List[Dict[str, Any]] = field(default_factory=list)
    performance_feedback: Dict[str, Any] = field(default_factory=dict)
    error_messages: List[str] = field(default_factory=list)
    safety_constraints: List[str] = field(default_factory=list)
    target_metrics: Dict[str, float] = field(default_factory=dict)
    market_regime: str = "normal"
    risk_tolerance: str = "moderate"


@dataclass
class ExecutionResult:
    """Result of code execution."""

    status: ExecutionStatus
    output: Any = None
    error_message: str = ""
    execution_time: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    safety_violations: List[str] = field(default_factory=list)


@dataclass
class IterationResult:
    """Result of a single iteration."""

    iteration: int
    generated_code: str
    execution_result: ExecutionResult
    performance_improvement: float = 0.0
    feedback_summary: str = ""
    convergence_signal: float = 0.0


@dataclass
class RefinementResult:
    """Final result of iterative refinement."""

    final_code: str
    best_performance: Dict[str, float]
    total_iterations: int
    convergence_reason: ConvergenceReason
    iteration_history: List[IterationResult]
    total_time: float
    llm_tokens_used: int = 0


class LLMIntegrationLayer:
    """Simplified LLM integration using centralized service."""

    def __init__(
        self,
        config: IterativePromptingConfig,
        llm_client: Optional[UniversalLLMClient] = None,
    ):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Use provided client or get global one
        self.llm_client = llm_client or get_global_llm_client()

    async def generate_strategy_code(self, prompt: str) -> str:
        """Generate strategy code using centralized LLM service."""
        try:
            # Create system prompt
            system_prompt = self._get_system_prompt()

            # Create LLM request
            request = LLMRequest(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                model=self.config.llm_model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                provider=(
                    self.config.llm_provider.value
                    if isinstance(self.config.llm_provider, LLMProvider)
                    else self.config.llm_provider
                ),
            )

            # Generate response
            response = await self.llm_client.generate(request)

            # Extract code from response
            return self._extract_code_from_response(response.content)

        except (LLMError, ProviderError) as e:
            self.logger.error(f"LLM generation failed: {e}")
            raise LLMIntegrationError(f"Failed to generate code: {str(e)}")
        except Exception as e:
            self.logger.error(f"Unexpected error in LLM generation: {e}")
            raise LLMIntegrationError(f"Unexpected error: {str(e)}")

    def _get_system_prompt(self) -> str:
        """Get system prompt for strategy generation."""
        return """You are an expert trading strategy developer. Generate code that:
1. Implements a complete trading_strategy(market_data) function
2. Takes market data as input and returns trading decisions
   as {'action': 'buy'|'sell'|'hold', 'quantity': int}
3. Uses only allowed imports and follows safety constraints
4. Includes proper error handling and data validation
5. Optimizes for risk-adjusted returns
6. Consider market conditions and risk management

Always wrap your code in ```python``` code blocks."""

    def _extract_code_from_response(self, content: str) -> str:
        """Extract Python code from LLM response."""
        # Look for code blocks
        lines = content.split("\n")
        code_lines = []
        in_code_block = False

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("```python"):
                in_code_block = True
                continue
            elif stripped.startswith("```") and in_code_block:
                in_code_block = False
                continue
            elif in_code_block:
                code_lines.append(line)

        if code_lines:
            return "\n".join(code_lines)
        else:
            # If no code blocks found, return original content
            return content


class CodeExecutionEnvironment:
    """Sandboxed code execution environment."""

    def __init__(self, config: IterativePromptingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.safety_monitor = SafetyMonitor(config)

    def execute_strategy_code(
        self, code: str, market_data: Dict[str, Any]
    ) -> ExecutionResult:
        """Execute strategy code in sandboxed environment."""
        start_time = time.time()

        # Pre-execution safety validation
        safety_violations = self.safety_monitor.validate_code(code)
        if safety_violations:
            return ExecutionResult(
                status=ExecutionStatus.SAFETY_VIOLATION,
                error_message="Safety violations detected",
                safety_violations=safety_violations,
            )

        # Create temporary execution environment
        with tempfile.TemporaryDirectory() as temp_dir:
            strategy_file = Path(temp_dir) / "strategy.py"
            strategy_file.write_text(code)

            try:
                return self._execute_in_subprocess(
                    strategy_file, market_data, start_time
                )
            except Exception as e:
                execution_time = time.time() - start_time
                return ExecutionResult(
                    status=ExecutionStatus.ERROR,
                    error_message=str(e),
                    execution_time=execution_time,
                )

    def _execute_in_subprocess(
        self,
        strategy_file: Path,
        market_data: Dict[str, Any],
        start_time: float,
    ) -> ExecutionResult:
        """Execute code in isolated subprocess with resource monitoring."""
        execution_script = self._create_execution_script(strategy_file, market_data)

        try:
            # Start process with resource limits
            process = subprocess.Popen(
                [self.config.python_path, "-c", execution_script],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=strategy_file.parent,
            )

            # Monitor resource usage if psutil available
            max_memory = 0.0
            max_cpu = 0.0

            if PSUTIL_AVAILABLE:
                ps_process = psutil.Process(process.pid)
                max_memory, max_cpu = self._monitor_process_resources(ps_process)

            # Wait for completion with timeout
            try:
                stdout, stderr = process.communicate(
                    timeout=self.config.execution_timeout
                )
                execution_time = time.time() - start_time

                if process.returncode == 0:
                    try:
                        result_data = json.loads(stdout)
                        return ExecutionResult(
                            status=ExecutionStatus.SUCCESS,
                            output=result_data,
                            execution_time=execution_time,
                            memory_usage_mb=max_memory,
                            cpu_usage_percent=max_cpu,
                        )
                    except json.JSONDecodeError:
                        return ExecutionResult(
                            status=ExecutionStatus.ERROR,
                            error_message="Invalid JSON output from strategy",
                            execution_time=execution_time,
                        )
                else:
                    return ExecutionResult(
                        status=ExecutionStatus.ERROR,
                        error_message=stderr or "Unknown execution error",
                        execution_time=execution_time,
                    )

            except subprocess.TimeoutExpired:
                process.kill()
                return ExecutionResult(
                    status=ExecutionStatus.TIMEOUT,
                    error_message=(
                        f"Execution timed out after {self.config.execution_timeout}s"
                    ),
                    execution_time=self.config.execution_timeout,
                )

        except Exception as e:
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                error_message=f"Subprocess execution failed: {str(e)}",
                execution_time=time.time() - start_time,
            )

    def _create_execution_script(
        self, strategy_file: Path, market_data: Dict[str, Any]
    ) -> str:
        """Create execution script for sandboxed environment."""
        return f"""
import json
import sys
import traceback
from pathlib import Path

# Add strategy directory to path
sys.path.insert(0, "{strategy_file.parent}")

try:
    # Import strategy module
    import strategy

    # Prepare market data
    market_data = {json.dumps(market_data)}

    # Execute strategy
    result = strategy.trading_strategy(market_data)

    # Return result as JSON
    print(json.dumps({{
        "success": True,
        "result": result,
        "market_data": market_data
    }}))

except Exception as e:
    print(json.dumps({{
        "success": False,
        "error": str(e),
        "traceback": traceback.format_exc()
    }}), file=sys.stderr)
    sys.exit(1)
"""

    def _monitor_process_resources(self, ps_process) -> Tuple[float, float]:
        """Monitor process resource usage."""
        max_memory = 0.0
        max_cpu = 0.0

        try:
            # Monitor for a short period
            for _ in range(10):
                if ps_process.is_running():
                    memory_mb = ps_process.memory_info().rss / (1024 * 1024)
                    cpu_percent = ps_process.cpu_percent()

                    max_memory = max(max_memory, memory_mb)
                    max_cpu = max(max_cpu, cpu_percent)

                    # Check resource limits
                    if memory_mb > self.config.memory_limit_mb:
                        ps_process.kill()
                        raise CodeExecutionError("Memory limit exceeded")

                    if cpu_percent > self.config.max_cpu_percent:
                        self.logger.warning(f"High CPU usage: {cpu_percent}%")

                    time.sleep(0.1)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

        return max_memory, max_cpu


class SafetyMonitor:
    """Safety monitoring and validation system."""

    def __init__(self, config: IterativePromptingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def validate_code(self, code: str) -> List[str]:
        """Validate code for safety violations."""
        violations = []

        if not self.config.enable_safety_checks:
            return violations

        # Check forbidden patterns
        for pattern in self.config.forbidden_patterns:
            if pattern in code:
                violations.append(f"Forbidden pattern detected: {pattern}")

        # AST-based validation
        try:
            tree = ast.parse(code)
            violations.extend(self._analyze_ast(tree))
        except SyntaxError as e:
            violations.append(f"Syntax error: {str(e)}")

        return violations

    def _analyze_ast(self, tree: ast.AST) -> List[str]:
        """Analyze AST for dangerous patterns."""
        violations = []

        for node in ast.walk(tree):
            # Check imports
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                violation = self._check_import_safety(node)
                if violation:
                    violations.append(violation)

            # Check function calls
            elif isinstance(node, ast.Call):
                violation = self._check_call_safety(node)
                if violation:
                    violations.append(violation)

        return violations

    def _check_import_safety(
        self, node: Union[ast.Import, ast.ImportFrom]
    ) -> Optional[str]:
        """Check if import is safe."""
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in self.config.allowed_imports:
                    return f"Disallowed import: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module not in self.config.allowed_imports:
                return f"Disallowed import from: {node.module}"
        return None

    def _check_call_safety(self, node: ast.Call) -> Optional[str]:
        """Check if function call is safe."""
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            dangerous_functions = [
                "eval",
                "exec",
                "__import__",
                "open",
                "input",
            ]
            if func_name in dangerous_functions:
                return f"Dangerous function call: {func_name}"
        return None


class PerformanceEvaluator:
    """Multi-metric performance evaluation system."""

    def __init__(self, config: IterativePromptingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def evaluate_strategy_performance(
        self, execution_result: ExecutionResult, market_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """Evaluate strategy performance across multiple metrics."""
        if execution_result.status != ExecutionStatus.SUCCESS:
            return {metric: 0.0 for metric in self.config.performance_metrics}

        try:
            strategy_output = execution_result.output
            return self._calculate_performance_metrics(strategy_output, market_data)

        except Exception as e:
            self.logger.error(f"Performance evaluation failed: {e}")
            raise PerformanceEvaluationError(
                f"Failed to evaluate performance: {str(e)}"
            )

    def _calculate_performance_metrics(
        self, strategy_output: Dict[str, Any], market_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        metrics = {}

        # Mock implementation - would integrate with backtesting engine
        metrics["sharpe_ratio"] = self._calculate_mock_sharpe_ratio(strategy_output)
        metrics["max_drawdown"] = self._calculate_mock_max_drawdown(strategy_output)
        metrics["total_return"] = self._calculate_mock_total_return(strategy_output)
        metrics["win_rate"] = self._calculate_mock_win_rate(strategy_output)
        metrics["profit_factor"] = self._calculate_mock_profit_factor(strategy_output)
        metrics["volatility"] = self._calculate_mock_volatility(strategy_output)

        return metrics

    def _calculate_mock_sharpe_ratio(self, strategy_output: Dict[str, Any]) -> float:
        """Mock Sharpe ratio calculation."""
        # In real implementation, would calculate from returns series
        return 1.2 if strategy_output.get("action") == "buy" else 0.8

    def _calculate_mock_max_drawdown(self, strategy_output: Dict[str, Any]) -> float:
        """Mock max drawdown calculation."""
        return 0.05 if strategy_output.get("action") != "hold" else 0.02

    def _calculate_mock_total_return(self, strategy_output: Dict[str, Any]) -> float:
        """Mock total return calculation."""
        return 0.15 if strategy_output.get("action") == "buy" else 0.02

    def _calculate_mock_win_rate(self, strategy_output: Dict[str, Any]) -> float:
        """Mock win rate calculation."""
        return 0.65 if strategy_output.get("action") == "buy" else 0.50

    def _calculate_mock_profit_factor(self, strategy_output: Dict[str, Any]) -> float:
        """Mock profit factor calculation."""
        return 1.8 if strategy_output.get("action") == "buy" else 1.1

    def _calculate_mock_volatility(self, strategy_output: Dict[str, Any]) -> float:
        """Mock volatility calculation."""
        return 0.12 if strategy_output.get("action") == "buy" else 0.08


class FeedbackSynthesizer:
    """Converts execution results into actionable improvement prompts."""

    def __init__(self, config: IterativePromptingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def synthesize_feedback(
        self, iteration_result: IterationResult, context: PromptContext
    ) -> str:
        """Synthesize feedback for strategy improvement."""
        feedback_parts = []

        # Performance feedback
        performance_feedback = self._generate_performance_feedback(
            iteration_result.execution_result, context
        )
        if performance_feedback:
            feedback_parts.append(performance_feedback)

        # Error feedback
        error_feedback = self._generate_error_feedback(
            iteration_result.execution_result
        )
        if error_feedback:
            feedback_parts.append(error_feedback)

        # Safety feedback
        safety_feedback = self._generate_safety_feedback(
            iteration_result.execution_result
        )
        if safety_feedback:
            feedback_parts.append(safety_feedback)

        # Resource usage feedback
        resource_feedback = self._generate_resource_feedback(
            iteration_result.execution_result
        )
        if resource_feedback:
            feedback_parts.append(resource_feedback)

        return (
            "\n\n".join(feedback_parts)
            if feedback_parts
            else "No specific feedback available."
        )

    def _generate_performance_feedback(
        self, execution_result: ExecutionResult, context: PromptContext
    ) -> str:
        """Generate performance-focused feedback."""
        if execution_result.status != ExecutionStatus.SUCCESS:
            return ""

        feedback = ["PERFORMANCE ANALYSIS:"]

        for metric, value in execution_result.performance_metrics.items():
            target = context.target_metrics.get(metric)
            if target:
                if value >= target:
                    feedback.append(
                        f"✓ {metric}: {value:.3f} (target: {target:.3f}) - GOOD"
                    )
                else:
                    improvement = target - value
                    feedback.append(
                        f"× {metric}: {value:.3f} (target: {target:.3f}) - "
                        f"IMPROVE by {improvement:.3f}"
                    )
            else:
                feedback.append(f"• {metric}: {value:.3f}")

        # Specific improvement suggestions
        suggestions = self._generate_performance_suggestions(
            execution_result.performance_metrics, context
        )
        if suggestions:
            feedback.append("\nSUGGESTIONS:")
            feedback.extend(suggestions)

        return "\n".join(feedback)

    def _generate_error_feedback(self, execution_result: ExecutionResult) -> str:
        """Generate error-focused feedback."""
        if execution_result.status == ExecutionStatus.SUCCESS:
            return ""

        feedback = ["ERROR ANALYSIS:"]
        feedback.append(f"Status: {execution_result.status.value}")

        if execution_result.error_message:
            feedback.append(f"Error: {execution_result.error_message}")

            # Suggest fixes based on error type
            suggestions = self._generate_error_suggestions(
                execution_result.error_message
            )
            if suggestions:
                feedback.append("\nSUGGESTED FIXES:")
                feedback.extend(suggestions)

        return "\n".join(feedback)

    def _generate_safety_feedback(self, execution_result: ExecutionResult) -> str:
        """Generate safety-focused feedback."""
        if not execution_result.safety_violations:
            return ""

        feedback = ["SAFETY VIOLATIONS:"]
        for violation in execution_result.safety_violations:
            feedback.append(f"× {violation}")

        feedback.append("\nREQUIRED ACTIONS:")
        feedback.append("- Remove or replace unsafe code patterns")
        feedback.append("- Use only allowed imports and functions")
        feedback.append("- Follow defensive programming practices")

        return "\n".join(feedback)

    def _generate_resource_feedback(self, execution_result: ExecutionResult) -> str:
        """Generate resource usage feedback."""
        feedback_parts = []

        if execution_result.execution_time > 10:
            feedback_parts.append(
                f"⚠ Slow execution: {execution_result.execution_time:.2f}s"
            )
            feedback_parts.append("Consider optimizing algorithm complexity")

        if execution_result.memory_usage_mb > 100:
            feedback_parts.append(
                f"⚠ High memory usage: {execution_result.memory_usage_mb:.1f}MB"
            )
            feedback_parts.append("Consider memory-efficient data structures")

        return "\n".join(feedback_parts) if feedback_parts else ""

    def _generate_performance_suggestions(
        self, metrics: Dict[str, float], context: PromptContext
    ) -> List[str]:
        """Generate specific performance improvement suggestions."""
        suggestions = []

        if metrics.get("sharpe_ratio", 0) < self.config.target_sharpe_ratio:
            suggestions.append(
                "- Improve risk-adjusted returns by optimizing entry/exit signals"
            )
            suggestions.append("- Consider position sizing based on volatility")

        if metrics.get("max_drawdown", 1) > self.config.max_drawdown_threshold:
            suggestions.append("- Implement stop-loss mechanisms to limit drawdowns")
            suggestions.append(
                "- Add trend-following filters to avoid adverse market conditions"
            )

        if metrics.get("win_rate", 0) < 0.5:
            suggestions.append("- Refine entry criteria to improve trade accuracy")
            suggestions.append("- Consider market regime filters")

        return suggestions

    def _generate_error_suggestions(self, error_message: str) -> List[str]:
        """Generate error-specific suggestions."""
        suggestions = []
        error_lower = error_message.lower()

        if "keyerror" in error_lower:
            suggestions.append("- Add proper error handling for missing data keys")
            suggestions.append("- Validate input data structure before processing")

        if "typeerror" in error_lower:
            suggestions.append("- Add type checking and conversion")
            suggestions.append("- Ensure consistent data types throughout strategy")

        if "indexerror" in error_lower:
            suggestions.append("- Add bounds checking for array/list access")
            suggestions.append("- Handle empty or insufficient data gracefully")

        if "zerodivisionerror" in error_lower:
            suggestions.append("- Add checks for zero values before division")
            suggestions.append("- Use safe division with default values")

        return suggestions


class StrategyRefinementEngine:
    """Orchestrates the iterative strategy improvement process."""

    def __init__(
        self,
        config: IterativePromptingConfig,
        llm_client: Optional[UniversalLLMClient] = None,
    ):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize components with optional LLM client
        self.llm_integration = LLMIntegrationLayer(config, llm_client)
        self.execution_env = CodeExecutionEnvironment(config)
        self.performance_evaluator = PerformanceEvaluator(config)
        self.feedback_synthesizer = FeedbackSynthesizer(config)

        # State tracking
        self.iteration_history: List[IterationResult] = []
        self.best_performance: Dict[str, float] = {}
        self.consecutive_failures = 0

    async def refine_strategy(self, context: PromptContext) -> RefinementResult:
        """Execute the complete iterative refinement process."""
        start_time = time.time()
        self.logger.info(
            f"Starting strategy refinement for: {context.task_description}"
        )

        self.iteration_history.clear()
        self.best_performance.clear()
        self.consecutive_failures = 0
        best_code = ""
        total_tokens = 0

        for iteration in range(self.config.max_iterations):
            self.logger.info(
                f"Starting iteration {iteration + 1}/{self.config.max_iterations}"
            )

            try:
                # Generate strategy code
                prompt = self._build_iteration_prompt(context, iteration)
                generated_code = await self.llm_integration.generate_strategy_code(
                    prompt
                )
                total_tokens += len(prompt.split()) + len(
                    generated_code.split()
                )  # Rough estimate

                # Execute and evaluate
                execution_result = self.execution_env.execute_strategy_code(
                    generated_code, context.market_data
                )

                if execution_result.status == ExecutionStatus.SUCCESS:
                    execution_result.performance_metrics = (
                        self.performance_evaluator.evaluate_strategy_performance(
                            execution_result, context.market_data
                        )
                    )

                # Create iteration result
                iteration_result = IterationResult(
                    iteration=iteration + 1,
                    generated_code=generated_code,
                    execution_result=execution_result,
                    performance_improvement=self._calculate_performance_improvement(
                        execution_result
                    ),
                    feedback_summary=self.feedback_synthesizer.synthesize_feedback(
                        IterationResult(
                            iteration + 1, generated_code, execution_result
                        ),
                        context,
                    ),
                    convergence_signal=0.0,  # Will be calculated after creation
                )

                self.iteration_history.append(iteration_result)

                # Update best performance and code
                if self._is_improvement(execution_result):
                    self.best_performance = execution_result.performance_metrics.copy()
                    best_code = generated_code
                    self.consecutive_failures = 0
                    self.logger.info(f"Iteration {iteration + 1}: Performance improved")
                else:
                    self.consecutive_failures += 1
                    self.logger.info(
                        f"Iteration {iteration + 1}: No improvement "
                        f"({self.consecutive_failures} consecutive)"
                    )

                # Check convergence criteria
                convergence_reason = self._check_convergence_criteria(iteration)
                if convergence_reason:
                    self.logger.info(
                        f"Converged after {iteration + 1} iterations: "
                        f"{convergence_reason.value}"
                    )
                    break

            except Exception as e:
                self.logger.error(f"Iteration {iteration + 1} failed: {e}")
                self.consecutive_failures += 1

                if self.consecutive_failures >= self.config.max_consecutive_failures:
                    convergence_reason = ConvergenceReason.ERROR_THRESHOLD
                    break

        else:
            convergence_reason = ConvergenceReason.MAX_ITERATIONS

        total_time = time.time() - start_time

        return RefinementResult(
            final_code=best_code,
            best_performance=self.best_performance,
            total_iterations=len(self.iteration_history),
            convergence_reason=convergence_reason,
            iteration_history=self.iteration_history,
            total_time=total_time,
            llm_tokens_used=total_tokens,
        )

    def _build_iteration_prompt(self, context: PromptContext, iteration: int) -> str:
        """Build comprehensive prompt for current iteration."""
        prompt_parts = [
            "TRADING STRATEGY GENERATION TASK",
            "=" * 40,
            f"Task: {context.task_description}",
            f"Market Regime: {context.market_regime}",
            f"Risk Tolerance: {context.risk_tolerance}",
            "",
            "MARKET DATA:",
        ]

        # Add market data context
        for key, value in context.market_data.items():
            prompt_parts.append(f"- {key}: {value}")

        # Add available skills
        if context.available_skills:
            prompt_parts.extend(
                [
                    "",
                    "AVAILABLE SKILLS:",
                    "- " + "\n- ".join(context.available_skills),
                ]
            )

        # Add target metrics
        if context.target_metrics:
            prompt_parts.extend(["", "TARGET METRICS:"])
            for metric, target in context.target_metrics.items():
                prompt_parts.append(f"- {metric}: {target}")

        # Add iteration-specific context
        if iteration > 0 and self.iteration_history:
            prompt_parts.extend(
                [
                    "",
                    "PREVIOUS ITERATION FEEDBACK:",
                    self.iteration_history[-1].feedback_summary,
                ]
            )

            if self.iteration_history[-1].generated_code:
                prompt_parts.extend(
                    [
                        "",
                        "PREVIOUS CODE (for reference):",
                        "```python",
                        self.iteration_history[-1].generated_code,
                        "```",
                    ]
                )

        # Add safety constraints
        if context.safety_constraints:
            prompt_parts.extend(
                [
                    "",
                    "SAFETY CONSTRAINTS:",
                    "- " + "\n- ".join(context.safety_constraints),
                ]
            )

        prompt_parts.extend(
            [
                "",
                "REQUIREMENTS:",
                "1. Generate a complete trading_strategy(market_data) function",
                "2. Return trading decisions as "
                "{'action': 'buy'|'sell'|'hold', 'quantity': int}",
                "3. Use only allowed imports and safe coding practices",
                "4. Include proper error handling and data validation",
                "5. Optimize for risk-adjusted returns",
                "6. Consider market regime and risk tolerance",
                "",
                "Please provide your improved trading strategy implementation:",
            ]
        )

        return "\n".join(prompt_parts)

    def _calculate_performance_improvement(
        self, execution_result: ExecutionResult
    ) -> float:
        """Calculate performance improvement over previous best."""
        if (
            execution_result.status != ExecutionStatus.SUCCESS
            or not self.best_performance
        ):
            return 0.0

        current_sharpe = execution_result.performance_metrics.get("sharpe_ratio", 0)
        best_sharpe = self.best_performance.get("sharpe_ratio", 0)

        if best_sharpe == 0:
            return 1.0 if current_sharpe > 0 else 0.0

        return (current_sharpe - best_sharpe) / best_sharpe

    def _calculate_convergence_signal(self, iteration_result: IterationResult) -> float:
        """Calculate convergence signal based on recent performance."""
        if len(self.iteration_history) < 3:
            return 0.0

        recent_improvements = [
            result.performance_improvement
            for result in self.iteration_history[-3:]
            if result.execution_result.status == ExecutionStatus.SUCCESS
        ]

        if not recent_improvements:
            return 1.0  # High convergence signal if no recent successes

        return 1.0 - (sum(recent_improvements) / len(recent_improvements))

    def _is_improvement(self, execution_result: ExecutionResult) -> bool:
        """Check if current result is an improvement."""
        if execution_result.status != ExecutionStatus.SUCCESS:
            return False

        if not self.best_performance:
            return True

        improvement = self._calculate_performance_improvement(execution_result)
        return improvement >= self.config.min_performance_improvement

    def _check_convergence_criteria(
        self, iteration: int
    ) -> Optional[ConvergenceReason]:
        """Check if any convergence criteria are met."""
        # Check performance target
        if self.best_performance:
            sharpe_ratio = self.best_performance.get("sharpe_ratio", 0)
            if sharpe_ratio >= self.config.target_sharpe_ratio:
                return ConvergenceReason.PERFORMANCE_TARGET

        # Check consecutive failures
        if self.consecutive_failures >= self.config.max_consecutive_failures:
            return ConvergenceReason.ERROR_THRESHOLD

        # Check convergence threshold
        if len(self.iteration_history) >= 3:
            convergence_signal = self._calculate_convergence_signal(
                self.iteration_history[-1]
            )
            if convergence_signal >= (1.0 - self.config.convergence_threshold):
                return ConvergenceReason.NO_IMPROVEMENT

        return None


class IterativePrompting:
    """Main iterative prompting system interface with curriculum and skill
    integration."""

    def __init__(
        self,
        config: IterativePromptingConfig,
        curriculum=None,
        skill_librarian=None,
        llm_client: Optional[UniversalLLMClient] = None,
    ):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.refinement_engine = StrategyRefinementEngine(config, llm_client)

        # Integration components
        self.curriculum = curriculum
        self.skill_librarian = skill_librarian
        self.llm_client = llm_client

    async def generate_strategy(self, context: PromptContext) -> RefinementResult:
        """Generate and refine a trading strategy through iterative prompting."""
        self.logger.info(
            f"Starting iterative strategy generation: {context.task_description}"
        )

        # Enhance context with curriculum and skills if available
        enhanced_context = await self._enhance_context_with_integrations(context)

        try:
            result = await self.refinement_engine.refine_strategy(enhanced_context)

            # Store successful strategy as skill if skill librarian available
            if self.skill_librarian and result.final_code and result.best_performance:
                await self._store_successful_strategy_as_skill(result, enhanced_context)

            self.logger.info(
                f"Strategy generation completed: {result.total_iterations} iterations, "
                f"best Sharpe ratio: "
                f"{result.best_performance.get('sharpe_ratio', 0):.3f}, "
                f"converged due to: {result.convergence_reason.value}"
            )

            return result

        except Exception as e:
            self.logger.error(f"Strategy generation failed: {e}")
            raise PromptingError(f"Failed to generate strategy: {str(e)}")

    async def _enhance_context_with_integrations(
        self, context: PromptContext
    ) -> PromptContext:
        """Enhance context with curriculum and skill library information."""
        enhanced_context = context

        # Add available skills from skill library
        if self.skill_librarian:
            try:
                available_skills = await self._get_relevant_skills(context)
                enhanced_context.available_skills.extend(available_skills)
                self.logger.info(
                    f"Added {len(available_skills)} relevant skills to context"
                )
            except Exception as e:
                self.logger.warning(f"Failed to fetch relevant skills: {e}")

        # Add curriculum-based constraints and targets
        if self.curriculum:
            try:
                curriculum_enhancements = await self._get_curriculum_enhancements(
                    context
                )
                enhanced_context.target_metrics.update(
                    curriculum_enhancements.get("target_metrics", {})
                )
                enhanced_context.safety_constraints.extend(
                    curriculum_enhancements.get("safety_constraints", [])
                )
                self.logger.info("Enhanced context with curriculum information")
            except Exception as e:
                self.logger.warning(f"Failed to enhance with curriculum: {e}")

        return enhanced_context

    async def _get_relevant_skills(self, context: PromptContext) -> List[str]:
        """Retrieve relevant skills from the skill library."""
        try:
            # Search for skills related to the task
            # This would use the SkillLibrarian's search functionality
            skill_descriptions = []

            # Mock implementation - would integrate with actual SkillLibrarian methods
            if hasattr(self.skill_librarian, "search_skills"):
                skills = self.skill_librarian.search_skills(
                    query=context.task_description,
                    market_regime=context.market_regime,
                    risk_tolerance=context.risk_tolerance,
                )

                for skill in skills[:5]:  # Limit to top 5 relevant skills
                    skill_descriptions.append(f"{skill.name}: {skill.description}")

            return skill_descriptions

        except Exception as e:
            self.logger.error(f"Error retrieving relevant skills: {e}")
            return []

    async def _get_curriculum_enhancements(
        self, context: PromptContext
    ) -> Dict[str, Any]:
        """Get curriculum-based enhancements for the context."""
        enhancements = {"target_metrics": {}, "safety_constraints": []}

        try:
            # Get current curriculum task if available
            if hasattr(self.curriculum, "get_current_task"):
                current_task = self.curriculum.get_current_task()
                if current_task:
                    # Set target metrics based on curriculum difficulty
                    difficulty_multiplier = 1.0
                    if hasattr(current_task, "difficulty_level"):
                        difficulty_map = {
                            "easy": 0.8,
                            "medium": 1.0,
                            "hard": 1.2,
                            "expert": 1.5,
                        }
                        difficulty_multiplier = difficulty_map.get(
                            current_task.difficulty_level, 1.0
                        )

                    # Adjust target metrics based on curriculum progression
                    enhancements["target_metrics"] = {
                        "sharpe_ratio": self.config.target_sharpe_ratio
                        * difficulty_multiplier,
                        "max_drawdown": self.config.max_drawdown_threshold
                        / difficulty_multiplier,
                        "win_rate": 0.55 * difficulty_multiplier,
                    }

                    # Add curriculum-specific safety constraints
                    enhancements["safety_constraints"] = [
                        "Follow risk management principles appropriate for "
                        "difficulty level",
                        f"Ensure strategy complexity matches "
                        f"{current_task.difficulty_level} level",
                        "Implement proper position sizing for learning stage",
                    ]

        except Exception as e:
            self.logger.error(f"Error getting curriculum enhancements: {e}")

        return enhancements

    async def _store_successful_strategy_as_skill(
        self, result: RefinementResult, context: PromptContext
    ):
        """Store successful strategy as a skill in the library."""
        try:
            if (
                result.best_performance.get("sharpe_ratio", 0)
                >= self.config.target_sharpe_ratio
            ):
                skill_name = (
                    f"strategy_{context.task_description[:30].replace(' ', '_')}"
                )
                skill_description = f"Trading strategy for {context.task_description}"

                # Create skill object
                skill = Skill(
                    name=skill_name,
                    description=skill_description,
                    code=result.final_code,
                    category=SkillCategory.TRADING,
                    complexity=SkillComplexity.INTERMEDIATE,
                    performance_metrics=result.best_performance,
                    market_conditions={
                        "regime": context.market_regime,
                        "risk_tolerance": context.risk_tolerance,
                    },
                )

                # Store in skill library
                if hasattr(self.skill_librarian, "store_skill"):
                    success = self.skill_librarian.store_skill(skill)
                    if success:
                        self.logger.info(
                            f"Successfully stored strategy as skill: {skill_name}"
                        )
                    else:
                        self.logger.warning(f"Failed to store skill: {skill_name}")

        except Exception as e:
            self.logger.error(f"Error storing strategy as skill: {e}")


class VoyagerIterativeSystem:
    """Complete VOYAGER iterative prompting system with full integration."""

    def __init__(
        self,
        config: IterativePromptingConfig,
        curriculum_config=None,
        skill_config=None,
        llm_client: Optional[UniversalLLMClient] = None,
    ):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize integrated components
        self.curriculum = (
            AutomaticCurriculum(curriculum_config) if curriculum_config else None
        )
        self.skill_librarian = SkillLibrarian(skill_config) if skill_config else None
        self.prompting_system = IterativePrompting(
            config, self.curriculum, self.skill_librarian, llm_client
        )
        self.llm_client = llm_client

    async def run_learning_episode(self, initial_task: str = None) -> Dict[str, Any]:
        """Run a complete learning episode with curriculum-guided strategy
        development."""
        episode_results = {
            "tasks_completed": 0,
            "strategies_generated": 0,
            "skills_acquired": 0,
            "total_iterations": 0,
            "episode_duration": 0.0,
        }

        start_time = time.time()

        try:
            # Get task from curriculum or use provided task
            if self.curriculum and not initial_task:
                task = await self._get_next_curriculum_task()
            else:
                task = initial_task or "Generate a basic trading strategy"

            self.logger.info(f"Starting learning episode with task: {task}")

            # Create context for the task
            context = PromptContext(
                task_description=task,
                market_data=await self._generate_mock_market_data(),
                available_skills=[],
                market_regime="normal",
                risk_tolerance="moderate",
            )

            # Run iterative prompting to generate strategy
            result = await self.prompting_system.generate_strategy(context)

            # Update episode results
            episode_results["tasks_completed"] = 1
            episode_results["strategies_generated"] = 1
            episode_results["total_iterations"] = result.total_iterations
            episode_results["episode_duration"] = time.time() - start_time

            # Update curriculum with results if available
            if self.curriculum:
                await self._update_curriculum_with_results(task, result)

            # Count acquired skills
            if (
                result.best_performance.get("sharpe_ratio", 0)
                >= self.config.target_sharpe_ratio
            ):
                episode_results["skills_acquired"] = 1

            self.logger.info(f"Learning episode completed: {episode_results}")
            return episode_results

        except Exception as e:
            self.logger.error(f"Learning episode failed: {e}")
            episode_results["episode_duration"] = time.time() - start_time
            return episode_results

    async def _get_next_curriculum_task(self) -> str:
        """Get the next task from the curriculum."""
        try:
            if hasattr(self.curriculum, "generate_next_task"):
                task = self.curriculum.generate_next_task()
                return task.description if hasattr(task, "description") else str(task)
            else:
                return "Generate a momentum-based trading strategy"
        except Exception as e:
            self.logger.error(f"Error getting curriculum task: {e}")
            return "Generate a basic trading strategy"

    async def _generate_mock_market_data(self) -> Dict[str, Any]:
        """Generate mock market data for testing."""
        return {
            "symbol": "SPY",
            "price": 450.0,
            "volume": 1000000,
            "trend": "up",
            "volatility": 0.15,
            "rsi": 55,
            "macd": 0.5,
            "bollinger_position": 0.3,
        }

    async def _update_curriculum_with_results(
        self, task: str, result: RefinementResult
    ):
        """Update curriculum with learning results."""
        try:
            if hasattr(self.curriculum, "update_progress"):
                success = (
                    result.best_performance.get("sharpe_ratio", 0)
                    >= self.config.target_sharpe_ratio
                )
                self.curriculum.update_progress(task, success, result.best_performance)
                self.logger.info(
                    f"Updated curriculum with task results: success={success}"
                )
        except Exception as e:
            self.logger.error(f"Error updating curriculum: {e}")


# Legacy compatibility functions


def generate_strategy_legacy(
    context_dict: Dict[str, Any], config_dict: Dict[str, Any]
) -> Tuple[str, Dict[str, Any]]:
    """Legacy function for backward compatibility."""
    import asyncio

    # Convert legacy inputs to new format
    config = IterativePromptingConfig(**config_dict)
    context = PromptContext(
        task_description=context_dict.get("task_description", ""),
        market_data=context_dict.get("market_data", {}),
        available_skills=context_dict.get("available_skills", []),
        previous_attempts=context_dict.get("previous_attempts", []),
        performance_feedback=context_dict.get("performance_feedback", {}),
        error_messages=context_dict.get("error_messages", []),
    )

    # Run async function in sync context
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        prompting_system = IterativePrompting(config)
        result = loop.run_until_complete(prompting_system.generate_strategy(context))

        return result.final_code, {
            "iterations": result.total_iterations,
            "convergence_reason": result.convergence_reason.value,
            "performance_metrics": result.best_performance,
            "total_time": result.total_time,
            "tokens_used": result.llm_tokens_used,
        }
    finally:
        loop.close()
