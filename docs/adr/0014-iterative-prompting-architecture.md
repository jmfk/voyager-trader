# 0014. Iterative Prompting Architecture

## Status

Accepted

## Context

The VOYAGER-Trader system requires an iterative prompting mechanism that enables autonomous strategy generation and improvement through continuous LLM interaction. This system must integrate with the existing curriculum system and skill library while ensuring safe execution of generated code.

Key requirements:
- LLM-driven strategy generation with iterative refinement
- Sandboxed execution environment for generated strategies
- Comprehensive performance evaluation and feedback synthesis
- Safety monitoring to prevent harmful or invalid strategies
- Integration with existing curriculum and skill library systems
- Support for multiple LLM providers (OpenAI, Anthropic, etc.)

## Decision

We will implement a comprehensive iterative prompting system with six core components:

### 1. LLM Integration Layer
- **LLMProvider**: Abstract base class for different LLM services
- **OpenAIProvider**: OpenAI GPT integration with rate limiting
- **AnthropicProvider**: Claude integration (future extension)
- **PromptTemplate**: Structured prompt generation with context injection
- **ResponseParser**: Robust code extraction from LLM responses

### 2. Code Execution Environment
- **SandboxedExecutor**: Isolated Python execution with resource limits
- **ExecutionContext**: Managed environment with market data simulation
- **ResourceMonitor**: CPU, memory, and execution time tracking
- **SecurityValidator**: AST-based code analysis for dangerous patterns

### 3. Performance Evaluation System
- **PerformanceEvaluator**: Multi-metric strategy assessment
- **BacktestEngine**: Historical market data testing
- **RiskAnalyzer**: Risk-adjusted performance metrics
- **ComparisonEngine**: Strategy performance comparison

### 4. Feedback Synthesis
- **FeedbackSynthesizer**: Converts results into actionable improvement prompts
- **ErrorAnalyzer**: Categorizes and prioritizes execution errors
- **PerformanceAnalyzer**: Identifies performance bottlenecks and improvements
- **ContextualFeedback**: Market condition-aware feedback generation

### 5. Strategy Refinement Engine
- **StrategyRefiner**: Orchestrates the iterative improvement process
- **IterationManager**: Controls refinement cycles and stopping criteria
- **ProgressTracker**: Monitors improvement across iterations
- **ConvergenceDetector**: Identifies when further refinement is unlikely

### 6. Safety Monitoring
- **SafetyMonitor**: Real-time safety validation during execution
- **RiskLimiter**: Prevents strategies from exceeding risk thresholds
- **CodeAnalyzer**: Static analysis for dangerous code patterns
- **ExecutionGuard**: Runtime protection against infinite loops and resource exhaustion

### Integration Points
- **CurriculumIntegration**: Receives tasks from curriculum system
- **SkillLibraryIntegration**: Stores successful strategies as skills
- **ObservabilityIntegration**: Comprehensive logging and metrics
- **ConfigurationManagement**: Centralized configuration for all components

## Architecture Flow

```
Curriculum Task → LLM Integration → Code Generation → Safety Validation →
Sandboxed Execution → Performance Evaluation → Feedback Synthesis →
Strategy Refinement → [Iteration Loop] → Skill Library Storage
```

## Implementation Strategy

1. **Phase 1**: Core LLM integration and basic sandboxed execution
2. **Phase 2**: Performance evaluation and feedback synthesis
3. **Phase 3**: Advanced safety monitoring and refinement engine
4. **Phase 4**: Full curriculum/skill library integration

## Configuration Structure

```python
@dataclass
class IterativePromptingConfig:
    # LLM Configuration
    llm_provider: str = "openai"
    llm_model: str = "gpt-4"
    max_tokens: int = 2000
    temperature: float = 0.7

    # Execution Configuration
    execution_timeout: int = 30
    memory_limit_mb: int = 512
    max_cpu_percent: float = 50.0

    # Iteration Configuration
    max_iterations: int = 10
    convergence_threshold: float = 0.01
    min_performance_improvement: float = 0.05

    # Safety Configuration
    enable_safety_checks: bool = True
    allowed_imports: List[str] = field(default_factory=list)
    forbidden_patterns: List[str] = field(default_factory=list)
```

## Consequences

**Positive:**
- Autonomous strategy generation and improvement
- Safe execution environment prevents system damage
- Comprehensive performance tracking enables data-driven refinement
- Modular architecture allows for easy extension and testing
- Integration with existing systems maintains consistency

**Negative:**
- Additional complexity in system architecture
- LLM API costs for iterative refinement
- Potential latency in strategy generation
- Requires robust error handling for LLM failures

**Risks:**
- LLM-generated code quality and consistency
- API rate limiting and availability issues
- Resource exhaustion from poorly optimized strategies
- Security vulnerabilities in generated code

**Mitigations:**
- Comprehensive testing with mock LLM responses
- Fallback mechanisms for API failures
- Strict resource limits and monitoring
- Multi-layer security validation
