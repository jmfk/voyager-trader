# 0015. Centralized LLM Service Architecture

## Status

Accepted

## Context

The current system has LLM integration scattered across different components, with each component handling its own LLM provider logic. This leads to:

- Code duplication across components
- Inconsistent error handling and retry logic
- Difficulty in managing multiple LLM providers
- No centralized configuration or monitoring
- Limited support for fallback mechanisms
- Hard to add new providers or local models

We need a centralized LLM service that provides a single point of integration for all AI/LLM interactions in the system.

## Decision

We will implement a centralized LLM Service with the following architecture:

### Core Components

1. **LLMService** - Main service interface providing unified API
2. **Provider Registry** - Manages available LLM providers
3. **Provider Implementations** - Concrete implementations for each LLM provider
4. **Configuration Management** - Centralized configuration for all providers
5. **Request Routing** - Intelligent routing based on model capabilities and availability
6. **Fallback Management** - Automatic failover between providers

### Provider Types

**Remote APIs:**
- OpenAI (GPT-3.5, GPT-4, GPT-4-turbo)
- Anthropic (Claude-2, Claude-3, Claude-3.5)
- Google (PaLM, Gemini)
- Azure OpenAI
- AWS Bedrock

**Local Models:**
- Ollama (Llama2, Llama3, CodeLlama, Mistral, etc.)
- LM Studio
- LocalAI
- Hugging Face Transformers

### Service Interface

```python
@dataclass
class LLMRequest:
    messages: List[Dict[str, str]]
    model: str
    temperature: float = 0.7
    max_tokens: int = 2000
    stream: bool = False
    provider: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass  
class LLMResponse:
    content: str
    model: str
    provider: str
    usage: Dict[str, int]
    metadata: Dict[str, Any] = field(default_factory=dict)

class LLMService:
    async def generate(self, request: LLMRequest) -> LLMResponse
    async def stream_generate(self, request: LLMRequest) -> AsyncIterator[LLMResponse]
    def get_available_models(self) -> List[str]
    def get_provider_status(self) -> Dict[str, bool]
```

### Configuration Structure

```yaml
llm_service:
  default_provider: "openai"
  fallback_chain: ["openai", "anthropic", "ollama"]

  providers:
    openai:
      api_key: "${OPENAI_API_KEY}"
      base_url: "https://api.openai.com/v1"
      models:
        - gpt-3.5-turbo
        - gpt-4
        - gpt-4-turbo
      rate_limits:
        requests_per_minute: 60
        tokens_per_minute: 40000

    anthropic:
      api_key: "${ANTHROPIC_API_KEY}"
      models:
        - claude-3-haiku
        - claude-3-sonnet
        - claude-3-opus

    ollama:
      base_url: "http://localhost:11434"
      models:
        - llama2
        - codellama
        - mistral
      timeout: 120
```

### Provider Abstract Base Class

```python
class LLMProvider(ABC):
    @abstractmethod
    async def generate(self, request: LLMRequest) -> LLMResponse

    @abstractmethod
    async def stream_generate(self, request: LLMRequest) -> AsyncIterator[LLMResponse]

    @abstractmethod
    def get_available_models(self) -> List[str]

    @abstractmethod
    async def health_check(self) -> bool
```

### Features

1. **Unified Interface** - Single API for all LLM interactions
2. **Provider Abstraction** - Easy to add new providers
3. **Intelligent Routing** - Route requests based on model capabilities
4. **Fallback Support** - Automatic failover on provider failure
5. **Rate Limiting** - Per-provider rate limiting and quotas
6. **Health Monitoring** - Real-time provider health checks
7. **Cost Tracking** - Track usage and costs across providers
8. **Caching** - Optional response caching for repeated requests
9. **Streaming Support** - Both batch and streaming responses
10. **Local Model Support** - Seamless integration with local models

## Implementation Plan

1. Create base LLMService and Provider interfaces
2. Implement OpenAI provider (migrate existing code)
3. Implement Ollama provider for local models
4. Add Anthropic provider
5. Implement provider registry and configuration
6. Add fallback and load balancing logic
7. Integrate with existing components (prompting.py, etc.)
8. Add monitoring and observability
9. Write comprehensive tests

## Integration Points

- **Iterative Prompting System** - Replace direct OpenAI calls
- **Curriculum System** - Use for task generation prompts
- **Skill Library** - Use for skill description generation
- **Core System** - Central point for all AI interactions

## Consequences

**Positive:**
- Single point of control for all LLM interactions
- Easy to add new providers and models
- Consistent error handling and retry logic
- Better monitoring and cost tracking
- Support for local models reduces API costs
- Improved reliability with fallback mechanisms

**Negative:**
- Additional abstraction layer complexity
- Migration effort for existing components
- Need to handle different provider capabilities
- Potential single point of failure (mitigated by design)

**Risks:**
- Provider API changes breaking integrations
- Local model performance and resource requirements
- Complexity in handling different response formats

**Mitigations:**
- Comprehensive provider health monitoring
- Extensive testing with mock providers
- Clear provider capability documentation
- Graceful degradation strategies
