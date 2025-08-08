# LLM Service Usage Guide

This guide explains how to use the centralized LLM service in VOYAGER-Trader for all AI/LLM interactions.

## Overview

The centralized LLM service provides a unified interface for interacting with multiple LLM providers including:
- **Remote APIs**: OpenAI, Anthropic Claude, Google, Azure OpenAI, AWS Bedrock
- **Local Models**: Ollama, LM Studio, LocalAI, Hugging Face

## Quick Start

### Basic Usage

```python
from src.voyager_trader.llm_service import chat_completion_create

# Simple chat completion
response = await chat_completion_create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful trading assistant."},
        {"role": "user", "content": "Analyze AAPL stock performance."}
    ],
    temperature=0.7
)

print(response.choices[0].message.content)
```

### Using Specific Providers

```python
from src.voyager_trader.llm_service import get_global_llm_client

client = get_global_llm_client()

# Force use of Ollama local model
response = await client.chat.create(
    model="llama2",
    messages=[{"role": "user", "content": "Hello"}],
    provider="ollama"
)

# Use Anthropic Claude
response = await client.chat.create(
    model="claude-3-haiku",
    messages=[{"role": "user", "content": "Analyze this data"}],
    provider="anthropic"
)
```

### Advanced Service Usage

```python
from src.voyager_trader.llm_service import (
    LLMService,
    LLMRequest,
    create_llm_service_from_config
)

# Create service with custom config
config = {
    "default_provider": "ollama",
    "fallback_chain": ["ollama", "openai", "anthropic"],
    "providers": {
        "ollama": {
            "enabled": True,
            "base_url": "http://localhost:11434",
            "models": ["llama2", "codellama"]
        }
    }
}

service = create_llm_service_from_config(config)

# Create detailed request
request = LLMRequest(
    messages=[{"role": "user", "content": "Generate trading strategy"}],
    model="llama2",
    temperature=0.8,
    max_tokens=1000,
    provider="ollama"
)

response = await service.generate(request)
```

## Configuration

### Environment Variables

Set these environment variables for provider authentication:

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# Anthropic Claude
export ANTHROPIC_API_KEY="sk-ant-..."

# Azure OpenAI (if using Azure)
export AZURE_OPENAI_API_KEY="..."
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
```

### Configuration File

Create `config/llm_service.yaml`:

```yaml
llm_service:
  default_provider: "openai"
  fallback_chain: ["openai", "anthropic", "ollama"]
  request_timeout: 60
  max_retries: 3

  providers:
    openai:
      enabled: true
      api_key: "${OPENAI_API_KEY}"
      base_url: "https://api.openai.com/v1"
      models:
        - gpt-3.5-turbo
        - gpt-4
        - gpt-4-turbo
      rate_limits:
        requests_per_minute: 60
        tokens_per_minute: 40000
      timeout: 60
      max_retries: 3

    anthropic:
      enabled: true
      api_key: "${ANTHROPIC_API_KEY}"
      models:
        - claude-3-haiku
        - claude-3-sonnet
        - claude-3-opus
        - claude-3-5-sonnet
      timeout: 60
      max_retries: 3

    ollama:
      enabled: true
      base_url: "http://localhost:11434"
      models:
        - llama2
        - llama3
        - codellama
        - mistral
        - gemma
      timeout: 120
      max_retries: 2

    azure_openai:
      enabled: false
      api_key: "${AZURE_OPENAI_API_KEY}"
      base_url: "${AZURE_OPENAI_ENDPOINT}"
      api_version: "2024-02-15-preview"
      models:
        - gpt-35-turbo
        - gpt-4
```

## Provider Setup

### OpenAI Setup

1. Get API key from https://platform.openai.com/api-keys
2. Set environment variable: `export OPENAI_API_KEY="sk-..."`
3. Available models: `gpt-3.5-turbo`, `gpt-4`, `gpt-4-turbo`

### Anthropic Claude Setup

1. Get API key from https://console.anthropic.com/
2. Set environment variable: `export ANTHROPIC_API_KEY="sk-ant-..."`
3. Available models: `claude-3-haiku`, `claude-3-sonnet`, `claude-3-opus`, `claude-3-5-sonnet`

### Ollama Local Models Setup

1. Install Ollama: https://ollama.ai/
2. Pull models:
   ```bash
   ollama pull llama2
   ollama pull codellama
   ollama pull mistral
   ```
3. Start Ollama server: `ollama serve` (usually runs on localhost:11434)

## Service Features

### Automatic Fallback

The service automatically tries providers in the fallback chain if one fails:

```python
# Will try: openai -> anthropic -> ollama
response = await chat_completion_create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello"}]
)
```

### Provider Health Monitoring

```python
from src.voyager_trader.llm_service import get_global_llm_service

service = get_global_llm_service()

# Check provider status
status = service.get_provider_status()
print(f"OpenAI healthy: {status['openai']['healthy']}")
print(f"Available models: {status['openai']['models']}")
print(f"Health check interval: {status['openai']['health_check_interval']}s")
print(f"Last health check: {status['openai']['last_health_check']}")

# Get all available models
models = service.get_available_models()
for provider, provider_models in models.items():
    print(f"{provider}: {provider_models}")
```

### Streaming Support

```python
from src.voyager_trader.llm_service import get_global_llm_client

client = get_global_llm_client()

# Stream response
async for chunk in await client.chat.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Tell a story"}],
    stream=True
):
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

## Integration Examples

### With Iterative Prompting

```python
from src.voyager_trader.prompting import IterativePrompting
from src.voyager_trader.llm_service import create_llm_service_from_config

# Create service with local model preference
config = {
    "default_provider": "ollama",
    "fallback_chain": ["ollama", "openai"]
}
service = create_llm_service_from_config(config)

# Use in iterative prompting
prompting = IterativePrompting(llm_service=service)
result = await prompting.generate_strategy(context)
```

### With Custom Provider

```python
from src.voyager_trader.llm_service import LLMProvider, LLMRequest, LLMResponse

class CustomProvider(LLMProvider):
    async def generate(self, request: LLMRequest) -> LLMResponse:
        # Your custom implementation
        pass

    async def health_check(self) -> bool:
        return True

    def get_available_models(self) -> List[str]:
        return ["custom-model"]

# Register custom provider
service = get_global_llm_service()
custom_provider = CustomProvider(ProviderConfig(name="custom"))
service.registry.register_provider(custom_provider)
```

## Error Handling

The service provides structured error handling:

```python
from src.voyager_trader.llm_service import (
    LLMError,
    ProviderError,
    RateLimitError,
    ModelNotAvailableError
)

try:
    response = await chat_completion_create(
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello"}]
    )
except RateLimitError as e:
    print(f"Rate limited: {e}")
    # Implement backoff strategy
except ModelNotAvailableError as e:
    print(f"Model not available: {e}")
    # Try different model
except ProviderError as e:
    print(f"Provider error: {e}")
    # Handle provider-specific error
except LLMError as e:
    print(f"General LLM error: {e}")
```

### Rate Limiting

The service supports standardized rate limiting across all providers:

```yaml
providers:
  openai:
    rate_limits:
      requests_per_minute: 60      # Maximum API requests per minute
      tokens_per_minute: 40000     # Maximum tokens per minute (input + estimated output)
```

**Rate Limiting Features:**
- **Request-based limiting**: Controls number of API calls per minute
- **Token-based limiting**: Controls token usage including input and estimated output
- **Per-provider configuration**: Each provider can have different limits
- **Automatic enforcement**: Service automatically waits when limits are reached
- **Token estimation**: Uses 4 characters ≈ 1 token approximation for rate limiting

### Security Features

The LLM service includes built-in security measures:

**Input Sanitization (Ollama Provider):**
- Removes role indicators that could confuse the model (`System:`, `User:`, etc.)
- Limits message length to prevent extremely long prompts (8000 chars max)
- Normalizes excessive newlines that could break prompt structure
- Provides safe defaults for empty or invalid messages
- Logs warnings for unknown message roles

This prevents prompt injection attacks where malicious input tries to manipulate the model by including fake role indicators or system instructions.

### Session Lifecycle Management

All providers support proper session lifecycle management with async context managers:

```python
from src.voyager_trader.llm_service import LLMService, create_default_llm_service

# Recommended: Use async context manager for automatic cleanup
async with create_default_llm_service() as service:
    response = await service.generate(LLMRequest(
        messages=[{"role": "user", "content": "Hello"}],
        model="gpt-3.5-turbo"
    ))
    print(response.content)
# Service and all provider sessions are automatically closed

# Alternative: Manual lifecycle management
service = create_default_llm_service()
try:
    response = await service.generate(request)
finally:
    await service.close()  # Ensure proper cleanup
```

**Session Management Features:**
- Automatic connection pooling and reuse
- Graceful session cleanup with `__aexit__` and `close()` methods
- Resource leak detection with `__del__` warnings
- Per-provider session isolation

### Configuration Validation

The service validates all configuration on startup to catch errors early:

```python
from src.voyager_trader.llm_service import create_llm_service_from_config, LLMError

config = {
    "default_provider": "openai",
    "providers": {
        "openai": {
            "enabled": True,
            "timeout": -5,  # Invalid: negative timeout
            "api_key": "sk-..."
        }
    }
}

try:
    service = create_llm_service_from_config(config)
except LLMError as e:
    print(f"Configuration error: {e}")
    # Fix the configuration and retry
```

**Validation Checks:**
- Provider existence and enablement
- API key availability (config or environment)
- Timeout and retry limits within reasonable bounds
- Valid URL formats for base_url settings
- Rate limiting configuration correctness
- Provider-specific requirements (e.g., Ollama timeout minimums)
- Fallback chain provider availability

## Best Practices

1. **Use Environment Variables**: Store API keys securely in environment variables
2. **Configure Fallbacks**: Set up fallback chains for reliability
3. **Local Models for Development**: Use Ollama for development and testing
4. **Monitor Usage**: Regularly check provider status and health
5. **Set Appropriate Rate Limits**: Configure limits based on your API plan and usage patterns
5. **Handle Errors**: Implement proper error handling and retry logic
6. **Rate Limiting**: Respect provider rate limits and quotas
7. **Cost Optimization**: Use local models when possible to reduce API costs

## Troubleshooting

### Common Issues

1. **Provider Not Available**
   ```
   ProviderError: Provider 'openai' not available
   ```
   - Check API key is set correctly
   - Verify internet connection
   - Check provider status

2. **Model Not Found**
   ```
   ModelNotAvailableError: Model 'gpt-5' not found
   ```
   - Use `service.get_available_models()` to see available models
   - Check provider configuration

3. **Ollama Connection Issues**
   ```
   ProviderError: Ollama API error (500)
   ```
   - Ensure Ollama is running: `ollama serve`
   - Check if model is pulled: `ollama list`
   - Verify base_url in config

4. **Rate Limiting**
   ```
   RateLimitError: Rate limit exceeded
   ```
   - Implement exponential backoff
   - Use local models as fallback
   - Check your API quota

### Debug Mode

Enable debug logging:

```python
import logging
logging.getLogger("src.voyager_trader.llm_service").setLevel(logging.DEBUG)
```

This will show detailed request/response information and provider selection logic.
