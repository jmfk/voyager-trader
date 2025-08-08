"""Centralized LLM Service for VOYAGER-Trader.

Provides unified interface for all LLM interactions including:
- Remote APIs (OpenAI, Anthropic, etc.)
- Local models (Ollama, LM Studio, etc.)
- Fallback mechanisms and load balancing
- Rate limiting and cost tracking
- Health monitoring and observability

Follows architecture defined in ADR-0015.
"""

import asyncio
import json
import logging
import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple, Union
from urllib.parse import urljoin

import aiohttp

try:
    import openai

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic

    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


class LLMProvider(str, Enum):
    """Available LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    AZURE_OPENAI = "azure_openai"
    GOOGLE = "google"
    LOCAL_AI = "local_ai"


class LLMError(Exception):
    """Base exception for LLM service errors."""


class ProviderError(LLMError):
    """Exception raised by LLM provider."""

    def __init__(
        self, message: str, provider: str, model: str = "", status_code: int = 0
    ):
        super().__init__(message)
        self.provider = provider
        self.model = model
        self.status_code = status_code


class RateLimitError(ProviderError):
    """Exception raised when rate limit is exceeded."""


class ModelNotAvailableError(ProviderError):
    """Exception raised when requested model is not available."""


@dataclass
class LLMRequest:
    """Request object for LLM service."""

    messages: List[Dict[str, str]]
    model: str = "gpt-3.5-turbo"
    temperature: float = 0.7
    max_tokens: int = 2000
    stream: bool = False
    provider: Optional[str] = None
    system_prompt: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Process request after initialization."""
        if self.system_prompt and self.messages:
            # Ensure system prompt is first message
            if not self.messages[0].get("role") == "system":
                self.messages.insert(
                    0, {"role": "system", "content": self.system_prompt}
                )
        elif self.system_prompt:
            self.messages = [{"role": "system", "content": self.system_prompt}]


@dataclass
class LLMResponse:
    """Response object from LLM service."""

    content: str
    model: str
    provider: str
    usage: Dict[str, int] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    finish_reason: str = "stop"
    created_at: float = field(default_factory=time.time)


@dataclass
class ProviderConfig:
    """Configuration for an LLM provider."""

    name: str
    enabled: bool = True
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    models: List[str] = field(default_factory=list)
    rate_limits: Dict[str, int] = field(default_factory=dict)
    timeout: int = 60
    max_retries: int = 3
    retry_delay: float = 1.0
    health_check_interval: int = 300  # 5 minutes
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, config: ProviderConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{config.name}")
        self._last_health_check = 0.0
        self._is_healthy = True
        self._request_count = 0
        self._token_count = 0
        self._last_request_time = time.time()
        self._token_window_start = time.time()

    @abstractmethod
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate response from LLM."""
        pass

    @abstractmethod
    async def stream_generate(self, request: LLMRequest) -> AsyncIterator[LLMResponse]:
        """Generate streaming response from LLM."""
        pass

    @abstractmethod
    def get_available_models(self) -> List[str]:
        """Get list of available models for this provider."""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if provider is healthy."""
        pass

    async def _enforce_rate_limits(self, estimated_tokens: int = 0) -> None:
        """Enforce rate limits for the provider.

        Args:
            estimated_tokens: Estimated tokens for request (for token-based limiting)
        """
        if not self.config.rate_limits:
            return

        current_time = time.time()
        time_window = 60.0  # 1 minute

        # Check requests per minute limit
        requests_per_minute = self.config.rate_limits.get("requests_per_minute", 0)
        if requests_per_minute > 0:
            time_since_last = current_time - self._last_request_time
            if (
                self._request_count >= requests_per_minute
                and time_since_last < time_window
            ):
                wait_time = time_window - time_since_last
                self.logger.info(
                    f"Request rate limit reached ({requests_per_minute}/min), "
                    f"waiting {wait_time:.2f} seconds"
                )
                await asyncio.sleep(wait_time)
                self._request_count = 0
                self._last_request_time = current_time

        # Check tokens per minute limit
        tokens_per_minute = self.config.rate_limits.get("tokens_per_minute", 0)
        if tokens_per_minute > 0 and estimated_tokens > 0:
            time_since_window_start = current_time - self._token_window_start

            # Reset token count if window has passed
            if time_since_window_start >= time_window:
                self._token_count = 0
                self._token_window_start = current_time
                time_since_window_start = 0

            # Check if adding this request would exceed token limit
            if self._token_count + estimated_tokens > tokens_per_minute:
                wait_time = time_window - time_since_window_start
                self.logger.info(
                    f"Token rate limit reached ({tokens_per_minute}/min), "
                    f"waiting {wait_time:.2f} seconds"
                )
                await asyncio.sleep(wait_time)
                self._token_count = 0
                self._token_window_start = current_time

        self._request_count += 1
        self._token_count += estimated_tokens
        self._last_request_time = current_time

    def _estimate_tokens(self, request: LLMRequest) -> int:
        """Estimate token count for rate limiting.

        Simple estimation: 4 characters = 1 token (rough approximation).
        More accurate implementations could use tokenizer libraries.
        """
        total_chars = 0
        for message in request.messages:
            content = message.get("content", "")
            total_chars += len(content)

        # Add estimated response tokens (max_tokens or a reasonable default)
        response_tokens = min(request.max_tokens, 500)  # Conservative estimate

        # 4 chars per token is a rough approximation
        estimated_tokens = (total_chars // 4) + response_tokens

        return max(estimated_tokens, 10)  # Minimum 10 tokens per request

    def _validate_request(self, request: LLMRequest) -> None:
        """Validate request parameters."""
        if request.model not in self.get_available_models():
            raise ModelNotAvailableError(
                f"Model {request.model} not available for provider {self.config.name}",
                provider=self.config.name,
                model=request.model,
            )

    async def _retry_on_failure(self, func, *args, **kwargs):
        """Retry function on failure with exponential backoff."""
        last_exception = None

        for attempt in range(self.config.max_retries):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt == self.config.max_retries - 1:
                    break

                wait_time = self.config.retry_delay * (2**attempt)
                self.logger.warning(
                    f"Attempt {attempt + 1} failed: {str(e)}, "
                    f"retrying in {wait_time:.2f}s"
                )
                await asyncio.sleep(wait_time)

        raise last_exception


class OpenAIProvider(BaseLLMProvider):
    """OpenAI LLM provider implementation."""

    def __init__(self, config: ProviderConfig):
        super().__init__(config)

        if not OPENAI_AVAILABLE:
            raise ProviderError("OpenAI library not available", provider=config.name)

        api_key = config.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ProviderError("OpenAI API key not provided", provider=config.name)

        self.client = openai.AsyncOpenAI(api_key=api_key, base_url=config.base_url)

        if not config.models:
            config.models = [
                "gpt-3.5-turbo",
                "gpt-3.5-turbo-16k",
                "gpt-4",
                "gpt-4-turbo",
                "gpt-4o",
            ]

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with cleanup."""
        await self.close()

    async def close(self):
        """Close OpenAI client connections."""
        if hasattr(self.client, "close"):
            try:
                await self.client.close()
            except Exception as e:
                self.logger.error(f"Error closing OpenAI client: {e}")

    def __del__(self):
        """Cleanup on deletion."""
        if hasattr(self, "client") and self.client:
            import warnings

            warnings.warn(
                "OpenAIProvider client was not properly closed. "
                "Use async context manager or call close() explicitly.",
                ResourceWarning,
                stacklevel=2,
            )

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate response from OpenAI."""
        self._validate_request(request)
        estimated_tokens = self._estimate_tokens(request)
        await self._enforce_rate_limits(estimated_tokens)

        try:
            response = await self._retry_on_failure(
                self.client.chat.completions.create,
                model=request.model,
                messages=request.messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stream=False,
            )

            return LLMResponse(
                content=response.choices[0].message.content,
                model=response.model,
                provider=self.config.name,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
                finish_reason=response.choices[0].finish_reason,
                metadata={"request_id": response.id},
            )

        except Exception as e:
            if hasattr(e, "status_code") and e.status_code == 429:
                raise RateLimitError(
                    "OpenAI rate limit exceeded",
                    provider=self.config.name,
                    model=request.model,
                    status_code=e.status_code,
                )
            raise ProviderError(
                f"OpenAI API error: {str(e)}",
                provider=self.config.name,
                model=request.model,
            )

    async def stream_generate(self, request: LLMRequest) -> AsyncIterator[LLMResponse]:
        """Generate streaming response from OpenAI."""
        self._validate_request(request)
        estimated_tokens = self._estimate_tokens(request)
        await self._enforce_rate_limits(estimated_tokens)

        try:
            stream = await self._retry_on_failure(
                self.client.chat.completions.create,
                model=request.model,
                messages=request.messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stream=True,
            )

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield LLMResponse(
                        content=chunk.choices[0].delta.content,
                        model=request.model,
                        provider=self.config.name,
                        finish_reason=chunk.choices[0].finish_reason or "continue",
                        metadata={"chunk_id": chunk.id},
                    )

        except Exception as e:
            raise ProviderError(
                f"OpenAI streaming error: {str(e)}",
                provider=self.config.name,
                model=request.model,
            )

    def get_available_models(self) -> List[str]:
        """Get available OpenAI models."""
        return self.config.models.copy()

    async def health_check(self) -> bool:
        """Check OpenAI API health."""
        try:
            # Simple test request
            response = await self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1,
            )
            return bool(response)
        except Exception as e:
            self.logger.error(f"OpenAI health check failed: {e}")
            return False


class OllamaProvider(BaseLLMProvider):
    """Ollama local model provider implementation."""

    def __init__(self, config: ProviderConfig):
        super().__init__(config)

        self.base_url = config.base_url or "http://localhost:11434"
        self.session = None

        if not config.models:
            config.models = [
                "llama2",
                "llama2:7b",
                "llama2:13b",
                "llama2:70b",
                "codellama",
                "codellama:7b",
                "codellama:13b",
                "codellama:34b",
                "mistral",
                "mistral:7b",
                "mistral:instruct",
                "llama3",
                "llama3:8b",
                "llama3:70b",
            ]

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with session cleanup."""
        await self.close()

    async def close(self):
        """Close aiohttp session."""
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None

    def __del__(self):
        """Cleanup on deletion."""
        if hasattr(self, "session") and self.session and not self.session.closed:
            import warnings

            warnings.warn(
                "OllamaProvider session was not properly closed. "
                "Use async context manager or call close() explicitly.",
                ResourceWarning,
                stacklevel=2,
            )

    async def _get_session(self):
        """Get or create aiohttp session."""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate response from Ollama."""
        self._validate_request(request)
        estimated_tokens = self._estimate_tokens(request)
        await self._enforce_rate_limits(estimated_tokens)

        session = await self._get_session()

        # Convert messages to Ollama format
        prompt = self._convert_messages_to_prompt(request.messages)

        payload = {
            "model": request.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": request.temperature,
                "num_predict": request.max_tokens,
            },
        }

        try:
            async with session.post(
                urljoin(self.base_url, "/api/generate"), json=payload
            ) as response:
                if response.status != 200:
                    text = await response.text()
                    raise ProviderError(
                        f"Ollama API error: {response.status} - {text}",
                        provider=self.config.name,
                        model=request.model,
                        status_code=response.status,
                    )

                result = await response.json()

                return LLMResponse(
                    content=result.get("response", ""),
                    model=request.model,
                    provider=self.config.name,
                    usage={
                        "prompt_tokens": result.get("prompt_eval_count", 0),
                        "completion_tokens": result.get("eval_count", 0),
                        "total_tokens": result.get("prompt_eval_count", 0)
                        + result.get("eval_count", 0),
                    },
                    metadata={
                        "eval_duration": result.get("eval_duration", 0),
                        "load_duration": result.get("load_duration", 0),
                    },
                )

        except aiohttp.ClientError as e:
            raise ProviderError(
                f"Ollama connection error: {str(e)}",
                provider=self.config.name,
                model=request.model,
            )

    async def stream_generate(self, request: LLMRequest) -> AsyncIterator[LLMResponse]:
        """Generate streaming response from Ollama."""
        self._validate_request(request)
        estimated_tokens = self._estimate_tokens(request)
        await self._enforce_rate_limits(estimated_tokens)

        session = await self._get_session()

        prompt = self._convert_messages_to_prompt(request.messages)

        payload = {
            "model": request.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": request.temperature,
                "num_predict": request.max_tokens,
            },
        }

        try:
            async with session.post(
                urljoin(self.base_url, "/api/generate"), json=payload
            ) as response:
                if response.status != 200:
                    text = await response.text()
                    raise ProviderError(
                        f"Ollama streaming error: {response.status} - {text}",
                        provider=self.config.name,
                        model=request.model,
                        status_code=response.status,
                    )

                async for line in response.content:
                    if line:
                        try:
                            chunk = json.loads(line.decode())
                            if chunk.get("response"):
                                yield LLMResponse(
                                    content=chunk["response"],
                                    model=request.model,
                                    provider=self.config.name,
                                    finish_reason="stop"
                                    if chunk.get("done")
                                    else "continue",
                                )
                        except json.JSONDecodeError:
                            continue

        except aiohttp.ClientError as e:
            raise ProviderError(
                f"Ollama streaming connection error: {str(e)}",
                provider=self.config.name,
                model=request.model,
            )

    def _sanitize_content(self, content: str) -> str:
        """Sanitize message content to prevent prompt injection attacks.

        Args:
            content: Raw message content

        Returns:
            Sanitized content safe for use in prompts
        """
        if not content:
            return ""

        # Remove potential prompt injection patterns
        sanitized = content

        # Remove role indicators that could confuse the model
        role_patterns = [
            "System:",
            "User:",
            "Assistant:",
            "Human:",
            "AI:",
            "Bot:",
        ]

        for pattern in role_patterns:
            # Case-insensitive replacement, but preserve the original text structure
            sanitized = re.sub(
                re.escape(pattern),
                pattern.replace(":", ""),
                sanitized,
                flags=re.IGNORECASE,
            )

        # Remove excessive newlines that could break prompt structure
        sanitized = re.sub(r"\n{3,}", "\n\n", sanitized)

        # Trim whitespace
        sanitized = sanitized.strip()

        # Limit content length to prevent extremely long prompts
        max_length = 8000  # Reasonable limit for single message
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length] + "...[truncated]"

        return sanitized

    def _convert_messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert OpenAI-style messages to Ollama format with sanitization."""
        prompt_parts = []

        for message in messages:
            role = message.get("role", "user")
            raw_content = message.get("content", "")

            # Sanitize content to prevent prompt injection
            content = self._sanitize_content(raw_content)

            if not content:  # Skip empty messages
                continue

            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
            else:
                # For unknown roles, default to user but log a warning
                self.logger.warning(f"Unknown message role '{role}', treating as user")
                prompt_parts.append(f"User: {content}")

        if not prompt_parts:  # If all messages were empty/invalid
            prompt_parts.append("User: Hello")  # Safe default

        return "\n\n".join(prompt_parts) + "\n\nAssistant:"

    def get_available_models(self) -> List[str]:
        """Get available Ollama models."""
        return self.config.models.copy()

    async def health_check(self) -> bool:
        """Check Ollama service health."""
        try:
            session = await self._get_session()
            async with session.get(urljoin(self.base_url, "/api/tags")) as response:
                return response.status == 200
        except Exception as e:
            self.logger.error(f"Ollama health check failed: {e}")
            return False


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude provider implementation."""

    def __init__(self, config: ProviderConfig):
        super().__init__(config)

        if not ANTHROPIC_AVAILABLE:
            raise ProviderError("Anthropic library not available", provider=config.name)

        api_key = config.api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ProviderError("Anthropic API key not provided", provider=config.name)

        self.client = anthropic.AsyncAnthropic(api_key=api_key)

        if not config.models:
            config.models = [
                "claude-3-haiku-20240307",
                "claude-3-sonnet-20240229",
                "claude-3-opus-20240229",
                "claude-3-5-sonnet-20241022",
            ]

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with cleanup."""
        await self.close()

    async def close(self):
        """Close Anthropic client connections."""
        if hasattr(self.client, "close"):
            try:
                await self.client.close()
            except Exception as e:
                self.logger.error(f"Error closing Anthropic client: {e}")

    def __del__(self):
        """Cleanup on deletion."""
        if hasattr(self, "client") and self.client:
            import warnings

            warnings.warn(
                "AnthropicProvider client was not properly closed. "
                "Use async context manager or call close() explicitly.",
                ResourceWarning,
                stacklevel=2,
            )

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate response from Anthropic."""
        self._validate_request(request)
        estimated_tokens = self._estimate_tokens(request)
        await self._enforce_rate_limits(estimated_tokens)

        # Convert messages format for Anthropic
        system_prompt = None
        messages = []

        for msg in request.messages:
            if msg.get("role") == "system":
                system_prompt = msg.get("content")
            else:
                messages.append(
                    {"role": msg.get("role"), "content": msg.get("content")}
                )

        try:
            response = await self._retry_on_failure(
                self.client.messages.create,
                model=request.model,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                system=system_prompt,
                messages=messages,
            )

            return LLMResponse(
                content=response.content[0].text if response.content else "",
                model=response.model,
                provider=self.config.name,
                usage={
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens
                    + response.usage.output_tokens,
                },
                finish_reason=response.stop_reason,
                metadata={"request_id": response.id},
            )

        except Exception as e:
            if hasattr(e, "status_code") and e.status_code == 429:
                raise RateLimitError(
                    "Anthropic rate limit exceeded",
                    provider=self.config.name,
                    model=request.model,
                    status_code=e.status_code,
                )
            raise ProviderError(
                f"Anthropic API error: {str(e)}",
                provider=self.config.name,
                model=request.model,
            )

    async def stream_generate(self, request: LLMRequest) -> AsyncIterator[LLMResponse]:
        """Generate streaming response from Anthropic."""
        self._validate_request(request)
        estimated_tokens = self._estimate_tokens(request)
        await self._enforce_rate_limits(estimated_tokens)

        # Convert messages format for Anthropic
        system_prompt = None
        messages = []

        for msg in request.messages:
            if msg.get("role") == "system":
                system_prompt = msg.get("content")
            else:
                messages.append(
                    {"role": msg.get("role"), "content": msg.get("content")}
                )

        try:
            stream = await self._retry_on_failure(
                self.client.messages.stream,
                model=request.model,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                system=system_prompt,
                messages=messages,
            )

            async with stream as stream_response:
                async for event in stream_response:
                    if event.type == "content_block_delta":
                        yield LLMResponse(
                            content=event.delta.text,
                            model=request.model,
                            provider=self.config.name,
                            finish_reason="continue",
                        )
                    elif event.type == "message_stop":
                        yield LLMResponse(
                            content="",
                            model=request.model,
                            provider=self.config.name,
                            finish_reason="stop",
                        )

        except Exception as e:
            raise ProviderError(
                f"Anthropic streaming error: {str(e)}",
                provider=self.config.name,
                model=request.model,
            )

    def get_available_models(self) -> List[str]:
        """Get available Anthropic models."""
        return self.config.models.copy()

    async def health_check(self) -> bool:
        """Check Anthropic API health."""
        try:
            # Simple test request
            response = await self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=1,
                messages=[{"role": "user", "content": "test"}],
            )
            return bool(response)
        except Exception as e:
            self.logger.error(f"Anthropic health check failed: {e}")
            return False


class ProviderRegistry:
    """Registry for managing LLM providers."""

    def __init__(self):
        self.providers: Dict[str, BaseLLMProvider] = {}
        self.logger = logging.getLogger(__name__)

    def register_provider(self, provider: BaseLLMProvider) -> None:
        """Register a new provider."""
        self.providers[provider.config.name] = provider
        self.logger.info(f"Registered provider: {provider.config.name}")

    def get_provider(self, name: str) -> Optional[BaseLLMProvider]:
        """Get provider by name."""
        return self.providers.get(name)

    def get_available_providers(self) -> List[str]:
        """Get list of available provider names."""
        return [
            name for name, provider in self.providers.items() if provider.config.enabled
        ]

    def get_provider_for_model(self, model: str) -> Optional[BaseLLMProvider]:
        """Find provider that supports the given model."""
        for provider in self.providers.values():
            if provider.config.enabled and model in provider.get_available_models():
                return provider
        return None

    async def health_check_all(self) -> Dict[str, bool]:
        """Run health check on all providers."""
        results = {}
        tasks = []

        for name, provider in self.providers.items():
            if provider.config.enabled:
                tasks.append(self._check_provider_health(name, provider))

        if tasks:
            results_list = await asyncio.gather(*tasks, return_exceptions=True)
            for name, result in results_list:
                results[name] = result

        return results

    async def _check_provider_health(
        self, name: str, provider: BaseLLMProvider
    ) -> Tuple[str, bool]:
        """Check health of a single provider."""
        try:
            health = await provider.health_check()
            return name, health
        except Exception as e:
            self.logger.error(f"Health check failed for {name}: {e}")
            return name, False


@dataclass
class LLMServiceConfig:
    """Configuration for LLM service."""

    default_provider: str = "openai"
    fallback_chain: List[str] = field(
        default_factory=lambda: ["openai", "anthropic", "ollama"]
    )
    enable_fallback: bool = True
    enable_caching: bool = False
    cache_ttl: int = 3600
    health_check_interval: int = 300
    providers: Dict[str, ProviderConfig] = field(default_factory=dict)


class LLMService:
    """Centralized LLM service with multi-provider support."""

    def __init__(self, config: LLMServiceConfig):
        self.config = config
        self.registry = ProviderRegistry()
        self.logger = logging.getLogger(__name__)
        self._initialize_providers()
        self._provider_health: Dict[str, bool] = {}
        self._provider_last_health_check: Dict[str, float] = {}

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with cleanup."""
        await self.close()

    async def close(self):
        """Close all provider sessions."""
        for provider in self.registry.providers.values():
            if hasattr(provider, "close"):
                try:
                    await provider.close()
                except Exception as e:
                    self.logger.error(
                        f"Error closing provider {provider.config.name}: {e}"
                    )

    def __del__(self):
        """Cleanup on deletion."""
        if hasattr(self, "registry") and self.registry:
            import warnings

            warnings.warn(
                "LLMService was not properly closed. "
                "Use async context manager or call close() explicitly.",
                ResourceWarning,
                stacklevel=2,
            )

    def _initialize_providers(self) -> None:
        """Initialize all configured providers."""
        for name, provider_config in self.config.providers.items():
            if not provider_config.enabled:
                continue

            try:
                provider = self._create_provider(provider_config)
                self.registry.register_provider(provider)
                self.logger.info(f"Initialized provider: {name}")
            except Exception as e:
                self.logger.error(f"Failed to initialize provider {name}: {e}")

    def _create_provider(self, config: ProviderConfig) -> BaseLLMProvider:
        """Create provider instance based on config."""
        provider_map = {
            "openai": OpenAIProvider,
            "anthropic": AnthropicProvider,
            "ollama": OllamaProvider,
        }

        provider_class = provider_map.get(config.name)
        if not provider_class:
            raise LLMError(f"Unknown provider type: {config.name}")

        return provider_class(config)

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate response using the best available provider."""
        await self._ensure_health_check()

        # Determine provider to use
        if request.provider:
            provider = self.registry.get_provider(request.provider)
            if not provider or not provider.config.enabled:
                raise ProviderError(
                    f"Requested provider {request.provider} not available",
                    provider=request.provider or "unknown",
                )
        else:
            provider = self._select_provider(request.model)

        if not provider:
            raise ProviderError(
                f"No provider available for model {request.model}",
                provider="none",
                model=request.model,
            )

        # Attempt generation with fallback
        return await self._generate_with_fallback(request, provider)

    async def stream_generate(self, request: LLMRequest) -> AsyncIterator[LLMResponse]:
        """Generate streaming response using the best available provider."""
        await self._ensure_health_check()

        # Determine provider to use
        if request.provider:
            provider = self.registry.get_provider(request.provider)
            if not provider or not provider.config.enabled:
                raise ProviderError(
                    f"Requested provider {request.provider} not available",
                    provider=request.provider or "unknown",
                )
        else:
            provider = self._select_provider(request.model)

        if not provider:
            raise ProviderError(
                f"No provider available for model {request.model}",
                provider="none",
                model=request.model,
            )

        # Stream with fallback support
        async for response in provider.stream_generate(request):
            yield response

    def get_available_models(self) -> Dict[str, List[str]]:
        """Get all available models grouped by provider."""
        models = {}
        for name in self.registry.get_available_providers():
            provider = self.registry.get_provider(name)
            if provider:
                models[name] = provider.get_available_models()
        return models

    def get_provider_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all providers."""
        status = {}
        for name in self.registry.get_available_providers():
            provider = self.registry.get_provider(name)
            if provider:
                status[name] = {
                    "enabled": provider.config.enabled,
                    "healthy": self._provider_health.get(name, False),
                    "models": provider.get_available_models(),
                    "last_health_check": self._provider_last_health_check.get(
                        name, 0.0
                    ),
                    "health_check_interval": provider.config.health_check_interval,
                }
        return status

    async def health_check(self) -> Dict[str, bool]:
        """Run health check on all providers."""
        current_time = time.time()
        health_results = await self.registry.health_check_all()
        self._provider_health = health_results

        # Update last check time for all providers
        for provider_name in health_results.keys():
            self._provider_last_health_check[provider_name] = current_time

        return health_results

    def _select_provider(self, model: str) -> Optional[BaseLLMProvider]:
        """Select best provider for the given model."""
        # First try model-specific provider
        provider = self.registry.get_provider_for_model(model)
        if provider and self._provider_health.get(provider.config.name, True):
            return provider

        # Fallback to default provider chain
        for provider_name in self.config.fallback_chain:
            provider = self.registry.get_provider(provider_name)
            if (
                provider
                and provider.config.enabled
                and self._provider_health.get(provider_name, True)
                and model in provider.get_available_models()
            ):
                return provider

        return None

    async def _generate_with_fallback(
        self, request: LLMRequest, primary_provider: BaseLLMProvider
    ) -> LLMResponse:
        """Generate response with fallback support."""
        providers_to_try = [primary_provider]

        # Add fallback providers if enabled
        if self.config.enable_fallback:
            for provider_name in self.config.fallback_chain:
                if provider_name != primary_provider.config.name:
                    provider = self.registry.get_provider(provider_name)
                    if (
                        provider
                        and provider.config.enabled
                        and request.model in provider.get_available_models()
                    ):
                        providers_to_try.append(provider)

        last_error = None
        for provider in providers_to_try:
            try:
                self.logger.debug(f"Trying provider: {provider.config.name}")
                return await provider.generate(request)
            except Exception as e:
                self.logger.warning(f"Provider {provider.config.name} failed: {e}")
                last_error = e
                # Mark provider as potentially unhealthy
                self._provider_health[provider.config.name] = False
                continue

        # All providers failed
        raise last_error or LLMError("All providers failed")

    async def _ensure_health_check(self) -> None:
        """Ensure health checks are up to date for all providers."""
        current_time = time.time()

        for name, provider in self.registry.providers.items():
            if not provider.config.enabled:
                continue

            last_check = self._provider_last_health_check.get(name, 0.0)
            interval = provider.config.health_check_interval

            if current_time - last_check > interval:
                try:
                    health = await provider.health_check()
                    self._provider_health[name] = health
                    self._provider_last_health_check[name] = current_time
                    self.logger.debug(f"Health check for {name}: {health}")
                except Exception as e:
                    self.logger.error(f"Health check failed for {name}: {e}")
                    self._provider_health[name] = False
                    self._provider_last_health_check[name] = current_time


# Factory functions for easy service creation


def create_llm_service_from_config(config_dict: Dict[str, Any]) -> LLMService:
    """Create LLM service from configuration dictionary."""
    # Convert dict to config objects
    providers = {}
    for name, provider_dict in config_dict.get("providers", {}).items():
        provider_dict["name"] = name
        providers[name] = ProviderConfig(**provider_dict)

    service_config = LLMServiceConfig(
        default_provider=config_dict.get("default_provider", "openai"),
        fallback_chain=config_dict.get("fallback_chain", ["openai"]),
        enable_fallback=config_dict.get("enable_fallback", True),
        providers=providers,
    )

    return LLMService(service_config)


def create_default_llm_service() -> LLMService:
    """Create LLM service with default configuration."""
    config = {
        "default_provider": "openai",
        "fallback_chain": ["openai", "ollama"],
        "providers": {
            "openai": {
                "enabled": True,
                "models": ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
                "rate_limits": {"requests_per_minute": 60},
            },
            "ollama": {
                "enabled": True,
                "base_url": "http://localhost:11434",
                "models": ["llama2", "codellama", "mistral"],
                "timeout": 120,
            },
        },
    }

    return create_llm_service_from_config(config)


# OpenAI-Compatible API Layer


class OpenAICompatibleClient:
    """OpenAI-compatible client routing calls through centralized LLM service.

    This allows any framework or library that expects OpenAI API to work seamlessly
    with our centralized LLM service, including local models and other providers.
    """

    def __init__(self, llm_service: LLMService, api_key: str = "voyager-internal"):
        self.llm_service = llm_service
        self.api_key = api_key

        # Create nested client structure to match OpenAI SDK
        self.chat = self.ChatCompletions(self)

    class ChatCompletions:
        """OpenAI chat completions compatible interface."""

        def __init__(self, client: "OpenAICompatibleClient"):
            self.client = client

        async def create(
            self,
            model: str,
            messages: List[Dict[str, str]],
            temperature: float = 0.7,
            max_tokens: int = 2000,
            stream: bool = False,
            provider: Optional[str] = None,
            **kwargs,
        ) -> Union["OpenAIResponse", AsyncIterator["OpenAIResponse"]]:
            """Create chat completion (OpenAI compatible)."""

            request = LLMRequest(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
                provider=provider,
                metadata=kwargs,
            )

            if stream:
                return self._stream_response(request)
            else:
                response = await self.client.llm_service.generate(request)
                return self._convert_to_openai_response(response)

        async def _stream_response(
            self, request: LLMRequest
        ) -> AsyncIterator["OpenAIStreamResponse"]:
            """Handle streaming response."""
            async for chunk in self.client.llm_service.stream_generate(request):
                yield OpenAIStreamResponse(
                    id=f"chatcmpl-{int(time.time())}",
                    object="chat.completion.chunk",
                    created=int(time.time()),
                    model=chunk.model,
                    choices=[
                        OpenAIChoice(
                            index=0,
                            delta=OpenAIDelta(content=chunk.content),
                            finish_reason=chunk.finish_reason
                            if chunk.finish_reason != "continue"
                            else None,
                        )
                    ],
                )

        def _convert_to_openai_response(
            self, response: LLMResponse
        ) -> "OpenAIResponse":
            """Convert LLMResponse to OpenAI format."""
            return OpenAIResponse(
                id=f"chatcmpl-{int(time.time())}",
                object="chat.completion",
                created=int(response.created_at),
                model=response.model,
                choices=[
                    OpenAIChoice(
                        index=0,
                        message=OpenAIMessage(
                            role="assistant", content=response.content
                        ),
                        finish_reason=response.finish_reason,
                    )
                ],
                usage=OpenAIUsage(
                    prompt_tokens=response.usage.get("prompt_tokens", 0),
                    completion_tokens=response.usage.get("completion_tokens", 0),
                    total_tokens=response.usage.get("total_tokens", 0),
                ),
                provider_metadata={
                    "provider": response.provider,
                    "original_metadata": response.metadata,
                },
            )


@dataclass
class OpenAIMessage:
    """OpenAI message format."""

    role: str
    content: str


@dataclass
class OpenAIDelta:
    """OpenAI delta format for streaming."""

    content: Optional[str] = None


@dataclass
class OpenAIChoice:
    """OpenAI choice format."""

    index: int
    message: Optional[OpenAIMessage] = None
    delta: Optional[OpenAIDelta] = None
    finish_reason: Optional[str] = None


@dataclass
class OpenAIUsage:
    """OpenAI usage format."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class OpenAIResponse:
    """OpenAI response format."""

    id: str
    object: str
    created: int
    model: str
    choices: List[OpenAIChoice]
    usage: OpenAIUsage
    provider_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OpenAIStreamResponse:
    """OpenAI streaming response format."""

    id: str
    object: str
    created: int
    model: str
    choices: List[OpenAIChoice]


class UniversalLLMClient:
    """Universal LLM client that can be used as drop-in replacement for OpenAI client.

    This client provides:
    - OpenAI SDK compatible interface
    - Automatic routing to best available provider
    - Local model support (Ollama, etc.)
    - Fallback mechanisms
    - Cost tracking and monitoring
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config:
            self.llm_service = create_llm_service_from_config(config)
        else:
            self.llm_service = create_default_llm_service()

        self.openai_client = OpenAICompatibleClient(self.llm_service)

        # Expose OpenAI compatible interface
        self.chat = self.openai_client.chat

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Direct LLM service access."""
        return await self.llm_service.generate(request)

    async def stream_generate(self, request: LLMRequest) -> AsyncIterator[LLMResponse]:
        """Direct streaming access."""
        async for response in self.llm_service.stream_generate(request):
            yield response

    def get_available_models(self) -> Dict[str, List[str]]:
        """Get available models."""
        return self.llm_service.get_available_models()

    def get_provider_status(self) -> Dict[str, Dict[str, Any]]:
        """Get provider status."""
        return self.llm_service.get_provider_status()

    async def health_check(self) -> Dict[str, bool]:
        """Run health check."""
        return await self.llm_service.health_check()


# Global service instance for easy access
_global_llm_service: Optional[LLMService] = None
_global_client: Optional[UniversalLLMClient] = None


def get_global_llm_service() -> LLMService:
    """Get or create global LLM service instance."""
    global _global_llm_service
    if _global_llm_service is None:
        _global_llm_service = create_default_llm_service()
    return _global_llm_service


def get_global_llm_client() -> UniversalLLMClient:
    """Get or create global LLM client instance."""
    global _global_client
    if _global_client is None:
        _global_client = UniversalLLMClient()
    return _global_client


def set_global_llm_config(config: Dict[str, Any]) -> None:
    """Set global LLM configuration."""
    global _global_llm_service, _global_client
    _global_llm_service = create_llm_service_from_config(config)
    _global_client = UniversalLLMClient(config)


# Convenience functions that mimic OpenAI SDK
async def chat_completion_create(
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.7,
    max_tokens: int = 2000,
    stream: bool = False,
    provider: Optional[str] = None,
    **kwargs,
) -> Union[OpenAIResponse, AsyncIterator[OpenAIStreamResponse]]:
    """Create chat completion using global service (OpenAI SDK compatible)."""
    client = get_global_llm_client()
    return await client.chat.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=stream,
        provider=provider,
        **kwargs,
    )


# Export commonly used symbols for easy importing
__all__ = [
    "LLMService",
    "LLMRequest",
    "LLMResponse",
    "UniversalLLMClient",
    "OpenAICompatibleClient",
    "create_llm_service_from_config",
    "create_default_llm_service",
    "get_global_llm_service",
    "get_global_llm_client",
    "set_global_llm_config",
    "chat_completion_create",
    "LLMError",
    "ProviderError",
    "RateLimitError",
    "ModelNotAvailableError",
]
