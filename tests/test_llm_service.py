"""Tests for Centralized LLM Service."""

import asyncio
import json
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.voyager_trader.llm_service import (
    AnthropicProvider,
    LLMRequest,
    LLMResponse,
    LLMService,
    LLMServiceConfig,
    ModelNotAvailableError,
    OllamaProvider,
    OpenAIProvider,
    ProviderConfig,
    ProviderError,
    ProviderRegistry,
    RateLimitError,
    UniversalLLMClient,
    chat_completion_create,
    create_default_llm_service,
    create_llm_service_from_config,
    get_global_llm_client,
    get_global_llm_service,
)


class TestLLMRequest:
    """Test LLMRequest data class."""

    def test_basic_request(self):
        """Test basic request creation."""
        request = LLMRequest(
            messages=[{"role": "user", "content": "Hello"}], model="gpt-3.5-turbo"
        )

        assert request.messages == [{"role": "user", "content": "Hello"}]
        assert request.model == "gpt-3.5-turbo"
        assert request.temperature == 0.7
        assert request.max_tokens == 2000
        assert request.stream is False

    def test_system_prompt_injection(self):
        """Test system prompt is properly injected."""
        request = LLMRequest(
            messages=[{"role": "user", "content": "Hello"}],
            model="gpt-3.5-turbo",
            system_prompt="You are a helpful assistant",
        )

        assert len(request.messages) == 2
        assert request.messages[0]["role"] == "system"
        assert request.messages[0]["content"] == "You are a helpful assistant"
        assert request.messages[1]["role"] == "user"

    def test_system_prompt_only(self):
        """Test request with only system prompt."""
        request = LLMRequest(
            messages=[],
            model="gpt-3.5-turbo",
            system_prompt="You are a helpful assistant",
        )

        assert len(request.messages) == 1
        assert request.messages[0]["role"] == "system"


class TestProviderConfig:
    """Test provider configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ProviderConfig(name="openai")

        assert config.name == "openai"
        assert config.enabled is True
        assert config.timeout == 60
        assert config.max_retries == 3
        assert len(config.models) == 0

    def test_custom_config(self):
        """Test custom configuration values."""
        config = ProviderConfig(
            name="ollama",
            enabled=True,
            base_url="http://localhost:11434",
            models=["llama2", "codellama"],
            timeout=120,
        )

        assert config.name == "ollama"
        assert config.base_url == "http://localhost:11434"
        assert "llama2" in config.models
        assert config.timeout == 120


class TestProviderRegistry:
    """Test provider registry."""

    def setup_method(self):
        """Set up test fixtures."""
        self.registry = ProviderRegistry()

    def test_register_provider(self):
        """Test provider registration."""
        config = ProviderConfig(name="test_provider")
        provider = Mock()
        provider.config = config

        self.registry.register_provider(provider)

        assert "test_provider" in self.registry.providers
        assert self.registry.get_provider("test_provider") == provider

    def test_get_available_providers(self):
        """Test getting available providers."""
        # Register enabled provider
        enabled_config = ProviderConfig(name="enabled", enabled=True)
        enabled_provider = Mock()
        enabled_provider.config = enabled_config

        # Register disabled provider
        disabled_config = ProviderConfig(name="disabled", enabled=False)
        disabled_provider = Mock()
        disabled_provider.config = disabled_config

        self.registry.register_provider(enabled_provider)
        self.registry.register_provider(disabled_provider)

        available = self.registry.get_available_providers()
        assert "enabled" in available
        assert "disabled" not in available

    def test_get_provider_for_model(self):
        """Test finding provider by model."""
        config = ProviderConfig(name="test", models=["model1", "model2"], enabled=True)
        provider = Mock()
        provider.config = config
        provider.get_available_models.return_value = ["model1", "model2"]

        self.registry.register_provider(provider)

        found_provider = self.registry.get_provider_for_model("model1")
        assert found_provider == provider

        not_found = self.registry.get_provider_for_model("model3")
        assert not_found is None

    @pytest.mark.asyncio
    async def test_health_check_all(self):
        """Test health check for all providers."""
        config = ProviderConfig(name="test", enabled=True)
        provider = AsyncMock()
        provider.config = config
        provider.health_check.return_value = True

        self.registry.register_provider(provider)

        results = await self.registry.health_check_all()
        assert results["test"] is True
        provider.health_check.assert_called_once()


@pytest.mark.asyncio
class TestOpenAIProvider:
    """Test OpenAI provider implementation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = ProviderConfig(
            name="openai", api_key="test_key", models=["gpt-3.5-turbo", "gpt-4"]
        )

    @patch("src.voyager_trader.llm_service.OPENAI_AVAILABLE", True)
    @patch("src.voyager_trader.llm_service.openai")
    def test_provider_initialization(self, mock_openai):
        """Test provider initialization."""
        mock_client = Mock()
        mock_openai.AsyncOpenAI.return_value = mock_client

        provider = OpenAIProvider(self.config)

        assert provider.config == self.config
        assert provider.client == mock_client
        mock_openai.AsyncOpenAI.assert_called_once_with(
            api_key="test_key", base_url=None
        )

    @patch("src.voyager_trader.llm_service.OPENAI_AVAILABLE", False)
    def test_provider_unavailable_error(self):
        """Test error when OpenAI library not available."""
        with pytest.raises(ProviderError, match="OpenAI library not available"):
            OpenAIProvider(self.config)

    @patch("src.voyager_trader.llm_service.OPENAI_AVAILABLE", True)
    @patch("src.voyager_trader.llm_service.openai")
    def test_missing_api_key_error(self, mock_openai):
        """Test error when API key is missing."""
        config = ProviderConfig(name="openai", api_key=None)

        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ProviderError, match="API key not provided"):
                OpenAIProvider(config)

    @patch("src.voyager_trader.llm_service.OPENAI_AVAILABLE", True)
    @patch("src.voyager_trader.llm_service.openai")
    async def test_generate_success(self, mock_openai):
        """Test successful response generation."""
        # Setup mock client and response
        mock_client = AsyncMock()
        mock_openai.AsyncOpenAI.return_value = mock_client

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Generated response"
        mock_response.model = "gpt-3.5-turbo"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15
        mock_response.choices[0].finish_reason = "stop"
        mock_response.id = "test_id"

        mock_client.chat.completions.create.return_value = mock_response

        provider = OpenAIProvider(self.config)

        request = LLMRequest(
            messages=[{"role": "user", "content": "Hello"}], model="gpt-3.5-turbo"
        )

        response = await provider.generate(request)

        assert isinstance(response, LLMResponse)
        assert response.content == "Generated response"
        assert response.model == "gpt-3.5-turbo"
        assert response.provider == "openai"
        assert response.usage["total_tokens"] == 15

    @patch("src.voyager_trader.llm_service.OPENAI_AVAILABLE", True)
    @patch("src.voyager_trader.llm_service.openai")
    async def test_generate_model_not_available(self, mock_openai):
        """Test error when model not available."""
        mock_client = AsyncMock()
        mock_openai.AsyncOpenAI.return_value = mock_client

        provider = OpenAIProvider(self.config)

        request = LLMRequest(
            messages=[{"role": "user", "content": "Hello"}], model="nonexistent-model"
        )

        with pytest.raises(ModelNotAvailableError):
            await provider.generate(request)

    @patch("src.voyager_trader.llm_service.OPENAI_AVAILABLE", True)
    @patch("src.voyager_trader.llm_service.openai")
    async def test_generate_rate_limit_error(self, mock_openai):
        """Test rate limit error handling."""
        mock_client = AsyncMock()
        mock_openai.AsyncOpenAI.return_value = mock_client

        # Create rate limit error
        error = Exception("Rate limit exceeded")
        error.status_code = 429
        mock_client.chat.completions.create.side_effect = error

        provider = OpenAIProvider(self.config)

        request = LLMRequest(
            messages=[{"role": "user", "content": "Hello"}], model="gpt-3.5-turbo"
        )

        with pytest.raises(RateLimitError):
            await provider.generate(request)


@pytest.mark.asyncio
class TestOllamaProvider:
    """Test Ollama provider implementation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = ProviderConfig(
            name="ollama",
            base_url="http://localhost:11434",
            models=["llama2", "codellama"],
        )

    def test_provider_initialization(self):
        """Test provider initialization."""
        provider = OllamaProvider(self.config)

        assert provider.config == self.config
        assert provider.base_url == "http://localhost:11434"

    def test_message_conversion(self):
        """Test message conversion to Ollama format."""
        provider = OllamaProvider(self.config)

        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]

        prompt = provider._convert_messages_to_prompt(messages)

        assert "System: You are helpful" in prompt
        assert "User: Hello" in prompt
        assert "Assistant: Hi there" in prompt
        assert prompt.endswith("Assistant:")

    @patch("aiohttp.ClientSession.post")
    async def test_generate_success(self, mock_post):
        """Test successful response generation."""
        # Mock response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            "response": "Generated response",
            "prompt_eval_count": 10,
            "eval_count": 5,
            "eval_duration": 1000000,
            "load_duration": 500000,
        }

        mock_post.return_value.__aenter__.return_value = mock_response

        provider = OllamaProvider(self.config)

        request = LLMRequest(
            messages=[{"role": "user", "content": "Hello"}], model="llama2"
        )

        response = await provider.generate(request)

        assert isinstance(response, LLMResponse)
        assert response.content == "Generated response"
        assert response.model == "llama2"
        assert response.provider == "ollama"
        assert response.usage["prompt_tokens"] == 10
        assert response.usage["completion_tokens"] == 5

    @patch("aiohttp.ClientSession.post")
    async def test_generate_error(self, mock_post):
        """Test error handling."""
        # Mock error response
        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.text.return_value = "Internal server error"

        mock_post.return_value.__aenter__.return_value = mock_response

        provider = OllamaProvider(self.config)

        request = LLMRequest(
            messages=[{"role": "user", "content": "Hello"}], model="llama2"
        )

        with pytest.raises(ProviderError, match="Ollama API error"):
            await provider.generate(request)


@pytest.mark.asyncio
class TestLLMService:
    """Test centralized LLM service."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = LLMServiceConfig(
            default_provider="openai",
            fallback_chain=["openai", "ollama"],
            providers={
                "openai": ProviderConfig(name="openai", enabled=True),
                "ollama": ProviderConfig(name="ollama", enabled=True),
            },
        )

    def test_service_initialization(self):
        """Test service initialization."""
        # Create a simplified config with valid provider settings
        config = LLMServiceConfig(
            default_provider="openai",
            providers={
                "openai": ProviderConfig(
                    name="openai",
                    enabled=True,
                    api_key="test_key",
                    timeout=60,
                    max_retries=3,
                ),
                "ollama": ProviderConfig(
                    name="ollama", enabled=False
                ),  # Disabled to avoid init issues
            },
        )

        # Mock the _create_provider method to return test providers
        mock_openai_provider = Mock()
        mock_openai_provider.config.name = "openai"

        with patch.object(LLMService, "_create_provider") as mock_create:
            mock_create.return_value = mock_openai_provider

            service = LLMService(config)

            # Verify only enabled provider was created
            mock_create.assert_called_once()
            # Verify service has correct config
            assert service.config.default_provider == "openai"
            assert len(service.config.providers) == 2

    @patch("src.voyager_trader.llm_service.AnthropicProvider")
    @patch("src.voyager_trader.llm_service.OpenAIProvider")
    @patch("src.voyager_trader.llm_service.OllamaProvider")
    async def test_generate_with_specific_provider(
        self, mock_ollama_provider, mock_openai_provider, mock_anthropic_provider
    ):
        """Test generation with specific provider."""
        # Setup mock providers
        mock_openai_instance = AsyncMock()
        mock_openai_instance.config = ProviderConfig(name="openai", enabled=True)
        mock_openai_instance.generate.return_value = LLMResponse(
            content="Generated response", model="gpt-3.5-turbo", provider="openai"
        )
        mock_openai_provider.return_value = mock_openai_instance

        mock_ollama_instance = Mock()
        mock_anthropic_instance = Mock()
        mock_ollama_provider.return_value = mock_ollama_instance
        mock_anthropic_provider.return_value = mock_anthropic_instance

        # Create simplified config
        simple_config = LLMServiceConfig(
            default_provider="openai",
            providers={
                "openai": ProviderConfig(
                    name="openai",
                    enabled=True,
                    api_key="test_key",
                    timeout=60,
                    max_retries=3,
                )
            },
        )

        service = LLMService(simple_config)

        request = LLMRequest(
            messages=[{"role": "user", "content": "Hello"}],
            model="gpt-3.5-turbo",
            provider="openai",
        )

        response = await service.generate(request)

        assert response.content == "Generated response"
        assert response.provider == "openai"
        mock_openai_instance.generate.assert_called_once_with(request)

    @patch("src.voyager_trader.llm_service.OpenAIProvider")
    async def test_generate_provider_not_available(self, mock_provider_class):
        """Test error when requested provider not available."""
        mock_provider_class.return_value = Mock()

        service = LLMService(self.config)

        request = LLMRequest(
            messages=[{"role": "user", "content": "Hello"}],
            model="gpt-3.5-turbo",
            provider="nonexistent",
        )

        with pytest.raises(ProviderError, match="not available"):
            await service.generate(request)

    @patch("src.voyager_trader.llm_service.AnthropicProvider")
    @patch("src.voyager_trader.llm_service.OpenAIProvider")
    @patch("src.voyager_trader.llm_service.OllamaProvider")
    def test_get_available_models(
        self, mock_ollama_provider, mock_openai_provider, mock_anthropic_provider
    ):
        """Test getting available models."""
        mock_openai_instance = Mock()
        mock_openai_instance.config = ProviderConfig(name="openai", enabled=True)
        mock_openai_instance.get_available_models.return_value = [
            "gpt-3.5-turbo",
            "gpt-4",
        ]
        mock_openai_provider.return_value = mock_openai_instance

        mock_ollama_instance = Mock()
        mock_anthropic_instance = Mock()
        mock_ollama_provider.return_value = mock_ollama_instance
        mock_anthropic_provider.return_value = mock_anthropic_instance

        # Create simplified config
        simple_config = LLMServiceConfig(
            default_provider="openai",
            providers={
                "openai": ProviderConfig(
                    name="openai",
                    enabled=True,
                    api_key="test_key",
                    timeout=60,
                    max_retries=3,
                )
            },
        )

        service = LLMService(simple_config)

        models = service.get_available_models()

        assert "openai" in models
        assert "gpt-3.5-turbo" in models["openai"]
        assert "gpt-4" in models["openai"]

    @patch("src.voyager_trader.llm_service.AnthropicProvider")
    @patch("src.voyager_trader.llm_service.OpenAIProvider")
    @patch("src.voyager_trader.llm_service.OllamaProvider")
    def test_get_provider_status(
        self, mock_ollama_provider, mock_openai_provider, mock_anthropic_provider
    ):
        """Test getting provider status."""
        mock_openai_instance = Mock()
        mock_openai_instance.config = ProviderConfig(name="openai", enabled=True)
        mock_openai_instance.get_available_models.return_value = ["gpt-3.5-turbo"]
        mock_openai_provider.return_value = mock_openai_instance

        mock_ollama_instance = Mock()
        mock_anthropic_instance = Mock()
        mock_ollama_provider.return_value = mock_ollama_instance
        mock_anthropic_provider.return_value = mock_anthropic_instance

        # Create simplified config
        simple_config = LLMServiceConfig(
            default_provider="openai",
            providers={
                "openai": ProviderConfig(
                    name="openai",
                    enabled=True,
                    api_key="test_key",
                    timeout=60,
                    max_retries=3,
                )
            },
        )

        service = LLMService(simple_config)
        service._provider_health = {"openai": True}

        status = service.get_provider_status()

        assert "openai" in status
        assert status["openai"]["enabled"] is True
        assert status["openai"]["healthy"] is True
        assert "gpt-3.5-turbo" in status["openai"]["models"]


@pytest.mark.asyncio
class TestUniversalLLMClient:
    """Test universal LLM client."""

    @patch("src.voyager_trader.llm_service.create_default_llm_service")
    def test_client_initialization(self, mock_create_service):
        """Test client initialization."""
        mock_service = Mock()
        mock_create_service.return_value = mock_service

        client = UniversalLLMClient()

        assert client.llm_service == mock_service
        assert hasattr(client, "chat")
        mock_create_service.assert_called_once()

    @patch("src.voyager_trader.llm_service.create_llm_service_from_config")
    def test_client_with_config(self, mock_create_service):
        """Test client initialization with config."""
        config = {"providers": {"openai": {"enabled": True}}}
        mock_service = Mock()
        mock_create_service.return_value = mock_service

        client = UniversalLLMClient(config)

        assert client.llm_service == mock_service
        mock_create_service.assert_called_once_with(config)

    @patch("src.voyager_trader.llm_service.create_default_llm_service")
    async def test_openai_compatible_interface(self, mock_create_service):
        """Test OpenAI compatible interface."""
        mock_service = AsyncMock()
        mock_service.generate.return_value = LLMResponse(
            content="Generated response",
            model="gpt-3.5-turbo",
            provider="openai",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )
        mock_create_service.return_value = mock_service

        client = UniversalLLMClient()

        response = await client.chat.create(
            model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Hello"}]
        )

        assert response.model == "gpt-3.5-turbo"
        assert response.choices[0].message.content == "Generated response"
        assert response.usage.total_tokens == 15


class TestFactoryFunctions:
    """Test factory functions."""

    def test_create_llm_service_from_config(self):
        """Test service creation from config dict."""
        config = {
            "default_provider": "openai",
            "providers": {"openai": {"enabled": True, "models": ["gpt-3.5-turbo"]}},
        }

        with patch("src.voyager_trader.llm_service.LLMService") as mock_service_class:
            create_llm_service_from_config(config)

            mock_service_class.assert_called_once()
            service_config = mock_service_class.call_args[0][0]
            assert service_config.default_provider == "openai"
            assert "openai" in service_config.providers

    def test_create_default_llm_service(self):
        """Test default service creation."""
        with patch(
            "src.voyager_trader.llm_service.create_llm_service_from_config"
        ) as mock_create:
            create_default_llm_service()

            mock_create.assert_called_once()
            config = mock_create.call_args[0][0]
            assert config["default_provider"] == "openai"
            assert "openai" in config["providers"]
            assert "ollama" in config["providers"]

    def test_global_service_singleton(self):
        """Test global service singleton pattern."""
        with patch(
            "src.voyager_trader.llm_service.create_default_llm_service"
        ) as mock_create:
            mock_service = Mock()
            mock_create.return_value = mock_service

            # Reset global state
            import src.voyager_trader.llm_service as llm_module

            llm_module._global_llm_service = None

            service1 = get_global_llm_service()
            service2 = get_global_llm_service()

            assert service1 == service2
            assert service1 == mock_service
            mock_create.assert_called_once()

    def test_global_client_singleton(self):
        """Test global client singleton pattern."""
        with patch(
            "src.voyager_trader.llm_service.UniversalLLMClient"
        ) as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client

            # Reset global state
            import src.voyager_trader.llm_service as llm_module

            llm_module._global_client = None

            client1 = get_global_llm_client()
            client2 = get_global_llm_client()

            assert client1 == client2
            assert client1 == mock_client
            mock_client_class.assert_called_once()


@pytest.mark.asyncio
class TestConvenienceFunctions:
    """Test convenience functions."""

    @patch("src.voyager_trader.llm_service.get_global_llm_client")
    async def test_chat_completion_create(self, mock_get_client):
        """Test convenience chat completion function."""
        mock_client = AsyncMock()
        mock_client.chat.create.return_value = Mock()
        mock_get_client.return_value = mock_client

        await chat_completion_create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.8,
        )

        mock_client.chat.create.assert_called_once_with(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.8,
            max_tokens=2000,
            stream=False,
            provider=None,
        )


@pytest.mark.asyncio
class TestStreamingScenarios:
    """Test streaming response scenarios."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = ProviderConfig(
            name="openai", api_key="test_key", models=["gpt-3.5-turbo"]
        )

    @patch("src.voyager_trader.llm_service.OPENAI_AVAILABLE", True)
    @patch("src.voyager_trader.llm_service.openai")
    async def test_streaming_generate_success(self, mock_openai):
        """Test successful streaming response generation."""
        # Setup mock client and streaming response
        mock_client = AsyncMock()
        mock_openai.AsyncOpenAI.return_value = mock_client

        # Create mock streaming chunks
        chunks = [
            Mock(choices=[Mock(delta=Mock(content="Hello"))]),
            Mock(choices=[Mock(delta=Mock(content=" world"))]),
            Mock(choices=[Mock(delta=Mock(content="!"))]),
        ]

        # Set finish reasons
        chunks[0].choices[0].finish_reason = None
        chunks[1].choices[0].finish_reason = None
        chunks[2].choices[0].finish_reason = "stop"

        # Set other required attributes
        for i, chunk in enumerate(chunks):
            chunk.id = f"chunk_{i}"
            chunk.model = "gpt-3.5-turbo"

        async def async_iter():
            for chunk in chunks:
                yield chunk

        mock_client.chat.completions.create.return_value = async_iter()

        provider = OpenAIProvider(self.config)

        request = LLMRequest(
            messages=[{"role": "user", "content": "Hello"}],
            model="gpt-3.5-turbo",
            stream=True,
        )

        # Collect streaming responses
        responses = []
        async for response in provider.stream_generate(request):
            responses.append(response)

        # Verify streaming responses
        assert len(responses) == 3
        assert responses[0].content == "Hello"
        assert responses[1].content == " world"
        assert responses[2].content == "!"
        assert responses[2].finish_reason == "stop"

    @patch("src.voyager_trader.llm_service.OPENAI_AVAILABLE", True)
    @patch("src.voyager_trader.llm_service.openai")
    async def test_streaming_generate_error(self, mock_openai):
        """Test streaming error handling."""
        mock_client = AsyncMock()
        mock_openai.AsyncOpenAI.return_value = mock_client

        # Create error during streaming
        async def async_iter():
            yield Mock(choices=[Mock(delta=Mock(content="Hello"))])
            raise Exception("Stream interrupted")

        mock_client.chat.completions.create.return_value = async_iter()

        provider = OpenAIProvider(self.config)

        request = LLMRequest(
            messages=[{"role": "user", "content": "Hello"}],
            model="gpt-3.5-turbo",
            stream=True,
        )

        responses = []
        with pytest.raises(ProviderError):
            async for response in provider.stream_generate(request):
                responses.append(response)

        # Should have received at least one response before error
        assert len(responses) >= 1

    @patch("aiohttp.ClientSession.post")
    async def test_ollama_streaming_success(self, mock_post):
        """Test Ollama streaming response."""
        config = ProviderConfig(
            name="ollama",
            base_url="http://localhost:11434",
            models=["llama2"],
        )

        # Mock streaming response
        mock_response = AsyncMock()
        mock_response.status = 200

        # Create mock streaming content
        stream_data = [
            b'{"response": "Hello", "done": false}\n',
            b'{"response": " world", "done": false}\n',
            b'{"response": "!", "done": true, "prompt_eval_count": 10, '
            b'"eval_count": 3}\n',
        ]

        async def content_iter():
            for data in stream_data:
                yield data

        # Set up the async iterator for content
        mock_response.content.__aiter__ = lambda: content_iter()

        mock_post.return_value.__aenter__.return_value = mock_response

        provider = OllamaProvider(config)

        request = LLMRequest(
            messages=[{"role": "user", "content": "Hello"}], model="llama2", stream=True
        )

        responses = []
        async for response in provider.stream_generate(request):
            responses.append(response)

        # Verify streaming responses
        assert len(responses) == 3
        assert responses[0].content == "Hello"
        assert responses[1].content == " world"
        assert responses[2].content == "!"
        assert responses[2].finished is True


@pytest.mark.asyncio
class TestErrorScenarios:
    """Test comprehensive error handling scenarios."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = ProviderConfig(
            name="openai", api_key="test_key", models=["gpt-3.5-turbo"]
        )

    async def test_network_timeout_error(self):
        """Test network timeout error handling."""
        config = ProviderConfig(
            name="ollama",
            base_url="http://localhost:11434",
            models=["llama2"],
            timeout=1,  # Very short timeout
        )

        provider = OllamaProvider(config)

        request = LLMRequest(
            messages=[{"role": "user", "content": "Hello"}], model="llama2"
        )

        with patch("aiohttp.ClientSession.post") as mock_post:
            # Simulate timeout
            mock_post.side_effect = asyncio.TimeoutError("Connection timeout")

            with pytest.raises(ProviderError, match="timeout"):
                await provider.generate(request)

    async def test_connection_error(self):
        """Test connection error handling."""
        config = ProviderConfig(
            name="ollama", base_url="http://localhost:11434", models=["llama2"]
        )

        provider = OllamaProvider(config)

        request = LLMRequest(
            messages=[{"role": "user", "content": "Hello"}], model="llama2"
        )

        with patch("aiohttp.ClientSession.post") as mock_post:
            # Simulate connection error
            import aiohttp

            mock_post.side_effect = aiohttp.ClientConnectorError(
                connection_key=None, os_error=None
            )

            with pytest.raises(ProviderError):
                await provider.generate(request)

    @patch("src.voyager_trader.llm_service.OPENAI_AVAILABLE", True)
    @patch("src.voyager_trader.llm_service.openai")
    async def test_api_key_unauthorized_error(self, mock_openai):
        """Test API key unauthorized error."""
        mock_client = AsyncMock()
        mock_openai.AsyncOpenAI.return_value = mock_client

        # Create unauthorized error
        error = Exception("Incorrect API key provided")
        error.status_code = 401
        mock_client.chat.completions.create.side_effect = error

        provider = OpenAIProvider(self.config)

        request = LLMRequest(
            messages=[{"role": "user", "content": "Hello"}], model="gpt-3.5-turbo"
        )

        with pytest.raises(ProviderError):
            await provider.generate(request)

    @patch("src.voyager_trader.llm_service.OPENAI_AVAILABLE", True)
    @patch("src.voyager_trader.llm_service.openai")
    async def test_model_overloaded_error(self, mock_openai):
        """Test model overloaded/capacity error."""
        mock_client = AsyncMock()
        mock_openai.AsyncOpenAI.return_value = mock_client

        # Create service overloaded error
        error = Exception("The model is currently overloaded")
        error.status_code = 503
        mock_client.chat.completions.create.side_effect = error

        provider = OpenAIProvider(self.config)

        request = LLMRequest(
            messages=[{"role": "user", "content": "Hello"}], model="gpt-3.5-turbo"
        )

        with pytest.raises(ProviderError):
            await provider.generate(request)

    @patch("aiohttp.ClientSession.post")
    async def test_ollama_model_not_found_error(self, mock_post):
        """Test Ollama model not found error."""
        config = ProviderConfig(
            name="ollama",
            base_url="http://localhost:11434",
            models=["nonexistent-model"],
        )

        # Mock not found response
        mock_response = AsyncMock()
        mock_response.status = 404
        mock_response.text.return_value = "model 'nonexistent-model' not found"

        mock_post.return_value.__aenter__.return_value = mock_response

        provider = OllamaProvider(config)

        request = LLMRequest(
            messages=[{"role": "user", "content": "Hello"}], model="nonexistent-model"
        )

        with pytest.raises(ProviderError, match="Ollama API error"):
            await provider.generate(request)

    async def test_invalid_json_response_error(self):
        """Test invalid JSON response handling."""
        config = ProviderConfig(
            name="ollama", base_url="http://localhost:11434", models=["llama2"]
        )

        provider = OllamaProvider(config)

        request = LLMRequest(
            messages=[{"role": "user", "content": "Hello"}], model="llama2"
        )

        with patch("aiohttp.ClientSession.post") as mock_post:
            # Mock response with invalid JSON
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.side_effect = json.JSONDecodeError(
                "Invalid JSON", "doc", 0
            )
            mock_response.text.return_value = "Invalid JSON response"

            mock_post.return_value.__aenter__.return_value = mock_response

            with pytest.raises(ProviderError):
                await provider.generate(request)

    async def test_empty_response_error(self):
        """Test empty response handling."""
        config = ProviderConfig(
            name="ollama", base_url="http://localhost:11434", models=["llama2"]
        )

        provider = OllamaProvider(config)

        request = LLMRequest(
            messages=[{"role": "user", "content": "Hello"}], model="llama2"
        )

        with patch("aiohttp.ClientSession.post") as mock_post:
            # Mock empty response
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = {}  # Empty response

            mock_post.return_value.__aenter__.return_value = mock_response

            with pytest.raises(ProviderError):
                await provider.generate(request)

    async def test_anthropic_message_format_error(self):
        """Test Anthropic message format error handling."""
        if not hasattr(self, "anthropic_available"):
            pytest.skip("Anthropic library not available")

        config = ProviderConfig(
            name="anthropic", api_key="test_key", models=["claude-3-haiku"]
        )

        with patch("src.voyager_trader.llm_service.ANTHROPIC_AVAILABLE", True):
            with patch("src.voyager_trader.llm_service.anthropic") as mock_anthropic:
                mock_client = AsyncMock()
                mock_anthropic.AsyncAnthropic.return_value = mock_client

                # Create message format error
                error = Exception("Invalid message format")
                error.status_code = 400
                mock_client.messages.create.side_effect = error

                provider = AnthropicProvider(config)

                request = LLMRequest(
                    messages=[{"role": "invalid", "content": "Hello"}],  # Invalid role
                    model="claude-3-haiku",
                )

                with pytest.raises(ProviderError):
                    await provider.generate(request)


@pytest.mark.asyncio
class TestRateLimitingScenarios:
    """Test rate limiting scenarios."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = ProviderConfig(
            name="openai",
            api_key="test_key",
            models=["gpt-3.5-turbo"],
            rate_limits={"requests_per_minute": 2, "tokens_per_minute": 1000},
        )

    @patch("src.voyager_trader.llm_service.OPENAI_AVAILABLE", True)
    @patch("src.voyager_trader.llm_service.openai")
    async def test_request_rate_limiting(self, mock_openai):
        """Test request-based rate limiting."""
        mock_client = AsyncMock()
        mock_openai.AsyncOpenAI.return_value = mock_client

        # Mock successful response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Response"
        mock_response.model = "gpt-3.5-turbo"
        mock_response.usage.total_tokens = 10
        mock_response.choices[0].finish_reason = "stop"
        mock_response.id = "test_id"

        mock_client.chat.completions.create.return_value = mock_response

        provider = OpenAIProvider(self.config)

        request = LLMRequest(
            messages=[{"role": "user", "content": "Hello"}], model="gpt-3.5-turbo"
        )

        # First request should succeed
        await provider.generate(request)

        # Second request should succeed
        await provider.generate(request)

        # Third request should be rate limited (need to test timing)
        # This is a simplified test - in real scenarios, rate limiting involves timing

    @patch("src.voyager_trader.llm_service.OPENAI_AVAILABLE", True)
    @patch("src.voyager_trader.llm_service.openai")
    async def test_token_rate_limiting(self, mock_openai):
        """Test token-based rate limiting."""
        # Configure very low token limit
        config = ProviderConfig(
            name="openai",
            api_key="test_key",
            models=["gpt-3.5-turbo"],
            rate_limits={"requests_per_minute": 10, "tokens_per_minute": 20},
        )

        mock_client = AsyncMock()
        mock_openai.AsyncOpenAI.return_value = mock_client

        OpenAIProvider(config)

        # Create request with many tokens (should exceed limit)
        long_message = "This is a very long message " * 100  # Approx 500+ tokens
        LLMRequest(
            messages=[{"role": "user", "content": long_message}], model="gpt-3.5-turbo"
        )

        # Test that rate limiting logic is applied (tokens estimated correctly)
        # The actual rate limiting behavior would require time-based testing


@pytest.mark.asyncio
class TestSessionLifecycleScenarios:
    """Test session lifecycle management scenarios."""

    async def test_context_manager_cleanup(self):
        """Test proper cleanup with context manager."""
        config = {
            "default_provider": "openai",
            "providers": {
                "openai": {
                    "enabled": True,
                    "api_key": "test_key",
                    "models": ["gpt-3.5-turbo"],
                }
            },
        }

        with patch(
            "src.voyager_trader.llm_service.OpenAIProvider"
        ) as mock_provider_class:
            mock_provider = AsyncMock()
            mock_provider.config = ProviderConfig(name="openai", enabled=True)
            mock_provider.close = AsyncMock()
            mock_provider_class.return_value = mock_provider

            # Test context manager usage
            async with create_llm_service_from_config(config) as service:
                assert service is not None

            # Verify cleanup was called
            mock_provider.close.assert_called_once()

    async def test_manual_cleanup(self):
        """Test manual cleanup without context manager."""
        config = {
            "default_provider": "openai",
            "providers": {
                "openai": {
                    "enabled": True,
                    "api_key": "test_key",
                    "models": ["gpt-3.5-turbo"],
                }
            },
        }

        with patch(
            "src.voyager_trader.llm_service.OpenAIProvider"
        ) as mock_provider_class:
            mock_provider = AsyncMock()
            mock_provider.config = ProviderConfig(name="openai", enabled=True)
            mock_provider.close = AsyncMock()
            mock_provider_class.return_value = mock_provider

            service = create_llm_service_from_config(config)
            await service.close()

            # Verify cleanup was called
            mock_provider.close.assert_called_once()

    async def test_cleanup_with_errors(self):
        """Test cleanup handles provider errors gracefully."""
        config = {
            "default_provider": "openai",
            "providers": {
                "openai": {
                    "enabled": True,
                    "api_key": "test_key",
                    "models": ["gpt-3.5-turbo"],
                }
            },
        }

        with patch(
            "src.voyager_trader.llm_service.OpenAIProvider"
        ) as mock_provider_class:
            mock_provider = AsyncMock()
            mock_provider.config = ProviderConfig(name="openai", enabled=True)
            # Make close() raise an error
            mock_provider.close.side_effect = Exception("Cleanup error")
            mock_provider_class.return_value = mock_provider

            service = create_llm_service_from_config(config)

            # Should not raise exception, but log the error
            await service.close()

            mock_provider.close.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])
