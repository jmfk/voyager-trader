#!/usr/bin/env python3
"""
Examples demonstrating the centralized LLM service usage.

These examples show how to use the LLM service for various scenarios
including basic completions, streaming, provider selection, and integration
with the VOYAGER-Trader system.
"""

import asyncio
import json
from typing import Dict, List

from src.voyager_trader.llm_service import (  # High-level convenience functions; Service classes; Configuration; Exceptions
    LLMError,
    LLMRequest,
    LLMResponse,
    LLMService,
    ModelNotAvailableError,
    ProviderError,
    RateLimitError,
    UniversalLLMClient,
    chat_completion_create,
    create_default_llm_service,
    create_llm_service_from_config,
    get_global_llm_client,
    get_global_llm_service,
)


async def example_basic_chat_completion():
    """Basic chat completion using convenience function."""
    print("=== Basic Chat Completion ===")

    try:
        response = await chat_completion_create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful trading assistant."},
                {
                    "role": "user",
                    "content": "What are the key metrics for evaluating a trading strategy?",
                },
            ],
            temperature=0.7,
            max_tokens=500,
        )

        print(f"Model: {response.model}")
        print(f"Response: {response.choices[0].message.content}")
        print(f"Tokens used: {response.usage.total_tokens}")

    except Exception as e:
        print(f"Error: {e}")


async def example_provider_selection():
    """Example of selecting specific providers."""
    print("\n=== Provider Selection ===")

    client = get_global_llm_client()

    # Try different providers
    providers_to_try = ["openai", "anthropic", "ollama"]

    for provider in providers_to_try:
        try:
            print(f"\nTrying {provider}...")

            # Select appropriate model for provider
            model_map = {
                "openai": "gpt-3.5-turbo",
                "anthropic": "claude-3-haiku",
                "ollama": "llama2",
            }

            response = await client.chat.create(
                model=model_map[provider],
                messages=[{"role": "user", "content": "Hello! How are you?"}],
                provider=provider,
                max_tokens=100,
            )

            print(
                f"✓ {provider} response: {response.choices[0].message.content[:50]}..."
            )

        except Exception as e:
            print(f"✗ {provider} failed: {e}")


async def example_streaming_response():
    """Example of streaming responses."""
    print("\n=== Streaming Response ===")

    try:
        client = get_global_llm_client()

        print("Streaming response from GPT-3.5-turbo:")

        stream = await client.chat.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "user",
                    "content": "Write a short story about algorithmic trading in exactly 3 sentences.",
                }
            ],
            stream=True,
            max_tokens=200,
        )

        full_response = ""
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
                full_response += content

        print(f"\n\nComplete response length: {len(full_response)} characters")

    except Exception as e:
        print(f"Streaming error: {e}")


async def example_custom_configuration():
    """Example of using custom service configuration."""
    print("\n=== Custom Configuration ===")

    # Create custom config favoring local models
    config = {
        "default_provider": "ollama",
        "fallback_chain": ["ollama", "openai", "anthropic"],
        "providers": {
            "ollama": {
                "enabled": True,
                "base_url": "http://localhost:11434",
                "models": ["llama2", "codellama", "mistral"],
                "timeout": 120,
            },
            "openai": {"enabled": True, "models": ["gpt-3.5-turbo", "gpt-4"]},
            "anthropic": {
                "enabled": True,
                "models": ["claude-3-haiku", "claude-3-sonnet"],
            },
        },
    }

    try:
        service = create_llm_service_from_config(config)

        request = LLMRequest(
            messages=[
                {
                    "role": "user",
                    "content": "Explain the concept of mean reversion in trading.",
                }
            ],
            model="llama2",  # Will try Ollama first
            temperature=0.8,
            max_tokens=300,
        )

        response = await service.generate(request)

        print(f"Response from {response.provider}:")
        print(f"Model: {response.model}")
        print(f"Content: {response.content[:100]}...")

    except Exception as e:
        print(f"Custom config error: {e}")


async def example_provider_health_monitoring():
    """Example of monitoring provider health."""
    print("\n=== Provider Health Monitoring ===")

    service = get_global_llm_service()

    # Get provider status
    status = service.get_provider_status()

    print("Provider Status:")
    for provider_name, provider_status in status.items():
        health = "🟢" if provider_status.get("healthy", False) else "🔴"
        enabled = "✓" if provider_status.get("enabled", False) else "✗"

        print(f"  {health} {provider_name}: {enabled} enabled")

        models = provider_status.get("models", [])
        if models:
            print(
                f"     Models: {', '.join(models[:3])}{'...' if len(models) > 3 else ''}"
            )

    # Get available models across all providers
    print("\nAvailable Models by Provider:")
    available_models = service.get_available_models()

    for provider, models in available_models.items():
        print(f"  {provider}: {len(models)} models")
        for model in models[:3]:  # Show first 3
            print(f"    - {model}")
        if len(models) > 3:
            print(f"    ... and {len(models) - 3} more")


async def example_error_handling():
    """Example of proper error handling."""
    print("\n=== Error Handling ===")

    scenarios = [
        # Scenario 1: Invalid model
        {
            "description": "Invalid model name",
            "model": "gpt-999-invalid",
            "messages": [{"role": "user", "content": "Hello"}],
        },
        # Scenario 2: Unavailable provider
        {
            "description": "Unavailable provider",
            "model": "gpt-3.5-turbo",
            "provider": "invalid_provider",
            "messages": [{"role": "user", "content": "Hello"}],
        },
    ]

    for scenario in scenarios:
        print(f"\nTesting: {scenario['description']}")

        try:
            response = await chat_completion_create(**scenario)
            print(
                f"✓ Unexpected success: {response.choices[0].message.content[:50]}..."
            )

        except ModelNotAvailableError as e:
            print(f"✓ Caught ModelNotAvailableError: {e}")

        except ProviderError as e:
            print(f"✓ Caught ProviderError: {e}")

        except RateLimitError as e:
            print(f"✓ Caught RateLimitError: {e}")

        except LLMError as e:
            print(f"✓ Caught general LLMError: {e}")

        except Exception as e:
            print(f"✗ Unexpected error: {type(e).__name__}: {e}")


async def example_trading_assistant():
    """Example of using LLM service for trading-specific tasks."""
    print("\n=== Trading Assistant Example ===")

    # Trading-specific prompts
    trading_prompts = [
        {
            "task": "Risk Assessment",
            "prompt": "Analyze the risk factors for a momentum trading strategy in volatile markets. Provide 3 key risks and mitigation strategies.",
        },
        {
            "task": "Technical Analysis",
            "prompt": "Explain how to combine RSI and MACD indicators for entry and exit signals in day trading.",
        },
        {
            "task": "Strategy Generation",
            "prompt": "Design a simple mean reversion strategy for cryptocurrency trading. Include entry/exit criteria and position sizing.",
        },
    ]

    for task_info in trading_prompts:
        print(f"\n--- {task_info['task']} ---")

        try:
            response = await chat_completion_create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert quantitative trading analyst with deep knowledge of financial markets and algorithmic trading strategies.",
                    },
                    {"role": "user", "content": task_info["prompt"]},
                ],
                temperature=0.3,  # Lower temperature for more focused responses
                max_tokens=400,
            )

            content = response.choices[0].message.content
            print(content[:200] + "..." if len(content) > 200 else content)
            print(f"(Used {response.usage.total_tokens} tokens)")

        except Exception as e:
            print(f"Error: {e}")


async def example_local_vs_remote_comparison():
    """Compare responses from local vs remote models."""
    print("\n=== Local vs Remote Model Comparison ===")

    question = "What is the difference between a bull market and a bear market?"

    model_configs = [
        {"name": "OpenAI GPT-3.5", "model": "gpt-3.5-turbo", "provider": "openai"},
        {
            "name": "Anthropic Claude",
            "model": "claude-3-haiku",
            "provider": "anthropic",
        },
        {"name": "Local Llama2", "model": "llama2", "provider": "ollama"},
    ]

    client = get_global_llm_client()

    for config in model_configs:
        print(f"\n--- {config['name']} ---")

        try:
            import time

            start_time = time.time()

            response = await client.chat.create(
                model=config["model"],
                messages=[{"role": "user", "content": question}],
                provider=config["provider"],
                max_tokens=150,
                temperature=0.7,
            )

            end_time = time.time()
            response_time = end_time - start_time

            content = response.choices[0].message.content
            print(f"Response ({response_time:.2f}s): {content[:100]}...")

        except Exception as e:
            print(f"Error with {config['name']}: {e}")


async def main():
    """Run all examples."""
    print("🚀 LLM Service Examples")
    print("=" * 50)

    examples = [
        example_basic_chat_completion,
        example_provider_selection,
        example_streaming_response,
        example_custom_configuration,
        example_provider_health_monitoring,
        example_error_handling,
        example_trading_assistant,
        example_local_vs_remote_comparison,
    ]

    for example_func in examples:
        try:
            await example_func()
            await asyncio.sleep(1)  # Brief pause between examples

        except KeyboardInterrupt:
            print("\n\nExamples interrupted by user.")
            break

        except Exception as e:
            print(f"\n❌ Example {example_func.__name__} failed: {e}")

    print("\n✅ All examples completed!")


if __name__ == "__main__":
    # Run the examples
    asyncio.run(main())
