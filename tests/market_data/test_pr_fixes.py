"""Test the PR review fixes to ensure they work correctly."""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import pytest

from src.voyager_trader.market_data.cache import DataCache
from src.voyager_trader.market_data.exceptions import (
    AuthenticationError,
    ConnectionError,
    DataSourceError,
    MarketDataError,
    RateLimitError,
    create_http_error,
)
from src.voyager_trader.market_data.monitoring import (
    MetricsCollector,
    record_request_metrics,
)
from src.voyager_trader.market_data.rate_limiter import RateLimiter
from src.voyager_trader.market_data.types import (
    Symbol,
    create_symbol,
    ensure_symbol_object,
    normalize_symbol,
)
from src.voyager_trader.models.types import AssetClass


class TestSymbolTypeHandling:
    """Test centralized Symbol type handling."""

    def test_symbol_creation(self):
        """Test symbol creation from string."""
        symbol = create_symbol("AAPL")
        assert symbol.code == "AAPL"
        assert symbol.asset_class == AssetClass.EQUITY

        # Test with default asset class (EQUITY is the default)
        crypto_symbol = create_symbol("BTC-USD")
        assert crypto_symbol.code == "BTC-USD"
        assert crypto_symbol.asset_class == AssetClass.EQUITY

    def test_symbol_normalization(self):
        """Test symbol normalization to string."""
        # String input
        assert normalize_symbol("AAPL") == "AAPL"

        # Symbol object input
        symbol_obj = create_symbol("GOOGL")
        assert normalize_symbol(symbol_obj) == "GOOGL"

    def test_ensure_symbol_object(self):
        """Test ensuring we have a Symbol object."""
        # String input
        result = ensure_symbol_object("MSFT")
        assert result.code == "MSFT"
        assert result.asset_class == AssetClass.EQUITY

        # Symbol object input
        original = create_symbol("TSLA")
        result = ensure_symbol_object(original)
        assert result is original


class TestExceptionHierarchy:
    """Test the new exception hierarchy."""

    def test_base_exception(self):
        """Test base MarketDataError."""
        error = MarketDataError("Test error", provider="test", symbol="AAPL")
        assert str(error) == "Test error"
        assert error.provider == "test"
        assert error.symbol == "AAPL"
        assert error.details == {}

    def test_data_source_error(self):
        """Test DataSourceError inheritance."""
        error = DataSourceError("Source error", provider="alpha_vantage")
        assert isinstance(error, MarketDataError)
        assert error.provider == "alpha_vantage"

    def test_rate_limit_error(self):
        """Test RateLimitError with specific fields."""
        error = RateLimitError("Rate limit exceeded", provider="yahoo", retry_after=60)
        assert isinstance(error, DataSourceError)
        assert error.retry_after == 60

    def test_http_error_creation(self):
        """Test HTTP error creation by status code."""
        # 401 -> AuthenticationError
        error = create_http_error(401, "Unauthorized", provider="test")
        assert isinstance(error, AuthenticationError)

        # 429 -> RateLimitError
        error = create_http_error(429, "Too many requests", provider="test")
        assert isinstance(error, RateLimitError)

        # 500 -> DataSourceError
        error = create_http_error(500, "Server error", provider="test")
        assert isinstance(error, DataSourceError)


class TestRateLimiterEnhancements:
    """Test rate limiter exponential backoff and burst capacity."""

    @pytest.fixture
    def rate_limiter(self):
        """Create a rate limiter for testing."""
        return RateLimiter()

    def test_exponential_backoff_tracking(self, rate_limiter):
        """Test failure tracking for exponential backoff."""
        provider = "test_provider"

        # Record failures
        rate_limiter.record_failure(provider)
        assert rate_limiter._failure_counts[provider] == 1

        rate_limiter.record_failure(provider)
        assert rate_limiter._failure_counts[provider] == 2

        # Record success should reduce count
        rate_limiter.record_success(provider)
        assert rate_limiter._failure_counts[provider] == 1

        # Another success should clear it
        rate_limiter.record_success(provider)
        assert rate_limiter._failure_counts[provider] == 0

    def test_burst_capacity_setting(self, rate_limiter):
        """Test burst capacity configuration."""
        provider = "test_provider"
        rate_limiter.set_limit(provider, 60, 5, burst_capacity=10)

        assert rate_limiter._burst_capacity[provider] == 10
        assert rate_limiter._burst_tokens[provider] == 10

    @pytest.mark.asyncio
    async def test_exponential_backoff_calculation(self, rate_limiter):
        """Test that exponential backoff works correctly."""
        import time

        provider = "test_provider"
        rate_limiter.set_limit(provider, 60, 1)

        # Simulate multiple failures that occurred a short time ago
        rate_limiter._failure_counts[provider] = 2
        # Set failure time to 1 second ago to ensure backoff triggers
        rate_limiter._last_failure[provider] = time.time() - 1.0

        # This should trigger backoff (2^2 = 4 seconds backoff, minus 1 second elapsed = ~3 seconds remaining)
        start_time = time.time()
        await rate_limiter.acquire(provider)
        end_time = time.time()

        # Should have waited for remaining backoff time (around 3+ seconds)
        duration = end_time - start_time
        assert duration > 2.5  # Should have significant delay due to backoff


class TestMetricsCollection:
    """Test metrics collection functionality."""

    def test_metrics_collector_creation(self):
        """Test creating a metrics collector."""
        collector = MetricsCollector(retention_hours=12)
        assert collector.retention_hours == 12
        assert len(collector._provider_metrics) == 0

    def test_request_recording(self):
        """Test recording request metrics."""
        collector = MetricsCollector()

        collector.record_request(
            provider="test",
            method="get_data",
            symbol="AAPL",
            duration_ms=150.5,
            success=True,
            cache_hit=False,
            data_size_bytes=1024,
        )

        assert len(collector._recent_requests) == 1
        assert "test" in collector._provider_metrics

        metrics = collector._provider_metrics["test"]
        assert metrics.total_requests == 1
        assert metrics.successful_requests == 1
        assert metrics.failed_requests == 0
        assert metrics.total_data_bytes == 1024

    def test_failure_recording(self):
        """Test recording failures."""
        collector = MetricsCollector()

        collector.record_request(
            provider="test",
            method="get_data",
            success=False,
            error_type="ConnectionError",
        )

        metrics = collector._provider_metrics["test"]
        assert metrics.failed_requests == 1
        assert metrics.current_failure_streak == 1
        assert len(metrics.recent_errors) == 1

    def test_system_metrics(self):
        """Test system-wide metrics."""
        collector = MetricsCollector()

        # Record some data
        collector.record_request(
            "provider1", "method1", success=True, data_size_bytes=500
        )
        collector.record_request("provider2", "method2", success=False)

        system_metrics = collector.get_system_metrics()

        assert system_metrics["total_requests"] == 2
        assert system_metrics["successful_requests"] == 1
        assert system_metrics["failed_requests"] == 1
        assert system_metrics["error_rate_percent"] == 50.0
        assert system_metrics["active_providers"] == 2

    def test_health_status(self):
        """Test health status calculation."""
        collector = MetricsCollector()

        # Healthy provider
        collector.record_request("healthy", "test", success=True, duration_ms=100)

        # Unhealthy provider with high error rate
        for _ in range(10):
            collector.record_request("unhealthy", "test", success=False)

        health = collector.get_health_status()

        assert not health[
            "overall_healthy"
        ]  # Should be unhealthy due to one bad provider
        assert health["providers"]["healthy"]["healthy"] is True
        assert health["providers"]["unhealthy"]["healthy"] is False
        assert len(health["alerts"]) > 0


class TestCacheKeyCollisionResistance:
    """Test cache key collision resistance."""

    def test_cache_key_generation(self):
        """Test that cache keys are collision-resistant."""
        cache = DataCache()

        # Test with different parameters that could collide
        key1 = cache._generate_key(
            "source1", "method1", ("arg1", "arg2"), {"param": "value"}
        )
        key2 = cache._generate_key(
            "source1", "method1", ("arg1", "arg3"), {"param": "value"}
        )
        key3 = cache._generate_key(
            "source2", "method1", ("arg1", "arg2"), {"param": "value"}
        )

        # All keys should be different
        assert key1 != key2
        assert key1 != key3
        assert key2 != key3

        # Keys should be consistent for same parameters
        key1_repeat = cache._generate_key(
            "source1", "method1", ("arg1", "arg2"), {"param": "value"}
        )
        assert key1 == key1_repeat

    def test_cache_key_length_and_format(self):
        """Test cache key format and length."""
        cache = DataCache()

        key = cache._generate_key("source", "method", ("arg",), {"param": "value"})

        # Should be in format: hash_uuid
        assert "_" in key
        parts = key.split("_")
        assert len(parts) == 2
        assert len(parts[0]) == 32  # SHA-256 truncated to 32 chars
        assert len(parts[1]) == 8  # UUID truncated to 8 chars


class TestConnectionPooling:
    """Test connection pooling configuration."""

    @pytest.mark.asyncio
    async def test_alpha_vantage_session_config(self):
        """Test Alpha Vantage session has proper connection pooling."""
        from src.voyager_trader.market_data.sources.alpha_vantage import (
            AlphaVantageDataSource,
        )

        # Mock the config to avoid needing real API key
        source = AlphaVantageDataSource({"api_key": "test_key"})

        session = await source._get_session()

        # Check that session was created with proper configuration
        assert session is not None
        assert session.connector is not None

        # Check connector settings (these should be set by our connection pooling)
        connector = session.connector
        assert hasattr(connector, "_limit")  # Should have connection limits

        # Clean up
        await source.close()


# Integration test for convenience functions
def test_convenience_functions():
    """Test convenience functions work correctly."""
    # This mainly tests that imports work and functions are accessible
    record_request_metrics(
        provider="test", method="test_method", symbol="AAPL", success=True
    )

    # Should not raise any errors
    from src.voyager_trader.market_data import get_health_status, get_system_metrics

    metrics = get_system_metrics()
    assert isinstance(metrics, dict)
    assert "total_requests" in metrics

    health = get_health_status()
    assert isinstance(health, dict)
    assert "overall_healthy" in health
