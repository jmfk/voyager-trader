"""
Tests for database circuit breaker functionality.

This module tests the CircuitBreaker class and its integration with
the DatabaseManager for preventing cascading failures.
"""

import asyncio
import tempfile
from pathlib import Path

import pytest

from src.voyager_trader.persistence.circuit_breaker import (
    CircuitBreakerConfig,
    CircuitBreakerException,
    CircuitBreakerManager,
    CircuitBreakerState,
    DatabaseCircuitBreaker,
    get_circuit_breaker_manager,
)
from src.voyager_trader.persistence.database import DatabaseManager


class TestCircuitBreakerConfig:
    """Test CircuitBreakerConfig functionality."""

    def test_default_config_values(self):
        """Test default configuration values."""
        config = CircuitBreakerConfig()

        assert config.failure_threshold == 5
        assert config.failure_rate_threshold == 0.5
        assert config.minimum_requests == 10
        assert config.timeout_duration == 60.0
        assert config.half_open_max_requests == 3
        assert config.sliding_window_size == 100
        assert config.sliding_window_duration == 300.0
        assert config.recovery_timeout == 30.0
        assert config.success_threshold == 3
        assert config.metrics_enabled is True
        assert config.log_state_changes is True

    def test_custom_config_values(self):
        """Test custom configuration values."""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            failure_rate_threshold=0.3,
            timeout_duration=30.0,
            minimum_requests=5,
        )

        assert config.failure_threshold == 3
        assert config.failure_rate_threshold == 0.3
        assert config.timeout_duration == 30.0
        assert config.minimum_requests == 5


class TestDatabaseCircuitBreaker:
    """Test DatabaseCircuitBreaker functionality."""

    @pytest.fixture
    def circuit_breaker_config(self):
        """Create test circuit breaker configuration."""
        return CircuitBreakerConfig(
            failure_threshold=2,
            minimum_requests=2,  # Require at least 2 requests before evaluating
            timeout_duration=1.0,
            half_open_max_requests=2,
            success_threshold=1,
        )

    @pytest.fixture
    def circuit_breaker(self, circuit_breaker_config):
        """Create DatabaseCircuitBreaker for testing."""
        return DatabaseCircuitBreaker("test_cb", circuit_breaker_config)

    def test_initial_state(self, circuit_breaker):
        """Test initial circuit breaker state."""
        assert circuit_breaker.name == "test_cb"
        assert circuit_breaker.state == CircuitBreakerState.CLOSED
        assert circuit_breaker.is_closed is True
        assert circuit_breaker.is_open is False
        assert circuit_breaker.is_half_open is False
        assert circuit_breaker.metrics.total_requests == 0
        assert circuit_breaker.metrics.failed_requests == 0

    @pytest.mark.asyncio
    async def test_successful_call(self, circuit_breaker):
        """Test successful function call through circuit breaker."""

        async def test_func():
            return "success"

        result = await circuit_breaker.call(test_func)

        assert result == "success"
        assert circuit_breaker.metrics.total_requests == 1
        assert circuit_breaker.metrics.successful_requests == 1
        assert circuit_breaker.metrics.failed_requests == 0
        assert circuit_breaker.state == CircuitBreakerState.CLOSED

    @pytest.mark.asyncio
    async def test_failed_call(self, circuit_breaker):
        """Test failed function call through circuit breaker."""

        async def failing_func():
            raise Exception("Test error")

        with pytest.raises(Exception, match="Test error"):
            await circuit_breaker.call(failing_func)

        assert circuit_breaker.metrics.total_requests == 1
        assert circuit_breaker.metrics.successful_requests == 0
        assert circuit_breaker.metrics.failed_requests == 1
        assert (
            circuit_breaker.state == CircuitBreakerState.CLOSED
        )  # Still closed, need more failures

    @pytest.mark.asyncio
    async def test_circuit_opens_after_failures(self, circuit_breaker):
        """Test circuit breaker opens after failure threshold."""

        async def failing_func():
            raise Exception("Test error")

        # First failure
        with pytest.raises(Exception):
            await circuit_breaker.call(failing_func)
        assert circuit_breaker.state == CircuitBreakerState.CLOSED

        # Second failure should open circuit
        with pytest.raises(Exception):
            await circuit_breaker.call(failing_func)
        assert circuit_breaker.state == CircuitBreakerState.OPEN

    @pytest.mark.asyncio
    async def test_circuit_rejects_when_open(self, circuit_breaker):
        """Test circuit breaker rejects requests when open."""

        async def failing_func():
            raise Exception("Test error")

        # Trigger failures to open circuit
        for _ in range(2):
            with pytest.raises(Exception):
                await circuit_breaker.call(failing_func)

        assert circuit_breaker.state == CircuitBreakerState.OPEN

        # Now requests should be rejected
        async def any_func():
            return "should not execute"

        with pytest.raises(CircuitBreakerException):
            await circuit_breaker.call(any_func)

        assert circuit_breaker.metrics.rejected_requests == 1

    @pytest.mark.asyncio
    async def test_half_open_state_transition(self, circuit_breaker):
        """Test transition from OPEN to HALF_OPEN state."""

        async def failing_func():
            raise Exception("Test error")

        # Open the circuit
        for _ in range(2):
            with pytest.raises(Exception):
                await circuit_breaker.call(failing_func)

        assert circuit_breaker.state == CircuitBreakerState.OPEN

        # Wait for timeout and try again
        await asyncio.sleep(1.1)  # Longer than timeout_duration

        async def test_func():
            return "test"

        # This should transition to half-open and succeed
        result = await circuit_breaker.call(test_func)
        assert result == "test"
        assert (
            circuit_breaker.state == CircuitBreakerState.CLOSED
        )  # Success should close circuit

    @pytest.mark.asyncio
    async def test_half_open_to_closed_on_success(self, circuit_breaker):
        """Test transition from HALF_OPEN to CLOSED on success."""

        # Open the circuit first
        async def failing_func():
            raise Exception("Test error")

        for _ in range(2):
            with pytest.raises(Exception):
                await circuit_breaker.call(failing_func)

        # Manually set to half-open for testing
        await circuit_breaker._transition_to_half_open()
        assert circuit_breaker.state == CircuitBreakerState.HALF_OPEN

        # Successful call should close circuit (success_threshold=1)
        async def success_func():
            return "success"

        result = await circuit_breaker.call(success_func)
        assert result == "success"
        assert circuit_breaker.state == CircuitBreakerState.CLOSED

    @pytest.mark.asyncio
    async def test_half_open_to_open_on_failure(self, circuit_breaker):
        """Test transition from HALF_OPEN back to OPEN on failure."""

        # Open the circuit first
        async def failing_func():
            raise Exception("Test error")

        for _ in range(2):
            with pytest.raises(Exception):
                await circuit_breaker.call(failing_func)

        # Manually set to half-open
        await circuit_breaker._transition_to_half_open()
        assert circuit_breaker.state == CircuitBreakerState.HALF_OPEN

        # Failed call should return to open
        with pytest.raises(Exception):
            await circuit_breaker.call(failing_func)

        assert circuit_breaker.state == CircuitBreakerState.OPEN

    @pytest.mark.asyncio
    async def test_circuit_breaker_reset(self, circuit_breaker):
        """Test manual circuit breaker reset."""

        # Open the circuit
        async def failing_func():
            raise Exception("Test error")

        for _ in range(2):
            with pytest.raises(Exception):
                await circuit_breaker.call(failing_func)

        assert circuit_breaker.state == CircuitBreakerState.OPEN

        # Reset should return to closed
        await circuit_breaker.reset()
        assert circuit_breaker.state == CircuitBreakerState.CLOSED

    @pytest.mark.asyncio
    async def test_failure_rate_threshold(self):
        """Test circuit breaker opens based on failure rate."""
        config = CircuitBreakerConfig(
            failure_threshold=10,  # High absolute threshold
            failure_rate_threshold=0.6,  # 60% failure rate
            minimum_requests=5,
        )
        cb = DatabaseCircuitBreaker("rate_test", config)

        async def sometimes_fails(should_fail=False):
            if should_fail:
                raise Exception("Failure")
            return "success"

        # Mix of successes and failures - 3 successes, 2 failures (40% failure rate)
        await cb.call(sometimes_fails, should_fail=False)
        await cb.call(sometimes_fails, should_fail=False)
        await cb.call(sometimes_fails, should_fail=False)

        with pytest.raises(Exception):
            await cb.call(sometimes_fails, should_fail=True)
        with pytest.raises(Exception):
            await cb.call(sometimes_fails, should_fail=True)

        # Should still be closed (40% < 60% threshold)
        assert cb.state == CircuitBreakerState.CLOSED

        # Add more failures to exceed rate threshold
        with pytest.raises(Exception):
            await cb.call(sometimes_fails, should_fail=True)
        with pytest.raises(Exception):
            await cb.call(sometimes_fails, should_fail=True)

        # Now should be open (4/7 = 57% failure rate, getting close)
        # Let's add one more failure to definitely exceed 60%
        with pytest.raises(Exception):
            await cb.call(sometimes_fails, should_fail=True)

        # Should be open now (5/8 = 62.5% > 60%)
        assert cb.state == CircuitBreakerState.OPEN

    def test_get_statistics(self, circuit_breaker):
        """Test circuit breaker statistics."""
        stats = circuit_breaker.get_statistics()

        assert stats["name"] == "test_cb"
        assert stats["state"] == "closed"
        assert stats["total_requests"] == 0
        assert stats["successful_requests"] == 0
        assert stats["failed_requests"] == 0
        assert stats["rejected_requests"] == 0
        assert stats["failure_rate"] == 0.0
        assert stats["success_rate"] == 1.0
        assert "config" in stats
        assert stats["config"]["failure_threshold"] == 2


class TestCircuitBreakerManager:
    """Test CircuitBreakerManager functionality."""

    @pytest.fixture
    def manager(self):
        """Create CircuitBreakerManager for testing."""
        return CircuitBreakerManager()

    def test_create_circuit_breaker(self, manager):
        """Test creating circuit breakers."""
        config = CircuitBreakerConfig(failure_threshold=3)
        cb = manager.create_circuit_breaker("test1", config)

        assert cb.name == "test1"
        assert cb.config.failure_threshold == 3
        assert len(manager.list_circuit_breakers()) == 1

    def test_get_circuit_breaker(self, manager):
        """Test retrieving circuit breakers."""
        cb = manager.create_circuit_breaker("test2")
        retrieved = manager.get_circuit_breaker("test2")

        assert retrieved is cb
        assert manager.get_circuit_breaker("nonexistent") is None

    def test_get_or_create_circuit_breaker(self, manager):
        """Test get or create functionality."""
        # Should create new one
        cb1 = manager.get_or_create_circuit_breaker("test3")
        assert cb1.name == "test3"

        # Should return existing one
        cb2 = manager.get_or_create_circuit_breaker("test3")
        assert cb2 is cb1

    def test_duplicate_circuit_breaker_name(self, manager):
        """Test error when creating duplicate circuit breaker names."""
        manager.create_circuit_breaker("duplicate")

        with pytest.raises(ValueError, match="already exists"):
            manager.create_circuit_breaker("duplicate")

    @pytest.mark.asyncio
    async def test_reset_all(self, manager):
        """Test resetting all circuit breakers."""
        cb1 = manager.create_circuit_breaker("cb1")
        cb2 = manager.create_circuit_breaker("cb2")

        # Open both circuits
        async def fail():
            raise Exception("fail")

        for cb in [cb1, cb2]:
            await cb._transition_to_open()
            assert cb.state == CircuitBreakerState.OPEN

        # Reset all
        await manager.reset_all()

        assert cb1.state == CircuitBreakerState.CLOSED
        assert cb2.state == CircuitBreakerState.CLOSED

    def test_global_statistics(self, manager):
        """Test global circuit breaker statistics."""
        cb1 = manager.create_circuit_breaker("cb1")
        manager.create_circuit_breaker("cb2")

        # Open one circuit breaker
        cb1._state = CircuitBreakerState.OPEN

        stats = manager.get_global_statistics()

        assert stats["total_breakers"] == 2
        assert stats["open_breakers"] == 1
        assert stats["closed_breakers"] == 1
        assert stats["half_open_breakers"] == 0


class TestDatabaseManagerCircuitBreakerIntegration:
    """Test DatabaseManager integration with circuit breakers."""

    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database path for testing."""
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        temp_path = temp_file.name
        temp_file.close()

        yield temp_path

        # Cleanup
        Path(temp_path).unlink(missing_ok=True)

    @pytest.fixture
    def circuit_breaker_config(self):
        """Create test circuit breaker configuration."""
        return CircuitBreakerConfig(
            failure_threshold=2,
            minimum_requests=1,
            timeout_duration=0.5,
        )

    @pytest.mark.asyncio
    async def test_database_manager_with_circuit_breaker(
        self, temp_db_path, circuit_breaker_config
    ):
        """Test DatabaseManager initialization with circuit breaker."""
        from src.voyager_trader.persistence.connection_health import HealthCheckConfig

        health_config = HealthCheckConfig(enabled=False)

        db_manager = DatabaseManager(
            database_url=f"sqlite:///{temp_db_path}",
            pool_size=1,
            health_check_config=health_config,
            circuit_breaker_config=circuit_breaker_config,
        )

        assert db_manager._circuit_breaker is not None
        assert db_manager._circuit_breaker.state == CircuitBreakerState.CLOSED

        await db_manager.close()

    @pytest.mark.asyncio
    async def test_circuit_breaker_protects_get_connection(
        self, temp_db_path, circuit_breaker_config
    ):
        """Test circuit breaker protects get_connection calls."""
        from src.voyager_trader.persistence.connection_health import HealthCheckConfig

        health_config = HealthCheckConfig(enabled=False)

        db_manager = DatabaseManager(
            database_url=f"sqlite:///{temp_db_path}",
            pool_size=1,
            health_check_config=health_config,
            circuit_breaker_config=circuit_breaker_config,
        )

        # Skip full initialization to avoid schema issues
        db_manager._initialized = True

        try:
            # Normal operation should work
            async with db_manager.get_connection() as conn:
                cursor = await conn.execute("SELECT 1")
                result = await cursor.fetchone()
                assert result == (1,)

            # Check circuit breaker recorded success
            stats = await db_manager.get_connection_stats()
            assert stats["circuit_breaker"]["total_requests"] == 1
            assert stats["circuit_breaker"]["successful_requests"] == 1

        finally:
            await db_manager.close()

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_on_connection_failures(
        self, temp_db_path, circuit_breaker_config
    ):
        """Test circuit breaker opens when connection creation fails."""
        from src.voyager_trader.persistence.connection_health import HealthCheckConfig

        health_config = HealthCheckConfig(enabled=False)

        db_manager = DatabaseManager(
            database_url="sqlite:///nonexistent/path/database.db",  # Invalid path
            pool_size=1,
            health_check_config=health_config,
            circuit_breaker_config=circuit_breaker_config,
        )

        # Skip initialization
        db_manager._initialized = True

        try:
            # First failure
            with pytest.raises(Exception):  # Connection creation will fail
                async with db_manager.get_connection():
                    pass

            # Second failure should open circuit
            with pytest.raises(Exception):
                async with db_manager.get_connection():
                    pass

            # Circuit should be open now
            assert db_manager._circuit_breaker.state == CircuitBreakerState.OPEN

            # Third attempt should be rejected by circuit breaker
            with pytest.raises(CircuitBreakerException):
                async with db_manager.get_connection():
                    pass

        finally:
            await db_manager.close()

    @pytest.mark.asyncio
    async def test_circuit_breaker_statistics_in_connection_stats(
        self, temp_db_path, circuit_breaker_config
    ):
        """Test circuit breaker statistics included in connection stats."""
        from src.voyager_trader.persistence.connection_health import HealthCheckConfig

        health_config = HealthCheckConfig(enabled=False)

        db_manager = DatabaseManager(
            database_url=f"sqlite:///{temp_db_path}",
            pool_size=1,
            health_check_config=health_config,
            circuit_breaker_config=circuit_breaker_config,
        )

        db_manager._initialized = True

        try:
            stats = await db_manager.get_connection_stats()

            # Should include circuit breaker stats
            assert "circuit_breaker" in stats
            cb_stats = stats["circuit_breaker"]
            assert "state" in cb_stats
            assert "total_requests" in cb_stats
            assert "failure_rate" in cb_stats
            assert "config" in cb_stats

        finally:
            await db_manager.close()


class TestGlobalCircuitBreakerManager:
    """Test global circuit breaker manager."""

    def test_get_global_manager(self):
        """Test getting global circuit breaker manager."""
        manager1 = get_circuit_breaker_manager()
        manager2 = get_circuit_breaker_manager()

        # Should return same instance
        assert manager1 is manager2

    def test_global_manager_persistence(self):
        """Test that circuit breakers persist across manager calls."""
        manager1 = get_circuit_breaker_manager()
        cb = manager1.create_circuit_breaker("persistent_test")

        manager2 = get_circuit_breaker_manager()
        retrieved = manager2.get_circuit_breaker("persistent_test")

        assert retrieved is cb


class TestCircuitBreakerEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_with_sync_function(self):
        """Test circuit breaker with synchronous function."""
        config = CircuitBreakerConfig(failure_threshold=1)
        cb = DatabaseCircuitBreaker("sync_test", config)

        def sync_func():
            return "sync_result"

        result = await cb.call(sync_func)
        assert result == "sync_result"

    @pytest.mark.asyncio
    async def test_concurrent_circuit_breaker_calls(self):
        """Test concurrent calls through circuit breaker."""
        config = CircuitBreakerConfig(failure_threshold=5)
        cb = DatabaseCircuitBreaker("concurrent_test", config)

        async def test_func(delay=0.1):
            await asyncio.sleep(delay)
            return "concurrent_result"

        # Execute concurrent calls
        tasks = [cb.call(test_func, delay=0.05) for _ in range(10)]
        results = await asyncio.gather(*tasks)

        assert all(result == "concurrent_result" for result in results)
        assert cb.metrics.total_requests == 10
        assert cb.metrics.successful_requests == 10

    @pytest.mark.asyncio
    async def test_circuit_breaker_exception_details(self):
        """Test CircuitBreakerException includes proper details."""
        config = CircuitBreakerConfig(failure_threshold=1)
        cb = DatabaseCircuitBreaker("exception_test", config)

        # Open the circuit
        async def fail():
            raise Exception("fail")

        with pytest.raises(Exception):
            await cb.call(fail)

        assert cb.state == CircuitBreakerState.OPEN

        # Now test the exception details
        with pytest.raises(CircuitBreakerException) as exc_info:
            await cb.call(lambda: "test")

        assert "exception_test" in str(exc_info.value)
        assert exc_info.value.state == CircuitBreakerState.OPEN
        assert exc_info.value.metrics is cb.metrics
