"""
Tests for database connection health checking functionality.

This module tests the HealthyConnection class, ConnectionHealthManager,
and DatabaseManager integration with health checking.
"""

import asyncio
import sqlite3
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import aiosqlite
import pytest

from src.voyager_trader.persistence.connection_health import (
    ConnectionHealthManager,
    ConnectionStatus,
    HealthCheckConfig,
    HealthyConnection,
)
from src.voyager_trader.persistence.database import DatabaseManager


class TestHealthCheckConfig:
    """Test HealthCheckConfig functionality."""

    def test_default_config_values(self):
        """Test default configuration values."""
        config = HealthCheckConfig()

        assert config.enabled is True
        assert config.timeout == 1.0
        assert config.query == "SELECT 1"
        assert config.interval == 30
        assert config.max_age == 3600
        assert config.max_usage == 1000
        assert config.max_idle_time == 300
        assert config.max_consecutive_failures == 3
        assert config.health_check_cache_duration == 5.0
        assert config.min_healthy_connections == 2
        assert config.health_check_batch_size == 5

    def test_custom_config_values(self):
        """Test custom configuration values."""
        config = HealthCheckConfig(
            enabled=False,
            timeout=2.0,
            query="PRAGMA integrity_check(1)",
            max_age=7200,
        )

        assert config.enabled is False
        assert config.timeout == 2.0
        assert config.query == "PRAGMA integrity_check(1)"
        assert config.max_age == 7200


class TestHealthyConnection:
    """Test HealthyConnection functionality."""

    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing."""
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        temp_path = temp_file.name
        temp_file.close()

        yield temp_path

        # Cleanup
        Path(temp_path).unlink(missing_ok=True)

    @pytest.fixture
    async def db_connection(self, temp_db):
        """Create database connection for testing."""
        conn = await aiosqlite.connect(temp_db)
        yield conn
        await conn.close()

    @pytest.fixture
    def health_config(self):
        """Create test health check configuration."""
        return HealthCheckConfig(
            timeout=0.5,
            max_age=60,
            max_usage=10,
            max_idle_time=30,
        )

    @pytest.fixture
    def healthy_connection(self, db_connection, health_config):
        """Create HealthyConnection for testing."""
        return HealthyConnection(
            connection=db_connection,
            config=health_config,
            connection_id="test_conn_1",
        )

    def test_initial_state(self, healthy_connection):
        """Test initial connection state."""
        assert healthy_connection.connection_id == "test_conn_1"
        assert healthy_connection.status == ConnectionStatus.HEALTHY
        assert healthy_connection.is_healthy is True
        assert healthy_connection.is_expired is False
        assert healthy_connection.is_stale is False
        assert healthy_connection.metrics.usage_count == 0
        assert healthy_connection.metrics.health_check_count == 0

    @pytest.mark.asyncio
    async def test_health_check_success(self, healthy_connection):
        """Test successful health check."""
        result = await healthy_connection.validate_health()

        assert result is True
        assert healthy_connection.metrics.health_check_count == 1
        assert healthy_connection.metrics.health_check_failures == 0
        assert healthy_connection.status == ConnectionStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_health_check_with_mock_failure(self, health_config):
        """Test health check with connection failure."""
        # Create mock connection that fails
        mock_conn = AsyncMock()
        mock_conn.execute.side_effect = sqlite3.Error("Connection failed")

        healthy_conn = HealthyConnection(
            connection=mock_conn,
            config=health_config,
            connection_id="test_conn_fail",
        )

        result = await healthy_conn.validate_health()

        assert result is False
        assert healthy_conn.metrics.health_check_count == 1
        assert healthy_conn.metrics.health_check_failures == 1
        assert healthy_conn.status == ConnectionStatus.UNHEALTHY

    @pytest.mark.asyncio
    async def test_health_check_timeout(self, health_config):
        """Test health check timeout."""
        # Create mock connection that takes too long
        mock_conn = AsyncMock()

        async def slow_execute(*args):
            await asyncio.sleep(1.0)  # Longer than timeout
            return MagicMock()

        mock_conn.execute = slow_execute

        healthy_conn = HealthyConnection(
            connection=mock_conn,
            config=health_config,
            connection_id="test_conn_timeout",
        )

        result = await healthy_conn.validate_health()

        assert result is False
        assert healthy_conn.status == ConnectionStatus.UNHEALTHY

    @pytest.mark.asyncio
    async def test_health_check_caching(self, healthy_connection):
        """Test health check result caching."""
        # First health check
        result1 = await healthy_connection.validate_health()
        assert result1 is True

        # Mock the connection to fail, but cached result should be used
        healthy_connection.connection = AsyncMock()
        healthy_connection.connection.execute.side_effect = sqlite3.Error("Fail")

        # Second health check should use cache
        result2 = await healthy_connection.validate_health()
        assert result2 is True  # Uses cached result

        # Forced health check should fail
        result3 = await healthy_connection.validate_health(force=True)
        assert result3 is False

    @pytest.mark.asyncio
    async def test_record_usage(self, healthy_connection):
        """Test usage recording and metrics."""
        initial_usage = healthy_connection.metrics.usage_count
        initial_time = healthy_connection.metrics.last_used

        # Record usage
        await healthy_connection.record_usage(query_time=0.1)

        assert healthy_connection.metrics.usage_count == initial_usage + 1
        assert healthy_connection.metrics.last_used > initial_time
        assert healthy_connection.metrics.total_query_time == 0.1

    @pytest.mark.asyncio
    async def test_connection_expiration_by_age(self, healthy_connection):
        """Test connection expiration by age."""
        # Manually set old creation time
        old_time = datetime.utcnow() - timedelta(seconds=120)  # 2 minutes ago
        healthy_connection._metrics.created_at = old_time

        # Record usage to trigger expiration check
        await healthy_connection.record_usage()

        assert healthy_connection.is_expired is True
        assert healthy_connection.status == ConnectionStatus.EXPIRED

    @pytest.mark.asyncio
    async def test_connection_expiration_by_usage(self, healthy_connection):
        """Test connection expiration by usage count."""
        # Set usage count to exceed limit
        healthy_connection._metrics.usage_count = 15  # Exceeds max_usage=10

        # Record usage to trigger expiration check
        await healthy_connection.record_usage()

        assert healthy_connection.is_expired is True
        assert healthy_connection.status == ConnectionStatus.EXPIRED

    @pytest.mark.asyncio
    async def test_connection_staleness(self, healthy_connection):
        """Test connection staleness detection."""
        # Set old last_used time
        old_time = datetime.utcnow() - timedelta(seconds=60)  # 1 minute ago
        healthy_connection._metrics.last_used = old_time

        # Record usage to trigger staleness check
        await healthy_connection.record_usage()

        assert healthy_connection.is_stale is True
        assert healthy_connection.status == ConnectionStatus.STALE

    @pytest.mark.asyncio
    async def test_connection_stats(self, healthy_connection):
        """Test connection statistics."""
        # Record some usage
        await healthy_connection.record_usage(0.1)
        await healthy_connection.record_usage(0.2)
        await healthy_connection.validate_health()

        stats = healthy_connection.get_stats()

        assert stats["connection_id"] == "test_conn_1"
        assert stats["status"] == "healthy"
        assert stats["usage_count"] == 2
        assert stats["health_check_count"] == 1
        assert stats["health_check_failures"] == 0
        assert stats["health_check_success_rate"] == 1.0
        assert stats["average_query_time"] == 0.15
        assert stats["is_healthy"] is True
        assert stats["is_expired"] is False
        assert stats["is_stale"] is False


class TestConnectionHealthManager:
    """Test ConnectionHealthManager functionality."""

    @pytest.fixture
    def health_config(self):
        """Create test health check configuration."""
        return HealthCheckConfig(interval=0.1)  # Short interval for testing

    @pytest.fixture
    async def health_manager(self, health_config):
        """Create ConnectionHealthManager for testing."""
        manager = ConnectionHealthManager(health_config)
        yield manager
        await manager.stop_monitoring()

    @pytest.fixture
    def mock_connections(self, health_config):
        """Create mock healthy connections for testing."""
        connections = []
        for i in range(3):
            mock_conn = AsyncMock()
            mock_cursor = AsyncMock()
            mock_cursor.fetchone.return_value = (1,)
            mock_conn.execute.return_value = mock_cursor

            healthy_conn = HealthyConnection(
                connection=mock_conn,
                config=health_config,
                connection_id=f"test_conn_{i}",
            )
            connections.append(healthy_conn)

        return connections

    @pytest.mark.asyncio
    async def test_start_stop_monitoring(self, health_manager, mock_connections):
        """Test starting and stopping health monitoring."""
        # Start monitoring
        await health_manager.start_monitoring(mock_connections)
        assert health_manager._monitoring_task is not None

        # Stop monitoring
        await health_manager.stop_monitoring()
        assert health_manager._monitoring_task is None

    @pytest.mark.asyncio
    async def test_monitoring_loop(self, health_manager, mock_connections):
        """Test monitoring loop execution."""
        # Start monitoring
        await health_manager.start_monitoring(mock_connections)

        # Wait for at least one monitoring cycle
        await asyncio.sleep(0.2)

        # Check that health checks were performed
        for conn in mock_connections:
            assert conn.metrics.health_check_count > 0

        # Stop monitoring
        await health_manager.stop_monitoring()

    def test_pool_statistics_empty(self, health_manager):
        """Test pool statistics with empty pool."""
        stats = health_manager.get_pool_statistics([])

        assert stats["total_connections"] == 0
        assert stats["healthy_connections"] == 0
        assert stats["unhealthy_connections"] == 0
        assert stats["health_check_success_rate"] == 0.0
        assert stats["average_age"] == 0.0
        assert stats["total_usage"] == 0

    @pytest.mark.asyncio
    async def test_pool_statistics_with_connections(
        self, health_manager, mock_connections
    ):
        """Test pool statistics with connections."""
        # Record some usage and health checks
        for i, conn in enumerate(mock_connections):
            await conn.record_usage(0.1 * (i + 1))
            await conn.validate_health()

        # Make one connection unhealthy
        mock_connections[1]._status = ConnectionStatus.UNHEALTHY
        mock_connections[1]._metrics.health_check_failures = 1

        stats = health_manager.get_pool_statistics(mock_connections)

        assert stats["total_connections"] == 3
        assert stats["healthy_connections"] == 2
        assert stats["unhealthy_connections"] == 1
        assert stats["total_usage"] == 3  # 1 + 1 + 1
        assert 0 < stats["health_check_success_rate"] < 1


class TestDatabaseManagerHealthIntegration:
    """Test DatabaseManager integration with health checking."""

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
    def health_config(self):
        """Create test health check configuration."""
        return HealthCheckConfig(
            enabled=True,
            timeout=1.0,
            max_age=300,  # 5 minutes
            max_usage=100,
            interval=1,  # 1 second for testing
        )

    @pytest.fixture
    async def db_manager(self, temp_db_path, health_config):
        """Create DatabaseManager with health checking."""
        manager = DatabaseManager(
            database_url=f"sqlite:///{temp_db_path}",
            pool_size=2,
            health_check_config=health_config,
        )
        yield manager
        await manager.close()

    @pytest.mark.asyncio
    async def test_database_manager_initialization_with_health_checks(self, db_manager):
        """Test DatabaseManager initialization with health checking."""
        await db_manager.initialize()

        assert db_manager._initialized is True
        assert len(db_manager._pool) == 2
        assert all(isinstance(conn, HealthyConnection) for conn in db_manager._pool)
        assert db_manager._health_manager is not None

    @pytest.mark.asyncio
    async def test_healthy_connection_retrieval(self, db_manager):
        """Test getting healthy connections from pool."""
        await db_manager.initialize()

        # Get connection and verify it's healthy
        async with db_manager.get_connection() as conn:
            assert conn is not None
            # Test that connection works
            cursor = await conn.execute("SELECT 1")
            result = await cursor.fetchone()
            assert result == (1,)

    @pytest.mark.asyncio
    async def test_connection_health_validation(self, db_manager):
        """Test connection health validation during retrieval."""
        await db_manager.initialize()

        # Get connection multiple times to test pool behavior
        for _ in range(3):
            async with db_manager.get_connection() as conn:
                cursor = await conn.execute("SELECT 1")
                result = await cursor.fetchone()
                assert result == (1,)

    @pytest.mark.asyncio
    async def test_connection_stats_with_health_metrics(self, db_manager):
        """Test connection statistics including health metrics."""
        await db_manager.initialize()

        # Use connections to generate metrics
        async with db_manager.get_connection() as conn:
            await conn.execute("SELECT 1")

        stats = await db_manager.get_connection_stats()

        # Verify basic stats
        assert stats["initialized"] is True
        assert stats["max_pool_size"] == 2

        # Verify health stats are included
        assert "total_connections" in stats
        assert "healthy_connections" in stats
        assert "health_check_success_rate" in stats
        assert "average_age" in stats
        assert "total_usage" in stats

    @pytest.mark.asyncio
    async def test_unhealthy_connection_removal(self, db_manager):
        """Test automatic removal of unhealthy connections."""
        await db_manager.initialize()

        # Get a connection and manually mark it as unhealthy
        healthy_conn = db_manager._pool[0]
        healthy_conn._status = ConnectionStatus.UNHEALTHY

        # Getting connection should create new healthy one
        async with db_manager.get_connection() as conn:
            cursor = await conn.execute("SELECT 1")
            result = await cursor.fetchone()
            assert result == (1,)

    @pytest.mark.asyncio
    async def test_connection_expiration_handling(self, db_manager):
        """Test handling of expired connections."""
        await db_manager.initialize()

        # Get connection and simulate expiration
        healthy_conn = db_manager._pool[0]
        healthy_conn._metrics.usage_count = 200  # Exceeds max_usage=100

        # Connection should be retired and new one created
        async with db_manager.get_connection() as conn:
            cursor = await conn.execute("SELECT 1")
            result = await cursor.fetchone()
            assert result == (1,)

    @pytest.mark.asyncio
    async def test_health_monitoring_integration(self, db_manager):
        """Test health monitoring integration."""
        await db_manager.initialize()

        # Wait for at least one monitoring cycle
        await asyncio.sleep(1.2)

        # Verify health checks were performed
        for conn in db_manager._pool:
            if conn.metrics.health_check_count > 0:
                assert conn.is_healthy

    @pytest.mark.asyncio
    async def test_database_manager_cleanup(self, db_manager):
        """Test proper cleanup of health monitoring."""
        await db_manager.initialize()

        # Verify monitoring is active
        assert db_manager._health_manager._monitoring_task is not None

        # Close database manager
        await db_manager.close()

        # Verify cleanup
        assert db_manager._health_manager._monitoring_task is None
        assert len(db_manager._pool) == 0
        assert db_manager._initialized is False

    @pytest.mark.asyncio
    async def test_disabled_health_checks(self, temp_db_path):
        """Test DatabaseManager with disabled health checks."""
        disabled_config = HealthCheckConfig(enabled=False)

        db_manager = DatabaseManager(
            database_url=f"sqlite:///{temp_db_path}",
            pool_size=2,
            health_check_config=disabled_config,
        )

        try:
            await db_manager.initialize()

            # Health monitoring should not be started
            assert db_manager._health_manager._monitoring_task is None

            # Connections should still work
            async with db_manager.get_connection() as conn:
                cursor = await conn.execute("SELECT 1")
                result = await cursor.fetchone()
                assert result == (1,)

        finally:
            await db_manager.close()


class TestHealthCheckErrorScenarios:
    """Test error scenarios in health checking."""

    @pytest.fixture
    def health_config(self):
        """Create test health check configuration."""
        return HealthCheckConfig(timeout=0.1)

    @pytest.mark.asyncio
    async def test_health_check_with_closed_connection(self, health_config):
        """Test health check with closed connection."""
        # Create connection and close it
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        temp_path = temp_file.name
        temp_file.close()

        try:
            conn = await aiosqlite.connect(temp_path)
            healthy_conn = HealthyConnection(
                connection=conn,
                config=health_config,
                connection_id="closed_conn",
            )

            # Close the connection
            await conn.close()

            # Health check should fail
            result = await healthy_conn.validate_health()
            assert result is False
            assert healthy_conn.status == ConnectionStatus.UNHEALTHY

        finally:
            Path(temp_path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_health_check_with_corrupted_database(self, health_config):
        """Test health check with corrupted database file."""
        # Create corrupted database file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        temp_file.write(b"corrupted data")
        temp_file.close()

        try:
            # Try to connect to corrupted database
            conn = await aiosqlite.connect(temp_file.name)
            healthy_conn = HealthyConnection(
                connection=conn,
                config=health_config,
                connection_id="corrupted_conn",
            )

            # Health check may fail due to corruption
            await healthy_conn.validate_health()
            # Result may vary depending on SQLite's behavior with corruption

        except Exception:
            # Connection creation may fail with corrupted database
            pass
        finally:
            Path(temp_file.name).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_concurrent_health_checks(self, health_config):
        """Test concurrent health checks on same connection."""
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        temp_path = temp_file.name
        temp_file.close()

        try:
            conn = await aiosqlite.connect(temp_path)
            healthy_conn = HealthyConnection(
                connection=conn,
                config=health_config,
                connection_id="concurrent_conn",
            )

            # Run concurrent health checks
            tasks = [healthy_conn.validate_health(force=True) for _ in range(5)]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # At least some should succeed
            successful = [r for r in results if r is True]
            assert len(successful) > 0

        finally:
            await conn.close()
            Path(temp_path).unlink(missing_ok=True)
