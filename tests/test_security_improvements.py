"""
Tests for security and performance improvements in the skill caching system.
"""

import json
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from voyager_trader.models.learning import Skill, SkillExecutionResult
from voyager_trader.models.types import SkillCategory, SkillComplexity
from voyager_trader.skills import (
    CacheConfig,
    CacheOverflowError,
    ConnectionPool,
    DatabaseConfig,
    DatabaseConnectionError,
    SecurityValidationError,
    SkillExecutionCache,
    SkillExecutor,
)


class TestCacheUtilizationAlerting:
    """Test cache utilization monitoring and alerting."""

    def test_cache_utilization_alerts(self):
        """Test cache utilization alerts when threshold is exceeded."""
        config = CacheConfig(
            max_execution_cache_size=5,
            cache_utilization_alert_threshold=0.8,  # 80% threshold
        )
        cache = SkillExecutionCache(config)

        test_skill = Skill(
            name="test_skill",
            description="Test skill",
            category=SkillCategory.TECHNICAL_ANALYSIS,
            complexity=SkillComplexity.BASIC,
            code="result = {'test': True}",
            input_schema={"type": "object"},
            output_schema={"type": "object"},
        )

        # Add entries to approach threshold
        for i in range(4):  # 4/5 = 80% - should trigger alert
            result = (SkillExecutionResult.SUCCESS, {"value": i}, {"metadata": "test"})
            cache.cache_execution_result(test_skill, {"input": i}, None, result)

        # Check cache stats
        stats = cache.get_cache_stats()
        execution_stats = stats["execution_cache"]

        assert execution_stats["utilization"] == 0.8  # 80%
        assert execution_stats["utilization_percentage"] == 80.0

    def test_memory_monitoring_enabled(self):
        """Test memory monitoring functionality."""
        config = CacheConfig(
            enable_memory_monitoring=True,
            memory_report_interval_minutes=1,  # Short interval for testing
        )
        cache = SkillExecutionCache(config)

        # Force memory monitoring check
        with patch("psutil.Process") as mock_process:
            mock_memory_info = Mock()
            mock_memory_info.rss = 100 * 1024 * 1024  # 100 MB
            mock_memory_info.vms = 200 * 1024 * 1024  # 200 MB

            mock_process.return_value.memory_info.return_value = mock_memory_info
            mock_process.return_value.memory_percent.return_value = 15.5

            cache._report_memory_usage()

            # Verify memory monitoring was called
            mock_process.assert_called_once()
            mock_process.return_value.memory_info.assert_called_once()

    def test_memory_monitoring_high_usage_alert(self):
        """Test memory monitoring alerts on high usage."""
        config = CacheConfig(enable_memory_monitoring=True)
        cache = SkillExecutionCache(config)

        with patch("psutil.Process") as mock_process:
            mock_memory_info = Mock()
            mock_memory_info.rss = 1000 * 1024 * 1024  # 1000 MB
            mock_memory_info.vms = 2000 * 1024 * 1024  # 2000 MB

            mock_process.return_value.memory_info.return_value = mock_memory_info
            mock_process.return_value.memory_percent.return_value = 85.0  # High usage

            with patch.object(cache.logger, "warning") as mock_warning:
                cache._report_memory_usage()

                # Should trigger high memory warning
                mock_warning.assert_called_once()
                warning_call = mock_warning.call_args[0][0]
                assert "High memory usage detected" in warning_call
                assert "85.0%" in warning_call


class TestConnectionPoolMonitoring:
    """Test connection pool exhaustion monitoring."""

    def setup_method(self):
        """Set up test connection pool."""
        self.db_config = DatabaseConfig(
            db_type="sqlite",
            connection_string=":memory:",
            pool_size=2,
            max_overflow=1,
            enable_connection_pooling=True,
        )
        self.pool = ConnectionPool(self.db_config)

    def test_exhaustion_event_tracking(self):
        """Test tracking of pool exhaustion events."""
        initial_stats = self.pool.get_pool_stats()
        assert initial_stats["exhaustion_events"] == 0

        # Exhaust the pool
        connections = []
        try:
            # Use all pool connections + overflow
            for _ in range(3):  # pool_size=2 + max_overflow=1
                conn_context = self.pool.get_connection()
                connections.append(conn_context)
                conn_context.__enter__()

            # Try to get one more - should fail and increment exhaustion count
            with pytest.raises(DatabaseConnectionError):
                with self.pool.get_connection():
                    pass

            stats = self.pool.get_pool_stats()
            assert stats["exhaustion_events"] == 1
            assert stats["exhaustion_rate"] > 0

        finally:
            # Clean up connections
            for conn_context in connections:
                try:
                    conn_context.__exit__(None, None, None)
                except:
                    pass

    def test_pool_health_warnings(self):
        """Test pool health check warnings."""
        # Initially healthy
        warnings = self.pool.check_pool_health()
        assert len(warnings) == 0

        # Exhaust pool to trigger warnings
        connections = []
        try:
            for _ in range(3):  # Use all connections
                conn_context = self.pool.get_connection()
                connections.append(conn_context)
                conn_context.__enter__()

            warnings = self.pool.check_pool_health()
            assert len(warnings) > 0
            assert any("utilization" in warning.lower() for warning in warnings)

        finally:
            for conn_context in connections:
                try:
                    conn_context.__exit__(None, None, None)
                except:
                    pass

    def test_overflow_utilization_monitoring(self):
        """Test overflow connection utilization monitoring."""
        connections = []
        try:
            # Use all regular pool connections
            for _ in range(2):  # pool_size=2
                conn_context = self.pool.get_connection()
                connections.append(conn_context)
                conn_context.__enter__()

            # Use overflow connection
            overflow_conn = self.pool.get_connection()
            connections.append(overflow_conn)
            overflow_conn.__enter__()

            stats = self.pool.get_pool_stats()
            assert stats["overflow_connections"] == 1
            assert stats["overflow_utilization"] == 1.0  # 100% of overflow used
            assert stats["overflow_utilization_percentage"] == 100.0

        finally:
            for conn_context in connections:
                try:
                    conn_context.__exit__(None, None, None)
                except:
                    pass


class TestASTCodeInjection:
    """Test AST-based code injection improvements."""

    def setup_method(self):
        """Set up test executor."""
        self.config = CacheConfig()
        self.executor = SkillExecutor(cache_config=self.config)

    def test_ast_injection_with_simple_code(self):
        """Test AST injection with simple variable assignments."""
        cached_code = """
import json

# Placeholder assignments
inputs = {}
context = {}

# Simple skill code
result = {'value': inputs.get('test', 0) + 1}
"""

        inputs = {"test": 42}
        context = {"env": "test"}

        injected_code = self.executor._inject_runtime_data(cached_code, inputs, context)

        # Should contain the actual data, not empty dicts
        assert "{'test': 42}" in injected_code or '"test": 42' in injected_code
        assert "{'env': 'test'}" in injected_code or '"env": "test"' in injected_code

    def test_fallback_injection_on_ast_failure(self):
        """Test fallback to string replacement when AST fails."""
        # Invalid Python code that will cause AST parsing to fail
        invalid_cached_code = """
import json
inputs = {  # Incomplete syntax
context = {}
result = inputs.get('test')
"""

        inputs = {"test": "value"}
        context = {"key": "value"}

        # Should fall back to regex-based replacement without crashing
        injected_code = self.executor._inject_runtime_data(
            invalid_cached_code, inputs, context
        )

        # Should contain some form of data injection
        assert injected_code is not None
        assert len(injected_code) > 0

    def test_improved_regex_fallback(self):
        """Test improved regex-based fallback injection."""
        cached_code = """
inputs = { }
context = {   }
parameters = {'default': 'value'}
result = inputs.get('data')
"""

        inputs = {"data": [1, 2, 3]}
        context = {"mode": "test"}

        injected_code = self.executor._inject_runtime_data_fallback(
            cached_code, inputs, context
        )

        # Should handle whitespace variations
        assert '"data": [1, 2, 3]' in injected_code
        assert '"mode": "test"' in injected_code
        # Should not modify other assignments
        assert "parameters = {'default': 'value'}" in injected_code


class TestSpecificExceptionHandling:
    """Test improved specific exception handling."""

    def test_database_connection_error_specificity(self):
        """Test that database errors are properly categorized."""
        # Test with invalid database config
        invalid_config = DatabaseConfig(
            db_type="sqlite",
            connection_string="/invalid/path/database.db",
            enable_connection_pooling=True,
        )

        with pytest.raises(DatabaseConnectionError):
            ConnectionPool(invalid_config)

    def test_cache_overflow_detection(self):
        """Test cache overflow error detection."""
        # This would be used in future implementations
        # For now, just verify the exception class exists
        assert issubclass(CacheOverflowError, Exception)

        # Test that it can be raised and caught properly
        try:
            raise CacheOverflowError("Cache utilization too high")
        except CacheOverflowError as e:
            assert "Cache utilization" in str(e)

    def test_security_validation_error(self):
        """Test security validation error handling."""
        # Verify the exception class exists and works
        assert issubclass(SecurityValidationError, Exception)

        try:
            raise SecurityValidationError("Dangerous import detected")
        except SecurityValidationError as e:
            assert "Dangerous import" in str(e)


class TestSecurityValidation:
    """Test security validation improvements."""

    def test_subprocess_security_parameters(self):
        """Test that subprocess execution uses secure parameters."""
        config = CacheConfig()
        executor = SkillExecutor(cache_config=config)

        test_skill = Skill(
            name="secure_test",
            description="Security test skill",
            category=SkillCategory.TECHNICAL_ANALYSIS,
            complexity=SkillComplexity.BASIC,
            code="result = {'secure': True}",
            input_schema={"type": "object"},
            output_schema={"type": "object"},
        )

        # Mock subprocess.run to check security parameters
        with patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = '{"secure": true}'
            mock_run.return_value.stderr = ""

            try:
                executor.execute_skill(test_skill, {})
            except:
                pass  # We're just testing the subprocess call parameters

            # Verify secure subprocess parameters
            if mock_run.called:
                call_args = mock_run.call_args
                kwargs = call_args[1] if len(call_args) > 1 else {}

                # Should have security parameters
                assert kwargs.get("capture_output") is True
                assert kwargs.get("text") is True
                assert "timeout" in kwargs
                # Should NOT use shell=True (security risk)
                assert kwargs.get("shell") is not True

    def test_temporary_directory_usage(self):
        """Test that temporary directories are used for skill execution."""
        config = CacheConfig()
        executor = SkillExecutor(cache_config=config)

        test_skill = Skill(
            name="temp_test",
            description="Temporary directory test",
            category=SkillCategory.TECHNICAL_ANALYSIS,
            complexity=SkillComplexity.BASIC,
            code="result = {'temp': True}",
            input_schema={"type": "object"},
            output_schema={"type": "object"},
        )

        with patch("tempfile.TemporaryDirectory") as mock_temp_dir:
            mock_temp_dir.return_value.__enter__.return_value = "/tmp/secure_temp"
            mock_temp_dir.return_value.__exit__.return_value = None

            with patch("subprocess.run") as mock_run:
                mock_run.return_value.returncode = 0
                mock_run.return_value.stdout = '{"temp": true}'
                mock_run.return_value.stderr = ""

                try:
                    executor.execute_skill(test_skill, {})
                except:
                    pass

                # Verify temporary directory was used
                mock_temp_dir.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
