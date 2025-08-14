"""
Tests for VOYAGER skill operations observability system.

Comprehensive test suite for monitoring, metrics collection, health checks,
and performance tracking functionality.
"""

import tempfile
import time
import unittest
from datetime import datetime
from unittest.mock import Mock, patch

import pytest

from src.voyager_trader.models.learning import Skill, SkillExecutionResult
from src.voyager_trader.models.types import SkillCategory, SkillComplexity
from src.voyager_trader.observability import (
    CorrelationContextManager,
    HealthCheckResult,
    ObservabilityConfig,
    PerformanceBaseline,
    SensitiveDataFilter,
    SkillObservabilityManager,
)
from src.voyager_trader.skills import SkillExecutor, VoyagerSkillLibrary


class TestSensitiveDataFilter(unittest.TestCase):
    """Test sensitive data filtering functionality."""

    def setUp(self):
        self.filter = SensitiveDataFilter()

    def test_filter_api_keys(self):
        """Test filtering of API key patterns."""
        text = "Use API key sk_1234567890abcdef1234567890 for authentication"
        filtered = self.filter.filter_string(text)
        self.assertIn("[FILTERED]", filtered)
        self.assertNotIn("sk_1234567890abcdef1234567890", filtered)

    def test_filter_email_addresses(self):
        """Test filtering of email addresses."""
        text = "Contact support at admin@example.com for help"
        filtered = self.filter.filter_string(text)
        self.assertIn("[FILTERED]", filtered)
        self.assertNotIn("admin@example.com", filtered)

    def test_filter_monetary_amounts(self):
        """Test filtering of monetary amounts (potential positions/balances)."""
        text = "Current balance is 12345.67 USD"
        filtered = self.filter.filter_string(text)
        self.assertIn("[FILTERED]", filtered)
        self.assertNotIn("12345.67", filtered)

    def test_filter_dict_recursively(self):
        """Test recursive filtering of dictionary structures."""
        data = {
            "user": "test@example.com",
            "nested": {"api_key": "sk_abcdef1234567890", "balance": 1234.56},
            "safe_data": "this should remain",
        }

        filtered = self.filter.filter_dict(data)

        # Sensitive data should be filtered
        self.assertEqual(filtered["user"], "[FILTERED]")
        self.assertEqual(filtered["nested"]["api_key"], "[FILTERED]")
        self.assertEqual(filtered["nested"]["balance"], "[FILTERED]")

        # Safe data should remain
        self.assertEqual(filtered["safe_data"], "this should remain")

    def test_no_filtering_for_safe_content(self):
        """Test that safe content passes through unchanged."""
        safe_text = "This is a normal log message with no sensitive data"
        filtered = self.filter.filter_string(safe_text)
        self.assertEqual(safe_text, filtered)


class TestCorrelationContextManager(unittest.TestCase):
    """Test correlation ID management functionality."""

    def setUp(self):
        self.manager = CorrelationContextManager()

    def test_generate_correlation_id(self):
        """Test correlation ID generation."""
        id1 = self.manager.generate_correlation_id()
        id2 = self.manager.generate_correlation_id()

        self.assertIsInstance(id1, str)
        self.assertIsInstance(id2, str)
        self.assertNotEqual(id1, id2)  # Should be unique
        self.assertTrue(len(id1) > 30)  # UUID should be reasonably long

    def test_correlation_context_manager(self):
        """Test correlation context manager functionality."""
        test_id = "test-correlation-id-123"

        # Initially no correlation ID
        self.assertIsNone(self.manager.get_correlation_id())

        # Within context, ID should be set
        with self.manager.correlation_context(test_id) as context_id:
            self.assertEqual(context_id, test_id)
            self.assertEqual(self.manager.get_correlation_id(), test_id)

        # After context, ID should be cleared
        self.assertIsNone(self.manager.get_correlation_id())

    def test_nested_correlation_contexts(self):
        """Test nested correlation contexts."""
        outer_id = "outer-id"
        inner_id = "inner-id"

        with self.manager.correlation_context(outer_id):
            self.assertEqual(self.manager.get_correlation_id(), outer_id)

            with self.manager.correlation_context(inner_id):
                self.assertEqual(self.manager.get_correlation_id(), inner_id)

            # Should restore outer ID
            self.assertEqual(self.manager.get_correlation_id(), outer_id)

    def test_auto_generate_correlation_id(self):
        """Test auto-generation of correlation ID when none provided."""
        with self.manager.correlation_context() as context_id:
            self.assertIsNotNone(context_id)
            self.assertEqual(self.manager.get_correlation_id(), context_id)


class TestSkillObservabilityManager(unittest.TestCase):
    """Test skill observability manager functionality."""

    def setUp(self):
        self.config = ObservabilityConfig(
            enable_tracing=False,  # Disable for testing
            enable_metrics=False,  # Disable for testing
            enable_correlation_ids=True,
            enable_sensitive_data_filtering=True,
        )
        self.observability = SkillObservabilityManager(self.config)

    def test_initialization(self):
        """Test observability manager initialization."""
        self.assertIsNotNone(self.observability.sensitive_filter)
        self.assertIsNotNone(self.observability.correlation_manager)
        self.assertEqual(self.observability.config, self.config)

    @patch("src.voyager_trader.observability.logger")
    def test_instrument_skill_execution_success(self, mock_logger):
        """Test skill execution instrumentation for successful execution."""
        skill_id = "test_skill"
        inputs = {"param1": "value1"}

        with self.observability.instrument_skill_execution(skill_id, inputs) as context:
            self.assertIn("correlation_id", context)
            self.assertIn("start_time", context)
            self.assertIsInstance(context["start_time"], float)

            # Simulate some work
            time.sleep(0.01)

        # Check that logging was called
        self.assertTrue(mock_logger.info.called)
        log_calls = [call.args[0] for call in mock_logger.info.call_args_list]
        self.assertTrue(any("Starting skill execution" in call for call in log_calls))
        self.assertTrue(any("Completed skill execution" in call for call in log_calls))

    @pytest.mark.skip(reason="Observability test needs update")
    @patch("src.voyager_trader.observability.logger")
    def test_instrument_skill_execution_with_error(self, mock_logger):
        """Test skill execution instrumentation when error occurs."""
        skill_id = "test_skill"
        inputs = {"param1": "value1"}

        with self.assertRaises(ValueError):
            with self.observability.instrument_skill_execution(skill_id, inputs):
                raise ValueError("Test error")

        # Check that error logging was called
        self.assertTrue(mock_logger.error.called)
        error_calls = [call.args[0] for call in mock_logger.error.call_args_list]
        self.assertTrue(any("Skill execution failed" in call for call in error_calls))

    def test_record_cache_events(self):
        """Test cache hit/miss recording."""
        # These should not raise exceptions
        self.observability.record_cache_hit("execution", "test_key")
        self.observability.record_cache_miss("metadata", "test_key_2")

    @patch("src.voyager_trader.observability.logger")
    def test_instrument_skill_composition(self, mock_logger):
        """Test skill composition instrumentation."""
        skill_ids = ["skill1", "skill2", "skill3"]

        with self.observability.instrument_skill_composition(skill_ids) as context:
            self.assertIn("composition_id", context)
            # Simulate composition work
            time.sleep(0.01)

        # Check logging
        self.assertTrue(mock_logger.info.called)
        log_calls = [call.args[0] for call in mock_logger.info.call_args_list]
        self.assertTrue(any("Starting skill composition" in call for call in log_calls))
        self.assertTrue(
            any("Completed skill composition" in call for call in log_calls)
        )

    def test_record_validation_events(self):
        """Test validation event recording."""
        skill_id = "test_skill"
        validation_type = "security"
        errors = ["Error 1", "Error 2"]

        # These should not raise exceptions
        self.observability.record_validation_attempt(skill_id, validation_type)
        self.observability.record_validation_failure(skill_id, validation_type, errors)

    def test_performance_baseline_updates(self):
        """Test performance baseline tracking."""
        metric_name = "test_metric"

        # Should not have baseline initially
        self.assertNotIn(metric_name, self.observability._performance_baselines)

        # Update baseline multiple times
        values = [1.0, 1.1, 0.9, 1.2, 0.8]
        for value in values:
            self.observability._update_performance_baseline(metric_name, value)

        # Should now have baseline
        self.assertIn(metric_name, self.observability._performance_baselines)
        baseline = self.observability._performance_baselines[metric_name]

        self.assertEqual(baseline.metric_name, metric_name)
        self.assertEqual(baseline.measurement_count, len(values))
        self.assertIsInstance(baseline.baseline_value, float)
        self.assertIsInstance(baseline.confidence_interval, tuple)
        self.assertEqual(len(baseline.confidence_interval), 2)

    def test_health_check_execution(self):
        """Test health check execution and caching."""

        def mock_health_check():
            return {"healthy": True, "message": "All good"}

        component = "test_component"

        # First call should execute check
        result1 = self.observability.perform_health_check(component, mock_health_check)
        self.assertEqual(result1.component, component)
        self.assertEqual(result1.status, "healthy")
        self.assertIn("All good", result1.message)

        # Second call within cache timeout should return cached result
        result2 = self.observability.perform_health_check(component, mock_health_check)
        self.assertEqual(
            result1.timestamp, result2.timestamp
        )  # Same timestamp = cached

    def test_system_metrics_summary(self):
        """Test system metrics summary generation."""
        summary = self.observability.get_system_metrics_summary()

        self.assertIn("timestamp", summary)
        self.assertIn("service_name", summary)
        self.assertIn("observability_enabled", summary)
        self.assertIn("performance_baselines", summary)

        # Check observability status
        obs_status = summary["observability_enabled"]
        self.assertEqual(obs_status["correlation_ids"], True)
        self.assertEqual(obs_status["sensitive_filtering"], True)


class TestSkillExecutorObservability(unittest.TestCase):
    """Test skill executor observability integration."""

    def setUp(self):
        self.config = ObservabilityConfig(
            enable_tracing=False, enable_metrics=False, enable_correlation_ids=True
        )

        self.executor = SkillExecutor(
            timeout_seconds=5, observability_config=self.config
        )

        self.test_skill = Skill(
            name="test_observability_skill",
            description="Test skill for observability",
            category=SkillCategory.MARKET_ANALYSIS,
            complexity=SkillComplexity.BASIC,
            code='print("Hello from skill")',
            input_schema={"type": "object"},
            output_schema={"type": "object"},
        )

    def test_executor_has_observability(self):
        """Test that executor has observability manager."""
        self.assertIsNotNone(self.executor.observability)
        self.assertIsInstance(self.executor.observability, SkillObservabilityManager)

    @patch("src.voyager_trader.skills.subprocess.run")
    def test_skill_execution_with_observability(self, mock_subprocess):
        """Test skill execution includes observability instrumentation."""
        # Mock successful subprocess execution
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = '{"result": "success"}'
        mock_result.stderr = ""
        mock_subprocess.return_value = mock_result

        inputs = {"test_input": "value"}
        result, output, metadata = self.executor.execute_skill(self.test_skill, inputs)

        # Check that correlation ID was added to metadata
        self.assertIn("correlation_id", metadata)
        self.assertIsInstance(metadata["correlation_id"], str)

        # Check execution result
        self.assertEqual(result, SkillExecutionResult.SUCCESS)

    def test_cache_observability_integration(self):
        """Test cache events are recorded through observability."""
        with patch.object(self.executor.observability, "record_cache_hit") as mock_hit:
            with patch.object(
                self.executor.observability, "record_cache_miss"
            ) as mock_miss:
                # Mock cache miss first time
                with patch.object(
                    self.executor.cache, "get_execution_result", return_value=None
                ):
                    with patch(
                        "src.voyager_trader.skills.subprocess.run"
                    ) as mock_subprocess:
                        mock_result = Mock()
                        mock_result.returncode = 0
                        mock_result.stdout = '{"test": "result"}'
                        mock_result.stderr = ""
                        mock_subprocess.return_value = mock_result

                        self.executor.execute_skill(self.test_skill, {})

                        # Should record cache miss
                        mock_miss.assert_called_once()
                        mock_hit.assert_not_called()


class TestVoyagerSkillLibraryObservability(unittest.TestCase):
    """Test VoyagerSkillLibrary observability integration."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.config = {
            "skill_library_path": self.test_dir,
            "enable_tracing": False,
            "enable_metrics": False,
            "enable_health_endpoints": True,
            "execution_timeout": 5,
        }

        self.skill_library = VoyagerSkillLibrary(self.config)

        self.test_skill = Skill(
            name="test_health_skill",
            description="Test skill for health checks",
            category=SkillCategory.MARKET_ANALYSIS,
            complexity=SkillComplexity.BASIC,
            code='print("Health check skill")',
            input_schema={"type": "object"},
            output_schema={"type": "object"},
        )

    def test_library_has_observability(self):
        """Test that skill library has observability manager."""
        self.assertIsNotNone(self.skill_library.observability)
        self.assertIsInstance(
            self.skill_library.observability, SkillObservabilityManager
        )

    def test_health_status_endpoint(self):
        """Test health status endpoint functionality."""
        health_status = self.skill_library.get_health_status()

        # Check overall structure
        self.assertIn("overall_status", health_status)
        self.assertIn("components", health_status)
        self.assertIn("unhealthy_components", health_status)
        self.assertIn("system_metrics", health_status)

        # Check component health checks
        components = health_status["components"]
        expected_components = ["executor", "librarian", "cache", "validator"]

        for component in expected_components:
            self.assertIn(component, components)
            health_result = components[component]
            self.assertIsInstance(health_result, HealthCheckResult)
            self.assertIn(health_result.status, ["healthy", "degraded", "unhealthy"])

    def test_executor_health_check(self):
        """Test executor health check functionality."""
        health_result = self.skill_library._check_executor_health()

        self.assertIn("healthy", health_result)
        self.assertIn("message", health_result)
        self.assertIn("details", health_result)
        self.assertIsInstance(health_result["healthy"], bool)

    def test_librarian_health_check(self):
        """Test librarian health check functionality."""
        health_result = self.skill_library._check_librarian_health()

        self.assertIn("healthy", health_result)
        self.assertIn("message", health_result)
        self.assertIn("details", health_result)
        self.assertIsInstance(health_result["healthy"], bool)

    @pytest.mark.skip(reason="Cache health check implementation needs update")
    def test_cache_health_check(self):
        """Test cache health check functionality."""
        health_result = self.skill_library._check_cache_health()

        self.assertIn("healthy", health_result)
        self.assertIn("message", health_result)
        self.assertIn("details", health_result)
        self.assertIsInstance(health_result["healthy"], bool)

        # Check cache utilization details
        details = health_result["details"]
        self.assertIn("executor_cache_utilization", details)
        self.assertIn("metadata_cache_utilization", details)
        self.assertIn("max_utilization", details)

    def test_validator_health_check(self):
        """Test validator health check functionality."""
        health_result = self.skill_library._check_validator_health()

        self.assertIn("healthy", health_result)
        self.assertIn("message", health_result)
        self.assertIn("details", health_result)
        self.assertIsInstance(health_result["healthy"], bool)

    def test_observability_cleanup(self):
        """Test observability cleanup functionality."""
        # Should not raise exceptions
        self.skill_library.cleanup_observability()

    def tearDown(self):
        """Clean up test environment."""
        import shutil

        try:
            shutil.rmtree(self.test_dir)
        except Exception:
            pass


class TestObservabilityConfig(unittest.TestCase):
    """Test observability configuration."""

    def test_default_config(self):
        """Test default observability configuration."""
        config = ObservabilityConfig()

        # Check defaults
        self.assertTrue(config.enable_tracing)
        self.assertTrue(config.enable_metrics)
        self.assertTrue(config.enable_correlation_ids)
        self.assertTrue(config.enable_sensitive_data_filtering)
        self.assertTrue(config.enable_health_endpoints)
        self.assertEqual(config.service_name, "voyager-trader-skills")
        self.assertEqual(config.metrics_export_interval_seconds, 10)

    def test_custom_config(self):
        """Test custom observability configuration."""
        custom_config = ObservabilityConfig(
            enable_tracing=False,
            enable_metrics=False,
            service_name="custom-service",
            otlp_endpoint="http://localhost:4317",
        )

        self.assertFalse(custom_config.enable_tracing)
        self.assertFalse(custom_config.enable_metrics)
        self.assertEqual(custom_config.service_name, "custom-service")
        self.assertEqual(custom_config.otlp_endpoint, "http://localhost:4317")


class TestPerformanceBaseline(unittest.TestCase):
    """Test performance baseline functionality."""

    def test_performance_baseline_creation(self):
        """Test performance baseline creation."""
        baseline = PerformanceBaseline(
            metric_name="test_metric",
            baseline_value=1.0,
            measurement_count=10,
            last_updated=datetime.utcnow(),
            confidence_interval=(0.9, 1.1),
        )

        self.assertEqual(baseline.metric_name, "test_metric")
        self.assertEqual(baseline.baseline_value, 1.0)
        self.assertEqual(baseline.measurement_count, 10)
        self.assertIsInstance(baseline.last_updated, datetime)
        self.assertEqual(baseline.confidence_interval, (0.9, 1.1))


class TestHealthCheckResult(unittest.TestCase):
    """Test health check result functionality."""

    def test_health_check_result_creation(self):
        """Test health check result creation."""
        timestamp = datetime.utcnow()
        result = HealthCheckResult(
            component="test_component",
            status="healthy",
            message="Component is working",
            response_time_ms=50.0,
            timestamp=timestamp,
            details={"extra": "info"},
        )

        self.assertEqual(result.component, "test_component")
        self.assertEqual(result.status, "healthy")
        self.assertEqual(result.message, "Component is working")
        self.assertEqual(result.response_time_ms, 50.0)
        self.assertEqual(result.timestamp, timestamp)
        self.assertEqual(result.details["extra"], "info")


if __name__ == "__main__":
    # Run tests with pytest for better output
    pytest.main([__file__, "-v"])
