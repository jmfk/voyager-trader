"""
VOYAGER Skill Operations Observability Module.

Comprehensive monitoring, metrics collection, and observability for skill library
operations using OpenTelemetry for distributed tracing and metrics, enhanced
structured logging with correlation IDs, and health check endpoints.

Key Features:
- OpenTelemetry metrics and distributed tracing
- Correlation IDs for tracking skill execution flows
- Sensitive data filtering and automatic masking
- Health check endpoints for system monitoring
- Performance regression detection
"""

import hashlib
import json
import re
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    from opentelemetry import metrics, trace
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
        OTLPMetricExporter,
    )
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.trace.status import Status, StatusCode

    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False

from loguru import logger


@dataclass
class ObservabilityConfig:
    """Configuration for skill observability system."""

    # OpenTelemetry configuration
    enable_tracing: bool = True
    enable_metrics: bool = True
    otlp_endpoint: Optional[str] = None  # If None, uses console exporters
    service_name: str = "voyager-trader-skills"
    service_version: str = "1.0.0"

    # Metrics configuration
    metrics_export_interval_seconds: int = 10
    enable_performance_metrics: bool = True
    enable_resource_metrics: bool = True

    # Logging configuration
    enable_correlation_ids: bool = True
    log_level: str = "INFO"
    enable_sensitive_data_filtering: bool = True

    # Health check configuration
    health_check_timeout_seconds: int = 5
    enable_health_endpoints: bool = True

    # Performance monitoring
    enable_performance_regression_detection: bool = True
    performance_baseline_window_hours: int = 24
    performance_degradation_threshold: float = 0.2  # 20% degradation threshold


@dataclass
class HealthCheckResult:
    """Health check result for a system component."""

    component: str
    status: str  # "healthy", "degraded", "unhealthy"
    message: str
    response_time_ms: float
    timestamp: datetime
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceBaseline:
    """Performance baseline for regression detection."""

    metric_name: str
    baseline_value: float
    measurement_count: int
    last_updated: datetime
    confidence_interval: Tuple[float, float]


class SensitiveDataFilter:
    """Filters sensitive data from logs and metrics."""

    def __init__(self):
        # Patterns for sensitive data detection
        self.sensitive_patterns = [
            re.compile(r"\b[A-Za-z0-9]{20,}\b"),  # API keys (20+ alphanumeric)
            re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b"),  # Credit cards
            re.compile(
                r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
            ),  # Email addresses
            re.compile(r"\bsk_[A-Za-z0-9]{20,}\b"),  # OpenAI API keys
            re.compile(
                r"\b(?:password|pwd|secret|token|key)[:=]\s*[^\s]+", re.IGNORECASE
            ),
            re.compile(
                r"\b\d+\.\d{2,}\b"
            ),  # Monetary amounts (potential positions/balances)
        ]

    def filter_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Filter sensitive data from dictionary."""
        if not isinstance(data, dict):
            return data

        filtered = {}
        for key, value in data.items():
            if isinstance(value, dict):
                filtered[key] = self.filter_dict(value)
            elif isinstance(value, str):
                filtered[key] = self.filter_string(value)
            elif isinstance(value, (list, tuple)):
                filtered[key] = [
                    self.filter_string(str(item)) if isinstance(item, str) else item
                    for item in value
                ]
            else:
                filtered[key] = value

        return filtered

    def filter_string(self, text: str) -> str:
        """Filter sensitive data from string."""
        if not isinstance(text, str):
            return text

        filtered_text = text
        for pattern in self.sensitive_patterns:
            filtered_text = pattern.sub("[FILTERED]", filtered_text)

        return filtered_text


class CorrelationContextManager:
    """Manages correlation IDs for tracing skill execution flows."""

    def __init__(self):
        self._local = {}

    def generate_correlation_id(self) -> str:
        """Generate a new correlation ID."""
        return str(uuid.uuid4())

    def set_correlation_id(self, correlation_id: str) -> None:
        """Set correlation ID for current context."""
        self._local["correlation_id"] = correlation_id

    def get_correlation_id(self) -> Optional[str]:
        """Get current correlation ID."""
        return self._local.get("correlation_id")

    @contextmanager
    def correlation_context(self, correlation_id: Optional[str] = None):
        """Context manager for correlation ID tracking."""
        if correlation_id is None:
            correlation_id = self.generate_correlation_id()

        old_id = self.get_correlation_id()
        self.set_correlation_id(correlation_id)

        try:
            yield correlation_id
        finally:
            if old_id:
                self.set_correlation_id(old_id)
            else:
                self._local.pop("correlation_id", None)


class SkillObservabilityManager:
    """Central observability management for skill operations."""

    def __init__(self, config: ObservabilityConfig):
        self.config = config
        self.logger = logger
        self.sensitive_filter = SensitiveDataFilter()
        self.correlation_manager = CorrelationContextManager()

        # Performance baselines for regression detection
        self._performance_baselines: Dict[str, PerformanceBaseline] = {}

        # Health check results cache
        self._health_cache: Dict[str, HealthCheckResult] = {}
        self._health_cache_timeout = timedelta(seconds=30)

        # Initialize OpenTelemetry if available
        self.tracer = None
        self.meter = None
        self._metrics = {}

        if OPENTELEMETRY_AVAILABLE and (config.enable_tracing or config.enable_metrics):
            self._setup_opentelemetry()

        # Setup enhanced logging
        self._setup_logging()

    def _setup_opentelemetry(self) -> None:
        """Setup OpenTelemetry tracing and metrics."""
        try:
            # Setup tracing
            if self.config.enable_tracing:
                # Check if tracer provider is already set to avoid override warnings
                current_provider = trace.get_tracer_provider()
                if isinstance(current_provider, TracerProvider):
                    self.tracer = trace.get_tracer(__name__)
                else:
                    # No provider set, create one
                    if self.config.otlp_endpoint:
                        span_exporter = OTLPSpanExporter(
                            endpoint=self.config.otlp_endpoint
                        )
                    else:
                        from opentelemetry.sdk.trace.export import (
                            ConsoleSpanExporter,
                        )

                        span_exporter = ConsoleSpanExporter()

                    trace_provider = TracerProvider()
                    trace_provider.add_span_processor(BatchSpanProcessor(span_exporter))
                    trace.set_tracer_provider(trace_provider)
                    self.tracer = trace.get_tracer(__name__)

            # Setup metrics
            if self.config.enable_metrics:
                # Check if meter provider is already set to avoid override warnings
                current_provider = metrics.get_meter_provider()
                if isinstance(current_provider, MeterProvider):
                    self.meter = metrics.get_meter(__name__)
                else:
                    # No provider set, create one
                    if self.config.otlp_endpoint:
                        metric_exporter = OTLPMetricExporter(
                            endpoint=self.config.otlp_endpoint
                        )
                    else:
                        from opentelemetry.sdk.metrics.export import (
                            ConsoleMetricExporter,
                        )

                        metric_exporter = ConsoleMetricExporter()

                    metric_reader = PeriodicExportingMetricReader(
                        exporter=metric_exporter,
                        export_interval_millis=self.config.metrics_export_interval_seconds
                        * 1000,
                    )

                    metrics_provider = MeterProvider(metric_readers=[metric_reader])
                    metrics.set_meter_provider(metrics_provider)
                    self.meter = metrics.get_meter(__name__)

                self._setup_metrics()

        except Exception as e:
            self.logger.error(f"Failed to setup OpenTelemetry: {e}")

    def _setup_metrics(self) -> None:
        """Setup skill-specific metrics."""
        if not self.meter:
            return

        # Execution metrics
        self._metrics["execution_counter"] = self.meter.create_counter(
            "skill_executions_total",
            description="Total number of skill executions",
            unit="1",
        )

        self._metrics["execution_duration"] = self.meter.create_histogram(
            "skill_execution_duration_seconds",
            description="Skill execution duration",
            unit="s",
        )

        self._metrics["execution_success"] = self.meter.create_counter(
            "skill_execution_success_total",
            description="Total number of successful skill executions",
            unit="1",
        )

        self._metrics["execution_errors"] = self.meter.create_counter(
            "skill_execution_errors_total",
            description="Total number of skill execution errors",
            unit="1",
        )

        # Cache metrics
        self._metrics["cache_hits"] = self.meter.create_counter(
            "skill_cache_hits_total",
            description="Total number of skill cache hits",
            unit="1",
        )

        self._metrics["cache_misses"] = self.meter.create_counter(
            "skill_cache_misses_total",
            description="Total number of skill cache misses",
            unit="1",
        )

        # Composition metrics
        self._metrics["composition_attempts"] = self.meter.create_counter(
            "skill_composition_attempts_total",
            description="Total number of skill composition attempts",
            unit="1",
        )

        self._metrics["composition_success"] = self.meter.create_counter(
            "skill_composition_success_total",
            description="Total number of successful skill compositions",
            unit="1",
        )

        # Validation metrics
        self._metrics["validation_attempts"] = self.meter.create_counter(
            "skill_validation_attempts_total",
            description="Total number of skill validation attempts",
            unit="1",
        )

        self._metrics["validation_failures"] = self.meter.create_counter(
            "skill_validation_failures_total",
            description="Total number of skill validation failures",
            unit="1",
        )

        # Resource metrics
        if PSUTIL_AVAILABLE and self.config.enable_resource_metrics:
            self._metrics["memory_usage"] = self.meter.create_gauge(
                "skill_memory_usage_bytes",
                description="Memory usage during skill execution",
                unit="By",
            )

            self._metrics["cpu_usage"] = self.meter.create_gauge(
                "skill_cpu_usage_percent",
                description="CPU usage during skill execution",
                unit="1",
            )

    def _setup_logging(self) -> None:
        """Setup enhanced structured logging."""
        # Configure loguru for structured logging
        logger.remove()  # Remove default handler

        log_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{extra[correlation_id]}</cyan> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )

        logger.add(
            sink=lambda record: print(record),
            format=log_format,
            level=self.config.log_level,
            filter=self._log_filter,
        )

    def _log_filter(self, record) -> bool:
        """Filter sensitive data from log records."""
        if not self.config.enable_sensitive_data_filtering:
            return True

        # Add correlation ID if not present
        if "correlation_id" not in record["extra"]:
            correlation_id = (
                self.correlation_manager.get_correlation_id() or "no-correlation"
            )
            record["extra"]["correlation_id"] = correlation_id

        # Filter sensitive data from message
        if isinstance(record["message"], str):
            record["message"] = self.sensitive_filter.filter_string(record["message"])

        return True

    @contextmanager
    def instrument_skill_execution(self, skill_id: str, inputs: Dict[str, Any]):
        """Context manager for instrumenting skill execution."""
        start_time = time.time()
        correlation_id = self.correlation_manager.generate_correlation_id()

        # Filter sensitive data from inputs for logging
        filtered_inputs = self.sensitive_filter.filter_dict(inputs)

        with self.correlation_manager.correlation_context(correlation_id):
            # Start tracing span if available
            span = None
            if self.tracer:
                span = self.tracer.start_span(f"skill_execution:{skill_id}")
                span.set_attribute("skill_id", skill_id)
                span.set_attribute("correlation_id", correlation_id)
                span.set_attribute("input_count", len(inputs))

            # Record metrics
            if self._metrics.get("execution_counter"):
                self._metrics["execution_counter"].add(1, {"skill_id": skill_id})

            # Log execution start
            self.logger.info(
                f"Starting skill execution: {skill_id}",
                extra={
                    "correlation_id": correlation_id,
                    "skill_id": skill_id,
                    "inputs": filtered_inputs,
                    "event": "skill_execution_start",
                },
            )

            success = False

            try:
                yield {
                    "correlation_id": correlation_id,
                    "start_time": start_time,
                    "span": span,
                }
                success = True

            except Exception as e:
                success = False

                # Record error metrics
                if self._metrics.get("execution_errors"):
                    self._metrics["execution_errors"].add(
                        1, {"skill_id": skill_id, "error_type": type(e).__name__}
                    )

                # Set span status if available
                if span:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)

                # Log error
                self.logger.error(
                    f"Skill execution failed: {skill_id} - {str(e)}",
                    extra={
                        "correlation_id": correlation_id,
                        "skill_id": skill_id,
                        "error": str(e),
                        "event": "skill_execution_error",
                    },
                )

                raise

            finally:
                execution_time = time.time() - start_time

                # Record execution time metrics
                if self._metrics.get("execution_duration"):
                    self._metrics["execution_duration"].record(
                        execution_time, {"skill_id": skill_id}
                    )

                # Record success metrics
                if success and self._metrics.get("execution_success"):
                    self._metrics["execution_success"].add(1, {"skill_id": skill_id})

                # Record resource metrics if available
                if PSUTIL_AVAILABLE and self.config.enable_resource_metrics:
                    try:
                        process = psutil.Process()
                        memory_info = process.memory_info()
                        cpu_percent = process.cpu_percent()

                        if self._metrics.get("memory_usage"):
                            self._metrics["memory_usage"].set(
                                memory_info.rss, {"skill_id": skill_id}
                            )

                        if self._metrics.get("cpu_usage"):
                            self._metrics["cpu_usage"].set(
                                cpu_percent, {"skill_id": skill_id}
                            )

                    except Exception:
                        pass  # Ignore resource monitoring errors

                # Update performance baselines for regression detection
                if self.config.enable_performance_regression_detection:
                    self._update_performance_baseline(
                        f"execution_time_{skill_id}", execution_time
                    )

                # Finish span if available
                if span:
                    span.set_attribute("execution_time_seconds", execution_time)
                    span.set_attribute("success", success)
                    if success:
                        span.set_status(Status(StatusCode.OK))
                    span.end()

                # Log execution completion
                self.logger.info(
                    f"Completed skill execution: {skill_id} ({'success' if success else 'failure'})",
                    extra={
                        "correlation_id": correlation_id,
                        "skill_id": skill_id,
                        "execution_time_seconds": execution_time,
                        "success": success,
                        "event": "skill_execution_complete",
                    },
                )

    def record_cache_hit(self, cache_type: str, key: str) -> None:
        """Record a cache hit event."""
        if self._metrics.get("cache_hits"):
            self._metrics["cache_hits"].add(1, {"cache_type": cache_type})

        self.logger.debug(
            f"Cache hit: {cache_type}",
            extra={
                "cache_type": cache_type,
                "cache_key": hashlib.sha256(key.encode()).hexdigest()[
                    :12
                ],  # Hash key for privacy
                "event": "cache_hit",
            },
        )

    def record_cache_miss(self, cache_type: str, key: str) -> None:
        """Record a cache miss event."""
        if self._metrics.get("cache_misses"):
            self._metrics["cache_misses"].add(1, {"cache_type": cache_type})

        self.logger.debug(
            f"Cache miss: {cache_type}",
            extra={
                "cache_type": cache_type,
                "cache_key": hashlib.sha256(key.encode()).hexdigest()[
                    :12
                ],  # Hash key for privacy
                "event": "cache_miss",
            },
        )

    @contextmanager
    def instrument_skill_composition(self, skill_ids: List[str]):
        """Context manager for instrumenting skill composition."""
        composition_id = self.correlation_manager.generate_correlation_id()

        with self.correlation_manager.correlation_context(composition_id):
            # Start tracing span if available
            span = None
            if self.tracer:
                span = self.tracer.start_span("skill_composition")
                span.set_attribute("composition_id", composition_id)
                span.set_attribute("skill_count", len(skill_ids))
                span.set_attribute("skill_ids", json.dumps(skill_ids))

            # Record metrics
            if self._metrics.get("composition_attempts"):
                self._metrics["composition_attempts"].add(
                    1, {"skill_count": len(skill_ids)}
                )

            # Log composition start
            self.logger.info(
                f"Starting skill composition: {len(skill_ids)} skills",
                extra={
                    "composition_id": composition_id,
                    "skill_ids": skill_ids,
                    "skill_count": len(skill_ids),
                    "event": "skill_composition_start",
                },
            )

            success = False

            try:
                yield {"composition_id": composition_id, "span": span}
                success = True

                # Record success metrics
                if self._metrics.get("composition_success"):
                    self._metrics["composition_success"].add(
                        1, {"skill_count": len(skill_ids)}
                    )

            except Exception as e:
                # Set span status if available
                if span:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)

                # Log error
                self.logger.error(
                    f"Skill composition failed: {str(e)}",
                    extra={
                        "composition_id": composition_id,
                        "error": str(e),
                        "event": "skill_composition_error",
                    },
                )

                raise

            finally:
                # Finish span if available
                if span:
                    span.set_attribute("success", success)
                    if success:
                        span.set_status(Status(StatusCode.OK))
                    span.end()

                # Log composition completion
                self.logger.info(
                    f"Completed skill composition: {'success' if success else 'failure'}",
                    extra={
                        "composition_id": composition_id,
                        "success": success,
                        "event": "skill_composition_complete",
                    },
                )

    def record_validation_attempt(self, skill_id: str, validation_type: str) -> None:
        """Record a skill validation attempt."""
        if self._metrics.get("validation_attempts"):
            self._metrics["validation_attempts"].add(
                1, {"skill_id": skill_id, "validation_type": validation_type}
            )

    def record_validation_failure(
        self, skill_id: str, validation_type: str, errors: List[str]
    ) -> None:
        """Record a skill validation failure."""
        if self._metrics.get("validation_failures"):
            self._metrics["validation_failures"].add(
                1,
                {
                    "skill_id": skill_id,
                    "validation_type": validation_type,
                    "error_count": len(errors),
                },
            )

        # Filter sensitive data from errors
        filtered_errors = [
            self.sensitive_filter.filter_string(error) for error in errors
        ]

        self.logger.warning(
            f"Skill validation failed: {skill_id} ({validation_type})",
            extra={
                "skill_id": skill_id,
                "validation_type": validation_type,
                "errors": filtered_errors,
                "event": "skill_validation_failure",
            },
        )

    def _update_performance_baseline(self, metric_name: str, value: float) -> None:
        """Update performance baseline for regression detection."""
        now = datetime.utcnow()

        if metric_name not in self._performance_baselines:
            # Initialize new baseline
            self._performance_baselines[metric_name] = PerformanceBaseline(
                metric_name=metric_name,
                baseline_value=value,
                measurement_count=1,
                last_updated=now,
                confidence_interval=(value * 0.9, value * 1.1),  # Initial 10% tolerance
            )
        else:
            baseline = self._performance_baselines[metric_name]

            # Update baseline using exponential moving average
            alpha = 0.1  # Smoothing factor
            baseline.baseline_value = (
                alpha * value + (1 - alpha) * baseline.baseline_value
            )
            baseline.measurement_count += 1
            baseline.last_updated = now

            # Update confidence interval (more sophisticated calculation could be used)
            std_factor = min(0.3, 0.05 + 0.1 / max(1, baseline.measurement_count - 1))
            baseline.confidence_interval = (
                baseline.baseline_value * (1 - std_factor),
                baseline.baseline_value * (1 + std_factor),
            )

            # Check for performance regression
            if value > baseline.confidence_interval[1] * (
                1 + self.config.performance_degradation_threshold
            ):
                self.logger.warning(
                    f"Performance regression detected: {metric_name}",
                    extra={
                        "metric_name": metric_name,
                        "current_value": value,
                        "baseline_value": baseline.baseline_value,
                        "confidence_interval": baseline.confidence_interval,
                        "regression_threshold": self.config.performance_degradation_threshold,
                        "event": "performance_regression",
                    },
                )

    def perform_health_check(self, component: str, check_function) -> HealthCheckResult:
        """Perform health check for a system component."""
        # Check cache first
        cache_key = f"health_{component}"
        if cache_key in self._health_cache:
            cached_result = self._health_cache[cache_key]
            if datetime.utcnow() - cached_result.timestamp < self._health_cache_timeout:
                return cached_result

        start_time = time.time()

        try:
            # Perform health check with timeout
            result = check_function()
            response_time = (time.time() - start_time) * 1000  # Convert to ms

            health_result = HealthCheckResult(
                component=component,
                status="healthy" if result.get("healthy", True) else "degraded",
                message=result.get("message", "Component is healthy"),
                response_time_ms=response_time,
                timestamp=datetime.utcnow(),
                details=result.get("details", {}),
            )

        except Exception as e:
            response_time = (time.time() - start_time) * 1000

            health_result = HealthCheckResult(
                component=component,
                status="unhealthy",
                message=f"Health check failed: {str(e)}",
                response_time_ms=response_time,
                timestamp=datetime.utcnow(),
                details={"error": str(e)},
            )

        # Cache result
        self._health_cache[cache_key] = health_result

        # Log health check result
        self.logger.info(
            f"Health check completed: {component} - {health_result.status}",
            extra={
                "component": component,
                "status": health_result.status,
                "response_time_ms": health_result.response_time_ms,
                "message": health_result.message,
                "event": "health_check",
            },
        )

        return health_result

    def get_system_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of current system metrics."""
        summary = {
            "timestamp": datetime.utcnow().isoformat(),
            "service_name": self.config.service_name,
            "service_version": self.config.service_version,
            "observability_enabled": {
                "tracing": self.config.enable_tracing and OPENTELEMETRY_AVAILABLE,
                "metrics": self.config.enable_metrics and OPENTELEMETRY_AVAILABLE,
                "correlation_ids": self.config.enable_correlation_ids,
                "sensitive_filtering": self.config.enable_sensitive_data_filtering,
            },
            "performance_baselines": len(self._performance_baselines),
            "health_cache_size": len(self._health_cache),
        }

        # Add resource information if available
        if PSUTIL_AVAILABLE:
            try:
                process = psutil.Process()
                memory_info = process.memory_info()
                summary["system_resources"] = {
                    "memory_rss_mb": memory_info.rss / 1024 / 1024,
                    "memory_vms_mb": memory_info.vms / 1024 / 1024,
                    "cpu_percent": process.cpu_percent(),
                    "num_threads": process.num_threads(),
                }
            except Exception:
                pass

        return summary

    def cleanup(self) -> None:
        """Cleanup observability resources."""
        # Clear caches
        self._health_cache.clear()
        self._performance_baselines.clear()

        # Cleanup OpenTelemetry resources
        if OPENTELEMETRY_AVAILABLE:
            try:
                # Force export any remaining metrics/traces
                if self.config.enable_metrics:
                    provider = metrics.get_meter_provider()
                    if hasattr(provider, "force_flush"):
                        provider.force_flush(timeout_millis=5000)

                if self.config.enable_tracing:
                    provider = trace.get_tracer_provider()
                    if hasattr(provider, "force_flush"):
                        provider.force_flush(timeout_millis=5000)

            except Exception as e:
                self.logger.error(f"Error during observability cleanup: {e}")
