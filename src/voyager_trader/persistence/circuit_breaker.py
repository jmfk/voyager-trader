"""
Circuit Breaker pattern implementation for database connections.

This module provides circuit breaker functionality to prevent cascading failures
when database connections are consistently failing, with automatic recovery
and comprehensive monitoring.
"""

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class CircuitBreakerState(Enum):
    """Circuit breaker states following the standard pattern."""

    CLOSED = "closed"  # Normal operation, requests pass through
    OPEN = "open"  # Failure threshold exceeded, requests are blocked
    HALF_OPEN = "half_open"  # Testing if service has recovered


@dataclass
class CircuitBreakerMetrics:
    """Metrics and statistics for circuit breaker monitoring."""

    # Request counters
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    rejected_requests: int = 0

    # State transition counters
    state_transitions: Dict[str, int] = field(
        default_factory=lambda: {
            "closed_to_open": 0,
            "open_to_half_open": 0,
            "half_open_to_closed": 0,
            "half_open_to_open": 0,
        }
    )

    # Timing metrics
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    last_state_change: datetime = field(default_factory=datetime.utcnow)

    # Recent failure tracking
    recent_failures: deque = field(default_factory=lambda: deque(maxlen=100))

    @property
    def failure_rate(self) -> float:
        """Calculate current failure rate."""
        if self.total_requests == 0:
            return 0.0
        return self.failed_requests / self.total_requests

    @property
    def success_rate(self) -> float:
        """Calculate current success rate."""
        return 1.0 - self.failure_rate

    @property
    def time_in_current_state(self) -> timedelta:
        """Get time spent in current state."""
        return datetime.utcnow() - self.last_state_change


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""

    # Failure detection
    failure_threshold: int = 5  # Number of failures to open circuit
    failure_rate_threshold: float = 0.5  # Percentage (0.0-1.0) failure rate to open
    minimum_requests: int = 10  # Minimum requests before evaluating failure rate

    # Timing configuration
    timeout_duration: float = 60.0  # Seconds to wait before trying half-open
    half_open_max_requests: int = 3  # Max requests to allow in half-open state

    # Sliding window configuration
    sliding_window_size: int = 100  # Number of recent requests to track
    sliding_window_duration: float = 300.0  # Time window in seconds

    # Recovery configuration
    recovery_timeout: float = 30.0  # Seconds to wait after last failure
    success_threshold: int = 3  # Consecutive successes needed to close circuit

    # Monitoring
    metrics_enabled: bool = True
    log_state_changes: bool = True


class CircuitBreakerException(Exception):
    """Exception raised when circuit breaker is open."""

    def __init__(
        self, message: str, state: CircuitBreakerState, metrics: CircuitBreakerMetrics
    ):
        super().__init__(message)
        self.state = state
        self.metrics = metrics


class DatabaseCircuitBreaker:
    """
    Circuit breaker implementation for database operations.

    Prevents cascading failures by monitoring database operation success/failure
    rates and temporarily blocking requests when failure thresholds are exceeded.
    """

    def __init__(self, name: str, config: CircuitBreakerConfig):
        """
        Initialize database circuit breaker.

        Args:
            name: Unique name for this circuit breaker
            config: Circuit breaker configuration
        """
        self.name = name
        self.config = config

        # Circuit breaker state
        self._state = CircuitBreakerState.CLOSED
        self._metrics = CircuitBreakerMetrics()
        self._lock = asyncio.Lock()

        # Half-open state tracking
        self._half_open_requests = 0
        self._consecutive_successes = 0
        self._last_failure_time: Optional[float] = None

        # Request tracking for sliding window
        self._request_history: deque = deque(maxlen=config.sliding_window_size)

        logger.info(f"Initialized DatabaseCircuitBreaker '{name}'")

    @property
    def state(self) -> CircuitBreakerState:
        """Get current circuit breaker state."""
        return self._state

    @property
    def metrics(self) -> CircuitBreakerMetrics:
        """Get current metrics."""
        return self._metrics

    @property
    def is_closed(self) -> bool:
        """Check if circuit breaker is closed (normal operation)."""
        return self._state == CircuitBreakerState.CLOSED

    @property
    def is_open(self) -> bool:
        """Check if circuit breaker is open (blocking requests)."""
        return self._state == CircuitBreakerState.OPEN

    @property
    def is_half_open(self) -> bool:
        """Check if circuit breaker is half-open (testing recovery)."""
        return self._state == CircuitBreakerState.HALF_OPEN

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute a function through the circuit breaker.

        Args:
            func: Function to execute
            *args: Function positional arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result if successful

        Raises:
            CircuitBreakerException: If circuit breaker is open
            Exception: Any exception raised by the function
        """
        async with self._lock:
            # Check if we should allow the request
            if not await self._should_allow_request():
                self._metrics.rejected_requests += 1
                raise CircuitBreakerException(
                    f"Circuit breaker '{self.name}' is {self._state.value}",
                    self._state,
                    self._metrics,
                )

        # Execute the function and handle results
        start_time = time.time()
        try:
            result = (
                await func(*args, **kwargs)
                if asyncio.iscoroutinefunction(func)
                else func(*args, **kwargs)
            )
            await self._record_success(start_time)
            return result

        except Exception as e:
            await self._record_failure(start_time, e)
            raise

    async def _should_allow_request(self) -> bool:
        """
        Determine if a request should be allowed through the circuit breaker.

        Returns:
            True if request should be allowed, False otherwise
        """
        current_time = time.time()

        if self._state == CircuitBreakerState.CLOSED:
            return True

        elif self._state == CircuitBreakerState.OPEN:
            # Check if we should transition to half-open
            if (
                self._last_failure_time
                and current_time - self._last_failure_time
                >= self.config.timeout_duration
            ):
                await self._transition_to_half_open()
                return True
            return False

        elif self._state == CircuitBreakerState.HALF_OPEN:
            # Allow limited requests in half-open state
            if self._half_open_requests < self.config.half_open_max_requests:
                self._half_open_requests += 1
                return True
            return False

        return False

    async def _record_success(self, start_time: float) -> None:
        """
        Record a successful operation.

        Args:
            start_time: Operation start time
        """
        async with self._lock:
            duration = time.time() - start_time
            current_time = datetime.utcnow()

            # Update metrics
            self._metrics.total_requests += 1
            self._metrics.successful_requests += 1
            self._metrics.last_success_time = current_time

            # Add to request history
            self._request_history.append(
                {"timestamp": current_time, "success": True, "duration": duration}
            )

            # Handle state-specific success logic
            if self._state == CircuitBreakerState.HALF_OPEN:
                self._consecutive_successes += 1

                # Check if we should close the circuit
                if self._consecutive_successes >= self.config.success_threshold:
                    await self._transition_to_closed()

            elif self._state == CircuitBreakerState.CLOSED:
                self._consecutive_successes += 1

            logger.debug(f"Circuit breaker '{self.name}' recorded success")

    async def _record_failure(self, start_time: float, error: Exception) -> None:
        """
        Record a failed operation.

        Args:
            start_time: Operation start time
            error: The exception that occurred
        """
        async with self._lock:
            duration = time.time() - start_time
            current_time = datetime.utcnow()

            # Update metrics
            self._metrics.total_requests += 1
            self._metrics.failed_requests += 1
            self._metrics.last_failure_time = current_time
            self._last_failure_time = time.time()

            # Add to request history and recent failures
            failure_record = {
                "timestamp": current_time,
                "success": False,
                "duration": duration,
                "error": str(error),
                "error_type": type(error).__name__,
            }

            self._request_history.append(failure_record)
            self._metrics.recent_failures.append(failure_record)

            # Reset consecutive successes
            self._consecutive_successes = 0

            # Handle state-specific failure logic
            if self._state == CircuitBreakerState.CLOSED:
                if await self._should_open_circuit():
                    await self._transition_to_open()

            elif self._state == CircuitBreakerState.HALF_OPEN:
                # Any failure in half-open state returns to open
                await self._transition_to_open()

            logger.warning(
                f"Circuit breaker '{self.name}' recorded failure: "
                f"{type(error).__name__}: {error}"
            )

    async def _should_open_circuit(self) -> bool:
        """
        Determine if the circuit should be opened based on failure criteria.

        Returns:
            True if circuit should be opened, False otherwise
        """
        # Check absolute failure threshold
        recent_failures = len(
            [
                req
                for req in self._request_history
                if not req["success"]
                and (datetime.utcnow() - req["timestamp"]).total_seconds()
                <= self.config.sliding_window_duration
            ]
        )

        if recent_failures >= self.config.failure_threshold:
            logger.info(
                f"Circuit breaker '{self.name}' failure threshold reached: "
                f"{recent_failures}"
            )
            return True

        # Check failure rate threshold (only if we have minimum requests)
        if self._metrics.total_requests >= self.config.minimum_requests:
            recent_requests = [
                req
                for req in self._request_history
                if (datetime.utcnow() - req["timestamp"]).total_seconds()
                <= self.config.sliding_window_duration
            ]

            if len(recent_requests) >= self.config.minimum_requests:
                recent_failure_rate = len(
                    [req for req in recent_requests if not req["success"]]
                ) / len(recent_requests)

                if recent_failure_rate >= self.config.failure_rate_threshold:
                    logger.info(
                        f"Circuit breaker '{self.name}' failure rate threshold "
                        f"reached: "
                        f"{recent_failure_rate:.2%}"
                    )
                    return True

        return False

    async def _transition_to_closed(self) -> None:
        """Transition circuit breaker to CLOSED state."""
        old_state = self._state
        self._state = CircuitBreakerState.CLOSED
        self._metrics.last_state_change = datetime.utcnow()
        self._metrics.state_transitions["half_open_to_closed"] += 1

        # Reset half-open state tracking
        self._half_open_requests = 0

        if self.config.log_state_changes:
            logger.info(
                f"Circuit breaker '{self.name}' transitioned from "
                f"{old_state.value} to CLOSED"
            )

    async def _transition_to_open(self) -> None:
        """Transition circuit breaker to OPEN state."""
        old_state = self._state
        self._state = CircuitBreakerState.OPEN
        self._metrics.last_state_change = datetime.utcnow()

        if old_state == CircuitBreakerState.CLOSED:
            self._metrics.state_transitions["closed_to_open"] += 1
        elif old_state == CircuitBreakerState.HALF_OPEN:
            self._metrics.state_transitions["half_open_to_open"] += 1

        # Reset tracking variables
        self._half_open_requests = 0
        self._consecutive_successes = 0

        if self.config.log_state_changes:
            logger.warning(
                f"Circuit breaker '{self.name}' transitioned from "
                f"{old_state.value} to OPEN"
            )

    async def _transition_to_half_open(self) -> None:
        """Transition circuit breaker to HALF_OPEN state."""
        old_state = self._state
        self._state = CircuitBreakerState.HALF_OPEN
        self._metrics.last_state_change = datetime.utcnow()
        self._metrics.state_transitions["open_to_half_open"] += 1

        # Reset half-open tracking
        self._half_open_requests = 0
        self._consecutive_successes = 0

        if self.config.log_state_changes:
            logger.info(
                f"Circuit breaker '{self.name}' transitioned from "
                f"{old_state.value} to HALF_OPEN"
            )

    async def reset(self) -> None:
        """Manually reset circuit breaker to CLOSED state."""
        async with self._lock:
            old_state = self._state
            self._state = CircuitBreakerState.CLOSED
            self._metrics.last_state_change = datetime.utcnow()

            # Reset all counters
            self._half_open_requests = 0
            self._consecutive_successes = 0
            self._last_failure_time = None

            logger.info(
                f"Circuit breaker '{self.name}' manually reset from "
                f"{old_state.value} to CLOSED"
            )

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive circuit breaker statistics.

        Returns:
            Dictionary with circuit breaker statistics
        """
        recent_requests = [
            req
            for req in self._request_history
            if (datetime.utcnow() - req["timestamp"]).total_seconds()
            <= self.config.sliding_window_duration
        ]

        return {
            # Current state
            "name": self.name,
            "state": self._state.value,
            "time_in_state": self._metrics.time_in_current_state.total_seconds(),
            # Request metrics
            "total_requests": self._metrics.total_requests,
            "successful_requests": self._metrics.successful_requests,
            "failed_requests": self._metrics.failed_requests,
            "rejected_requests": self._metrics.rejected_requests,
            "failure_rate": self._metrics.failure_rate,
            "success_rate": self._metrics.success_rate,
            # Recent activity (sliding window)
            "recent_requests_count": len(recent_requests),
            "recent_failures_count": len(
                [req for req in recent_requests if not req["success"]]
            ),
            "recent_failure_rate": (
                len([req for req in recent_requests if not req["success"]])
                / len(recent_requests)
                if recent_requests
                else 0.0
            ),
            # State transitions
            "state_transitions": dict(self._metrics.state_transitions),
            # Timing information
            "last_failure_time": self._metrics.last_failure_time,
            "last_success_time": self._metrics.last_success_time,
            "consecutive_successes": self._consecutive_successes,
            # Configuration
            "config": {
                "failure_threshold": self.config.failure_threshold,
                "failure_rate_threshold": self.config.failure_rate_threshold,
                "timeout_duration": self.config.timeout_duration,
                "minimum_requests": self.config.minimum_requests,
            },
        }


class CircuitBreakerManager:
    """
    Manager for multiple circuit breakers with centralized monitoring.

    Provides centralized management and monitoring of circuit breakers
    across different database operations and components.
    """

    def __init__(self):
        """Initialize circuit breaker manager."""
        self._circuit_breakers: Dict[str, DatabaseCircuitBreaker] = {}
        self._global_metrics = {
            "total_breakers": 0,
            "open_breakers": 0,
            "half_open_breakers": 0,
            "closed_breakers": 0,
        }

        logger.info("Initialized CircuitBreakerManager")

    def create_circuit_breaker(
        self, name: str, config: Optional[CircuitBreakerConfig] = None
    ) -> DatabaseCircuitBreaker:
        """
        Create a new circuit breaker.

        Args:
            name: Unique name for the circuit breaker
            config: Optional configuration (uses defaults if not provided)

        Returns:
            Created circuit breaker instance

        Raises:
            ValueError: If circuit breaker with same name already exists
        """
        if name in self._circuit_breakers:
            raise ValueError(f"Circuit breaker '{name}' already exists")

        config = config or CircuitBreakerConfig()
        circuit_breaker = DatabaseCircuitBreaker(name, config)
        self._circuit_breakers[name] = circuit_breaker
        self._global_metrics["total_breakers"] += 1

        logger.info(f"Created circuit breaker '{name}'")
        return circuit_breaker

    def get_circuit_breaker(self, name: str) -> Optional[DatabaseCircuitBreaker]:
        """
        Get circuit breaker by name.

        Args:
            name: Circuit breaker name

        Returns:
            Circuit breaker instance if found, None otherwise
        """
        return self._circuit_breakers.get(name)

    def get_or_create_circuit_breaker(
        self, name: str, config: Optional[CircuitBreakerConfig] = None
    ) -> DatabaseCircuitBreaker:
        """
        Get existing circuit breaker or create new one.

        Args:
            name: Circuit breaker name
            config: Configuration for new circuit breaker (ignored if exists)

        Returns:
            Circuit breaker instance
        """
        existing = self.get_circuit_breaker(name)
        if existing:
            return existing

        return self.create_circuit_breaker(name, config)

    async def reset_all(self) -> None:
        """Reset all circuit breakers to CLOSED state."""
        for cb in self._circuit_breakers.values():
            await cb.reset()

        logger.info("Reset all circuit breakers")

    def get_global_statistics(self) -> Dict[str, Any]:
        """
        Get global statistics for all circuit breakers.

        Returns:
            Dictionary with global circuit breaker statistics
        """
        # Update global metrics
        self._global_metrics.update(
            {
                "total_breakers": len(self._circuit_breakers),
                "open_breakers": sum(
                    1 for cb in self._circuit_breakers.values() if cb.is_open
                ),
                "half_open_breakers": sum(
                    1 for cb in self._circuit_breakers.values() if cb.is_half_open
                ),
                "closed_breakers": sum(
                    1 for cb in self._circuit_breakers.values() if cb.is_closed
                ),
            }
        )

        # Aggregate request metrics
        total_requests = sum(
            cb.metrics.total_requests for cb in self._circuit_breakers.values()
        )
        total_failures = sum(
            cb.metrics.failed_requests for cb in self._circuit_breakers.values()
        )
        total_rejections = sum(
            cb.metrics.rejected_requests for cb in self._circuit_breakers.values()
        )

        return {
            **self._global_metrics,
            "total_requests": total_requests,
            "total_failures": total_failures,
            "total_rejections": total_rejections,
            "global_failure_rate": total_failures / total_requests
            if total_requests > 0
            else 0.0,
            "circuit_breakers": [
                cb.get_statistics() for cb in self._circuit_breakers.values()
            ],
        }

    def list_circuit_breakers(self) -> List[str]:
        """
        Get list of all circuit breaker names.

        Returns:
            List of circuit breaker names
        """
        return list(self._circuit_breakers.keys())


# Global circuit breaker manager instance
_global_cb_manager: Optional[CircuitBreakerManager] = None


def get_circuit_breaker_manager() -> CircuitBreakerManager:
    """
    Get the global circuit breaker manager instance.

    Returns:
        Global circuit breaker manager
    """
    global _global_cb_manager
    if _global_cb_manager is None:
        _global_cb_manager = CircuitBreakerManager()
    return _global_cb_manager
