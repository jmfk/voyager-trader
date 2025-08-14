"""
Global test configuration for VOYAGER Trader tests.

This file automatically sets up the test environment for all tests,
including disabling OpenTelemetry to prevent CI/test issues.
"""

# IMPORTANT: Set environment variables BEFORE any imports
# to ensure they're available when modules are loaded
import os

os.environ["DISABLE_OPENTELEMETRY"] = "true"
os.environ["TESTING"] = "true"
os.environ["ENVIRONMENT"] = "test"

import pytest  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def disable_opentelemetry():
    """Automatically disable OpenTelemetry for all test sessions."""
    # This runs once per test session and ensures OpenTelemetry is disabled
    # even if the environment variable isn't set externally
    original_value = os.environ.get("DISABLE_OPENTELEMETRY")
    os.environ["DISABLE_OPENTELEMETRY"] = "true"

    yield

    # Restore original value if it existed
    if original_value is not None:
        os.environ["DISABLE_OPENTELEMETRY"] = original_value
    else:
        os.environ.pop("DISABLE_OPENTELEMETRY", None)


@pytest.fixture(autouse=True)
def test_environment():
    """Set up test environment for each test."""
    # Ensure we're always in test mode
    os.environ["TESTING"] = "true"
    yield
