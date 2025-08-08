"""Market data specific exceptions for granular error handling."""

from typing import Any, Dict, Optional


class MarketDataError(Exception):
    """Base exception for all market data related errors."""

    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        symbol: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.provider = provider
        self.symbol = symbol
        self.details = details or {}


class DataSourceError(MarketDataError):
    """Base exception for data source related errors."""

    pass


class ConnectionError(DataSourceError):
    """Connection-related errors (network, timeout, etc.)."""

    pass


class AuthenticationError(DataSourceError):
    """Authentication/authorization errors (API key, permissions, etc.)."""

    pass


class RateLimitError(DataSourceError):
    """Rate limiting errors."""

    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        retry_after: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(message, provider=provider, **kwargs)
        self.retry_after = retry_after


class QuotaExceededError(DataSourceError):
    """API quota exceeded errors."""

    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        quota_type: Optional[str] = None,
        reset_time: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(message, provider=provider, **kwargs)
        self.quota_type = quota_type
        self.reset_time = reset_time


class DataValidationError(MarketDataError):
    """Data validation and quality errors."""

    def __init__(
        self,
        message: str,
        validation_type: Optional[str] = None,
        expected_value: Optional[Any] = None,
        actual_value: Optional[Any] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.validation_type = validation_type
        self.expected_value = expected_value
        self.actual_value = actual_value


class DataNotFoundError(MarketDataError):
    """Data not available/found errors."""

    pass


class SymbolNotSupportedError(MarketDataError):
    """Symbol not supported by provider errors."""

    pass


class TimeframeNotSupportedError(MarketDataError):
    """Timeframe not supported by provider errors."""

    def __init__(
        self,
        message: str,
        timeframe: Optional[str] = None,
        supported_timeframes: Optional[list] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.timeframe = timeframe
        self.supported_timeframes = supported_timeframes or []


class CacheError(MarketDataError):
    """Cache-related errors."""

    pass


class CacheCorruptionError(CacheError):
    """Cache data corruption errors."""

    pass


class ServiceUnavailableError(DataSourceError):
    """Service temporarily unavailable errors."""

    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        estimated_recovery_time: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(message, provider=provider, **kwargs)
        self.estimated_recovery_time = estimated_recovery_time


class ConfigurationError(MarketDataError):
    """Configuration-related errors."""

    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        expected_type: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.config_key = config_key
        self.expected_type = expected_type


class SecurityError(MarketDataError):
    """Security-related errors (oversized responses, malformed data, etc.)."""

    def __init__(self, message: str, security_type: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.security_type = security_type


# Convenience function to create appropriate exception based on HTTP status
def create_http_error(
    status_code: int, message: str, provider: Optional[str] = None, **kwargs
) -> DataSourceError:
    """Create appropriate exception based on HTTP status code."""
    if status_code == 401:
        return AuthenticationError(
            f"Authentication failed: {message}", provider=provider, **kwargs
        )
    elif status_code == 403:
        return AuthenticationError(
            f"Permission denied: {message}", provider=provider, **kwargs
        )
    elif status_code == 404:
        return DataNotFoundError(
            f"Data not found: {message}", provider=provider, **kwargs
        )
    elif status_code == 429:
        return RateLimitError(
            f"Rate limit exceeded: {message}", provider=provider, **kwargs
        )
    elif status_code == 503:
        return ServiceUnavailableError(
            f"Service unavailable: {message}", provider=provider, **kwargs
        )
    elif 500 <= status_code < 600:
        return DataSourceError(
            f"Server error ({status_code}): {message}", provider=provider, **kwargs
        )
    elif 400 <= status_code < 500:
        return DataSourceError(
            f"Client error ({status_code}): {message}", provider=provider, **kwargs
        )
    else:
        return ConnectionError(
            f"HTTP error ({status_code}): {message}", provider=provider, **kwargs
        )
