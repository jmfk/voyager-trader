"""Market data integration system for VOYAGER-Trader."""

from .cache import DataCache
from .cache_warming import (
    AdaptiveWarming,
    CacheWarmer,
    PopularSymbolsWarming,
    ScheduledWarming,
)
from .exceptions import (
    AuthenticationError,
    CacheError,
    ConfigurationError,
    ConnectionError,
    DataNotFoundError,
    DataSourceError,
    DataValidationError,
    MarketDataError,
    QuotaExceededError,
    RateLimitError,
    SecurityError,
    ServiceUnavailableError,
    SymbolNotSupportedError,
    TimeframeNotSupportedError,
)
from .interfaces import DataFetcher, DataSource
from .manager import DataSourceManager
from .monitoring import (
    MetricsCollector,
    get_health_status,
    get_metrics_collector,
    get_system_metrics,
)
from .normalizer import DataNormalizer
from .rate_limiter import RateLimiter
from .service import MarketDataService
from .sources import AlphaVantageDataSource, MockDataSource, YahooFinanceDataSource
from .types import Symbol, create_symbol, ensure_symbol_object, normalize_symbol
from .validator import DataValidator

__all__ = [
    "DataSource",
    "DataFetcher",
    "MarketDataService",
    "DataSourceManager",
    "AlphaVantageDataSource",
    "YahooFinanceDataSource",
    "MockDataSource",
    "DataCache",
    "CacheWarmer",
    "PopularSymbolsWarming",
    "ScheduledWarming",
    "AdaptiveWarming",
    "DataValidator",
    "DataNormalizer",
    "RateLimiter",
    "MetricsCollector",
    "get_metrics_collector",
    "get_system_metrics",
    "get_health_status",
    "Symbol",
    "create_symbol",
    "ensure_symbol_object",
    "normalize_symbol",
    # Exceptions
    "MarketDataError",
    "DataSourceError",
    "ConnectionError",
    "AuthenticationError",
    "RateLimitError",
    "QuotaExceededError",
    "DataValidationError",
    "DataNotFoundError",
    "SymbolNotSupportedError",
    "TimeframeNotSupportedError",
    "CacheError",
    "ServiceUnavailableError",
    "ConfigurationError",
    "SecurityError",
]
