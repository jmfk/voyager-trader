"""
Configuration management for VOYAGER Trader.

This module provides centralized configuration management with environment
variable support, validation, and default values for all system components.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class DatabaseConfig(BaseModel):
    """Database configuration settings."""

    # Database connection settings
    url: str = Field(
        default="sqlite:///voyager_trader.db", description="Database connection URL"
    )
    pool_size: int = Field(
        default=10, ge=1, le=100, description="Database connection pool size"
    )
    max_overflow: int = Field(
        default=20, ge=0, le=100, description="Maximum pool overflow connections"
    )
    echo: bool = Field(default=False, description="Whether to echo SQL statements")

    # Connection timeout settings
    connect_timeout: float = Field(
        default=30.0, gt=0, description="Database connection timeout in seconds"
    )
    query_timeout: float = Field(
        default=60.0, gt=0, description="Query execution timeout in seconds"
    )

    # Backup and maintenance settings
    backup_enabled: bool = Field(
        default=True, description="Whether to enable automatic backups"
    )
    backup_directory: Optional[str] = Field(
        default=None, description="Directory for database backups"
    )
    backup_interval_hours: int = Field(
        default=24, ge=1, le=168, description="Backup interval in hours"
    )
    backup_retention_days: int = Field(
        default=30, ge=1, le=365, description="Backup retention period in days"
    )

    # Performance settings
    enable_wal_mode: bool = Field(
        default=True, description="Enable WAL mode for SQLite"
    )
    enable_foreign_keys: bool = Field(
        default=True, description="Enable foreign key constraints"
    )
    cache_size_mb: int = Field(
        default=64, ge=1, le=1024, description="Database cache size in MB"
    )

    @field_validator("url")
    @classmethod
    def validate_database_url(cls, v: str) -> str:
        """Validate database URL format."""
        if not v:
            raise ValueError("Database URL cannot be empty")

        # Basic URL validation
        if not (
            v.startswith(("sqlite:///", "postgresql://", "mysql://", "mongodb://"))
        ):
            raise ValueError("Unsupported database URL scheme")

        return v

    @field_validator("backup_directory")
    @classmethod
    def validate_backup_directory(cls, v: Optional[str]) -> Optional[str]:
        """Validate backup directory path."""
        if v is not None:
            backup_path = Path(v)
            if not backup_path.exists():
                try:
                    backup_path.mkdir(parents=True, exist_ok=True)
                except (OSError, PermissionError) as e:
                    raise ValueError(f"Cannot create backup directory {v}: {e}")
        return v


class AuditConfig(BaseModel):
    """Audit logging configuration settings."""

    enabled: bool = Field(default=True, description="Whether audit logging is enabled")
    log_level: str = Field(default="INFO", description="Audit log level")
    retention_days: int = Field(
        default=365,
        ge=1,
        le=2555,  # 7 years max
        description="Audit log retention period in days",
    )

    # What to audit
    audit_all_entities: bool = Field(
        default=True, description="Audit all entity operations"
    )
    audit_user_actions: bool = Field(default=True, description="Audit user actions")
    audit_system_events: bool = Field(default=True, description="Audit system events")
    audit_trading_operations: bool = Field(
        default=True, description="Audit trading operations"
    )
    audit_configuration_changes: bool = Field(
        default=True, description="Audit configuration changes"
    )

    # Performance settings
    batch_size: int = Field(
        default=100, ge=1, le=1000, description="Batch size for audit log processing"
    )
    async_logging: bool = Field(default=True, description="Use asynchronous logging")

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level. Must be one of: {valid_levels}")
        return v.upper()


class SecurityConfig(BaseModel):
    """Security configuration settings."""

    # JWT settings
    jwt_secret: Optional[str] = Field(default=None, description="JWT secret key")
    jwt_algorithm: str = Field(default="HS256", description="JWT signing algorithm")
    jwt_expire_minutes: int = Field(
        default=30,
        ge=1,
        le=1440,  # 24 hours max
        description="JWT expiration time in minutes",
    )

    # Password settings
    password_min_length: int = Field(
        default=8, ge=6, le=128, description="Minimum password length"
    )
    require_password_complexity: bool = Field(
        default=True, description="Require complex passwords"
    )

    # Rate limiting
    rate_limit_enabled: bool = Field(default=True, description="Enable rate limiting")
    rate_limit_requests_per_minute: int = Field(
        default=60, ge=1, le=10000, description="Rate limit: requests per minute"
    )

    # CORS settings
    cors_origins: List[str] = Field(
        default_factory=lambda: ["http://localhost:3001"],
        description="Allowed CORS origins",
    )
    cors_credentials: bool = Field(default=True, description="Allow CORS credentials")


class PerformanceConfig(BaseModel):
    """Performance and optimization configuration."""

    # Cache settings
    cache_enabled: bool = Field(default=True, description="Enable caching")
    cache_ttl_seconds: int = Field(
        default=300,
        ge=1,
        le=86400,  # 24 hours max
        description="Default cache TTL in seconds",
    )
    cache_size_mb: int = Field(
        default=128, ge=1, le=1024, description="Cache size limit in MB"
    )

    # Connection pooling
    max_connections: int = Field(
        default=50, ge=1, le=500, description="Maximum concurrent connections"
    )
    connection_timeout: float = Field(
        default=30.0, gt=0, description="Connection timeout in seconds"
    )

    # Query optimization
    query_batch_size: int = Field(
        default=1000,
        ge=1,
        le=10000,
        description="Default batch size for bulk operations",
    )
    enable_query_logging: bool = Field(
        default=False, description="Enable query performance logging"
    )


class TradingConfig(BaseModel):
    """Trading system configuration."""

    # Risk management
    max_position_size_percent: float = Field(
        default=5.0,
        ge=0.1,
        le=50.0,
        description="Maximum position size as percentage of portfolio",
    )
    max_portfolio_risk_percent: float = Field(
        default=2.0, ge=0.1, le=10.0, description="Maximum portfolio risk as percentage"
    )
    daily_loss_limit_percent: float = Field(
        default=5.0, ge=0.1, le=20.0, description="Daily loss limit as percentage"
    )

    # Order management
    default_order_timeout_minutes: int = Field(
        default=60, ge=1, le=1440, description="Default order timeout in minutes"
    )
    enable_position_tracking: bool = Field(
        default=True, description="Enable position tracking"
    )
    enable_pnl_calculation: bool = Field(
        default=True, description="Enable P&L calculation"
    )


class VoyagerTraderConfig(BaseModel):
    """Main configuration for VOYAGER Trader system."""

    # System settings
    environment: str = Field(default="development", description="Runtime environment")
    debug: bool = Field(default=False, description="Enable debug mode")
    log_level: str = Field(default="INFO", description="System log level")

    # Component configurations
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    audit: AuditConfig = Field(default_factory=AuditConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    trading: TradingConfig = Field(default_factory=TradingConfig)

    # Feature flags
    enable_persistence: bool = Field(
        default=True, description="Enable persistent storage"
    )
    enable_audit_logging: bool = Field(default=True, description="Enable audit logging")
    enable_metrics: bool = Field(default=True, description="Enable metrics collection")

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """Validate environment setting."""
        valid_environments = {"development", "testing", "staging", "production"}
        if v.lower() not in valid_environments:
            raise ValueError(
                f"Invalid environment. Must be one of: {valid_environments}"
            )
        return v.lower()

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level. Must be one of: {valid_levels}")
        return v.upper()

    @classmethod
    def from_env(cls) -> "VoyagerTraderConfig":
        """
        Create configuration from environment variables.

        Returns:
            Configuration instance with values from environment
        """
        # System settings
        config_data = {
            "environment": os.getenv("VOYAGER_ENVIRONMENT", "development"),
            "debug": os.getenv("VOYAGER_DEBUG", "false").lower() == "true",
            "log_level": os.getenv("VOYAGER_LOG_LEVEL", "INFO"),
            "enable_persistence": os.getenv(
                "VOYAGER_ENABLE_PERSISTENCE", "true"
            ).lower()
            == "true",
            "enable_audit_logging": os.getenv(
                "VOYAGER_ENABLE_AUDIT_LOGGING", "true"
            ).lower()
            == "true",
            "enable_metrics": os.getenv("VOYAGER_ENABLE_METRICS", "true").lower()
            == "true",
        }

        # Database configuration
        config_data["database"] = {
            "url": os.getenv("DATABASE_URL", "sqlite:///voyager_trader.db"),
            "pool_size": int(os.getenv("DB_POOL_SIZE", "10")),
            "max_overflow": int(os.getenv("DB_MAX_OVERFLOW", "20")),
            "echo": os.getenv("DB_ECHO", "false").lower() == "true",
            "connect_timeout": float(os.getenv("DB_CONNECT_TIMEOUT", "30.0")),
            "query_timeout": float(os.getenv("DB_QUERY_TIMEOUT", "60.0")),
            "backup_enabled": os.getenv("DB_BACKUP_ENABLED", "true").lower() == "true",
            "backup_directory": os.getenv("DB_BACKUP_DIRECTORY"),
            "backup_interval_hours": int(os.getenv("DB_BACKUP_INTERVAL_HOURS", "24")),
            "backup_retention_days": int(os.getenv("DB_BACKUP_RETENTION_DAYS", "30")),
            "enable_wal_mode": os.getenv("DB_ENABLE_WAL_MODE", "true").lower()
            == "true",
            "enable_foreign_keys": os.getenv("DB_ENABLE_FOREIGN_KEYS", "true").lower()
            == "true",
            "cache_size_mb": int(os.getenv("DB_CACHE_SIZE_MB", "64")),
        }

        # Audit configuration
        config_data["audit"] = {
            "enabled": os.getenv("AUDIT_ENABLED", "true").lower() == "true",
            "log_level": os.getenv("AUDIT_LOG_LEVEL", "INFO"),
            "retention_days": int(os.getenv("AUDIT_RETENTION_DAYS", "365")),
            "audit_all_entities": os.getenv("AUDIT_ALL_ENTITIES", "true").lower()
            == "true",
            "audit_user_actions": os.getenv("AUDIT_USER_ACTIONS", "true").lower()
            == "true",
            "audit_system_events": os.getenv("AUDIT_SYSTEM_EVENTS", "true").lower()
            == "true",
            "audit_trading_operations": os.getenv(
                "AUDIT_TRADING_OPERATIONS", "true"
            ).lower()
            == "true",
            "audit_configuration_changes": os.getenv(
                "AUDIT_CONFIG_CHANGES", "true"
            ).lower()
            == "true",
            "batch_size": int(os.getenv("AUDIT_BATCH_SIZE", "100")),
            "async_logging": os.getenv("AUDIT_ASYNC_LOGGING", "true").lower() == "true",
        }

        # Security configuration
        cors_origins = os.getenv("VOYAGER_CORS_ORIGINS", "http://localhost:3001")
        config_data["security"] = {
            "jwt_secret": os.getenv("VOYAGER_JWT_SECRET"),
            "jwt_algorithm": os.getenv("VOYAGER_JWT_ALGORITHM", "HS256"),
            "jwt_expire_minutes": int(os.getenv("VOYAGER_JWT_EXPIRE_MINUTES", "30")),
            "password_min_length": int(os.getenv("PASSWORD_MIN_LENGTH", "8")),
            "require_password_complexity": os.getenv(
                "REQUIRE_PASSWORD_COMPLEXITY", "true"
            ).lower()
            == "true",
            "rate_limit_enabled": os.getenv("RATE_LIMIT_ENABLED", "true").lower()
            == "true",
            "rate_limit_requests_per_minute": int(
                os.getenv("RATE_LIMIT_PER_MINUTE", "60")
            ),
            "cors_origins": [origin.strip() for origin in cors_origins.split(",")],
            "cors_credentials": os.getenv("CORS_CREDENTIALS", "true").lower() == "true",
        }

        # Performance configuration
        config_data["performance"] = {
            "cache_enabled": os.getenv("CACHE_ENABLED", "true").lower() == "true",
            "cache_ttl_seconds": int(os.getenv("CACHE_TTL_SECONDS", "300")),
            "cache_size_mb": int(os.getenv("CACHE_SIZE_MB", "128")),
            "max_connections": int(os.getenv("MAX_CONNECTIONS", "50")),
            "connection_timeout": float(os.getenv("CONNECTION_TIMEOUT", "30.0")),
            "query_batch_size": int(os.getenv("QUERY_BATCH_SIZE", "1000")),
            "enable_query_logging": os.getenv("ENABLE_QUERY_LOGGING", "false").lower()
            == "true",
        }

        # Trading configuration
        config_data["trading"] = {
            "max_position_size_percent": float(
                os.getenv("MAX_POSITION_SIZE_PERCENT", "5.0")
            ),
            "max_portfolio_risk_percent": float(
                os.getenv("MAX_PORTFOLIO_RISK_PERCENT", "2.0")
            ),
            "daily_loss_limit_percent": float(
                os.getenv("DAILY_LOSS_LIMIT_PERCENT", "5.0")
            ),
            "default_order_timeout_minutes": int(
                os.getenv("DEFAULT_ORDER_TIMEOUT_MINUTES", "60")
            ),
            "enable_position_tracking": os.getenv(
                "ENABLE_POSITION_TRACKING", "true"
            ).lower()
            == "true",
            "enable_pnl_calculation": os.getenv(
                "ENABLE_PNL_CALCULATION", "true"
            ).lower()
            == "true",
        }

        return cls.model_validate(config_data)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.

        Returns:
            Configuration as dictionary
        """
        return self.model_dump()

    def get_database_url(self) -> str:
        """
        Get database connection URL.

        Returns:
            Database URL string
        """
        return self.database.url

    def is_production(self) -> bool:
        """
        Check if running in production environment.

        Returns:
            True if production environment
        """
        return self.environment == "production"

    def is_debug_enabled(self) -> bool:
        """
        Check if debug mode is enabled.

        Returns:
            True if debug mode enabled
        """
        return self.debug or self.environment == "development"


# Global configuration instance
_config: Optional[VoyagerTraderConfig] = None


def get_settings() -> VoyagerTraderConfig:
    """
    Get the global configuration instance.

    Returns:
        Global configuration instance
    """
    global _config
    if _config is None:
        _config = VoyagerTraderConfig.from_env()

    return _config


def reload_settings() -> VoyagerTraderConfig:
    """
    Reload configuration from environment variables.

    Returns:
        Reloaded configuration instance
    """
    global _config
    _config = VoyagerTraderConfig.from_env()
    return _config


def update_settings(**kwargs) -> VoyagerTraderConfig:
    """
    Update configuration with provided values.

    Args:
        **kwargs: Configuration values to update

    Returns:
        Updated configuration instance
    """
    global _config
    if _config is None:
        _config = VoyagerTraderConfig.from_env()

    # Create updated configuration
    updated_data = _config.model_dump()
    updated_data.update(kwargs)

    _config = VoyagerTraderConfig.model_validate(updated_data)
    return _config
