# Database Connection Health Checking Guide

This document describes the comprehensive connection health checking system implemented for VOYAGER-Trader's database layer.

## Overview

The connection health checking system provides robust connection validation, lifecycle management, and automatic recovery for the database connection pool. This ensures reliable database operations and prevents issues with stale or corrupted connections.

## Key Components

### HealthyConnection Class

The `HealthyConnection` class wraps standard `aiosqlite.Connection` objects with health checking capabilities:

```python
from voyager_trader.persistence.connection_health import HealthyConnection, HealthCheckConfig

# Create health check configuration
config = HealthCheckConfig(
    enabled=True,
    timeout=1.0,
    max_age=3600,  # 1 hour
    max_usage=1000,
)

# Wrap existing connection
healthy_conn = HealthyConnection(
    connection=raw_connection,
    config=config,
    connection_id="conn_1"
)

# Validate health
is_healthy = await healthy_conn.validate_health()
if is_healthy:
    # Use connection safely
    result = await healthy_conn.connection.execute("SELECT * FROM users")
```

### ConnectionHealthManager

Manages periodic health monitoring and pool-wide statistics:

```python
from voyager_trader.persistence.connection_health import ConnectionHealthManager

# Initialize health manager
health_manager = ConnectionHealthManager(config)

# Start monitoring pool
await health_manager.start_monitoring(connection_pool)

# Get pool statistics
stats = health_manager.get_pool_statistics(connection_pool)
print(f"Healthy connections: {stats['healthy_connections']}")
```

### Enhanced DatabaseManager

The `DatabaseManager` now integrates health checking automatically:

```python
from voyager_trader.persistence.database import DatabaseManager
from voyager_trader.persistence.connection_health import HealthCheckConfig

# Configure health checking
health_config = HealthCheckConfig(
    enabled=True,
    timeout=2.0,
    max_age=7200,
    interval=60,
)

# Create database manager with health checking
db_manager = DatabaseManager(
    database_url="sqlite:///app.db",
    pool_size=10,
    health_check_config=health_config,
)

await db_manager.initialize()

# Get healthy connection (automatically validated)
async with db_manager.get_connection() as conn:
    # Connection is guaranteed to be healthy
    result = await conn.execute("SELECT 1")
```

## Configuration Options

### HealthCheckConfig Parameters

```python
@dataclass
class HealthCheckConfig:
    # Health check settings
    enabled: bool = True                    # Enable/disable health checks
    timeout: float = 1.0                    # Health check timeout (seconds)
    query: str = "SELECT 1"                 # Health check query
    interval: int = 30                      # Periodic check interval (seconds)

    # Connection lifecycle limits
    max_age: int = 3600                     # Max connection age (seconds)
    max_usage: int = 1000                   # Max operations per connection
    max_idle_time: int = 300                # Max idle time (seconds)

    # Health check thresholds
    max_consecutive_failures: int = 3       # Max failures before marking unhealthy
    health_check_cache_duration: float = 5.0  # Cache duration (seconds)

    # Pool management
    min_healthy_connections: int = 2        # Minimum healthy connections
    health_check_batch_size: int = 5        # Batch size for monitoring
```

### Environment Configuration

You can configure health checking through environment variables or settings:

```python
# In your settings configuration
class DatabaseSettings:
    # Database connection
    DATABASE_URL: str = "sqlite:///voyager_trader.db"
    DB_POOL_SIZE: int = 10

    # Health check settings
    DB_HEALTH_CHECK_ENABLED: bool = True
    DB_HEALTH_CHECK_TIMEOUT: float = 1.0
    DB_HEALTH_CHECK_INTERVAL: int = 30
    DB_MAX_CONNECTION_AGE: int = 3600
    DB_MAX_CONNECTION_USAGE: int = 1000

    def get_health_config(self) -> HealthCheckConfig:
        return HealthCheckConfig(
            enabled=self.DB_HEALTH_CHECK_ENABLED,
            timeout=self.DB_HEALTH_CHECK_TIMEOUT,
            interval=self.DB_HEALTH_CHECK_INTERVAL,
            max_age=self.DB_MAX_CONNECTION_AGE,
            max_usage=self.DB_MAX_CONNECTION_USAGE,
        )
```

## Health Check Strategies

### 1. Quick Health Check (Default)

Uses `SELECT 1` for fast validation:

```python
config = HealthCheckConfig(query="SELECT 1", timeout=1.0)
```

**Pros:** Very fast, minimal overhead
**Cons:** May not detect all connection issues

### 2. Table Validation Check

Validates table access:

```python
config = HealthCheckConfig(
    query="SELECT COUNT(*) FROM sqlite_master LIMIT 1",
    timeout=2.0
)
```

**Pros:** Validates table access
**Cons:** Slightly more overhead

### 3. Write Validation Check

Tests write capabilities:

```python
config = HealthCheckConfig(
    query="CREATE TEMP TABLE health_check (id INTEGER); DROP TABLE health_check;",
    timeout=5.0
)
```

**Pros:** Validates full database functionality
**Cons:** Higher overhead, potential for interference

### 4. Integrity Check

Thorough database validation:

```python
config = HealthCheckConfig(
    query="PRAGMA integrity_check(1)",
    timeout=10.0
)
```

**Pros:** Comprehensive validation
**Cons:** Expensive, should be used sparingly

## Connection Lifecycle Management

### Connection States

The system tracks connections through various states:

```python
class ConnectionStatus(Enum):
    HEALTHY = "healthy"      # Connection is working properly
    UNHEALTHY = "unhealthy"  # Connection failed health checks
    STALE = "stale"          # Connection idle too long
    EXPIRED = "expired"      # Connection exceeded age/usage limits
```

### Automatic Connection Management

```python
# Connections are automatically managed based on:

# Age limits
if connection.age > config.max_age:
    # Connection is marked as EXPIRED and replaced

# Usage limits  
if connection.usage_count > config.max_usage:
    # Connection is marked as EXPIRED and replaced

# Idle time limits
if connection.idle_time > config.max_idle_time:
    # Connection is marked as STALE and may be replaced

# Health check failures
if consecutive_failures > config.max_consecutive_failures:
    # Connection is marked as UNHEALTHY and replaced
```

## Monitoring and Metrics

### Connection Statistics

Get comprehensive connection pool statistics:

```python
async def monitor_database_health():
    db_manager = await get_database()
    stats = await db_manager.get_connection_stats()

    # Basic pool metrics
    print(f"Pool size: {stats['pool_size']}/{stats['max_pool_size']}")
    print(f"Available connections: {stats['available_connections']}")

    # Health metrics
    print(f"Healthy connections: {stats['healthy_connections']}")
    print(f"Unhealthy connections: {stats['unhealthy_connections']}")
    print(f"Success rate: {stats['health_check_success_rate']:.2%}")
    print(f"Average age: {stats['average_age']:.1f}s")
    print(f"Total usage: {stats['total_usage']}")
```

### Individual Connection Statistics

```python
# Get detailed stats for specific connection
connection_stats = healthy_connection.get_stats()

print(f"Connection ID: {connection_stats['connection_id']}")
print(f"Status: {connection_stats['status']}")
print(f"Age: {connection_stats['age_seconds']:.1f}s")
print(f"Usage count: {connection_stats['usage_count']}")
print(f"Health checks: {connection_stats['health_check_count']}")
print(f"Success rate: {connection_stats['health_check_success_rate']:.2%}")
print(f"Avg query time: {connection_stats['average_query_time']:.3f}s")
```

### Logging and Alerts

The system provides detailed logging for monitoring:

```python
# Configure logging for health checking
import logging

# Enable debug logging for detailed health check information
logging.getLogger("voyager_trader.persistence.connection_health").setLevel(logging.DEBUG)

# Enable warning logging for connection issues
logging.getLogger("voyager_trader.persistence.database").setLevel(logging.WARNING)
```

## Performance Considerations

### Health Check Overhead

- **Cached Results:** Health check results are cached to avoid overhead
- **Batch Processing:** Periodic checks process connections in batches
- **Lightweight Queries:** Default `SELECT 1` query is very fast
- **Configurable Intervals:** Adjust check frequency based on needs

### Memory Usage

Connection health tracking adds minimal memory overhead:

```python
# Per connection overhead (approximate):
# - ConnectionMetrics: ~200 bytes
# - HealthyConnection wrapper: ~100 bytes
# - Total per connection: ~300 bytes

# For 100 connections: ~30KB additional memory
```

### CPU Usage

Health checking CPU usage is minimal:

```python
# Default configuration impact:
# - Health check query: <1ms per connection
# - Periodic monitoring: ~10ms per batch (5 connections)
# - For 100 connections with 30s interval: <1% CPU usage
```

## Best Practices

### 1. Configure Appropriate Timeouts

```python
# For local databases
config = HealthCheckConfig(timeout=0.5)

# For remote databases
config = HealthCheckConfig(timeout=5.0)

# For high-latency connections
config = HealthCheckConfig(timeout=10.0)
```

### 2. Set Reasonable Connection Limits

```python
# For high-traffic applications
config = HealthCheckConfig(
    max_age=1800,      # 30 minutes
    max_usage=2000,    # Higher usage limit
    max_idle_time=600, # 10 minutes
)

# For low-traffic applications
config = HealthCheckConfig(
    max_age=3600,     # 1 hour
    max_usage=500,    # Lower usage limit
    max_idle_time=300, # 5 minutes
)
```

### 3. Adjust Monitoring Frequency

```python
# For critical applications
config = HealthCheckConfig(interval=15)  # Check every 15 seconds

# For normal applications
config = HealthCheckConfig(interval=30)  # Check every 30 seconds

# For low-priority applications
config = HealthCheckConfig(interval=120) # Check every 2 minutes
```

### 4. Handle Health Check Failures

```python
async def robust_database_operation():
    max_retries = 3
    retry_delay = 1.0

    for attempt in range(max_retries):
        try:
            async with db_manager.get_connection() as conn:
                return await conn.execute("SELECT * FROM users")
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"Database operation failed (attempt {attempt + 1}): {e}")
                await asyncio.sleep(retry_delay * (2 ** attempt))
            else:
                logger.error(f"Database operation failed after {max_retries} attempts: {e}")
                raise
```

## Troubleshooting

### Common Issues

#### High Health Check Failure Rate

```python
# Check configuration
stats = await db_manager.get_connection_stats()
if stats['health_check_success_rate'] < 0.9:
    # Increase timeout
    config.timeout = 5.0

    # Check database load
    # Check network connectivity
    # Review health check query complexity
```

#### Excessive Connection Creation/Destruction

```python
# Monitor connection churn
if stats['average_age'] < config.max_age / 10:
    # Connections are being replaced too frequently
    # Investigate health check issues
    # Review connection limits
    # Check for database connectivity problems
```

#### Memory Leaks

```python
# Ensure proper cleanup
async def cleanup_database():
    db_manager = await get_database()
    await db_manager.close()  # This stops monitoring and closes connections
```

### Debugging Health Checks

```python
# Enable detailed logging
logging.getLogger("voyager_trader.persistence.connection_health").setLevel(logging.DEBUG)

# Test individual connection health
async def debug_connection_health():
    async with db_manager.get_connection() as conn:
        # Get the healthy connection wrapper
        for healthy_conn in db_manager._pool:
            if healthy_conn.connection == conn:
                # Force health check
                result = await healthy_conn.validate_health(force=True)
                print(f"Health check result: {result}")
                print(f"Connection stats: {healthy_conn.get_stats()}")
                break
```

### Performance Monitoring

```python
# Monitor health check performance
async def monitor_health_performance():
    start_time = time.time()

    # Perform health checks
    tasks = []
    for healthy_conn in db_manager._pool:
        tasks.append(healthy_conn.validate_health(force=True))

    results = await asyncio.gather(*tasks)

    duration = time.time() - start_time
    success_count = sum(results)

    print(f"Health checks completed in {duration:.3f}s")
    print(f"Success rate: {success_count}/{len(results)}")
```

## Migration from Non-Health-Checked Connections

### Backward Compatibility

The health checking system is fully backward compatible:

```python
# Old code continues to work unchanged
async with db_manager.get_connection() as conn:
    result = await conn.execute("SELECT * FROM users")

# New health check features are automatically enabled
```

### Gradual Migration

```python
# Phase 1: Enable health checking with permissive settings
config = HealthCheckConfig(
    enabled=True,
    timeout=10.0,      # Generous timeout
    max_age=7200,      # 2 hours
    max_usage=10000,   # High usage limit
)

# Phase 2: Tighten settings based on observed metrics
config = HealthCheckConfig(
    timeout=2.0,       # Stricter timeout
    max_age=3600,      # 1 hour
    max_usage=1000,    # Normal usage limit
)

# Phase 3: Enable advanced features
config = HealthCheckConfig(
    interval=30,       # Periodic monitoring
    query="PRAGMA quick_check(1)",  # More thorough checks
)
```

## Related Documentation

- [Database Management Guide](./database-management.md)
- [SQLite Error Handling Guide](./sqlite-error-handling.md)
- [Performance Monitoring](./performance-monitoring.md)
- [Persistent Storage Architecture](./persistent-storage.md)

## API Reference

### HealthCheckConfig

Configuration class for connection health checking parameters.

### HealthyConnection

Wrapper class that provides health checking capabilities for database connections.

#### Methods

- `validate_health(force=False) -> bool`: Validate connection health
- `record_usage(query_time=None) -> None`: Record connection usage
- `get_stats() -> Dict`: Get connection statistics
- `close() -> None`: Close the underlying connection

#### Properties

- `status: ConnectionStatus`: Current connection status
- `metrics: ConnectionMetrics`: Connection usage metrics
- `is_healthy: bool`: Whether connection is healthy
- `is_expired: bool`: Whether connection has exceeded limits
- `is_stale: bool`: Whether connection has been idle too long

### ConnectionHealthManager

Manager for connection pool health monitoring.

#### Methods

- `start_monitoring(pool) -> None`: Start periodic health monitoring
- `stop_monitoring() -> None`: Stop health monitoring
- `get_pool_statistics(pool) -> Dict`: Get pool health statistics

### DatabaseManager (Enhanced)

Enhanced database manager with integrated health checking.

#### New Parameters

- `health_check_config: HealthCheckConfig`: Health check configuration

#### Enhanced Methods

- `get_connection_stats() -> Dict`: Now includes health metrics
- `initialize() -> None`: Starts health monitoring if enabled
- `close() -> None`: Stops health monitoring

## Examples

See the `examples/` directory for complete implementation examples:

- `examples/basic_health_checking.py`: Basic usage patterns
- `examples/advanced_monitoring.py`: Advanced monitoring setup
- `examples/custom_health_checks.py`: Custom health check implementations
- `examples/performance_optimization.py`: Performance tuning examples
