# Persistent Storage for VOYAGER Trader

This document describes the persistent storage system implemented for VOYAGER Trader, providing comprehensive data persistence and audit logging capabilities for all trading operations.

## Overview

The persistent storage system provides:

- **Complete data persistence** for accounts, portfolios, orders, trades, and positions
- **Comprehensive audit logging** for all system actions and changes
- **High-performance database layer** with connection pooling and optimization
- **Configurable storage backends** (SQLite, PostgreSQL, MySQL support)
- **Automatic backup and recovery** capabilities
- **Thread-safe operations** with transaction support

## Architecture

### Core Components

1. **Database Manager** (`persistence/database.py`)
   - Connection pooling and management
   - Schema creation and migration
   - Transaction handling
   - Performance optimization

2. **Repository Layer** (`persistence/repositories.py`)
   - Entity-specific data access patterns
   - CRUD operations for all trading entities
   - Query optimization and batching
   - Data serialization/deserialization

3. **Trading Service** (`persistence/trading_service.py`)
   - High-level service layer for trading operations
   - Automatic persistence integration
   - Business logic coordination
   - Performance metrics and reporting

4. **Audit Service** (`persistence/audit_service.py`)
   - Comprehensive audit trail logging
   - System event tracking
   - User action monitoring
   - Risk management event logging

## Database Schema

### Core Tables

#### Accounts
Stores trading account information including balances, risk parameters, and configuration.

```sql
CREATE TABLE accounts (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    account_type TEXT NOT NULL,
    base_currency TEXT NOT NULL,
    cash_balance_amount DECIMAL(20, 8),
    total_equity_amount DECIMAL(20, 8),
    buying_power_amount DECIMAL(20, 8),
    -- ... additional fields
);
```

#### Portfolios
Manages portfolio-level data including positions mapping and performance metrics.

#### Orders
Tracks all trading orders with complete lifecycle management.

#### Trades
Records all executed trades with detailed execution information.

#### Positions
Maintains position data with entry/exit tracking and P&L calculations.

#### Audit Logs
Comprehensive audit trail for all system activities.

### Key Features

- **Foreign key constraints** ensure data integrity
- **Automatic timestamps** track creation and modification
- **Version tracking** for optimistic locking
- **JSON fields** for flexible metadata storage
- **Indexes** optimized for common query patterns
- **Views** for complex reporting queries

## Configuration

### Environment Variables

Key configuration options available via environment variables:

```bash
# Database connection
DATABASE_URL=sqlite:///voyager_trader.db
DB_POOL_SIZE=10
DB_MAX_OVERFLOW=20

# Backup settings  
DB_BACKUP_ENABLED=true
DB_BACKUP_DIRECTORY=./backups
DB_BACKUP_INTERVAL_HOURS=24

# Audit settings
AUDIT_ENABLED=true
AUDIT_RETENTION_DAYS=365
AUDIT_BATCH_SIZE=100

# Performance settings
CACHE_ENABLED=true
CACHE_TTL_SECONDS=300
QUERY_BATCH_SIZE=1000
```

### Configuration Management

The system uses a comprehensive configuration system (`config.py`) with:

- **Pydantic validation** for type safety and validation
- **Environment variable integration** for easy deployment
- **Default values** for all settings
- **Validation rules** for configuration integrity
- **Hot reloading** support for configuration changes

## Usage Examples

### Basic Usage

```python
from voyager_trader.persistence import get_trading_service

# Get the trading service
service = await get_trading_service()

# Create a new account
account = Account(
    name="My Trading Account",
    account_type="margin",
    base_currency=Currency.USD,
    cash_balance=Money(amount=Decimal("50000"), currency=Currency.USD),
    total_equity=Money(amount=Decimal("50000"), currency=Currency.USD),
    buying_power=Money(amount=Decimal("100000"), currency=Currency.USD),
    maintenance_margin=Money(amount=Decimal("0"), currency=Currency.USD),
)

# Save with audit logging
saved_account = await service.create_account(
    account, 
    user_id="user-123"
)

# Create a portfolio
portfolio = Portfolio(
    name="Growth Portfolio",
    account_id=saved_account.id,
    base_currency=Currency.USD,
    cash_balance=Money(amount=Decimal("25000"), currency=Currency.USD),
    total_value=Money(amount=Decimal("50000"), currency=Currency.USD),
)

saved_portfolio = await service.create_portfolio(
    portfolio,
    user_id="user-123"
)
```

### Order Management

```python
# Create an order
order = Order(
    symbol=Symbol(code="AAPL", asset_class="equity"),
    order_type=OrderType.LIMIT,
    side=OrderSide.BUY,
    quantity=Quantity(amount=Decimal("100")),
    price=Decimal("150.00"),
)

# Save with strategy tracking
saved_order = await service.create_order(
    order,
    strategy_id="momentum-v1",
    user_id="trader-123"
)

# Execute partial fill
executed_order = saved_order.execute_partial(
    quantity=Quantity(amount=Decimal("50")),
    price=Decimal("151.00")
)

# Update with audit trail
updated_order = await service.update_order(
    executed_order,
    strategy_id="momentum-v1"
)
```

### Trade Recording

```python
# Record a trade execution
trade = Trade(
    symbol=Symbol(code="AAPL", asset_class="equity"),
    side=OrderSide.BUY,
    quantity=Quantity(amount=Decimal("50")),
    price=Decimal("151.00"),
    timestamp=datetime.now(timezone.utc),
    order_id=saved_order.id,
    commission=Money(amount=Decimal("1.00"), currency=Currency.USD),
)

saved_trade = await service.create_trade(
    trade,
    strategy_id="momentum-v1"
)
```

### Position Tracking

```python
# Create a position
position = Position(
    symbol=Symbol(code="AAPL", asset_class="equity"),
    position_type=PositionType.LONG,
    quantity=Quantity(amount=Decimal("50")),
    entry_price=Decimal("151.00"),
    entry_timestamp=datetime.now(timezone.utc),
    portfolio_id=saved_portfolio.id,
)

saved_position = await service.create_position(
    position,
    strategy_id="momentum-v1"
)

# Update position price
updated_position = saved_position.update_price(Decimal("155.00"))
await service.update_position(updated_position)

# Close position
closed_position = updated_position.close_position(
    exit_price=Decimal("160.00"),
    exit_timestamp=datetime.now(timezone.utc)
)
await service.update_position(closed_position)
```

## Audit Logging

### Automatic Audit Trail

The system automatically logs:

- **Entity operations**: Create, update, delete for all entities
- **Trading events**: Order submissions, fills, cancellations
- **Position events**: Opening, closing, modifications
- **System events**: Startup, shutdown, errors
- **User actions**: Logins, logouts, configuration changes
- **Risk events**: Limit breaches, circuit breaker triggers

### Custom Audit Logging

```python
from voyager_trader.persistence.audit_service import get_audit_service

audit = await get_audit_service()

# Log custom events
await audit.log_order_submitted(
    order_id="order-123",
    symbol="AAPL",
    side="buy",
    quantity=100.0,
    order_type="limit",
    price=150.00,
    strategy_id="momentum-v1"
)

await audit.log_risk_limit_breached(
    limit_type="max_drawdown",
    limit_value=5.0,
    current_value=7.2,
    entity_type="portfolio",
    entity_id="portfolio-123",
    action_taken="suspend_trading"
)
```

### Audit Queries

```python
# Get entity history
history = await audit.get_entity_history("account", "account-123")

# Get user activity
user_logs = await audit.get_user_activity(
    "user-123", 
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 12, 31)
)

# Get trading activity
trading_logs = await audit.get_audit_trail(
    action="trade_executed",
    start_date=datetime.now() - timedelta(days=7),
    limit=100
)
```

## Performance Optimization

### Connection Pooling

```python
# Optimized for high-throughput operations
db_manager = DatabaseManager(
    database_url="sqlite:///voyager_trader.db",
    pool_size=10,        # Base pool size
    max_overflow=20,     # Additional connections under load
    echo=False          # Disable SQL logging in production
)
```

### Batch Operations

```python
# Efficient batch processing
await db_manager.execute_many(
    "INSERT INTO trades (id, symbol, side, quantity, price) VALUES (?, ?, ?, ?, ?)",
    trades_data
)
```

### Query Optimization

- **Indexes** on frequently queried columns
- **Views** for complex reporting queries  
- **Batch processing** for bulk operations
- **Connection pooling** for concurrent access
- **Query caching** for repeated operations

## Backup and Recovery

### Automatic Backups

```bash
# Configure automatic backups
DB_BACKUP_ENABLED=true
DB_BACKUP_DIRECTORY=./backups
DB_BACKUP_INTERVAL_HOURS=24
DB_BACKUP_RETENTION_DAYS=30
```

### Manual Backup

```python
from voyager_trader.persistence.database import get_database

db = await get_database()

# Create manual backup
backup_path = await db.create_backup("manual_backup_20241201.db")
print(f"Backup created: {backup_path}")
```

### Recovery Procedures

1. **Stop the trading system**
2. **Restore from backup**:
   ```bash
   cp backups/voyager_trader_20241201.db voyager_trader.db
   ```
3. **Verify data integrity**
4. **Restart the system**

## Migration and Schema Evolution

The system supports schema migrations for database evolution:

```python
# Future: Migration support
from voyager_trader.persistence.migrations import run_migrations

await run_migrations(db_manager)
```

## Monitoring and Maintenance

### Database Statistics

```python
# Get connection pool statistics
stats = await db_manager.get_connection_stats()
print(f"Pool size: {stats['pool_size']}")
print(f"Available connections: {stats['available_connections']}")

# Get table information
table_info = await db_manager.get_table_info("trades")
```

### Performance Monitoring

```python
# Enable query logging for performance analysis
ENABLE_QUERY_LOGGING=true

# Monitor slow queries
DB_QUERY_TIMEOUT=60.0
```

## Security Considerations

### Data Protection

- **Foreign key constraints** prevent orphaned records
- **Input validation** prevents SQL injection
- **Connection encryption** for network databases
- **Audit logging** for security monitoring
- **Access control** through repository pattern

### Sensitive Data

- **Password hashing** with bcrypt
- **JWT secrets** from environment variables
- **Database credentials** not stored in code
- **Audit log retention** for compliance

## Testing

### Test Coverage

The persistence layer includes comprehensive tests:

- **Unit tests** for all repository operations
- **Integration tests** for service layer
- **Audit logging tests** for compliance verification
- **Performance tests** for load validation
- **Migration tests** for schema evolution

### Running Tests

```bash
# Run persistence tests
python -m pytest tests/persistence/ -v

# Run with coverage
python -m pytest tests/persistence/ --cov=src.voyager_trader.persistence

# Performance tests
python -m pytest tests/persistence/test_performance.py -v
```

## Troubleshooting

### Common Issues

1. **Database locked error (SQLite)**
   - Enable WAL mode: `DB_ENABLE_WAL_MODE=true`
   - Increase timeout: `DB_CONNECT_TIMEOUT=60.0`

2. **Connection pool exhaustion**
   - Increase pool size: `DB_POOL_SIZE=20`
   - Add overflow: `DB_MAX_OVERFLOW=30`

3. **Slow query performance**
   - Enable query logging: `ENABLE_QUERY_LOGGING=true`
   - Increase cache: `DB_CACHE_SIZE_MB=128`

4. **Audit log growth**
   - Set retention: `AUDIT_RETENTION_DAYS=90`
   - Enable batch processing: `AUDIT_BATCH_SIZE=1000`

### Diagnostic Commands

```python
# Check database connectivity
db = await get_database()
assert await db.table_exists("accounts")

# Verify audit logging
audit = await get_audit_service()
logs = await audit.get_audit_trail(limit=10)
assert len(logs) > 0

# Test repository operations
service = await get_trading_service()
accounts = await service.get_active_accounts()
```

## Production Deployment

### Database Setup

For production environments:

1. **Use PostgreSQL or MySQL** for better performance
2. **Enable connection pooling** with appropriate sizes
3. **Configure automated backups** with encryption
4. **Set up monitoring** and alerting
5. **Enable audit logging** with appropriate retention

### Environment Configuration

```bash
# Production settings
VOYAGER_ENVIRONMENT=production
DATABASE_URL=postgresql://voyager:password@db.company.com:5432/voyager_prod
DB_POOL_SIZE=50
DB_MAX_OVERFLOW=100
AUDIT_RETENTION_DAYS=2555  # 7 years for compliance
```

### Security Hardening

1. **Use strong JWT secrets**
2. **Enable database encryption**
3. **Configure secure backup storage**
4. **Set up audit log monitoring**
5. **Enable rate limiting**

## Future Enhancements

Planned improvements include:

- **Read replicas** for query performance
- **Sharding support** for horizontal scaling  
- **Real-time event streaming** with Kafka
- **Data archiving** for historical data
- **Advanced analytics** and reporting
- **Multi-tenant support** for SaaS deployments

## Support

For issues or questions:

1. Check the troubleshooting section above
2. Review the test suite for examples
3. Examine the configuration options
4. Create an issue in the GitHub repository

The persistent storage system provides a robust foundation for the VOYAGER Trader system, ensuring data integrity, audit compliance, and high performance for all trading operations.