# SQLite Error Handling Guide

This document describes the robust error handling system implemented for SQLite operations in VOYAGER-Trader.

## Overview

The system replaces fragile string-based error checking with proper SQLite error codes, providing reliable, locale-independent, and version-stable error handling.

## Problem Solved

**Before (Problematic)**:
```python
except sqlite3.Error as e:
    # Fragile string matching
    if "already exists" not in str(e):
        raise
```

**After (Robust)**:
```python
from voyager_trader.persistence.error_handling import SQLiteErrorHandler

except sqlite3.Error as e:
    should_continue, log_message = SQLiteErrorHandler.handle_database_error(
        e, "table creation", ignore_table_exists=True
    )

    if should_continue:
        logger.debug(log_message)
    else:
        logger.error(log_message)
        raise
```

## Key Components

### SQLiteErrorCode Enum

Defines all SQLite error codes with their numeric values:

```python
from voyager_trader.persistence.error_handling import SQLiteErrorCode

# Core error codes
SQLiteErrorCode.SQLITE_OK = 0           # Success
SQLiteErrorCode.SQLITE_ERROR = 1        # Generic error
SQLiteErrorCode.SQLITE_BUSY = 5         # Database locked
SQLiteErrorCode.SQLITE_CONSTRAINT = 19  # Constraint violation
SQLiteErrorCode.SQLITE_READONLY = 8     # Read-only database
SQLiteErrorCode.SQLITE_FULL = 13        # Database full

# Extended error codes
SQLiteErrorCode.SQLITE_CONSTRAINT_UNIQUE = 2067  # UNIQUE constraint failed
```

### SQLiteErrorHandler Class

Provides static methods for error analysis and handling:

#### Error Detection Methods

```python
from voyager_trader.persistence.error_handling import SQLiteErrorHandler

# Check specific error types
if SQLiteErrorHandler.is_table_exists_error(error):
    # Handle table already exists
    pass

if SQLiteErrorHandler.is_constraint_violation(error):
    # Handle constraint violations (including UNIQUE, FOREIGN KEY, etc.)
    pass

if SQLiteErrorHandler.is_database_busy(error):
    # Handle database locked/busy
    pass

if SQLiteErrorHandler.is_database_full(error):
    # Handle disk full errors
    pass

if SQLiteErrorHandler.is_readonly_error(error):
    # Handle read-only database
    pass
```

#### Unified Error Handling

```python
# Handle any database error with context
should_continue, log_message = SQLiteErrorHandler.handle_database_error(
    error,
    operation="user operation description",
    ignore_table_exists=False  # Whether to ignore "table exists" errors
)

if should_continue:
    logger.debug(log_message)
    # Continue operation
else:
    logger.error(log_message)
    raise  # Re-raise the error
```

#### Error Logging

```python
# Log database error with proper context
SQLiteErrorHandler.log_database_error(error, "operation description")
```

## Usage Examples

### Database Table Creation

```python
from voyager_trader.persistence.error_handling import SQLiteErrorHandler

async def create_tables(conn):
    statements = ["CREATE TABLE users (id INTEGER PRIMARY KEY)", ...]

    for statement in statements:
        try:
            await conn.execute(statement)
        except sqlite3.Error as e:
            should_continue, log_message = SQLiteErrorHandler.handle_database_error(
                e, "table creation", ignore_table_exists=True
            )

            if should_continue:
                logger.debug(log_message)
            else:
                logger.error(log_message)
                raise
```

### Repository Operations

```python
from voyager_trader.persistence.error_handling import SQLiteErrorHandler

async def save_user(self, user_data):
    try:
        await self.db.execute(
            "INSERT INTO users (name, email) VALUES (?, ?)",
            (user_data['name'], user_data['email'])
        )
    except sqlite3.Error as e:
        if SQLiteErrorHandler.is_constraint_violation(e):
            # Handle duplicate email, etc.
            raise UserAlreadyExistsError(f"User already exists: {user_data['email']}")
        else:
            # Log and re-raise unexpected errors
            SQLiteErrorHandler.log_database_error(e, "saving user")
            raise
```

### Connection Pool Error Handling

```python
from voyager_trader.persistence.error_handling import SQLiteErrorHandler

async def execute_with_retry(self, query, params, max_retries=3):
    for attempt in range(max_retries):
        try:
            return await self.db.execute(query, params)
        except sqlite3.Error as e:
            if SQLiteErrorHandler.is_database_busy(e):
                if attempt < max_retries - 1:
                    await asyncio.sleep(0.1 * (2 ** attempt))  # Exponential backoff
                    continue

            # Log error and re-raise
            SQLiteErrorHandler.log_database_error(e, f"executing query (attempt {attempt + 1})")
            raise
```

## Error Code Reference

### Core SQLite Errors

| Code | Constant | Description | Recommended Action |
|------|----------|-------------|-------------------|
| 0 | SQLITE_OK | Success | Continue |
| 1 | SQLITE_ERROR | Generic error | Check specific case |
| 5 | SQLITE_BUSY | Database locked | Retry with backoff |
| 8 | SQLITE_READONLY | Read-only database | Check permissions |
| 13 | SQLITE_FULL | Database/disk full | Free space or cleanup |
| 19 | SQLITE_CONSTRAINT | Constraint violation | Handle specific constraint |

### Extended Error Codes

| Code | Constant | Description | Recommended Action |
|------|----------|-------------|-------------------|
| 2067 | SQLITE_CONSTRAINT_UNIQUE | UNIQUE constraint failed | Handle duplicate data |

## Benefits

### Reliability Improvements

- **Locale Independence**: Error codes work regardless of system language
- **Version Stability**: Error codes remain consistent across SQLite versions
- **Precise Handling**: Target specific error conditions accurately
- **Better Debugging**: Clear error classification and logging

### Code Quality

- **Maintainable**: Easier to understand and modify error handling logic
- **Testable**: Can mock specific error codes for unit testing
- **Documented**: Error codes are well-documented in SQLite documentation
- **Performance**: No string operations in error handling paths

## Testing

The error handling system includes comprehensive tests:

```bash
# Run error handling tests
python -m pytest tests/persistence/test_error_handling.py -v

# Test coverage
python -m pytest tests/persistence/test_error_handling.py --cov=src.voyager_trader.persistence.error_handling
```

### Test Categories

1. **Unit Tests**: Test individual error detection methods
2. **Integration Tests**: Test with real SQLite operations
3. **Error Code Tests**: Verify error code constants and comparisons
4. **Logging Tests**: Ensure proper error logging functionality

## Migration from String-Based Checking

### Before
```python
except sqlite3.Error as e:
    if "already exists" not in str(e):
        raise
```

### After
```python
from voyager_trader.persistence.error_handling import SQLiteErrorHandler

except sqlite3.Error as e:
    should_continue, log_message = SQLiteErrorHandler.handle_database_error(
        e, "operation description", ignore_table_exists=True
    )

    if should_continue:
        logger.debug(log_message)
    else:
        logger.error(log_message)
        raise
```

## Best Practices

1. **Always Use Error Codes**: Never rely on error message text for logic
2. **Provide Context**: Include operation description in error handling
3. **Log Appropriately**: Use debug for expected errors, error for unexpected
4. **Handle Extended Codes**: Check for both primary and extended error codes
5. **Test Error Scenarios**: Write tests for each error condition

## Related Documentation

- [Database Management Guide](./database-management.md)
- [Persistent Storage Architecture](./persistent-storage.md)
- [SQLite Documentation](https://www.sqlite.org/rescode.html)
