"""
Database connection and management for VOYAGER Trader.

This module provides database connection pooling, schema management,
and migration capabilities for the trading system persistence layer.
"""

import asyncio
import json
import logging
import sqlite3
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

import aiosqlite

from .circuit_breaker import CircuitBreakerConfig, get_circuit_breaker_manager
from .connection_health import (
    ConnectionHealthManager,
    HealthCheckConfig,
    HealthyConnection,
)
from .error_handling import SQLiteErrorHandler

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Database connection manager with connection pooling and schema management.

    Provides async/await interface for database operations with automatic
    connection pooling, transaction management, and schema migration.
    """

    def __init__(
        self,
        database_url: str = "sqlite:///voyager_trader.db",
        pool_size: int = 10,
        max_overflow: int = 20,
        echo: bool = False,
        health_check_config: Optional[HealthCheckConfig] = None,
        circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
    ):
        """
        Initialize database manager.

        Args:
            database_url: Database connection URL
            pool_size: Size of connection pool
            max_overflow: Maximum pool overflow
            echo: Whether to echo SQL statements
            health_check_config: Optional health check configuration
            circuit_breaker_config: Optional circuit breaker configuration
        """
        self.database_url = database_url
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.echo = echo

        # Extract database path from URL
        if database_url.startswith("sqlite:///"):
            self.db_path = database_url[10:]  # Remove 'sqlite:///'
        else:
            raise ValueError(f"Unsupported database URL: {database_url}")

        # Health check configuration
        self.health_config = health_check_config or HealthCheckConfig()

        # Circuit breaker configuration and setup
        self.circuit_breaker_config = circuit_breaker_config or CircuitBreakerConfig()
        cb_manager = get_circuit_breaker_manager()
        self._circuit_breaker = cb_manager.get_or_create_circuit_breaker(
            name=f"database_{self.db_path.replace('/', '_').replace('.', '_')}",
            config=self.circuit_breaker_config,
        )

        # Connection pool with health checking
        self._pool: List[HealthyConnection] = []
        self._pool_semaphore = asyncio.Semaphore(pool_size + max_overflow)
        self._initialized = False

        # Health monitoring
        self._health_manager = ConnectionHealthManager(self.health_config)
        self._next_connection_id = 0

        logger.info(f"Initialized DatabaseManager with path: {self.db_path}")

    async def initialize(self) -> None:
        """Initialize database and create tables if needed."""
        if self._initialized:
            return

        # Ensure database directory exists
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)

        # Create initial connection pool with health checking
        for _ in range(self.pool_size):
            healthy_conn = await self._create_healthy_connection()
            self._pool.append(healthy_conn)

        # Create tables if they don't exist
        await self._create_tables()

        # Start health monitoring if enabled
        if self.health_config.enabled:
            await self._health_manager.start_monitoring(self._pool)

        self._initialized = True
        logger.info("Database initialized successfully")

    async def close(self) -> None:
        """Close all database connections."""
        # Stop health monitoring
        await self._health_manager.stop_monitoring()

        # Close all healthy connections
        for healthy_conn in self._pool:
            await healthy_conn.close()
        self._pool.clear()
        self._initialized = False
        logger.info("Database connections closed")

    @asynccontextmanager
    async def get_connection(self) -> AsyncGenerator[aiosqlite.Connection, None]:
        """
        Get a healthy database connection from the pool with circuit breaker
        protection.

        Yields:
            Validated database connection from the pool

        Raises:
            CircuitBreakerException: If circuit breaker is open
        """

        async def _get_connection_operation():
            """Internal operation to get connection."""
            async with self._pool_semaphore:
                # Try to get a healthy connection from pool
                healthy_conn = await self._get_healthy_connection_from_pool()

                # Create new connection if none available or all unhealthy
                if healthy_conn is None:
                    healthy_conn = await self._create_healthy_connection()

                return healthy_conn

        # Execute connection retrieval through circuit breaker
        healthy_conn = await self._circuit_breaker.call(_get_connection_operation)
        start_time = time.time()

        try:
            # Record connection usage start
            conn = healthy_conn.connection
            yield conn

            # Record successful usage (helps health checking and circuit breaker)
            query_time = time.time() - start_time
            await healthy_conn.record_usage(query_time)

        except Exception:
            # Let circuit breaker know about the failure
            # Note: The circuit breaker will handle this automatically
            raise
        finally:
            # Return connection to pool with health validation
            await self._return_connection_to_pool(healthy_conn)

    @asynccontextmanager
    async def transaction(self) -> AsyncGenerator[aiosqlite.Connection, None]:
        """
        Execute operations within a database transaction.

        Yields:
            Database connection within a transaction context
        """
        async with self.get_connection() as conn:
            await conn.execute("BEGIN")
            try:
                yield conn
                await conn.commit()
            except Exception:
                await conn.rollback()
                raise

    async def execute(
        self, query: str, parameters: tuple = (), fetch: str = "none"
    ) -> Any:
        """
        Execute a database query.

        Args:
            query: SQL query to execute
            parameters: Query parameters
            fetch: Type of fetch ("none", "one", "all")

        Returns:
            Query result based on fetch type
        """
        try:
            async with self.get_connection() as conn:
                cursor = await conn.execute(query, parameters)

                if fetch == "one":
                    result = await cursor.fetchone()
                elif fetch == "all":
                    result = await cursor.fetchall()
                else:
                    result = cursor.lastrowid

                await conn.commit()
                return result
        except sqlite3.Error as e:
            SQLiteErrorHandler.log_database_error(
                e, f"executing query: {query[:100]}..."
            )
            raise

    async def execute_many(self, query: str, parameters_list: List[tuple]) -> None:
        """
        Execute a query multiple times with different parameters.

        Args:
            query: SQL query to execute
            parameters_list: List of parameter tuples
        """
        try:
            async with self.transaction() as conn:
                await conn.executemany(query, parameters_list)
        except sqlite3.Error as e:
            SQLiteErrorHandler.log_database_error(
                e, f"executing batch query: {query[:100]}..."
            )
            raise

    async def _create_tables(self) -> None:
        """Create database tables from schema file."""
        schema_path = Path(__file__).parent / "simple_schema.sql"

        if not schema_path.exists():
            raise FileNotFoundError(f"Schema file not found: {schema_path}")

        # Read schema file
        schema_sql = schema_path.read_text(encoding="utf-8")

        # Execute schema statements
        async with self.transaction() as conn:
            # Split schema into individual statements
            statements = []
            current_statement = []

            for line in schema_sql.split("\n"):
                line = line.strip()
                if not line or line.startswith("--"):
                    continue

                current_statement.append(line)

                if line.endswith(";"):
                    statements.append(" ".join(current_statement))
                    current_statement = []

            # Execute each statement
            for statement in statements:
                if statement.strip():
                    try:
                        await conn.execute(statement)
                    except sqlite3.Error as e:
                        # Handle error using proper error codes
                        (
                            should_continue,
                            log_message,
                        ) = SQLiteErrorHandler.handle_database_error(
                            e, "table creation", ignore_table_exists=True
                        )

                        if should_continue:
                            logger.debug(log_message)
                        else:
                            logger.error(log_message)
                            raise

        logger.info("Database tables created successfully")

    async def get_table_info(self, table_name: str) -> List[Dict[str, Any]]:
        """
        Get information about a database table.

        Args:
            table_name: Name of the table

        Returns:
            List of column information dictionaries
        """
        async with self.get_connection() as conn:
            cursor = await conn.execute(f"PRAGMA table_info({table_name})")
            columns = await cursor.fetchall()

            return [
                {
                    "name": col[1],
                    "type": col[2],
                    "not_null": bool(col[3]),
                    "default": col[4],
                    "primary_key": bool(col[5]),
                }
                for col in columns
            ]

    async def table_exists(self, table_name: str) -> bool:
        """
        Check if a table exists in the database.

        Args:
            table_name: Name of the table

        Returns:
            True if table exists, False otherwise
        """
        query = """
            SELECT name FROM sqlite_master
            WHERE type='table' AND name=?
        """
        result = await self.execute(query, (table_name,), fetch="one")
        return result is not None

    async def get_connection_stats(self) -> Dict[str, Any]:
        """
        Get connection pool statistics including health metrics.

        Returns:
            Dictionary with comprehensive pool statistics
        """
        # Get basic pool stats
        basic_stats = {
            "pool_size": len(self._pool),
            "max_pool_size": self.pool_size,
            "max_overflow": self.max_overflow,
            "available_connections": self._pool_semaphore._value,
            "initialized": self._initialized,
        }

        # Get health statistics
        health_stats = self._health_manager.get_pool_statistics(self._pool)

        # Get circuit breaker statistics
        circuit_breaker_stats = {
            "circuit_breaker": self._circuit_breaker.get_statistics()
        }

        # Combine all stats
        return {**basic_stats, **health_stats, **circuit_breaker_stats}

    async def _create_healthy_connection(self) -> HealthyConnection:
        """
        Create a new healthy connection with proper configuration.

        Returns:
            New HealthyConnection instance
        """
        # Create underlying connection
        conn = await aiosqlite.connect(self.db_path)

        # Configure connection
        await conn.execute("PRAGMA foreign_keys = ON")
        await conn.execute("PRAGMA journal_mode = WAL")
        await conn.execute("PRAGMA synchronous = NORMAL")

        # Generate unique connection ID
        self._next_connection_id += 1
        connection_id = f"conn_{self._next_connection_id}"

        # Wrap in healthy connection
        healthy_conn = HealthyConnection(
            connection=conn,
            config=self.health_config,
            connection_id=connection_id,
        )

        logger.debug(f"Created new healthy connection: {connection_id}")
        return healthy_conn

    async def _get_healthy_connection_from_pool(self) -> Optional[HealthyConnection]:
        """
        Get a healthy connection from the pool.

        Returns:
            Healthy connection if available, None otherwise
        """
        # Try to find healthy connections in pool
        healthy_connections = []
        unhealthy_connections = []

        while self._pool:
            healthy_conn = self._pool.pop()

            # Check if connection is healthy
            if await healthy_conn.validate_health():
                healthy_connections.append(healthy_conn)
            else:
                unhealthy_connections.append(healthy_conn)
                continue

            # Return the first healthy connection found
            if healthy_connections:
                conn_to_use = healthy_connections.pop(0)

                # Return other healthy connections to pool
                self._pool.extend(healthy_connections)

                # Close unhealthy connections
                for unhealthy_conn in unhealthy_connections:
                    await unhealthy_conn.close()

                return conn_to_use

        # Close any remaining unhealthy connections
        for unhealthy_conn in unhealthy_connections:
            await unhealthy_conn.close()

        return None

    async def _return_connection_to_pool(self, healthy_conn: HealthyConnection) -> None:
        """
        Return a connection to the pool after health validation.

        Args:
            healthy_conn: The healthy connection to return
        """
        # Validate health before returning to pool
        if await healthy_conn.validate_health():
            # Check if connection should be retired
            if healthy_conn.is_expired or healthy_conn.is_stale:
                await healthy_conn.close()
                logger.debug(f"Retired connection: {healthy_conn.connection_id}")
            else:
                # Return to pool if there's space
                if len(self._pool) < self.pool_size:
                    self._pool.append(healthy_conn)
                else:
                    # Pool is full, close connection
                    await healthy_conn.close()
        else:
            # Connection is unhealthy, close it
            await healthy_conn.close()
            logger.warning(f"Closed unhealthy connection: {healthy_conn.connection_id}")

    def serialize_json_field(self, data: Any) -> str:
        """
        Serialize data to JSON string for database storage.

        Args:
            data: Data to serialize

        Returns:
            JSON string representation
        """
        if data is None:
            return "{}"
        if isinstance(data, str):
            return data
        return json.dumps(data, default=str, ensure_ascii=False)

    def deserialize_json_field(self, json_str: Optional[str]) -> Any:
        """
        Deserialize JSON string from database.

        Args:
            json_str: JSON string to deserialize

        Returns:
            Deserialized data
        """
        if not json_str:
            return {}
        try:
            return json.loads(json_str)
        except (json.JSONDecodeError, TypeError):
            logger.warning(f"Failed to deserialize JSON: {json_str}")
            return {}


# Global database manager instance
db_manager: Optional[DatabaseManager] = None


async def get_database() -> DatabaseManager:
    """
    Get the global database manager instance.

    Returns:
        Initialized database manager
    """
    global db_manager
    if db_manager is None:
        # Import settings here to avoid circular imports
        try:
            from ..config import get_settings

            settings = get_settings()
            db_manager = DatabaseManager(
                database_url=getattr(
                    settings, "DATABASE_URL", "sqlite:///voyager_trader.db"
                ),
                pool_size=getattr(settings, "DB_POOL_SIZE", 10),
                max_overflow=getattr(settings, "DB_MAX_OVERFLOW", 20),
                echo=getattr(settings, "DB_ECHO", False),
            )
        except ImportError:
            # Fallback to default configuration
            db_manager = DatabaseManager()

        await db_manager.initialize()

    return db_manager


async def close_database() -> None:
    """Close the global database manager."""
    global db_manager
    if db_manager:
        await db_manager.close()
        db_manager = None
