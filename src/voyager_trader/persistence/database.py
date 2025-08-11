"""
Database connection and management for VOYAGER Trader.

This module provides database connection pooling, schema management,
and migration capabilities for the trading system persistence layer.
"""

import asyncio
import json
import logging
import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

import aiosqlite

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
    ):
        """
        Initialize database manager.

        Args:
            database_url: Database connection URL
            pool_size: Size of connection pool
            max_overflow: Maximum pool overflow
            echo: Whether to echo SQL statements
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

        # Connection pool
        self._pool: List[aiosqlite.Connection] = []
        self._pool_semaphore = asyncio.Semaphore(pool_size + max_overflow)
        self._initialized = False

        logger.info(f"Initialized DatabaseManager with path: {self.db_path}")

    async def initialize(self) -> None:
        """Initialize database and create tables if needed."""
        if self._initialized:
            return

        # Ensure database directory exists
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)

        # Create initial connection pool
        for _ in range(self.pool_size):
            conn = await aiosqlite.connect(self.db_path)
            await conn.execute("PRAGMA foreign_keys = ON")
            await conn.execute("PRAGMA journal_mode = WAL")
            await conn.execute("PRAGMA synchronous = NORMAL")
            self._pool.append(conn)

        # Create tables if they don't exist
        await self._create_tables()

        self._initialized = True
        logger.info("Database initialized successfully")

    async def close(self) -> None:
        """Close all database connections."""
        for conn in self._pool:
            await conn.close()
        self._pool.clear()
        self._initialized = False
        logger.info("Database connections closed")

    @asynccontextmanager
    async def get_connection(self) -> AsyncGenerator[aiosqlite.Connection, None]:
        """
        Get a database connection from the pool.

        Yields:
            Database connection from the pool
        """
        async with self._pool_semaphore:
            if self._pool:
                conn = self._pool.pop()
            else:
                # Create new connection if pool is empty
                conn = await aiosqlite.connect(self.db_path)
                await conn.execute("PRAGMA foreign_keys = ON")
                await conn.execute("PRAGMA journal_mode = WAL")
                await conn.execute("PRAGMA synchronous = NORMAL")

            try:
                yield conn
            finally:
                # Return connection to pool if there's space
                if len(self._pool) < self.pool_size:
                    self._pool.append(conn)
                else:
                    await conn.close()

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

    async def execute_many(self, query: str, parameters_list: List[tuple]) -> None:
        """
        Execute a query multiple times with different parameters.

        Args:
            query: SQL query to execute
            parameters_list: List of parameter tuples
        """
        async with self.transaction() as conn:
            await conn.executemany(query, parameters_list)

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
                        # Ignore "table already exists" errors
                        if "already exists" not in str(e):
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
        Get connection pool statistics.

        Returns:
            Dictionary with pool statistics
        """
        return {
            "pool_size": len(self._pool),
            "max_pool_size": self.pool_size,
            "max_overflow": self.max_overflow,
            "available_connections": self._pool_semaphore._value,
            "initialized": self._initialized,
        }

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
