"""Tests for database manager."""

import pytest
import pytest_asyncio
import tempfile
from pathlib import Path

from src.voyager_trader.persistence.database import DatabaseManager


@pytest_asyncio.fixture
async def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    
    db_manager = DatabaseManager(
        database_url=f"sqlite:///{db_path}",
        pool_size=2,
        max_overflow=2,
        echo=False
    )
    
    await db_manager.initialize()
    
    yield db_manager
    
    await db_manager.close()
    Path(db_path).unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_database_initialization(temp_db):
    """Test database initialization."""
    db = temp_db
    
    assert db._initialized
    assert len(db._pool) >= 0
    
    # Test table creation
    assert await db.table_exists("accounts")
    assert await db.table_exists("portfolios")
    assert await db.table_exists("orders")
    assert await db.table_exists("trades")
    assert await db.table_exists("positions")
    assert await db.table_exists("audit_logs")


@pytest.mark.asyncio
async def test_database_connection_pool(temp_db):
    """Test database connection pool."""
    db = temp_db
    
    # Test getting connection
    async with db.get_connection() as conn:
        assert conn is not None
        
        # Test basic query
        cursor = await conn.execute("SELECT 1")
        result = await cursor.fetchone()
        assert result[0] == 1


@pytest.mark.asyncio
async def test_database_transactions(temp_db):
    """Test database transactions."""
    db = temp_db
    
    # Test successful transaction
    async with db.transaction() as conn:
        await conn.execute(
            "INSERT INTO accounts (id, name, account_type) VALUES (?, ?, ?)",
            ("test-account", "Test Account", "cash")
        )
    
    # Verify data was committed
    result = await db.execute(
        "SELECT name FROM accounts WHERE id = ?",
        ("test-account",),
        fetch="one"
    )
    assert result[0] == "Test Account"
    
    # Test transaction rollback
    try:
        async with db.transaction() as conn:
            await conn.execute(
                "INSERT INTO accounts (id, name, account_type) VALUES (?, ?, ?)",
                ("test-account-2", "Test Account 2", "cash")
            )
            # Simulate error
            raise Exception("Test error")
    except Exception:
        pass
    
    # Verify data was rolled back
    result = await db.execute(
        "SELECT name FROM accounts WHERE id = ?",
        ("test-account-2",),
        fetch="one"
    )
    assert result is None


@pytest.mark.asyncio
async def test_database_table_info(temp_db):
    """Test getting table information."""
    db = temp_db
    
    account_info = await db.get_table_info("accounts")
    
    assert len(account_info) > 0
    
    # Check for key columns
    column_names = [col["name"] for col in account_info]
    assert "id" in column_names
    assert "name" in column_names
    assert "account_type" in column_names
    assert "created_at" in column_names


@pytest.mark.asyncio
async def test_database_json_serialization(temp_db):
    """Test JSON field serialization/deserialization."""
    db = temp_db
    
    # Test serialization
    test_data = {"key1": "value1", "key2": 42, "key3": [1, 2, 3]}
    serialized = db.serialize_json_field(test_data)
    assert isinstance(serialized, str)
    
    # Test deserialization
    deserialized = db.deserialize_json_field(serialized)
    assert deserialized == test_data
    
    # Test empty data
    assert db.serialize_json_field(None) == "{}"
    assert db.deserialize_json_field(None) == {}
    assert db.deserialize_json_field("") == {}


@pytest.mark.asyncio
async def test_database_connection_stats(temp_db):
    """Test connection pool statistics."""
    db = temp_db
    
    stats = await db.get_connection_stats()
    
    assert isinstance(stats, dict)
    assert "pool_size" in stats
    assert "max_pool_size" in stats
    assert "initialized" in stats
    assert stats["initialized"] is True


@pytest.mark.asyncio
async def test_database_execute_many(temp_db):
    """Test batch execution."""
    db = temp_db
    
    # Insert multiple accounts
    accounts_data = [
        ("account-1", "Account 1", "cash"),
        ("account-2", "Account 2", "margin"),
        ("account-3", "Account 3", "cash"),
    ]
    
    await db.execute_many(
        "INSERT INTO accounts (id, name, account_type) VALUES (?, ?, ?)",
        accounts_data
    )
    
    # Verify all accounts were inserted
    results = await db.execute("SELECT COUNT(*) FROM accounts", fetch="one")
    assert results[0] == 3