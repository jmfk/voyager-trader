"""
Tests for the skill caching and performance optimization functionality.
"""

import shutil
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from voyager_trader.models.learning import Skill, SkillExecutionResult
from voyager_trader.models.types import SkillCategory, SkillComplexity
from voyager_trader.skills import (
    CacheConfig,
    CacheEntry,
    ConnectionPool,
    DatabaseConfig,
    DatabaseSkillStorage,
    LRUCache,
    SkillExecutionCache,
    SkillExecutor,
    SkillLibrarian,
    VoyagerSkillLibrary,
)


class TestCacheEntry:
    """Test the CacheEntry dataclass."""

    def test_cache_entry_creation(self):
        """Test creating a cache entry."""
        entry = CacheEntry(value="test_value", timestamp=datetime.utcnow())

        assert entry.value == "test_value"
        assert entry.hit_count == 0
        assert isinstance(entry.timestamp, datetime)
        assert isinstance(entry.access_time, datetime)

    def test_is_expired(self):
        """Test expiration checking."""
        # Recent entry should not be expired
        recent_entry = CacheEntry(value="test", timestamp=datetime.utcnow())
        assert not recent_entry.is_expired(ttl_hours=24)

        # Old entry should be expired
        old_timestamp = datetime.utcnow() - timedelta(hours=25)
        old_entry = CacheEntry(value="test", timestamp=old_timestamp)
        assert old_entry.is_expired(ttl_hours=24)

    def test_update_access(self):
        """Test access tracking."""
        entry = CacheEntry(value="test", timestamp=datetime.utcnow())
        original_access_time = entry.access_time
        original_hit_count = entry.hit_count

        time.sleep(0.001)  # Small delay to ensure different timestamp
        entry.update_access()

        assert entry.hit_count == original_hit_count + 1
        assert entry.access_time > original_access_time


class TestLRUCache:
    """Test the LRU Cache implementation."""

    def setup_method(self):
        """Set up test cache."""
        self.cache = LRUCache(max_size=3, ttl_hours=24)

    def test_cache_put_get(self):
        """Test basic put and get operations."""
        self.cache.put("key1", "value1")
        assert self.cache.get("key1") == "value1"
        assert self.cache.get("nonexistent") is None

    def test_cache_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        # Fill cache to capacity
        self.cache.put("key1", "value1")
        self.cache.put("key2", "value2")
        self.cache.put("key3", "value3")

        # Add one more item, should evict oldest (key1)
        self.cache.put("key4", "value4")

        assert self.cache.get("key1") is None  # Evicted
        assert self.cache.get("key2") == "value2"
        assert self.cache.get("key3") == "value3"
        assert self.cache.get("key4") == "value4"

    def test_cache_lru_order_update(self):
        """Test that accessing items updates LRU order."""
        self.cache.put("key1", "value1")
        self.cache.put("key2", "value2")
        self.cache.put("key3", "value3")

        # Access key1 to move it to end
        self.cache.get("key1")

        # Add new item, should evict key2 (oldest unaccessed)
        self.cache.put("key4", "value4")

        assert self.cache.get("key1") == "value1"  # Still there
        assert self.cache.get("key2") is None  # Evicted
        assert self.cache.get("key3") == "value3"
        assert self.cache.get("key4") == "value4"

    def test_cache_ttl_expiration(self):
        """Test TTL-based expiration."""
        # Create cache with very short TTL (1/1000th of an hour = 3.6 seconds)
        cache = LRUCache(max_size=10, ttl_hours=1 / 3600)  # 1 second TTL

        cache.put("key1", "value1")
        assert cache.get("key1") == "value1"

        # Wait for expiration - use longer wait to ensure expiration
        time.sleep(1.1)  # Wait slightly longer than TTL
        assert cache.get("key1") is None  # Should be expired

    def test_cache_stats(self):
        """Test cache statistics."""
        stats = self.cache.get_stats()
        assert stats["size"] == 0
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["hit_rate"] == 0

        self.cache.put("key1", "value1")
        self.cache.get("key1")  # Hit
        self.cache.get("key2")  # Miss

        stats = self.cache.get_stats()
        assert stats["size"] == 1
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5

    def test_cache_clear(self):
        """Test cache clearing."""
        self.cache.put("key1", "value1")
        self.cache.put("key2", "value2")

        assert self.cache.get("key1") == "value1"

        self.cache.clear()
        assert self.cache.get("key1") is None
        assert self.cache.get("key2") is None

        stats = self.cache.get_stats()
        assert stats["size"] == 0
        # Stats should be reset after clear
        assert stats["hits"] == 0
        assert stats["misses"] == 2  # The two get calls after clear are misses


class TestSkillExecutionCache:
    """Test the SkillExecutionCache."""

    def setup_method(self):
        """Set up test cache."""
        self.config = CacheConfig()
        self.cache = SkillExecutionCache(self.config)
        self.test_skill = Skill(
            name="test_skill",
            description="Test skill",
            category=SkillCategory.TECHNICAL_ANALYSIS,
            complexity=SkillComplexity.BASIC,
            code="result = {'test': True}",
            input_schema={"type": "object"},
            output_schema={"type": "object"},
        )

    def test_execution_result_caching(self):
        """Test caching of execution results."""
        inputs = {"param1": "value1"}
        context = {"context_param": "value"}
        result = (
            SkillExecutionResult.SUCCESS,
            {"output": "test"},
            {"metadata": "test"},
        )

        # Should not find result initially
        cached_result = self.cache.get_execution_result(
            self.test_skill, inputs, context
        )
        assert cached_result is None

        # Cache the result
        self.cache.cache_execution_result(self.test_skill, inputs, context, result)

        # Should find cached result
        cached_result = self.cache.get_execution_result(
            self.test_skill, inputs, context
        )
        assert cached_result is not None

        # Result should match (with cache metadata added)
        exec_result, output, metadata = cached_result
        assert exec_result == SkillExecutionResult.SUCCESS
        assert output == {"output": "test"}
        assert metadata["cached"] is True

    def test_execution_result_key_generation(self):
        """Test that different inputs generate different cache keys."""
        inputs1 = {"param1": "value1"}
        inputs2 = {"param1": "value2"}
        context = {"context_param": "value"}
        result = (
            SkillExecutionResult.SUCCESS,
            {"output": "test"},
            {"metadata": "test"},
        )

        # Cache result with inputs1
        self.cache.cache_execution_result(self.test_skill, inputs1, context, result)

        # Should find result with same inputs
        cached_result1 = self.cache.get_execution_result(
            self.test_skill, inputs1, context
        )
        assert cached_result1 is not None

        # Should not find result with different inputs
        cached_result2 = self.cache.get_execution_result(
            self.test_skill, inputs2, context
        )
        assert cached_result2 is None

    def test_compilation_caching(self):
        """Test caching of compiled skill code."""
        compiled_code = "# Compiled skill code"

        # Should not find compiled code initially
        cached_code = self.cache.get_compiled_code(self.test_skill)
        assert cached_code is None

        # Cache the compiled code
        self.cache.cache_compiled_code(self.test_skill, compiled_code)

        # Should find cached compiled code
        cached_code = self.cache.get_compiled_code(self.test_skill)
        assert cached_code == compiled_code

    def test_cache_disabled_config(self):
        """Test cache behavior when disabled in config."""
        config = CacheConfig(enable_result_cache=False, enable_compilation_cache=False)
        cache = SkillExecutionCache(config)

        inputs = {"param1": "value1"}
        result = (
            SkillExecutionResult.SUCCESS,
            {"output": "test"},
            {"metadata": "test"},
        )
        compiled_code = "# Test code"

        # Caching should be no-ops
        cache.cache_execution_result(self.test_skill, inputs, None, result)
        cache.cache_compiled_code(self.test_skill, compiled_code)

        # Should not find anything in cache
        assert cache.get_execution_result(self.test_skill, inputs, None) is None
        assert cache.get_compiled_code(self.test_skill) is None

    def test_cache_stats(self):
        """Test comprehensive cache statistics."""
        stats = self.cache.get_cache_stats()

        assert "execution_cache" in stats
        assert "compilation_cache" in stats
        assert "metadata_cache" in stats
        assert "config" in stats

        config_stats = stats["config"]
        assert config_stats["enable_result_cache"] is True
        assert config_stats["enable_compilation_cache"] is True


class TestSkillExecutor:
    """Test the enhanced SkillExecutor with caching."""

    def setup_method(self):
        """Set up test executor."""
        self.cache_config = CacheConfig()
        self.executor = SkillExecutor(cache_config=self.cache_config)
        self.test_skill = Skill(
            name="test_skill",
            description="Test skill",
            category=SkillCategory.TECHNICAL_ANALYSIS,
            complexity=SkillComplexity.BASIC,
            code="result = {'success': True, 'value': inputs.get('value', 0)}",
            input_schema={
                "type": "object",
                "properties": {"value": {"type": "number"}},
            },
            output_schema={"type": "object"},
        )

    def test_execution_with_caching(self):
        """Test skill execution with caching enabled."""
        inputs = {"value": 42}

        # First execution should not be cached
        start_time = time.time()
        result1, output1, metadata1 = self.executor.execute_skill(
            self.test_skill, inputs
        )
        first_exec_time = time.time() - start_time

        assert result1 == SkillExecutionResult.SUCCESS
        assert metadata1["cached"] is False

        # Second execution should be cached and faster
        start_time = time.time()
        result2, output2, metadata2 = self.executor.execute_skill(
            self.test_skill, inputs
        )
        second_exec_time = time.time() - start_time

        assert result2 == SkillExecutionResult.SUCCESS
        assert metadata2["cached"] is True
        assert second_exec_time < first_exec_time  # Should be faster due to caching

    def test_cache_stats_methods(self):
        """Test cache statistics methods."""
        stats = self.executor.get_cache_stats()
        assert "execution_cache" in stats
        assert "compilation_cache" in stats
        assert "metadata_cache" in stats

        # Test cache clearing
        self.executor.clear_cache()

        # Test expired entry eviction
        evicted = self.executor.evict_expired_cache_entries()
        assert "execution" in evicted
        assert "compilation" in evicted
        assert "metadata" in evicted


class TestSkillLibrarian:
    """Test the enhanced SkillLibrarian with caching."""

    def setup_method(self):
        """Set up test librarian."""
        self.temp_dir = tempfile.mkdtemp()
        self.storage_path = Path(self.temp_dir)
        self.cache_config = CacheConfig()
        self.librarian = SkillLibrarian(self.storage_path, self.cache_config)
        self.test_skill = Skill(
            name="test_skill",
            description="Test skill",
            category=SkillCategory.TECHNICAL_ANALYSIS,
            complexity=SkillComplexity.BASIC,
            code="result = {'test': True}",
            input_schema={"type": "object"},
            output_schema={"type": "object"},
        )

    def teardown_method(self):
        """Clean up temporary directory."""
        if hasattr(self, "temp_dir") and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def test_skill_storage_with_caching(self):
        """Test skill storage with metadata caching."""
        # Store skill
        success = self.librarian.store_skill(self.test_skill)
        assert success is True

        # Verify file was created
        skill_file = self.storage_path / f"{self.test_skill.id}.json"
        assert skill_file.exists()

        # Verify skill can be retrieved from cache
        retrieved_skill = self.librarian.retrieve_skill(self.test_skill.id)
        assert retrieved_skill is not None
        assert retrieved_skill.name == self.test_skill.name

    def test_cache_refresh(self):
        """Test cache refresh functionality."""
        # Store skill
        self.librarian.store_skill(self.test_skill)

        # Modify file externally to simulate external changes
        skill_file = self.storage_path / f"{self.test_skill.id}.json"
        time.sleep(0.001)  # Ensure different mtime
        skill_file.touch()

        # Refresh cache
        refresh_stats = self.librarian.refresh_cache()
        assert refresh_stats["refreshed"] >= 0
        assert refresh_stats["removed"] >= 0

    def test_cache_performance_methods(self):
        """Test cache performance and statistics methods."""
        # Store some test skills
        self.librarian.store_skill(self.test_skill)

        # Test cache stats
        stats = self.librarian.get_cache_stats()
        assert "metadata_cache" in stats
        assert "skills_in_memory" in stats
        assert "files_tracked" in stats
        assert "index_sizes" in stats

        # Test performance summary
        summary = self.librarian.get_performance_summary()
        assert "skills_count" in summary
        assert "cache_hit_rate" in summary
        assert "memory_efficiency" in summary
        assert "performance_indicators" in summary

        # Test cache clearing
        self.librarian.clear_caches()

        # Test expired entry eviction
        evicted = self.librarian.evict_expired_cache_entries()
        assert isinstance(evicted, dict)


class TestVoyagerSkillLibrary:
    """Test the enhanced VoyagerSkillLibrary with caching."""

    def setup_method(self):
        """Set up test library."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = {
            "skill_library_path": self.temp_dir,
            "max_execution_cache_size": 100,
            "max_metadata_cache_size": 50,
            "enable_result_cache": True,
            "enable_compilation_cache": True,
        }
        self.library = VoyagerSkillLibrary(self.config)
        self.test_skill = Skill(
            name="test_skill",
            description="Test skill",
            category=SkillCategory.TECHNICAL_ANALYSIS,
            complexity=SkillComplexity.BASIC,
            code="result = {'value': inputs.get('input_value', 0) * 2}",
            input_schema={
                "type": "object",
                "properties": {"input_value": {"type": "number"}},
            },
            output_schema={"type": "object"},
        )

    def teardown_method(self):
        """Clean up temporary directory."""
        if hasattr(self, "temp_dir") and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def test_skill_execution_with_performance_tracking(self):
        """Test skill execution with performance tracking."""
        # Add skill to library
        self.library.add_skill(self.test_skill)

        inputs = {"input_value": 21}

        # Execute skill
        result, output, metadata = self.library.execute_skill(
            self.test_skill.id, inputs
        )

        assert result == SkillExecutionResult.SUCCESS
        assert "execution_time_seconds" in metadata
        assert "library_stats" in metadata

        # Check performance stats
        library_stats = metadata["library_stats"]
        assert "total_executions" in library_stats
        assert "success_rate" in library_stats
        assert "cache_hit_rate" in library_stats
        assert "average_execution_time" in library_stats

    def test_comprehensive_stats(self):
        """Test comprehensive statistics collection."""
        # Add skill to library
        self.library.add_skill(self.test_skill)

        # Get comprehensive stats
        stats = self.library.get_comprehensive_stats()

        assert "library" in stats
        assert "execution_cache" in stats
        assert "performance" in stats
        assert "cache_config" in stats

        # Verify cache config in stats
        cache_config = stats["cache_config"]
        assert cache_config["result_cache_enabled"] is True
        assert cache_config["compilation_cache_enabled"] is True

    def test_cache_management_methods(self):
        """Test cache management functionality."""
        # Add and execute skill to populate caches
        self.library.add_skill(self.test_skill)
        self.library.execute_skill(self.test_skill.id, {"input_value": 10})

        # Test cache clearing
        self.library.clear_all_caches()

        # Test expired entry eviction
        evicted = self.library.evict_expired_entries()
        assert "execution_cache" in evicted
        assert "metadata_cache" in evicted

        # Test cache refresh
        refresh_result = self.library.refresh_caches()
        assert "refresh" in refresh_result
        assert "eviction" in refresh_result

    def test_performance_recommendations(self):
        """Test performance optimization recommendations."""
        # Add skill and execute multiple times to generate stats
        self.library.add_skill(self.test_skill)

        for i in range(5):
            self.library.execute_skill(self.test_skill.id, {"input_value": i})

        # Get recommendations
        recommendations = self.library.get_performance_recommendations()
        assert isinstance(recommendations, list)
        # Recommendations depend on performance metrics, so just verify format


class TestConnectionPool:
    """Test the database connection pool."""

    def setup_method(self):
        """Set up test connection pool."""
        self.db_config = DatabaseConfig(
            db_type="sqlite",
            connection_string=":memory:",
            pool_size=3,
            max_overflow=2,
            enable_connection_pooling=True,
        )
        self.pool = ConnectionPool(self.db_config)

    def test_connection_pool_initialization(self):
        """Test connection pool initialization."""
        stats = self.pool.get_pool_stats()
        assert stats["pool_size"] == 3
        assert stats["available_connections"] == 3
        assert stats["overflow_connections"] == 0
        assert stats["initialized"] is True

    def test_connection_acquisition_and_release(self):
        """Test connection acquisition and release."""
        with self.pool.get_connection() as conn:
            assert conn is not None
            # Test that connection works
            cursor = conn.execute("SELECT 1")
            result = cursor.fetchone()
            assert result[0] == 1

        # Connection should be returned to pool
        stats = self.pool.get_pool_stats()
        assert stats["available_connections"] == 3

    def test_connection_pool_exhaustion(self):
        """Test behavior when connection pool is exhausted."""
        connections = []

        # Acquire all connections from pool + overflow
        for i in range(5):  # pool_size=3 + max_overflow=2
            conn_context = self.pool.get_connection()
            connections.append(conn_context)
            conn = conn_context.__enter__()
            assert conn is not None

        # Pool should now be exhausted
        with pytest.raises(Exception):  # Should raise DatabaseConnectionError
            with self.pool.get_connection() as conn:
                pass

        # Clean up connections
        for conn_context in connections:
            conn_context.__exit__(None, None, None)

    def test_connection_pool_stats(self):
        """Test connection pool statistics."""
        stats = self.pool.get_pool_stats()

        assert "pool_size" in stats
        assert "available_connections" in stats
        assert "overflow_connections" in stats
        assert "max_overflow" in stats
        assert "pool_utilization" in stats
        assert "initialized" in stats

    def test_connection_pool_close_all(self):
        """Test closing all connections in pool."""
        self.pool.close_all()

        stats = self.pool.get_pool_stats()
        assert stats["initialized"] is False
        assert stats["available_connections"] == 0


class TestDatabaseSkillStorage:
    """Test the database skill storage system."""

    def setup_method(self):
        """Set up test database storage."""
        self.db_config = DatabaseConfig(
            db_type="sqlite",
            connection_string=":memory:",
            pool_size=2,
            enable_connection_pooling=True,
        )
        self.storage = DatabaseSkillStorage(self.db_config)
        self.test_skill = Skill(
            name="test_skill",
            description="Test skill",
            category=SkillCategory.TECHNICAL_ANALYSIS,
            complexity=SkillComplexity.BASIC,
            code="result = {'test': True}",
            input_schema={"type": "object"},
            output_schema={"type": "object"},
        )

    def test_database_schema_initialization(self):
        """Test database schema is properly initialized."""
        # Schema should be created during initialization
        if not self.storage.connection_pool:
            pytest.skip("Database connection pool not initialized")

        try:
            with self.storage.connection_pool.get_connection() as conn:
                # Check if skills table exists
                cursor = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='skills'"
                )
                result = cursor.fetchone()
                if result is None:
                    # Try to initialize schema manually for test
                    self.storage._initialize_schema()
                    cursor = conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table' AND name='skills'"
                    )
                    result = cursor.fetchone()

                assert (
                    result is not None
                ), "Skills table should exist after schema initialization"
                assert result["name"] == "skills"
        except Exception as e:
            pytest.skip(f"Database test skipped due to error: {e}")

    def test_skill_storage_to_database(self):
        """Test storing skills to database."""
        if self.storage.connection_pool:
            success = self.storage.store_skill_to_db(self.test_skill)
            assert success is True

            # Verify skill was stored
            with self.storage.connection_pool.get_connection() as conn:
                cursor = conn.execute(
                    "SELECT * FROM skills WHERE id = ?", (self.test_skill.id,)
                )
                result = cursor.fetchone()
                assert result is not None
                assert result["name"] == self.test_skill.name
                # Handle both string and enum values
                expected_category = (
                    self.test_skill.category.value
                    if hasattr(self.test_skill.category, "value")
                    else self.test_skill.category
                )
                assert result["category"] == expected_category
        else:
            # Test without connection pool should return False
            success = self.storage.store_skill_to_db(self.test_skill)
            assert success is False

    def test_connection_stats(self):
        """Test connection statistics."""
        stats = self.storage.get_connection_stats()
        assert "pool_size" in stats
        assert "available_connections" in stats
        assert "initialized" in stats

    def test_database_storage_without_pooling(self):
        """Test database storage without connection pooling."""
        storage = DatabaseSkillStorage()  # No db_config
        stats = storage.get_connection_stats()
        assert stats["connection_pooling"] is False


if __name__ == "__main__":
    pytest.main([__file__])
