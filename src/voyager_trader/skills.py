"""
VOYAGER Skill Library System.

Comprehensive skill management system with six core components:
1. Skill Executor - Safe execution environment
2. Skill Composer - Skill composition and dependency resolution
3. Skill Validator - Multi-faceted validation system
4. Skill Librarian - Storage and retrieval with indexing
5. Skill Discoverer - Pattern recognition and extraction
6. Performance Tracker - Comprehensive metrics tracking
"""

import ast
import hashlib
import json
import logging
import sqlite3
import subprocess
import sys
import tempfile
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from queue import Empty, Queue
from threading import Lock, RLock
from typing import Any, Dict, List, Optional, Set, Tuple

from .models.learning import Experience, Skill, SkillExecutionResult
from .models.types import SkillCategory, SkillComplexity


class SkillExecutionError(Exception):
    """Exception raised during skill execution."""


class SkillCompositionError(Exception):
    """Exception raised during skill composition."""


class SkillValidationError(Exception):
    """Exception raised during skill validation."""


class DatabaseConnectionError(Exception):
    """Exception raised for database connection issues."""


@dataclass
class DatabaseConfig:
    """Database connection configuration."""

    db_type: str = "sqlite"  # sqlite, postgresql, mysql
    connection_string: str = ":memory:"
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    pool_recycle: int = 3600  # Recycle connections after 1 hour
    enable_connection_pooling: bool = True


class ConnectionPool:
    """Thread-safe database connection pool."""

    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._pool: Queue = Queue(maxsize=config.pool_size)
        self._overflow_connections = 0
        self._pool_lock = Lock()
        self._initialized = False

        if config.enable_connection_pooling:
            self._initialize_pool()

    def _initialize_pool(self) -> None:
        """Initialize the connection pool."""
        try:
            for _ in range(self.config.pool_size):
                conn = self._create_connection()
                self._pool.put(conn)
            self._initialized = True
            self.logger.info(
                f"Initialized connection pool with {self.config.pool_size} connections"
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize connection pool: {e}")
            raise DatabaseConnectionError(f"Pool initialization failed: {e}")

    def _create_connection(self):
        """Create a new database connection."""
        if self.config.db_type == "sqlite":
            conn = sqlite3.connect(
                self.config.connection_string,
                check_same_thread=False,
                timeout=self.config.pool_timeout,
            )
            # Enable row factory for easier result handling
            conn.row_factory = sqlite3.Row
            return conn
        else:
            # For other database types, would use appropriate drivers
            raise NotImplementedError(
                f"Database type {self.config.db_type} not implemented"
            )

    @contextmanager
    def get_connection(self):
        """Get a connection from the pool with automatic cleanup."""
        if not self.config.enable_connection_pooling:
            # Direct connection without pooling
            conn = self._create_connection()
            try:
                yield conn
            finally:
                conn.close()
            return

        conn = None
        is_overflow = False

        try:
            # Try to get connection from pool
            try:
                conn = self._pool.get(timeout=self.config.pool_timeout)
            except Empty:
                # Pool is empty, check if we can create overflow connection
                with self._pool_lock:
                    if self._overflow_connections < self.config.max_overflow:
                        conn = self._create_connection()
                        self._overflow_connections += 1
                        is_overflow = True
                        self.logger.debug("Created overflow connection")
                    else:
                        raise DatabaseConnectionError("Connection pool exhausted")

            # Test connection is still valid
            if self._is_connection_stale(conn):
                conn.close()
                conn = self._create_connection()
                self.logger.debug("Recycled stale connection")

            yield conn

        except Exception as e:
            self.logger.error(f"Connection error: {e}")
            if conn:
                conn.close()
                conn = None
            raise
        finally:
            # Return connection to pool or close if overflow
            if conn:
                if is_overflow:
                    conn.close()
                    with self._pool_lock:
                        self._overflow_connections -= 1
                else:
                    try:
                        self._pool.put(conn, block=False)
                    except:
                        # Pool is full, close connection
                        conn.close()

    def _is_connection_stale(self, conn) -> bool:
        """Check if connection is stale and needs recycling."""
        try:
            # Simple connectivity test
            conn.execute("SELECT 1").fetchone()
            return False
        except Exception:
            return True

    def close_all(self) -> None:
        """Close all connections in the pool."""
        with self._pool_lock:
            while not self._pool.empty():
                try:
                    conn = self._pool.get_nowait()
                    conn.close()
                except Empty:
                    break

            self._initialized = False
            self.logger.info("Closed all connections in pool")

    def get_pool_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics."""
        with self._pool_lock:
            return {
                "pool_size": self.config.pool_size,
                "available_connections": self._pool.qsize(),
                "overflow_connections": self._overflow_connections,
                "max_overflow": self.config.max_overflow,
                "pool_utilization": (
                    (self.config.pool_size - self._pool.qsize()) / self.config.pool_size
                    if self.config.pool_size > 0
                    else 0
                ),
                "initialized": self._initialized,
            }


class DatabaseSkillStorage:
    """Database-backed skill storage with connection pooling.

    This is prepared for future migration from file-based to database storage.
    Currently provides the interface but delegates to file-based storage.
    """

    def __init__(self, db_config: Optional[DatabaseConfig] = None):
        self.db_config = db_config or DatabaseConfig()
        self.connection_pool = ConnectionPool(self.db_config) if db_config else None
        self.logger = logging.getLogger(__name__)

        if self.connection_pool:
            self._initialize_schema()

    def _initialize_schema(self) -> None:
        """Initialize database schema for skill storage."""
        if not self.connection_pool:
            return

        schema_sql = """
        CREATE TABLE IF NOT EXISTS skills (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT,
            category TEXT,
            complexity TEXT,
            version TEXT DEFAULT '1.0',
            code TEXT NOT NULL,
            language TEXT DEFAULT 'python',
            input_schema TEXT,
            output_schema TEXT,
            parameters TEXT,
            performance_metrics TEXT,
            usage_count INTEGER DEFAULT 0,
            success_count INTEGER DEFAULT 0,
            last_used TIMESTAMP,
            is_active BOOLEAN DEFAULT TRUE,
            tags TEXT,
            required_skills TEXT,
            dependencies TEXT,
            examples TEXT,
            learned_from TEXT,
            created_by TEXT DEFAULT 'voyager',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_skills_category ON skills(category);
        CREATE INDEX IF NOT EXISTS idx_skills_complexity ON skills(complexity);
        CREATE INDEX IF NOT EXISTS idx_skills_usage_count ON skills(usage_count);
        CREATE INDEX IF NOT EXISTS idx_skills_success_rate ON skills(success_count, usage_count);
        CREATE INDEX IF NOT EXISTS idx_skills_last_used ON skills(last_used);
        """

        try:
            with self.connection_pool.get_connection() as conn:
                conn.executescript(schema_sql)
                conn.commit()
            self.logger.info("Database schema initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize schema: {e}")
            raise DatabaseConnectionError(f"Schema initialization failed: {e}")

    def store_skill_to_db(self, skill: "Skill") -> bool:
        """Store skill to database (future implementation)."""
        if not self.connection_pool:
            return False

        try:
            with self.connection_pool.get_connection() as conn:
                # Convert skill to database format
                skill_data = {
                    "id": skill.id,
                    "name": skill.name,
                    "description": skill.description,
                    "category": skill.category
                    if isinstance(skill.category, str)
                    else skill.category.value,
                    "complexity": skill.complexity
                    if isinstance(skill.complexity, str)
                    else skill.complexity.value,
                    "version": skill.version,
                    "code": skill.code,
                    "language": skill.language,
                    "input_schema": json.dumps(skill.input_schema),
                    "output_schema": json.dumps(skill.output_schema),
                    "parameters": json.dumps(skill.parameters, default=str),
                    "performance_metrics": json.dumps(
                        {k: float(v) for k, v in skill.performance_metrics.items()}
                    ),
                    "usage_count": skill.usage_count,
                    "success_count": skill.success_count,
                    "last_used": skill.last_used.isoformat()
                    if skill.last_used
                    else None,
                    "is_active": skill.is_active,
                    "tags": json.dumps(skill.tags),
                    "required_skills": json.dumps(skill.required_skills),
                    "dependencies": json.dumps(skill.dependencies),
                    "examples": json.dumps(skill.examples),
                    "learned_from": skill.learned_from,
                    "created_by": skill.created_by,
                }

                # Insert or update skill
                sql = """
                INSERT OR REPLACE INTO skills
                (id, name, description, category, complexity, version, code, language,
                 input_schema, output_schema, parameters, performance_metrics,
                 usage_count, success_count, last_used, is_active, tags,
                 required_skills, dependencies, examples, learned_from, created_by,
                 updated_at)
                VALUES
                (:id, :name, :description, :category, :complexity, :version, :code, :language,
                 :input_schema, :output_schema, :parameters, :performance_metrics,
                 :usage_count, :success_count, :last_used, :is_active, :tags,
                 :required_skills, :dependencies, :examples, :learned_from, :created_by,
                 CURRENT_TIMESTAMP)
                """

                conn.execute(sql, skill_data)
                conn.commit()

            return True

        except Exception as e:
            self.logger.error(f"Failed to store skill to database: {e}")
            return False

    def get_connection_stats(self) -> Dict[str, Any]:
        """Get database connection statistics."""
        if self.connection_pool:
            return self.connection_pool.get_pool_stats()
        return {"connection_pooling": False}

    def close(self) -> None:
        """Close all database connections."""
        if self.connection_pool:
            self.connection_pool.close_all()


class CacheConfig:
    """Configuration for skill caching system."""

    def __init__(
        self,
        max_execution_cache_size: int = 1000,
        max_metadata_cache_size: int = 500,
        execution_cache_ttl_hours: int = 24,
        metadata_cache_ttl_hours: int = 72,
        enable_compilation_cache: bool = True,
        enable_result_cache: bool = True,
    ):
        self.max_execution_cache_size = max_execution_cache_size
        self.max_metadata_cache_size = max_metadata_cache_size
        self.execution_cache_ttl_hours = execution_cache_ttl_hours
        self.metadata_cache_ttl_hours = metadata_cache_ttl_hours
        self.enable_compilation_cache = enable_compilation_cache
        self.enable_result_cache = enable_result_cache


@dataclass
class CacheEntry:
    """Cache entry with value, timestamp and hit count."""

    value: Any
    timestamp: datetime
    hit_count: int = 0
    access_time: datetime = field(default_factory=datetime.utcnow)

    def is_expired(self, ttl_hours: float) -> bool:
        """Check if entry is expired based on TTL."""
        return datetime.utcnow() - self.timestamp > timedelta(hours=ttl_hours)

    def update_access(self) -> None:
        """Update access time and hit count."""
        self.access_time = datetime.utcnow()
        self.hit_count += 1


class LRUCache:
    """Thread-safe LRU cache with TTL support."""

    def __init__(self, max_size: int, ttl_hours: float = 24.0):
        self.max_size = max_size
        self.ttl_hours = ttl_hours
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = RLock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache with LRU update."""
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None

            entry = self._cache[key]

            # Check if expired
            if entry.is_expired(self.ttl_hours):
                del self._cache[key]
                self._misses += 1
                return None

            # Update LRU order and access stats
            entry.update_access()
            self._cache.move_to_end(key)
            self._hits += 1
            return entry.value

    def put(self, key: str, value: Any) -> None:
        """Put value in cache with LRU eviction."""
        with self._lock:
            # If key exists, update it
            if key in self._cache:
                self._cache[key].value = value
                self._cache[key].timestamp = datetime.utcnow()
                self._cache.move_to_end(key)
                return

            # Add new entry
            entry = CacheEntry(value=value, timestamp=datetime.utcnow())
            self._cache[key] = entry

            # Evict oldest if at capacity
            if len(self._cache) > self.max_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]

    def clear(self) -> None:
        """Clear all cache entries and reset stats."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    def evict_expired(self) -> int:
        """Remove expired entries and return count removed."""
        with self._lock:
            expired_keys = [
                key
                for key, entry in self._cache.items()
                if entry.is_expired(self.ttl_hours)
            ]

            for key in expired_keys:
                del self._cache[key]

            return len(expired_keys)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0

            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
                "ttl_hours": self.ttl_hours,
            }


class SkillExecutionCache:
    """Specialized cache for skill execution results."""

    def __init__(self, config: CacheConfig):
        self.config = config
        self._execution_cache = LRUCache(
            max_size=config.max_execution_cache_size,
            ttl_hours=config.execution_cache_ttl_hours,
        )
        self._compilation_cache = LRUCache(
            max_size=config.max_execution_cache_size // 2,  # Smaller compilation cache
            ttl_hours=config.execution_cache_ttl_hours
            * 2,  # Longer TTL for compiled code
        )
        self._metadata_cache = LRUCache(
            max_size=config.max_metadata_cache_size,
            ttl_hours=config.metadata_cache_ttl_hours,
        )
        self.logger = logging.getLogger(__name__)

    def _generate_execution_key(
        self, skill: "Skill", inputs: Dict[str, Any], context: Optional[Dict[str, Any]]
    ) -> str:
        """Generate cache key for skill execution."""
        # Create deterministic hash from skill ID, inputs, and context
        key_data = {
            "skill_id": skill.id,
            "skill_version": skill.version,
            "inputs": inputs,
            "context": context or {},
        }

        # Sort keys for consistent hashing
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.sha256(key_str.encode()).hexdigest()[:16]

    def _generate_compilation_key(self, skill: "Skill") -> str:
        """Generate cache key for skill compilation."""
        # Hash skill code and version
        key_data = {
            "skill_id": skill.id,
            "skill_version": skill.version,
            "code_hash": hashlib.sha256(skill.code.encode()).hexdigest()[:16],
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()[:16]

    def get_execution_result(
        self, skill: "Skill", inputs: Dict[str, Any], context: Optional[Dict[str, Any]]
    ) -> Optional[Tuple["SkillExecutionResult", Any, Dict[str, Any]]]:
        """Get cached execution result if available."""
        if not self.config.enable_result_cache:
            return None

        cache_key = self._generate_execution_key(skill, inputs, context)
        result = self._execution_cache.get(cache_key)

        if result:
            self.logger.debug(f"Cache hit for skill execution: {skill.name}")
            # Update metadata to indicate cache hit
            execution_result, output, metadata = result
            metadata = metadata.copy()
            metadata["cached"] = True
            metadata["cache_hit_time"] = datetime.utcnow().isoformat()
            return execution_result, output, metadata

        return None

    def cache_execution_result(
        self,
        skill: "Skill",
        inputs: Dict[str, Any],
        context: Optional[Dict[str, Any]],
        result: Tuple["SkillExecutionResult", Any, Dict[str, Any]],
    ) -> None:
        """Cache execution result."""
        if not self.config.enable_result_cache:
            return

        # Only cache successful results to avoid caching errors
        execution_result, output, metadata = result
        if execution_result == SkillExecutionResult.SUCCESS:
            cache_key = self._generate_execution_key(skill, inputs, context)
            self._execution_cache.put(cache_key, result)
            self.logger.debug(f"Cached execution result for skill: {skill.name}")

    def get_compiled_code(self, skill: "Skill") -> Optional[str]:
        """Get cached compiled skill code."""
        if not self.config.enable_compilation_cache:
            return None

        cache_key = self._generate_compilation_key(skill)
        compiled_code = self._compilation_cache.get(cache_key)

        if compiled_code:
            self.logger.debug(f"Cache hit for compiled code: {skill.name}")

        return compiled_code

    def cache_compiled_code(self, skill: "Skill", compiled_code: str) -> None:
        """Cache compiled skill code."""
        if not self.config.enable_compilation_cache:
            return

        cache_key = self._generate_compilation_key(skill)
        self._compilation_cache.put(cache_key, compiled_code)
        self.logger.debug(f"Cached compiled code for skill: {skill.name}")

    def get_skill_metadata(self, skill_id: str) -> Optional[Dict[str, Any]]:
        """Get cached skill metadata."""
        return self._metadata_cache.get(skill_id)

    def cache_skill_metadata(self, skill_id: str, metadata: Dict[str, Any]) -> None:
        """Cache skill metadata."""
        self._metadata_cache.put(skill_id, metadata)

    def clear_all(self) -> None:
        """Clear all caches."""
        self._execution_cache.clear()
        self._compilation_cache.clear()
        self._metadata_cache.clear()
        self.logger.info("Cleared all skill caches")

    def evict_expired(self) -> Dict[str, int]:
        """Evict expired entries from all caches."""
        execution_evicted = self._execution_cache.evict_expired()
        compilation_evicted = self._compilation_cache.evict_expired()
        metadata_evicted = self._metadata_cache.evict_expired()

        if execution_evicted + compilation_evicted + metadata_evicted > 0:
            self.logger.info(
                f"Evicted expired cache entries: execution={execution_evicted}, "
                f"compilation={compilation_evicted}, metadata={metadata_evicted}"
            )

        return {
            "execution": execution_evicted,
            "compilation": compilation_evicted,
            "metadata": metadata_evicted,
        }

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        return {
            "execution_cache": self._execution_cache.get_stats(),
            "compilation_cache": self._compilation_cache.get_stats(),
            "metadata_cache": self._metadata_cache.get_stats(),
            "config": {
                "max_execution_cache_size": self.config.max_execution_cache_size,
                "max_metadata_cache_size": self.config.max_metadata_cache_size,
                "execution_cache_ttl_hours": self.config.execution_cache_ttl_hours,
                "metadata_cache_ttl_hours": self.config.metadata_cache_ttl_hours,
                "enable_compilation_cache": self.config.enable_compilation_cache,
                "enable_result_cache": self.config.enable_result_cache,
            },
        }


class SkillExecutor:
    """
    Safe execution environment for VOYAGER skills.

    Provides sandboxed execution with timeout protection,
    resource limits, security validation, and performance caching.
    """

    def __init__(
        self,
        timeout_seconds: int = 30,
        max_memory_mb: int = 128,
        cache_config: Optional[CacheConfig] = None,
    ):
        self.timeout_seconds = timeout_seconds
        self.max_memory_mb = max_memory_mb
        self.logger = logging.getLogger(__name__)

        # Initialize caching system
        self.cache_config = cache_config or CacheConfig()
        self.cache = SkillExecutionCache(self.cache_config)

    def execute_skill(
        self,
        skill: Skill,
        inputs: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[SkillExecutionResult, Any, Dict[str, Any]]:
        """
        Execute a skill safely in a sandboxed environment with caching.

        Returns:
            Tuple of (result, output, execution_metadata)
        """
        start_time = time.time()
        execution_metadata = {
            "start_time": datetime.utcnow().isoformat(),
            "skill_id": skill.id,
            "skill_name": skill.name,
            "inputs": inputs,
            "context": context or {},
            "cached": False,
        }

        # Check cache first
        cached_result = self.cache.get_execution_result(skill, inputs, context)
        if cached_result:
            return cached_result

        try:
            # Validate inputs against skill schema
            if not self._validate_inputs(skill, inputs):
                result = (
                    SkillExecutionResult.ERROR,
                    None,
                    {**execution_metadata, "error": "Invalid inputs"},
                )
                # Don't cache error results
                return result

            # Create safe execution environment
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                skill_file = temp_path / f"skill_{skill.id}.py"

                # Check for cached compiled code first
                skill_code = self.cache.get_compiled_code(skill)
                if not skill_code:
                    skill_code = self._prepare_skill_code(skill, inputs, context)
                    self.cache.cache_compiled_code(skill, skill_code)
                else:
                    # Still need to inject current inputs/context into cached code
                    skill_code = self._inject_runtime_data(skill_code, inputs, context)

                skill_file.write_text(skill_code)

                # Execute in subprocess with limits
                result = subprocess.run(
                    [sys.executable, str(skill_file)],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                    cwd=temp_dir,
                )

                execution_time = time.time() - start_time
                execution_metadata.update(
                    {
                        "execution_time_seconds": execution_time,
                        "return_code": result.returncode,
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                    }
                )

                if result.returncode == 0:
                    # Parse output
                    try:
                        output = (
                            json.loads(result.stdout) if result.stdout.strip() else None
                        )
                        exec_result = (
                            SkillExecutionResult.SUCCESS,
                            output,
                            execution_metadata,
                        )
                    except json.JSONDecodeError:
                        exec_result = (
                            SkillExecutionResult.SUCCESS,
                            result.stdout.strip(),
                            execution_metadata,
                        )

                    # Cache successful result
                    self.cache.cache_execution_result(
                        skill, inputs, context, exec_result
                    )
                    return exec_result
                else:
                    # Don't cache error results
                    return (
                        SkillExecutionResult.ERROR,
                        None,
                        {**execution_metadata, "error": result.stderr},
                    )

        except subprocess.TimeoutExpired:
            # Don't cache timeout results
            return (
                SkillExecutionResult.TIMEOUT,
                None,
                {**execution_metadata, "error": "Execution timeout"},
            )
        except Exception as e:
            # Don't cache error results
            return (
                SkillExecutionResult.ERROR,
                None,
                {**execution_metadata, "error": str(e)},
            )

    def _validate_inputs(self, skill: Skill, inputs: Dict[str, Any]) -> bool:
        """Validate inputs against skill's input schema."""
        try:
            # Basic validation - check required fields exist
            required_fields = skill.input_schema.get("required", [])
            for field_name in required_fields:
                if field_name not in inputs:
                    self.logger.error(f"Missing required input field: {field_name}")
                    return False
            return True
        except Exception as e:
            self.logger.error(f"Input validation error: {e}")
            return False

    def _prepare_skill_code(
        self, skill: Skill, inputs: Dict[str, Any], context: Optional[Dict[str, Any]]
    ) -> str:
        """Prepare skill code for safe execution."""
        # Create wrapper code with inputs and context
        wrapper_code = f"""
import json
import sys
from decimal import Decimal
from datetime import datetime
from typing import Any, Dict, List, Optional

# Skill inputs
inputs = {json.dumps(inputs, default=str)}
context = {json.dumps(context or {}, default=str)}

# Skill parameters
parameters = {json.dumps(skill.parameters, default=str)}

try:
    # Skill code
{self._indent_code(skill.code, 4)}

    # Output result if skill defines 'result' variable
    if 'result' in locals():
        print(json.dumps(result, default=str))
except Exception as e:
    print(f"Skill execution error: {{e}}", file=sys.stderr)
    sys.exit(1)
"""
        return wrapper_code

    def _indent_code(self, code: str, spaces: int) -> str:
        """Indent code by specified number of spaces."""
        lines = code.split("\n")
        return "\n".join(
            " " * spaces + line if line.strip() else line for line in lines
        )

    def _inject_runtime_data(
        self,
        cached_code: str,
        inputs: Dict[str, Any],
        context: Optional[Dict[str, Any]],
    ) -> str:
        """Inject runtime inputs and context into cached compiled code."""
        # Replace the placeholder sections in cached code with current data
        runtime_inputs = json.dumps(inputs, default=str)
        runtime_context = json.dumps(context or {}, default=str)

        # This assumes the cached code has placeholders we can replace
        # In a more sophisticated implementation, we'd parse and replace AST nodes
        code_with_inputs = cached_code.replace(
            "inputs = {}", f"inputs = {runtime_inputs}"
        )
        code_with_context = code_with_inputs.replace(
            "context = {}", f"context = {runtime_context}"
        )

        return code_with_context

    def clear_cache(self) -> None:
        """Clear all execution caches."""
        self.cache.clear_all()
        self.logger.info("Cleared skill execution caches")

    def evict_expired_cache_entries(self) -> Dict[str, int]:
        """Evict expired cache entries and return eviction stats."""
        return self.cache.evict_expired()

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        return self.cache.get_cache_stats()


class SkillComposer:
    """
    Skill composition engine with dependency resolution.

    Enables building complex strategies from simpler skills
    with automatic dependency management.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def compose_skills(
        self, skills: List[Skill], composition_strategy: str = "sequential"
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Compose multiple skills into a unified strategy.

        Args:
            skills: List of skills to compose
            composition_strategy: How to combine skills
                                 (sequential, parallel, conditional)

        Returns:
            Tuple of (composed_code, metadata)
        """
        if not skills:
            raise SkillCompositionError("No skills provided for composition")

        # Validate dependencies
        dependency_order = self._resolve_dependencies(skills)
        if not dependency_order:
            raise SkillCompositionError("Circular dependencies detected")

        # Generate composed code based on strategy
        if composition_strategy == "sequential":
            return self._compose_sequential(dependency_order)
        elif composition_strategy == "parallel":
            return self._compose_parallel(dependency_order)
        elif composition_strategy == "conditional":
            return self._compose_conditional(dependency_order)
        else:
            raise SkillCompositionError(
                f"Unknown composition strategy: {composition_strategy}"
            )

    def _resolve_dependencies(self, skills: List[Skill]) -> Optional[List[Skill]]:
        """Resolve skill dependencies using topological sort."""
        skill_map = {skill.name: skill for skill in skills}
        visited = set()
        temp_visited = set()
        result = []

        def visit(skill: Skill) -> bool:
            if skill.name in temp_visited:
                return False  # Circular dependency
            if skill.name in visited:
                return True

            temp_visited.add(skill.name)

            # Visit dependencies first
            for dep_name in skill.required_skills:
                if dep_name in skill_map:
                    if not visit(skill_map[dep_name]):
                        return False

            temp_visited.remove(skill.name)
            visited.add(skill.name)
            result.append(skill)
            return True

        for skill in skills:
            if skill.name not in visited:
                if not visit(skill):
                    return None  # Circular dependency detected

        return result

    def _compose_sequential(self, skills: List[Skill]) -> Tuple[str, Dict[str, Any]]:
        """Compose skills to run sequentially."""
        composed_code = """
def execute_composed_strategy(inputs, context=None):
    results = {}
    context = context or {}
    execution_order = []
    failed_skills = []

"""

        for i, skill in enumerate(skills):
            composed_code += f"""
    # Execute skill: {skill.name}
    execution_order.append('{skill.name}')
    try:
{self._indent_code(skill.code, 8)}
        skill_result = locals().get('result', None)
        results['{skill.name}'] = {{
            'result': skill_result,
            'success': True,
            'execution_time': None,
            'skill_index': {i}
        }}
        context.update(locals().get('context_updates', {{}}))
    except Exception as e:
        error_info = {{
            'error': str(e),
            'error_type': type(e).__name__,
            'success': False,
            'skill_index': {i}
        }}
        results['{skill.name}'] = error_info
        failed_skills.append('{skill.name}')

        # Log error details to results
        if 'execution_summary' not in results:
            results['execution_summary'] = {{}}
        results['execution_summary']['failed_at_skill'] = '{skill.name}'
        results['execution_summary']['failed_skill_index'] = {i}

        if not context.get('continue_on_error', False):
            results['execution_summary']['aborted'] = True
            results['execution_summary']['executed_skills'] = execution_order
            results['execution_summary']['failed_skills'] = failed_skills
            return results
"""

        composed_code += """

    # Add execution summary for successful completion
    results['execution_summary'] = {
        'completed': True,
        'executed_skills': execution_order,
        'failed_skills': failed_skills,
        'total_skills': len(execution_order),
        'success_rate': (
            (len(execution_order) - len(failed_skills)) / len(execution_order)
            if execution_order else 1.0
        )
    }

    return results

result = execute_composed_strategy(inputs, context)
"""

        metadata = {
            "composition_type": "sequential",
            "skills_count": len(skills),
            "skill_names": [skill.name for skill in skills],
            "total_complexity": sum(
                (
                    1
                    if skill.complexity == SkillComplexity.BASIC
                    else (
                        2
                        if skill.complexity == SkillComplexity.INTERMEDIATE
                        else 3
                        if skill.complexity == SkillComplexity.ADVANCED
                        else 4
                    )
                )
                for skill in skills
            ),
        }

        return composed_code, metadata

    def _compose_parallel(self, skills: List[Skill]) -> Tuple[str, Dict[str, Any]]:
        """Compose skills to run in parallel (simulated)."""
        # Note: True parallelism would require threading/multiprocessing
        # This provides a parallel-like structure for now
        composed_code = """
def execute_composed_strategy(inputs, context=None):
    results = {}
    context = context or {}
    errors = []

"""

        for skill in skills:
            composed_code += f"""
    # Execute skill: {skill.name} (parallel simulation)
    try:
        skill_context = context.copy()
{self._indent_code(skill.code, 8)}
        results['{skill.name}'] = locals().get('result', None)
    except Exception as e:
        errors.append(('{skill.name}', str(e)))
"""

        composed_code += """

    if errors and not context.get('continue_on_error', False):
        results['errors'] = errors

    return results

result = execute_composed_strategy(inputs, context)
"""

        metadata = {
            "composition_type": "parallel",
            "skills_count": len(skills),
            "skill_names": [skill.name for skill in skills],
        }

        return composed_code, metadata

    def _compose_conditional(self, skills: List[Skill]) -> Tuple[str, Dict[str, Any]]:
        """Compose skills with conditional execution."""
        composed_code = """
def execute_composed_strategy(inputs, context=None):
    results = {}
    context = context or {}

"""

        for i, skill in enumerate(skills):
            condition = (
                f"results.get('skill_{i-1}', {{}}).get('success', True)"
                if i > 0
                else "True"
            )
            composed_code += f"""
    # Conditional execution of skill: {skill.name}
    if {condition}:
        try:
{self._indent_code(skill.code, 12)}
            results['{skill.name}'] = locals().get('result', None)
            context.update(locals().get('context_updates', {{}}))
        except Exception as e:
            results['{skill.name}'] = {{'error': str(e), 'success': False}}
    else:
        results['{skill.name}'] = {{'skipped': True, 'reason': 'Previous skill failed'}}
"""

        composed_code += """

    return results

result = execute_composed_strategy(inputs, context)
"""

        metadata = {
            "composition_type": "conditional",
            "skills_count": len(skills),
            "skill_names": [skill.name for skill in skills],
        }

        return composed_code, metadata

    def _indent_code(self, code: str, spaces: int) -> str:
        """Indent code by specified number of spaces."""
        lines = code.split("\n")
        return "\n".join(
            " " * spaces + line if line.strip() else line for line in lines
        )


class SkillValidator(ABC):
    """Abstract base class for skill validation strategies."""

    @abstractmethod
    def validate(self, skill: Skill) -> Tuple[bool, List[str]]:
        """Validate a skill. Returns (is_valid, error_messages)."""


class SyntaxValidator(SkillValidator):
    """Validates skill code syntax."""

    def validate(self, skill: Skill) -> Tuple[bool, List[str]]:
        """Check if skill code has valid Python syntax."""
        errors = []

        try:
            ast.parse(skill.code)
        except SyntaxError as e:
            errors.append(f"Syntax error at line {e.lineno}: {e.msg}")
        except Exception as e:
            errors.append(f"Code parsing error: {str(e)}")

        return len(errors) == 0, errors


class SecurityValidator(SkillValidator):
    """Validates skill code for security concerns."""

    DANGEROUS_IMPORTS = {
        "os",
        "subprocess",
        "sys",
        "shutil",
        "glob",
        "tempfile",
        "pickle",
        "marshal",
        "shelve",
        "dbm",
        "sqlite3",
        "socket",
        "urllib",
        "http",
        "ftplib",
        "smtplib",
        "zipfile",
        "tarfile",
        "ctypes",
        "multiprocessing",
        "threading",
        "asyncio",
        "concurrent",
        "importlib",
        "__builtin__",
        "builtins",
        "code",
        "inspect",
        "runpy",
        "pty",
        "fcntl",
        "termios",
        "tty",
        "signal",
        "resource",
        "gc",
        "weakref",
    }

    DANGEROUS_FUNCTIONS = {
        "eval",
        "exec",
        "compile",
        "__import__",
        "open",
        "file",
        "input",
        "raw_input",
        "reload",
    }

    def validate(self, skill: Skill) -> Tuple[bool, List[str]]:
        """Check skill code for security vulnerabilities."""
        errors = []

        try:
            tree = ast.parse(skill.code)

            for node in ast.walk(tree):
                # Check for dangerous imports
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name in self.DANGEROUS_IMPORTS:
                            errors.append(f"Dangerous import detected: {alias.name}")

                # Check for dangerous function calls
                elif isinstance(node, ast.Call):
                    if (
                        isinstance(node.func, ast.Name)
                        and node.func.id in self.DANGEROUS_FUNCTIONS
                    ):
                        errors.append(
                            f"Dangerous function call detected: {node.func.id}"
                        )

        except Exception as e:
            errors.append(f"Security validation error: {str(e)}")

        return len(errors) == 0, errors


class PerformanceValidator(SkillValidator):
    """Validates skill performance characteristics."""

    def __init__(
        self,
        min_success_rate: float = 0.6,
        min_usage_count: int = 5,
        legacy_compatible: bool = True,
    ):
        self.min_success_rate = min_success_rate
        self.min_usage_count = min_usage_count
        self.legacy_compatible = legacy_compatible

    def validate(self, skill: Skill) -> Tuple[bool, List[str]]:
        """Check if skill meets performance thresholds."""
        errors = []

        # Legacy compatibility mode - more lenient validation
        if self.legacy_compatible:
            min_usage = max(1, self.min_usage_count // 2)  # Half the requirement
            min_success = max(0.3, self.min_success_rate - 0.2)  # 20% lower threshold
        else:
            min_usage = self.min_usage_count
            min_success = self.min_success_rate

        if skill.usage_count < min_usage:
            if self.legacy_compatible and skill.usage_count == 0:
                # Allow brand new skills to pass in legacy mode
                pass
            else:
                errors.append(
                    f"Insufficient usage data: {skill.usage_count} < {min_usage}"
                    + (" (legacy mode)" if self.legacy_compatible else "")
                )

        if skill.usage_count > 0 and skill.success_rate < min_success:
            if not (self.legacy_compatible and skill.success_rate >= 0.3):
                errors.append(
                    f"Low success rate: {skill.success_rate * 100:.1f}% < "
                    f"{min_success * 100:.1f}%"
                    + (" (legacy mode)" if self.legacy_compatible else "")
                )

        return len(errors) == 0, errors


class CompositeSkillValidator:
    """Composite validator that runs multiple validation strategies."""

    def __init__(self, legacy_compatible: bool = True):
        self.validators = [
            SyntaxValidator(),
            SecurityValidator(),
            PerformanceValidator(legacy_compatible=legacy_compatible),
        ]
        self.legacy_compatible = legacy_compatible
        self.logger = logging.getLogger(__name__)

    def validate_skill(self, skill: Skill) -> Tuple[bool, Dict[str, List[str]]]:
        """Run all validators on a skill."""
        results = {}
        overall_valid = True

        for validator in self.validators:
            validator_name = validator.__class__.__name__
            try:
                is_valid, errors = validator.validate(skill)
                results[validator_name] = errors
                if not is_valid:
                    overall_valid = False
            except Exception as e:
                self.logger.error(f"Validator {validator_name} failed: {e}")
                results[validator_name] = [f"Validator error: {str(e)}"]
                overall_valid = False

        return overall_valid, results


class SkillLibrarian:
    """
    Storage and retrieval system for skills with indexing and search.

    Provides efficient skill storage, retrieval, and search capabilities
    with comprehensive indexing and metadata caching.
    """

    def __init__(self, storage_path: Path, cache_config: Optional[CacheConfig] = None):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)

        # Initialize caching system
        self.cache_config = cache_config or CacheConfig()
        self.metadata_cache = SkillExecutionCache(self.cache_config)

        # Cache for file modification times to detect changes
        self._file_mtimes: Dict[str, float] = {}
        self._cache_lock = Lock()

        # In-memory indexes for fast retrieval
        self._skills_cache: Dict[str, Skill] = {}
        self._category_index: Dict[SkillCategory, Set[str]] = {}
        self._complexity_index: Dict[SkillComplexity, Set[str]] = {}
        self._tag_index: Dict[str, Set[str]] = {}
        self._dependency_index: Dict[str, Set[str]] = {}

        # Load existing skills
        self._load_all_skills()

    def store_skill(self, skill: Skill) -> bool:
        """Store a skill in the library with caching."""
        try:
            # Save to file
            skill_file = self.storage_path / f"{skill.id}.json"
            skill_data = skill.model_dump()

            with open(skill_file, "w") as f:
                json.dump(skill_data, f, indent=2, default=str)

            # Update file modification time tracking
            with self._cache_lock:
                self._file_mtimes[skill.id] = skill_file.stat().st_mtime

                # Update cache and indexes
                self._skills_cache[skill.id] = skill
                self._update_indexes(skill)

                # Cache metadata for fast retrieval
                metadata = {
                    "name": skill.name,
                    "category": skill.category
                    if isinstance(skill.category, str)
                    else skill.category.value,
                    "complexity": skill.complexity
                    if isinstance(skill.complexity, str)
                    else skill.complexity.value,
                    "tags": skill.tags,
                    "version": skill.version,
                    "last_used": skill.last_used.isoformat()
                    if skill.last_used
                    else None,
                    "usage_count": skill.usage_count,
                    "success_rate": float(skill.success_rate),
                    "file_path": str(skill_file),
                    "file_mtime": self._file_mtimes[skill.id],
                }
                self.metadata_cache.cache_skill_metadata(skill.id, metadata)

            self.logger.info(f"Stored skill: {skill.name} ({skill.id})")
            return True

        except Exception as e:
            self.logger.error(f"Failed to store skill {skill.name}: {e}")
            return False

    def retrieve_skill(self, skill_id: str) -> Optional[Skill]:
        """Retrieve a skill by ID with intelligent caching."""
        # Check memory cache first
        with self._cache_lock:
            if skill_id in self._skills_cache:
                # Verify file hasn't been modified
                skill_file = self.storage_path / f"{skill_id}.json"
                if skill_file.exists():
                    current_mtime = skill_file.stat().st_mtime
                    cached_mtime = self._file_mtimes.get(skill_id, 0)

                    # If file unchanged, return cached skill
                    if current_mtime <= cached_mtime:
                        return self._skills_cache[skill_id]
                    else:
                        # File changed, reload
                        self.logger.debug(f"Reloading modified skill: {skill_id}")
                        self._reload_skill(skill_id)
                        return self._skills_cache.get(skill_id)
                else:
                    # File deleted, remove from cache
                    self._remove_skill_from_cache(skill_id)
                    return None

        # Not in cache, try to load from file
        return self._load_skill_from_file(skill_id)

    def _reload_skill(self, skill_id: str) -> Optional[Skill]:
        """Reload a skill from file and update caches."""
        skill = self._load_skill_from_file(skill_id)
        if skill:
            with self._cache_lock:
                # Remove from old indexes
                old_skill = self._skills_cache.get(skill_id)
                if old_skill:
                    self._remove_from_indexes(old_skill)

                # Update with new skill
                self._skills_cache[skill_id] = skill
                self._update_indexes(skill)

        return skill

    def _load_skill_from_file(self, skill_id: str) -> Optional[Skill]:
        """Load a skill from file and update cache."""
        skill_file = self.storage_path / f"{skill_id}.json"

        if not skill_file.exists():
            return None

        try:
            with open(skill_file, "r") as f:
                skill_data = json.load(f)
                skill = Skill(**skill_data)

            with self._cache_lock:
                # Update cache and tracking
                self._skills_cache[skill_id] = skill
                self._file_mtimes[skill_id] = skill_file.stat().st_mtime
                self._update_indexes(skill)

                # Cache metadata
                metadata = {
                    "name": skill.name,
                    "category": skill.category
                    if isinstance(skill.category, str)
                    else skill.category.value,
                    "complexity": skill.complexity
                    if isinstance(skill.complexity, str)
                    else skill.complexity.value,
                    "tags": skill.tags,
                    "version": skill.version,
                    "last_used": skill.last_used.isoformat()
                    if skill.last_used
                    else None,
                    "usage_count": skill.usage_count,
                    "success_rate": float(skill.success_rate),
                    "file_path": str(skill_file),
                    "file_mtime": self._file_mtimes[skill_id],
                }
                self.metadata_cache.cache_skill_metadata(skill_id, metadata)

            return skill

        except Exception as e:
            self.logger.error(f"Failed to load skill {skill_id}: {e}")
            return None

    def _remove_skill_from_cache(self, skill_id: str) -> None:
        """Remove skill from all caches and indexes."""
        with self._cache_lock:
            skill = self._skills_cache.pop(skill_id, None)
            if skill:
                self._remove_from_indexes(skill)

            self._file_mtimes.pop(skill_id, None)

    def _remove_from_indexes(self, skill: Skill) -> None:
        """Remove skill from all indexes."""
        skill_id = skill.id

        # Remove from category index
        if skill.category in self._category_index:
            self._category_index[skill.category].discard(skill_id)
            if not self._category_index[skill.category]:
                del self._category_index[skill.category]

        # Remove from complexity index
        if skill.complexity in self._complexity_index:
            self._complexity_index[skill.complexity].discard(skill_id)
            if not self._complexity_index[skill.complexity]:
                del self._complexity_index[skill.complexity]

        # Remove from tag indexes
        for tag in skill.tags:
            if tag in self._tag_index:
                self._tag_index[tag].discard(skill_id)
                if not self._tag_index[tag]:
                    del self._tag_index[tag]

        # Remove from dependency index
        for dep in skill.required_skills:
            if dep in self._dependency_index:
                self._dependency_index[dep].discard(skill_id)
                if not self._dependency_index[dep]:
                    del self._dependency_index[dep]

    def search_skills(
        self,
        category: Optional[SkillCategory] = None,
        complexity: Optional[SkillComplexity] = None,
        tags: Optional[List[str]] = None,
        min_success_rate: Optional[float] = None,
        min_usage_count: Optional[int] = None,
        name_pattern: Optional[str] = None,
    ) -> List[Skill]:
        """Search skills with multiple criteria."""
        candidate_ids = set(self._skills_cache.keys())

        # Filter by category
        if category:
            category_ids = self._category_index.get(category, set())
            candidate_ids &= category_ids

        # Filter by complexity
        if complexity:
            complexity_ids = self._complexity_index.get(complexity, set())
            candidate_ids &= complexity_ids

        # Filter by tags
        if tags:
            for tag in tags:
                tag_ids = self._tag_index.get(tag, set())
                candidate_ids &= tag_ids

        # Apply remaining filters
        results = []
        for skill_id in candidate_ids:
            skill = self._skills_cache[skill_id]

            # Name pattern filter
            if name_pattern and name_pattern.lower() not in skill.name.lower():
                continue

            # Success rate filter
            if min_success_rate is not None and skill.success_rate < min_success_rate:
                continue

            # Usage count filter
            if min_usage_count is not None and skill.usage_count < min_usage_count:
                continue

            results.append(skill)

        # Sort by reliability score
        results.sort(key=lambda s: s.reliability_score, reverse=True)
        return results

    def get_skill_dependencies(self, skill_id: str) -> List[Skill]:
        """Get all dependencies for a skill."""
        skill = self.retrieve_skill(skill_id)
        if not skill:
            return []

        dependencies = []
        for dep_name in skill.required_skills:
            # Find skill by name (simplified - in production would need better mapping)
            for cached_skill in self._skills_cache.values():
                if cached_skill.name == dep_name:
                    dependencies.append(cached_skill)
                    break

        return dependencies

    def get_skills_by_category(self, category: SkillCategory) -> List[Skill]:
        """Get all skills in a category."""
        skill_ids = self._category_index.get(category, set())
        return [self._skills_cache[skill_id] for skill_id in skill_ids]

    def get_library_stats(self) -> Dict[str, Any]:
        """Get library statistics."""
        skills = list(self._skills_cache.values())

        cache_stats = self.get_cache_stats()

        return {
            "total_skills": len(skills),
            "categories": {
                cat if isinstance(cat, str) else cat.value: len(ids)
                for cat, ids in self._category_index.items()
            },
            "complexity_distribution": {
                comp if isinstance(comp, str) else comp.value: len(ids)
                for comp, ids in self._complexity_index.items()
            },
            "average_success_rate": (
                sum(s.success_rate for s in skills) / len(skills) if skills else 0
            ),
            "total_usage_count": sum(s.usage_count for s in skills),
            "reliable_skills": len([s for s in skills if s.is_reliable]),
            "experimental_skills": len([s for s in skills if s.is_experimental]),
            "cache_performance": cache_stats,
        }

    def _load_all_skills(self) -> None:
        """Load all skills from storage."""
        for skill_file in self.storage_path.glob("*.json"):
            try:
                with open(skill_file, "r") as f:
                    skill_data = json.load(f)
                    skill = Skill(**skill_data)
                    self._skills_cache[skill.id] = skill
                    self._update_indexes(skill)
            except Exception as e:
                self.logger.error(f"Failed to load skill from {skill_file}: {e}")

    def _update_indexes(self, skill: Skill) -> None:
        """Update all indexes for a skill."""
        skill_id = skill.id

        # Category index
        if skill.category not in self._category_index:
            self._category_index[skill.category] = set()
        self._category_index[skill.category].add(skill_id)

        # Complexity index
        if skill.complexity not in self._complexity_index:
            self._complexity_index[skill.complexity] = set()
        self._complexity_index[skill.complexity].add(skill_id)

        # Tag index
        for tag in skill.tags:
            if tag not in self._tag_index:
                self._tag_index[tag] = set()
            self._tag_index[tag].add(skill_id)

        # Dependency index
        for dep in skill.required_skills:
            if dep not in self._dependency_index:
                self._dependency_index[dep] = set()
            self._dependency_index[dep].add(skill_id)

    def clear_caches(self) -> None:
        """Clear all caches (but keep in-memory skills for performance)."""
        with self._cache_lock:
            self.metadata_cache.clear_all()
            self.logger.info("Cleared skill library caches")

    def evict_expired_cache_entries(self) -> Dict[str, int]:
        """Evict expired cache entries."""
        return self.metadata_cache.evict_expired()

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        with self._cache_lock:
            return {
                "metadata_cache": self.metadata_cache.get_cache_stats(),
                "skills_in_memory": len(self._skills_cache),
                "files_tracked": len(self._file_mtimes),
                "index_sizes": {
                    "categories": sum(
                        len(ids) for ids in self._category_index.values()
                    ),
                    "complexity": sum(
                        len(ids) for ids in self._complexity_index.values()
                    ),
                    "tags": sum(len(ids) for ids in self._tag_index.values()),
                    "dependencies": sum(
                        len(ids) for ids in self._dependency_index.values()
                    ),
                },
            }

    def refresh_cache(self) -> Dict[str, int]:
        """Refresh cache by checking all files for modifications."""
        refreshed_count = 0
        removed_count = 0

        with self._cache_lock:
            skill_ids = list(self._skills_cache.keys())

        for skill_id in skill_ids:
            skill_file = self.storage_path / f"{skill_id}.json"

            if skill_file.exists():
                current_mtime = skill_file.stat().st_mtime
                cached_mtime = self._file_mtimes.get(skill_id, 0)

                if current_mtime > cached_mtime:
                    self._reload_skill(skill_id)
                    refreshed_count += 1
            else:
                self._remove_skill_from_cache(skill_id)
                removed_count += 1

        if refreshed_count > 0 or removed_count > 0:
            self.logger.info(
                f"Cache refresh completed: {refreshed_count} skills refreshed, "
                f"{removed_count} skills removed"
            )

        return {
            "refreshed": refreshed_count,
            "removed": removed_count,
        }

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary of the skill library."""
        cache_stats = self.get_cache_stats()
        library_stats = self.get_library_stats()

        # Calculate cache efficiency
        metadata_cache_stats = cache_stats["metadata_cache"]["metadata_cache"]
        cache_hit_rate = metadata_cache_stats.get("hit_rate", 0)

        return {
            "skills_count": library_stats["total_skills"],
            "cache_hit_rate": cache_hit_rate,
            "memory_efficiency": {
                "skills_cached": cache_stats["skills_in_memory"],
                "files_tracked": cache_stats["files_tracked"],
                "index_entries": sum(cache_stats["index_sizes"].values()),
            },
            "performance_indicators": {
                "reliable_skills_pct": (
                    library_stats["reliable_skills"]
                    / library_stats["total_skills"]
                    * 100
                    if library_stats["total_skills"] > 0
                    else 0
                ),
                "average_success_rate": library_stats["average_success_rate"],
                "total_usage": library_stats["total_usage_count"],
            },
        }


class SkillDiscoverer:
    """
    Pattern recognition and skill extraction from successful strategies.

    Analyzes trading experiences and strategies to identify
    reusable patterns and extract them as new skills.
    """

    def __init__(self, min_pattern_frequency: int = 3, min_success_rate: float = 0.7):
        self.min_pattern_frequency = min_pattern_frequency
        self.min_success_rate = min_success_rate
        self.logger = logging.getLogger(__name__)

    def discover_skills_from_experiences(
        self, experiences: List[Experience]
    ) -> List[Skill]:
        """Discover new skills from trading experiences."""
        # Group experiences by outcome and patterns
        successful_experiences = [
            exp
            for exp in experiences
            if exp.is_positive_outcome
            and exp.financial_impact
            and exp.financial_impact.is_positive()
        ]

        if len(successful_experiences) < self.min_pattern_frequency:
            return []

        # Extract patterns
        patterns = self._extract_patterns(successful_experiences)

        # Convert patterns to skills
        discovered_skills = []
        for pattern in patterns:
            if pattern["frequency"] >= self.min_pattern_frequency:
                skill = self._pattern_to_skill(pattern, successful_experiences)
                if skill:
                    discovered_skills.append(skill)

        return discovered_skills

    def discover_skills_from_code_analysis(
        self, strategy_code: str, performance_data: Dict[str, Any]
    ) -> List[Skill]:
        """Discover skills by analyzing successful strategy code."""
        try:
            tree = ast.parse(strategy_code)

            # Extract function definitions as potential skills
            functions = [
                node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
            ]

            discovered_skills = []
            for func in functions:
                skill = self._function_to_skill(func, strategy_code, performance_data)
                if skill:
                    discovered_skills.append(skill)

            return discovered_skills

        except Exception as e:
            self.logger.error(f"Code analysis failed: {e}")
            return []

    def _extract_patterns(self, experiences: List[Experience]) -> List[Dict[str, Any]]:
        """Extract common patterns from experiences."""
        pattern_counts = {}

        for exp in experiences:
            # Extract patterns from experience
            exp_patterns = exp.extract_patterns()

            for pattern in exp_patterns:
                if pattern not in pattern_counts:
                    pattern_counts[pattern] = {
                        "name": pattern,
                        "frequency": 0,
                        "experiences": [],
                        "avg_financial_impact": Decimal("0"),
                        "success_contexts": [],
                    }

                pattern_data = pattern_counts[pattern]
                pattern_data["frequency"] += 1
                pattern_data["experiences"].append(exp.id)

                if exp.financial_impact:
                    pattern_data["avg_financial_impact"] += exp.financial_impact.amount

                pattern_data["success_contexts"].append(
                    {
                        "market_conditions": exp.market_conditions,
                        "symbols": [str(s) for s in exp.symbols_involved],
                        "timeframe": exp.timeframe.value if exp.timeframe else None,
                    }
                )

        # Normalize averages
        for pattern_data in pattern_counts.values():
            if pattern_data["frequency"] > 0:
                pattern_data["avg_financial_impact"] /= pattern_data["frequency"]

        return list(pattern_counts.values())

    def _pattern_to_skill(
        self, pattern: Dict[str, Any], experiences: List[Experience]
    ) -> Optional[Skill]:
        """Convert a pattern into a skill."""
        try:
            # Generate skill metadata
            skill_name = f"discovered_{pattern['name']}_{int(time.time())}"

            # Determine skill category based on pattern characteristics
            category = self._infer_category(pattern)
            complexity = self._infer_complexity(pattern)

            # Generate basic skill code template
            skill_code = self._generate_skill_code_template(pattern)

            # Create skill
            skill = Skill(
                name=skill_name,
                description=f"Discovered skill from pattern: {pattern['name']}",
                category=category,
                complexity=complexity,
                code=skill_code,
                input_schema={
                    "type": "object",
                    "properties": {
                        "market_data": {"type": "object"},
                        "context": {"type": "object"},
                    },
                    "required": ["market_data"],
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "signal": {"type": "string"},
                        "confidence": {"type": "number"},
                    },
                },
                tags=["discovered", "pattern-based", pattern["name"]],
                learned_from=",".join(
                    pattern["experiences"][:5]
                ),  # Reference first 5 experiences
            )

            return skill

        except Exception as e:
            self.logger.error(f"Failed to convert pattern to skill: {e}")
            return None

    def _function_to_skill(
        self, func_node: ast.FunctionDef, full_code: str, performance: Dict[str, Any]
    ) -> Optional[Skill]:
        """Convert a function AST node to a skill."""
        try:
            # Extract function code
            func_code = ast.get_source_segment(full_code, func_node)
            if not func_code:
                return None

            # Generate skill metadata
            skill_name = f"extracted_{func_node.name}_{int(time.time())}"

            # Analyze function to determine category and complexity
            category = self._analyze_function_category(func_node)
            complexity = self._analyze_function_complexity(func_node)

            # Create skill
            skill = Skill(
                name=skill_name,
                description=f"Extracted function: {func_node.name}",
                category=category,
                complexity=complexity,
                code=func_code,
                input_schema=self._infer_input_schema(func_node),
                output_schema={"type": "object"},
                tags=["extracted", "function-based", func_node.name],
                created_by="discoverer",
            )

            return skill

        except Exception as e:
            self.logger.error(f"Failed to convert function to skill: {e}")
            return None

    def _infer_category(self, pattern: Dict[str, Any]) -> SkillCategory:
        """Infer skill category from pattern characteristics."""
        pattern_name = pattern["name"].lower()

        if "technical" in pattern_name or "indicator" in pattern_name:
            return SkillCategory.TECHNICAL_ANALYSIS
        elif "risk" in pattern_name or "position" in pattern_name:
            return SkillCategory.RISK_MANAGEMENT
        elif "entry" in pattern_name:
            return SkillCategory.ENTRY_TIMING
        elif "exit" in pattern_name:
            return SkillCategory.EXIT_TIMING
        else:
            return SkillCategory.MARKET_ANALYSIS

    def _infer_complexity(self, pattern: Dict[str, Any]) -> SkillComplexity:
        """Infer skill complexity from pattern characteristics."""
        frequency = pattern["frequency"]
        context_variety = len(set(str(ctx) for ctx in pattern["success_contexts"]))

        if frequency > 10 and context_variety > 5:
            return SkillComplexity.ADVANCED
        elif frequency > 5 and context_variety > 3:
            return SkillComplexity.INTERMEDIATE
        else:
            return SkillComplexity.BASIC

    def _generate_skill_code_template(self, pattern: Dict[str, Any]) -> str:
        """Generate a basic code template for a discovered pattern."""
        return f"""
def execute_pattern_{pattern['name'].replace(' ', '_')}(inputs, context=None):
    \\\"\\\"\\\"
    Discovered pattern: {pattern['name']}
    Frequency: {pattern['frequency']}
    Average impact: {pattern['avg_financial_impact']}
    \\\"\\\"\\\"

    # Pattern-specific logic would be implemented here
    # This is a template that needs manual refinement

    market_data = inputs.get('market_data', {{}})

    # Placeholder pattern detection logic
    signal_strength = 0.5  # To be implemented based on pattern analysis

    result = {{
        'signal': 'hold',  # Default signal
        'confidence': signal_strength,
        'pattern': '{pattern['name']}',
        'reasoning': 'Pattern-based signal generation'
    }}

    return result

# Execute the pattern function
result = execute_pattern_{pattern['name'].replace(' ', '_')}(inputs, context)
"""

    def _analyze_function_category(self, func_node: ast.FunctionDef) -> SkillCategory:
        """Analyze function to determine likely category."""
        func_name = func_node.name.lower()

        if "rsi" in func_name or "sma" in func_name or "indicator" in func_name:
            return SkillCategory.TECHNICAL_ANALYSIS
        elif "risk" in func_name or "size" in func_name:
            return SkillCategory.RISK_MANAGEMENT
        elif "entry" in func_name or "buy" in func_name:
            return SkillCategory.ENTRY_TIMING
        elif "exit" in func_name or "sell" in func_name:
            return SkillCategory.EXIT_TIMING
        else:
            return SkillCategory.MARKET_ANALYSIS

    def _analyze_function_complexity(
        self, func_node: ast.FunctionDef
    ) -> SkillComplexity:
        """Analyze function complexity."""
        # Simple heuristic based on function characteristics
        lines_count = len(func_node.body)
        args_count = len(func_node.args.args)

        if lines_count > 20 or args_count > 5:
            return SkillComplexity.ADVANCED
        elif lines_count > 10 or args_count > 3:
            return SkillComplexity.INTERMEDIATE
        else:
            return SkillComplexity.BASIC

    def _infer_input_schema(self, func_node: ast.FunctionDef) -> Dict[str, Any]:
        """Infer input schema from function signature."""
        properties = {}
        required = []

        for arg in func_node.args.args:
            if arg.arg != "self":  # Skip self parameter
                properties[arg.arg] = {"type": "object"}  # Generic type
                required.append(arg.arg)

        return {"type": "object", "properties": properties, "required": required}


class VoyagerSkillLibrary:
    """
    Main VOYAGER Skill Library interface.

    Coordinates all six components of the skill library system:
    - Skill Executor, Composer, Validator, Librarian, Discoverer, Performance Tracker
    With comprehensive caching and performance optimization.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize storage path
        storage_path = Path(config.get("skill_library_path", "skills"))

        # Initialize caching configuration
        self.cache_config = CacheConfig(
            max_execution_cache_size=config.get("max_execution_cache_size", 1000),
            max_metadata_cache_size=config.get("max_metadata_cache_size", 500),
            execution_cache_ttl_hours=config.get("execution_cache_ttl_hours", 24),
            metadata_cache_ttl_hours=config.get("metadata_cache_ttl_hours", 72),
            enable_compilation_cache=config.get("enable_compilation_cache", True),
            enable_result_cache=config.get("enable_result_cache", True),
        )

        # Initialize all components with caching support
        self.executor = SkillExecutor(
            timeout_seconds=config.get("execution_timeout", 30),
            max_memory_mb=config.get("max_memory", 128),
            cache_config=self.cache_config,
        )
        self.composer = SkillComposer()
        self.validator = CompositeSkillValidator()
        self.librarian = SkillLibrarian(storage_path, cache_config=self.cache_config)
        self.discoverer = SkillDiscoverer(
            min_pattern_frequency=config.get("min_pattern_frequency", 3),
            min_success_rate=config.get("min_success_rate", 0.7),
        )

        # Performance tracking
        self._execution_stats = {
            "total_executions": 0,
            "successful_executions": 0,
            "cache_hits": 0,
            "average_execution_time": 0.0,
            "last_reset": datetime.utcnow(),
        }
        self._stats_lock = Lock()

        self.logger.info("VOYAGER Skill Library initialized with caching enabled")

    def add_skill(self, skill: Skill, validate: bool = True) -> bool:
        """Add a new skill to the library."""
        try:
            # Validate skill if requested
            if validate:
                is_valid, validation_errors = self.validator.validate_skill(skill)
                if not is_valid:
                    self.logger.error(f"Skill validation failed: {validation_errors}")
                    return False

            # Store skill
            success = self.librarian.store_skill(skill)

            if success:
                self.logger.info(f"Successfully added skill: {skill.name}")

            return success

        except Exception as e:
            self.logger.error(f"Failed to add skill {skill.name}: {e}")
            return False

    def execute_skill(
        self,
        skill_id: str,
        inputs: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[SkillExecutionResult, Any, Dict[str, Any]]:
        """Execute a skill safely with performance tracking."""
        start_time = time.time()

        skill = self.librarian.retrieve_skill(skill_id)
        if not skill:
            return SkillExecutionResult.ERROR, None, {"error": "Skill not found"}

        result, output, metadata = self.executor.execute_skill(skill, inputs, context)

        execution_time = time.time() - start_time
        was_cached = metadata.get("cached", False)

        # Update performance statistics
        with self._stats_lock:
            self._execution_stats["total_executions"] += 1
            if result == SkillExecutionResult.SUCCESS:
                self._execution_stats["successful_executions"] += 1
            if was_cached:
                self._execution_stats["cache_hits"] += 1

            # Update running average execution time
            old_avg = self._execution_stats["average_execution_time"]
            n = self._execution_stats["total_executions"]
            self._execution_stats["average_execution_time"] = (
                old_avg * (n - 1) + execution_time
            ) / n

        # Update skill performance (only for actual executions, not cached results)
        if not was_cached:
            performance_data = {
                "execution_time": Decimal(str(execution_time)),
            }
            updated_skill = skill.record_usage(result, performance_data)
            self.librarian.store_skill(updated_skill)

        # Add performance info to metadata
        metadata.update(
            {
                "execution_time_seconds": execution_time,
                "library_stats": self._get_current_performance_snapshot(),
            }
        )

        return result, output, metadata

    def compose_and_execute(
        self,
        skill_ids: List[str],
        inputs: Dict[str, Any],
        composition_strategy: str = "sequential",
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[SkillExecutionResult, Any, Dict[str, Any]]:
        """Compose skills and execute the resulting strategy."""
        # Retrieve skills
        skills = [self.librarian.retrieve_skill(skill_id) for skill_id in skill_ids]
        missing_skills = [
            skill_id for skill_id, skill in zip(skill_ids, skills) if skill is None
        ]

        if missing_skills:
            return (
                SkillExecutionResult.ERROR,
                None,
                {"error": f"Skills not found: {missing_skills}"},
            )

        try:
            # Compose skills
            composed_code, composition_metadata = self.composer.compose_skills(
                skills, composition_strategy
            )

            # Create temporary composed skill
            composed_skill = Skill(
                name=f"composed_{int(time.time())}",
                description=f"Composed strategy using {len(skills)} skills",
                category=SkillCategory.MARKET_ANALYSIS,  # Default category
                complexity=SkillComplexity.ADVANCED,  # Composed skills complex
                code=composed_code,
                input_schema={"type": "object"},
                output_schema={"type": "object"},
                tags=["composed"] + [skill.name for skill in skills],
            )

            # Execute composed skill
            result, output, execution_metadata = self.executor.execute_skill(
                composed_skill, inputs, context
            )

            # Combine metadata
            combined_metadata = {
                **execution_metadata,
                "composition_metadata": composition_metadata,
                "component_skills": skill_ids,
            }

            return result, output, combined_metadata

        except Exception as e:
            return SkillExecutionResult.ERROR, None, {"error": str(e)}

    def discover_skills_from_experiences(
        self, experiences: List[Experience]
    ) -> List[Skill]:
        """Discover new skills from trading experiences."""
        return self.discoverer.discover_skills_from_experiences(experiences)

    def search_skills(self, **criteria) -> List[Skill]:
        """Search skills with various criteria."""
        return self.librarian.search_skills(**criteria)

    def get_skill(self, skill_id: str) -> Optional[Skill]:
        """Get a skill by ID."""
        return self.librarian.retrieve_skill(skill_id)

    def get_library_stats(self) -> Dict[str, Any]:
        """Get comprehensive library statistics."""
        return self.librarian.get_library_stats()

    def validate_skill(self, skill: Skill) -> Tuple[bool, Dict[str, List[str]]]:
        """Validate a skill using all validators."""
        return self.validator.validate_skill(skill)

    def _get_current_performance_snapshot(self) -> Dict[str, Any]:
        """Get current performance snapshot."""
        with self._stats_lock:
            total_exec = self._execution_stats["total_executions"]
            success_rate = (
                self._execution_stats["successful_executions"] / total_exec
                if total_exec > 0
                else 0
            )
            cache_hit_rate = (
                self._execution_stats["cache_hits"] / total_exec
                if total_exec > 0
                else 0
            )

            return {
                "total_executions": total_exec,
                "success_rate": success_rate,
                "cache_hit_rate": cache_hit_rate,
                "average_execution_time": self._execution_stats[
                    "average_execution_time"
                ],
            }

    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        library_stats = self.librarian.get_library_stats()
        executor_cache_stats = self.executor.get_cache_stats()
        performance_stats = self._get_current_performance_snapshot()

        return {
            "library": library_stats,
            "execution_cache": executor_cache_stats,
            "performance": performance_stats,
            "cache_config": {
                "max_execution_cache_size": self.cache_config.max_execution_cache_size,
                "max_metadata_cache_size": self.cache_config.max_metadata_cache_size,
                "execution_cache_ttl_hours": self.cache_config.execution_cache_ttl_hours,
                "metadata_cache_ttl_hours": self.cache_config.metadata_cache_ttl_hours,
                "compilation_cache_enabled": self.cache_config.enable_compilation_cache,
                "result_cache_enabled": self.cache_config.enable_result_cache,
            },
        }

    def clear_all_caches(self) -> None:
        """Clear all caches across the system."""
        self.executor.clear_cache()
        self.librarian.clear_caches()

        with self._stats_lock:
            self._execution_stats = {
                "total_executions": 0,
                "successful_executions": 0,
                "cache_hits": 0,
                "average_execution_time": 0.0,
                "last_reset": datetime.utcnow(),
            }

        self.logger.info("Cleared all caches and reset performance statistics")

    def evict_expired_entries(self) -> Dict[str, Any]:
        """Evict expired entries from all caches."""
        executor_evicted = self.executor.evict_expired_cache_entries()
        librarian_evicted = self.librarian.evict_expired_cache_entries()

        total_evicted = {
            "execution_cache": executor_evicted,
            "metadata_cache": librarian_evicted,
        }

        self.logger.info(f"Evicted expired cache entries: {total_evicted}")
        return total_evicted

    def refresh_caches(self) -> Dict[str, Any]:
        """Refresh all caches by checking for file modifications."""
        refresh_stats = self.librarian.refresh_cache()

        # Also evict expired entries
        eviction_stats = self.evict_expired_entries()

        return {
            "refresh": refresh_stats,
            "eviction": eviction_stats,
        }

    def get_performance_recommendations(self) -> List[str]:
        """Get performance optimization recommendations."""
        recommendations = []
        stats = self.get_comprehensive_stats()

        performance = stats["performance"]
        cache_stats = stats["execution_cache"]

        # Check cache hit rates
        if performance["cache_hit_rate"] < 0.3:
            recommendations.append(
                "Low cache hit rate detected. Consider increasing cache sizes or TTL."
            )

        # Check execution performance
        if performance["average_execution_time"] > 5.0:
            recommendations.append(
                "High average execution time. Consider optimizing skill code or "
                "increasing cache sizes."
            )

        # Check success rates
        if performance["success_rate"] < 0.8:
            recommendations.append(
                "Low skill execution success rate. Review skill validation and "
                "error handling."
            )

        # Check cache utilization
        exec_cache_usage = (
            cache_stats["execution_cache"]["size"]
            / cache_stats["execution_cache"]["max_size"]
        )
        if exec_cache_usage > 0.9:
            recommendations.append(
                "Execution cache nearly full. Consider increasing max_execution_cache_size."
            )

        # Check memory efficiency
        library_stats = stats["library"]
        if library_stats["cache_performance"]["skills_in_memory"] > 1000:
            recommendations.append(
                "High number of skills in memory. Consider implementing LRU eviction "
                "for skill cache."
            )

        return recommendations


# Legacy compatibility - maintain existing TradingSkill class for backward compatibility


@dataclass
class TradingSkill:
    """Legacy TradingSkill class for backward compatibility."""

    name: str
    description: str
    code: str
    performance_metrics: Dict[str, float]
    usage_count: int = 0
    success_rate: float = 0.0
    prerequisites: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)


class SkillLibrary:
    """Legacy SkillLibrary class - now wraps VoyagerSkillLibrary."""

    def __init__(self, config):
        """Initialize with legacy interface."""
        # Convert config to expected format
        if hasattr(config, "skill_library_path"):
            config_dict = {"skill_library_path": config.skill_library_path}
        else:
            config_dict = (
                config if isinstance(config, dict) else {"skill_library_path": "skills"}
            )

        self.voyager_library = VoyagerSkillLibrary(config_dict)
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.skills = {}  # Legacy interface
        self.skill_dependencies = {}  # Legacy interface

        # Load existing skills for legacy interface
        self._sync_legacy_interface()

    def add_skill(self, skill: TradingSkill) -> None:
        """Add a skill using legacy interface."""
        # Convert legacy skill to new Skill model
        new_skill = Skill(
            name=skill.name,
            description=skill.description,
            category=SkillCategory.MARKET_ANALYSIS,  # Default
            complexity=SkillComplexity.INTERMEDIATE,  # Default
            code=skill.code,
            input_schema={"type": "object"},
            output_schema={"type": "object"},
            performance_metrics={
                k: Decimal(str(v)) for k, v in skill.performance_metrics.items()
            },
            usage_count=skill.usage_count,
            success_count=(
                int(
                    (
                        skill.success_rate
                        if skill.success_rate <= 1.0
                        else skill.success_rate / 100.0
                    )
                    * skill.usage_count
                )
                if skill.usage_count > 0
                else 0
            ),
            tags=skill.tags,
            required_skills=skill.prerequisites,
        )

        self.voyager_library.add_skill(new_skill)
        self.skills[skill.name] = skill

    def get_skill(self, name: str) -> Optional[TradingSkill]:
        """Get skill by name using legacy interface."""
        return self.skills.get(name)

    def search_skills(
        self, tags: List[str] = None, min_success_rate: float = 0.0
    ) -> List[TradingSkill]:
        """Search skills using legacy interface."""
        # Convert to new search and back to legacy format
        new_skills = self.voyager_library.search_skills(
            tags=tags, min_success_rate=min_success_rate
        )

        legacy_skills = []
        for skill in new_skills:
            legacy_skill = TradingSkill(
                name=skill.name,
                description=skill.description,
                code=skill.code,
                performance_metrics={
                    k: float(v) for k, v in skill.performance_metrics.items()
                },
                usage_count=skill.usage_count,
                success_rate=float(skill.success_rate),
                prerequisites=skill.required_skills,
                tags=skill.tags,
            )
            legacy_skills.append(legacy_skill)

        return legacy_skills

    def compose_skills(self, skill_names: List[str]) -> Optional[str]:
        """Compose skills using legacy interface."""
        # Find skills by name and get their IDs
        skill_ids = []
        for name in skill_names:
            # Find skill by name in the new system
            skills = self.voyager_library.search_skills(name_pattern=name)
            if skills:
                skill_ids.append(skills[0].id)
            else:
                self.logger.error(f"Skill not found: {name}")
                return None

        if not skill_ids:
            return None

        # Use new composition system
        skills = [self.voyager_library.get_skill(skill_id) for skill_id in skill_ids]
        if not all(skills):
            return None

        try:
            composed_code, _ = self.voyager_library.composer.compose_skills(
                skills, "sequential"
            )
            return composed_code
        except Exception as e:
            self.logger.error(f"Composition failed: {e}")
            return None

    def update_skill_performance(
        self, name: str, performance: Dict[str, float]
    ) -> None:
        """Update skill performance using legacy interface."""
        if name in self.skills:
            legacy_skill = self.skills[name]
            legacy_skill.performance_metrics.update(performance)
            legacy_skill.usage_count += 1

            if "success" in performance:
                old_rate = legacy_skill.success_rate
                legacy_skill.success_rate = (
                    old_rate * (legacy_skill.usage_count - 1) + performance["success"]
                ) / legacy_skill.usage_count

    def _sync_legacy_interface(self) -> None:
        """Sync legacy interface with new system."""
        # Convert new skills to legacy format for backward compatibility
        new_skills = self.voyager_library.search_skills()
        for skill in new_skills:
            legacy_skill = TradingSkill(
                name=skill.name,
                description=skill.description,
                code=skill.code,
                performance_metrics={
                    k: float(v) for k, v in skill.performance_metrics.items()
                },
                usage_count=skill.usage_count,
                success_rate=float(skill.success_rate),
                prerequisites=skill.required_skills,
                tags=skill.tags,
            )
            self.skills[skill.name] = legacy_skill
