"""Multi-level caching system for market data."""

import asyncio
import hashlib
import json
import logging
import pickle
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union

logger = logging.getLogger(__name__)


class CacheEntry:
    """Represents a cache entry with metadata."""

    def __init__(self, data: Any, ttl: int, created_at: Optional[datetime] = None):
        self.data = data
        self.ttl = ttl  # Time to live in seconds
        self.created_at = created_at or datetime.now(timezone.utc)
        self.access_count = 0
        self.last_accessed = self.created_at

    @property
    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        if self.ttl <= 0:  # TTL of 0 or negative means no expiration
            return False

        age = (datetime.now(timezone.utc) - self.created_at).total_seconds()
        return age > self.ttl

    def access(self) -> Any:
        """Access the cached data and update metadata."""
        self.access_count += 1
        self.last_accessed = datetime.now(timezone.utc)
        return self.data


class MemoryCache:
    """In-memory cache with TTL and size limits."""

    def __init__(self, max_size: int = 1000, default_ttl: int = 300):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[Any]:
        """Get data from cache."""
        async with self._lock:
            if key not in self._cache:
                return None

            entry = self._cache[key]
            if entry.is_expired:
                del self._cache[key]
                return None

            return entry.access()

    async def set(self, key: str, data: Any, ttl: Optional[int] = None) -> None:
        """Set data in cache."""
        ttl = ttl if ttl is not None else self.default_ttl

        async with self._lock:
            # Remove expired entries if at capacity
            if len(self._cache) >= self.max_size:
                await self._evict_entries()

            self._cache[key] = CacheEntry(data, ttl)

    async def delete(self, key: str) -> bool:
        """Delete entry from cache."""
        async with self._lock:
            return self._cache.pop(key, None) is not None

    async def clear(self) -> None:
        """Clear all cache entries."""
        async with self._lock:
            self._cache.clear()

    async def _evict_entries(self) -> None:
        """Evict expired or least recently used entries."""
        # First, remove expired entries
        expired_keys = [key for key, entry in self._cache.items() if entry.is_expired]

        for key in expired_keys:
            del self._cache[key]

        # If still at capacity, remove LRU entries
        while len(self._cache) >= self.max_size:
            lru_key = min(
                self._cache.keys(), key=lambda k: self._cache[k].last_accessed
            )
            del self._cache[lru_key]

    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        async with self._lock:
            total_entries = len(self._cache)
            expired_count = sum(1 for entry in self._cache.values() if entry.is_expired)

            return {
                "total_entries": total_entries,
                "expired_entries": expired_count,
                "max_size": self.max_size,
                "default_ttl": self.default_ttl,
                "utilization": (
                    total_entries / self.max_size if self.max_size > 0 else 0
                ),
            }


class DiskCache:
    """Persistent disk cache for long-term storage."""

    def __init__(self, cache_dir: Union[str, Path] = "cache", max_size_mb: int = 1000):
        self.cache_dir = Path(cache_dir)
        self.max_size_mb = max_size_mb
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Metadata file
        self.metadata_file = self.cache_dir / "metadata.json"
        self._metadata: Dict[str, Dict] = self._load_metadata()

    def _load_metadata(self) -> Dict[str, Dict]:
        """Load cache metadata from disk."""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, "r") as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load cache metadata: {e}")

        return {}

    def _save_metadata(self) -> None:
        """Save cache metadata to disk."""
        try:
            with open(self.metadata_file, "w") as f:
                json.dump(self._metadata, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save cache metadata: {e}")

    def _get_cache_path(self, key: str) -> Path:
        """Get file path for cache key."""
        # Hash the key to create a filename
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.pickle"

    async def get(self, key: str) -> Optional[Any]:
        """Get data from disk cache."""
        if key not in self._metadata:
            return None

        metadata = self._metadata[key]

        # Check if expired
        created_at = datetime.fromisoformat(metadata["created_at"])
        ttl = metadata["ttl"]
        if ttl > 0:
            age = (datetime.utcnow() - created_at).total_seconds()
            if age > ttl:
                await self.delete(key)
                return None

        # Load data from file
        cache_path = self._get_cache_path(key)
        if not cache_path.exists():
            # Metadata exists but file doesn't - cleanup
            del self._metadata[key]
            self._save_metadata()
            return None

        try:
            with open(cache_path, "rb") as f:
                data = pickle.load(f)

            # Update access metadata
            self._metadata[key]["access_count"] += 1
            self._metadata[key]["last_accessed"] = datetime.utcnow().isoformat()

            return data
        except Exception as e:
            logger.error(f"Failed to load cached data for key {key}: {e}")
            await self.delete(key)
            return None

    async def set(self, key: str, data: Any, ttl: int = 3600) -> None:
        """Set data in disk cache."""
        cache_path = self._get_cache_path(key)

        try:
            # Save data to file
            with open(cache_path, "wb") as f:
                pickle.dump(data, f)

            # Update metadata
            self._metadata[key] = {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "ttl": ttl,
                "access_count": 0,
                "last_accessed": datetime.now(timezone.utc).isoformat(),
                "file_size": cache_path.stat().st_size,
            }

            self._save_metadata()

            # Check cache size and cleanup if necessary
            await self._cleanup_if_needed()

        except Exception as e:
            logger.error(f"Failed to cache data for key {key}: {e}")

    async def delete(self, key: str) -> bool:
        """Delete entry from disk cache."""
        if key not in self._metadata:
            return False

        cache_path = self._get_cache_path(key)

        try:
            if cache_path.exists():
                cache_path.unlink()

            del self._metadata[key]
            self._save_metadata()
            return True
        except Exception as e:
            logger.error(f"Failed to delete cached data for key {key}: {e}")
            return False

    async def clear(self) -> None:
        """Clear all disk cache."""
        try:
            for cache_file in self.cache_dir.glob("*.pickle"):
                cache_file.unlink()

            self._metadata.clear()
            self._save_metadata()
        except Exception as e:
            logger.error(f"Failed to clear disk cache: {e}")

    async def _cleanup_if_needed(self) -> None:
        """Clean up cache if size exceeds limit."""
        total_size_mb = self._get_total_size_mb()

        if total_size_mb <= self.max_size_mb:
            return

        logger.info(
            f"Disk cache size ({total_size_mb:.1f}MB) exceeds limit "
            f"({self.max_size_mb}MB), cleaning up..."
        )

        # Sort by last accessed (oldest first)
        entries_by_age = sorted(
            self._metadata.items(), key=lambda x: x[1]["last_accessed"]
        )

        # Remove entries until under size limit
        for key, metadata in entries_by_age:
            await self.delete(key)

            total_size_mb = self._get_total_size_mb()
            if total_size_mb <= self.max_size_mb * 0.8:  # Leave some buffer
                break

    def _get_total_size_mb(self) -> float:
        """Get total cache size in MB."""
        total_bytes = sum(
            metadata.get("file_size", 0) for metadata in self._metadata.values()
        )
        return total_bytes / (1024 * 1024)

    async def get_stats(self) -> Dict[str, Any]:
        """Get disk cache statistics."""
        total_entries = len(self._metadata)
        total_size_mb = self._get_total_size_mb()

        # Count expired entries
        expired_count = 0
        now = datetime.utcnow()
        for metadata in self._metadata.values():
            created_at = datetime.fromisoformat(metadata["created_at"])
            ttl = metadata["ttl"]
            if ttl > 0 and (now - created_at).total_seconds() > ttl:
                expired_count += 1

        return {
            "total_entries": total_entries,
            "expired_entries": expired_count,
            "total_size_mb": total_size_mb,
            "max_size_mb": self.max_size_mb,
            "utilization": (
                total_size_mb / self.max_size_mb if self.max_size_mb > 0 else 0
            ),
            "cache_directory": str(self.cache_dir),
        }


class DataCache:
    """Multi-level cache combining memory and disk storage."""

    def __init__(
        self,
        memory_cache_size: int = 1000,
        memory_default_ttl: int = 300,  # 5 minutes
        disk_cache_size_mb: int = 1000,  # 1GB
        disk_default_ttl: int = 3600,  # 1 hour
        cache_dir: Union[str, Path] = "cache",
    ):
        self.memory_cache = MemoryCache(memory_cache_size, memory_default_ttl)
        self.disk_cache = DiskCache(cache_dir, disk_cache_size_mb)
        self.disk_default_ttl = disk_default_ttl

    def _generate_key(self, source: str, method: str, args: tuple, kwargs: Dict) -> str:
        """Generate collision-resistant cache key from parameters."""
        # Create a consistent, detailed key from the parameters
        key_data = {
            "source": source,
            "method": method,
            "args": args,  # Keep as tuple for proper serialization
            "kwargs": kwargs,
            "version": "v1",  # Version for cache invalidation if needed
        }

        # Use JSON with sorted keys for consistent serialization
        try:
            key_str = json.dumps(
                key_data, sort_keys=True, default=str, separators=(",", ":")
            )
        except (TypeError, ValueError):
            # Fallback for non-serializable objects
            key_str = f"{source}:{method}:{str(args)}:{str(sorted(kwargs.items()))}"

        # Use SHA-256 for better collision resistance than MD5
        hash_obj = hashlib.sha256(key_str.encode("utf-8"))

        # Include a small UUID component to further reduce collision risk
        # This is deterministic based on the content
        seed = int(hash_obj.hexdigest()[:8], 16)
        collision_id = str(uuid.UUID(int=seed))

        # Return hash with collision ID
        return f"{hash_obj.hexdigest()[:32]}_{collision_id[:8]}"

    async def get(
        self, source: str, method: str, args: tuple = (), kwargs: Optional[Dict] = None
    ) -> Optional[Any]:
        """Get data from cache (memory first, then disk)."""
        kwargs = kwargs or {}
        key = self._generate_key(source, method, args, kwargs)

        # Try memory cache first
        data = await self.memory_cache.get(key)
        if data is not None:
            return data

        # Try disk cache
        data = await self.disk_cache.get(key)
        if data is not None:
            # Store in memory cache for faster access
            await self.memory_cache.set(key, data)
            return data

        return None

    async def set(
        self,
        source: str,
        method: str,
        data: Any,
        args: tuple = (),
        kwargs: Optional[Dict] = None,
        memory_ttl: Optional[int] = None,
        disk_ttl: Optional[int] = None,
    ) -> None:
        """Set data in both memory and disk cache."""
        kwargs = kwargs or {}
        key = self._generate_key(source, method, args, kwargs)

        # Store in memory cache
        await self.memory_cache.set(key, data, memory_ttl)

        # Store in disk cache
        disk_ttl = disk_ttl if disk_ttl is not None else self.disk_default_ttl
        await self.disk_cache.set(key, data, disk_ttl)

    async def delete(
        self, source: str, method: str, args: tuple = (), kwargs: Optional[Dict] = None
    ) -> bool:
        """Delete data from both caches."""
        kwargs = kwargs or {}
        key = self._generate_key(source, method, args, kwargs)

        memory_deleted = await self.memory_cache.delete(key)
        disk_deleted = await self.disk_cache.delete(key)

        return memory_deleted or disk_deleted

    async def clear(self) -> None:
        """Clear both caches."""
        await self.memory_cache.clear()
        await self.disk_cache.clear()

    async def get_stats(self) -> Dict[str, Any]:
        """Get combined cache statistics."""
        memory_stats = await self.memory_cache.get_stats()
        disk_stats = await self.disk_cache.get_stats()

        return {
            "memory_cache": memory_stats,
            "disk_cache": disk_stats,
        }
