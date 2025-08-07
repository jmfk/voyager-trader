# VOYAGER Skill Caching - Quick Reference

## 🚀 Quick Start

```python
from voyager_trader.skills import VoyagerSkillLibrary, CacheConfig

# Basic configuration with caching enabled
config = {
    "skill_library_path": "skills",
    "enable_result_cache": True,
    "enable_compilation_cache": True,
    "max_execution_cache_size": 1000,  # Adjust based on memory
}

library = VoyagerSkillLibrary(config)
```

## 📊 Performance Monitoring

```python
# Quick performance check
stats = library.get_comprehensive_stats()
print(f"Cache hit rate: {stats['performance']['cache_hit_rate']:.1%}")

# Get optimization recommendations
for rec in library.get_performance_recommendations():
    print(f"💡 {rec}")
```

## 🔧 Cache Management

```python
# Clear all caches
library.clear_all_caches()

# Evict expired entries
library.evict_expired_entries()

# Refresh caches (check for file changes)
library.refresh_caches()
```

## ⚙️ Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_execution_cache_size` | 1000 | Max cached execution results |
| `max_metadata_cache_size` | 500 | Max cached metadata entries |
| `execution_cache_ttl_hours` | 24 | Cache TTL in hours |
| `enable_result_cache` | True | Enable execution result caching |
| `enable_compilation_cache` | True | Enable code compilation caching |

## 🔍 Debugging

```python
# Enable debug logging
import logging
logging.getLogger('voyager_trader.skills').setLevel(logging.DEBUG)

# Check cache statistics
cache_stats = library.executor.get_cache_stats()
print(f"Execution cache utilization: {cache_stats['execution_cache']['size']}")
```

## 🎯 Performance Tips

1. **Monitor hit rates**: Aim for >70% cache hit rate
2. **Consistent inputs**: Slight input variations create different cache keys
3. **Batch operations**: Group similar operations for better cache effectiveness
4. **Regular maintenance**: Run `evict_expired_entries()` periodically

## 🚨 Common Issues

| Issue | Solution |
|-------|----------|
| Low cache hit rate | Check input consistency, increase TTL |
| High memory usage | Reduce cache sizes, enable aggressive TTL |
| Slow despite caching | Verify caching enabled, check cache stats |

## 📈 Benchmark Template

```python
import time

def benchmark_caching():
    inputs = {"value": 42}

    # First execution (uncached)
    start = time.time()
    result1 = library.execute_skill(skill_id, inputs)
    uncached_time = time.time() - start

    # Second execution (cached)
    start = time.time()
    result2 = library.execute_skill(skill_id, inputs)
    cached_time = time.time() - start

    speedup = uncached_time / cached_time
    print(f"Speedup: {speedup:.1f}x")

benchmark_caching()
```
