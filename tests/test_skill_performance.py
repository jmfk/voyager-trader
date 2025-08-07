"""
Performance and stress tests for the skill caching system.
"""

import concurrent.futures
import tempfile
import time
from decimal import Decimal
from pathlib import Path
from threading import Thread

import pytest

from voyager_trader.models.learning import Skill, SkillExecutionResult
from voyager_trader.models.types import SkillCategory, SkillComplexity
from voyager_trader.skills import (
    CacheConfig,
    LRUCache,
    SkillExecutionCache,
    SkillExecutor,
    SkillLibrarian,
    VoyagerSkillLibrary,
)


class TestCachePerformance:
    """Performance tests for caching components."""

    def test_lru_cache_performance_large_dataset(self):
        """Test LRU cache performance with large dataset."""
        cache = LRUCache(max_size=1000, ttl_hours=24)

        # Measure insertion performance
        start_time = time.time()
        for i in range(10000):
            cache.put(f"key_{i}", f"value_{i}")
        insertion_time = time.time() - start_time

        # Should handle 10k insertions reasonably quickly
        assert insertion_time < 5.0  # 5 seconds threshold

        # Verify LRU eviction worked correctly
        assert cache.get_stats()["size"] == 1000

        # Measure retrieval performance
        start_time = time.time()
        for i in range(9000, 10000):  # Get last 1000 items
            result = cache.get(f"key_{i}")
            assert result == f"value_{i}"
        retrieval_time = time.time() - start_time

        # Should retrieve 1000 items quickly
        assert retrieval_time < 1.0  # 1 second threshold

    def test_cache_concurrent_access(self):
        """Test cache thread safety under concurrent access."""
        cache = LRUCache(max_size=100, ttl_hours=24)
        results = []

        def worker(thread_id: int) -> None:
            """Worker function for concurrent testing."""
            thread_results = []
            for i in range(100):
                key = f"thread_{thread_id}_key_{i}"
                value = f"thread_{thread_id}_value_{i}"

                # Put and immediately get
                cache.put(key, value)
                retrieved = cache.get(key)
                thread_results.append(retrieved == value)

            results.extend(thread_results)

        # Create multiple threads
        threads = []
        for i in range(5):
            thread = Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify all operations succeeded
        success_rate = sum(results) / len(results)
        assert success_rate > 0.95  # Allow for some race conditions

    def test_skill_execution_cache_stress(self):
        """Stress test the skill execution cache."""
        config = CacheConfig(
            max_execution_cache_size=500,
            max_metadata_cache_size=200,
        )
        cache = SkillExecutionCache(config)

        # Create test skills
        skills = []
        for i in range(10):
            skill = Skill(
                name=f"test_skill_{i}",
                description=f"Test skill {i}",
                category=SkillCategory.TECHNICAL_ANALYSIS,
                complexity=SkillComplexity.BASIC,
                code=f"result = {{'skill_id': {i}, 'value': inputs.get('value', 0)}}",
                input_schema={"type": "object"},
                output_schema={"type": "object"},
            )
            skills.append(skill)

        # Stress test with many cache operations
        start_time = time.time()
        cache_hits = 0
        total_operations = 0

        for iteration in range(100):
            for skill_idx, skill in enumerate(skills):
                for value in range(10):
                    inputs = {"value": value}
                    result = (
                        SkillExecutionResult.SUCCESS,
                        {"output": skill_idx * value},
                        {"metadata": "test"},
                    )

                    # Check cache first
                    cached = cache.get_execution_result(skill, inputs, None)
                    if cached:
                        cache_hits += 1
                    else:
                        # Cache the result
                        cache.cache_execution_result(skill, inputs, None, result)

                    total_operations += 1

        execution_time = time.time() - start_time
        cache_hit_rate = cache_hits / total_operations if total_operations > 0 else 0

        # Performance assertions
        assert execution_time < 10.0  # Should complete within 10 seconds
        assert cache_hit_rate > 0.8  # Should have high cache hit rate after warmup

        # Verify cache stats
        stats = cache.get_cache_stats()
        assert stats["execution_cache"]["size"] > 0


class TestSkillExecutorPerformance:
    """Performance tests for SkillExecutor with caching."""

    def setup_method(self):
        """Set up test executor."""
        self.cache_config = CacheConfig(
            max_execution_cache_size=200,
            enable_result_cache=True,
            enable_compilation_cache=True,
        )
        self.executor = SkillExecutor(
            timeout_seconds=10,
            cache_config=self.cache_config,
        )
        self.test_skill = Skill(
            name="performance_test_skill",
            description="Skill for performance testing",
            category=SkillCategory.TECHNICAL_ANALYSIS,
            complexity=SkillComplexity.BASIC,
            code="""
import time
# Simulate some computation
time.sleep(0.01)  # 10ms delay
result = {
    'computed_value': sum(range(inputs.get('range_size', 100))),
    'input_value': inputs.get('value', 0) * 2
}
""",
            input_schema={"type": "object"},
            output_schema={"type": "object"},
        )

    def test_execution_performance_with_caching(self):
        """Test execution performance improvement with caching."""
        inputs = {"value": 42, "range_size": 1000}

        # First execution (not cached)
        start_time = time.time()
        result1, output1, metadata1 = self.executor.execute_skill(
            self.test_skill, inputs
        )
        first_execution_time = time.time() - start_time

        assert result1 == SkillExecutionResult.SUCCESS
        assert metadata1["cached"] is False

        # Second execution (should be cached)
        start_time = time.time()
        result2, output2, metadata2 = self.executor.execute_skill(
            self.test_skill, inputs
        )
        second_execution_time = time.time() - start_time

        assert result2 == SkillExecutionResult.SUCCESS
        assert metadata2["cached"] is True
        assert output1 == output2  # Results should be identical

        # Cached execution should be significantly faster
        speedup_ratio = first_execution_time / second_execution_time
        assert speedup_ratio > 2.0  # At least 2x faster

    def test_cache_effectiveness_over_time(self):
        """Test cache effectiveness over multiple executions."""
        execution_times = []
        cache_hit_count = 0

        # Execute same skill multiple times with same inputs
        inputs = {"value": 10, "range_size": 500}

        for i in range(10):
            start_time = time.time()
            result, output, metadata = self.executor.execute_skill(
                self.test_skill, inputs
            )
            execution_time = time.time() - start_time
            execution_times.append(execution_time)

            if metadata.get("cached", False):
                cache_hit_count += 1

            assert result == SkillExecutionResult.SUCCESS

        # First execution should be slowest
        assert execution_times[0] > execution_times[-1]

        # Should have cache hits after first execution
        assert cache_hit_count >= 8  # Allow for some variance


class TestSkillLibrarianPerformance:
    """Performance tests for SkillLibrarian with caching."""

    def setup_method(self):
        """Set up test librarian."""
        self.temp_dir = tempfile.mkdtemp()
        self.storage_path = Path(self.temp_dir)
        self.cache_config = CacheConfig(
            max_metadata_cache_size=1000,
            metadata_cache_ttl_hours=1,
        )
        self.librarian = SkillLibrarian(self.storage_path, self.cache_config)

    def test_bulk_skill_storage_performance(self):
        """Test performance of storing many skills."""
        skills = []
        for i in range(100):
            skill = Skill(
                name=f"bulk_skill_{i}",
                description=f"Bulk test skill {i}",
                category=SkillCategory.TECHNICAL_ANALYSIS,
                complexity=SkillComplexity.BASIC,
                code=f"result = {{'skill_number': {i}}}",
                input_schema={"type": "object"},
                output_schema={"type": "object"},
            )
            skills.append(skill)

        # Measure bulk storage performance
        start_time = time.time()
        for skill in skills:
            success = self.librarian.store_skill(skill)
            assert success is True
        storage_time = time.time() - start_time

        # Should store 100 skills within reasonable time
        assert storage_time < 5.0  # 5 seconds threshold

        # Verify all skills were stored
        library_stats = self.librarian.get_library_stats()
        assert library_stats["total_skills"] == 100

    def test_skill_retrieval_performance(self):
        """Test performance of retrieving skills."""
        # Store test skills
        skills = []
        skill_ids = []
        for i in range(50):
            skill = Skill(
                name=f"retrieval_test_skill_{i}",
                description=f"Retrieval test skill {i}",
                category=SkillCategory.TECHNICAL_ANALYSIS,
                complexity=SkillComplexity.BASIC,
                code=f"result = {{'skill_number': {i}}}",
                input_schema={"type": "object"},
                output_schema={"type": "object"},
            )
            skills.append(skill)
            skill_ids.append(skill.id)
            self.librarian.store_skill(skill)

        # Measure retrieval performance
        start_time = time.time()
        for skill_id in skill_ids:
            retrieved_skill = self.librarian.retrieve_skill(skill_id)
            assert retrieved_skill is not None
        retrieval_time = time.time() - start_time

        # Should retrieve 50 skills quickly
        assert retrieval_time < 1.0  # 1 second threshold

        # Test repeated retrieval (should use cache)
        start_time = time.time()
        for skill_id in skill_ids:
            retrieved_skill = self.librarian.retrieve_skill(skill_id)
            assert retrieved_skill is not None
        second_retrieval_time = time.time() - start_time

        # Second retrieval should be faster due to caching
        assert second_retrieval_time < retrieval_time

    def test_search_performance_with_large_dataset(self):
        """Test search performance with large skill dataset."""
        # Create skills with various categories and tags
        categories = list(SkillCategory)
        complexities = list(SkillComplexity)

        for i in range(200):
            skill = Skill(
                name=f"search_test_skill_{i}",
                description=f"Search test skill {i}",
                category=categories[i % len(categories)],
                complexity=complexities[i % len(complexities)],
                code=f"result = {{'skill_number': {i}}}",
                input_schema={"type": "object"},
                output_schema={"type": "object"},
                tags=[f"tag_{i % 10}", f"group_{i // 50}"],
            )
            self.librarian.store_skill(skill)

        # Measure search performance
        start_time = time.time()

        # Test various search queries
        results1 = self.librarian.search_skills(
            category=SkillCategory.TECHNICAL_ANALYSIS
        )
        results2 = self.librarian.search_skills(complexity=SkillComplexity.BASIC)
        results3 = self.librarian.search_skills(tags=["tag_0"])
        results4 = self.librarian.search_skills(name_pattern="skill_1")

        search_time = time.time() - start_time

        # Should complete searches within reasonable time
        assert search_time < 2.0  # 2 seconds threshold

        # Verify search results
        assert len(results1) > 0
        assert len(results2) > 0
        assert len(results3) > 0
        assert len(results4) > 0


class TestVoyagerSkillLibraryPerformance:
    """End-to-end performance tests for VoyagerSkillLibrary."""

    def setup_method(self):
        """Set up test library."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = {
            "skill_library_path": self.temp_dir,
            "max_execution_cache_size": 500,
            "max_metadata_cache_size": 300,
            "enable_result_cache": True,
            "enable_compilation_cache": True,
            "execution_timeout": 5,
        }
        self.library = VoyagerSkillLibrary(self.config)

    def test_end_to_end_performance_workflow(self):
        """Test complete workflow performance from skill creation to execution."""
        # Create multiple test skills
        skills = []
        for i in range(20):
            skill = Skill(
                name=f"workflow_skill_{i}",
                description=f"Workflow test skill {i}",
                category=SkillCategory.TECHNICAL_ANALYSIS,
                complexity=SkillComplexity.BASIC,
                code=f"""
import time
time.sleep(0.001)  # 1ms delay
result = {{
    'skill_id': {i},
    'input_multiplied': inputs.get('value', 1) * {i + 1},
    'computed_sum': sum(range(inputs.get('range', 10)))
}}
""",
                input_schema={"type": "object"},
                output_schema={"type": "object"},
            )
            skills.append(skill)

        # Measure skill addition performance
        start_time = time.time()
        for skill in skills:
            success = self.library.add_skill(skill, validate=True)
            assert success is True
        addition_time = time.time() - start_time

        assert addition_time < 3.0  # 3 seconds for adding 20 skills

        # Measure skill execution performance (with caching)
        skill_ids = [skill.id for skill in skills]
        inputs = {"value": 5, "range": 20}

        # First round of executions (not cached)
        start_time = time.time()
        first_round_results = []
        for skill_id in skill_ids:
            result, output, metadata = self.library.execute_skill(skill_id, inputs)
            first_round_results.append((result, output, metadata))
            assert result == SkillExecutionResult.SUCCESS
        first_round_time = time.time() - start_time

        # Second round of executions (should be cached)
        start_time = time.time()
        second_round_results = []
        for skill_id in skill_ids:
            result, output, metadata = self.library.execute_skill(skill_id, inputs)
            second_round_results.append((result, output, metadata))
            assert result == SkillExecutionResult.SUCCESS
            assert metadata.get("cached", False) is True
        second_round_time = time.time() - start_time

        # Cached executions should be significantly faster
        speedup_ratio = first_round_time / second_round_time
        assert speedup_ratio > 3.0  # At least 3x faster with caching

        # Verify results consistency
        for i, ((r1, o1, m1), (r2, o2, m2)) in enumerate(
            zip(first_round_results, second_round_results)
        ):
            assert r1 == r2 == SkillExecutionResult.SUCCESS
            assert o1 == o2  # Outputs should be identical

    def test_system_performance_under_load(self):
        """Test system performance under concurrent load."""
        # Create test skill
        skill = Skill(
            name="load_test_skill",
            description="Skill for load testing",
            category=SkillCategory.TECHNICAL_ANALYSIS,
            complexity=SkillComplexity.BASIC,
            code="""
import time
time.sleep(0.005)  # 5ms delay to simulate work
result = {
    'input_squared': inputs.get('value', 0) ** 2,
    'timestamp': str(time.time())
}
""",
            input_schema={"type": "object"},
            output_schema={"type": "object"},
        )

        self.library.add_skill(skill)

        def execute_skill_multiple_times(skill_id: str, thread_id: int) -> list:
            """Execute skill multiple times from a thread."""
            results = []
            for i in range(10):
                inputs = {"value": thread_id * 10 + i}
                start_time = time.time()
                result, output, metadata = self.library.execute_skill(skill_id, inputs)
                execution_time = time.time() - start_time

                results.append(
                    {
                        "result": result,
                        "execution_time": execution_time,
                        "cached": metadata.get("cached", False),
                        "thread_id": thread_id,
                        "iteration": i,
                    }
                )
            return results

        # Execute with multiple threads concurrently
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for thread_id in range(5):
                future = executor.submit(
                    execute_skill_multiple_times, skill.id, thread_id
                )
                futures.append(future)

            # Collect all results
            all_results = []
            for future in concurrent.futures.as_completed(futures):
                thread_results = future.result()
                all_results.extend(thread_results)

        total_time = time.time() - start_time

        # Verify all executions succeeded
        successful_executions = [
            r for r in all_results if r["result"] == SkillExecutionResult.SUCCESS
        ]
        assert len(successful_executions) == 50  # 5 threads * 10 executions

        # Check performance metrics
        cache_hit_rate = sum(1 for r in all_results if r["cached"]) / len(all_results)
        average_execution_time = sum(r["execution_time"] for r in all_results) / len(
            all_results
        )

        # Performance assertions
        assert total_time < 15.0  # Should complete within 15 seconds
        assert cache_hit_rate > 0.3  # Should have some cache hits
        assert average_execution_time < 0.1  # Average execution should be fast

        # Verify system stats are reasonable
        comprehensive_stats = self.library.get_comprehensive_stats()
        performance_stats = comprehensive_stats["performance"]

        assert performance_stats["total_executions"] == 50
        assert performance_stats["success_rate"] == 1.0
        assert performance_stats["cache_hit_rate"] >= cache_hit_rate


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
