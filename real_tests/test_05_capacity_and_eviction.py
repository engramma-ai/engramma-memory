"""
Test 05: Capacity Limits and Eviction
======================================
Tests behavior at capacity boundaries and importance-based eviction.

Real use cases:
- Long-running agents hitting 1000 pattern limit
- Smart eviction keeping important memories
- Graceful behavior under memory pressure
"""

import warnings

import numpy as np
import pytest

from conftest import make_embedding

from engramma_memory import EngrammaMemory


class TestCapacityLimits:
    """Test behavior at and beyond max_patterns."""

    def test_fill_to_capacity(self):
        mem = EngrammaMemory(dim=32, backend="local", max_patterns=100)
        for i in range(100):
            key = make_embedding(32, seed=i)
            mem.store(key=key, value=key)
        assert mem.count == 100

    def test_over_capacity_evicts(self):
        mem = EngrammaMemory(dim=32, backend="local", max_patterns=50)
        for i in range(75):
            key = make_embedding(32, seed=i + 1000)
            mem.store(key=key, value=key)
        # Count should not exceed max
        assert mem.count <= 50

    def test_capacity_warning_at_90_percent(self):
        mem = EngrammaMemory(dim=32, backend="local", max_patterns=100)
        with pytest.warns(UserWarning, match="approaching the local storage limit"):
            for i in range(91):
                key = make_embedding(32, seed=i + 2000)
                mem.store(key=key, value=key)

    def test_max_1000_hard_limit(self):
        with pytest.warns(UserWarning, match="limited to 1000"):
            mem = EngrammaMemory(dim=32, backend="local", max_patterns=2000)
        # Should be capped at 1000
        stats = mem.stats()
        assert stats["max_patterns"] == 1000


class TestImportanceEviction:
    """Test that frequently accessed patterns survive eviction."""

    def test_frequently_accessed_survives(self):
        mem = EngrammaMemory(dim=32, backend="local", max_patterns=20)

        # Store important pattern first
        important_key = make_embedding(32, seed=0)
        important_value = np.ones(32, dtype=np.float32)
        mem.store(key=important_key, value=important_value)

        # Access it many times to boost importance
        for _ in range(100):
            mem.query(important_key, top_k=1)

        # Now flood with new patterns
        for i in range(1, 50):
            k = make_embedding(32, seed=i + 3000)
            mem.store(key=k, value=k)

        # The important pattern should still be accessible
        results = mem.query(important_key, top_k=1)
        assert results[0]["score"] > 0.5

    def test_rarely_accessed_gets_evicted(self):
        mem = EngrammaMemory(dim=32, backend="local", max_patterns=10)

        # Store a rarely-used pattern
        rare_key = make_embedding(32, seed=0)
        mem.store(key=rare_key, value=rare_key)

        # Store many others and access them frequently
        for i in range(1, 10):
            k = make_embedding(32, seed=i + 4000)
            mem.store(key=k, value=k)
            for _ in range(20):
                mem.query(k, top_k=1)

        # Add more to trigger eviction
        for i in range(10, 20):
            k = make_embedding(32, seed=i + 4000)
            mem.store(key=k, value=k)

        # Rare pattern may be evicted
        results = mem.query(rare_key, top_k=1)
        # Score should be lower (either evicted or overshadowed)
        assert results[0]["score"] < 0.95


class TestStressOperations:
    """Stress tests for stability under heavy load."""

    def test_rapid_store_query_cycle(self, mem):
        """Simulate real agent usage: store then query repeatedly."""
        for i in range(200):
            key = make_embedding(128, seed=i)
            mem.store(key=key, value=key)
            if i % 10 == 0:
                results = mem.query(key, top_k=3)
                assert len(results) > 0

    def test_concurrent_operations(self, mem):
        """Store, query, compose, forget in rapid succession."""
        keys = [make_embedding(128, seed=i) for i in range(50)]
        for k in keys:
            mem.store(key=k, value=k)

        # Mix of operations
        for i in range(50):
            mem.query(keys[i % 50], top_k=3)
            if i % 5 == 0:
                mem.compose([keys[i % 50], keys[(i + 1) % 50]])
            if i % 10 == 0:
                mem.forget(keys[i % 50], strategy="decay")

        # Should still be operational
        stats = mem.stats()
        assert stats["n_stored"] == 50
