"""
Test 01: Core Operations - Store, Query, Retrieve, Stats
=========================================================
Tests the fundamental CRUD operations of EngrammaMemory with the local backend.
Real use cases: storing embeddings, retrieving similar patterns, checking stats.
"""

import numpy as np
import pytest

from conftest import make_embedding, text_to_embedding

from engramma_memory import EngrammaMemory


class TestStoreAndCount:
    """Test basic storage and counting."""

    def test_store_single_pattern(self, mem):
        key = make_embedding(128, seed=1)
        value = make_embedding(128, seed=2)
        mem.store(key=key, value=value)
        assert mem.count == 1

    def test_store_multiple_patterns(self, mem):
        for i in range(50):
            key = make_embedding(128, seed=i)
            value = make_embedding(128, seed=i + 1000)
            mem.store(key=key, value=value)
        assert mem.count == 50

    def test_store_with_list_input(self, mem):
        key = [0.1] * 128
        value = [0.2] * 128
        mem.store(key=key, value=value)
        assert mem.count == 1

    def test_store_duplicate_key_updates(self, mem):
        key = make_embedding(128, seed=1)
        value1 = make_embedding(128, seed=2)
        value2 = make_embedding(128, seed=3)
        mem.store(key=key, value=value1)
        mem.store(key=key, value=value2)
        # Duplicate key within threshold (0.1) should update, not add
        assert mem.count == 1

    def test_empty_memory_count(self, mem):
        assert mem.count == 0


class TestQuery:
    """Test query/retrieval operations."""

    def test_query_exact_match(self, mem):
        key = make_embedding(128, seed=10)
        mem.store(key=key, value=key)
        results = mem.query(key, top_k=1)
        assert len(results) == 1
        assert results[0]["score"] > 0.9

    def test_query_top_k(self, mem):
        for i in range(20):
            key = make_embedding(128, seed=i)
            mem.store(key=key, value=key)

        query = make_embedding(128, seed=0)
        results = mem.query(query, top_k=5)
        assert len(results) == 5
        # First result should have highest score
        assert results[0]["score"] >= results[1]["score"]

    def test_query_returns_correct_value(self, mem):
        key = make_embedding(128, seed=42)
        value = np.ones(128, dtype=np.float32) * 0.5
        mem.store(key=key, value=value)
        results = mem.query(key, top_k=1)
        np.testing.assert_allclose(results[0]["value"], value, atol=1e-5)

    def test_query_empty_memory(self, mem):
        query = make_embedding(128, seed=1)
        results = mem.query(query, top_k=1)
        assert len(results) == 1
        assert results[0]["score"] == 0.0

    def test_query_similar_vectors(self, mem):
        base = make_embedding(128, seed=1)
        noise = np.random.default_rng(99).standard_normal(128).astype(np.float32) * 0.05
        similar = base + noise
        similar = similar / (np.linalg.norm(similar) + 1e-8)

        mem.store(key=base, value=base)
        results = mem.query(similar, top_k=1)
        assert results[0]["score"] > 0.5


class TestRetrieve:
    """Test smart routing retrieval."""

    def test_retrieve_exact_pattern(self, mem):
        key = make_embedding(128, seed=5)
        value = make_embedding(128, seed=6)
        mem.store(key=key, value=value)
        result = mem.retrieve(key)
        # Should retrieve something close to the stored value
        similarity = float(np.dot(result, value) / (np.linalg.norm(result) * np.linalg.norm(value) + 1e-8))
        assert similarity > 0.5

    def test_retrieve_with_multiple_patterns(self, mem):
        keys = [make_embedding(128, seed=i) for i in range(10)]
        values = [make_embedding(128, seed=i + 100) for i in range(10)]
        for k, v in zip(keys, values):
            mem.store(key=k, value=v)

        result = mem.retrieve(keys[3])
        assert result.shape == (128,)
        assert not np.allclose(result, 0)

    def test_retrieve_empty_memory(self, mem):
        query = make_embedding(128, seed=1)
        result = mem.retrieve(query)
        assert result.shape == (128,)


class TestStats:
    """Test stats reporting."""

    def test_stats_empty(self, mem):
        stats = mem.stats()
        assert stats["n_stored"] == 0
        assert stats["exact_count"] == 0
        assert stats["dim"] == 128
        assert stats["max_patterns"] == 1000

    def test_stats_after_storing(self, mem):
        for i in range(25):
            key = make_embedding(128, seed=i)
            mem.store(key=key, value=key)
        stats = mem.stats()
        assert stats["n_stored"] == 25
        assert stats["exact_count"] == 25
        assert stats["usage_ratio"] == pytest.approx(0.025)

    def test_repr(self, mem):
        repr_str = repr(mem)
        assert "EngrammaMemory" in repr_str
        assert "dim=128" in repr_str
        assert "backend='local'" in repr_str


class TestInitialization:
    """Test various initialization configurations."""

    def test_default_backend_is_local(self):
        mem = EngrammaMemory(dim=64)
        assert mem.backend_name == "local"

    def test_custom_dimensions(self):
        for dim in [32, 64, 128, 256, 512]:
            mem = EngrammaMemory(dim=dim)
            assert mem.dim == dim

    def test_cloud_without_api_key_raises(self):
        with pytest.raises(ValueError, match="Cloud backend requires an API key"):
            EngrammaMemory(dim=128, backend="cloud")

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            EngrammaMemory(dim=128, backend="nonexistent")

    def test_max_patterns_over_limit_warns(self):
        with pytest.warns(UserWarning, match="limited to 1000"):
            EngrammaMemory(dim=128, max_patterns=5000)
