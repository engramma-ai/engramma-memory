"""Tests for the Engramma Memory Engine."""
import numpy as np
import pytest

from engramma_memory.engine import (
    ExactMemory,
    EnergyMemory,
    MultiHeadAttentionMemory,
    ConfidenceRouter,
    EngrammaEngine,
)


DIM = 64
RNG = np.random.default_rng(123)


def random_vec(dim=DIM):
    v = RNG.standard_normal(dim).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


class TestExactMemory:
    def test_store_and_retrieve_single(self):
        mem = ExactMemory(DIM, DIM, max_entries=100)
        key = random_vec()
        value = random_vec()
        mem.store(key, value)

        result, conf = mem.retrieve(key)
        assert conf > 0.9
        np.testing.assert_allclose(result, value, atol=1e-5)

    def test_retrieve_nearest(self):
        mem = ExactMemory(DIM, DIM, max_entries=100)
        keys = [random_vec() for _ in range(10)]
        values = [random_vec() for _ in range(10)]

        for k, v in zip(keys, values):
            mem.store(k, v)

        result, _ = mem.retrieve(keys[5])
        np.testing.assert_allclose(result, values[5], atol=1e-5)

    def test_eviction_when_full(self):
        mem = ExactMemory(DIM, DIM, max_entries=5)
        keys = [random_vec() for _ in range(10)]
        values = [random_vec() for _ in range(10)]

        for k, v in zip(keys, values):
            mem.store(k, v)

        assert mem.count <= 5

    def test_retrieve_top_k(self):
        mem = ExactMemory(DIM, DIM, max_entries=100)
        for _ in range(20):
            mem.store(random_vec(), random_vec())

        results = mem.retrieve_top_k(random_vec(), k=5)
        assert len(results) == 5
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)

    def test_count_property(self):
        mem = ExactMemory(DIM, DIM, max_entries=100)
        assert mem.count == 0
        mem.store(random_vec(), random_vec())
        assert mem.count == 1


class TestEnergyMemory:
    def test_store_and_retrieve(self):
        mem = EnergyMemory(DIM, DIM, max_patterns=100)
        key = random_vec()
        value = random_vec()
        mem.store(key, value)

        result, conf = mem.retrieve(key)
        assert conf > 0.0
        assert result.shape == (DIM,)

    def test_soft_generalization(self):
        mem = EnergyMemory(DIM, DIM, max_patterns=100)
        key_a = random_vec()
        key_b = random_vec()
        mem.store(key_a, key_a)
        mem.store(key_b, key_b)

        mid = (key_a + key_b) / 2
        mid /= np.linalg.norm(mid) + 1e-8
        result, _ = mem.retrieve(mid)
        assert result.shape == (DIM,)

    def test_empty_retrieve(self):
        mem = EnergyMemory(DIM, DIM, max_patterns=100)
        result, conf = mem.retrieve(random_vec())
        assert conf == 0.0
        np.testing.assert_array_equal(result, np.zeros(DIM))


class TestMultiHeadAttention:
    def test_store_and_attend(self):
        mem = MultiHeadAttentionMemory(DIM, n_heads=4, max_patterns=100)
        key = random_vec()
        value = random_vec()
        mem.store(key, value)

        result, conf = mem.attend(key)
        assert result.shape == (DIM,)
        assert conf > 0.0

    def test_compositional_attention(self):
        mem = MultiHeadAttentionMemory(DIM, n_heads=4, max_patterns=100)
        key_a = random_vec()
        key_b = random_vec()
        val_a = random_vec()
        val_b = random_vec()
        mem.store(key_a, val_a)
        mem.store(key_b, val_b)

        result, conf = mem.attend_compositional(key_a, key_b)
        assert result.shape == (DIM,)
        assert conf > 0.0

    def test_empty_attend(self):
        mem = MultiHeadAttentionMemory(DIM, n_heads=4, max_patterns=100)
        result, conf = mem.attend(random_vec())
        assert conf == 0.0
        np.testing.assert_array_equal(result, np.zeros(DIM))


class TestConfidenceRouter:
    def test_route_returns_weights(self):
        router = ConfidenceRouter()
        weights = router.route(0.9, 0.5, 0.3)
        assert weights.shape == (3,)
        assert abs(weights.sum() - 1.0) < 1e-5

    def test_update_shifts_preference(self):
        router = ConfidenceRouter()
        initial = router.success_ema.copy()
        router.update(0, 1.0)
        assert router.success_ema[0] > initial[0]


class TestEngrammaEngine:
    def test_store_and_retrieve(self):
        engine = EngrammaEngine(dim=DIM, max_patterns=100)
        key = random_vec()
        value = random_vec()
        engine.store(key, value)

        result = engine.retrieve(key)
        assert result.shape == (DIM,)
        cos_sim = np.dot(result, value) / (np.linalg.norm(result) * np.linalg.norm(value) + 1e-8)
        assert cos_sim > 0.5

    def test_compose_two_keys(self):
        engine = EngrammaEngine(dim=DIM, max_patterns=100)
        key_a = random_vec()
        key_b = random_vec()
        engine.store(key_a, key_a)
        engine.store(key_b, key_b)

        result = engine.compose([key_a, key_b])
        assert result.shape == (DIM,)
        assert np.linalg.norm(result) > 0

    def test_compose_three_keys(self):
        engine = EngrammaEngine(dim=DIM, max_patterns=100)
        keys = [random_vec() for _ in range(3)]
        for k in keys:
            engine.store(k, k)

        result = engine.compose(keys)
        assert result.shape == (DIM,)

    def test_compose_single_key_fallback(self):
        engine = EngrammaEngine(dim=DIM, max_patterns=100)
        key = random_vec()
        engine.store(key, key)
        result = engine.compose([key])
        assert result.shape == (DIM,)

    def test_forget_immediate(self):
        engine = EngrammaEngine(dim=DIM, max_patterns=100)
        key = random_vec()
        engine.store(key, key)
        assert engine.exact.count == 1
        engine.forget(key, strategy="immediate")
        assert engine.exact.count == 0

    def test_forget_decay(self):
        engine = EngrammaEngine(dim=DIM, max_patterns=100)
        key = random_vec()
        engine.store(key, key)
        initial_count = engine.exact.access_count[0]
        engine.forget(key, strategy="decay")
        assert engine.exact.access_count[0] < initial_count

    def test_get_stats(self):
        engine = EngrammaEngine(dim=DIM, max_patterns=100)
        engine.store(random_vec(), random_vec())
        stats = engine.get_stats()
        assert stats["n_stored"] == 1
        assert stats["exact_count"] == 1
        assert stats["max_patterns"] == 100
        assert stats["dim"] == DIM

    def test_retrieve_top_k(self):
        engine = EngrammaEngine(dim=DIM, max_patterns=100)
        for _ in range(10):
            engine.store(random_vec(), random_vec())
        results = engine.retrieve_top_k(random_vec(), k=3)
        assert len(results) == 3
