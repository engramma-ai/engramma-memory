"""
Test 04: Engine Pathways - Exact, Energy, Multi-Head Attention
===============================================================
Tests the three internal pathways directly and their routing behavior.

Real use cases:
- Exact recall: looking up a known fact
- Energy-based: generalizing from noisy/partial inputs
- Multi-head: blending related concepts for creative output
"""

import numpy as np
import pytest

from conftest import make_embedding

from engramma_memory.engine import (
    ConfidenceRouter,
    EnergyMemory,
    EngrammaEngine,
    ExactMemory,
    MultiHeadAttentionMemory,
)


class TestExactMemory:
    """Test the kNN exact memory pathway."""

    def test_perfect_recall(self):
        exact = ExactMemory(key_dim=64, value_dim=64)
        key = make_embedding(64, seed=1)
        value = make_embedding(64, seed=2)
        exact.store(key, value)

        result, conf = exact.retrieve(key)
        np.testing.assert_allclose(result, value, atol=1e-5)
        assert conf > 0.9

    def test_nearest_neighbor(self):
        exact = ExactMemory(key_dim=64, value_dim=64)
        keys = [make_embedding(64, seed=i) for i in range(10)]
        values = [make_embedding(64, seed=i + 100) for i in range(10)]

        for k, v in zip(keys, values):
            exact.store(k, v)

        # Query with slight noise added to key[3]
        noisy = keys[3] + np.random.default_rng(42).standard_normal(64).astype(np.float32) * 0.01
        result, conf = exact.retrieve(noisy)
        np.testing.assert_allclose(result, values[3], atol=0.1)

    def test_retrieve_top_k(self):
        exact = ExactMemory(key_dim=64, value_dim=64)
        for i in range(20):
            k = make_embedding(64, seed=i)
            exact.store(k, k)

        query = make_embedding(64, seed=0)
        results = exact.retrieve_top_k(query, k=5)
        assert len(results) == 5
        # Scores should be descending
        scores = [r[1] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_importance_eviction(self):
        exact = ExactMemory(key_dim=32, value_dim=32, max_entries=5)

        # Store 5 patterns
        keys = [make_embedding(32, seed=i) for i in range(5)]
        for k in keys:
            exact.store(k, k)

        # Access key[0] many times to boost importance
        for _ in range(50):
            exact.retrieve(keys[0])

        # Store a 6th pattern - should evict least important
        new_key = make_embedding(32, seed=99)
        exact.store(new_key, new_key)

        # key[0] should still be there (high importance)
        result, conf = exact.retrieve(keys[0])
        assert conf > 0.8

    def test_count(self):
        exact = ExactMemory(key_dim=32, value_dim=32)
        assert exact.count == 0
        exact.store(make_embedding(32, seed=1), make_embedding(32, seed=1))
        assert exact.count == 1


class TestEnergyMemory:
    """Test the energy-based (Hopfield) memory pathway."""

    def test_store_and_retrieve(self):
        energy = EnergyMemory(key_dim=64, value_dim=64)
        key = make_embedding(64, seed=1)
        value = make_embedding(64, seed=2)
        energy.store(key, value)

        result, conf = energy.retrieve(key)
        assert result.shape == (64,)
        assert conf > 0.5

    def test_generalization_from_noisy_input(self):
        """Energy memory should generalize even with noisy queries."""
        energy = EnergyMemory(key_dim=64, value_dim=64, beta=4.0)
        key = make_embedding(64, seed=10)
        value = make_embedding(64, seed=20)
        energy.store(key, value)

        # Add significant noise
        noisy_query = key + np.random.default_rng(77).standard_normal(64).astype(np.float32) * 0.3
        result, conf = energy.retrieve(noisy_query)

        # Should still produce a reasonable result
        sim = np.dot(result, value) / (np.linalg.norm(result) * np.linalg.norm(value) + 1e-8)
        assert sim > 0.3

    def test_soft_blending_multiple_patterns(self):
        """With multiple patterns, energy memory should softly blend."""
        energy = EnergyMemory(key_dim=64, value_dim=64, beta=2.0)

        # Store two patterns
        key_a = make_embedding(64, seed=1)
        key_b = make_embedding(64, seed=2)
        val_a = np.ones(64, dtype=np.float32)
        val_b = -np.ones(64, dtype=np.float32)

        energy.store(key_a, val_a)
        energy.store(key_b, val_b)

        # Query midpoint
        mid = (key_a + key_b) / 2.0
        result, _ = energy.retrieve(mid)
        # Should be a blend, not pure +1 or -1
        assert abs(np.mean(result)) < 0.9

    def test_temperature_effect(self):
        """Higher beta should make retrieval sharper."""
        key = make_embedding(64, seed=1)
        val = make_embedding(64, seed=2)

        low_temp = EnergyMemory(key_dim=64, value_dim=64, beta=1.0)
        high_temp = EnergyMemory(key_dim=64, value_dim=64, beta=10.0)

        for e in [low_temp, high_temp]:
            e.store(key, val)
            e.store(make_embedding(64, seed=3), make_embedding(64, seed=4))

        _, conf_low = low_temp.retrieve(key)
        _, conf_high = high_temp.retrieve(key)

        # Higher beta = more confident (sharper distribution)
        assert conf_high > conf_low


class TestMultiHeadAttention:
    """Test the multi-head attention pathway."""

    def test_attend_single_pattern(self):
        mha = MultiHeadAttentionMemory(dim=64, n_heads=4)
        key = make_embedding(64, seed=1)
        value = make_embedding(64, seed=2)
        mha.store(key, value)

        result, conf = mha.attend(key)
        assert result.shape == (64,)
        assert conf > 0.0

    def test_attend_compositional(self):
        """Split-query attention for native composition."""
        mha = MultiHeadAttentionMemory(dim=64, n_heads=4)

        key_a = make_embedding(64, seed=1)
        key_b = make_embedding(64, seed=2)
        val_a = make_embedding(64, seed=10)
        val_b = make_embedding(64, seed=20)

        mha.store(key_a, val_a)
        mha.store(key_b, val_b)

        result, conf = mha.attend_compositional(key_a, key_b)
        assert result.shape == (64,)
        assert conf > 0.0
        # Result should not be identical to either value
        assert not np.allclose(result, val_a, atol=0.1)

    def test_multiple_heads_different_temperatures(self):
        mha = MultiHeadAttentionMemory(dim=32, n_heads=4)
        # Temperatures should range from 2.0 to 8.0
        np.testing.assert_allclose(mha.temperatures, [2.0, 4.0, 6.0, 8.0])

    def test_head_weights_sum_to_one(self):
        mha = MultiHeadAttentionMemory(dim=32, n_heads=4)
        assert np.isclose(mha.head_weights.sum(), 1.0)


class TestConfidenceRouter:
    """Test the confidence-based routing system."""

    def test_route_initial_bias_to_exact(self):
        router = ConfidenceRouter()
        # Initial EMA: [0.7, 0.5, 0.5] - biased toward exact
        weights = router.route(0.9, 0.5, 0.5)
        assert weights[0] > weights[1]
        assert weights[0] > weights[2]

    def test_route_sums_to_one(self):
        router = ConfidenceRouter()
        weights = router.route(0.3, 0.6, 0.8)
        assert np.isclose(weights.sum(), 1.0)

    def test_update_shifts_preference(self):
        router = ConfidenceRouter()
        original = router.success_ema.copy()

        # Repeatedly reward pathway 2 (attention)
        for _ in range(20):
            router.update(2, 0.95)

        # EMA for pathway 2 should increase
        assert router.success_ema[2] > original[2]

    def test_high_confidence_routing(self):
        router = ConfidenceRouter()
        # Very high exact confidence should dominate
        weights = router.route(0.99, 0.1, 0.1)
        assert weights[0] > 0.7


class TestEngineIntegration:
    """Test the full engine with all pathways working together."""

    def test_engine_stores_in_all_pathways(self):
        engine = EngrammaEngine(dim=64, max_patterns=100)
        key = make_embedding(64, seed=1)
        value = make_embedding(64, seed=2)
        engine.store(key, value)

        assert engine.exact.count == 1
        assert engine.energy.n == 1
        assert engine.attention.n == 1

    def test_engine_routing_exact_match(self):
        """High-confidence exact matches bypass other pathways."""
        engine = EngrammaEngine(dim=64)
        key = make_embedding(64, seed=1)
        value = make_embedding(64, seed=2)
        engine.store(key, value)

        result = engine.retrieve(key)
        # Should be very close to value (exact pathway dominates)
        sim = np.dot(result, value) / (np.linalg.norm(result) * np.linalg.norm(value) + 1e-8)
        assert sim > 0.9

    def test_engine_routing_uncertain_query(self):
        """Uncertain queries should blend pathways."""
        engine = EngrammaEngine(dim=64)
        for i in range(20):
            k = make_embedding(64, seed=i)
            engine.store(k, k)

        # Query with a vector far from stored patterns
        far_query = make_embedding(64, seed=9999)
        result = engine.retrieve(far_query)
        # Should still produce a finite result
        assert np.isfinite(result).all()

    def test_engine_composition(self):
        engine = EngrammaEngine(dim=64)
        keys = [make_embedding(64, seed=i) for i in range(5)]
        for k in keys:
            engine.store(k, k)

        composed = engine.compose([keys[0], keys[1]])
        assert composed.shape == (64,)
        assert not np.allclose(composed, 0)
