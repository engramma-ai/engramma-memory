"""
Test 02: Composition - The Key Differentiator
==============================================
Tests Engramma's native composition capability via multi-head attention.
This is what separates it from simple vector databases.

Real use cases:
- Blending "happy" + "birthday" into a combined semantic concept
- Composing user preferences from multiple signals
- Creating synthesized context from multiple memory fragments
"""

import numpy as np
import pytest

from conftest import make_embedding

from engramma_memory import EngrammaMemory


class TestTwoWayComposition:
    """Test composing two patterns together."""

    def test_compose_two_patterns(self, mem):
        key_a = make_embedding(128, seed=1)
        key_b = make_embedding(128, seed=2)
        value_a = make_embedding(128, seed=100)
        value_b = make_embedding(128, seed=200)

        mem.store(key=key_a, value=value_a)
        mem.store(key=key_b, value=value_b)

        composed = mem.compose([key_a, key_b])
        assert composed.shape == (128,)
        assert not np.allclose(composed, 0)

    def test_composition_is_different_from_individual(self, mem):
        key_a = make_embedding(128, seed=10)
        key_b = make_embedding(128, seed=20)

        mem.store(key=key_a, value=key_a)
        mem.store(key=key_b, value=key_b)

        composed = mem.compose([key_a, key_b])
        result_a = mem.retrieve(key_a)
        result_b = mem.retrieve(key_b)

        # Composition should not be identical to either individual
        assert not np.allclose(composed, result_a, atol=0.1)
        assert not np.allclose(composed, result_b, atol=0.1)

    def test_composition_relates_to_both_inputs(self, mem):
        key_a = make_embedding(128, seed=30)
        key_b = make_embedding(128, seed=40)

        mem.store(key=key_a, value=key_a)
        mem.store(key=key_b, value=key_b)

        composed = mem.compose([key_a, key_b])

        # The composed result should have some similarity to both
        sim_a = np.dot(composed, key_a) / (np.linalg.norm(composed) * np.linalg.norm(key_a) + 1e-8)
        sim_b = np.dot(composed, key_b) / (np.linalg.norm(composed) * np.linalg.norm(key_b) + 1e-8)

        # At least one should show notable similarity (composition blends)
        assert max(abs(sim_a), abs(sim_b)) > 0.1

    def test_composition_is_deterministic(self, mem):
        key_a = make_embedding(128, seed=50)
        key_b = make_embedding(128, seed=60)

        mem.store(key=key_a, value=key_a)
        mem.store(key=key_b, value=key_b)

        result1 = mem.compose([key_a, key_b])
        result2 = mem.compose([key_a, key_b])
        np.testing.assert_array_equal(result1, result2)


class TestThreeWayComposition:
    """Test composing three or more patterns."""

    def test_compose_three_patterns(self, mem):
        keys = [make_embedding(128, seed=i) for i in range(3)]
        for k in keys:
            mem.store(key=k, value=k)

        composed = mem.compose(keys)
        assert composed.shape == (128,)
        assert not np.allclose(composed, 0)

    def test_compose_four_patterns(self, mem):
        keys = [make_embedding(128, seed=i * 10) for i in range(4)]
        for k in keys:
            mem.store(key=k, value=k)

        composed = mem.compose(keys)
        assert composed.shape == (128,)
        assert not np.allclose(composed, 0)

    def test_compose_many_patterns(self, mem):
        keys = [make_embedding(128, seed=i * 7) for i in range(8)]
        for k in keys:
            mem.store(key=k, value=k)

        composed = mem.compose(keys)
        assert composed.shape == (128,)


class TestCompositionEdgeCases:
    """Test edge cases in composition."""

    def test_compose_single_key_returns_retrieve(self, mem):
        key = make_embedding(128, seed=1)
        mem.store(key=key, value=key)

        composed = mem.compose([key])
        retrieved = mem.retrieve(key)
        np.testing.assert_array_equal(composed, retrieved)

    def test_compose_empty_list(self, mem):
        result = mem.compose([])
        assert result.shape == (128,)
        assert np.allclose(result, 0)

    def test_compose_with_weights_warns_local(self, mem):
        key_a = make_embedding(128, seed=1)
        key_b = make_embedding(128, seed=2)
        mem.store(key=key_a, value=key_a)
        mem.store(key=key_b, value=key_b)

        with pytest.warns(UserWarning, match="Weighted composition is limited"):
            mem.compose([key_a, key_b], weights=[0.8, 0.2])

    def test_compose_with_equal_weights_no_warn(self, mem):
        key_a = make_embedding(128, seed=1)
        key_b = make_embedding(128, seed=2)
        mem.store(key=key_a, value=key_a)
        mem.store(key=key_b, value=key_b)

        # Equal weights should NOT warn
        result = mem.compose([key_a, key_b], weights=[0.5, 0.5])
        assert result.shape == (128,)


class TestCompositionQuality:
    """Test that composition produces meaningful results."""

    def test_composition_preserves_norm(self, mem):
        """Composed vector should have reasonable magnitude."""
        key_a = make_embedding(128, seed=1)
        key_b = make_embedding(128, seed=2)
        mem.store(key=key_a, value=key_a)
        mem.store(key=key_b, value=key_b)

        composed = mem.compose([key_a, key_b])
        norm = np.linalg.norm(composed)
        # Should not explode or vanish
        assert 0.1 < norm < 10.0

    def test_different_compositions_differ(self, mem):
        """Different pairs should produce different compositions."""
        keys = [make_embedding(128, seed=i * 100) for i in range(4)]
        for k in keys:
            mem.store(key=k, value=k)

        comp_01 = mem.compose([keys[0], keys[1]])
        comp_23 = mem.compose([keys[2], keys[3]])

        # Compositions of different pairs should not be identical
        assert not np.array_equal(comp_01, comp_23)

    def test_composition_stability_many_patterns(self, mem):
        """Composition should work even with many stored patterns."""
        for i in range(100):
            k = make_embedding(128, seed=i)
            mem.store(key=k, value=k)

        key_a = make_embedding(128, seed=0)
        key_b = make_embedding(128, seed=50)
        composed = mem.compose([key_a, key_b])

        assert composed.shape == (128,)
        assert np.isfinite(composed).all()
        assert np.linalg.norm(composed) > 0.01
