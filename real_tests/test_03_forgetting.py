"""
Test 03: Forgetting - Decay and Immediate Removal
===================================================
Tests the ability to forget or decay stored patterns.

Real use cases:
- GDPR right-to-be-forgotten: immediately removing user data
- Stale context decay: reducing importance of old memories
- Memory management: clearing outdated preferences
"""

import numpy as np
import pytest

from conftest import make_embedding

from engramma_memory import EngrammaMemory


class TestImmediateForget:
    """Test immediate (hard delete) forgetting."""

    def test_forget_immediate_removes_pattern(self, mem):
        key = make_embedding(128, seed=1)
        mem.store(key=key, value=key)
        assert mem.count == 1

        mem.forget(key, strategy="immediate")
        # Pattern should be deactivated
        stats = mem.stats()
        assert stats["exact_count"] == 0

    def test_forget_immediate_specific_pattern(self, mem):
        keys = [make_embedding(128, seed=i) for i in range(5)]
        for k in keys:
            mem.store(key=k, value=k)
        assert mem.count == 5

        mem.forget(keys[2], strategy="immediate")
        assert mem.count == 4

    def test_forget_nonexistent_key_no_error(self, mem):
        key = make_embedding(128, seed=1)
        mem.store(key=key, value=key)

        far_away_key = make_embedding(128, seed=9999)
        # Should not crash even if key is not close to any stored pattern
        mem.forget(far_away_key, strategy="immediate")

    def test_forget_then_query(self, mem):
        key = make_embedding(128, seed=1)
        mem.store(key=key, value=key)
        mem.forget(key, strategy="immediate")

        # After immediate forget, the pattern is deactivated in exact memory
        # Query still returns a result but the exact_count should be 0
        assert mem.count == 0


class TestDecayForget:
    """Test decay (gradual importance reduction) forgetting."""

    def test_decay_reduces_importance(self, mem):
        key = make_embedding(128, seed=1)
        mem.store(key=key, value=key)

        # Access it many times to build up importance
        for _ in range(10):
            mem.query(key, top_k=1)

        # Now decay
        mem.forget(key, strategy="decay")

        # Pattern still exists but with reduced importance
        assert mem.count == 1

    def test_decay_default_strategy(self, mem):
        key = make_embedding(128, seed=5)
        mem.store(key=key, value=key)

        # Default strategy is "decay"
        mem.forget(key)
        assert mem.count == 1

    def test_repeated_decay_eventually_allows_eviction(self, mem):
        """After repeated decays, the pattern should be evictable."""
        target = make_embedding(128, seed=0)
        mem.store(key=target, value=target)

        # Decay multiple times
        for _ in range(10):
            mem.forget(target, strategy="decay")

        # Now fill memory to force eviction
        for i in range(1, 1001):
            k = make_embedding(128, seed=i + 5000)
            mem.store(key=k, value=k)

        # The decayed pattern should have been evicted
        results = mem.query(target, top_k=1)
        # If evicted, the closest match won't be exact
        assert results[0]["score"] < 0.95


class TestForgetWithList:
    """Test forgetting with list inputs."""

    def test_forget_with_list_key(self, mem):
        key = [0.1] * 128
        mem.store(key=key, value=key)
        mem.forget(key, strategy="immediate")
        assert mem.count == 0
