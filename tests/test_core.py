"""Tests for the public EngrammaMemory API."""
import numpy as np
import pytest

from engramma_memory import EngrammaMemory


DIM = 64
RNG = np.random.default_rng(456)


def random_vec(dim=DIM):
    v = RNG.standard_normal(dim).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


class TestEngrammaMemoryLocal:
    def test_init_defaults(self):
        mem = EngrammaMemory(dim=DIM)
        assert mem.dim == DIM
        assert mem.backend_name == "local"
        assert mem.count == 0

    def test_store_and_query(self):
        mem = EngrammaMemory(dim=DIM)
        key = random_vec()
        mem.store(key=key, value=key)

        results = mem.query(key, top_k=1)
        assert len(results) == 1
        assert results[0]["score"] > 0.9

    def test_query_top_k(self):
        mem = EngrammaMemory(dim=DIM)
        for _ in range(10):
            v = random_vec()
            mem.store(key=v, value=v)

        results = mem.query(random_vec(), top_k=5)
        assert len(results) == 5

    def test_retrieve(self):
        mem = EngrammaMemory(dim=DIM)
        key = random_vec()
        mem.store(key=key, value=key)
        result = mem.retrieve(key)
        assert result.shape == (DIM,)

    def test_compose(self):
        mem = EngrammaMemory(dim=DIM)
        key_a = random_vec()
        key_b = random_vec()
        mem.store(key=key_a, value=key_a)
        mem.store(key=key_b, value=key_b)

        result = mem.compose([key_a, key_b])
        assert result.shape == (DIM,)
        assert np.linalg.norm(result) > 0

    def test_forget_decay(self):
        mem = EngrammaMemory(dim=DIM)
        key = random_vec()
        mem.store(key=key, value=key)
        mem.forget(key, strategy="decay")
        assert mem.count == 1

    def test_forget_immediate(self):
        mem = EngrammaMemory(dim=DIM)
        key = random_vec()
        mem.store(key=key, value=key)
        mem.forget(key, strategy="immediate")
        assert mem.count == 0

    def test_stats(self):
        mem = EngrammaMemory(dim=DIM)
        mem.store(key=random_vec(), value=random_vec())
        stats = mem.stats()
        assert "n_stored" in stats or "exact_count" in stats
        assert stats.get("exact_count", stats.get("n_stored")) == 1

    def test_count_property(self):
        mem = EngrammaMemory(dim=DIM)
        assert mem.count == 0
        mem.store(key=random_vec(), value=random_vec())
        assert mem.count == 1

    def test_repr(self):
        mem = EngrammaMemory(dim=DIM)
        r = repr(mem)
        assert "EngrammaMemory" in r
        assert "local" in r

    def test_list_input(self):
        mem = EngrammaMemory(dim=DIM)
        key = random_vec().tolist()
        mem.store(key=key, value=key)
        results = mem.query(key, top_k=1)
        assert len(results) == 1

    def test_max_patterns_cap(self):
        with pytest.warns(UserWarning):
            mem = EngrammaMemory(dim=DIM, max_patterns=5000)

    def test_metadata_ignored_locally(self):
        mem = EngrammaMemory(dim=DIM)
        key = random_vec()
        mem.store(key=key, value=key, metadata={"tag": "test"})
        assert mem.count == 1


class TestEngrammaMemoryCloud:
    def test_cloud_requires_api_key(self):
        with pytest.raises(ValueError, match="Cloud backend requires"):
            EngrammaMemory(dim=DIM, backend="cloud")

    def test_cloud_invalid_key(self):
        with pytest.raises(ValueError, match="Invalid API key"):
            EngrammaMemory(dim=DIM, backend="cloud", api_key="invalid_key")

    def test_unknown_backend(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            EngrammaMemory(dim=DIM, backend="redis")
