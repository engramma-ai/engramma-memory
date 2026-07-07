"""
Test 09: Performance Benchmarks
=================================
Tests latency and throughput to verify performance claims.
Not strict assertions but sanity checks that operations complete in reasonable time.

Benchmarks from docs:
- Local p50 latency @ 1000 vectors: ~8.8ms
- Store: should complete in <50ms per operation
- Query: should complete in <20ms per operation
"""

import time

import numpy as np
import pytest

from conftest import make_embedding

from engramma_memory import EngrammaMemory


class TestStoreThroughput:
    """Benchmark store operations."""

    def test_store_1000_patterns_under_10s(self):
        mem = EngrammaMemory(dim=128, backend="local")
        start = time.perf_counter()

        for i in range(1000):
            key = make_embedding(128, seed=i)
            mem.store(key=key, value=key)

        elapsed = time.perf_counter() - start
        assert elapsed < 10.0, f"Storing 1000 patterns took {elapsed:.2f}s (expected <10s)"
        assert mem.count == 1000

    def test_store_latency_per_op(self):
        mem = EngrammaMemory(dim=256, backend="local")
        latencies = []

        for i in range(100):
            key = make_embedding(256, seed=i)
            start = time.perf_counter()
            mem.store(key=key, value=key)
            latencies.append(time.perf_counter() - start)

        p50 = np.percentile(latencies, 50)
        p99 = np.percentile(latencies, 99)
        assert p50 < 0.05, f"Store p50 = {p50*1000:.1f}ms (expected <50ms)"
        assert p99 < 0.2, f"Store p99 = {p99*1000:.1f}ms (expected <200ms)"


class TestQueryLatency:
    """Benchmark query operations."""

    def test_query_latency_at_100_patterns(self):
        mem = EngrammaMemory(dim=128, backend="local")
        for i in range(100):
            key = make_embedding(128, seed=i)
            mem.store(key=key, value=key)

        latencies = []
        for i in range(50):
            query = make_embedding(128, seed=i + 5000)
            start = time.perf_counter()
            mem.query(query, top_k=5)
            latencies.append(time.perf_counter() - start)

        p50 = np.percentile(latencies, 50)
        assert p50 < 0.02, f"Query p50 @ 100 patterns = {p50*1000:.1f}ms (expected <20ms)"

    def test_query_latency_at_500_patterns(self):
        mem = EngrammaMemory(dim=128, backend="local")
        for i in range(500):
            key = make_embedding(128, seed=i)
            mem.store(key=key, value=key)

        latencies = []
        for i in range(50):
            query = make_embedding(128, seed=i + 6000)
            start = time.perf_counter()
            mem.query(query, top_k=5)
            latencies.append(time.perf_counter() - start)

        p50 = np.percentile(latencies, 50)
        assert p50 < 0.05, f"Query p50 @ 500 patterns = {p50*1000:.1f}ms (expected <50ms)"

    def test_query_latency_at_1000_patterns(self):
        mem = EngrammaMemory(dim=128, backend="local")
        for i in range(1000):
            key = make_embedding(128, seed=i)
            mem.store(key=key, value=key)

        latencies = []
        for i in range(50):
            query = make_embedding(128, seed=i + 7000)
            start = time.perf_counter()
            mem.query(query, top_k=5)
            latencies.append(time.perf_counter() - start)

        p50 = np.percentile(latencies, 50)
        assert p50 < 0.1, f"Query p50 @ 1000 patterns = {p50*1000:.1f}ms (expected <100ms)"


class TestRetrieveLatency:
    """Benchmark smart routing retrieval."""

    def test_retrieve_latency(self):
        mem = EngrammaMemory(dim=128, backend="local")
        for i in range(200):
            key = make_embedding(128, seed=i)
            mem.store(key=key, value=key)

        latencies = []
        for i in range(50):
            query = make_embedding(128, seed=i + 8000)
            start = time.perf_counter()
            mem.retrieve(query)
            latencies.append(time.perf_counter() - start)

        p50 = np.percentile(latencies, 50)
        assert p50 < 0.05, f"Retrieve p50 = {p50*1000:.1f}ms (expected <50ms)"


class TestCompositionLatency:
    """Benchmark composition operations."""

    def test_two_way_composition_latency(self):
        mem = EngrammaMemory(dim=128, backend="local")
        for i in range(100):
            key = make_embedding(128, seed=i)
            mem.store(key=key, value=key)

        key_a = make_embedding(128, seed=0)
        key_b = make_embedding(128, seed=50)

        latencies = []
        for _ in range(50):
            start = time.perf_counter()
            mem.compose([key_a, key_b])
            latencies.append(time.perf_counter() - start)

        p50 = np.percentile(latencies, 50)
        assert p50 < 0.05, f"Compose p50 = {p50*1000:.1f}ms (expected <50ms)"

    def test_three_way_composition_latency(self):
        mem = EngrammaMemory(dim=128, backend="local")
        for i in range(100):
            key = make_embedding(128, seed=i)
            mem.store(key=key, value=key)

        keys = [make_embedding(128, seed=i * 10) for i in range(3)]

        latencies = []
        for _ in range(30):
            start = time.perf_counter()
            mem.compose(keys)
            latencies.append(time.perf_counter() - start)

        p50 = np.percentile(latencies, 50)
        assert p50 < 0.1, f"3-way compose p50 = {p50*1000:.1f}ms (expected <100ms)"


class TestMemoryFootprint:
    """Check memory usage stays reasonable."""

    def test_memory_footprint_1000_patterns(self):
        import sys
        mem = EngrammaMemory(dim=128, backend="local")

        # Store 1000 patterns
        for i in range(1000):
            key = make_embedding(128, seed=i)
            mem.store(key=key, value=key)

        # Engine size should be reasonable (< 50MB for 128-dim x 1000 patterns)
        engine = mem._backend.engine
        total_bytes = (
            engine.exact.keys.nbytes +
            engine.exact.values.nbytes +
            engine.energy.keys.nbytes +
            engine.energy.values.nbytes +
            engine.attention.keys.nbytes +
            engine.attention.values.nbytes
        )
        total_mb = total_bytes / (1024 * 1024)
        assert total_mb < 50, f"Memory footprint = {total_mb:.1f}MB (expected <50MB)"
