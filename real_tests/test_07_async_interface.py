"""
Test 07: Async Interface
=========================
Tests the async core with the local backend (async wrapping sync operations).

Real use cases:
- Async web frameworks (FastAPI, aiohttp)
- Async agent loops
- Concurrent memory operations
"""

import numpy as np
import pytest

from conftest import make_embedding

from engramma_memory import EngrammaMemoryAsync


@pytest.fixture
def async_mem():
    return EngrammaMemoryAsync(dim=128, backend="local")


class TestAsyncCoreOperations:
    """Test basic async store/query/retrieve."""

    @pytest.mark.asyncio
    async def test_async_store(self, async_mem):
        key = make_embedding(128, seed=1)
        await async_mem.store(key=key, value=key)

    @pytest.mark.asyncio
    async def test_async_query(self, async_mem):
        key = make_embedding(128, seed=1)
        await async_mem.store(key=key, value=key)
        results = await async_mem.query(key, top_k=1)
        assert len(results) == 1
        assert results[0]["score"] > 0.9

    @pytest.mark.asyncio
    async def test_async_retrieve(self, async_mem):
        key = make_embedding(128, seed=5)
        value = make_embedding(128, seed=6)
        await async_mem.store(key=key, value=value)
        result = await async_mem.retrieve(key)
        assert result.shape == (128,)

    @pytest.mark.asyncio
    async def test_async_compose(self, async_mem):
        key_a = make_embedding(128, seed=1)
        key_b = make_embedding(128, seed=2)
        await async_mem.store(key=key_a, value=key_a)
        await async_mem.store(key=key_b, value=key_b)
        result = await async_mem.compose([key_a, key_b])
        assert result.shape == (128,)

    @pytest.mark.asyncio
    async def test_async_forget(self, async_mem):
        key = make_embedding(128, seed=1)
        await async_mem.store(key=key, value=key)
        await async_mem.forget(key, strategy="immediate")


class TestAsyncContextManager:
    """Test async context manager lifecycle."""

    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        async with EngrammaMemoryAsync(dim=64, backend="local") as mem:
            key = make_embedding(64, seed=1)
            await mem.store(key=key, value=key)
            results = await mem.query(key, top_k=1)
            assert len(results) == 1

    @pytest.mark.asyncio
    async def test_close_without_context_manager(self):
        mem = EngrammaMemoryAsync(dim=64, backend="local")
        await mem.close()  # Should not raise


class TestAsyncCloudGuards:
    """Test that cloud-only methods raise RuntimeError on local backend."""

    @pytest.mark.asyncio
    async def test_get_modulation_state_local_raises(self, async_mem):
        with pytest.raises(RuntimeError, match="cloud-only"):
            await async_mem.get_modulation_state()

    @pytest.mark.asyncio
    async def test_explain_local_raises(self, async_mem):
        with pytest.raises(RuntimeError, match="cloud-only"):
            query = make_embedding(128, seed=1)
            await async_mem.explain(query)

    @pytest.mark.asyncio
    async def test_consolidate_local_raises(self, async_mem):
        with pytest.raises(RuntimeError, match="cloud-only"):
            await async_mem.consolidate()

    @pytest.mark.asyncio
    async def test_store_text_local_raises(self, async_mem):
        with pytest.raises(RuntimeError, match="cloud-only"):
            await async_mem.store_text("hello world")

    @pytest.mark.asyncio
    async def test_query_text_local_raises(self, async_mem):
        with pytest.raises(RuntimeError, match="cloud-only"):
            await async_mem.query_text("hello")

    @pytest.mark.asyncio
    async def test_get_current_regime_local_raises(self, async_mem):
        with pytest.raises(RuntimeError, match="cloud-only"):
            await async_mem.get_current_regime()

    @pytest.mark.asyncio
    async def test_snapshot_local_raises(self, async_mem):
        with pytest.raises(RuntimeError, match="cloud-only"):
            await async_mem.snapshot()

    @pytest.mark.asyncio
    async def test_analytics_local_raises(self, async_mem):
        with pytest.raises(RuntimeError, match="cloud-only"):
            await async_mem.analytics()

    @pytest.mark.asyncio
    async def test_enable_prefetch_local_raises(self, async_mem):
        with pytest.raises(RuntimeError, match="cloud-only"):
            await async_mem.enable_prefetch()

    @pytest.mark.asyncio
    async def test_compose_fractional_local_raises(self, async_mem):
        with pytest.raises(RuntimeError, match="cloud-only"):
            key_a = make_embedding(128, seed=1)
            key_b = make_embedding(128, seed=2)
            await async_mem.compose_fractional(key_a, key_b, alpha=0.5)


class TestAsyncInitialization:
    """Test async initialization edge cases."""

    @pytest.mark.asyncio
    async def test_cloud_without_key_raises(self):
        with pytest.raises(ValueError, match="API key"):
            EngrammaMemoryAsync(dim=128, backend="cloud")

    @pytest.mark.asyncio
    async def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            EngrammaMemoryAsync(dim=128, backend="fake")
