"""
Test 06: Framework Integrations
================================
Tests all integration adapters (CrewAI, OpenAI Assistants, FastAPI, LangChain).
These run WITHOUT actual framework dependencies using local mocks where needed.

Real use cases:
- CrewAI agent with long-term memory
- OpenAI Assistant with tool-calling memory
- FastAPI endpoint serving memory queries
- LangChain chain with conversational memory
"""

import json

import numpy as np
import pytest

from conftest import text_to_embedding

from engramma_memory.integrations.crewai import EngrammaCrewMemory
from engramma_memory.integrations.openai_assistants import (
    EngrammaToolHandler,
    engramma_tool_definitions,
)


class TestCrewAIIntegration:
    """Test CrewAI memory adapter."""

    def test_save_and_search(self):
        memory = EngrammaCrewMemory(dim=128)
        memory.save("The user prefers dark mode interfaces")
        memory.save("The project deadline is next Friday")
        memory.save("Use Python 3.11 for all new services")

        results = memory.search("user interface preferences")
        assert len(results) > 0

    def test_search_relevance(self):
        memory = EngrammaCrewMemory(dim=128)
        memory.save("Alice likes Python programming")
        memory.save("Bob works on the frontend with React")
        memory.save("The deployment uses Kubernetes on AWS")

        results = memory.search("Python")
        assert any("Python" in r["content"] for r in results)

    def test_compose_context(self):
        memory = EngrammaCrewMemory(dim=128)
        memory.save("User wants fast response times")
        memory.save("System uses PostgreSQL database")
        memory.save("Deployed on AWS us-east-1")

        context = memory.compose_context(["performance", "infrastructure"])
        assert isinstance(context, str)

    def test_reset(self):
        memory = EngrammaCrewMemory(dim=128)
        memory.save("Some fact")
        assert memory._engramma.count == 1
        memory.reset()
        assert memory._engramma.count == 0

    def test_search_empty(self):
        memory = EngrammaCrewMemory(dim=128)
        results = memory.search("anything")
        assert results == []

    def test_custom_embed_fn(self):
        def custom_embed(text: str) -> np.ndarray:
            rng = np.random.default_rng(len(text))
            return rng.standard_normal(64).astype(np.float32)

        memory = EngrammaCrewMemory(dim=64, embed_fn=custom_embed)
        memory.save("Test content")
        results = memory.search("Test")
        assert isinstance(results, list)


class TestOpenAIAssistantsIntegration:
    """Test OpenAI Assistants tool handler."""

    def test_tool_definitions_structure(self):
        tools = engramma_tool_definitions()
        assert len(tools) == 4
        names = {t["function"]["name"] for t in tools}
        assert names == {"engramma_store", "engramma_query", "engramma_compose", "engramma_forget"}

    def test_tool_definitions_valid_schema(self):
        tools = engramma_tool_definitions()
        for tool in tools:
            assert tool["type"] == "function"
            assert "name" in tool["function"]
            assert "description" in tool["function"]
            assert "parameters" in tool["function"]
            assert "required" in tool["function"]["parameters"]

    def test_handle_store(self):
        handler = EngrammaToolHandler(dim=128)
        result = json.loads(handler.handle("engramma_store", {"content": "User likes coffee"}))
        assert result["status"] == "stored"
        assert result["memory_count"] == 1

    def test_handle_query_after_store(self):
        handler = EngrammaToolHandler(dim=128)
        handler.handle("engramma_store", {"content": "Python is a programming language"})
        handler.handle("engramma_store", {"content": "JavaScript runs in browsers"})
        handler.handle("engramma_store", {"content": "Rust is for systems programming"})

        result = json.loads(handler.handle("engramma_query", {"query": "programming language", "top_k": 3}))
        assert "memories" in result
        assert result["count"] >= 0

    def test_handle_compose(self):
        handler = EngrammaToolHandler(dim=128)
        handler.handle("engramma_store", {"content": "Fast delivery"})
        handler.handle("engramma_store", {"content": "High quality"})

        result = json.loads(
            handler.handle("engramma_compose", {"queries": ["speed", "quality"]})
        )
        assert "composed" in result
        assert "inputs" in result

    def test_handle_forget(self):
        handler = EngrammaToolHandler(dim=128)
        handler.handle("engramma_store", {"content": "Old information"})
        result = json.loads(
            handler.handle("engramma_forget", {"query": "Old information", "strategy": "immediate"})
        )
        assert result["status"] == "forgotten"
        assert result["strategy"] == "immediate"

    def test_handle_unknown_tool(self):
        handler = EngrammaToolHandler(dim=128)
        result = json.loads(handler.handle("nonexistent_tool", {}))
        assert "error" in result

    def test_full_assistant_workflow(self):
        """Simulate a full assistant conversation with memory."""
        handler = EngrammaToolHandler(dim=128)

        # User tells assistant their preferences
        handler.handle("engramma_store", {"content": "User name is Alice"})
        handler.handle("engramma_store", {"content": "Alice works at TechCorp"})
        handler.handle("engramma_store", {"content": "Alice prefers concise answers"})

        # Later, assistant needs context
        result = json.loads(
            handler.handle("engramma_query", {"query": "who is the user", "top_k": 3})
        )
        assert result["count"] > 0

        # Compose context from multiple aspects
        result = json.loads(
            handler.handle("engramma_compose", {"queries": ["user name", "work"]})
        )
        assert "composed" in result


class TestFastAPIIntegration:
    """Test FastAPI router creation (without running a server)."""

    def test_create_router_default(self):
        try:
            from engramma_memory.integrations.fastapi import create_memory_router
            router = create_memory_router(dim=128)
            assert router is not None
            # Should have routes
            assert len(router.routes) >= 5
        except ImportError:
            pytest.skip("FastAPI not installed")

    def test_create_router_custom_dim(self):
        try:
            from engramma_memory.integrations.fastapi import create_memory_router
            router = create_memory_router(dim=64)
            assert router is not None
        except ImportError:
            pytest.skip("FastAPI not installed")

    def test_middleware_creation(self):
        try:
            from engramma_memory.integrations.fastapi import EngrammaMiddleware

            class MockApp:
                async def __call__(self, scope, receive, send):
                    pass

            middleware = EngrammaMiddleware(MockApp(), dim=128)
            assert middleware._engramma is not None
            assert middleware._engramma.dim == 128
        except ImportError:
            pytest.skip("FastAPI not installed")

    def test_default_embed_factory(self):
        from engramma_memory.integrations.fastapi import _default_embed_factory
        embed = _default_embed_factory(128)
        vec = embed("hello world")
        assert vec.shape == (128,)
        assert np.isclose(np.linalg.norm(vec), 1.0, atol=0.01)
        # Deterministic
        vec2 = embed("hello world")
        np.testing.assert_array_equal(vec, vec2)


class TestLangChainIntegration:
    """Test LangChain memory adapter (without langchain-core)."""

    def test_langchain_import_guard(self):
        """Should raise ImportError if langchain-core not installed."""
        from engramma_memory.integrations.langchain import _HAS_LANGCHAIN
        if not _HAS_LANGCHAIN:
            with pytest.raises(ImportError, match="langchain-core"):
                from engramma_memory.integrations.langchain import EngrammaLangChainMemory
                EngrammaLangChainMemory(dim=128)

    def test_default_embed_is_deterministic(self):
        """The fallback embedding function should be deterministic."""
        from engramma_memory.integrations.langchain import EngrammaLangChainMemory
        try:
            mem = EngrammaLangChainMemory(dim=128)
            e1 = mem._default_embed("hello")
            e2 = mem._default_embed("hello")
            np.testing.assert_array_equal(e1, e2)
        except ImportError:
            pytest.skip("langchain-core not installed")
