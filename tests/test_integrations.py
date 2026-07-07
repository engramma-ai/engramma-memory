"""Tests for framework integrations (import-level checks)."""
import pytest
import numpy as np


DIM = 64


class TestLangChainIntegration:
    @pytest.fixture(autouse=True)
    def skip_if_no_langchain(self):
        pytest.importorskip("langchain_core")

    def test_import_and_init(self):
        from engramma_memory.integrations.langchain import EngrammaLangChainMemory

        def embed_fn(text):
            np.random.seed(hash(text) % 2**31)
            return np.random.randn(DIM).astype(np.float32)

        mem = EngrammaLangChainMemory(dim=DIM, embed_fn=embed_fn)
        assert mem is not None


class TestLlamaIndexIntegration:
    @pytest.fixture(autouse=True)
    def skip_if_no_llamaindex(self):
        pytest.importorskip("llama_index.core")

    def test_import_and_init(self):
        from engramma_memory.integrations.llamaindex import EngrammaRetriever

        def embed_fn(text):
            np.random.seed(hash(text) % 2**31)
            return np.random.randn(DIM).astype(np.float32)

        retriever = EngrammaRetriever(dim=DIM, embed_fn=embed_fn, top_k=5)
        assert retriever is not None


class TestOpenAIIntegration:
    @pytest.fixture(autouse=True)
    def skip_if_no_openai(self):
        pytest.importorskip("openai")

    def test_import_tool_definitions(self):
        from engramma_memory.integrations.openai_assistants import engramma_tool_definitions
        tools = engramma_tool_definitions()
        assert isinstance(tools, list)
        assert len(tools) > 0


class TestCrewAIIntegration:
    @pytest.fixture(autouse=True)
    def skip_if_no_crewai(self):
        pytest.importorskip("crewai")

    def test_import(self):
        from engramma_memory.integrations.crewai import EngrammaCrewMemory
        assert EngrammaCrewMemory is not None


class TestFastAPIIntegration:
    @pytest.fixture(autouse=True)
    def skip_if_no_fastapi(self):
        pytest.importorskip("fastapi")

    def test_import_router_factory(self):
        from engramma_memory.integrations.fastapi import create_memory_router
        assert callable(create_memory_router)
