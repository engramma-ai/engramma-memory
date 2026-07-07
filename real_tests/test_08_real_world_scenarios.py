"""
Test 08: Real-World Scenarios
==============================
End-to-end tests simulating actual production use cases.

Scenarios:
- Chatbot with persistent memory
- RAG agent storing/retrieving documents
- Multi-agent shared memory
- User preference learning
- Context window augmentation
"""

import numpy as np
import pytest

from conftest import make_embedding, text_to_embedding

from engramma_memory import EngrammaMemory
from engramma_memory.integrations.crewai import EngrammaCrewMemory
from engramma_memory.integrations.openai_assistants import EngrammaToolHandler


class TestChatbotMemory:
    """Simulate a chatbot that remembers conversation context."""

    def test_conversation_memory(self):
        mem = EngrammaMemory(dim=128, backend="local")
        dim = 128

        # Simulate storing conversation turns
        turns = [
            "Hello, I'm Alice and I work at TechCorp",
            "I need help with our Python microservices",
            "We're migrating from monolith to microservices",
            "The deadline is end of Q3",
            "We use PostgreSQL and Redis",
            "Our team has 5 developers",
        ]

        for turn in turns:
            emb = text_to_embedding(turn, dim)
            mem.store(key=emb, value=emb)

        assert mem.count == 6

        # Later query about the project
        query = text_to_embedding("what database does the team use", dim)
        results = mem.query(query, top_k=3)
        assert len(results) == 3
        assert all(r["score"] > 0 for r in results)

    def test_memory_grows_over_session(self):
        mem = EngrammaMemory(dim=64, backend="local")

        counts = []
        for i in range(100):
            emb = make_embedding(64, seed=i)
            mem.store(key=emb, value=emb)
            counts.append(mem.count)

        # Count should monotonically increase (no duplicates with unique seeds)
        assert counts == sorted(counts)
        assert counts[-1] == 100


class TestRAGAgent:
    """Simulate a RAG agent indexing and retrieving documents."""

    def test_document_indexing(self):
        mem = EngrammaMemory(dim=128, backend="local")

        documents = [
            "Python was created by Guido van Rossum in 1991",
            "JavaScript was created by Brendan Eich in 1995",
            "Rust was created by Graydon Hoare in 2010",
            "Go was created by Robert Griesemer, Rob Pike, Ken Thompson in 2009",
            "TypeScript was created by Microsoft in 2012",
        ]

        for doc in documents:
            emb = text_to_embedding(doc, 128)
            mem.store(key=emb, value=emb)

        # Query for a specific topic
        query = text_to_embedding("Who created Python", 128)
        results = mem.query(query, top_k=2)
        assert len(results) == 2

    def test_composition_for_synthesis(self):
        """Use composition to blend related documents."""
        mem = EngrammaMemory(dim=128, backend="local")

        # Store related facts
        facts = [
            "The weather in Paris is mild in spring",
            "Paris has many outdoor cafes",
            "Spring is from March to May",
        ]

        embeddings = []
        for fact in facts:
            emb = text_to_embedding(fact, 128)
            mem.store(key=emb, value=emb)
            embeddings.append(emb)

        # Compose "Paris + Spring" for a blended context
        composed = mem.compose([embeddings[0], embeddings[2]])
        assert composed.shape == (128,)
        assert np.linalg.norm(composed) > 0


class TestMultiAgentSharedMemory:
    """Simulate multiple agents sharing memory."""

    def test_shared_memory_instance(self):
        shared_mem = EngrammaMemory(dim=128, backend="local")

        # Agent 1 stores findings
        for i in range(10):
            emb = make_embedding(128, seed=i)
            shared_mem.store(key=emb, value=emb)

        # Agent 2 queries the shared memory
        query = make_embedding(128, seed=5)
        results = shared_mem.query(query, top_k=3)
        assert len(results) == 3
        assert results[0]["score"] > 0.9

    def test_crew_multi_agent_workflow(self):
        """CrewAI-style multi-agent memory sharing."""
        memory = EngrammaCrewMemory(dim=128)

        # Research agent stores findings
        memory.save("The API has a rate limit of 100 requests/minute")
        memory.save("Authentication uses OAuth 2.0 with JWT tokens")
        memory.save("The database schema has 15 tables")

        # Writer agent searches for context
        results = memory.search("API authentication")
        assert len(results) > 0

        # Compose agent creates unified context
        context = memory.compose_context(["API limits", "authentication"])
        assert isinstance(context, str)


class TestUserPreferenceLearning:
    """Simulate learning user preferences over time."""

    def test_preference_accumulation(self):
        handler = EngrammaToolHandler(dim=128)

        preferences = [
            "User prefers dark mode",
            "User likes concise responses",
            "User works in Python primarily",
            "User is in timezone UTC+1",
            "User prefers technical explanations",
        ]

        for pref in preferences:
            handler.handle("engramma_store", {"content": pref})

        # Query for relevant preferences
        result = handler.handle("engramma_query", {"query": "what language does user code in", "top_k": 3})
        parsed = __import__("json").loads(result)
        assert parsed["count"] > 0

    def test_preference_override(self):
        """New preferences should be accessible alongside old ones."""
        handler = EngrammaToolHandler(dim=128)

        handler.handle("engramma_store", {"content": "User prefers light mode"})
        handler.handle("engramma_store", {"content": "User now prefers dark mode"})

        result = handler.handle("engramma_query", {"query": "theme preference", "top_k": 2})
        parsed = __import__("json").loads(result)
        # Both should be queryable
        assert parsed["count"] >= 1


class TestContextWindowAugmentation:
    """Simulate using memory to augment limited context windows."""

    def test_store_and_retrieve_long_context(self):
        mem = EngrammaMemory(dim=256, backend="local")

        # Store many context chunks
        for i in range(200):
            chunk = make_embedding(256, seed=i)
            mem.store(key=chunk, value=chunk)

        # Retrieve relevant context for current query
        query = make_embedding(256, seed=42)
        results = mem.query(query, top_k=10)
        assert len(results) == 10
        # First result should be the exact match
        assert results[0]["score"] > 0.9

    def test_compose_multiple_context_windows(self):
        """Blend multiple relevant contexts into one."""
        mem = EngrammaMemory(dim=128, backend="local")

        # Store conversation segments
        segments = [make_embedding(128, seed=i) for i in range(20)]
        for seg in segments:
            mem.store(key=seg, value=seg)

        # Compose recent context
        recent = segments[-3:]
        composed = mem.compose(recent)
        assert composed.shape == (128,)
        assert np.isfinite(composed).all()


class TestContinualLearning:
    """Test that memory adapts as new information is added."""

    def test_new_patterns_accessible_immediately(self, mem):
        for i in range(50):
            key = make_embedding(128, seed=i)
            mem.store(key=key, value=key)

            # New pattern should be immediately queryable
            results = mem.query(key, top_k=1)
            assert results[0]["score"] > 0.9

    def test_old_patterns_still_accessible(self, mem):
        keys = [make_embedding(128, seed=i) for i in range(100)]
        for k in keys:
            mem.store(key=k, value=k)

        # Older patterns should still be retrievable
        for i in [0, 10, 50, 99]:
            results = mem.query(keys[i], top_k=1)
            assert results[0]["score"] > 0.8

    def test_retrieve_routing_adapts(self, mem):
        """Router should learn from confidence signals."""
        # Store patterns with distinct characteristics
        for i in range(30):
            key = make_embedding(128, seed=i)
            mem.store(key=key, value=key)
            # Exercise retrieve to train router
            mem.retrieve(key)

        # After training, retrieve should work well
        test_key = make_embedding(128, seed=5)
        result = mem.retrieve(test_key)
        # Should be close to stored value
        stored = make_embedding(128, seed=5)
        sim = np.dot(result, stored) / (np.linalg.norm(result) * np.linalg.norm(stored) + 1e-8)
        assert sim > 0.5
