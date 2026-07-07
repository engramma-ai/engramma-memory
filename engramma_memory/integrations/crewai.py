"""
CrewAI integration for Engramma Memory.

Provides a memory class compatible with CrewAI's agent memory system.

Usage:
    from engramma_memory.integrations.crewai import EngrammaCrewMemory

    memory = EngrammaCrewMemory(dim=256)
    # Use with CrewAI agents
"""
from typing import Optional, List, Dict, Any

import numpy as np

from ..core import EngrammaMemory


class EngrammaCrewMemory:
    """
    CrewAI-compatible memory backed by Engramma.

    Provides long-term memory for CrewAI agents with native composition
    capabilities - agents can blend related memories for richer context.

    Parameters
    ----------
    dim : int
        Embedding dimension.
    embed_fn : callable, optional
        Text-to-embedding function.
    backend : str
        "local" or "cloud".
    api_key : str, optional
        API key for cloud backend.
    """

    def __init__(self, dim: int = 256, embed_fn=None,
                 backend: str = "local", api_key: Optional[str] = None):
        self._dim = dim
        self._embed_fn = embed_fn or self._default_embed
        self._engramma = EngrammaMemory(dim=dim, backend=backend, api_key=api_key)
        self._texts: List[str] = []

    def save(self, value: str, metadata: Optional[Dict] = None) -> None:
        """Save a memory entry."""
        embedding = self._embed_fn(value)
        self._engramma.store(key=embedding, value=embedding)
        self._texts.append(value)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant memories."""
        if self._engramma.count == 0:
            return []

        embedding = self._embed_fn(query)
        results = self._engramma.query(embedding, top_k=top_k)

        memories = []
        for r in results:
            if r["score"] > 0.2:
                text = self._find_text(r["value"])
                if text:
                    memories.append({
                        "content": text,
                        "score": r["score"],
                    })
        return memories

    def compose_context(self, queries: List[str]) -> str:
        """
        Compose multiple memory queries into a unified context.

        This leverages Engramma's native composition - not available
        with standard vector stores.
        """
        if not queries or self._engramma.count == 0:
            return ""

        embeddings = [self._embed_fn(q) for q in queries]
        composed = self._engramma.compose(embeddings)
        text = self._find_text(composed)
        return text or ""

    def reset(self) -> None:
        """Clear all memories."""
        self._engramma = EngrammaMemory(dim=self._dim, backend="local")
        self._texts = []

    def _find_text(self, value: np.ndarray) -> Optional[str]:
        if not self._texts:
            return None
        value = value.flatten()[:self._dim]
        best_dist = float('inf')
        best_text = None
        for text in self._texts:
            emb = self._embed_fn(text)
            dist = float(np.linalg.norm(emb.flatten()[:self._dim] - value))
            if dist < best_dist:
                best_dist = dist
                best_text = text
        return best_text if best_dist < 2.0 else None

    def _default_embed(self, text: str) -> np.ndarray:
        rng = np.random.default_rng(hash(text) % (2**32))
        vec = rng.standard_normal(self._dim).astype(np.float32)
        return vec / (np.linalg.norm(vec) + 1e-8)
