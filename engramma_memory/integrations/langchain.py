"""
LangChain integration for Engramma Memory.

Implements BaseMemory interface for drop-in use with LangChain chains and agents.

Usage:
    from engramma_memory.integrations.langchain import EngrammaLangChainMemory

    memory = EngrammaLangChainMemory(dim=256)
    chain = ConversationChain(llm=llm, memory=memory)
"""
from typing import Dict, List, Any, Optional

import numpy as np

from ..core import EngrammaMemory


try:
    from langchain_core.memory import BaseMemory
    _HAS_LANGCHAIN = True
except ImportError:
    _HAS_LANGCHAIN = False

    class BaseMemory:
        pass


class EngrammaLangChainMemory(BaseMemory):
    """
    LangChain-compatible memory backed by Engramma.

    Stores conversation history as embeddings and retrieves relevant context
    using Engramma's compositional memory engine.

    Parameters
    ----------
    dim : int
        Embedding dimension.
    memory_key : str
        Key used to pass memory context to the chain.
    input_key : str
        Key for incoming messages.
    embed_fn : callable, optional
        Function to convert text to embeddings. If None, uses a simple hash-based
        embedding (for testing only - use a real embedder in production).
    backend : str
        "local" or "cloud".
    api_key : str, optional
        API key for cloud backend.
    top_k : int
        Number of memories to retrieve per query.
    """

    memory_key: str = "history"
    input_key: str = "input"

    def __init__(self, dim: int = 256, memory_key: str = "history",
                 input_key: str = "input", embed_fn=None,
                 backend: str = "local", api_key: Optional[str] = None,
                 top_k: int = 5, **kwargs):
        if not _HAS_LANGCHAIN:
            raise ImportError(
                "LangChain integration requires langchain-core. "
                "Install with: pip install engramma-memory[langchain]"
            )
        super().__init__(**kwargs)
        self._dim = dim
        self._memory_key = memory_key
        self._input_key = input_key
        self._top_k = top_k
        self._embed_fn = embed_fn or self._default_embed
        self._engramma = EngrammaMemory(dim=dim, backend=backend, api_key=api_key)
        self._buffer: List[str] = []

    @property
    def memory_variables(self) -> List[str]:
        return [self._memory_key]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        query_text = inputs.get(self._input_key, "")
        if not query_text or self._engramma.count == 0:
            return {self._memory_key: ""}

        query_emb = self._embed_fn(query_text)
        results = self._engramma.query(query_emb, top_k=self._top_k)

        relevant = []
        for i, r in enumerate(results):
            if r["score"] > 0.3:
                if i < len(self._buffer):
                    relevant.append(self._buffer[-(i + 1)])

        return {self._memory_key: "\n".join(relevant[-self._top_k:])}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        input_text = inputs.get(self._input_key, "")
        output_text = outputs.get("output", outputs.get("response", ""))

        for text in [input_text, output_text]:
            if text:
                embedding = self._embed_fn(text)
                self._engramma.store(key=embedding, value=embedding)
                self._buffer.append(text)

    def clear(self) -> None:
        self._engramma = EngrammaMemory(dim=self._dim, backend="local")
        self._buffer = []

    def _default_embed(self, text: str) -> np.ndarray:
        """Simple deterministic embedding for testing. Use a real embedder in production."""
        rng = np.random.default_rng(hash(text) % (2**32))
        vec = rng.standard_normal(self._dim).astype(np.float32)
        return vec / (np.linalg.norm(vec) + 1e-8)
