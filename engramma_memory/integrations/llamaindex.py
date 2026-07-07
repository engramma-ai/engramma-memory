"""
LlamaIndex integration for Engramma Memory.

Implements BaseRetriever interface for use with LlamaIndex query engines.

Usage:
    from engramma_memory.integrations.llamaindex import EngrammaRetriever

    retriever = EngrammaRetriever(dim=256, top_k=5)
    query_engine = RetrieverQueryEngine(retriever=retriever)
"""

from typing import Any, List, Optional

import numpy as np

from ..core import EngrammaMemory

try:
    from llama_index.core.retrievers import BaseRetriever
    from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

    _HAS_LLAMAINDEX = True
except ImportError:
    _HAS_LLAMAINDEX = False

    class BaseRetriever:
        pass

    class NodeWithScore:
        pass

    class TextNode:
        pass

    class QueryBundle:
        pass


class EngrammaRetriever(BaseRetriever):
    """
    LlamaIndex-compatible retriever backed by Engramma Memory.

    Stores document embeddings and retrieves using Engramma's hybrid
    architecture (exact + energy + multi-head composition).

    Parameters
    ----------
    dim : int
        Embedding dimension.
    top_k : int
        Number of nodes to retrieve.
    embed_fn : callable, optional
        Text-to-embedding function.
    backend : str
        "local" or "cloud".
    api_key : str, optional
        API key for cloud backend.
    """

    def __init__(
        self,
        dim: int = 256,
        top_k: int = 5,
        embed_fn=None,
        backend: str = "local",
        api_key: Optional[str] = None,
        **kwargs,
    ):
        if not _HAS_LLAMAINDEX:
            raise ImportError(
                "LlamaIndex integration requires llama-index-core. "
                "Install with: pip install engramma-memory[llamaindex]"
            )
        super().__init__(**kwargs)
        self._dim = dim
        self._top_k = top_k
        self._embed_fn = embed_fn
        self._engramma = EngrammaMemory(dim=dim, backend=backend, api_key=api_key)
        self._texts: List[str] = []
        self._embeddings: List[np.ndarray] = []

    def add_documents(self, texts: List[str], embeddings: List[np.ndarray]) -> None:
        """Add documents with their embeddings to the Engramma memory."""
        for text, emb in zip(texts, embeddings):
            emb = np.asarray(emb, dtype=np.float32)
            self._engramma.store(key=emb, value=emb)
            self._texts.append(text)
            self._embeddings.append(emb)

    def _retrieve(self, query_bundle: Any) -> List[Any]:
        if self._engramma.count == 0:
            return []

        if hasattr(query_bundle, "embedding") and query_bundle.embedding is not None:
            query_emb = np.asarray(query_bundle.embedding, dtype=np.float32)
        elif self._embed_fn and hasattr(query_bundle, "query_str"):
            query_emb = self._embed_fn(query_bundle.query_str)
        else:
            return []

        results = self._engramma.query(query_emb, top_k=self._top_k)

        nodes = []
        for r in results:
            value = r["value"]
            score = r["score"]

            best_idx = self._find_closest_text(value)
            if best_idx is not None:
                text = self._texts[best_idx]
            else:
                text = ""

            node = TextNode(text=text)
            nodes.append(NodeWithScore(node=node, score=score))

        return nodes

    def _find_closest_text(self, value: np.ndarray) -> Optional[int]:
        if not self._embeddings:
            return None
        value = value.flatten()[: self._dim]
        dists = [np.linalg.norm(e.flatten()[: self._dim] - value) for e in self._embeddings]
        best = int(np.argmin(dists))
        return best if dists[best] < 2.0 else None


class EngrammaVectorStore:
    """
    Simplified vector store interface for LlamaIndex.

    Use this when you want Engramma as a drop-in replacement for
    ChromaVectorStore or FaissVectorStore.
    """

    def __init__(self, dim: int = 256, backend: str = "local", api_key: Optional[str] = None):
        self._engramma = EngrammaMemory(dim=dim, backend=backend, api_key=api_key)
        self._dim = dim
        self._nodes: List[Any] = []

    def add(self, nodes: List[Any]) -> List[str]:
        ids = []
        for node in nodes:
            if hasattr(node, "embedding") and node.embedding is not None:
                emb = np.asarray(node.embedding, dtype=np.float32)
                self._engramma.store(key=emb, value=emb)
                self._nodes.append(node)
                ids.append(getattr(node, "node_id", str(len(self._nodes))))
        return ids

    def query(self, query_embedding: List[float], top_k: int = 5) -> List[Any]:
        emb = np.asarray(query_embedding, dtype=np.float32)
        return self._engramma.query(emb, top_k=top_k)
