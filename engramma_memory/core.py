"""
EngrammaMemory - The public API.

One interface, two backends. Switching from local to cloud is one line.
"""
import numpy as np
from numpy.typing import NDArray
from typing import Optional, List, Dict, Any, Union


class EngrammaMemory:
    """
    A composable memory engine for AI systems.

    Parameters
    ----------
    dim : int
        Dimension of key/value vectors.
    backend : str
        "local" (default, free, open source) or "cloud" (production, paid).
    api_key : str, optional
        Required for cloud backend. Get yours at https://engramma-memory.dev/signup
    max_patterns : int, optional
        Maximum stored patterns. Local is capped at 1000.
    **kwargs
        Additional backend-specific options.

    Examples
    --------
    >>> from engramma_memory import EngrammaMemory
    >>> mem = EngrammaMemory(dim=256, backend="local")
    >>> mem.store(key=embedding, value=data)
    >>> result = mem.query(embedding, top_k=3)

    To switch to cloud (one line change):
    >>> mem = EngrammaMemory(dim=256, backend="cloud", api_key="nx_live_...")
    """

    def __init__(self, dim: int, backend: str = "local",
                 api_key: Optional[str] = None,
                 max_patterns: int = 1000,
                 **kwargs):
        self.dim = dim
        self.backend_name = backend

        if backend == "local":
            from .backends.local import LocalBackend
            self._backend = LocalBackend(dim=dim, max_patterns=max_patterns, **kwargs)

        elif backend == "cloud":
            if not api_key:
                raise ValueError(
                    "Cloud backend requires an API key.\n"
                    "Get your free key: https://engramma-memory.dev/signup\n"
                    "Usage: EngrammaMemory(dim=256, backend='cloud', api_key='nx_live_...')"
                )
            from .backends.cloud import CloudBackend
            self._backend = CloudBackend(dim=dim, api_key=api_key, **kwargs)

        else:
            raise ValueError(
                f"Unknown backend '{backend}'. Use 'local' or 'cloud'."
            )

    def store(self, key: Union[NDArray, List[float]],
              value: Union[NDArray, List[float], Any],
              metadata: Optional[Dict] = None) -> None:
        """
        Store a key-value pair in memory.

        Parameters
        ----------
        key : array-like
            The embedding vector used for retrieval.
        value : array-like
            The data to store (typically an embedding or payload).
        metadata : dict, optional
            Additional metadata (cloud backend only).
        """
        key = np.asarray(key, dtype=np.float32)
        if hasattr(self._backend, 'store'):
            if metadata and self.backend_name == "cloud":
                self._backend.store(key, value, metadata=metadata)
            else:
                self._backend.store(key, value)

    def query(self, query: Union[NDArray, List[float]],
              top_k: int = 1,
              filters: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Query memory for the most relevant stored patterns.

        Parameters
        ----------
        query : array-like
            The query embedding vector.
        top_k : int
            Number of results to return.
        filters : dict, optional
            Metadata filters (cloud backend only).

        Returns
        -------
        list of dict
            Each dict contains 'value' (NDArray) and 'score' (float).
        """
        query = np.asarray(query, dtype=np.float32)
        if filters and self.backend_name == "cloud":
            return self._backend.query(query, top_k=top_k, filters=filters)
        return self._backend.query(query, top_k=top_k)

    def retrieve(self, query: Union[NDArray, List[float]]) -> NDArray:
        """
        Retrieve the best-matching pattern (smart routing).

        Uses confidence-based routing between exact recall, soft generalization,
        and multi-head composition pathways.

        Parameters
        ----------
        query : array-like
            The query embedding vector.

        Returns
        -------
        NDArray
            The retrieved/composed value vector.
        """
        query = np.asarray(query, dtype=np.float32)
        return self._backend.retrieve(query)

    def compose(self, keys: List[Union[NDArray, List[float]]],
                weights: Optional[List[float]] = None) -> NDArray:
        """
        Compose multiple stored patterns into a blended result.

        This is Engramma's key differentiator: native compositional memory.
        Vector databases require manual retrieval + blending; Engramma does it
        natively through multi-head attention.

        Parameters
        ----------
        keys : list of array-like
            The component keys to compose.
        weights : list of float, optional
            Composition weights. Local version supports equal weights only.
            For weighted composition, use Engramma Cloud.

        Returns
        -------
        NDArray
            The composed value vector.
        """
        keys_arr = [np.asarray(k, dtype=np.float32) for k in keys]
        return self._backend.compose(keys_arr, weights)

    def forget(self, key: Union[NDArray, List[float]],
               strategy: str = "decay") -> None:
        """
        Remove or decay a pattern from memory.

        Parameters
        ----------
        key : array-like
            The key of the pattern to forget.
        strategy : str
            "decay" (reduce importance) or "immediate" (delete now).
        """
        key = np.asarray(key, dtype=np.float32)
        self._backend.forget(key, strategy)

    def stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        return self._backend.stats()

    @property
    def count(self) -> int:
        """Number of active patterns in memory."""
        s = self.stats()
        return s.get("exact_count", s.get("n_stored", 0))

    def __repr__(self) -> str:
        return (
            f"EngrammaMemory(dim={self.dim}, backend='{self.backend_name}', "
            f"count={self.count})"
        )
