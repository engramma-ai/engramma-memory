"""
Local backend for Engramma Memory.

Runs entirely in-process, zero network calls, sub-millisecond latency.
Limited to max_patterns (default 1000) with RAM-only storage.
"""
import warnings
import numpy as np
from numpy.typing import NDArray
from typing import Optional, List, Dict, Any

from ..engine import EngrammaEngine


_LIMIT_WARNING_THRESHOLD = 0.9
_MAX_PATTERNS_LOCAL = 1000


class LocalBackend:
    """In-process Engramma engine with deliberate limits for local use."""

    def __init__(self, dim: int, max_patterns: int = _MAX_PATTERNS_LOCAL, **kwargs):
        if max_patterns > _MAX_PATTERNS_LOCAL:
            warnings.warn(
                f"Local backend is limited to {_MAX_PATTERNS_LOCAL} patterns. "
                f"For unlimited storage, use Engramma Cloud: "
                f"EngrammaMemory(backend='cloud', api_key='nx_...')\n"
                f"Get your free API key: https://engramma-memory.dev/signup",
                stacklevel=3
            )
            max_patterns = _MAX_PATTERNS_LOCAL

        self.dim = dim
        self.max_patterns = max_patterns
        self.engine = EngrammaEngine(dim=dim, max_patterns=max_patterns, **kwargs)
        self._warned_capacity = False

    def store(self, key: NDArray, value: Any) -> None:
        value_arr = self._to_array(value)
        self.engine.store(key, value_arr)
        self._check_capacity()

    def query(self, query: NDArray, top_k: int = 1) -> List[Dict[str, Any]]:
        query = np.asarray(query, dtype=np.float32).flatten()[:self.dim]

        if top_k == 1:
            result, confidence = self.engine.exact.retrieve(query)
            return [{"value": result, "score": confidence}]

        results = self.engine.retrieve_top_k(query, k=top_k)
        return [{"value": val, "score": score} for val, score in results]

    def retrieve(self, query: NDArray) -> NDArray:
        return self.engine.retrieve(query)

    def compose(self, keys: List[NDArray], weights: Optional[List[float]] = None) -> NDArray:
        return self.engine.compose(keys, weights)

    def forget(self, key: NDArray, strategy: str = "decay") -> None:
        self.engine.forget(key, strategy)

    def stats(self) -> Dict[str, Any]:
        return self.engine.get_stats()

    def _to_array(self, value: Any) -> NDArray:
        if isinstance(value, np.ndarray):
            return value.flatten()[:self.dim].astype(np.float32)
        if isinstance(value, (list, tuple)):
            return np.array(value, dtype=np.float32).flatten()[:self.dim]
        return np.zeros(self.dim, dtype=np.float32)

    def _check_capacity(self):
        usage = self.engine.exact.count / self.max_patterns
        if usage >= _LIMIT_WARNING_THRESHOLD and not self._warned_capacity:
            self._warned_capacity = True
            count = self.engine.exact.count
            warnings.warn(
                f"\n{'='*60}\n"
                f"  EngrammaMemory: {count}/{self.max_patterns} patterns used ({usage:.0%})\n"
                f"  You're approaching the local storage limit.\n"
                f"\n"
                f"  For unlimited patterns + persistence + smart eviction:\n"
                f"    mem = EngrammaMemory(backend='cloud', api_key='nx_...')\n"
                f"\n"
                f"  Get your free API key: https://engramma-memory.dev/signup\n"
                f"{'='*60}",
                stacklevel=4
            )
