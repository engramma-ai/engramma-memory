"""
Engramma Memory Engine v2 (Public Edition)

Hybrid architecture combining:
  1. Exact kNN Memory - perfect recall for stored patterns
  2. Energy-based Memory - soft generalization via temperature-scaled softmax
  3. Multi-Head Attention Memory - native composition via simultaneous attention

All learning is LOCAL. No GPU required. Sub-millisecond retrieval.
"""
import logging
import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Optional, Dict, List

logger = logging.getLogger(__name__)


class ExactMemory:
    """Exact kNN memory with importance-based eviction."""

    def __init__(self, key_dim: int, value_dim: int, max_entries: int = 1000):
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.max_entries = max_entries

        self.keys = np.zeros((max_entries, key_dim), dtype=np.float32)
        self.values = np.zeros((max_entries, value_dim), dtype=np.float32)
        self.active = np.zeros(max_entries, dtype=bool)
        self.access_count = np.zeros(max_entries, dtype=np.float32)
        self.timestamps = np.zeros(max_entries, dtype=np.float32)
        self.n = 0
        self.ops_count = 0

    def store(self, key: NDArray, value: NDArray):
        key = key.flatten()[:self.key_dim].astype(np.float32)
        value = value.flatten()[:self.value_dim].astype(np.float32)

        if self.n > 0:
            active_idx = np.where(self.active)[0]
            dists = np.linalg.norm(self.keys[active_idx] - key, axis=1)
            if len(dists) > 0 and np.min(dists) < 0.1:
                best = active_idx[np.argmin(dists)]
                self.values[best] = value
                self.access_count[best] += 1
                self.timestamps[best] = self.ops_count
                self.ops_count += 1
                return

        if self.n < self.max_entries:
            idx = self.n
        else:
            scores = self._importance_scores()
            scores[~self.active] = float('inf')
            idx = int(np.argmin(scores))

        self.keys[idx] = key
        self.values[idx] = value
        self.active[idx] = True
        self.access_count[idx] = 1.0
        self.timestamps[idx] = self.ops_count
        self.n = min(self.n + 1, self.max_entries)
        self.ops_count += 1

    def retrieve(self, query: NDArray, k: int = 1) -> Tuple[NDArray, float]:
        if self.n == 0:
            return np.zeros(self.value_dim, dtype=np.float32), 0.0

        query = query.flatten()[:self.key_dim].astype(np.float32)
        active_idx = np.where(self.active)[0]
        dists = np.linalg.norm(self.keys[active_idx] - query, axis=1)
        best_local = int(np.argmin(dists))
        best_idx = active_idx[best_local]

        self.access_count[best_idx] += 1
        confidence = float(np.exp(-dists[best_local]))

        if k == 1:
            return self.values[best_idx].copy(), confidence
        else:
            top_k_local = np.argsort(dists)[:k]
            top_k_idx = active_idx[top_k_local]
            weights = np.exp(-dists[top_k_local])
            weights /= weights.sum() + 1e-8
            result = (weights[:, None] * self.values[top_k_idx]).sum(axis=0)
            return result, float(np.max(weights))

    def retrieve_top_k(self, query: NDArray, k: int = 5) -> List[Tuple[NDArray, float]]:
        if self.n == 0:
            return []

        query = query.flatten()[:self.key_dim].astype(np.float32)
        active_idx = np.where(self.active)[0]
        dists = np.linalg.norm(self.keys[active_idx] - query, axis=1)
        top_k_local = np.argsort(dists)[:k]

        results = []
        for local_idx in top_k_local:
            idx = active_idx[local_idx]
            self.access_count[idx] += 0.1
            conf = float(np.exp(-dists[local_idx]))
            results.append((self.values[idx].copy(), conf))
        return results

    def _importance_scores(self) -> NDArray:
        recency = np.exp(-0.01 * (self.ops_count - self.timestamps))
        return self.access_count * recency

    @property
    def count(self) -> int:
        return int(np.sum(self.active))


class EnergyMemory:
    """Energy-based memory with temperature-scaled softmax retrieval."""

    def __init__(self, key_dim: int, value_dim: int, max_patterns: int = 1000,
                 beta: float = 4.0):
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.max_patterns = max_patterns
        self.beta = beta

        self.keys = np.zeros((max_patterns, key_dim), dtype=np.float32)
        self.values = np.zeros((max_patterns, value_dim), dtype=np.float32)
        self.active = np.zeros(max_patterns, dtype=bool)
        self.importance = np.zeros(max_patterns, dtype=np.float32)
        self.n = 0

    def store(self, key: NDArray, value: NDArray):
        key = key.flatten()[:self.key_dim].astype(np.float32)
        value = value.flatten()[:self.value_dim].astype(np.float32)
        key_norm = key / (np.linalg.norm(key) + 1e-8)

        if self.n < self.max_patterns:
            idx = self.n
        else:
            active_idx = np.where(self.active)[0]
            idx = active_idx[int(np.argmin(self.importance[active_idx]))]

        self.keys[idx] = key_norm
        self.values[idx] = value
        self.active[idx] = True
        self.importance[idx] = 1.0
        self.n = min(self.n + 1, self.max_patterns)

    def retrieve(self, query: NDArray) -> Tuple[NDArray, float]:
        if self.n == 0:
            return np.zeros(self.value_dim, dtype=np.float32), 0.0

        query = query.flatten()[:self.key_dim].astype(np.float32)
        query_norm = query / (np.linalg.norm(query) + 1e-8)

        active_idx = np.where(self.active)[0]
        sims = self.beta * (self.keys[active_idx] @ query_norm)
        sims -= np.max(sims)
        weights = np.exp(sims)
        weights /= weights.sum() + 1e-8

        result = weights @ self.values[active_idx]
        confidence = float(np.max(weights))

        top_idx = active_idx[int(np.argmax(weights))]
        self.importance[top_idx] += 0.1

        return result, confidence


class MultiHeadAttentionMemory:
    """
    Multi-head attention over stored patterns for native composition.

    Each head uses full-dim projections with different temperatures:
    - Sharp heads (high temp): recall single patterns precisely
    - Soft heads (low temp): blend neighbors for composition

    When head 1 attends to pattern A and head 2 attends to pattern B,
    the weighted output is approximately (A+B)/2 = native composition.
    """

    def __init__(self, dim: int, n_heads: int = 4, max_patterns: int = 1000):
        self.dim = dim
        self.n_heads = n_heads
        self.max_patterns = max_patterns

        self.W_q = np.zeros((n_heads, dim, dim), dtype=np.float32)
        self.W_k = np.zeros((n_heads, dim, dim), dtype=np.float32)
        rng = np.random.default_rng(42)

        for h in range(n_heads):
            self.W_q[h] = np.eye(dim, dtype=np.float32)
            self.W_k[h] = np.eye(dim, dtype=np.float32)
            self.W_q[h] += rng.standard_normal((dim, dim)).astype(np.float32) * 0.05
            self.W_k[h] += rng.standard_normal((dim, dim)).astype(np.float32) * 0.05

        self.keys = np.zeros((max_patterns, dim), dtype=np.float32)
        self.values = np.zeros((max_patterns, dim), dtype=np.float32)
        self.n = 0

        self.temperatures = np.linspace(2.0, 8.0, n_heads).astype(np.float32)
        self.head_weights = np.ones(n_heads, dtype=np.float32) / n_heads

    def store(self, key: NDArray, value: NDArray):
        key = key.flatten()[:self.dim].astype(np.float32)
        value = value.flatten()[:self.dim].astype(np.float32)

        if self.n < self.max_patterns:
            idx = self.n
            self.n += 1
        else:
            idx = self.n % self.max_patterns

        self.keys[idx] = key / (np.linalg.norm(key) + 1e-8)
        self.values[idx] = value

    def attend(self, query: NDArray) -> Tuple[NDArray, float]:
        if self.n == 0:
            return np.zeros(self.dim, dtype=np.float32), 0.0

        query = query.flatten()[:self.dim].astype(np.float32)
        n = min(self.n, self.max_patterns)
        stored_keys = self.keys[:n]
        stored_values = self.values[:n]

        head_outputs = []
        total_max_weight = 0.0

        for h in range(self.n_heads):
            Q = query @ self.W_q[h]
            K = stored_keys @ self.W_k[h]

            scores = self.temperatures[h] * (Q @ K.T) / np.sqrt(self.dim)
            scores -= np.max(scores)
            weights = np.exp(scores)
            weights /= weights.sum() + 1e-8

            head_out = weights @ stored_values
            head_outputs.append(head_out)
            total_max_weight += float(np.max(weights))

        result = np.zeros(self.dim, dtype=np.float32)
        for h in range(self.n_heads):
            result += self.head_weights[h] * head_outputs[h]

        confidence = total_max_weight / self.n_heads
        return result, confidence

    def attend_compositional(self, query_a: NDArray, query_b: NDArray) -> Tuple[NDArray, float]:
        """Split-query attention: half heads attend via A, half via B."""
        if self.n == 0:
            return np.zeros(self.dim, dtype=np.float32), 0.0

        query_a = query_a.flatten()[:self.dim].astype(np.float32)
        query_b = query_b.flatten()[:self.dim].astype(np.float32)
        n = min(self.n, self.max_patterns)
        stored_keys = self.keys[:n]
        stored_values = self.values[:n]

        head_outputs = []
        head_confidences = []
        mid = self.n_heads // 2

        for h in range(self.n_heads):
            q = query_a if h < mid else query_b
            Q = q @ self.W_q[h]
            K = stored_keys @ self.W_k[h]

            temp = max(self.temperatures[h], 5.0)
            scores = temp * (Q @ K.T) / np.sqrt(self.dim)
            scores -= np.max(scores)
            weights = np.exp(scores)
            weights /= weights.sum() + 1e-8

            head_out = weights @ stored_values
            head_outputs.append(head_out)
            head_confidences.append(float(np.max(weights)))

        head_conf_arr = np.array(head_confidences, dtype=np.float32)
        head_conf_arr /= head_conf_arr.sum() + 1e-8
        result = np.zeros(self.dim, dtype=np.float32)
        for h in range(self.n_heads):
            result += head_conf_arr[h] * head_outputs[h]

        avg_norm = np.mean([np.linalg.norm(h) for h in head_outputs])
        result_norm = np.linalg.norm(result)
        if result_norm > 1e-10:
            result = result * (avg_norm / result_norm)

        return result, float(np.mean(head_confidences))


class ConfidenceRouter:
    """Routes queries to the best pathway based on confidence signals."""

    def __init__(self):
        self.success_ema = np.array([0.7, 0.5, 0.5], dtype=np.float32)

    def route(self, exact_conf: float, energy_conf: float, attn_conf: float) -> NDArray:
        raw = np.array([exact_conf, energy_conf, attn_conf], dtype=np.float32)
        weighted = raw * self.success_ema
        weighted -= np.max(weighted)
        weights = np.exp(5.0 * weighted)
        weights /= weights.sum() + 1e-8
        return weights

    def update(self, pathway_idx: int, confidence: float):
        lr = 0.05
        self.success_ema[pathway_idx] = (
            (1 - lr) * self.success_ema[pathway_idx] + lr * confidence
        )


class EngrammaEngine:
    """
    Engramma Memory Engine v2 - the core hybrid architecture.

    Three pathways sharing the same stored patterns:
      1. EXACT (kNN) - perfect recall for exact matches
      2. ENERGY (softmax) - soft generalization
      3. MULTI-HEAD ATTENTION - native composition

    The key insight: multi-head attention allows each head to attend to
    DIFFERENT stored patterns simultaneously. This enables native composition
    that vector databases cannot achieve.
    """

    def __init__(self, dim: int, max_patterns: int = 1000, beta: float = 4.0,
                 n_heads: int = 4):
        self.dim = dim
        self.max_patterns = max_patterns

        self.exact = ExactMemory(dim, dim, max_patterns)
        self.energy = EnergyMemory(dim, dim, max_patterns, beta)
        self.attention = MultiHeadAttentionMemory(dim, n_heads, max_patterns)
        self.router = ConfidenceRouter()
        self.n_stored = 0

    def store(self, key: NDArray, value: NDArray):
        key = np.asarray(key, dtype=np.float32).flatten()[:self.dim]
        value = np.asarray(value, dtype=np.float32).flatten()[:self.dim]
        self.exact.store(key, value)
        self.energy.store(key, value)
        self.attention.store(key, value)
        self.n_stored += 1

    def retrieve(self, query: NDArray) -> NDArray:
        query = np.asarray(query, dtype=np.float32).flatten()[:self.dim]

        exact_result, exact_conf = self.exact.retrieve(query)

        if exact_conf > 0.95:
            self.router.update(0, exact_conf)
            return exact_result

        energy_result, energy_conf = self.energy.retrieve(query)
        attn_result, attn_conf = self.attention.attend(query)

        if exact_conf > 0.7:
            alpha = exact_conf
            return alpha * exact_result + (1 - alpha) * energy_result

        weights = self.router.route(exact_conf, energy_conf, attn_conf)
        result = (weights[0] * exact_result +
                  weights[1] * energy_result +
                  weights[2] * attn_result)
        return result

    def retrieve_top_k(self, query: NDArray, k: int = 5) -> List[Tuple[NDArray, float]]:
        query = np.asarray(query, dtype=np.float32).flatten()[:self.dim]
        return self.exact.retrieve_top_k(query, k)

    def compose(self, keys: List[NDArray], weights: Optional[List[float]] = None) -> NDArray:
        """
        Compositional retrieval: blend multiple stored patterns.

        In the free version, only equal-weight blending is supported.
        For weighted composition, upgrade to Engramma Cloud.
        """
        if len(keys) < 2:
            if keys:
                return self.retrieve(keys[0])
            return np.zeros(self.dim, dtype=np.float32)

        keys_arr = [np.asarray(k, dtype=np.float32).flatten()[:self.dim] for k in keys]

        if weights is not None:
            # Local version: only 50/50 blend supported
            w = np.array(weights, dtype=np.float32)
            if not np.allclose(w, w[0]):
                import warnings
                warnings.warn(
                    "Weighted composition is limited in the local version. "
                    "All weights will be treated as equal. "
                    "For custom weights, upgrade to Engramma Cloud: "
                    "mem = EngrammaMemory(backend='cloud', api_key='nx_...')",
                    stacklevel=2
                )

        if len(keys_arr) == 2:
            return self.attention.attend_compositional(keys_arr[0], keys_arr[1])[0]

        # For 3+ keys: pairwise composition then merge
        results = []
        for i in range(0, len(keys_arr) - 1, 2):
            r, _ = self.attention.attend_compositional(keys_arr[i], keys_arr[i + 1])
            results.append(r)
        if len(keys_arr) % 2 == 1:
            results.append(self.retrieve(keys_arr[-1]))

        combined = np.mean(results, axis=0)
        return combined

    def forget(self, key: NDArray, strategy: str = "decay"):
        """Remove or decay a pattern from memory."""
        key = np.asarray(key, dtype=np.float32).flatten()[:self.dim]

        if strategy == "immediate":
            active_idx = np.where(self.exact.active)[0]
            if len(active_idx) == 0:
                return
            dists = np.linalg.norm(self.exact.keys[active_idx] - key, axis=1)
            best = active_idx[int(np.argmin(dists))]
            if dists[int(np.argmin(dists))] < 0.5:
                self.exact.active[best] = False
                self.exact.access_count[best] = 0
        elif strategy == "decay":
            active_idx = np.where(self.exact.active)[0]
            if len(active_idx) == 0:
                return
            dists = np.linalg.norm(self.exact.keys[active_idx] - key, axis=1)
            best = active_idx[int(np.argmin(dists))]
            if dists[int(np.argmin(dists))] < 0.5:
                self.exact.access_count[best] *= 0.1

    def get_stats(self) -> Dict:
        return {
            "n_stored": self.n_stored,
            "exact_count": self.exact.count,
            "max_patterns": self.max_patterns,
            "dim": self.dim,
            "usage_ratio": self.exact.count / self.max_patterns,
        }
