"""
EngrammaMemory Async - Async-first interface for modern frameworks.

Full Phase 1-10 feature set with async/await.

Usage:
    from engramma_memory import EngrammaMemoryAsync

    async def main():
        mem = EngrammaMemoryAsync(dim=256, backend="cloud", api_key="nx_live_...")
        await mem.store(key=embedding, value=data)
        results = await mem.query(embedding, top_k=5)

        # Cloud-exclusive premium features
        explanation = await mem.explain(query)
        regime = await mem.get_current_regime()
        await mem.consolidate()
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
from numpy.typing import NDArray


class EngrammaMemoryAsync:
    """
    Async interface for Engramma Memory (cloud backend recommended).

    Parameters
    ----------
    dim : int
        Dimension of key/value vectors.
    backend : str
        "cloud" (recommended for async) or "local" (runs sync, no premium features).
    api_key : str, optional
        Required for cloud backend.
    """

    def __init__(
        self,
        dim: int,
        backend: str = "cloud",
        api_key: Optional[str] = None,
        max_patterns: int = 1000,
        **kwargs,
    ):
        self.dim = dim
        self.backend_name = backend
        self._is_local = backend == "local"

        if backend == "cloud":
            if not api_key:
                raise ValueError(
                    "Cloud backend requires an API key.\n"
                    "Get your free key: https://www.engramma-memory.com/signup"
                )
            from .backends.async_cloud import AsyncCloudBackend

            self._backend = AsyncCloudBackend(dim=dim, api_key=api_key, **kwargs)
        elif backend == "local":
            from .backends.local import LocalBackend

            self._backend = LocalBackend(dim=dim, max_patterns=max_patterns, **kwargs)
        else:
            raise ValueError(f"Unknown backend '{backend}'. Use 'local' or 'cloud'.")

    def _require_cloud(self, method_name: str):
        if self._is_local:
            raise RuntimeError(
                f"{method_name}() is a cloud-only feature.\n"
                f"Switch to cloud: EngrammaMemoryAsync(backend='cloud', api_key='nx_...')"
            )

    # ═══════════════════════════════════════════════════════════════════
    # CORE OPERATIONS
    # ═══════════════════════════════════════════════════════════════════

    async def store(
        self,
        key: Union[NDArray, List[float]],
        value: Union[NDArray, List[float], Any],
        metadata: Optional[Dict] = None,
    ) -> None:
        key = np.asarray(key, dtype=np.float32)
        if self._is_local:
            self._backend.store(key, value)
        else:
            await self._backend.store(key, value, metadata=metadata)

    async def query(
        self,
        query: Union[NDArray, List[float]],
        top_k: int = 1,
        filters: Optional[Dict] = None,
        use_phi_b: bool = False,
    ) -> List[Dict[str, Any]]:
        query = np.asarray(query, dtype=np.float32)
        if self._is_local:
            return self._backend.query(query, top_k=top_k)
        return await self._backend.query(query, top_k=top_k, filters=filters, use_phi_b=use_phi_b)

    async def retrieve(self, query: Union[NDArray, List[float]]) -> NDArray:
        query = np.asarray(query, dtype=np.float32)
        if self._is_local:
            return self._backend.retrieve(query)
        return await self._backend.retrieve(query)

    async def compose(
        self,
        keys: List[Union[NDArray, List[float]]],
        weights: Optional[List[float]] = None,
        mode: str = "attention",
    ) -> NDArray:
        keys_arr = [np.asarray(k, dtype=np.float32) for k in keys]
        if self._is_local:
            return self._backend.compose(keys_arr, weights)
        return await self._backend.compose(keys_arr, weights, mode=mode)

    async def forget(self, key: Union[NDArray, List[float]], strategy: str = "decay") -> None:
        key = np.asarray(key, dtype=np.float32)
        if self._is_local:
            self._backend.forget(key, strategy)
        else:
            await self._backend.forget(key, strategy)

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 1 — NEUROMODULATION
    # ═══════════════════════════════════════════════════════════════════

    async def get_modulation_state(self) -> Dict[str, Any]:
        """Get current neuromodulation state M(t). Cloud-only."""
        self._require_cloud("get_modulation_state")
        return await self._backend.get_modulation_state()

    async def get_surprise_history(self, window: int = 100) -> Dict[str, Any]:
        """Return surprise trajectory. Cloud-only."""
        self._require_cloud("get_surprise_history")
        return await self._backend.get_surprise_history(window)

    async def configure_neuromodulation(
        self, baseline: float = 0.5, sensitivity: float = 2.0, tau: float = 10.0
    ) -> Dict[str, Any]:
        """Fine-tune plasticity gate. Cloud-only."""
        self._require_cloud("configure_neuromodulation")
        return await self._backend.configure_neuromodulation(baseline, sensitivity, tau)

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 2 — PHI_B ROUTING
    # ═══════════════════════════════════════════════════════════════════

    async def get_router_trace(self, query: Union[NDArray, List[float]]) -> Dict[str, Any]:
        """Explain pathway selection for a query. Cloud-only."""
        self._require_cloud("get_router_trace")
        return await self._backend.get_router_trace(np.asarray(query, dtype=np.float32))

    async def set_routing_geometry(self, geometry: str = "phi_b") -> Dict[str, Any]:
        """Switch routing geometry. Cloud-only."""
        self._require_cloud("set_routing_geometry")
        return await self._backend.set_routing_geometry(geometry)

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 3 — EFE STRATEGIC ROUTING
    # ═══════════════════════════════════════════════════════════════════

    async def query_with_epistemic_weight(
        self,
        query: Union[NDArray, List[float]],
        epistemic_w: float = 1.0,
        pragmatic_w: float = 0.5,
        top_k: int = 1,
    ) -> List[Dict[str, Any]]:
        """Query with exploration/exploitation tradeoff. Cloud-only."""
        self._require_cloud("query_with_epistemic_weight")
        return await self._backend.query_with_epistemic_weight(
            np.asarray(query, dtype=np.float32), epistemic_w, pragmatic_w, top_k
        )

    async def get_efe_scores(self, query: Union[NDArray, List[float]]) -> Dict[str, Any]:
        """Get EFE scores per pathway. Cloud-only."""
        self._require_cloud("get_efe_scores")
        return await self._backend.get_efe_scores(np.asarray(query, dtype=np.float32))

    async def set_pathway_strategy(self, strategy: str = "balanced") -> Dict[str, Any]:
        """Set exploit/explore/balanced. Cloud-only."""
        self._require_cloud("set_pathway_strategy")
        return await self._backend.set_pathway_strategy(strategy)

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 4 — STDP
    # ═══════════════════════════════════════════════════════════════════

    async def get_head_specialization(self) -> Dict[str, Any]:
        """Which heads specialized for what. Cloud-only."""
        self._require_cloud("get_head_specialization")
        return await self._backend.get_head_specialization()

    async def get_head_temperatures(self) -> Dict[str, Any]:
        """Inspect head temperatures. Cloud-only."""
        self._require_cloud("get_head_temperatures")
        return await self._backend.get_head_temperatures()

    async def enable_stdp_learning(
        self, enabled: bool = True, eta: float = 0.01, tau: float = 5.0
    ) -> Dict[str, Any]:
        """Configure STDP plasticity. Cloud-only."""
        self._require_cloud("enable_stdp_learning")
        return await self._backend.enable_stdp_learning(enabled, eta, tau)

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 6 — CAUSAL & ADVANCED COMPOSITION
    # ═══════════════════════════════════════════════════════════════════

    async def compose_fractional(
        self,
        key_a: Union[NDArray, List[float]],
        key_b: Union[NDArray, List[float]],
        alpha: float = 0.5,
    ) -> NDArray:
        """Continuous interpolation between patterns. Cloud-only."""
        self._require_cloud("compose_fractional")
        return await self._backend.compose_fractional(
            np.asarray(key_a, dtype=np.float32), np.asarray(key_b, dtype=np.float32), alpha
        )

    async def get_causal_graph(self) -> Dict[str, Any]:
        """Get discovered causal DAG. Cloud-only."""
        self._require_cloud("get_causal_graph")
        return await self._backend.get_causal_graph()

    async def predict_causal_effect(
        self, cause_key: Union[NDArray, List[float]], effect_key: Union[NDArray, List[float]]
    ) -> Dict[str, Any]:
        """Predict causal effect of perturbing a pattern. Cloud-only."""
        self._require_cloud("predict_causal_effect")
        return await self._backend.predict_causal_effect(
            np.asarray(cause_key, dtype=np.float32),
            np.asarray(effect_key, dtype=np.float32),
        )

    async def is_confounded(
        self,
        key_a: Union[NDArray, List[float]],
        key_b: Union[NDArray, List[float]],
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """Detect hidden confounders. Cloud-only."""
        self._require_cloud("is_confounded")
        return await self._backend.is_confounded(
            np.asarray(key_a, dtype=np.float32), np.asarray(key_b, dtype=np.float32), threshold
        )

    async def enable_active_exploration(self, enabled: bool = True) -> Dict[str, Any]:
        """System autonomously probes uncertain relationships. Cloud-only."""
        self._require_cloud("enable_active_exploration")
        return await self._backend.enable_active_exploration(enabled)

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 7 — STRUCTURE DISCOVERY
    # ═══════════════════════════════════════════════════════════════════

    async def get_skeleton_entropy(self) -> Dict[str, Any]:
        """Quantify structural uncertainty. Cloud-only."""
        self._require_cloud("get_skeleton_entropy")
        return await self._backend.get_skeleton_entropy()

    async def get_uncertain_pairs(self, n_top: int = 10) -> Dict[str, Any]:
        """Which relationships need testing. Cloud-only."""
        self._require_cloud("get_uncertain_pairs")
        return await self._backend.get_uncertain_pairs(n_top)

    async def auto_select_thresholds(self) -> Dict[str, Any]:
        """Auto-calibrate thresholds via StARS. Cloud-only."""
        self._require_cloud("auto_select_thresholds")
        return await self._backend.auto_select_thresholds()

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 8 — SAFETY & CONSOLIDATION
    # ═══════════════════════════════════════════════════════════════════

    async def get_current_regime(self) -> Dict[str, Any]:
        """Get A/B/C regime status. Cloud-only."""
        self._require_cloud("get_current_regime")
        return await self._backend.get_current_regime()

    async def enable_anomaly_protection(self, enabled: bool = True) -> Dict[str, Any]:
        """Activate safety circuit. Cloud-only."""
        self._require_cloud("enable_anomaly_protection")
        return await self._backend.enable_anomaly_protection(enabled)

    async def consolidate(self) -> Dict[str, Any]:
        """Trigger Sleep/Wake consolidation. Cloud-only."""
        self._require_cloud("consolidate")
        return await self._backend.consolidate()

    async def set_regime_thresholds(
        self, theta_b: float = 1.0, theta_c: float = 2.0
    ) -> Dict[str, Any]:
        """Configure regime entry thresholds. Cloud-only."""
        self._require_cloud("set_regime_thresholds")
        return await self._backend.set_regime_thresholds(theta_b, theta_c)

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 9 — TEMPORAL CAUSALITY
    # ═══════════════════════════════════════════════════════════════════

    async def test_granger_causality(
        self,
        key_a: Union[NDArray, List[float]],
        key_b: Union[NDArray, List[float]],
        max_lag: int = 10,
    ) -> Dict[str, Any]:
        """Test temporal Granger causality. Cloud-only."""
        self._require_cloud("test_granger_causality")
        return await self._backend.test_granger_causality(
            np.asarray(key_a, dtype=np.float32), np.asarray(key_b, dtype=np.float32), max_lag
        )

    async def get_causal_predictions(
        self, query: Union[NDArray, List[float]], n_predictions: int = 3
    ) -> Dict[str, Any]:
        """Predict next pattern accesses. Cloud-only."""
        self._require_cloud("get_causal_predictions")
        return await self._backend.get_causal_predictions(
            np.asarray(query, dtype=np.float32), n_predictions
        )

    async def enable_prefetch(self, enabled: bool = True) -> Dict[str, Any]:
        """Enable predictive prefetching. Cloud-only."""
        self._require_cloud("enable_prefetch")
        return await self._backend.enable_prefetch(enabled)

    async def get_prefetch_hit_rate(self) -> Dict[str, Any]:
        """Cache hit rate from predictions. Cloud-only."""
        self._require_cloud("get_prefetch_hit_rate")
        return await self._backend.get_prefetch_hit_rate()

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 10 — TEXT, TIERS, XAI
    # ═══════════════════════════════════════════════════════════════════

    async def store_text(
        self, text: str, value_embedding: Optional[NDArray] = None, metadata: Optional[Dict] = None
    ) -> None:
        """Store semantic text (HDC auto-encoding). Cloud-only."""
        self._require_cloud("store_text")
        await self._backend.store_text(text, value_embedding, metadata)

    async def query_text(
        self, query_text: str, top_k: int = 5, filters: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """Query by natural language. Cloud-only."""
        self._require_cloud("query_text")
        return await self._backend.query_text(query_text, top_k, filters)

    async def compose_text(
        self, texts: List[str], weights: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """Compose blends of text patterns. Cloud-only."""
        self._require_cloud("compose_text")
        return await self._backend.compose_text(texts, weights)

    async def get_text_encoding(self, text: str) -> NDArray:
        """Get HDC encoding of text. Cloud-only."""
        self._require_cloud("get_text_encoding")
        return await self._backend.get_text_encoding(text)

    async def move_to_tier(
        self, key: Union[NDArray, List[float]], tier: str = "warm"
    ) -> Dict[str, Any]:
        """Move pattern between hot/warm/cold tiers. Cloud-only."""
        self._require_cloud("move_to_tier")
        return await self._backend.move_to_tier(np.asarray(key, dtype=np.float32), tier)

    async def get_tier_stats(self) -> Dict[str, Any]:
        """Storage tier statistics. Cloud-only."""
        self._require_cloud("get_tier_stats")
        return await self._backend.get_tier_stats()

    async def explain(self, query: Union[NDArray, List[float]]) -> Dict[str, Any]:
        """Full XAI explanation of retrieval. Cloud-only."""
        self._require_cloud("explain")
        return await self._backend.explain(np.asarray(query, dtype=np.float32))

    async def get_xai_dashboard(self) -> Dict[str, Any]:
        """Full dashboard data for visualization. Cloud-only."""
        self._require_cloud("get_xai_dashboard")
        return await self._backend.get_xai_dashboard()

    async def snapshot(self, name: Optional[str] = None) -> Dict[str, Any]:
        """Create persistent memory snapshot. Cloud-only."""
        self._require_cloud("snapshot")
        return await self._backend.snapshot(name)

    async def restore(self, snapshot_id: str) -> Dict[str, Any]:
        """Restore from snapshot. Cloud-only."""
        self._require_cloud("restore")
        return await self._backend.restore(snapshot_id)

    async def analytics(self, period: str = "24h") -> Dict[str, Any]:
        """Usage analytics. Cloud-only."""
        self._require_cloud("analytics")
        return await self._backend.analytics(period)

    # ═══════════════════════════════════════════════════════════════════
    # LIFECYCLE
    # ═══════════════════════════════════════════════════════════════════

    async def close(self):
        """Close the async HTTP client."""
        if not self._is_local and hasattr(self._backend, "close"):
            await self._backend.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()
