"""
Cloud backend for Engramma Memory.

Production-grade memory engine powered by NEXUSCompose v4 (Phases 1-10):

  Core:
    - Unlimited pattern storage (tiered: hot/warm/cold)
    - Persistent memory across restarts
    - Active Inference routing (Phase 1)
    - phi_B geometric routing (Phase 2)
    - EFE strategic pathway selection (Phase 3)
    - STDP temporal plasticity (Phase 4)

  Advanced (Phases 6-10):
    - Fractional composition with causal graph
    - Active structure discovery
    - Regime detection & anomaly safety
    - Temporal causality & predictive prefetch
    - Text interface & XAI dashboard
"""
import time
import logging
import numpy as np
from numpy.typing import NDArray
from typing import Optional, List, Dict, Any


logger = logging.getLogger(__name__)

_CLOUD_API_BASE = "https://api.engramma-memory.dev/v1"
_MAX_RETRIES = 3
_BACKOFF_BASE = 0.5


class CloudBackend:
    """Engramma Cloud backend - production-grade memory engine (v4)."""

    def __init__(self, dim: int, api_key: str, endpoint: Optional[str] = None,
                 max_retries: int = _MAX_RETRIES, timeout: float = 30.0,
                 **kwargs):
        if not api_key or not api_key.startswith("nx_"):
            raise ValueError(
                "Invalid API key. Get your key at: https://engramma-memory.dev/signup\n"
                "Keys start with 'nx_live_' (production) or 'nx_test_' (sandbox)."
            )

        self.dim = dim
        self.api_key = api_key
        self.endpoint = endpoint or _CLOUD_API_BASE
        self.max_retries = max_retries
        self.timeout = timeout
        self._session_id: Optional[str] = None

        try:
            import httpx
            self._client = httpx.Client(
                base_url=self.endpoint,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "X-Engramma-Dim": str(dim),
                    "X-Engramma-SDK": "python/0.1.0",
                },
                timeout=timeout,
            )
            self._http_lib = "httpx"
        except ImportError:
            try:
                import requests
                self._client = None
                self._requests = requests
                self._headers = {
                    "Authorization": f"Bearer {api_key}",
                    "X-Engramma-Dim": str(dim),
                    "X-Engramma-SDK": "python/0.1.0",
                }
                self._http_lib = "requests"
            except ImportError:
                raise ImportError(
                    "Cloud backend requires 'httpx' or 'requests'. "
                    "Install with: pip install engramma-memory[cloud]"
                )

    # ═══════════════════════════════════════════════════════════════════
    # CORE MEMORY OPERATIONS
    # ═══════════════════════════════════════════════════════════════════

    def store(self, key: NDArray, value: Any, metadata: Optional[Dict] = None) -> None:
        payload = {
            "key": key.flatten().tolist(),
            "value": self._serialize_value(value),
        }
        if metadata:
            payload["metadata"] = metadata
        self._post("/memory/store", payload)

    def query(self, query: NDArray, top_k: int = 1,
              filters: Optional[Dict] = None,
              use_phi_b: bool = False) -> List[Dict[str, Any]]:
        """
        Query memory with optional phi_B geometric routing (Phase 2).

        Parameters
        ----------
        use_phi_b : bool
            Use information-theoretic Hellinger routing instead of
            cosine confidence. More accurate for ambiguous queries.
        """
        payload = {
            "query": query.flatten().tolist(),
            "top_k": top_k,
            "use_phi_b": use_phi_b,
        }
        if filters:
            payload["filters"] = filters
        response = self._post("/memory/query", payload)
        return response.get("results", [])

    def retrieve(self, query: NDArray) -> NDArray:
        """Smart retrieval with Active Inference routing (Phase 1+3)."""
        payload = {
            "query": query.flatten().tolist(),
            "mode": "active_inference",
        }
        response = self._post("/memory/retrieve", payload)
        return np.array(response.get("result", np.zeros(self.dim).tolist()),
                        dtype=np.float32)

    def compose(self, keys: List[NDArray],
                weights: Optional[List[float]] = None,
                mode: str = "attention") -> NDArray:
        """
        Weighted composition with multiple HDC modes (Phase 2+6).

        Parameters
        ----------
        weights : list of float, optional
            Custom blend ratios (e.g., [0.7, 0.3]).
        mode : str
            Composition operation:
            - "attention" (default): multi-head split-query attention
            - "bundle": HDC superposition (additive, preserves all)
            - "bind": HDC binding (multiplicative, creates new concept)
            - "sequential": position-encoded temporal composition
        """
        payload = {
            "keys": [k.flatten().tolist() for k in keys],
            "mode": mode,
        }
        if weights is not None:
            payload["weights"] = weights
        response = self._post("/memory/compose", payload)
        return np.array(response.get("result", np.zeros(self.dim).tolist()),
                        dtype=np.float32)

    def forget(self, key: NDArray, strategy: str = "decay") -> None:
        payload = {
            "key": key.flatten().tolist(),
            "strategy": strategy,
        }
        self._post("/memory/forget", payload)

    def stats(self) -> Dict[str, Any]:
        return self._get("/memory/stats")

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 1 — ACTIVE INFERENCE & NEUROMODULATION
    # ═══════════════════════════════════════════════════════════════════

    def get_modulation_state(self) -> Dict[str, Any]:
        """
        Get current neuromodulation state M(t).

        Returns
        -------
        dict with:
            - modulation: float (0.0-3.0) — current plasticity multiplier
            - surprise: float — last surprise signal
            - baseline: float — resting modulation level
            - regime: str — "low_surprise" | "normal" | "high_surprise"
        """
        return self._get("/neuromodulation/state")

    def get_surprise_history(self, window: int = 100) -> Dict[str, Any]:
        """
        Return surprise trajectory for monitoring learning dynamics.

        Parameters
        ----------
        window : int
            Number of recent operations to include.
        """
        return self._get(f"/neuromodulation/surprise_history?window={window}")

    def configure_neuromodulation(self, baseline: float = 0.5,
                                  sensitivity: float = 2.0,
                                  tau: float = 10.0) -> Dict[str, Any]:
        """
        Fine-tune the plasticity gate parameters.

        Parameters
        ----------
        baseline : float
            Resting modulation level (default 0.5).
        sensitivity : float
            How aggressively M(t) responds to surprise.
        tau : float
            Decay time constant for returning to baseline.
        """
        return self._post("/neuromodulation/configure", {
            "baseline": baseline,
            "sensitivity": sensitivity,
            "tau": tau,
        })

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 2 — PHI_B GEOMETRIC ROUTING
    # ═══════════════════════════════════════════════════════════════════

    def get_phi_b_encoding(self, query: NDArray) -> Dict[str, Any]:
        """
        Inspect the phi_B Hellinger encoding used for routing decisions.

        Returns the geometric representation on the information manifold.
        """
        payload = {"query": query.flatten().tolist()}
        return self._post("/routing/phi_b_encoding", payload)

    def set_routing_geometry(self, geometry: str = "phi_b") -> Dict[str, Any]:
        """
        Switch routing geometry.

        Parameters
        ----------
        geometry : str
            "phi_b" (Hellinger, default) | "cosine" | "euclidean"
        """
        return self._post("/routing/set_geometry", {"geometry": geometry})

    def get_router_trace(self, query: NDArray) -> Dict[str, Any]:
        """
        Explain which pathway was chosen and why for a given query.

        Returns
        -------
        dict with:
            - selected_pathway: str ("exact" | "energy" | "attention")
            - phi_b_scores: dict per pathway
            - confidence_breakdown: dict
            - routing_time_us: int
        """
        payload = {"query": query.flatten().tolist()}
        return self._post("/routing/trace", payload)

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 3 — EFE STRATEGIC ROUTING
    # ═══════════════════════════════════════════════════════════════════

    def query_with_epistemic_weight(self, query: NDArray,
                                    epistemic_w: float = 1.0,
                                    pragmatic_w: float = 0.5,
                                    top_k: int = 1) -> List[Dict[str, Any]]:
        """
        Query with explicit exploration/exploitation tradeoff.

        Parameters
        ----------
        epistemic_w : float
            Weight on information gain (exploration).
        pragmatic_w : float
            Weight on expected reward (exploitation).
        """
        payload = {
            "query": query.flatten().tolist(),
            "epistemic_weight": epistemic_w,
            "pragmatic_weight": pragmatic_w,
            "top_k": top_k,
        }
        response = self._post("/memory/query_efe", payload)
        return response.get("results", [])

    def get_efe_scores(self, query: NDArray) -> Dict[str, Any]:
        """
        Get Expected Free Energy scores for each pathway.

        Returns per-pathway: epistemic value, pragmatic value, total EFE.
        """
        payload = {"query": query.flatten().tolist()}
        return self._post("/routing/efe_scores", payload)

    def set_pathway_strategy(self, strategy: str = "balanced") -> Dict[str, Any]:
        """
        Configure exploration/exploitation balance.

        Parameters
        ----------
        strategy : str
            "exploit" — maximize immediate accuracy
            "explore" — prioritize information gain
            "balanced" — adaptive tradeoff (default)
        """
        return self._post("/routing/set_strategy", {"strategy": strategy})

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 4 — STDP TEMPORAL PLASTICITY
    # ═══════════════════════════════════════════════════════════════════

    def get_head_specialization(self) -> Dict[str, Any]:
        """
        Which attention heads have specialized and for what.

        Returns
        -------
        dict mapping head_id → {specialization_type, temperature, weight}
        """
        return self._get("/stdp/head_specialization")

    def get_head_temperatures(self) -> Dict[str, Any]:
        """
        Inspect temperature of each head (sharp for recall, soft for composition).
        """
        return self._get("/stdp/temperatures")

    def enable_stdp_learning(self, enabled: bool = True,
                             eta: float = 0.01,
                             tau: float = 5.0) -> Dict[str, Any]:
        """
        Configure STDP temporal plasticity.

        Parameters
        ----------
        enabled : bool
            Enable/disable STDP weight updates.
        eta : float
            Learning rate for spike-timing dependent updates.
        tau : float
            Time window (in operations) for co-activation detection.
        """
        return self._post("/stdp/configure", {
            "enabled": enabled,
            "eta": eta,
            "tau": tau,
        })

    def get_head_activation_timeline(self, window: int = 50) -> Dict[str, Any]:
        """
        Spike raster: which heads fired when (for debugging specialization).
        """
        return self._get(f"/stdp/activation_timeline?window={window}")

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 5 — SYSTEM CONFIGURATION
    # ═══════════════════════════════════════════════════════════════════

    def get_system_config(self) -> Dict[str, Any]:
        """Get full configuration of all active components."""
        return self._get("/system/config")

    def set_system_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Bulk-configure all components at once."""
        return self._post("/system/config", config)

    def get_architecture_stats(self) -> Dict[str, Any]:
        """
        Summary of routing/belief/consolidation state.

        Returns n_stored, pathway distribution, belief entropy,
        consolidation status, regime, head weights, etc.
        """
        return self._get("/system/architecture_stats")

    def enable_benchmarking(self, enabled: bool = True) -> Dict[str, Any]:
        """Enable per-component latency breakdown in responses."""
        return self._post("/system/benchmarking", {"enabled": enabled})

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 6 — ADVANCED COMPOSITION & CAUSAL GRAPH
    # ═══════════════════════════════════════════════════════════════════

    def compose_fractional(self, key_a: NDArray, key_b: NDArray,
                           alpha: float = 0.5) -> NDArray:
        """
        Continuous interpolation between two stored patterns.

        Unlike equal-weight composition, this uses learned semantic
        interpolation in the energy landscape — NOT linear blending.

        Parameters
        ----------
        alpha : float
            Interpolation factor (0.0 = pure A, 1.0 = pure B).
        """
        payload = {
            "key_a": key_a.flatten().tolist(),
            "key_b": key_b.flatten().tolist(),
            "alpha": alpha,
        }
        response = self._post("/memory/compose_fractional", payload)
        return np.array(response.get("result", np.zeros(self.dim).tolist()),
                        dtype=np.float32)

    def get_causal_graph(self) -> Dict[str, Any]:
        """
        Return the discovered causal DAG of pattern relationships.

        Returns
        -------
        dict with:
            - nodes: list of pattern IDs
            - edges: list of {source, target, strength, orientation_confidence}
            - confounders: list of detected hidden confounders
        """
        return self._get("/causal/graph")

    def predict_causal_effect(self, cause_key: NDArray,
                              effect_key: NDArray) -> Dict[str, Any]:
        """
        Predict: "If I perturb pattern A, what happens to pattern B?"

        Returns causal strength, direction, and confidence interval.
        """
        payload = {
            "cause_key": cause_key.flatten().tolist(),
            "effect_key": effect_key.flatten().tolist(),
        }
        return self._post("/causal/predict_effect", payload)

    def get_causal_strength(self, key_a: NDArray, key_b: NDArray) -> Dict[str, Any]:
        """SNR-based measure of causal coupling between two patterns."""
        payload = {
            "key_a": key_a.flatten().tolist(),
            "key_b": key_b.flatten().tolist(),
        }
        return self._post("/causal/strength", payload)

    def is_confounded(self, key_a: NDArray, key_b: NDArray,
                      threshold: float = 0.5) -> Dict[str, Any]:
        """
        Detect whether A↔B association is driven by a hidden confounder.

        Returns
        -------
        dict with:
            - confounded: bool
            - confounder_candidates: list of pattern IDs
            - confidence: float
        """
        payload = {
            "key_a": key_a.flatten().tolist(),
            "key_b": key_b.flatten().tolist(),
            "threshold": threshold,
        }
        return self._post("/causal/is_confounded", payload)

    def get_semantic_groups(self) -> Dict[str, Any]:
        """Automatic clustering of semantically-related patterns."""
        return self._get("/semantic/groups")

    def semantic_boost(self, query: NDArray, alpha: float = 0.3) -> List[Dict[str, Any]]:
        """
        Re-rank results by learned semantic similarity.

        Parameters
        ----------
        alpha : float
            Blend between raw retrieval score and semantic boost.
        """
        payload = {
            "query": query.flatten().tolist(),
            "alpha": alpha,
        }
        response = self._post("/memory/semantic_boost", payload)
        return response.get("results", [])

    def enable_active_exploration(self, enabled: bool = True) -> Dict[str, Any]:
        """
        Let the system autonomously test uncertain compositions.

        When enabled, the engine probes uncertain pattern relationships
        during idle periods to strengthen the causal graph.
        """
        return self._post("/system/active_exploration", {"enabled": enabled})

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 7 — ACTIVE STRUCTURE DISCOVERY
    # ═══════════════════════════════════════════════════════════════════

    def get_skeleton_entropy(self) -> Dict[str, Any]:
        """
        Quantify structural uncertainty of the memory graph.

        Returns
        -------
        dict with:
            - total_entropy: float (lower = more certain)
            - n_edges_discovered: int
            - n_edges_uncertain: int
            - convergence_ratio: float
        """
        return self._get("/structure/skeleton_entropy")

    def get_uncertain_pairs(self, n_top: int = 10) -> Dict[str, Any]:
        """
        Which pattern relationships should be tested next?

        Returns the top-N most uncertain edges for targeted exploration.
        """
        return self._get(f"/structure/uncertain_pairs?n_top={n_top}")

    def get_causal_orientations(self) -> Dict[str, Any]:
        """
        Which directional arrows A→B vs B→A are confirmed.

        Includes both directly observed and Meek-propagated orientations.
        """
        return self._get("/structure/orientations")

    def explain_orientation(self, key_a: NDArray, key_b: NDArray) -> Dict[str, Any]:
        """Why do we think A→B? Returns evidence and reasoning."""
        payload = {
            "key_a": key_a.flatten().tolist(),
            "key_b": key_b.flatten().tolist(),
        }
        return self._post("/structure/explain_orientation", payload)

    def auto_select_thresholds(self) -> Dict[str, Any]:
        """
        Calibrate routing/composition thresholds via StARS bootstrap.

        Zero manual tuning — the system finds optimal thresholds
        for the current data distribution.

        Returns
        -------
        dict with:
            - thresholds: dict of component → optimal_value
            - stability_score: float
            - n_bootstrap_samples: int
        """
        return self._post("/structure/auto_thresholds", {})

    def get_threshold_stability(self) -> Dict[str, Any]:
        """How stable is the current threshold selection?"""
        return self._get("/structure/threshold_stability")

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 8 — REGIME DETECTION & SAFETY
    # ═══════════════════════════════════════════════════════════════════

    def get_current_regime(self) -> Dict[str, Any]:
        """
        Get current operating regime.

        Returns
        -------
        dict with:
            - regime: "A" (normal), "B" (cautious), or "C" (anomaly)
            - anomaly_signal: float (KL divergence from expected)
            - modulation: float (current M(t))
            - since: str (ISO timestamp of last regime change)
        """
        return self._get("/safety/regime")

    def get_regime_history(self, window: int = 100) -> Dict[str, Any]:
        """Timeline of regime changes with triggers."""
        return self._get(f"/safety/regime_history?window={window}")

    def enable_anomaly_protection(self, enabled: bool = True) -> Dict[str, Any]:
        """
        Activate amygdala-like safety circuit.

        When enabled, OOD queries trigger Regime C which:
        - Falls back to exact-only retrieval (safest)
        - Disables composition (prevents hallucinated blends)
        - Logs anomaly for review
        """
        return self._post("/safety/anomaly_protection", {"enabled": enabled})

    def set_regime_thresholds(self, theta_b: float = 1.0,
                              theta_c: float = 2.0) -> Dict[str, Any]:
        """
        Configure when to enter cautious (B) and anomaly (C) regimes.

        Parameters
        ----------
        theta_b : float
            KL threshold for entering Regime B (cautious).
        theta_c : float
            KL threshold for entering Regime C (anomaly/lockdown).
        """
        return self._post("/safety/set_thresholds", {
            "theta_b": theta_b,
            "theta_c": theta_c,
        })

    def consolidate(self) -> Dict[str, Any]:
        """
        Trigger a Sleep/Wake consolidation cycle (Phase 8).

        Replays stored patterns through STDP-based plasticity:
        - LTP: strengthen frequently co-activated patterns
        - LTD: weaken rarely-accessed patterns
        - Prune: remove dead-weight patterns below threshold

        Returns
        -------
        dict with:
            - patterns_strengthened: int (LTP)
            - patterns_weakened: int (LTD)
            - patterns_pruned: int
            - consolidation_time_ms: float
            - memory_freed_mb: float
        """
        return self._post("/memory/consolidate", {})

    def get_consolidation_actions(self) -> Dict[str, Any]:
        """What LTP/LTD/prune decisions were made in last consolidation."""
        return self._get("/memory/consolidation_actions")

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 9 — TEMPORAL CAUSALITY & PREFETCH
    # ═══════════════════════════════════════════════════════════════════

    def test_granger_causality(self, key_a: NDArray, key_b: NDArray,
                               max_lag: int = 10) -> Dict[str, Any]:
        """
        Test temporal Granger causality between two patterns.

        "Does accessing A predict future access of B?"

        Returns
        -------
        dict with:
            - lag: int (optimal temporal offset)
            - strength: float (SNR of temporal coupling)
            - significant: bool
            - p_value: float
        """
        payload = {
            "key_a": key_a.flatten().tolist(),
            "key_b": key_b.flatten().tolist(),
            "max_lag": max_lag,
        }
        return self._post("/temporal/granger_test", payload)

    def get_causal_predictions(self, query: NDArray,
                               n_predictions: int = 3) -> Dict[str, Any]:
        """
        Predict which patterns will be accessed next.

        Based on learned temporal transition model.

        Returns
        -------
        dict with:
            - predictions: list of {pattern_id, probability, expected_lag}
        """
        payload = {
            "query": query.flatten().tolist(),
            "n_predictions": n_predictions,
        }
        return self._post("/temporal/predictions", payload)

    def enable_prefetch(self, enabled: bool = True) -> Dict[str, Any]:
        """
        Enable predictive prefetching.

        Automatically prefetches predicted next patterns into hot tier
        for sub-millisecond access.
        """
        return self._post("/temporal/prefetch", {"enabled": enabled})

    def get_prefetch_hit_rate(self) -> Dict[str, Any]:
        """Cache hit % from temporal predictions."""
        return self._get("/temporal/prefetch_stats")

    def record_temporal_access(self, key: NDArray) -> None:
        """Explicitly mark a temporal access for Granger analysis."""
        payload = {"key": key.flatten().tolist()}
        self._post("/temporal/record_access", payload)

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 10 — SCALING, TEXT INTERFACE & EXPLAINABILITY
    # ═══════════════════════════════════════════════════════════════════

    # --- Tiered Storage ---

    def move_to_tier(self, key: NDArray, tier: str = "warm") -> Dict[str, Any]:
        """
        Explicitly promote/demote a pattern between storage tiers.

        Parameters
        ----------
        tier : str
            "hot" (~1ms, in-memory) | "warm" (~10ms, SSD) | "cold" (~100ms, archive)
        """
        payload = {
            "key": key.flatten().tolist(),
            "tier": tier,
        }
        return self._post("/storage/move_tier", payload)

    def get_tier_stats(self) -> Dict[str, Any]:
        """
        How many patterns in each tier with hit rates.

        Returns
        -------
        dict with:
            - hot: {count, hit_rate, avg_latency_ms}
            - warm: {count, hit_rate, avg_latency_ms}
            - cold: {count, hit_rate, avg_latency_ms}
            - total_storage_mb: float
        """
        return self._get("/storage/tier_stats")

    # --- Text Interface (HDC Tokenizer) ---

    def store_text(self, text: str, value_embedding: Optional[NDArray] = None,
                   metadata: Optional[Dict] = None) -> None:
        """
        Store semantic text (auto-encoded to HDC representation).

        Parameters
        ----------
        text : str
            Natural language text to store.
        value_embedding : NDArray, optional
            Custom value vector. If None, uses text encoding as value.
        metadata : dict, optional
            Additional metadata.
        """
        payload = {"text": text}
        if value_embedding is not None:
            payload["value"] = value_embedding.flatten().tolist()
        if metadata:
            payload["metadata"] = metadata
        self._post("/memory/store_text", payload)

    def query_text(self, query_text: str, top_k: int = 5,
                   filters: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Query memory by natural language text.

        Auto-encodes query to HDC representation and searches.
        """
        payload = {
            "query_text": query_text,
            "top_k": top_k,
        }
        if filters:
            payload["filters"] = filters
        response = self._post("/memory/query_text", payload)
        return response.get("results", [])

    def compose_text(self, texts: List[str],
                     weights: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Compose blends of semantic text patterns.

        Example: compose_text(["python", "machine learning"]) → blended concept
        """
        payload = {"texts": texts}
        if weights:
            payload["weights"] = weights
        return self._post("/memory/compose_text", payload)

    def get_text_encoding(self, text: str) -> NDArray:
        """Inspect the HDC encoding of a text string."""
        response = self._post("/text/encode", {"text": text})
        return np.array(response.get("encoding", np.zeros(self.dim).tolist()),
                        dtype=np.float32)

    # --- Explainability (XAI) ---

    def explain(self, query: NDArray) -> Dict[str, Any]:
        """
        XAI: Full explanation of a retrieval decision.

        Returns
        -------
        dict with:
            - pathway_selected: str
            - confidence_breakdown: dict (per pathway)
            - attention_map: list of {pattern_id, weight} (top-10 contributors)
            - energy_landscape: dict (local basin data)
            - head_contributions: list of {head_id, weight, attended_pattern}
            - regime: str (current safety regime)
            - modulation: float (M(t) at query time)
        """
        payload = {"query": query.flatten().tolist()}
        return self._post("/xai/explain", payload)

    def get_xai_dashboard(self) -> Dict[str, Any]:
        """
        Full XAI dashboard data for visualization.

        Returns JSON with belief sphere coordinates, causal graph,
        regime history, head specialization map, and tier distribution.
        """
        return self._get("/xai/dashboard")

    def explain_head_contributions(self, query: NDArray) -> Dict[str, Any]:
        """Which attention heads contributed how much to the result."""
        payload = {"query": query.flatten().tolist()}
        return self._post("/xai/head_contributions", payload)

    def visualize_belief_state(self) -> Dict[str, Any]:
        """
        Get belief state projection for 2D/3D visualization.

        Returns coordinates on the information hypersphere.
        """
        return self._get("/xai/belief_state")

    # --- Snapshots & Versioning ---

    def snapshot(self, name: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a persistent snapshot of current memory state.

        Snapshots can be restored later, enabling versioned memory.
        """
        payload = {}
        if name:
            payload["name"] = name
        return self._post("/memory/snapshot", payload)

    def restore(self, snapshot_id: str) -> Dict[str, Any]:
        """Restore memory from a previously created snapshot."""
        return self._post("/memory/restore", {"snapshot_id": snapshot_id})

    def list_snapshots(self) -> Dict[str, Any]:
        """List all available snapshots."""
        return self._get("/memory/snapshots")

    # --- Analytics ---

    def analytics(self, period: str = "24h") -> Dict[str, Any]:
        """
        Get memory usage analytics and health metrics.

        Parameters
        ----------
        period : str
            Time window: "1h", "24h", "7d", "30d"

        Returns
        -------
        dict with:
            - total_queries: int
            - avg_latency_ms: float
            - pathway_distribution: dict (exact/energy/attention %)
            - composition_requests: int
            - cache_hit_rate: float
            - storage_used_mb: float
            - regime_changes: int
            - prefetch_hit_rate: float
        """
        return self._get(f"/analytics/usage?period={period}")

    # ═══════════════════════════════════════════════════════════════════
    # HTTP TRANSPORT WITH RETRY & BACKOFF
    # ═══════════════════════════════════════════════════════════════════

    def _serialize_value(self, value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.flatten().tolist()
        if isinstance(value, (list, tuple)):
            return list(value)
        return value

    def _post(self, path: str, payload: Dict) -> Dict:
        return self._request("POST", path, json=payload)

    def _get(self, path: str) -> Dict:
        return self._request("GET", path)

    def _request(self, method: str, path: str, json: Optional[Dict] = None) -> Dict:
        last_exc = None
        for attempt in range(self.max_retries):
            try:
                if self._client is not None:
                    if method == "POST":
                        resp = self._client.post(path, json=json)
                    else:
                        resp = self._client.get(path)
                    resp.raise_for_status()
                    return resp.json()
                else:
                    url = f"{self.endpoint}{path}"
                    if method == "POST":
                        resp = self._requests.post(
                            url, json=json, headers=self._headers, timeout=self.timeout
                        )
                    else:
                        resp = self._requests.get(
                            url, headers=self._headers, timeout=self.timeout
                        )
                    resp.raise_for_status()
                    return resp.json()
            except Exception as exc:
                last_exc = exc
                if attempt < self.max_retries - 1:
                    wait = _BACKOFF_BASE * (2 ** attempt)
                    logger.warning(
                        "Engramma Cloud request failed (attempt %d/%d), "
                        "retrying in %.1fs: %s",
                        attempt + 1, self.max_retries, wait, exc
                    )
                    time.sleep(wait)
        raise ConnectionError(
            f"Engramma Cloud request failed after {self.max_retries} attempts: {last_exc}"
        ) from last_exc
