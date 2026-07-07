"""
Async Cloud backend for Engramma Memory.

Mirrors all CloudBackend features (Phases 1-10) with async/await.
Requires httpx: pip install engramma-memory[cloud]
"""
import logging
import numpy as np
from numpy.typing import NDArray
from typing import Optional, List, Dict, Any

try:
    import httpx
except ImportError:
    raise ImportError(
        "Async cloud backend requires 'httpx'. "
        "Install with: pip install engramma-memory[cloud]"
    )


logger = logging.getLogger(__name__)

_CLOUD_API_BASE = "https://api.engramma-memory.dev/v1"
_MAX_RETRIES = 3
_BACKOFF_BASE = 0.5


class AsyncCloudBackend:
    """Async Engramma Cloud backend — full Phase 1-10 feature set."""

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
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.endpoint,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "X-Engramma-Dim": str(self.dim),
                    "X-Engramma-SDK": "python-async/0.1.0",
                },
                timeout=self.timeout,
            )
        return self._client

    # ─── Core Operations ───────────────────────────────────────────────

    async def store(self, key: NDArray, value: Any,
                    metadata: Optional[Dict] = None) -> None:
        payload = {
            "key": key.flatten().tolist(),
            "value": self._serialize_value(value),
        }
        if metadata:
            payload["metadata"] = metadata
        await self._post("/memory/store", payload)

    async def query(self, query: NDArray, top_k: int = 1,
                    filters: Optional[Dict] = None,
                    use_phi_b: bool = False) -> List[Dict[str, Any]]:
        payload = {
            "query": query.flatten().tolist(),
            "top_k": top_k,
            "use_phi_b": use_phi_b,
        }
        if filters:
            payload["filters"] = filters
        response = await self._post("/memory/query", payload)
        return response.get("results", [])

    async def retrieve(self, query: NDArray) -> NDArray:
        payload = {"query": query.flatten().tolist(), "mode": "active_inference"}
        response = await self._post("/memory/retrieve", payload)
        return np.array(response.get("result", np.zeros(self.dim).tolist()),
                        dtype=np.float32)

    async def compose(self, keys: List[NDArray],
                      weights: Optional[List[float]] = None,
                      mode: str = "attention") -> NDArray:
        payload = {"keys": [k.flatten().tolist() for k in keys], "mode": mode}
        if weights is not None:
            payload["weights"] = weights
        response = await self._post("/memory/compose", payload)
        return np.array(response.get("result", np.zeros(self.dim).tolist()),
                        dtype=np.float32)

    async def forget(self, key: NDArray, strategy: str = "decay") -> None:
        await self._post("/memory/forget", {"key": key.flatten().tolist(), "strategy": strategy})

    async def stats(self) -> Dict[str, Any]:
        return await self._get("/memory/stats")

    # ─── Phase 1: Neuromodulation ──────────────────────────────────────

    async def get_modulation_state(self) -> Dict[str, Any]:
        return await self._get("/neuromodulation/state")

    async def get_surprise_history(self, window: int = 100) -> Dict[str, Any]:
        return await self._get(f"/neuromodulation/surprise_history?window={window}")

    async def configure_neuromodulation(self, baseline: float = 0.5,
                                        sensitivity: float = 2.0,
                                        tau: float = 10.0) -> Dict[str, Any]:
        return await self._post("/neuromodulation/configure", {
            "baseline": baseline, "sensitivity": sensitivity, "tau": tau,
        })

    # ─── Phase 2: phi_B Routing ────────────────────────────────────────

    async def get_phi_b_encoding(self, query: NDArray) -> Dict[str, Any]:
        return await self._post("/routing/phi_b_encoding", {"query": query.flatten().tolist()})

    async def get_router_trace(self, query: NDArray) -> Dict[str, Any]:
        return await self._post("/routing/trace", {"query": query.flatten().tolist()})

    async def set_routing_geometry(self, geometry: str = "phi_b") -> Dict[str, Any]:
        return await self._post("/routing/set_geometry", {"geometry": geometry})

    # ─── Phase 3: EFE Strategic Routing ────────────────────────────────

    async def query_with_epistemic_weight(self, query: NDArray,
                                          epistemic_w: float = 1.0,
                                          pragmatic_w: float = 0.5,
                                          top_k: int = 1) -> List[Dict[str, Any]]:
        payload = {
            "query": query.flatten().tolist(),
            "epistemic_weight": epistemic_w,
            "pragmatic_weight": pragmatic_w,
            "top_k": top_k,
        }
        response = await self._post("/memory/query_efe", payload)
        return response.get("results", [])

    async def get_efe_scores(self, query: NDArray) -> Dict[str, Any]:
        return await self._post("/routing/efe_scores", {"query": query.flatten().tolist()})

    async def set_pathway_strategy(self, strategy: str = "balanced") -> Dict[str, Any]:
        return await self._post("/routing/set_strategy", {"strategy": strategy})

    # ─── Phase 4: STDP ─────────────────────────────────────────────────

    async def get_head_specialization(self) -> Dict[str, Any]:
        return await self._get("/stdp/head_specialization")

    async def get_head_temperatures(self) -> Dict[str, Any]:
        return await self._get("/stdp/temperatures")

    async def enable_stdp_learning(self, enabled: bool = True,
                                   eta: float = 0.01,
                                   tau: float = 5.0) -> Dict[str, Any]:
        return await self._post("/stdp/configure", {"enabled": enabled, "eta": eta, "tau": tau})

    # ─── Phase 6: Causal & Composition ─────────────────────────────────

    async def compose_fractional(self, key_a: NDArray, key_b: NDArray,
                                 alpha: float = 0.5) -> NDArray:
        payload = {
            "key_a": key_a.flatten().tolist(),
            "key_b": key_b.flatten().tolist(),
            "alpha": alpha,
        }
        response = await self._post("/memory/compose_fractional", payload)
        return np.array(response.get("result", np.zeros(self.dim).tolist()),
                        dtype=np.float32)

    async def get_causal_graph(self) -> Dict[str, Any]:
        return await self._get("/causal/graph")

    async def predict_causal_effect(self, cause_key: NDArray,
                                    effect_key: NDArray) -> Dict[str, Any]:
        return await self._post("/causal/predict_effect", {
            "cause_key": cause_key.flatten().tolist(),
            "effect_key": effect_key.flatten().tolist(),
        })

    async def is_confounded(self, key_a: NDArray, key_b: NDArray,
                            threshold: float = 0.5) -> Dict[str, Any]:
        return await self._post("/causal/is_confounded", {
            "key_a": key_a.flatten().tolist(),
            "key_b": key_b.flatten().tolist(),
            "threshold": threshold,
        })

    async def enable_active_exploration(self, enabled: bool = True) -> Dict[str, Any]:
        return await self._post("/system/active_exploration", {"enabled": enabled})

    # ─── Phase 7: Structure Discovery ──────────────────────────────────

    async def get_skeleton_entropy(self) -> Dict[str, Any]:
        return await self._get("/structure/skeleton_entropy")

    async def get_uncertain_pairs(self, n_top: int = 10) -> Dict[str, Any]:
        return await self._get(f"/structure/uncertain_pairs?n_top={n_top}")

    async def auto_select_thresholds(self) -> Dict[str, Any]:
        return await self._post("/structure/auto_thresholds", {})

    # ─── Phase 8: Safety & Consolidation ───────────────────────────────

    async def get_current_regime(self) -> Dict[str, Any]:
        return await self._get("/safety/regime")

    async def enable_anomaly_protection(self, enabled: bool = True) -> Dict[str, Any]:
        return await self._post("/safety/anomaly_protection", {"enabled": enabled})

    async def consolidate(self) -> Dict[str, Any]:
        return await self._post("/memory/consolidate", {})

    async def set_regime_thresholds(self, theta_b: float = 1.0,
                                    theta_c: float = 2.0) -> Dict[str, Any]:
        return await self._post("/safety/set_thresholds", {
            "theta_b": theta_b, "theta_c": theta_c,
        })

    # ─── Phase 9: Temporal Causality ───────────────────────────────────

    async def test_granger_causality(self, key_a: NDArray, key_b: NDArray,
                                     max_lag: int = 10) -> Dict[str, Any]:
        return await self._post("/temporal/granger_test", {
            "key_a": key_a.flatten().tolist(),
            "key_b": key_b.flatten().tolist(),
            "max_lag": max_lag,
        })

    async def get_causal_predictions(self, query: NDArray,
                                     n_predictions: int = 3) -> Dict[str, Any]:
        return await self._post("/temporal/predictions", {
            "query": query.flatten().tolist(),
            "n_predictions": n_predictions,
        })

    async def enable_prefetch(self, enabled: bool = True) -> Dict[str, Any]:
        return await self._post("/temporal/prefetch", {"enabled": enabled})

    async def get_prefetch_hit_rate(self) -> Dict[str, Any]:
        return await self._get("/temporal/prefetch_stats")

    # ─── Phase 10: Text, Tiers, XAI ───────────────────────────────────

    async def store_text(self, text: str, value_embedding: Optional[NDArray] = None,
                         metadata: Optional[Dict] = None) -> None:
        payload = {"text": text}
        if value_embedding is not None:
            payload["value"] = value_embedding.flatten().tolist()
        if metadata:
            payload["metadata"] = metadata
        await self._post("/memory/store_text", payload)

    async def query_text(self, query_text: str, top_k: int = 5,
                         filters: Optional[Dict] = None) -> List[Dict[str, Any]]:
        payload = {"query_text": query_text, "top_k": top_k}
        if filters:
            payload["filters"] = filters
        response = await self._post("/memory/query_text", payload)
        return response.get("results", [])

    async def compose_text(self, texts: List[str],
                           weights: Optional[List[float]] = None) -> Dict[str, Any]:
        payload = {"texts": texts}
        if weights:
            payload["weights"] = weights
        return await self._post("/memory/compose_text", payload)

    async def get_text_encoding(self, text: str) -> NDArray:
        response = await self._post("/text/encode", {"text": text})
        return np.array(response.get("encoding", np.zeros(self.dim).tolist()),
                        dtype=np.float32)

    async def move_to_tier(self, key: NDArray, tier: str = "warm") -> Dict[str, Any]:
        return await self._post("/storage/move_tier", {
            "key": key.flatten().tolist(), "tier": tier,
        })

    async def get_tier_stats(self) -> Dict[str, Any]:
        return await self._get("/storage/tier_stats")

    async def explain(self, query: NDArray) -> Dict[str, Any]:
        return await self._post("/xai/explain", {"query": query.flatten().tolist()})

    async def get_xai_dashboard(self) -> Dict[str, Any]:
        return await self._get("/xai/dashboard")

    async def snapshot(self, name: Optional[str] = None) -> Dict[str, Any]:
        payload = {}
        if name:
            payload["name"] = name
        return await self._post("/memory/snapshot", payload)

    async def restore(self, snapshot_id: str) -> Dict[str, Any]:
        return await self._post("/memory/restore", {"snapshot_id": snapshot_id})

    async def analytics(self, period: str = "24h") -> Dict[str, Any]:
        return await self._get(f"/analytics/usage?period={period}")

    # ─── HTTP Transport ────────────────────────────────────────────────

    def _serialize_value(self, value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.flatten().tolist()
        if isinstance(value, (list, tuple)):
            return list(value)
        return value

    async def _post(self, path: str, payload: Dict) -> Dict:
        return await self._request("POST", path, json=payload)

    async def _get(self, path: str) -> Dict:
        return await self._request("GET", path)

    async def _request(self, method: str, path: str,
                       json: Optional[Dict] = None) -> Dict:
        import asyncio
        last_exc = None
        client = await self._get_client()

        for attempt in range(self.max_retries):
            try:
                if method == "POST":
                    resp = await client.post(path, json=json)
                else:
                    resp = await client.get(path)
                resp.raise_for_status()
                return resp.json()
            except Exception as exc:
                last_exc = exc
                if attempt < self.max_retries - 1:
                    wait = _BACKOFF_BASE * (2 ** attempt)
                    logger.warning(
                        "Engramma Cloud async request failed (attempt %d/%d), "
                        "retrying in %.1fs: %s",
                        attempt + 1, self.max_retries, wait, exc
                    )
                    await asyncio.sleep(wait)

        raise ConnectionError(
            f"Engramma Cloud request failed after {self.max_retries} attempts: {last_exc}"
        ) from last_exc

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()
