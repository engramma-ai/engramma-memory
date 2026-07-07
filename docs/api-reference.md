# API Reference

## EngrammaMemory (Sync)

The main entry point. Works with both local and cloud backends.

```python
from engramma_memory import EngrammaMemory
```

### Constructor

```python
EngrammaMemory(
    dim: int,
    backend: str = "local",
    api_key: str = None,
    max_patterns: int = 1000,
    **kwargs
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dim` | int | required | Dimension of key/value vectors |
| `backend` | str | `"local"` | `"local"` or `"cloud"` |
| `api_key` | str | None | Required for cloud backend |
| `max_patterns` | int | 1000 | Max stored patterns (local capped at 1000) |

---

### Core Methods

#### store()

```python
mem.store(key, value, metadata=None) -> None
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `key` | NDArray / list | Embedding vector for retrieval |
| `value` | NDArray / list | Data to store |
| `metadata` | dict | Additional metadata (cloud only) |

#### query()

```python
mem.query(query, top_k=1, filters=None, use_phi_b=False) -> list[dict]
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `query` | NDArray / list | Query embedding |
| `top_k` | int | Number of results |
| `filters` | dict | Metadata filters (cloud only) |
| `use_phi_b` | bool | Use Hellinger routing (cloud only) |

**Returns:** List of `{"value": NDArray, "score": float}`

#### retrieve()

```python
mem.retrieve(query) -> NDArray
```

Smart retrieval with confidence-based routing (local) or Active Inference (cloud).

#### compose()

```python
mem.compose(keys, weights=None, mode="attention") -> NDArray
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `keys` | list | Component keys to compose |
| `weights` | list | Blend weights (cloud only) |
| `mode` | str | `"attention"`, `"bundle"`, `"bind"`, `"sequential"` (cloud only) |

#### forget()

```python
mem.forget(key, strategy="decay") -> None
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `key` | NDArray / list | Key of the pattern |
| `strategy` | str | `"decay"` or `"immediate"` |

#### stats()

```python
mem.stats() -> dict
```

#### count (property)

```python
mem.count -> int
```

---

## EngrammaMemoryAsync

Async-first interface. Same API as sync with `await`.

```python
from engramma_memory import EngrammaMemoryAsync

async with EngrammaMemoryAsync(dim=256, backend="cloud", api_key="nx_...") as mem:
    await mem.store(key=emb, value=emb)
    results = await mem.query(emb, top_k=5)
```

All core methods have identical signatures with `async/await`.

---

## Cloud-Exclusive Methods

These methods are only available with `backend="cloud"`. Calling them on a local backend raises `RuntimeError`.

### Phase 1 — Neuromodulation

| Method | Returns | Description |
|--------|---------|-------------|
| `get_modulation_state()` | dict | Current M(t), surprise, baseline, regime |
| `get_surprise_history(window=100)` | dict | Surprise trajectory |
| `configure_neuromodulation(baseline, sensitivity, tau)` | dict | Tune plasticity gate |

### Phase 2 — phi_B Routing

| Method | Returns | Description |
|--------|---------|-------------|
| `get_phi_b_encoding(query)` | dict | Inspect Hellinger encoding |
| `set_routing_geometry(geometry)` | dict | Switch: `"phi_b"`, `"cosine"`, `"euclidean"` |
| `get_router_trace(query)` | dict | Explain pathway selection |

### Phase 3 — EFE Strategy

| Method | Returns | Description |
|--------|---------|-------------|
| `query_with_epistemic_weight(query, epistemic_w, pragmatic_w, top_k)` | list | Query with explore/exploit balance |
| `get_efe_scores(query)` | dict | Per-pathway EFE scores |
| `set_pathway_strategy(strategy)` | dict | `"exploit"`, `"explore"`, `"balanced"` |

### Phase 4 — STDP Plasticity

| Method | Returns | Description |
|--------|---------|-------------|
| `get_head_specialization()` | dict | Per-head specialization type + weight |
| `get_head_temperatures()` | dict | Temperature of each head |
| `enable_stdp_learning(enabled, eta, tau)` | dict | Configure temporal plasticity |
| `get_head_activation_timeline(window)` | dict | Spike raster |

### Phase 5 — System Config

| Method | Returns | Description |
|--------|---------|-------------|
| `get_system_config()` | dict | Full component configuration |
| `set_system_config(config)` | dict | Bulk-configure all components |
| `get_architecture_stats()` | dict | System state summary |
| `enable_benchmarking(enabled)` | dict | Enable latency breakdown |

### Phase 6 — Causal & Composition

| Method | Returns | Description |
|--------|---------|-------------|
| `compose_fractional(key_a, key_b, alpha)` | NDArray | Continuous interpolation |
| `get_causal_graph()` | dict | Full causal DAG |
| `predict_causal_effect(cause_key, effect_key)` | dict | Interventional prediction |
| `get_causal_strength(key_a, key_b)` | dict | SNR coupling measure |
| `is_confounded(key_a, key_b, threshold)` | dict | Confounder detection |
| `get_semantic_groups()` | dict | Auto-clustering |
| `semantic_boost(query, alpha)` | list | Semantic re-ranking |
| `enable_active_exploration(enabled)` | dict | Autonomous probing |

### Phase 7 — Structure Discovery

| Method | Returns | Description |
|--------|---------|-------------|
| `get_skeleton_entropy()` | dict | Structural uncertainty |
| `get_uncertain_pairs(n_top)` | dict | Top uncertain edges |
| `get_causal_orientations()` | dict | Confirmed A→B directions |
| `explain_orientation(key_a, key_b)` | dict | Why A→B? |
| `auto_select_thresholds()` | dict | StARS auto-calibration |
| `get_threshold_stability()` | dict | Threshold stability score |

### Phase 8 — Safety & Consolidation

| Method | Returns | Description |
|--------|---------|-------------|
| `get_current_regime()` | dict | A/B/C regime + anomaly signal |
| `get_regime_history(window)` | dict | Regime change timeline |
| `enable_anomaly_protection(enabled)` | dict | Activate safety circuit |
| `set_regime_thresholds(theta_b, theta_c)` | dict | Configure KL thresholds |
| `consolidate()` | dict | Sleep/Wake cycle (LTP/LTD/prune) |
| `get_consolidation_actions()` | dict | Last consolidation decisions |

### Phase 9 — Temporal Causality

| Method | Returns | Description |
|--------|---------|-------------|
| `test_granger_causality(key_a, key_b, max_lag)` | dict | Temporal causality test |
| `get_causal_predictions(query, n_predictions)` | dict | Predict next accesses |
| `enable_prefetch(enabled)` | dict | Predictive prefetching |
| `get_prefetch_hit_rate()` | dict | Prediction accuracy |
| `record_temporal_access(key)` | None | Mark temporal access |

### Phase 10 — Scale, Text & XAI

#### Tiered Storage

| Method | Returns | Description |
|--------|---------|-------------|
| `move_to_tier(key, tier)` | dict | Move to `"hot"` / `"warm"` / `"cold"` |
| `get_tier_stats()` | dict | Per-tier count, hit rate, latency |

#### Text Interface

| Method | Returns | Description |
|--------|---------|-------------|
| `store_text(text, value_embedding, metadata)` | None | Store semantic text |
| `query_text(query_text, top_k, filters)` | list | Query by natural language |
| `compose_text(texts, weights)` | dict | Compose text concepts |
| `get_text_encoding(text)` | NDArray | Inspect HDC encoding |

#### Explainability (XAI)

| Method | Returns | Description |
|--------|---------|-------------|
| `explain(query)` | dict | Full retrieval explanation |
| `get_xai_dashboard()` | dict | Dashboard visualization data |
| `explain_head_contributions(query)` | dict | Per-head contribution analysis |
| `visualize_belief_state()` | dict | Belief sphere coordinates |

#### Snapshots

| Method | Returns | Description |
|--------|---------|-------------|
| `snapshot(name)` | dict | Create persistent snapshot |
| `restore(snapshot_id)` | dict | Restore from snapshot |
| `list_snapshots()` | dict | List all snapshots |

#### Analytics

| Method | Returns | Description |
|--------|---------|-------------|
| `analytics(period)` | dict | Usage metrics (1h/24h/7d/30d) |

---

## Backend Comparison

| Feature | Local | Cloud |
|---------|-------|-------|
| Max patterns | 1000 | Unlimited (tiered) |
| Persistence | RAM only | Persistent |
| Composition weights | Equal only | Custom ratios |
| Composition modes | attention only | attention, bundle, bind, sequential |
| Routing | ConfidenceRouter | phi_B + EFE + Active Inference |
| Text interface | No | Yes (HDC tokenizer) |
| XAI | No | Full explainability |
| Safety | No | 3-regime anomaly detection |
| Temporal | No | Granger + prefetch |
| Causal graph | No | Full DAG discovery |
| Async | N/A | Full async support |

---

## Error Handling

```python
# No API key for cloud
EngrammaMemory(dim=256, backend="cloud")
# → ValueError: Cloud backend requires an API key.

# Invalid API key
EngrammaMemory(dim=256, backend="cloud", api_key="bad_key")
# → ValueError: Invalid API key.

# Invalid backend
EngrammaMemory(dim=256, backend="redis")
# → ValueError: Unknown backend 'redis'. Use 'local' or 'cloud'.

# Cloud-only method on local
mem = EngrammaMemoryAsync(dim=256, backend="local")
await mem.explain(emb)
# → RuntimeError: explain() is a cloud-only feature.

# Network failure (after 3 retries with exponential backoff)
# → ConnectionError: Engramma Cloud request failed after 3 attempts: ...

# Capacity warning (local, at 90%)
# → UserWarning: EngrammaMemory: 900/1000 patterns used (90%)...
```
