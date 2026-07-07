# Engramma Cloud — Feature Overview

Engramma Cloud extends the open-source engine with 10 phases of advanced capabilities. Switch with one line:

```python
mem = EngrammaMemory(dim=256, backend="cloud", api_key="nx_live_...")
```

---

## Feature Map by Phase

| Phase | Category | What It Does | Local | Cloud |
|-------|----------|--------------|-------|-------|
| 1 | Neuromodulation | Auto-tunes learning rate based on surprise | - | Yes |
| 2 | phi_B Routing | Information-theoretic pathway selection | - | Yes |
| 3 | EFE Strategy | Optimal explore/exploit decisions | - | Yes |
| 4 | STDP Plasticity | Attention heads self-specialize over time | - | Yes |
| 5 | Unified Config | Full system orchestration & benchmarking | - | Yes |
| 6 | Causal Graph | Discover & exploit causal relationships | - | Yes |
| 7 | Structure Discovery | Zero-config auto-calibration (StARS) | - | Yes |
| 8 | Safety Regimes | Anomaly detection + Sleep/Wake consolidation | - | Yes |
| 9 | Temporal Causality | Granger causality + predictive prefetch | - | Yes |
| 10 | Scale & XAI | Tiered storage, text interface, explainability | - | Yes |

---

## Phase 1 — Neuromodulation (Active Inference)

The system automatically learns aggressively when surprised and conservatively when confident.

```python
# Check current plasticity state
state = mem.get_modulation_state()
# → {"modulation": 1.4, "surprise": 0.8, "baseline": 0.5, "regime": "high_surprise"}

# Monitor learning dynamics
history = mem.get_surprise_history(window=100)

# Fine-tune plasticity (advanced)
mem.configure_neuromodulation(baseline=0.5, sensitivity=2.0, tau=10.0)
```

**Why it matters:** No more hand-tuning learning rates. The system auto-scales based on data novelty.

---

## Phase 2 — phi_B Geometric Routing

Routes queries using Hellinger distance on the information manifold — mathematically principled, not empirical thresholds.

```python
# Use phi_B routing for more accurate queries
results = mem.query(embedding, top_k=5, use_phi_b=True)

# Understand routing decisions
trace = mem.get_router_trace(query_embedding)
# → {"selected_pathway": "attention", "phi_b_scores": {...}, "routing_time_us": 42}

# Switch geometry
mem.set_routing_geometry("phi_b")  # or "cosine", "euclidean"

# Inspect encoding
encoding = mem.get_phi_b_encoding(query_embedding)
```

**Why it matters:** Reduces silent routing failures. Queries go to the right pathway every time.

---

## Phase 3 — EFE Strategic Routing

Expected Free Energy optimization — the system makes **strategic** decisions about which pathway to use.

```python
# Balance exploration vs exploitation
results = mem.query_with_epistemic_weight(
    embedding,
    epistemic_w=1.0,   # information gain (exploration)
    pragmatic_w=0.5,   # expected reward (exploitation)
    top_k=5
)

# Inspect per-pathway EFE scores
scores = mem.get_efe_scores(embedding)
# → {"exact": {"epistemic": 0.1, "pragmatic": 0.9}, "energy": {...}, "attention": {...}}

# Set global strategy
mem.set_pathway_strategy("explore")   # prioritize learning
mem.set_pathway_strategy("exploit")   # prioritize accuracy
mem.set_pathway_strategy("balanced")  # adaptive (default)
```

**Why it matters:** Agent stops making random routing decisions. Better long-term memory performance.

---

## Phase 4 — STDP Temporal Plasticity

Attention heads specialize through biologically-plausible Spike-Timing Dependent Plasticity.

```python
# See what each head has learned
spec = mem.get_head_specialization()
# → {"head_0": {"type": "exact_recall", "temperature": 2.1, "weight": 0.35}, ...}

# Inspect temperatures (sharp=recall, soft=composition)
temps = mem.get_head_temperatures()

# Configure STDP learning
mem.enable_stdp_learning(enabled=True, eta=0.01, tau=5.0)

# View spike raster (debugging)
timeline = mem.get_head_activation_timeline(window=50)
```

**Why it matters:** Emergent specialization. No manual head tuning. Better generalization.

---

## Phase 5 — System Configuration

Full system orchestration with unified configuration.

```python
# Get everything at once
config = mem.get_system_config()

# Bulk-configure
mem.set_system_config({
    "neuromodulation": {"sensitivity": 1.5},
    "routing": {"geometry": "phi_b"},
    "stdp": {"enabled": True},
})

# Architecture overview
stats = mem.get_architecture_stats()
# → {n_stored, pathway_distribution, belief_entropy, regime, head_weights, ...}

# Enable latency breakdown in responses
mem.enable_benchmarking(True)
```

---

## Phase 6 — Causal Graph & Advanced Composition

Discover and exploit causal relationships. Go beyond correlation.

```python
# Continuous interpolation (not just 50/50)
blend = mem.compose_fractional(key_a, key_b, alpha=0.7)  # 70% A, 30% B

# HDC composition modes
blend = mem.compose([key_a, key_b], mode="bundle")     # superposition (preserves all)
blend = mem.compose([key_a, key_b], mode="bind")       # binding (new concept)
blend = mem.compose([key_a, key_b], mode="sequential") # temporal ordering

# Discover the causal DAG
graph = mem.get_causal_graph()
# → {nodes: [...], edges: [{source, target, strength, orientation_confidence}]}

# Predict causal effects
effect = mem.predict_causal_effect(cause_key=emb_a, effect_key=emb_b)
# → {"strength": 0.7, "direction": "a→b", "confidence": 0.92}

# Detect confounders
result = mem.is_confounded(key_a, key_b, threshold=0.5)
# → {"confounded": True, "confounder_candidates": ["p_17"], "confidence": 0.85}

# Semantic clustering
groups = mem.get_semantic_groups()

# Re-rank by semantics
results = mem.semantic_boost(query, alpha=0.3)

# Let system explore uncertain relationships autonomously
mem.enable_active_exploration(enabled=True)
```

**Why it matters:** The system becomes an active researcher, not just a retriever.

---

## Phase 7 — Active Structure Discovery

Zero-config auto-calibration. The system finds optimal thresholds for your data.

```python
# How certain is the memory structure?
entropy = mem.get_skeleton_entropy()
# → {"total_entropy": 0.34, "n_edges_discovered": 42, "n_edges_uncertain": 7}

# Which relationships should we test next?
uncertain = mem.get_uncertain_pairs(n_top=10)

# Auto-calibrate all thresholds (StARS bootstrap)
result = mem.auto_select_thresholds()
# → {"thresholds": {"exact_confidence": 0.82, ...}, "stability_score": 0.94}

# Check threshold stability
stability = mem.get_threshold_stability()

# Understand edge orientations
orientations = mem.get_causal_orientations()
mem.explain_orientation(key_a, key_b)
# → {"direction": "a→b", "evidence": "Meek rule R1", "confidence": 0.91}
```

**Why it matters:** Zero manual tuning. System self-calibrates to your data distribution.

---

## Phase 8 — Regime Detection & Safety

Production safety. Three operating regimes with automatic fallback.

```python
# Check current regime
regime = mem.get_current_regime()
# → {"regime": "A", "anomaly_signal": 0.12, "modulation": 0.6, "since": "..."}
# Regimes: A (normal), B (cautious), C (anomaly/lockdown)

# Enable safety circuit (amygdala-like)
mem.enable_anomaly_protection(enabled=True)
# In Regime C: falls back to exact-only, disables composition, logs anomaly

# Configure when to enter each regime
mem.set_regime_thresholds(theta_b=1.0, theta_c=2.0)  # KL thresholds

# View regime history
history = mem.get_regime_history(window=100)

# Trigger Sleep/Wake consolidation
result = mem.consolidate()
# → {
#     "patterns_strengthened": 42,  (LTP)
#     "patterns_weakened": 15,      (LTD)
#     "patterns_pruned": 7,
#     "consolidation_time_ms": 230,
#     "memory_freed_mb": 1.2
# }

# Review consolidation decisions
actions = mem.get_consolidation_actions()
```

**Why it matters:** Production safety. Your agent won't hallucinate compositions on OOD data.

---

## Phase 9 — Temporal Causality & Prefetch

Discover temporal patterns and predict future memory accesses.

```python
# Test: "Does accessing A predict future access of B?"
result = mem.test_granger_causality(key_a, key_b, max_lag=10)
# → {"lag": 3, "strength": 0.85, "significant": True, "p_value": 0.002}

# Predict next accesses
predictions = mem.get_causal_predictions(query_embedding, n_predictions=3)
# → {"predictions": [{"pattern_id": "p_12", "probability": 0.7, "expected_lag": 2}]}

# Enable predictive prefetching (sub-ms for predicted patterns)
mem.enable_prefetch(enabled=True)

# Monitor prediction accuracy
stats = mem.get_prefetch_hit_rate()
# → {"hit_rate": 0.73, "avg_latency_saved_ms": 8.2}

# Explicitly mark temporal access
mem.record_temporal_access(key)
```

**Why it matters:** Sub-millisecond latencies for predictable access patterns.

---

## Phase 10 — Scaling, Text & Explainability

### Tiered Storage

```python
# Move patterns between performance tiers
mem.move_to_tier(key, tier="hot")   # ~1ms, in-memory
mem.move_to_tier(key, tier="warm")  # ~10ms, SSD
mem.move_to_tier(key, tier="cold")  # ~100ms, archive

# Monitor tier distribution
stats = mem.get_tier_stats()
# → {"hot": {"count": 500, "hit_rate": 0.92}, "warm": {...}, "cold": {...}}
```

### Text Interface (HDC Tokenizer)

```python
# Store and query with natural language — no manual embedding
mem.store_text("The user prefers Python over JavaScript")
mem.store_text("Project deadline is March 15th", metadata={"type": "fact"})

# Query by text
results = mem.query_text("what language does the user prefer?", top_k=3)

# Compose text concepts
blend = mem.compose_text(["Python", "API design"], weights=[0.7, 0.3])

# Inspect encoding
encoding = mem.get_text_encoding("machine learning")
```

### Explainability (XAI)

```python
# Full explanation of a retrieval decision
explanation = mem.explain(query_embedding)
# → {
#     "pathway_selected": "attention",
#     "confidence_breakdown": {"exact": 0.3, "energy": 0.5, "attention": 0.9},
#     "attention_map": [{"pattern_id": "p_42", "weight": 0.6}, ...],
#     "head_contributions": [{"head_id": 0, "weight": 0.4}, ...],
#     "regime": "A",
#     "modulation": 0.7
# }

# Full dashboard data (for custom visualization UIs)
dashboard = mem.get_xai_dashboard()

# Per-head analysis
contributions = mem.explain_head_contributions(query_embedding)

# Belief state on information hypersphere
belief = mem.visualize_belief_state()
```

### Snapshots & Versioning

```python
# Create a checkpoint
snap = mem.snapshot(name="before-experiment")

# Restore if things go wrong
mem.restore(snapshot_id=snap["snapshot_id"])

# List all versions
snapshots = mem.list_snapshots()
```

### Analytics

```python
# Usage metrics
metrics = mem.analytics(period="24h")
# → {
#     "total_queries": 12847,
#     "avg_latency_ms": 3.2,
#     "pathway_distribution": {"exact": 45, "energy": 30, "attention": 25},
#     "composition_requests": 1205,
#     "prefetch_hit_rate": 0.73,
#     "storage_used_mb": 42.5
# }
```

---

## Getting Started with Cloud

1. **Sign up** at [engramma-memory.dev/signup](https://engramma-memory.dev/signup)
2. **Get your key** (starts with `nx_live_` or `nx_test_`)
3. **One-line switch:**
```python
mem = EngrammaMemory(dim=256, backend="cloud", api_key="nx_live_...")
```

All local code continues to work unchanged. Cloud features are additive.
