# Migrating to Engramma Cloud

## Why Cloud?

The local backend is powerful for development. But when your app grows, you'll hit walls:

| | Local | Cloud |
|--|-------|-------|
| Max patterns | 1000 | Unlimited (tiered) |
| Persistence | RAM only (lost on restart) | Persistent |
| Composition | Equal weights, attention only | Custom weights + 4 modes |
| Routing | Confidence-based | Active Inference + phi_B + EFE |
| Causal reasoning | None | Full DAG discovery |
| Safety | None | 3-regime anomaly detection |
| Temporal | None | Granger causality + prefetch |
| Text interface | None | HDC tokenizer |
| Explainability | None | Full XAI dashboard |
| Concurrency | Single-threaded | Multi-tenant + async |

## The Migration: One Line

```python
# Before (local)
mem = EngrammaMemory(dim=256, backend="local")

# After (cloud)
mem = EngrammaMemory(dim=256, backend="cloud", api_key="nx_live_...")
```

The API is identical. Every existing method works the same way. Your code doesn't change.

## Getting Your API Key

1. Sign up at [engramma-memory.com/signup](https://www.engramma-memory.com/signup)
2. Create a project in the dashboard
3. Copy your API key:
    - `nx_live_...` — production (persistent, billed)
    - `nx_test_...` — sandbox (ephemeral, free)

## What You Unlock

### Immediate (same methods, better behavior)

- `query()` uses Active Inference routing instead of simple confidence
- `compose()` supports custom weights and 4 composition modes
- `retrieve()` uses phi_B geometric selection
- All patterns persist across restarts
- No 1000-pattern limit

### New Methods (cloud-exclusive)

```python
# Phase 1: Introspect learning dynamics
state = mem.get_modulation_state()

# Phase 2: Understand routing
trace = mem.get_router_trace(query)

# Phase 3: Strategic queries
results = mem.query_with_epistemic_weight(query, epistemic_w=1.0)

# Phase 4: See head evolution
spec = mem.get_head_specialization()

# Phase 6: Causal reasoning
graph = mem.get_causal_graph()
effect = mem.predict_causal_effect(cause, effect)
blend = mem.compose_fractional(key_a, key_b, alpha=0.7)

# Phase 7: Auto-calibration
mem.auto_select_thresholds()

# Phase 8: Safety + consolidation
regime = mem.get_current_regime()
mem.enable_anomaly_protection(True)
mem.consolidate()

# Phase 9: Temporal prediction
granger = mem.test_granger_causality(key_a, key_b)
mem.enable_prefetch(True)

# Phase 10: Text + XAI
mem.store_text("user prefers Python")
results = mem.query_text("programming language preference")
explanation = mem.explain(query)
```

## Cloud-Exclusive Features

### Weighted Composition + Modes

```python
# Local: equal weights only, attention mode only
blend = mem.compose([key_a, key_b])

# Cloud: custom weights + multiple HDC modes
blend = mem.compose([key_a, key_b, key_c], weights=[0.6, 0.3, 0.1])
blend = mem.compose([key_a, key_b], mode="bind")       # new concept
blend = mem.compose([key_a, key_b], mode="bundle")     # superposition
blend = mem.compose([key_a, key_b], mode="sequential") # temporal
```

### Text Interface

```python
# No embedding needed — HDC tokenizer handles it
mem.store_text("The user is a senior Python developer")
mem.store_text("They work on distributed systems")

results = mem.query_text("what does the user do?", top_k=3)
blend = mem.compose_text(["Python", "distributed systems"])
```

### Safety Regimes

```python
# Enable automatic protection
mem.enable_anomaly_protection(True)

# Monitor operating regime
regime = mem.get_current_regime()
if regime["regime"] == "C":
    # System is in lockdown — OOD data detected
    # Composition is disabled, only exact recall available
    alert_team(regime["anomaly_signal"])
```

### Sleep/Wake Consolidation

```python
# Strengthen important patterns, prune dead weight
result = mem.consolidate()
print(f"Strengthened: {result['patterns_strengthened']}")
print(f"Pruned: {result['patterns_pruned']}")
print(f"Freed: {result['memory_freed_mb']} MB")
```

### Explainability

```python
# Understand any retrieval decision
explanation = mem.explain(query)
print(f"Pathway: {explanation['pathway_selected']}")
print(f"Top contributor: {explanation['attention_map'][0]}")
```

## Metadata Filtering

```python
mem.store(key=embedding, value=embedding,
          metadata={"user_id": "u_123", "topic": "python"})

results = mem.query(query, top_k=10,
                    filters={"user_id": "u_123", "topic": "python"})
```

## Async Support

For production async frameworks:

```python
from engramma_memory import EngrammaMemoryAsync

async with EngrammaMemoryAsync(dim=256, backend="cloud", api_key=key) as mem:
    await mem.store(key=emb, value=emb)
    results = await mem.query(emb, top_k=5)
    regime = await mem.get_current_regime()
```

## Environment Configuration

```python
import os

mem = EngrammaMemory(
    dim=256,
    backend="cloud",
    api_key=os.environ["ENGRAMMA_API_KEY"],
)
```

!!! warning "Never hardcode API keys"
    Use environment variables or a secrets manager.

## Hybrid Approach

Use local for development, cloud for production:

```python
import os

backend = "cloud" if os.environ.get("ENGRAMMA_API_KEY") else "local"
mem = EngrammaMemory(
    dim=256,
    backend=backend,
    api_key=os.environ.get("ENGRAMMA_API_KEY"),
)
```

## Pricing

| Tier | Patterns | Queries/month | Features | Price |
|------|----------|---------------|----------|-------|
| Free Trial | 5,000 | 50,000 | Phases 1-5 | $0 (14 days) |
| Starter | 10,000 | 100,000 | Phases 1-8 | $29/mo |
| Pro | 100,000 | Unlimited | All Phases | $79/mo |
| Scale | 1,000,000 | Unlimited | All + SLA 99.9% | $249/mo |
| Enterprise | Unlimited | Unlimited | On-premise option | Custom |

All paid tiers include: persistence, async, XAI dashboard, text interface, and safety regimes.

## FAQ

**Q: Will my local code break when switching to cloud?**

No. The API is identical. All local methods work the same. Cloud features are purely additive.

**Q: What about latency?**

Cloud adds network latency (~20-50ms). For latency-sensitive apps:
- Enable predictive prefetch (`mem.enable_prefetch(True)`)
- Use the hot tier for frequently accessed patterns
- Use async for non-blocking calls

**Q: Can I export data from cloud back to local?**

Yes. The dashboard includes bulk export. You can also query all patterns programmatically.

**Q: Is my data isolated?**

Yes. Each API key has its own isolated memory space. No cross-tenant access. SOC 2 compliant.

**Q: What happens during a cloud outage?**

Requests retry automatically (3 attempts with exponential backoff). For mission-critical apps, consider a local cache fallback.
