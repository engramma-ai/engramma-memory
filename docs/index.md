# Engramma Memory

**A composable memory engine for AI systems that learns, adapts, and reasons — not just retrieves.**

---

## Why Engramma?

Vector databases find the nearest neighbor. That's it.

Engramma is a **memory engine** — it retrieves, composes, generalizes, and adapts:

| Feature | Vector DBs | Engramma Local | Engramma Cloud |
|---------|-----------|----------------|----------------|
| Exact recall | Yes | Yes | Yes |
| **Native composition** | No | **Yes** | **Yes (weighted)** |
| Soft generalization | No | Yes | Yes |
| Adaptive routing | No | Confidence-based | **Active Inference** |
| Causal reasoning | No | No | **Yes** |
| Safety regimes | No | No | **Yes** |
| Text interface | No | No | **Yes** |
| XAI explainability | No | No | **Yes** |

## Quick Example

```python
from engramma_memory import EngrammaMemory
import numpy as np

mem = EngrammaMemory(dim=256, backend="local")

# Store embeddings
mem.store(key=embedding_a, value=embedding_a)
mem.store(key=embedding_b, value=embedding_b)

# Native composition — no manual blending
blend = mem.compose([embedding_a, embedding_b])
```

## One Line to Production

```python
# Switch from local to cloud — same API, unlimited power
mem = EngrammaMemory(dim=256, backend="cloud", api_key="nx_live_...")

# Now you have access to 40+ cloud-exclusive features:
mem.explain(query)                # XAI: why was this returned?
mem.consolidate()                 # Sleep/Wake memory consolidation
mem.get_causal_graph()            # Discover causal structure
mem.query_text("user preferences") # Natural language queries
mem.get_current_regime()          # Safety regime monitoring
```

## Architecture

```
Query ──┬──> [Exact Memory]     ──> Perfect recall (kNN)
        ├──> [Energy Memory]    ──> Soft generalization (Hopfield)
        └──> [Multi-Head Attn]  ──> Native composition
                    │
            ConfidenceRouter ──> Best result       ← Local
            phi_B + EFE     ──> Optimal routing   ← Cloud
```

## Next Steps

- [5-Minute Quickstart](getting-started/quickstart.md) — Get running in 20 lines
- [Engramma vs Vector Databases](guides/engramma-vs-vectordbs.md) — Understand the difference
- [Building a Chatbot](guides/chatbot-memory.md) — Real-world example
- [Migrating to Cloud](guides/migrating-to-cloud.md) — Unlock Phases 1-10
- [Cloud Features](cloud/overview.md) — Full premium feature reference
- [API Reference](api-reference.md) — Full method documentation
