# Async API

Engramma provides a fully async interface for modern Python frameworks.

## Setup

```python
from engramma_memory import EngrammaMemoryAsync

# Use as context manager (recommended)
async with EngrammaMemoryAsync(dim=256, backend="cloud", api_key="nx_live_...") as mem:
    await mem.store(key=embedding, value=data)
    results = await mem.query(embedding, top_k=5)

# Or manage lifecycle manually
mem = EngrammaMemoryAsync(dim=256, backend="cloud", api_key="nx_live_...")
# ... use ...
await mem.close()
```

## All Features Available

Every cloud feature has an async equivalent:

```python
async with EngrammaMemoryAsync(dim=256, backend="cloud", api_key=key) as mem:
    # Core
    await mem.store(key=emb, value=emb)
    results = await mem.query(emb, top_k=5, use_phi_b=True)
    result = await mem.retrieve(emb)
    blend = await mem.compose([emb_a, emb_b], weights=[0.7, 0.3])

    # Phase 1 - Neuromodulation
    state = await mem.get_modulation_state()
    history = await mem.get_surprise_history(window=50)

    # Phase 2 - Routing
    trace = await mem.get_router_trace(emb)

    # Phase 3 - EFE
    results = await mem.query_with_epistemic_weight(emb, epistemic_w=1.0)
    await mem.set_pathway_strategy("explore")

    # Phase 4 - STDP
    spec = await mem.get_head_specialization()

    # Phase 6 - Causal
    blend = await mem.compose_fractional(emb_a, emb_b, alpha=0.7)
    graph = await mem.get_causal_graph()
    effect = await mem.predict_causal_effect(emb_a, emb_b)

    # Phase 7 - Discovery
    entropy = await mem.get_skeleton_entropy()
    await mem.auto_select_thresholds()

    # Phase 8 - Safety
    regime = await mem.get_current_regime()
    await mem.enable_anomaly_protection(True)
    await mem.consolidate()

    # Phase 9 - Temporal
    granger = await mem.test_granger_causality(emb_a, emb_b)
    predictions = await mem.get_causal_predictions(emb, n_predictions=3)
    await mem.enable_prefetch(True)

    # Phase 10 - Text & XAI
    await mem.store_text("user prefers Python")
    results = await mem.query_text("programming preferences")
    explanation = await mem.explain(emb)
    dashboard = await mem.get_xai_dashboard()
```

## With FastAPI

```python
from fastapi import FastAPI
from engramma_memory import EngrammaMemoryAsync

app = FastAPI()
mem: EngrammaMemoryAsync = None

@app.on_event("startup")
async def startup():
    global mem
    mem = EngrammaMemoryAsync(dim=256, backend="cloud", api_key="nx_live_...")

@app.on_event("shutdown")
async def shutdown():
    await mem.close()

@app.post("/remember")
async def remember(text: str):
    await mem.store_text(text)
    return {"status": "stored"}

@app.get("/recall")
async def recall(query: str, top_k: int = 5):
    return await mem.query_text(query, top_k=top_k)

@app.get("/health")
async def health():
    regime = await mem.get_current_regime()
    return {"regime": regime["regime"], "anomaly": regime["anomaly_signal"]}
```

## Local Backend (Sync Fallback)

The async interface also supports the local backend. Operations run synchronously under the hood but maintain the async signature for code consistency:

```python
# For testing/development
mem = EngrammaMemoryAsync(dim=256, backend="local")
await mem.store(key=emb, value=emb)  # runs sync internally
results = await mem.query(emb, top_k=3)

# Cloud-only features raise RuntimeError with local backend:
await mem.explain(emb)  # RuntimeError: explain() is a cloud-only feature.
```
