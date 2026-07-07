# 5-Minute Quickstart

Get Engramma Memory running in 20 lines. By the end, you'll understand: store, query, compose, and retrieve.

## Setup

```python
from engramma_memory import EngrammaMemory
import numpy as np

mem = EngrammaMemory(dim=64, backend="local")
```

## Store Embeddings

Store key-value pairs. Keys are used for retrieval, values are what you get back.

```python
rng = np.random.default_rng(42)

# Create and store concept embeddings
concepts = {}
for name in ["python", "javascript", "ml", "web_dev", "data_science"]:
    embedding = rng.standard_normal(64).astype(np.float32)
    embedding /= np.linalg.norm(embedding)
    mem.store(key=embedding, value=embedding)
    concepts[name] = embedding
```

## Query — Find Relevant Memories

```python
results = mem.query(concepts["python"], top_k=2)
print(f"Top match score: {results[0]['score']:.4f}")
# Returns list of {"value": ndarray, "score": float}
```

## Compose — Blend Multiple Patterns

This is Engramma's killer feature. Vector databases can't do this natively.

```python
# Compose "python" + "data_science" into a single blended pattern
blend = mem.compose([concepts["python"], concepts["data_science"]])
print(f"Composed vector norm: {np.linalg.norm(blend):.4f}")
```

The composed result attends to both patterns simultaneously through multi-head attention — it's not a naive average.

## Retrieve — Smart Routing

```python
# ConfidenceRouter picks the best pathway automatically
result = mem.retrieve(concepts["ml"])
similarity = np.dot(result, concepts["ml"]) / (
    np.linalg.norm(result) * np.linalg.norm(concepts["ml"]) + 1e-8
)
print(f"Similarity: {similarity:.4f}")
```

## Forget — Remove Patterns

```python
mem.forget(concepts["javascript"], strategy="decay")     # Reduce importance
mem.forget(concepts["web_dev"], strategy="immediate")    # Delete now
```

## Check Stats

```python
print(mem.stats())
# {'exact_count': 3, 'capacity': 1000, 'dim': 64, ...}
print(f"Patterns stored: {mem.count}")
```

## Next Steps

- [Engramma vs Vector Databases](../guides/engramma-vs-vectordbs.md) — Why composition matters
- [Building a Chatbot](../guides/chatbot-memory.md) — Real-world use case
- [API Reference](../api-reference.md) — Full documentation

!!! tip "Ready for production?"
    The local backend is capped at 1000 patterns. For unlimited storage, persistence, and weighted composition, switch to [Engramma Cloud](../guides/migrating-to-cloud.md) — one line change.
