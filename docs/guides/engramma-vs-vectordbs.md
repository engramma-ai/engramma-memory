# Engramma vs Vector Databases (Pinecone, Chroma, FAISS)

## The Core Difference

Vector databases are **retrieval engines** — they find the single nearest neighbor to your query. That's their entire job.

Engramma is a **memory engine** — it retrieves, composes, generalizes, and adapts. The key difference is **native composition**: the ability to blend multiple stored patterns into coherent results without manual post-processing.

## Head-to-Head Comparison

| Capability | Pinecone/Chroma/FAISS | Engramma Memory |
|-----------|----------------------|--------------|
| Exact recall | Yes | Yes |
| Native composition | No | **Yes** |
| Soft generalization | No | Yes (Hopfield energy) |
| Adaptive routing | No | Yes (ConfidenceRouter) |
| Continual learning | No (static index) | Yes (Hebbian) |
| GPU required | Often | Never |
| Managed cloud | Yes (paid) | Yes (Engramma Cloud) |
| Open source local | Chroma/FAISS yes | Yes |

## The Composition Problem

Consider a chatbot that knows:

- "User likes Python" (stored as embedding A)
- "User works in data science" (stored as embedding B)

Now you ask: *"What projects would this user enjoy?"*

**Vector DB approach:**
```python
# Retrieve A and B separately, then manually combine
result_a = index.query(embed("python"))
result_b = index.query(embed("data science"))
combined = (result_a + result_b) / 2  # Naive average — not meaningful!
```

This naive average doesn't represent any stored pattern. It's a point in space that may not correspond to anything useful.

**Engramma approach:**
```python
# Native composition through multi-head attention
composed = mem.compose([embed("python"), embed("data science")])
```

Each attention head attends to different aspects of both patterns. The result is a coherent composition that reflects the relationship between the inputs, not just their midpoint.

## Benchmark Results (Real Data)

| Task | Engramma | FAISS | ChromaDB | NumPy kNN |
|------|-------|-------|----------|-----------|
| Exact recall @1000 | 100% | 100% | 100% | 100% |
| **Composition (50/50 blend)** | **81.4%** | 70.6% | 70.0% | 70.3% |
| **Composition (triple blend)** | **68.4%** | 56.6% | 57.0% | 55.9% |
| **Continual learning** | **8.6%** | 1.1% | 1.1% | 1.1% |
| Noisy retrieval (sigma=0.3) | 70.0% | 70.5% | 72.0% | 62.5% |

## Where Engramma Wins

**Composition (+15-20% over kNN)**

Engramma's multi-head attention naturally composes patterns. Each head has a different temperature: sharp heads recall exact patterns, soft heads blend across stored memories. The combined output is meaningful composition — not a geometric midpoint.

**Continual Learning (8x better retention)**

When memory is full and new patterns push out old ones, Engramma retains 8.6% of original patterns vs 1.1% for vector databases. The energy-based pathway (modern Hopfield network) creates soft attractors that preserve important patterns even as the exact memory evicts them.

## Where Engramma Loses (Honestly)

**Latency**

| System | p50 latency @1000 patterns |
|--------|---------------------------|
| FAISS | 0.02 ms |
| NumPy kNN | 0.50 ms |
| ChromaDB | 0.75 ms |
| Engramma | 8.8 ms |

Engramma routes through three pathways (exact, energy, attention). This adds overhead. For latency-critical applications with simple retrieval needs, FAISS is faster.

**Memory Footprint**

| System | MB per 1000 patterns |
|--------|---------------------|
| ChromaDB | 0.46 MB |
| FAISS / NumPy | 0.72 MB |
| Engramma | 3.14 MB |

Three pathways = ~3x storage. Each pathway maintains its own representation of the stored patterns.

**Noisy Retrieval**

On corrupted queries (sigma=0.3 noise), Engramma scores 70.0% vs ChromaDB's 72.0%. The multi-pathway routing doesn't help with noise — it's roughly equivalent to direct kNN.

## When to Use Engramma

Use Engramma when you need:

- **Compositional queries**: "What connects topic A and topic B?"
- **Adaptive memory**: System should learn what's important
- **Agent memory**: Long-running agents that build context over time
- **Conversational AI**: Blending multiple user preferences/history

## When to Use a Vector Database

Use Pinecone/Chroma/FAISS when you need:

- **Pure similarity search**: Find the nearest document, nothing more
- **Sub-millisecond latency**: Engramma is 10-100x slower for simple lookups
- **Minimal memory**: Storage-constrained environments
- **Billion-scale**: Engramma local caps at 1000 patterns (use Engramma Cloud for unlimited)

## Migration Path

Already using a vector database? Engramma works alongside it:

```python
from engramma_memory import EngrammaMemory

# Use Engramma for compositional queries only
engramma = EngrammaMemory(dim=256, backend="local")

# Your existing vector DB handles simple retrieval
# Engramma handles "connect the dots" queries
composed = engramma.compose([query_a, query_b, query_c])
```

Or replace entirely with one line:

```python
mem = EngrammaMemory(dim=256, backend="cloud", api_key="nx_live_...")
```
