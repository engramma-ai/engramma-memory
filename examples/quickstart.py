"""
Engramma Memory - 5-Minute Quickstart

This example shows the core API in 20 lines.
"""
import numpy as np
from engramma_memory import EngrammaMemory

# 1. Create a memory engine
mem = EngrammaMemory(dim=64, backend="local")

# 2. Store some embeddings (these would come from your embedding model)
rng = np.random.default_rng(42)
concepts = {}
for name in ["python", "javascript", "machine_learning", "web_dev", "data_science"]:
    embedding = rng.standard_normal(64).astype(np.float32)
    embedding /= np.linalg.norm(embedding)
    mem.store(key=embedding, value=embedding)
    concepts[name] = embedding
    print(f"Stored: {name}")

# 3. Query - find the most relevant memory
results = mem.query(concepts["python"], top_k=2)
print(f"\nQuery 'python' -> top result score: {results[0]['score']:.4f}")

# 4. Compose - blend two concepts natively
blend = mem.compose([concepts["python"], concepts["data_science"]])
print(f"\nComposed 'python' + 'data_science' -> vector norm: {np.linalg.norm(blend):.4f}")

# 5. Smart retrieval with routing
result = mem.retrieve(concepts["machine_learning"])
sim = np.dot(result, concepts["machine_learning"]) / (
    np.linalg.norm(result) * np.linalg.norm(concepts["machine_learning"]) + 1e-8
)
print(f"Smart retrieve 'machine_learning' -> similarity: {sim:.4f}")

# 6. Check stats
print(f"\nMemory stats: {mem.stats()}")
print(f"Patterns stored: {mem.count}")
