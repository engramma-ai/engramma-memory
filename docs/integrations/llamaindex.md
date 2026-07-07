# LlamaIndex Integration

## Installation

```bash
pip install engramma-memory[llamaindex]
```

## As a Retriever

```python
from engramma_memory.integrations.llamaindex import EngrammaRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

def my_embedder(text: str) -> list:
    ...

retriever = EngrammaRetriever(dim=256, embed_fn=my_embedder, top_k=5)

# Ingest documents
retriever.add_texts(["Document 1 content", "Document 2 content", ...])

# Use with a query engine
query_engine = RetrieverQueryEngine(retriever=retriever)
response = query_engine.query("What is the relationship between X and Y?")
```

## As a Vector Store

```python
from engramma_memory.integrations.llamaindex import EngrammaVectorStore

vector_store = EngrammaVectorStore(dim=256, embed_fn=my_embedder)

# Drop-in replacement for ChromaVectorStore or FaissVectorStore
vector_store.add(nodes)
results = vector_store.query(query_bundle)
```

## Compositional Queries

The real power is compositional retrieval — not available in standard LlamaIndex vector stores:

```python
# Standard: retrieve docs similar to ONE query
results = retriever.retrieve("Python")

# Engramma: retrieve docs that bridge MULTIPLE topics
composed = retriever.compose_query(["Python", "machine learning", "APIs"])
```

## Cloud Backend

```python
retriever = EngrammaRetriever(
    dim=256,
    embed_fn=my_embedder,
    backend="cloud",
    api_key="nx_live_...",
)
```
