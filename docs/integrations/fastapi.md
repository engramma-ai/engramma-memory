# FastAPI Integration

## Installation

```bash
pip install engramma-memory[fastapi]
```

## Quick Setup

```python
from fastapi import FastAPI
from engramma_memory.integrations.fastapi import create_memory_router

app = FastAPI()

def my_embedder(text: str) -> list:
    ...

router = create_memory_router(dim=256, embed_fn=my_embedder)
app.include_router(router, prefix="/memory")
```

This gives you 5 REST endpoints instantly.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/memory/store` | Store content |
| POST | `/memory/query` | Query memories |
| POST | `/memory/compose` | Compose topics |
| POST | `/memory/forget` | Forget content |
| GET | `/memory/stats` | Memory statistics |

## Request/Response Examples

### Store

```bash
curl -X POST http://localhost:8000/memory/store \
  -H "Content-Type: application/json" \
  -d '{"content": "User prefers dark mode", "metadata": {}}'
```

### Query

```bash
curl -X POST http://localhost:8000/memory/query \
  -H "Content-Type: application/json" \
  -d '{"query": "user preferences", "top_k": 5}'
```

Response:
```json
{
  "results": [
    {"text": "User prefers dark mode", "score": 0.89},
    ...
  ]
}
```

### Compose

```bash
curl -X POST http://localhost:8000/memory/compose \
  -H "Content-Type: application/json" \
  -d '{"topics": ["UI preferences", "accessibility"]}'
```

### Stats

```bash
curl http://localhost:8000/memory/stats
```

Response:
```json
{
  "exact_count": 42,
  "capacity": 1000,
  "dim": 256,
  "backend": "local"
}
```

## Middleware

For injecting Engramma into every request:

```python
from engramma_memory.integrations.fastapi import EngrammaMiddleware

app.add_middleware(EngrammaMiddleware, dim=256, embed_fn=my_embedder)

# Access in any route:
@app.get("/my-route")
async def my_route(request: Request):
    mem = request.state.engramma_memory
    results = mem.query(embedding, top_k=3)
```

## Cloud Backend

```python
router = create_memory_router(
    dim=256,
    embed_fn=my_embedder,
    backend="cloud",
    api_key="nx_live_...",
)
```

## Full Example

```python
from fastapi import FastAPI
from engramma_memory.integrations.fastapi import create_memory_router
import numpy as np

app = FastAPI(title="My App with Memory")

def embed(text: str) -> list:
    rng = np.random.default_rng(hash(text) % (2**32))
    vec = rng.standard_normal(256).astype(np.float32)
    vec /= np.linalg.norm(vec)
    return vec.tolist()

router = create_memory_router(dim=256, embed_fn=embed)
app.include_router(router, prefix="/memory")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```
