"""
FastAPI integration for Engramma Memory.

Provides a ready-to-use router and middleware for adding memory
capabilities to any FastAPI application.

Usage:
    from fastapi import FastAPI
    from engramma_memory.integrations.fastapi import create_memory_router

    app = FastAPI()
    memory_router = create_memory_router(dim=256)
    app.include_router(memory_router, prefix="/memory")
"""
from typing import Optional, List, Dict, Any

import numpy as np

from ..core import EngrammaMemory


def create_memory_router(dim: int = 256, backend: str = "local",
                         api_key: Optional[str] = None,
                         embed_fn=None):
    """
    Create a FastAPI router with Engramma memory endpoints.

    Endpoints:
        POST /store    - Store a key-value pair
        POST /query    - Query memory
        POST /compose  - Compose multiple memories
        POST /forget   - Forget a memory
        GET  /stats    - Memory statistics

    Parameters
    ----------
    dim : int
        Embedding dimension.
    backend : str
        "local" or "cloud".
    api_key : str, optional
        API key for cloud backend.
    embed_fn : callable, optional
        Text-to-embedding function for text-based endpoints.
    """
    try:
        from fastapi import APIRouter, HTTPException
        from pydantic import BaseModel, Field
    except ImportError:
        raise ImportError(
            "FastAPI integration requires fastapi and pydantic. "
            "Install with: pip install engramma-memory[fastapi]"
        )

    router = APIRouter(tags=["memory"])
    engramma = EngrammaMemory(dim=dim, backend=backend, api_key=api_key)

    _embed_fn = embed_fn or _default_embed_factory(dim)

    class StoreRequest(BaseModel):
        key: Optional[List[float]] = None
        text: Optional[str] = None
        value: Optional[List[float]] = None

    class QueryRequest(BaseModel):
        query: Optional[List[float]] = None
        text: Optional[str] = None
        top_k: int = Field(default=5, ge=1, le=100)

    class ComposeRequest(BaseModel):
        keys: Optional[List[List[float]]] = None
        texts: Optional[List[str]] = None
        weights: Optional[List[float]] = None

    class ForgetRequest(BaseModel):
        key: Optional[List[float]] = None
        text: Optional[str] = None
        strategy: str = "decay"

    @router.post("/store")
    async def store(req: StoreRequest):
        if req.key is not None:
            key = np.array(req.key, dtype=np.float32)
        elif req.text is not None:
            key = _embed_fn(req.text)
        else:
            raise HTTPException(400, "Provide 'key' (vector) or 'text'")

        value = np.array(req.value, dtype=np.float32) if req.value else key
        engramma.store(key=key, value=value)
        return {"status": "stored", "count": engramma.count}

    @router.post("/query")
    async def query(req: QueryRequest):
        if req.query is not None:
            q = np.array(req.query, dtype=np.float32)
        elif req.text is not None:
            q = _embed_fn(req.text)
        else:
            raise HTTPException(400, "Provide 'query' (vector) or 'text'")

        results = engramma.query(q, top_k=req.top_k)
        return {
            "results": [
                {"value": r["value"].tolist(), "score": float(r["score"])}
                for r in results
            ]
        }

    @router.post("/compose")
    async def compose(req: ComposeRequest):
        if req.keys is not None:
            keys = [np.array(k, dtype=np.float32) for k in req.keys]
        elif req.texts is not None:
            keys = [_embed_fn(t) for t in req.texts]
        else:
            raise HTTPException(400, "Provide 'keys' (vectors) or 'texts'")

        result = engramma.compose(keys, weights=req.weights)
        return {"result": result.tolist()}

    @router.post("/forget")
    async def forget(req: ForgetRequest):
        if req.key is not None:
            key = np.array(req.key, dtype=np.float32)
        elif req.text is not None:
            key = _embed_fn(req.text)
        else:
            raise HTTPException(400, "Provide 'key' (vector) or 'text'")

        engramma.forget(key, strategy=req.strategy)
        return {"status": "forgotten", "strategy": req.strategy}

    @router.get("/stats")
    async def stats():
        return engramma.stats()

    return router


def _default_embed_factory(dim: int):
    def embed(text: str) -> np.ndarray:
        rng = np.random.default_rng(hash(text) % (2**32))
        vec = rng.standard_normal(dim).astype(np.float32)
        return vec / (np.linalg.norm(vec) + 1e-8)
    return embed


class EngrammaMiddleware:
    """
    ASGI middleware that injects Engramma memory into request state.

    Usage:
        app.add_middleware(EngrammaMiddleware, dim=256)

        @app.get("/chat")
        async def chat(request: Request):
            mem = request.state.engramma_memory
            result = mem.query(embedding, top_k=5)
    """

    def __init__(self, app, dim: int = 256, backend: str = "local",
                 api_key: Optional[str] = None):
        self.app = app
        self._engramma = EngrammaMemory(dim=dim, backend=backend, api_key=api_key)

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            scope.setdefault("state", {})
            scope["state"]["engramma_memory"] = self._engramma
        await self.app(scope, receive, send)
