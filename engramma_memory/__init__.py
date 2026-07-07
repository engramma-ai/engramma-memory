"""
Engramma Memory - A composable memory engine for AI systems.

Quick start:
    >>> from engramma_memory import EngrammaMemory
    >>> mem = EngrammaMemory(dim=256, backend="local")
    >>> mem.store(key=embedding, value=data)
    >>> result = mem.query(embedding, top_k=3)
    >>> blend = mem.compose([key_a, key_b])

Async:
    >>> from engramma_memory import EngrammaMemoryAsync
    >>> mem = EngrammaMemoryAsync(dim=256, backend="cloud", api_key="nx_live_...")
    >>> await mem.store(key=embedding, value=data)
"""

from .async_core import EngrammaMemoryAsync
from .core import EngrammaMemory

__version__ = "0.1.0"
__all__ = ["EngrammaMemory", "EngrammaMemoryAsync"]
