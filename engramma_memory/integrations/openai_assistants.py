"""
OpenAI Assistants API integration for Engramma Memory.

Provides tool definitions and handlers for using Engramma as a tool
in OpenAI Assistants / Function Calling.

Usage:
    from engramma_memory.integrations.openai_assistants import (
        engramma_tool_definitions,
        EngrammaToolHandler,
    )

    tools = engramma_tool_definitions()
    handler = EngrammaToolHandler(dim=256)

    # In your assistant loop:
    result = handler.handle(tool_name, arguments)
"""
import json
from typing import Dict, Any, Optional, List

import numpy as np

from ..core import EngrammaMemory


def engramma_tool_definitions() -> List[Dict[str, Any]]:
    """
    Returns OpenAI-compatible tool definitions for Engramma Memory.

    Add these to your Assistant's tools list.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "engramma_store",
                "description": "Store information in long-term memory for later retrieval. Use this to remember facts, context, or user preferences.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "The content/information to store in memory."
                        },
                        "category": {
                            "type": "string",
                            "description": "Optional category tag (e.g., 'user_preference', 'fact', 'context').",
                            "enum": ["user_preference", "fact", "context", "conversation", "task"]
                        }
                    },
                    "required": ["content"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "engramma_query",
                "description": "Search long-term memory for relevant information. Use this to recall stored facts or context.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "What to search for in memory."
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of results to return (default: 3).",
                            "default": 3
                        }
                    },
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "engramma_compose",
                "description": "Compose/blend multiple memories together. Useful for combining related concepts or creating a synthesized view.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "queries": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of memory queries to compose together."
                        }
                    },
                    "required": ["queries"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "engramma_forget",
                "description": "Remove or decay a specific memory. Use when information is outdated or user requests deletion.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The memory to forget."
                        },
                        "strategy": {
                            "type": "string",
                            "enum": ["decay", "immediate"],
                            "description": "How to forget: 'decay' (gradual) or 'immediate' (delete now).",
                            "default": "decay"
                        }
                    },
                    "required": ["query"]
                }
            }
        }
    ]


class EngrammaToolHandler:
    """
    Handles Engramma tool calls from OpenAI Assistants.

    Parameters
    ----------
    dim : int
        Embedding dimension.
    embed_fn : callable, optional
        Text-to-embedding function. If None, uses a hash-based embedder.
    backend : str
        "local" or "cloud".
    api_key : str, optional
        API key for cloud backend.
    """

    def __init__(self, dim: int = 256, embed_fn=None,
                 backend: str = "local", api_key: Optional[str] = None):
        self._dim = dim
        self._embed_fn = embed_fn or self._default_embed
        self._engramma = EngrammaMemory(dim=dim, backend=backend, api_key=api_key)
        self._text_store: Dict[int, str] = {}
        self._store_count = 0

    def handle(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """
        Handle a tool call and return the result as a string.

        Parameters
        ----------
        tool_name : str
            One of: engramma_store, engramma_query, engramma_compose, engramma_forget
        arguments : dict
            The parsed arguments from the tool call.

        Returns
        -------
        str
            JSON-formatted result.
        """
        handlers = {
            "engramma_store": self._handle_store,
            "engramma_query": self._handle_query,
            "engramma_compose": self._handle_compose,
            "engramma_forget": self._handle_forget,
        }

        handler = handlers.get(tool_name)
        if handler is None:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})

        try:
            result = handler(arguments)
            return json.dumps(result)
        except Exception as e:
            return json.dumps({"error": str(e)})

    def _handle_store(self, args: Dict) -> Dict:
        content = args["content"]
        embedding = self._embed_fn(content)
        self._engramma.store(key=embedding, value=embedding)
        self._text_store[self._store_count] = content
        self._store_count += 1
        return {"status": "stored", "memory_count": self._engramma.count}

    def _handle_query(self, args: Dict) -> Dict:
        query = args["query"]
        top_k = args.get("top_k", 3)
        embedding = self._embed_fn(query)
        results = self._engramma.query(embedding, top_k=top_k)

        memories = []
        for r in results:
            if r["score"] > 0.2:
                text = self._find_text(r["value"])
                if text:
                    memories.append({"content": text, "relevance": round(r["score"], 3)})

        return {"memories": memories, "count": len(memories)}

    def _handle_compose(self, args: Dict) -> Dict:
        queries = args["queries"]
        embeddings = [self._embed_fn(q) for q in queries]
        result = self._engramma.compose(embeddings)
        text = self._find_text(result)
        return {"composed": text or "(synthesized memory)", "inputs": queries}

    def _handle_forget(self, args: Dict) -> Dict:
        query = args["query"]
        strategy = args.get("strategy", "decay")
        embedding = self._embed_fn(query)
        self._engramma.forget(embedding, strategy=strategy)
        return {"status": "forgotten", "strategy": strategy}

    def _find_text(self, value: np.ndarray) -> Optional[str]:
        if not self._text_store:
            return None
        value = value.flatten()[:self._dim]
        best_dist = float('inf')
        best_text = None
        for idx, text in self._text_store.items():
            emb = self._embed_fn(text)
            dist = float(np.linalg.norm(emb.flatten()[:self._dim] - value))
            if dist < best_dist:
                best_dist = dist
                best_text = text
        return best_text if best_dist < 2.0 else None

    def _default_embed(self, text: str) -> np.ndarray:
        rng = np.random.default_rng(hash(text) % (2**32))
        vec = rng.standard_normal(self._dim).astype(np.float32)
        return vec / (np.linalg.norm(vec) + 1e-8)
