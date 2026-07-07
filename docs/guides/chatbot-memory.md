# Building a Chatbot with Long-Term Memory

This guide shows how to give your chatbot persistent, compositional memory. The bot remembers past conversations and can blend related memories to build richer context.

## The Problem

Standard chatbots have two memory options:

1. **Context window** — limited to the last N messages, then forgotten
2. **Vector store** — retrieves individual past messages, no synthesis

Neither handles: *"Based on everything you know about me, what should I try next?"*

Engramma adds a third option: **compositional memory** that attends to multiple stored patterns simultaneously.

## Basic Setup

```python
import numpy as np
from engramma_memory import EngrammaMemory


def embed(text: str, dim: int = 128) -> np.ndarray:
    """Replace with your embedding model (OpenAI, sentence-transformers, etc.)"""
    rng = np.random.default_rng(hash(text) % (2**32))
    vec = rng.standard_normal(dim).astype(np.float32)
    return vec / (np.linalg.norm(vec) + 1e-8)


class ChatbotWithMemory:
    def __init__(self, dim: int = 128):
        self.mem = EngrammaMemory(dim=dim, backend="local")
        self.dim = dim
        self.history: list = []

    def remember(self, text: str):
        """Store a message in long-term memory."""
        embedding = embed(text, self.dim)
        self.mem.store(key=embedding, value=embedding)
        self.history.append(text)

    def recall(self, query: str, top_k: int = 3) -> list:
        """Find relevant past messages."""
        embedding = embed(query, self.dim)
        results = self.mem.query(embedding, top_k=top_k)
        return [self._match(r["value"]) for r in results if r["score"] > 0.2]

    def compose_context(self, topics: list) -> str:
        """Blend multiple topics into unified context (Engramma-exclusive)."""
        embeddings = [embed(t, self.dim) for t in topics]
        composed = self.mem.compose(embeddings)
        return self._match(composed)

    def _match(self, value: np.ndarray) -> str:
        """Find the stored message closest to a value vector."""
        value = value.flatten()[:self.dim]
        best_dist, best_text = float('inf'), ""
        for text in self.history:
            dist = float(np.linalg.norm(embed(text, self.dim) - value))
            if dist < best_dist:
                best_dist, best_text = dist, text
        return best_text
```

## Usage

```python
bot = ChatbotWithMemory()

# User tells the bot things over time
bot.remember("I love programming in Python")
bot.remember("Machine learning is fascinating")
bot.remember("I'm building a recommendation system")
bot.remember("My project deadline is next Friday")
bot.remember("I need help with PyTorch")
```

### Simple Recall

```python
# "What does this user know about AI?"
recalled = bot.recall("AI projects", top_k=3)
# -> ["Machine learning is fascinating", "I'm building a recommendation system", ...]
```

### Compositional Context (Engramma Advantage)

```python
# "What connects Python AND machine learning for this user?"
context = bot.compose_context(["Python", "machine learning"])
# -> Returns the message that best bridges both topics
```

This is fundamentally different from retrieving "Python" results and "ML" results separately. Engramma's multi-head attention finds patterns that relate to **both** topics simultaneously.

## With a Real LLM

```python
import openai

class SmartChatbot(ChatbotWithMemory):
    def respond(self, user_message: str) -> str:
        # Remember what the user said
        self.remember(user_message)

        # Recall relevant context
        context = self.recall(user_message, top_k=5)

        # Build prompt with memory
        system = (
            "You are a helpful assistant with memory of past conversations.\n"
            f"Relevant memories:\n" + "\n".join(f"- {c}" for c in context)
        )

        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_message},
            ]
        )

        answer = response.choices[0].message.content

        # Remember the response too
        self.remember(answer)
        return answer
```

## With LangChain

```python
from engramma_memory.integrations.langchain import EngrammaLangChainMemory
from langchain.chains import ConversationChain
from langchain_openai import ChatOpenAI

memory = EngrammaLangChainMemory(dim=128, embed_fn=embed)
chain = ConversationChain(llm=ChatOpenAI(), memory=memory)

response = chain.invoke({"input": "Tell me about Python"})
```

## Forgetting Old Information

```python
# User updates their preferences
bot.remember("Actually, I switched from PyTorch to JAX")

# Decay the old information
old_embedding = embed("I need help with PyTorch")
bot.mem.forget(old_embedding, strategy="decay")
```

## Production Considerations

The local backend is great for prototyping but has limits:

| Limit | Local | Cloud |
|-------|-------|-------|
| Max memories | 1000 | Unlimited |
| Persistence | RAM only | Persistent |
| Weighted composition | Equal only | Custom weights |

For a production chatbot, switch to cloud:

```python
class ProductionChatbot(ChatbotWithMemory):
    def __init__(self):
        self.mem = EngrammaMemory(dim=128, backend="cloud", api_key="nx_live_...")
        # Everything else stays the same
```

See [Migrating to Engramma Cloud](migrating-to-cloud.md) for the full guide.
