# LangChain Integration

## Installation

```bash
pip install engramma-memory[langchain]
```

## Usage

```python
from engramma_memory.integrations.langchain import EngrammaLangChainMemory
from langchain.chains import ConversationChain
from langchain_openai import ChatOpenAI

def my_embedder(text: str) -> list:
    """Your embedding function (OpenAI, sentence-transformers, etc.)"""
    # Return a list/array of floats
    ...

memory = EngrammaLangChainMemory(dim=256, embed_fn=my_embedder)
chain = ConversationChain(llm=ChatOpenAI(), memory=memory)

# Use as normal — Engramma handles memory behind the scenes
response = chain.invoke({"input": "I love Python programming"})
response = chain.invoke({"input": "Tell me about data science"})

# Memory now contains both messages and can compose them
```

## How It Works

`EngrammaLangChainMemory` implements LangChain's `BaseMemory` interface:

- `save_context()` — Stores each message exchange as embeddings in Engramma
- `load_memory_variables()` — Retrieves relevant past context for the current query
- `clear()` — Resets memory

The key difference from LangChain's built-in memory types: Engramma uses compositional retrieval, not just recency or simple similarity.

## With ConversationBufferMemory-style behavior

```python
# Stores full conversation history, retrieves compositionally
memory = EngrammaLangChainMemory(
    dim=256,
    embed_fn=my_embedder,
    memory_key="history",     # Variable name in prompt
    top_k=5,                  # How many memories to retrieve
)
```

## Cloud Backend

```python
memory = EngrammaLangChainMemory(
    dim=256,
    embed_fn=my_embedder,
    backend="cloud",
    api_key="nx_live_...",
)
```
