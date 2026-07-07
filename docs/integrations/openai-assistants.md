# OpenAI Assistants Integration

## Installation

No extra dependencies needed — works with the base `engramma-memory` package.

```bash
pip install engramma-memory
```

## Setup

```python
from engramma_memory.integrations.openai_assistants import (
    engramma_tool_definitions,
    EngrammaToolHandler,
)

# Get tool definitions for your Assistant
tools = engramma_tool_definitions()

# Create handler
handler = EngrammaToolHandler(dim=256, embed_fn=my_embedder)
```

## Tool Definitions

`engramma_tool_definitions()` returns 4 function-calling tools:

| Tool | Description |
|------|-------------|
| `engramma_store` | Store content in memory |
| `engramma_query` | Query for relevant memories |
| `engramma_compose` | Compose multiple topics |
| `engramma_forget` | Remove a memory |

## Handling Tool Calls

```python
import openai

client = openai.OpenAI()

# Create assistant with Engramma tools
assistant = client.beta.assistants.create(
    name="Assistant with Memory",
    instructions="You have long-term memory. Use engramma_store to remember and engramma_query to recall.",
    model="gpt-4",
    tools=tools,
)

# In your run loop, handle tool calls:
def handle_tool_call(tool_call):
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)
    result = handler.handle(name, args)
    return result  # Returns JSON string
```

## Example Flow

```python
# Assistant decides to remember something
handler.handle("engramma_store", {"content": "User prefers Python over JavaScript"})

# Later, assistant recalls
result = handler.handle("engramma_query", {"query": "user language preferences"})
# -> {"results": [{"text": "User prefers Python over JavaScript", "score": 0.92}]}

# Compositional query
result = handler.handle("engramma_compose", {"topics": ["Python", "web development"]})
# -> {"composed_text": "...", "topics": ["Python", "web development"]}
```

## Cloud Backend

```python
handler = EngrammaToolHandler(
    dim=256,
    embed_fn=my_embedder,
    backend="cloud",
    api_key="nx_live_...",
)
```
