# CrewAI Integration

## Installation

```bash
pip install engramma-memory[crewai]
```

## Usage

```python
from engramma_memory.integrations.crewai import EngrammaCrewMemory

memory = EngrammaCrewMemory(dim=256, embed_fn=my_embedder)

# Store knowledge
memory.save("The project uses FastAPI for the backend")
memory.save("Deployment target is AWS ECS")
memory.save("Database is PostgreSQL with pgvector extension")

# Search relevant memories
results = memory.search("infrastructure setup", top_k=3)
for r in results:
    print(f"[{r['score']:.2f}] {r['text']}")
```

## Compositional Context (Engramma-Exclusive)

This is not possible with standard CrewAI memory backends:

```python
# Find knowledge that bridges multiple topics
context = memory.compose_context(["FastAPI", "PostgreSQL", "deployment"])
# Returns the memory that best connects all three topics
```

Use this to give agents richer context when their task spans multiple domains.

## With CrewAI Agents

```python
from crewai import Agent, Task, Crew

memory = EngrammaCrewMemory(dim=256, embed_fn=my_embedder)

# Pre-load shared knowledge
memory.save("Company coding standard: Python 3.11+, type hints required")
memory.save("API versioning: /v1/ prefix, semver for breaking changes")

# Agent can access composed context
researcher = Agent(
    role="Technical Researcher",
    goal="Find relevant technical context",
    backstory="You have access to team knowledge via Engramma memory.",
)

# In your task logic, inject composed context
context = memory.compose_context(["Python", "API design", "best practices"])
```

## Reset

```python
memory.reset()  # Clear all stored memories
```

## Cloud Backend

```python
memory = EngrammaCrewMemory(
    dim=256,
    embed_fn=my_embedder,
    backend="cloud",
    api_key="nx_live_...",
)
```
