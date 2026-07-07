# Installation

## Requirements

- Python 3.9+
- NumPy (installed automatically)

## Basic Install

```bash
pip install engramma-memory
```

## With Integrations

Install only what you need:

=== "LangChain"

    ```bash
    pip install engramma-memory[langchain]
    ```

=== "LlamaIndex"

    ```bash
    pip install engramma-memory[llamaindex]
    ```

=== "OpenAI Assistants"

    ```bash
    pip install engramma-memory[openai]
    ```

=== "CrewAI"

    ```bash
    pip install engramma-memory[crewai]
    ```

=== "FastAPI"

    ```bash
    pip install engramma-memory[fastapi]
    ```

=== "Everything"

    ```bash
    pip install engramma-memory[all]
    ```

## Development Install

```bash
git clone https://github.com/engramma-ai/engramma-memory.git
cd engramma-memory
pip install -e ".[dev]"
```

## Verify Installation

```python
from engramma_memory import EngrammaMemory
mem = EngrammaMemory(dim=64, backend="local")
print(mem)  # EngrammaMemory(dim=64, backend='local', count=0)
```
