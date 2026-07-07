# Contributing to Engramma Memory

Thank you for your interest in contributing! This guide will help you get started.

## Development Setup

```bash
git clone https://github.com/engramma-ai/engramma-memory.git
cd engramma-memory
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e ".[dev,docs]"
```

## Running Tests

```bash
pytest tests/ -v
pytest tests/ -v --cov=engramma_memory  # with coverage
```

## Code Style

We use [Ruff](https://docs.astral.sh/ruff/) for linting and formatting:

```bash
ruff check engramma_memory/
ruff format engramma_memory/
```

## Type Checking

```bash
mypy engramma_memory/ --ignore-missing-imports
```

## Pull Request Process

1. Fork the repository and create a feature branch from `main`.
2. Write tests for any new functionality.
3. Ensure all tests pass and code is formatted.
4. Update documentation if your change affects the public API.
5. Submit a PR with a clear description of what changed and why.

## What We're Looking For

- Bug fixes with regression tests
- Performance improvements (with benchmarks)
- New framework integrations
- Documentation improvements
- Examples and tutorials

## Architecture Overview

```
engramma_memory/
├── core.py          # Public API (EngrammaMemory class)
├── engine.py        # Hybrid memory engine (Exact + Energy + Attention)
├── backends/        # Local and Cloud backend implementations
└── integrations/    # Framework adapters (LangChain, LlamaIndex, etc.)
```

The engine uses three pathways:
- **ExactMemory** — kNN with importance-based eviction
- **EnergyMemory** — Hopfield network with temperature-scaled softmax
- **MultiHeadAttentionMemory** — Compositional retrieval via split-query attention

## Reporting Issues

Please use [GitHub Issues](https://github.com/engramma-ai/engramma-memory/issues) and include:
- Python version
- Engramma version (`engramma_memory.__version__`)
- Minimal reproducible example
- Expected vs actual behavior

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
