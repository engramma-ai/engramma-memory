# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2026-07-07

### Added

- Comprehensive real-world test suite (`real_tests/`) with 152 tests covering:
  - Core operations (store, query, retrieve, stats)
  - Native composition (2-way, 3-way, multi-pattern)
  - Forgetting (immediate delete, decay)
  - Engine pathways (exact kNN, energy-based, multi-head attention, router)
  - Capacity limits and importance-based eviction
  - Framework integrations (CrewAI, OpenAI Assistants, FastAPI, LangChain)
  - Async interface with cloud-only guards
  - Real-world scenarios (chatbot, RAG, multi-agent, preferences)
  - Performance benchmarks (latency, throughput, memory footprint)
  - Edge cases and numerical robustness

## [0.1.0] - 2026-07-06

### Added

- Initial public release of Engramma Memory SDK
- Hybrid memory engine with three pathways:
  - Exact kNN Memory with importance-based eviction
  - Energy-based Memory (Modern Hopfield network)
  - Multi-Head Attention Memory for native composition
- Confidence-based routing between pathways
- Local backend (in-process, zero dependencies beyond NumPy)
- Cloud backend (production-grade, unlimited storage)
- Framework integrations:
  - LangChain (BaseMemory adapter)
  - LlamaIndex (BaseRetriever adapter)
  - OpenAI Assistants (tool definitions + handler)
  - CrewAI (memory integration)
  - FastAPI (router middleware)
- Examples: quickstart, chatbot memory, RAG agent
- Jupyter notebooks for interactive exploration
- Full MkDocs documentation site
- MIT License
