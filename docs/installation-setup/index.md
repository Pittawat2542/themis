# Installation & Setup

## Base Install

```bash
uv add themis-eval
```

## Optional Extras

Install only what your workflow needs:

- `stats`: richer statistical comparisons and report tooling
- `compression`: compressed artifact storage
- `extractors`: additional built-in parsing helpers
- `datasets`: dataset integrations
- `providers-openai`, `providers-litellm`, `providers-vllm`: provider SDKs
- `telemetry`: external observability callbacks
- `storage-postgres`: Postgres storage backend
- `docs`: local docs build toolchain
- `dev`: tests, lint, and contributor tooling

Example:

```bash
uv add "themis-eval[stats,compression]"
```

## Smoke Import

```bash
uv run python -c "from themis import Orchestrator, ProjectSpec, BenchmarkSpec; print('ok')"
```

## Authoring Imports

Root-package authoring:

```python
from themis import BenchmarkSpec, Orchestrator, PluginRegistry, ProjectSpec
```

Supporting spec imports:

```python
from themis.specs import DatasetQuerySpec, DatasetSpec, GenerationSpec, SliceSpec
```

## Local Docs

```bash
uv sync --group docs
uv run mkdocs build --strict
```

## Local Verification

```bash
uv run pytest tests/benchmark -q
uv run python examples/01_hello_world.py
```
