# Themis

> Benchmark-first orchestration for reproducible LLM evaluation.

[![CI](https://github.com/Pittawat2542/themis/actions/workflows/ci.yml/badge.svg)](https://github.com/Pittawat2542/themis/actions/workflows/ci.yml)
[![Docs](https://github.com/Pittawat2542/themis/actions/workflows/docs.yml/badge.svg)](https://github.com/Pittawat2542/themis/actions/workflows/docs.yml)
[![PyPI version](https://img.shields.io/pypi/v/themis-eval.svg)](https://pypi.org/project/themis-eval/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

Themis now documents and supports one public authoring flow:

- `ProjectSpec` for shared storage and execution policy
- `BenchmarkSpec` for benchmark slices, prompt variants, parse pipelines, scores, and agent-style prompt flows
- `PluginRegistry` for engines, parsers, metrics, judges, and hooks
- `Orchestrator` for planning, execution, handoffs, and imports
- `BenchmarkResult` for aggregation, paired comparison, artifact bundles, and timelines
- `generate_config_report(...)` for reproducibility snapshots
- `themis-quickcheck` for fast SQLite inspection by slice and benchmark dimension

## Why Themis

- Benchmark-native authoring instead of experiment-matrix bookkeeping
- Query-aware dataset providers for subset, filter, and pushdown sampling
- Explicit prompt variants and parse pipelines instead of payload hacks
- Bootstrap prompt sequences, scripted follow-up turns, and first-class tool passing for agent-capable engines
- Projection-backed results with `slice_id`, `prompt_variant_id`, and semantic dimensions
- Local-first storage and deterministic reuse of completed work
- Seed-aware planning and per-candidate deterministic execution defaults

## Installation

```bash
uv add themis-eval
```

Add extras only when needed:

- `stats` for paired comparisons and richer report tooling
- `compression` for compressed artifact storage
- `extractors` for additional built-in parsing helpers
- `math` for math-equivalence scoring via `math-verify`
- `datasets` for dataset integrations
- `providers-openai`, `providers-litellm`, `providers-vllm` for provider SDKs
- `telemetry` for external observability callbacks
- `storage-postgres` for Postgres-backed storage

## Quick Start

Start with a zero-friction smoke evaluation:

```bash
themis quick-eval inline \
  --model demo-model \
  --provider demo \
  --input "2 + 2" \
  --expected "4" \
  --format json
```

That writes a SQLite store under:

```text
.cache/themis/quick-eval/inline-demo-model-exact-match/themis.sqlite3
```

Initialize a real project scaffold when you want editable code and project files:

```bash
themis init starter-eval
```

Or start from a built-in benchmark definition:

```bash
themis quick-eval benchmark \
  --benchmark mmlu_pro \
  --model demo-model \
  --provider demo \
  --preview \
  --format json
```

```bash
themis init starter-mmlu --benchmark mmlu_pro
```

Math benchmarks are available as built-ins too:

```bash
themis quick-eval benchmark \
  --benchmark aime_2026 \
  --model demo-model \
  --provider demo \
  --preview \
  --format json
```

Then run the shipped hello-world benchmark when you want the smallest code-first example:

```bash
uv run python examples/01_hello_world.py
```

Expected output:

```text
{'model_id': 'demo-model', 'slice_id': 'arithmetic', 'metric_id': 'exact_match', 'source': 'synthetic', 'prompt_variant_id': 'qa-default', 'mean': 1.0, 'count': 1}
```

That script shows the full benchmark-first loop:

- define a `DatasetProvider.scan(slice_spec, query)`
- register one engine and one metric
- build a `BenchmarkSpec`
- run `orchestrator.run_benchmark(...)`
- inspect the returned `BenchmarkResult`

The complete script is embedded in [docs/quick-start/index.md](docs/quick-start/index.md).

## Examples

Runnable examples live in [`examples/`](examples/):

- `01_hello_world.py`
- `02_project_file.py`
- `03_custom_extractor_metric.py`
- `04_compare_models.py`
- `05_resume_run.py`
- `06_hooks_and_timeline.py`
- `07_judge_metric.py`
- `08_external_stage_handoff.py`
- `09_experiment_evolution.py`
- `10_agent_eval.py`
- `11_quick_benchmark.py`
- `12_iter_and_estimate.py`
- `13_catalog_builtin_benchmark.py`

`10_agent_eval.py` is the canonical advanced example for bootstrap prompts,
follow-up turns, tool declaration and selection, and returned agent traces.

`13_catalog_builtin_benchmark.py` is the catalog-specific example for running a
shipped builtin benchmark through `themis.catalog.build_catalog_benchmark_project(...)`
with a local fixture dataset loader.

To discover all shipped builtin benchmark ids from Python, use:

```python
from themis.catalog import list_catalog_benchmarks

print(list_catalog_benchmarks())
```

The canonical benchmark list and Python usage notes live in
[docs/guides/builtin-benchmarks.md](docs/guides/builtin-benchmarks.md).

`examples/medical_reasoning_eval` is intentionally left untouched as a handoff
reference. It is not the recommended public authoring pattern after the
benchmark-first redesign.

## Documentation

- Docs site: https://pittawat2542.github.io/themis/
- Quick Start: [docs/quick-start/index.md](docs/quick-start/index.md)
- Tutorials: [docs/tutorials/index.md](docs/tutorials/index.md)
- Concepts: [docs/concepts/index.md](docs/concepts/index.md)
- Guides: [docs/guides/index.md](docs/guides/index.md)
- API Reference: [docs/api-reference/index.md](docs/api-reference/index.md)
- FAQ: [docs/faq/index.md](docs/faq/index.md)

## Development

```bash
uv sync --all-extras --dev
uv run pytest
uv run mkdocs build --strict
uv run ruff check
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Citation

If you use Themis in research, cite via [`CITATION.cff`](CITATION.cff).

## License

MIT. See [LICENSE](LICENSE).
