# Themis

> Lightweight, practical evaluation workflows for LLM experiments.

[![CI](https://github.com/Pittawat2542/themis/actions/workflows/ci.yml/badge.svg)](https://github.com/Pittawat2542/themis/actions/workflows/ci.yml)
[![Docs](https://github.com/Pittawat2542/themis/actions/workflows/docs.yml/badge.svg)](https://github.com/Pittawat2542/themis/actions/workflows/docs.yml)
[![PyPI version](https://img.shields.io/pypi/v/themis-eval.svg)](https://pypi.org/project/themis-eval/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

Themis gives you two clean entry points:

- `themis.evaluate(...)` for quick benchmark and custom dataset evaluation.
- `ExperimentSession().run(...)` for explicit, step-by-step evaluation pipelines.

It includes built-in benchmarks, auto-configured metric pipelines, caching and resumability, statistical comparison utilities, and a web server for run inspection.

## Why Themis

- **Fast start**: run your first evaluation in a few lines.
- **Structured control**: `ExperimentSession` API for reproducible workflows.
- **Built-in presets**: Curated benchmark definitions with prompts, metrics, and extractors included.
- **Extensible**: Register your own datasets, custom metrics, LLM providers, and benchmark presets.
- **Practical storage**: Local cache, resumable runs, and a robust storage backend.
- **Production-minded CI/CD**: Strict docs build, package validation, and release automation.

## Installation

```bash
# stable release
uv add themis-eval

# with optional extras
uv add "themis-eval[math,nlp,code,server]"
```

## Quick Start (No API key required)

Use our built-in `fake` model provider with the `demo` benchmark to ensure everything is installed correctly:

```python
from themis import evaluate

report = evaluate(
    "demo",
    model="fake:fake-math-llm", # Uses the built-in fake provider
    limit=10,
)

exact_match = report.evaluation_report.metrics["ExactMatch"]
print(f"ExactMatch: {exact_match.mean:.2%} (n={exact_match.count})")
```

## Quick Start (Real models)

Evaluating real models requires the corresponding provider's API key (e.g., `OPENAI_API_KEY`). By default, Themis uses [LiteLLM](https://github.com/BerriAI/litellm) for robust multi-provider routing.

```python
import os
from themis import evaluate

os.environ["OPENAI_API_KEY"] = "sk-..."

# Run the GSM8K math benchmark with GPT-4
report = evaluate(
    "gsm8k",
    model="openai/gpt-4o", # Model string parsed by LiteLLM
    limit=100,
)

# Extract and print the aggregated metric score
accuracy = report.evaluation_report.metrics["ExactMatch"].mean
print(f"Accuracy: {accuracy:.2%}")
```

## CLI Workflow

```bash
# Run two experiments
themis eval gsm8k --model gpt-4 --limit 100 --run-id run-a
themis eval gsm8k --model gpt-4 --temperature 0.7 --limit 100 --run-id run-b

# Compare them
themis compare run-a run-b

# Explore in browser
themis serve --storage .cache/experiments
```

Helpful commands:

```bash
themis list benchmarks
themis list runs --storage .cache/experiments
themis list metrics
```

## Session API (Advanced Workflow)

Use the `ExperimentSession` API when you want explicit programmatic control over the dataset, metric pipeline, execution parallelization, and storage backend.

```python
from themis.evaluation.metric_pipeline import MetricPipeline
from themis.presets import get_benchmark_preset
from themis.session import ExperimentSession
from themis.specs import ExecutionSpec, ExperimentSpec, StorageSpec

# 1. Load the benchmark preset
preset = get_benchmark_preset("gsm8k")
pipeline = MetricPipeline(extractor=preset.extractor, metrics=preset.metrics)

# 2. Define the explicit specification
spec = ExperimentSpec(
    dataset=preset.load_dataset(limit=100),
    prompt=preset.prompt_template.template,
    model="litellm:gpt-4o",
    sampling={"temperature": 0.0, "max_tokens": 512},
    pipeline=pipeline,
    run_id="gsm8k-gpt4o-eval",
)

# 3. Run the session with explicit execution and storage configurations
report = ExperimentSession().run(
    spec,
    execution=ExecutionSpec(workers=8),
    storage=StorageSpec(path=".cache/experiments", cache=True),
)
```

## Built-in Coverage

Themis ships with math, reasoning, science, and QA presets (for example: `gsm8k`, `math500`, `aime24`, `aime25`, `mmlu-pro`, `supergpqa`, `gpqa`, `commonsense_qa`, `coqa`, `demo`).

List everything from CLI:

```bash
themis list benchmarks
```

Supported metric families include:

- exact/verification metrics (for math/structured outputs)
- NLP metrics (`BLEU`, `ROUGE`, `BERTScore`, `METEOR`)
- code metrics (`PassAtK`, `CodeBLEU`, execution-based checks)

## Extending Themis

Top-level extension APIs are available directly from `themis`:

```python
import themis

# themis.register_metric(name, metric_cls)
# themis.register_dataset(name, factory)
# themis.register_provider(name, factory)
# themis.register_benchmark(preset)
```

See the extension guides:

- [Extending Themis](https://pittawat2542.github.io/themis/)
- [API Backends Reference](docs/api/backends.md)

## Documentation

- Docs site: https://pittawat2542.github.io/themis/
- Getting started: [docs/getting-started/quickstart.md](docs/getting-started/quickstart.md)
- Evaluation guide: [docs/guides/evaluation.md](docs/guides/evaluation.md)
- Comparison guide: [docs/guides/comparison.md](docs/guides/comparison.md)

- CI/CD and release process: [docs/guides/ci-cd.md](docs/guides/ci-cd.md)

## Examples

Runnable examples live in [`examples/`](examples/):

- `01_quickstart.py`
- `02_custom_dataset.py`
- `04_comparison.py`
- `05_api_server.py`
- `07_provider_ready.py`
- `08_resume_cache.py`
- `09_research_loop.py`

Run one:

```bash
uv run python examples/01_quickstart.py
```

## Development

```bash
# install all dev + feature dependencies
uv sync --all-extras --dev

# test
uv run pytest

# strict docs build
uv run mkdocs build --strict

# baseline syntax/runtime lint used in CI
uv run ruff check --select E9,F63,F7 themis tests
```

## Contributing

Contributions are welcome. Start with [CONTRIBUTING.md](CONTRIBUTING.md).

## Citation

If you use Themis in research, cite via [`CITATION.cff`](CITATION.cff).

## License

MIT. See [LICENSE](LICENSE).
