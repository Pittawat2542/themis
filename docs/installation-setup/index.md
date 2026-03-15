# Installation & Setup

## Requirements

- Python `3.12+`
- `uv` recommended for package and environment management

## Install Themis

=== "Base Package"

    ```bash
    uv add themis-eval
    ```

=== "With Common Extras"

    ```bash
    uv add "themis-eval[stats,compression]"
    ```

=== "Everything"

    ```bash
    uv add "themis-eval[all]"
    ```

## Optional Extras

| Extra | Adds |
| --- | --- |
| `dev` | Contributor tooling (`pytest`, `ruff`, `mypy`, coverage helpers) |
| `compression` | `zstandard` support for compressed artifact blobs |
| `datasets` | `datasets` library support for custom Hugging Face-backed loaders |
| `extractors` | `jsonschema` support for the built-in `json_schema` extractor |
| `providers-openai` | OpenAI SDK for custom inference engines |
| `providers-litellm` | LiteLLM + retry helpers for custom inference engines |
| `providers-vllm` | vLLM runtime dependency for custom inference engines |
| `stats` | `numpy`, `scipy`, and `pandas` for comparisons and reports |
| `telemetry` | Langfuse callback support plus observability-link dependencies (`langfuse`, `wandb`) |
| `storage-postgres` | `psycopg` driver for the `postgres_blob` storage backend |
| `docs` | MkDocs + mkdocstrings toolchain |
| `all` | Every runtime extra above except `dev` |

## Choose Extras by Workflow

### `dev`

Install this when you want the package's lint, test, and type-check tools
without cloning the repository:

```bash
uv add "themis-eval[dev]"
```

For a source checkout, prefer the repository workflow instead:

```bash
uv sync --all-extras --dev
```

### `compression`

Install this when your `StorageConfig` uses `compression="zstd"` or you expect
large artifact blobs such as judge audits and structured payload snapshots.

```bash
uv add "themis-eval[compression]"
```

See [Storage and Resume](../concepts/storage-and-resume.md) for how compressed
artifacts are laid out on disk.

### `datasets`

Install this when your custom dataset loader imports the Hugging Face
`datasets` package.

```bash
uv add "themis-eval[datasets]"
```

Themis does not auto-register a built-in Hugging Face loader. Provide the
`dataset_loader` object and its `load_task_items(task)` method yourself.

See [Write a Dataset Loader](../guides/dataset-loaders.md) for the loader
contract and a Hugging Face example.

### `extractors`

Install this when you want to use the built-in `json_schema` extractor.

```bash
uv add "themis-eval[extractors]"
```

The other built-ins (`regex`, `first_number`, `choice_letter`) do not require
an extra.

See [Add a Minimal Plugin Set](../guides/plugins.md) and
[Extractors API](../api-reference/extractors.md).

### `providers-openai`

Install this when your custom inference engine imports the `openai` SDK.

```bash
uv add "themis-eval[providers-openai]"
```

Themis does not ship a built-in OpenAI engine in this package. The extra only
installs the dependency; register an `InferenceEngine` under your chosen
provider name, such as `openai`.

### `providers-litellm`

Install this when your custom inference engine or router imports LiteLLM.

```bash
uv add "themis-eval[providers-litellm]"
```

This extra is intended for custom plugin code that wants LiteLLM request
routing plus retry helpers from `tenacity`.

### `providers-vllm`

Install this when your custom inference engine imports the Python `vllm`
runtime or talks to a colocated vLLM worker.

```bash
uv add "themis-eval[providers-vllm]"
```

As with the other provider extras, provide the engine implementation and
register it with `PluginRegistry`.

### `stats`

Install this for paired comparisons, report building, and other statistical
analysis paths.

```bash
uv add "themis-eval[stats]"
```

See [Compare and Export Results](../guides/compare-and-export.md),
[Analyze Results](../tutorials/analyze-results.md), and
[Reporting & Stats API](../api-reference/reporting-and-stats.md).

### `telemetry`

Install this when you want the optional `LangfuseCallback` integration and
stored observability links.

```bash
uv add "themis-eval[telemetry]"
```

See [Attach Telemetry & Observability](../guides/telemetry-and-observability.md)
and [Telemetry API](../api-reference/telemetry.md).

### `storage-postgres`

Install this when your project uses `backend="postgres_blob"`.

```bash
uv add "themis-eval[storage-postgres]"
```

### `docs`

Install this when you want to build or edit the documentation site locally.

```bash
uv add "themis-eval[docs]"
```

Then verify the site build:

```bash
uv run mkdocs build --strict
```

### `all`

Install this when you want every runtime extra in one command.

```bash
uv add "themis-eval[all]"
```

`all` includes `compression`, `datasets`, `docs`, `extractors`,
`providers-openai`, `providers-litellm`, `providers-vllm`, `stats`,
`telemetry`, and `storage-postgres`. It does not include `dev`.

## Install From Source

```bash
git clone https://github.com/Pittawat2542/themis.git
cd themis
uv sync --all-extras --dev
```

## Verify the Environment

Run a simple import check:

```bash
uv run python -c "from themis import Orchestrator, ProjectSpec, ExperimentSpec; print('ok')"
```

If you installed the `stats` extra, also verify the operator CLI:

```bash
uv run themis-quickcheck --help
```

If you installed the `telemetry` extra, verify the telemetry primitives:

```bash
uv run python -c "from themis.telemetry import TelemetryBus, LangfuseCallback; print('ok')"
```

## Pick a Storage Configuration Early

Every `ProjectSpec` needs either a SQLite storage root, or a Postgres URL plus
blob root. The backend-neutral type is `StorageConfig`; `StorageSpec` remains
available as the SQLite convenience alias.

```python
from themis import (
    ExecutionPolicySpec,
    ProjectSpec,
    SqliteBlobStorageSpec,
)

project = ProjectSpec(
    project_name="docs-demo",
    researcher_id="team-docs",
    global_seed=7,
    storage=SqliteBlobStorageSpec(root_dir=".cache/themis/docs-demo"),
    execution_policy=ExecutionPolicySpec(),
)
```

!!! tip
    Treat the storage root as part of the experiment contract. Resume and
    quickcheck behavior depend on it.
