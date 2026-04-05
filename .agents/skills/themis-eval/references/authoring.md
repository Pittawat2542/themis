# Choose The Authoring Surface

Use this file when changing how a project defines or launches runs.

## API Surface Chooser

Use `evaluate(...)` when the project wants the shortest path from dataset plus inline arguments to a completed run.

Use `Experiment(...)` when the project needs one or more of these:

- `compile()` to freeze a `RunSnapshot`
- `run()` or `run_async()` on an explicit experiment object
- `replay()` or `rejudge()` for downstream reruns
- config loading with `Experiment.from_config(...)`
- reusable experiment definitions that persist across commands or tests

Use `themis.catalog` when the task is about shipped benchmark recipes or
reusable builtin catalog components.

Use custom extension protocols when builtin generators, reducers, parsers, or metrics are not sufficient.

## Python Versus Config And CLI

Prefer Python authoring when the repo wants:

- direct imports and typed objects
- inline custom logic without module-path indirection
- small scripts, tests, notebooks, and local debugging

Prefer config plus CLI when the repo wants:

- checked-in experiment definitions
- shell-friendly automation
- worker or batch submission flows
- environment-specific overrides without changing Python code

Prefer `themis.catalog` when the repo wants:

- shipped benchmark names instead of hand-authored datasets
- reusable shipped parsers, metrics, reducers, selectors, or judge workflows
- benchmark-materialization convenience through `themis.catalog.run(...)` or
  `quick-eval benchmark`

### Config Rules

- `Experiment.from_config(...)` supports YAML and TOML.
- Config component fields accept builtin ids like `builtin/exact_match` or importable module paths like `package.module:factory`.
- Config files carry strings and JSON-like values, not live Python objects.
- Relative paths in storage and runtime settings resolve relative to the config file directory.
- Dotlist overrides can be passed before compile or run time.

## Minimal Canonical Shapes

Smallest inline run:

```python
from themis import evaluate
from themis.core.models import Case, Dataset

result = evaluate(
    model="builtin/demo_generator",
    data=[
        Dataset(
            dataset_id="sample",
            cases=[
                Case(
                    case_id="case-1",
                    input={"question": "2+2"},
                    expected_output={"answer": "4"},
                )
            ],
        )
    ],
    metric="builtin/exact_match",
    parser="builtin/json_identity",
)
```

Explicit experiment:

```python
from themis import Experiment, RuntimeConfig
from themis.core.config import EvaluationConfig, GenerationConfig, StorageConfig
from themis.core.models import Case, Dataset

experiment = Experiment(
    generation=GenerationConfig(generator="builtin/demo_generator"),
    evaluation=EvaluationConfig(
        metrics=["builtin/exact_match"],
        parsers=["builtin/json_identity"],
    ),
    storage=StorageConfig(store="memory"),
    datasets=[
        Dataset(
            dataset_id="sample",
            cases=[Case(case_id="case-1", input={"question": "2+2"})],
        )
    ],
)
result = experiment.run(runtime=RuntimeConfig(max_concurrent_tasks=4))
```

Catalog-backed benchmark usage:

```python
from themis.catalog import load, run

benchmark = load("mmlu_pro")
dataset = benchmark.materialize_dataset()
result = run("mmlu_pro")
```

Config-backed experiment skeleton:

```yaml
generation:
  generator: builtin/demo_generator
evaluation:
  metrics:
    - builtin/exact_match
  parsers:
    - builtin/json_identity
storage:
  store: sqlite
  parameters:
    path: runs/themis.sqlite3
datasets:
  - dataset_id: sample
    cases:
      - case_id: case-1
        input:
          question: 2+2
        expected_output:
          answer: "4"
```

## Public Surface To Prefer

Prefer root-package exports from `themis` unless the project already depends on deeper internals:

- authoring: `Experiment`, `evaluate`
- prompts: `PromptSpec`
- persistence: `InMemoryRunStore`, `SqliteRunStore`, `RunStore`, `sqlite_store`
- run inspection: `get_run_snapshot`, `get_execution_state`, `get_evaluation_execution`
- reporting: `Reporter`, `StatsEngine`, `snapshot_report`, `quickcheck`
- bundles: `export_*_bundle(...)` and `import_*_bundle(...)`

Prefer catalog exports from `themis.catalog` when the task is about shipped
components or benchmark recipes:

- `load`
- `run`
- `list_component_ids`
- `builtin_component_refs`

Common data models that appear in user code:

- `Dataset`
- `Case`
- `GenerationResult`
- `ParsedOutput`
- `Score`
- `TraceStep`
