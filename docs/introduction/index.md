# Public Surface

Themis now documents one benchmark-first public API.

## Main Objects

| Object | Role |
| --- | --- |
| `ProjectSpec` | Shared storage, seed, and execution policy |
| `BenchmarkSpec` | Models, slices, prompt variants, parse pipelines, and scores |
| `SliceSpec` | One benchmark slice with dataset config, dimensions, and allowed prompts |
| `DatasetQuerySpec` | Subset, filters, item pinning, and sampling hints |
| `PromptVariantSpec` | A reusable prompt family and message template |
| `ParseSpec` | A named parser pipeline |
| `ScoreSpec` | A named scoring overlay, optionally tied to a parse pipeline |
| `PluginRegistry` | Runtime lookup for engines, parsers, metrics, judges, and hooks |
| `Orchestrator` | Planning, execution, export, import, resume, and progress |
| `BenchmarkResult` | Aggregation, paired comparisons, timelines, and artifact bundles |

## Mental Model

`BenchmarkSpec` is the public authoring model. Internally, Themis compiles it to
a private execution IR before planning trials. That lower layer is an
implementation detail, not a second public API.

Dataset access uses the benchmark-first provider contract
`DatasetProvider.scan(slice_spec, query)`.

Use this split when deciding where logic belongs:

- project-wide runtime policy: `ProjectSpec`
- benchmark semantics: `BenchmarkSpec`
- provider-specific execution: `InferenceEngine`
- answer parsing: `ParseSpec` + extractor chain
- scoring: `ScoreSpec` + metrics
- read-side analysis: `BenchmarkResult`

## Public Imports

Use the root package for the main entry points:

```python
from themis import (
    BenchmarkResult,
    BenchmarkSpec,
    DatasetQuerySpec,
    Orchestrator,
    ParseSpec,
    PluginRegistry,
    ProjectSpec,
    PromptMessage,
    PromptVariantSpec,
    ScoreSpec,
    SliceSpec,
    generate_config_report,
)
```

Use `themis.specs` for supporting spec models that are still public but not
curated into the root package:

```python
from themis.specs import DatasetSpec, GenerationSpec, JudgeInferenceSpec
```
