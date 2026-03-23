# Public Surface

Themis now documents one benchmark-first public API.

## Main Objects

| Object | Role |
| --- | --- |
| `ProjectSpec` | Shared storage, seed, and execution policy |
| `BenchmarkSpec` | Models, slices, prompt variants, parse pipelines, and scores |
| `SliceSpec` | One benchmark slice with dataset config, dimensions, allowed prompts, and explicit tool selection |
| `DatasetQuerySpec` | Subset, filters, item pinning, and sampling hints |
| `PromptVariantSpec` | A reusable prompt family with bootstrap messages and optional follow-up turns |
| `ToolSpec` | A serializable tool definition passed to selected agent-capable trials |
| `McpServerSpec` | A serializable MCP server definition passed to selected MCP-capable trials |
| `ParseSpec` | A named parser pipeline |
| `ScoreSpec` | A named scoring overlay, optionally tied to a parse pipeline |
| `BenchmarkDefinition` | A reusable packaged benchmark definition for starter workflows and the built-in catalog |
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
- benchmark packaging or starter assembly: `BenchmarkDefinition` and `build_benchmark_definition_project(...)`
- agent and tool authoring: bootstrap prompt variants plus `ToolSpec`, `McpServerSpec`, `SliceSpec.tool_ids`, and `SliceSpec.mcp_server_ids`
- provider-specific execution: `InferenceEngine`
- answer parsing: `ParseSpec` + extractor chain
- scoring: `ScoreSpec` + metrics
- read-side analysis: `BenchmarkResult`

Use the CLI starter paths when they fit the job:

- `themis quick-eval inline ...` for a one-prompt smoke test
- `themis quick-eval benchmark ...` for a built-in benchmark preview or quick run
- `themis init ...` for an editable scaffold backed by project files

For the full agent and tool authoring flow, see
[Author Agent Evaluations and Tools](../guides/agent-evals-and-tools.md).

## Public Imports

Use the root package for the main entry points:

```python
from themis import (
    BenchmarkDefinition,
    BenchmarkResult,
    BenchmarkSpec,
    DatasetQuerySpec,
    EngineCapabilities,
    McpServerSpec,
    ModelSpec,
    Orchestrator,
    ParseSpec,
    PluginRegistry,
    ProjectSpec,
    PromptMessage,
    PromptTurnSpec,
    PromptVariantSpec,
    ScoreSpec,
    SliceSpec,
    StorageConfig,
    ToolSpec,
    build_benchmark_definition_project,
    generate_config_report,
)
```

Use `themis.specs` for supporting spec models that are still public but not
curated into the root package:

```python
from themis.specs import DatasetSpec, GenerationSpec, JudgeInferenceSpec
```
