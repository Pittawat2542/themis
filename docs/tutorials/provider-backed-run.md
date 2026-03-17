# Provider-backed Run

This tutorial shows the minimum provider-facing pattern without relying on the
legacy dataset-loader API.

## Core Shape

```python
from themis import (
    BenchmarkSpec,
    DatasetQuerySpec,
    InferenceGridSpec,
    InferenceParamsSpec,
    ModelSpec,
    Orchestrator,
    PluginRegistry,
    ProjectSpec,
    PromptMessage,
    PromptVariantSpec,
    ScoreSpec,
    SliceSpec,
)
from themis.specs import DatasetSpec, GenerationSpec


class RemoteDatasetProvider:
    def scan(self, slice_spec, query):
        # push the subset or filter into the remote source when possible
        del slice_spec
        return [{"item_id": "item-1", "question": "2 + 2", "answer": "4"}]


class ProviderEngine:
    def infer(self, trial, context, runtime):
        # Messages are already rendered by orchestration before the engine runs.
        prompt = trial.prompt.messages
        del prompt, context, runtime
        ...
```

## Why It Matters

- `DatasetQuerySpec` keeps subset and filter intent outside dataset payloads
- engines receive already-rendered benchmark prompts plus preserved prompt metadata
- slice dimensions stay queryable later through `BenchmarkResult.aggregate(...)`

## Full Next Step

Pair this pattern with [Build a Dataset Provider](../guides/dataset-loaders.md)
and [Build a Provider Engine](../guides/provider-engines.md).
