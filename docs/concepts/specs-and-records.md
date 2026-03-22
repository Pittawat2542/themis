# Specs and Records

## Public Spec Model

| Type | Purpose |
| --- | --- |
| `ProjectSpec` | Storage root, seed, and execution policy |
| `BenchmarkSpec` | Top-level benchmark definition |
| `SliceSpec` | One dataset slice, prompt scope, dimensions, parses, scores, and explicit tool selection |
| `DatasetQuerySpec` | Subset, item IDs, metadata filters, and sampling hints |
| `PromptVariantSpec` | Prompt family plus bootstrap messages and optional follow-up turns |
| `ToolSpec` | Serializable tool definition selected onto agent-capable trials |
| `McpServerSpec` | Serializable MCP server definition selected onto MCP-capable trials |
| `ParseSpec` | Named parser pipeline |
| `ScoreSpec` | Named scoring overlay |

## Runtime Records

The persisted records are still trial-shaped internally, but the public analysis
surface speaks benchmark language:

- `BenchmarkResult.iter_trial_summaries()` returns rows with `benchmark_id`, `slice_id`, `prompt_variant_id`, and dimensions
- `BenchmarkResult.aggregate(...)` groups score rows by benchmark semantics
- `RecordTimelineView` exposes concrete inference, parse, score, and judge data for one candidate

## Quick-Start Factory

For exploratory work before you need multiple models or custom pipelines, use
`BenchmarkSpec.simple()` to build a minimal single-model benchmark in one call:

```python
benchmark = BenchmarkSpec.simple(
    benchmark_id="gsm8k-quick",
    model_id="gpt-4o",
    dataset_source=DatasetSource.HUGGINGFACE,
    dataset_id="openai/gsm8k",
    prompt="Solve step by step: {item.question}",
    metric="exact_match",
)
```

Graduate to a full `BenchmarkSpec` when you need multiple models, prompt
variants, parse pipelines, or inference parameter sweeps.

## Prompt Variable Namespaces

Prompt message templates support two distinct namespaces:

| Namespace | Source | Example |
| --- | --- | --- |
| `{item.<field>}` | Dataset item payload, resolved per trial | `{item.question}` |
| `{prompt.<key>}` | Static `variables` dict on the variant spec | `{prompt.tone}` |

`variables` are defined once per variant and never change across items.
`{item.*}` values change for every row in the dataset.

```python
PromptVariantSpec(
    id="formal-qa",
    messages=[PromptMessage(role=PromptRole.USER,
        content="({prompt.tone}) {item.question}")],
    variables={"tone": "concise"},
)
```

## Inspect Prompts Without Running Inference

Use `BenchmarkSpec.preview(item)` to render all prompt variants against a
sample item dict before committing to a full run:

```python
for entry in benchmark.preview({"question": "2 + 2"}):
    print(entry["prompt_variant_id"])
    for msg in entry["messages"]:
        print(f"  [{msg['role']}] {msg['content']}")
```

Pass `prompt_variant_ids=[...]` to filter to a specific subset.

## DatasetSliceSpec vs SliceSpec

These two types are related but distinct:

- **`SliceSpec`** is the authoring spec inside `BenchmarkSpec`.  It owns the
  full pipeline: dataset identity, query controls, prompt variant selection,
  tool selection, MCP server selection, parse pipelines, and score passes.

- **`DatasetSliceSpec`** is what `DatasetProvider.scan()` receives.  It is a
  narrower read-only view carrying only the fields needed for a provider to
  decide *which* items to return (dataset identity and semantic dimensions).

If you are writing a `BenchmarkSpec`, use `SliceSpec`.  If you are implementing
a `DatasetProvider`, the argument to `scan()` is a `DatasetSliceSpec`.

## Guiding Rule

Put semantics in the benchmark spec, not in payload conventions:

- use `dimensions={"source": "synthetic"}` instead of encoding source into IDs
- use `prompt_variant_ids=[...]` instead of manual prompt compatibility logic
- use `ParseSpec` instead of reparsing model text inside a metric
