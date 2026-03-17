# Specs and Records

## Public Spec Model

| Type | Purpose |
| --- | --- |
| `ProjectSpec` | Storage root, seed, and execution policy |
| `BenchmarkSpec` | Top-level benchmark definition |
| `SliceSpec` | One dataset slice, prompt scope, dimensions, parses, and scores |
| `DatasetQuerySpec` | Subset, item IDs, metadata filters, and sampling hints |
| `PromptVariantSpec` | Prompt family plus message template |
| `ParseSpec` | Named parser pipeline |
| `ScoreSpec` | Named scoring overlay |

## Runtime Records

The persisted records are still trial-shaped internally, but the public analysis
surface speaks benchmark language:

- `BenchmarkResult.iter_trial_summaries()` returns rows with `benchmark_id`, `slice_id`, `prompt_variant_id`, and dimensions
- `BenchmarkResult.aggregate(...)` groups score rows by benchmark semantics
- `RecordTimelineView` exposes concrete inference, parse, score, and judge data for one candidate

## Guiding Rule

Put semantics in the benchmark spec, not in payload conventions:

- use `dimensions={"source": "synthetic"}` instead of encoding source into IDs
- use `prompt_variant_ids=[...]` instead of manual prompt compatibility logic
- use `ParseSpec` instead of reparsing model text inside a metric
