# FAQ

## Why is the public API benchmark-first now?

Because serious eval authors need first-class slices, prompt variants, parse
pipelines, semantic dimensions, and benchmark-native reporting.

## Why does `BenchmarkSpec` compile to something private?

Planning and execution still run on a lower-level IR, but that layer is an
implementation detail. The public contract is the benchmark surface.

## What replaced the old dataset loader contract?

Use `DatasetProvider.scan(slice_spec, query)`.

## What should I do with `examples/medical_reasoning_eval`?

Treat it as a handoff and acceptance reference. It was intentionally not
rewritten during the benchmark-first overhaul.

## How do I inspect results without importing Python?

Use `themis-quickcheck` against the SQLite database.

## How do I group results by benchmark semantics?

Use `BenchmarkResult.aggregate(...)` and include `slice_id`,
`prompt_variant_id`, or dimension keys in `group_by`.
