---
title: Run benchmarks
diataxis: how-to
audience: users evaluating against catalog entries
goal: Show how to execute named benchmark entries and understand their constraints.
---

# Run benchmarks

Goal: execute a named benchmark from the catalog instead of wiring the dataset yourself.

When to use this:

Use this guide when a shipped benchmark entry already matches the task you want to run.

## Procedure

Run the shortest benchmark workflow:

```bash
themis quick-eval benchmark --name mmlu_pro
```

Or run the same named benchmark from Python through the catalog API using `themis.catalog.run(...)`.

Then inspect the benchmark catalog for prerequisites such as optional dataset dependencies or adapter-specific execution constraints.

## Variants

- quick local check: `quick-eval benchmark`
- Python catalog execution: `themis.catalog.run(...)`
- custom experiment around the same dataset: load the definition first and move to Python or config-driven experiments

## Expected result

You should get a completed run keyed by the named benchmark entry and know whether the benchmark requires extra setup.

## Troubleshooting

- [Benchmark catalog](../reference/benchmark-catalog.md)
- [Benchmark adapters](../explanation/benchmark-adapters.md)
