---
title: Python API reference
diataxis: reference
audience: Python users of Themis
goal: Describe the supported v6 Python contract.
---

# Python API reference

The v6 root package is deliberately small. These are its complete stable exports:

## Root exports

| Name | Purpose |
| --- | --- |
| `Case`, `Dataset`, `DatasetSource` | Define evaluation inputs and their provenance |
| `Generation` | Configure a generator, sampling, selection, and reduction |
| `Evaluation` | Configure parsing, metrics, and judge models |
| `Experiment` | Define, compile, run, replay, or rerun an evaluation |
| `RunOptions` | Validate runtime concurrency, timeout, retry, and reuse controls |
| `RunSnapshot` | Inspect the immutable compiled boundary passed to runtime planning and execution |
| `RunResult`, `MetricResult`, `MetricInterpretation` | Consume outcomes and declare metric direction, range, and correctness semantics |
| `evaluate` | Run a deliberately small one-off evaluation |
| `__version__` | Read the installed package version |

## Focused modules

Use focused modules for the rest:

- `themis.adapters` for provider integrations;
- `themis.storage` for memory, SQLite, JSONL, PostgreSQL, and MongoDB stores;
- `themis.metrics` for metric constructors;
- `themis.analysis` for inspection, reporting, and statistics;
- `themis.artifacts` for portable generation and evaluation bundles;
- `themis.catalog` for reviewed benchmark and suite definitions;
- `themis.components` for custom component protocols and their input/output models;
- `themis.presets` for immutable experiment and runtime preset application;
- `themis.runtime` for snapshot estimates and resource plans.

`themis.core` is private implementation. It has no compatibility contract.

## Generated modules

::: themis

::: themis.api

::: themis.storage

::: themis.analysis

::: themis.artifacts

::: themis.adapters

::: themis.catalog

::: themis.components

::: themis.presets

::: themis.metrics

::: themis.runtime
