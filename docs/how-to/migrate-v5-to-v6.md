---
title: Migrate from v5 to v6
diataxis: how-to
audience: users upgrading existing Themis integrations
goal: Map v5 Python and CLI usage to the v6 public contract.
---

# Migrate from v5 to v6

Goal: update v5 application code and CLI consumers to the supported v6 public
boundary without migrating persisted runs.

When to use this guide: when upgrading a project that imports Themis Python
APIs, implements a store, uses presets or benchmark builders, or parses CLI
JSON.

Themis 6.0.0 deliberately changes public Python and machine-readable CLI
contracts. Existing stored snapshots, store schemas, and `run_id` values do not
need migration.

## Procedure

Apply each relevant migration below, then run your existing experiment and
compare its `run_id` with the v5 value.

### Replace private imports

Use the root package for authoring models and focused public modules for the
rest:

| v5 source | v6 source |
| --- | --- |
| experiment, dataset, and result models | `themis` |
| component protocols, contexts, and workflow models | `themis.components` |
| store protocol, base, backends, and factories | `themis.storage` |
| planning functions and result types | `themis.runtime` |
| bundle models and import/export helpers | `themis.artifacts` |
| inspection, reporting, and statistics | `themis.analysis` |

Application code should contain no `themis.core` imports.

### Update benchmark builders

Catalog and suite expansion now return `themis.Experiment`. Remove `storage=`
and `runtime=` from benchmark builders, then provide execution provenance at
the boundary:

```python
from themis import RunOptions
from themis.catalog import build_benchmark_experiment
from themis.storage import sqlite_store

experiment = build_benchmark_experiment("mmlu_pro")
store = sqlite_store("runs.sqlite3")
snapshot = experiment.compile(store=store, options=RunOptions(max_concurrency=8))
result = experiment.run(store=store, options=RunOptions(max_concurrency=8))
```

### Update preset use

`apply_preset` returns all resolved values instead of embedding runtime state in
the experiment:

```python
from themis.presets import apply_preset

application = apply_preset(experiment, "runtime/local-careful")
snapshot = application.experiment.compile(
    store=store,
    options=application.options,
)
```

Generation and evaluation presets change identity. Runtime presets change
provenance but not `run_id`.

### Update planning calls

Compile first, then pass the snapshot to `themis.runtime.estimate(snapshot)` or
`themis.runtime.resource_plan(snapshot)`. The internal planner is not public.

### Update custom stores

Keep `RunStore` in consumer type annotations. Backend authors should subclass
`RunStoreBase`, expose a `StorageConfig` through `storage_config`, and implement
only its documented snapshot, event, run-id, document, blob, and deletion
primitives.

### Update CLI JSON parsing

Read ordinary JSON results from `payload["data"]` after checking
`payload["schema_version"] == "1"`. Artifact exports remain raw bundle
documents and retain their bundle-level schema version.

## Variants

- If you only use Python authoring, migrate imports, compile/run options, and
  presets.
- If you only consume CLI output, update the JSON envelope parser; text and
  tabular output need no change.
- If you own a backend, migrate it to `RunStoreBase` independently of stored
  data because the persisted schema is unchanged.

## Expected result

Application code uses only `themis` and its focused public modules. Compilation,
Python execution, CLI execution, worker execution, and batch execution retain
the same logical `run_id` for an unchanged experiment.

## Troubleshooting

- If a launcher fails, confirm its `definition` resolves to an importable public
  `Experiment` instance or zero-argument factory.
- If a custom store has no provenance, expose a public `StorageConfig` as
  `storage_config`.
- If CLI keys appear missing, read them from the envelope's `data` member; do
  not unwrap raw artifact bundles.
