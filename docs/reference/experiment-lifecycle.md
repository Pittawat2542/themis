---
title: Experiment lifecycle reference
diataxis: reference
audience: Python users authoring and executing experiments
goal: Document the primary v6 experiment lifecycle.
---

# Experiment lifecycle reference

## Lifecycle methods

| API | Purpose | Notes |
| --- | --- | --- |
| `Experiment.compile(store=..., options=...)` | Freeze logical identity plus execution provenance and return a snapshot | Store and runtime options are optional and do not change `run_id` |
| `Experiment.run(store=..., options=...)` | Execute synchronously | Store is explicit; runtime options do not change identity |
| `Experiment.run_async(...)` | Execute inside an event loop | Same contract as `run()` |
| `Experiment.replay(store=..., from_stage=Stage...)` | Recompute downstream stages from evidence | Accepts `REDUCE`, `PARSE`, `SCORE`, or `JUDGE`; requires persisted upstream artifacts |
| `Experiment.rerun(store=..., from_stage=Stage...)` | Re-execute selected cases or failures | Accepts all persisted boundaries except `SELECT`; records lineage |

Launcher configuration is handled by the CLI. It imports a reviewed Python
`Experiment`; `Experiment` itself does not load YAML or TOML.
