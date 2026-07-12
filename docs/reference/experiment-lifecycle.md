---
title: Experiment lifecycle reference
diataxis: reference
audience: Python users authoring and executing experiments
goal: Document the primary v5 experiment lifecycle.
---

# Experiment lifecycle reference

## Lifecycle methods

| API | Purpose | Notes |
| --- | --- | --- |
| `Experiment.compile()` | Freeze logical identity and return a snapshot | Does not execute work |
| `Experiment.run(store=..., options=...)` | Execute synchronously | Store is explicit; runtime options do not change identity |
| `Experiment.run_async(...)` | Execute inside an event loop | Same contract as `run()` |
| `Experiment.replay(store=..., from_stage=...)` | Recompute downstream stages from evidence | Requires persisted upstream artifacts |
| `Experiment.rerun(...)` | Re-execute selected cases or failures | Records lineage |

Launcher configuration is handled by the CLI. It imports a reviewed Python
`Experiment`; `Experiment` itself does not load YAML or TOML.
