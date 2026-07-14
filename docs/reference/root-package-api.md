---
title: Root package API
diataxis: reference
audience: Python users of the public package surface
goal: Enumerate the intentionally small root-package contract.
---

# Root package API

The root package contains only the nouns needed for the common authoring path:

`Case`, `Dataset`, `DatasetSource`, `Generation`, `Evaluation`, `Experiment`,
`RunOptions`, `RunSnapshot`, `RunResult`, `MetricResult`, `MetricInterpretation`,
`evaluate`, and `__version__`.

Storage, adapters, analysis, artifacts, catalog, components, metrics, presets,
and runtime planning live in their named modules. The `themis.core` package is
private contributor implementation and must not be used by application code.
