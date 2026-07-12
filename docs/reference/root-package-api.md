---
title: Root package API
diataxis: reference
audience: Python users of the public package surface
goal: Enumerate the intentionally small root-package contract.
---

# Root package API

The root package contains only the nouns needed for the common authoring path:

`Case`, `Dataset`, `DatasetSource`, `Generation`, `Evaluation`, `Experiment`,
`RunOptions`, `RunResult`, `MetricResult`, `evaluate`, and `__version__`.

Storage, adapters, analysis, metrics, and catalog functionality live in their
named modules. Importing an internal object from `themis.core` is possible for
Themis development, but it is not a supported user contract.
