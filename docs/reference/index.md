---
title: Reference overview
diataxis: reference
audience: active Themis users and contributors
goal: Route readers to the right lookup page for symbols, commands, models, and catalog entries.
---

# Reference overview

Use reference docs when you already know what concept you need and want precise technical details.

## Reference map

| Name | Kind | Use when | Key constraints / notes |
| --- | --- | --- | --- |
| [Python API](python-api.md) | Reference page | You want the generated API reference entry point | Best when you already know the symbol family you need |
| [Root package API](root-package-api.md) | Reference page | You want a category-level view of top-level exports from `themis` | Pairs well with the generated API page |
| [Experiment lifecycle](experiment-lifecycle.md) | Reference page | You want `evaluate`, `Experiment`, `compile`, `run`, or `rejudge` details | Focused on authoring and execution flow |
| [Stores and inspection](stores-and-inspection.md) | Reference page | You want persistence, reporting, bundle, or inspection helpers | Best for resume, reporting, and export workflows |
| [Adapters](adapters.md) | Reference page | You want provider-backed generator adapter details | Covers OpenAI, vLLM, and LangGraph adapters |
| [CLI](cli.md) | Reference page | You want commands, inputs, and output shapes | Best for shell-driven workflows |
| [Config schema](config-schema.md) | Reference page | You want config field behavior and identity implications | Best for config-driven experiments |
| [Protocols](protocols.md) | Reference page | You want extension contract details | Best for custom components and instrumentation |
| [Data models](data-models.md) | Reference page | You want runtime or projection model details | Best for inspection and downstream tooling |
| [Builtins and adapters](builtins-and-adapters.md) | Reference page | You want builtin component ids and adapter-family guidance | Best for choosing shipped components |
| [Benchmark catalog](benchmark-catalog.md) | Reference page | You want named shipped benchmarks plus `themis.catalog.load(...)` and `themis.catalog.run(...)` | Best for catalog-backed evaluation workflows |
