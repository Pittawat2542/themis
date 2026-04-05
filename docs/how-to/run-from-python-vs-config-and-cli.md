---
title: Run from Python vs config and CLI
diataxis: how-to
audience: users deciding how to author and execute experiments
goal: Compare Python-first and config-driven execution styles.
---

# Run from Python vs config and CLI

Goal: choose the execution style that best matches how you manage experiments.

When to use this:

Use this guide when you already know Themis concepts but need to decide how to organize real runs in code, config, or shell workflows.

## Procedure

Use Python when you want:

- direct imports and type-checked objects
- custom components without module-path indirection
- the shortest path to experiments and local debugging

Use config and CLI when you want:

- reproducible checked-in experiment definitions
- shell-friendly automation
- submission flows such as worker-pool and batch

Config-backed execution details:

- `Experiment.from_config(...)` supports `YAML` (`.yaml` / `.yml`) and `TOML` (`.toml`)
- config component fields accept builtin ids or importable module paths such as `package.module:factory`
- config files carry strings, not live Python objects; object instances belong in Python authoring only
- relative storage and runtime paths resolve relative to the config file directory
- CLI or Python callers can pass dotlist `overrides` before compile/run time

Use the config-backed external execution example when you want one runnable path from config file to execution:

```python
--8<-- "examples/docs/external_execution.py"
```

## Variants

| Variant | Best when | Tradeoff | Related APIs / commands |
| --- | --- | --- | --- |
| Ad hoc scripts and notebooks | You want direct imports, live objects, and quick local debugging | Harder to standardize across repeated runs | Python authoring, `evaluate(...)`, `Experiment(...)` |
| Checked-in experiment specs and automation | You want reproducible definitions that work well with shell workflows and deferred execution | Component references must be config-loadable rather than live objects | `Experiment.from_config(...)`, `themis run`, `themis submit` |
| Mixed approach | You want checked-in configs for repeatable runs but still keep custom component logic in Python | Requires discipline about what lives in config vs code | Config files plus importable module paths |

## Expected result

You should know whether the next example or guide you follow should be code-first or config-first.

## Troubleshooting

- [Config schema](../reference/config-schema.md)
- [CLI reference](../reference/cli.md)
- [First external execution](../tutorials/first-external-execution.md)
