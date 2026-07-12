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
- reviewed importable modules as the canonical experiment definition
- the shortest path to experiments and local debugging

Use config and CLI when you want:

- shell-friendly transport or overrides around importable experiment code
- shell-friendly automation
- submission flows such as worker-pool and batch

Config-backed execution details:

- launchers support `YAML` (`.yaml` / `.yml`) and `TOML` (`.toml`)
- `definition: package.module:experiment` imports an `Experiment` object or factory
- launchers carry operational settings, not component or dataset definitions
- config files are automation surfaces, not the canonical source of serious experiment meaning
- relative storage and runtime paths resolve relative to the config file directory
- CLI or Python callers can pass dotlist `overrides` before compile/run time

Use the external execution example when you want one runnable path from a reviewed Python module through a config automation surface to worker execution:

```python
--8<-- "examples/docs/external_execution.py"
```

## Variants

| Variant | Best when | Tradeoff | Related APIs / commands |
| --- | --- | --- | --- |
| Reviewed Python module | You want the experiment meaning to live in versioned, importable code | More explicit structure than notebook-only exploration | Python authoring with `Experiment(...)`, `Experiment.run()`, and `Experiment.run_async()` |
| Config and CLI automation | You want shell workflows, deferred execution, or environment-specific overrides around reviewed Python code | The Python definition must be importable | launcher config, `themis run`, `themis submit` |
| Mixed approach | You want config transport for repeatable runs but still keep experiment meaning and custom components in Python | Requires discipline about what lives in config vs code | Config files plus importable module paths |

## Expected result

You should know whether the next example or guide you follow should be code-first or config-first.

## Troubleshooting

- [Config schema](../reference/config-schema.md)
- [CLI reference](../reference/cli.md)
- [First external execution](../tutorials/first-external-execution.md)
