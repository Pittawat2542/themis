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

## Variants

- ad hoc scripts and notebooks: Python
- checked-in experiment specs and automation: config + CLI

## Expected result

You should know whether the next example or guide you follow should be code-first or config-first.

## Troubleshooting

- [Config schema](../reference/config-schema.md)
- [CLI reference](../reference/cli.md)
- [First external execution](../tutorials/first-external-execution.md)
