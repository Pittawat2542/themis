---
title: Use submit, worker, and batch
diataxis: how-to
audience: users adopting deferred execution flows
goal: Show how to submit experiments and execute them through worker-pool or batch flows.
---

# Use submit, worker, and batch

Goal: hand off execution through manifest-backed deferred workflows.

When to use this:

Use this guide when in-process `run()` is not the right operational shape and you need queued or request-based execution.

## Procedure

Worker-pool flow:

```bash
themis submit --config experiment.yaml --mode worker-pool
themis worker run --queue-root runs/queue
```

Batch flow:

```bash
themis submit --config experiment.yaml --mode batch
themis batch run --request runs/batch/requests/<run-id>.json
```

## Variants

- single-host queued work: worker-pool
- request/completed manifest flow: batch

## Expected result

The run should execute from a manifest without direct in-process orchestration at the point of execution.

## Troubleshooting

- [First external execution](../tutorials/first-external-execution.md)
- [CLI reference](../reference/cli.md)
