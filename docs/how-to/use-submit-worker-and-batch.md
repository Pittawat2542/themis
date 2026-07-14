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

Use this comparison when you need to decide whether execution is queue-driven or request-manifest-driven.

```mermaid
sequenceDiagram
    participant U as User
    participant Q as Worker-pool queue
    participant W as Worker
    participant B as Batch request
    participant R as Batch runner
    U->>Q: themis submit --mode worker-pool
    W->>Q: themis worker run --definition-root ...
    Q-->>W: queued manifest
    U->>B: themis submit --mode batch
    R->>B: themis batch run --request ... --definition-root ...
```

Both flows hand execution off through a manifest, but they differ in whether work is pulled from a queue or invoked by an explicit request file.

Submission manifests persist the compiled snapshot, definition digest, execution target, timestamps, and resource plan. They do not replace the compiled `RunSnapshot`, and workers validate the definition root, digest, and stored snapshot before execution.

Worker-pool flow:

```bash
themis submit --config experiment.yaml --mode worker-pool
themis worker run --queue-root runs/queue --definition-root .
```

Batch flow:

```bash
themis submit --config experiment.yaml --mode batch
themis batch run \
  --request runs/batch/requests/<run-id>.json \
  --definition-root .
```

## Variants

| Variant | Best when | Tradeoff | Related APIs / commands |
| --- | --- | --- | --- |
| Single-host queued work | Workers should pull manifests from a shared queue root | Requires a worker process to keep polling | `themis submit --mode worker-pool`, `themis worker run --definition-root ...` |
| Request and completed manifest flow | Each run should execute from an explicit request file | Less queue-like than worker-pool mode | `themis submit --mode batch`, `themis batch run --request ... --definition-root ...` |
| Signed worker queue | The queue crosses a trust boundary | Requires shared secret management | `themis worker run --signing-key-env ... --require-signature` |

## Expected result

The run should execute from a manifest without direct in-process orchestration at the point of execution.

## Troubleshooting

- [First external execution](../tutorials/first-external-execution.md)
- [CLI reference](../reference/cli.md)
