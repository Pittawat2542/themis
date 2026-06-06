---
title: Observe runs and instrumentation
diataxis: how-to
audience: users who need runtime visibility into stage execution
goal: Show how to attach subscribers and tracing providers without changing run identity.
---

# Observe runs and instrumentation

Goal: observe stage activity and emit tracing spans while keeping the logical run definition unchanged.

When to use this:

Use this guide when you need callback hooks, span emission, or lightweight runtime telemetry around generation and evaluation.

## Procedure

Use `LifecycleSubscriber` when you want callbacks around stage boundaries or raw `on_event(...)` notifications.

Use `TracingProvider` when you want span-oriented tracing around the run, generation, reduction, parsing, scoring, or judging stages.

Wire them into `Experiment.run(...)` or `Experiment.rejudge(...)` at execution time:

```python
--8<-- "examples/docs/observability.py"
```

Instrumentation is runtime-only. Swapping subscribers or tracing backends changes what you observe, not `run_id`.

## Variants

| Variant | Best when | Tradeoff | Related APIs / commands |
| --- | --- | --- | --- |
| Experiment flow | You want observability on reusable experiments, replay, or rejudge flows | Requires explicit experiment construction | `Experiment.run(...)`, `Experiment.rejudge(...)` |
| Config and CLI flow | You want observability in automation-oriented runs | Instrumentation is configured at the Python boundary that launches the run | `Experiment.from_config(...)`, `Experiment.run(...)` |
| No-op default | You do not need explicit instrumentation for this run | No trace or subscriber output to inspect later | Omit `subscribers` and `tracing_provider` |

## Expected result

You should get a completed run plus callback records and span names you can inspect or forward to your own tracing backend.

## Troubleshooting

- [Protocols reference](../reference/protocols.md)
- [Identity vs provenance](../explanation/identity-vs-provenance.md)
- [Failure, retry, and resume](../explanation/failure-retry-and-resume.md)
