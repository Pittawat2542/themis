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

Wire them into `Experiment.run(...)`, `Experiment.rejudge(...)`, or `evaluate(...)` at execution time:

```python
--8<-- "examples/docs/observability.py"
```

Instrumentation is runtime-only. Swapping subscribers or tracing backends changes what you observe, not `run_id`.

## Variants

- layer-1 convenience flow: pass `subscribers=` or `tracing_provider=` to `evaluate(...)`
- experiment flow: pass them to `Experiment.run(...)` or `Experiment.rejudge(...)`
- no-op default: omit both and let the built-in `NoOpTracingProvider` handle execution silently

## Expected result

You should get a completed run plus callback records and span names you can inspect or forward to your own tracing backend.

## Troubleshooting

- [Protocols reference](../reference/protocols.md)
- [Identity vs provenance](../explanation/identity-vs-provenance.md)
- [Failure, retry, and resume](../explanation/failure-retry-and-resume.md)
