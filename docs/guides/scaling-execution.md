# Scale Execution

Use this guide when the bottleneck is throughput rather than correctness:
multiple models, many benchmark items, multiple GPUs, or API-backed engines with
rate limits.

## Start with the Scheduler Knob

Themis schedules work through one bounded global queue per run. The main local
throughput control is `ExecutionPolicySpec.max_in_flight_work_items`.

```python
project = ProjectSpec(
    ...,
    execution_policy=ExecutionPolicySpec(
        max_in_flight_work_items=16,
        max_retries=3,
        retry_backoff_factor=1.5,
    ),
)
```

Use this when you want more local concurrency across planned generation,
transform, or evaluation work items.

## Local Concurrency vs Engine Concurrency

Themis can schedule API-based models and local GPU models through the same
work-item scheduler. The important difference is where the real bottleneck lives:

- local GPU engines are usually bound by device memory and worker placement
- API engines are usually bound by rate limits, quotas, and provider latency

At the orchestration layer, both are just planned work items. Actual throughput,
batching, and rate limiting still belong to the engine implementation.

## Retry Behavior

Themis exposes persisted retry controls at the project level:

```python
policy = ExecutionPolicySpec(
    max_retries=4,
    retry_backoff_factor=2.0,
    retryable_error_codes=["provider_timeout", "rate_limited"],
)
```

Use this for transient failures that your engine maps onto stable error codes.
Provider SDK retries, quota backoff, and timeout classification still live in
your engine or SDK adapter.

## Worker-Pool Backend

Use the worker-pool backend when one shared store coordinates work across
machines or device pools:

```toml
[execution_backend]
kind = "worker_pool"
lease_ttl_seconds = 180
poll_interval_seconds = 5
worker_tags = ["gpu:a100", "region:us-east-1"]
```

What Themis does:

- persists a run manifest plus normalized stage work items
- stores lease metadata and pending/completed status in the shared store
- keeps the same deterministic trial identities as local runs

What you still provide:

- the worker processes or job system that actually claim and execute those work
  items
- routing logic that matches worker tags to the right hardware or provider pool

## Multi-GPU and Multi-Machine Use

Themis does not hardcode a GPU topology. The typical pattern is:

1. use a shared storage backend
2. plan or submit the experiment once
3. run external workers on multiple machines
4. route work with `worker_tags`
5. import or materialize results back into the same store

That makes local GPU, multi-GPU, and multi-machine runs share the same storage
and resume behavior.

## Batch Backend

Use the batch backend when work is asynchronous and completed by an externally
polled provider or job system:

```toml
[execution_backend]
kind = "batch"
provider = "openai"
poll_interval_seconds = 30
max_batch_items = 250
```

What this means today:

- `submit()` persists a pending run handle instead of executing locally
- run manifests and stage work items remain visible in storage
- external job IDs can be attached to work items

What it does not mean today:

- there is no built-in OpenAI Batch submit/poll/import adapter in Themis
- Themis does not automatically resume provider batch jobs on your behalf

For provider batch APIs, pair this backend with the export/import workflow in
[Hand Off Generation or Evaluation](external-stage-handoffs.md).

## API-Based Models and Parallelism

You can parallelize API-backed models at the orchestration layer the same way
you parallelize local models: by increasing the number of planned work items and
the scheduler's in-flight budget.

Treat this as partial support rather than full abstraction:

- Themis parallelizes work-item scheduling
- your engine decides how many concurrent provider requests are safe
- your engine decides whether to use request batching, provider retries, or
  serialized access under a quota

## Cost and Capacity Planning

Before increasing concurrency, estimate the planned work:

```python
estimate = orchestrator.estimate(experiment)
print(estimate.work_items_by_stage)
print(estimate.estimated_total_tokens)
```

`estimate()` is best-effort, but it is useful for catching obviously oversized
runs before they reach your provider or cluster.

## Recommended Operational Pattern

- keep one stable storage root per experiment lineage
- use `plan()` or `submit()` before large changes
- treat provider-specific throttling as engine logic, not doc-only magic
- use batch or worker-pool backends only when you actually have external workers
  or providers to complete the pending items
- rely on resume semantics to recover from interruptions instead of manually
  restarting whole runs
