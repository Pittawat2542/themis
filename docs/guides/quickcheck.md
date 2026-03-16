# Use the Quickcheck CLI

`themis-quickcheck` reads the SQLite projection tables produced by the runtime.
The same commands are also available under the parent CLI as `themis quickcheck
...`.
Generation rows use `gen`, transform rows use `tf:<transform_hash>`, and
evaluation rows use `ev:<evaluation_hash>`.

The deterministic example in these docs uses:

```text
.cache/themis-examples/01-hello-world/themis.sqlite3
```

## Failures

```bash
themis-quickcheck failures --db .cache/themis-examples/01-hello-world/themis.sqlite3 --limit 20
themis quickcheck failures --db .cache/themis-examples/01-hello-world/themis.sqlite3 --limit 20
```

Shows recent failed trials with:

- `trial_hash`
- `overlay_key`
- `model_id`
- `task_id`
- `item_id`
- error fingerprint
- error preview

Use `--transform-hash` or `--evaluation-hash` to inspect one overlay directly:

```bash
themis-quickcheck failures \
  --db .cache/themis-examples/01-hello-world/themis.sqlite3 \
  --evaluation-hash <evaluation_hash>
```

Get the hash from Python first:

```python
print(result.evaluation_hashes[0])
```

## Scores

```bash
themis-quickcheck scores --db .cache/themis-examples/01-hello-world/themis.sqlite3 --metric exact_match
themis quickcheck scores --db .cache/themis-examples/01-hello-world/themis.sqlite3 --metric exact_match
```

Aggregates scores by `overlay_key`, model, task, and metric.

Expected output for the hello-world example:

```text
ev:fc7ad3e8b3e2	demo-model	arithmetic	exact_match	1.0000	2
```

Use `--task` when you want to inspect one task only:

```bash
themis-quickcheck scores \
  --db .cache/themis-examples/01-hello-world/themis.sqlite3 \
  --metric exact_match \
  --task arithmetic
```

To inspect one evaluation overlay directly:

```bash
themis-quickcheck scores \
  --db .cache/themis-examples/01-hello-world/themis.sqlite3 \
  --metric exact_match \
  --evaluation-hash <evaluation_hash>
```

## Latency

```bash
themis-quickcheck latency --db .cache/themis-examples/01-hello-world/themis.sqlite3
themis quickcheck latency --db .cache/themis-examples/01-hello-world/themis.sqlite3
```

Defaults to the generation overlay. Use `--transform-hash` or
`--evaluation-hash` to inspect a specific overlay:

```bash
themis-quickcheck latency \
  --db .cache/themis-examples/01-hello-world/themis.sqlite3 \
  --evaluation-hash <evaluation_hash>
```

Prints count plus average and percentile latency/token summaries for the
selected overlay.

Expected output for the hello-world example:

```text
count=2 latency_ms(avg=2.00,p50=2.00,p95=2.00) tokens_in(avg=n/a) tokens_out(avg=n/a)
```

For a fully successful run like the hello-world example, `failures` prints no
rows.
