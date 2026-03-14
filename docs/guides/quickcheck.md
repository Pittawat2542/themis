# Use the Quickcheck CLI

`themis-quickcheck` reads the SQLite projection tables produced by the runtime.
Generation rows use `gen`, transform rows use `tf:<transform_hash>`, and
evaluation rows use `ev:<evaluation_hash>`.

## Failures

```bash
themis-quickcheck failures --db .cache/themis/hello-world/themis.sqlite3 --limit 20
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
  --db .cache/themis/hello-world/themis.sqlite3 \
  --evaluation-hash <evaluation_hash>
```

## Scores

```bash
themis-quickcheck scores --db .cache/themis/hello-world/themis.sqlite3 --metric exact_match
```

Aggregates scores by `overlay_key`, model, task, and metric.

To inspect one evaluation overlay directly:

```bash
themis-quickcheck scores \
  --db .cache/themis/hello-world/themis.sqlite3 \
  --metric exact_match \
  --evaluation-hash <evaluation_hash>
```

## Latency

```bash
themis-quickcheck latency --db .cache/themis/hello-world/themis.sqlite3
```

Defaults to the generation overlay. Use `--transform-hash` or
`--evaluation-hash` to inspect a specific overlay:

```bash
themis-quickcheck latency \
  --db .cache/themis/hello-world/themis.sqlite3 \
  --evaluation-hash <evaluation_hash>
```

Prints count plus average and percentile latency/token summaries for the
selected overlay.
