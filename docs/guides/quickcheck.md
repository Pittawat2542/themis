# Use the Quickcheck CLI

`themis-quickcheck` reads the SQLite summary tables produced by the runtime.

## Failures

```bash
themis-quickcheck failures --db .cache/themis/hello-world/themis.sqlite3 --limit 20
```

Shows recent failed trials with:

- `trial_hash`
- `model_id`
- `task_id`
- `item_id`
- error fingerprint
- error preview

## Scores

```bash
themis-quickcheck scores --db .cache/themis/hello-world/themis.sqlite3 --metric exact_match
```

Aggregates scores by model, task, and metric.

## Latency

```bash
themis-quickcheck latency --db .cache/themis/hello-world/themis.sqlite3
```

Prints count plus average and percentile latency/token summaries.
