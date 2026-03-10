# FAQ / Troubleshooting

## Why does `Orchestrator.run()` fail with a dataset-loader error?

`TrialPlanner` needs dataset items to expand an `ExperimentSpec`. Pass a
`dataset_loader` object to the orchestrator, and make sure it implements
`load_task_items(task)`.

## Why are my prompt placeholders not being interpolated automatically?

The current runtime stores prompt messages and dataset context separately. Your
inference engine can render those inputs however it wants, but Themis does not
apply a built-in string templating step during trial execution.

## Why is a rerun not skipping completed work?

Resume checks are tied to:

- the trial hash
- the evaluation revision
- the presence of a completed projection

Changing the spec, storage root, or `eval_revision` makes the runtime treat the
work as new.

## Why does `result.compare()` raise an optional dependency error?

Comparisons and report building need the `stats` extra:

```bash
uv add "themis-eval[stats]"
```

## Why is `view_timeline(...).item_payload` empty?

That happens when `StorageSpec.store_item_payloads` is `False`. The runtime still
stores events and projections, but omits dataset payload blobs.

## Why is `view_timeline(...).observability` empty?

Passing a `TelemetryBus` is not enough by itself. External URLs only appear in
`RecordTimelineView.observability` when a callback such as `LangfuseCallback`
persists refs through a `SqliteObservabilityStore` wired to the same database.

## Why does the built-in `json_schema` extractor say an optional dependency is missing?

Install the `extractors` extra:

```bash
uv add "themis-eval[extractors]"
```

## How do I inspect failures without loading all artifacts?

Use the quickcheck CLI against the SQLite database:

```bash
themis-quickcheck failures --db path/to/themis.sqlite3
```
