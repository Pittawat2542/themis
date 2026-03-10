# Storage and Resume

Themis stores run state under `StorageSpec.root_dir`.

## Layout

At minimum you will see:

- `themis.sqlite3` for specs, events, summaries, and score rows
- `artifacts/` when blob compression is enabled

The SQLite database holds:

- canonical serialized specs
- append-only `trial_events`
- `trial_summary` and `candidate_summary` projections
- flattened `metric_scores`
- `observability_refs` for external trace links such as Langfuse URLs

## Artifact Storage

Large payloads such as prompts, dataset items, and inference outputs can be
stored as content-addressed blobs. When compression is enabled, Themis uses
deduplicated artifact hashes instead of rewriting large JSON payloads for every
event row.

Judge-audit trails also live in artifact storage so expensive judge traces stay
inspectable without bloating the SQLite summary tables.

## Resume Semantics

`TrialExecutor` skips a trial when both conditions hold:

1. the latest terminal event is `trial_completed`
2. a projection already exists for the requested `eval_revision`

That means resume is tied to both the trial hash and the evaluation revision.

## Privacy and Storage Control

If you set `StorageSpec.store_item_payloads=False`, the runtime still stores the
trial and candidate projections, but `RecordTimelineView.item_payload` will be
empty. This is useful when dataset items contain sensitive fields.

## What Timeline Views Can Hydrate

`RecordTimelineView` combines the stage timeline with related stored payloads.
Depending on what the run produced, a candidate view can include:

- `conversation`
- `inference`
- `extractions`
- `evaluation`
- `judge_audit`
- `observability`
