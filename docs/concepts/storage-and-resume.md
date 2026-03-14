# Storage and Resume

Themis stores run state in either a local SQLite store plus blob directory, or
in Postgres plus a local blob directory.

## Layout

For `sqlite_blob`, at minimum you will see:

- `themis.sqlite3` for specs, events, summaries, and score rows
- `artifacts/` when blob compression is enabled

The event/projection database holds:

- canonical serialized specs
- append-only `trial_events`
- persisted `run_manifests`
- normalized `stage_work_items`
- `trial_summary` and `candidate_summary` projections
- flattened `metric_scores`
- `observability_links` for provider-neutral external trace links

Trial-event metadata is stored as JSON for compatibility, but the runtime
decodes known event shapes into typed metadata models before replay and overlay
selection.

Spec, transform, and evaluation hashes remain the current 12-character public
identifiers. Internally, Themis also computes full canonical hashes and rejects
any write or task-resolution step that would alias two different canonical
payloads to the same short hash.

New databases use store format `stage_overlays_v2`. Older
`stage_overlays_v1` databases are rejected instead of being migrated in place.

## Artifact Storage

Large payloads such as prompts, dataset items, and inference outputs can be
stored as content-addressed blobs. When compression is enabled, Themis uses
deduplicated artifact hashes instead of rewriting large JSON payloads for every
event row.

Judge-audit trails also live in artifact storage so expensive judge traces stay
inspectable without bloating the SQLite summary tables.

Run manifests are stored as canonical JSON snapshots plus normalized stage work
items. That lets Themis export only the missing generation or evaluation work
without turning read-side inspection into a write path.

## Intermediate Artifacts Stay Reusable

Themis keeps stage outputs separate on purpose:

- generation stores raw candidates keyed by `trial_hash`
- transforms store normalized candidates keyed by `trial_hash + transform_hash`
- evaluations store metric-backed candidates keyed by `trial_hash + evaluation_hash`

That separation is what makes these workflows possible without regenerating
everything:

- rerun evaluation after adding a new metric
- add a new output transform and re-extract existing candidates
- export only missing evaluation work to an external judge system
- inspect generated text now and score it later

`ExperimentResult` mirrors the same separation on the read side:

- the default view reads raw generated candidates
- `result.for_transform(transform_hash)` reads transformed outputs
- `result.for_evaluation(evaluation_hash)` reads scored candidates

## Resume Semantics

`TrialExecutor` skips a trial only when the requested overlay has a successful
materialized projection for the current deterministic stage identity.

That means resume is tied to deterministic stage identity:

- generation resume uses `trial_hash`
- output-transform resume uses `trial_hash + transform_hash`
- evaluation resume uses `trial_hash + evaluation_hash`

Overlay cache state is independent. A failed transform or evaluation overlay
does not mark the generation overlay as failed, and failed overlays remain
rerunnable on the next resume attempt.

`Orchestrator.plan()` and the external handoff APIs use stage work items to make
that resume story visible. Each work item records one deterministic
generation/transform/evaluation unit and whether it is still pending or already
completed in the current store.

## Re-Extract and Re-Evaluate Without Re-Generating

Because overlays are independent, you can change downstream stages while keeping
the existing generation overlay:

- `transform()` runs declared output transforms against stored generated
  candidates
- `evaluate()` materializes required transforms if they are missing, then runs
  the evaluation overlay
- `export_evaluation_bundle()` exports only the evaluation items that are still
  pending for the current store

This is the normal path when you add a metric, refine an extractor, or score the
same generated text in a separate system.

## Privacy and Storage Control

If your `StorageConfig` sets `store_item_payloads=False`, the runtime stores the
trial and candidate projections, but `RecordTimelineView.item_payload` is
empty. This is useful when dataset items contain sensitive fields.

## Read Projections First, Hydrate Full Records on Demand

The storage layout is optimized so common post-run workflows do not need to load
every large artifact back into memory.

Prefer projection-backed helpers when you want summaries:

- `leaderboard()` for aggregate model/task/metric tables
- `compare()` for paired statistical comparisons
- `themis-quickcheck` for fast SQLite-only inspection
- trial and candidate summary rows for downstream SQL queries

Hydrate full trials or timelines only when you need provenance-heavy drilldown,
such as:

- one candidate's raw response
- extraction failures or warnings
- judge audit traces
- one trial's timeline and stored payloads

## Large Responses and Storage Efficiency

Large prompts, dataset items, and inference outputs can live in the blob store
instead of being duplicated inside every event row. In practice that means:

- compressed blob storage reduces repeated large payloads
- judge traces stay inspectable without bloating summary tables
- summary and score queries stay small even when generated text is large

If you need to analyze a few large responses, pull targeted trials or timeline
views. If you need aggregate reporting, stay on the projection-backed APIs so
the read path does not materialize every stored artifact.

## What Timeline Views Can Hydrate

`RecordTimelineView` combines the stage timeline with related stored payloads.
Depending on what the run produced, a candidate view can include:

- `conversation`
- `inference`
- `extractions`
- `evaluation`
- `judge_audit`
- `observability`
