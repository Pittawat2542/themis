# Storage

Storage, summaries, projections, and artifact persistence.

## Storage Model

- The event repository is the source of truth. It stores append-only lifecycle
  events plus canonical serialized specs.
- Projection tables exist for read-heavy workflows: trial summaries, candidate
  summaries, metric scores, timelines, and observability references.
- Terminal-state checks and projection revision checks use direct SQL lookups
  instead of full event hydration. That keeps malformed historical payloads from
  breaking unrelated read paths and avoids unnecessary replay work.
- Reporting and comparison paths should prefer projected `trial_summary` rows
  and `metric_scores` rows. Full `TrialRecord` materialization remains available
  for provenance-heavy workflows, but it is no longer the default aggregate
  analysis path.

::: themis.storage
    options:
      show_root_heading: false

::: themis.storage.artifact_store
    options:
      show_root_heading: false

::: themis.storage.event_repo
    options:
      show_root_heading: false

::: themis.storage.events
    options:
      show_root_heading: false

::: themis.storage.projection_repo
    options:
      show_root_heading: false

::: themis.storage.observability
    options:
      show_root_heading: false

::: themis.storage.sqlite_schema
    options:
      show_root_heading: false
