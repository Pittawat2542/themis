# Storage

Storage, summaries, projections, and artifact persistence.

!!! warning "Implementation detail pages"
    `ProjectSpec.storage`, `StorageConfig`, and the documented blob-storage
    specs are the stable user-facing surface. The repository, projection, and
    schema modules on this page are importable for inspection and debugging, but
    they are implementation detail rather than the stable extension surface.

## Storage Model

- `ProjectSpec.storage` accepts `StorageConfig`, the backend-neutral union of
  `SqliteBlobStorageSpec` and `PostgresBlobStorageSpec`.
- `StorageSpec` remains a compatibility alias for `SqliteBlobStorageSpec` when
  you want the legacy short name for SQLite-only projects.
- The event repository is the source of truth. It stores append-only lifecycle
  events plus canonical serialized specs.
- Projection tables exist for read-heavy workflows: trial summaries, candidate
  summaries, metric scores, timelines, and observability references.
- Trial and candidate summaries are keyed by overlay so generation,
  transform, and evaluation projections can succeed or fail independently.
- Terminal-state checks and projection overlay checks use direct SQL lookups
  instead of full event hydration. That keeps read paths fast and avoids
  unnecessary replay work.
- SQLite stores use format `stage_overlays_v2`. Stores without that format are
  rejected rather than migrated in place.
- Migration utilities copy persisted run manifests and normalized stage work
  items as well as specs, events, artifacts, and observability links.
- Reporting and comparison paths should prefer projected `trial_summary` rows
  and `metric_scores` rows. Full `TrialRecord` materialization is
  available for provenance-heavy workflows.
- Event payload and metadata types live on the dedicated
  [Types](types.md) reference page to avoid duplicated anchors.

::: themis.storage
    options:
      show_root_heading: false

::: themis.storage.artifact_store
    options:
      show_root_heading: false

::: themis.storage.event_repo
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
