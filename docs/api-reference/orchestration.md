# Orchestration

Planning, execution, projection, and orchestration facades.

## Execution Semantics

- `Orchestrator` exposes stage-specific entry points: `generate()`,
  `transform()`, `evaluate()`, `run()`, and `import_candidates()`.
- Stage-specific entry points validate only the plugins required for the stages
  they execute. `generate()` validates generation plugins, `transform()`
  validates transforms, and `evaluate()` validates transforms plus evaluations.
- Candidate work is scheduled through one bounded global queue configured by
  `ExecutionPolicySpec.max_in_flight_work_items`.
- Resume behavior is projection-aware: completed generation, transform, and
  evaluation overlays are skipped independently when matching projections exist.
- Synchronous orchestration entry points are safe to call from running event
  loops such as notebooks and async application shells.
- Projection refresh happens after stage execution so `ExperimentResult` can
  serve reports, comparisons, and timelines from read-side tables instead of
  forcing every consumer down the full replay path.

::: themis.orchestration.orchestrator
    options:
      show_root_heading: false

::: themis.orchestration.trial_planner
    options:
      show_root_heading: false

::: themis.orchestration.executor
    options:
      show_root_heading: false

::: themis.orchestration.trial_runner
    options:
      show_root_heading: false

::: themis.orchestration.candidate_pipeline
    options:
      show_root_heading: false
