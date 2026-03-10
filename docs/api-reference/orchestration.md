# Orchestration

Planning, execution, projection, and orchestration facades.

## Execution Semantics

- Trial execution is sequential at the orchestrator level in the current public
  API. `parallel_trials` has been removed because it did not provide real
  trial-level concurrency.
- Candidate execution remains concurrent within a trial through
  `parallel_candidates`, which must be a positive integer and now fails fast
  during construction if misconfigured.
- Resume behavior is projection-aware: completed trials with matching revision
  markers are skipped, while partial candidate work is replayed from the event
  log and continued.
- Projection refresh happens after execution so `ExperimentResult` can serve
  reports, comparisons, and timelines from read-side tables instead of forcing
  every consumer down the full replay path.

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
