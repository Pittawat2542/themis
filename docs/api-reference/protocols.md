# Protocols

Execution and storage interfaces implemented by plugins and repositories.

## Contract Notes

- `InferenceEngine.infer(...)` and judge-backed engine calls always receive a
  concrete `RuntimeContext`, not an untyped mapping.
- Plugin boundaries are typed. Engines, extractors, and metrics should return
  the declared record models instead of loose dict payloads.
- Repository protocols split write and read concerns deliberately: the event
  repository is append-only, while the projection repository exposes query
  surfaces such as candidate scores, timelines, and trial summaries.
- Use `ProjectionRepository.iter_trial_summaries(...)` when you only need
  projected metadata for reports or comparisons. It exists specifically to avoid
  hydrating full `TrialRecord` objects on aggregate read paths.

::: themis.contracts.protocols
    options:
      show_root_heading: false
