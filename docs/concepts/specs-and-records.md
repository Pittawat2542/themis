# Specs and Records

Themis draws a hard boundary between input configuration and output artifacts.

## Specs

Specs are immutable Pydantic models used on the write side:

- `ProjectSpec`
- `ExperimentSpec`
- `TrialSpec`
- `ModelSpec`
- `TaskSpec`
- `PromptTemplateSpec`
- `InferenceParamsSpec`

Shared traits:

- frozen and strict
- canonically hashable through `SpecBase`
- validated at construction time

`RuntimeContext` is adjacent to specs but deliberately excluded from the hashable
configuration surface. It carries secrets, environment labels, and resume state.

## Records

Records are immutable output artifacts produced by the runtime:

- `InferenceRecord`
- `ExtractionRecord`
- `EvaluationRecord`
- `CandidateRecord`
- `TrialRecord`
- `RecordTimelineView`

The central relationship is:

```text
TrialRecord
  -> candidates[]
  -> CandidateRecord
     -> inference
     -> extractions[]
     -> evaluation
     -> conversation
```

## Why `spec_hash` Appears Everywhere

Specs and records both link back to stable hashes:

- specs use canonical hashing for identity and deduplication
- records keep those hashes so storage and projections can connect write-side and
  read-side data without fragile object references

This is what allows projections, reports, and the artifact store to stay
consistent across resume and replay flows.
