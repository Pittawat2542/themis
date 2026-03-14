# Specs and Records

Themis draws a hard boundary between input configuration and output artifacts.

## Specs

Specs are immutable Pydantic models used on the write side:

- `ProjectSpec`
- `ExperimentSpec`
- `TrialSpec`
- `ModelSpec`
- `TaskSpec`
- `GenerationSpec`
- `OutputTransformSpec`
- `EvaluationSpec`
- `PromptTemplateSpec`
- `InferenceParamsSpec`

Shared traits:

- frozen and strict
- canonically hashable through `SpecBase`
- validated at construction time

`RuntimeContext` is adjacent to specs but deliberately excluded from the hashable
configuration surface. It carries secrets, environment labels, and resume state.

`ProjectSpec` also carries the execution backend configuration. Local execution
is the default, and worker-pool or batch backends are selected through the
discriminated `execution_backend` field instead of a separate runtime config
system.

`TaskSpec` is stage-oriented:

- `generation` declares whether the task can produce raw candidates
- `output_transforms` declare named normalization passes over generated output
- `evaluations` declare named scoring passes, optionally bound to one transform

That separation is reflected in deterministic identities:

- generation uses `trial_hash`
- transforms use `transform_hash`
- evaluations use `evaluation_hash`

`ItemSamplingSpec` can further narrow the dataset before subset or stratified
sampling by `item_ids` and exact-match `metadata_filters`. That keeps benchmark
slices declarative and reproducible.

## Code as Config, with File-Backed Project Policy

Themis is intentionally code-first for the experiment matrix:

- `ExperimentSpec` is authored in Python
- `ProjectSpec` can be authored in Python or loaded from `project.toml` /
  `project.json`

That split keeps the dynamic parts of an experiment in code:

- model lists
- prompt-template sweeps
- inference parameter grids
- task-local metrics and transforms

while still allowing shared infrastructure policy to live in a reusable config
file:

- storage root or database URL
- retry policy
- backend selection
- project-wide seed and metadata

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

- specs use 12-character public aliases for identity and deduplication
- records keep those hashes so storage and projections can connect write-side and
  read-side data without fragile object references

Internally, those short aliases are backed by full canonical hashes. The runtime
keeps the short identifiers stable for user-facing APIs and rejects collisions
if two different canonical payloads would resolve to the same public hash.

This is what allows projections, reports, and the artifact store to stay
consistent across resume and replay flows.

## Version Pinning and Reproducibility

Exact reproduction depends on versioning the same inputs that feed those hashes:

- the Python code that builds `ExperimentSpec`
- the `ProjectSpec` payload or project file
- dataset identity and `DatasetSpec.revision` when the source supports it
- model/provider IDs and provider `extras`
- prompt templates and inference params

Themis persists the concrete `ProjectSpec` and `ExperimentSpec` inside the run
manifest, so a completed run always keeps the exact planning snapshot that
produced it.

Project-level determinism and item sampling are stable under `global_seed`, but
provider-side determinism still depends on whether the engine honors
`InferenceParamsSpec.seed` and deterministic sampling settings such as
`temperature=0.0`.

## Overlay-Scoped Reads

`ExperimentResult` can read the same trial set through different overlays:

- the default generation view reads raw generated candidates
- `result.for_transform(transform_hash)` reads transformed candidates
- `result.for_evaluation(evaluation_hash)` reads evaluation-scored candidates

That keeps generation identity stable while allowing post-generation stages to
be recomputed independently.

Planning produces one more immutable artifact: `RunManifest`. It is not a
projection record, but it is persisted alongside the event log so runs can be
diffed, exported, and resumed against a concrete snapshot of the experiment.

## Diffable Configurations

Because specs are immutable and canonically hashed, you can compare two planned
experiments before running them:

```python
diff = orchestrator.diff_specs(baseline_experiment, treatment_experiment)

print(diff.changed_experiment_fields)
print(diff.added_trial_hashes[:3])
print(diff.added_evaluation_hashes[:3])
```

Use that when you want to verify that:

- a new model only adds new trial hashes
- a new metric only adds evaluation overlays
- a new dataset slice changes the trial set in the way you expected
