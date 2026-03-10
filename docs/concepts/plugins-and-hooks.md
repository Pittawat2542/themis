# Plugins and Hooks

Themis uses protocols for runtime behavior and a registry for binding names to
implementations.

## Plugin Types

| Plugin | Protocol | Purpose |
| --- | --- | --- |
| Inference engine | `InferenceEngine` | Produces `InferenceResult` for one trial + dataset item |
| Extractor | `Extractor` | Parses a candidate into structured output |
| Metric | `Metric` | Scores a candidate against task context |
| Judge service | `JudgeService` | Performs extra model calls for judge-backed metrics |
| Candidate selector | `CandidateSelectionStrategy` | Picks a best candidate from a set |
| Report exporter | `ReportExporter` | Writes assembled reports to disk |

## Registry Behavior

`PluginRegistry` is instance-scoped. That matters because it:

- avoids global test pollution
- lets separate experiments register different plugin sets
- makes plugin metadata part of an explicit runtime object instead of process-wide state

It also auto-registers four built-in extractors:

- `regex`
- `json_schema`
- `first_number`
- `choice_letter`

## Hooks

Hooks are lightweight transforms around the pipeline:

- `pre_inference`
- `post_inference`
- `pre_extraction`
- `post_extraction`
- `pre_eval`
- `post_eval`

Use hooks when you want to adjust prompts or candidate objects without replacing
the engine, extractor, or metric itself.

## Planning-Time Compatibility

`TrialPlanner` validates each planned `TrialSpec` against the current registry
snapshot before execution starts. That compatibility pass checks:

- whether the requested provider, extractors, and metrics are registered
- whether plugin API major versions match the supported runtime
- whether `response_format` and `logprobs` fit the engine capabilities
- whether a prompt token estimator reports a prompt larger than the engine's
  declared context window

That is why provider and metric mismatches fail during planning instead of after
part of a run has already executed.

## Choose Hooks vs Plugins

Choose a plugin when you need to own a full execution stage:

- inference behavior
- structured extraction
- scoring logic
- judge-model calls

Choose a hook when you want to wrap or tweak an existing stage:

- prepend or rewrite prompt messages
- normalize candidate objects before extraction
- attach lightweight metadata before or after scoring
