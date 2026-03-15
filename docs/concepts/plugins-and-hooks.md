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

## What Lives in Specs vs the Registry

Not every moving part is a plugin. The division is:

- specs describe the experiment matrix: models, tasks, prompts, transforms,
  evaluations, and inference params
- the registry binds execution behavior: engines, extractors, metrics, judges,
  and hooks

In practice:

- add a new model by extending `ExperimentSpec.models`
- add a new prompt by extending `ExperimentSpec.prompt_templates`
- add a new extractor, metric, or judge behavior by registering a plugin

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

## Planning-Time Validation

`TrialPlanner` validates each planned `TrialSpec` against the current registry
snapshot before execution starts. That validation pass checks:

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

## Compose Multiple Metrics and Judges

`EvaluationSpec.metrics` is a list, so the same generated candidate can be
scored by more than one metric in the same evaluation pass.

That includes judge-backed metrics. Each metric can:

- call `judge_service` independently
- use its own `JudgeInferenceSpec`
- use a different judge prompt
- parse the judge response in its own metric logic

This is the intended way to run multiple judges or multiple judge prompts over
the same candidate set without duplicating generation.

## Qualitative Tags Belong in Metric Output

When you want qualitative analysis on top of scalar scores, keep the structured
labels in metric output rather than inventing a second reporting system.

Typical patterns are:

- `details={"tags": ["hallucination", "format_error"]}`
- `details={"tag": "refusal"}`
- candidate-local warnings attached by extractors or metrics

Those tags become available later through `ExperimentResult.iter_tagged_examples()`.
