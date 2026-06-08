---
title: Glossary
diataxis: reference
audience: all readers
goal: Provide a shared vocabulary for Themis concepts that appear across tutorials, guides, reference, and explanation pages.
---

# Glossary

- `artifact`: persisted payload produced by generation or evaluation and stored for later inspection
- `Automation Surface`: a config file, CLI argument, submission manifest, worker queue, or batch request that transports or overrides an Executable Definition for automation
- `benchmark`: named catalog entry that resolves to a dataset plus any adapter-specific behavior
- `candidate`: one generation attempt for a case
- `candidate set subject`: a workflow-backed evaluation subject containing one or more candidates
- `case`: one dataset row evaluated by the runtime
- `component`: a generator, reducer, parser, metric, or judge model with `component_id`, `version`, and `fingerprint()`
- `Evidence`: durable material preserved so a result can be inspected later, including snapshots, events, artifacts, traces, telemetry, and projections
- `Executable Definition`: an importable Python module that constructs an Experiment and is the canonical document of what the experiment means
- `Experiment`: a versioned research definition that describes the dataset, generation behavior, evaluation behavior, defaults, and arguments needed to produce evidence-backed results
- `explanation`: a concept document that helps the reader understand why Themis works the way it does
- `how-to guide`: a task-oriented document that helps an already-motivated user achieve a specific outcome
- `provenance`: metadata recorded with a run that does not change `run_id`
- `Replaceable Boundary`: a named point where user code can replace Themis defaults while preserving runtime semantics
- `reduced candidate`: the candidate selected or synthesized after reduction
- `reference`: lookup-oriented documentation for exact commands, types, fields, and interfaces
- `Run Identity`: the identity-bearing part of a RunSnapshot that defines the logical experiment and determines `run_id`
- `RunSnapshot`: the immutable compiled artifact that captures identity, provenance, datasets, and resolved component refs
- `run`: one execution of a compiled snapshot
- `score`: the metric output or score row recorded for a metric on a case
- `Score Claim`: a derived statement about model or workflow quality that must remain traceable back to Evidence
- `subject`: the object passed into workflow-backed scoring, such as a candidate set, trace, or conversation
- `trace`: structured step-by-step execution data captured from generation or evaluation
- `tutorial`: a learning-oriented lesson that guides a newcomer to a successful outcome

Score claims are conclusions over evidence; raw scores and metric rows are not substitutes for the evidence that produced them.
