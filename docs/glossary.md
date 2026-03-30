---
title: Glossary
diataxis: reference
audience: all readers
goal: Provide a shared vocabulary for Themis concepts that appear across tutorials, guides, reference, and explanation pages.
---

# Glossary

- `artifact`: persisted payload produced by generation or evaluation and stored for later inspection
- `benchmark`: named catalog entry that resolves to a dataset plus any adapter-specific behavior
- `candidate`: one generation attempt for a case
- `candidate set subject`: a workflow-backed evaluation subject containing one or more candidates
- `case`: one dataset row evaluated by the runtime
- `component`: a generator, reducer, parser, metric, or judge model with `component_id`, `version`, and `fingerprint()`
- `explanation`: a concept document that helps the reader understand why Themis works the way it does
- `how-to guide`: a task-oriented document that helps an already-motivated user achieve a specific outcome
- `provenance`: metadata recorded with a run that does not change `run_id`
- `reduced candidate`: the candidate selected or synthesized after reduction
- `reference`: lookup-oriented documentation for exact commands, types, fields, and interfaces
- `RunSnapshot`: the immutable compiled artifact that captures identity, provenance, datasets, and resolved component refs
- `run`: one execution of a compiled snapshot
- `score`: the final metric output recorded for a metric on a case
- `subject`: the object passed into workflow-backed scoring, such as a candidate set, trace, or conversation
- `trace`: structured step-by-step execution data captured from generation or evaluation
- `tutorial`: a learning-oriented lesson that guides a newcomer to a successful outcome
