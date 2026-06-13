# Themis

Themis is a Python-canonical LLM evaluation experiment platform for researchers. It exists to make rapid iteration possible without losing the rigor needed to defend a result later.

## Language

**Experiment**:
A versioned research definition that describes the dataset, generation behavior, evaluation behavior, defaults, and arguments needed to produce evidence-backed results.
_Avoid_: job, task, benchmark run

**Executable Definition**:
An importable Python module that constructs an **Experiment** and is the canonical document of what the experiment means.
_Avoid_: ad hoc script, notebook-only definition, config-as-source-of-truth

**RunSnapshot**:
The immutable compiled artifact that freezes an **Experiment** into executable identity and provenance.
_Avoid_: config dump, runtime plan

**Run Identity**:
The identity-bearing part of a **RunSnapshot** that defines the logical experiment and determines `run_id`.
_Avoid_: full config, environment

**Provenance**:
The recorded execution context for a run that explains where and how it happened without redefining the logical experiment.
_Avoid_: identity, metadata bucket

**Evidence**:
Durable material preserved so a result can be inspected later, including snapshots, events, artifacts, traces, telemetry, and projections.
_Avoid_: output, logs

**Score Claim**:
A derived statement about model or workflow quality that must remain traceable back to **Evidence**.
_Avoid_: truth, result

**Replaceable Boundary**:
A named point where user code can replace Themis defaults while preserving runtime semantics, such as dataset, generator, selector, reducer, parser, metric, judge, store, or reporter.
_Avoid_: arbitrary plugin hook, internal override

**Automation Surface**:
A config file, CLI argument, submission manifest, worker queue, or batch request that transports or overrides an **Executable Definition** for automation.
_Avoid_: canonical experiment definition

**Benchmark Kit**:
A Themis-owned Python helper that turns catalog benchmark metadata into a reviewed **Experiment** with dataset, generation, parsing, metric, and judge defaults.
_Avoid_: imported task class, benchmark script, config-only benchmark

**Suite Definition**:
A named **Automation Surface** that expands to multiple benchmark or executable definitions while preserving one **RunSnapshot** per executable run.
_Avoid_: mega-run, aggregate experiment, canonical suite source

**Runtime Preset**:
A named convenience layer that resolves to normal session, evaluation, runtime, or storage configuration for an **Experiment**.
_Avoid_: harness runtime, hidden execution mode

**Execution Resource Plan**:
Provenance describing expected work, concurrency, provider calls, and backend requirements for a run.
_Avoid_: run identity, scheduler truth

**Provider Telemetry**:
Evidence captured from generator or judge provider calls, such as latency, retries, token usage, model identifiers, and failure categories.
_Avoid_: score, identity metadata

## Product philosophy

- Themis is an LLM evaluation experiment platform, not a generic workflow engine.
- Python modules are the canonical source of truth for serious experiments.
- Config files and CLI arguments are automation surfaces, useful for transport, submission, and environment-specific overrides.
- Defaults are part of the executable contract once resolved; they support fast iteration by making common choices explicit.
- Runtime controls such as concurrency, retry settings, store paths, and duplicate-run policy are provenance unless they change the logical experiment.
- Scores are claims over evidence, not substitutes for evidence.
- Themis should ship complete defaults and make each important boundary replaceable.

## Engineering philosophy

- Prefer concrete, typed interfaces with small public surfaces.
- Keep modules deep: callers should understand the concept they are using, not the internal choreography behind it.
- Add extension points only at stable, named replaceable boundaries.
- Keep conventions consistent across Python API, config, CLI, docs, examples, and tests.
- Treat code, docs, runnable examples, and release metadata as one product surface.
- Test behavior through public interfaces and durable evidence, not incidental implementation details.

## Relationships

- An **Executable Definition** constructs one **Experiment**.
- An **Experiment** compiles to one **RunSnapshot** for a resolved set of defaults and arguments.
- A **RunSnapshot** contains **Run Identity** and **Provenance**.
- **Run Identity** determines `run_id`; **Provenance** documents execution context.
- **Evidence** supports one or more **Score Claims**.
- A **Replaceable Boundary** lets user code customize an **Experiment** without changing Themis-owned runtime semantics.
- A **Reporter** is a Replaceable Boundary over **Evidence** and projections; changing it does not redefine **Run Identity**.
- An **Automation Surface** may launch, transport, or override an **Executable Definition**, but it is not the canonical definition.
- A **Benchmark Kit** may construct an **Experiment**, but the resulting importable Python **Executable Definition** remains the canonical source for serious research.
- A **Suite Definition** expands into separate executable runs; it does not collapse multiple logical experiments into one **RunSnapshot**.
- A **Runtime Preset** is resolved before compilation and is recorded as provenance when it only changes runtime behavior.
- An **Execution Resource Plan** and **Provider Telemetry** are **Evidence** or provenance; they do not redefine **Run Identity** unless their resolved choices change the logical experiment.

## Example dialogue

> **Dev:** "Can we put this experiment in a YAML file so workers can submit it?"
> **Domain expert:** "Use YAML as an **Automation Surface** if it helps submission, but keep the reviewed Python **Executable Definition** as the source of truth."
>
> **Dev:** "The score improved, so can we call the model better?"
> **Domain expert:** "Only as a **Score Claim**. Show the **Evidence**: snapshot, artifacts, failures, traces, and projections."

## Flagged ambiguities

- "Experiment platform" means an LLM evaluation experiment platform, not a domain-general workflow engine.
- "Documented by code" means reviewed importable Python modules, not throwaway scripts or notebook-only definitions.
- "Config" means an automation surface for transport and overrides, not the canonical source of truth for serious experiments.
- "Result" is overloaded; use **Score Claim** for derived metric statements and **Evidence** for durable inspection material.
