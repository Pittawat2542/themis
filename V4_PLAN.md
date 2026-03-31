# Themis V4 Full Rewrite Plan

## Table of Contents

- [Summary](#summary)
- [Design Principles](#design-principles)
- [Concepts: Architecture & Details](#concepts-architecture--details)
  - [System Overview](#system-overview)
  - [RunSnapshot Structure](#runsnapshot-structure)
  - [Execution Pipeline](#execution-pipeline)
  - [Concurrency & Fan-Out Model](#concurrency--fan-out-model)
  - [Storage & Event Architecture](#storage--event-architecture)
  - [Public API Layers](#public-api-layers)
  - [Observability & Telemetry](#observability--telemetry)
- [Target Architecture](#target-architecture)
- [Generation Architecture](#generation-architecture)
- [Evaluation Workflow And Judge Metrics Architecture](#evaluation-workflow-and-judge-metrics-architecture)
- [Public API And Extension Model](#public-api-and-extension-model)
- [Execution, Lifecycle, And State](#execution-lifecycle-and-state)
- [Concurrency And Resource Management](#concurrency-and-resource-management)
- [Storage, Reproducibility, And Provenance](#storage-reproducibility-and-provenance)
- [Configuration Loading](#configuration-loading)
- [Feature Coverage Requirements](#feature-coverage-requirements)
- [V2 To V4 Feature Mapping](#v2-to-v4-feature-mapping)
- [Detailed Implementation Plan](#detailed-implementation-plan)
  - [Phase 1: Foundation](#phase-1-foundation-weeks-12)
  - [Phase 2: Execution Engine](#phase-2-execution-engine-weeks-34)
  - [Phase 3: Evaluation & Metrics](#phase-3-evaluation--metrics-weeks-56)
  - [Phase 4: Read Models, Reporting & Catalog](#phase-4-read-models-reporting--catalog-weeks-78)
  - [Phase 5: CLI & Integration](#phase-5-cli--integration-weeks-910)
  - [Phase 6: Hardening & Release](#phase-6-hardening--release-weeks-1112)
- [Test Plan And Acceptance Criteria](#test-plan-and-acceptance-criteria)
- [Documentation And Examples Follow-Up](#documentation-and-examples-follow-up)
- [Design Decisions And Rationale](#design-decisions-and-rationale)

---

## Summary

Build `v4` as a **clean-slate rewrite with zero backward compatibility and zero legacy support**. Do not ship compatibility adapters, config translators, storage importers, or migration utilities. `v4` only preserves **feature coverage**: every important capability in today's package must exist in the new system, but behind a new architecture centered on one immutable, fully serializable `RunSnapshot`. Documentation and examples are also a ground-up rewrite, but as a dedicated later pass using the **Diátaxis** framework.

## Design Principles

1. **Single execution artifact.** `RunSnapshot` is the only executable artifact. Every input needed to reproduce a run is captured in or referenced by the snapshot. *(Referenced by: Target Architecture, Execution Engine, Storage)*
2. **Generation is a blackbox; evaluation is owned.** Themis treats generation as an opaque function that conforms to a typed input/output contract (`Generator` protocol). Themis owns candidate fan-out, reduction, and the full evaluation pipeline including judge execution. Built-in adapters wrap common providers (OpenAI, vLLM, LangGraph) but users can plug in any generation system. *(Referenced by: Generation Architecture, Evaluation Architecture)*
3. **Evaluation owns its runtime.** Judge-backed metrics compile to evaluation workflows executed by a Themis-owned `WorkflowRunner`. Retry, seeding, artifact persistence, and tracing for judge calls are first-class. *(Referenced by: Evaluation Architecture, Execution Engine)*
4. **Candidate fan-out is an evaluation concern.** Independent repeated generations are separate candidates managed by the planner. The `Generator` produces one candidate per call; Themis calls it N times. *(Referenced by: Generation Architecture, Planning)*
5. **Domain boundaries over shared abstractions.** Generation and evaluation are semantically distinct layers. Judge calls are evaluation, not generation. *(Referenced by: Evaluation Architecture)*
6. **Typed boundaries, no mutable bags.** All cross-stage data passes through typed, read-only boundary objects. No mutable context dicts, no hidden service injection. *(Referenced by: Public API, Metric Interfaces)*
7. **Artifacts persisted by default.** Raw responses, judge artifacts, generation traces, and evaluation traces are stored automatically. Any reported score must be inspectable without re-running the producer. *(Referenced by: Storage, Judge Artifact Persistence)*
8. **Every component is fingerprinted.** Extension points expose `component_id`, `version`, and `fingerprint()`. Snapshot identity derives from these fingerprints. Fingerprints are computed once at `RunSnapshot` compile time and cached — they are never recomputed during execution. *(Referenced by: Public API, RunSnapshot)*
9. **Prove need before building.** Metric families, subject types, and extension points ship only when a concrete use case requires them. Speculative combinations are deferred. *(Referenced by: Metric Families, Typed Evaluation Subjects)*
10. **Clean break, no compatibility layer.** `v4` preserves feature coverage, not APIs, schemas, storage layouts, or naming. No adapters, translators, or importers. *(Referenced by: Release Boundary, Summary)*

---

## Concepts: Architecture & Details

### System Overview

The following diagram shows the high-level architecture of Themis v4 and how the major components relate to each other.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER-FACING APIs                              │
│                                                                             │
│   Layer 1: evaluate(model=..., data=..., metric=...)                        │
│   Layer 2: Experiment(generation=..., evaluation=..., storage=..., ...)     │
│   Layer 3: Extension protocols (themis.core / themis.plugins)               │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Experiment.compile()                               │
│                                                                             │
│   Assembles all inputs into a single immutable RunSnapshot                  │
│   (dataset refs, generator config, metrics, seeds, provenance)              │
│   Fingerprints computed once here and cached for the run lifetime           │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                             RunSnapshot                                     │
│                                                                             │
│   ┌─────────────────────┐    ┌──────────────────────────────────────┐       │
│   │   Identity Fields   │    │        Provenance Fields             │       │
│   │   (determine run_id)│    │   (recorded, don't affect run_id)   │       │
│   │                     │    │                                      │       │
│   │ • dataset fp+rev    │    │ • themis version                     │       │
│   │ • prompt config     │    │ • git commit                         │       │
│   │ • parser fps        │    │ • Python version                     │       │
│   │ • metric fps        │    │ • platform                           │       │
│   │ • generator fp      │    │ • dependency versions                │       │
│   │ • sampling policy   │    │ • environment metadata               │       │
│   │ • reducer config    │    │ • storage config                     │       │
│   │ • judge fan-out     │    │ • runtime config                     │       │
│   │ • seeds             │    │                                      │       │
│   └─────────────────────┘    └──────────────────────────────────────┘       │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
                          ┌────────┴────────┐
                          ▼                 ▼
                   ┌────────────┐    ┌─────────────┐
                   │  Planner   │    │  RunStore    │
                   │            │    │  .resume()   │
                   └─────┬──────┘    └──────┬──────┘
                         │                  │
                         ▼                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            Orchestrator                                     │
│                                                                             │
│   Orchestrates planned work items across all stages                         │
│   Manages concurrency, backpressure, and progress                           │
│   Sole writer of stage-level and run-level events to RunStore               │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
                    ┌──────────────┼──────────────┐
                    ▼              ▼              ▼
┌───────────────────────┐ ┌──────────────┐ ┌──────────────────────────────────┐
│      Generator        │ │   Parser /   │ │        WorkflowRunner            │
│   (blackbox — user    │ │   Reducer    │ │   (evaluation execution only)    │
│    or built-in)       │ │              │ │                                   │
│                       │ │              │ │   • Judge model calls             │
│  Input: Case + ctx    │ │              │ │   • Retries with policy           │
│  Output:              │ │              │ │   • Deterministic seeding         │
│    GenerationResult   │ │              │ │   • Tracing                       │
│    ├── final_output   │ │              │ │   • Step output storage           │
│    ├── trace?         │ │              │ │                                   │
│    ├── conversation?  │ │              │ │   Sole writer of step-level       │
│    └── artifacts?     │ │              │ │   events to RunStore              │
└───────────────────────┘ └──────────────┘ └──────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    RunStore + Read Models + Reporter                        │
│                                                                             │
│   Events ──► Projections ──► RunResult / BenchmarkResult / TimelineView     │
│                                 ──► JSON / MD / CSV / LaTeX exports         │
└─────────────────────────────────────────────────────────────────────────────┘
```

### RunSnapshot Structure

`RunSnapshot` is the single executable artifact (Principle 1). Understanding what makes two runs "the same run" vs "the same run in a different environment" is critical.

**Fingerprint stability rule:** All component fingerprints are computed once at `Experiment.compile()` time and frozen into the snapshot. Fingerprints are never recomputed during execution. This means a bug fix to a `Generator` or `Parser` after compilation does not silently invalidate an in-progress run — the snapshot records what was compiled, and resume always uses the original fingerprints. To pick up a component change, the user must recompile (creating a new `run_id`).

```
RunSnapshot
├── Identity Fields ──────────────────── hash(identity) = run_id
│   │
│   │   Two snapshots with identical identity fields
│   │   represent the SAME logical run.
│   │
│   ├── dataset_fingerprint + revision
│   ├── prompt_config
│   ├── parser_fingerprints[]
│   ├── metric_fingerprints[]
│   │   └── (includes judge model id/version/fp for LLM metrics)
│   ├── generator_id + version + fingerprint
│   ├── candidate_sampling_policy
│   ├── reducer_config
│   ├── judge_fanout_policy
│   └── seeds
│
└── Provenance Fields ────────────────── recorded, NOT part of run_id
    │
    │   Changing these does NOT create a new logical run.
    │   They exist for reproducibility auditing.
    │
    ├── themis_version
    ├── git_commit
    ├── python_version
    ├── platform
    ├── dependency_versions[]
    ├── environment_metadata (non-secret)
    ├── storage_config
    └── runtime_config (parallelism, timeouts, retry)
```

### Execution Pipeline

The following diagram shows the complete data flow through all execution stages, with typed boundary objects at each transition.

```
  Case (from Dataset)
    │
    ▼
┌──────────────────┐
│  Generation      │     ┌─────────────────────────────────────────┐
│                  │     │  Fan-out: N candidates per case          │
│  Generator       │     │  (num_samples, self-consistency, pass@k)│
│  (blackbox)      │     │                                         │
│                  │     │  Orchestrator calls Generator N times    │
└────────┬─────────┘     └─────────────────────────────────────────┘
         │
         │ GenerationResult[]
         │   ├── final_output (required)
         │   ├── trace: list[TraceStep] (optional)
         │   ├── conversation: list[Message] (optional)
         │   └── artifacts: dict (optional)
         ▼
┌──────────────────┐
│  Reduction       │───► ReducedCandidate
│  CandidateReducer│     (best-of-n / majority-vote / synthesis)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Parsing         │───► ParsedOutput
│  Parser          │     (normalize model output for scoring)
└────────┬─────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────┐
│  Scoring / Evaluation                                    │
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │  PureMetric   │  │  LLMMetric   │  │ TraceMetric  │   │
│  │              │  │              │  │              │   │
│  │ Deterministic │  │ Compiles to  │  │ Compiles to  │   │
│  │ scoring      │  │ EvalWorkflow │  │ EvalWorkflow │   │
│  │              │  │ ──► WFRunner │  │ ──► WFRunner │   │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘   │
│         │                 │                 │            │
│         └────────┬────────┴────────┬────────┘            │
│                  ▼                 ▼                      │
│               Score[]    EvaluationExecution[]            │
└──────────────────────────┬───────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────┐
│  RunStore (persist)  ──►  Read Models  ──►  Reporter     │
│                                                          │
│  RunResult · BenchmarkResult · TimelineView · TraceView  │
└──────────────────────────────────────────────────────────┘
```

### Concurrency & Fan-Out Model

The system involves multiplicative fan-out: `cases × candidates × judge_calls × rubric_dimensions`. This diagram shows how concurrency is bounded at each level.

```
    Cases (from Dataset)
     │
     │  N cases
     ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                        Planner                              │
    │                                                             │
    │  Produces work items LAZILY (async generator)               │
    │  Memory: O(concurrency_cap), not O(N×M×K)                  │
    │                                                             │
    │  Axes:                                                      │
    │  ├── Candidate multiplicity (M per case)                    │
    │  └── Judge fan-out (K per scored item)                      │
    └──────────────────────────┬──────────────────────────────────┘
                               │ work items (lazy)
                               ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                      Orchestrator                            │
    │                                                             │
    │  Concurrency Controls:                                      │
    │  ┌─────────────────────────────────────────────────────┐    │
    │  │ Level 1: Per-provider rate limiting                  │    │
    │  │   Token-bucket/leaky-bucket per endpoint             │    │
    │  │   Shared across generation + judge calls             │    │
    │  ├─────────────────────────────────────────────────────┤    │
    │  │ Level 2: Global concurrency cap                     │    │
    │  │   max_concurrent_tasks bounds total parallelism      │    │
    │  ├─────────────────────────────────────────────────────┤    │
    │  │ Level 3: Per-stage concurrency                      │    │
    │  │   Independent limits for generation / evaluation     │    │
    │  │   Prevents evaluation from starving generation       │    │
    │  └─────────────────────────────────────────────────────┘    │
    │                                                             │
    │  Sensible Defaults (zero-config):                           │
    │  • Per-provider: auto-detect from API response headers      │
    │  • Global cap: 32 concurrent tasks                          │
    │  • Per-stage: proportional split (no config needed)         │
    │  Users only touch these knobs when they have a reason to.   │
    │                                                             │
    │  Backpressure:                                              │
    │  • Queue when limits reached (never drop)                   │
    │  • Pause on store write failure                             │
    │  • Stream artifacts to RunStore (don't hold in memory)      │
    └─────────────────────────────────────────────────────────────┘

    Fan-Out Layers (do not conflate):
    ┌───────────────────────────┐    ┌──────────────────────────────┐
    │   GENERATION FAN-OUT      │    │   EVALUATION FAN-OUT          │
    │   (Principle 4)           │    │   (separate layer)            │
    │                           │    │                               │
    │   • num_samples           │    │   • rubric dimensions         │
    │   • self-consistency      │    │   • panel of judges           │
    │   • best-of-n pool        │    │   • repeated judge calls      │
    │   • pass@k pool           │    │   • pairwise tournaments      │
    │                           │    │   • majority-vote judging     │
    └───────────────────────────┘    └──────────────────────────────┘
```

### Storage & Event Architecture

```
    ┌─────────────────────────────────────────────────────────────┐
    │                      RunStore Interface                     │
    │                                                             │
    │   .initialize()    .persist_event()    .resume()            │
    │   .query_events()  .get_projection()   .store_blob()       │
    └──────┬──────────────────────────────────────────┬───────────┘
           │                                          │
           ▼                                          ▼
    ┌──────────────┐                          ┌──────────────┐
    │ SQLite Store │                          │Postgres Store│
    │              │                          │              │
    │ sqlite_store │                          │postgres_store│
    │ (path)       │                          │(url,blob_root│
    │ .initialize()│                          │).initialize()│
    └──────────────┘                          └──────────────┘

    Write Ownership Boundaries:
    ┌───────────────────────────────────────────────────────────────┐
    │  WorkflowRunner  │ Writes step-level events (step start,     │
    │                  │ step complete, step error) during eval     │
    │                  │ workflow execution only.                   │
    ├──────────────────┼───────────────────────────────────────────┤
    │  Orchestrator    │ Writes stage-level events (generation     │
    │                  │ complete, reduction complete, parse        │
    │                  │ complete, score complete) and run-level    │
    │                  │ events (run start, run complete/fail).     │
    │                  │ Single write path — no implicit ordering   │
    │                  │ dependency with WorkflowRunner.            │
    ├──────────────────┼───────────────────────────────────────────┤
    │  Contract        │ WorkflowRunner writes step events to a    │
    │                  │ scoped event sink. Orchestrator writes     │
    │                  │ stage/run events. They never write the     │
    │                  │ same event types.                          │
    └──────────────────┴───────────────────────────────────────────┘

    Event Flow:
    ┌─────────┐     ┌──────────┐     ┌────────────┐     ┌──────────┐
    │RunEvents│────►│ Versioned │────►│ Projections│────►│Read Models│
    │(writes) │     │Event Store│     │  (derived) │     │(queries) │
    └─────────┘     └──────────┘     └────────────┘     └──────────┘

    Event Schema Evolution Rules:
    ┌──────────────────────────────────────────────────────────────┐
    │  • Additive-only: new event types may be added; existing    │
    │    event types may gain optional fields but never remove    │
    │    or change the type of existing fields.                   │
    │  • Schema version stamped on every event. Readers that      │
    │    encounter an unknown event type or unknown fields must   │
    │    skip gracefully (forward compatibility).                 │
    │  • Breaking changes require a new major schema version      │
    │    and a store migration tool.                              │
    │  • Schema version is independent of package version.        │
    └──────────────────────────────────────────────────────────────┘

    Artifact Storage:
    ┌──────────────────────────────────────────────────────────┐
    │  Persisted by default (Principle 7):                     │
    │                                                          │
    │  Generation artifacts             Evaluation artifacts:  │
    │  (from GenerationResult):         • rendered judge prompt│
    │  • final_output                   • judge model id/ver/fp│
    │  • trace (if provided)            • effective seed        │
    │  • conversation (if provided)   • raw judge response      │
    │  • artifacts (if provided)      • parsed judgment         │
    │  • token usage (if provided)    • per-call scores         │
    │                                 • aggregation output      │
    │                                 • retry history           │
    │                                 • judge conversation trace│
    │                                                          │
    │  Large artifacts stored as blob refs, not inline         │
    └──────────────────────────────────────────────────────────┘

    Read Models (projection-backed):
    ┌──────────────┐  ┌─────────────────┐  ┌──────────────┐  ┌───────────┐
    │  RunResult   │  │BenchmarkResult  │  │ TimelineView │  │ TraceView │
    └──────────────┘  └─────────────────┘  └──────────────┘  └───────────┘
```

### Public API Layers

```
    ┌─────────────────────────────────────────────────────────────────┐
    │  Layer 1: Quick Start (5 lines or fewer)                       │
    │                                                                 │
    │    result = evaluate(                                            │
    │        model=openai("gpt-4o"),                                  │
    │        data=[{"input": "...", "expected": "..."}],              │
    │        metric=exact_match,                                      │
    │    )                                                            │
    │                                                                 │
    │    Note: Layer 1 internally constructs an Experiment, compiles  │
    │    a RunSnapshot, and runs it. No workflow graph is exposed.    │
    │    The built-in openai() adapter is the Generator.              │
    ├─────────────────────────────────────────────────────────────────┤
    │  Layer 2: Configurable Experiments                              │
    │                                                                 │
    │    exp = Experiment(                                             │
    │        model="builtin/demo_generator",                           │
    │            generator=openai("gpt-4o"),                           │
    │            candidate_policy=SamplingPolicy(num_samples=3),       │
    │            reducer=majority_vote(),                              │
    │        ),                                                        │
    │        evaluation=EvaluationConfig(                              │
    │            metrics=[exact_match, llm_rubric(judge=...)],         │
    │            parsers=[json_parser()],                              │
    │        ),                                                        │
    │        storage=StorageConfig(                                    │
    │            store=sqlite_store("./results.db"),                   │
    │        ),                                                        │
    │        datasets=[...],                                           │
    │    )                                                             │
    │    result = exp.run()                                            │
    ├─────────────────────────────────────────────────────────────────┤
    │  Layer 3: Extension Protocols                                   │
    │                                                                 │
    │    themis.core:                  themis.plugins:                 │
    │    ├── Generator                 ├── custom generators            │
    │    ├── EvaluationWorkflow        ├── custom metrics              │
    │    ├── PureMetric                ├── custom parsers              │
    │    ├── LLMMetric                 ├── custom reducers             │
    │    ├── SelectionMetric           └── custom reporters            │
    │    ├── TraceMetric                                               │
    │    ├── Parser                    Component refs:                 │
    │    ├── CandidateReducer          ├── builtin/exact_match         │
    │    └── Reporter                  ├── builtin/llm_rubric          │
    │                                  ├── openai/chat                 │
    │    Every extension exposes:      ├── catalog/mmlu_pro            │
    │    • component_id                └── lab/my_metric               │
    │    • version                                                     │
    │    • fingerprint()                                               │
    │                                                                 │
    │    Built-in Generator adapters:                                  │
    │    ├── openai(model_id)          (wraps OpenAI API)              │
    │    ├── vllm(endpoint)            (wraps vLLM server)             │
    │    └── langgraph(graph)          (wraps LangGraph graph)         │
    └─────────────────────────────────────────────────────────────────┘

    Lifecycle Hooks (replacing monolithic PipelineHook):
    ┌──────────────────────────────────────────────────────────┐
    │  BeforeGenerate ──► AfterGenerate                       │
    │  BeforeReduce   ──► AfterReduce                         │
    │  BeforeParse    ──► AfterParse                          │
    │  BeforeScore    ──► AfterScore                          │
    │  BeforeJudge    ──► AfterJudge                          │
    │  OnEvent (catch-all subscriber)                         │
    └──────────────────────────────────────────────────────────┘

    Typed Contexts (replacing mutable dicts):
    ┌────────────────────┬──────────────────────────────────────────┐
    │  GenerateContext   │  Read-only; available during generation  │
    │  ReduceContext     │  Read-only; available during reduction   │
    │  ParseContext      │  Read-only; available during parsing     │
    │  ScoreContext      │  Read-only; base context for all metrics │
    │                    │  Contains: case, parsed_output,          │
    │                    │  dataset_metadata, run_id, seed          │
    │  EvalScoreContext  │  Extends ScoreContext for LLM metrics.   │
    │  (extends Score-   │  Adds: judge_model_ref, judge_seed,     │
    │   Context)         │  eval_workflow_config. PureMetric never  │
    │                    │  sees these fields.                      │
    └────────────────────┴──────────────────────────────────────────┘
```

### Observability & Telemetry

Observability is a cross-cutting concern that must be designed into the system from the start, not retrofitted.

```
    Instrumentation Strategy:
    ┌──────────────────────────────────────────────────────────────┐
    │  • Structured logging with correlation IDs (run_id,         │
    │    case_id, candidate_id) at every stage boundary.          │
    │  • OpenTelemetry-compatible span emission at:               │
    │    - Orchestrator stage transitions                          │
    │    - Generator calls (one span per candidate)               │
    │    - WorkflowRunner eval steps (one span per judge call)    │
    │    - RunStore writes                                        │
    │  • Optional integration with langfuse, OpenTelemetry, or    │
    │    custom span collectors via a TracingProvider protocol.    │
    │  • Default: structured JSON logs to stderr. No external     │
    │    dependency required for basic observability.              │
    │  • TracingProvider is an extension protocol (component_id,  │
    │    version, fingerprint) but is NOT an identity field —     │
    │    changing the tracing backend does not change run_id.     │
    └──────────────────────────────────────────────────────────────┘

    TracingProvider Protocol:
    ┌──────────────────────────────────────────────────────────────┐
    │  class TracingProvider(Protocol):                            │
    │      def start_span(name, attributes) -> Span               │
    │      def end_span(span, status) -> None                     │
    │                                                              │
    │  Built-in: LogTracingProvider (default), LangfuseProvider,   │
    │  OTelProvider                                                │
    └──────────────────────────────────────────────────────────────┘
```

---

## Target Architecture

- Replace the current spec/compiler/orchestrator stack with a single execution model:
  - `Experiment.compile() -> RunSnapshot`
  - `Orchestrator.run(snapshot) -> RunResult`
  - `RunStore.resume(snapshot_id) -> RunResult | RunHandle`
- Remove the current public authoring and internal-IR types from the new public API:
  - `ProjectSpec`
  - `BenchmarkSpec`
  - `ExperimentSpec`
  - `TrialSpec`
  - `SliceSpec`
  - `PromptVariantSpec`
  - `ParseSpec`
  - `ScoreSpec`
  - `TraceScoreSpec`
- Introduce a smaller core model:
  - `evaluate(...)`
  - `Experiment` (composed from `GenerationConfig`, `EvaluationConfig`, `StorageConfig`)
  - `RunSnapshot`
  - `Case`
  - `Dataset`
  - `Prompt`
  - `Generator` (blackbox generation protocol)
  - `GenerationResult` (typed output from `Generator`)
  - `EvaluationWorkflow`
  - `WorkflowRunner` (evaluation execution only)
  - `CandidateReducer`
  - `Parser`
  - `Metric` (with families: `PureMetric`, `LLMMetric`, `SelectionMetric`, `TraceMetric`)
  - `EvaluationSubject` (with types: `CandidateSetSubject`, `ConversationSubject`, `TraceSubject`)
  - `RunStore`
  - `Orchestrator`
  - `Reporter`
  - `LifecycleSubscriber`
  - `TracingProvider`
- Make `RunSnapshot` the only executable artifact (Principle 1).
  - **Identity fields** — these determine `run_id` (two snapshots with identical identity fields are "the same run"). All fingerprints are computed once at compile time and frozen:
    - dataset fingerprint and revision
    - prompt config
    - parser fingerprints
    - metric fingerprints (including judge model id/version/fingerprint and evaluation workflow definitions for LLM-backed metrics)
    - generator id/version/fingerprint
    - candidate sampling policy
    - reducer configuration
    - judge fan-out policy
    - seeds
  - **Provenance fields** — recorded for reproducibility and auditing but do **not** affect `run_id`:
    - `themis` package version
    - git commit
    - Python version
    - platform
    - dependency versions for registered components
    - non-secret environment metadata
    - storage config
    - runtime config (parallelism, timeouts, retry policy)
  - `run_id` must derive from identity fields only. Provenance fields are captured but changing them does not create a new logical run.
  - **Fingerprint stability rule:** A component change after compile does not silently invalidate an in-progress run. To pick up a change, the user must recompile (creating a new `run_id`). This eliminates the fingerprint-vs-resume tension.
- Preserve projection-backed read models, but rebuild them on top of a versioned event schema (additive-only evolution, schema version stamped on every event):
  - `RunResult`
  - `BenchmarkResult`
  - `TimelineView`
  - `TraceView`

## Generation Architecture

Generation is a **blackbox** — Themis treats it as an opaque function that conforms to a typed input/output contract. Themis is an evaluation framework; it does not need to own the generation execution graph. Users can use any generation system (direct API calls, LangChain, LangGraph, vLLM, custom code) as long as it implements the `Generator` protocol.

Per Principles 2 and 4: Themis owns candidate fan-out (calling the generator N times) and the evaluation pipeline. The generator is responsible for its own internal control flow (multi-turn, tool use, agent loops).

### Generation Stack

1. **Generator** (protocol — the blackbox)
   - Takes a `Case` + `GenerateContext`, returns a `GenerationResult`.
   - The generator owns its internal implementation: single call, multi-turn, agent loop, tool use — Themis does not prescribe this.
   - Must expose `component_id`, `version`, `fingerprint()` for snapshot identity.

2. **Built-in Generator Adapters** (convenience wrappers)
   - `openai(model_id)` — wraps OpenAI/compatible API. Auto-captures trace and conversation.
   - `vllm(endpoint)` — wraps vLLM server.
   - `langgraph(graph)` — wraps a LangGraph graph, auto-captures trace steps.
   - These are reference implementations showing how to bridge external systems into the `Generator` protocol. Users can implement their own with minimal effort.

3. **CandidateSet**
   - Outer fan-out over repeated generator calls.
   - Created by planner/sample policy, not by the generator.
   - The generator produces one candidate per call; the `Orchestrator` calls it N times.

### Trace Contract for Evaluation Support

To support evaluation beyond final-output scoring (trace-level, conversation-level), the `GenerationResult` carries optional structured data:

- `final_output` (required) — the primary output for standard metrics.
- `trace: list[TraceStep]` (optional) — structured execution trace for `TraceMetric`. Each `TraceStep` has a `step_name`, `step_type`, `input`, `output`, `metadata`.
- `conversation: list[Message]` (optional) — conversation history for `ConversationSubject` evaluation.
- `artifacts: dict[str, Any]` (optional) — arbitrary additional data (tool call logs, reasoning chains, intermediate outputs).

If a user wants trace-level evaluation, their generator must populate the trace. Built-in adapters auto-capture this. For custom generators, the user bridges it — which they're already doing when using external frameworks.

### What Themis Does NOT Own in Generation

- Internal control flow (multi-turn, agent loops, tool routing)
- Model call retry logic within a generation (the generator handles this)
- Prompt rendering within complex workflows (the generator handles this)

### What Themis DOES Own

- Candidate fan-out: calling the generator N times per case
- Candidate reduction: selecting/merging across candidates
- Seeding: providing a deterministic seed per candidate call
- Persistence: storing `GenerationResult` artifacts
- Evaluation: everything after generation completes
### Candidate Reduction

Do **not** put selection/reduction logic into `Parser`. A separate stage sits between generation and parsing/scoring:

- **`CandidateReducer`**

Examples:
- Choose best of N with an LLM.
- Majority-vote collapse.
- Synthesize one final answer from several candidates.
- Rank candidates before scoring.

This enforces clean separation:
- Generation produces candidates (via blackbox `Generator`).
- Reducer derives a chosen or merged candidate view.
- Parser normalizes one chosen view.
- Metric scores it.

### Concrete Generation Interfaces

```python
class Generator(Protocol):
    """Blackbox generation — users implement this or use built-in adapters."""
    component_id: str
    version: str

    def fingerprint(self) -> str: ...

    async def generate(
        self,
        case: Case,
        ctx: GenerateContext,
    ) -> GenerationResult:
        ...
```

```python
class GenerationResult(BaseModel):
    """Typed output from a Generator. Only final_output is required."""
    candidate_id: str
    final_output: str | dict[str, Any]
    trace: list[TraceStep] | None = None           # for TraceMetric
    conversation: list[Message] | None = None       # for ConversationSubject
    artifacts: dict[str, Any] | None = None         # arbitrary extra data
    token_usage: TokenUsage | None = None
    latency_ms: float | None = None
```

```python
class TraceStep(BaseModel):
    """One step in a generation trace. Enables trace-level evaluation."""
    step_name: str
    step_type: str           # e.g., "model_call", "tool_call", "agent_loop"
    input: dict[str, Any]
    output: dict[str, Any]
    metadata: dict[str, Any] | None = None
    timestamp: datetime | None = None
```

```python
class WorkflowRunner(Protocol):
    """Executes evaluation workflows only (judge calls)."""
    def run_evaluation(
        self,
        workflow: EvaluationWorkflow,
        subject: EvaluationSubject,
        ctx: EvalScoreContext,
    ) -> EvaluationExecution:
        ...
```

```python
class CandidateReducer(Protocol):
    component_id: str
    version: str

    def fingerprint(self) -> str: ...
    def reduce(
        self,
        candidates: list[GenerationResult],
        ctx: ReduceContext,
    ) -> ReducedCandidate:
        ...
```

### Generation Fan-Out Rules

Candidate fan-out is an evaluation concern managed by the `Orchestrator`:
- Multiple samples
- Self-consistency candidate generation
- Best-of-n candidate pool
- pass@k candidate pool

The `Generator` produces one candidate per call. The `Orchestrator` calls it N times with different seeds.

### Planning Implications

Planner plans along two axes:

- **Candidate multiplicity**: how many independent candidates per case (`num_samples`, self-consistency sampling, pass@k fan-out stay in planning/sampling policy).
- **Judge fan-out**: how many judge calls per scored item (rubric dimensions, panel of judges, repeated calls).

The generator's internal complexity is opaque to the planner. This is much easier to estimate and resume.

### Generation Architecture Rules

See Principles 2, 4, 5. In summary: generation is a blackbox conforming to a typed contract, fan-out at the candidate level is owned by Themis, evaluation is fully owned by Themis, and strict stage separation (generation -> reduction -> parsing -> scoring, with judging in evaluation).

## Evaluation Workflow And Judge Metrics Architecture

Judge metrics are a **first-class evaluation pattern**, not a special-case service and not a separate runtime stack. They use the shared `WorkflowRunner` but remain semantically in the **evaluation layer**.

Per Principles 2, 5, and 6: one shared runtime (no parallel judge engine/runner/store), judge calls stay in the evaluation layer, no mutable context dicts or hidden service injection, no candidate selection in parsers, no synthetic task/trial objects.

### Metric Families

Metrics are classified into explicit families:

- **`PureMetric`** — no model calls. Deterministic scoring from parsed output and reference data (e.g., exact match, F1, BLEU).
- **`LLMMetric`** — one or more LM judge calls per scored item. Compiles to an `EvaluationWorkflow` (e.g., LLM-as-judge grading, rubric scoring, faithfulness evaluation).
- **`SelectionMetric`** — ranking or choosing across candidate sets, possibly LM-backed. Operates on `CandidateSetSubject` with cardinality constraints (e.g., pairwise comparison when `size=2`, tournament/ranking when `size>2`, best-of-n selection by judge).
- **`TraceMetric`** — trace or workflow evaluation, possibly LM-backed. Operates on `TraceSubject` or `ConversationSubject` (e.g., agent trajectory scoring, tool-use quality assessment).

This is clearer than a generic `JudgeService` abstraction. Not all metrics require an LM judge; the family system makes this explicit.

```
    Metric Family Decision Tree:

    Is the metric deterministic (no model calls)?
    ├── YES ──► PureMetric
    │           score(parsed, case, ctx) -> Score | ScoreError
    │
    └── NO (requires LM judge)
        │
        What does it score?
        ├── Single candidate output ──► LLMMetric
        │   build_workflow(CandidateSetSubject[size=1]) -> EvaluationWorkflow
        │
        ├── Candidate pair/set ──► SelectionMetric
        │   build_workflow(CandidateSetSubject[size≥2]) -> EvaluationWorkflow
        │
        └── Execution trace/conversation ──► TraceMetric
            build_workflow(TraceSubject | ConversationSubject) -> EvaluationWorkflow
```

**Speculative-complexity warning (Principle 9).** The full metric x subject matrix (4 metric families x 3 subject types = 12 cells) is not all equally justified. Ship only the combinations with concrete, demonstrated use cases in the initial release:

| | CandidateSetSubject | ConversationSubject | TraceSubject |
|---|---|---|---|
| **PureMetric** | **Ship** (size=1) | — | — |
| **LLMMetric** | **Ship** (size=1) | Defer | Defer |
| **SelectionMetric** | **Ship** (size≥2) | — | — |
| **TraceMetric** | — | **Ship** | **Ship** |

Deferred cells are designed in the protocol but not implemented or tested until a concrete need arises. The protocol surface should make later addition non-breaking.

### Typed Evaluation Subjects

Judge metrics operate on explicit subject types, not ad hoc bags. The hierarchy is consolidated to three types:

- **`CandidateSetSubject`** — one or more candidate outputs, with cardinality constraints. Covers:
  - Single-answer grading (size=1, replaces the old `CandidateSubject`)
  - Pairwise comparison (size=2, replaces the old `CandidatePairSubject`)
  - Best-of-n selection and ranking (size>2)
  - Cardinality is validated at compile time: `LLMMetric` requires `size=1`, `SelectionMetric` requires `size≥2`.
- **`ConversationSubject`** — a full conversation trace (from `GenerationResult.conversation`), for dialogue quality judging.
- **`TraceSubject`** — an execution trace (from `GenerationResult.trace`), for agent/tool-use trajectory judging.

The subject type hierarchy is intentionally kept flat and minimal. Do not add new subject types without a concrete metric that requires them. The consolidation of candidate subjects into one type with cardinality constraints eliminates three types that were slight variations on "some candidate outputs."

### Evaluation Workflows

`LLMMetric`, `SelectionMetric`, and `TraceMetric` compile to `EvaluationWorkflow` instances. Unlike generation (which is a blackbox), evaluation workflows are **owned by Themis** — they define the judge execution pipeline with explicit steps for prompt rendering, model calls, response parsing, and score aggregation.

```python
class EvaluationWorkflow(Protocol):
    """Declares the judge execution graph. Workflows are data objects (Pydantic models)
    with a list of steps, not opaque callables."""
    workflow_id: str
    version: str

    def fingerprint(self) -> str: ...
    def steps(self) -> list[EvalStep]: ...
```

```python
class EvalStep(BaseModel):
    """One step in an evaluation workflow (prompt render, model call, parse, aggregate)."""
    step_type: str                    # "render_prompt", "model_call", "parse_response", "aggregate"
    config: dict[str, Any]
```

```python
class EvaluationExecution(BaseModel):
    execution_id: str
    subject_ref: SubjectRef
    rendered_prompts: list[RenderedJudgePrompt]
    judge_responses: list[JudgeResponse]
    parsed_judgments: list[ParsedJudgment]
    scores: list[Score]
    aggregation_output: AggregationResult | None
    trace: WorkflowTrace
```

```python
class JudgeResponse(BaseModel):
    judge_model_id: str
    judge_model_version: str
    judge_model_fingerprint: str
    effective_seed: int | None
    raw_response: str
    token_usage: TokenUsage
    latency_ms: float
    provider_request_id: str | None
    retry_history: list[RetryRecord]
    conversation_trace: ConversationTrace | None
```

The `WorkflowRunner` executes `EvaluationWorkflow` instances with full support for model calls, retries, seeding, event persistence, tracing, and resume/replay.

```
    LLMMetric Execution Flow:

    LLMMetric.build_workflow(subject, ctx)
         │
         ▼
    EvaluationWorkflow
         │
         ▼
    WorkflowRunner.run_evaluation()
         │
         ├──► Render judge prompt(s)
         │         │
         │         ▼
         ├──► Judge model call ◄── judge model adapter
         │         │
         │         ▼
         ├──► Parse judge response
         │         │
         │         ▼
         ├──► Compute scores per rubric dimension
         │         │
         │         ▼
         └──► Aggregate (if multi-dimension/multi-judge)
                   │
                   ▼
         EvaluationExecution
         ├── rendered_prompts[]
         ├── judge_responses[]
         ├── parsed_judgments[]
         ├── scores[]
         ├── aggregation_output
         └── trace
```

### Judge Fan-Out Rules

Judge fan-out is evaluation-level, strictly separate from candidate fan-out:

- Rubric dimensions (multiple rubric criteria per scored item)
- Panel of judges (multiple judge models per scored item)
- Repeated judge calls (for judge agreement / consistency)
- Pairwise tournaments (all-pairs or bracket-style comparisons)
- Majority-vote judging
- Candidate-set ranking or selection

The same sample may have many candidates and many judge calls, but they belong to different layers.

### Judge Artifact Persistence

Judge evaluation must be fully replayable from storage. Persist by default:

- Rendered judge prompt
- Judge model id/version/fingerprint
- Effective seed
- Raw judge response
- Parsed judge response (structured judgment)
- Per-call scores or rubric outputs
- Aggregation logic output
- Retry history
- Token usage, latency, provider request ids
- Judge trace if multi-turn or tool-using

A reported score must be inspectable without re-running the judge.

### Naming Rules

Do **not** use `Judge*` classes as primary architecture names. Prefer:

- `EvaluationWorkflow`
- `EvaluationSubject`
- `EvaluationExecution`
- `LLMMetric`
- `SelectionMetric`
- `TraceMetric`
- `WorkflowRunner` (shared)

Use `judge` in user-facing config where it is meaningful (e.g., `judge_model`, `judge_prompt`), but avoid a duplicate runtime subsystem.

### Error Contracts

All metric protocols have explicit error behavior:

- **`PureMetric.score()`** returns `Score | ScoreError`. It never raises — failures are captured as `ScoreError` with a reason string. The `Orchestrator` records `ScoreError` as a failed score event and continues with other work items.
- **`LLMMetric.build_workflow()` / `SelectionMetric.build_workflow()` / `TraceMetric.build_workflow()`** may raise `WorkflowBuildError` if the subject is invalid or the workflow cannot be constructed. This is a compile-time error, not a runtime error. Runtime failures during workflow execution (judge timeout, parse failure) are handled by the `WorkflowRunner` and recorded as `StepError` events.

### Concrete Metric Interfaces

```python
class PureMetric(Protocol):
    component_id: str
    version: str

    def fingerprint(self) -> str: ...
    def score(self, parsed: ParsedOutput, case: Case, ctx: ScoreContext) -> Score | ScoreError: ...
```

```python
class LLMMetric(Protocol):
    component_id: str
    version: str

    def fingerprint(self) -> str: ...
    def build_workflow(self, subject: CandidateSetSubject, ctx: EvalScoreContext) -> EvaluationWorkflow: ...
    # subject must have size=1; validated at compile time
```

```python
class SelectionMetric(Protocol):
    component_id: str
    version: str

    def fingerprint(self) -> str: ...
    def build_workflow(self, subject: CandidateSetSubject, ctx: EvalScoreContext) -> EvaluationWorkflow: ...
    # subject must have size>=2; validated at compile time
```

```python
class TraceMetric(Protocol):
    component_id: str
    version: str

    def fingerprint(self) -> str: ...
    def build_workflow(self, subject: TraceSubject | ConversationSubject, ctx: EvalScoreContext) -> EvaluationWorkflow: ...
```

## Public API And Extension Model

- Layer 1 API:
  - `evaluate(model=..., data=..., metric=...)`
  - Must support a minimal eval in 5 lines or fewer.
  - Must work with both pure metrics and LLM-backed metrics transparently.
  - Internally constructs an `Experiment`, compiles a `RunSnapshot`, and runs it. No workflow graph is exposed.
- Layer 2 API:
  - `Experiment(generation=GenerationConfig(...), evaluation=EvaluationConfig(...), storage=StorageConfig(...), datasets=[...])` for configurable experiments.
  - `GenerationConfig` groups: generator, candidate sampling policy, reducer.
  - `EvaluationConfig` groups: metrics, parsers, judge configuration, evaluation workflow overrides.
  - `StorageConfig` groups: store backend, resume config, blob storage.
  - This composed structure prevents `Experiment` from becoming a flat bag of 15+ parameters.
- Layer 3 API:
  - typed extension protocols under `themis.core` and `themis.plugins`.
- Standardize naming:
  - `Parser` replaces `Extractor`
  - `Metric` and `Score` replace mixed `evaluation/score/trace_score` vocabulary
  - `PureMetric`, `LLMMetric`, `SelectionMetric`, `TraceMetric` replace the generic `JudgeService` pattern
  - `EvaluationWorkflow` and `EvaluationSubject` replace ad hoc judge call orchestration
  - `Dataset` and `Case` replace `DatasetProvider` vs `DatasetLoader`
  - `Generator` replaces `ModelAdapter` at the public API boundary (built-in adapters like `openai()` implement `Generator`)
  - `CandidateReducer` is a new stage with no v2 equivalent
  - `Orchestrator` replaces `Runner` to avoid confusion with `WorkflowRunner`
- Replace flat string plugin ids with namespaced component refs:
  - `builtin/exact_match`
  - `builtin/llm_rubric`
  - `builtin/pairwise_judge`
  - `openai/chat`
  - `catalog/mmlu_pro`
  - `lab/my_metric`
- Require every extension point to expose:
  - `component_id`
  - `version`
  - `fingerprint()`
- Replace the monolithic `PipelineHook` contract with stage-scoped optional protocols plus an event subscriber:
  - `BeforeGenerate`
  - `AfterGenerate`
  - `BeforeReduce`
  - `AfterReduce`
  - `BeforeParse`
  - `AfterParse`
  - `BeforeScore`
  - `AfterScore`
  - `BeforeJudge`
  - `AfterJudge`
  - `OnEvent`
- Replace raw context dicts with typed read-only contexts:
  - `GenerateContext` — case, prompt config, seed, dataset metadata
  - `ReduceContext` — candidate set metadata, reduction policy
  - `ParseContext` — parser config, case reference
  - `ScoreContext` — base context for all metrics: case, parsed_output, dataset_metadata, run_id, seed
  - `EvalScoreContext` (extends `ScoreContext`) — adds judge_model_ref, judge_seed, eval_workflow_config. Only used by LLM-backed metrics (`LLMMetric`, `SelectionMetric`, `TraceMetric`). `PureMetric` never sees these fields.

Remove the separate `JudgeContext`; judge-specific data is carried by `EvalScoreContext` and the `EvaluationWorkflow`/`EvaluationExecution`, not injected via mutable context dicts. The `ScoreContext`/`EvalScoreContext` split ensures pure metrics have a narrow interface without tramp data from judge configuration.

## Execution, Lifecycle, And State

- Preserve all current capabilities:
  - generation (now via workflow runner and candidate fan-out)
  - candidate reduction (new stage)
  - parsing / output normalization
  - evaluation (now including first-class LLM-backed judging via evaluation workflows)
  - trace scoring (now via `TraceMetric` over typed subjects)
  - report generation
  - external handoff/import
  - resume / crash recovery
  - progress tracking
  - estimate / planning
- Split responsibilities cleanly:
  - `Planner` computes candidate multiplicity, judge fan-out, and work items from `RunSnapshot`
  - `Generator` (blackbox) produces one candidate per call
  - `WorkflowRunner` executes `EvaluationWorkflow` instances (judge calls only)
  - `Orchestrator` orchestrates planned work items across all stages, owns stage-level event writes
  - `Importer` validates and imports external generation/evaluation results
  - `RunStore` persists events, manifests, and projections
  - `Reporter` builds outputs from read models
- Persist raw response artifacts by default:
  - raw text
  - structured output
  - reasoning trace
  - token usage
  - finish reason
  - provider request id
  - conversation/tool traces
  - `GenerationResult` fields (final_output, trace, conversation, artifacts) per candidate
  - rendered judge prompts, raw judge responses, parsed judgments, per-call scores, aggregation outputs, retry history, and judge traces from `EvaluationExecution`
- Support replay from checkpoints:
  - re-reduce without re-generation
  - re-parse without re-generation
  - re-score without re-generation (including re-judge without re-generation)
  - re-run trace metrics without re-generation
  - rebuild reports from stored events and projections
  - inspect any judge score without re-running the judge
- Preserve execution modes:
  - local
  - worker-pool
  - batch
  - generation export/import
  - evaluation export/import

```
    Replay Capabilities (Principle 7):

    Stored Artifacts Allow Re-entry At Any Stage:

    ┌───────────┐   ┌───────────┐   ┌───────────┐   ┌───────────┐
    │ Generation│──►│ Reduction │──►│  Parsing  │──►│  Scoring  │
    │           │   │           │   │           │   │           │
    │  stored   │   │  stored   │   │  stored   │   │  stored   │
    │  ✓        │   │  ✓        │   │  ✓        │   │  ✓        │
    └───────────┘   └───────────┘   └───────────┘   └───────────┘
         ▲               ▲               ▲               ▲
         │               │               │               │
    Full re-run    Re-reduce only   Re-parse only   Re-score only
                   (skip generation) (skip gen+red)  (skip gen+red+parse)

    Each stage's outputs are persisted independently.
    Any stage can be re-entered without re-running upstream stages.
```

## Concurrency And Resource Management

The system involves massive fan-out: cases x candidates x judge calls x rubric dimensions. Without an explicit concurrency model this becomes either sequentially slow or uncontrollably parallel.

### Async Model

- The `WorkflowRunner`, `Orchestrator`, `Generator`, and all I/O-bound operations (model calls, store writes) are `async`-native, built on `asyncio`.
- Sync wrappers are provided for Layer 1 (`evaluate(...)`) and Layer 2 (`Experiment.run()`) so users without async code can call them directly. Internally these enter an event loop.
- CPU-bound operations (pure metrics, parsing) may run in a thread pool executor to avoid blocking the event loop.

### Concurrency Control

Fan-out is bounded at three levels:

1. **Per-provider rate limiting.** Each `Generator` (or judge model adapter) declares (or auto-detects) its rate limits (requests/min, tokens/min). The orchestrator enforces these via a token-bucket or leaky-bucket limiter shared across all tasks targeting the same provider endpoint. This applies to both generation and judge model calls.
2. **Global concurrency cap.** `Orchestrator` enforces a configurable maximum number of concurrent tasks (`max_concurrent_tasks`). This bounds memory usage regardless of fan-out depth.
3. **Per-stage concurrency.** Each stage (generation, reduction, parsing, evaluation) can have an independent concurrency limit so that, e.g., evaluation does not starve generation of model call slots when sharing a provider.

### Sensible Defaults

The concurrency model works without configuration. The defaults are:

- **Per-provider rate limiting:** auto-detect from API response headers (e.g., `x-ratelimit-*`). If unavailable, default to 60 requests/min.
- **Global concurrency cap:** 32 concurrent tasks.
- **Per-stage concurrency:** proportional split of the global cap (no explicit config needed).

Users only touch these knobs when they have a reason to (e.g., hitting provider limits, running on constrained hardware, or needing to prioritize generation over evaluation).

### Backpressure

- When the global or per-provider limit is reached, new work items are queued, not dropped. The planner produces work items lazily (iterator/async generator) so that unbounded fan-out does not materialize all work items in memory at once.
- Store write failures trigger backpressure: the orchestrator pauses new task starts until the store is healthy, rather than buffering unboundedly.

### Memory Management

- `GenerationResult` and `EvaluationExecution` artifacts are streamed to the `RunStore` as they complete. The orchestrator does not hold all completed executions in memory simultaneously.
- Large artifacts (raw model responses, conversation traces) are stored as blob references, not inline in event payloads.
- The planner's work-item iterator ensures that for a run with N cases x M candidates x K judges, memory usage is O(concurrency_cap), not O(N x M x K).

### Failure And Retry

- Transient failures (provider timeout, rate limit 429) are retried per the workflow's retry policy with exponential backoff and jitter.
- Persistent failures (malformed response, parse error) are recorded as `StepError` and do not block other work items.
- Partial fan-out failure (e.g., 3 of 5 rubric dimensions succeed) is recorded faithfully; the run completes with `PARTIAL_FAILURE` status rather than aborting.

## Storage, Reproducibility, And Provenance

- Replace construction-time storage side effects with explicit initialization:
  - `sqlite_store(path).initialize()`
  - `postgres_store(url, blob_root).initialize()`
- Implement true backend separation behind a single `RunStore` interface.
- Version the storage schema and event schema independently from package version. Event schema evolution is additive-only: new event types may be added, existing types may gain optional fields, but fields are never removed or changed. Schema version is stamped on every event. Readers skip unknown event types/fields gracefully (forward compatibility). Breaking changes require a new major schema version and a migration tool.
- Capture provenance automatically for every run:
  - `themis` version
  - git commit
  - Python version
  - platform
  - dependency versions for registered components
  - dataset revisions and fingerprints
  - provider/model endpoint metadata
  - generator fingerprint and version
  - evaluation workflow fingerprints and versions (including judge model id/version/fingerprint)
  - candidate sampling policy configuration
  - judge fan-out policy configuration
  - non-secret runtime config
- Persist secrets as references only, never values.
- Replace the current config report with a snapshot report generated from `RunSnapshot` plus persisted run metadata.
- Preserve export capabilities:
  - JSON
  - Markdown
  - CSV
  - LaTeX
  - quick machine-readable score tables
- Preserve quick inspection workflows with a rebuilt `quickcheck` on the new store schema.

## Configuration Loading

v4 supports two modes of experiment definition:

1. **Python-first (primary).** `Experiment(...)` is the primary API. This is the most expressive and type-safe way to define experiments. All examples and documentation lead with Python.

2. **Declarative config files (optional, Phase 5).** Support loading experiments from YAML/TOML files via `Experiment.from_config("experiment.yaml")`. Config files map to the same `GenerationConfig`/`EvaluationConfig`/`StorageConfig` composition. This enables:
   - Non-Python users running experiments via CLI
   - Version-controlled experiment definitions
   - CI/CD integration without Python scripts

   Config file support is a convenience layer, not a separate system. The config loader validates, resolves component refs (e.g., `builtin/exact_match`), and produces a standard `Experiment` object. Custom Python components (custom generators, custom metrics) are referenced by importable module paths in the config file.

**Decision:** Python-first. Config files are a Phase 5 addition, not a blocking requirement for the core system. Do not design the core around config file constraints.

---

## Feature Coverage Requirements

- Preserve and redesign these workflows:
  - quick-eval inline
  - quick-eval file
  - quick-eval huggingface
  - quick-eval benchmark
  - init/scaffold
  - report generation
  - quickcheck
  - estimate
  - submit/resume
  - compare/export
  - snapshot reporting
- Preserve all extension categories:
  - custom generators (any system implementing the `Generator` protocol)
  - custom evaluation workflows
  - custom candidate reducers
  - custom parsers
  - custom pure metrics
  - custom LLM-backed metrics (with custom judge prompts, judge models, rubrics)
  - custom selection metrics (ranking, pairwise, tournament)
  - custom trace metrics (agent trajectory, conversation quality)
  - dataset integrations
  - custom prompts
  - custom reports
  - telemetry/observability
  - tool-using and MCP-enabled evaluations
- Preserve catalog functionality, but behind a cleaner abstraction:
  - `catalog.load("mmlu_pro")`
  - `catalog.run("mmlu_pro", model=..., store=...)`
- Preserve benchmark-native analysis behavior on top of generic result models.
- Preserve statistics as an optional extra behind a narrow `StatsEngine` boundary.

## V2 To V4 Feature Mapping

Every v2 feature must have a v4 equivalent or an explicit intentional removal (acceptance criterion). This table tracks the mapping.

| v2 Feature / Concept | v4 Equivalent | Status |
|---|---|---|
| `ProjectSpec` / `BenchmarkSpec` / `ExperimentSpec` | `Experiment` + `RunSnapshot` | Replaced |
| `TrialSpec` / `SliceSpec` | `Case` + `Dataset` + planner work items | Replaced |
| `PromptVariantSpec` | `Prompt` | Replaced |
| `ParseSpec` | `Parser` | Renamed |
| `ScoreSpec` / `TraceScoreSpec` | `Metric` families (`PureMetric`, `LLMMetric`, `SelectionMetric`, `TraceMetric`) | Replaced |
| `DatasetProvider` / `DatasetLoader` | `Dataset` | Merged |
| `Extractor` | `Parser` | Renamed |
| `PipelineHook` | Stage-scoped lifecycle protocols + `LifecycleSubscriber` | Replaced |
| Raw provider strings | `Generator` with namespaced component ref | Replaced |
| `JudgeService` / `JudgeContext` | `LLMMetric` + `EvaluationWorkflow` + `EvalScoreContext` | Replaced |
| Flat string plugin ids | Namespaced component refs (`builtin/exact_match`) | Replaced |
| Mutable context dicts | Typed read-only contexts (`GenerateContext`, `ScoreContext`, `EvalScoreContext`, etc.) | Replaced |
| Construction-time storage init | Explicit `store.initialize()` | Replaced |
| Config report | Snapshot report from `RunSnapshot` | Replaced |
| `ModelAdapter` (generation) | `Generator` protocol (blackbox with typed I/O contract) | Replaced |
| Quick-eval (inline, file, HF, benchmark) | `evaluate(...)` / `Experiment(...)` thin wrappers | Preserved |
| Init/scaffold | CLI rebuild on Layer 1/2 | Preserved |
| Submit/resume | `RunStore.resume(snapshot_id)` | Preserved |
| Compare/export | Rebuilt on new read models | Preserved |
| Quickcheck | Rebuilt on new store schema | Preserved |
| Estimate/planning | `Planner` with multi-axis planning | Preserved |
| Report generation (JSON, MD, CSV, LaTeX) | `Reporter` with same export formats | Preserved |
| Catalog benchmarks | `themis.catalog` with declarative manifests | Preserved |
| Statistics extras | `StatsEngine` behind narrow boundary | Preserved |
| _(no v2 equivalent)_ | `Generator` protocol + built-in adapters (openai, vllm, langgraph) | New |
| _(no v2 equivalent)_ | `CandidateReducer` | New |
| _(no v2 equivalent)_ | `EvaluationWorkflow` + typed `EvaluationSubject` (3 types) | New |
| _(no v2 equivalent)_ | Concurrency control + backpressure | New |
| _(no v2 equivalent)_ | `Orchestrator` (replaces ad hoc orchestration) | New |
| _(no v2 equivalent)_ | `TracingProvider` + observability hooks | New |
| _(no v2 equivalent)_ | Declarative config file loading (Phase 5) | New |

**Intentional removals:** Internal IR types (`ProjectSpec`, `BenchmarkSpec`, `ExperimentSpec`, `TrialSpec`, `SliceSpec`, `PromptVariantSpec`, `ParseSpec`, `ScoreSpec`, `TraceScoreSpec`) are removed from the public API. Their functionality is subsumed by the new core model.

---

## Detailed Implementation Plan

The implementation is organized into six phases with explicit entry/exit criteria. Phases are sequential — each phase's exit criteria must be met before the next begins. Within each phase, tasks are ordered by dependency.

The blackbox generation model significantly simplifies Phases 1-2 compared to the previous workflow-engine approach.

```
    Phase Timeline:

    Phase 1          Phase 2          Phase 3          Phase 4        Phase 5       Phase 6
    Foundation       Execution        Evaluation &     Read Models,   CLI &         Hardening
                     Engine           Metrics          Reporting &    Integration   & Release
                                                       Catalog
    ┌────────────┐   ┌────────────┐   ┌────────────┐   ┌──────────┐   ┌──────────┐  ┌──────────┐
    │ Core types │   │ Planner    │   │ Metric     │   │ Result   │   │ CLI      │  │ Perf     │
    │ Protocols  │   │ Orchestratr│   │  families  │   │  facades │   │  rebuild │  │ Security │
    │ RunSnapshot│   │ Generator  │   │ Eval WF    │   │ Reporter │   │ quick-   │  │ Docs     │
    │ Events     │   │  adapters  │   │  WF Runner │   │ Catalog  │   │  eval    │  │ Release  │
    │ Store iface│   │ Candidate  │   │ Judge      │   │ Built-ins│   │ Export   │  │          │
    │ Observ.    │   │  fan-out   │   │  fan-out   │   │ Stats    │   │ Config   │  │          │
    └────────────┘   │ Reducer    │   │ Subjects   │   │          │   │  files   │  │          │
                     │ Resume     │   │            │   │          │   │          │  │          │
                     │ Concurrency│   │            │   │          │   │          │  │          │
                     └────────────┘   └────────────┘   └──────────┘   └──────────┘  └──────────┘

    Dependencies:
    Phase 1 ──► Phase 2 ──► Phase 3 ──► Phase 4 ──► Phase 5 ──► Phase 6
                                    ╲              ╱
                                     ╲            ╱
                                      ► Phase 4 ──
```

### Phase 1: Foundation (Weeks 1-2)

**Goal:** Establish all core types, protocols, and storage interface so that subsequent phases build on stable foundations.

**Entry criteria:** Approved migration plan. Clean `v4` branch or repo.

#### 1.1 Core Domain Models (Week 1)

- Create `themis.core` package structure.
- Define immutable domain models using Pydantic `BaseModel` (frozen):
  - `Case`, `Dataset`
  - `GenerationResult`, `TraceStep`, `Message`
  - `ParsedOutput`, `Score`, `ScoreError`
  - `ReducedCandidate`
  - `WorkflowTrace`, `ConversationTrace`
- Define `RunEvent` base type and initial event variants (with schema version stamp).
- Define typed context objects:
  - `GenerateContext`, `ReduceContext`, `ParseContext`
  - `ScoreContext` (base: case, parsed_output, dataset_metadata, run_id, seed)
  - `EvalScoreContext` (extends `ScoreContext`: adds judge_model_ref, judge_seed, eval_workflow_config)
- Define composed config objects: `GenerationConfig`, `EvaluationConfig`, `StorageConfig`.
- **Tests:** Unit tests for model immutability, serialization round-trip, fingerprint stability.

#### 1.2 Extension Protocols (Week 1)

- Define core protocols with `component_id`, `version`, `fingerprint()`:
  - `Generator` — `generate(case, ctx) -> GenerationResult`
  - `Parser`
  - `CandidateReducer` — `reduce(candidates, ctx) -> ReducedCandidate`
  - `EvaluationWorkflow` — `steps() -> list[EvalStep]`
- Define metric family protocols:
  - `PureMetric` — `score(parsed, case, ctx) -> Score | ScoreError`
  - `LLMMetric` — `build_workflow(subject, ctx) -> EvaluationWorkflow`
  - `SelectionMetric` — `build_workflow(subject, ctx) -> EvaluationWorkflow`
  - `TraceMetric` — `build_workflow(subject, ctx) -> EvaluationWorkflow`
- Define evaluation subject types (3 types):
  - `CandidateSetSubject` (with cardinality constraints)
  - `ConversationSubject`, `TraceSubject`
- Define `EvaluationExecution`, `JudgeResponse`, `RenderedJudgePrompt`, `ParsedJudgment`, `AggregationResult`.
- Define lifecycle subscriber protocols: `BeforeGenerate`, `AfterGenerate`, `BeforeReduce`, `AfterReduce`, `BeforeParse`, `AfterParse`, `BeforeScore`, `AfterScore`, `BeforeJudge`, `AfterJudge`, `OnEvent`.
- Define `TracingProvider` protocol for observability.
- **Tests:** Protocol compliance tests with dummy implementations. Verify fingerprint determinism. Verify `CandidateSetSubject` cardinality validation.

#### 1.3 RunSnapshot (Week 1-2)

- Define `RunSnapshot` with explicit identity/provenance field separation.
- Implement `run_id` derivation from identity fields only (content-addressable hash).
- Implement fingerprint computation and caching at compile time.
- Implement `RunSnapshot` serialization/deserialization (JSON).
- Implement `Experiment.compile() -> RunSnapshot` with composed config objects.
- **Tests:** Golden tests for snapshot serialization stability. Verify that provenance changes do not alter `run_id`. Verify that identity changes do alter `run_id`. Verify fingerprints are frozen at compile time.

#### 1.4 Storage Interface (Week 2)

- Define `RunStore` protocol: `initialize()`, `persist_event()`, `query_events()`, `get_projection()`, `store_blob()`, `resume()`.
- Define versioned event schema (schema version independent of package version, additive-only evolution, forward-compatible readers).
- Implement in-memory `RunStore` for testing.
- Implement `sqlite_store(path)` with explicit `.initialize()`.
- **Tests:** Store contract test suite (runs against both in-memory and SQLite). Event round-trip. Blob storage. Schema version validation. Unknown event type skip test.

**Phase 1 exit criteria:**
- All core types serialize/deserialize cleanly.
- `RunSnapshot` identity/provenance separation is tested with golden tests.
- `Generator`, `EvaluationWorkflow`, and all metric protocols have compliance tests.
- SQLite store passes the full contract test suite.
- A dummy `Experiment` can compile to a `RunSnapshot` and persist it.

---

### Phase 2: Execution Engine (Weeks 3-4)

**Goal:** Build the runtime that can call generators, manage candidate fan-out, perform reduction, and persist results — with concurrency control and resume.

**Entry criteria:** Phase 1 exit criteria met.

#### 2.1 Built-in Generator Adapters (Week 3)

- Implement `openai(model_id)` generator adapter:
  - Wraps OpenAI API into `Generator` protocol.
  - Auto-captures conversation trace and token usage in `GenerationResult`.
  - Exposes `component_id`, `version`, `fingerprint()`.
- Implement `vllm(endpoint)` generator adapter.
- Implement `langgraph(graph)` generator adapter (wraps LangGraph, auto-captures trace steps).
- **Tests:** Each adapter produces valid `GenerationResult`. Fingerprints are deterministic. Trace data is captured correctly.

#### 2.2 Concurrency & Rate Limiting (Week 3)

- Implement per-provider token-bucket rate limiter (auto-detect from response headers, default 60 req/min).
- Implement global concurrency semaphore (`max_concurrent_tasks`, default 32).
- Implement per-stage concurrency limits (proportional split default).
- Implement backpressure: queue when limits reached, pause on store failure.
- Implement lazy work-item generation (async generator from `Planner`).
- **Tests:** Rate limiter respects limits under concurrent load. Global cap bounds memory. Backpressure pauses correctly on simulated store failure.

#### 2.3 Planner & Candidate Fan-Out (Week 3-4)

- Implement `Planner` that reads `RunSnapshot` and produces a lazy stream of work items.
- Implement candidate multiplicity planning: `num_samples`, self-consistency, pass@k, best-of-n.
- Implement deterministic candidate-level seeding.
- **Tests:** Planner produces correct work item count. Seeds are deterministic. Work items are lazy (memory test).

#### 2.4 Orchestrator (Week 4)

- Implement `Orchestrator` that consumes planner work items and dispatches to `Generator`.
- Implement stage pipeline: generation -> reduction -> parsing -> scoring.
- Implement `CandidateReducer` invocation between generation and parsing.
- Implement lifecycle hook dispatch: `BeforeGenerate`, `AfterGenerate`, `BeforeReduce`, `AfterReduce`, `BeforeParse`, `AfterParse`.
- Implement progress tracking and status reporting.
- Implement stage-level and run-level event writes to `RunStore`.
- Implement observability span emission via `TracingProvider`.
- **Tests:** End-to-end: dataset -> generation -> reduction -> parsing with mock components. Lifecycle hooks fire in correct order. Events written to store at correct granularity.

#### 2.5 Resume & Crash Recovery (Week 4)

- Implement `RunStore.resume(snapshot_id)` that reconstructs execution state from persisted events.
- Implement stage-level resume: skip completed work items, restart failed/pending ones.
- Implement `PARTIAL_FAILURE` terminal status.
- **Tests:** Kill mid-run, resume, verify completion. Re-score without re-generation. Re-reduce without re-generation.

#### 2.6 Importer (Week 4)

- Implement `Importer` for external generation results (validate `GenerationResult` schema, persist events).
- Implement generation bundle export format.
- **Tests:** Export a generation bundle, import it, verify events match.

**Phase 2 exit criteria:**
- A full generation pipeline runs end-to-end: dataset -> generator call -> candidate fan-out -> reduction -> parsing -> `PureMetric` scoring.
- Built-in generator adapters (openai, vllm, langgraph) work correctly.
- Concurrency controls are enforced (rate limiting, global cap, sensible defaults).
- Resume works after simulated crash.
- Generation export/import round-trips cleanly.

---

### Phase 3: Evaluation & Metrics (Weeks 5-6)

**Goal:** Add LLM-backed evaluation, judge fan-out, and all metric families via the `WorkflowRunner`.

**Entry criteria:** Phase 2 exit criteria met.

#### 3.1 WorkflowRunner for Evaluation (Week 5)

- Implement `WorkflowRunner.run_evaluation()` for executing `EvaluationWorkflow` instances.
- Implement `EvaluationExecution` assembly: rendered prompts, judge responses, parsed judgments, scores, aggregation, trace.
- Implement `JudgeResponse` capture: model id/version/fingerprint, seed, raw response, token usage, latency, retry history.
- Implement retry, seeding, event persistence (step-level events), and tracing.
- **Tests:** Single LLM judge call end-to-end. Verify all `EvaluationExecution` fields populated. Verify judge artifacts persisted to store. Verify step-level events written by `WorkflowRunner` (not `Orchestrator`).

#### 3.2 Metric Family Integration (Week 5)

- Integrate `PureMetric.score()` into the orchestrator pipeline (already partially done in Phase 2). Verify `ScoreError` handling.
- Integrate `LLMMetric.build_workflow()` -> `WorkflowRunner.run_evaluation()` pipeline with `EvalScoreContext`.
- Integrate `SelectionMetric.build_workflow()` for `CandidateSetSubject` (size≥2).
- Integrate `TraceMetric.build_workflow()` for `TraceSubject` and `ConversationSubject`.
- Implement mixed metric runs (PureMetric + LLMMetric on the same experiment).
- Add `BeforeScore`, `AfterScore`, `BeforeJudge`, `AfterJudge` lifecycle hooks.
- **Tests:** Each metric family executes correctly. Mixed metric run produces both pure and LLM-backed scores. Lifecycle hooks fire for judge calls. `CandidateSetSubject` cardinality enforced.

#### 3.3 Judge Fan-Out (Week 6)

- Implement rubric-dimension fan-out (multiple criteria per scored item).
- Implement panel-of-judges fan-out (multiple judge models per scored item).
- Implement repeated judge calls (consistency/agreement).
- Implement pairwise tournament for `SelectionMetric` (CandidateSetSubject size=2).
- Implement majority-vote aggregation.
- Implement judge fan-out planning in `Planner`.
- **Tests:** Multi-dimension rubric produces per-dimension scores. Panel of judges with majority vote. Pairwise tournament completes. Judge fan-out respects concurrency limits.

#### 3.4 Evaluation Import/Export & Re-Judge (Week 6)

- Implement evaluation bundle export/import.
- Implement re-judge without re-generation (re-run evaluation workflow from stored `GenerationResult` outputs).
- Implement judge score inspection from storage without re-running the judge.
- **Tests:** Export evaluation bundle, import, verify. Re-judge produces new scores without re-generation. Inspect judge artifacts from store.

**Phase 3 exit criteria:**
- All four metric families execute correctly through the `WorkflowRunner`.
- Judge fan-out (rubric, panel, repeated, pairwise) works with correct concurrency.
- Re-judge without re-generation works.
- Every judge score is inspectable from storage.
- Mixed metric runs produce correct results.
- Error contracts (`ScoreError`, `WorkflowBuildError`) are tested.

---

### Phase 4: Read Models, Reporting & Catalog (Weeks 7-8)

**Goal:** Build the read side, reporting, and migrate all built-in components.

**Entry criteria:** Phase 3 exit criteria met.

#### 4.1 Read Model Projections (Week 7)

- Implement projection builders from event streams:
  - `RunResult`
  - `BenchmarkResult`
  - `TimelineView`
  - `TraceView`
- Ensure projections support `GenerationResult` drill-down (final_output, trace, conversation, artifacts).
- Ensure projections support full `EvaluationExecution` drill-down (judge prompts, responses, judgments, scores, aggregation, traces).
- **Tests:** Project from recorded events. Verify drill-down to individual judge calls. Verify timeline accuracy.

#### 4.2 Reporter & Exporters (Week 7)

- Implement `Reporter` that builds outputs from read models.
- Implement export formats: JSON, Markdown, CSV, LaTeX, quick machine-readable score tables.
- Implement snapshot report from `RunSnapshot` + run metadata.
- Implement `quickcheck` on new store schema.
- **Tests:** Each export format produces valid output. Snapshot report includes identity and provenance. Quickcheck works on a completed run.

#### 4.3 Catalog & Built-Ins (Week 8)

- Create `themis.catalog` package.
- Migrate built-in parsers with namespaced component refs.
- Migrate built-in pure metrics: `builtin/exact_match`, `builtin/f1`, `builtin/bleu`, etc.
- Implement built-in evaluation workflows / LLM metrics: `builtin/llm_rubric`, `builtin/pairwise_judge`, `builtin/panel_of_judges`, `builtin/majority_vote_judge`.
- Implement built-in candidate reducer `builtin/majority_vote` and candidate selector `builtin/best_of_n`.
- Migrate benchmark definitions with stable ids, dataset fingerprint rules, prompt/parser/metric sets, summary adapters.
- Replace ad hoc registration with declarative catalog manifests + code-backed loaders.
- Implement `catalog.load("mmlu_pro")` and `catalog.run("mmlu_pro", model=..., store=...)`.
- **Tests:** Each built-in component has a unit test. Catalog load/run works. Benchmark definitions are stable.

#### 4.4 Postgres Store (Week 8)

- Implement `postgres_store(url, blob_root)` with explicit `.initialize()`.
- Implement blob storage to filesystem or object store.
- Run the full store contract test suite against Postgres.
- **Tests:** All store contract tests pass. Blob round-trip. Schema migration. Event schema forward-compatibility test.

#### 4.5 Statistics (Week 8)

- Implement `StatsEngine` behind a narrow boundary (optional extra).
- Preserve benchmark-native analysis behavior on top of generic result models.
- **Tests:** Stats computations match expected values on known data.

**Phase 4 exit criteria:**
- All read models project correctly from events.
- All export formats produce valid output.
- All built-in components are in the catalog with stable ids and fingerprints.
- Both SQLite and Postgres stores pass the full contract suite.
- `catalog.load()` and `catalog.run()` work.

---

### Phase 5: CLI & Integration (Weeks 9-10)

**Goal:** Rebuild the CLI on top of the library APIs, add config file support, and verify all user-facing workflows work end-to-end.

**Entry criteria:** Phase 4 exit criteria met.

#### 5.1 CLI Core (Week 9)

- Rebuild CLI framework on top of Layer 1 and Layer 2 APIs.
- Ensure CLI and Python use exactly the same execution model and snapshot generation path.
- Implement `quick-eval` as a thin wrapper around `evaluate(...)` / `Experiment(...)`:
  - `quick-eval inline`
  - `quick-eval file`
  - `quick-eval huggingface`
  - `quick-eval benchmark`

#### 5.2 CLI Commands (Week 9-10)

- Implement all CLI commands:
  - `init` / scaffold
  - `run` / `submit`
  - `resume`
  - `estimate`
  - `report`
  - `quickcheck`
  - `compare`
  - `export`
- **Tests:** CLI integration tests for each command. Verify CLI produces same results as Python API.

#### 5.3 Configuration File Loading (Week 10)

- Implement `Experiment.from_config("experiment.yaml")` for declarative experiment definitions.
- Support YAML and TOML formats.
- Config files map to `GenerationConfig`/`EvaluationConfig`/`StorageConfig` composition.
- Custom Python components referenced by importable module paths.
- **Tests:** Config file round-trip (load config, compile, verify `RunSnapshot` matches Python-defined equivalent).

#### 5.4 End-to-End Integration Tests (Week 10)

- Run the full acceptance test suite (see Test Plan) through both CLI and Python API.
- Test tool/MCP-enabled evaluations end-to-end.
- Test worker-pool and batch execution modes.
- **Tests:** All acceptance scenarios pass. CLI and Python produce identical results.

**Phase 5 exit criteria:**
- Every CLI command works and produces correct output.
- All acceptance scenarios pass through both CLI and Python API.
- `quick-eval` modes are thin wrappers, not a separate framework.
- Config file loading works for standard experiments.

---

### Phase 6: Hardening & Release (Weeks 11-12)

**Goal:** Performance optimization, security review, final documentation pass, and release.

**Entry criteria:** Phase 5 exit criteria met.

#### 6.1 Performance & Load Testing (Week 13)

- Profile memory usage under large fan-out (10K cases x 5 candidates x 3 judges).
- Verify O(concurrency_cap) memory bound holds.
- Benchmark throughput with rate limiting enabled.
- Optimize hot paths (event serialization, fingerprint computation, projection building).
- **Tests:** Memory stays bounded under load. Throughput meets target.

#### 6.2 Security Review (Week 13)

- Verify secrets are reference-only (never persisted as values).
- Review all external inputs (dataset rows, model responses, judge responses) for injection.
- Review blob storage permissions.
- **Tests:** No secrets in stored artifacts. Input fuzzing for parsers and importers.

#### 6.3 Failure Mode Testing (Week 13-14)

- Run all failure-mode tests from the Test Plan:
  - Provider timeout, rate limit, malformed responses.
  - Parser/metric exceptions on individual samples.
  - Judge failures (timeout, parse failure, partial fan-out failure).
  - Store write failure, interrupted run recovery.
  - Workflow step failure mid-candidate, reducer failure.
- **Tests:** All failure modes handled gracefully. No data loss. Resume works after each failure type.

#### 6.4 Release Packaging (Week 14)

- Final pass on public API surface: verify no internal types leak.
- Verify no v2 types, names, or patterns remain in v4.
- Package as fully separate major version.
- No compatibility layer, no dual runtime, no config translators, no store importers.
- Tag release.

**Phase 6 exit criteria (release criteria):**
- All acceptance tests pass.
- All failure-mode tests pass.
- Memory is bounded under load.
- No secrets in stored artifacts.
- No internal types in public API.
- Clean release tag with no v2 residue.

---

## Test Plan And Acceptance Criteria

- Build a behavior-first acceptance suite for every current user-facing capability.
- Required end-to-end scenarios:
  - 5-line hello world with in-memory dataset
  - custom model adapter with exact-match metric (PureMetric)
  - custom parser plus parsed metric
  - multiple models on one benchmark
  - interrupted run resume
  - re-score without re-generation
  - re-reduce without re-generation
  - re-judge without re-generation (inspect persisted judge artifacts, re-run evaluation workflow only)
  - LLM-backed metric with deterministic judge seeding (`LLMMetric` over `CandidateSetSubject` size=1)
  - pairwise judge comparison (`SelectionMetric` over `CandidateSetSubject` size=2)
  - panel-of-judges with majority vote aggregation (judge fan-out with multiple judge models)
  - rubric-based multi-dimension judge scoring (evaluation workflow with per-dimension scores)
  - candidate-set ranking by judge (`SelectionMetric` over `CandidateSetSubject` size>2)
  - trace/conversation quality judging (`TraceMetric` over `TraceSubject` and `ConversationSubject`)
  - mixed metric run combining PureMetric and LLMMetric on the same experiment
  - lifecycle subscribers modifying prompts and observing events (including `BeforeJudge` / `AfterJudge`)
  - generation bundle export/import
  - evaluation bundle export/import
  - quick-eval CLI modes
  - report generation and export
  - timeline and trace inspection (including judge trace inspection)
  - catalog benchmark execution
  - tool/MCP-enabled evaluation
  - SQLite backend
  - Postgres backend
  - custom generator producing multi-turn conversation trace
  - custom generator producing agent execution trace (for TraceMetric evaluation)
  - generator adapter integration (openai, vllm, langgraph built-in adapters)
  - self-consistency sampling with majority-vote candidate reducer
  - best-of-n generation with LLM-backed candidate reducer
  - pass@k evaluation over candidate sets
  - config file loading and execution matching Python API result
- Add golden tests for `RunSnapshot` stability and serialization.
- Add golden tests for `Generator` and `EvaluationWorkflow` fingerprint stability.
- Add failure-mode tests:
  - provider timeout
  - provider rate limit
  - malformed dataset rows
  - parser failure on one sample
  - metric exception on one sample
  - judge model timeout or rate limit (evaluation workflow retry and recovery)
  - judge response parse failure (malformed judge output)
  - partial judge fan-out failure (some rubric dimensions succeed, others fail)
  - disk/store write failure
  - interrupted run recovery
  - generator failure (timeout, exception) for individual candidates
  - reducer failure with fallback behavior
  - ScoreError handling for PureMetric failures
- Acceptance criteria:
  - every shipped `v2` feature has a `v4` equivalent or an explicit intentional removal
  - no primary `v4` workflow requires internal IR objects
  - a run is reproducible from persisted snapshot and referenced resources alone
  - raw response lifecycle artifacts are replayable end to end
  - generators are fingerprinted and their results (including trace/conversation) are persisted
  - evaluation workflows are serializable and fingerprinted end to end
  - candidate-level `GenerationResult` data is persisted and replayable
  - every judge score is inspectable from storage without re-running the judge
  - no metric receives mutable context dicts with hidden judge service injection
  - `PureMetric` receives only `ScoreContext` (never `EvalScoreContext` fields)
  - `CandidateSetSubject` cardinality constraints are enforced at compile time

## Documentation And Examples Follow-Up

- Documentation is a dedicated later pass and will be rewritten from the ground up using **Diátaxis**.
- The docs set must be usable by someone who did not read the architecture plan first. Each doc track should optimize for user questions, not internal design vocabulary.
- The docs landing experience should include:
  - a short "start here" page that routes users to tutorial vs guide vs reference vs explanation
  - a glossary for core terms (`case`, `candidate`, `reduced candidate`, `RunSnapshot`, `artifact`, `subject`, `score`, `run`, `benchmark`)
  - a "choose your API layer" decision page (`evaluate(...)` vs `Experiment(...)` vs custom extension protocols)
  - an FAQ for common questions, confusing behaviors, and frequent user mistakes
- The docs rewrite must produce four distinct tracks:
  - **Tutorials**: linear, hands-on learning paths for new users. These should teach one end-to-end workflow at a time, with copy-pasteable code and expected outcomes.
    - first evaluation: `evaluate(...)` against a tiny local dataset with one built-in metric
    - first experiment: `Experiment(...)` with explicit generation/evaluation/storage configuration
    - first persisted run: execute, inspect stored artifacts, and generate a report
    - first LLM-judged evaluation: add a judge-backed metric and inspect judge artifacts
    - first advanced run: multi-candidate generation plus reduction plus mixed metrics
  - **How-to guides**: task-oriented instructions for users who already know what they want to accomplish.
    - choose the right API layer (`evaluate`, `Experiment`, extension protocols)
    - configure OpenAI, vLLM, LangGraph, or a fully custom generator
    - author a custom `Generator`, `Parser`, `CandidateReducer`, or `Metric`
    - enable trace/conversation capture for trace-level evaluation
    - run from Python vs run from config file and CLI
    - resume interrupted runs and inspect run state
    - reproduce a run from `RunSnapshot` and stored artifacts
    - compare experiments, export reports, and drill into score details
    - control concurrency, retries, and cost-related settings safely
  - **Reference**: precise API and schema documentation, optimized for lookup rather than teaching.
    - public Python APIs: `evaluate`, `Experiment`, `Experiment.compile`, `RunResult`, `BenchmarkResult`
    - config models: `GenerationConfig`, `EvaluationConfig`, `StorageConfig`, retry/parallelism policies
    - core contracts: `Generator`, `CandidateReducer`, `Parser`, `PureMetric`, `LLMMetric`, `TraceMetric`, `WorkflowRunner`
    - data models: `RunSnapshot`, `GenerationResult`, `TraceStep`, evaluation subjects, score objects, artifact records
    - CLI commands, config-file schema, and component reference syntax
    - built-in components catalog: generators, reducers, metrics, reporters, storage backends
  - **Explanation**: concept and architecture documents that explain why the system is designed the way it is.
    - user mental model: the lifecycle of one case from dataset row to final score
    - glossary-level concept pages for runs, candidates, subjects, metrics, artifacts, and reports
    - snapshot-centric architecture and why `RunSnapshot` is the executable artifact
    - identity vs provenance: what changes a `run_id` and what does not
    - compiled experiment vs executed run: what gets frozen at compile time and what happens at runtime
    - generation as blackbox vs evaluation as Themis-owned runtime
    - candidate fan-out, reduction, parsing, and scoring as separate stages
    - reducer vs parser vs metric: where each responsibility belongs and common mistakes
    - event/store/read-model architecture and why artifacts are persisted by default
    - artifact model: what is stored by default, what users can inspect later, and how this supports debugging
    - fingerprinting, reproducibility, and replay/re-score workflows
    - failure, retry, and resume semantics from the user's point of view
    - API layer model and why Python-first is the primary surface
    - extension boundaries: what users own vs what Themis owns
- Explanation docs should explicitly answer the user-facing questions "What is this?", "When do I need to care?", "What do I provide?", "What does Themis provide?", and "What should I inspect when something goes wrong?"
- The docs rewrite should include an explicit FAQ section for common questions and mistakes, and this FAQ should be treated as a living part of the docs that grows over time as repeated user confusion appears in issues, support requests, examples, and onboarding feedback.
- All existing examples are replaced, not migrated in place.
- New examples must be authored directly against `v4` APIs and reflect the new layer model:
  - minimal usage
  - configurable experiments (with `GenerationConfig`, `EvaluationConfig`, `StorageConfig`)
  - custom generators (wrapping OpenAI, vLLM, LangGraph, and fully custom)
  - generator trace contract (populating trace/conversation for trace-level evaluation)
  - candidate reduction (majority-vote, best-of-n, LLM-backed selection)
  - pure metrics (exact match, F1, BLEU)
  - LLM-backed metrics (rubric grading, faithfulness, custom judge prompt)
  - selection metrics (pairwise comparison, candidate-set ranking)
  - trace metrics (agent trajectory scoring, conversation quality)
  - mixed metric experiments (pure + LLM-backed on the same run)
  - advanced extensibility
  - external execution
  - reporting and analysis (including judge score drill-down)
  - each example should be small, runnable, and focused on one primary idea
  - examples should use tiny fixture datasets and deterministic settings where feasible
  - examples should explicitly show what artifacts or outputs the user should inspect afterward
  - examples should be split between "quickstart-quality" examples and "reference-quality" advanced examples
  - examples should say what prerequisite setup is assumed (API keys, local models, installed packages)
  - examples should call out the decision they illustrate, not just the API they invoke
  - no example should depend on internal IR types or undocumented helper APIs
- Agent skills are a separate future pass after the docs rewrite.
  - Update project/Codex skills to remove `v2` concepts and align automation and planning guidance with `v4` APIs, CLI flows, and release process.

## Design Decisions And Rationale

These are active design decisions, not passive assumptions. Each includes the rationale so future readers can evaluate whether the decision still holds.

1. **Clean-break release with zero backward compatibility.**
   _Rationale:_ The v2 public surface (spec types, flat plugin ids, mutable context dicts) is too entangled to evolve incrementally. A compatibility layer would add permanent maintenance cost while slowing adoption of the new model. Users who need v2 keep v2.

2. **Feature parity means capabilities, not API/schema/naming preservation.**
   _Rationale:_ Preserving APIs would force the new architecture to mimic the old one. The goal is that every _workflow_ a user can accomplish in v2 is also accomplishable in v4, behind better abstractions.

3. **Retain the projection-backed read side.**
   _Rationale:_ The event-sourced read model is the strongest architectural asset in v2. It cleanly separates write-path complexity from read-path flexibility and already supports the replay/re-score workflows that v4 expands.

4. **Store raw responses by default (Principle 7).**
   _Rationale:_ Re-parsing and re-scoring are high-value workflows. Without raw artifacts, any change to a parser or metric requires re-running generation, which is the most expensive stage.

5. **Blackbox generation with typed contract (Principle 2).**
   _Rationale:_ Building a custom 9-node-type workflow engine for generation means Themis competes with LangGraph, LangChain, etc. on their turf — and will always be worse at it since generation orchestration isn't Themis's core concern. A blackbox `Generator` protocol with a structured `GenerationResult` (including optional trace and conversation data) is more honest about what Themis owns. Users can use any generation system; Themis provides built-in adapters as convenience. Fingerprinting and provenance are maintained at the `Generator` interface boundary.

6. **Candidate fan-out is an evaluation concern, not a generation concern (Principle 4).**
   _Rationale:_ The `Generator` produces one candidate per call. Themis calls it N times for pass@k, self-consistency, etc. This separates the evaluation concern (how many candidates do we need?) from the generation concern (how do we produce one candidate?). Planning, estimation, and resume are tractable because the planner has a simple cardinality axis.

7. **Evaluation workflows are Themis-owned (Principle 3).**
   _Rationale:_ Unlike generation (where users have diverse frameworks), judge execution is core evaluation infrastructure. Themis owns the evaluation workflow: prompt rendering, model calls, response parsing, score aggregation, retry, seeding, and artifact persistence. This ensures reproducibility and inspectability of every judge decision.

8. **Candidate reduction is a distinct stage.**
   _Rationale:_ v2 conflates selection with parsing, making it impossible to re-reduce without re-parsing, and muddying the boundary between "what the model said" and "which answer we chose."

9. **Consolidated evaluation subjects (3 types, not 5).**
   _Rationale:_ `CandidateSubject`, `CandidatePairSubject`, and `CandidateSetSubject` are slight variations on "some candidate outputs." Collapsing them into `CandidateSetSubject` with cardinality constraints (size=1 for grading, size=2 for pairwise, size>2 for ranking) eliminates accidental type proliferation while preserving compile-time validation.

10. **Judge-backed metrics as evaluation workflows over typed subjects (Principles 5, 6).**
    _Rationale:_ A generic `JudgeService` hides what the metric actually does. Typed subjects and explicit evaluation workflows make the judge's behavior inspectable, serializable, and fingerprintable.

11. **No mutable context dicts with hidden service injection (Principle 6).**
    _Rationale:_ v2's `runtime_context` and `judge_service` injection via dicts is a source of subtle bugs and makes metrics impossible to test in isolation. Typed contexts (`ScoreContext` for pure metrics, `EvalScoreContext` for LLM metrics) eliminate this. The split ensures pure metrics never see judge-specific configuration.

12. **Fingerprints frozen at compile time (Principle 8).**
    _Rationale:_ Computing fingerprints once at `Experiment.compile()` and freezing them into the `RunSnapshot` eliminates the tension between fingerprint stability and resume. A component change after compilation does not silently invalidate an in-progress run. To pick up a change, the user must recompile (creating a new `run_id`). This also avoids performance concerns from recomputing fingerprints on every access.

13. **Observability designed in, not retrofitted.**
    _Rationale:_ Retrofitting tracing into a workflow/evaluation engine is expensive and invasive. The `TracingProvider` protocol and structured span emission are designed from the start, with a zero-dependency default (JSON logs to stderr) so observability works out of the box.
