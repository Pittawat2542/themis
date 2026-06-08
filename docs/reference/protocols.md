---
title: Protocols reference
diataxis: reference
audience: users and contributors implementing extensions
goal: Document the extension contracts exposed by Themis.
---

# Protocols reference

Use this page when you are implementing custom components rather than using builtin ids.

## Important runtime instrumentation contracts

| Name | Kind | Use when | Key constraints / notes |
| --- | --- | --- | --- |
| `LifecycleSubscriber` | Instrumentation protocol | You want callbacks around stage boundaries or raw `on_event(...)` notifications | Observes execution without changing `run_id` |
| `TracingProvider` | Instrumentation protocol | You want span-oriented tracing around runs or stages | Implements `start_span(...)` and `end_span(...)` hooks |

## Important config/runtime contracts

| Name | Kind | Use when | Key constraints / notes |
| --- | --- | --- | --- |
| `Generator` | Generation protocol | Candidate production logic belongs in your own code rather than a builtin or adapter | Used in both direct Python authoring and config-loaded experiments |
| `CandidateSelector` | Selection protocol | Multi-candidate fan-out needs a custom pre-reduction selection rule | Runs before reduction and participates in identity through component refs |
| `CandidateReducer` | Reduction protocol | Multi-candidate output needs custom collapse or synthesis logic | Pair with fan-out generation |
| `Parser` | Parsing protocol | Reduced output needs custom normalization before scoring | Keep parser responsibility separate from metric logic |
| `JudgeModel` | Evaluation protocol | Workflow-backed metrics need a custom judge model | Used by judge workflows and participates in identity through component refs |
| Metric protocols | Evaluation protocols | You need custom deterministic, workflow-backed, or trace-aware scoring | Choose the smallest metric protocol that matches the task |
| `WorkflowRunner` | Runtime protocol | Workflow-backed metrics need custom execution semantics | Keeps workflow execution separate from observation hooks |
| `ReporterProtocol` | Reporting protocol | Stored evidence needs a custom export or reporting view | Read-side boundary; reporter selection does not change `run_id` |

Generated contracts:

::: themis.core.protocols
