---
title: Python API reference
diataxis: reference
audience: Python users of Themis
goal: Provide a generated entry point for the public Python surface.
---

# Python API Reference

This page is the generated entry point into the public Python API. Use the smaller reference pages in this section when you already know the category of symbol you need.

## Root exports

The root package exposes the stable graph/runtime, storage, reporting, and inspection surface only:

| Name | Kind | Use when | Key constraints / notes |
| --- | --- | --- | --- |
| `__version__`, `Experiment`, `DatasetSourceSpec` | Core identity | You need package version, experiment authoring, or dataset-source identity | Root exports are intentionally explicit |
| `EvaluationGraph`, `EvaluationStep`, `StepInput`, `StepOutput`, `GraphRuntime`, `GraphRunResult` | Graph runtime | You need typed graph execution primitives | Replaces flat workflow execution |
| `ExecutionBackend`, `ExecutionRequest`, `InMemoryExecutionBackend`, `FilesystemExecutionBackend`, `QueueExecutionBackend`, `CodeExecutionRequest`, `CodeExecutionResult`, `CodeExecutionLimits`, `CodeExecutionStatus`, `LocalSubprocessExecutionBackend`, `DockerExecutionBackend` | Execution backend | You need local, file-backed, queue-backed, or code-execution primitives | Docker-backed code execution is optional; default tests use local subprocess execution |
| `ExecutionCheckpoint`, `ProjectionCursor`, `InMemoryRunStore`, `SqliteRunStore`, `RunStore`, `sqlite_store` | Storage | You need run storage, checkpoint, or projection cursor APIs | Events remain authoritative |
| `PromptSpec`, `RuntimeConfig`, `SessionConfig`, `RuntimePreset`, `SessionPreset`, `EvaluationPreset`, `ExperimentPreset`, `apply_preset`, `get_preset`, `list_presets` | Configuration | You need prompt, runtime, session-native settings, or named operational presets | Runtime-only presets do not change logical identity |
| `Reporter`, `ReporterProtocol`, `register_reporter`, `create_reporter`, `available_reporters`, `FailureSlice`, `FailureSliceSummary`, `ReliabilitySummary`, `RegressionPolicy`, `RegressionFinding`, `RegressionSummary`, `TrendPoint`, `TrendView`, `SuiteCoverageSummary`, `PairwiseComparisonReport`, `PairwiseMetricClaim`, `StatsEngine` | Reporting and analysis | You need summaries, custom reporters, suite coverage, slices, trends, regressions, or paired comparisons | Pairwise reports are Score Claims over stored Evidence and use canonical dataset-scoped case keys |
| `RerunPlan`, `RerunSelector`, `RunEstimate`, `ExecutionResourcePlan`, `RunResult`, `RunSnapshot`, `RunStatus`, `RunLineage`, `RunQuery`, `RunRecord` | Run models | You need typed run results, registry records, rerun payloads, or resource plans | Runtime resource plans are provenance unless they change logical experiment behavior |
| `ProviderTelemetry`, `StageTelemetry`, `ProviderCallStartedEvent`, `ProviderCallCompletedEvent`, `ProviderCallFailedEvent` | Telemetry | You need provider-call evidence and aggregate runtime summaries | Missing provider telemetry degrades to empty summaries instead of failed reporting |
| `get_case_audit`, `get_evaluation_execution`, `get_execution_state`, `get_run_record`, `get_run_snapshot`, `get_telemetry_summary` | Inspection | You need stored case, workflow, state, registry, identity, or telemetry details | Prefer dataset-scoped case keys when duplicate case IDs exist |

`__version__`, `Experiment`, `DatasetSourceSpec`, `EvaluationGraph`, `EvaluationStep`, `StepInput`, `StepOutput`, `GraphRuntime`, `GraphRunResult`, `ExecutionBackend`, `ExecutionRequest`, `CodeExecutionRequest`, `CodeExecutionResult`, `CodeExecutionLimits`, `CodeExecutionStatus`, `LocalSubprocessExecutionBackend`, `DockerExecutionBackend`, `InMemoryExecutionBackend`, `FilesystemExecutionBackend`, `QueueExecutionBackend`, `ExecutionCheckpoint`, `ProjectionCursor`, `InMemoryRunStore`, `SqliteRunStore`, `RunStore`, `sqlite_store`, `PromptSpec`, `RuntimeConfig`, `SessionConfig`, `RuntimePreset`, `SessionPreset`, `EvaluationPreset`, `ExperimentPreset`, `apply_preset`, `get_preset`, `list_presets`, `Reporter`, `ReporterProtocol`, `register_reporter`, `create_reporter`, `available_reporters`, `FailureSlice`, `FailureSliceSummary`, `ReliabilitySummary`, `RegressionPolicy`, `RegressionFinding`, `RegressionSummary`, `TrendPoint`, `TrendView`, `SuiteCoverageSummary`, `PairwiseComparisonReport`, `PairwiseMetricClaim`, `StatsEngine`, `RerunPlan`, `RerunSelector`, `RunEstimate`, `ExecutionResourcePlan`, `RunResult`, `RunSnapshot`, `RunStatus`, `RunLineage`, `RunQuery`, `RunRecord`, `ProviderTelemetry`, `StageTelemetry`, `ProviderCallStartedEvent`, `ProviderCallCompletedEvent`, `ProviderCallFailedEvent`, `get_case_audit`, `get_evaluation_execution`, `get_execution_state`, `get_run_record`, `get_run_snapshot`, and `get_telemetry_summary`.

Removed convenience helpers such as one-call evaluation, root bundle import/export helpers, `quickcheck`, and `snapshot_report` are not part of the root API. Use `Experiment`, `Reporter`, inspection helpers, CLI export commands, or lower-level core modules where appropriate.

## Generated Modules

Root package:

::: themis

Catalog namespace:

::: themis.catalog

Core namespace:

::: themis.core

Adapters:

::: themis.adapters
