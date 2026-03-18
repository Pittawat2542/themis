# Concepts

Themis has one public mental model now: benchmark authoring on the write side,
projection-backed analysis on the read side.

## Core Split

| Layer | Main types |
| --- | --- |
| Authoring | `ProjectSpec`, `BenchmarkSpec`, `SliceSpec`, `PromptVariantSpec`, `ParseSpec`, `ScoreSpec` |
| Runtime extension | `DatasetProvider`, `InferenceEngine`, `Extractor`, `Metric`, `JudgeService`, `PipelineHook` |
| Read side | `BenchmarkResult`, `RecordTimelineView`, `themis-quickcheck` |

## Start Here

- [Architecture](architecture.md) for the end-to-end flow
- [Specs and Records](specs-and-records.md) for the public object model
- [Plugins and Hooks](plugins-and-hooks.md) for extension boundaries
- [Storage and Resume](storage-and-resume.md) for persistence and reuse
- [Statistical Comparisons](statistical-comparisons.md) for aggregation semantics
