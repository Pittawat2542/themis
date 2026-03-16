# API Reference

The API reference is generated from the Python source. Use this section when you
need exact fields, signatures, and docstrings.

## Main Areas

| Area | Includes |
| --- | --- |
| Root Package | curated top-level imports from `themis` |
| Specs | write-side configuration models |
| Orchestration | planner, executor, runner, and facade |
| Runtime | result-side facades and comparison tables |
| Registry | plugin metadata and registration helpers |
| Extractors | built-in shipped extractors auto-registered by `PluginRegistry` |
| Protocols | interfaces for engines, metrics, hooks, repositories, and exporters |
| Telemetry | in-process event bus plus Langfuse callback wiring |
| Records | immutable output models |
| Storage | SQLite repositories and artifact storage |
| Reporting & Stats | report assembly, exporters, and statistical comparisons |
| Config Reports | collection models, typed render APIs, and renderer registration |
| CLI | the `themis` parent CLI plus `themis-quickcheck` |

!!! note
    The generated docs are only as good as the source docstrings. This docset
    intentionally keeps the API reference close to the code instead of copying it
    into separate prose pages.
