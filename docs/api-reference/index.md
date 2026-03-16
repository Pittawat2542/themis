# API Reference

The API reference is generated from the Python source. Use this section when you
need exact fields, signatures, and docstrings.

## Main Areas

| Area | Includes |
| --- | --- |
| [Root Package](root.md) | curated top-level imports from `themis` |
| [Errors](errors.md) | stable exception hierarchy plus error-record mapping helpers |
| [Specs](specs.md) | write-side configuration models |
| [Orchestration](orchestration.md) | orchestrator facade plus linked implementation-detail internals |
| [Run Planning](run-planning.md) | run manifests, run handles, cost estimates, and exported work bundles |
| [Runtime](runtime.md) | result-side facades and comparison tables |
| [Progress](progress.md) | progress snapshots, renderer config, and the in-process event bus |
| [Registry](registry.md) | plugin metadata and registration helpers |
| [Extractors](extractors.md) | built-in shipped extractors auto-registered by `PluginRegistry` |
| [Protocols](protocols.md) | interfaces for engines, metrics, hooks, repositories, and exporters |
| [Telemetry](telemetry.md) | in-process event bus plus Langfuse callback wiring |
| [Records](records.md) | immutable output models |
| [Storage](storage.md) | SQLite repositories and artifact storage internals |
| [Postgres Storage](postgres-storage.md) | Postgres connection management for shared-store deployments |
| [Types](types.md) | shared enums, event models, and JSON/value helpers |
| [Reporting & Stats](reporting-and-stats.md) | report assembly, exporters, and statistical comparisons |
| [Config Reports](config-reports.md) | collection models, typed render APIs, and renderer registration |
| [CLI](cli.md) | the `themis` parent CLI plus `themis-quickcheck` |

!!! note
    The generated docs are only as good as the source docstrings. This docset
    intentionally keeps the API reference close to the code instead of copying it
    into separate prose pages.

    Pages that include planner, executor, repository, or schema modules keep the
    implementation details in view for power users, but those modules are not
    the stable extension surface. Prefer the root package, namespace re-exports,
    and documented protocol interfaces when choosing imports for application
    code.
