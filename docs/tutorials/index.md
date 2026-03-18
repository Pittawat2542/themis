# Tutorials

Use these when you want a complete script, not a single recipe.

| Tutorial | What you build | Key APIs |
| --- | --- | --- |
| [Hello World Walkthrough](hello-world.md) | A minimal benchmark run from scratch | `BenchmarkSpec`, `SliceSpec`, `PromptVariantSpec`, `BenchmarkResult` |
| [Load a Project File](project-files.md) | A benchmark with file-backed shared policy | `ProjectSpec`, `Orchestrator.from_project_file(...)` |
| [Analyze a Stored Run](analyze-results.md) | Aggregate, compare, and inspect stored results | `BenchmarkResult`, `themis-quickcheck` |
| [Provider-backed Run](provider-backed-run.md) | A benchmark that uses dataset queries and orchestration-rendered prompts | `DatasetProvider`, rendered `trial.prompt.messages` |
