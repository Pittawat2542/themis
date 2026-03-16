# Tutorials

Tutorials are guided lessons. Use them when you want to learn Themis by building
something step by step, not when you just need a quick answer.

| Tutorial | Goal | Main APIs |
| --- | --- | --- |
| [Hello World Walkthrough](hello-world.md) | Build and run a minimal experiment from scratch | `PluginRegistry`, `ProjectSpec`, `ExperimentSpec`, `Orchestrator` |
| [Load a Project File](project-files.md) | Convert inline project policy into reusable file-based config | `ProjectSpec`, `Orchestrator.from_project_file()` |
| [Analyze a Stored Run](analyze-results.md) | Run a shipped experiment, inspect a report, and confirm the stored outputs | `ExperimentResult`, `ReportBuilder`, `themis-quickcheck` |
| [Provider-backed Run](provider-backed-run.md) | Wire a minimal provider engine, authenticate with env vars, and run one tiny task | `InferenceEngine`, `PluginRegistry`, `Orchestrator` |

## Example Progression

The examples directory mirrors this progression:

1. `examples/01_hello_world.py`
2. `examples/02_project_file.py`
3. `examples/04_compare_models.py`

The provider-backed tutorial intentionally uses a tiny one-item task and notes
that real provider outputs vary by model version and account settings.
