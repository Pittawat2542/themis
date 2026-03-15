# Tutorials

Tutorials are guided lessons. Use them when you want to learn Themis by building
something step by step, not when you just need a quick answer.

| Tutorial | Goal | Main APIs |
| --- | --- | --- |
| [Hello World Walkthrough](hello-world.md) | Build and run a minimal experiment from scratch | `PluginRegistry`, `ProjectSpec`, `ExperimentSpec`, `Orchestrator` |
| [Load a Project File](project-files.md) | Convert inline project policy into reusable file-based config | `ProjectSpec`, `Orchestrator.from_project_file()` |
| [Analyze Results](analyze-results.md) | Extend a comparison run into reports, statistics, and operator checks | `ExperimentResult`, `ReportBuilder`, `themis-quickcheck` |

## Example Progression

The examples directory mirrors this progression:

1. `examples/01_hello_world.py`
2. `examples/02_project_file.py`
3. `examples/04_compare_models.py`
4. `examples/05_resume_run.py`
5. `examples/06_hooks_and_timeline.py`
6. `examples/07_judge_metric.py`
