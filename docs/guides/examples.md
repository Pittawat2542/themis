# Example Catalog

These runnable scripts are the canonical workflow sources for the docs. Prefer
them over copy-pasting isolated snippets when you need a complete starting
point.

| Example | What it teaches | Prerequisites | Related docs |
| --- | --- | --- | --- |
| `examples/01_hello_world.py` | minimal local run with one loader, one engine, one metric | base install | [Quick Start](../quick-start/index.md), [Hello World Walkthrough](../tutorials/hello-world.md) |
| `examples/02_project_file.py` | loading `ProjectSpec` from TOML | base install | [Load a Project File](../tutorials/project-files.md), [Author Project Files](project-files.md) |
| `examples/03_custom_extractor_metric.py` | custom extractor plus metric over parsed output | base install | [Add a Minimal Plugin Set](plugins.md) |
| `examples/04_compare_models.py` | paired comparison and report export | `themis-eval[stats]` | [Compare and Export Results](compare-and-export.md), [Analyze Results](analyze-results.md) |
| `examples/05_resume_run.py` | rerun skip behavior on the same storage root | base install | [Resume and Inspect Runs](resume-and-inspect.md) |
| `examples/06_hooks_and_timeline.py` | hooks, telemetry events, and timeline inspection | base install | [Attach Telemetry & Observability](telemetry-and-observability.md) |
| `examples/07_judge_metric.py` | judge-backed metric and audit-trail inspection | `themis-eval[compression]` | [Add a Minimal Plugin Set](plugins.md) |
| `examples/08_external_stage_handoff.py` | external evaluation handoff and re-import | `themis-eval[stats]` | [Hand Off Generation or Evaluation](external-stage-handoffs.md) |
| `examples/09_experiment_evolution.py` | adding metrics, prompts, and slices across repeated runs | base install | [Evolve an Experiment](evolve-an-experiment.md) |

## How To Use This Catalog

- Start with the smallest example that matches your real task.
- Use the linked guide when you need the surrounding explanation.
- Treat the script output and storage roots as the canonical values used
  elsewhere in the docs.
