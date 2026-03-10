# Examples

This directory is intentionally small and progressive. Every script is
self-contained, well-commented, and aligned with the current implementation.

## Recommended Order

1. `01_hello_world.py`
   Minimal end-to-end run. Mirrors the Quick Start.

2. `02_project_file.py`
   Moves shared policy into a TOML project file and loads it with
   `Orchestrator.from_project_file(...)`.

3. `03_custom_extractor_metric.py`
   Shows how to author and register a custom extractor plus a metric that scores
   parsed output instead of raw text.

4. `04_compare_models.py`
   Runs a paired comparison and exports a Markdown report.
   Requires `themis-eval[stats]`.

5. `05_resume_run.py`
   Demonstrates how repeated runs skip completed trials when storage, specs, and
   evaluation revision match.

6. `06_hooks_and_timeline.py`
   Shows how hooks change prompts, how telemetry events flow, and how to inspect
   a timeline view after execution.

7. `07_judge_metric.py`
   Demonstrates judge-backed metrics and audit-trail inspection.
   Requires `themis-eval[compression]`.

## Run an Example

```bash
uv run python examples/01_hello_world.py
```

## Design Rules For This Folder

- Use only the current `ProjectSpec` + `ExperimentSpec` + `Orchestrator` flow.
- Prefer local fake plugins over external providers unless the example is
  explicitly about provider integration.
- Keep each script runnable on its own.
