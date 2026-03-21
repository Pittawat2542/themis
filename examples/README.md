# Examples

Every numbered example in this directory uses the benchmark-first public API.

## Recommended Order

1. `01_hello_world.py`
   Smallest end-to-end benchmark run.

2. `02_project_file.py`
   Moves shared policy into a TOML project file.

3. `03_custom_extractor_metric.py`
   Adds a parse pipeline and scores parsed output.

4. `04_compare_models.py`
   Aggregates by slice and runs a paired comparison.

5. `05_resume_run.py`
   Reuses the same storage root across repeated benchmark runs.

6. `06_hooks_and_timeline.py`
   Shows hooks, prompt mutation, and timeline inspection.

7. `07_judge_metric.py`
   Demonstrates a judge-backed metric through the benchmark API.

8. `08_external_stage_handoff.py`
   Exports evaluation work, scores it externally, and imports it back.

9. `09_experiment_evolution.py`
   Evolves a benchmark by adding models and prompt variants.

10. `10_agent_eval.py`
   Demonstrates bootstrap prompts, scripted follow-up turns, benchmark tool overrides, slice-level tool selection, runtime tool handlers, and agent traces.

## Run One

```bash
uv run python examples/01_hello_world.py
```

The published examples catalog is [docs/guides/examples.md](../docs/guides/examples.md).

## Scope

- Use the current `ProjectSpec` + `BenchmarkSpec` + `Orchestrator` flow only.
- Prefer local fake plugins unless the example is explicitly about provider integration.
- Keep each script runnable on its own.
- Leave `examples/medical_reasoning_eval` untouched; it is a handoff artifact, not the recommended pattern.
