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
   Demonstrates bootstrap prompts, scripted follow-up turns, benchmark tool overrides, slice-level tool selection, runtime tool handlers, agent traces, and persisted trace scoring.

11. `11_quick_benchmark.py`
   Demonstrates `BenchmarkSpec.simple()`, preview, and `PluginRegistry.from_dict()`.

12. `12_iter_and_estimate.py`
   Demonstrates streaming iteration, estimate output, and resume invalidation checks.

13. `13_catalog_builtin_benchmark.py`
   Runs a shipped builtin benchmark through `themis.catalog` with a local fixture dataset loader.

14. `14_mcp_openai.py`
   Demonstrates the OpenAI Responses MCP path with a remote MCP server and no local runtime tool handlers.

## Run One

```bash
uv run python examples/01_hello_world.py
```

When an example shows `DatasetQuerySpec.subset(..., seed=...)`, the explicit
seed is there to make subset selection reproducible. Omitting the seed keeps
count-based sampling deterministic and order-based.

The published examples catalog is [docs/guides/examples.md](../docs/guides/examples.md).

## Scope

- Use the current `ProjectSpec` + `BenchmarkSpec` + `Orchestrator` flow only.
- Prefer local fake plugins unless the example is explicitly about provider integration.
- Keep each script runnable on its own.
- Leave `examples/medical_reasoning_eval` untouched; it is a handoff artifact, not the recommended pattern.
