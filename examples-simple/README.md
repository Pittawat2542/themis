# Themis Examples

These examples cover both high-level `themis.evaluate(...)` and explicit
`ExperimentSpec` + `ExperimentSession.run(...)` workflows.

## Run

```bash
uv run python examples-simple/01_quickstart.py
```

## Files

- `01_quickstart.py`: benchmark preset with explicit specs/session.
- `02_custom_dataset.py`: custom inline dataset with explicit pipeline.
- `03_distributed.py`: execution tuning (workers/retries) via `ExecutionSpec`.
- `04_comparison.py`: run two experiments and compare statistically.
- `05_api_server.py`: API server usage and REST/WebSocket examples.
- `06_custom_metrics.py`: custom metric class in a spec pipeline.
- `07_provider_ready.py`: real-provider-ready run with fake fallback.
- `08_resume_cache.py`: cache/resume behavior with shared `run_id`.
- `09_research_loop.py`: run, export, compare, and share in one script.

## Notes

- Results are written under `.cache/experiments` by default.
- Use `fake:fake-math-llm` for local smoke tests without API keys.
