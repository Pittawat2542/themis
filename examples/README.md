# Themis Examples

These examples demonstrate the core workflows of Themis using the unified `themis.evaluate()` API and advanced features like comparison loops and internal tooling.

## Quick Start

You can run any of the standalone examples directly. For instance:

```bash
uv run python examples/01_quickstart.py
```

## Available Examples

### Standalone Snippets
- `01_quickstart.py`: basic evaluation using benchmark presets.
- `02_custom_dataset.py`: custom inline dataset evaluation.
- `03_distributed.py`: execution tuning (parallel workers and retries).
- `04_comparison.py`: run multiple experiments and compare them statistically.
- `05_api_server.py`: launching the Themis FastAPI server with REST/WebSocket interfaces.
- `06_custom_metrics.py`: registering and using custom evaluation metrics.
- `07_provider_ready.py`: real-provider execution scaffolding with fake fallback.
- `08_resume_cache.py`: cache hit/resume behavior with shared `run_id`.
- `09_research_loop.py`: run, export to bundle, compare, and generate shareable markdown in one script.
- `10_logging_demo.py`: using built-in structured logging and tracing utilities.

### In-Depth Tutorials
- **`countdown/`**: A dense, multi-part internal tutorial showcasing advanced R&D pipelines (SLURM orchestration, dataset synthesis, manifest tracking, reproducibility gates). See [its README](countdown/README.md) for sequence details.

## Notes

- Results and metrics are written to `.cache/experiments` by default.
- If you don't provide real API keys, the examples frequently default the model string to `fake:fake-math-llm` for local, cost-free smoke testing.
