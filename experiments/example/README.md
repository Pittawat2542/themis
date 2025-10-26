# Example Experiment

This folder demonstrates how to build a reproducible experiment on top of the
Themis package. The goal is to make it trivial to plug in new models or
datasets: edit a config file, run the CLI, and rely on Themis for caching,
resume, and evaluation.

## Structure

- `config.py` – Pydantic models and a default configuration. Add new models or
  datasets by appending to the `DEFAULT_CONFIG` lists.
- `datasets.py` – dataset adapters (demo rows, MATH-500 via Hugging Face or a
  local mirror).
- `experiment.py` – wires prompts, plans, model providers, evaluation, and
  storage together.
- `cli.py` – Cyclopts-powered CLI for running the experiment end-to-end.
- `config.sample.json` – editable JSON config for experimentation.

## Running

```bash
uv run python -m experiments.example.cli run --dry-run
uv run python -m experiments.example.cli run --run-id demo --storage-dir .cache/example-demo
```

To customize the run, copy `config.sample.json`, tweak the `models`, `samplings`,
 or `datasets` arrays, and pass the path via `--config-path`.
