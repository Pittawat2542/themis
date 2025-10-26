# Advanced Example Experiment

This example demonstrates how to extend or mix in custom components on top of the
base Themis abstractions. It reuses the dataset loaders and provider router from
the simpler example, but swaps in a custom evaluation pipeline to compute
subject-level breakdowns and an instrumented provider router that records per-
model invocation counts.

## Highlights

- **Custom configuration** – Extends the base config with `prompt_style` and
  `enable_subject_breakdown` flags.
- **Subject-aware evaluation** – Subclasses `EvaluationPipeline` to compute
  averaged metrics per subject and exposes them via metadata.
- **Instrumented provider router** – Mixes in additional bookkeeping while still
  delegating all core logic to the package's `ModelProvider` interface.
- **Cyclopts CLI** – `experiments/advanced_example/cli.py` wires these parts into
  a reproducible command-line workflow.

## Usage

```bash
uv run python -m experiments.advanced_example.cli run --prompt-style cot
```

Copy `config.sample.json` to customize models/datasets/prompt styles and pass it
via `--config-path`.
