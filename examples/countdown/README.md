# Countdown Example Scripts

These scripts back the command sequences used in the Countdown tutorial.

## Core scripts

- `countdown_tutorial_run.py`: Part 1 baseline tutorial run.
- `run_countdown_baseline.py`: Part 5 baseline run (`countdown-baseline-v1`).
- `run_countdown_candidate.py`: Part 5 candidate run (`countdown-candidate-v1`).
- `verify_reproducibility.py`: Manifest checks for candidate run.
- `gate_candidate.py`: Baseline vs candidate quality gate.
- `build_research_bundle.py`: Bundle export + `index.json`.

## Part 6 scripts

- `run_countdown_part6_backends.py`: Explicit execution/storage backends + bundle export.
- `publish_countdown_part6.py`: Optional W&B/MLflow publish from bundle.

## Part 9 scripts

- `build_manifest_index.py`: Collect run manifests into `manifest_index.json`.
- `diff_manifests.py`: Compute manifest diff payload.
- `gate_reproducibility.py`: Enforce allowed drift and emit audit JSON.

## Wrapper scripts

- `run_part5_pipeline.sh`: Run baseline/candidate + reproducibility + quality gate + bundle.
- `run_part7_pipeline.sh`: Run operations/observability workflow.
- `run_part8_pipeline.sh`: Run statistical reliability workflow.
- `run_part9_pipeline.sh`: Run manifest index + diff + reproducibility gate.

Run from repository root, for example:

```bash
uv run python examples/countdown/run_countdown_baseline.py

# or wrappers
bash examples/countdown/run_part5_pipeline.sh
```
