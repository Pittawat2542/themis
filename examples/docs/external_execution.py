from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from themis.core.submission import run_worker_once, submit_experiment
from themis.core.experiment import Experiment


CONFIG_TEMPLATE = """
generation:
  generator: builtin/demo_generator
  candidate_policy:
    num_samples: 1
  reducer: builtin/majority_vote
evaluation:
  metrics:
    - builtin/exact_match
  parsers:
    - builtin/json_identity
storage:
  store: sqlite
  parameters:
    path: runs/themis.sqlite3
runtime:
  queue_root: runs/queue
datasets:
  - dataset_id: sample
    cases:
      - case_id: case-1
        input:
          question: 2+2
        expected_output:
          answer: "4"
""".strip()


def run_example() -> dict[str, object]:
    """Submit an experiment to the worker-pool flow and execute one worker cycle."""

    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        config_path = root / "experiment.yaml"
        config_path.write_text(CONFIG_TEMPLATE, encoding="utf-8")
        experiment = Experiment.from_config(config_path)
        manifest = submit_experiment(experiment, config_path=str(config_path), mode="worker_pool")
        result = run_worker_once(root / "runs" / "queue")
        assert result is not None
        return {
            "run_id": result.run_id,
            "status": result.status.value,
            "manifest_path": str(manifest.manifest_path),
        }


if __name__ == "__main__":
    print(run_example())
