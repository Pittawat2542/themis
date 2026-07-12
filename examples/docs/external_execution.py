from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from themis.core.submission import run_worker_once, submit_experiment
from themis.launcher import load_core_experiment


DEFINITION = '''from themis import Case, Dataset, Evaluation, Experiment, Generation

experiment = Experiment(
    datasets=[Dataset(
        dataset_id="sample",
        cases=[Case(
            case_id="case-1",
            input={"question": "2+2"},
            expected_output={"answer": "4"},
        )],
    )],
    generation=Generation(
        generator="builtin/demo_generator",
        reducer="builtin/majority_vote",
    ),
    evaluation=Evaluation(
        metrics=["builtin/exact_match"],
        parser="builtin/json_identity",
    ),
)
'''

LAUNCHER = '''definition: definition:experiment
storage:
  target: sqlite
  kwargs:
    path: runs/themis.sqlite3
runtime:
  queue_root: runs/queue
'''


def run_example() -> dict[str, object]:
    """Submit a Python-defined experiment and execute one worker cycle."""

    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "definition.py").write_text(DEFINITION, encoding="utf-8")
        config_path = root / "experiment.yaml"
        config_path.write_text(LAUNCHER, encoding="utf-8")
        experiment = load_core_experiment(config_path)
        manifest = submit_experiment(
            experiment, config_path=str(config_path), mode="worker_pool"
        )
        result = run_worker_once(root / "runs" / "queue")
        assert result is not None
        return {
            "run_id": result.run_id,
            "status": result.status.value,
            "manifest_path": str(manifest.manifest_path),
        }


if __name__ == "__main__":
    print(run_example())
