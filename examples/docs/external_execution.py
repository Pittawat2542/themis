from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory


DEFINITION = """from themis import Case, Dataset, Evaluation, Experiment, Generation

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
"""

LAUNCHER = """definition: definition:experiment
storage:
  target: sqlite
  kwargs:
    path: runs/themis.sqlite3
runtime:
  queue_root: runs/queue
"""


def run_example() -> dict[str, object]:
    """Submit a Python-defined experiment and execute one worker cycle."""

    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "definition.py").write_text(DEFINITION, encoding="utf-8")
        config_path = root / "experiment.yaml"
        config_path.write_text(LAUNCHER, encoding="utf-8")
        submitted = subprocess.run(
            [
                sys.executable,
                "-m",
                "themis.cli",
                "submit",
                "--config",
                str(config_path),
                "--mode",
                "worker-pool",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        worker = subprocess.run(
            [
                sys.executable,
                "-m",
                "themis.cli",
                "worker",
                "run",
                "--queue-root",
                str(root / "runs" / "queue"),
                "--definition-root",
                str(root),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        submission_data = json.loads(submitted.stdout)["data"]
        result_data = json.loads(worker.stdout)["data"]
        return {
            "run_id": result_data["run_id"],
            "status": result_data["status"],
            "manifest_path": submission_data["manifest_path"],
        }


if __name__ == "__main__":
    print(run_example())
