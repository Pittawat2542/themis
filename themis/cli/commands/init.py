"""Scaffolding CLI command."""

from __future__ import annotations

from pathlib import Path


def init(*, path: str) -> int:
    root = Path(path)
    (root / "data").mkdir(parents=True, exist_ok=True)
    (root / "experiment.yaml").write_text(
        """
definition: experiment:experiment
storage:
  target: sqlite
  kwargs:
    path: runs/themis.sqlite3
""".strip()
    )
    (root / "experiment.py").write_text(
        '''from themis import Case, Dataset, Evaluation, Experiment, Generation

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
    seeds=[7],
)
'''
    )
    (root / "data" / "sample.jsonl").write_text(
        '{"case_id":"case-1","input":{"question":"2+2"},"expected_output":{"answer":"4"}}\n'
    )
    (root / "run.py").write_text(
        """
from pathlib import Path

from experiment import experiment
from themis.storage import sqlite_store


if __name__ == "__main__":
    result = experiment.run(
        store=sqlite_store(Path(__file__).with_name("runs/themis.sqlite3"))
    )
    print(result.run_id)
""".strip()
        + "\n"
    )
    print(root)
    return 0
