"""Scaffolding CLI command."""

from __future__ import annotations

from pathlib import Path


def init(*, path: str) -> int:
    root = Path(path)
    (root / "data").mkdir(parents=True, exist_ok=True)
    (root / "experiment.yaml").write_text(
        """
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
datasets:
  - dataset_id: sample
    cases: []
seeds: [7]
""".strip()
    )
    (root / "data" / "sample.jsonl").write_text(
        '{"case_id":"case-1","input":{"question":"2+2"},"expected_output":{"answer":"4"}}\n'
    )
    (root / "run.py").write_text(
        """
from pathlib import Path

from themis import Experiment


if __name__ == "__main__":
    result = Experiment.from_config(Path(__file__).with_name("experiment.yaml")).run()
    print(result.run_id)
""".strip()
        + "\n"
    )
    print(root)
    return 0
