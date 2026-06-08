from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from themis.core.contexts import GenerateContext
from themis.core.submission import run_worker_once, submit_experiment
from themis.core.experiment import Experiment
from themis.core.models import Case, GenerationResult


class ExternalExecutionGenerator:
    component_id = "generator/external-execution"
    version = "1.0"

    def fingerprint(self) -> str:
        return "external-execution-generator"

    async def generate(self, case: Case, ctx: GenerateContext) -> GenerationResult:
        del ctx
        return GenerationResult(
            candidate_id=f"{case.case_id}-candidate",
            final_output=case.expected_output,
        )


def build_generator() -> ExternalExecutionGenerator:
    """Return the reviewed Python generator used by the automation config."""

    return ExternalExecutionGenerator()


CONFIG_TEMPLATE = """
generation:
  generator: examples.docs.external_execution:build_generator
  candidate_policy:
    num_samples: 1
  reducer: builtin/majority_vote
evaluation:
  metrics:
    - builtin/exact_match
  parsers:
    - builtin/json_identity
storage:
  target: sqlite
  kwargs:
    path: runs/themis.sqlite3
runtime:
  queue_root: runs/queue
dataset_sources:
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
