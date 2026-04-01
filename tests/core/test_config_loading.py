from __future__ import annotations

from pathlib import Path

from themis import Experiment
from themis.core.models import Case, Dataset, GenerationResult


class ConfigGenerator:
    component_id = "generator/config"
    version = "1.0"

    def fingerprint(self) -> str:
        return "generator-config"

    async def generate(self, case: Case, ctx: object) -> GenerationResult:
        del ctx
        return GenerationResult(
            candidate_id=f"{case.case_id}-candidate", final_output=case.expected_output
        )


CONFIG_GENERATOR = ConfigGenerator()


def test_experiment_from_yaml_matches_python_defined_equivalent(tmp_path: Path) -> None:
    path = tmp_path / "experiment.yaml"
    path.write_text(
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
  store: memory
datasets:
  - dataset_id: dataset-1
    revision: r1
    cases:
      - case_id: case-1
        input:
          question: 2+2
        expected_output:
          answer: "4"
seeds: [7]
""".strip()
    )

    loaded = Experiment.from_config(path)
    explicit = Experiment(
        generation=loaded.generation,
        evaluation=loaded.evaluation,
        storage=loaded.storage,
        datasets=[
            Dataset(
                dataset_id="dataset-1",
                revision="r1",
                cases=[
                    Case(
                        case_id="case-1",
                        input={"question": "2+2"},
                        expected_output={"answer": "4"},
                    )
                ],
            )
        ],
        seeds=[7],
    )

    assert loaded.compile() == explicit.compile()


def test_experiment_from_toml_supports_overrides(tmp_path: Path) -> None:
    path = tmp_path / "experiment.toml"
    path.write_text(
        """
[generation]
generator = "builtin/demo_generator"
reducer = "builtin/majority_vote"

[generation.candidate_policy]
num_samples = 1

[evaluation]
metrics = ["builtin/exact_match"]
parsers = ["builtin/json_identity"]

[storage]
store = "memory"

[[datasets]]
dataset_id = "dataset-1"

[[datasets.cases]]
case_id = "case-1"

[datasets.cases.input]
question = "2+2"

[datasets.cases.expected_output]
answer = "4"
""".strip()
    )

    experiment = Experiment.from_config(
        path, overrides=["generation.candidate_policy.num_samples=2"]
    )

    assert experiment.generation.candidate_policy["num_samples"] == 2


def test_experiment_from_config_loads_custom_component_symbols(tmp_path: Path) -> None:
    path = tmp_path / "experiment.yaml"
    path.write_text(
        """
generation:
  generator: tests.core.test_config_loading:CONFIG_GENERATOR
evaluation:
  metrics: []
  parsers: []
storage:
  store: memory
datasets:
  - dataset_id: dataset-1
    cases:
      - case_id: case-1
        input: "hello"
""".strip()
    )

    experiment = Experiment.from_config(path)

    assert not isinstance(experiment.generation.generator, str)
    assert experiment.generation.generator.component_id == CONFIG_GENERATOR.component_id
    assert (
        experiment.generation.generator.fingerprint() == CONFIG_GENERATOR.fingerprint()
    )
    assert (
        experiment.compile().component_refs.generator.component_id == "generator/config"
    )


def test_experiment_from_config_resolves_relative_paths_from_config_directory(
    tmp_path: Path,
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    path = root / "experiment.yaml"
    path.write_text(
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
runtime:
  queue_root: runs/queue
  batch_root: runs/batch
datasets:
  - dataset_id: dataset-1
    cases:
      - case_id: case-1
        input:
          question: 2+2
        expected_output:
          answer: "4"
""".strip()
    )

    experiment = Experiment.from_config(path)

    assert experiment.storage.parameters["path"] == str(
        root / "runs" / "themis.sqlite3"
    )
    assert experiment.runtime.queue_root == str(root / "runs" / "queue")
    assert experiment.runtime.batch_root == str(root / "runs" / "batch")


def test_experiment_from_config_applies_overrides_before_normalizing_paths(
    tmp_path: Path,
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    path = root / "experiment.yaml"
    path.write_text(
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
runtime:
  queue_root: runs/queue
  batch_root: runs/batch
datasets:
  - dataset_id: dataset-1
    cases:
      - case_id: case-1
        input:
          question: 2+2
        expected_output:
          answer: "4"
""".strip()
    )

    experiment = Experiment.from_config(
        path,
        overrides=[
            "storage.parameters.path=alt/store.sqlite3",
            "runtime.queue_root=alt/queue",
            "runtime.batch_root=alt/batch",
        ],
    )

    assert experiment.storage.parameters["path"] == str(root / "alt" / "store.sqlite3")
    assert experiment.runtime.queue_root == str(root / "alt" / "queue")
    assert experiment.runtime.batch_root == str(root / "alt" / "batch")
