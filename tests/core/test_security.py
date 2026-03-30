from __future__ import annotations

import pytest

from themis.core.base import JSONValue
from themis.core.config import EvaluationConfig, GenerationConfig, StorageConfig
from themis.core.experiment import Experiment
from themis.core.models import Case, Dataset


def _experiment(
    *,
    storage: StorageConfig | None = None,
    environment_metadata: dict[str, str] | None = None,
    judge_config: dict[str, JSONValue] | None = None,
) -> Experiment:
    return Experiment(
        generation=GenerationConfig(generator="builtin/demo_generator"),
        evaluation=EvaluationConfig(
            metrics=["builtin/exact_match"],
            parsers=["builtin/json_identity"],
            judge_config=judge_config or {},
        ),
        storage=storage or StorageConfig(store="memory"),
        datasets=[
            Dataset(
                dataset_id="dataset-1",
                cases=[Case(case_id="case-1", input={"question": "2+2"}, expected_output={"answer": "4"})],
            )
        ],
        environment_metadata=environment_metadata or {},
    )


def test_compile_rejects_literal_secret_in_environment_metadata() -> None:
    experiment = _experiment(environment_metadata={"OPENAI_API_KEY": "sk-test-secret"})

    with pytest.raises(ValueError, match="literal secret"):
        experiment.compile()


def test_compile_rejects_literal_secret_in_judge_config() -> None:
    experiment = _experiment(judge_config={"api_key": "top-secret-token"})

    with pytest.raises(ValueError, match="literal secret"):
        experiment.compile()


def test_compile_allows_reference_style_secret_values() -> None:
    experiment = _experiment(
        environment_metadata={
            "OPENAI_API_KEY": "${OPENAI_API_KEY}",
            "ANTHROPIC_API_KEY": "env:ANTHROPIC_API_KEY",
            "JUDGE_TOKEN": "secret://vault/judge-token",
        }
    )

    snapshot = experiment.compile()

    assert snapshot.provenance.environment_metadata == {
        "OPENAI_API_KEY": "${OPENAI_API_KEY}",
        "ANTHROPIC_API_KEY": "env:ANTHROPIC_API_KEY",
        "JUDGE_TOKEN": "secret://vault/judge-token",
    }


def test_compile_redacts_credentials_in_storage_urls() -> None:
    experiment = _experiment(
        storage=StorageConfig(
            store="postgres",
            parameters={"url": "postgresql://themis:swordfish@db.example.com:5432/themis"},
        )
    )

    snapshot = experiment.compile()

    assert snapshot.provenance.storage.parameters["url"] == "postgresql://themis:<redacted>@db.example.com:5432/themis"
