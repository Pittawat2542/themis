from __future__ import annotations

import sqlite3

import pytest

from themis.core.config import EvidenceRetention, RuntimeConfig
from themis.core.config import EvaluationConfig, GenerationConfig, StorageConfig
from themis.core.experiment import Experiment
from themis.core.events import RunStartedEvent
from themis.core.models import (
    MetricDirection,
    MetricInterpretation,
    MetricResult,
    Case,
    Candidate,
    Dataset,
    ScoreOutcome,
)
from themis.core.projections import _metric_value_error, _score_outcome
from themis.core.security import EvidenceSanitizer
from themis.core.store import ProjectionConsistency, ProjectionFreshness
from themis.core.stores.memory import InMemoryRunStore
from themis.core.stores.sqlite import SqliteRunStore


class UnsupportedSeedGenerator:
    component_id = "generator/no-seed"
    version = "1.0"

    def fingerprint(self) -> str:
        return "no-seed"

    async def generate(self, case, ctx) -> Candidate:
        return Candidate(candidate_id=case.case_id, final_output=case.input)


def test_continuous_and_lower_is_better_metric_outcomes_follow_contract() -> None:
    continuous = MetricInterpretation(
        direction=MetricDirection.HIGHER_IS_BETTER,
        valid_range=(0.0, 1.0),
    )
    lower_binary = MetricInterpretation(
        direction=MetricDirection.LOWER_IS_BETTER,
        valid_range=(0.0, 10.0),
        correctness_threshold=2.0,
    )

    assert _score_outcome(MetricResult(metric_id="continuous", value=0.4), continuous) is ScoreOutcome.SCORED
    assert _score_outcome(MetricResult(metric_id="error", value=1.5), lower_binary) is ScoreOutcome.CORRECT
    assert _score_outcome(MetricResult(metric_id="error", value=2.5), lower_binary) is ScoreOutcome.INCORRECT
    assert _metric_value_error(MetricResult(metric_id="continuous", value=2.0), continuous)


def test_evidence_sanitizer_redacts_secrets_and_applies_retention() -> None:
    standard = EvidenceSanitizer(EvidenceRetention.STANDARD).value(
        {
            "authorization": "Bearer secret",
            "raw_response": {"answer": "sensitive"},
            "result": {"final_output": "answer"},
        }
    )
    minimal = EvidenceSanitizer(EvidenceRetention.MINIMAL).value(
        {"result": {"final_output": "answer"}}
    )

    assert standard == {
        "authorization": "<redacted>",
        "raw_response": {},
        "result": {"final_output": "answer"},
    }
    assert minimal == {"result": {}}


def test_memory_append_is_idempotent_and_projection_freshness_is_explicit() -> None:
    store = InMemoryRunStore()
    event = RunStartedEvent(run_id="run-1")

    first = store.persist_event(event)
    duplicate = store.persist_event(event)

    assert first.inserted is True
    assert duplicate == first.model_copy(update={"inserted": False})
    assert store.count_events("run-1") == 1
    read = store.read_projection(
        "run-1", "benchmark_result", consistency=ProjectionConsistency.EVENTUAL
    )
    assert read.freshness is ProjectionFreshness.MISSING


def test_sqlite_rejects_schema_v1_store(tmp_path) -> None:
    path = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE run_events (id INTEGER PRIMARY KEY, run_id TEXT, event_type TEXT, event_json TEXT)"
        )

    with pytest.raises(RuntimeError, match="schema-v1"):
        SqliteRunStore(path).initialize()


def test_runtime_defaults_to_standard_evidence_and_unlimited_rpm() -> None:
    runtime = RuntimeConfig()

    assert runtime.evidence_retention is EvidenceRetention.STANDARD
    assert runtime.provider_rate_limits == {}
    assert runtime.strict_determinism is False


def test_strict_determinism_rejects_unsupported_provider_before_run() -> None:
    experiment = Experiment(
        generation=GenerationConfig(generator=UnsupportedSeedGenerator()),
        evaluation=EvaluationConfig(
            metrics=["builtin/exact_match"],
            parsers=["builtin/json_identity"],
        ),
        storage=StorageConfig(),
        runtime=RuntimeConfig(strict_determinism=True),
        dataset_sources=[
            Dataset(
                dataset_id="dataset",
                cases=[Case(case_id="case", input="answer", expected_output="answer")],
            )
        ],
        seeds=[7],
    )

    with pytest.raises(ValueError, match="seed-capable"):
        experiment.compile()
