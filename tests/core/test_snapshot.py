from __future__ import annotations

import json
from pathlib import Path

import pytest

from themis.catalog.registry import ComponentSpec
from themis.core.base import JSONValue
from themis.core.config import (
    EvaluationConfig,
    GenerationConfig,
    RuntimeConfig,
    Stage,
    StorageConfig,
)
from themis.core.dataset_sources import DatasetSourceSpec
from themis.core.experiment import Experiment
from themis.core.contexts import (
    GenerationContext,
    ParseContext,
    ReduceContext,
    ScoreContext,
)
from themis.core.models import (
    Case,
    Dataset,
    ParsedOutput,
    ReducedCandidate,
    MetricResult,
    MetricInterpretation,
    Candidate,
)
from themis.core.snapshot import BUILTIN_COMPONENT_REFS
from tests.release import CURRENT_VERSION


class DummyGenerator:
    component_id = "builtin/demo_generator"
    version = "1.0"

    def __init__(self, fingerprint_value: str) -> None:
        self.fingerprint_value = fingerprint_value

    def fingerprint(self) -> str:
        return self.fingerprint_value

    async def generate(self, case: Case, ctx: GenerationContext) -> Candidate:
        return Candidate(
            candidate_id=f"{case.case_id}-candidate", final_output={"seed": ctx.seed}
        )


class DummyReducer:
    component_id = "builtin/majority_vote"
    version = "1.0"

    def fingerprint(self) -> str:
        return "reducer-fingerprint"

    async def reduce(
        self,
        candidates: list[Candidate],
        ctx: ReduceContext,
    ) -> ReducedCandidate:
        return ReducedCandidate(
            candidate_id=f"{ctx.case_id}-reduced",
            source_candidate_ids=[candidate.candidate_id for candidate in candidates],
            final_output=candidates[0].final_output,
        )


class DummyParser:
    component_id = "builtin/json_identity"
    version = "1.0"

    def fingerprint(self) -> str:
        return "parser-fingerprint"

    def parse(self, candidate: ReducedCandidate, ctx: ParseContext) -> ParsedOutput:
        return ParsedOutput(
            value={"candidate_id": candidate.candidate_id, "run_id": ctx.run_id}
        )


class DummyMetric:
    interpretation = MetricInterpretation()
    component_id = "builtin/exact_match"
    version = "1.0"

    def fingerprint(self) -> str:
        return "metric-fingerprint"

    def score(
        self, parsed: ParsedOutput, case: Case, ctx: ScoreContext
    ) -> MetricResult:
        del ctx
        return MetricResult(
            metric_id="builtin/exact_match",
            value=float(parsed.value == case.expected_output),
        )


def _experiment(
    *,
    revision: str = "r1",
    generator: DummyGenerator | None = None,
    workflow_overrides: dict[str, JSONValue] | None = None,
) -> Experiment:
    return Experiment(
        generation=GenerationConfig(
            generator=generator or DummyGenerator("generator-fingerprint"),
            candidate_policy={"num_samples": 2},
            reducer=DummyReducer(),
        ),
        evaluation=EvaluationConfig(
            metrics=[DummyMetric()],
            parsers=[DummyParser()],
            judge_config={"panel_size": 1},
            workflow_overrides=workflow_overrides or {},
        ),
        storage=StorageConfig(target="sqlite", kwargs={"path": "runs/themis.sqlite3"}),
        runtime=RuntimeConfig(
            max_concurrent_tasks=16,
            stage_concurrency={Stage.GENERATE: 8},
            provider_concurrency={"openai:https://api.openai.com/v1": 4},
            provider_rate_limits={"openai:https://api.openai.com/v1": 120},
            store_retry_attempts=7,
            store_retry_delay=0.25,
        ),
        dataset_sources=[
            Dataset(
                dataset_id="dataset-1",
                cases=[
                    Case(
                        case_id="case-1", input={"question": "2+2"}, expected_output="4"
                    )
                ],
                revision=revision,
                metadata={"owner": "tests"},
            )
        ],
        seeds=[7, 11],
        environment_metadata={"env": "test"},
        themis_version=CURRENT_VERSION,
        python_version="3.12.9",
        platform="macos",
        git_commit="abc123",
        dependency_versions={"themis-eval": CURRENT_VERSION},
        provider_metadata={},
    )


def test_run_id_uses_identity_fields_only() -> None:
    compiled = _experiment().compile()
    changed_provenance = compiled.model_copy(
        update={
            "provenance": compiled.provenance.model_copy(
                update={
                    "platform": "linux",
                    "environment_metadata": {"env": "prod"},
                    "runtime": RuntimeConfig(max_concurrent_tasks=64),
                }
            )
        }
    )

    assert compiled.run_id == changed_provenance.run_id


def test_identity_changes_alter_run_id() -> None:
    first = _experiment(revision="r1").compile()
    second = _experiment(revision="r2").compile()

    assert first.run_id != second.run_id


def test_dataset_source_identity_tracks_source_fingerprint_and_materialization_receipt() -> (
    None
):
    experiment = Experiment(
        generation=GenerationConfig(
            generator="builtin/demo_generator",
            candidate_policy={"num_samples": 1},
            reducer="builtin/majority_vote",
        ),
        evaluation=EvaluationConfig(
            metrics=["builtin/exact_match"],
            parsers=["builtin/json_identity"],
        ),
        storage=StorageConfig(target="memory"),
        dataset_sources=[
            DatasetSourceSpec(
                target="inline",
                dataset_id="dataset-1",
                source_id="inline-fixture",
                source_revision="source-r1",
                source_fingerprint="source-fingerprint-v1",
                revision="dataset-r1",
                kwargs={
                    "cases": [
                        {
                            "case_id": "case-1",
                            "input": {"question": "2+2"},
                            "expected_output": {"answer": "4"},
                            "metadata": {"difficulty": "easy"},
                        }
                    ]
                },
                provenance_metadata={"source_kind": "inline"},
            )
        ],
        seeds=[7],
    )

    snapshot = experiment.compile()
    source_ref = snapshot.identity.dataset_source_refs[0]
    manifest = snapshot.dataset_manifests[0]

    assert source_ref.dataset_id == "dataset-1"
    assert source_ref.source_id == "inline-fixture"
    assert source_ref.source_revision == "source-r1"
    assert source_ref.source_fingerprint == "source-fingerprint-v1"
    assert manifest.source_id == "inline-fixture"
    assert manifest.source_revision == "source-r1"
    assert manifest.source_fingerprint == "source-fingerprint-v1"
    assert manifest.materialization_receipt["case_count"] == 1


def test_workflow_overrides_change_run_id() -> None:
    first = _experiment(workflow_overrides={"timeout": 10}).compile()
    second = _experiment(workflow_overrides={"timeout": 20}).compile()

    assert first.run_id != second.run_id


def test_component_fingerprints_are_frozen_at_compile_time() -> None:
    generator = DummyGenerator("fingerprint-before")
    compiled = _experiment(generator=generator).compile()

    generator.fingerprint_value = "fingerprint-after"

    assert compiled.component_refs.generator.fingerprint == "fingerprint-before"


def test_runtime_config_is_recorded_in_snapshot_provenance() -> None:
    compiled = _experiment().compile()

    assert compiled.provenance.runtime.max_concurrent_tasks == 16
    assert compiled.provenance.runtime.stage_concurrency == {"generate": 8}
    assert compiled.provenance.runtime.provider_rate_limits == {
        "openai:https://api.openai.com/v1": 120
    }


def test_storage_dsn_credentials_are_redacted_in_snapshot_provenance() -> None:
    compiled = Experiment(
        generation=GenerationConfig(
            generator="builtin/demo_generator",
            candidate_policy={"num_samples": 1},
            reducer="builtin/majority_vote",
        ),
        evaluation=EvaluationConfig(
            metrics=["builtin/exact_match"],
            parsers=["builtin/json_identity"],
        ),
        storage=StorageConfig(
            target="postgres",
            kwargs={"url": "postgresql://themis:swordfish@db.example.com:5432/themis"},
        ),
        dataset_sources=[
            Dataset(
                dataset_id="dataset-1",
                cases=[Case(case_id="case-1", input={"q": "2+2"})],
            )
        ],
    ).compile()

    assert (
        compiled.provenance.storage.kwargs["url"]
        == "postgresql://themis:<redacted>@db.example.com:5432/themis"
    )


def test_snapshot_serialization_matches_golden_file() -> None:
    compiled = _experiment().compile()
    golden_path = Path("tests/core/golden/run_snapshot_minimal.json")

    expected = json.loads(golden_path.read_text())

    assert compiled.model_dump(mode="json") == expected


def test_builtin_component_strings_resolve_to_registry_entries() -> None:
    experiment = Experiment(
        generation=GenerationConfig(
            generator="builtin/demo_generator",
            candidate_policy={"num_samples": 1},
            reducer="builtin/majority_vote",
        ),
        evaluation=EvaluationConfig(
            metrics=["builtin/exact_match"],
            parsers=["builtin/json_identity"],
            judge_config={"panel_size": 1},
        ),
        storage=StorageConfig(target="memory", kwargs={"path": ":memory:"}),
        dataset_sources=[
            Dataset(
                dataset_id="dataset-1",
                cases=[Case(case_id="case-1", input={"q": "2+2"})],
            )
        ],
    )

    snapshot = experiment.compile()

    assert (
        snapshot.component_refs.generator
        == BUILTIN_COMPONENT_REFS["builtin/demo_generator"]
    )
    assert (
        snapshot.component_refs.reducer
        == BUILTIN_COMPONENT_REFS["builtin/majority_vote"]
    )
    assert len(snapshot.component_refs.parsers) == 1
    assert snapshot.component_refs.parsers[0].id == "default"
    assert (
        snapshot.component_refs.parsers[0].parser
        == BUILTIN_COMPONENT_REFS["builtin/json_identity"]
    )
    assert [ref.component_id for ref in snapshot.component_refs.metrics] == [
        "builtin/exact_match"
    ]
    assert (
        snapshot.component_refs.metrics[0].interpretation.correctness_threshold == 1.0
    )


def test_unknown_builtin_component_strings_fail_fast() -> None:
    experiment = Experiment(
        generation=GenerationConfig(generator="generator/unknown"),
        evaluation=EvaluationConfig(metrics=["builtin/exact_match"]),
        storage=StorageConfig(target="memory"),
        dataset_sources=[
            Dataset(
                dataset_id="dataset-1",
                cases=[Case(case_id="case-1", input={"q": "2+2"})],
            )
        ],
    )

    with pytest.raises(ValueError, match="Unknown component"):
        experiment.compile()


def test_builtin_registry_changes_alter_component_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import themis.catalog.registry as registry

    original = registry.component_specs()["builtin/demo_generator"]
    original_manifest_specs = registry._manifest_component_specs
    first = Experiment(
        generation=GenerationConfig(generator="builtin/demo_generator"),
        evaluation=EvaluationConfig(metrics=["builtin/exact_match"]),
        storage=StorageConfig(target="memory"),
        dataset_sources=[
            Dataset(
                dataset_id="dataset-1",
                cases=[Case(case_id="case-1", input={"q": "2+2"})],
            )
        ],
    ).compile()

    def patched_manifest_specs() -> dict[str, ComponentSpec]:
        specs = original_manifest_specs()
        specs["builtin/demo_generator"] = ComponentSpec(
            component_id=original.component_id,
            kind=original.kind,
            target=original.target,
            version="2.0",
            fingerprint="generator-demo-fingerprint-v2",
        )
        return specs

    monkeypatch.setattr(registry, "_manifest_component_specs", patched_manifest_specs)

    second = Experiment(
        generation=GenerationConfig(generator="builtin/demo_generator"),
        evaluation=EvaluationConfig(metrics=["builtin/exact_match"]),
        storage=StorageConfig(target="memory"),
        dataset_sources=[
            Dataset(
                dataset_id="dataset-1",
                cases=[Case(case_id="case-1", input={"q": "2+2"})],
            )
        ],
    ).compile()

    assert first.component_refs.generator != second.component_refs.generator
    assert first.run_id != second.run_id
