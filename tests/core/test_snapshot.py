from __future__ import annotations

import json
from pathlib import Path

import pytest

from themis.core.config import EvaluationConfig, GenerationConfig, RuntimeConfig, StorageConfig
from themis.core.experiment import Experiment
from themis.core.contexts import GenerateContext, ParseContext, ReduceContext, ScoreContext
from themis.core.models import Case, Dataset, GenerationResult, ParsedOutput, ReducedCandidate, Score
from themis.core.snapshot import BUILTIN_COMPONENT_REFS, ComponentRef


class DummyGenerator:
    component_id = "generator/demo"
    version = "1.0"

    def __init__(self, fingerprint_value: str) -> None:
        self.fingerprint_value = fingerprint_value

    def fingerprint(self) -> str:
        return self.fingerprint_value

    async def generate(self, case: Case, ctx: GenerateContext) -> GenerationResult:
        return GenerationResult(candidate_id=f"{case.case_id}-candidate", final_output={"seed": ctx.seed})


class DummyReducer:
    component_id = "reducer/demo"
    version = "1.0"

    def fingerprint(self) -> str:
        return "reducer-fingerprint"

    def reduce(
        self,
        candidates: list[GenerationResult],
        ctx: ReduceContext,
    ) -> ReducedCandidate:
        return ReducedCandidate(
            candidate_id=f"{ctx.case_id}-reduced",
            source_candidate_ids=[candidate.candidate_id for candidate in candidates],
            final_output=candidates[0].final_output,
        )


class DummyParser:
    component_id = "parser/demo"
    version = "1.0"

    def fingerprint(self) -> str:
        return "parser-fingerprint"

    def parse(self, candidate: ReducedCandidate, ctx: ParseContext) -> ParsedOutput:
        return ParsedOutput(value={"candidate_id": candidate.candidate_id, "run_id": ctx.run_id})


class DummyMetric:
    component_id = "metric/demo"
    version = "1.0"

    def fingerprint(self) -> str:
        return "metric-fingerprint"

    def score(self, parsed: ParsedOutput, case: Case, ctx: ScoreContext) -> Score:
        del ctx
        return Score(metric_id="metric/demo", value=float(parsed.value == case.expected_output))


def _experiment(
    *,
    revision: str = "r1",
    generator: DummyGenerator | None = None,
    workflow_overrides: dict[str, object] | None = None,
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
        storage=StorageConfig(store="sqlite", parameters={"path": "runs/themis.sqlite3"}),
        runtime=RuntimeConfig(
            max_concurrent_tasks=16,
            stage_concurrency={"generation": 8},
            provider_concurrency={"openai:https://api.openai.com/v1": 4},
            provider_rate_limits={"openai:https://api.openai.com/v1": 120},
            store_retry_attempts=7,
            store_retry_delay=0.25,
        ),
        datasets=[
            Dataset(
                dataset_id="dataset-1",
                cases=[Case(case_id="case-1", input={"question": "2+2"}, expected_output="4")],
                revision=revision,
                metadata={"owner": "tests"},
            )
        ],
        seeds=[7, 11],
        environment_metadata={"env": "test"},
        themis_version="4.0.0a0",
        python_version="3.12.9",
        platform="macos",
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
    assert compiled.provenance.runtime.stage_concurrency == {"generation": 8}
    assert compiled.provenance.runtime.provider_rate_limits == {
        "openai:https://api.openai.com/v1": 120
    }


def test_snapshot_serialization_matches_golden_file() -> None:
    compiled = _experiment().compile()
    golden_path = Path("tests/core/golden/run_snapshot_minimal.json")

    expected = json.loads(golden_path.read_text())

    assert compiled.model_dump(mode="json") == expected


def test_builtin_component_strings_resolve_to_registry_entries() -> None:
    experiment = Experiment(
        generation=GenerationConfig(
            generator="generator/demo",
            candidate_policy={"num_samples": 1},
            reducer="reducer/demo",
        ),
        evaluation=EvaluationConfig(
            metrics=["metric/demo"],
            parsers=["parser/demo"],
            judge_config={"panel_size": 1},
        ),
        storage=StorageConfig(store="memory", parameters={"path": ":memory:"}),
        datasets=[Dataset(dataset_id="dataset-1", cases=[Case(case_id="case-1", input={"q": "2+2"})])],
    )

    snapshot = experiment.compile()

    assert snapshot.component_refs.generator == BUILTIN_COMPONENT_REFS["generator/demo"]
    assert snapshot.component_refs.reducer == BUILTIN_COMPONENT_REFS["reducer/demo"]
    assert snapshot.component_refs.parsers == [BUILTIN_COMPONENT_REFS["parser/demo"]]
    assert snapshot.component_refs.metrics == [BUILTIN_COMPONENT_REFS["metric/demo"]]


def test_unknown_builtin_component_strings_fail_fast() -> None:
    experiment = Experiment(
        generation=GenerationConfig(generator="generator/unknown"),
        evaluation=EvaluationConfig(metrics=["metric/demo"]),
        storage=StorageConfig(store="memory"),
        datasets=[Dataset(dataset_id="dataset-1", cases=[Case(case_id="case-1", input={"q": "2+2"})])],
    )

    with pytest.raises(ValueError, match="Unknown builtin component"):
        experiment.compile()


def test_builtin_registry_changes_alter_component_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = BUILTIN_COMPONENT_REFS["generator/demo"]
    first = Experiment(
        generation=GenerationConfig(generator="generator/demo"),
        evaluation=EvaluationConfig(metrics=["metric/demo"]),
        storage=StorageConfig(store="memory"),
        datasets=[Dataset(dataset_id="dataset-1", cases=[Case(case_id="case-1", input={"q": "2+2"})])],
    ).compile()

    monkeypatch.setitem(
        BUILTIN_COMPONENT_REFS,
        "generator/demo",
        ComponentRef(
            component_id=original.component_id,
            version="2.0",
            fingerprint="generator-demo-fingerprint-v2",
        ),
    )

    second = Experiment(
        generation=GenerationConfig(generator="generator/demo"),
        evaluation=EvaluationConfig(metrics=["metric/demo"]),
        storage=StorageConfig(store="memory"),
        datasets=[Dataset(dataset_id="dataset-1", cases=[Case(case_id="case-1", input={"q": "2+2"})])],
    ).compile()

    assert first.component_refs.generator != second.component_refs.generator
    assert first.run_id != second.run_id
