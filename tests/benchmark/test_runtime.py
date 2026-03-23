from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from themis.runtime.benchmark_result import BenchmarkResult
from themis.types.json_types import JSONDict
from themis.types.enums import PValueCorrection
from themis.types.enums import RecordStatus
from themis.types.events import ScoreRow, TrialSummaryRow


class FakeProjectionRepository:
    def __init__(self) -> None:
        self.score_calls: list[dict[str, object]] = []
        self.score_rows = [
            ScoreRow(
                trial_hash="trial-1",
                candidate_id="cand-1",
                metric_id="accuracy",
                score=1.0,
                details={},
            ),
            ScoreRow(
                trial_hash="trial-2",
                candidate_id="cand-2",
                metric_id="accuracy",
                score=0.0,
                details={},
            ),
            ScoreRow(
                trial_hash="trial-ignored",
                candidate_id="cand-ignored",
                metric_id="accuracy",
                score=0.25,
                details={},
            ),
        ]
        self.summaries = [
            TrialSummaryRow(
                trial_hash="trial-1",
                model_id="model-a",
                slice_id="slice-qa",
                item_id="item-1",
                status=RecordStatus.OK,
                benchmark_id="bench-1",
                prompt_variant_id="qa-default",
                dimensions={"source": "medqa", "format": "qa"},
            ),
            TrialSummaryRow(
                trial_hash="trial-2",
                model_id="model-a",
                slice_id="slice-qa",
                item_id="item-2",
                status=RecordStatus.OK,
                benchmark_id="bench-1",
                prompt_variant_id="qa-default",
                dimensions={"source": "medqa", "format": "qa"},
            ),
        ]

    def get_trial_record(self, *args, **kwargs):
        return None

    def get_timeline_view(self, *args, **kwargs):
        return None

    def iter_candidate_scores(self, **kwargs):
        self.score_calls.append(dict(kwargs))
        trial_hashes = kwargs.get("trial_hashes")
        metric_id = kwargs.get("metric_id")
        for row in self.score_rows:
            if trial_hashes is not None and row.trial_hash not in trial_hashes:
                continue
            if metric_id is not None and row.metric_id != metric_id:
                continue
            yield row

    def iter_trial_summaries(self, **kwargs):
        trial_hashes = kwargs.get("trial_hashes")
        for row in self.summaries:
            if trial_hashes is not None and row.trial_hash not in trial_hashes:
                continue
            yield row


def test_benchmark_result_aggregate_groups_by_semantic_fields(tmp_path: Path) -> None:
    result = BenchmarkResult(
        projection_repo=FakeProjectionRepository(),
        trial_hashes=["trial-1", "trial-2"],
        benchmark_id="bench-1",
        slice_ids=["slice-qa"],
        prompt_variant_ids=["qa-default"],
    )

    rows = result.aggregate(group_by=["model_id", "slice_id", "metric_id", "source"])

    assert rows == [
        {
            "count": 2,
            "mean": 0.5,
            "metric_id": "accuracy",
            "model_id": "model-a",
            "slice_id": "slice-qa",
            "source": "medqa",
        }
    ]


def test_benchmark_result_persist_artifacts_writes_bundle(tmp_path: Path) -> None:
    result = BenchmarkResult(
        projection_repo=FakeProjectionRepository(),
        trial_hashes=["trial-1", "trial-2"],
        benchmark_id="bench-1",
        slice_ids=["slice-qa"],
        prompt_variant_ids=["qa-default"],
    )

    bundle = result.persist_artifacts(storage_root=tmp_path)

    assert bundle.aggregate_json_path.exists()
    assert bundle.summary_markdown_path.exists()
    payload = json.loads(bundle.aggregate_json_path.read_text())
    assert payload["scope"] == {"overlay_key": "gen"}
    assert "scope=gen" in bundle.summary_markdown_path.read_text()


def test_benchmark_result_persist_artifacts_uses_unique_scope_paths(
    tmp_path: Path,
) -> None:
    result = BenchmarkResult(
        projection_repo=FakeProjectionRepository(),
        trial_hashes=["trial-1", "trial-2"],
        transform_hashes=["transform-1"],
        evaluation_hashes=["evaluation-1"],
        benchmark_id="bench-1",
        slice_ids=["slice-qa"],
        prompt_variant_ids=["qa-default"],
    )

    generation_bundle = result.persist_artifacts(storage_root=tmp_path)
    evaluation_bundle = result.for_evaluation("evaluation-1").persist_artifacts(
        storage_root=tmp_path
    )

    assert (
        generation_bundle.aggregate_json_path != evaluation_bundle.aggregate_json_path
    )
    assert (
        generation_bundle.summary_markdown_path
        != evaluation_bundle.summary_markdown_path
    )
    assert generation_bundle.aggregate_json_path.name == "benchmark-aggregate-gen.json"
    assert evaluation_bundle.aggregate_json_path.name == (
        "benchmark-aggregate-ev-evaluation-1.json"
    )
    evaluation_payload = json.loads(evaluation_bundle.aggregate_json_path.read_text())
    assert evaluation_payload["scope"] == {
        "overlay_key": "ev:evaluation-1",
        "evaluation_hash": "evaluation-1",
    }


def test_benchmark_result_export_json_scopes_scores_to_active_trial_hashes() -> None:
    repo = FakeProjectionRepository()
    result = BenchmarkResult(
        projection_repo=repo,
        trial_hashes=["trial-1", "trial-2"],
        benchmark_id="bench-1",
        slice_ids=["slice-qa"],
        prompt_variant_ids=["qa-default"],
    )

    payload = result.export_json(include_trials=False)
    score_rows = cast(list[JSONDict], payload["score_rows"])

    assert {row["trial_hash"] for row in score_rows} == {
        "trial-1",
        "trial-2",
    }
    assert repo.score_calls[-1]["trial_hashes"] == ["trial-1", "trial-2"]


def test_benchmark_result_overlay_views_preserve_benchmark_metadata() -> None:
    result = BenchmarkResult(
        projection_repo=FakeProjectionRepository(),
        trial_hashes=["trial-1", "trial-2"],
        transform_hashes=["transform-1"],
        evaluation_hashes=["evaluation-1"],
        benchmark_id="bench-1",
        slice_ids=["slice-qa"],
        prompt_variant_ids=["qa-default"],
        scan_stats={"loaded_count": 2},
    )

    transform_view = result.for_transform("transform-1")
    evaluation_view = result.for_evaluation("evaluation-1")

    assert isinstance(transform_view, BenchmarkResult)
    assert transform_view.benchmark_id == "bench-1"
    assert transform_view.active_transform_hash == "transform-1"
    assert transform_view.scan_stats == {"loaded_count": 2}
    assert isinstance(evaluation_view, BenchmarkResult)
    assert evaluation_view.benchmark_id == "bench-1"
    assert evaluation_view.active_evaluation_hash == "evaluation-1"
    assert evaluation_view.scan_stats == {"loaded_count": 2}


def test_benchmark_result_aggregate_handles_missing_dimension_values() -> None:
    repo = FakeProjectionRepository()
    repo.score_rows.append(
        ScoreRow(
            trial_hash="trial-3",
            candidate_id="cand-3",
            metric_id="accuracy",
            score=0.5,
            details={},
        )
    )
    repo.summaries.append(
        TrialSummaryRow(
            trial_hash="trial-3",
            model_id="model-a",
            slice_id="slice-qa",
            item_id="item-3",
            status=RecordStatus.OK,
            benchmark_id="bench-1",
            prompt_variant_id="qa-default",
            dimensions={"format": "qa"},
        )
    )
    result = BenchmarkResult(
        projection_repo=repo,
        trial_hashes=["trial-1", "trial-2", "trial-3"],
        benchmark_id="bench-1",
        slice_ids=["slice-qa"],
        prompt_variant_ids=["qa-default"],
    )

    rows = result.aggregate(group_by=["source", "metric_id"])

    assert rows == [
        {
            "count": 1,
            "mean": 0.5,
            "metric_id": "accuracy",
            "source": None,
        },
        {
            "count": 2,
            "mean": 0.5,
            "metric_id": "accuracy",
            "source": "medqa",
        },
    ]


def test_benchmark_result_aggregate_rejects_unknown_group_by_keys() -> None:
    result = BenchmarkResult(
        projection_repo=FakeProjectionRepository(),
        trial_hashes=["trial-1", "trial-2"],
        benchmark_id="bench-1",
        slice_ids=["slice-qa"],
        prompt_variant_ids=["qa-default"],
    )

    with pytest.raises(ValueError, match="Unsupported group_by key"):
        result.aggregate(group_by=["sice_id"])


@pytest.mark.filterwarnings("ignore:.*stats.*:UserWarning")
def test_benchmark_result_paired_compare_uses_statistical_engine() -> None:
    class ComparisonProjectionRepository(FakeProjectionRepository):
        def __init__(self) -> None:
            self.score_calls = []
            self.score_rows = [
                ScoreRow(
                    trial_hash="trial-a1",
                    candidate_id="cand-a1",
                    metric_id="accuracy",
                    score=0.0,
                    details={},
                ),
                ScoreRow(
                    trial_hash="trial-b1",
                    candidate_id="cand-b1",
                    metric_id="accuracy",
                    score=1.0,
                    details={},
                ),
                ScoreRow(
                    trial_hash="trial-a2",
                    candidate_id="cand-a2",
                    metric_id="accuracy",
                    score=0.0,
                    details={},
                ),
                ScoreRow(
                    trial_hash="trial-b2",
                    candidate_id="cand-b2",
                    metric_id="accuracy",
                    score=1.0,
                    details={},
                ),
            ]
            self.summaries = [
                TrialSummaryRow(
                    trial_hash="trial-a1",
                    model_id="model-a",
                    task_id="slice-qa",
                    slice_id="slice-qa",
                    item_id="item-1",
                    status=RecordStatus.OK,
                    benchmark_id="bench-1",
                    prompt_variant_id="qa-default",
                    dimensions={},
                ),
                TrialSummaryRow(
                    trial_hash="trial-b1",
                    model_id="model-b",
                    task_id="slice-qa",
                    slice_id="slice-qa",
                    item_id="item-1",
                    status=RecordStatus.OK,
                    benchmark_id="bench-1",
                    prompt_variant_id="qa-default",
                    dimensions={},
                ),
                TrialSummaryRow(
                    trial_hash="trial-a2",
                    model_id="model-a",
                    task_id="slice-qa",
                    slice_id="slice-qa",
                    item_id="item-2",
                    status=RecordStatus.OK,
                    benchmark_id="bench-1",
                    prompt_variant_id="qa-default",
                    dimensions={},
                ),
                TrialSummaryRow(
                    trial_hash="trial-b2",
                    model_id="model-b",
                    task_id="slice-qa",
                    slice_id="slice-qa",
                    item_id="item-2",
                    status=RecordStatus.OK,
                    benchmark_id="bench-1",
                    prompt_variant_id="qa-default",
                    dimensions={},
                ),
            ]

    result = BenchmarkResult(
        projection_repo=ComparisonProjectionRepository(),
        trial_hashes=["trial-a1", "trial-b1", "trial-a2", "trial-b2"],
        benchmark_id="bench-1",
        slice_ids=["slice-qa"],
        prompt_variant_ids=["qa-default"],
    )

    rows = result.paired_compare(
        metric_id="accuracy",
        baseline_model_id="model-a",
        treatment_model_id="model-b",
        p_value_correction=PValueCorrection.HOLM,
    )

    assert len(rows) == 1
    assert rows[0]["slice_id"] == "slice-qa"
    assert rows[0]["baseline_model_id"] == "model-a"
    assert rows[0]["treatment_model_id"] == "model-b"
    assert rows[0]["pair_count"] == 2
    assert "p_value" in rows[0]
    assert "adjusted_p_value" in rows[0]
    assert "ci_lower" in rows[0]
    assert "ci_upper" in rows[0]
    assert rows[0]["adjustment_method"] == PValueCorrection.HOLM
