from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from themis.records.candidate import CandidateRecord
from themis.records.extraction import ExtractionRecord
from themis.records.inference import InferenceRecord
from themis.records.trial import TrialRecord
from themis.runtime.benchmark_result import BenchmarkResult
from themis.runtime.corpus_metrics import SUPPORTED_CORPUS_METRIC_IDS
from themis.specs.experiment import InferenceParamsSpec, PromptTemplateSpec, TrialSpec
from themis.specs.foundational import DatasetSpec, GenerationSpec, ModelSpec, TaskSpec
from themis.types.enums import DatasetSource
from themis.types.json_types import JSONDict
from themis.types.enums import PValueCorrection
from themis.types.enums import RecordStatus
from themis.types.events import ScoreRow, TraceScoreRow, TrialSummaryRow


class FakeProjectionRepository:
    def __init__(self) -> None:
        self.score_calls: list[dict[str, object]] = []
        self.trace_score_calls: list[dict[str, object]] = []
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
        self.trace_score_rows = [
            TraceScoreRow(
                trial_hash="trial-1",
                trace_id="cand-1",
                trace_scope="candidate_trace",
                trace_score_hash="trace-score-1",
                metric_id="tool_presence",
                score=1.0,
                details={"tool_name": "search"},
            ),
            TraceScoreRow(
                trial_hash="trial-2",
                trace_id="cand-2",
                trace_scope="candidate_trace",
                trace_score_hash="trace-score-1",
                metric_id="tool_presence",
                score=0.0,
                details={"tool_name": "search"},
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

    def iter_trace_scores(self, **kwargs):
        self.trace_score_calls.append(dict(kwargs))
        trial_hashes = kwargs.get("trial_hashes")
        metric_id = kwargs.get("metric_id")
        trace_score_hash = kwargs.get("trace_score_hash")
        for row in self.trace_score_rows:
            if trial_hashes is not None and row.trial_hash not in trial_hashes:
                continue
            if metric_id is not None and row.metric_id != metric_id:
                continue
            if (
                trace_score_hash is not None
                and row.trace_score_hash != trace_score_hash
            ):
                continue
            yield row


def _trial_record(
    *,
    trial_hash: str,
    item_id: str,
    raw_text: str,
    answer: str,
    parsed_answer: object | None = None,
) -> TrialRecord:
    trial_spec = TrialSpec(
        trial_id=trial_hash,
        model=ModelSpec(model_id="model-a", provider="demo"),
        task=TaskSpec(
            task_id="classification-task",
            dataset=DatasetSpec(source=DatasetSource.MEMORY),
            generation=GenerationSpec(),
        ),
        item_id=item_id,
        prompt=PromptTemplateSpec(id="baseline", messages=[]),
        params=InferenceParamsSpec(),
    )
    extractions = []
    if parsed_answer is not None:
        extractions.append(
            ExtractionRecord(
                spec_hash=f"{trial_hash}-extract",
                extractor_id="parsed_label",
                success=True,
                parsed_answer=parsed_answer,
            )
        )
    return TrialRecord(
        spec_hash=trial_hash,
        trial_spec=trial_spec,
        status=RecordStatus.OK,
        candidates=[
            CandidateRecord(
                spec_hash=f"{trial_hash}-candidate-0",
                candidate_id=f"{trial_hash}-candidate-0",
                sample_index=0,
                status=RecordStatus.OK,
                inference=InferenceRecord(
                    spec_hash=f"{trial_hash}-inference-0",
                    raw_text=raw_text,
                ),
                extractions=extractions,
            )
        ],
    )


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


def test_benchmark_result_aggregate_trace_groups_by_semantic_fields() -> None:
    repo = FakeProjectionRepository()
    result = BenchmarkResult(
        projection_repo=repo,
        trial_hashes=["trial-1", "trial-2"],
        benchmark_id="bench-1",
        slice_ids=["slice-qa"],
        prompt_variant_ids=["qa-default"],
        trace_score_hashes=["trace-score-1"],
    )

    rows = result.aggregate_trace(group_by=["model_id", "slice_id", "metric_id"])

    assert rows == [
        {
            "count": 2,
            "mean": 0.5,
            "metric_id": "tool_presence",
            "model_id": "model-a",
            "slice_id": "slice-qa",
        }
    ]
    assert repo.trace_score_calls[-1]["trial_hashes"] == ["trial-1", "trial-2"]
    assert repo.trace_score_calls[-1]["evaluation_hash"] is None


def test_benchmark_result_aggregate_trace_scopes_to_selected_evaluation() -> None:
    repo = FakeProjectionRepository()
    result = BenchmarkResult(
        projection_repo=repo,
        trial_hashes=["trial-1", "trial-2"],
        evaluation_hashes=["evaluation-1"],
        benchmark_id="bench-1",
    )

    list(result.for_evaluation("evaluation-1").iter_trace_scores())

    assert repo.trace_score_calls[-1]["evaluation_hash"] == "evaluation-1"


def test_benchmark_result_aggregate_corpus_uses_parsed_answer_then_raw_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeSklearnMetrics:
        @staticmethod
        def f1_score(expected_labels, predicted_labels, *, average, zero_division):
            del average, zero_division
            correct = sum(
                expected == predicted
                for expected, predicted in zip(
                    expected_labels,
                    predicted_labels,
                    strict=True,
                )
            )
            return correct / len(expected_labels)

    class CorpusProjectionRepository(FakeProjectionRepository):
        def __init__(self) -> None:
            super().__init__()
            self.trials = {
                "trial-1": _trial_record(
                    trial_hash="trial-1",
                    item_id="item-1",
                    raw_text="wrong raw answer",
                    parsed_answer="cat",
                    answer="cat",
                ),
                "trial-2": _trial_record(
                    trial_hash="trial-2",
                    item_id="item-2",
                    raw_text="dog",
                    answer="dog",
                ),
            }
            self.timeline_views = {
                "trial-1": SimpleNamespace(item_payload={"answer": "cat"}),
                "trial-2": SimpleNamespace(item_payload={"answer": "dog"}),
            }

        def get_trial_record(self, trial_hash: str, **kwargs):
            del kwargs
            return self.trials.get(trial_hash)

        def get_timeline_view(self, record_id: str, *args, **kwargs):
            del args, kwargs
            return self.timeline_views.get(record_id)

    monkeypatch.setattr(
        "themis.runtime.corpus_metrics.import_optional",
        lambda module_name, *, extra: FakeSklearnMetrics(),
    )
    result = BenchmarkResult(
        projection_repo=CorpusProjectionRepository(),
        trial_hashes=["trial-1", "trial-2"],
        benchmark_id="bench-1",
        slice_ids=["slice-qa"],
        prompt_variant_ids=["qa-default"],
    )

    rows = result.aggregate_corpus(
        group_by=["model_id", "slice_id"],
        metric_id="f1_micro",
        candidate_selector="anchor_candidate",
    )

    assert rows == [
        {
            "count": 2,
            "metric_id": "f1_micro",
            "model_id": "model-a",
            "score": 1.0,
            "slice_id": "slice-qa",
        }
    ]


def test_benchmark_result_aggregate_corpus_supports_classification_metric_suite() -> (
    None
):
    assert SUPPORTED_CORPUS_METRIC_IDS == {
        "f1_micro",
        "f1_macro",
        "f1_weighted",
        "precision_micro",
        "precision_macro",
        "precision_weighted",
        "recall_micro",
        "recall_macro",
        "recall_weighted",
        "cohen_kappa",
    }


def test_benchmark_result_aggregate_corpus_requires_candidate_selector() -> None:
    result = BenchmarkResult(
        projection_repo=FakeProjectionRepository(),
        trial_hashes=["trial-1"],
        benchmark_id="bench-1",
    )

    with pytest.raises(ValueError, match="candidate_selector"):
        result.aggregate_corpus(group_by=["model_id"], metric_id="f1_micro")


def test_benchmark_result_aggregate_corpus_rejects_unknown_candidate_selector() -> None:
    result = BenchmarkResult(
        projection_repo=FakeProjectionRepository(),
        trial_hashes=["trial-1"],
        benchmark_id="bench-1",
    )

    with pytest.raises(ValueError, match="anchor_candidate"):
        result.aggregate_corpus(
            group_by=["model_id"],
            metric_id="f1_micro",
            candidate_selector="all_candidates",
        )


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
