from __future__ import annotations

from themis.records.candidate import CandidateRecord
from themis.records.error import ErrorRecord
from themis.records.evaluation import EvaluationRecord, MetricScore
from themis.records.extraction import ExtractionRecord
from themis.records.trial import TrialRecord
from themis.runtime.result_services import (
    ExperimentResultAnalysisService,
    ExperimentResultDiagnosticsService,
    ResultOverlayContext,
)
from themis.specs.experiment import InferenceParamsSpec, PromptTemplateSpec, TrialSpec
from themis.specs.foundational import DatasetSpec, GenerationSpec, ModelSpec, TaskSpec
from themis.types.enums import ErrorCode, ErrorWhere, RecordStatus, DatasetSource
from themis.types.events import ScoreRow, TrialSummaryRow


def _trial_spec(trial_id: str) -> TrialSpec:
    return TrialSpec(
        trial_id=trial_id,
        model=ModelSpec(model_id="gpt-4o-mini", provider="openai"),
        task=TaskSpec(
            task_id="math",
            dataset=DatasetSpec(source=DatasetSource.MEMORY),
            generation=GenerationSpec(),
        ),
        item_id="item-1",
        prompt=PromptTemplateSpec(id="baseline", messages=[]),
        params=InferenceParamsSpec(),
    )


def test_result_services_cover_analysis_and_diagnostics():
    trial_spec = _trial_spec("trial-service")
    candidate = CandidateRecord(
        spec_hash="candidate-service",
        candidate_id="candidate-service",
        sample_index=0,
        status=RecordStatus.ERROR,
        extractions=[
            ExtractionRecord(
                spec_hash="extract-service",
                extractor_id="json",
                success=False,
                failure_reason="missing field",
            )
        ],
        evaluation=EvaluationRecord(
            spec_hash="eval-service",
            metric_scores=[
                MetricScore(
                    metric_id="exact_match",
                    value=0.0,
                    details={"tags": ["format_error"]},
                )
            ],
            aggregate_scores={"exact_match": 0.0},
        ),
        error=ErrorRecord(
            code=ErrorCode.PARSE_ERROR,
            where=ErrorWhere.INFERENCE,
            message="bad candidate",
            retryable=False,
        ),
    )
    trial = TrialRecord(
        spec_hash=trial_spec.spec_hash,
        trial_spec=trial_spec,
        status=RecordStatus.ERROR,
        candidates=[candidate],
        error=ErrorRecord(
            code=ErrorCode.METRIC_COMPUTATION,
            where=ErrorWhere.METRIC,
            message="trial failed",
            retryable=False,
        ),
    )

    context = ResultOverlayContext(
        trial_hashes=[trial.spec_hash],
        transform_hashes=[],
        evaluation_hashes=["eval-service"],
        active_transform_hash=None,
        active_evaluation_hash="eval-service",
    )
    analysis = ExperimentResultAnalysisService(
        projection_repo=_ScoreRepo(
            [
                ScoreRow(
                    trial_hash=trial.spec_hash,
                    candidate_id="candidate-service",
                    metric_id="exact_match",
                    score=0.0,
                )
            ]
        ),
        context=context,
        iter_trials=lambda: iter([trial]),
        iter_trial_summaries=lambda: iter(
            [
                TrialSummaryRow(
                    trial_hash=trial.spec_hash,
                    model_id="gpt-4o-mini",
                    task_id="math",
                    item_id="item-1",
                    status=RecordStatus.ERROR,
                )
            ]
        ),
    )
    diagnostics = ExperimentResultDiagnosticsService(
        context=context,
        iter_trials=lambda: iter([trial]),
    )

    assert analysis.leaderboard() == [
        {
            "model_id": "gpt-4o-mini",
            "task_id": "math",
            "metric_id": "exact_match",
            "mean": 0.0,
            "count": 1,
            "min": 0.0,
            "max": 0.0,
        }
    ]
    failures = list(diagnostics.iter_failures())
    assert failures[0]["level"] == "trial"
    assert failures[1]["level"] == "candidate"
    assert list(diagnostics.iter_invalid_extractions())[0]["extractor_id"] == "json"
    assert list(diagnostics.iter_tagged_examples())[0]["tags"] == ["format_error"]


class _ScoreRepo:
    def __init__(self, score_rows: list[ScoreRow]) -> None:
        self.score_rows = score_rows

    def iter_candidate_scores(
        self,
        *,
        trial_hash: str | None = None,
        metric_id: str | None = None,
        evaluation_hash: str | None = None,
    ):
        del evaluation_hash
        for row in self.score_rows:
            if trial_hash is not None and row.trial_hash != trial_hash:
                continue
            if metric_id is not None and row.metric_id != metric_id:
                continue
            yield row
