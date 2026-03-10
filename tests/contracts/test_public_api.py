from __future__ import annotations

from datetime import datetime, timezone

import themis
import themis.specs

from themis.records.candidate import CandidateRecord
from themis.records.evaluation import EvaluationRecord, MetricScore
from themis.records.timeline import RecordTimeline, TimelineStageRecord
from themis.records.trial import TrialRecord
from themis.runtime import ComparisonTable, ExperimentResult, RecordTimelineView
from themis.specs.experiment import TrialSpec, PromptTemplateSpec, InferenceParamsSpec
from themis.specs.foundational import DatasetSpec, ModelSpec, TaskSpec
from themis.storage.events import ScoreRow, TrialSummaryRow
from themis.types.enums import RecordStatus


EXPECTED_ROOT_EXPORTS = {
    "__version__",
    "Orchestrator",
    "ExperimentResult",
    "ProjectSpec",
    "ExperimentSpec",
    "StorageSpec",
    "ExecutionPolicySpec",
    "InferenceGridSpec",
    "InferenceParamsSpec",
    "ItemSamplingSpec",
    "PromptMessage",
    "PromptTemplateSpec",
    "TrialSpec",
    "ModelSpec",
    "ExtractorRefSpec",
    "ExtractorChainSpec",
    "TaskSpec",
    "DatasetSpec",
    "RuntimeContext",
    "PluginRegistry",
    "ThemisError",
}


def _make_trial_record() -> TrialRecord:
    trial_spec = TrialSpec(
        trial_id="trial_1",
        model=ModelSpec(model_id="gpt-4o-mini", provider="openai"),
        task=TaskSpec(
            task_id="math",
            dataset=DatasetSpec(source="memory"),
            default_metrics=["exact_match"],
        ),
        item_id="item-1",
        prompt=PromptTemplateSpec(id="baseline", messages=[]),
        params=InferenceParamsSpec(),
    )
    candidate = CandidateRecord(
        spec_hash="candidate_1",
        evaluation=EvaluationRecord(
            spec_hash="candidate_1",
            metric_scores=[MetricScore(metric_id="exact_match", value=1.0)],
        ),
    )
    return TrialRecord(
        spec_hash=trial_spec.spec_hash,
        status=RecordStatus.OK,
        candidates=[candidate],
        trial_spec=trial_spec,
    )


def _make_timeline_view(trial: TrialRecord) -> RecordTimelineView:
    timeline = RecordTimeline(
        record_id="candidate_1",
        record_type="candidate",
        trial_hash=trial.spec_hash,
        candidate_id="candidate_1",
        item_id=trial.trial_spec.item_id,
        stages=[
            TimelineStageRecord(
                stage="prompt_render",
                status=RecordStatus.OK,
                started_at=datetime.now(timezone.utc),
                ended_at=datetime.now(timezone.utc),
                duration_ms=1,
            )
        ],
        source_event_range=(1, 2),
    )
    return RecordTimelineView(
        record_id="candidate_1",
        record_type="candidate",
        trial_hash=trial.spec_hash,
        candidate_id="candidate_1",
        lineage={
            "task": trial.trial_spec.task.task_id,
            "model": trial.trial_spec.model.model_id,
        },
        trial_spec=trial.trial_spec,
        timeline=timeline,
        extractions=[],
        related_events=[],
    )


class FakeProjectionRepository:
    def __init__(self, trial: TrialRecord, view: RecordTimelineView):
        self.trial = trial
        self.view = view
        self.trial_summaries = [
            TrialSummaryRow(
                trial_hash=trial.spec_hash,
                model_id=trial.trial_spec.model.model_id,
                task_id=trial.trial_spec.task.task_id,
                item_id=trial.trial_spec.item_id,
                status=trial.status,
            )
        ]
        self.score_rows = [
            ScoreRow(
                trial_hash=trial.spec_hash,
                candidate_id="candidate_1",
                metric_id="exact_match",
                score=1.0,
            )
        ]

    def get_trial_record(
        self, trial_hash: str, eval_revision: str
    ) -> TrialRecord | None:
        if trial_hash == self.trial.spec_hash:
            return self.trial
        return None

    def get_timeline_view(
        self,
        record_id: str,
        record_type: str,
        eval_revision: str,
    ) -> RecordTimelineView | None:
        if record_id == self.view.record_id and record_type == self.view.record_type:
            return self.view
        return None

    def iter_candidate_scores(self, **_: object):
        return iter(self.score_rows)

    def iter_trial_summaries(self, **_: object):
        return iter(self.trial_summaries)


class PairwiseProjectionRepository:
    def __init__(self, trials: list[TrialRecord], score_rows: list[ScoreRow]):
        self._trials = {trial.spec_hash: trial for trial in trials}
        self._score_rows = list(score_rows)
        self._trial_summaries = [
            TrialSummaryRow(
                trial_hash=trial.spec_hash,
                model_id=trial.trial_spec.model.model_id,
                task_id=trial.trial_spec.task.task_id,
                item_id=trial.trial_spec.item_id,
                status=trial.status,
            )
            for trial in trials
        ]

    def get_trial_record(
        self, trial_hash: str, eval_revision: str
    ) -> TrialRecord | None:
        return self._trials.get(trial_hash)

    def iter_candidate_scores(self, **_: object):
        return iter(self._score_rows)

    def iter_trial_summaries(self, **_: object):
        return iter(self._trial_summaries)


class SummaryOnlyProjectionRepository:
    def __init__(
        self, trial_summaries: list[TrialSummaryRow], score_rows: list[ScoreRow]
    ):
        self._trial_summaries = list(trial_summaries)
        self._score_rows = list(score_rows)

    def get_trial_record(
        self, trial_hash: str, eval_revision: str
    ) -> TrialRecord | None:
        del trial_hash, eval_revision
        raise AssertionError("compare() should not hydrate full trial records")

    def iter_candidate_scores(self, **_: object):
        return iter(self._score_rows)

    def iter_trial_summaries(self, **_: object):
        return iter(self._trial_summaries)


def test_root_public_api_exports_curated_v2_surface() -> None:
    assert set(themis.__all__) == EXPECTED_ROOT_EXPORTS


def test_root_public_api_excludes_removed_v1_helpers_and_internal_storage_types() -> (
    None
):
    forbidden = {
        "DatabaseManager",
        "SqliteEventRepository",
        "SqliteProjectionRepository",
        "ArtifactStore",
        "evaluate",
        "quick_evaluate",
        "register_provider",
        "register_metric",
        "ExperimentSession",
    }
    assert forbidden.isdisjoint(set(themis.__all__))


def test_runtime_context_exported_from_root_and_specs_packages() -> None:
    assert hasattr(themis, "RuntimeContext")
    assert "RuntimeContext" in themis.specs.__all__
    assert hasattr(themis, "PromptMessage")
    assert "PromptMessage" in themis.specs.__all__


def test_experiment_result_reads_trials_timelines_and_projection_hooks() -> None:
    trial = _make_trial_record()
    view = _make_timeline_view(trial)
    repo = FakeProjectionRepository(trial, view)
    result = ExperimentResult(
        projection_repo=repo,
        trial_hashes=[trial.spec_hash],
        eval_revision="latest",
    )

    assert list(result.iter_trials()) == [trial]
    assert result.get_trial(trial.spec_hash) == trial
    assert result.view_timeline("candidate_1") == view

    report_builder = result.report()
    assert report_builder.trials == [trial]


def test_experiment_result_compare_returns_statistical_rows() -> None:
    trials: list[TrialRecord] = []
    score_rows: list[ScoreRow] = []
    treatment_scores = [1.0, 0.9, 1.0, 0.8, 0.9]
    baseline_scores = [0.4, 0.5, 0.6, 0.4, 0.5]

    for index, (treatment_score, baseline_score) in enumerate(
        zip(treatment_scores, baseline_scores), start=1
    ):
        for model_id, score in (
            ("baseline", baseline_score),
            ("treatment", treatment_score),
        ):
            trial_spec = TrialSpec(
                trial_id=f"{model_id}_{index}",
                model=ModelSpec(model_id=model_id, provider="openai"),
                task=TaskSpec(
                    task_id="math",
                    dataset=DatasetSpec(source="memory"),
                    default_metrics=["exact_match"],
                ),
                item_id=f"item-{index}",
                prompt=PromptTemplateSpec(id="baseline", messages=[]),
                params=InferenceParamsSpec(),
            )
            trial = TrialRecord(
                spec_hash=trial_spec.spec_hash,
                status=RecordStatus.OK,
                candidates=[],
                trial_spec=trial_spec,
            )
            trials.append(trial)
            score_rows.append(
                ScoreRow(
                    trial_hash=trial_spec.spec_hash,
                    candidate_id=f"{model_id}_cand_{index}",
                    metric_id="exact_match",
                    score=score,
                )
            )

    result = ExperimentResult(
        projection_repo=PairwiseProjectionRepository(trials, score_rows),
        trial_hashes=[trial.spec_hash for trial in trials],
        eval_revision="latest",
    )

    comparison = result.compare(metric_id="exact_match")

    assert isinstance(comparison, ComparisonTable)
    assert len(comparison.rows) == 1
    row = comparison.rows[0]
    assert row.metric_id == "exact_match"
    assert row.task_id == "math"
    assert row.baseline_model_id == "baseline"
    assert row.treatment_model_id == "treatment"
    assert row.pair_count == 5
    assert row.delta_mean > 0.0
    assert row.ci_upper >= row.ci_lower


def test_experiment_result_compare_uses_trial_summaries_without_hydration() -> None:
    trial_summaries = [
        TrialSummaryRow(
            trial_hash="trial-baseline",
            model_id="baseline",
            task_id="math",
            item_id="item-1",
            status=RecordStatus.OK,
        ),
        TrialSummaryRow(
            trial_hash="trial-treatment",
            model_id="treatment",
            task_id="math",
            item_id="item-1",
            status=RecordStatus.OK,
        ),
    ]
    score_rows = [
        ScoreRow(
            trial_hash="trial-baseline",
            candidate_id="cand-baseline",
            metric_id="exact_match",
            score=0.5,
        ),
        ScoreRow(
            trial_hash="trial-treatment",
            candidate_id="cand-treatment",
            metric_id="exact_match",
            score=1.0,
        ),
    ]
    result = ExperimentResult(
        projection_repo=SummaryOnlyProjectionRepository(trial_summaries, score_rows),
        trial_hashes=[row.trial_hash for row in trial_summaries],
        eval_revision="latest",
    )

    comparison = result.compare(metric_id="exact_match")

    assert isinstance(comparison, ComparisonTable)
    assert len(comparison.rows) == 1
    assert comparison.rows[0].pair_count == 1
