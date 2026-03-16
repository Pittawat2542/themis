from __future__ import annotations

from datetime import datetime, timezone
import json

import themis
import themis.config_report
import themis.progress
import themis.records
import themis.specs
import themis.stats
import themis.types

from themis.records.candidate import CandidateRecord
from themis.records.evaluation import EvaluationRecord, MetricScore
from themis.records.extraction import ExtractionRecord
from themis.records.timeline import RecordTimeline, TimelineStageRecord
from themis.records.trial import TrialRecord
from themis.runtime import ComparisonTable, ExperimentResult, RecordTimelineView
from themis.specs.experiment import TrialSpec, PromptTemplateSpec, InferenceParamsSpec
from themis.specs.foundational import (
    DatasetSpec,
    EvaluationSpec,
    ExtractorChainSpec,
    ExtractorRefSpec,
    GenerationSpec,
    ModelSpec,
    OutputTransformSpec,
    TaskSpec,
)
from themis.types.enums import (
    PValueCorrection,
    RecordStatus,
    RecordType,
    DatasetSource,
    ErrorWhere,
    ErrorCode,
)
from themis.types.events import ScoreRow, TimelineStage, TrialSummaryRow


EXPECTED_ROOT_EXPORTS = {
    "__version__",
    "Orchestrator",
    "ExperimentResult",
    "generate_config_report",
    "ProjectSpec",
    "ExperimentSpec",
    "StorageConfig",
    "StorageSpec",
    "SqliteBlobStorageSpec",
    "PostgresBlobStorageSpec",
    "ExecutionPolicySpec",
    "InferenceGridSpec",
    "InferenceParamsSpec",
    "ItemSamplingSpec",
    "PromptMessage",
    "PromptTemplateSpec",
    "TrialSpec",
    "ModelSpec",
    "GenerationSpec",
    "OutputTransformSpec",
    "EvaluationSpec",
    "ExtractorRefSpec",
    "ExtractorChainSpec",
    "TaskSpec",
    "DatasetSpec",
    "RuntimeContext",
    "PluginRegistry",
    "ThemisError",
}

EXPECTED_CONFIG_REPORT_EXPORTS = {
    "ConfigReportDocument",
    "ConfigReportHeader",
    "ConfigReportNode",
    "ConfigReportParameter",
    "ConfigReportOptions",
    "ConfigReportMixin",
    "ConfigReportRenderer",
    "ConfigReportFormat",
    "ConfigReportVerbosity",
    "JsonConfigReportRenderer",
    "YamlConfigReportRenderer",
    "MarkdownConfigReportRenderer",
    "LatexConfigReportRenderer",
    "build_config_report_document",
    "get_config_report_renderer",
    "list_config_report_renderers",
    "render_config_report",
    "register_config_report_renderer",
    "generate_config_report",
    "config_reportable",
}


def _make_trial_record() -> TrialRecord:
    trial_spec = TrialSpec(
        trial_id="trial_1",
        model=ModelSpec(model_id="gpt-4o-mini", provider="openai"),
        task=TaskSpec(
            task_id="math",
            dataset=DatasetSpec(source=DatasetSource.MEMORY),
            generation=GenerationSpec(),
            output_transforms=[
                OutputTransformSpec(
                    name="default",
                    extractor_chain=ExtractorChainSpec(
                        extractors=[ExtractorRefSpec(id="json")]
                    ),
                )
            ],
            evaluations=[
                EvaluationSpec(
                    name="default",
                    transform="default",
                    metrics=["exact_match"],
                )
            ],
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


def _make_analysis_trial_record() -> TrialRecord:
    trial_spec = TrialSpec(
        trial_id="trial_analysis",
        model=ModelSpec(model_id="gpt-4o-mini", provider="openai"),
        task=TaskSpec(
            task_id="math",
            dataset=DatasetSpec(source=DatasetSource.MEMORY),
            generation=GenerationSpec(),
            output_transforms=[
                OutputTransformSpec(
                    name="default",
                    extractor_chain=ExtractorChainSpec(
                        extractors=[ExtractorRefSpec(id="json")]
                    ),
                )
            ],
            evaluations=[
                EvaluationSpec(
                    name="default",
                    transform="default",
                    metrics=["exact_match"],
                )
            ],
        ),
        item_id="item-analysis",
        prompt=PromptTemplateSpec(id="baseline", messages=[]),
        params=InferenceParamsSpec(),
    )
    candidate = CandidateRecord(
        spec_hash="candidate_analysis",
        status=RecordStatus.ERROR,
        error=themis.records.ErrorRecord(
            where=ErrorWhere.EXTRACTOR,
            code=ErrorCode.PARSE_ERROR,
            message="Extractor could not parse response",
            retryable=False,
            details={"reason": "missing field"},
        ),
        extractions=[
            ExtractionRecord(
                spec_hash="extract_bad",
                extractor_id="json",
                success=False,
                parsed_answer=None,
                failure_reason="missing field",
            )
        ],
        evaluation=EvaluationRecord(
            spec_hash="eval_analysis",
            metric_scores=[
                MetricScore(
                    metric_id="exact_match",
                    value=0.0,
                    details={"tags": ["hallucination", "format_error"]},
                )
            ],
        ),
    )
    return TrialRecord(
        spec_hash=trial_spec.spec_hash,
        status=RecordStatus.ERROR,
        error=themis.records.ErrorRecord(
            where=ErrorWhere.EXECUTOR,
            code=ErrorCode.METRIC_COMPUTATION,
            message="One or more candidates failed",
            retryable=False,
            details={},
        ),
        candidates=[candidate],
        trial_spec=trial_spec,
    )


def _make_timeline_view(trial: TrialRecord) -> RecordTimelineView:
    assert trial.trial_spec is not None
    timeline = RecordTimeline(
        record_id="candidate_1",
        record_type=RecordType.CANDIDATE,
        trial_hash=trial.spec_hash,
        candidate_id="candidate_1",
        item_id=trial.trial_spec.item_id,
        stages=[
            TimelineStageRecord(
                stage=TimelineStage.PROMPT_RENDER,
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
        record_type=RecordType.CANDIDATE,
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
        self.last_trial_summary_kwargs: dict[str, object] | None = None
        self.last_get_trial_kwargs: dict[str, object] | None = None
        self.last_timeline_kwargs: dict[str, object] | None = None
        assert trial.trial_spec is not None
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
        self,
        trial_hash: str,
        *,
        transform_hash: str | None = None,
        evaluation_hash: str | None = None,
    ) -> TrialRecord | None:
        self.last_get_trial_kwargs = {
            "trial_hash": trial_hash,
            "transform_hash": transform_hash,
            "evaluation_hash": evaluation_hash,
        }
        del transform_hash, evaluation_hash
        if trial_hash == self.trial.spec_hash:
            return self.trial
        return None

    def get_timeline_view(
        self,
        record_id: str,
        record_type: str,
        *,
        transform_hash: str | None = None,
        evaluation_hash: str | None = None,
    ) -> RecordTimelineView | None:
        self.last_timeline_kwargs = {
            "record_id": record_id,
            "record_type": record_type,
            "transform_hash": transform_hash,
            "evaluation_hash": evaluation_hash,
        }
        del transform_hash, evaluation_hash
        if record_id == self.view.record_id and record_type == self.view.record_type:
            return self.view
        return None

    def iter_candidate_scores(self, **_: object):
        return iter(self.score_rows)

    def iter_trial_summaries(self, **kwargs: object):
        self.last_trial_summary_kwargs = dict(kwargs)
        return iter(self.trial_summaries)


class PairwiseProjectionRepository:
    def __init__(self, trials: list[TrialRecord], score_rows: list[ScoreRow]):
        typed_trials: list[TrialRecord] = []
        for trial in trials:
            assert trial.trial_spec is not None
            typed_trials.append(trial)
        self._trials = {trial.spec_hash: trial for trial in typed_trials}
        self._score_rows = list(score_rows)
        self._trial_summaries = []
        for trial in typed_trials:
            trial_spec = trial.trial_spec
            assert trial_spec is not None
            self._trial_summaries.append(
                TrialSummaryRow(
                    trial_hash=trial.spec_hash,
                    model_id=trial_spec.model.model_id,
                    task_id=trial_spec.task.task_id,
                    item_id=trial_spec.item_id,
                    status=trial.status,
                )
            )

    def get_trial_record(
        self,
        trial_hash: str,
        *,
        transform_hash: str | None = None,
        evaluation_hash: str | None = None,
    ) -> TrialRecord | None:
        del transform_hash, evaluation_hash
        return self._trials.get(trial_hash)

    def iter_candidate_scores(self, **_: object):
        return iter(self._score_rows)

    def iter_trial_summaries(self, **_: object):
        return iter(self._trial_summaries)

    def get_timeline_view(
        self,
        record_id: str,
        record_type: str,
        *,
        transform_hash: str | None = None,
        evaluation_hash: str | None = None,
    ) -> RecordTimelineView | None:
        del record_id, record_type, transform_hash, evaluation_hash
        return None


class SummaryOnlyProjectionRepository:
    def __init__(
        self, trial_summaries: list[TrialSummaryRow], score_rows: list[ScoreRow]
    ):
        self._trial_summaries = list(trial_summaries)
        self._score_rows = list(score_rows)

    def get_trial_record(
        self,
        trial_hash: str,
        *,
        transform_hash: str | None = None,
        evaluation_hash: str | None = None,
    ) -> TrialRecord | None:
        del trial_hash, transform_hash, evaluation_hash
        raise AssertionError("compare() should not hydrate full trial records")

    def iter_candidate_scores(self, **_: object):
        return iter(self._score_rows)

    def iter_trial_summaries(self, **_: object):
        return iter(self._trial_summaries)

    def get_timeline_view(
        self,
        record_id: str,
        record_type: str,
        *,
        transform_hash: str | None = None,
        evaluation_hash: str | None = None,
    ) -> RecordTimelineView | None:
        del record_id, record_type, transform_hash, evaluation_hash
        return None


def test_root_public_api_exports_curated_v2_surface() -> None:
    assert set(themis.__all__) == EXPECTED_ROOT_EXPORTS


def test_root_public_api_exports_stage_specs() -> None:
    assert hasattr(themis, "GenerationSpec")
    assert hasattr(themis, "OutputTransformSpec")
    assert hasattr(themis, "EvaluationSpec")
    assert "GenerationSpec" in themis.specs.__all__
    assert "OutputTransformSpec" in themis.specs.__all__
    assert "EvaluationSpec" in themis.specs.__all__
    assert "StorageConfig" in themis.specs.__all__


def test_config_report_public_api_exports_curated_surface() -> None:
    assert set(themis.config_report.__all__) == EXPECTED_CONFIG_REPORT_EXPORTS


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
    assert hasattr(themis, "StorageConfig")
    assert "RuntimeContext" in themis.specs.__all__
    assert hasattr(themis, "PromptMessage")
    assert "PromptMessage" in themis.specs.__all__


def test_lazy_public_namespaces_are_discoverable_via_dir() -> None:
    assert set(themis.records.__all__).issubset(set(dir(themis.records)))
    assert set(themis.types.__all__).issubset(set(dir(themis.types)))
    assert set(themis.stats.__all__).issubset(set(dir(themis.stats)))


def test_progress_public_module_exports_curated_surface() -> None:
    assert {
        "ProgressBus",
        "ProgressConfig",
        "ProgressEvent",
        "ProgressEventType",
        "ProgressRendererType",
        "ProgressVerbosity",
        "RunProgressTracker",
        "RunProgressSnapshot",
        "StageProgressSnapshot",
    } == set(themis.progress.__all__)
    assert "StageProgressSnapshot" not in themis.__all__


def test_experiment_result_reads_trials_timelines_and_projection_hooks() -> None:
    trial = _make_trial_record()
    view = _make_timeline_view(trial)
    repo = FakeProjectionRepository(trial, view)
    result = ExperimentResult(
        projection_repo=repo,
        trial_hashes=[trial.spec_hash],
    )

    assert list(result.iter_trials()) == [trial]
    assert result.get_trial(trial.spec_hash) == trial
    assert result.view_timeline("candidate_1") == view
    assert result.view_timeline("candidate_1", record_type=RecordType.CANDIDATE) == view

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
                    dataset=DatasetSpec(source=DatasetSource.MEMORY),
                    generation=GenerationSpec(),
                    output_transforms=[
                        OutputTransformSpec(
                            name="default",
                            extractor_chain=ExtractorChainSpec(
                                extractors=[ExtractorRefSpec(id="json")]
                            ),
                        )
                    ],
                    evaluations=[
                        EvaluationSpec(
                            name="default",
                            transform="default",
                            metrics=["exact_match"],
                        )
                    ],
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
    )

    comparison = result.compare(
        metric_id="exact_match",
        p_value_correction=PValueCorrection.NONE,
    )

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
    )

    comparison = result.compare(metric_id="exact_match")

    assert isinstance(comparison, ComparisonTable)
    assert len(comparison.rows) == 1
    assert comparison.rows[0].pair_count == 1


def test_experiment_result_passes_active_overlay_to_trial_summary_iteration() -> None:
    trial = _make_trial_record()
    view = _make_timeline_view(trial)
    repo = FakeProjectionRepository(trial, view)
    result = ExperimentResult(
        projection_repo=repo,
        trial_hashes=[trial.spec_hash],
        evaluation_hashes=["eval_1"],
    ).for_evaluation("eval_1")

    list(result.iter_trial_summaries())

    assert repo.last_trial_summary_kwargs == {
        "trial_hashes": [trial.spec_hash],
        "transform_hash": None,
        "evaluation_hash": "eval_1",
    }


def test_experiment_result_overlay_views_forward_only_the_active_overlay() -> None:
    trial = _make_trial_record()
    view = _make_timeline_view(trial)
    repo = FakeProjectionRepository(trial, view)
    result = ExperimentResult(
        projection_repo=repo,
        trial_hashes=[trial.spec_hash],
        transform_hashes=["tx_1"],
        evaluation_hashes=["eval_1"],
    )

    transform_view = result.for_transform("tx_1")
    evaluation_view = result.for_evaluation("eval_1")

    assert result.active_transform_hash is None
    assert result.active_evaluation_hash is None
    assert transform_view.get_trial(trial.spec_hash) == trial
    assert repo.last_get_trial_kwargs == {
        "trial_hash": trial.spec_hash,
        "transform_hash": "tx_1",
        "evaluation_hash": None,
    }
    assert evaluation_view.view_timeline("candidate_1") == view
    assert repo.last_timeline_kwargs == {
        "record_id": "candidate_1",
        "record_type": RecordType.CANDIDATE,
        "transform_hash": None,
        "evaluation_hash": "eval_1",
    }


def test_experiment_result_leaderboard_aggregates_score_rows() -> None:
    trial = _make_trial_record()
    view = _make_timeline_view(trial)
    repo = FakeProjectionRepository(trial, view)
    result = ExperimentResult(
        projection_repo=repo,
        trial_hashes=[trial.spec_hash],
        evaluation_hashes=["eval_1"],
    ).for_evaluation("eval_1")

    leaderboard = result.leaderboard()

    assert leaderboard == [
        {
            "model_id": "gpt-4o-mini",
            "task_id": "math",
            "metric_id": "exact_match",
            "mean": 1.0,
            "count": 1,
            "min": 1.0,
            "max": 1.0,
        }
    ]


def test_experiment_result_iter_helpers_surface_invalid_examples_and_failures() -> None:
    trial = _make_analysis_trial_record()
    view = _make_timeline_view(trial)
    repo = FakeProjectionRepository(trial, view)
    result = ExperimentResult(
        projection_repo=repo,
        trial_hashes=[trial.spec_hash],
        evaluation_hashes=["eval_analysis"],
    ).for_evaluation("eval_analysis")

    invalid_extractions = list(result.iter_invalid_extractions())
    failures = list(result.iter_failures())
    tagged = list(result.iter_tagged_examples())
    filtered = list(result.iter_tagged_examples(tag="hallucination"))

    assert invalid_extractions == [
        {
            "trial_hash": trial.spec_hash,
            "candidate_id": "candidate_analysis",
            "model_id": "gpt-4o-mini",
            "task_id": "math",
            "item_id": "item-analysis",
            "extractor_id": "json",
            "failure_reason": "missing field",
            "warnings": [],
            "transform_hash": None,
            "evaluation_hash": "eval_analysis",
        }
    ]
    assert failures[0]["level"] == "trial"
    assert failures[0]["message"] == "One or more candidates failed"
    assert failures[1]["level"] == "candidate"
    assert failures[1]["candidate_id"] == "candidate_analysis"
    assert tagged == [
        {
            "trial_hash": trial.spec_hash,
            "candidate_id": "candidate_analysis",
            "model_id": "gpt-4o-mini",
            "task_id": "math",
            "item_id": "item-analysis",
            "tags": ["format_error", "hallucination"],
            "source": "metric_details",
            "transform_hash": None,
            "evaluation_hash": "eval_analysis",
        }
    ]
    assert filtered == tagged


def test_experiment_result_export_json_writes_overlay_payload(tmp_path) -> None:
    trial = _make_trial_record()
    view = _make_timeline_view(trial)
    repo = FakeProjectionRepository(trial, view)
    result = ExperimentResult(
        projection_repo=repo,
        trial_hashes=[trial.spec_hash],
        evaluation_hashes=["eval_1"],
    ).for_evaluation("eval_1")

    output_path = tmp_path / "result.json"
    payload = result.export_json(str(output_path))

    written = json.loads(output_path.read_text())
    assert payload == written
    assert written["overlay"]["evaluation_hash"] == "eval_1"
    assert written["trial_hashes"] == [trial.spec_hash]
    assert written["trial_summaries"][0]["trial_hash"] == trial.spec_hash
    assert written["trials"][0]["spec_hash"] == trial.spec_hash
