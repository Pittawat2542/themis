from themis.report.builder import ReportBuilder
from themis.report.metric_frame_builder import MetricFrameBuilder
from themis.report.report_metadata_builder import ReportMetadataBuilder
import themis
from themis.overlays import OverlaySelection
from themis.records.trial import TrialRecord
from themis.records.candidate import CandidateRecord
from themis.records.evaluation import EvaluationRecord, MetricScore
from themis.records.provenance import ProvenanceRecord
from themis.records.report import ReportMetadata
from themis.runtime.experiment_result import ExperimentResult
from themis.specs.experiment import InferenceParamsSpec, PromptTemplateSpec, TrialSpec
from themis.specs.foundational import DatasetSpec, GenerationSpec, ModelSpec, TaskSpec
from themis.types.events import ScoreRow, TrialSummaryRow
from themis.types.enums import PValueCorrection, RecordStatus, DatasetSource
from themis.types.json_types import JSONDict
import tempfile
import os
from typing import cast


def test_report_builder_basic():
    # Setup flat mock
    eval_rec = EvaluationRecord(
        spec_hash="e1",
        metric_scores=[
            MetricScore(metric_id="accuracy", value=1.0),
            MetricScore(metric_id="latency", value=150.0),
        ],
    )
    cand = CandidateRecord(spec_hash="c1", status=RecordStatus.OK, evaluation=eval_rec)
    trial = TrialRecord(
        spec_hash="t1",
        status=RecordStatus.OK,
        candidates=[cand, cand],  # Duplicated for count=2
    )

    builder = ReportBuilder([trial])
    report = builder.build()

    # Assert
    assert len(report.tables) == 1
    t = report.tables[0]
    assert t.id == "main_results"

    assert any(r["metric_id"] == "accuracy" for r in t.data)
    assert any(r["metric_id"] == "latency" for r in t.data)

    acc_row = next(r for r in t.data if r["metric_id"] == "accuracy")
    assert acc_row["mean"] == 1.0

    lat_row = next(r for r in t.data if r["metric_id"] == "latency")
    assert lat_row["count"] == 2

    assert "t1" in report.metadata.spec_hashes


def test_report_builder_shortcuts():
    builder = ReportBuilder([])
    # Even empty trials should build an empty report safely (0 tables)
    with tempfile.TemporaryDirectory() as d:
        csv_path = os.path.join(d, "out.csv")
        md_path = os.path.join(d, "out.md")
        tex_path = os.path.join(d, "out.tex")

        builder.to_csv(csv_path)
        builder.to_markdown(md_path)
        builder.to_latex(tex_path)

        assert os.path.exists(csv_path)
        assert os.path.exists(md_path)
        assert os.path.exists(tex_path)


class ProjectionBackedReportRepo:
    def __init__(self, trial: TrialRecord, rows: list[ScoreRow]):
        self.trial = trial
        self.rows = rows
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

    def get_trial_record(
        self,
        trial_hash: str,
        *,
        transform_hash: str | None = None,
        evaluation_hash: str | None = None,
    ):
        del transform_hash, evaluation_hash
        if trial_hash == self.trial.spec_hash:
            return self.trial
        return None

    def iter_candidate_scores(self, **kwargs):
        return iter(self.rows)

    def iter_trial_summaries(self, **kwargs):
        return iter(self.trial_summaries)


def test_report_builder_builds_from_projection_rows_via_experiment_result():
    trial_spec = TrialSpec(
        trial_id="projection_report",
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
    trial = TrialRecord(
        spec_hash=trial_spec.spec_hash,
        status=RecordStatus.OK,
        candidates=[],
        trial_spec=trial_spec,
    )
    repo = ProjectionBackedReportRepo(
        trial,
        rows=[
            ScoreRow(
                trial_hash=trial_spec.spec_hash,
                candidate_id="cand_1",
                metric_id="accuracy",
                score=0.5,
                details={"source": "projection"},
            )
        ],
    )
    result = ExperimentResult(projection_repo=repo, trial_hashes=[trial_spec.spec_hash])

    report = result.report().build()

    assert len(report.tables) == 1
    row = report.tables[0].data[0]
    assert row["metric_id"] == "accuracy"
    assert row["mean"] == 0.5
    assert row["model_id"] == "gpt-4o-mini"
    assert row["task_id"] == "math"


def test_report_builder_builds_from_trial_summaries_without_trials():
    builder = ReportBuilder(
        [],
        trial_summaries=[
            TrialSummaryRow(
                trial_hash="trial_1",
                model_id="gpt-4o-mini",
                task_id="math",
                item_id="item-1",
                status=RecordStatus.OK,
            )
        ],
        score_rows=[
            ScoreRow(
                trial_hash="trial_1",
                candidate_id="cand_1",
                metric_id="accuracy",
                score=0.75,
            )
        ],
    )

    report = builder.build()

    assert report.tables[0].data[0]["model_id"] == "gpt-4o-mini"
    assert report.tables[0].data[0]["task_id"] == "math"


def test_metric_frame_builder_reuses_trial_summary_metadata_for_report_and_comparison_frames():
    frame_builder = MetricFrameBuilder()
    summaries = [
        TrialSummaryRow(
            trial_hash="trial_1",
            model_id="gpt-4o-mini",
            task_id="math",
            item_id="item-1",
            status=RecordStatus.OK,
        )
    ]
    score_rows = [
        ScoreRow(
            trial_hash="trial_1",
            candidate_id="cand_1",
            metric_id="accuracy",
            score=0.75,
        )
    ]

    report_frame = frame_builder.build_report_frame(summaries, score_rows)
    comparison_frame = frame_builder.build_comparison_frame(summaries, score_rows)

    report_row = report_frame.to_dict(orient="records")[0]
    comparison_row = comparison_frame.to_dict(orient="records")[0]

    assert report_row["model_id"] == "gpt-4o-mini"
    assert report_row["metric_value"] == 0.75
    assert comparison_row["task_id"] == "math"
    assert comparison_row["score"] == 0.75


def test_report_metadata_builder_collects_overlay_and_provenance_metadata():
    trial_spec = TrialSpec(
        trial_id="report_meta",
        model=ModelSpec(model_id="gpt-4o-mini", provider="openai"),
        task=TaskSpec(
            task_id="math",
            dataset=DatasetSpec(source="memory", revision="rev-a"),
            generation=GenerationSpec(),
        ),
        item_id="item-1",
        prompt=PromptTemplateSpec(id="baseline", messages=[]),
        params=InferenceParamsSpec(),
    )
    trial = TrialRecord(
        spec_hash=trial_spec.spec_hash,
        status=RecordStatus.OK,
        candidates=[],
        trial_spec=trial_spec,
        provenance=ProvenanceRecord(
            themis_version=themis.__version__,
            git_commit="abc123",
            python_version="3.12.0",
            platform="macOS",
            library_versions={"openai": "1.0.0"},
            model_endpoint_meta={},
        ),
    )

    metadata = ReportMetadataBuilder().build(
        [trial],
        [
            TrialSummaryRow(
                trial_hash=trial.spec_hash,
                model_id=trial_spec.model.model_id,
                task_id=trial_spec.task.task_id,
                item_id=trial_spec.item_id,
                status=trial.status,
            )
        ],
        OverlaySelection(evaluation_hash="eval_1"),
    )

    assert metadata.spec_hashes == [trial.spec_hash]
    assert metadata.extras["dataset_revisions"] == ["rev-a"]
    assert metadata.extras["evaluation_hash"] == "eval_1"
    assert metadata.extras["provenance"]["git_commits"] == ["abc123"]


def test_report_builder_includes_pairwise_comparisons_and_provenance() -> None:
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
                    dataset=DatasetSpec(
                        source=DatasetSource.MEMORY,
                        revision="rev-a",
                    ),
                    generation=GenerationSpec(),
                ),
                item_id=f"item-{index}",
                prompt=PromptTemplateSpec(id="baseline", messages=[]),
                params=InferenceParamsSpec(),
            )
            trials.append(
                TrialRecord(
                    spec_hash=trial_spec.spec_hash,
                    status=RecordStatus.OK,
                    candidates=[],
                    trial_spec=trial_spec,
                    provenance=ProvenanceRecord(
                        themis_version=themis.__version__,
                        git_commit="abc123",
                        python_version="3.12.0",
                        platform="macOS",
                        library_versions={"openai": "1.0.0"},
                        model_endpoint_meta={},
                    ),
                )
            )
            score_rows.append(
                ScoreRow(
                    trial_hash=trial_spec.spec_hash,
                    candidate_id=f"{model_id}_cand_{index}",
                    metric_id="exact_match",
                    score=score,
                )
            )

    report = ReportBuilder(trials, score_rows=score_rows).build(
        p_value_correction=PValueCorrection.HOLM
    )

    assert {table.id for table in report.tables} >= {
        "main_results",
        "paired_comparisons",
    }
    comparisons = report.get_table("paired_comparisons")
    assert comparisons is not None
    comparison = cast(JSONDict, comparisons.data[0])
    assert comparison["baseline_model_id"] == "baseline"
    assert comparison["treatment_model_id"] == "treatment"
    delta_mean = float(cast(float, comparison["delta_mean"]))
    adjusted_p_value = float(cast(float, comparison["adjusted_p_value"]))
    p_value = float(cast(float, comparison["p_value"]))
    assert delta_mean > 0.0
    assert comparison["adjustment_method"] == "holm"
    assert adjusted_p_value >= p_value
    assert report.metadata.themis_version == themis.__version__
    assert report.metadata.extras["dataset_revisions"] == ["rev-a"]
    provenance = cast(JSONDict, report.metadata.extras["provenance"])
    assert provenance["themis_versions"] == [themis.__version__]


def test_report_metadata_defaults_to_package_version():
    metadata = ReportMetadata(spec_hash="meta")

    assert metadata.themis_version == themis.__version__


def test_report_builder_records_overlay_metadata_for_evaluation_views():
    builder = ReportBuilder(
        [],
        trial_summaries=[
            TrialSummaryRow(
                trial_hash="trial_1",
                model_id="gpt-4o-mini",
                task_id="math",
                item_id="item-1",
                status=RecordStatus.OK,
            )
        ],
        score_rows=[
            ScoreRow(
                trial_hash="trial_1",
                candidate_id="cand_1",
                metric_id="accuracy",
                score=0.75,
            )
        ],
        overlay_key="ev:eval_1",
        evaluation_hash="eval_1",
    )

    report = builder.build()

    assert report.metadata.extras["overlay_key"] == "ev:eval_1"
    assert report.metadata.extras["evaluation_hash"] == "eval_1"
    assert "eval_revision" not in report.metadata.extras
