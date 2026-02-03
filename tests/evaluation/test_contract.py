from __future__ import annotations

from themis.evaluation import extractors, metrics
from themis.evaluation.pipeline import (
    EvaluationPipeline,
    EvaluationPipelineContract,
    ComposableEvaluationPipeline,
    ComposableEvaluationReportPipeline,
)
from tests.factories import make_record


def test_standard_pipeline_implements_contract():
    pipeline = EvaluationPipeline(
        extractor=extractors.IdentityExtractor(),
        metrics=[metrics.ResponseLength()],
    )
    assert isinstance(pipeline, EvaluationPipelineContract)


def test_composable_report_pipeline_implements_contract():
    comp = (
        ComposableEvaluationPipeline()
        .extract(extractors.IdentityExtractor())
        .compute_metrics([metrics.ResponseLength()], references=[])
    )
    pipeline = ComposableEvaluationReportPipeline(comp)
    assert isinstance(pipeline, EvaluationPipelineContract)


def test_evaluation_records_use_metric_score_list():
    pipeline = EvaluationPipeline(
        extractor=extractors.IdentityExtractor(),
        metrics=[metrics.ResponseLength()],
    )
    report = pipeline.evaluate([make_record()])
    assert report.records
    assert isinstance(report.records[0].scores, list)
    assert report.records[0].scores[0].metric_name == "ResponseLength"
