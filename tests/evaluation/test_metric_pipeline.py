from __future__ import annotations

from themis.evaluation import extractors, metrics
from themis.evaluation.pipeline import EvaluationPipeline, EvaluationPipelineContract
from tests.factories import make_record


def test_metric_pipeline_evaluates_records():
    pipeline = EvaluationPipeline(
        extractor=extractors.IdentityExtractor(),
        metrics=[metrics.ResponseLength()],
    )

    report = pipeline.evaluate([make_record()])

    assert report.metrics["ResponseLength"].count == 1
    assert isinstance(pipeline, EvaluationPipelineContract)
