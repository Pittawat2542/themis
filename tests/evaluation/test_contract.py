from __future__ import annotations

from themis.core import entities as core_entities
from themis.evaluation import extractors, metrics
from themis.evaluation.pipeline import (
    EvaluationPipeline,
    EvaluationPipelineContract,
    ComposableEvaluationPipeline,
    ComposableEvaluationReportPipeline,
)


def _make_record() -> core_entities.GenerationRecord:
    prompt_spec = core_entities.PromptSpec(name="t", template="Q")
    prompt = core_entities.PromptRender(spec=prompt_spec, text="Q")
    model = core_entities.ModelSpec(identifier="model-x", provider="fake")
    sampling = core_entities.SamplingConfig(temperature=0.0, top_p=1.0, max_tokens=8)
    task = core_entities.GenerationTask(
        prompt=prompt,
        model=model,
        sampling=sampling,
        metadata={"dataset_id": "s1"},
    )
    return core_entities.GenerationRecord(
        task=task,
        output=core_entities.ModelOutput(text="ok"),
        error=None,
    )


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
    report = pipeline.evaluate([_make_record()])
    assert report.records
    assert isinstance(report.records[0].scores, list)
    assert report.records[0].scores[0].metric_name == "ResponseLength"
