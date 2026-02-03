from __future__ import annotations

from themis.core import entities as core_entities
from themis.evaluation import extractors, metrics
from themis.evaluation.metric_pipeline import MetricPipeline
from themis.evaluation.pipeline import EvaluationPipelineContract


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


def test_metric_pipeline_evaluates_records():
    pipeline = MetricPipeline(
        extractor=extractors.IdentityExtractor(),
        metrics=[metrics.ResponseLength()],
    )

    report = pipeline.evaluate([_make_record()])

    assert report.metrics["ResponseLength"].count == 1
    assert isinstance(pipeline, EvaluationPipelineContract)
