from themis.core import entities as core_entities
from themis.evaluation import extractors, metrics, pipeline


def make_generation_record(
    sample_id: str, raw_output: str, reference_value: str
) -> core_entities.GenerationRecord:
    sampling = core_entities.SamplingConfig(temperature=0.5, top_p=0.9, max_tokens=128)
    prompt_spec = core_entities.PromptSpec(
        name="qa", template="Answer the question: {question}"
    )
    prompt_render = core_entities.PromptRender(
        spec=prompt_spec,
        text="Answer the question: Capital?",
        context={"question": "Capital of France?"},
    )
    model_spec = core_entities.ModelSpec(identifier="gpt-4o", provider="test")
    reference = core_entities.Reference(kind="answer", value=reference_value)
    task = core_entities.GenerationTask(
        prompt=prompt_render,
        model=model_spec,
        sampling=sampling,
        metadata={"sample_id": sample_id},
        reference=reference,
    )
    return core_entities.GenerationRecord(
        task=task,
        output=core_entities.ModelOutput(text=raw_output),
        error=None,
        metrics={},
    )


def test_pipeline_returns_metric_aggregates():
    extractor = extractors.JsonFieldExtractor(field_path="answer")
    metric = metrics.ExactMatch()
    eval_pipeline = pipeline.EvaluationPipeline(extractor=extractor, metrics=[metric])

    generations = [
        make_generation_record("sample-1", '{"answer": "Paris"}', "Paris"),
        make_generation_record("sample-2", '{"answer": "Lyon"}', "Paris"),
    ]

    report = eval_pipeline.evaluate(generations)

    exact_report = report.metrics["ExactMatch"]
    assert exact_report.mean == 0.5
    assert exact_report.count == 2
    assert {score.metadata["sample_id"] for score in exact_report.per_sample} == {
        "sample-1",
        "sample-2",
    }
    assert not report.failures
    for score in exact_report.per_sample:
        assert "evaluation_time_ms" in score.metadata
        assert "extraction_time_ms" in score.metadata
    assert len(report.records) == 2
    assert report.records[0].scores


def test_pipeline_collects_extraction_failures_and_continues():
    extractor = extractors.JsonFieldExtractor(field_path="answer.value")
    metric = metrics.ExactMatch()
    eval_pipeline = pipeline.EvaluationPipeline(extractor=extractor, metrics=[metric])

    generations = [
        make_generation_record("sample-1", '{"answer": {"value": "Madrid"}}', "Madrid"),
        make_generation_record("sample-2", '{"prediction": "Madrid"}', "Madrid"),
    ]

    report = eval_pipeline.evaluate(generations)

    assert len(report.failures) == 1
    assert report.failures[0].sample_id == "sample-2"
    assert "answer.value" in report.failures[0].message
    exact = report.metrics["ExactMatch"]
    assert exact.count == 1
    assert exact.mean == 1.0
    assert report.records[1].failures
    assert report.records[1].failures


def test_pipeline_can_use_custom_reference_selector():
    extractor = extractors.JsonFieldExtractor(field_path="answer")
    metric = metrics.ExactMatch()

    def reference_selector(record: core_entities.GenerationRecord):
        return record.task.metadata["reference"]

    eval_pipeline = pipeline.EvaluationPipeline(
        extractor=extractor,
        metrics=[metric],
        reference_selector=reference_selector,
    )

    sampling = core_entities.SamplingConfig(0.1, 0.9, 64)
    prompt_spec = core_entities.PromptSpec(name="tmp", template="")
    prompt_render = core_entities.PromptRender(spec=prompt_spec, text="", context={})
    model_spec = core_entities.ModelSpec(identifier="gpt-4o", provider="test")
    task = core_entities.GenerationTask(
        prompt=prompt_render,
        model=model_spec,
        sampling=sampling,
        metadata={"sample_id": "sample-3", "reference": "banana"},
    )
    generations = [
        core_entities.GenerationRecord(
            task=task,
            output=core_entities.ModelOutput(text='{ "answer": "banana" }'),
            error=None,
            metrics={},
        )
    ]

    report = eval_pipeline.evaluate(generations)

    assert report.metrics["ExactMatch"].mean == 1.0


def test_pipeline_isolates_metric_failures_without_stopping_pipeline():
    extractor = extractors.JsonFieldExtractor(field_path="answer")

    class BrokenMetric:
        name = "BrokenMetric"

        def compute(self, *, prediction, references, metadata):
            raise RuntimeError("metric bug")

    eval_pipeline = pipeline.EvaluationPipeline(
        extractor=extractor,
        metrics=[metrics.ExactMatch(), BrokenMetric()],
    )

    generations = [make_generation_record("sample-99", '{"answer": "Paris"}', "Paris")]

    report = eval_pipeline.evaluate(generations)

    assert any("BrokenMetric" in failure.message for failure in report.failures)
    broken = report.metrics["BrokenMetric"]
    assert broken.count == 0
    assert broken.mean == 0.0
    assert report.records[0].failures
    assert "BrokenMetric" in report.records[0].failures[0]


def test_pipeline_allows_reference_free_metrics():
    extractor = extractors.IdentityExtractor()
    response_length = metrics.ResponseLength()
    exact_match = metrics.ExactMatch()
    eval_pipeline = pipeline.EvaluationPipeline(
        extractor=extractor,
        metrics=[response_length, exact_match],
    )

    sampling = core_entities.SamplingConfig(0.1, 0.9, 64)
    prompt_spec = core_entities.PromptSpec(name="tmp", template="")
    prompt_render = core_entities.PromptRender(spec=prompt_spec, text="", context={})
    model_spec = core_entities.ModelSpec(identifier="gpt-4o", provider="test")
    task = core_entities.GenerationTask(
        prompt=prompt_render,
        model=model_spec,
        sampling=sampling,
        metadata={"sample_id": "sample-4"},
        reference=None,
    )
    generations = [
        core_entities.GenerationRecord(
            task=task,
            output=core_entities.ModelOutput(text="Hello world"),
            error=None,
            metrics={},
        )
    ]

    report = eval_pipeline.evaluate(generations)

    assert report.metrics["ResponseLength"].count == 1
    assert report.metrics["ExactMatch"].count == 0
    assert any("ExactMatch" in failure.message for failure in report.failures)
