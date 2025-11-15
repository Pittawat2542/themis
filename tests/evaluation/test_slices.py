from themis.core import entities as core_entities
from themis.evaluation import extractors, metrics, pipeline, reports


def make_generation_record(sample_id: str, raw_output: str, reference_value: str) -> core_entities.GenerationRecord:
    sampling = core_entities.SamplingConfig(temperature=0.5, top_p=0.9, max_tokens=128)
    prompt_spec = core_entities.PromptSpec(name="qa", template="Answer the question: {question}")
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


def test_pipeline_slices_aggregate_correctly():
    extractor = extractors.JsonFieldExtractor(field_path="answer")
    metric = metrics.ExactMatch()
    eval_pipeline = pipeline.EvaluationPipeline(extractor=extractor, metrics=[metric])

    # Register a slice for only sample-1
    eval_pipeline.register_slice(
        "only_sample_1",
        lambda rec: rec.task.metadata.get("sample_id") == "sample-1",
    )

    generations = [
        make_generation_record("sample-1", '{"answer": "Paris"}', "Paris"),
        make_generation_record("sample-2", '{"answer": "Lyon"}', "Paris"),
    ]

    report = eval_pipeline.evaluate(generations)

    assert "only_sample_1" in report.slices
    exact_slice = report.slices["only_sample_1"]["ExactMatch"]
    assert exact_slice.count == 1
    assert exact_slice.mean == 1.0

    # CI for slice
    ci = reports.ci_for_slice_metric(report, "only_sample_1", "ExactMatch", n_bootstrap=500)
    assert ci.statistic == 1.0
    assert ci.ci_lower <= ci.statistic <= ci.ci_upper

    # Confusion matrix utility
    cm = reports.confusion_matrix(["correct", "wrong", "correct"], ["correct", "wrong", "wrong"])
    assert cm["correct"]["correct"] == 1
    assert cm["correct"]["wrong"] == 1
    assert cm["wrong"]["wrong"] == 1