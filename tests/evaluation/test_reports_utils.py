from themis.core import entities as core_entities
from themis.evaluation import metrics, extractors, pipeline, reports


def make_generation_record(sample_id: str, raw_output: str, reference_value: str) -> core_entities.GenerationRecord:
    sampling = core_entities.SamplingConfig(temperature=0.5, top_p=0.9, max_tokens=64)
    prompt_spec = core_entities.PromptSpec(name="qa", template="Answer: {answer}")
    prompt_render = core_entities.PromptRender(spec=prompt_spec, text=raw_output, context={"answer": reference_value})
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


def test_reports_ci_and_permutation_and_effect_sizes():
    extractor = extractors.IdentityExtractor()
    metric = metrics.ExactMatch()
    eval_pipeline = pipeline.EvaluationPipeline(extractor=extractor, metrics=[metric])

    # Two small groups with clear difference (A: all correct, B: mixed)
    group_a = [
        make_generation_record("a1", "Paris", "Paris"),
        make_generation_record("a2", "Paris", "Paris"),
    ]
    group_b = [
        make_generation_record("b1", "Paris", "Paris"),
        make_generation_record("b2", "Lyon", "Paris"),
    ]

    report_a = eval_pipeline.evaluate(group_a)
    report_b = eval_pipeline.evaluate(group_b)

    # CI
    ci_a = reports.ci_for_metric(report_a, "ExactMatch", n_bootstrap=500)
    assert 0.0 <= ci_a.ci_lower <= ci_a.ci_upper <= 1.0

    # Permutation test & Holm correction
    cmp = reports.compare_reports_with_holm(report_a, report_b, ["ExactMatch"], n_permutations=500, seed=42)
    holm_flags = cmp["holm_significant"]
    assert isinstance(holm_flags, list)
    assert len(holm_flags) == 1

    # Effect sizes
    h = reports.cohens_h_for_metric(report_a, report_b, "ExactMatch")
    assert isinstance(h.value, float)
