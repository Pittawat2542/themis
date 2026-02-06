import pytest

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

    # Two small paired groups with clear difference (A: all correct, B: mixed)
    group_a = [
        make_generation_record("sample-1", "Paris", "Paris"),
        make_generation_record("sample-2", "Paris", "Paris"),
    ]
    group_b = [
        make_generation_record("sample-1", "Paris", "Paris"),
        make_generation_record("sample-2", "Lyon", "Paris"),
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


def test_reports_align_by_sample_id_for_paired_tests():
    extractor = extractors.IdentityExtractor()
    metric = metrics.ExactMatch()
    eval_pipeline = pipeline.EvaluationPipeline(extractor=extractor, metrics=[metric])

    group_a = [
        make_generation_record("sample-1", "Paris", "Paris"),
        make_generation_record("sample-2", "Paris", "Paris"),
    ]
    group_b = [
        make_generation_record("sample-2", "Paris", "Paris"),
        make_generation_record("sample-1", "Lyon", "Paris"),
    ]

    report_a = eval_pipeline.evaluate(group_a)
    report_b = eval_pipeline.evaluate(group_b)

    values_a, values_b = reports.aligned_metric_values(
        report_a, report_b, "ExactMatch"
    )
    assert values_a == [1.0, 1.0]
    assert values_b == [0.0, 1.0]

    paired = reports.paired_permutation_test_for_metric(
        report_a, report_b, "ExactMatch", n_permutations=200, seed=1
    )
    assert 0.0 <= paired.p_value <= 1.0


def test_reports_alignment_averages_duplicate_sample_ids():
    extractor = extractors.IdentityExtractor()
    metric = metrics.ExactMatch()
    eval_pipeline = pipeline.EvaluationPipeline(extractor=extractor, metrics=[metric])

    # Duplicate sample_id values are intentionally present and should be aggregated
    # deterministically (mean across duplicates) before pairing.
    report_a = eval_pipeline.evaluate(
        [
            make_generation_record("sample-1", "Paris", "Paris"),  # 1.0
            make_generation_record("sample-1", "Lyon", "Paris"),   # 0.0
        ]
    )
    report_b = eval_pipeline.evaluate(
        [
            make_generation_record("sample-1", "Paris", "Paris"),  # 1.0
            make_generation_record("sample-1", "Paris", "Paris"),  # 1.0
        ]
    )

    values_a, values_b = reports.aligned_metric_values(report_a, report_b, "ExactMatch")
    assert values_a == [0.5]
    assert values_b == [1.0]


def test_reports_alignment_requires_overlap():
    extractor = extractors.IdentityExtractor()
    metric = metrics.ExactMatch()
    eval_pipeline = pipeline.EvaluationPipeline(extractor=extractor, metrics=[metric])

    report_a = eval_pipeline.evaluate(
        [make_generation_record("sample-1", "Paris", "Paris")]
    )
    report_b = eval_pipeline.evaluate(
        [make_generation_record("sample-2", "Paris", "Paris")]
    )

    with pytest.raises(ValueError):
        reports.aligned_metric_values(report_a, report_b, "ExactMatch")

    with pytest.raises(ValueError):
        reports.paired_t_test_for_metric(report_a, report_b, "ExactMatch")


def test_inferential_helpers_reject_truncated_per_sample_vectors():
    truncated = reports.EvaluationReport(
        metrics={
            "ExactMatch": reports.MetricAggregate(
                name="ExactMatch",
                count=10,
                mean=0.8,
                per_sample=[
                    core_entities.MetricScore(metric_name="ExactMatch", value=1.0),
                    core_entities.MetricScore(metric_name="ExactMatch", value=0.0),
                ],
                per_sample_complete=False,
                truncated_count=8,
            )
        },
        failures=[],
        records=[],
        metadata={"per_sample_metrics_truncated": True},
    )

    with pytest.raises(ValueError, match="truncated subset"):
        reports.ci_for_metric(truncated, "ExactMatch")

    with pytest.raises(ValueError, match="truncated subset"):
        reports.permutation_test_for_metric(truncated, truncated, "ExactMatch")
