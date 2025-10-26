
from themis.core import entities as core_entities
from themis.evaluation import extractors, metrics
from themis.experiment import builder as experiment_builder
from themis.generation import templates
from themis.generation import strategies


def test_experiment_builder_respects_strategy_resolver(tmp_path):
    template = templates.PromptTemplate(name="t", template="Explain {topic}")
    sampling = core_entities.SamplingConfig(temperature=0.1, top_p=0.9, max_tokens=32)
    model_spec = core_entities.ModelSpec(identifier="fake-model", provider="fake")
    binding = experiment_builder.ModelBinding(spec=model_spec, provider_name="fake")

    definition = experiment_builder.ExperimentDefinition(
        templates=[template],
        sampling_parameters=[sampling],
        model_bindings=[binding],
        dataset_id_field="id",
        reference_field="answer",
        metadata_fields=("subject",),
        context_builder=lambda row: {"topic": row["topic"]},
    )

    resolver_invocations = {"count": 0}

    def resolver(task):
        resolver_invocations["count"] += 1
        return strategies.RepeatedSamplingStrategy(attempts=2)

    builder = experiment_builder.ExperimentBuilder(
        extractor=extractors.JsonFieldExtractor(field_path="answer"),
        metrics=[metrics.ExactMatch()],
        strategy_resolver=resolver,
    )

    built = builder.build(definition, storage_dir=tmp_path / "storage")
    dataset = [
        {"id": "row-1", "topic": "graphs", "answer": "ok", "subject": "math"},
    ]

    report = built.orchestrator.run(
        dataset=dataset, run_id="builder-test", resume=False
    )

    assert report.metadata["total_samples"] == len(dataset)
    assert resolver_invocations["count"] == len(dataset)
    record = report.generation_results[0]
    assert record.metrics["attempt_count"] == 2
