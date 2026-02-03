from __future__ import annotations

from themis.core import entities as core_entities
from themis.evaluation import extractors, metrics
from themis.evaluation.pipelines.composable_pipeline import (
    ComposableEvaluationPipeline,
    ComposableEvaluationReportPipeline,
)
from themis.experiment import builder as experiment_builder
from themis.generation import templates


def test_experiment_builder_uses_pipeline_factory(tmp_path):
    template = templates.PromptTemplate(name="t", template="What is 2+2?")
    sampling = core_entities.SamplingConfig(temperature=0.1, top_p=0.9, max_tokens=32)
    model_spec = core_entities.ModelSpec(identifier="fake-model", provider="fake")
    binding = experiment_builder.ModelBinding(spec=model_spec, provider_name="fake")

    definition = experiment_builder.ExperimentDefinition(
        templates=[template],
        sampling_parameters=[sampling],
        model_bindings=[binding],
        dataset_id_field="id",
        reference_field=None,
    )

    factory_called = {"value": False}

    def pipeline_factory(extractor, metric_list):
        factory_called["value"] = True
        pipeline = (
            ComposableEvaluationPipeline()
            .extract(extractor)
            .compute_metrics(metric_list, references=[])
        )
        return ComposableEvaluationReportPipeline(pipeline)

    builder = experiment_builder.ExperimentBuilder(
        extractor=extractors.IdentityExtractor(),
        metrics=[metrics.ResponseLength()],
        pipeline_factory=pipeline_factory,
    )

    built = builder.build(definition, storage_dir=tmp_path / "storage")
    dataset = [{"id": "row-1"}]

    report = built.orchestrator.run(dataset=dataset, run_id="builder-factory-test", resume=False)

    assert factory_called["value"] is True
    assert "ResponseLength" in report.evaluation_report.metrics
