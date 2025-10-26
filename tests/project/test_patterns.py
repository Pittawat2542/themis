import pytest

from themis.core import entities as core_entities
from themis.evaluation import reports as evaluation_reports
from themis.experiment import builder as experiment_builder
from themis.experiment import orchestrator
from themis.generation import templates
from themis.project import (
    AblationVariant,
    Project,
    XAbationPattern,
)


def _definition_for_variant(variant):
    template = templates.PromptTemplate(
        name=f"tmpl-{variant.slug()}",
        template="Explain {problem}",
    )
    sampling = core_entities.SamplingConfig(temperature=0.0, top_p=1.0, max_tokens=32)
    model_spec = core_entities.ModelSpec(
        identifier=f"model-{variant.slug()}", provider="fake"
    )
    binding = experiment_builder.ModelBinding(spec=model_spec, provider_name="fake")
    return experiment_builder.ExperimentDefinition(
        templates=[template],
        sampling_parameters=[sampling],
        model_bindings=[binding],
    )


def _mock_report(metric_name: str, mean: float) -> orchestrator.ExperimentReport:
    metric = evaluation_reports.MetricAggregate(
        name=metric_name, count=10, mean=mean, per_sample=[]
    )
    eval_report = evaluation_reports.EvaluationReport(
        metrics={metric_name: metric}, failures=[], records=[]
    )
    return orchestrator.ExperimentReport(
        generation_results=[],
        evaluation_report=eval_report,
        failures=[],
        metadata={},
    )


def test_x_ablation_materialize_registers_experiments():
    project = Project(project_id="proj", name="Demo")
    pattern = XAbationPattern(
        name="temperature-sweep",
        parameter_name="temperature",
        values=[0.0, 0.2, 0.4],
        definition_builder=_definition_for_variant,
        metric_name="ExactMatch",
        x_axis_label="Temperature",
        y_axis_label="Exact match",
    )

    application = pattern.materialize(
        project,
        description_template="temperature={value_label}",
        base_tags=("demo",),
    )

    assert tuple(project.list_experiment_names()) == tuple(
        experiment.name for experiment in application.experiments
    )
    assert len(application.experiments) == 3
    sample = application.experiments[1]
    assert sample.metadata["parameter_value"] == 0.2
    assert sample.metadata["pattern"] == "x-ablation"
    assert "x-ablation" in sample.tags
    assert sample.description == "temperature=0.2"


def test_x_ablation_builds_chart_from_reports():
    project = Project(project_id="proj", name="Demo")
    pattern = XAbationPattern(
        name="prompt-style",
        parameter_name="style",
        values=[
            AblationVariant(value="system"),
            AblationVariant(value="cot", label="Chain-of-thought"),
        ],
        definition_builder=_definition_for_variant,
        metric_name="ExactMatch",
        x_axis_label="Style",
        y_axis_label="Score",
    )
    application = pattern.materialize(project)

    reports = {
        application.experiments[0].name: _mock_report("ExactMatch", 0.5),
        application.experiments[1].name: _mock_report("ExactMatch", 0.7),
    }

    chart = application.build_chart(reports)

    assert chart.metric_name == "ExactMatch"
    assert [point.label for point in chart.points] == ["system", "Chain-of-thought"]
    assert chart.points[0].metric_value == pytest.approx(0.5)
    assert chart.points[1].metric_value == pytest.approx(0.7)


def test_x_ablation_chart_requires_metric():
    project = Project(project_id="proj", name="Demo")
    pattern = XAbationPattern(
        name="prompt-style",
        parameter_name="style",
        values=["baseline"],
        definition_builder=_definition_for_variant,
        metric_name="ExactMatch",
    )
    application = pattern.materialize(project)

    missing_metric_report = orchestrator.ExperimentReport(
        generation_results=[],
        evaluation_report=evaluation_reports.EvaluationReport(
            metrics={}, failures=[], records=[]
        ),
        failures=[],
        metadata={},
    )

    reports = {application.experiments[0].name: missing_metric_report}
    with pytest.raises(ValueError):
        application.build_chart(reports)
