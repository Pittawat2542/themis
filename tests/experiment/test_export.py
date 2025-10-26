import csv
import json

from themis.core import entities as core_entities
from themis.evaluation import reports as evaluation_reports
from themis.experiment import export as experiment_export
from themis.experiment import orchestrator
from themis.generation import templates
from themis.project import AblationChart, AblationChartPoint


def _sample_report() -> orchestrator.ExperimentReport:
    template = templates.PromptTemplate(name="demo", template="Explain {topic}")
    sampling = core_entities.SamplingConfig(temperature=0.2, top_p=0.95, max_tokens=32)
    model = core_entities.ModelSpec(identifier="demo-model", provider="fake")
    prompt = core_entities.PromptRender(spec=template, text="Explain graphs", context={})
    task = core_entities.GenerationTask(
        prompt=prompt,
        model=model,
        sampling=sampling,
        metadata={"dataset_id": "sample-1", "subject": "math"},
    )
    generation_record = core_entities.GenerationRecord(
        task=task,
        output=core_entities.ModelOutput(text="42"),
        error=None,
    )

    metric_score = core_entities.MetricScore(metric_name="ExactMatch", value=0.8)
    evaluation_record = core_entities.EvaluationRecord(
        sample_id="sample-1",
        scores=[metric_score],
        failures=[],
    )
    aggregate = evaluation_reports.MetricAggregate(
        name="ExactMatch",
        count=1,
        mean=0.8,
        per_sample=[metric_score],
    )
    evaluation_report = evaluation_reports.EvaluationReport(
        metrics={"ExactMatch": aggregate},
        failures=[],
        records=[evaluation_record],
    )

    return orchestrator.ExperimentReport(
        generation_results=[generation_record],
        evaluation_report=evaluation_report,
        failures=[],
        metadata={"run_id": "demo", "total_samples": 1},
    )


def test_export_report_csv_writes_expected_columns(tmp_path):
    report = _sample_report()
    csv_path = tmp_path / "report.csv"

    experiment_export.export_report_csv(report, csv_path)

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert rows
    row = rows[0]
    assert row["sample_id"] == "sample-1"
    assert row["subject"] == "math"
    assert row["model_identifier"] == "demo-model"
    assert row["metric:ExactMatch"] == "0.8"


def test_render_html_report_includes_chart_and_tables():
    report = _sample_report()
    chart = AblationChart(
        title="Temperature sweep",
        x_label="temp",
        y_label="score",
        metric_name="ExactMatch",
        points=(
            AblationChartPoint(x_value=0.0, label="0.0", metric_value=0.4, metric_name="ExactMatch", count=2),
            AblationChartPoint(x_value=0.2, label="0.2", metric_value=0.8, metric_name="ExactMatch", count=2),
        ),
    )

    html_doc = experiment_export.render_html_report(report, charts=[chart], title="Demo report")

    assert "Demo report" in html_doc
    assert "ExactMatch" in html_doc
    assert "chart-section" in html_doc
    assert "svg" in html_doc


def test_export_report_json_serializes_samples_and_charts(tmp_path):
    report = _sample_report()
    chart = AblationChart(
        title="Temperature sweep",
        x_label="temp",
        y_label="score",
        metric_name="ExactMatch",
        points=(
            AblationChartPoint(
                x_value=0.0,
                label="0.0",
                metric_value=0.4,
                metric_name="ExactMatch",
                count=2,
            ),
        ),
    )
    json_path = tmp_path / "report.json"

    experiment_export.export_report_json(
        report,
        json_path,
        charts=[chart],
        title="Demo",
        sample_limit=10,
    )

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["title"] == "Demo"
    assert payload["charts"][0]["title"] == "Temperature sweep"
    assert payload["metrics"][0]["name"] == "ExactMatch"
    assert payload["samples"][0]["scores"][0]["value"] == 0.8
