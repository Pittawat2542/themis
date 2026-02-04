"""Compare two vNext runs."""

from pathlib import Path

from themis.comparison import compare_runs
from themis.comparison.statistics import StatisticalTest
from themis.evaluation.metric_pipeline import MetricPipeline
from themis.presets import get_benchmark_preset
from themis.session import ExperimentSession
from themis.specs import ExecutionSpec, ExperimentSpec, StorageSpec
from themis.storage import ExperimentStorage


def run_experiment(run_id: str, temperature: float) -> None:
    preset = get_benchmark_preset("demo")
    pipeline = MetricPipeline(extractor=preset.extractor, metrics=preset.metrics)
    spec = ExperimentSpec(
        dataset=preset.load_dataset(limit=10),
        prompt=preset.prompt_template.template,
        model="fake:fake-math-llm",
        sampling={"temperature": temperature, "max_tokens": 128},
        pipeline=pipeline,
        run_id=run_id,
    )
    ExperimentSession().run(
        spec,
        execution=ExecutionSpec(workers=2),
        storage=StorageSpec(path=".cache/experiments", cache=False),
    )


def main() -> None:
    run_a = "demo-vnext-a"
    run_b = "demo-vnext-b"
    storage = ExperimentStorage(".cache/experiments")
    for run_id in (run_a, run_b):
        try:
            storage.delete_run(run_id)
        except FileNotFoundError:
            pass

    run_experiment(run_a, temperature=0.0)
    run_experiment(run_b, temperature=0.7)

    report = compare_runs(
        run_ids=[run_a, run_b],
        storage_path=Path(".cache/experiments"),
        statistical_test=StatisticalTest.BOOTSTRAP,
        alpha=0.05,
    )

    print(report.summary(include_details=False))


if __name__ == "__main__":
    main()
