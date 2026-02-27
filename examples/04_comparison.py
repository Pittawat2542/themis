"""Compare two runs."""

from pathlib import Path

import themis
from themis.comparison import compare_runs
from themis.comparison import StatisticalTest
from themis.presets import get_benchmark_preset
from themis.storage import ExperimentStorage


def run_experiment(run_id: str, temperature: float) -> None:
    preset = get_benchmark_preset("demo")
    themis.evaluate(
        preset.load_dataset(limit=10),
        model="fake:fake-math-llm",
        prompt=preset.prompt_template.template,
        metrics=[m.name for m in preset.metrics],
        temperature=temperature,
        max_tokens=128,
        workers=2,
        run_id=run_id,
        storage=".cache/experiments",
        resume=True,
    )


def main() -> None:
    run_a = "demo-compare-a"
    run_b = "demo-compare-b"
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
