"""Execution tuning example (workers + retries) with explicit specs.

This replaces the old distributed placeholder.
"""

from themis.evaluation.metric_pipeline import MetricPipeline
from themis.presets import get_benchmark_preset
from themis.session import ExperimentSession
from themis.specs import ExecutionSpec, ExperimentSpec, StorageSpec

preset = get_benchmark_preset("demo")
pipeline = MetricPipeline(extractor=preset.extractor, metrics=preset.metrics)

spec = ExperimentSpec(
    dataset=preset.load_dataset(limit=10),
    prompt=preset.prompt_template.template,
    model="fake:fake-math-llm",
    sampling={"temperature": 0.0, "max_tokens": 128},
    pipeline=pipeline,
)

report = ExperimentSession().run(
    spec,
    execution=ExecutionSpec(workers=4, max_retries=2),
    storage=StorageSpec(path=".cache/experiments", cache=True),
)

print("Completed", len(report.generation_results), "samples")
print("Failures:", len(report.failures))
