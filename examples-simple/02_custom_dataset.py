"""Custom dataset example using ExperimentSpec + ExperimentSession."""

from themis.evaluation import extractors, metrics
from themis.evaluation.metric_pipeline import MetricPipeline
from themis.session import ExperimentSession
from themis.specs import ExecutionSpec, ExperimentSpec, StorageSpec

DATASET = [
    {"id": "q1", "question": "What is 10 + 5?", "answer": "15"},
    {"id": "q2", "question": "What is 20 - 8?", "answer": "12"},
    {"id": "q3", "question": "What is 6 * 7?", "answer": "42"},
]

pipeline = MetricPipeline(
    extractor=extractors.IdentityExtractor(),
    metrics=[metrics.ExactMatch(), metrics.ResponseLength()],
)

spec = ExperimentSpec(
    dataset=DATASET,
    prompt="Q: {question}\nA:",
    model="fake:fake-math-llm",
    sampling={"temperature": 0.0, "max_tokens": 128},
    pipeline=pipeline,
)

report = ExperimentSession().run(
    spec,
    execution=ExecutionSpec(workers=2),
    storage=StorageSpec(path=".cache/experiments", cache=False),
)

print("Samples:", len(report.generation_results))
for name, aggregate in sorted(report.evaluation_report.metrics.items()):
    print(f"{name}: mean={aggregate.mean:.4f} (n={aggregate.count})")
