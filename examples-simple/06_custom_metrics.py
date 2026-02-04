"""Custom metric example with ExperimentSpec + ExperimentSession."""

from dataclasses import dataclass
from typing import Any, Sequence

from themis.core.entities import MetricScore
from themis.evaluation import extractors, metrics
from themis.evaluation.metric_pipeline import MetricPipeline
from themis.interfaces import Metric
from themis.session import ExperimentSession
from themis.specs import ExecutionSpec, ExperimentSpec, StorageSpec


@dataclass
class ContainsKeywordMetric(Metric):
    keyword: str = "because"

    def __post_init__(self) -> None:
        self.name = f"contains_{self.keyword}"

    def compute(
        self,
        *,
        prediction: Any,
        references: Sequence[Any],
        metadata: dict[str, Any] | None = None,
    ) -> MetricScore:
        text = str(prediction).lower()
        hit = self.keyword.lower() in text
        return MetricScore(
            metric_name=self.name,
            value=1.0 if hit else 0.0,
            details={"keyword": self.keyword, "found": hit},
            metadata=metadata or {},
        )


DATASET = [
    {"id": "1", "question": "Why is the sky blue?", "answer": "Rayleigh scattering"},
    {"id": "2", "question": "What is 2+2?", "answer": "4"},
]

pipeline = MetricPipeline(
    extractor=extractors.IdentityExtractor(),
    metrics=[metrics.ExactMatch(), ContainsKeywordMetric(keyword="because")],
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

for name, aggregate in sorted(report.evaluation_report.metrics.items()):
    print(f"{name}: mean={aggregate.mean:.4f} (n={aggregate.count})")
