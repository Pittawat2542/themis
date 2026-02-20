"""Custom metric example with themis.evaluate()."""

from dataclasses import dataclass
from typing import Any, Sequence

import themis
from themis.core.entities import MetricScore
from themis.interfaces import Metric


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


@dataclass
class ContainsBecauseMetric(ContainsKeywordMetric):
    keyword: str = "because"


themis.register_metric("contains_because", ContainsBecauseMetric)

report = themis.evaluate(
    DATASET,
    model="fake:fake-math-llm",
    prompt="Q: {question}\nA:",
    metrics=["exact_match", "contains_because"],
    temperature=0.0,
    max_tokens=128,
    workers=2,
    storage=".cache/experiments",
    resume=False,
)

for name, aggregate in sorted(report.evaluation_report.metrics.items()):
    print(f"{name}: mean={aggregate.mean:.4f} (n={aggregate.count})")
