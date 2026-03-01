"""Tests for themis.evaluation.conditional module.

Covers ConditionalMetric, AdaptiveEvaluationPipeline, and selector helpers.
"""

from themis.core.entities import (
    GenerationRecord,
    GenerationTask,
    MetricScore,
    ModelOutput,
    ModelSpec,
    PromptRender,
    PromptSpec,
    Reference,
    SamplingConfig,
)
from themis.evaluation.conditional import (
    AdaptiveEvaluationPipeline,
    ConditionalMetric,
    combine_selectors,
    select_by_condition,
    select_by_metadata_field,
)
from themis.interfaces import Metric


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class DummyMetric(Metric):
    name = "dummy"
    requires_reference = False

    def compute(self, *, prediction, references, metadata=None) -> MetricScore:
        return MetricScore(self.name, 1.0)


class FailingConditionMetric(Metric):
    name = "failing"
    requires_reference = True

    def compute(self, *, prediction, references, metadata=None) -> MetricScore:
        return MetricScore(self.name, float(prediction == references[0]))


def _make_record(
    text: str = "output",
    metadata: dict | None = None,
    sample_id: str = "s1",
    reference: str | None = None,
) -> GenerationRecord:
    prompt = PromptRender(
        spec=PromptSpec(name="test", template=""),
        text="prompt",
        context={},
        metadata={},
    )
    md = {"dataset_id": sample_id, **(metadata or {})}
    ref = Reference(kind="answer", value=reference) if reference else None
    return GenerationRecord(
        task=GenerationTask(
            prompt=prompt,
            model=ModelSpec(identifier="mock", provider="mock"),
            sampling=SamplingConfig(),
            metadata=md,
            reference=ref,
        ),
        output=ModelOutput(text=text),
        error=None,
    )


class IdentityExtractor:
    """Pass-through extractor."""

    def extract(self, text: str) -> str:
        return text


# ---------------------------------------------------------------------------
# ConditionalMetric
# ---------------------------------------------------------------------------


class TestConditionalMetric:
    def test_condition_met_computes(self):
        cm = ConditionalMetric(
            metric=DummyMetric(),
            condition=lambda r: True,
        )
        score = cm.compute_or_default(
            _make_record(),
            prediction="x",
            references=[],
        )
        assert score.value == 1.0
        assert score.metadata.get("skipped") is not True

    def test_condition_not_met_returns_default(self):
        cm = ConditionalMetric(
            metric=DummyMetric(),
            condition=lambda r: False,
            default_score=0.5,
        )
        score = cm.compute_or_default(
            _make_record(),
            prediction="x",
            references=[],
        )
        assert score.value == 0.5
        assert score.metadata["skipped"] is True

    def test_condition_raises_returns_default(self):
        cm = ConditionalMetric(
            metric=DummyMetric(),
            condition=lambda r: 1 / 0,  # ZeroDivisionError
        )
        assert cm.should_evaluate(_make_record()) is False

    def test_name_defaults_to_conditional_prefix(self):
        cm = ConditionalMetric(
            metric=DummyMetric(),
            condition=lambda r: True,
        )
        assert cm.name == "conditional_dummy"

    def test_name_can_be_overridden(self):
        cm = ConditionalMetric(
            metric=DummyMetric(),
            condition=lambda r: True,
            name="custom_name",
        )
        assert cm.name == "custom_name"

    def test_requires_reference_delegates(self):
        cm = ConditionalMetric(
            metric=DummyMetric(),
            condition=lambda r: True,
        )
        assert (
            cm.requires_reference is False
        )  # DummyMetric has requires_reference=False

        cm2 = ConditionalMetric(
            metric=FailingConditionMetric(),
            condition=lambda r: True,
        )
        assert cm2.requires_reference is True


# ---------------------------------------------------------------------------
# Selectors
# ---------------------------------------------------------------------------


class TestSelectors:
    def test_select_by_metadata_field(self):
        m1, m2 = DummyMetric(), FailingConditionMetric()
        selector = select_by_metadata_field(
            "type",
            {"math": [m1], "code": [m2]},
            default=[m1],
        )
        math_record = _make_record(metadata={"type": "math"})
        code_record = _make_record(metadata={"type": "code"})
        other_record = _make_record(metadata={"type": "text"})

        assert selector(math_record) == [m1]
        assert selector(code_record) == [m2]
        assert selector(other_record) == [m1]  # default

    def test_select_by_condition(self):
        m1, m2 = DummyMetric(), FailingConditionMetric()
        selector = select_by_condition(
            condition=lambda r: r.task.metadata.get("hard") is True,
            metrics_if_true=[m1, m2],
            metrics_if_false=[m1],
        )
        hard = _make_record(metadata={"hard": True})
        easy = _make_record(metadata={"hard": False})

        assert len(selector(hard)) == 2
        assert len(selector(easy)) == 1

    def test_combine_selectors_deduplicates(self):
        m1 = DummyMetric()
        s1 = lambda r: [m1]  # noqa: E731
        s2 = lambda r: [m1]  # noqa: E731

        combined = combine_selectors(s1, s2)
        result = combined(_make_record())
        assert len(result) == 1  # deduplicated by name


# ---------------------------------------------------------------------------
# AdaptiveEvaluationPipeline
# ---------------------------------------------------------------------------


class TestAdaptiveEvaluationPipeline:
    def test_groups_and_evaluates(self):
        m1 = DummyMetric()

        def selector(record):
            return [m1]

        pipe = AdaptiveEvaluationPipeline(
            extractor=IdentityExtractor(),
            metric_selector=selector,
        )

        records = [_make_record(text="a", sample_id="s1")]
        report = pipe.evaluate(records)
        assert "dummy" in report.metrics
        assert report.metrics["dummy"].count == 1
