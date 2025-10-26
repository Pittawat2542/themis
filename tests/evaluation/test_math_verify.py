import pytest

from themis.evaluation import extractors, math_verify_utils, metrics

pytestmark = pytest.mark.skipif(
    not math_verify_utils.math_verify_available(),
    reason="math-verify not installed",
)


def test_math_verify_extractor_grabs_last_boxed():
    extractor = extractors.MathVerifyExtractor()
    text = "Intermediate \\boxed{2} final \\boxed{3}" + " and extra"
    assert extractor.extract(text).endswith("3")


def test_math_verify_metric_validates_equivalence():
    metric = metrics.MathVerifyAccuracy()
    score = metric.compute(prediction="\\boxed{2+2}", references=["4"])
    assert score.value == 1.0
