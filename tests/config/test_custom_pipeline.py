from themis.config.schema import (
    PipelineConfig,
    ExtractorConfig,
    MetricConfig,
)
from themis.config.runtime import build_pipeline_from_config
from themis.evaluation.extractors import RegexExtractor
from themis.evaluation.metrics import ExactMatch, RubricJudgeMetric


def test_build_pipeline_from_config():
    cfg = PipelineConfig(
        extractor=ExtractorConfig(
            name="regex", options={"pattern": r"<answer>(.*?)</answer>"}
        ),
        metrics=[
            MetricConfig(name="exact_match", options={"case_sensitive": False}),
            MetricConfig(
                name="rubric_judge",
                options={
                    "rubric": ["formatting", "correctness"],
                    "judge_model": {"identifier": "gpt-4o", "provider": "fake"},
                    "judge_executor": {"name": "fake"},
                },
            ),
        ],
    )

    pipeline = build_pipeline_from_config(cfg)

    assert isinstance(pipeline._extractor, RegexExtractor)
    assert pipeline._extractor.pattern == r"<answer>(.*?)</answer>"

    assert len(pipeline._metrics) == 2
    assert isinstance(pipeline._metrics[0], ExactMatch)
    assert pipeline._metrics[0].case_sensitive is False

    assert isinstance(pipeline._metrics[1], RubricJudgeMetric)
    assert pipeline._metrics[1].judge_model.identifier == "gpt-4o"
    assert "formatting" in pipeline._metrics[1].rubric[0]
