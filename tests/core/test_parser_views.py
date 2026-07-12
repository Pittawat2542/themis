from __future__ import annotations

from themis.core.config import (
    EvaluationConfig,
    GenerationConfig,
    ParserView,
    StorageConfig,
    TargetSpec,
)
from themis.core.experiment import Experiment
from themis.core.models import Case, Dataset


def test_parser_view_accepts_explicit_fallback_parsers() -> None:
    view = ParserView(
        id="answer",
        parser=TargetSpec(target="builtin/regex", kwargs={"pattern": r"Answer: (.+)"}),
        fallbacks=[TargetSpec(target="builtin/text")],
    )

    config = EvaluationConfig(parsers=[view])

    assert config.parser_views[0].id == "answer"
    assert config.parser_views[0].fallbacks == [TargetSpec(target="builtin/text")]


def test_parser_view_fallback_executes_when_primary_parser_fails() -> None:
    experiment = Experiment(
        generation=GenerationConfig(
            generator="builtin/demo_generator",
            candidate_policy={"num_samples": 1},
            reducer="builtin/majority_vote",
        ),
        evaluation=EvaluationConfig(
            parsers=[
                ParserView(
                    id="default",
                    parser=TargetSpec(
                        target="builtin/regex",
                        kwargs={"pattern": r"Answer: (.+)", "group": 1},
                    ),
                    fallbacks=[TargetSpec(target="builtin/text")],
                )
            ],
            metrics=["builtin/exact_match"],
        ),
        storage=StorageConfig(target="memory"),
        dataset_sources=[
            Dataset(
                dataset_id="dataset-1",
                cases=[
                    Case(
                        case_id="case-1", input={"question": "2+2"}, expected_output="4"
                    )
                ],
            )
        ],
    )

    result = experiment.run()

    assert result.cases[0].parse_errors == {}
    assert result.cases[0].parsed_views["default"].value == "4"
    assert result.cases[0].metric_results[0].value == 1.0
