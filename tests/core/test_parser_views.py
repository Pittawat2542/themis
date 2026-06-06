from __future__ import annotations

from themis.core.config import EvaluationConfig, ParserView, TargetSpec


def test_parser_view_accepts_explicit_fallback_parsers() -> None:
    view = ParserView(
        id="answer",
        parser=TargetSpec(target="builtin/regex", kwargs={"pattern": r"Answer: (.+)"}),
        fallbacks=[TargetSpec(target="builtin/text")],
    )

    config = EvaluationConfig(parsers=[view])

    assert config.parser_views[0].id == "answer"
    assert config.parser_views[0].fallbacks == [TargetSpec(target="builtin/text")]
