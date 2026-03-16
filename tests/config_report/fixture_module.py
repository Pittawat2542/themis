"""Fixtures with source comments that config-report tests can introspect."""

from dataclasses import dataclass

from themis.config_report import config_reportable


@dataclass(frozen=True)
class CommentedConfig:
    alpha: int = 1  # Inline comment attached to alpha.
    # Leading comment attached to beta.
    beta: str = "two"


@config_reportable(
    paper_fields={"forced_param"},
    non_paper_fields={"ordinary_param"},
)
@dataclass(frozen=True)
class VerbosityOverrideConfig:
    required_param: str
    ordinary_param: int = 1
    forced_param: int = 1
