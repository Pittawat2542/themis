"""Optional metadata hooks for non-Pydantic config objects."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ConfigReportOptions:
    """Optional display and traversal hints for config reporting."""

    display_name: str | None = None
    hidden_fields: set[str] = field(default_factory=set)
    child_fields: set[str] = field(default_factory=set)
    redacted_fields: set[str] = field(default_factory=set)
    paper_fields: set[str] = field(default_factory=set)
    non_paper_fields: set[str] = field(default_factory=set)


class ConfigReportMixin:
    """Mixin that lets custom config objects provide report hints."""

    __config_report_options__ = ConfigReportOptions()


def config_reportable(
    *,
    display_name: str | None = None,
    hidden_fields: set[str] | None = None,
    child_fields: set[str] | None = None,
    redacted_fields: set[str] | None = None,
    paper_fields: set[str] | None = None,
    non_paper_fields: set[str] | None = None,
):
    """Attach lightweight config-report hints to a class."""

    options = ConfigReportOptions(
        display_name=display_name,
        hidden_fields=set(hidden_fields or set()),
        child_fields=set(child_fields or set()),
        redacted_fields=set(redacted_fields or set()),
        paper_fields=set(paper_fields or set()),
        non_paper_fields=set(non_paper_fields or set()),
    )

    def decorator(cls):
        cls.__config_report_options__ = options
        return cls

    return decorator


def get_config_report_options(cls: type[object]) -> ConfigReportOptions:
    """Merge report options across the class MRO."""

    merged = ConfigReportOptions()
    for base in reversed(cls.__mro__):
        options = getattr(base, "__config_report_options__", None)
        if options is None:
            continue
        new_paper = (merged.paper_fields | set(options.paper_fields)) - set(
            options.non_paper_fields
        )
        new_non_paper = (merged.non_paper_fields | set(options.non_paper_fields)) - set(
            options.paper_fields
        )
        merged = ConfigReportOptions(
            display_name=options.display_name or merged.display_name,
            hidden_fields=merged.hidden_fields | set(options.hidden_fields),
            child_fields=merged.child_fields | set(options.child_fields),
            redacted_fields=merged.redacted_fields | set(options.redacted_fields),
            paper_fields=new_paper,
            non_paper_fields=new_non_paper,
        )
    return merged
