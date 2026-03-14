"""Optional statistics namespace for paired-comparison tooling."""

from themis._optional import import_optional
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from themis.stats.stats_engine import ComparisonResult, StatsEngine

__all__ = ["StatsEngine", "ComparisonResult"]


def __getattr__(name: str) -> object:
    if name in __all__:
        module = import_optional("themis.stats.stats_engine", extra="stats")
        return getattr(module, name)
    raise AttributeError(f"module 'themis.stats' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
