"""Convenient constructors for shipped metrics."""

from themis.catalog.builtins.metrics import ExactMatchMetric


def exact_match() -> ExactMatchMetric:
    """Create the shipped exact-match metric."""

    return ExactMatchMetric()


__all__ = ["exact_match"]
