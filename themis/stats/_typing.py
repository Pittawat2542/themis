"""Internal typing protocols for optional stats/reporting dependencies."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Protocol

from themis.types.json_types import JSONDict


class MetricAggregation(Protocol):
    """Subset of a grouped metric column used for aggregate calculations."""

    def agg(self, aggregations: Sequence[str]) -> "AggregatedMetricFrame":
        """Aggregate one grouped metric column."""
        ...

    def mean(self) -> "MeanMetricFrame":
        """Return grouped means for one metric column."""
        ...


class GroupedMetricFrame(Protocol):
    """Subset of a grouped DataFrame used by `StatsEngine.aggregate()`."""

    def __getitem__(self, column: str) -> MetricAggregation:
        """Return one grouped metric column."""
        ...

    def __iter__(self) -> Iterator[tuple[tuple[object, ...], "MetricFrame"]]:
        """Iterate grouped frames as `(group_key, frame)` tuples."""
        ...


class RecordsFrame(Protocol):
    """Subset of a DataFrame used when serializing report rows."""

    def to_dict(self, orient: str = "dict") -> list[JSONDict]:
        """Return JSON-serializable record rows."""
        ...


class AggregatedMetricFrame(Protocol):
    """Subset of an aggregated DataFrame used by tests and report building."""

    @property
    def loc(self) -> object:
        """Label-based row/column accessor."""
        ...

    def reset_index(self) -> RecordsFrame:
        """Return the aggregated rows as a flat DataFrame."""
        ...


class MeanMetricFrame(Protocol):
    """Subset of a grouped-mean frame used in comparison building."""

    def reset_index(self, name: str = "index") -> "MetricFrame":
        """Return grouped means as a flat frame with a named value column."""
        ...


class MetricFrame(Protocol):
    """Subset of a DataFrame used for metrics aggregation."""

    @property
    def empty(self) -> bool:
        """Return whether the frame contains any rows."""
        ...

    @property
    def columns(self) -> Sequence[str]:
        """Return frame column labels."""
        ...

    def __getitem__(self, key: object) -> object:
        """Return one column or a filtered frame."""
        ...

    def groupby(
        self,
        by: list[str],
        dropna: bool = True,
    ) -> GroupedMetricFrame:
        """Group the frame by the requested columns."""
        ...

    def pivot_table(
        self,
        *,
        index: str,
        columns: str,
        values: str,
        aggfunc: str,
    ) -> "MetricFrame":
        """Pivot grouped metric rows into a comparison frame."""
        ...

    def dropna(self) -> "MetricFrame":
        """Drop rows with null values."""
        ...

    def to_numpy(self) -> NumericVector:
        """Return the underlying numeric values for one selected column."""
        ...


class HashSeries(Protocol):
    """Subset of a pandas hash-series result."""

    def sum(self) -> object:
        """Return a deterministic aggregate hash seed."""
        ...


class PandasHashing(Protocol):
    """Subset of `pandas.util` used by report hashing."""

    def hash_pandas_object(self, frame: MetricFrame) -> HashSeries:
        """Hash rows in a deterministic pandas-native way."""
        ...


class PandasNamespace(Protocol):
    """Subset of pandas used by report building."""

    util: PandasHashing

    def DataFrame(self, data: list[JSONDict]) -> MetricFrame:
        """Build a DataFrame from JSON-compatible rows."""
        ...


class NumericVector(Protocol):
    """Subset of ndarray/Series behavior used by paired bootstrap."""

    def __len__(self) -> int:
        """Return the number of paired observations."""
        ...

    def __getitem__(self, index: int) -> float:
        """Return one numeric observation."""
        ...

    def __sub__(self, other: object) -> "NumericVector":
        """Subtract one vector from another."""
        ...


class BootstrapConfidenceInterval(Protocol):
    """Subset of scipy's bootstrap confidence interval wrapper."""

    low: float
    high: float


class BootstrapResult(Protocol):
    """Subset of scipy's bootstrap result."""

    confidence_interval: BootstrapConfidenceInterval


class NumpyNamespace(Protocol):
    """Subset of NumPy used by the stats engine."""

    def mean(self, values: object) -> float:
        """Return the arithmetic mean."""
        ...

    def all(self, values: object) -> bool:
        """Return whether all values evaluate truthy."""
        ...


class ScipyStatsNamespace(Protocol):
    """Subset of SciPy stats used by the stats engine."""

    def bootstrap(
        self,
        data: tuple[object, ...],
        statistic: object,
        *,
        n_resamples: int,
        confidence_level: float,
        method: str,
        random_state: int,
    ) -> BootstrapResult:
        """Run a bootstrap interval estimation."""
        ...

    def wilcoxon(self, values: object) -> tuple[float, float]:
        """Run a Wilcoxon signed-rank test."""
        ...
