"""Reusable experiment patterns for organizing projects."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from themis.experiment.definitions import ExperimentDefinition
from themis.experiment.orchestrator import ExperimentReport
from themis.project.definitions import Project, ProjectExperiment


def _slugify(value: str) -> str:
    text = value.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = text.strip("-")
    return text or "variant"


@dataclass
class AblationVariant:
    value: Any
    label: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def display_label(self) -> str:
        return self.label or str(self.value)

    def slug(self) -> str:
        return _slugify(self.display_label())


@dataclass(frozen=True)
class AblationChartPoint:
    x_value: Any
    label: str
    metric_value: float
    metric_name: str
    count: int


@dataclass(frozen=True)
class AblationChart:
    title: str
    x_label: str
    y_label: str
    metric_name: str
    points: tuple[AblationChartPoint, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "x_label": self.x_label,
            "y_label": self.y_label,
            "metric": self.metric_name,
            "points": [
                {
                    "label": point.label,
                    "x": point.x_value,
                    "value": point.metric_value,
                    "count": point.count,
                }
                for point in self.points
            ],
        }


@dataclass(frozen=True)
class XAbationPatternApplication:
    pattern_name: str
    parameter_name: str
    experiments: tuple[ProjectExperiment, ...]
    variant_by_name: Mapping[str, AblationVariant]
    _pattern: "XAblationPattern" = field(repr=False)

    def build_chart(self, reports: Mapping[str, ExperimentReport]) -> AblationChart:
        return self._pattern._build_chart(reports, self.variant_by_name)


class XAbationPattern:
    """Vary a single factor across values to compare performance."""

    pattern_type = "x-ablation"

    def __init__(
        self,
        *,
        name: str,
        parameter_name: str,
        values: Sequence[AblationVariant | Any],
        definition_builder: Callable[[AblationVariant], ExperimentDefinition],
        metric_name: str,
        x_axis_label: str | None = None,
        y_axis_label: str | None = None,
        title: str | None = None,
    ) -> None:
        if not values:
            raise ValueError("XAblationPattern requires at least one value")
        self.name = name
        self.parameter_name = parameter_name
        self._variants = [self._normalize_variant(value) for value in values]
        self._definition_builder = definition_builder
        self.metric_name = metric_name
        self.x_axis_label = x_axis_label or parameter_name
        self.y_axis_label = y_axis_label or metric_name
        self.title = title or f"{name} ({parameter_name} ablation)"

    def materialize(
        self,
        project: Project,
        *,
        name_template: str | None = None,
        description_template: str | None = None,
        base_tags: Sequence[str] | None = None,
    ) -> XAbationPatternApplication:
        template = name_template or "{pattern}-{value_slug}"
        tags = tuple(base_tags or ()) + (self.pattern_type,)
        experiments: list[ProjectExperiment] = []
        variant_map: dict[str, AblationVariant] = {}
        for index, variant in enumerate(self._variants):
            experiment_name = template.format(
                pattern=self.name,
                parameter=self.parameter_name,
                value=variant.value,
                value_label=variant.display_label(),
                value_slug=variant.slug(),
                index=index,
            )
            description: str | None = None
            if description_template is not None:
                description = description_template.format(
                    pattern=self.name,
                    parameter=self.parameter_name,
                    value=variant.value,
                    value_label=variant.display_label(),
                    index=index,
                )
            metadata = {
                "pattern": self.pattern_type,
                "pattern_name": self.name,
                "parameter_name": self.parameter_name,
                "parameter_value": variant.value,
                "parameter_label": variant.display_label(),
                "pattern_index": index,
            }
            metadata.update(dict(variant.metadata))
            definition = self._definition_builder(variant)
            project_experiment = project.add_experiment(
                ProjectExperiment(
                    name=experiment_name,
                    description=description,
                    definition=definition,
                    metadata=metadata,
                    tags=tuple(dict.fromkeys(tags)),
                )
            )
            experiments.append(project_experiment)
            variant_map[project_experiment.name] = variant
        return XAbationPatternApplication(
            pattern_name=self.name,
            parameter_name=self.parameter_name,
            experiments=tuple(experiments),
            variant_by_name=variant_map,
            _pattern=self,
        )

    def _build_chart(
        self,
        reports: Mapping[str, ExperimentReport],
        variant_by_name: Mapping[str, AblationVariant],
    ) -> AblationChart:
        points: list[AblationChartPoint] = []
        for experiment in variant_by_name:
            variant = variant_by_name[experiment]
            report = reports.get(experiment)
            if report is None:
                raise KeyError(
                    f"Missing report for experiment '{experiment}' in pattern '{self.name}'"
                )
            metric = report.evaluation_report.metrics.get(self.metric_name)
            if metric is None:
                raise ValueError(
                    f"Metric '{self.metric_name}' not found for experiment '{experiment}'"
                )
            points.append(
                AblationChartPoint(
                    x_value=variant.value,
                    label=variant.display_label(),
                    metric_value=metric.mean,
                    metric_name=metric.name,
                    count=metric.count,
                )
            )
        ordered_points = self._order_points(points, variant_by_name)
        return AblationChart(
            title=self.title,
            x_label=self.x_axis_label,
            y_label=self.y_axis_label,
            metric_name=self.metric_name,
            points=tuple(ordered_points),
        )

    def _order_points(
        self,
        points: Sequence[AblationChartPoint],
        variant_by_name: Mapping[str, AblationVariant],
    ) -> list[AblationChartPoint]:
        order: dict[Any, int] = {
            variant.value: index for index, variant in enumerate(self._variants)
        }
        return sorted(points, key=lambda point: order.get(point.x_value, 0))

    def _normalize_variant(self, value: AblationVariant | Any) -> AblationVariant:
        if isinstance(value, AblationVariant):
            return AblationVariant(
                value=value.value,
                label=value.label,
                metadata=dict(value.metadata),
            )
        return AblationVariant(value=value)


__all__ = [
    "AblationChart",
    "AblationChartPoint",
    "AblationVariant",
    "XAblationPattern",
    "XAblationPatternApplication",
]
