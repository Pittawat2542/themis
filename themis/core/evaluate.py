"""Layer 1 convenience API for Themis experiments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

from themis.core.base import JSONValue
from themis.core.config import (
    EvaluationConfig,
    GenerationConfig,
    GeneratorComponent,
    JudgeModelComponent,
    MetricComponent,
    ParserComponent,
    ReducerComponent,
    RuntimeConfig,
    StorageConfig,
)
from themis.core.experiment import Experiment
from themis.core.models import Case, Dataset
from themis.core.protocols import LifecycleSubscriber, TracingProvider
from themis.core.store import RunStore


def evaluate(
    *,
    model: object,
    data: Dataset | Sequence[Dataset] | Sequence[Mapping[str, Any]],
    metric: object | Sequence[object],
    parser: object | Sequence[object] | None = None,
    judge: object | Sequence[object] | None = None,
    samples: int = 1,
    reducer: object | None = None,
    storage: StorageConfig | None = None,
    runtime: RuntimeConfig | None = None,
    seeds: list[int] | None = None,
    workflow_overrides: dict[str, object] | None = None,
    judge_config: dict[str, object] | None = None,
    environment_metadata: dict[str, str] | None = None,
    themis_version: str = "4.0.0",
    python_version: str = "3.12",
    platform: str = "unknown",
    store: RunStore | None = None,
    subscribers: list[LifecycleSubscriber] | None = None,
    tracing_provider: TracingProvider | None = None,
):
    """Compile and run a Themis experiment through the Layer 1 API."""

    datasets = _normalize_datasets(data)
    metrics = _normalize_component_list(metric)
    parsers = _normalize_component_list(parser)
    judge_models = _normalize_component_list(judge)
    candidate_policy: dict[str, JSONValue] = {"num_samples": max(1, samples)}
    if reducer is None:
        reducer = "builtin/majority_vote"

    experiment = Experiment(
        generation=GenerationConfig(
            generator=cast(GeneratorComponent, model),
            candidate_policy=candidate_policy,
            reducer=cast(ReducerComponent | None, reducer),
        ),
        evaluation=EvaluationConfig(
            metrics=cast(list[MetricComponent], metrics),
            parsers=cast(list[ParserComponent], parsers),
            judge_models=cast(list[JudgeModelComponent], judge_models),
            judge_config=cast(dict[str, JSONValue], dict(judge_config or {})),
            workflow_overrides=cast(
                dict[str, JSONValue], dict(workflow_overrides or {})
            ),
        ),
        storage=storage or StorageConfig(store="memory"),
        runtime=runtime or RuntimeConfig(),
        datasets=datasets,
        seeds=list(seeds or []),
        environment_metadata=dict(environment_metadata or {}),
        themis_version=themis_version,
        python_version=python_version,
        platform=platform,
    )
    return experiment.run(
        store=store,
        subscribers=subscribers,
        tracing_provider=tracing_provider,
    )


def _normalize_datasets(
    data: Dataset | Sequence[Dataset] | Sequence[Mapping[str, Any]],
) -> list[Dataset]:
    if isinstance(data, Dataset):
        return [data]
    items = list(data)
    if not items:
        return [Dataset(dataset_id="inline", revision="inline", cases=[])]
    if all(isinstance(item, Dataset) for item in items):
        return [cast(Dataset, item) for item in items]
    return [
        Dataset(
            dataset_id="inline",
            revision="inline",
            cases=[
                _case_from_mapping(index, cast(Mapping[str, Any], item))
                for index, item in enumerate(items, start=1)
            ],
        )
    ]


def _case_from_mapping(index: int, payload: Mapping[str, Any]) -> Case:
    metadata = payload.get("metadata", {})
    return Case(
        case_id=str(payload.get("case_id", f"case-{index}")),
        input=payload["input"],
        expected_output=payload.get("expected_output"),
        metadata={key: str(value) for key, value in metadata.items()}
        if isinstance(metadata, Mapping)
        else {},
    )


def _normalize_component_list(value: object | Sequence[object] | None) -> list[object]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        return [value]
    if isinstance(value, Sequence):
        return list(value)
    return [value]
