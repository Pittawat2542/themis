"""Serialization helpers for Themis core entities using a generic introspective approach."""

from __future__ import annotations

import dataclasses
import types
import typing
from typing import Any, Dict, TypeVar, get_args, get_origin

from themis.core import entities as core_entities

T = TypeVar("T")

# =========================================================================
# GENERIC SERIALIZATION
# =========================================================================


def serialize(entity: Any) -> Any:
    """Recursively serialize a dataclass or standard collection."""
    if dataclasses.is_dataclass(entity):
        return dataclasses.asdict(entity)
    elif isinstance(entity, (list, tuple, set)):
        return [serialize(x) for x in entity]
    elif isinstance(entity, dict):
        return {k: serialize(v) for k, v in entity.items()}
    return entity


def _resolve_type_hint(hint: Any) -> Any:
    """Resolve forward references and extract underlying types from generics."""
    if isinstance(hint, str):
        hint = hint.strip("'\"")
        return getattr(core_entities, hint, hint)

    origin = get_origin(hint)

    # Handle Optional[T] / Union[T, None]
    if origin is getattr(types, "UnionType", type(None)) or origin is typing.Union:
        args = get_args(hint)
        non_none_args = [arg for arg in args if arg is not type(None)]
        if non_none_args:
            return _resolve_type_hint(non_none_args[0])

    # Handle list[T]
    if origin is list or origin is typing.List:
        args = get_args(hint)
        if args:
            return list, _resolve_type_hint(args[0])

    # Handle dict[K, V]
    if origin is dict or origin is typing.Dict:
        return dict, None

    return hint


def deserialize(entity_cls: type[T], data: Any) -> T:
    """Recursively deserialize a dictionary into a dataclass."""
    if data is None:
        return None  # type: ignore

    if isinstance(entity_cls, str):
        entity_cls = getattr(core_entities, entity_cls)

    origin = get_origin(entity_cls)
    if origin is not None:
        if origin is list or origin is typing.List:
            args = get_args(entity_cls)
            if args:
                item_cls = _resolve_type_hint(args[0])
                if isinstance(item_cls, tuple):
                    item_cls = item_cls[1]
                return [deserialize(item_cls, item) for item in data]  # type: ignore
            return data  # type: ignore
        elif origin is dict:
            return data  # type: ignore
        elif hasattr(origin, "__dataclass_fields__"):
            entity_cls = origin  # Fallback for generics like Reference[T]

    if not dataclasses.is_dataclass(entity_cls):
        return data  # type: ignore

    kwargs = {}
    type_hints = typing.get_type_hints(entity_cls)

    for field in dataclasses.fields(entity_cls):
        if field.name not in data:
            continue

        value = data[field.name]
        if value is None:
            kwargs[field.name] = None
            continue

        hint = type_hints.get(field.name, field.type)
        resolved_hint = _resolve_type_hint(hint)

        if isinstance(resolved_hint, tuple) and resolved_hint[0] is list:
            kwargs[field.name] = [deserialize(resolved_hint[1], item) for item in value]
        elif resolved_hint is dict:
            kwargs[field.name] = value
        elif dataclasses.is_dataclass(resolved_hint):
            kwargs[field.name] = deserialize(resolved_hint, value)
        else:
            kwargs[field.name] = value

    return entity_cls(**kwargs)  # type: ignore


# =========================================================================
# BACKWARDS COMPATIBILITY WRAPPERS
# =========================================================================


def serialize_sampling(config: core_entities.SamplingConfig) -> Dict[str, Any]:
    return serialize(config)


def deserialize_sampling(data: Dict[str, Any]) -> core_entities.SamplingConfig:
    return deserialize(core_entities.SamplingConfig, data)


def serialize_model_spec(spec: core_entities.ModelSpec) -> Dict[str, Any]:
    return serialize(spec)


def deserialize_model_spec(data: Dict[str, Any]) -> core_entities.ModelSpec:
    return deserialize(core_entities.ModelSpec, data)


def serialize_prompt_spec(spec: core_entities.PromptSpec) -> Dict[str, Any]:
    return serialize(spec)


def deserialize_prompt_spec(data: Dict[str, Any]) -> core_entities.PromptSpec:
    return deserialize(core_entities.PromptSpec, data)


def serialize_prompt_render(render: core_entities.PromptRender) -> Dict[str, Any]:
    return serialize(render)


def deserialize_prompt_render(data: Dict[str, Any]) -> core_entities.PromptRender:
    return deserialize(core_entities.PromptRender, data)


def serialize_reference(
    reference: core_entities.Reference | None,
) -> Dict[str, Any] | None:
    return serialize(reference) if reference else None


def deserialize_reference(
    data: Dict[str, Any] | None,
) -> core_entities.Reference | None:
    return deserialize(core_entities.Reference, data) if data else None


def serialize_generation_task(task: core_entities.GenerationTask) -> Dict[str, Any]:
    return serialize(task)


def deserialize_generation_task(data: Dict[str, Any]) -> core_entities.GenerationTask:
    return deserialize(core_entities.GenerationTask, data)


def serialize_generation_record(
    record: core_entities.GenerationRecord,
) -> Dict[str, Any]:
    return serialize(record)


def deserialize_generation_record(
    data: Dict[str, Any],
) -> core_entities.GenerationRecord:
    return deserialize(core_entities.GenerationRecord, data)


def serialize_metric_score(score: core_entities.MetricScore) -> Dict[str, Any]:
    return serialize(score)


def deserialize_metric_score(data: Dict[str, Any]) -> core_entities.MetricScore:
    return deserialize(core_entities.MetricScore, data)


def serialize_evaluation_record(
    record: core_entities.EvaluationRecord,
) -> Dict[str, Any]:
    return serialize(record)


def deserialize_evaluation_record(
    data: Dict[str, Any],
) -> core_entities.EvaluationRecord:
    return deserialize(core_entities.EvaluationRecord, data)


__all__ = [
    "serialize",
    "deserialize",
    "serialize_generation_record",
    "deserialize_generation_record",
    "serialize_generation_task",
    "deserialize_generation_task",
    "serialize_evaluation_record",
    "deserialize_evaluation_record",
    "serialize_metric_score",
    "deserialize_metric_score",
    "serialize_sampling",
    "deserialize_sampling",
    "serialize_model_spec",
    "deserialize_model_spec",
    "serialize_prompt_spec",
    "deserialize_prompt_spec",
    "serialize_prompt_render",
    "deserialize_prompt_render",
    "serialize_reference",
    "deserialize_reference",
]
