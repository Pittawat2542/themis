"""Serialization helpers for Themis core entities."""

from __future__ import annotations

import copy
from typing import Any, Dict

from themis.core import entities as core_entities


def serialize_sampling(config: core_entities.SamplingConfig) -> Dict[str, Any]:
    return {
        "temperature": config.temperature,
        "top_p": config.top_p,
        "max_tokens": config.max_tokens,
    }


def deserialize_sampling(data: Dict[str, Any]) -> core_entities.SamplingConfig:
    return core_entities.SamplingConfig(
        temperature=data["temperature"],
        top_p=data["top_p"],
        max_tokens=data["max_tokens"],
    )


def serialize_model_spec(spec: core_entities.ModelSpec) -> Dict[str, Any]:
    return {
        "identifier": spec.identifier,
        "provider": spec.provider,
        "metadata": copy.deepcopy(spec.metadata),
        "default_sampling": serialize_sampling(spec.default_sampling)
        if spec.default_sampling
        else None,
    }


def deserialize_model_spec(data: Dict[str, Any]) -> core_entities.ModelSpec:
    default_sampling = (
        deserialize_sampling(data["default_sampling"])
        if data.get("default_sampling")
        else None
    )
    return core_entities.ModelSpec(
        identifier=data["identifier"],
        provider=data["provider"],
        metadata=copy.deepcopy(data.get("metadata", {})),
        default_sampling=default_sampling,
    )


def serialize_prompt_spec(spec: core_entities.PromptSpec) -> Dict[str, Any]:
    return {
        "name": spec.name,
        "template": spec.template,
        "metadata": copy.deepcopy(spec.metadata),
    }


def deserialize_prompt_spec(data: Dict[str, Any]) -> core_entities.PromptSpec:
    return core_entities.PromptSpec(
        name=data["name"],
        template=data["template"],
        metadata=copy.deepcopy(data.get("metadata", {})),
    )


def serialize_prompt_render(render: core_entities.PromptRender) -> Dict[str, Any]:
    return {
        "spec": serialize_prompt_spec(render.spec),
        "text": render.text,
        "context": copy.deepcopy(render.context),
        "metadata": copy.deepcopy(render.metadata),
    }


def deserialize_prompt_render(data: Dict[str, Any]) -> core_entities.PromptRender:
    return core_entities.PromptRender(
        spec=deserialize_prompt_spec(data["spec"]),
        text=data["text"],
        context=copy.deepcopy(data.get("context", {})),
        metadata=copy.deepcopy(data.get("metadata", {})),
    )


def serialize_reference(
    reference: core_entities.Reference | None,
) -> Dict[str, Any] | None:
    if reference is None:
        return None
    return {"kind": reference.kind, "value": reference.value}


def deserialize_reference(
    data: Dict[str, Any] | None,
) -> core_entities.Reference | None:
    if data is None:
        return None
    return core_entities.Reference(kind=data["kind"], value=data.get("value"))


def serialize_generation_task(task: core_entities.GenerationTask) -> Dict[str, Any]:
    return {
        "prompt": serialize_prompt_render(task.prompt),
        "model": serialize_model_spec(task.model),
        "sampling": serialize_sampling(task.sampling),
        "metadata": copy.deepcopy(task.metadata),
        "reference": serialize_reference(task.reference),
    }


def deserialize_generation_task(data: Dict[str, Any]) -> core_entities.GenerationTask:
    return core_entities.GenerationTask(
        prompt=deserialize_prompt_render(data["prompt"]),
        model=deserialize_model_spec(data["model"]),
        sampling=deserialize_sampling(data["sampling"]),
        metadata=copy.deepcopy(data.get("metadata", {})),
        reference=deserialize_reference(data.get("reference")),
    )


def serialize_generation_record(
    record: core_entities.GenerationRecord,
) -> Dict[str, Any]:
    return {
        "task": serialize_generation_task(record.task),
        "output": {
            "text": record.output.text,
            "raw": record.output.raw,
        }
        if record.output
        else None,
        "error": {
            "message": record.error.message,
            "kind": record.error.kind,
            "details": copy.deepcopy(record.error.details),
        }
        if record.error
        else None,
        "metrics": copy.deepcopy(record.metrics),
        "attempts": [
            serialize_generation_record(attempt) for attempt in record.attempts
        ],
    }


def deserialize_generation_record(
    data: Dict[str, Any],
) -> core_entities.GenerationRecord:
    output_data = data.get("output")
    error_data = data.get("error")
    return core_entities.GenerationRecord(
        task=deserialize_generation_task(data["task"]),
        output=core_entities.ModelOutput(
            text=output_data["text"], raw=output_data.get("raw")
        )
        if output_data
        else None,
        error=core_entities.ModelError(
            message=error_data["message"],
            kind=error_data.get("kind", "model_error"),
            details=copy.deepcopy(error_data.get("details", {})),
        )
        if error_data
        else None,
        metrics=copy.deepcopy(data.get("metrics", {})),
        attempts=[
            deserialize_generation_record(attempt)
            for attempt in data.get("attempts", [])
        ],
    )


def serialize_metric_score(score: core_entities.MetricScore) -> Dict[str, Any]:
    return {
        "metric_name": score.metric_name,
        "value": score.value,
        "details": copy.deepcopy(score.details),
        "metadata": copy.deepcopy(score.metadata),
    }


def deserialize_metric_score(data: Dict[str, Any]) -> core_entities.MetricScore:
    return core_entities.MetricScore(
        metric_name=data["metric_name"],
        value=data["value"],
        details=copy.deepcopy(data.get("details", {})),
        metadata=copy.deepcopy(data.get("metadata", {})),
    )


def serialize_evaluation_record(
    record: core_entities.EvaluationRecord,
) -> Dict[str, Any]:
    return {
        "sample_id": record.sample_id,
        "scores": [serialize_metric_score(score) for score in record.scores],
        "failures": list(record.failures),
    }


def deserialize_evaluation_record(
    data: Dict[str, Any],
) -> core_entities.EvaluationRecord:
    return core_entities.EvaluationRecord(
        sample_id=data.get("sample_id"),
        scores=[deserialize_metric_score(score) for score in data.get("scores", [])],
        failures=list(data.get("failures", [])),
    )


__all__ = [
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
