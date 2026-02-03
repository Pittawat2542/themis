from __future__ import annotations

from typing import Iterable

from themis.core import entities as core_entities


def make_prompt(text: str = "Q") -> core_entities.PromptRender:
    spec = core_entities.PromptSpec(name="t", template=text)
    return core_entities.PromptRender(spec=spec, text=text)


def make_task(
    *,
    sample_id: str = "s1",
    provider: str = "fake",
    model_id: str = "model-x",
    prompt_text: str = "Q",
) -> core_entities.GenerationTask:
    prompt = make_prompt(prompt_text)
    model = core_entities.ModelSpec(identifier=model_id, provider=provider)
    sampling = core_entities.SamplingConfig(temperature=0.0, top_p=1.0, max_tokens=8)
    return core_entities.GenerationTask(
        prompt=prompt,
        model=model,
        sampling=sampling,
        metadata={"dataset_id": sample_id},
    )


def make_record(
    *,
    sample_id: str = "s1",
    provider: str = "fake",
    model_id: str = "model-x",
    prompt_text: str = "Q",
    output_text: str = "ok",
) -> core_entities.GenerationRecord:
    task = make_task(
        sample_id=sample_id,
        provider=provider,
        model_id=model_id,
        prompt_text=prompt_text,
    )
    return core_entities.GenerationRecord(
        task=task,
        output=core_entities.ModelOutput(text=output_text),
        error=None,
    )


def make_evaluation_record(
    *,
    sample_id: str = "s1",
    metric_name: str = "ExactMatch",
    value: float = 1.0,
) -> core_entities.EvaluationRecord:
    return core_entities.EvaluationRecord(
        sample_id=sample_id,
        scores=[core_entities.MetricScore(metric_name=metric_name, value=value)],
    )


def make_records(values: Iterable[float]) -> list[core_entities.GenerationRecord]:
    return [
        make_record(sample_id=f"s{i}", output_text="ok")
        for i, _ in enumerate(values, start=1)
    ]


__all__ = [
    "make_prompt",
    "make_task",
    "make_record",
    "make_evaluation_record",
    "make_records",
]
