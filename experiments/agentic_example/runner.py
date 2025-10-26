"""Agentic generation runner that performs plan+answer steps."""

from __future__ import annotations


from themis.core import entities as core_entities
from themis.generation.runner import GenerationRunner


class AgenticRunner(GenerationRunner):
    def __init__(
        self,
        *,
        provider,
        planner_prompt: str,
        final_prompt_prefix: str,
        **kwargs,
    ) -> None:
        super().__init__(provider=provider, **kwargs)
        self._planner_prompt = planner_prompt
        self._final_prompt_prefix = final_prompt_prefix

    def _run_single_attempt(
        self, task: core_entities.GenerationTask
    ) -> core_entities.GenerationRecord:
        context = dict(task.prompt.context)
        context.setdefault("problem", task.prompt.text)

        plan_prompt = self._planner_prompt.format(**context)
        plan_task = _clone_task(task, text=plan_prompt, stage="plan")
        plan_record = super()._run_single_attempt(plan_task)
        plan_text = plan_record.output.text if plan_record.output else ""

        answer_prompt = (
            f"{task.prompt.text}\nPlan:\n{plan_text}\n{self._final_prompt_prefix}"
        )
        answer_task = _clone_task(task, text=answer_prompt, stage="answer")
        answer_record = super()._run_single_attempt(answer_task)
        answer_record.metrics["plan"] = plan_text
        answer_record.attempts = [plan_record] + answer_record.attempts
        return answer_record


def _clone_task(
    task: core_entities.GenerationTask, *, text: str, stage: str
) -> core_entities.GenerationTask:
    prompt = core_entities.PromptRender(
        spec=task.prompt.spec,
        text=text,
        context=dict(task.prompt.context),
        metadata={**task.prompt.metadata, "stage": stage},
    )
    metadata = {**task.metadata, "stage": stage}
    return core_entities.GenerationTask(
        prompt=prompt,
        model=task.model,
        sampling=task.sampling,
        metadata=metadata,
        reference=task.reference,
    )


__all__ = ["AgenticRunner"]
