from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Sequence

from themis.core import entities as core_entities
from themis.interfaces import Metric as MetricInterface


@dataclass
class PairwiseJudgeMetric(MetricInterface):
    judge_model: core_entities.ModelSpec
    judge_provider: Any
    sampling: core_entities.SamplingConfig | None = None
    rubric: dict[str, str] | Sequence[str] = ()

    def __post_init__(self) -> None:
        self.name = "PairwiseJudge"

    def compute(
        self,
        *,
        prediction: Any,
        references: Sequence[Any],
        metadata: dict[str, Any] | None = None,
    ) -> core_entities.MetricScore:
        from themis.generation.runner import GenerationRunner
        from themis.generation.templates import PromptTemplate

        md = dict(metadata or {})
        try:
            a_text, b_text = (
                prediction
                if isinstance(prediction, (list, tuple))
                else (str(prediction), "")
            )
        except Exception:
            a_text, b_text = str(prediction), ""
        reference = str(references[0]) if references else ""

        rubric_lines = (
            [f"- {k}: {v}" for k, v in self.rubric.items()]
            if isinstance(self.rubric, dict)
            else [f"- {str(item)}" for item in self.rubric]
        )
        rubric_text = (
            "\n".join(rubric_lines)
            or "- correctness\n- reasoning quality\n- formatting"
        )

        template = PromptTemplate(
            name="PairwiseJudgeMetric",
            template=(
                "You are an impartial evaluator. Compare two candidate responses (A and B) using the rubric below.\n"
                "Rubric:\n{rubric}\n\n"
                "If a reference answer is provided, consider it for correctness but judge reasoning quality and formatting separately.\n"
                'Return strict JSON: {{"preference": "A"|"B"|"tie", "confidence": float, "rationale": str}}.\n\n'
                "A:\n{a}\n\nB:\n{b}\n\nReference:\n{reference}\n"
            ),
        )
        prompt = template.render_prompt(
            {
                "rubric": rubric_text,
                "a": str(a_text),
                "b": str(b_text),
                "reference": reference,
            }
        )

        sampling = self.sampling or core_entities.SamplingConfig(
            temperature=0.0, top_p=1.0, max_tokens=512
        )
        task = core_entities.GenerationTask(
            prompt=prompt,
            model=self.judge_model,
            sampling=sampling,
            metadata={"metric": self.name, **md},
            reference=None,
        )

        try:
            runner = GenerationRunner(provider=self.judge_provider)
            record = next(iter(runner.run([task])))
            raw_text = record.output.text if record.output else ""
        except Exception as exc:  # pragma: no cover - provider failure
            return core_entities.MetricScore(
                metric_name=self.name,
                value=0.5,
                details={"error": str(exc), "preference": "tie"},
                metadata=md,
            )

        preference = "tie"
        confidence = 0.0
        rationale = ""
        try:
            payload = json.loads(raw_text)
            preference = str(payload.get("preference", "tie")).lower().strip()
            confidence = float(payload.get("confidence", 0.0))
            rationale = str(payload.get("rationale", "")).strip()
        except Exception:
            pass

        value = 0.5
        if preference == "a":
            value = 1.0
        elif preference == "b":
            value = 0.0

        return core_entities.MetricScore(
            metric_name=self.name,
            value=value,
            details={
                "preference": preference,
                "confidence": confidence,
                "rationale": rationale,
                "raw_judge_output": raw_text,
            },
            metadata=md,
        )
