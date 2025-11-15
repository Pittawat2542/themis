from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence
import json

from themis.core import entities as core_entities
from themis.interfaces import Metric as MetricInterface


@dataclass
class RubricJudgeMetric(MetricInterface):
    judge_model: core_entities.ModelSpec
    judge_provider: Any
    sampling: core_entities.SamplingConfig | None = None
    rubric: dict[str, str] | Sequence[str] = ()

    def __post_init__(self) -> None:
        self.name = "RubricJudge"

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
        candidate = str(prediction)
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
            name="RubricJudgeMetric",
            template=(
                "You are an impartial evaluator. Using the rubric below, score the candidate response.\n"
                "Rubric:\n{rubric}\n\n"
                "If a reference answer is provided, consider it for correctness but judge reasoning quality and formatting separately.\n"
                "Return a strict JSON object with keys: scores (dict of floats 0..1), verdict ('pass'|'fail'|'abstain'), rationale (string).\n\n"
                "Candidate:\n{candidate}\n\n"
                "Reference:\n{reference}\n"
            ),
        )
        prompt = template.render_prompt(
            {"rubric": rubric_text, "candidate": candidate, "reference": reference}
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
                value=0.0,
                details={"error": str(exc), "verdict": "abstain"},
                metadata=md,
            )

        verdict = "abstain"
        scores: dict[str, float] = {}
        rationale = ""
        try:
            payload = json.loads(raw_text)
            verdict = str(payload.get("verdict", "abstain")).lower().strip()
            rationale = str(payload.get("rationale", "")).strip()
            raw_scores = payload.get("scores") or {}
            if isinstance(raw_scores, dict):
                for k, v in raw_scores.items():
                    try:
                        fv = float(v)
                    except Exception:
                        fv = 0.0
                    scores[str(k)] = max(0.0, min(1.0, fv))
        except Exception:
            pass

        value = (
            sum(scores.values()) / max(1, len(scores))
            if scores
            else (1.0 if verdict == "pass" else 0.0)
        )

        return core_entities.MetricScore(
            metric_name=self.name,
            value=value,
            details={
                "verdict": verdict,
                "scores": scores,
                "rationale": rationale,
                "raw_judge_output": raw_text,
            },
            metadata=md,
        )
