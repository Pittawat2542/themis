from collections.abc import Callable, Mapping
from typing import cast

from themis.contracts.protocols import (
    DatasetContext,
    InferenceEngine,
    InferenceResult,
    JudgeService,
)
from themis.orchestration.seeding import derive_judge_seed, judge_call_id
from themis.records.candidate import CandidateRecord
from themis.records.inference import InferenceRecord
from themis.records.judge import JudgeCallRecord, JudgeAuditTrail
from themis.specs.experiment import PromptTemplateSpec, RuntimeContext, TrialSpec
from themis.specs.foundational import (
    DatasetSpec,
    GenerationSpec,
    JudgeInferenceSpec,
    TaskSpec,
)
from themis.types.enums import DatasetSource


def _coerce_runtime_context(value: object) -> RuntimeContext:
    if value is None:
        return RuntimeContext()
    if isinstance(value, RuntimeContext):
        return value
    if isinstance(value, Mapping):
        return RuntimeContext.model_validate(dict(value))

    # Strictly validate against the model type and let Pydantic handle it
    # Pydantic validation handles coercing dicts if needed, or raises ValidationError.
    return RuntimeContext.model_validate(value)


def _coerce_inference_result(
    value: InferenceResult | InferenceRecord,
) -> InferenceRecord:
    if isinstance(value, InferenceResult):
        return value.inference
    return value


class DefaultJudgeService(JudgeService):
    """
    Standard implementation of JudgeService.
    Executes inference for judge-dependent metrics and records the calls for auditing.
    """

    def __init__(
        self,
        *,
        engine_resolver: Callable[[str], InferenceEngine],
    ) -> None:
        self.engine_resolver = engine_resolver
        self.calls: dict[str, list[JudgeCallRecord]] = {}

    def judge(
        self,
        metric_id: str,
        parent_candidate: CandidateRecord,
        judge_spec: JudgeInferenceSpec,
        prompt: PromptTemplateSpec,
        runtime: Mapping[str, object],
    ) -> InferenceRecord:
        """
        Executes a judge model endpoint, capturing the audit trail.
        """
        engine = self._resolve_engine(judge_spec.model.provider)

        # We need a temporary trial spec for the judge call to pass to the engine
        # We synthesize a TrialSpec based on the judge config
        task_spec = runtime.get("task_spec")
        if not isinstance(task_spec, TaskSpec):
            task_spec = TaskSpec(
                task_id="judge_fallback_task",
                dataset=DatasetSpec(source=DatasetSource.MEMORY),
                generation=GenerationSpec(),
            )

        candidate_hash = parent_candidate.candidate_id or parent_candidate.spec_hash
        judge_call_index = len(self.calls.get(parent_candidate.spec_hash, []))
        engine_runtime = _coerce_runtime_context(runtime.get("runtime_context"))
        effective_params = judge_spec.params
        if judge_spec.params.seed is None:
            effective_params = judge_spec.params.model_copy(
                update={
                    "seed": derive_judge_seed(
                        engine_runtime.candidate_seed,
                        metric_id,
                        judge_call_index,
                    )
                }
            )
        effective_judge_spec = judge_spec.model_copy(
            update={"params": effective_params}
        )

        judge_trial = TrialSpec(
            trial_id=judge_call_id(candidate_hash, metric_id, judge_call_index),
            model=effective_judge_spec.model,
            task=task_spec,
            item_id=parent_candidate.spec_hash,
            prompt=prompt,
            params=effective_judge_spec.params,
            candidate_count=1,
        )

        # Run inference
        dataset_context = runtime.get("dataset_context", {})
        if not isinstance(dataset_context, Mapping):
            dataset_context = {}
        inf_record = _coerce_inference_result(
            engine.infer(
                judge_trial,
                cast(DatasetContext, dataset_context),
                engine_runtime,
            )
        )

        # Record the call to our internal audit trail before returning to the metric
        call_record = JudgeCallRecord(
            spec_hash=inf_record.spec_hash,
            metric_id=metric_id,
            judge_spec=effective_judge_spec,
            rendered_prompt=list(prompt.messages),
            inference=inf_record,
        )
        if parent_candidate.spec_hash not in self.calls:
            self.calls[parent_candidate.spec_hash] = []
        self.calls[parent_candidate.spec_hash].append(call_record)

        return inf_record

    def consume_audit_trail(self, candidate_hash: str) -> JudgeAuditTrail | None:
        """
        Retrieves the bundled JudgeAuditTrail for all calls made during a candidate's evaluation,
        clearing the internal list for the next candidate.
        """
        calls = self.calls.pop(candidate_hash, [])
        if not calls:
            return None

        trail = JudgeAuditTrail(
            spec_hash=f"audit_{candidate_hash}",
            candidate_hash=candidate_hash,
            judge_calls=calls,
        )
        return trail

    def _resolve_engine(self, provider: str) -> InferenceEngine:
        return self.engine_resolver(provider)
