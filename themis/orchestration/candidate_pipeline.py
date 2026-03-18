"""Pure candidate execution pipeline split into generation, transform, and evaluation."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping

from themis.contracts.protocols import (
    DatasetContext,
    InferenceResult,
    JudgeService,
)
from themis.orchestration.resolved_plugins import (
    ResolvedEvaluationPlugins,
    ResolvedGenerationPlugins,
    ResolvedTransformPlugins,
    resolve_evaluation_plugins,
    resolve_generation_plugins,
    resolve_transform_plugins,
)
from themis.errors import ExtractionError, InferenceError, MetricError
from themis.orchestration.task_resolution import (
    ResolvedEvaluation,
    ResolvedOutputTransform,
)
from themis.records.candidate import CandidateRecord
from themis.records.error import ErrorRecord
from themis.records.evaluation import EvaluationRecord, MetricScore
from themis.records.inference import InferenceRecord
from themis.registry.plugin_registry import PluginRegistry
from themis.specs.experiment import DataItemContext, RuntimeContext, TrialSpec
from themis.telemetry.bus import TelemetryBus, TelemetryEventName
from themis.types.enums import ErrorCode, ErrorWhere, RecordStatus


def candidate_hash_for_index(trial: TrialSpec, cand_index: int) -> str:
    """Derive a deterministic candidate hash for one sample index."""
    base_hash = trial.compute_hash(short=False)
    return hashlib.sha256(f"{base_hash}_{cand_index}".encode()).hexdigest()[:12]


def _dataset_payload(dataset_context: DatasetContext) -> dict[str, object]:
    if isinstance(dataset_context, DataItemContext):
        return dict(dataset_context.payload)
    if isinstance(dataset_context, Mapping):
        return dict(dataset_context)
    return {}


def generate_candidate(
    trial: TrialSpec,
    registry: PluginRegistry,
    dataset_context: DatasetContext,
    runtime_context: RuntimeContext,
    cand_index: int,
    *,
    prepared_trial: TrialSpec | None = None,
    resolved_generation: ResolvedGenerationPlugins | None = None,
) -> CandidateRecord:
    """Run the generation stage only for one candidate."""
    candidate_hash = candidate_hash_for_index(trial, cand_index)
    candidate = CandidateRecord(
        spec_hash=candidate_hash,
        candidate_id=candidate_hash,
        sample_index=cand_index,
    )
    resolved = resolved_generation or resolve_generation_plugins(trial, registry)
    execution_trial = (
        prepared_trial
        if prepared_trial is not None
        else resolved.hooks.apply_pre_inference(trial)
    )
    engine = resolved.engine
    try:
        inf_result = _coerce_inference_result(
            engine.infer(execution_trial, dataset_context, runtime_context)
        )
        inf_result = resolved.hooks.apply_post_inference(execution_trial, inf_result)
        return candidate.model_copy(
            update={
                "status": RecordStatus.OK,
                "inference": inf_result.inference,
                "conversation": inf_result.conversation
                or inf_result.inference.conversation,
            }
        )
    except InferenceError:
        raise
    except Exception as exc:
        raise InferenceError(
            code=ErrorCode.PROVIDER_UNAVAILABLE,
            message=f"{type(exc).__name__}: {exc}",
            details={},
        ) from exc


def transform_candidate(
    trial: TrialSpec,
    registry: PluginRegistry,
    candidate: CandidateRecord,
    transform: ResolvedOutputTransform,
    telemetry_bus: TelemetryBus | None = None,
    *,
    resolved_transform: ResolvedTransformPlugins | None = None,
) -> CandidateRecord:
    """Apply one resolved output transform to an existing generated candidate."""
    resolved = resolved_transform or resolve_transform_plugins(transform, registry)
    working_candidate = candidate

    for attempt_index, extractor_step in enumerate(resolved.extractors):
        working_candidate = resolved.hooks.apply_pre_extraction(
            trial, working_candidate
        )
        try:
            extraction_record = extractor_step.extractor.extract(
                trial,
                working_candidate,
                extractor_step.config,
            )
        except ExtractionError:
            raise
        except Exception as exc:
            raise ExtractionError(
                code=ErrorCode.PARSE_ERROR,
                message=f"{type(exc).__name__}: {exc}",
                details={},
            ) from exc

        working_candidate = working_candidate.model_copy(
            update={"extractions": [*working_candidate.extractions, extraction_record]}
        )
        working_candidate = resolved.hooks.apply_post_extraction(
            trial, working_candidate
        )
        if telemetry_bus is not None:
            telemetry_bus.emit(
                TelemetryEventName.EXTRACTOR_ATTEMPT,
                trial_hash=trial.spec_hash,
                candidate_id=working_candidate.candidate_id
                or working_candidate.spec_hash,
                extractor_id=extractor_step.extractor_id,
                attempt_index=attempt_index,
                success=extraction_record.success,
            )
        if extraction_record.success:
            return working_candidate.model_copy(
                update={"status": RecordStatus.OK, "error": None}
            )

    failure_reason = None
    if working_candidate.extractions:
        failure_reason = working_candidate.extractions[-1].failure_reason
    message = failure_reason or (
        f"All extractors failed for output transform '{transform.spec.name}'."
    )
    return working_candidate.model_copy(
        update={
            "status": RecordStatus.ERROR,
            "error": ErrorRecord(
                code=ErrorCode.PARSE_ERROR,
                where=ErrorWhere.EXTRACTOR,
                message=message,
                retryable=False,
                details={"transform": transform.spec.name},
            ),
        }
    )


def evaluate_candidate(
    trial: TrialSpec,
    registry: PluginRegistry,
    dataset_context: DatasetContext,
    runtime_context: RuntimeContext,
    candidate: CandidateRecord,
    evaluation: ResolvedEvaluation,
    judge_service: JudgeService | None = None,
    telemetry_bus: TelemetryBus | None = None,
    *,
    resolved_evaluation: ResolvedEvaluationPlugins | None = None,
) -> CandidateRecord:
    """Run one evaluation stage against an existing generated or transformed candidate."""
    resolved = resolved_evaluation or resolve_evaluation_plugins(evaluation, registry)
    candidate_id = candidate.candidate_id or candidate.spec_hash
    scores: list[MetricScore] = []
    aggregate_scores: dict[str, float] = {}
    working_candidate = resolved.hooks.apply_pre_eval(trial, candidate)

    for metric_step in resolved.metrics:
        try:
            ctx = _dataset_payload(dataset_context)
            if judge_service is not None:
                ctx["judge_service"] = judge_service
                ctx["runtime_context"] = runtime_context
            if telemetry_bus is not None:
                telemetry_bus.emit(
                    TelemetryEventName.METRIC_START,
                    trial_hash=trial.spec_hash,
                    candidate_id=candidate_id,
                    metric_id=metric_step.metric_id,
                )

            score_record = metric_step.metric.score(trial, working_candidate, ctx)
            scores.append(score_record)
            aggregate_scores[score_record.metric_id] = score_record.value
            if telemetry_bus is not None:
                telemetry_bus.emit(
                    TelemetryEventName.METRIC_END,
                    trial_hash=trial.spec_hash,
                    candidate_id=candidate_id,
                    metric_id=score_record.metric_id,
                    score=score_record.value,
                )
        except MetricError:
            raise
        except Exception as exc:
            raise MetricError(
                code=ErrorCode.METRIC_COMPUTATION,
                message=f"{type(exc).__name__}: {exc}",
                details={},
            ) from exc

    evaluation_record = EvaluationRecord(
        spec_hash=evaluation.evaluation_hash,
        metric_scores=scores,
        aggregate_scores=aggregate_scores,
    )
    working_candidate = working_candidate.model_copy(
        update={"evaluation": evaluation_record}
    )
    working_candidate = resolved.hooks.apply_post_eval(trial, working_candidate)
    return working_candidate.model_copy(update={"status": RecordStatus.OK})


def _coerce_inference_result(
    value: InferenceResult | InferenceRecord,
) -> InferenceResult:
    if isinstance(value, InferenceResult):
        return value
    if isinstance(value, InferenceRecord):
        return InferenceResult(inference=value, conversation=value.conversation)
    raise TypeError("InferenceEngine.infer() must return InferenceResult.")
