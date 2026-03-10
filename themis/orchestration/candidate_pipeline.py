"""Pure candidate execution pipeline for inference, extraction, and scoring."""

import hashlib
import inspect
from collections.abc import Mapping

from themis.contracts.protocols import (
    DatasetContext,
    Extractor,
    InferenceResult,
    JudgeService,
    RenderedPrompt,
)
from themis.errors.exceptions import InferenceError, ExtractionError, MetricError
from themis.records.candidate import CandidateRecord
from themis.records.evaluation import EvaluationRecord, MetricScore
from themis.records.extraction import ExtractionRecord
from themis.records.inference import InferenceRecord
from themis.registry.plugin_registry import PluginRegistry
from themis.specs.experiment import DataItemContext, RuntimeContext, TrialSpec
from themis.telemetry.bus import TelemetryBus
from themis.types.enums import RecordStatus, ErrorCode
from themis.types.json_types import JSONValueType


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


def execute_candidate_pipeline(
    trial: TrialSpec,
    registry: PluginRegistry,
    dataset_context: DatasetContext,
    runtime_context: RuntimeContext,
    cand_index: int,
    judge_service: JudgeService | None = None,
    telemetry_bus: TelemetryBus | None = None,
) -> CandidateRecord:
    """
    Pure pipeline function for one candidate execution.

    The pipeline runs inference, optional extraction, and metric scoring in that
    order, returning a `CandidateRecord` with the intermediate artifacts attached.
    """
    candidate_hash = candidate_hash_for_index(trial, cand_index)
    cand = CandidateRecord(
        spec_hash=candidate_hash,
        candidate_id=candidate_hash,
        sample_index=cand_index,
    )
    execution_trial = _apply_pre_inference_hooks(trial, registry)

    # 1. Inference
    engine = registry.get_inference_engine(trial.model.provider)
    try:
        inf_result = _coerce_inference_result(
            engine.infer(execution_trial, dataset_context, runtime_context)
        )
        inf_result = _apply_post_inference_hooks(execution_trial, registry, inf_result)
        cand = cand.model_copy(
            update={
                "inference": inf_result.inference,
                "conversation": inf_result.conversation
                or inf_result.inference.conversation,
            }
        )
    except InferenceError:
        raise
    except Exception as e:
        raise InferenceError(
            code=ErrorCode.PROVIDER_UNAVAILABLE,
            message=f"{type(e).__name__}: {str(e)}",
            details={},
        ) from e

    # 2. Extraction (Fallback Chain)
    extractions: list[ExtractionRecord] = []
    extractor_refs = []
    if trial.task.default_extractor_chain:
        extractor_refs = trial.task.default_extractor_chain.extractors

    if extractor_refs:
        for attempt_index, extractor_ref in enumerate(extractor_refs):
            cand = _apply_candidate_hooks(
                execution_trial, registry, "pre_extraction", cand
            )
            extractor = registry.get_extractor(extractor_ref.id)
            try:
                extraction_record = _invoke_extractor(
                    extractor, execution_trial, cand, extractor_ref.config
                )
                extractions.append(extraction_record)
                cand = cand.model_copy(update={"extractions": extractions})
                cand = _apply_candidate_hooks(
                    execution_trial, registry, "post_extraction", cand
                )
                if telemetry_bus is not None:
                    telemetry_bus.emit(
                        "extractor_attempt",
                        trial_hash=trial.spec_hash,
                        candidate_id=candidate_hash,
                        extractor_id=extractor_ref.id,
                        attempt_index=attempt_index,
                        success=extraction_record.success,
                    )
                if extraction_record.success:
                    break
            except ExtractionError:
                raise
            except Exception as e:
                raise ExtractionError(
                    code=ErrorCode.PARSE_ERROR,
                    message=f"{type(e).__name__}: {str(e)}",
                    details={},
                ) from e

    # 3. Evaluation
    scores: list[MetricScore] = []
    agg_scores: dict[str, float] = {}
    cand = _apply_candidate_hooks(execution_trial, registry, "pre_eval", cand)

    for metric_id in trial.task.default_metrics:
        metric = registry.get_metric(metric_id)
        try:
            # We inject the judge_service into the context so metrics can access it
            ctx = _dataset_payload(dataset_context)
            if judge_service is not None:
                ctx["judge_service"] = judge_service
                ctx["runtime_context"] = runtime_context
            if telemetry_bus is not None:
                telemetry_bus.emit(
                    "metric_start",
                    trial_hash=trial.spec_hash,
                    candidate_id=candidate_hash,
                    metric_id=metric_id,
                )

            score_record = metric.score(execution_trial, cand, ctx)
            scores.append(score_record)
            agg_scores[score_record.metric_id] = score_record.value
            if telemetry_bus is not None:
                telemetry_bus.emit(
                    "metric_end",
                    trial_hash=trial.spec_hash,
                    candidate_id=candidate_hash,
                    metric_id=score_record.metric_id,
                    score=score_record.value,
                )
        except MetricError:
            raise
        except Exception as e:
            raise MetricError(
                code=ErrorCode.METRIC_COMPUTATION,
                message=f"{type(e).__name__}: {str(e)}",
                details={},
            ) from e

    eval_record = EvaluationRecord(
        spec_hash=candidate_hash,  # Evaluation binds to candidate context
        metric_scores=scores,
        aggregate_scores=agg_scores,
    )
    cand = cand.model_copy(update={"evaluation": eval_record})
    cand = _apply_candidate_hooks(execution_trial, registry, "post_eval", cand)

    return cand.model_copy(update={"status": RecordStatus.OK})


def _coerce_inference_result(
    value: InferenceResult | InferenceRecord,
) -> InferenceResult:
    if isinstance(value, InferenceResult):
        return value
    if isinstance(value, InferenceRecord):
        return InferenceResult(inference=value, conversation=value.conversation)
    raise TypeError("InferenceEngine.infer() must return InferenceResult.")


def _apply_pre_inference_hooks(trial: TrialSpec, registry: PluginRegistry) -> TrialSpec:
    prompt = RenderedPrompt(messages=list(trial.prompt.messages))
    for hook in registry.iter_hooks():
        pre_inference = getattr(hook, "pre_inference", None)
        if callable(pre_inference):
            prompt = pre_inference(trial, prompt)
    return trial.model_copy(
        update={"prompt": trial.prompt.model_copy(update={"messages": prompt.messages})}
    )


def _apply_post_inference_hooks(
    trial: TrialSpec,
    registry: PluginRegistry,
    result: InferenceResult,
) -> InferenceResult:
    updated = result
    for hook in registry.iter_hooks():
        post_inference = getattr(hook, "post_inference", None)
        if callable(post_inference):
            updated = post_inference(trial, updated)
    return updated


def _apply_candidate_hooks(
    trial: TrialSpec,
    registry: PluginRegistry,
    hook_name: str,
    candidate: CandidateRecord,
) -> CandidateRecord:
    updated = candidate
    for hook in registry.iter_hooks():
        method = getattr(hook, hook_name, None)
        if callable(method):
            updated = method(trial, updated)
    return updated


def _invoke_extractor(
    extractor: Extractor,
    trial: TrialSpec,
    candidate: CandidateRecord,
    config: Mapping[str, JSONValueType],
) -> ExtractionRecord:
    signature = inspect.signature(extractor.extract)
    if "config" in signature.parameters or len(signature.parameters) >= 3:
        return extractor.extract(trial, candidate, config)
    return extractor.extract(trial, candidate)
