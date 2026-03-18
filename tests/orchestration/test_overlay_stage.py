from __future__ import annotations

from types import SimpleNamespace

from themis.orchestration.overlay_stage import OverlayStageExecutor
from themis.orchestration.runner_state import TrialExecutionSession
from themis.orchestration.task_resolution import (
    ResolvedEvaluation,
    ResolvedOutputTransform,
    ResolvedTaskStages,
)
from themis.records.candidate import CandidateRecord
from themis.records.error import ErrorRecord
from themis.records.provenance import ProvenanceRecord
from themis.specs.experiment import (
    DataItemContext,
    InferenceParamsSpec,
    PromptTemplateSpec,
    RuntimeContext,
    TrialSpec,
)
from themis.specs.foundational import (
    DatasetSpec,
    EvaluationSpec,
    ExtractorChainSpec,
    ExtractorRefSpec,
    GenerationSpec,
    ModelSpec,
    OutputTransformSpec,
    TaskSpec,
)
from themis.types.enums import DatasetSource, ErrorCode, ErrorWhere, RecordStatus
from themis.types.events import ArtifactRef


class _NullEventEmitter:
    def emit_output_transform_events(self, *args, **kwargs) -> None:
        del args, kwargs

    def emit_evaluation_candidate_events(self, *args, **kwargs) -> None:
        del args, kwargs

    def emit_candidate_failure_event(self, *args, **kwargs) -> None:
        del args, kwargs


class _ResolvedPlugins:
    def output_transform_for(self, transform_hash: str) -> object:
        return {"transform_hash": transform_hash}

    def evaluation_for(self, evaluation_hash: str) -> object:
        return {"evaluation_hash": evaluation_hash}

    def create_judge_service(self) -> object:
        return SimpleNamespace(consume_audit_trail=lambda candidate_id: None)


def _trial_spec() -> TrialSpec:
    return TrialSpec(
        trial_id="trial-overlay",
        model=ModelSpec(model_id="demo-model", provider="demo"),
        task=TaskSpec(
            task_id="qa",
            dataset=DatasetSpec(source=DatasetSource.MEMORY),
            generation=GenerationSpec(),
            output_transforms=[
                OutputTransformSpec(
                    name="parsed",
                    extractor_chain=ExtractorChainSpec(
                        extractors=[ExtractorRefSpec(id="mock-extractor")]
                    ),
                )
            ],
            evaluations=[
                EvaluationSpec(
                    name="score",
                    transform="parsed",
                    metrics=["exact_match"],
                )
            ],
        ),
        item_id="item-1",
        prompt=PromptTemplateSpec(id="baseline", messages=[]),
        params=InferenceParamsSpec(max_tokens=8),
    )


def _session() -> TrialExecutionSession:
    return TrialExecutionSession(
        trial=_trial_spec(),
        prepared_trial=_trial_spec(),
        dataset_context=DataItemContext(item_id="item-1", payload={"answer": "42"}),
        base_runtime=RuntimeContext(),
        provenance=ProvenanceRecord(
            themis_version="test",
            git_commit=None,
            python_version="3.12",
            platform="test",
            library_versions={},
            model_endpoint_meta={},
        ),
        resolved_stages=ResolvedTaskStages(output_transforms=(), evaluations=()),
        prompt_payload={"messages": []},
        prompt_artifact=(
            ArtifactRef(
                artifact_hash="sha256:test-prompt",
                media_type="application/json",
                label="rendered_prompt",
            ),
            "sha256:test-prompt",
        ),
        item_payload={"answer": "42"},
        dataset_metadata={},
        event_seq=0,
        resolved_plugins=_ResolvedPlugins(),  # type: ignore[arg-type]
    )


def _candidate() -> CandidateRecord:
    return CandidateRecord(
        spec_hash="candidate-1",
        candidate_id="candidate-1",
        sample_index=0,
        status=RecordStatus.OK,
    )


def _retryable_error(message: str) -> ErrorRecord:
    return ErrorRecord(
        code=ErrorCode.PROVIDER_TIMEOUT,
        message=message,
        retryable=True,
        where=ErrorWhere.EXECUTOR,
    )


def _executor() -> OverlayStageExecutor:
    return OverlayStageExecutor(
        registry=SimpleNamespace(),  # type: ignore[arg-type]
        event_emitter=_NullEventEmitter(),  # type: ignore[arg-type]
        artifact_store=None,
        max_retries=2,
        retry_backoff_factor=1.0,
        retryable_error_codes=(),
        telemetry_bus=None,
        append_session_event=lambda *args, **kwargs: None,
    )


def test_run_output_transform_returns_latest_failed_candidate_after_retry_exhaustion(
    monkeypatch,
) -> None:
    executor = _executor()
    session = _session()
    candidate = _candidate()
    transform = ResolvedOutputTransform(
        spec=session.trial.task.output_transforms[0],
        transform_hash="transform-1",
    )
    attempts = {"count": 0}
    first_failure = candidate.model_copy(
        update={
            "status": RecordStatus.ERROR,
            "error": _retryable_error("stale transform failure"),
        }
    )

    def _transform_candidate(*args, **kwargs):
        del args, kwargs
        attempts["count"] += 1
        if attempts["count"] == 1:
            return first_failure
        raise RuntimeError("latest transform failure")

    monkeypatch.setattr(
        "themis.orchestration.overlay_stage.transform_candidate",
        _transform_candidate,
    )
    monkeypatch.setattr(
        "themis.orchestration.overlay_stage.map_exception_to_error_record",
        lambda exc, **kwargs: _retryable_error(str(exc)),
    )

    result = executor.run_output_transform(session, candidate, transform)

    assert result.status == RecordStatus.ERROR
    assert result.error is not None
    assert result.error.message == "latest transform failure"


def test_run_evaluation_candidate_returns_latest_failed_candidate_after_retry_exhaustion(
    monkeypatch,
) -> None:
    executor = _executor()
    session = _session()
    candidate = _candidate()
    transform = ResolvedOutputTransform(
        spec=session.trial.task.output_transforms[0],
        transform_hash="transform-1",
    )
    evaluation = ResolvedEvaluation(
        spec=session.trial.task.evaluations[0],
        transform=transform,
        evaluation_hash="evaluation-1",
    )
    attempts = {"count": 0}
    first_failure = candidate.model_copy(
        update={
            "status": RecordStatus.ERROR,
            "error": _retryable_error("stale evaluation failure"),
        }
    )

    def _evaluate_candidate(*args, **kwargs):
        del args, kwargs
        attempts["count"] += 1
        if attempts["count"] == 1:
            return first_failure
        raise RuntimeError("latest evaluation failure")

    monkeypatch.setattr(
        "themis.orchestration.overlay_stage.evaluate_candidate",
        _evaluate_candidate,
    )
    monkeypatch.setattr(
        "themis.orchestration.overlay_stage.map_exception_to_error_record",
        lambda exc, **kwargs: _retryable_error(str(exc)),
    )

    result = executor.run_evaluation_candidate(session, candidate, evaluation)

    assert result.status == RecordStatus.ERROR
    assert result.error is not None
    assert result.error.message == "latest evaluation failure"
