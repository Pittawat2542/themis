"""Generation and evaluation bundle export/import helpers."""

from __future__ import annotations

import json
import hashlib

from themis.core.events import (
    EvaluationCompletedEvent,
    GenerationCompletedEvent,
    ParseCompletedEvent,
    ReductionCompletedEvent,
    ScoreCompletedEvent,
)
from themis.core.models import GenerationResult, ParsedOutput, ReducedCandidate, Score
from themis.core.results import (
    EvaluationBundle,
    EvaluationBundleRecord,
    ParseBundle,
    ParseBundleRecord,
    ReductionBundle,
    ReductionBundleRecord,
    ScoreBundle,
    ScoreBundleRecord,
    GenerationBundle,
    GenerationBundleRecord,
)
from themis.core.store import RunStore
from themis.core.workflows import EvaluationExecution


def export_generation_bundle(store: RunStore, run_id: str) -> GenerationBundle:
    """Export stored generation artifacts into a portable bundle."""

    stored = store.resume(run_id)
    if stored is None:
        raise ValueError(f"Unknown run_id: {run_id}")

    records: list[GenerationBundleRecord] = []
    for event in stored.events:
        if isinstance(event, GenerationCompletedEvent) and event.result is not None:
            records.append(
                GenerationBundleRecord(
                    case_id=event.case_id,
                    candidate_id=event.candidate_id,
                    candidate_index=event.candidate_index,
                    seed=event.seed,
                    result_blob_ref=_blob_ref(
                        GenerationResult.model_validate(event.result).model_dump(
                            mode="json"
                        )
                    ),
                    result=GenerationResult.model_validate(event.result),
                )
            )

    return GenerationBundle(run_id=run_id, snapshot=stored.snapshot, records=records)


def import_generation_bundle(store: RunStore, bundle: GenerationBundle) -> None:
    """Import generation artifacts from a bundle into a store."""

    snapshot = bundle.snapshot
    if snapshot.run_id != bundle.run_id:
        raise ValueError("Bundle snapshot run_id does not match bundle.run_id")

    valid_case_ids = {
        case.case_id for dataset in snapshot.datasets for case in dataset.cases
    }
    store.persist_snapshot(snapshot)
    for record in bundle.records:
        if record.case_id not in valid_case_ids:
            raise ValueError(f"Unknown case_id in generation bundle: {record.case_id}")
        result_payload = record.result.model_dump(mode="json")
        blob_ref = store.store_blob(
            json.dumps(result_payload, sort_keys=True).encode("utf-8"),
            "application/json",
        )
        if record.result_blob_ref is not None and blob_ref != record.result_blob_ref:
            raise ValueError(
                "Generation bundle blob ref does not match serialized result payload"
            )
        store.persist_event(
            GenerationCompletedEvent(
                run_id=bundle.run_id,
                case_id=record.case_id,
                candidate_id=record.candidate_id,
                candidate_index=record.candidate_index,
                seed=record.seed,
                result=result_payload,
                result_blob_ref=record.result_blob_ref or blob_ref,
            )
        )


def export_reduction_bundle(store: RunStore, run_id: str) -> ReductionBundle:
    """Export stored reduction artifacts into a portable bundle."""

    stored = store.resume(run_id)
    if stored is None:
        raise ValueError(f"Unknown run_id: {run_id}")

    records: list[ReductionBundleRecord] = []
    for event in stored.events:
        if isinstance(event, ReductionCompletedEvent) and event.result is not None:
            records.append(
                ReductionBundleRecord(
                    case_id=event.case_id,
                    candidate_id=event.candidate_id,
                    result=ReducedCandidate.model_validate(event.result),
                )
            )

    return ReductionBundle(run_id=run_id, snapshot=stored.snapshot, records=records)


def import_reduction_bundle(store: RunStore, bundle: ReductionBundle) -> None:
    """Import reduction artifacts from a bundle into a store."""

    snapshot = bundle.snapshot
    if snapshot.run_id != bundle.run_id:
        raise ValueError("Bundle snapshot run_id does not match bundle.run_id")

    store.persist_snapshot(snapshot)
    for record in bundle.records:
        store.persist_event(
            ReductionCompletedEvent(
                run_id=bundle.run_id,
                case_id=record.case_id,
                candidate_id=record.candidate_id,
                source_candidate_ids=list(record.result.source_candidate_ids),
                result=record.result.model_dump(mode="json"),
            )
        )


def export_parse_bundle(store: RunStore, run_id: str) -> ParseBundle:
    """Export stored parse artifacts into a portable bundle."""

    stored = store.resume(run_id)
    if stored is None:
        raise ValueError(f"Unknown run_id: {run_id}")

    records: list[ParseBundleRecord] = []
    for event in stored.events:
        if isinstance(event, ParseCompletedEvent) and event.result is not None:
            records.append(
                ParseBundleRecord(
                    case_id=event.case_id,
                    candidate_id=event.candidate_id,
                    result=ParsedOutput.model_validate(event.result),
                )
            )

    return ParseBundle(run_id=run_id, snapshot=stored.snapshot, records=records)


def import_parse_bundle(store: RunStore, bundle: ParseBundle) -> None:
    """Import parse artifacts from a bundle into a store."""

    snapshot = bundle.snapshot
    if snapshot.run_id != bundle.run_id:
        raise ValueError("Bundle snapshot run_id does not match bundle.run_id")

    store.persist_snapshot(snapshot)
    for record in bundle.records:
        store.persist_event(
            ParseCompletedEvent(
                run_id=bundle.run_id,
                case_id=record.case_id,
                candidate_id=record.candidate_id,
                result=record.result.model_dump(mode="json"),
            )
        )


def export_score_bundle(store: RunStore, run_id: str) -> ScoreBundle:
    """Export stored score artifacts into a portable bundle."""

    stored = store.resume(run_id)
    if stored is None:
        raise ValueError(f"Unknown run_id: {run_id}")

    records: list[ScoreBundleRecord] = []
    for event in stored.events:
        if isinstance(event, ScoreCompletedEvent) and event.score is not None:
            records.append(
                ScoreBundleRecord(
                    case_id=event.case_id,
                    candidate_id=event.candidate_id,
                    metric_id=event.metric_id,
                    score=Score.model_validate(event.score),
                )
            )

    return ScoreBundle(run_id=run_id, snapshot=stored.snapshot, records=records)


def import_score_bundle(store: RunStore, bundle: ScoreBundle) -> None:
    """Import score artifacts from a bundle into a store."""

    snapshot = bundle.snapshot
    if snapshot.run_id != bundle.run_id:
        raise ValueError("Bundle snapshot run_id does not match bundle.run_id")

    store.persist_snapshot(snapshot)
    for record in bundle.records:
        store.persist_event(
            ScoreCompletedEvent(
                run_id=bundle.run_id,
                case_id=record.case_id,
                candidate_id=record.candidate_id,
                metric_id=record.metric_id,
                score=record.score.model_dump(mode="json"),
            )
        )


def export_evaluation_bundle(store: RunStore, run_id: str) -> EvaluationBundle:
    """Export stored evaluation artifacts into a portable bundle."""

    stored = store.resume(run_id)
    if stored is None:
        raise ValueError(f"Unknown run_id: {run_id}")

    records: list[EvaluationBundleRecord] = []
    for event in stored.events:
        if isinstance(event, EvaluationCompletedEvent) and event.execution is not None:
            records.append(
                EvaluationBundleRecord(
                    case_id=event.case_id,
                    metric_id=event.metric_id,
                    candidate_id=event.candidate_id,
                    execution_blob_ref=_blob_ref(
                        EvaluationExecution.model_validate(event.execution).model_dump(
                            mode="json"
                        )
                    ),
                    execution=EvaluationExecution.model_validate(event.execution),
                )
            )

    return EvaluationBundle(run_id=run_id, snapshot=stored.snapshot, records=records)


def import_evaluation_bundle(store: RunStore, bundle: EvaluationBundle) -> None:
    """Import evaluation artifacts from a bundle into a store."""

    snapshot = bundle.snapshot
    if snapshot.run_id != bundle.run_id:
        raise ValueError("Bundle snapshot run_id does not match bundle.run_id")

    valid_case_ids = {
        case.case_id for dataset in snapshot.datasets for case in dataset.cases
    }
    store.persist_snapshot(snapshot)
    for record in bundle.records:
        if record.case_id not in valid_case_ids:
            raise ValueError(f"Unknown case_id in evaluation bundle: {record.case_id}")
        execution_payload = record.execution.model_dump(mode="json")
        blob_ref = store.store_blob(
            json.dumps(execution_payload, sort_keys=True).encode("utf-8"),
            "application/json",
        )
        if (
            record.execution_blob_ref is not None
            and blob_ref != record.execution_blob_ref
        ):
            raise ValueError(
                "Evaluation bundle blob ref does not match serialized execution payload"
            )
        store.persist_event(
            EvaluationCompletedEvent(
                run_id=bundle.run_id,
                case_id=record.case_id,
                candidate_id=record.candidate_id,
                metric_id=record.metric_id,
                execution=execution_payload,
                execution_blob_ref=record.execution_blob_ref or blob_ref,
            )
        )
        final_score = _final_score(record.metric_id, record.execution)
        if record.candidate_id is not None and final_score is not None:
            store.persist_event(
                ScoreCompletedEvent(
                    run_id=bundle.run_id,
                    case_id=record.case_id,
                    candidate_id=record.candidate_id,
                    metric_id=record.metric_id,
                    score=final_score.model_dump(mode="json"),
                )
            )


def _final_score(metric_id: str, execution: EvaluationExecution) -> Score | None:
    if execution.aggregation_output is not None and isinstance(
        execution.aggregation_output.value, (int, float)
    ):
        return Score(
            metric_id=metric_id,
            value=float(execution.aggregation_output.value),
            details=execution.aggregation_output.details,
        )
    if not execution.failures and len(execution.scores) == 1:
        return execution.scores[-1]
    return None


def _blob_ref(payload: dict[str, object]) -> str:
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return f"sha256:{hashlib.sha256(blob).hexdigest()}"
