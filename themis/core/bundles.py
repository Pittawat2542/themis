"""Generation and evaluation bundle export/import helpers."""

from __future__ import annotations

import json
import hashlib

from themis.core.events import EvaluationCompletedEvent, GenerationCompletedEvent, ScoreCompletedEvent
from themis.core.models import GenerationResult, Score
from themis.core.results import (
    EvaluationBundle,
    EvaluationBundleRecord,
    GenerationBundle,
    GenerationBundleRecord,
)
from themis.core.store import RunStore
from themis.core.workflows import EvaluationExecution


def export_generation_bundle(store: RunStore, run_id: str) -> GenerationBundle:
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
                    result_blob_ref=_blob_ref(GenerationResult.model_validate(event.result).model_dump(mode="json")),
                    result=GenerationResult.model_validate(event.result),
                )
            )

    return GenerationBundle(run_id=run_id, snapshot=stored.snapshot, records=records)


def import_generation_bundle(store: RunStore, bundle: GenerationBundle) -> None:
    snapshot = bundle.snapshot
    if snapshot.run_id != bundle.run_id:
        raise ValueError("Bundle snapshot run_id does not match bundle.run_id")

    valid_case_ids = {case.case_id for dataset in snapshot.datasets for case in dataset.cases}
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
            raise ValueError("Generation bundle blob ref does not match serialized result payload")
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


def export_evaluation_bundle(store: RunStore, run_id: str) -> EvaluationBundle:
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
                    execution_blob_ref=_blob_ref(EvaluationExecution.model_validate(event.execution).model_dump(mode="json")),
                    execution=EvaluationExecution.model_validate(event.execution),
                )
            )

    return EvaluationBundle(run_id=run_id, snapshot=stored.snapshot, records=records)


def import_evaluation_bundle(store: RunStore, bundle: EvaluationBundle) -> None:
    snapshot = bundle.snapshot
    if snapshot.run_id != bundle.run_id:
        raise ValueError("Bundle snapshot run_id does not match bundle.run_id")

    valid_case_ids = {case.case_id for dataset in snapshot.datasets for case in dataset.cases}
    store.persist_snapshot(snapshot)
    for record in bundle.records:
        if record.case_id not in valid_case_ids:
            raise ValueError(f"Unknown case_id in evaluation bundle: {record.case_id}")
        execution_payload = record.execution.model_dump(mode="json")
        blob_ref = store.store_blob(
            json.dumps(execution_payload, sort_keys=True).encode("utf-8"),
            "application/json",
        )
        if record.execution_blob_ref is not None and blob_ref != record.execution_blob_ref:
            raise ValueError("Evaluation bundle blob ref does not match serialized execution payload")
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
        if record.candidate_id is not None:
            store.persist_event(
                ScoreCompletedEvent(
                    run_id=bundle.run_id,
                    case_id=record.case_id,
                    candidate_id=record.candidate_id,
                    metric_id=record.metric_id,
                    score=_final_score(record.metric_id, record.execution).model_dump(mode="json"),
                )
            )


def _final_score(metric_id: str, execution: EvaluationExecution) -> Score:
    if execution.aggregation_output is not None and isinstance(execution.aggregation_output.value, (int, float)):
        return Score(
            metric_id=metric_id,
            value=float(execution.aggregation_output.value),
            details=execution.aggregation_output.details,
        )
    if execution.scores:
        return execution.scores[-1]
    return Score(metric_id=metric_id, value=0.0)


def _blob_ref(payload: dict[str, object]) -> str:
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return f"sha256:{hashlib.sha256(blob).hexdigest()}"
