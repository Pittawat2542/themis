"""Generation and evaluation bundle export/import helpers."""

from __future__ import annotations

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
        store.persist_event(
            GenerationCompletedEvent(
                run_id=bundle.run_id,
                case_id=record.case_id,
                candidate_id=record.candidate_id,
                candidate_index=record.candidate_index,
                seed=record.seed,
                result=record.result.model_dump(mode="json"),
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
        store.persist_event(
            EvaluationCompletedEvent(
                run_id=bundle.run_id,
                case_id=record.case_id,
                candidate_id=record.candidate_id,
                metric_id=record.metric_id,
                execution=record.execution.model_dump(mode="json"),
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
