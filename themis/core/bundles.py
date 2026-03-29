"""Generation bundle export/import helpers."""

from __future__ import annotations

from themis.core.events import GenerationCompletedEvent
from themis.core.models import GenerationResult
from themis.core.results import GenerationBundle, GenerationBundleRecord


def export_generation_bundle(store, run_id: str) -> GenerationBundle:
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


def import_generation_bundle(store, bundle: GenerationBundle) -> None:
    snapshot = bundle.snapshot
    if getattr(snapshot, "run_id", None) != bundle.run_id:
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
