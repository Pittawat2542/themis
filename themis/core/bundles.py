"""Generation and evaluation bundle export/import helpers."""

from __future__ import annotations

import json
import hashlib
from typing import TypeVar

from pydantic import BaseModel

from themis.core.case_refs import CaseRef, resolve_case_key
from themis.core.config import EvidenceRetention
from themis.core.events import (
    EvaluationCompletedEvent,
    GenerationCompletedEvent,
    ParseCompletedEvent,
    ReductionCompletedEvent,
    ScoreCompletedEvent,
)
from themis.core.models import (
    ParsedOutput,
    ReducedCandidate,
    MetricResult,
    Candidate,
)
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
from themis.core.security import EvidenceSanitizer
from themis.core.workflows import EvaluationExecution

BundleT = TypeVar("BundleT", bound=BaseModel)
_RETENTION_RANK = {
    EvidenceRetention.MINIMAL: 0,
    EvidenceRetention.STANDARD: 1,
    EvidenceRetention.FULL: 2,
}


def export_generation_bundle(
    store: RunStore,
    run_id: str,
    *,
    retention: EvidenceRetention | None = None,
) -> GenerationBundle:
    """Export stored generation artifacts into a portable bundle."""

    stored = store.resume(run_id)
    if stored is None:
        raise ValueError(f"Unknown run_id: {run_id}")

    case_refs = _bundle_case_refs(stored.snapshot)
    records: list[GenerationBundleRecord] = []
    for event in stored.events:
        if (
            isinstance(event, GenerationCompletedEvent)
            and event.result is not None
        ):
            case_ref = _bundle_record_case_ref(case_refs, event)
            records.append(
                GenerationBundleRecord(
                    case_id=event.case_id,
                    dataset_id=case_ref.dataset_id
                    if case_ref is not None
                    else getattr(event, "dataset_id", None),
                    case_key=case_ref.case_key
                    if case_ref is not None
                    else getattr(event, "case_key", None),
                    candidate_id=event.candidate_id,
                    candidate_index=event.candidate_index,
                    seed=event.seed,
                    result_blob_ref=_blob_ref(
                        Candidate.model_validate(event.result).model_dump(
                            mode="json"
                        )
                    ),
                    result=Candidate.model_validate(event.result),
                )
            )

    return _sanitize_export_bundle(
        GenerationBundle(run_id=run_id, snapshot=stored.snapshot, records=records),
        retention,
    )


def import_generation_bundle(
    store: RunStore,
    bundle: GenerationBundle,
    *,
    retention: EvidenceRetention = EvidenceRetention.STANDARD,
) -> None:
    """Import generation artifacts from a bundle into a store."""

    bundle = _sanitize_bundle(bundle, retention)
    snapshot = bundle.snapshot
    if snapshot.run_id != bundle.run_id:
        raise ValueError("Bundle snapshot run_id does not match bundle.run_id")

    valid_case_refs = _bundle_case_refs(snapshot)
    store.persist_snapshot(snapshot)
    for record in bundle.records:
        record_case_key = resolve_case_key(
            case_id=record.case_id,
            dataset_id=record.dataset_id,
            case_key=record.case_key,
        )
        if record_case_key not in valid_case_refs:
            raise ValueError(f"Unknown case_id in generation bundle: {record.case_id}")
        case_ref = valid_case_refs[record_case_key]
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
                case_id=case_ref.case_id,
                dataset_id=case_ref.dataset_id,
                case_key=case_ref.case_key,
                candidate_id=record.candidate_id,
                candidate_index=record.candidate_index,
                seed=record.seed,
                result=result_payload,
                result_blob_ref=record.result_blob_ref or blob_ref,
            )
        )


def export_reduction_bundle(
    store: RunStore, run_id: str, *, retention: EvidenceRetention | None = None
) -> ReductionBundle:
    """Export stored reduction artifacts into a portable bundle."""

    stored = store.resume(run_id)
    if stored is None:
        raise ValueError(f"Unknown run_id: {run_id}")

    case_refs = _bundle_case_refs(stored.snapshot)
    records: list[ReductionBundleRecord] = []
    for event in stored.events:
        if isinstance(event, ReductionCompletedEvent) and event.result is not None:
            case_ref = _bundle_record_case_ref(case_refs, event)
            records.append(
                ReductionBundleRecord(
                    case_id=event.case_id,
                    dataset_id=case_ref.dataset_id
                    if case_ref is not None
                    else getattr(event, "dataset_id", None),
                    case_key=case_ref.case_key
                    if case_ref is not None
                    else getattr(event, "case_key", None),
                    candidate_id=event.candidate_id,
                    result=ReducedCandidate.model_validate(event.result),
                )
            )

    return _sanitize_export_bundle(
        ReductionBundle(run_id=run_id, snapshot=stored.snapshot, records=records),
        retention,
    )


def import_reduction_bundle(
    store: RunStore,
    bundle: ReductionBundle,
    *,
    retention: EvidenceRetention = EvidenceRetention.STANDARD,
) -> None:
    """Import reduction artifacts from a bundle into a store."""

    bundle = _sanitize_bundle(bundle, retention)
    snapshot = bundle.snapshot
    if snapshot.run_id != bundle.run_id:
        raise ValueError("Bundle snapshot run_id does not match bundle.run_id")

    valid_case_refs = _bundle_case_refs(snapshot)
    store.persist_snapshot(snapshot)
    for record in bundle.records:
        record_case_key = resolve_case_key(
            case_id=record.case_id,
            dataset_id=record.dataset_id,
            case_key=record.case_key,
        )
        if record_case_key not in valid_case_refs:
            raise ValueError(f"Unknown case_id in reduction bundle: {record.case_id}")
        case_ref = valid_case_refs[record_case_key]
        store.persist_event(
            ReductionCompletedEvent(
                run_id=bundle.run_id,
                case_id=case_ref.case_id,
                dataset_id=case_ref.dataset_id,
                case_key=case_ref.case_key,
                candidate_id=record.candidate_id,
                source_candidate_ids=list(record.result.source_candidate_ids),
                result=record.result.model_dump(mode="json"),
            )
        )


def export_parse_bundle(
    store: RunStore, run_id: str, *, retention: EvidenceRetention | None = None
) -> ParseBundle:
    """Export stored parse artifacts into a portable bundle."""

    stored = store.resume(run_id)
    if stored is None:
        raise ValueError(f"Unknown run_id: {run_id}")

    case_refs = _bundle_case_refs(stored.snapshot)
    records: list[ParseBundleRecord] = []
    for event in stored.events:
        if isinstance(event, ParseCompletedEvent) and event.result is not None:
            case_ref = _bundle_record_case_ref(case_refs, event)
            records.append(
                ParseBundleRecord(
                    case_id=event.case_id,
                    dataset_id=case_ref.dataset_id
                    if case_ref is not None
                    else getattr(event, "dataset_id", None),
                    case_key=case_ref.case_key
                    if case_ref is not None
                    else getattr(event, "case_key", None),
                    candidate_id=event.candidate_id,
                    result=ParsedOutput.model_validate(event.result),
                )
            )

    return _sanitize_export_bundle(
        ParseBundle(run_id=run_id, snapshot=stored.snapshot, records=records),
        retention,
    )


def import_parse_bundle(
    store: RunStore,
    bundle: ParseBundle,
    *,
    retention: EvidenceRetention = EvidenceRetention.STANDARD,
) -> None:
    """Import parse artifacts from a bundle into a store."""

    bundle = _sanitize_bundle(bundle, retention)
    snapshot = bundle.snapshot
    if snapshot.run_id != bundle.run_id:
        raise ValueError("Bundle snapshot run_id does not match bundle.run_id")

    valid_case_refs = _bundle_case_refs(snapshot)
    store.persist_snapshot(snapshot)
    for record in bundle.records:
        record_case_key = resolve_case_key(
            case_id=record.case_id,
            dataset_id=record.dataset_id,
            case_key=record.case_key,
        )
        if record_case_key not in valid_case_refs:
            raise ValueError(f"Unknown case_id in parse bundle: {record.case_id}")
        case_ref = valid_case_refs[record_case_key]
        store.persist_event(
            ParseCompletedEvent(
                run_id=bundle.run_id,
                case_id=case_ref.case_id,
                dataset_id=case_ref.dataset_id,
                case_key=case_ref.case_key,
                candidate_id=record.candidate_id,
                result=record.result.model_dump(mode="json"),
            )
        )


def export_score_bundle(
    store: RunStore, run_id: str, *, retention: EvidenceRetention | None = None
) -> ScoreBundle:
    """Export stored score artifacts into a portable bundle."""

    stored = store.resume(run_id)
    if stored is None:
        raise ValueError(f"Unknown run_id: {run_id}")

    case_refs = _bundle_case_refs(stored.snapshot)
    records: list[ScoreBundleRecord] = []
    for event in stored.events:
        if isinstance(event, ScoreCompletedEvent) and event.metric_result is not None:
            case_ref = _bundle_record_case_ref(case_refs, event)
            records.append(
                ScoreBundleRecord(
                    case_id=event.case_id,
                    dataset_id=case_ref.dataset_id
                    if case_ref is not None
                    else getattr(event, "dataset_id", None),
                    case_key=case_ref.case_key
                    if case_ref is not None
                    else getattr(event, "case_key", None),
                    candidate_id=event.candidate_id,
                    metric_id=event.metric_id,
                    metric_result=MetricResult.model_validate(event.metric_result),
                )
            )

    return _sanitize_export_bundle(
        ScoreBundle(run_id=run_id, snapshot=stored.snapshot, records=records),
        retention,
    )


def import_score_bundle(
    store: RunStore,
    bundle: ScoreBundle,
    *,
    retention: EvidenceRetention = EvidenceRetention.STANDARD,
) -> None:
    """Import score artifacts from a bundle into a store."""

    bundle = _sanitize_bundle(bundle, retention)
    snapshot = bundle.snapshot
    if snapshot.run_id != bundle.run_id:
        raise ValueError("Bundle snapshot run_id does not match bundle.run_id")

    valid_case_refs = _bundle_case_refs(snapshot)
    store.persist_snapshot(snapshot)
    for record in bundle.records:
        record_case_key = resolve_case_key(
            case_id=record.case_id,
            dataset_id=record.dataset_id,
            case_key=record.case_key,
        )
        if record_case_key not in valid_case_refs:
            raise ValueError(f"Unknown case_id in score bundle: {record.case_id}")
        case_ref = valid_case_refs[record_case_key]
        store.persist_event(
            ScoreCompletedEvent(
                run_id=bundle.run_id,
                case_id=case_ref.case_id,
                dataset_id=case_ref.dataset_id,
                case_key=case_ref.case_key,
                candidate_id=record.candidate_id,
                metric_id=record.metric_id,
                metric_result=record.metric_result.model_dump(mode="json"),
            )
        )


def export_evaluation_bundle(
    store: RunStore, run_id: str, *, retention: EvidenceRetention | None = None
) -> EvaluationBundle:
    """Export stored evaluation artifacts into a portable bundle."""

    stored = store.resume(run_id)
    if stored is None:
        raise ValueError(f"Unknown run_id: {run_id}")

    case_refs = _bundle_case_refs(stored.snapshot)
    records: list[EvaluationBundleRecord] = []
    for event in stored.events:
        if isinstance(event, EvaluationCompletedEvent) and event.execution is not None:
            case_ref = _bundle_record_case_ref(case_refs, event)
            records.append(
                EvaluationBundleRecord(
                    case_id=event.case_id,
                    dataset_id=case_ref.dataset_id
                    if case_ref is not None
                    else getattr(event, "dataset_id", None),
                    case_key=case_ref.case_key
                    if case_ref is not None
                    else getattr(event, "case_key", None),
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

    return _sanitize_export_bundle(
        EvaluationBundle(run_id=run_id, snapshot=stored.snapshot, records=records),
        retention,
    )


def import_evaluation_bundle(
    store: RunStore,
    bundle: EvaluationBundle,
    *,
    retention: EvidenceRetention = EvidenceRetention.STANDARD,
) -> None:
    """Import evaluation artifacts from a bundle into a store."""

    bundle = _sanitize_bundle(bundle, retention)
    snapshot = bundle.snapshot
    if snapshot.run_id != bundle.run_id:
        raise ValueError("Bundle snapshot run_id does not match bundle.run_id")

    valid_case_refs = _bundle_case_refs(snapshot)
    store.persist_snapshot(snapshot)
    for record in bundle.records:
        record_case_key = resolve_case_key(
            case_id=record.case_id,
            dataset_id=record.dataset_id,
            case_key=record.case_key,
        )
        if record_case_key not in valid_case_refs:
            raise ValueError(f"Unknown case_id in evaluation bundle: {record.case_id}")
        case_ref = valid_case_refs[record_case_key]
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
                case_id=case_ref.case_id,
                dataset_id=case_ref.dataset_id,
                case_key=case_ref.case_key,
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
                    case_id=case_ref.case_id,
                    dataset_id=case_ref.dataset_id,
                    case_key=case_ref.case_key,
                    candidate_id=record.candidate_id,
                    metric_id=record.metric_id,
                    metric_result=final_score.model_dump(mode="json"),
                )
            )


def _sanitize_bundle(bundle: BundleT, retention: EvidenceRetention) -> BundleT:
    payload = EvidenceSanitizer(retention).value(bundle.model_dump(mode="json"))
    if isinstance(payload, dict):
        snapshot_payload = payload.get("snapshot")
        if isinstance(snapshot_payload, dict):
            snapshot_payload.pop("run_id", None)
    return type(bundle).model_validate(payload)


def _sanitize_export_bundle(
    bundle: BundleT, requested: EvidenceRetention | None
) -> BundleT:
    source = bundle.snapshot.provenance.runtime.evidence_retention  # type: ignore[attr-defined]
    retention = requested or source
    if _RETENTION_RANK[retention] > _RETENTION_RANK[source]:
        raise ValueError(
            f"Cannot elevate bundle retention from {source.value} to {retention.value}"
        )
    return _sanitize_bundle(bundle, retention)


def _final_score(metric_id: str, execution: EvaluationExecution) -> MetricResult | None:
    if execution.aggregation_output is not None:
        if execution.aggregation_output.metric_result is not None:
            return execution.aggregation_output.metric_result
        if isinstance(execution.aggregation_output.value, (int, float)):
            return MetricResult(
                metric_id=metric_id,
                value=float(execution.aggregation_output.value),
                metadata=execution.aggregation_output.metadata,
            )
    if not execution.failures and len(execution.metric_results) == 1:
        return execution.metric_results[-1]
    return None


def _bundle_case_refs(snapshot) -> dict[str, CaseRef]:
    case_refs: dict[str, CaseRef] = {}
    duplicate_case_ids: set[str] = set()
    for dataset in snapshot.datasets:
        for case in dataset.cases:
            case_ref = CaseRef(dataset_id=dataset.dataset_id, case_id=case.case_id)
            case_refs[case_ref.case_key] = case_ref
            if case.case_id in case_refs:
                duplicate_case_ids.add(case.case_id)
            else:
                case_refs[case.case_id] = case_ref
    for case_id in duplicate_case_ids:
        case_refs.pop(case_id, None)
    return case_refs


def _bundle_record_case_ref(
    case_refs: dict[str, CaseRef], record_or_event
) -> CaseRef | None:
    record_case_key = resolve_case_key(
        case_id=record_or_event.case_id,
        dataset_id=getattr(record_or_event, "dataset_id", None),
        case_key=getattr(record_or_event, "case_key", None),
    )
    return case_refs.get(record_case_key)


def _blob_ref(payload: dict[str, object]) -> str:
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return f"sha256:{hashlib.sha256(blob).hexdigest()}"
