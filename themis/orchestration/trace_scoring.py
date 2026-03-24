"""Trace-view normalization and trace-metric scoring helpers."""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import TypeAdapter

from themis.records.conversation import ConversationEvent
from themis.records.trial import TrialRecord
from themis.runtime.trace_view import TraceEvent, TraceView
from themis.specs.foundational import MetricRefSpec
from themis.types.enums import RecordType
from themis.types.events import TraceScoreRow, TrialEvent, TrialEventType

_CONVERSATION_EVENT_ADAPTER: TypeAdapter[ConversationEvent] = TypeAdapter(
    ConversationEvent
)


def score_trial_traces(
    *,
    projection_repo,
    registry,
    record: TrialRecord,
    trial_hash: str,
    evaluation_hash: str | None = None,
) -> list[TraceScoreRow]:
    """Build normalized trace views for one trial and score all trace evaluations."""

    trial_spec = record.trial_spec
    if trial_spec is None or not trial_spec.task.trace_evaluations:
        return []

    from themis.orchestration.task_resolution import resolve_task_stages

    resolved_trace_evaluations = resolve_task_stages(trial_spec.task).trace_evaluations
    resolved_by_name = {
        resolved.spec.name: resolved for resolved in resolved_trace_evaluations
    }
    trial_view = projection_repo.get_timeline_view(
        trial_hash,
        RecordType.TRIAL,
        evaluation_hash=evaluation_hash,
    )
    if trial_view is None:
        return []

    candidate_views = [
        projection_repo.get_timeline_view(
            candidate.spec_hash,
            RecordType.CANDIDATE,
            evaluation_hash=evaluation_hash,
        )
        for candidate in record.candidates
    ]
    populated_candidate_views = [view for view in candidate_views if view is not None]

    rows: list[TraceScoreRow] = []
    for trace_evaluation in trial_spec.task.trace_evaluations:
        resolved = resolved_by_name.get(trace_evaluation.name)
        if resolved is None:
            continue
        trace_views = build_trace_views(
            trial_view=trial_view,
            candidate_views=populated_candidate_views,
            scope=trace_evaluation.scope,
        )
        for trace_view in trace_views:
            for raw_metric_ref in trace_evaluation.metrics:
                metric_ref = (
                    raw_metric_ref
                    if isinstance(raw_metric_ref, MetricRefSpec)
                    else MetricRefSpec.model_validate(raw_metric_ref)
                )
                metric = registry.get_metric(metric_ref.id)
                if not hasattr(metric, "score_trace"):
                    raise TypeError(
                        f"Metric '{metric_ref.id}' does not implement score_trace()."
                    )
                score = metric.score_trace(
                    trace_view,
                    {
                        "metric_ref": metric_ref,
                        "metric_config": metric_ref.config,
                        "metric_registry": registry,
                        "trace_scope": trace_view.trace_scope,
                    },
                )
                rows.append(
                    TraceScoreRow(
                        trial_hash=trial_hash,
                        trace_scope=trace_view.trace_scope,
                        trace_id=trace_view.trace_id,
                        trace_score_hash=resolved.trace_score_hash,
                        metric_id=score.metric_id,
                        score=score.value,
                        details=score.details,
                    )
                )
    return rows


def build_trace_views(
    *,
    trial_view,
    candidate_views: Sequence,
    scope: str,
) -> list[TraceView]:
    """Build candidate- or trial-scoped trace views from materialized timeline views."""

    if scope == "trial_trace":
        return [
            TraceView(
                trace_id=trial_view.trial_hash,
                trace_scope=scope,
                trial_hash=trial_view.trial_hash,
                trial_view=trial_view,
                candidate_view=None,
                candidate_views=list(candidate_views),
                trace_events=normalize_trace_events(trial_view.related_events),
            )
        ]
    if scope != "candidate_trace":
        raise ValueError(f"Unsupported trace scope '{scope}'.")
    return [
        TraceView(
            trace_id=view.record_id,
            trace_scope=scope,
            trial_hash=view.trial_hash,
            trial_view=trial_view,
            candidate_view=view,
            candidate_views=[view],
            trace_events=normalize_trace_events(view.related_events),
        )
        for view in candidate_views
    ]


def normalize_trace_events(events: Sequence[TrialEvent]) -> list[TraceEvent]:
    """Normalize persisted trial events into trace events used by trace metrics."""

    normalized: list[TraceEvent] = []
    ordered_events = sorted(
        events,
        key=lambda event: (
            event.event_seq,
            _conversation_event_index(event),
        ),
    )
    for event in ordered_events:
        if (
            event.event_type != TrialEventType.CONVERSATION_EVENT
            or event.payload is None
        ):
            continue
        conversation_event = _CONVERSATION_EVENT_ADAPTER.validate_python(event.payload)
        tool_name = None
        node_id = None
        if hasattr(conversation_event.payload, "tool_name"):
            tool_name = conversation_event.payload.tool_name
        if hasattr(conversation_event.payload, "node_id"):
            node_id = conversation_event.payload.node_id
        normalized.append(
            TraceEvent(
                event_key=event.event_id,
                source_kind="conversation",
                kind=conversation_event.kind,
                trial_hash=event.trial_hash,
                candidate_id=event.candidate_id,
                event_seq=event.event_seq,
                conversation_event_index=conversation_event.event_index,
                stage=None if event.stage is None else event.stage.value,
                timestamp=conversation_event.timestamp,
                tool_name=tool_name,
                node_id=node_id,
                source_metadata={"event_type": event.event_type.value},
            )
        )
    return normalized


def _conversation_event_index(event: TrialEvent) -> int:
    if event.event_type != TrialEventType.CONVERSATION_EVENT or event.payload is None:
        return -1
    payload = _CONVERSATION_EVENT_ADAPTER.validate_python(event.payload)
    return payload.event_index
