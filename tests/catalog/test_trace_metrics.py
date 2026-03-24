from __future__ import annotations

from datetime import UTC, datetime

from themis.catalog.metrics import (
    EventSequenceTraceMetric,
    NodePresenceTraceMetric,
    ToolPresenceTraceMetric,
    ToolStageTraceMetric,
)
from themis.runtime.trace_view import TraceEvent, TraceView
from themis.runtime.timeline_view import RecordTimelineView
from themis.records.timeline import RecordTimeline
from themis.specs.experiment import InferenceParamsSpec, PromptTemplateSpec, TrialSpec
from themis.specs.foundational import DatasetSpec, GenerationSpec, ModelSpec, TaskSpec
from themis.types.enums import DatasetSource, RecordType
from themis.types.events import TimelineStage


def _trace_view(*events: TraceEvent, scope: str = "candidate_trace") -> TraceView:
    trial = TrialSpec(
        trial_id="trial-trace-metric",
        model=ModelSpec(model_id="demo-model", provider="demo"),
        task=TaskSpec(
            task_id="agent-task",
            dataset=DatasetSpec(source=DatasetSource.MEMORY),
            generation=GenerationSpec(),
        ),
        item_id="item-1",
        prompt=PromptTemplateSpec(id="agent", messages=[]),
        params=InferenceParamsSpec(),
    )
    timeline = RecordTimeline(
        record_id="candidate-1" if scope == "candidate_trace" else trial.spec_hash,
        record_type=(
            RecordType.CANDIDATE if scope == "candidate_trace" else RecordType.TRIAL
        ),
        trial_hash=trial.spec_hash,
        candidate_id="candidate-1" if scope == "candidate_trace" else None,
        item_id="item-1",
        stages=[],
    )
    trial_view = RecordTimelineView(
        record_id=trial.spec_hash,
        record_type=RecordType.TRIAL,
        trial_hash=trial.spec_hash,
        trial_spec=trial,
        timeline=timeline,
        related_events=[],
    )
    candidate_views = []
    candidate_view = None
    if scope == "candidate_trace":
        candidate_view = RecordTimelineView(
            record_id="candidate-1",
            record_type=RecordType.CANDIDATE,
            trial_hash=trial.spec_hash,
            candidate_id="candidate-1",
            trial_spec=trial,
            timeline=timeline,
            related_events=[],
        )
        candidate_views = [candidate_view]
    return TraceView(
        trace_id="candidate-1" if scope == "candidate_trace" else trial.spec_hash,
        trace_scope=scope,
        trial_hash=trial.spec_hash,
        trial_view=trial_view,
        candidate_view=candidate_view,
        candidate_views=candidate_views,
        trace_events=list(events),
    )


def _event(
    *,
    kind: str,
    event_seq: int,
    stage: str | None = None,
    candidate_id: str | None = "candidate-1",
    tool_name: str | None = None,
    node_id: str | None = None,
) -> TraceEvent:
    return TraceEvent(
        event_key=f"evt-{event_seq}",
        source_kind="conversation",
        kind=kind,
        trial_hash="trial-trace-metric",
        candidate_id=candidate_id,
        event_seq=event_seq,
        conversation_event_index=event_seq,
        stage=stage,
        timestamp=datetime(2026, 3, 24, 12, 0, event_seq, tzinfo=UTC),
        tool_name=tool_name,
        node_id=node_id,
    )


def test_tool_presence_trace_metric_detects_tool_call() -> None:
    trace = _trace_view(_event(kind="tool_call", event_seq=1, tool_name="search"))

    score = ToolPresenceTraceMetric().score_trace(
        trace,
        {"metric_config": {"tool_name": "search"}},
    )

    assert score.value == 1.0
    assert score.error is None


def test_tool_stage_trace_metric_requires_matching_stage() -> None:
    trace = _trace_view(
        _event(
            kind="tool_call",
            event_seq=1,
            tool_name="search",
            stage=TimelineStage.INFERENCE.value,
        )
    )

    score = ToolStageTraceMetric().score_trace(
        trace,
        {
            "metric_config": {
                "tool_name": "search",
                "stage": TimelineStage.INFERENCE.value,
            }
        },
    )

    assert score.value == 1.0
    assert score.error is None


def test_event_sequence_trace_metric_tracks_order_across_trial_trace() -> None:
    trace = _trace_view(
        _event(
            kind="tool_call",
            event_seq=1,
            candidate_id="candidate-1",
            tool_name="search",
        ),
        _event(
            kind="node_enter",
            event_seq=2,
            candidate_id="candidate-2",
            node_id="reviewer",
        ),
        scope="trial_trace",
    )

    score = EventSequenceTraceMetric().score_trace(
        trace,
        {
            "metric_config": {
                "steps": [
                    {"kind": "tool_call", "tool_name": "search"},
                    {"kind": "node_enter", "node_id": "reviewer"},
                ]
            }
        },
    )

    assert score.value == 1.0


def test_node_presence_trace_metric_reports_missing_trace_signal() -> None:
    trace = _trace_view(_event(kind="message", event_seq=1))

    score = NodePresenceTraceMetric().score_trace(
        trace,
        {"metric_config": {"node_id": "planner"}},
    )

    assert score.value == 0.0
    assert score.error == "missing_trace_signal"
