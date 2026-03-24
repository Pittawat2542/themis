"""Trace-level metric plugins for agentic and workflow evaluation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from themis.records.evaluation import MetricScore
from themis.runtime.trace_view import TraceEvent, TraceView
from themis.types.json_validation import validate_json_dict


def _metric_config(context: Mapping[str, object]) -> Mapping[str, object]:
    value = context.get("metric_config", {})
    if not isinstance(value, Mapping):
        raise TypeError("metric_config must be a mapping.")
    return value


def _config_str(
    context: Mapping[str, object],
    *,
    key: str,
) -> str:
    value = _metric_config(context).get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"metric_config['{key}'] must be a non-empty string.")
    return value


def _config_steps(context: Mapping[str, object]) -> list[dict[str, str]]:
    raw_steps = _metric_config(context).get("steps")
    if not isinstance(raw_steps, Sequence) or isinstance(raw_steps, (str, bytes)):
        raise ValueError("metric_config['steps'] must be a non-empty list.")
    steps: list[dict[str, str]] = []
    for index, raw_step in enumerate(raw_steps):
        if not isinstance(raw_step, Mapping):
            raise ValueError(f"metric_config['steps'][{index}] must be a mapping.")
        step: dict[str, str] = {}
        for key in ("kind", "tool_name", "node_id", "stage", "candidate_id"):
            value = raw_step.get(key)
            if value is None:
                continue
            if not isinstance(value, str) or not value:
                raise ValueError(
                    f"metric_config['steps'][{index}]['{key}'] must be a non-empty string."
                )
            step[key] = value
        if "kind" not in step:
            raise ValueError(
                f"metric_config['steps'][{index}'] must define a 'kind' field."
            )
        steps.append(step)
    if not steps:
        raise ValueError("metric_config['steps'] must not be empty.")
    return steps


def _matches(event: TraceEvent, expected: Mapping[str, str]) -> bool:
    for key, expected_value in expected.items():
        if key == "kind" and event.kind != expected_value:
            return False
        if key == "tool_name" and event.tool_name != expected_value:
            return False
        if key == "node_id" and event.node_id != expected_value:
            return False
        if key == "stage" and event.stage != expected_value:
            return False
        if key == "candidate_id" and event.candidate_id != expected_value:
            return False
    return True


def _missing_signal_details(
    *,
    metric_id: str,
    signal: str,
    extra_details: Mapping[str, object] | None = None,
) -> MetricScore:
    details: dict[str, object] = {"required_signal": signal}
    if extra_details is not None:
        details.update(dict(extra_details))
    return MetricScore(
        metric_id=metric_id,
        value=0.0,
        error="missing_trace_signal",
        details=validate_json_dict(details, label=f"{metric_id} details"),
    )


class ToolPresenceTraceMetric:
    """Return 1 when the required tool appears in the normalized trace."""

    def score_trace(
        self,
        trace: TraceView,
        context: Mapping[str, object],
    ) -> MetricScore:
        tool_name = _config_str(context, key="tool_name")
        matching_events = [
            event
            for event in trace.trace_events
            if event.kind == "tool_call" and event.tool_name == tool_name
        ]
        metric_config = _metric_config(context)
        min_calls = metric_config.get("min_calls", 1)
        max_calls = metric_config.get("max_calls")
        if not isinstance(min_calls, int) or min_calls < 0:
            raise ValueError("metric_config['min_calls'] must be a non-negative int.")
        if max_calls is not None and (
            not isinstance(max_calls, int) or max_calls < min_calls
        ):
            raise ValueError(
                "metric_config['max_calls'] must be an int >= metric_config['min_calls']."
            )
        count = len(matching_events)
        matched = count >= min_calls and (max_calls is None or count <= max_calls)
        return MetricScore(
            metric_id="tool_presence",
            value=1.0 if matched else 0.0,
            details=validate_json_dict(
                {
                    "trace_scope": trace.trace_scope,
                    "tool_name": tool_name,
                    "matched_calls": count,
                    "required_min_calls": min_calls,
                    "required_max_calls": max_calls,
                },
                label="tool_presence details",
            ),
        )


class ToolStageTraceMetric:
    """Return 1 when a required tool call appears during the required stage."""

    def score_trace(
        self,
        trace: TraceView,
        context: Mapping[str, object],
    ) -> MetricScore:
        tool_name = _config_str(context, key="tool_name")
        stage = _config_str(context, key="stage")
        matched = any(
            event.kind == "tool_call"
            and event.tool_name == tool_name
            and event.stage == stage
            for event in trace.trace_events
        )
        return MetricScore(
            metric_id="tool_stage",
            value=1.0 if matched else 0.0,
            details=validate_json_dict(
                {
                    "trace_scope": trace.trace_scope,
                    "tool_name": tool_name,
                    "stage": stage,
                },
                label="tool_stage details",
            ),
        )


class EventSequenceTraceMetric:
    """Return 1 when the trace contains the requested ordered event subsequence."""

    def score_trace(
        self,
        trace: TraceView,
        context: Mapping[str, object],
    ) -> MetricScore:
        steps = _config_steps(context)
        matched_steps: list[str] = []
        step_index = 0
        for event in trace.trace_events:
            if step_index >= len(steps):
                break
            if _matches(event, steps[step_index]):
                matched_steps.append(event.event_key)
                step_index += 1
        return MetricScore(
            metric_id="event_sequence",
            value=1.0 if step_index == len(steps) else 0.0,
            details=validate_json_dict(
                {
                    "trace_scope": trace.trace_scope,
                    "matched_step_count": step_index,
                    "required_step_count": len(steps),
                    "matched_event_keys": matched_steps,
                },
                label="event_sequence details",
            ),
        )


class NodePresenceTraceMetric:
    """Return 1 when a required workflow node appears in the normalized trace."""

    def score_trace(
        self,
        trace: TraceView,
        context: Mapping[str, object],
    ) -> MetricScore:
        node_id = _config_str(context, key="node_id")
        node_events = [
            event
            for event in trace.trace_events
            if event.kind in {"node_enter", "node_exit"}
        ]
        if not node_events:
            return _missing_signal_details(
                metric_id="node_presence",
                signal="node_events",
                extra_details={"node_id": node_id, "trace_scope": trace.trace_scope},
            )
        matched = any(event.node_id == node_id for event in node_events)
        return MetricScore(
            metric_id="node_presence",
            value=1.0 if matched else 0.0,
            details=validate_json_dict(
                {
                    "trace_scope": trace.trace_scope,
                    "node_id": node_id,
                    "matched": matched,
                },
                label="node_presence details",
            ),
        )
