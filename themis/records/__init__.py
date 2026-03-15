"""Public record models and re-exports for persisted output artifacts."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from themis.records.base import RecordBase
    from themis.records.candidate import CandidateRecord
    from themis.records.conversation import (
        Conversation,
        ConversationEvent,
        MessageEvent,
        MessagePayload,
        NodeEnterEvent,
        NodeEnterPayload,
        NodeExitEvent,
        NodeExitPayload,
        ToolCallEvent,
        ToolCallPayload,
        ToolResultEvent,
        ToolResultPayload,
    )
    from themis.records.error import ErrorRecord
    from themis.records.evaluation import EvaluationRecord, MetricScore
    from themis.records.extraction import ExtractionRecord
    from themis.records.human_eval import AdjudicationRecord, AnnotationRecord
    from themis.records.inference import InferenceRecord, TokenUsage
    from themis.records.provenance import ProvenanceRecord
    from themis.records.timeline import RecordTimeline, TimelineStageRecord
    from themis.records.trial import TrialRecord
    from themis.records.observability import ObservabilityLink, ObservabilitySnapshot
    from themis.records.observability import ObservabilityRefs

__all__ = [
    "CandidateRecord",
    "Conversation",
    "ConversationEvent",
    "ErrorRecord",
    "EvaluationRecord",
    "ExtractionRecord",
    "InferenceRecord",
    "AnnotationRecord",
    "AdjudicationRecord",
    "ObservabilityLink",
    "ObservabilitySnapshot",
    "MessageEvent",
    "MessagePayload",
    "MetricScore",
    "NodeEnterEvent",
    "NodeEnterPayload",
    "NodeExitEvent",
    "NodeExitPayload",
    "ProvenanceRecord",
    "ObservabilityRefs",
    "RecordBase",
    "RecordTimeline",
    "TimelineStageRecord",
    "TokenUsage",
    "ToolCallEvent",
    "ToolCallPayload",
    "ToolResultEvent",
    "ToolResultPayload",
    "TrialRecord",
]

_RE_EXPORTS = {
    "RecordBase": ("themis.records.base", "RecordBase"),
    "ErrorRecord": ("themis.records.error", "ErrorRecord"),
    "InferenceRecord": ("themis.records.inference", "InferenceRecord"),
    "TokenUsage": ("themis.records.inference", "TokenUsage"),
    "ExtractionRecord": ("themis.records.extraction", "ExtractionRecord"),
    "EvaluationRecord": ("themis.records.evaluation", "EvaluationRecord"),
    "MetricScore": ("themis.records.evaluation", "MetricScore"),
    "AnnotationRecord": ("themis.records.human_eval", "AnnotationRecord"),
    "AdjudicationRecord": ("themis.records.human_eval", "AdjudicationRecord"),
    "CandidateRecord": ("themis.records.candidate", "CandidateRecord"),
    "RecordTimeline": ("themis.records.timeline", "RecordTimeline"),
    "TimelineStageRecord": ("themis.records.timeline", "TimelineStageRecord"),
    "TrialRecord": ("themis.records.trial", "TrialRecord"),
    "Conversation": ("themis.records.conversation", "Conversation"),
    "ConversationEvent": ("themis.records.conversation", "ConversationEvent"),
    "MessageEvent": ("themis.records.conversation", "MessageEvent"),
    "MessagePayload": ("themis.records.conversation", "MessagePayload"),
    "ToolCallEvent": ("themis.records.conversation", "ToolCallEvent"),
    "ToolCallPayload": ("themis.records.conversation", "ToolCallPayload"),
    "ToolResultEvent": ("themis.records.conversation", "ToolResultEvent"),
    "ToolResultPayload": ("themis.records.conversation", "ToolResultPayload"),
    "NodeEnterEvent": ("themis.records.conversation", "NodeEnterEvent"),
    "NodeEnterPayload": ("themis.records.conversation", "NodeEnterPayload"),
    "NodeExitEvent": ("themis.records.conversation", "NodeExitEvent"),
    "NodeExitPayload": ("themis.records.conversation", "NodeExitPayload"),
    "ProvenanceRecord": ("themis.records.provenance", "ProvenanceRecord"),
    "ObservabilityLink": ("themis.records.observability", "ObservabilityLink"),
    "ObservabilitySnapshot": ("themis.records.observability", "ObservabilitySnapshot"),
    "ObservabilityRefs": ("themis.records.observability", "ObservabilityRefs"),
}


def __getattr__(name: str) -> object:
    try:
        module_name, attr_name = _RE_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    module = __import__(module_name, fromlist=[attr_name])
    return getattr(module, attr_name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
