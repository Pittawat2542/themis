"""Public execution-control and instrumentation contracts."""

from themis.core.config import EvidenceRetention, ExistingRunPolicy, Stage
from themis.core.planner import Planner as _Planner
from themis.core.protocols import EventSubscriber, TracingProvider
from themis.core.results import ExecutionResourcePlan, RunEstimate
from themis.core.snapshot import RunSnapshot


def estimate(snapshot: RunSnapshot) -> RunEstimate:
    """Estimate work and token demand for a compiled snapshot."""

    return _Planner().estimate(snapshot)


def resource_plan(snapshot: RunSnapshot) -> ExecutionResourcePlan:
    """Estimate operational resources for a compiled snapshot."""

    return _Planner().resource_plan(snapshot, snapshot.provenance.runtime)


__all__ = [
    "EventSubscriber",
    "ExecutionResourcePlan",
    "EvidenceRetention",
    "ExistingRunPolicy",
    "RunEstimate",
    "Stage",
    "TracingProvider",
    "estimate",
    "resource_plan",
]
