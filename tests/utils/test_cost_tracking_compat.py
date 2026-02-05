from __future__ import annotations

import pytest

from tests.factories import make_record
from themis.utils.cost_tracking import CostTracker


def test_utils_cost_tracker_tracks_generation_with_legacy_api():
    tracker = CostTracker()
    record = make_record(model_id="gpt-4o")

    tracked = tracker.track_generation(record, input_tokens=1000, output_tokens=200)
    summary = tracker.get_summary()

    assert tracked.total_cost > 0
    assert summary.num_requests == 1
    assert summary.total_cost == pytest.approx(tracked.total_cost)


def test_utils_cost_tracker_exposes_runtime_breakdown_api():
    tracker = CostTracker()
    tracker.record_generation(
        model="gpt-4o",
        prompt_tokens=500,
        completion_tokens=250,
        cost=0.0125,
    )

    breakdown = tracker.get_breakdown()
    summary = tracker.get_summary()
    assert breakdown.total_cost == pytest.approx(0.0125)
    assert breakdown.api_calls == 1
    assert summary.total_cost == pytest.approx(0.0125)
    assert summary.num_requests == 1
