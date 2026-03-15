from __future__ import annotations

from themis.records import AdjudicationRecord, AnnotationRecord


def test_human_eval_records_are_exported_and_structured():
    annotation = AnnotationRecord(
        spec_hash="annotation_1",
        rater_id="rater-1",
        rubric_version="rubric-2026-03-01",
        label="pass",
        notes="Clear and correct.",
        time_spent_s=12.5,
    )
    adjudication = AdjudicationRecord(
        spec_hash="adjudication_1",
        final_label="pass",
        adjudicator_id="lead-rater",
        rationale="Consensus after review.",
    )

    assert annotation.rater_id == "rater-1"
    assert annotation.label == "pass"
    assert adjudication.final_label == "pass"
    assert adjudication.adjudicator_id == "lead-rater"
