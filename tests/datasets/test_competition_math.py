import json

import pytest

from themis.datasets import competition_math


@pytest.mark.parametrize(
    "dataset_name",
    [
        "math-ai/aime24",
        "math-ai/aime25",
        "math-ai/amc23",
        "math-ai/olympiadbench",
        "ByteDance-Seed/BeyondAIME",
    ],
)
def test_load_competition_math_from_local(tmp_path, dataset_name):
    data_dir = tmp_path / "competition"
    data_dir.mkdir()
    sample_path = data_dir / "sample.json"
    sample_payload = {
        "problem": "Solve for x: 1 + 1",
        "solution": "Add the two numbers.",
        "answer": "2",
        "subject": "algebra",
        "difficulty": "easy",
        "extra_field": "metadata",
    }
    sample_path.write_text(json.dumps(sample_payload), encoding="utf-8")

    samples = competition_math.load_competition_math(
        dataset=dataset_name,
        source="local",
        data_dir=data_dir,
    )

    assert len(samples) == 1
    payload = samples[0].to_generation_example()
    assert payload["answer"] == "2"
    assert payload["problem"]
    assert payload["subject"] == "algebra"
    assert payload["level"] == "easy"
    assert payload["extra_field"] == "metadata"


def test_competition_math_subject_filter(tmp_path):
    data_dir = tmp_path / "filter"
    data_dir.mkdir()
    payload_a = {
        "problem": "Find 2+2",
        "solution": "Add",
        "answer": "4",
        "subject": "algebra",
    }
    payload_b = {
        "problem": "Find 3+5",
        "solution": "Add",
        "answer": "8",
        "subject": "number theory",
    }
    (data_dir / "a.json").write_text(json.dumps(payload_a), encoding="utf-8")
    (data_dir / "b.json").write_text(json.dumps(payload_b), encoding="utf-8")

    samples = competition_math.load_competition_math(
        dataset="math-ai/aime24",
        source="local",
        data_dir=data_dir,
        subjects=["algebra"],
    )

    assert len(samples) == 1
    assert samples[0].subject == "algebra"
