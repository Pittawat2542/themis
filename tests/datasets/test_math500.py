import json

import pytest
from pydantic import ValidationError

from themis.datasets import math500


def test_load_math500_from_local_directory(tmp_path):
    sample_dir = tmp_path / "test" / "algebra"
    sample_dir.mkdir(parents=True)
    sample_path = sample_dir / "sample.json"
    sample_payload = {
        "problem": "What is 3 + 4?",
        "solution": "Add the numbers.",
        "answer": "7",
        "subject": "algebra",
        "level": 1,
        "unique_id": "local-1",
    }
    sample_path.write_text(json.dumps(sample_payload), encoding="utf-8")

    samples = math500.load_math500(source="local", data_dir=tmp_path)

    assert len(samples) == 1
    assert samples[0].unique_id == "local-1"
    payload = samples[0].to_generation_example()
    assert payload["answer"] == "7"
    assert payload["subject"] == "algebra"


def test_math_sample_validates_missing_fields():
    with pytest.raises(ValidationError):
        math500.MathSample.model_validate(
            {
                "problem": "What is 1+1?",
                "solution": "Add",
                "answer": "2",
            }
        )


def test_math_sample_level_normalization():
    sample = math500.MathSample(
        unique_id="s1",
        problem="",
        solution="",
        answer="",
        level="3",
    )
    assert sample.level == 3
