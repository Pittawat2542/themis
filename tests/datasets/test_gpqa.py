"""Tests for GPQA dataset."""

from unittest.mock import patch

import pytest
from themis.datasets import gpqa, registry


@pytest.fixture
def mock_load_dataset():
    with patch("datasets.load_dataset") as mock:
        yield mock


def test_gpqa_loading(mock_load_dataset):
    # Mock data
    mock_data = [
        {
            "Question": "What is the capital of France?",
            "Correct Answer": "Paris",
            "Incorrect Answer 1": "London",
            "Incorrect Answer 2": "Berlin",
            "Incorrect Answer 3": "Madrid",
            "extra": "metadata",
        }
    ]
    mock_load_dataset.return_value = mock_data

    # Test loading
    samples = gpqa.load_gpqa(split="test", limit=1)

    assert len(samples) == 1
    assert samples[0].question == "What is the capital of France?"
    assert samples[0].answer == "D"
    assert "Paris" in samples[0].choices
    assert "London" in samples[0].choices
    assert samples[0].metadata["extra"] == "metadata"
    assert samples[0].unique_id == "gpqa-00001"

    # Test generation example
    gen_example = samples[0].to_generation_example()
    assert gen_example["question"] == "What is the capital of France?"
    assert gen_example["answer"] == "D"
    assert len(gen_example["choices"]) == 4


def test_create_dataset_gpqa(mock_load_dataset):
    mock_load_dataset.return_value = [
        {
            "Question": "Q",
            "Correct Answer": "A",
            "Incorrect Answer 1": "B",
            "Incorrect Answer 2": "C",
            "Incorrect Answer 3": "D",
        }
    ]
    samples = registry.create_dataset("gpqa", limit=1)
    assert len(samples) == 1
    assert samples[0]["question"] == "Q"
