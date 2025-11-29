"""Tests for GSM8K dataset."""

from unittest.mock import patch

import pytest
from themis.datasets import gsm8k, registry


@pytest.fixture
def mock_load_dataset():
    with patch("datasets.load_dataset") as mock:
        yield mock


def test_gsm8k_loading(mock_load_dataset):
    # Mock data
    mock_data = [
        {"question": "What is 2+2?", "answer": "4", "extra": "info"},
        {"question": "What is 3+3?", "answer": "6"},
    ]
    mock_load_dataset.return_value = mock_data

    # Test loading
    samples = gsm8k.load_gsm8k(split="test", limit=2)

    assert len(samples) == 2
    assert samples[0].question == "What is 2+2?"
    assert samples[0].answer == "4"
    assert samples[0].metadata == {"extra": "info"}
    assert samples[0].unique_id == "gsm8k-00001"

    # Test generation example
    gen_example = samples[0].to_generation_example()
    assert gen_example["question"] == "What is 2+2?"
    assert gen_example["answer"] == "4"
    assert gen_example["extra"] == "info"


def test_create_dataset_gsm8k(mock_load_dataset):
    mock_load_dataset.return_value = [{"question": "Q", "answer": "A"}]
    samples = registry.create_dataset("gsm8k", limit=1)
    assert len(samples) == 1
    assert samples[0]["question"] == "Q"
