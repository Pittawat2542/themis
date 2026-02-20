"""Tests for batch 2 benchmarks."""

from unittest.mock import patch

import pytest
from themis.datasets import (
    commonsense_qa,
    coqa,
    gsm_symbolic,
    med_qa,
    medmcqa,
    piqa,
    registry,
    sciq,
    social_i_qa,
)


@pytest.fixture
def mock_load_dataset():
    with patch("datasets.load_dataset") as mock:
        yield mock


def test_gsm_symbolic_loading(mock_load_dataset):
    mock_load_dataset.return_value = [{"question": "Q", "answer": "A", "id": "1"}]
    samples = gsm_symbolic.load_gsm_symbolic(split="test", limit=1)
    assert len(samples) == 1
    assert samples[0].question == "Q"
    assert samples[0].answer == "A"


def test_medmcqa_loading(mock_load_dataset):
    mock_load_dataset.return_value = [
        {
            "question": "Q",
            "opa": "A",
            "opb": "B",
            "opc": "C",
            "opd": "D",
            "cop": 1,
            "id": "1",
        }
    ]
    samples = medmcqa.load_medmcqa(split="test", limit=1)
    assert len(samples) == 1
    assert samples[0].question == "Q"
    assert samples[0].choices == ["A", "B", "C", "D"]
    assert samples[0].answer == "A"


def test_med_qa_loading(mock_load_dataset):
    mock_load_dataset.return_value = [
        {
            "question": "Q",
            "choices": [{"key": "A", "text": "OptA"}, {"key": "B", "text": "OptB"}],
            "answer": "A",
            "id": "1",
        }
    ]
    samples = med_qa.load_med_qa(split="test", limit=1)
    assert len(samples) == 1
    assert samples[0].question == "Q"
    assert samples[0].choices == ["OptA", "OptB"]
    assert samples[0].answer == "A"


def test_sciq_loading(mock_load_dataset):
    mock_load_dataset.return_value = [
        {
            "question": "Q",
            "correct_answer": "C",
            "distractor1": "D1",
            "distractor2": "D2",
            "distractor3": "D3",
            "support": "S",
            "id": "1",
        }
    ]
    samples = sciq.load_sciq(split="test", limit=1)
    assert len(samples) == 1
    assert samples[0].question == "Q"
    assert "C" in samples[0].choices
    assert samples[0].answer == "C"
    assert samples[0].support == "S"


def test_commonsense_qa_loading(mock_load_dataset):
    mock_load_dataset.return_value = [
        {
            "question": "Q",
            "choices": {"label": ["A", "B"], "text": ["T1", "T2"]},
            "answerKey": "A",
            "id": "1",
        }
    ]
    samples = commonsense_qa.load_commonsense_qa(split="validation", limit=1)
    assert len(samples) == 1
    assert samples[0].question == "Q"
    assert samples[0].choices == ["T1", "T2"]
    assert samples[0].answer == "A"


def test_piqa_loading(mock_load_dataset):
    mock_load_dataset.return_value = [
        {"goal": "G", "sol1": "S1", "sol2": "S2", "label": 0, "id": "1"}
    ]
    samples = piqa.load_piqa(split="validation", limit=1)
    assert len(samples) == 1
    assert samples[0].goal == "G"
    assert samples[0].choices == ["S1", "S2"]
    assert samples[0].answer == "S1"


def test_social_i_qa_loading(mock_load_dataset):
    mock_load_dataset.return_value = [
        {
            "context": "C",
            "question": "Q",
            "answerA": "A",
            "answerB": "B",
            "answerC": "C",
            "label": "1",
            "id": "1",
        }
    ]
    samples = social_i_qa.load_social_i_qa(split="validation", limit=1)
    assert len(samples) == 1
    assert samples[0].context == "C"
    assert samples[0].question == "Q"
    assert samples[0].choices == ["A", "B", "C"]
    assert samples[0].answer == "A"


def test_coqa_loading(mock_load_dataset):
    mock_load_dataset.return_value = [
        {
            "story": "S",
            "questions": ["Q1", "Q2"],
            "answers": {"input_text": ["A1", "A2"]},
            "id": "1",
        }
    ]
    samples = coqa.load_coqa(split="validation", limit=2)
    # Should flatten to 2 samples
    assert len(samples) == 2
    assert samples[0].question == "Q1"
    assert samples[0].answer == "A1"
    assert samples[1].question == "Q2"
    assert samples[1].answer == "A2"


def test_registry_entries_batch2():
    assert registry.is_dataset_registered("gsm-symbolic")
    assert registry.is_dataset_registered("medmcqa")
    assert registry.is_dataset_registered("med_qa")
    assert registry.is_dataset_registered("sciq")
    assert registry.is_dataset_registered("commonsense_qa")
    assert registry.is_dataset_registered("piqa")
    assert registry.is_dataset_registered("social_i_qa")
    assert registry.is_dataset_registered("coqa")
