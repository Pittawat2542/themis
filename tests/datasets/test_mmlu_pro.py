import json

from themis.datasets import mmlu_pro


def test_load_mmlu_pro_from_local_directory(tmp_path):
    data_dir = tmp_path / "mmlu-pro"
    data_dir.mkdir()
    sample_path = data_dir / "example.json"
    sample_payload = {
        "question": "Which color is the sky on a clear day?",
        "choices": ["Green", "Blue", "Red", "Yellow"],
        "answer": 1,
        "subject": "science",
        "difficulty": "easy",
    }
    sample_path.write_text(json.dumps(sample_payload), encoding="utf-8")

    samples = mmlu_pro.load_mmlu_pro(
        source="local", data_dir=data_dir, subjects=["science"]
    )

    assert len(samples) == 1
    record = samples[0].to_generation_example()
    assert record["answer"] == "B"
    assert record["choices"][1] == "Blue"
    assert record["choice_labels"] == ["A", "B", "C", "D"]


def test_load_mmlu_pro_subject_filter(tmp_path):
    data_dir = tmp_path / "subjects"
    data_dir.mkdir()
    sample_a = {
        "question": "Capital of France?",
        "choices": ["Paris", "Rome", "Berlin", "Madrid"],
        "answer": "A",
        "subject": "geography",
    }
    sample_b = {
        "question": "Capital of Italy?",
        "choices": ["Paris", "Rome", "Berlin", "Madrid"],
        "answer": "B",
        "subject": "history",
    }
    (data_dir / "row_a.json").write_text(json.dumps(sample_a), encoding="utf-8")
    (data_dir / "row_b.json").write_text(json.dumps(sample_b), encoding="utf-8")

    samples = mmlu_pro.load_mmlu_pro(
        source="local", data_dir=data_dir, subjects=["geography"]
    )

    assert len(samples) == 1
    assert samples[0].subject == "geography"
