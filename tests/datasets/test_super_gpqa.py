import json

from themis.datasets import super_gpqa


def test_load_super_gpqa_from_local_directory(tmp_path):
    data_dir = tmp_path / "supergpqa"
    data_dir.mkdir()
    sample_path = data_dir / "sample.json"
    sample_payload = {
        "id": "gpqa-1",
        "question": "What is 2 + 2?",
        "options": {"A": "3", "B": "4", "C": "5", "D": "6"},
        "answer": "B",
        "field": "arithmetic",
        "difficulty": "hard",
    }
    sample_path.write_text(json.dumps(sample_payload), encoding="utf-8")

    samples = super_gpqa.load_super_gpqa(
        source="local", data_dir=data_dir, subjects=["arithmetic"]
    )

    assert len(samples) == 1
    record = samples[0].to_generation_example()
    assert record["unique_id"] == "gpqa-1"
    assert record["answer"] == "B"
    assert record["choices"] == ["3", "4", "5", "6"]
    assert record["choice_labels"] == ["A", "B", "C", "D"]
    assert record["subject"] == "arithmetic"


def test_load_super_gpqa_from_jsonl(tmp_path):
    data_dir = tmp_path / "gpqa-jsonl"
    data_dir.mkdir()
    sample_path = data_dir / "split.jsonl"
    rows = [
        json.dumps(
            {
                "question": "What is 3 + 3?",
                "choices": ["5", "6", "7", "8"],
                "answer": 1,
                "category": "math",
            }
        )
    ]
    sample_path.write_text("\n".join(rows), encoding="utf-8")

    samples = super_gpqa.load_super_gpqa(
        source="local", data_dir=data_dir, subjects=None
    )

    assert len(samples) == 1
    payload = samples[0].to_generation_example()
    assert payload["answer"] == "B"
    assert payload["choices"][1] == "6"
