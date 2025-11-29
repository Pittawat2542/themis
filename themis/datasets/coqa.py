"""Helpers for working with the stanfordnlp/coqa dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator, List, Sequence

from pydantic import BaseModel, Field, field_validator

_DATASET_NAME = "stanfordnlp/coqa"


class CoQaSample(BaseModel):
    unique_id: str
    story: str
    question: str
    answer: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("metadata", mode="before")
    @classmethod
    def _ensure_metadata(cls, value: Any) -> dict[str, Any]:
        return dict(value or {})

    def to_generation_example(self) -> dict[str, Any]:
        return {
            "unique_id": self.unique_id,
            "story": self.story,
            "question": self.question,
            "answer": self.answer,
            "metadata": dict(self.metadata),
        }


def load_coqa(
    *,
    split: str = "validation", # Test set usually has no labels
    limit: int | None = None,
    source: str = "huggingface",
    data_dir: str | Path | None = None,
) -> List[CoQaSample]:
    """Load CoQA samples from Hugging Face or a local directory."""

    if source not in {"huggingface", "local"}:
        raise ValueError(
            f"Unsupported source '{source}'. Expected one of: 'huggingface', 'local'."
        )

    if source == "huggingface":
        rows = _load_from_huggingface(split=split)
    else:
        if data_dir is None:
            raise ValueError(
                "data_dir must be provided when source='local'. "
                "Pass dataset.data_dir in configs or --data-dir on the CLI."
            )
        rows = _load_from_local(Path(data_dir))

    samples: list[CoQaSample] = []
    for index, row in enumerate(rows, start=1):
        # CoQA has multiple questions per story. We need to flatten them.
        # But wait, usually we want to evaluate turn-by-turn or just single turn.
        # For simplicity, let's flatten: each question is a sample.
        # Or maybe just take the first one? No, that's wasteful.
        # Let's see the structure:
        # 'questions': ['q1', 'q2'], 'answers': {'input_text': ['a1', 'a2'], ...}
        
        story = row.get("story") or ""
        questions = row.get("questions") or []
        answers_data = row.get("answers") or {}
        answers = answers_data.get("input_text") or []
        
        if len(questions) != len(answers):
            # Mismatch, skip or warn?
            # Let's just take the minimum length
            min_len = min(len(questions), len(answers))
            questions = questions[:min_len]
            answers = answers[:min_len]
            
        for i, (q, a) in enumerate(zip(questions, answers)):
            sample = CoQaSample(
                unique_id=f"coqa-{index:05d}-{i:02d}",
                story=story,
                question=str(q),
                answer=str(a),
                metadata={"turn": i, "source": row.get("source")},
            )
            samples.append(sample)
            if limit is not None and len(samples) >= limit:
                break
        
        if limit is not None and len(samples) >= limit:
            break
            
    return samples


def _load_from_huggingface(*, split: str) -> Iterable[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "datasets is required to load CoQA from Hugging Face. Install it via `uv pip install '.[hf]'`."
        ) from exc

    dataset = load_dataset(_DATASET_NAME, split=split)
    for row in dataset:
        yield dict(row)


def _load_from_local(root: Path) -> Iterator[dict[str, Any]]:
    if not root.exists():
        raise FileNotFoundError(f"Local dataset directory not found: {root}")

    for path in root.rglob("*"):
        if path.suffix.lower() == ".json":
            with path.open("r", encoding="utf-8") as handle:
                row = json.load(handle)
                row.setdefault("id", path.stem)
                yield row
        elif path.suffix.lower() in {".jsonl", ".ndjson"}:
            with path.open("r", encoding="utf-8") as handle:
                for line_num, line in enumerate(handle, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    row.setdefault("id", f"{path.stem}-{line_num}")
                    yield row


__all__ = ["CoQaSample", "load_coqa"]
