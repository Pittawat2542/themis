"""Helpers for working with the allenai/sciq dataset."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterable, Iterator, List, Sequence

from pydantic import BaseModel, Field, field_validator

_DATASET_NAME = "allenai/sciq"


class SciQSample(BaseModel):
    unique_id: str
    question: str
    choices: list[str]
    answer: str
    support: str = Field(default="")
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("choices", mode="before")
    @classmethod
    def _ensure_choices(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            return [str(item) for item in value]
        raise TypeError("choices must be a sequence")

    def to_generation_example(self) -> dict[str, Any]:
        return {
            "unique_id": self.unique_id,
            "question": self.question,
            "choices": list(self.choices),
            "answer": self.answer,
            "support": self.support,
            "metadata": dict(self.metadata),
        }


def load_sciq(
    *,
    split: str = "test",
    limit: int | None = None,
    source: str = "huggingface",
    data_dir: str | Path | None = None,
) -> List[SciQSample]:
    """Load SciQ samples from Hugging Face or a local directory."""

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

    samples: list[SciQSample] = []
    for index, row in enumerate(rows, start=1):
        sample = _row_to_sample(row, index=index)
        samples.append(sample)
        if limit is not None and len(samples) >= limit:
            break
    return samples


def _row_to_sample(row: dict[str, Any], *, index: int) -> SciQSample:
    unique_id = (
        row.get("id")
        or row.get("unique_id")
        or f"sciq-{index:05d}"
    )
    question = row.get("question") or ""
    
    # SciQ has 'correct_answer', 'distractor1', 'distractor2', 'distractor3'
    correct = str(row.get("correct_answer") or "")
    distractors = [
        str(row.get("distractor1") or ""),
        str(row.get("distractor2") or ""),
        str(row.get("distractor3") or ""),
    ]
    
    # Filter empty distractors just in case
    distractors = [d for d in distractors if d]
    
    choices = [correct] + distractors
    # Sort to be deterministic
    choices.sort()
    
    support = str(row.get("support") or "")

    metadata_keys = {
        "question", "correct_answer", "distractor1", "distractor2", "distractor3", "support", "id"
    }
    metadata = {key: value for key, value in row.items() if key not in metadata_keys}
    
    return SciQSample(
        unique_id=str(unique_id),
        question=str(question),
        choices=choices,
        answer=correct,
        support=support,
        metadata=metadata,
    )


def _load_from_huggingface(*, split: str) -> Iterable[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "datasets is required to load SciQ from Hugging Face. Install it via `uv pip install '.[hf]'`."
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


__all__ = ["SciQSample", "load_sciq"]
