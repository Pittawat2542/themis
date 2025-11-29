"""Helpers for working with the openai/gsm8k dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator, List, Sequence

from pydantic import BaseModel, Field, field_validator

_DATASET_NAME = "openai/gsm8k"


class Gsm8kSample(BaseModel):
    unique_id: str
    question: str
    answer: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("metadata", mode="before")
    @classmethod
    def _ensure_metadata(cls, value: Any) -> dict[str, Any]:
        return dict(value or {})

    def to_generation_example(self) -> dict[str, Any]:
        payload = {
            "unique_id": self.unique_id,
            "question": self.question,
            "answer": self.answer,
        }
        payload.update(self.metadata)
        return payload


def load_gsm8k(
    *,
    split: str = "test",
    limit: int | None = None,
    source: str = "huggingface",
    data_dir: str | Path | None = None,
    subset: str = "main",
) -> List[Gsm8kSample]:
    """Load GSM8K samples from Hugging Face or a local directory."""

    if source not in {"huggingface", "local"}:
        raise ValueError(
            f"Unsupported source '{source}'. Expected one of: 'huggingface', 'local'."
        )

    if source == "huggingface":
        rows = _load_from_huggingface(split=split, subset=subset)
    else:
        if data_dir is None:
            raise ValueError(
                "data_dir must be provided when source='local'. "
                "Pass dataset.data_dir in configs or --data-dir on the CLI."
            )
        rows = _load_from_local(Path(data_dir))

    samples: list[Gsm8kSample] = []
    for index, row in enumerate(rows, start=1):
        sample = _row_to_sample(row, index=index)
        samples.append(sample)
        if limit is not None and len(samples) >= limit:
            break
    return samples


def _row_to_sample(row: dict[str, Any], *, index: int) -> Gsm8kSample:
    unique_id = (
        row.get("id")
        or row.get("unique_id")
        or f"gsm8k-{index:05d}"
    )
    question = row.get("question") or row.get("problem") or ""
    answer = row.get("answer") or ""
    
    core_keys = {"id", "unique_id", "question", "problem", "answer"}
    metadata = {key: value for key, value in row.items() if key not in core_keys}
    
    return Gsm8kSample(
        unique_id=str(unique_id),
        question=str(question),
        answer=str(answer),
        metadata=metadata,
    )


def _load_from_huggingface(*, split: str, subset: str) -> Iterable[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "datasets is required to load GSM8K from Hugging Face. Install it via `uv pip install '.[hf]'`."
        ) from exc

    dataset = load_dataset(_DATASET_NAME, subset, split=split)
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


__all__ = ["Gsm8kSample", "load_gsm8k"]
